import numpy as np
import os
import plyfile
from plyfile import PlyData, PlyElement
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import math
import random
import torch.optim.lr_scheduler as lr_scheduler # Import the scheduler

# --- DEFINITIVE FIX: Integrate FAISS for high-speed search ---
try:
    import faiss
    FAISS_AVAILABLE = True
    print("FAISS library found. Using for high-speed nearest neighbor search.")
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS is not installed. This script requires it for performance.")
    print("Please install with: conda install -c pytorch faiss-gpu")


# Ensure CUDA is available
if not torch.cuda.is_available():
    print("CUDA is not available. This script will run on CPU.")
    device = "cpu"
else:
    device = "cuda"
    print(f"Using device: {device}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# --- Configuration ---
INPUT_PLY_FILE = "/home/chz/3DGS/gaussian-splatting/output/069683f5-b/point_cloud/iteration_30000/point_cloud.ply"
CAMERAS_JSON_FILE = "/home/chz/3DGS/gaussian-splatting/output/069683f5-b/cameras.json"

# Training options
N_RAYS_PER_BATCH = 1024
N_SAMPLES_PER_RAY = 128
NUM_TRAINING_ITERATIONS = 250000 
LEARNING_RATE = 5e-5 
PERTURB_SAMPLES = 1.0
USE_VIEWDIRS = True
K_NEAREST_GAUSSIANS = 8 # Kept from previous step

# MLP options
MLP_DEPTH = 8
MLP_WIDTH = 512
POS_ENCODE_FREQS = 12
VIEW_ENCODE_FREQS = 4
NN_QUERY_CHUNK = 1024 * 32 

# --- Output Path for Saved Model ---
OUTPUT_DIR = "nerf_checkpoints"
MODEL_FILENAME = "GS_NeRF_DEFINITIVE.ckpt" 

# --- Positional Encoding (No changes) ---
class Embedder(nn.Module):
    def __init__(self, input_dims, include_input, max_freq_log2, num_freqs, log_sampling=True):
        super(Embedder, self).__init__()
        self.include_input, self.max_freq_log2, self.num_freqs = include_input, max_freq_log2, num_freqs
        self.out_dim = input_dims * (1 + 2 * num_freqs) if include_input else input_dims * 2 * num_freqs
        self.freq_bands = 2.**torch.linspace(0., max_freq_log2, steps=num_freqs)
    def forward(self, inputs):
        outputs = [inputs] if self.include_input else []
        for freq in self.freq_bands:
            outputs.extend([torch.sin(inputs * freq), torch.cos(inputs * freq)])
        return torch.cat(outputs, dim=-1)

def get_embedder(multires, i_embed=0):
    if i_embed == -1: return nn.Identity(), 3
    embed_kwargs = {'include_input': True, 'max_freq_log2': multires - 1, 'num_freqs': multires}
    embedder_obj = Embedder(3, **embed_kwargs)
    return embedder_obj, embedder_obj.out_dim

# --- NeRF-like MLP Model (with bias fix) ---
class NeRFMLP(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, use_viewdirs=False, skips=[4]):
        super(NeRFMLP, self).__init__()
        self.D, self.W, self.input_ch, self.input_ch_views = D, W, input_ch, input_ch_views
        self.use_viewdirs, self.skips = use_viewdirs, skips
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        if use_viewdirs:
            self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            torch.nn.init.constant_(self.alpha_linear.bias, -1.5)
            self.rgb_linear = nn.Linear(W//2, 3)
            
    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1) if self.use_viewdirs else (x, None)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = F.relu(l(h))
            if i in self.skips: h = torch.cat([input_pts, h], -1)
        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
            for l in self.views_linears:
                h = F.relu(l(h))
            rgb = torch.sigmoid(self.rgb_linear(h))
            outputs = torch.cat([rgb, alpha], -1)
        return outputs

# --- Helper Functions ---
def get_rays(H, W, fx, fy, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W - 1, int(W)), torch.linspace(0, H - 1, int(H)), indexing='xy')
    i, j = i.to(device), j.to(device)
    cx, cy = W * 0.5, H * 0.5
    dirs = torch.stack([(i - cx) / fx, (j - cy) / fy, torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_d = F.normalize(rays_d, dim=-1)
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return rays_o, rays_d

def eval_sh_basis_torch(degree: int, view_dir: torch.Tensor) -> torch.Tensor:
    N = view_dir.shape[0]
    sh_basis = torch.zeros((N, (degree + 1)**2), device=view_dir.device, dtype=view_dir.dtype)
    x, y, z = view_dir.unbind(-1)
    C0,C1,C2,C3,C4 = 0.28209479177,0.4886025119,1.09254843059,0.31539156525,0.54627421529
    if degree >= 0: sh_basis[:, 0] = C0
    if degree >= 1: sh_basis[:, 1],sh_basis[:, 2],sh_basis[:, 3] = -C1*y,C1*z,-C1*x
    if degree >= 2: sh_basis[:, 4],sh_basis[:, 5],sh_basis[:, 6],sh_basis[:, 7],sh_basis[:, 8] = C2*x*y,-C2*y*z,C3*(3*z*z-1),-C2*x*z,C4*(x*x-y*y)
    return sh_basis

def sh_to_view_dependent_rgb_torch(sh_coeffs_all: torch.Tensor, view_dir_normalized: torch.Tensor, sh_degree: int) -> torch.Tensor:
    if sh_degree < 0: raise ValueError("SH degree must be non-negative.")
    dc_term = sh_coeffs_all[..., 0:3]
    base_color = torch.sigmoid(dc_term)
    if sh_degree == 0: return base_color
    num_coeffs_per_channel = (sh_degree + 1)**2
    sh_rest_coeffs = sh_coeffs_all[..., 3:].reshape(*sh_coeffs_all.shape[:-1], 3, num_coeffs_per_channel - 1)
    sh_basis = eval_sh_basis_torch(sh_degree, view_dir_normalized)[..., 1:]
    if sh_basis.dim() < sh_rest_coeffs.dim(): sh_basis = sh_basis.unsqueeze(-2)
    view_dependent_offset = torch.sum(sh_rest_coeffs * sh_basis, dim=-1)
    final_color = base_color + view_dependent_offset
    return torch.clamp(final_color, 0.0, 1.0)

def sample_points_along_rays(rays_o, rays_d, N_samples, near, far, perturb=0.):
    t_vals = torch.linspace(0., 1., N_samples, device=rays_o.device)
    z_vals = near * (1.-t_vals) + far * t_vals
    z_vals = z_vals.expand([rays_o.shape[0], N_samples])
    if perturb > 0.:
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper, lower = torch.cat([mids, z_vals[..., -1:]], -1), torch.cat([z_vals[..., :1], mids], -1)
        z_vals = lower + (upper - lower) * torch.rand(z_vals.shape, device=rays_o.device)
    return rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

def get_dynamic_near_far(camera_position: torch.Tensor, points: torch.Tensor, percentile_min=5.0, percentile_max=95.0):
    dists = torch.norm(points - camera_position, dim=-1)
    near_plane = torch.quantile(dists, percentile_min / 100.0)
    far_plane = torch.quantile(dists, percentile_max / 100.0)
    return near_plane.item() * 0.9, far_plane.item() * 1.1

# --- FAISS-accelerated function to get ground truth data from 3D Gaussians ---
def get_3dgs_ground_truth_raw(pts: torch.Tensor, viewdirs: torch.Tensor, gaussians_data: dict, sh_degree: int, faiss_index, chunk: int, k_neighbors: int) -> torch.Tensor:
    gauss_pos = gaussians_data['pos']
    gauss_opacity_logits = gaussians_data['opacity']
    gauss_sh = gaussians_data['sh_coeffs_all']
    gauss_scales = gaussians_data['scale']
    
    num_query_pts = pts.shape[0]
    all_raw_gt_chunks = []

    for i in range(0, num_query_pts, chunk):
        pts_chunk = pts[i:i+chunk]
        viewdirs_chunk = viewdirs[i:i+chunk]
        chunk_size = pts_chunk.shape[0]
        
        pts_chunk_np = pts_chunk.cpu().numpy()
        distances, indices = faiss_index.search(pts_chunk_np, k_neighbors)
        
        indices_tensor = torch.from_numpy(indices).to(device=device, dtype=torch.long)
        
        neighbor_pos = gauss_pos[indices_tensor]
        neighbor_opacity_logits = gauss_opacity_logits[indices_tensor]
        neighbor_sh = gauss_sh[indices_tensor]
        neighbor_scales = gauss_scales[indices_tensor]

        query_pts_expanded = pts_chunk.unsqueeze(1)
        dist_sq = torch.sum((query_pts_expanded - neighbor_pos)**2, dim=-1)
        
        per_gauss_avg_scale_sq = torch.prod(torch.abs(neighbor_scales), dim=-1) + 1e-8
        density_weights = torch.exp(-0.5 * dist_sq / per_gauss_avg_scale_sq)
        
        viewdirs_expanded = viewdirs_chunk.unsqueeze(1).expand(-1, k_neighbors, -1)
        flat_sh_coeffs = neighbor_sh.reshape(-1, neighbor_sh.shape[-1])
        flat_viewdirs = viewdirs_expanded.reshape(-1, 3)
        flat_colors = sh_to_view_dependent_rgb_torch(flat_sh_coeffs, flat_viewdirs, sh_degree)
        neighbor_colors = flat_colors.reshape(chunk_size, k_neighbors, 3)
        
        weights_for_color = density_weights
        weights_sum = torch.sum(weights_for_color, dim=-1, keepdim=True) + 1e-8
        normalized_weights = weights_for_color / weights_sum
        final_color = torch.sum(normalized_weights.unsqueeze(-1) * neighbor_colors, dim=1)

        neighbor_opacities = torch.sigmoid(neighbor_opacity_logits)
        blended_density = torch.sum(density_weights * neighbor_opacities, dim=1)
        
        target_sigma = F.relu(blended_density)
        
        raw_gt_chunk = torch.cat([final_color, target_sigma.unsqueeze(-1)], dim=-1)
        all_raw_gt_chunks.append(raw_gt_chunk)
        
    return torch.cat(all_raw_gt_chunks, dim=0)

# --- Main Training Function ---
def train_nerf_from_3dgs():
    if not FAISS_AVAILABLE:
        print("ERROR: FAISS is required for this script. Please install it and try again.")
        return

    print(f"Reading trained 3D Gaussians from '{INPUT_PLY_FILE}'...")
    try:
        plydata = plyfile.PlyData.read(INPUT_PLY_FILE)
        gaussians_data_np = plydata['vertex'].data
    except FileNotFoundError:
        print(f"ERROR: Input PLY file not found at '{INPUT_PLY_FILE}'")
        return

    num_f_rest = sum(1 for name in gaussians_data_np.dtype.names if name.startswith('f_rest_'))
    sh_degree = int(np.sqrt(num_f_rest / 3 + 1) - 1) if num_f_rest > 0 else 0
    print(f"Inferred Spherical Harmonic Degree: {sh_degree}")
    sh_prop_names = [f'f_dc_{j}' for j in range(3)] + [f'f_rest_{j}' for j in range(num_f_rest)]
    
    gaussians_data = {
        'pos': torch.from_numpy(np.stack([gaussians_data_np[c] for c in ['x','y','z']], -1)).float().to(device),
        'opacity': torch.from_numpy(gaussians_data_np['opacity']).float().to(device),
        'sh_coeffs_all': torch.from_numpy(np.stack([gaussians_data_np[n] for n in sh_prop_names], -1)).float().to(device),
        'scale': torch.from_numpy(np.stack([gaussians_data_np[s] for s in ['scale_0', 'scale_1', 'scale_2']], -1)).float().to(device)
    }

    print(f"Loading cameras from 3DGS-style '{CAMERAS_JSON_FILE}'...")
    try:
        with open(CAMERAS_JSON_FILE, 'r') as f: cameras_data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Cameras JSON file not found at '{CAMERAS_JSON_FILE}'")
        return

    first_cam = cameras_data[0]
    H, W = first_cam['height'], first_cam['width']
    fx, fy = first_cam['fx'], first_cam['fy']
    print(f"Derived H={H}, W={W}, fx={fx:.2f}, fy={fy:.2f} from cameras.json")

    all_c2w_matrices = []
    for frame in cameras_data:
        R = torch.tensor(frame['rotation'], dtype=torch.float32)
        t = torch.tensor(frame['position'], dtype=torch.float32)
        c2w = torch.eye(4)
        c2w[:3, :3], c2w[:3, 3] = R, t
        all_c2w_matrices.append(c2w.to(device))
    
    print("Building high-performance FAISS index (IndexIVFFlat)...")
    gauss_pos_np = gaussians_data['pos'].cpu().numpy()
    d = gauss_pos_np.shape[1]
    nlist = 4096 
    
    quantizer = faiss.IndexFlatL2(d)
    faiss_index = faiss.IndexIVFFlat(quantizer, d, nlist)
    
    if device == "cuda":
        res = faiss.StandardGpuResources()
        faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)
        
    print("Training FAISS index...")
    faiss_index.train(gauss_pos_np)
    
    print("Adding data to FAISS index...")
    faiss_index.add(gauss_pos_np)
    
    faiss_index.nprobe = 10 
    print(f"✅ FAISS index built successfully. nprobe set to {faiss_index.nprobe}.")

    embed_fn, input_ch = get_embedder(POS_ENCODE_FREQS)
    embeddirs_fn, input_ch_views = get_embedder(VIEW_ENCODE_FREQS) if USE_VIEWDIRS else (None, 0)
    model = NeRFMLP(D=MLP_DEPTH, W=MLP_WIDTH, input_ch=input_ch, input_ch_views=input_ch_views, use_viewdirs=USE_VIEWDIRS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.L1Loss()
    
    # --- CHANGE 1: Add Cosine Annealing LR Scheduler ---
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_TRAINING_ITERATIONS)
    print("✅ Log-space sigma loss and Cosine Annealing LR Scheduler enabled.")


    print("\n🚀 Starting NeRF-like training from 3DGS ground truth...")
    pbar = tqdm(range(NUM_TRAINING_ITERATIONS), desc="Training MLP", ncols=100)
    for i in pbar:
        c2w = random.choice(all_c2w_matrices)
        rays_o_full, rays_d_full = get_rays(H, W, fx, fy, c2w)
        
        pix_x = torch.randint(0, int(W), (N_RAYS_PER_BATCH,), device=device)
        pix_y = torch.randint(0, int(H), (N_RAYS_PER_BATCH,), device=device)
        rays_o, rays_d = rays_o_full[pix_y, pix_x], rays_d_full[pix_y, pix_x]

        current_camera_pos = c2w[:3, 3]
        near, far = get_dynamic_near_far(current_camera_pos, gaussians_data['pos'])
        
        pts_sampled = sample_points_along_rays(rays_o, rays_d, N_SAMPLES_PER_RAY, near, far, perturb=PERTURB_SAMPLES)
        pts_flat = pts_sampled.reshape(-1, 3)
        viewdirs_flat = F.normalize(rays_d.unsqueeze(1).expand_as(pts_sampled), dim=-1).reshape(-1, 3)

        with torch.no_grad():
            raw_gt = get_3dgs_ground_truth_raw(pts_flat, viewdirs_flat, gaussians_data, sh_degree, faiss_index, chunk=NN_QUERY_CHUNK, k_neighbors=K_NEAREST_GAUSSIANS)

        mlp_input = torch.cat([embed_fn(pts_flat), embeddirs_fn(viewdirs_flat)], -1) if USE_VIEWDIRS else embed_fn(pts_flat)
        predicted_raw = model(mlp_input)
        
        predicted_color = predicted_raw[..., :3]
        predicted_sigma = F.softplus(predicted_raw[..., 3])

        gt_color = raw_gt[..., :3]
        gt_sigma = raw_gt[..., 3]
        
        color_loss = loss_fn(predicted_color, gt_color)
        
        # --- CHANGE 2: Implement the Log-Density Trick for Sigma Loss ---
        epsilon = 1e-6 # Add a small value to prevent log(0)
        sigma_loss = loss_fn(torch.log(predicted_sigma + epsilon), torch.log(gt_sigma + epsilon))
        sparsity = 1e-4
        sparsity_loss = torch.mean(predicted_sigma)
        
        # Keep the loss weighting
        total_loss = (10 * color_loss) + sigma_loss + (sparsity * sparsity_loss)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # --- CHANGE 3: Step the scheduler ---
        scheduler.step()
        
        pbar.set_postfix(loss=f"{total_loss.item():.4f}", color=f"{color_loss.item():.4f}", sigma=f"{sigma_loss.item():.4f}", sparsity=f"{sparsity_loss.item():.4f}")

        if (i + 1) % 1000 == 0: 
            print(f"\n--- Iteration {i+1}/{NUM_TRAINING_ITERATIONS} ---", flush=True)
            print(f"Losses -> Total: {total_loss.item():.4f}, Color: {color_loss.item():.4f}, Sigma: {sigma_loss.item():.4f}", flush=True)

    print("\n✅ Training complete.")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_model_path = os.path.join(OUTPUT_DIR, MODEL_FILENAME)
    torch.save({'model_state_dict': model.state_dict()}, output_model_path)
    print(f"Trained model checkpoint saved to '{output_model_path}'")

if __name__ == "__main__":
    train_nerf_from_3dgs()
