# Bridging Explicit and Implicit 3D Representations: From 3DGS to NeRF for Multi-Robot SLAM

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Nerfstudio](https://img.shields.io/badge/Framework-Nerfstudio-yellow)
![Status](https://img.shields.io/badge/Status-Active_Research-success)

## 📖 About The Project

Welcome to the central hub for my MSc Robotics and Artificial Intelligence project. This repository documents my systematic exploration of converting **3D Gaussian Splatting (3DGS)** into **Neural Radiance Fields (NeRF)**. 

**The Motivation:** While 3DGS offers state-of-the-art tracking and rendering speeds, its map sizes are massive (often >100MB per scene), creating a severe bandwidth bottleneck for multi-robot collaborative SLAM. NeRF, conversely, provides a highly compact continuous density field (~10MB) ideal for transmission. This project aims to bridge the two representations, combining the best of both worlds.

This repository serves as my **Research Showcase**, highlighting the core concepts, insights, and engineering bottlenecks discovered during my 2-phase exploration.

*(Note: For the full executable framework, please visit my heavily modified [Integration_3DGS2NeRF_nerfstudio Repository](https://github.com/zcemcui/Integration_3DGS2NeRF_nerfstudio/tree/phase2-online#)).*

---

## 🚀 The 2-Phase Exploration & Core Findings

### ❌ Phase 1: Direct 3D Conversion (The Mathematical Illusion)
Initially, I hypothesized that explicit ellipsoids could be directly mapped to an implicit continuous field without costly 2D rendering. I implemented a FAISS-accelerated KNN search to fuse 3DGS parameters directly into a NeRF MLP in 3D spatial coordinates.

* **What went wrong:** The conversion resulted in a blurry, bounded box-like shadow.
* **The Insight:** Explicit Gaussian parameters (especially the quaternion rotation matrices) cannot be trivially mapped to scalar volume density ($\sigma$). The mathematical dimensions of 3DGS opacity ($\alpha$) and NeRF volume density ($\sigma$) are fundamentally mismatched in 3D space.
* **Core Code:** [`/Phase1_Direct_Conversion`](./Phase1_Direct_Conversion)

---

### 🚧 Phase 2: Online Knowledge Distillation (The Engineering Bottleneck)
To bypass the 3D mathematical conflict, I pivoted to **2D Image-Level Distillation** within the Nerfstudio framework. The strategy was to dynamically render 2D pseudo-ground-truth images from a frozen 3DGS "Teacher" to supervise a NeRF "Student" in real-time.

<div align="center">
  <table>
    <tr>
      <td align="center"><b>NeRF Student (Predicted)</b></td>
      <td align="center"><b>3DGS Teacher (Ground Truth)</b></td>
    </tr>
    <tr>
      <td width="50%"><img src="https://github.com/user-attachments/assets/a97ae01a-cf50-4afb-8260-03a058f6f0f4" alt="NeRF Student"></td>
      <td width="50%"><img src="https://github.com/user-attachments/assets/6c0f544c-3583-412e-b5ba-354f702c74f6" alt="3DGS Teacher"></td>
    </tr>
  </table>
  <p><em>Fig 1. Qualitative comparison: The NeRF student (left) exhibits geometric blurriness compared to the sharp 3DGS teacher (right).</em></p>
</div>

* **The Result:** The model successfully converged quantitatively (PSNR > 25), but the 3D geometry suffered from noticeable blurriness and lacked high-frequency details.
* **The Insight:** We hit a hard hardware VRAM limit. 3DGS requires rendering full images, while standard NeRF relies on sampling random rays globally across multiple cameras to triangulate sharp geometry. To prevent Out-Of-Memory (OOM) crashes, we were forced to compromise and sample rays from only **one camera per batch**. This single-view training induced micro-scale catastrophic forgetting, preventing the NeRF from locking in a globally consistent, sharp 3D surface.
* **Core Code:** [`/Phase2_Online_Distillation`](./Phase2_Online_Distillation)

---

## 🔮 Future Work
The VRAM limitations of online rasterization-raytracing synchronization make sharp 3D reconstruction unfeasible on standard hardware. The logical next step for this research is to pivot to **Offline Cached Distillation** combined with **Depth Supervision**. By pre-rendering and caching high-precision RGB and Depth maps from the 3DGS teacher, we could bypass the online VRAM bottleneck, restore global multi-view ray sampling, and strictly constrain the NeRF geometry.

---

## 📁 Repository Navigation

* **`Phase1_Direct_Conversion/`**: Contains the FAISS-based KNN 3D spatial mapping scripts.
* **`Phase2_Online_Distillation/`**: Contains the core `gs_datamanager.py` and `gs_distill_pipeline.py` implementations for online 2D supervision.

## 🔗 External Links
For full reproducibility and access to the complete modified framework:
* 🖥️ **[Integration_3DGS2NeRF_nerfstudio (Phase 2 Online)](https://github.com/zcemcui/Integration_3DGS2NeRF_nerfstudio/tree/phase2-online#)**

---
*This project is conducted as part of the MSc Robotics and Artificial Intelligence program.*
