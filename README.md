# Bridging Explicit and Implicit 3D Representations: From 3DGS to NeRF for Multi-Robot SLAM

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Nerfstudio](https://img.shields.io/badge/Framework-Nerfstudio-yellow)
![Status](https://img.shields.io/badge/Status-Active_Research-success)

## 📖 About The Project

Welcome to the central hub for my MSc Robotics and Artificial Intelligence project. This repository documents my systematic exploration of converting **3D Gaussian Splatting (3DGS)** into **Neural Radiance Fields (NeRF)**. 

**The Motivation:** While 3DGS offers state-of-the-art tracking and rendering speeds, its map sizes are massive (often >100MB per scene), creating a severe bandwidth bottleneck for multi-robot collaborative SLAM. NeRF, conversely, provides a highly compact continuous density field (~10MB) ideal for transmission. This project aims to bridge the two representations, combining the best of both worlds.

This repository serves as my **Research Showcase**. It highlights the core scripts, mathematical reflections, and visual results of my 2-phase exploration.

*(Note: For the full executable framework, please visit my heavily modified [Integration_3DGS2NeRF_nerfstudio Repository](https://github.com/zcemcui/Integration_3DGS2NeRF_nerfstudio/tree/phase2-online#)).*

---

## 🚀 The 2-Phase Journey & Core Findings

### ❌ Phase 1: Direct 3D Conversion (The Mathematical Illusion)
Initially, I hypothesized that explicit ellipsoids could be directly mapped to an implicit continuous field without costly 2D rendering. I implemented a FAISS-accelerated KNN search to fuse 3DGS parameters directly into a NeRF MLP in 3D spatial coordinates.

* **What went wrong:** The conversion resulted in a blurry, bounded box-like shadow.
* **The Insight:** Explicit Gaussian parameters (especially the quaternion rotation matrices) cannot be trivially mapped to scalar volume density ($\sigma$). The mathematical dimensions of 3DGS opacity ($\alpha$) and NeRF volume density ($\sigma$) are fundamentally mismatched in 3D space.
* **Core Code:** [`/Phase1_Direct_Conversion`](./Phase1_Direct_Conversion)

---

### 🚧 Phase 2: 2D Image-Level Distillation (The Engineering Pivot)
To bypass the 3D mathematical conflict, I pivoted to **2D Image-Level Distillation**. The strategy was to let a frozen 3DGS "Teacher" render 2D images to supervise a NeRF "Student".

* **Online Distillation (via Nerfstudio):** Attempted real-time rendering supervision. **Result:** Catastrophic VRAM consumption. The architectural clash between 3DGS rasterization and NeRF ray-tracing in the same training loop caused severe memory bottlenecks.
* **Offline Cached Distillation (via Instant-NGP):** Implemented a `build_gpu_cache` to pre-render 3DGS outputs, solving the VRAM issue. **Result:** Achieved extremely fast training, but the low-capacity Instant-NGP model lacked geometric constraints, resulting in "floaters".
* **Core Code:** [`/Phase2_Online_Distillation`](./Phase2_Online_Distillation)

<p align="center">
  <img src="[在这里填入Phase2效果一般或者报错的图片链接]" alt="Phase 2 Floaters or Bottleneck" width="60%">
  <br>
  <em>Fig 1. Offline NGP distillation resolved memory issues but introduced floaters due to unconstrained geometry.</em>
</p>

---

## 📁 Repository Navigation

* **`Phase1_Direct_Conversion/`**: Contains the FAISS-based KNN 3D spatial mapping scripts.
* **`Phase2_Online_Distillation/`**: Contains the code for real-time and cached 2D distillation trials.

## 🔗 External Links
For full reproducibility and access to the complete modified framework:
* 🖥️ **[Integration_3DGS2NeRF_nerfstudio (Phase 2 Online)](https://github.com/zcemcui/Integration_3DGS2NeRF_nerfstudio/tree/phase2-online#)**

---
*This project is conducted as part of the MSc Robotics and Artificial Intelligence program.*
