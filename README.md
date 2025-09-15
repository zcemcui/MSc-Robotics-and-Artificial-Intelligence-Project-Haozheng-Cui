
# GS2NeRF: Direct Conversion from 3D Gaussian Splatting to NeRF

This repository contains the source code for the MSc Robotics and Artificial Intelligence project by Haozheng Cui, titled "GS2NeRF: Direct Conversion from 3D Gaussian Splatting to NeRF."

The project proposes a novel one-step method to convert a 3D Gaussian Splatting (3DGS) scene directly into a Neural Radiance Field (NeRF) representation. This work provides an efficient bridge between these two leading 3D scene reconstruction methods.

## Project Overview

In a heterogeneous robotic system where different 3D reconstruction methods are employed, the ability to convert between them is crucial. The current methods often rely on a two-step process that requires pre-rendered images. This project presents a new approach that distills the radiance field by training a lightweight MLP, similar to Instant-NGP, from a tensor storing pixel information.

## Methods

This project implements two core methods based on the referenced articles to achieve the conversion:

-   **Method 1 :** A novel direct conversion approach that skips the intermediate image rendering step.
-   **Method 2 :** An efficient training pipeline that leverages a tensor-based data representation to accelerate the distillation process. This method trains an Instant-NGP-like MLP for high-quality radiance field reconstruction.


### Training

To train the model using **Method 2**, simply run the main training script. This script will handle the data loading, model initialization, and training loop.

```bash
python Method2_train.py
