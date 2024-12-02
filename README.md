# MA_thesis
---

# Towards Improving Generalization Performance of Vision Transformers on Microscopic Data

## Overview

This repository contains code and resources for my Master's thesis on improving the generalization of Vision Transformers (ViTs), specifically DINOv2, for microscopic imaging tasks in histopathology and hematology. I explore domain generalization techniques and benchmark ViTs against traditional convolutional models like ResNet.

---

## Data

- **CAMELYON17-WILDS**: Histological image patches for cancer detection.  
- **White Blood Cell Datasets**: Harmonized datasets combining Matek-19, INT-20, and Acevedo-20.
- **Own CAMELYON17 Patch Dataset: Patches for cancer detection were extracted from the whole slide images:
  <img width="488" alt="Bildschirm­foto 2024-12-02 um 16 37 53" src="https://github.com/user-attachments/assets/8f339c2e-8c49-4433-a48e-1c1c2ff8b37b">

- Data augmentation techniques like cropping, flipping, and jittering were applied to address imbalances.

---

## Key Methods

- Fine-tuned **DINOv2 Vision Transformer** on medical imaging data.  
- Implemented domain generalization techniques (e.g., DANN, IRM).  
- Benchmarked against ResNet-50 for comparative analysis.

---

## Results

- **DINOv2** demonstrated strong baseline performance for both, blood and histophathology data.  
- Domain generalization methods provided limited or no improvement.  
- Findings suggest task-specific adaptations are critical for medical imaging.

---

## Acknowledgments

Thanks to **Prof. Dr. Dr. Fabian Theis**, **Dr. Carsten Marr**, and **Dr. Xudong Sun** for their guidance throughout this project.
