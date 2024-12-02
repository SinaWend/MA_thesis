# MA_thesis
---

# Towards Improving Generalization Performance of Vision Transformers on Microscopic Data

## Overview

This repository contains code and resources for my Master's thesis on improving the generalization of Vision Transformers (ViTs), specifically DINOv2, for microscopic imaging tasks in histopathology and hematology. I explore domain generalization techniques and benchmark ViTs against traditional convolutional models like ResNet.

---

## Data

- **CAMELYON17-WILDS**: Histological image patches for cancer detection.  
- **White Blood Cell Datasets**: Harmonized datasets combining Matek-19, INT-20, and Acevedo-20.  
- Data augmentation techniques like cropping, flipping, and jittering were applied to address imbalances.

---

## Key Methods

- Fine-tuned **DINOv2 Vision Transformer** on medical imaging data.  
- Implemented domain generalization techniques (e.g., DANN, IRM).  
- Benchmarked against ResNet-50 for comparative analysis.

---

## Results

- **DINOv2** demonstrated strong baseline performance.  
- Domain generalization methods provided limited or no improvement.  
- Findings suggest task-specific adaptations are critical for medical imaging.

---

## Acknowledgments

Thanks to **Prof. Dr. Dr. Fabian Theis**, **Dr. Carsten Marr**, and **Dr. Xudong Sun** for their guidance throughout this project.
