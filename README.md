**DynaQ: Content-Aware Image Compression System**

**Overview**

This project presents a content-adaptive image compression system that combines classical Discrete Cosine Transform (DCT) with a lightweight neural network to dynamically optimize compression quality.

Unlike standard JPEG compression (fixed quality), this system:

Analyzes image complexity
Predicts optimal compression level
Balances file size vs visual quality

**Key Features**
Block-based DCT compression (JPEG model)
Feature-based image complexity analysis
Custom 2-layer neural network (built from scratch)
Adaptive quantization parameter prediction
Evaluation using PSNR, SSIM, compression ratio

**System Pipeline**
Image Preprocessing
Feature Extraction
Spatial Variance
High-Frequency Energy
Edge Strength
Neural Network Prediction
DCT-based Compression
Performance Evaluation

**Results**
Compression Level	PSNR	SSIM	Compression Ratio
Low (Q=2)	>38 dB	>0.96	3×–6×
Medium (Q=8)	32–36 dB	>0.88	~6×–10×
High (Q=20)	Lower	Visible loss	10×–20×
Adaptive (AI)	Balanced	High	Optimized

<img width="940" height="500" alt="image" src="https://github.com/user-attachments/assets/f1c474c2-ef20-4ff8-b65f-dc89240763e5" />

<img width="940" height="501" alt="image" src="https://github.com/user-attachments/assets/3bec7ca0-304d-4101-b299-9e6a70fa7a48" />

<img width="940" height="501" alt="image" src="https://github.com/user-attachments/assets/30556674-4e6b-4937-9282-048067e0ff19" />





**Evaluation Metrics**
PSNR (Peak Signal-to-Noise Ratio)
SSIM (Structural Similarity Index)
File Size & Compression Ratio

**Innovation**
Combines classical DSP + machine learning
Uses lightweight model (no heavy frameworks)
Demonstrates real-world trade-off optimization

**Tech Stack**
MATLAB
Image Processing Toolbox
Custom Neural Network (no DL toolbox)

**Future Scope**
CNN-based feature learning
Region-wise adaptive compression
Real-time embedded system deployment
Integration with IoT/edge devices
📌 Applications
Smart devices & IoT imaging
Medical imaging
Satellite communication
Web image optimization
