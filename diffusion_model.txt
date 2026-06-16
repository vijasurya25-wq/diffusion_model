Algorithm: Denoising Diffusion Probabilistic Model (DDPM) with DDIM sampling for MRI-to-CT synthesis — Task 1 Brain.
Architecture: Conditional U-Net (base_dim=64, time_dim=256) with sinusoidal time embeddings and cross-attention conditioning on the input MRI. The model predicts the clean image x₀ directly from noisy input at each denoising step.
Training:

Dataset: SynthRAD2023 Task 1 Brain — 180 paired MRI/CT cases
Epochs: 150
Optimizer: Adam (lr=1e-4)
Loss: MSE noise prediction + 0.3 × L1 x₀ prediction
Batch size: 4 (2D axial slices, 256×256)
Hardware: NVIDIA A100-PCIE-40GB

Inference:

DDIM sampling with 50 steps (accelerated from T=1000 training steps)
EMA (Exponential Moving Average) weights used for inference

Preprocessing:

Resampled to 1mm isotropic resolution
MRI registered to CT space
Slices normalised to [-1, 1]
CT clipped to [-1000, 2000] HU

Results on training set (SynthRAD2023-style evaluation, within body mask):

SSIM: 0.7830
PSNR: 38.26 dB
MAE: 0.0086 (normalised)
Dice (Mean): 0.7805
