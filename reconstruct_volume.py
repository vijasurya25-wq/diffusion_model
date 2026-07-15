import numpy as np
import nibabel as nib
import os
import torch
import sys

sys.path.append('/media/rvcse22/CSERV/SVARA/diffusion_model')
from scripts.unet import ConditionalUNet
from scripts.diffusion import DDPM

DEVICE   = 'cuda:0'
BASE_DIR = '/media/rvcse22/CSERV/SVARA/diffusion_model'
CASE_ID  = '1BA001'  # change to any case in your val set
MRI_DIR  = '/media/rvcse22/CSERV/SVARA/processed/brain/train/mri'
CT_DIR   = '/media/rvcse22/CSERV/SVARA/processed/brain/train/ct'
OUT_DIR  = os.path.join(BASE_DIR, 'nifti_output')
os.makedirs(OUT_DIR, exist_ok=True)

# Load model
model = ConditionalUNet(base_dim=64, time_dim=256).to(DEVICE)
ckpt  = torch.load(f'{BASE_DIR}/checkpoints/latest.pth',
                   map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['ema'])
model.eval()
ddpm = DDPM(T=1000, device=DEVICE)

# Gather slices for this case
files = sorted([f for f in os.listdir(MRI_DIR) if f.startswith(CASE_ID)])
print(f'Found {len(files)} slices for {CASE_ID}')

fake_slices = []
real_slices = []

for f in files:
    mri = np.load(os.path.join(MRI_DIR, f)).astype(np.float32)
    ct  = np.load(os.path.join(CT_DIR,  f)).astype(np.float32) if os.path.exists(os.path.join(CT_DIR, f)) else None

    mri_tensor = torch.from_numpy(mri).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        fake_ct = ddpm.sample(model, mri_tensor, steps=100)

    # Convert [-1,1] to HU
    fake_hu = ((fake_ct[0,0].cpu().numpy() + 1) / 2 * 3000 - 1000)
    fake_slices.append(fake_hu)

    if ct is not None:
        real_hu = (ct + 1) / 2 * 3000 - 1000
        real_slices.append(real_hu)

# Stack into 3D volume
fake_vol = np.stack(fake_slices, axis=2).astype(np.float32)
print(f'Synthetic CT volume shape: {fake_vol.shape}')

# Save as NIfTI
affine = np.eye(4)
nib.save(nib.Nifti1Image(fake_vol, affine),
         os.path.join(OUT_DIR, f'{CASE_ID}_synthetic_ct.nii.gz'))

if real_slices:
    real_vol = np.stack(real_slices, axis=2).astype(np.float32)
    nib.save(nib.Nifti1Image(real_vol, affine),
             os.path.join(OUT_DIR, f'{CASE_ID}_real_ct.nii.gz'))
    print(f'Real CT volume shape: {real_vol.shape}')

print(f'Saved NIfTI volumes to {OUT_DIR}')
