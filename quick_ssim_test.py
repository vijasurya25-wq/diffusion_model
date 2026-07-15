import numpy as np, torch, sys, os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
sys.path.append('/media/rvcse22/CSERV/SVARA/diffusion_model')
from scripts.unet import ConditionalUNet
from scripts.diffusion import DDPM

DEVICE = 'cuda:0'
model  = ConditionalUNet(base_dim=64, time_dim=256).to(DEVICE)
ckpt   = torch.load('/media/rvcse22/CSERV/SVARA/diffusion_model/checkpoints/latest.pth',
                    map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['ema'])
model.eval()
ddpm = DDPM(T=1000, device=DEVICE)

MRI_DIR = '/media/rvcse22/CSERV/SVARA/processed/brain/train/mri'
CT_DIR  = '/media/rvcse22/CSERV/SVARA/processed/brain/train/ct'
files   = sorted([f for f in os.listdir(MRI_DIR) if f.startswith('1BA001')])[:30]

for steps in [100, 200, 500, 1000]:
    ssims, psnrs, maes = [], [], []
    for f in files:
        mri = np.load(os.path.join(MRI_DIR, f)).astype(np.float32)
        ct  = np.load(os.path.join(CT_DIR,  f)).astype(np.float32)
        mri_t = torch.from_numpy(mri).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            fake = ddpm.sample(model, mri_t, steps=steps)
        fake_hu = (fake[0,0].cpu().numpy() + 1) / 2 * 3000 - 1000
        real_hu = (ct + 1) / 2 * 3000 - 1000
        dr = real_hu.max() - real_hu.min()
        rn = (real_hu - real_hu.min()) / (dr + 1e-8)
        fn = np.clip((fake_hu - real_hu.min()) / (dr + 1e-8), 0, 1)
        ssims.append(ssim(rn, fn, data_range=1.0))
        psnrs.append(psnr(rn, fn, data_range=1.0))
        maes.append(np.mean(np.abs(real_hu - fake_hu)))
    print(f'Steps={steps:4d} | SSIM={np.mean(ssims):.4f} | PSNR={np.mean(psnrs):.2f}dB | MAE={np.mean(maes):.1f}HU')
