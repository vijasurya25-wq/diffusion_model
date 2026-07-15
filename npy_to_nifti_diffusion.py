"""
npy_to_nifti_diffusion.py — Convert Diffusion Model evaluate.py inferred CT
                             .npy slices to NIfTI volumes for viewing in skull_3d.m

Works with the CycleGAN-style evaluate.py generated for the diffusion model,
which saves individual .npy slices to eval_results/epoch_N/inferred_ct/npy/
(identical output structure to CycleGAN evaluate.py).

NOTE: This does NOT work with the ORIGINAL diffusion evaluate.py which saves
      only final_results.csv and per_slice_metrics.csv — not individual slices.
      Use the CycleGAN-style diffusion evaluate.py to generate the .npy files first.

How evaluate.py saves files:
  - Synthetic CT: eval_results/epoch_N/inferred_ct/npy/{case_id}_{slice}.npy
                  Each .npy is ONE slice of synthetic CT only
                  Values are in [-1, 1] normalised range
  - MRI:          processed/brain/val/mri/{case_id}_{slice}.npy
                  Each .npy is ONE slice of MRI
                  Values are in [-1, 1] normalised range

This script (identical to CycleGAN npy_to_nifti.py — only paths differ):
  1. Groups slices by case_id
  2. Sorts slices using NATURAL SORT (avoids distorted skull from wrong order)
  3. Stacks them into 3D volumes
  4. Denormalizes synthetic CT from [-1,1] to HU values
  5. Saves as .nii.gz for skull_3d.m

Usage:
    python npy_to_nifti_diffusion.py --epoch 1
    python npy_to_nifti_diffusion.py --epoch 1 --case 1BB001
    python npy_to_nifti_diffusion.py --epoch 1 --all

After running — update skull_3d.m CONFIG:
    USE_DIRECT_PATH = true;
    CT_PATH = '.../nifti_output/epoch_1/1BB001_synthetic_ct.nii.gz';
    MR_PATH = '.../nifti_output/epoch_1/1BB001_mri.nii.gz';
"""

import os
import re
import argparse
import numpy as np
import nibabel as nib
from collections import defaultdict
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — must match paths in diffusion model evaluate.py
# ─────────────────────────────────────────────────────────────────────────────

# Where diffusion evaluate.py --mode infer saved synthetic CT slices
# (same structure as CycleGAN: eval_results/epoch_N/inferred_ct/npy/)
EVAL_RESULTS_DIR = "/media/rvcse22/CSERV/SVARA/diffusion_model/eval_results"

# Where preprocessed val MRI slices are stored
# (same val set as CycleGAN — shared preprocessing)
VAL_MRI_DIR      = "/media/rvcse22/CSERV/SVARA/processed/brain/val/mri"

# Where NIfTI output will be saved
OUTPUT_DIR       = "/media/rvcse22/CSERV/SVARA/diffusion_model/nifti_output"

# CT HU range used in preprocess.py — needed to reverse normalisation
# Must match the formula in diffusion evaluate.py:
#   fake_hu = fake_01 * 3000 - 1000  (CT_HU_MIN=-1000, CT_HU_MAX=2000)
CT_HU_MIN = -1000
CT_HU_MAX =  2000

# Voxel spacing in mm — SynthRAD brain is 1mm isotropic
VOXEL_SIZE = (1.0, 1.0, 1.0)

# ─────────────────────────────────────────────────────────────────────────────


def natural_sort_key(filename):
    """
    Sort key for natural (numeric) sorting of filenames.
    Ensures _2 comes before _10 (not alphabetically after _19).

    Example:
        Alphabetical: 1BB001_1.npy, 1BB001_10.npy, 1BB001_2.npy  ← WRONG
        Natural:      1BB001_1.npy, 1BB001_2.npy,  1BB001_10.npy  ← CORRECT

    This is the ROOT CAUSE of the distorted skull — wrong slice order.
    """
    parts = re.split(r"(\d+)", filename)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def get_case_ids(npy_dir):
    """
    Get all unique patient case IDs from a folder of .npy files.
    Files are named like: 1BB001_0001.npy, 1BB001_0002.npy etc.
    Case ID is the part before the first underscore.
    """
    files = sorted([f for f in os.listdir(npy_dir) if f.endswith(".npy")],
                   key=natural_sort_key)
    if not files:
        raise FileNotFoundError(f"No .npy files found in {npy_dir}")
    case_ids = sorted(set(f.split("_")[0] for f in files))
    return case_ids


def get_slices_for_case(npy_dir, case_id):
    """
    Get all slice files for a specific case, sorted by slice index
    using NATURAL SORT (numeric order not alphabetical order).

    IMPORTANT: Python sorted() sorts alphabetically by default:
        _1.npy, _10.npy, _100.npy, _2.npy  ← alphabetical (WRONG)
        _1.npy, _2.npy, _10.npy, _100.npy  ← natural (CORRECT)

    Without natural sort, the skull is distorted because slices
    are stacked in wrong anatomical order.
    """
    files = sorted(
        [f for f in os.listdir(npy_dir)
         if f.startswith(case_id + "_") and f.endswith(".npy")],
        key=natural_sort_key   # ← KEY FIX: natural sort not alphabetical
    )
    if not files:
        raise FileNotFoundError(
            f"No slices found for case '{case_id}' in:\n  {npy_dir}"
        )
    print(f"    Slice order: {files[0]} → ... → {files[-1]} ({len(files)} slices)")
    return [os.path.join(npy_dir, f) for f in files]


def stack_volume(slice_paths):
    """
    Load all slices and stack into a 3D volume.
    Each slice is (H, W) → stacked to (H, W, N_slices).
    """
    slices = []
    for path in slice_paths:
        arr = np.load(path).astype(np.float32)
        # Handle case where slice might be 3D with extra dim
        if arr.ndim == 3:
            arr = arr[0]   # take first channel if (1, H, W)
        slices.append(arr)
    volume = np.stack(slices, axis=2)   # (H, W, N_slices)
    return volume


def denormalize_ct(arr_norm):
    """
    Reverse [-1, 1] normalisation back to Hounsfield Units (HU).

    evaluate.py saves synthetic CT in [-1, 1] range.
    skull_3d.m needs HU values for bone thresholding (HU_MIN=150, HU_MAX=1900).

    Formula (inverse of preprocess.py):
        arr_01 = (arr_norm + 1) / 2
        arr_hu = arr_01 * (CT_HU_MAX - CT_HU_MIN) + CT_HU_MIN
    """
    arr_01 = (arr_norm + 1.0) / 2.0
    arr_hu = arr_01 * (CT_HU_MAX - CT_HU_MIN) + CT_HU_MIN
    return arr_hu.astype(np.float32)


def denormalize_mri(arr_norm):
    """
    Reverse [-1, 1] normalisation for MRI to [0, 1] range.
    skull_3d.m normalises MRI internally so [0,1] is fine.
    """
    arr_01 = (arr_norm + 1.0) / 2.0
    return np.clip(arr_01, 0.0, 1.0).astype(np.float32)


def save_nifti(volume, out_path, voxel_size=VOXEL_SIZE):
    """Save numpy volume as NIfTI with correct voxel spacing."""
    affine          = np.diag([voxel_size[0], voxel_size[1], voxel_size[2], 1.0])
    img             = nib.Nifti1Image(volume, affine)
    img.header.set_zooms(voxel_size)
    nib.save(img, out_path)


def convert_case(case_id, inferred_ct_dir, epoch_out_dir):
    """
    Convert one patient case to NIfTI:
      1. Load synthetic CT slices → denormalize to HU → save as NIfTI
      2. Load val MRI slices → denormalize to [0,1] → save as NIfTI

    Returns True if successful, False if failed.
    """
    print(f"\n  Processing case: {case_id}")

    # ── Synthetic CT ──────────────────────────────────────────────────────────
    try:
        ct_paths  = get_slices_for_case(inferred_ct_dir, case_id)
        ct_norm   = stack_volume(ct_paths)
        ct_hu     = denormalize_ct(ct_norm)
        ct_out    = os.path.join(epoch_out_dir,
                                 f"{case_id}_synthetic_ct.nii.gz")
        save_nifti(ct_hu, ct_out)
        print(f"    Synthetic CT : {ct_norm.shape}  "
              f"HU range: [{ct_hu.min():.0f}, {ct_hu.max():.0f}]")
        print(f"    Saved        : {ct_out}")
    except Exception as e:
        print(f"    [ERROR] Synthetic CT failed: {e}")
        return False

    # ── MRI ───────────────────────────────────────────────────────────────────
    try:
        mri_paths = get_slices_for_case(VAL_MRI_DIR, case_id)
        mri_norm  = stack_volume(mri_paths)
        mri_01    = denormalize_mri(mri_norm)
        mri_out   = os.path.join(epoch_out_dir,
                                  f"{case_id}_mri.nii.gz")
        save_nifti(mri_01, mri_out)
        print(f"    MRI          : {mri_norm.shape}")
        print(f"    Saved        : {mri_out}")
    except Exception as e:
        print(f"    [WARN] MRI failed: {e}")
        print(f"           Synthetic CT saved — MRI panel will be empty in skull_3d.m")

    return True


def print_matlab_instructions(success_cases, epoch_out_dir):
    """Print exact MATLAB code to paste into skull_3d.m"""
    print(f"\n{'='*65}")
    print(f" Copy these paths into skull_3d.m CONFIG section")
    print(f"{'='*65}\n")

    for case_id in success_cases[:5]:
        ct_win = os.path.join(epoch_out_dir,
                              f"{case_id}_synthetic_ct.nii.gz").replace("/", "\\")
        mr_win = os.path.join(epoch_out_dir,
                              f"{case_id}_mri.nii.gz").replace("/", "\\")
        print(f"  %% Case {case_id}:")
        print(f"  ct_path = '{ct_win}';")
        print(f"  mr_path = '{mr_win}';")
        print()

    if len(success_cases) > 5:
        print(f"  ... and {len(success_cases)-5} more cases in:")
        print(f"  {epoch_out_dir}/\n")

    print(f"{'='*65}")
    print(f" skull_3d.m also needs these lines changed in STEP 1:")
    print(f"{'='*65}")
    print(f"""
  Replace the existing STEP 1 loading code with:

  fprintf('[1/6] Loading CT volume...\\n');
  [ct_volume, voxel_size] = load_nifti_robust(ct_path);
  ct_volume = double(ct_volume);

  mr_volume = [];
  if exist(mr_path, 'file')
      [mr_raw, ~] = load_nifti_robust(mr_path);
      mr_raw = double(mr_raw);
      mn = min(mr_raw(:)); mx = max(mr_raw(:));
      if mx > mn; mr_volume = (mr_raw - mn) / (mx - mn); end
  end
""")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert Diffusion Model evaluate.py inferred CT .npy slices\n"
            "to NIfTI (.nii.gz) for viewing synthetic CT + MRI in skull_3d.m.\n"
            "Identical to CycleGAN npy_to_nifti.py — only paths differ."
        )
    )
    parser.add_argument(
        "--epoch", type=str, required=True,
        help="Epoch number matching evaluate.py output (e.g. 160)"
    )
    parser.add_argument(
        "--case", type=str, default=None,
        help="Convert single case only (e.g. 1BB001). Omit to convert all."
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Convert all cases found in inferred CT folder"
    )
    args = parser.parse_args()

    # ── Paths ─────────────────────────────────────────────────────────────────
    inferred_ct_dir = os.path.join(
        EVAL_RESULTS_DIR, f"epoch_{args.epoch}", "inferred_ct", "npy"
    )
    epoch_out_dir   = os.path.join(OUTPUT_DIR, f"epoch_{args.epoch}")

    # Validate
    if not os.path.exists(inferred_ct_dir):
        raise FileNotFoundError(
            f"\nInferred CT folder not found:\n  {inferred_ct_dir}\n\n"
            f"Run the diffusion evaluate.py first:\n"
            f"  python evaluate.py --mode infer --checkpoint latest\n\n"
            f"Note: Use the CycleGAN-style evaluate.py for the diffusion model,\n"
            f"      NOT the original evaluate.py (which does not save individual slices)."
        )
    if not os.path.exists(VAL_MRI_DIR):
        print(f"[WARN] Val MRI dir not found: {VAL_MRI_DIR}")
        print(f"       Only synthetic CT will be converted — no MRI NIfTI")

    os.makedirs(epoch_out_dir, exist_ok=True)

    # ── Print header ──────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f" NPY → NIfTI Conversion  |  Epoch {args.epoch}")
    print(f"{'='*65}")
    print(f"  Inferred CT : {inferred_ct_dir}")
    print(f"  Val MRI     : {VAL_MRI_DIR}")
    print(f"  Output      : {epoch_out_dir}")
    print(f"{'='*65}")

    # ── Get case IDs ──────────────────────────────────────────────────────────
    if args.case:
        case_ids = [args.case]
        print(f"\n  Converting single case: {args.case}")
    else:
        case_ids = get_case_ids(inferred_ct_dir)
        print(f"\n  Found {len(case_ids)} cases: {case_ids}")

    # ── Convert ───────────────────────────────────────────────────────────────
    success = []
    for case_id in tqdm(case_ids, desc="Converting"):
        ok = convert_case(case_id, inferred_ct_dir, epoch_out_dir)
        if ok:
            success.append(case_id)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f" Done — {len(success)}/{len(case_ids)} cases converted")
    print(f"{'='*65}")
    print(f"\n  Output files per case:")
    print(f"    {{case_id}}_synthetic_ct.nii.gz  ← load as ct_path in skull_3d.m")
    print(f"    {{case_id}}_mri.nii.gz           ← load as mr_path in skull_3d.m")

    if success:
        print_matlab_instructions(success, epoch_out_dir)


if __name__ == "__main__":
    main()
