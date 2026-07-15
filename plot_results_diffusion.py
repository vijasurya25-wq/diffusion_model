"""
plot_results.py — Publication-quality graphs for Diffusion Model Brain MRI->CT project.

Reads:
  eval_results/epoch_N/metrics_per_slice.csv     (per-epoch slice metrics)
  eval_results/all_epochs_summary.csv            (epoch-wise training progress)
  metrics_log.csv                                (training losses — learning curve)
  dice_results/all_epochs_dice_summary.csv       (Dice scores — optional)

Outputs (saved to eval_results/plots/):
  ── Per-slice metric plots (from best epoch) ──────────────────────────────
  01. ssim_distribution.png       — SSIM histogram with quality zones
  02. psnr_distribution.png       — PSNR histogram
  03. mae_distribution.png        — MAE histogram
  04. metrics_boxplot.png         — Side-by-side boxplots for all 3 metrics
  05. ssim_vs_psnr.png            — SSIM vs PSNR scatter coloured by MAE
  06. cumulative_ssim.png         — Cumulative SSIM distribution
  07. ssim_vs_mae.png             — SSIM vs MAE scatter (new)
  08. metric_correlation_heatmap  — Pearson correlation matrix (new)
  ── Learning curves (from metrics_log.csv) ────────────────────────────────
  09. learning_curve_losses.png   — Generator + Discriminator losses over 160 epochs
  10. learning_curve_metrics.png  — Val SSIM + Val PSNR over 160 epochs
  11. cycle_identity_loss.png     — Cycle consistency + Identity loss curves (new)
  ── Epoch-wise progression (from all_epochs_summary.csv) ──────────────────
  12. epoch_ssim_progression.png  — Mean SSIM across evaluated epochs (new)
  13. epoch_psnr_progression.png  — Mean PSNR across evaluated epochs (new)
  14. epoch_mae_progression.png   — Mean MAE across evaluated epochs (new)
  15. epoch_all_metrics.png       — All 3 metrics in one multi-panel plot (new)
  ── Dice evaluation plots (from dice_results/) ────────────────────────────
  16. dice_per_structure.png      — Bar chart of Dice per brain structure (new)
  17. dice_epoch_progression.png  — Dice score across epochs (new)
  ── Summary ───────────────────────────────────────────────────────────────
  18. summary_card.png            — Full summary card (presentation ready)

Usage:
    # Plot best epoch (160) results:
    python plot_results.py --epoch 160

    # Plot a specific epoch:
    python plot_results.py --epoch 130

    # Plot all epochs found:
    python plot_results.py --all_epochs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os
import argparse

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
EVAL_RESULTS_DIR = "/media/rvcse22/CSERV/SVARA/diffusion_model/eval_results"
METRICS_LOG      = "/media/rvcse22/CSERV/SVARA/diffusion_model/logs/metrics_log.csv"
DICE_SUMMARY     = "/media/rvcse22/CSERV/SVARA/diffusion_model/dice_results/all_epochs_dice_summary.csv"
PLOTS_DIR        = "/media/rvcse22/CSERV/SVARA/diffusion_model/eval_results/plots"

# ─────────────────────────────────────────────────────────────────────────────
# STYLE — clean, publication-friendly dark theme (matches reference file)
# ─────────────────────────────────────────────────────────────────────────────
ACCENT      = "#2E86AB"    # blue
ACCENT2     = "#A23B72"    # magenta
ACCENT3     = "#F18F01"    # orange
ACCENT4     = "#2ECC71"    # green
ACCENT5     = "#9B59B6"    # purple
BG          = "#0F1117"
BG_PANEL    = "#1A1D27"
TEXT        = "#E8EAF0"
GRID_COLOR  = "#2A2D3A"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG_PANEL,
    "axes.edgecolor":    GRID_COLOR,
    "axes.labelcolor":   TEXT,
    "axes.titlecolor":   TEXT,
    "xtick.color":       TEXT,
    "ytick.color":       TEXT,
    "grid.color":        GRID_COLOR,
    "grid.linewidth":    0.6,
    "text.color":        TEXT,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "legend.facecolor":  BG_PANEL,
    "legend.edgecolor":  GRID_COLOR,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
    "savefig.facecolor": BG,
})


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_slice_metrics(epoch):
    path = os.path.join(EVAL_RESULTS_DIR, f"epoch_{epoch}", "metrics_per_slice.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Slice metrics not found: {path}\n"
            f"Run: python evaluate.py --mode metrics --checkpoint epoch_{epoch}"
        )
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} slice records from epoch {epoch}")
    return df


def load_all_epochs_summary():
    path = os.path.join(EVAL_RESULTS_DIR, "all_epochs_summary.csv")
    if not os.path.exists(path):
        print(f"  [WARN] all_epochs_summary.csv not found at {path} — skipping epoch progression plots.")
        return None
    df = pd.read_csv(path)
    df = df.sort_values("epoch")
    print(f"Loaded epoch summary: {len(df)} epochs")
    return df


def load_metrics_log():
    """
    Load diffusion model training log.
    train.py saves: epoch, loss, val_ssim, val_psnr_dB, val_mae_HU
    """
    if not os.path.exists(METRICS_LOG):
        print(f"  [WARN] metrics_log.csv not found at {METRICS_LOG} — skipping learning curves.")
        return None
    df = pd.read_csv(METRICS_LOG)
    print(f"  Raw CSV columns found: {list(df.columns)}")

    # ── Step 1: ensure epoch column exists ────────────────────────────────────
    epoch_col = next((c for c in df.columns if "epoch" in c.lower()), None)
    if epoch_col is None:
        epoch_col = df.columns[0]   # fallback to first column
    if epoch_col != "epoch":
        df = df.rename(columns={epoch_col: "epoch"})

    df = df.sort_values("epoch").reset_index(drop=True)

    # ── Step 2: build rename map for ALL column variations ────────────────────
    # Map every possible column name variant to the standard name
    rename_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ("val_ssim", "ssim", "val_ssim_score"):
            rename_map[c] = "val_SSIM"
        elif cl in ("val_psnr_db", "val_psnr", "psnr", "val_psnr_score",
                    "val_psnr_dB", "psnr_db"):
            rename_map[c] = "val_PSNR_dB"
        elif cl in ("val_mae_hu", "val_mae", "mae", "mae_hu"):
            rename_map[c] = "val_MAE_HU"
        elif cl in ("loss", "train_loss", "epoch_loss", "avg_loss",
                    "training_loss"):
            rename_map[c] = "loss"

    if rename_map:
        df = df.rename(columns=rename_map)

    print(f"  Columns after rename     : {list(df.columns)}")
    print(f"  Epochs in log            : {len(df)}")
    return df


def load_dice_summary():
    if not os.path.exists(DICE_SUMMARY):
        print(f"  [WARN] Dice summary not found — skipping Dice plots.")
        return None
    df = pd.read_csv(DICE_SUMMARY)
    df = df.sort_values("epoch")
    print(f"Loaded Dice summary: {len(df)} epochs")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ── SECTION 1: Per-slice metric plots ────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def plot_ssim_distribution(df, out_dir, epoch):
    fig, ax = plt.subplots(figsize=(9, 5))
    counts, bins, patches = ax.hist(
        df["ssim"], bins=60, color=ACCENT, alpha=0.85, edgecolor="none"
    )
    for patch, left in zip(patches, bins[:-1]):
        if left >= 0.90:
            patch.set_facecolor(ACCENT4)
        elif left >= 0.80:
            patch.set_facecolor(ACCENT)
        else:
            patch.set_facecolor(ACCENT2)

    mean_ssim = df["ssim"].mean()
    ax.axvline(mean_ssim, color="#F1C40F", linewidth=2,
               linestyle="--", label=f"Mean: {mean_ssim:.4f}")
    ax.axvline(0.90, color=ACCENT4, linewidth=1.5,
               linestyle=":", alpha=0.7, label="Target: 0.90")

    pct_above = 100 * (df["ssim"] >= 0.90).mean()
    ax.text(0.02, 0.95, f"{pct_above:.1f}% of slices ≥ 0.90 SSIM",
            transform=ax.transAxes, fontsize=10, color=ACCENT4, va="top")

    ax.set_xlabel("SSIM")
    ax.set_ylabel("Number of Slices")
    ax.set_title(f"SSIM Distribution — Diffusion Model Brain sCT (Epoch {epoch})")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(out_dir, "01_ssim_distribution.png")
    plt.savefig(path); plt.close()
    print(f"  Saved: {path}")


def plot_psnr_distribution(df, out_dir, epoch):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["psnr_dB"], bins=60, color=ACCENT2, alpha=0.85, edgecolor="none")

    mean_psnr = df["psnr_dB"].mean()
    ax.axvline(mean_psnr, color="#F1C40F", linewidth=2,
               linestyle="--", label=f"Mean: {mean_psnr:.2f} dB")
    ax.axvline(28.0, color=ACCENT4, linewidth=1.5,
               linestyle=":", alpha=0.7, label="Target: 28 dB")

    ax.set_xlabel("PSNR (dB)")
    ax.set_ylabel("Number of Slices")
    ax.set_title(f"PSNR Distribution — Diffusion Model Brain sCT (Epoch {epoch})")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(out_dir, "02_psnr_distribution.png")
    plt.savefig(path); plt.close()
    print(f"  Saved: {path}")


def plot_mae_distribution(df, out_dir, epoch):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["mae"], bins=60, color=ACCENT3, alpha=0.85, edgecolor="none")

    mean_mae = df["mae"].mean()
    ax.axvline(mean_mae, color="#F1C40F", linewidth=2,
               linestyle="--", label=f"Mean: {mean_mae:.4f}")
    ax.axvline(0.05, color=ACCENT4, linewidth=1.5,
               linestyle=":", alpha=0.7, label="Target: < 0.05")

    ax.set_xlabel("MAE (normalised)")
    ax.set_ylabel("Number of Slices")
    ax.set_title(f"MAE Distribution — Diffusion Model Brain sCT (Epoch {epoch})")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(out_dir, "03_mae_distribution.png")
    plt.savefig(path); plt.close()
    print(f"  Saved: {path}")


def plot_metrics_boxplot(df, out_dir, epoch):
    fig, axes = plt.subplots(1, 3, figsize=(13, 6))
    metrics = [
        ("ssim",    "SSIM",      ACCENT,  [0, 1]),
        ("psnr_dB", "PSNR (dB)", ACCENT2, None),
        ("mae",     "MAE",       ACCENT3, None),
    ]
    for ax, (col, label, color, ylim) in zip(axes, metrics):
        data = df[col].dropna().values
        ax.boxplot(data, patch_artist=True, notch=True, widths=0.5,
                   boxprops=dict(facecolor=color, alpha=0.7, linewidth=1.5),
                   medianprops=dict(color="#F1C40F", linewidth=2.5),
                   whiskerprops=dict(color=TEXT, linewidth=1.2),
                   capprops=dict(color=TEXT, linewidth=1.5),
                   flierprops=dict(marker="o", color=color, alpha=0.3, markersize=3))

        sample = data[::max(1, len(data)//500)]
        ax.scatter(np.ones_like(sample) + np.random.uniform(-0.15, 0.15, len(sample)),
                   sample, alpha=0.15, s=4, color=color, zorder=2)

        ax.set_title(label)
        ax.set_ylabel(label)
        ax.set_xticks([])
        ax.grid(axis="y", alpha=0.4)
        if ylim:
            ax.set_ylim(ylim)
        ax.text(0.5, 0.02,
                f"Mean: {np.mean(data):.4f}\nMedian: {np.median(data):.4f}",
                transform=ax.transAxes, fontsize=9,
                ha="center", va="bottom", color=TEXT,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, alpha=0.8))

    fig.suptitle(f"Metric Distribution — Diffusion Model Brain sCT (Epoch {epoch})",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, "04_metrics_boxplot.png")
    plt.savefig(path); plt.close()
    print(f"  Saved: {path}")


def plot_ssim_vs_psnr(df, out_dir, epoch):
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(df["ssim"], df["psnr_dB"],
                    c=df["mae"], cmap="plasma",
                    s=6, alpha=0.4, linewidths=0)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("MAE", color=TEXT)
    cbar.ax.yaxis.set_tick_params(color=TEXT)

    m, b = np.polyfit(df["ssim"].dropna(), df["psnr_dB"].dropna(), 1)
    x_line = np.linspace(df["ssim"].min(), df["ssim"].max(), 100)
    ax.plot(x_line, m * x_line + b, color="#F1C40F",
            linewidth=1.5, linestyle="--", alpha=0.8, label="Trend")

    corr = df["ssim"].corr(df["psnr_dB"])
    ax.text(0.05, 0.95, f"Pearson r = {corr:.3f}",
            transform=ax.transAxes, fontsize=10, color=TEXT, va="top")

    ax.set_xlabel("SSIM")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"SSIM vs PSNR Correlation — Epoch {epoch}\n(colour = MAE)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "05_ssim_vs_psnr.png")
    plt.savefig(path); plt.close()
    print(f"  Saved: {path}")


def plot_cumulative_ssim(df, out_dir, epoch):
    fig, ax = plt.subplots(figsize=(9, 5))
    sorted_ssim = np.sort(df["ssim"].dropna().values)
    cumulative  = np.arange(1, len(sorted_ssim) + 1) / len(sorted_ssim) * 100

    ax.plot(sorted_ssim, cumulative, color=ACCENT, linewidth=2.5)
    ax.fill_between(sorted_ssim, cumulative, alpha=0.15, color=ACCENT)

    for thresh, color, label in [(0.80, ACCENT3, "80%"),
                                  (0.90, ACCENT4, "90%"),
                                  (0.95, "#F1C40F", "95%")]:
        pct = 100 * (df["ssim"] >= thresh).mean()
        ax.axvline(thresh, color=color, linewidth=1.5, linestyle=":", alpha=0.8)
        ax.axhline(pct,    color=color, linewidth=1.5, linestyle=":", alpha=0.8)
        ax.text(thresh + 0.002, pct + 1, f"{pct:.1f}% ≥ {thresh}",
                color=color, fontsize=9)

    ax.set_xlabel("SSIM Threshold")
    ax.set_ylabel("% of Slices Achieving This SSIM or Better")
    ax.set_title(f"Cumulative SSIM Distribution — Diffusion Model Brain sCT (Epoch {epoch})")
    ax.set_xlim([df["ssim"].min() - 0.02, 1.01])
    ax.set_ylim([0, 102])
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "06_cumulative_ssim.png")
    plt.savefig(path); plt.close()
    print(f"  Saved: {path}")


def plot_ssim_vs_mae(df, out_dir, epoch):
    """
    NEW — SSIM vs MAE scatter.
    Shows inverse relationship: high SSIM slices should have low MAE.
    Good for papers to demonstrate metric consistency.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(df["ssim"], df["mae"],
                    c=df["psnr_dB"], cmap="viridis",
                    s=6, alpha=0.4, linewidths=0)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("PSNR (dB)", color=TEXT)
    cbar.ax.yaxis.set_tick_params(color=TEXT)

    corr = df["ssim"].corr(df["mae"])
    ax.text(0.05, 0.95, f"Pearson r = {corr:.3f}",
            transform=ax.transAxes, fontsize=10, color=TEXT, va="top")

    ax.set_xlabel("SSIM")
    ax.set_ylabel("MAE (normalised)")
    ax.set_title(f"SSIM vs MAE — Epoch {epoch}\n(colour = PSNR)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "07_ssim_vs_mae.png")
    plt.savefig(path); plt.close()
    print(f"  Saved: {path}")


def plot_metric_correlation_heatmap(df, out_dir, epoch):
    """
    NEW — Pearson correlation matrix of SSIM, PSNR, MAE.
    Standard inclusion in research papers to show metric relationships.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    cols    = ["ssim", "psnr_dB", "mae"]
    labels  = ["SSIM", "PSNR", "MAE"]
    corr    = df[cols].corr()

    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Pearson r")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr.values[i, j]:.3f}",
                    ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color="white" if abs(corr.values[i, j]) > 0.5 else TEXT)

    ax.set_title(f"Metric Correlation Matrix — Epoch {epoch}")
    plt.tight_layout()
    path = os.path.join(out_dir, "08_metric_correlation_heatmap.png")
    plt.savefig(path); plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# ── SECTION 2: Learning curves (from metrics_log.csv) ────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def plot_learning_curve_losses(log_df, out_dir):
    """
    Diffusion Model — training loss curve.
    Loss = MSE noise prediction + 0.3 x L1 x0 prediction (from train.py).
    """
    if "loss" not in log_df.columns:
        print("  [WARN] loss column not found — skipping plot 09")
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    window      = 5
    loss_smooth = log_df["loss"].rolling(window, min_periods=1).mean()

    ax.plot(log_df["epoch"], log_df["loss"],
            color=ACCENT, linewidth=0.8, alpha=0.35, label="Training Loss (raw)")
    ax.plot(log_df["epoch"], loss_smooth,
            color=ACCENT, linewidth=2.5,
            label=f"Training Loss (smoothed, w={window})")

    best_idx   = log_df["loss"].idxmin()
    best_epoch = log_df.loc[best_idx, "epoch"]
    best_loss  = log_df.loc[best_idx, "loss"]
    ax.axvline(best_epoch, color="#F1C40F", linewidth=1.5, linestyle="--",
               alpha=0.8, label=f"Best: {best_loss:.5f} @ epoch {int(best_epoch)}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE + 0.3 x L1)")
    ax.set_title("Diffusion Model — Training Loss over Epochs")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "09_training_loss.png")
    plt.savefig(path); plt.close()
    print(f"  Saved: {path}")

def plot_learning_curve_metrics(log_df, out_dir):
    """
    Validation SSIM and PSNR tracked every epoch during training.
    Shows when the model converged — critical for learning curve analysis.
    """
    # Guard — skip if required columns missing
    missing = [c for c in ["val_SSIM", "val_PSNR_dB"] if c not in log_df.columns]
    if missing:
        print(f"  [WARN] plot10 skipped — columns not found: {missing}")
        print(f"         Available columns: {list(log_df.columns)}")
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Smooth with rolling average for clarity
    window = 5
    ssim_smooth = log_df["val_SSIM"].rolling(window, min_periods=1).mean()
    psnr_smooth = log_df["val_PSNR_dB"].rolling(window, min_periods=1).mean()

    # SSIM
    axes[0].plot(log_df["epoch"], log_df["val_SSIM"],
                 color=ACCENT, linewidth=0.8, alpha=0.35, label="Val SSIM (raw)")
    axes[0].plot(log_df["epoch"], ssim_smooth,
                 color=ACCENT, linewidth=2.0, label=f"Val SSIM (smoothed, w={window})")
    axes[0].axhline(0.85, color=ACCENT4, linewidth=1.2,
                    linestyle=":", alpha=0.8, label="Target: 0.85")
    axes[0].axvspan(0, 5, alpha=0.1, color=ACCENT4, label="Warmup")

    best_ssim_epoch = log_df.loc[log_df["val_SSIM"].idxmax(), "epoch"]
    best_ssim_val   = log_df["val_SSIM"].max()
    axes[0].axvline(best_ssim_epoch, color="#F1C40F", linewidth=1.5,
                    linestyle="--", alpha=0.8, label=f"Best: {best_ssim_val:.4f} @ epoch {best_ssim_epoch}")

    axes[0].set_ylabel("SSIM")
    axes[0].set_title("Validation SSIM over Training")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # PSNR
    axes[1].plot(log_df["epoch"], log_df["val_PSNR_dB"],
                 color=ACCENT2, linewidth=0.8, alpha=0.35, label="Val PSNR (raw)")
    axes[1].plot(log_df["epoch"], psnr_smooth,
                 color=ACCENT2, linewidth=2.0, label=f"Val PSNR (smoothed, w={window})")
    axes[1].axhline(28.0, color=ACCENT4, linewidth=1.2,
                    linestyle=":", alpha=0.8, label="Target: 28 dB")
    axes[1].axvspan(0, 5, alpha=0.1, color=ACCENT4, label="Warmup")

    best_psnr_epoch = log_df.loc[log_df["val_PSNR_dB"].idxmax(), "epoch"]
    best_psnr_val   = log_df["val_PSNR_dB"].max()
    axes[1].axvline(best_psnr_epoch, color="#F1C40F", linewidth=1.5,
                    linestyle="--", alpha=0.8, label=f"Best: {best_psnr_val:.2f} dB @ epoch {best_psnr_epoch}")

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].set_title("Validation PSNR over Training")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.suptitle("Diffusion Model — Validation Metric Learning Curves (160 Epochs)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "10_learning_curve_metrics.png")
    plt.savefig(path); plt.close()
    print(f"  Saved: {path}")


def plot_cycle_identity_loss(log_df, out_dir):
    """
    Diffusion Model — Validation MAE over epochs (HU units).
    Replaces CycleGAN cycle+identity loss (not applicable to diffusion).
    """
    if "val_MAE_HU" not in log_df.columns:
        print("  [WARN] val_MAE_HU column not found — skipping plot 11")
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    window     = 5
    mae_smooth = log_df["val_MAE_HU"].rolling(window, min_periods=1).mean()

    ax.plot(log_df["epoch"], log_df["val_MAE_HU"],
            color=ACCENT3, linewidth=0.8, alpha=0.35, label="Val MAE HU (raw)")
    ax.plot(log_df["epoch"], mae_smooth,
            color=ACCENT3, linewidth=2.5,
            label=f"Val MAE HU (smoothed, w={window})")

    best_idx   = log_df["val_MAE_HU"].idxmin()
    best_epoch = log_df.loc[best_idx, "epoch"]
    best_mae   = log_df.loc[best_idx, "val_MAE_HU"]
    ax.axvline(best_epoch, color="#F1C40F", linewidth=1.5, linestyle="--",
               alpha=0.8, label=f"Best: {best_mae:.1f} HU @ epoch {int(best_epoch)}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (Hounsfield Units)")
    ax.set_title("Diffusion Model — Validation MAE over Epochs")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "11_val_mae_curve.png")
    plt.savefig(path); plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# ── SECTION 3: Epoch-wise progression ────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def plot_epoch_progression(epoch_df, out_dir):
    """
    NEW — SSIM, PSNR, MAE mean values across evaluated epochs (130,140,150,160).
    Shows how the model improved from epoch to epoch — very useful in papers
    to justify the choice of final epoch.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = [
        ("ssim_mean",  "ssim_std",  "SSIM",      ACCENT,  "12_epoch_ssim_progression.png"),
        ("psnr_mean",  "psnr_std",  "PSNR (dB)", ACCENT2, "13_epoch_psnr_progression.png"),
        ("mae_mean",   "mae_std",   "MAE",        ACCENT3, "14_epoch_mae_progression.png"),
    ]

    for ax, (mean_col, std_col, label, color, fname) in zip(axes, metrics):
        epochs = epoch_df["epoch"].values
        means  = epoch_df[mean_col].values
        stds   = epoch_df[std_col].values

        ax.plot(epochs, means, color=color, linewidth=2.5,
                marker="o", markersize=8, label=label)
        ax.fill_between(epochs, means - stds, means + stds,
                        alpha=0.2, color=color, label="± 1 std")

        # Annotate each point
        for e, m in zip(epochs, means):
            ax.annotate(f"{m:.4f}", (e, m),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=8, color=TEXT)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(f"{label} Across Evaluated Epochs")
        ax.set_xticks(epochs)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Diffusion Model — Metric Progression Across Epochs",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "15_epoch_all_metrics.png")
    plt.savefig(path); plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# ── SECTION 4: Dice evaluation plots ─────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

STRUCTURE_COLORS_PLOT = [
    ACCENT, ACCENT2, ACCENT3, ACCENT4, ACCENT5,
    "#E74C3C", "#1ABC9C", "#F39C12"
]

def plot_dice_per_structure(dice_df, out_dir):
    """
    NEW — Bar chart of mean Dice score per brain structure for the best epoch.
    Standard plot in medical image segmentation papers.
    """
    best_row   = dice_df.loc[dice_df["mean_dice"].idxmax()]
    best_epoch = int(best_row["epoch"])
    structures = ["brain", "skull", "white_matter", "gray_matter",
                  "csf", "ventricles", "cerebellum", "brainstem"]

    values = [float(best_row.get(s, 0)) for s in structures]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(structures, values,
                  color=STRUCTURE_COLORS_PLOT[:len(structures)],
                  alpha=0.85, edgecolor="none")

    ax.axhline(0.85, color="#F1C40F", linewidth=1.5,
               linestyle="--", alpha=0.8, label="Target: 0.85")
    ax.axhline(float(best_row["mean_dice"]), color=ACCENT4, linewidth=2,
               linestyle="-", alpha=0.8,
               label=f"Mean Dice: {float(best_row['mean_dice']):.4f}")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, color=TEXT)

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Dice Score")
    ax.set_title(f"Dice Score per Brain Structure — Best Epoch ({best_epoch})")
    ax.set_xticklabels(structures, rotation=25, ha="right")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "16_dice_per_structure.png")
    plt.savefig(path); plt.close()
    print(f"  Saved: {path}")


def plot_dice_epoch_progression(dice_df, out_dir):
    """
    NEW — Mean Dice score across evaluated epochs.
    Shows whether segmentation quality (clinical usability) improved with more training.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs     = dice_df["epoch"].values
    mean_dice  = dice_df["mean_dice"].values.astype(float)

    ax.plot(epochs, mean_dice, color=ACCENT4, linewidth=2.5,
            marker="o", markersize=9, label="Mean Dice (all structures)")

    for e, d in zip(epochs, mean_dice):
        ax.annotate(f"{d:.4f}", (e, d),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9, color=TEXT)

    ax.axhline(0.85, color="#F1C40F", linewidth=1.5,
               linestyle="--", alpha=0.8, label="Target: 0.85")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Dice Score")
    ax.set_title("Diffusion Model — Dice Score Progression Across Epochs\n"
                 "(Clinical usability measured by TotalSegmentator)")
    ax.set_xticks(epochs)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "17_dice_epoch_progression.png")
    plt.savefig(path); plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# ── SECTION 5: Summary card ───────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary_card(df, log_df, out_dir, epoch):
    """
    Single clean summary figure — Pix2Pix style.
    Shows mean metrics, benchmark comparison, and distributions.
    """
    fig = plt.figure(figsize=(16, 8))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Top row: three metric cards ────────────────────────────────────────────
    metric_data = [
        ("SSIM",      df["ssim"].mean(),    df["ssim"].std(),
         ACCENT,  0.87, 0.94, "Published range: 0.87–0.94"),
        ("PSNR (dB)", df["psnr_dB"].mean(), df["psnr_dB"].std(),
         ACCENT2, 28.0, 32.0, "Published range: 28–32 dB"),
        ("MAE",       df["mae"].mean(),     df["mae"].std(),
         ACCENT3, None, None, "Lower is better"),
    ]

    for col, (metric, mean, std, color, lo, hi, note) in enumerate(metric_data):
        ax = fig.add_subplot(gs[0, col])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

        rect = FancyBboxPatch((0.05, 0.05), 0.90, 0.90,
                               boxstyle="round,pad=0.02",
                               facecolor=BG, edgecolor=color,
                               linewidth=3, alpha=0.9)
        ax.add_patch(rect)

        val_str = f"{mean:.4f}" if isinstance(mean, float) else f"{int(mean)}"
        ax.text(0.5, 0.70, val_str,
                ha="center", va="center",
                fontsize=32, fontweight="bold", color=color)
        if std > 0:
            ax.text(0.5, 0.50, f"± {std:.4f}",
                    ha="center", va="center",
                    fontsize=12, color=TEXT, alpha=0.8)
        ax.text(0.5, 0.88, metric,
                ha="center", va="center",
                fontsize=14, fontweight="bold", color=TEXT)
        ax.text(0.5, 0.25, note,
                ha="center", va="center",
                fontsize=9, color=TEXT, alpha=0.7, style="italic")

        if lo is not None and hi is not None:
            in_range = lo <= mean <= hi
            status   = "✓ Within published range" if in_range else ("↑ Above" if mean > hi else "↓ Below")
            s_color  = ACCENT4 if in_range else ACCENT3
            ax.text(0.5, 0.08, status,
                    ha="center", va="center",
                    fontsize=9, color=s_color, fontweight="bold")

    # ── Bottom row: distributions ─────────────────────────────────────────────
    for col, (col_name, color, xlabel) in enumerate([
        ("ssim",    ACCENT,  "SSIM"),
        ("psnr_dB", ACCENT2, "PSNR (dB)"),
        ("mae",     ACCENT3, "MAE"),
    ]):
        ax = fig.add_subplot(gs[1, col])
        data = df[col_name].dropna()
        ax.hist(data, bins=50, color=color, alpha=0.85, edgecolor="none")
        ax.axvline(data.mean(), color="#F1C40F", linewidth=2.5,
                   linestyle="--", alpha=0.95, label=f"Mean: {data.mean():.4f}")
        ax.set_xlabel(xlabel, fontsize=10, fontweight="bold")
        ax.set_ylabel("Slices", fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.tick_params(labelsize=9)
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"Diffusion Model — Synthetic CT Evaluation Summary\n"
        f"Brain MRI → CT  |  SynthRAD2023  |  Epoch {epoch}",
        fontsize=14, fontweight="bold", y=0.99
    )
    path = os.path.join(out_dir, "18_summary_card.png")
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epoch", type=int, default=160,
        help="Epoch to use for per-slice metric plots (default: 160)"
    )
    args = parser.parse_args()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"Saving all plots to: {PLOTS_DIR}/\n")

    # Load data
    df        = load_slice_metrics(args.epoch)
    epoch_df  = load_all_epochs_summary()
    log_df    = load_metrics_log()
    dice_df   = load_dice_summary()
    print()

    # ── Section 1: Per-slice plots ────────────────────────────────────────────
    print("Section 1: Per-slice metric plots...")
    plot_ssim_distribution(df, PLOTS_DIR, args.epoch)
    plot_psnr_distribution(df, PLOTS_DIR, args.epoch)
    plot_mae_distribution(df, PLOTS_DIR, args.epoch)
    plot_metrics_boxplot(df, PLOTS_DIR, args.epoch)
    plot_ssim_vs_psnr(df, PLOTS_DIR, args.epoch)
    plot_cumulative_ssim(df, PLOTS_DIR, args.epoch)
    plot_ssim_vs_mae(df, PLOTS_DIR, args.epoch)
    plot_metric_correlation_heatmap(df, PLOTS_DIR, args.epoch)

    # ── Section 2: Learning curves ────────────────────────────────────────────
    if log_df is not None:
        print("\nSection 2: Learning curves...")
        plot_learning_curve_losses(log_df, PLOTS_DIR)
        plot_learning_curve_metrics(log_df, PLOTS_DIR)
        plot_cycle_identity_loss(log_df, PLOTS_DIR)

    # ── Section 3: Epoch progression ──────────────────────────────────────────
    if epoch_df is not None:
        print("\nSection 3: Epoch progression plots...")
        plot_epoch_progression(epoch_df, PLOTS_DIR)

    # ── Section 4: Dice plots ─────────────────────────────────────────────────
    if dice_df is not None:
        print("\nSection 4: Dice evaluation plots...")
        plot_dice_per_structure(dice_df, PLOTS_DIR)
        plot_dice_epoch_progression(dice_df, PLOTS_DIR)

    # ── Section 5: Summary card ───────────────────────────────────────────────
    print("\nSection 5: Summary card...")
    plot_summary_card(df, log_df, PLOTS_DIR, args.epoch)

    # ── Final report ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"All plots saved to: {PLOTS_DIR}/")
    print(f"{'='*60}")
    print(f"\nQuick summary (epoch {args.epoch}):")
    print(f"  SSIM  — mean={df['ssim'].mean():.4f}  std={df['ssim'].std():.4f}")
    print(f"  PSNR  — mean={df['psnr_dB'].mean():.3f}  std={df['psnr_dB'].std():.3f}")
    print(f"  MAE   — mean={df['mae'].mean():.4f}  std={df['mae'].std():.4f}")
    print(f"  Slices with SSIM >= 0.90: {100*(df['ssim']>=0.90).mean():.1f}%")
    print(f"\nMost impactful for research paper:")
    print(f"  18_summary_card.png          — single slide with all key results")
    print(f"  10_learning_curve_metrics.png — shows training convergence")
    print(f"  11_cycle_identity_loss.png   — CycleGAN-specific, unique contribution")
    print(f"  06_cumulative_ssim.png        — % of slices above quality threshold")
    print(f"  08_metric_correlation_heatmap — metric consistency proof")
    print(f"  16_dice_per_structure.png     — clinical usability evidence")


if __name__ == "__main__":
    main()
