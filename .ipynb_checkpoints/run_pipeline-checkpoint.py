import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
import tempfile
import os
import torch
import glob

from config import *
from train import train_model
from eval import evaluate_model
from models import OmniSR
from utils import visualize_sample


def generate_report(model, loss_history, eval_results, upscale_factor):
    """Build a multi-page PDF report combining plots and metrics."""
    report_path = f"Reports/OmniSR_Report_x{upscale_factor}.pdf"
    os.makedirs("Reports", exist_ok=True)
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(report_path, pagesize=letter)
    elements = []

    # --- Title ---
    elements.append(Paragraph(f"<b>OmniSR Super-Resolution Report (×{upscale_factor})</b>", styles['Title']))
    elements.append(Spacer(1, 12))

    # --- Model info ---
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    elements.append(Paragraph(
        f"Trained for <b>{NUM_ITERATIONS:,}</b> iterations on <b>{DEVICE}</b> "
        f"using <b>{OPTIMIZER}</b> (LR={LEARNING_RATE}, WD={WEIGHT_DECAY})<br/>"
        f"Loss Function: <b>{LOSS_FN}</b> &nbsp;&nbsp; "
        f"Model Parameters: <b>{param_count:.2f}M</b>",
        styles['Normal']
    ))
    elements.append(Spacer(1, 12))

    # --- Loss plot (if available) ---
    if loss_history:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(loss_history, color='blue', linewidth=2)
        ax.set_xlabel('Checkpoint Index')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Curve (Smoothed)')
        loss_img = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        fig.savefig(loss_img.name, dpi=200, bbox_inches='tight')
        plt.close(fig)
        elements.append(Paragraph("<b>Training Loss Curve</b>", styles['Heading2']))
        elements.append(RLImage(loss_img.name, width=400, height=240))
        elements.append(Spacer(1, 12))

    # --- Evaluation Results ---
    elements.append(Paragraph("<b>Evaluation Metrics</b>", styles['Heading2']))
    table_text = "<br/>".join(
        [f"{r['dataset']}: PSNR = {r['psnr']:.2f} dB, SSIM = {r['ssim']:.4f}, MSE = {r['mse']:.6f}"
         for r in eval_results]
    )
    elements.append(Paragraph(table_text, styles['Normal']))
    elements.append(Spacer(1, 12))

    # --- PSNR/SSIM Comparison Plot ---
    fig, ax1 = plt.subplots(figsize=(6, 3))
    datasets = [r["dataset"] for r in eval_results]
    psnr_vals = [r["psnr"] for r in eval_results]
    ssim_vals = [r["ssim"] for r in eval_results]

    ax1.bar(datasets, psnr_vals, color="skyblue", label="PSNR (dB)")
    ax2 = ax1.twinx()
    ax2.plot(datasets, ssim_vals, color="orange", marker="o", label="SSIM")
    ax1.set_ylabel("PSNR (dB)")
    ax2.set_ylabel("SSIM")
    ax1.set_title("Evaluation Performance Across Datasets")
    plt.tight_layout()
    perf_img = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    fig.savefig(perf_img.name, dpi=200, bbox_inches='tight')
    plt.close(fig)
    elements.append(RLImage(perf_img.name, width=420, height=240))
    elements.append(Spacer(1, 12))

    # --- Sample Visualizations ---
    elements.append(Paragraph("<b>Visualizations</b>", styles['Heading2']))
    for ds in TEST_DATASETS:
        fig = visualize_sample(model, dataset_name=ds, device=DEVICE)
        tmp_img = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        fig.savefig(tmp_img.name, dpi=150, bbox_inches='tight')
        plt.close(fig)
        elements.append(Paragraph(f"{ds} Example", styles['Heading3']))
        elements.append(RLImage(tmp_img.name, width=450, height=180))
        elements.append(Spacer(1, 10))

    doc.build(elements)
    print(f"✅ Report saved to: {report_path}")
    return report_path


def main():
    # 1. Automatically find the latest checkpoint in the Models folder
    checkpoints = glob.glob(os.path.join(MODEL_SAVE_DIR, "iter_*.pth"))
    
    latest_ckpt = None
    if checkpoints:
        # Sort by iteration number to find the highest one
        # Assumes format: Models/iter_50000.pth
        checkpoints.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        latest_ckpt = checkpoints[-1]
        print(f"--- Found latest checkpoint: {latest_ckpt} ---")
    else:
        print("--- No checkpoint found. Starting training from scratch. ---")

    # 2. Pass the latest checkpoint to the training function
    model, loss_history = train_model(resume_path=latest_ckpt)

    # 3. Proceed to evaluation and report generation
    eval_results = []
    for name in TEST_DATASETS:
        res = evaluate_model(model, name)
        eval_results.append(res)
    
    generate_report(model, loss_history, eval_results, UPSCALE_FACTOR)


if __name__ == "__main__":
    main()
