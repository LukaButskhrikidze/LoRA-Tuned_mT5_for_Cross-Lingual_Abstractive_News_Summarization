"""
Training visualization utilities
Usage: python -m src.visualization.training_plots --log_dir outputs/checkpoints/mt5_lora_en/logs
"""

import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

def plot_training_curves(log_dirs, labels, output_path="outputs/figures/training_curves.png"):
    """Plot training loss curves from multiple runs"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for log_dir, label in zip(log_dirs, labels):
        # Load tensorboard logs
        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()
        
        # Extract training loss
        train_loss = [(s.step, s.value) for s in ea.Scalars('train/loss')]
        if train_loss:
            steps, losses = zip(*train_loss)
            ax1.plot(steps, losses, label=label, linewidth=2)
        
        # Extract eval loss
        eval_loss = [(s.step, s.value) for s in ea.Scalars('eval/loss')]
        if eval_loss:
            steps, losses = zip(*eval_loss)
            ax2.plot(steps, losses, label=label, linewidth=2)
    
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Validation Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved training curves to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dirs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", default="outputs/figures/training_curves.png")
    args = parser.parse_args()
    
    plot_training_curves(args.log_dirs, args.labels, args.output)