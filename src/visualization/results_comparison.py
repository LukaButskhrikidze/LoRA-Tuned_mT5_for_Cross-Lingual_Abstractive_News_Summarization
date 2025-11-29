"""
Compare results across models
Usage: python -m src.visualization.results_comparison
"""

import json
import matplotlib.pyplot as plt
import numpy as np

def plot_rouge_comparison(results_files, output_path="outputs/figures/rouge_comparison.png"):
    """Create bar chart comparing ROUGE scores"""
    
    # Load results
    data = []
    for file in results_files:
        with open(file, 'r') as f:
            data.append(json.load(f))
    
    # Extract data
    labels = [d.get('label', d['model_path'].split('/')[-1]) for d in data]
    rouge1 = [d['rouge1'] for d in data]
    rouge2 = [d['rouge2'] for d in data]
    rougeL = [d['rougeL'] for d in data]
    
    # Create plot
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - width, rouge1, width, label='ROUGE-1', color='#2E86AB')
    ax.bar(x, rouge2, width, label='ROUGE-2', color='#A23B72')
    ax.bar(x + width, rougeL, width, label='ROUGE-L', color='#F18F01')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('ROUGE Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved comparison to {output_path}")

if __name__ == "__main__":
    # Example usage (update paths after training)
    results_files = [
        "outputs/results/full_en_results.json",
        "outputs/results/lora_en_results.json",
        "outputs/results/full_ru_results.json",
        "outputs/results/lora_ru_results.json",
    ]
    plot_rouge_comparison(results_files)