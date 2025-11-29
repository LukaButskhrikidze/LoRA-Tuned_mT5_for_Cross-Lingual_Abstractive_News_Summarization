"""
Create comprehensive visualizations for the project
Usage: python create_visualizations.py
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'

# Create output directory
os.makedirs('outputs/figures', exist_ok=True)

# Data from your results
data = {
    'English Full FT': {
        'rouge1': 31.09,
        'rouge2': 9.42,
        'rougeL': 24.02,
        'model_size_mb': 300,
        'trainable_params_pct': 100,
        'training_time_h': 1.47
    },
    'English LoRA': {
        'rouge1': 26.31,
        'rouge2': 5.93,
        'rougeL': 20.18,
        'model_size_mb': 5,
        'trainable_params_pct': 0.3,
        'training_time_h': 1.5
    },
    'Russian Full FT': {
        'rouge1': 5.22,
        'rouge2': 1.14,
        'rougeL': 5.17,
        'model_size_mb': 300,
        'trainable_params_pct': 100,
        'training_time_h': 1.0
    },
    'Russian LoRA': {
        'rouge1': 3.08,
        'rouge2': 0.71,
        'rougeL': 3.07,
        'model_size_mb': 5,
        'trainable_params_pct': 0.3,
        'training_time_h': 0.9
    }
}

def plot_rouge_comparison():
    """ROUGE scores comparison - English only (Russian failed)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['Full Fine-Tuning', 'LoRA (r=8)']
    rouge1 = [data['English Full FT']['rouge1'], data['English LoRA']['rouge1']]
    rouge2 = [data['English Full FT']['rouge2'], data['English LoRA']['rouge2']]
    rougeL = [data['English Full FT']['rougeL'], data['English LoRA']['rougeL']]
    
    x = np.arange(len(models))
    width = 0.25
    
    bars1 = ax.bar(x - width, rouge1, width, label='ROUGE-1', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x, rouge2, width, label='ROUGE-2', color='#A23B72', alpha=0.8)
    bars3 = ax.bar(x + width, rougeL, width, label='ROUGE-L', color='#F18F01', alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('ROUGE Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('English Summarization: ROUGE Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 35)
    
    plt.tight_layout()
    plt.savefig('outputs/figures/rouge_comparison.png', bbox_inches='tight')
    print("✅ Saved: outputs/figures/rouge_comparison.png")
    plt.close()

def plot_efficiency_comparison():
    """Model size vs Performance"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # English models
    models = ['Full FT', 'LoRA']
    sizes = [data['English Full FT']['model_size_mb'], data['English LoRA']['model_size_mb']]
    performance = [data['English Full FT']['rougeL'], data['English LoRA']['rougeL']]
    
    # Create scatter plot with size bubbles
    colors = ['#2E86AB', '#F18F01']
    for i, (model, size, perf) in enumerate(zip(models, sizes, performance)):
        ax.scatter(size, perf, s=1000, alpha=0.6, c=colors[i], edgecolors='black', linewidth=2)
        ax.annotate(f'{model}\n{size}MB\n{perf:.1f} ROUGE-L', 
                   (size, perf), 
                   ha='center', 
                   va='center',
                   fontsize=10,
                   fontweight='bold')
    
    ax.set_xlabel('Model Size (MB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('ROUGE-L Score', fontsize=12, fontweight='bold')
    ax.set_title('Performance vs Model Size Trade-off', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-20, 320)
    ax.set_ylim(15, 26)
    
    # Add efficiency annotation
    ax.annotate('60× smaller\n84% performance', 
               xy=(5, 20.18), 
               xytext=(80, 17),
               arrowprops=dict(arrowstyle='->', color='green', lw=2),
               fontsize=11,
               color='green',
               fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('outputs/figures/efficiency_comparison.png', bbox_inches='tight')
    print("✅ Saved: outputs/figures/efficiency_comparison.png")
    plt.close()

def plot_parameter_efficiency():
    """Trainable parameters comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Bar chart of trainable parameters
    models = ['Full Fine-Tuning', 'LoRA (r=8)']
    params = [data['English Full FT']['trainable_params_pct'], 
              data['English LoRA']['trainable_params_pct']]
    
    bars = ax1.bar(models, params, color=['#2E86AB', '#F18F01'], alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add percentage labels
    for bar, param in zip(bars, params):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height/2,
               f'{param}%\n({param*3:.1f}M params)' if param == 100 else f'{param}%\n(0.9M params)',
               ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    
    ax1.set_ylabel('Trainable Parameters (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Trainable Parameters Comparison', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 110)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Right: Pie chart for LoRA
    lora_frozen = 100 - data['English LoRA']['trainable_params_pct']
    lora_trainable = data['English LoRA']['trainable_params_pct']
    
    wedges, texts, autotexts = ax2.pie(
        [lora_frozen, lora_trainable],
        labels=['Frozen\nBase Model', 'LoRA\nAdapters'],
        autopct='%1.1f%%',
        colors=['#CCCCCC', '#F18F01'],
        explode=(0, 0.1),
        startangle=90,
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    
    ax2.set_title('LoRA Parameter Distribution', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/figures/parameter_efficiency.png', bbox_inches='tight')
    print("✅ Saved: outputs/figures/parameter_efficiency.png")
    plt.close()

def plot_cross_lingual_comparison():
    """English vs Russian performance (showing failure)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    languages = ['English', 'Russian']
    full_ft = [data['English Full FT']['rougeL'], data['Russian Full FT']['rougeL']]
    lora = [data['English LoRA']['rougeL'], data['Russian LoRA']['rougeL']]
    
    x = np.arange(len(languages))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, full_ft, width, label='Full Fine-Tuning', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, lora, width, label='LoRA (r=8)', color='#F18F01', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add failure annotation for Russian
    ax.annotate('Both methods\nfailed for Russian', 
               xy=(1, 5), 
               xytext=(1.3, 12),
               arrowprops=dict(arrowstyle='->', color='red', lw=2),
               fontsize=11,
               color='red',
               fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFE5E5', alpha=0.9))
    
    ax.set_ylabel('ROUGE-L Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Language', fontsize=12, fontweight='bold')
    ax.set_title('Cross-Lingual Performance: mT5-small Capacity Limitation', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(languages)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 28)
    
    # Add horizontal line for "acceptable" threshold
    ax.axhline(y=15, color='green', linestyle='--', alpha=0.5, label='Acceptable threshold')
    ax.text(0.5, 16, 'Minimum acceptable (~15)', ha='center', fontsize=9, color='green')
    
    plt.tight_layout()
    plt.savefig('outputs/figures/cross_lingual_comparison.png', bbox_inches='tight')
    print("✅ Saved: outputs/figures/cross_lingual_comparison.png")
    plt.close()

def plot_summary_dashboard():
    """Comprehensive summary dashboard"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Top left: ROUGE comparison
    ax1 = fig.add_subplot(gs[0, :2])
    models = ['Full FT', 'LoRA']
    rouge1 = [data['English Full FT']['rouge1'], data['English LoRA']['rouge1']]
    rouge2 = [data['English Full FT']['rouge2'], data['English LoRA']['rouge2']]
    rougeL = [data['English Full FT']['rougeL'], data['English LoRA']['rougeL']]
    
    x = np.arange(len(models))
    width = 0.25
    ax1.bar(x - width, rouge1, width, label='ROUGE-1', color='#2E86AB', alpha=0.8)
    ax1.bar(x, rouge2, width, label='ROUGE-2', color='#A23B72', alpha=0.8)
    ax1.bar(x + width, rougeL, width, label='ROUGE-L', color='#F18F01', alpha=0.8)
    ax1.set_ylabel('Score')
    ax1.set_title('English Performance', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Top right: Model size comparison
    ax2 = fig.add_subplot(gs[0, 2])
    sizes = [data['English Full FT']['model_size_mb'], data['English LoRA']['model_size_mb']]
    ax2.bar(models, sizes, color=['#2E86AB', '#F18F01'], alpha=0.8)
    ax2.set_ylabel('Size (MB)')
    ax2.set_title('Model Size', fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y')
    for i, (model, size) in enumerate(zip(models, sizes)):
        ax2.text(i, size, f'{size}MB', ha='center', va='bottom', fontweight='bold')
    
    # Middle left: Performance retention
    ax3 = fig.add_subplot(gs[1, 0])
    retention = (data['English LoRA']['rougeL'] / data['English Full FT']['rougeL']) * 100
    ax3.pie([retention, 100-retention], 
           labels=['Retained', 'Lost'],
           autopct='%1.1f%%',
           colors=['#F18F01', '#EEEEEE'],
           startangle=90)
    ax3.set_title('LoRA Performance\nRetention', fontweight='bold')
    
    # Middle center: Size reduction
    ax4 = fig.add_subplot(gs[1, 1])
    reduction = (data['English Full FT']['model_size_mb'] / data['English LoRA']['model_size_mb'])
    ax4.pie([1, reduction-1], 
           labels=['LoRA', f'{reduction:.0f}× reduction'],
           autopct=lambda p: f'{p:.0f}%' if p > 5 else '',
           colors=['#F18F01', '#EEEEEE'],
           startangle=90)
    ax4.set_title('Storage Savings', fontweight='bold')
    
    # Middle right: Trainable params
    ax5 = fig.add_subplot(gs[1, 2])
    params = [data['English Full FT']['trainable_params_pct'], 
              data['English LoRA']['trainable_params_pct']]
    ax5.bar(models, params, color=['#2E86AB', '#F18F01'], alpha=0.8)
    ax5.set_ylabel('Trainable %')
    ax5.set_title('Parameter Efficiency', fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Bottom: Cross-lingual comparison
    ax6 = fig.add_subplot(gs[2, :])
    languages = ['English\nFull FT', 'English\nLoRA', 'Russian\nFull FT', 'Russian\nLoRA']
    scores = [data['English Full FT']['rougeL'], 
              data['English LoRA']['rougeL'],
              data['Russian Full FT']['rougeL'],
              data['Russian LoRA']['rougeL']]
    colors_list = ['#2E86AB', '#F18F01', '#CCCCCC', '#CCCCCC']
    
    bars = ax6.bar(languages, scores, color=colors_list, alpha=0.8)
    ax6.set_ylabel('ROUGE-L Score')
    ax6.set_title('Cross-Lingual Comparison (Model Capacity Limitation)', fontweight='bold')
    ax6.axhline(y=15, color='green', linestyle='--', alpha=0.3)
    ax6.text(2.5, 16, 'Acceptable threshold', fontsize=9, color='green')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add "FAILED" annotation
    ax6.text(2.5, 8, 'BOTH FAILED\n(Model too small)', 
            ha='center', fontsize=11, fontweight='bold', color='red',
            bbox=dict(boxstyle='round', facecolor='#FFE5E5', alpha=0.8))
    
    plt.suptitle('LoRA vs Full Fine-Tuning: Comprehensive Analysis', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig('outputs/figures/summary_dashboard.png', bbox_inches='tight')
    print("✅ Saved: outputs/figures/summary_dashboard.png")
    plt.close()

def main():
    print("\n" + "="*70)
    print("Creating Visualizations")
    print("="*70 + "\n")
    
    plot_rouge_comparison()
    plot_efficiency_comparison()
    plot_parameter_efficiency()
    plot_cross_lingual_comparison()
    plot_summary_dashboard()
    
    print("\n" + "="*70)
    print("✅ All visualizations created successfully!")
    print("="*70)
    print("\nGenerated files:")
    print("  - outputs/figures/rouge_comparison.png")
    print("  - outputs/figures/efficiency_comparison.png")
    print("  - outputs/figures/parameter_efficiency.png")
    print("  - outputs/figures/cross_lingual_comparison.png")
    print("  - outputs/figures/summary_dashboard.png")
    print("\n")

if __name__ == "__main__":
    main()