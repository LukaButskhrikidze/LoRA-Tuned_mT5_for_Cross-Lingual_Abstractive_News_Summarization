"""
Data exploration script for XL-Sum datasets
Analyzes text lengths, generates visualizations, and prints statistics
Usage: python src/explore_data.py
"""

from datasets import load_from_disk
import matplotlib.pyplot as plt
import os
import numpy as np


def analyze_dataset(dataset, language_name, num_samples=1000):
    """Analyze text and summary lengths"""
    print(f"\n{'='*70}")
    print(f"Analyzing {language_name} Dataset")
    print(f"{'='*70}")
    
    # Get lengths (use min to avoid errors if dataset is smaller)
    num_samples = min(num_samples, len(dataset['train']))
    
    text_lengths = []
    summary_lengths = []
    
    for i in range(num_samples):
        sample = dataset['train'][i]
        text_lengths.append(len(sample['text']))
        summary_lengths.append(len(sample['summary']))
    
    # Calculate statistics
    stats = {
        'text_mean': np.mean(text_lengths),
        'text_median': np.median(text_lengths),
        'text_max': np.max(text_lengths),
        'text_min': np.min(text_lengths),
        'summary_mean': np.mean(summary_lengths),
        'summary_median': np.median(summary_lengths),
        'summary_max': np.max(summary_lengths),
        'summary_min': np.min(summary_lengths),
        'compression_ratio': np.mean(text_lengths) / np.mean(summary_lengths)
    }
    
    # Print statistics
    print(f"\n📊 Text Statistics (based on {num_samples} samples):")
    print(f"   Mean length:   {stats['text_mean']:>8,.0f} characters")
    print(f"   Median length: {stats['text_median']:>8,.0f} characters")
    print(f"   Max length:    {stats['text_max']:>8,.0f} characters")
    print(f"   Min length:    {stats['text_min']:>8,.0f} characters")
    
    print(f"\n📝 Summary Statistics:")
    print(f"   Mean length:   {stats['summary_mean']:>8,.0f} characters")
    print(f"   Median length: {stats['summary_median']:>8,.0f} characters")
    print(f"   Max length:    {stats['summary_max']:>8,.0f} characters")
    print(f"   Min length:    {stats['summary_min']:>8,.0f} characters")
    
    print(f"\n📉 Compression Ratio: {stats['compression_ratio']:.2f}x")
    print(f"   (Original text is {stats['compression_ratio']:.2f}x longer than summary)")
    
    return text_lengths, summary_lengths, stats


def plot_distributions(en_text, en_summ, ru_text, ru_summ, output_dir="outputs/figures"):
    """Create distribution plots"""
    print(f"\n📊 Generating distribution plots...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('XL-Sum Dataset: Text and Summary Length Distributions', 
                 fontsize=16, fontweight='bold')
    
    # English text lengths
    axes[0, 0].hist(en_text, bins=50, alpha=0.7, color='#2E86AB', edgecolor='black')
    axes[0, 0].set_title('English: Article Lengths', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Characters')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(np.mean(en_text), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(en_text):.0f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # English summary lengths
    axes[0, 1].hist(en_summ, bins=50, alpha=0.7, color='#A23B72', edgecolor='black')
    axes[0, 1].set_title('English: Summary Lengths', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Characters')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(np.mean(en_summ), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(en_summ):.0f}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Russian text lengths
    axes[1, 0].hist(ru_text, bins=50, alpha=0.7, color='#18A558', edgecolor='black')
    axes[1, 0].set_title('Russian: Article Lengths', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Characters')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].axvline(np.mean(ru_text), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(ru_text):.0f}')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Russian summary lengths
    axes[1, 1].hist(ru_summ, bins=50, alpha=0.7, color='#F18F01', edgecolor='black')
    axes[1, 1].set_title('Russian: Summary Lengths', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Characters')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].axvline(np.mean(ru_summ), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(ru_summ):.0f}')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'data_distributions.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved plot to: {output_path}")
    
    plt.close()


def print_examples(dataset, language_name, num_examples=3):
    """Print example articles and summaries"""
    print(f"\n{'='*70}")
    print(f"Sample {language_name} Examples")
    print(f"{'='*70}")
    
    for i in range(min(num_examples, len(dataset['train']))):
        sample = dataset['train'][i]
        
        print(f"\n--- Example {i+1} ---")
        print(f"\n📰 Article ({len(sample['text'])} characters):")
        print("-" * 70)
        # Print first 300 characters
        text_preview = sample['text'][:300] + "..." if len(sample['text']) > 300 else sample['text']
        print(text_preview)
        
        print(f"\n📝 Summary ({len(sample['summary'])} characters):")
        print("-" * 70)
        print(sample['summary'])
        print()


def main():
    """Main exploration function"""
    print("\n" + "="*70)
    print("XL-Sum Data Exploration")
    print("="*70)
    
    # Load datasets
    print("\n📂 Loading datasets...")
    try:
        ds_english = load_from_disk("data/xlsum_english")
        print(f"✅ Loaded English dataset")
    except Exception as e:
        print(f"❌ Error loading English dataset: {e}")
        return
    
    try:
        ds_russian = load_from_disk("data/xlsum_russian")
        print(f"✅ Loaded Russian dataset")
    except Exception as e:
        print(f"❌ Error loading Russian dataset: {e}")
        return
    
    # Analyze datasets
    en_text, en_summ, en_stats = analyze_dataset(ds_english, "English")
    ru_text, ru_summ, ru_stats = analyze_dataset(ds_russian, "Russian")
    
    # Create visualizations
    plot_distributions(en_text, en_summ, ru_text, ru_summ)
    
    # Print examples
    print_examples(ds_english, "English", num_examples=2)
    print_examples(ds_russian, "Russian", num_examples=2)
    
    # Print comparison
    print("\n" + "="*70)
    print("Comparison Summary")
    print("="*70)
    print(f"\n{'Metric':<30} {'English':>15} {'Russian':>15}")
    print("-" * 70)
    print(f"{'Mean Article Length':<30} {en_stats['text_mean']:>15,.0f} {ru_stats['text_mean']:>15,.0f}")
    print(f"{'Mean Summary Length':<30} {en_stats['summary_mean']:>15,.0f} {ru_stats['summary_mean']:>15,.0f}")
    print(f"{'Compression Ratio':<30} {en_stats['compression_ratio']:>15.2f}x {ru_stats['compression_ratio']:>15.2f}x")
    print(f"{'Training Samples':<30} {len(ds_english['train']):>15,} {len(ds_russian['train']):>15,}")
    print(f"{'Validation Samples':<30} {len(ds_english['validation']):>15,} {len(ds_russian['validation']):>15,}")
    print(f"{'Test Samples':<30} {len(ds_english['test']):>15,} {len(ds_russian['test']):>15,}")
    print("="*70)
    
    print("\n✅ Exploration complete!")
    print("📊 Visualization saved to: outputs/figures/data_distributions.png\n")


if __name__ == "__main__":
    main()