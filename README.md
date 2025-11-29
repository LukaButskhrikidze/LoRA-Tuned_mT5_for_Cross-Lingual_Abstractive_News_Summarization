# LoRA vs Full Fine-Tuning: mT5 for Cross-Lingual Abstractive Summarization

**Course:** DS 5690-01 Gen AI Models in Theory & Practice  
**Author:** Luka Butskhrikidze  
**Institution:** Vanderbilt University

## 📋 Project Overview

This project compares **Parameter-Efficient Fine-Tuning (LoRA)** with **Full Fine-Tuning** for multilingual abstractive summarization using mT5-small on the XL-Sum dataset. We investigate the performance-efficiency trade-offs and cross-lingual transfer capabilities of both approaches.

## 🎯 Key Findings

### English Results (High-Resource Language)

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Model Size | Trainable Parameters |
|-------|---------|---------|---------|------------|---------------------|
| **Full Fine-Tuning** | 31.09 | 9.42 | **24.02** | 300 MB | 100% (300M) |
| **LoRA (r=8, α=16)** | 26.31 | 5.93 | **20.18** | 5 MB | 0.3% (0.9M) |

**Key Insights:**
- ✅ **LoRA achieves 84% of full fine-tuning performance** (ROUGE-L: 20.18 vs 24.02)
- ✅ **60× smaller model size** (5 MB vs 300 MB)
- ✅ **Similar training time** (~1.5 hours for both)
- ✅ **Massive deployment advantages** for multi-task scenarios

### Russian Results (Medium-Resource Language)

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Status |
|-------|---------|---------|---------|--------|
| **Full Fine-Tuning** | 5.22 | 1.14 | 5.17 | ❌ Failed |
| **LoRA (r=8, α=16)** | 3.08 | 0.71 | 3.07 | ❌ Failed |

**Critical Finding:**
Both methods failed systematically on Russian, revealing that **mT5-small (300M parameters) lacks sufficient capacity for Cyrillic-script languages**, regardless of training approach. This highlights the importance of model scale in multilingual NLP.

## 🔬 Methodology

### Dataset
- **XL-Sum** (Cross-Lingual Summarization Dataset)
  - English: 306,522 training samples
  - Russian: 62,243 training samples

### Model
- **mT5-small** (300M parameters)
- Multilingual Text-to-Text Transfer Transformer

### Training Configuration

**English:**
- Epochs: 3
- Batch size: 4
- Learning rate: 5e-5 (Full) / 1e-4 (LoRA)
- Max sequence length: 512 (input) / 128 (output)

**Russian:**
- Epochs: 5 (to compensate for smaller dataset)
- Other parameters same as English

**LoRA Configuration:**
- Rank (r): 8
- Alpha (α): 16
- Target modules: Query and Value projection layers
- Dropout: 0.1

## 📊 Detailed Analysis

### Performance Trade-offs

1. **Quality vs Efficiency:**
   - LoRA sacrifices ~16% ROUGE-L performance
   - Gains 60× storage reduction
   - Enables multi-task deployment (one base model + many adapters)

2. **Training Dynamics:**
   - Similar training time for small models (mT5-small)
   - LoRA advantages emerge with larger models (>1B parameters)

3. **Cross-Lingual Transfer Limitations:**
   - Model capacity bottleneck more critical than training method
   - Latin-script bias in mT5 pretraining affects Cyrillic performance

### When to Use LoRA vs Full Fine-Tuning

**Use LoRA when:**
- ✅ Working with large models (>1B parameters)
- ✅ Need multiple task-specific models
- ✅ Storage/deployment constraints
- ✅ Acceptable to trade 10-20% performance for efficiency

**Use Full Fine-Tuning when:**
- ✅ Maximum performance required
- ✅ Single-task deployment
- ✅ Model is small enough to fit in memory
- ✅ Storage is not a constraint

## 🚀 Getting Started

### Installation

```bash
# Clone repository
git clone https://github.com/LukaButskhrikidze/LoRA-Tuned_mT5_for_Cross-Lingual_Abstractive_News_Summarization.git
cd LoRA-Tuned_mT5_for_Cross-Lingual_Abstractive_News_Summarization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Data

```bash
python data/download_xlsum.py
```

This downloads English and Russian XL-Sum datasets to `data/xlsum_english/` and `data/xlsum_russian/`.

### Training

**Full Fine-Tuning (English):**
```bash
python src/train_mt5.py \
  --mode full \
  --language english \
  --output_dir outputs/checkpoints/mt5_full_en \
  --num_epochs 3 \
  --batch_size 4 \
  --train_samples 306522 \
  --val_samples 11535
```

**LoRA Fine-Tuning (English):**
```bash
python src/train_mt5.py \
  --mode lora \
  --language english \
  --output_dir outputs/checkpoints/mt5_lora_en \
  --num_epochs 3 \
  --batch_size 4 \
  --train_samples 306522 \
  --val_samples 11535 \
  --lora_r 8 \
  --lora_alpha 16
```

### Evaluation

```bash
python src/evaluation/evaluate_model.py \
  --model_path outputs/checkpoints/mt5_full_en \
  --language english \
  --test_samples 1000 \
  --output_file outputs/results/full_en_test.json
```

### Generate Sample Predictions

```bash
python src/generate_samples.py \
  --model_path outputs/checkpoints/mt5_full_en \
  --language english \
  --num_samples 10 \
  --output_file outputs/results/samples_full_en.txt
```

## 📁 Project Structure

```
.
├── data/
│   ├── xlsum_english/          # English dataset
│   ├── xlsum_russian/          # Russian dataset
│   └── download_xlsum.py       # Data download script
├── src/
│   ├── train_mt5.py           # Main training script
│   ├── evaluation/
│   │   └── evaluate_model.py  # Model evaluation
│   ├── visualization/
│   │   ├── results_comparison.py
│   │   └── training_plots.py
│   └── utils/
├── outputs/
│   ├── checkpoints/           # Trained models
│   ├── results/              # Evaluation results
│   └── figures/              # Visualizations
├── notebooks/
│   └── 01_data_exploration.ipynb
├── scripts/                   # Bash training scripts
├── requirements.txt
└── README.md
```

## 🔍 Key Takeaways

1. **LoRA is Viable for High-Resource Languages:** Achieves 84% of full fine-tuning performance with minimal trainable parameters.

2. **Storage Efficiency Matters:** 60× reduction enables practical multi-task deployment scenarios.

3. **Model Scale is Critical:** For underrepresented languages (Russian/Cyrillic), model capacity matters more than training efficiency.

4. **Future Work:**
   - Test with mT5-base (580M) or mT5-large (1.2B) for Russian
   - Explore higher LoRA ranks (r=16, r=32)
   - Investigate language-specific adapters
   - Test on additional low-resource languages

## 📚 References

- **mT5:** Xue et al. (2021). "mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer"
- **LoRA:** Hu et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models"
- **XL-Sum:** Hasan et al. (2021). "XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages"

## 📧 Contact

**Luka Butskhrikidze**  
Vanderbilt University  
DS 5690-01 Gen AI Models in Theory & Practice

## 📄 License

This project is for academic purposes as part of DS 5690 coursework.
