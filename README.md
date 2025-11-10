# Cross-Lingual Summarization: mT5 Fine-Tuning vs LoRA

**Course**: DS 5690-01 Gen AI Models in Theory & Practice (2025F)  
**Focus**: Parameter-Efficient Fine-Tuning with LoRA on Encoder-Decoder Transformers

---

## 📋 Project Overview

This project implements and compares **full fine-tuning** vs **LoRA (Low-Rank Adaptation)** for cross-lingual abstractive summarization using mT5 on the XL-Sum dataset (English and Italian).

**Research Question**: How does LoRA's parameter efficiency compare to full fine-tuning in terms of performance, training cost, and generalization for multilingual seq2seq tasks?

---

## 🎯 Problem Statement

Large multilingual models like mT5 contain billions of parameters, making full fine-tuning computationally expensive and impractical for many applications. LoRA offers a parameter-efficient alternative by training only low-rank adapter matrices. This project empirically evaluates the trade-offs between these approaches for cross-lingual news summarization.

---

## 🏗️ Repository Structure

```
mt5-lora-summarization/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup.sh                          # Environment setup script
│
├── data/
│   ├── download_xlsum.py             # Script to download XL-Sum data
│   └── preprocess.py                 # Data preprocessing utilities
│
├── src/
│   ├── models/
│   │   ├── mt5_full.py              # Full fine-tuning trainer
│   │   ├── mt5_lora.py              # LoRA fine-tuning trainer
│   │   └── config.py                # Model configurations
│   │
│   ├── evaluation/
│   │   ├── rouge_eval.py            # ROUGE metric computation
│   │   └── analysis.py              # Result analysis utilities
│   │
│   ├── visualization/
│   │   ├── attention_viz.py         # Attention pattern visualization
│   │   └── training_plots.py        # Training metrics visualization
│   │
│   └── utils/
│       ├── tokenizer.py             # Tokenization utilities
│       └── logger.py                # Logging setup
│
├── scripts/
│   ├── train_full.sh                # Run full fine-tuning
│   ├── train_lora.sh                # Run LoRA fine-tuning
│   ├── evaluate.sh                  # Run evaluation
│   └── cross_lingual_test.sh        # Cross-lingual transfer test
│
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Dataset analysis
│   ├── 02_model_comparison.ipynb    # Results comparison
│   └── 03_attention_analysis.ipynb  # Attention visualization
│
├── outputs/
│   ├── checkpoints/                 # Model checkpoints
│   ├── results/                     # Evaluation results
│   └── figures/                     # Generated plots
│
├── docs/
│   ├── MODEL_CARD.md               # Model documentation
│   ├── DATA_CARD.md                # Dataset documentation
│   └── METHODOLOGY.md              # Detailed methodology
│
└── presentation/
    ├── slides.pdf                   # Final presentation
    └── demo_examples.md             # Demo examples for presentation
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- CUDA-capable GPU (16GB+ VRAM recommended)
- 50GB free disk space

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

# Download data
python data/download_xlsum.py --languages en it --output_dir data/xlsum
```

### Training

**Full Fine-Tuning (English)**
```bash
bash scripts/train_full.sh \
  --language en \
  --model_name google/mt5-small \
  --output_dir outputs/checkpoints/mt5_full_en \
  --num_epochs 5 \
  --batch_size 8
```

**LoRA Fine-Tuning (English)**
```bash
bash scripts/train_lora.sh \
  --language en \
  --model_name google/mt5-small \
  --lora_r 8 \
  --lora_alpha 16 \
  --output_dir outputs/checkpoints/mt5_lora_en \
  --num_epochs 5 \
  --batch_size 8
```

### Evaluation

```bash
bash scripts/evaluate.sh \
  --model_path outputs/checkpoints/mt5_lora_en \
  --test_language en \
  --output_file outputs/results/lora_en_results.json
```

---

## 📊 Methodology

### Connection to Course Content

**1. Transformer Architecture (Encoder-Decoder)**
- mT5 implements the full encoder-decoder architecture covered in class
- Encoder processes input (article), decoder generates output (summary)
- Cross-attention mechanism connects encoder and decoder

**2. LoRA (Parameter-Efficient Fine-Tuning)**
- Applies low-rank decomposition: ΔW = BA where B ∈ ℝ^(d×r), A ∈ ℝ^(r×k)
- Targets attention matrices (Q, K, V, O) in both encoder and decoder
- Rank r controls the trade-off between capacity and efficiency

**3. Training Configuration**
```python
Full Fine-Tuning:
- All parameters trainable (~300M for mT5-small)
- Standard AdamW optimizer
- Learning rate: 5e-5

LoRA:
- Only adapter parameters trainable (~0.5M, <1% of model)
- Rank r = 8, alpha = 16
- Learning rate: 1e-4 (can use higher LR for LoRA)
```

### Experimental Design

**Languages**: English (EN), Italian (IT)
**Dataset**: XL-Sum BBC news articles (train/val/test splits)
**Metrics**: 
- ROUGE-1, ROUGE-2, ROUGE-L (primary)
- Trainable parameters (%)
- Training time per epoch
- GPU memory usage
- Inference speed

**Comparisons**:
1. Full FT (EN) vs LoRA (EN) - Same language performance
2. Full FT (IT) vs LoRA (IT) - Generalization to second language
3. Cross-lingual: Train EN → Test IT (optional if time permits)

---

## 📈 Expected Results

### Performance Comparison Table
| Model | Language | ROUGE-1 | ROUGE-2 | ROUGE-L | Trainable Params | Train Time |
|-------|----------|---------|---------|---------|------------------|------------|
| Full FT | EN | TBD | TBD | TBD | 100% | TBD |
| LoRA | EN | TBD | TBD | TBD | ~0.5% | TBD |
| Full FT | IT | TBD | TBD | TBD | 100% | TBD |
| LoRA | IT | TBD | TBD | TBD | ~0.5% | TBD |

### Efficiency Metrics
- **Memory**: LoRA uses ~30-40% less VRAM during training
- **Speed**: LoRA trains ~20-30% faster per epoch
- **Storage**: LoRA checkpoints are 200x smaller (~2MB vs 400MB)

---

## 🔍 Analysis & Visualization

### 1. Attention Pattern Analysis
- Visualize encoder self-attention for source article
- Visualize decoder cross-attention (which source tokens influence summary)
- Compare attention distributions between full FT and LoRA

### 2. Training Dynamics
- Loss curves comparison
- Learning rate schedules
- Convergence speed analysis

### 3. Qualitative Examples
- Side-by-side summary comparisons
- Error analysis (hallucinations, faithfulness issues)
- Language-specific observations

---

## 📄 Model & Data Cards

See detailed documentation:
- **Model Card**: [docs/MODEL_CARD.md](docs/MODEL_CARD.md)
- **Data Card**: [docs/DATA_CARD.md](docs/DATA_CARD.md)

### Quick Summary

**Model**:
- Base: google/mt5-small (300M parameters)
- Versions: Full fine-tuned, LoRA-adapted
- Languages: English, Italian
- License: Apache 2.0

**Data**:
- XL-Sum BBC news corpus
- Train: 1,000 articles per language
- Val: 200 articles per language
- Test: 200 articles per language
- License: CC BY-NC-SA 4.0

---

## 🎓 Course Connections

### Topics Covered
1. **Attention Mechanisms**: Multi-head self-attention and cross-attention in encoder-decoder
2. **Transformer Architecture**: Positional encoding, layer normalization, feed-forward networks
3. **Fine-Tuning Strategies**: Full fine-tuning vs parameter-efficient methods
4. **LoRA Theory**: Low-rank matrix decomposition, adapter injection points
5. **Multilingual Models**: Shared vocabulary, cross-lingual transfer

### Learning Objectives Demonstrated
- Implement and compare training paradigms for large language models
- Analyze attention patterns in transformer architectures
- Evaluate trade-offs between model performance and computational efficiency
- Apply parameter-efficient fine-tuning techniques to real-world tasks

---

## 🔮 Critical Analysis & Future Work

### Impact
- Demonstrates practical viability of LoRA for resource-constrained environments
- Provides empirical evidence for when LoRA matches full fine-tuning performance
- Enables easier deployment and iteration for multilingual applications

### Key Findings
1. [To be filled after experiments]
2. LoRA achieves XX% of full FT performance with <1% trainable parameters
3. Cross-attention patterns show [observation]

### Limitations
- Limited to 2 languages (EN, IT)
- Small subset of XL-Sum dataset
- mT5-small may not capture full multilingual capabilities

### Next Steps
- Extend to more typologically diverse languages (e.g., Arabic, Chinese)
- Experiment with LoRA rank ablation (r = 4, 8, 16, 32)
- Implement QLoRA for even more memory efficiency
- Add faithfulness metrics (BERTScore, QuestEval)

---

## 📚 References

### Papers
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)
- [mT5: A massively multilingual pre-trained text-to-text transformer](https://arxiv.org/abs/2010.11934) (Xue et al., 2020)
- [XL-Sum: Large-Scale Multilingual Abstractive Summarization](https://arxiv.org/abs/2106.13822) (Hasan et al., 2021)

### Code Resources
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [PEFT: Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)
- [XL-Sum Dataset](https://huggingface.co/datasets/csebuetnlp/xlsum)

---

## 👥 Author

**Luka Butskhrikidze**  
Course: DS 5690-01 Gen AI Models in Theory & Practice (2025F)  
Instructor: Jesse Spencer-Smith  
Date: November 2025

---

## 📝 License

This project is for educational purposes as part of coursework at Vanderbilt University.

Code: MIT License  
Models: Subject to original model licenses (Apache 2.0 for mT5)  
Data: CC BY-NC-SA 4.0 (XL-Sum)
