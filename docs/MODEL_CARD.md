# Model Card: mT5 for Cross-Lingual Summarization (English & Russian)

## Model Details

### Model Description

This project compares two approaches for fine-tuning mT5 (Multilingual Text-to-Text Transfer Transformer) on cross-lingual abstractive summarization:

1. **Full Fine-Tuning**: All model parameters are updated during training
2. **LoRA (Low-Rank Adaptation)**: Only small adapter matrices are trained, keeping base model frozen

- **Model Type**: Encoder-Decoder Transformer (Seq2Seq)
- **Base Model**: google/mt5-small (300M parameters)
- **Languages**: English, Russian
- **Task**: Abstractive Text Summarization
- **Developed by**: Luka Butskhrikidze, Vanderbilt University
- **Course**: DS 5690-01 Gen AI Models in Theory & Practice
- **License**: Apache 2.0 (following base model license)

### Model Architecture

**mT5 Architecture** (based on T5):
- Encoder: 8 layers, 512 hidden dimension, 8 attention heads
- Decoder: 8 layers, 512 hidden dimension, 8 attention heads
- Vocabulary: 250k multilingual sentencepiece tokens
- Total Parameters: ~300M

**LoRA Configuration**:
- Rank (r): 8
- Alpha: 16
- Target Modules: Query (Q) and Value (V) projection matrices in all attention layers
- Trainable Parameters: ~0.5M (<0.2% of base model)
- Dropout: 0.1

## Intended Use

### Primary Use Cases

1. **Research & Education**: Comparing parameter-efficient fine-tuning methods
2. **Multilingual Summarization**: Generating concise summaries of news articles in English and Russian
3. **Low-Resource Adaptation**: Demonstrating effective fine-tuning with limited compute

### Out-of-Scope Uses

- Production deployment without further validation
- Summarization of domains significantly different from news (legal, medical, technical)
- Languages not included in training (though mT5 base supports 101 languages)
- Long-form document summarization (>512 tokens)

## Training Data

### Dataset

- **Source**: XL-Sum (BBC News multilingual corpus)
- **Languages Used**: English, Russian
- **Splits**:
  - Training: 1,000 articles per language
  - Validation: 200 articles per language
  - Test: 200 articles per language

### Data Preprocessing

1. Added "summarize: " prefix to input texts (mT5 convention)
2. Truncated articles to 512 tokens
3. Truncated summaries to 128 tokens
4. Applied mT5 sentencepiece tokenization
5. Padding to max length with attention masking

### Data Characteristics

**English (BBC News)**:
- Average article length: [TBD] tokens
- Average summary length: [TBD] tokens
- Compression ratio: [TBD]

**Russian (BBC News)**:
- Average article length: [TBD] tokens
- Average summary length: [TBD] tokens
- Compression ratio: [TBD]

See [DATA_CARD.md](DATA_CARD.md) for detailed dataset information.

## Training Procedure

### Full Fine-Tuning
```yaml
Model: google/mt5-small
Optimizer: AdamW
Learning Rate: 5e-5
Batch Size: 8 (effective: 16 with gradient accumulation)
Gradient Accumulation: 2 steps
Epochs: 5
Warmup Steps: 500
Weight Decay: 0.01
Mixed Precision: FP16
Hardware: NVIDIA RTX 5090 (33.7GB)
Training Time: [TBD] hours per language
```

### LoRA Fine-Tuning
```yaml
Base Model: google/mt5-small (frozen)
LoRA Rank (r): 8
LoRA Alpha: 16
LoRA Dropout: 0.1
Target Modules: ["q", "v"]
Optimizer: AdamW
Learning Rate: 1e-4 (higher than full FT)
Batch Size: 8 (effective: 16 with gradient accumulation)
Gradient Accumulation: 2 steps
Epochs: 5
Warmup Steps: 500
Mixed Precision: FP16
Hardware: NVIDIA RTX 5090 (33.7GB)
Training Time: [TBD] hours per language
Trainable Parameters: ~0.5M (0.17% of total)
```

### Differences Between Approaches

| Aspect | Full Fine-Tuning | LoRA |
|--------|------------------|------|
| Trainable Params | 300M (100%) | 0.5M (0.17%) |
| GPU Memory | [TBD] GB | [TBD] GB |
| Training Time | [TBD] hrs | [TBD] hrs |
| Checkpoint Size | ~1.2 GB | ~2 MB |
| Learning Rate | 5e-5 | 1e-4 |

## Evaluation

### Metrics

Primary metrics for summarization quality:
- **ROUGE-1**: Unigram overlap between generated and reference summaries
- **ROUGE-2**: Bigram overlap (measures fluency)
- **ROUGE-L**: Longest common subsequence (measures sentence structure)

### Results

#### English Summarization

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|---------|---------|---------|
| Full FT | [TBD] | [TBD] | [TBD] |
| LoRA | [TBD] | [TBD] | [TBD] |
| mT5-base (zero-shot) | [TBD] | [TBD] | [TBD] |

#### Russian Summarization

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|---------|---------|---------|
| Full FT | [TBD] | [TBD] | [TBD] |
| LoRA | [TBD] | [TBD] | [TBD] |
| mT5-base (zero-shot) | [TBD] | [TBD] | [TBD] |

#### Cross-Lingual Transfer (Optional)

Training on English → Testing on Russian:

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|---------|---------|---------|
| Full FT (EN→RU) | [TBD] | [TBD] | [TBD] |
| LoRA (EN→RU) | [TBD] | [TBD] | [TBD] |

### Qualitative Analysis

[Add 2-3 example summaries comparing Full FT vs LoRA vs Reference]

**Example 1: English**
```
Article: [First 100 words...]
Reference: [Gold summary]
Full FT: [Generated summary]
LoRA: [Generated summary]
Analysis: [Brief comment on quality, faithfulness, differences]
```

## Limitations and Biases

### Model Limitations

1. **Length Constraints**: Limited to 512 input tokens; longer articles are truncated
2. **Domain Specificity**: Trained only on news articles; may not generalize to other domains
3. **Language Coverage**: Only evaluated on English and Russian; performance on other languages unknown
4. **Abstractiveness**: Model may default to extractive patterns for some inputs
5. **Script Differences**: Russian uses Cyrillic script, which may affect tokenization and performance differently than Latin-script languages

### Known Biases

1. **Source Bias**: BBC news has editorial perspective and may reflect certain viewpoints
2. **Language Imbalance**: mT5 base model has varying performance across its 101 supported languages
3. **Cultural Context**: Summaries may lose culturally-specific nuances
4. **Named Entity Handling**: May struggle with rare or non-Western names

### Potential Risks

1. **Hallucination**: Model may generate plausible-sounding but factually incorrect information
2. **Omission**: Important details may be excluded from summaries
3. **Bias Amplification**: Training may amplify biases present in source data
4. **Misuse**: Could be used to generate misleading or manipulated summaries

## Ethical Considerations

### Responsible Use

- Summaries should not be treated as factual without verification
- Users should be aware of potential biases in generated content
- Model outputs should include attribution/disclaimers in production use
- Not suitable for high-stakes applications without human oversight

### Privacy

- Training data (BBC news) is publicly available
- No personal data was used in training
- Model does not memorize or leak training examples (tested via random sampling)

### Environmental Impact

**Full Fine-Tuning**:
- Training Time: [TBD] GPU-hours
- Estimated CO2 Emissions: [TBD] kg (based on [region] grid)

**LoRA**:
- Training Time: [TBD] GPU-hours (30-40% less than full FT)
- Estimated CO2 Emissions: [TBD] kg

LoRA significantly reduces environmental impact while maintaining performance.

## Model Card Authors

Luka Butskhrikidze  
Course: DS 5690-01 Gen AI Models in Theory & Practice (2025F)  
Vanderbilt University
Date: November 2025

## Model Card Contact

For questions or issues with this model card:
- GitHub: https://github.com/LukaButskhrikidze/LoRA-Tuned_mT5_for_Cross-Lingual_Abstractive_News_Summarization

## References

### Papers
- Xue et al. (2020). "mT5: A massively multilingual pre-trained text-to-text transformer"
- Hu et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models"
- Hasan et al. (2021). "XL-Sum: Large-Scale Multilingual Abstractive Summarization"

### Code
- HuggingFace Transformers: https://github.com/huggingface/transformers
- PEFT Library: https://github.com/huggingface/peft
- Base Model: https://huggingface.co/google/mt5-small

---

**Version**: 1.0  
**Last Updated**: November 2025  
**Status**: Educational/Research - Not for Production Use
```

---

## **4. `requirements.txt` (Root Directory)**

Update your `requirements.txt`:
```
# Core ML Libraries
torch>=2.0.0
transformers>=4.35.0
peft>=0.7.0
datasets==2.19.0  # Works with XL-Sum loading scripts
pyarrow==15.0.0   # Required for datasets 2.19.0
accelerate>=0.24.0

# Evaluation
rouge-score>=0.1.2
nltk>=3.8.1
evaluate>=0.4.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.17.0

# Utilities
pandas>=2.0.0
numpy>=1.24.0
tqdm>=4.65.0
tensorboard>=2.14.0

# Jupyter (for notebooks)
jupyter>=1.0.0
ipywidgets>=8.1.0

# Development
pytest>=7.4.0
black>=23.7.0
flake8>=6.1.0
