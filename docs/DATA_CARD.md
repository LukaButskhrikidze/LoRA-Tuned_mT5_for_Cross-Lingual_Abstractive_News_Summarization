# ModelCard: mT5 Fine-Tuned Models for Cross-Lingual Summarization

## Model Overview

This project trains and compares four mT5-small models for abstractive news summarization using two fine-tuning approaches (Full Fine-Tuning and LoRA) on two languages (English and Russian).

### Models Trained

| Model ID | Base Model | Method | Language | Status |
|----------|------------|--------|----------|--------|
| `mt5_full_en` | mT5-small | Full Fine-Tuning | English | ✅ Successful |
| `mt5_lora_en` | mT5-small | LoRA (r=8, α=16) | English | ✅ Successful |
| `mt5_full_ru` | mT5-small | Full Fine-Tuning | Russian | ❌ Failed |
| `mt5_lora_ru` | mT5-small | LoRA (r=8, α=16) | Russian | ❌ Failed |

## Base Model Information

### mT5-small

**Source:** Google Research  
**Paper:** Xue et al. (2021) - "mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer"  
**Architecture:** Encoder-Decoder Transformer (T5 variant)  
**Parameters:** 300 million (300M)  
**Pretraining:** Multilingual C4 (mC4) covering 101 languages  
**Tokenizer:** SentencePiece with 250k vocabulary  

**Model Specifications:**
- Encoder layers: 8
- Decoder layers: 8
- Hidden size: 512
- Feed-forward size: 1,024
- Attention heads: 6
- Max sequence length: 1,024 tokens

## Training Details

### English Models (Successful)

#### Full Fine-Tuning Configuration
```yaml
Training Data: 306,522 samples (XL-Sum English)
Validation Data: 11,535 samples
Epochs: 3
Batch Size: 4
Gradient Accumulation: 1
Learning Rate: 5e-5
Warmup Steps: 100
Weight Decay: 0.01
Optimizer: AdamW
LR Schedule: Linear decay with warmup
Max Source Length: 512 tokens
Max Target Length: 128 tokens
Training Time: 1h 28m (RTX 5090)
GPU Memory: ~18 GB peak
Total Steps: 229,893
Trainable Parameters: 300M (100%)
```

**Final Metrics:**
- Validation ROUGE-1: 31.09
- Validation ROUGE-2: 9.42
- Validation ROUGE-L: 24.02
- Validation Loss: 2.26
- Model Size: 300 MB

#### LoRA Fine-Tuning Configuration
```yaml
Training Data: 306,522 samples (XL-Sum English)
Validation Data: 11,535 samples
Epochs: 3
Batch Size: 4
Gradient Accumulation: 1
Learning Rate: 1e-4  # 2× higher than full FT
Warmup Steps: 100
Weight Decay: 0.01
Optimizer: AdamW
LR Schedule: Linear decay with warmup
Max Source Length: 512 tokens
Max Target Length: 128 tokens
Training Time: ~1h 30m (RTX 5090)
GPU Memory: ~12 GB peak
Total Steps: 229,893
Trainable Parameters: 0.9M (0.3%)

LoRA Parameters:
  Rank (r): 8
  Alpha (α): 16
  Dropout: 0.1
  Target Modules: ['q', 'v']  # Query and Value projections
  Task Type: SEQ_2_SEQ_LM
```

**Final Metrics:**
- Validation ROUGE-1: 26.31
- Validation ROUGE-2: 5.93
- Validation ROUGE-L: 20.18
- Validation Loss: 2.90
- Model Size: 5 MB (adapter only)

### Russian Models (Failed)

#### Training Configuration
```yaml
Training Data: 62,243 samples (XL-Sum Russian)
Validation Data: 7,780 samples
Epochs: 5  # Increased due to smaller dataset
Batch Size: 4
Other parameters: Same as English
```

**Full Fine-Tuning Results:**
- Validation ROUGE-L: 5.17 ❌
- Loss: 2.31 (reasonable, but generates nonsense)

**LoRA Results:**
- Validation ROUGE-L: 3.07 ❌
- Loss: 2.88 (reasonable, but generates nonsense)

**Failure Analysis:** Both methods failed systematically, indicating mT5-small lacks capacity for Russian/Cyrillic summarization. Generated fluent but factually incorrect text with severe hallucinations.

## Performance Comparison

### English Models

| Metric | Full FT | LoRA | Gap | Retention |
|--------|---------|------|-----|-----------|
| **ROUGE-1** | 31.09 | 26.31 | -4.78 | 84.6% |
| **ROUGE-2** | 9.42 | 5.93 | -3.49 | 63.0% |
| **ROUGE-L** | 24.02 | 20.18 | -3.84 | **84.0%** |
| **Model Size** | 300 MB | 5 MB | -295 MB | 1.7% |
| **Trainable %** | 100% | 0.3% | -99.7% | 0.3% |

**Key Finding:** LoRA retains 84% of full fine-tuning performance with 60× smaller model size.

### Russian Models

Both approaches failed with ROUGE-L scores near random baseline (<5), confirming that model capacity is the limiting factor for Cyrillic languages, not training method.

## Model Architecture

### Full Fine-Tuning
```
Input → mT5 Encoder (fine-tuned) → 
        mT5 Decoder (fine-tuned) → 
        Output
        
All 300M parameters updated during training
```

### LoRA Fine-Tuning
```
Input → mT5 Encoder (frozen) + LoRA Adapters →
        mT5 Decoder (frozen) + LoRA Adapters →
        Output

Base model frozen (299.1M params)
Only LoRA adapters trained (0.9M params)

LoRA Structure per attention layer:
  Q_base + (LoRA_down_Q @ LoRA_up_Q) * (α/r)
  V_base + (LoRA_down_V @ LoRA_up_V) * (α/r)
  
Where:
  LoRA_down: [hidden_size × r] = [512 × 8]
  LoRA_up: [r × hidden_size] = [8 × 512]
  Scaling factor: α/r = 16/8 = 2
```

## Intended Use

### Primary Use Cases
- ✅ **Abstractive summarization** of English news articles
- ✅ **Research comparison** of Full vs LoRA fine-tuning
- ✅ **Educational demonstration** of PEFT methods
- ✅ **Multi-task deployment** scenarios (LoRA advantage)

### Out-of-Scope Use
- ❌ Production deployment without further testing
- ❌ Russian summarization (requires larger model)
- ❌ Non-news domains without adaptation
- ❌ Extractive summarization
- ❌ Languages other than English

## Limitations and Biases

### Known Limitations

**English Models:**
1. **Performance gap:** LoRA is 16% lower than full fine-tuning (ROUGE-L)
2. **Domain specificity:** Trained only on news articles
3. **Length bias:** Optimized for summaries ~25 words
4. **Abstractiveness:** May over-abstract or hallucinate facts

**Russian Models:**
1. **Complete failure:** ROUGE-L ~3-5 (near random)
2. **Hallucination:** Generates fluent but factually incorrect text
3. **Model capacity:** mT5-small insufficient for Cyrillic
4. **Recommendation:** Use mT5-base (580M) or larger

### Inherited Biases

From mT5 pretraining:
- **Language bias:** Latin-script languages overrepresented
- **Geographic bias:** Western sources dominant in training
- **Temporal bias:** Training data from specific time period
- **Domain bias:** Web text may not represent all writing styles

From XL-Sum dataset:
- **Source bias:** BBC articles only (UK perspective)
- **Topic bias:** Mainstream news topics
- **Style bias:** Professional journalistic writing

### Performance Characteristics

**What the models do well:**
- ✅ Capture main events and entities
- ✅ Generate grammatically correct summaries
- ✅ Maintain factual accuracy (for English)
- ✅ Handle various news topics

**What the models struggle with:**
- ⚠️ Complex multi-sentence reasoning
- ⚠️ Exact quote preservation
- ⚠️ Very long documents (>512 tokens)
- ⚠️ Domain-specific jargon
- ⚠️ Rare entities or events

## Evaluation Methodology

### Automatic Metrics

**ROUGE Scores (primary metrics):**
- ROUGE-1: Unigram overlap
- ROUGE-2: Bigram overlap
- ROUGE-L: Longest common subsequence
- Configuration: With stemming for English, without for Russian

**Evaluation Set:**
- English validation: 11,535 samples
- Russian validation: 7,780 samples
- Test set: Reserved for final evaluation

### Qualitative Analysis

Sample predictions available in:
- `outputs/results/samples/full_en_samples.txt`
- `outputs/results/samples/lora_en_samples.txt`

**Evaluation criteria:**
1. Factual accuracy
2. Fluency and grammar
3. Coverage of key information
4. Abstractiveness vs extractiveness
5. Length appropriateness

## Environmental Impact

### Carbon Footprint (Estimated)

**Training Compute:**
- Hardware: 1× NVIDIA RTX 5090 (575W TDP)
- Training time: ~5 hours total (all 4 models)
- Energy consumption: ~2.9 kWh
- Estimated CO₂: ~1.2 kg CO₂eq (US average grid)

**Inference Compute:**
- Generation time: ~0.5 seconds per summary
- Batch inference: ~100 summaries/minute
- Relatively efficient for deployment

## Usage Examples

### Loading the Models

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# Load Full Fine-Tuned Model
tokenizer = AutoTokenizer.from_pretrained("outputs/checkpoints/mt5_full_en")
model = AutoModelForSeq2SeqLM.from_pretrained("outputs/checkpoints/mt5_full_en")

# Load LoRA Model
base_model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
model = PeftModel.from_pretrained(base_model, "outputs/checkpoints/mt5_lora_en")
```

### Generating Summaries

```python
def generate_summary(text, model, tokenizer, max_length=128):
    """Generate abstractive summary"""
    # Prepare input
    input_text = f"summarize: {text}"
    inputs = tokenizer(
        input_text,
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )
    
    # Generate
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3
    )
    
    # Decode
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

# Example usage
article = "Your news article text here..."
summary = generate_summary(article, model, tokenizer)
print(f"Summary: {summary}")
```

### Model Selection Guide

```python
# Choose model based on requirements:

# Scenario 1: Single task, maximum quality
model_path = "outputs/checkpoints/mt5_full_en"  # ROUGE-L: 24.02

# Scenario 2: Multiple tasks, storage constrained
model_path = "outputs/checkpoints/mt5_lora_en"  # ROUGE-L: 20.18, 60× smaller

# Scenario 3: Russian summarization
# DON'T use mT5-small - requires mT5-base or larger
```

## Reproducibility

### Hardware Requirements
- **Minimum:** 16 GB GPU memory (for inference)
- **Recommended:** 24 GB GPU memory (for training)
- **Used in project:** NVIDIA RTX 5090 (32 GB)

### Software Environment
```
Python: 3.10+
PyTorch: 2.0+
Transformers: 4.35+
PEFT: 0.7+
Datasets: 2.14+
Evaluate: 0.4+
```

See `requirements.txt` for exact versions.

### Random Seeds
- Training seed: Default (varies per run)
- Evaluation seed: 42 (for sample generation)

### Reproducibility Notes
- Results may vary slightly due to hardware differences
- GPU type affects training speed but not final metrics
- Batch size affects training dynamics (we used 4)

## Model Updates and Versions

**Current Version:** 1.0 (November 2024)

**Future Improvements:**
- [ ] Test set evaluation (currently validation only)
- [ ] Larger batch sizes with gradient accumulation
- [ ] Higher LoRA ranks (r=16, r=32)
- [ ] mT5-base for Russian
- [ ] Beam search tuning
- [ ] Domain adaptation experiments

## Citation

If you use these models or findings in your research:

```bibtex
@misc{butskhrikidze2024lora_mt5,
  author = {Butskhrikidze, Luka},
  title = {LoRA vs Full Fine-Tuning: mT5 for Cross-Lingual Summarization},
  year = {2024},
  institution = {Vanderbilt University},
  course = {DS 5690-01 Gen AI Models in Theory & Practice},
  url = {https://github.com/LukaButskhrikidze/LoRA-Tuned_mT5_for_Cross-Lingual_Abstractive_News_Summarization}
}
```

Also cite the base model and dataset:

```bibtex
@article{xue2021mt5,
  title={mT5: A massively multilingual pre-trained text-to-text transformer},
  author={Xue, Linting and Constant, Noah and Roberts, Adam and Kale, Mihir and Al-Rfou, Rami and Siddhant, Aditya and Barua, Aditya and Raffel, Colin},
  journal={arXiv preprint arXiv:2010.11934},
  year={2020}
}

@inproceedings{hasan-etal-2021-xl,
    title = "{XL}-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages",
    author = "Hasan, Tahmid and others",
    booktitle = "Findings of ACL-IJCNLP 2021",
    year = "2021"
}
```

## Contact and Support

**Author:** Luka Butskhrikidze  
**Institution:** Vanderbilt University  
**Course:** DS 5690-01 Gen AI Models in Theory & Practice  
**Date:** November 2024

**Repository:** https://github.com/LukaButskhrikidze/LoRA-Tuned_mT5_for_Cross-Lingual_Abstractive_News_Summarization

For questions about this project, please refer to the repository documentation or course materials.

## Acknowledgments

- Google Research for mT5
- Hugging Face for transformers and PEFT libraries
- XL-Sum dataset authors
- Vanderbilt University DS 5690 course staff
- Open-source ML community

---

**Model Status:** Research/Educational  
**Last Updated:** November 29, 2024  
**Model Version:** 1.0
