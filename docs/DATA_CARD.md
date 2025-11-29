# DataCard: XL-Sum Dataset

## Dataset Overview

**Name:** XL-Sum (Cross-Lingual Summarization Dataset)  
**Source:** [Hugging Face Datasets](https://huggingface.co/datasets/csebuetnlp/xlsum)  
**Paper:** Hasan et al. (2021) - "XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages"  
**License:** CC BY-NC-SA 4.0

## Dataset Description

XL-Sum is a comprehensive dataset for abstractive text summarization covering 44 languages. It was created by crawling and extracting professionally written summaries from BBC news articles. The dataset is designed to support multilingual and cross-lingual summarization research.

## Languages Used in This Project

### English (High-Resource)
- **Language Code:** `en`
- **Script:** Latin
- **Training Samples:** 306,522
- **Validation Samples:** 11,535
- **Test Samples:** 11,535
- **Domain:** News articles (BBC)
- **Average Article Length:** ~450 words
- **Average Summary Length:** ~25 words

### Russian (Medium-Resource)
- **Language Code:** `ru`
- **Script:** Cyrillic
- **Training Samples:** 62,243
- **Validation Samples:** 7,780
- **Test Samples:** 7,780
- **Domain:** News articles (BBC)
- **Average Article Length:** ~400 words
- **Average Summary Length:** ~22 words

## Data Structure

Each example in the dataset contains:

```json
{
    "id": "unique_identifier",
    "url": "original_article_url",
    "title": "article_title",
    "summary": "professional_summary",
    "text": "full_article_text"
}
```

## Data Splits

### English
| Split | Samples | Percentage | Usage |
|-------|---------|------------|-------|
| Train | 306,522 | 90.4% | Model training |
| Validation | 11,535 | 3.4% | Hyperparameter tuning |
| Test | 11,535 | 3.4% | Final evaluation |
| **Total** | **329,592** | **100%** | |

### Russian
| Split | Samples | Percentage | Usage |
|-------|---------|------------|-------|
| Train | 62,243 | 80.3% | Model training |
| Validation | 7,780 | 10.0% | Hyperparameter tuning |
| Test | 7,780 | 10.0% | Final evaluation |
| **Total** | **77,803** | **100%** | |

## Dataset Characteristics

### Summary Statistics

**English:**
- Source length (chars): Mean: 2,842 | Median: 2,456 | Max: 15,234
- Summary length (chars): Mean: 156 | Median: 142 | Max: 512
- Compression ratio: ~18:1

**Russian:**
- Source length (chars): Mean: 2,634 | Median: 2,198 | Max: 14,856
- Summary length (chars): Mean: 148 | Median: 135 | Max: 498
- Compression ratio: ~18:1

### Content Domains

Articles cover diverse topics including:
- International news
- Politics
- Business & Economics
- Science & Technology
- Sports
- Entertainment
- Health

### Language-Specific Characteristics

**English:**
- ✅ Latin script - well-represented in mT5 pretraining
- ✅ Large training corpus (306k samples)
- ✅ SentencePiece tokenization efficient
- Expected ROUGE-L: 20-30 for mT5-small

**Russian:**
- ⚠️ Cyrillic script - underrepresented in mT5 pretraining
- ⚠️ Smaller training corpus (62k samples)
- ⚠️ SentencePiece tokenization less efficient
- ⚠️ Morphologically rich (inflections, cases)
- Expected ROUGE-L: 15-25 for mT5-base (failed with mT5-small)

## Data Quality

### Strengths
- ✅ **Professional summaries:** Written by BBC journalists
- ✅ **High coverage:** Diverse news topics
- ✅ **Consistent format:** Standardized structure across languages
- ✅ **Clean text:** Minimal preprocessing needed
- ✅ **Reliable source:** Reputable news organization

### Limitations
- ⚠️ **Domain-specific:** News summarization only
- ⚠️ **Abstractive bias:** Summaries are abstractive, not extractive
- ⚠️ **Temporal bias:** Articles from specific time period
- ⚠️ **Language imbalance:** English has 5× more samples than Russian
- ⚠️ **Single source:** All from BBC (publication bias)

## Data Processing Pipeline

### 1. Download
```bash
python data/download_xlsum.py
```
Downloads and saves to:
- `data/xlsum_english/`
- `data/xlsum_russian/`

### 2. Preprocessing
- Remove empty articles
- Filter by length (min: 100 chars, max: 15,000 chars)
- No text normalization (preserve original)
- No stemming or lemmatization

### 3. Tokenization
- Tokenizer: mT5 SentencePiece (250k vocabulary)
- Max source length: 512 tokens
- Max target length: 128 tokens
- Padding: Right padding to max length
- Special tokens: `<pad>`, `</s>`

### 4. Format for Training
```python
input: "summarize: {article_text}"
target: "{summary_text}"
```

## Usage in This Project

### Training Configuration

**English:**
```python
train_samples = 306,522  # Full dataset
val_samples = 11,535     # Full validation set
test_samples = 11,535    # Reserved for final evaluation
```

**Russian:**
```python
train_samples = 62,243   # Full dataset
val_samples = 7,780      # Full validation set
test_samples = 7,780     # Reserved for final evaluation
```

### Data Loading

```python
from datasets import load_from_disk

# Load dataset
dataset = load_from_disk("data/xlsum_english")

# Access splits
train_data = dataset["train"]
val_data = dataset["validation"]
test_data = dataset["test"]

# Example
example = train_data[0]
print(f"Text: {example['text']}")
print(f"Summary: {example['summary']}")
```

## Ethical Considerations

### Bias
- **Geographic bias:** BBC perspective (UK-centric)
- **Language bias:** English overrepresented (5× more data)
- **Temporal bias:** News from specific time period
- **Topic bias:** Mainstream news topics

### Privacy
- All content from public BBC articles
- No personal data collected
- Summaries written by professional journalists

### Intended Use
- ✅ Academic research
- ✅ Summarization model training
- ✅ Cross-lingual NLP evaluation
- ❌ Not for production without additional validation
- ❌ Not representative of all news sources

## Citation

If you use XL-Sum in your research, please cite:

```bibtex
@inproceedings{hasan-etal-2021-xl,
    title = "{XL}-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages",
    author = "Hasan, Tahmid and Bhattacharjee, Abhik and Islam, Md. Saiful and Mubasshir, Kazi and Li, Yuan-Fang and Kang, Yong-Bin and Rahman, M. Sohel and Shahriyar, Rifat",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.413",
    pages = "4693--4703",
}
```

## Dataset Access

**Hugging Face:** `csebuetnlp/xlsum`  
**Languages:** 44 total (we use: `english`, `russian`)  
**Format:** Arrow/Parquet  
**Size:** ~2.3 GB (English), ~450 MB (Russian)

## Version Information

- **Dataset Version:** 1.0
- **Download Date:** November 2024
- **Processing Version:** As used in this project
- **Last Updated:** November 29, 2024

## Observed Performance

Based on our experiments with mT5-small:

| Language | Full Fine-Tuning | LoRA | Data Quality |
|----------|------------------|------|--------------|
| English | 24.02 ROUGE-L | 20.18 ROUGE-L | ✅ Excellent |
| Russian | 5.17 ROUGE-L | 3.07 ROUGE-L | ⚠️ Model capacity issue |

**Note:** Russian performance indicates model capacity limitation, not data quality issue. The dataset itself is high-quality but requires larger models (mT5-base or mT5-large) for effective learning.

## Contact

For questions about dataset usage in this project:
- **Author:** Luka Butskhrikidze
- **Course:** DS 5690-01 Gen AI Models in Theory & Practice
- **Institution:** Vanderbilt University

For questions about the XL-Sum dataset itself, refer to the original paper or Hugging Face dataset page.
