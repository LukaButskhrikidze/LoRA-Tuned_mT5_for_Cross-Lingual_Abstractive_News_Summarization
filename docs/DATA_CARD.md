# Data Card: XL-Sum Dataset (English & Russian)

## Dataset Description

**Name**: XL-Sum (Cross-Lingual Summarization)  
**Source**: BBC News Articles  
**Languages**: English, Russian  
**Task**: Abstractive Text Summarization  
**License**: CC BY-NC-SA 4.0

## Dataset Structure

### Languages Used

**English**
- Training samples: 306,522
- Validation samples: 11,535
- Test samples: 11,535

**Russian**
- Training samples: 62,243
- Validation samples: 2,334
- Test samples: 2,334

### Data Fields

Each example contains:
- `id`: Unique identifier (string)
- `url`: Source article URL (string)
- `title`: Article title (string)
- `text`: Full article text (string)
- `summary`: Reference summary (string)

### Data Splits
```python
{
    'train': Dataset,      # Training set
    'validation': Dataset,  # Validation set
    'test': Dataset        # Test set
}
```

## Data Collection

**Source**: BBC News multilingual website  
**Collection Period**: Various dates (see original paper)  
**Collection Method**: Web scraping and preprocessing  
**Annotators**: Professional journalists (original article authors)

## Data Characteristics

### English Statistics
- Average article length: ~700 tokens
- Average summary length: ~30 tokens
- Compression ratio: ~23:1

### Russian Statistics  
- Average article length: ~650 tokens
- Average summary length: ~25 tokens
- Compression ratio: ~26:1

### Language-Specific Notes

**English**:
- Standard modern English
- News domain vocabulary
- Formal writing style

**Russian**:
- Cyrillic script
- Standard modern Russian
- News domain vocabulary
- Formal writing style

## Preprocessing

### Applied Transformations
1. Text normalization (whitespace, encoding)
2. Added "summarize: " prefix for mT5 (at training time)
3. Truncation: max 512 tokens (articles), max 128 tokens (summaries)
4. Tokenization: mT5 sentencepiece tokenizer

### Quality Filters
- Removed empty articles/summaries
- Removed duplicates
- Length constraints applied

## Ethical Considerations

### Potential Biases

**Source Bias**:
- BBC editorial perspective
- Western-centric news coverage
- May not represent all viewpoints equally

**Language Bias**:
- English has 5x more samples than Russian
- May lead to better English performance

**Domain Limitation**:
- News domain only
- May not generalize to other genres

### Privacy
- All data is publicly available news articles
- No personal data collection
- URLs preserved for verification

## Limitations

1. **Domain**: Limited to news articles only
2. **Temporal**: May contain dated references
3. **Cultural Context**: News framing may differ by language
4. **Quality**: Some articles may have formatting artifacts
5. **Balance**: English >> Russian in dataset size

## Usage Recommendations

✅ **Appropriate Uses**:
- News summarization research
- Cross-lingual NLP experiments
- Model comparison studies
- Educational purposes

❌ **Inappropriate Uses**:
- Production systems without validation
- High-stakes decision making
- Domains outside news (medical, legal)
- Generating misleading content

## Citation
```bibtex
@inproceedings{hasan-etal-2021-xl,
    title = "{XL}-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages",
    author = "Hasan, Tahmid and others",
    booktitle = "Findings of ACL-IJCNLP 2021",
    year = "2021",
    publisher = "Association for Computational Linguistics",
}
```

## Access

**HuggingFace Hub**: `csebuetnlp/xlsum`  
**Website**: https://github.com/csebuetnlp/xl-sum

## Contact

For dataset issues: See original XL-Sum repository  
For project-specific questions: Luka Butskhrikidze

---

**Last Updated**: November 2025  
**Version**: 1.0
