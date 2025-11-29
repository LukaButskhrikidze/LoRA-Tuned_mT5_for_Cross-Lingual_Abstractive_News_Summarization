"""
Generate sample predictions from trained models for qualitative analysis
Usage: python src/generate_samples.py --model_path outputs/checkpoints/mt5_full_en --language english
"""

import argparse
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--language", type=str, required=True, choices=["english", "russian"])
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--output_file", type=str, default=None, help="Output file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    
    print(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    
    # Try loading as LoRA model first, fall back to regular model
    try:
        base_model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
        model = PeftModel.from_pretrained(base_model, args.model_path)
        print("✅ Loaded LoRA model")
    except:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
        print("✅ Loaded full fine-tuned model")
    
    model = model.cuda() if torch.cuda.is_available() else model
    model.eval()
    
    # Load test data
    print(f"Loading {args.language} test dataset...")
    dataset = load_from_disk(f"data/xlsum_{args.language}")
    test_data = dataset["test"]
    
    # Sample random indices
    indices = random.sample(range(len(test_data)), min(args.num_samples, len(test_data)))
    
    output_lines = []
    output_lines.append("="*100)
    output_lines.append(f"SAMPLE PREDICTIONS - {args.model_path}")
    output_lines.append(f"Language: {args.language.upper()}")
    output_lines.append("="*100)
    output_lines.append("")
    
    print(f"\nGenerating {len(indices)} sample predictions...")
    
    for i, idx in enumerate(indices, 1):
        example = test_data[idx]
        
        # Prepare input
        input_text = "summarize: " + example["text"]
        inputs = tokenizer(input_text, max_length=512, truncation=True, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate summary
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=128,
                num_beams=4,  # Use beam search for better quality
                early_stopping=True
            )
        
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Format output
        output_lines.append(f"SAMPLE {i} (Index: {idx})")
        output_lines.append("-"*100)
        output_lines.append(f"SOURCE TEXT (first 500 chars):")
        output_lines.append(example["text"][:500] + "..." if len(example["text"]) > 500 else example["text"])
        output_lines.append("")
        output_lines.append(f"REFERENCE SUMMARY:")
        output_lines.append(example["summary"])
        output_lines.append("")
        output_lines.append(f"GENERATED SUMMARY:")
        output_lines.append(prediction)
        output_lines.append("")
        output_lines.append("="*100)
        output_lines.append("")
        
        # Print progress
        print(f"  Generated {i}/{len(indices)}")
    
    # Write to file or print
    output_text = "\n".join(output_lines)
    
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"\n✅ Samples saved to {args.output_file}")
    else:
        print("\n" + output_text)

if __name__ == "__main__":
    main()