"""
Evaluate trained mT5 models
Usage: python src/evaluate_model.py --model_path outputs/checkpoints/mt5_lora_en --language english
"""

import argparse
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from evaluate import load
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--language", type=str, required=True, choices=["english", "russian"])
    parser.add_argument("--test_samples", type=int, default=200)
    parser.add_argument("--output_file", type=str, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    model = model.cuda() if torch.cuda.is_available() else model
    model.eval()
    
    # Load test data
    dataset = load_from_disk(f"data/xlsum_{args.language}")
    test_data = dataset["test"].select(range(min(args.test_samples, len(dataset["test"]))))
    
    # Generate summaries
    print("Generating summaries...")
    predictions = []
    references = []
    
    for example in test_data:
        input_text = "summarize: " + example["text"]
        inputs = tokenizer(input_text, max_length=512, truncation=True, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=128)
        
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(pred)
        references.append(example["summary"])
    
    # Compute ROUGE
    rouge = load("rouge")
    results = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    
    results = {k: round(v * 100, 2) for k, v in results.items()}
    
    print("\nResults:")
    print(f"  ROUGE-1: {results['rouge1']}")
    print(f"  ROUGE-2: {results['rouge2']}")
    print(f"  ROUGE-L: {results['rougeL']}")
    
    # Save results
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump({
                "model_path": args.model_path,
                "language": args.language,
                "test_samples": len(test_data),
                "rouge1": results['rouge1'],
                "rouge2": results['rouge2'],
                "rougeL": results['rougeL'],
            }, f, indent=2)
        print(f"\nResults saved to {args.output_file}")

if __name__ == "__main__":
    main()