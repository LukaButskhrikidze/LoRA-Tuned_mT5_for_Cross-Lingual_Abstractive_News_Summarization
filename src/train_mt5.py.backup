"""
mT5 Training Script for Full Fine-Tuning and LoRA
Course: DS 5690-01 Gen AI Models in Theory & Practice
Author: Luka Butskhrikidze

Usage:
    # Full fine-tuning
    python src/train_mt5.py --mode full --language english --output_dir outputs/full_en
    
    # LoRA fine-tuning
    python src/train_mt5.py --mode lora --language english --lora_r 8 --output_dir outputs/lora_en
    
    # Resume from checkpoint
    python src/train_mt5.py --mode full --language english --output_dir outputs/full_en --resume_from_checkpoint outputs/full_en/checkpoint-12345
"""

import argparse
import os
import time
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from evaluate import load


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Basic arguments
    parser.add_argument("--mode", type=str, required=True, choices=["full", "lora"])
    parser.add_argument("--language", type=str, required=True, choices=["english", "russian"])
    parser.add_argument("--model_name", type=str, default="google/mt5-small")
    parser.add_argument("--output_dir", type=str, required=True)
    
    # Training hyperparameters
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    
    # LoRA-specific arguments
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    
    # Data arguments
    parser.add_argument("--train_samples", type=int, default=1000)
    parser.add_argument("--val_samples", type=int, default=200)
    
    # Evaluation frequency arguments
    parser.add_argument("--eval_strategy", type=str, default="epoch", 
                        choices=["steps", "epoch"],
                        help="Evaluation strategy: 'epoch' or 'steps'")
    parser.add_argument("--eval_steps", type=int, default=25000,
                        help="Evaluate every N steps (only used if eval_strategy='steps')")
    
    # Resume training argument
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    
    return parser.parse_args()


def prepare_dataset(examples, tokenizer, max_source_length, max_target_length):
    """Tokenize the dataset"""
    # mT5 expects "summarize: " prefix for summarization
    inputs = ["summarize: " + doc for doc in examples["text"]]
    
    model_inputs = tokenizer(
        inputs,
        max_length=max_source_length,
        padding="max_length",
        truncation=True,
    )
    
    # Tokenize targets
    labels = tokenizer(
        examples["summary"],
        max_length=max_target_length,
        padding="max_length",
        truncation=True,
    )
    
    # Replace padding token id's of the labels by -100 so they are ignored by loss
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_preds, tokenizer):
    """Compute ROUGE metrics with error handling"""
    rouge = load("rouge")
    
    preds, labels = eval_preds
    
    # Decode predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    
    # Replace -100 in labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Clip predictions to valid token range to avoid IndexError
    vocab_size = tokenizer.vocab_size
    preds = np.clip(preds, 0, vocab_size - 1)
    
    try:
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    except Exception as e:
        print(f"\n⚠️  Warning: Decoding error during evaluation: {e}")
        # Return dummy metrics if decoding fails
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    # Compute ROUGE
    try:
        result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True
        )
        
        # Extract scores
        result = {key: value * 100 for key, value in result.items()}
        
        return {
            "rouge1": round(result["rouge1"], 2),
            "rouge2": round(result["rouge2"], 2),
            "rougeL": round(result["rougeL"], 2),
        }
    except Exception as e:
        print(f"\n⚠️  Warning: ROUGE computation error: {e}")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def main():
    args = parse_args()
    start_time = time.time()
    
    # Calculate steps per epoch for informative output
    steps_per_epoch = (args.train_samples // args.batch_size) // args.gradient_accumulation_steps
    
    # Print configuration
    print("="*70)
    print("mT5 Training Configuration")
    print("="*70)
    print(f"Mode: {args.mode.upper()}")
    print(f"Language: {args.language}")
    print(f"Model: {args.model_name}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Train samples: {args.train_samples}")
    print(f"Steps per epoch: ~{steps_per_epoch:,}")
    print(f"Eval strategy: {args.eval_strategy}")
    if args.eval_strategy == "steps":
        evals_per_epoch = steps_per_epoch // args.eval_steps
        print(f"Eval steps: {args.eval_steps:,} (~{evals_per_epoch} evals/epoch)")
    else:
        print(f"Eval frequency: Once per epoch")
    if args.resume_from_checkpoint:
        print(f"Resuming from: {args.resume_from_checkpoint}")
    if args.mode == "lora":
        print(f"LoRA rank: {args.lora_r}")
        print(f"LoRA alpha: {args.lora_alpha}")
    print("="*70)
    
    # Set up device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load tokenizer (use slow tokenizer for compatibility)
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    
    # Load dataset
    print(f"\nLoading {args.language} dataset...")
    dataset = load_from_disk(f"data/xlsum_{args.language}")
    
    # Subsample for training
    train_dataset = dataset["train"].select(range(min(args.train_samples, len(dataset["train"]))))
    val_dataset = dataset["validation"].select(range(min(args.val_samples, len(dataset["validation"]))))
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Tokenize datasets
    print("\nTokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda x: prepare_dataset(x, tokenizer, args.max_source_length, args.max_target_length),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    
    val_dataset = val_dataset.map(
        lambda x: prepare_dataset(x, tokenizer, args.max_source_length, args.max_target_length),
        batched=True,
        remove_columns=dataset["validation"].column_names,
    )
    
    print("✅ Tokenization complete")
    
    # Load model
    print(f"\nLoading model: {args.model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    
    # Apply LoRA if specified
    if args.mode == "lora":
        print(f"\nApplying LoRA...")
        print(f"  Rank (r): {args.lora_r}")
        print(f"  Alpha: {args.lora_alpha}")
        print(f"  Target modules: ['q', 'v']")
        
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q", "v"],
            inference_mode=False,
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Use higher learning rate for LoRA
        if args.learning_rate == 5e-5:
            args.learning_rate = 1e-4
            print(f"\n✅ Adjusted learning rate for LoRA: {args.learning_rate}")
    else:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} (100%)")
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=50,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps if args.eval_strategy == "steps" else None,
        save_strategy=args.eval_strategy,
        save_steps=args.eval_steps if args.eval_strategy == "steps" else None,
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        fp16=False,  # Disabled - causes NaN gradients on RTX 5090
        bf16=False,  # Also disabled for compatibility
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        report_to="tensorboard",
        push_to_hub=False,
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, tokenizer),
    )
    
    # Train
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    # Resume from checkpoint if specified
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # Save final model
    print(f"\n{'='*70}")
    print("SAVING MODEL")
    print("="*70)
    print(f"Output directory: {args.output_dir}")
    
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Final evaluation
    print(f"\n{'='*70}")
    print("FINAL EVALUATION")
    print("="*70)
    
    metrics = trainer.evaluate()
    
    print(f"\nResults:")
    print(f"  ROUGE-1: {metrics.get('eval_rouge1', 0):.2f}")
    print(f"  ROUGE-2: {metrics.get('eval_rouge2', 0):.2f}")
    print(f"  ROUGE-L: {metrics.get('eval_rougeL', 0):.2f}")
    
    # Save metrics
    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/metrics.txt", "w") as f:
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Language: {args.language}\n")
        f.write(f"Train samples: {len(train_dataset)}\n")
        f.write(f"Val samples: {len(val_dataset)}\n")
        f.write(f"Epochs: {args.num_epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        if args.mode == "lora":
            f.write(f"LoRA rank: {args.lora_r}\n")
            f.write(f"LoRA alpha: {args.lora_alpha}\n")
        f.write("\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    
    # Calculate training time
    training_time = time.time() - start_time
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    
    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"Total time: {hours}h {minutes}m")
    print(f"Model saved to: {args.output_dir}")
    print(f"Metrics saved to: {args.output_dir}/metrics.txt")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()