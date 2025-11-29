echo "======================================================================"
echo "Training mT5-small with LoRA (English)"
echo "======================================================================"

python src/train_mt5.py \
  --mode lora \
  --language english \
  --model_name google/mt5-small \
  --output_dir outputs/checkpoints/mt5_lora_en \
  --num_epochs 3 \
  --batch_size 16 \
  --lora_r 8 \
  --lora_alpha 16 \
  --train_samples 1000 \
  --val_samples 200