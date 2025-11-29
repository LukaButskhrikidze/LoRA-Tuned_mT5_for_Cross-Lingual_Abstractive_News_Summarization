echo "======================================================================"
echo "Training mT5-small with Full Fine-Tuning (English)"
echo "======================================================================"

python src/train_mt5.py \
  --mode full \
  --language english \
  --model_name google/mt5-small \
  --output_dir outputs/checkpoints/mt5_full_en \
  --num_epochs 3 \
  --batch_size 16 \
  --train_samples 1000 \
  --val_samples 200