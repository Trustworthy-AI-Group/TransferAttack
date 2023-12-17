INPUT_DIR=/path/to/adv_data
CHECKPOINT_PATH=/path/to/noise_0.50/checkpoint.pth.tar


python rs/predict.py "${INPUT_DIR}"  "${CHECKPOINT_PATH}"  0.50 prediction_outupt --alpha 0.001 --N 1000 --skip 100 --batch 1