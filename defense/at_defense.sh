INPUT_DIR=/path/to/adv_data
OUTPUT_FILE=at_result.txt
CHECKPOINT_DIR_PATH=/path/to/checkpoint_dir
LABEL_FILE=/path/to/data/labels.csv


cd at
CUDA_VISIBLE_DEVICES=0 \
python main_fast.py "${INPUT_DIR}" --config configs/configs_fast_4px_evaluate.yml --output_prefix "${OUTPUT_FILE}" --resume "${CHECKPOINT_DIR_PATH}/imagenet_model_weights_4px.pth.tar" --evaluate --restarts 10

cd ..
python check_output.py \
  --output_file=at/"${OUTPUT_FILE}" \
  --label_file="${LABEL_FILE}"