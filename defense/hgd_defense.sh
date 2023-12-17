INPUT_DIR=/path/to/adv_data
OUTPUT_FILE=hgd_result.txt
CHECKPOINT_DIR_PATH=/path/to/checkpoint_dir
LABEL_FILE=/path/to/data/labels.csv


cd hgd
python defense.py \
  --input_dir="${INPUT_DIR}" \
  --output_file="${OUTPUT_FILE}" \
  --checkpoint_dir_path="${CHECKPOINT_DIR_PATH}"

cd ..
python check_output.py \
  --output_file=hgd/"${OUTPUT_FILE}" \
  --label_file="${LABEL_FILE}"