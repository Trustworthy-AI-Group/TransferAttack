# Please change the name of the "ATTACK_METHOD" to eval your method!
# You can run this file directly!
ATTACK_METHOD=mifgsm
SOURCE_MODEL=resnet18
INPUT_DIR=../../adv_data/${ATTACK_METHOD}/${SOURCE_MODEL}
OUTPUT_FILE=hgd_results/${ATTACK_METHOD}_hgd_results.txt
CHECKPOINT_DIR_PATH=../../defense/models
LABEL_FILE=../path/to/data/labels.csv
GPU_ID=0


cd hgd
python defense.py \
  --input_dir="${INPUT_DIR}" \
  --output_file="${OUTPUT_FILE}" \
  --checkpoint_dir_path="${CHECKPOINT_DIR_PATH}" \
  --GPU_ID="${GPU_ID}"

cd ..
python check_output.py \
  --output_file=hgd/"${OUTPUT_FILE}" \
  --label_file="${LABEL_FILE}" \
  # --targeted