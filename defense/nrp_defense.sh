INPUT_DIR=/path/to/adv_data
OUTPUT_DIR=/path/to/purified_data
MODEL_PATH=/path/to/NRP.pth
EVAL_DIR=/path/to/data


python nrp/purify.py --dir=${INPUT_DIR} --output=${OUTPUT_DIR} --purifier NRP --model_pth ${MODEL_PATH}  --dynamic

cd ..
python main.py --input_dir ${EVAL_DIR} --output_dir ${OUTPUT_DIR} --eval