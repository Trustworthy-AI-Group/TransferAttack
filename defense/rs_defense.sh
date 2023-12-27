# Please change the name of the "ATTACK_METHOD" to eval your method!
# You can run this file directly!
ATTACK_METHOD=mifgsm
INPUT_DIR=../adv_data/${ATTACK_METHOD}/resnet18
LABEL_FILE=../path/to/data/labels.csv
CHECKPOINT_PATH=../defense/models/rs_imagenet/resnet50/noise_0.50/checkpoint.pth.tar
GPU_ID='0'

python rs/predict.py "${INPUT_DIR}" "${LABEL_FILE}" "${CHECKPOINT_PATH}"  0.50 prediction_outupt --alpha 0.001 --N 1000 --skip 100 --batch 1 --GPU_ID $GPU_ID # --targeted