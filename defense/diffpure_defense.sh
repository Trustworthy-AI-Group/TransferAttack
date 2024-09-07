# Please change the name of the "ATTACK_METHOD" to eval your method!
# You can run this file directly!
ATTACK_METHOD=mifgsm
SOURCE_MODEL=resnet18
INPUT_DIR=../../adv_data/${ATTACK_METHOD}/${SOURCE_MODEL}
IMAGE_FOLDER=../../path/to/data

CUDA_VISIBLE_DEVICES=0,1,2,3

cd diffpure
python diffpure.py --image_folder "${IMAGE_FOLDER}" --adv_dir "${INPUT_DIR}" --config imagenet.yml \
    --t 150 --adv_eps 0.0157 --adv_batch_size 4 --num_sub 16 --domain imagenet --classifier_name resnet101 \
    --diffusion_type sde \
    #--targeted
 cd ..