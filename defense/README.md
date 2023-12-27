## Requirements:

- easydict for AT
- statsmodels for RS
- opencv-python for NRP

## Quick Start:
1. Please change the name of the "ATTACK_METHOD" in '.sh' files to evaluate your method!
2. Directly run the '.sh' files according to the following instructions!

Notes: **We've already set up the relatively path to run, so you can run these commands directly without making many changes!**

## AT: https://github.com/locuslab/fast_adversarial/tree/master/ImageNet

```
sh at_defense.sh
```
We use default 4eps evaluation.

## HGD: https://github.com/lfz/Guided-Denoise/tree/master/nips_deploy

```
sh hgd_defense.sh
```

## RS: https://github.com/locuslab/smoothing

```
sh rs_defense.sh
```
```
python defense/rs/predict.py /path/to/adv_data /path/to/noise_0.50/checkpoint.pth.tar  0.50 prediction_outupt --alpha 0.001 --N 1000 --skip 100 --batch 1
```

Time: ~ 1 hour for 1000 samples on a single 4090 GPU

## NRP: https://github.com/Muzammal-Naseer/NRP

```
sh nrp_defense.sh
```

```
python defense/nrp/purify.py --dir=/path/to/adv_data --output=/path/to/save/purified_data --purifier NRP --model_pth /path/to/NRP.pth  --dynamic
```

Then, evaluate the purified_data, we report the ASR on ResNet101 target model.

