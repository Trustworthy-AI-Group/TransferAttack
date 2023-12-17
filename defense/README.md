## Requirements:

- easydict for AT
- statsmodels for RS
- opencv-python for NRP


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
sh hgd_defense.sh
```
```
python defense/rs/predict.py /path/to/adv_data /path/to/noise_0.50/checkpoint.pth.tar  0.50 prediction_outupt --alpha 0.001 --N 1000 --skip 100 --batch 1
```

Time: ~ 1 hour for 1000 samples on a single 4090 GPU

## NRP: https://github.com/Muzammal-Naseer/NRP

```
sh hgd_defense.sh
```

```
python defense/nrp/purify.py --dir=/path/to/adv_data --output=/path/to/save/purified_data --purifier NRP --model_pth /path/to/NRP.pth  --dynamic
```

Then, evaluate the purified_data, we report the ASR on ResNet101 target model.

