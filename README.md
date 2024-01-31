<h1 align="center">TransferAttack</h1>

## About

TransferAttack is a pytorch framework to boost the adversarial transferability for image classification.

[Devling into Adversarial Transferability on Image Classification: A Review, Benchmark and Evaluation](./README.md) will be released soon.

![Overview](./figs/overview.png)

We also release a list of papers about transfer-based attacks [here](https://xiaosenwang.com/transfer_based_attack_papers.html).

### Why TransferAttack

There are a lot of reasons for TransferAttack, such as:

+ **A benchmark for evaluating new transfer-based attacks**: TransferAttack categorizes existing transfer-based attacks into several types and fairly evaluates various transfer-based attacks under the same setting.
+ **Evaluate the robustness of deep models**: TransferAttack provides a plug-and-play interface to verify the robustness of models, such as CNNs and ViTs.
+ **A summary of transfer-based attacks**: TransferAttack reviews numerous transfer-based attacks, making it easy to get the whole picture of transfer-based attacks for practitioners.

## Requirements
+ Python >= 3.6
+ PyTorch >= 1.12.1
+ Torchvision >= 0.13.1
+ timm >= 0.6.12

```bash
pip install -r requirements.txt
```


## Usage
We randomly sample 1,000 images from ImageNet validate set, in which each image is from one category and can be correctly classified by the adopted models. Download the [data](https://drive.google.com/file/d/1VJbWlmcKRVei8rXbBtkjL0Ja_S2tvctp/view?usp=sharing) into `/path/to/data`. Then you can run the attack as follows:

```
python main.py --input_dir ./path/to/data --output_dir adv_data/mifgsm/resnet18 --attack mifgsm --model=resnet18
python main.py --input_dir ./path/to/data --output_dir adv_data/mifgsm/resnet18 --eval
```

## Attacks and Models

### Transfer-based Attacks

<table  style="width:100%" border="1">
<thead>
<tr class="header">
<th><strong>Category</strong></th>
<th><strong>Attack </strong></th>
<th><strong>Main Idea</strong></th>
</tr>
</thead>

<tr>
<th rowspan="19"><sub><strong>Gradient-based</strong></sub></th>
<td><a href="https://arxiv.org/abs/1412.6572" target="_blank" rel="noopener noreferrer">FGSM (Goodfellow et al., 2015)</a></td>
<td ><sub>Add a small perturbation in the direction of gradient</sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/abs/1607.02533" target="_blank" rel="noopener noreferrer">I-FGSM (Kurakin et al., 2015)</a></td>
<td ><sub>Iterative version of FGSM</sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/abs/1710.06081" target="_blank" rel="noopener noreferrer">MI-FGSM (Dong et al., 2018)</a></td>
<td ><sub>Integrate the momentum term into the I-FGSM</sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/abs/1908.06281" target="_blank" rel="noopener noreferrer">NI-FGSM (Lin et al., 2020)</a></td>
<td ><sub>Integrate the Nesterov's accelerated gradient into I-FGSM</sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/abs/2007.06765" target="_blank" rel="noopener noreferrer">PI-FGSM (Gao et al., 2020)</a></td>
<td ><sub>Reusing the cut noise and apply a heuristic project strategy to generate patch-wise noise</sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/abs/2103.15571" target="_blank" rel="noopener noreferrer">VMI-FGSM (Wang et al., 2021)</a></td>
<td ><sub>Variance tuning MI-FGSM</sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/abs/2103.15571" target="_blank" rel="noopener noreferrer">VNI-FGSM (Wang et al., 2021)</a></td>
<td ><sub>Variance tuning NI-FGSM</sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/abs/2103.10609" target="_blank" rel="noopener noreferrer">EMI-FGSM (Wang et al., 2021)</a></td>
<td ><sub>Accumulate the gradients of several data points linearly sampled in the direction of previous gradient</sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/abs/2104.09722" target="_blank" rel="noopener noreferrer">I-FGS²M (Zhang et al., 2021)</a></td>
<td ><sub>Assigning staircase weights to each interval of the gradient</sub></td>
</tr>

<tr>
<td><a href="https://www.ijcai.org/proceedings/2022/0227.pdf" target="_blank" rel="noopener noreferrer">VA-I-FGSM (Zhang et al., 2022)</a></td>
<td ><sub>Adopt a larger step size and auxiliary gradients from other categories</sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/abs/2007.03838" target="_blank" rel="noopener noreferrer">AI-FGTM (Zou et al., 2022)</a></td>
<td ><sub>Adopt Adam to adjust the step size and momentum using the tanh function</sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/abs/2210.05968" target="_blank" rel="noopener noreferrer">RAP (Qin et al., 2022)</a></td>
<td ><sub> Inject the worst-case perturbation when calculating the gradient.</sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/abs/2211.11236" target="_blank" rel="noopener noreferrer">GI-FGSM (Wang et al., 2022)</a></td>
<td ><sub>Use global momentum initialization to better stablize update direction.</sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/abs/2306.01809" target="_blank" rel="noopener noreferrer">PC-I-FGSM (Wan et al., 2023)</a></td>
<td ><sub>Gradient Prediction-Correction on MI-FGSM</sub></td>
</tr>

<tr>
<td><a href="https://ieeexplore.ieee.org/document/10096558" target="_blank" rel="noopener noreferrer">IE-FGSM (Peng et al., 2023)</a></td>
<td ><sub> Integrate anticipatory data point to stabilize the update direction.</sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/abs/2303.15109" target="_blank" rel="noopener noreferrer">DTA (Yang et al., 2023)</a></td>
<td ><sub>Calculate the gradient on several examples using small stepsize</sub></td>
</tr>

<tr>
<td><a href="https://openaccess.thecvf.com/content/ICCV2023/papers/Zhu_Boosting_Adversarial_Transferability_via_Gradient_Relevance_Attack_ICCV_2023_paper.pdf" target="_blank" rel="noopener noreferrer">GRA (Zhu et al., 2023)</a></td>
<td ><sub>Correct the gradient using the average gradient of several data points sampled in the neighborhood and adjust the update gradient with a decay indicator</sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/abs/2306.05225" target="_blank" rel="noopener noreferrer">PGN (Ge et al., 2023)</a></td>
<td ><sub>Penalizing gradient norm on the original loss function</sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/abs/2307.02828" target="_blank" rel="noopener noreferrer">SMI-FGRM (Han et al., 2023)</a></td>
<td ><sub> Substitute sign function with data rescaling and use the depth first sampling technique to stabilize the update direction.</sub></td>
</tr>

<tr>
<th rowspan="11"><sub><strong>Input transformation-based</strong></sub></th>
<td><a href="https://arxiv.org/abs/1803.06978" target="_blank" rel="noopener noreferrer">DIM (Xie et al., 2019)</a></td>
<td ><sub>Random resize and add padding to the input sample</sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/abs/1904.02884" target="_blank" rel="noopener noreferrer">TIM (Dong et al., 2019)</a></td>
<td ><sub>Adopt a Gaussian kernel to smooth the gradient before updating the perturbation</sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/abs/1908.06281" target="_blank" rel="noopener noreferrer">SIM (Ling et al., 2020)</a></td>
<td ><sub>Calculate the average gradient of several scaled images</sub></td>
</tr>

<tr>
<td><a href="https://openaccess.thecvf.com/content/CVPR2021/papers/Wu_Improving_the_Transferability_of_Adversarial_Samples_With_Adversarial_Transformations_CVPR_2021_paper" target="_blank" rel="noopener noreferrer">ATTA (Wu et al., 2021)</a></td>
<td ><sub>Train an adversarial transformation network to perform the input-transformation</sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/abs/2102.00436" target="_blank" rel="noopener noreferrer">Admix (Wang et al., 2021)</a></td>
<td ><sub>Mix up the images from other categories</sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/abs/2112.06011" target="_blank" rel="noopener noreferrer">DEM (Zou et al., 2021)</a></td>
<td ><sub>Calculate the average gradient of several DIM's transformed images</sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/abs/2207.05382" target="_blank" rel="noopener noreferrer">SSM (Long et al., 2022)</a></td>
<td ><sub>Randomly scale images and add noise in the frequency domain</sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/abs/2208.06538" target="_blank" rel="noopener noreferrer">MaskBlock (Fan et al., 2022)</a></td>
<td ><sub>Calculate the average gradients of multiply randomly block-level masked images.</sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/abs/2309.14700" target="_blank" rel="noopener noreferrer">SIA (Wang et al., 2023)</a></td>
<td ><sub> Split the image into blocks and apply various transformations to each block</sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/abs/2308.10601" target="_blank" rel="noopener noreferrer">STM (Ge et al., 2023)</a></td>
<td ><sub>Transform the image using a style transfer network</sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/abs/2308.10299" target="_blank" rel="noopener noreferrer">BSR (Wang et al., 2023)</a></td>
<td ><sub>Randomly shuffles and rotates the image blocks</sub></td>
</tr>

<tr>
<th rowspan="14"><sub><strong>Advanced objective</strong></sub></th>
<td><a href="https://doi.org/10.1007/978-3-030-01264-9_28" target="_blank" rel="noopener noreferrer">TAP (Zhou et al., 2018)</a></td>
<td ><sub>Maximize the difference of feature maps between benign sample and adversarial example and smooth the perturbation </sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/pdf/1907.10823.pdf" target="_blank" rel="noopener noreferrer">ILA (Huang et al., 2019)</a></td>
<td ><sub>Enlarge the similarity of feature difference between the original adversarial example and benign sample </sub></td>
</tr>

<tr>
<td><a href="https://ieeexplore.ieee.org/document/9156367" target="_blank" rel="noopener noreferrer">PoTrip (Li et al., 2020)</a></td>
<td ><sub>Introduce the Poincare distance as the similarity metric to make the magnitude of gradient self-adaptive</sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/abs/2008.08847" target="_blank" rel="noopener noreferrer">YAILA (Wu et al., 2020)</a></td>
<td ><sub>Establishe a linear map between intermediate-level discrepancies and classification loss</sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/abs/2012.11207" target="_blank" rel="noopener noreferrer">Logit (Zhao et al., 2021)</a></td>
<td ><sub>Replace the cross-entropy loss with logit loss</sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/pdf/2107.14185.pdf" target="_blank" rel="noopener noreferrer">FIA (Wang et al., 2021)</a></td>
<td ><sub>Minimize a weighted feature map in the intermediate layer</sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/pdf/2108.07033v1" target="_blank" rel="noopener noreferrer">TRAP (Wang et al., 2021)</a></td>
<td ><sub>Utilize affine transformations and reference feature map </sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/pdf/2204.00008.pdf" target="_blank" rel="noopener noreferrer">NAA (Zhang et al., 2022)</a></td>
<td ><sub>Compute the feature importance of each neuron with decomposition on integral </sub></td>
</tr>

<tr>
<td><a href="https://www.ijcai.org/proceedings/2022/0233.pdf" target="_blank" rel="noopener noreferrer">RPA (Zhang et al., 2022)</a></td>
<td ><sub>Calculate the weight matrix in FIA on randomly patch-wise masked images </sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/abs/2205.13152" target="_blank" rel="noopener noreferrer">TAIG (Huang et al., 2022)</a></td>
<td ><sub>Adopt the integrated gradient to update perturbation </sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/pdf/2204.10606.pdf" target="_blank" rel="noopener noreferrer">FMAA (He et al., 2022)</a></td>
<td ><sub>Utilize momentum to calculate the weight matrix in FIA</sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/abs/2303.03680" target="_blank" rel="noopener noreferrer">Logit-Margin (Weng et al., 2023)</a></td>
<td ><sub>Downscale the logits using a temperature factor and an adaptive margin</sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/abs/2304.13410" target="_blank" rel="noopener noreferrer">ILPD (Li et al., 2023)</a></td>
<td ><sub>Decays the intermediate-level perturbation from the benign features by mixing the features of benign samples and adversarial examples</sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/abs/2401.02727" target="_blank" rel="noopener noreferrer">FFT (Zeng et al., 2023)</a></td>
<td ><sub>Fine-tuning a crafted adversarial example in the feature space</sub></td>
</tr>

<tr>
<th rowspan="10"><sub><strong>Model-related</strong></sub></th>
<td><a href="https://arxiv.org/abs/1812.03413" target="_blank" rel="noopener noreferrer">Ghost (Li et al., 2020)</a></td>
<td ><sub>Densely apply dropout and random scaling on the skip connection to generate several ghost networks to average the gradient</sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/abs/2002.05990" target="_blank" rel="noopener noreferrer">SGM (Wu et al., 2021)</a></td>
<td ><sub>Utilize more gradients from the skip connections in the residual blocks</sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/abs/2206.08316" target="_blank" rel="noopener noreferrer">DSM (Yang et al., 2022)</a></td>
<td ><sub>Train surrogate models in a knowledge distillation manner and adopt CutMix on the input</sub></td>
</tr>

<tr>
<td><a href="https://ojs.aaai.org/index.php/AAAI/article/view/26139" target="_blank" rel="noopener noreferrer">MTA (Qin et al., 2023)</a></td>
<td ><sub>Train a meta-surrogate model (MSM), whose adversarial examples can maximize the loss on a single or a set of pre-trained surrogate models </sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/abs/2304.06908" target="_blank" rel="noopener noreferrer">MUP (Yang et al., 2023)</a></td>
<td ><sub>Mask unimportant parameters of surrogate models </sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/abs/2306.12685" target="_blank" rel="noopener noreferrer">BPA (Wang et al., 2023)</a></td>
<td ><sub>Recover the trunctaed gradient of non-linear layers </sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/abs/2304.10136" target="_blank" rel="noopener noreferrer">DHF (Wang et al., 2023)</a></td>
<td ><sub>Mixup the feature of current examples and benign samples and randomly replaces the features with their means.</sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/abs/2109.04176" target="_blank" rel="noopener noreferrer">PNA-PatchOut (Wei et al., 2021)</a></td>
<td ><sub>Ignore gradient of attention and randomly drop patches among the perturbation</sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/abs/2204.12680" target="_blank" rel="noopener noreferrer">SAPR (Zhou et al., 2022)</a></td>
<td ><sub>Randomly permute input tokens at each attention layer</sub></td>
</tr>

<tr>
<td><a href="https://arxiv.org/abs/2303.15754" target="_blank" rel="noopener noreferrer">TGR (Zhang et al., 2023)</a></td>
<td ><sub>Scale the gradient and mask the maximum or minimum gradient magnitude</sub></td>
</tr>

</table>

### Models

To thoroughly evaluate existing attacks, we have included various popular models, including both CNNs ([ResNet-18](https://arxiv.org/abs/1512.03385), [ResNet-101](https://arxiv.org/abs/1512.03385), [ResNeXt-50](https://arxiv.org/abs/1611.05431), [DenseNet-121](https://arxiv.org/abs/1608.06993)) and ViTs ([ViT](https://arxiv.org/abs/2010.11929), [PiT](https://arxiv.org/abs/2103.16302), [Visformer](https://arxiv.org/abs/2104.12533), [Swin](https://arxiv.org/abs/2103.14030)). Moreover, we also adopted four defense methods, namely [AT](https://arxiv.org/abs/1705.07204), [HGD](https://arxiv.org/abs/1712.02976), [RS](https://arxiv.org/abs/1902.02918), [NRP](https://arxiv.org/abs/2006.04924).
The defense models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1NfSjLzc-MtkYHLumcKYs6OqC2X_zWy3g?usp=share_link).

## Evaluation

### Untargeted Attack
**Note**: We adopt $\epsilon=16/255$ with the number of iterations $T=10$. The base attack for other types of attack is [MI-FGSM](https://arxiv.org/abs/1710.06081). The defaut surrogate model is ResNet-18. For [YAILA](#yaila), we adopt ResNet-50 as the surrogate model. For [PNA-PatchOUt](#pna), [SAPR](#sapr), [TGR](#tgr), we adopt ViT as the surrogate model.

<table  style="width:100%" border="1">
<thead>
<tr class="header">
<th rowspan="2"><strong>Category</strong></th>
<th rowspan="2"><strong>Attacks</strong></th>
<th colspan="4"><strong>CNNs</strong></th>
<th colspan="4"><strong>ViTs</strong></th>
<th colspan="4"><strong>Defenses</strong></th>
</tr>
<th> ResNet-18 </th>
<th> ResNet-101 </th>
<th> ResNeXt-50 </th>
<th> DenseNet-101 </th>
<th> ViT </th>
<th> PiT </th>
<th> Visformer </th>
<th> Swin </th>
<th> AT </th>
<th> HGD </th>
<th> RS </th>
<th> NRP </th>
</thead>

<tr>
<th rowspan="19"><sub><strong>Gradient-based</strong></sub></th>
<td><a href="./transferattack/gradient/fgsm.py" target="_blank" rel="noopener noreferrer">FGSM</a></td>
<td >97.4</td>
<td >36.2</td>
<td >43.8</td>
<td >61.0</td>
<td >15.2</td>
<td >21.2</td>
<td >28.8</td>
<td >34.4</td>
<td >31.0</td>
<td >28.0</td>
<td >20.1</td>
<td >29.8</td>
</tr>

<tr>
<td><a href="./transferattack/gradient/ifgsm.py" target="_blank" rel="noopener noreferrer">I-FGSM</a></td>
<td >100.0</td>
<td >13.9</td>
<td >16.1</td>
<td >37.4</td>
<td >5.4</td>
<td >8.3</td>
<td >11.5</td>
<td >17.0</td>
<td >27.9</td>
<td >9.9</td>
<td >16.2</td>
<td >21.2</td>
</tr>

<tr>
<td><a href="./transferattack/gradient/mifgsm.py" target="_blank" rel="noopener noreferrer">MI-FGSM</a></td>
<td >100.0</td>
<td >41.3</td>
<td >48.4</td>
<td >77.2</td>
<td >16.3</td>
<td >23.9</td>
<td >34.6</td>
<td >42.0</td>
<td >30.4</td>
<td >33.9</td>
<td >19.3</td>
<td >27.6</td>
</tr>

<tr>
<td><a href="./transferattack/gradient/nifgsm.py" target="_blank" rel="noopener noreferrer">NI-FGSM</a></td>
<td >100.0</td>
<td >43.9</td>
<td >49.8</td>
<td >79.5</td>
<td >16.8</td>
<td >23.4</td>
<td >35.3</td>
<td >41.2</td>
<td >30.1</td>
<td >36.2</td>
<td >19.7</td>
<td >28.2</td>
</tr>

<tr>
<td><a href="./transferattack/gradient/pifgsm.py" target="_blank" rel="noopener noreferrer">PI-FGSM</a></td>
<td >100.0</td>
<td >37.3</td>
<td >46.7</td>
<td >74.9</td>
<td >19.9</td>
<td >18.4</td>
<td >26.3</td>
<td >35.7</td>
<td >34.1</td>
<td >35.7</td>
<td >30.0</td>
<td >34.1</td>
</tr>

<tr>
<td><a href="./transferattack/gradient/vmifgsm.py" target="_blank" rel="noopener noreferrer">VMI-FGSM</a></td>
<td >100.0</td>
<td >62.4</td>
<td >68.8</td>
<td >91.2</td>
<td >28.3</td>
<td >41.3</td>
<td >54.5</td>
<td >58.9</td>
<td >32.9</td>
<td >55.6 </td>
<td >23.7</td>
<td >47.6</td>
</tr>

<tr>
<td><a href="./transferattack/gradient/vnifgsm.py" target="_blank" rel="noopener noreferrer">VNI-FGSM</a></td>
<td >100.0</td>
<td >61.4</td>
<td >68.5</td>
<td >92.6</td>
<td >25.3</td>
<td >38.6</td>
<td >52.0</td>
<td >56.9</td>
<td >32.3</td>
<td >52.3</td>
<td >21.5</td>
<td >36.9</td>
</tr>

<tr>
<td><a href="./transferattack/gradient/emifgsm.py" target="_blank" rel="noopener noreferrer">EMI-FGSM</a></td>
<td >100.0</td>
<td >56.6</td>
<td >62.4</td>
<td >90.4</td>
<td >20.9</td>
<td >32.6</td>
<td >46.8</td>
<td >53.1</td>
<td >32.4</td>
<td >46.7</td>
<td >21.3</td>
<td >34.2</td>
</tr>

<tr>
<td><a href="./transferattack/gradient/ifgssm.py" target="_blank" rel="noopener noreferrer">I-FGS²M</a></td>
<td >100.0</td>
<td >18.9</td>
<td >24.2</td>
<td >52.3</td>
<td >8.1</td>
<td >11.9</td>
<td >16.1</td>
<td >23.4</td>
<td >28.4</td>
<td >14.2</td>
<td >16.8</td>
<td >14.3</td>
</tr>

<tr>
<td><a href="./transferattack/gradient/vaifgsm.py" target="_blank" rel="noopener noreferrer">VA-I-FGSM</a></td>
<td >100.0</td>
<td >19.4</td>
<td >23.0</td>
<td >44.6</td>
<td >6.8</td>
<td >11.5</td>
<td >14.3</td>
<td >21.1</td>
<td >28.8</td>
<td >11.5</td>
<td >16.9</td>
<td >18.4</td>
</tr>

<tr>
<td><a href="./transferattack/gradient/aifgtm.py" target="_blank" rel="noopener noreferrer">AI-FGTM</a></td>
<td >100.0</td>
<td >34.6</td>
<td >40.5</td>
<td >70.1</td>
<td >12.7</td>
<td >20.1</td>
<td >28.9</td>
<td >34.9</td>
<td >29.8</td>
<td >26.4</td>
<td >18.2</td>
<td >20.4</td>
</tr>

<tr>
<td><a href="./transferattack/gradient/rap.py" target="_blank" rel="noopener noreferrer">RAP</a></td>
<td >100.0</td>
<td >51.8</td>
<td >58.5</td>
<td >87.5</td>
<td >21.1</td>
<td >26.9</td>
<td >43.1</td>
<td >49.3</td>
<td >32.4</td>
<td >39.7</td>
<td >22.8</td>
<td >31.0</td>
</tr>

<td><a href="./transferattack/gradient/gifgsm.py" target="_blank" rel="noopener noreferrer">GI-FGSM</a></td>
<td >100.0</td>
<td >49.5</td>
<td >54.6</td>
<td >83.7</td>
<td >18.5</td>
<td >27.0</td>
<td >38.7</td>
<td >46.6</td>
<td >31.3</td>
<td >39.0</td>
<td >20.2</td>
<td >31.2</td>
</tr>

<tr>
<td><a href="./transferattack/gradient/pcifgsm.py" target="_blank" rel="noopener noreferrer">PC-I-FGSM</a></td>
<td >100.0</td>
<td >41.3</td>
<td >48.4</td>
<td >76.7</td>
<td >16.7</td>
<td >25.0</td>
<td >35.1</td>
<td >41.4</td>
<td >30.2</td>
<td >34.1</td>
<td >19.3</td>
<td >26.6</td>
</tr>

<tr>
<td><a href="./transferattack/gradient/dta.py" target="_blank" rel="noopener noreferrer">DTA</a></td>
<td >100.0</td>
<td >50.0</td>
<td >57.4</td>
<td >84.8</td>
<td >19.4</td>
<td >28.5</td>
<td >42.5</td>
<td >45.0</td>
<td >31.2 </td>
<td >41.7</td>
<td >19.7</td>
<td >38.1</td>
</tr>

<tr>
<td><a href="./transferattack/gradient/gra.py" target="_blank" rel="noopener noreferrer">GRA</a></td>
<td >100.0</td>
<td >65.1</td>
<td >70.6</td>
<td >93.6</td>
<td >32.6</td>
<td >39.2</td>
<td >54.0</td>
<td >63.1</td>
<td >38.3</td>
<td >59.0</td>
<td >31.2</td>
<td >49.7</td>
</tr>

<tr>
<td><a href="./transferattack/gradient/pgn.py" target="_blank" rel="noopener noreferrer">PGN</a></td>
<td >100.0</td>
<td >68.4</td>
<td >73.6</td>
<td >94.5</td>
<td >31.6</td>
<td >43.6</td>
<td >57.3</td>
<td >65.0</td>
<td >38.8</td>
<td >60.7</td>
<td >32.1</td>
<td >51.7</td>
</tr>

<tr>
<td><a href="./transferattack/gradient/iefgsm.py" target="_blank" rel="noopener noreferrer">IE-FGSM</a></td>
<td >100.0</td>
<td >50.8</td>
<td >56.8</td>
<td >85.9</td>
<td >22.2</td>
<td >26.9</td>
<td >41.4</td>
<td >47.0</td>
<td >30.3</td>
<td >40.9</td>
<td >19.5</td>
<td >29.0</td>
</tr>

<tr>
<td><a href="./transferattack/gradient/smifgrm.py" target="_blank" rel="noopener noreferrer">SMI-FGRM</a></td>
<td >99.7</td>
<td >37.4</td>
<td >41.0</td>
<td >74.5</td>
<td >15.2</td>
<td >21.8</td>
<td >29.7</td>
<td >38.8</td>
<td >32.8</td>
<td >31.1</td>
<td >24.1</td>
<td >31.3</td>
</tr>




<tr>
<th rowspan="11"><sub><strong>Input transformation-based</strong></sub></th>
<td><a href="./transferattack/input_transformation/dim.py" target="_blank" rel="noopener noreferrer">DIM</a></td>
<td >100.0</td>
<td >62.2</td>
<td >68.1</td>
<td >91.9</td>
<td >28.1</td>
<td >36.6</td>
<td >52.8</td>
<td >57.7</td>
<td > 33.5</td>
<td > 59.8</td>
<td > 22.8</td>
<td > 44.7</td>
</tr>


<tr>
<td><a href="./transferattack/input_transformation/tim.py" target="_blank" rel="noopener noreferrer">TIM</a></td>
<td >100.0</td>
<td >35.6</td>
<td >46.4</td>
<td >72.3</td>
<td >15.0</td>
<td >17.4</td>
<td >26.2</td>
<td >35.6</td>
<td >33.7</td>
<td >32.5</td>
<td >29.6</td>
<td >34.1</td>
</tr>


<tr>
<td><a href="./transferattack/input_transformation/sim.py" target="_blank" rel="noopener noreferrer">SIM</a></td>
<td >100.0</td>
<td >58.4</td>
<td >64.9</td>
<td >91.3</td>
<td >22.9</td>
<td >34.4</td>
<td >47.2</td>
<td >53.5</td>
<td >33.6</td>
<td >50.1</td>
<td >22.9</td>
<td >38.2</td>
</tr>

<tr>
<td><a href="./transferattack/input_transformation/atta.py" target="_blank" rel="noopener noreferrer">ATTA</a></td>
<td >100.0</td>
<td >44.2</td>
<td >51.1</td>
<td >80.6</td>
<td >18.9</td>
<td >25.9</td>
<td >37.4</td>
<td >43.4</td>
<td >31.0</td>
<td >37.6</td>
<td >20.0</td>
<td >28.8</td>
</tr>

<tr>
<td><a href="./transferattack/input_transformation/admix.py" target="_blank" rel="noopener noreferrer">Admix</a></td>
<td >100.0</td>
<td >70.1</td>
<td >74.4</td>
<td >96.0</td>
<td >28.6</td>
<td >40.5</td>
<td >58.4</td>
<td >62.1</td>
<td >35.6</td>
<td >62.0</td>
<td >24.8</td>
<td >43.6</td>
</tr>

<tr>
<td><a href="./transferattack/input_transformation/dem.py" target="_blank" rel="noopener noreferrer">DEM</a></td>
<td >100.0</td>
<td >74.5</td>
<td >80.7</td>
<td >98.0</td>
<td >40.0</td>
<td >45.9</td>
<td >64.9</td>
<td >65.4</td>
<td >36.7</td>
<td >78.2</td>
<td >29.0</td>
<td >45.5</td>
</tr>

<tr>
<td><a href="./transferattack/input_transformation/ssm.py" target="_blank" rel="noopener noreferrer">SSM</a></td>
<td >100.0</td>
<td >69.8</td>
<td >73.5</td>
<td >94.2</td>
<td >30.5</td>
<td >41.3</td>
<td >56.7</td>
<td >64.1</td>
<td >35.9</td>
<td >61.2</td>
<td >26.1</td>
<td >48.3</td>
</tr>

<tr>
<td><a href="./transferattack/input_transformation/maskblock.py" target="_blank" rel="noopener noreferrer">MaskBlock</a></td>
<td >100.0</td>
<td >46.8</td>
<td >54.5</td>
<td >82.9</td>
<td >17.5</td>
<td >27.3</td>
<td >39.2</td>
<td >45.4</td>
<td >30.8</td>
<td >38.9</td>
<td >20.5</td>
<td >30.0</td>
</tr>

<tr>
<td><a href="./transferattack/input_transformation/sia.py" target="_blank" rel="noopener noreferrer">SIA</a></td>
<td >100.0</td>
<td >88.8</td>
<td >92.1</td>
<td >99.5</td>
<td >45.1</td>
<td >61.4</td>
<td >80.7</td>
<td >80.6</td>
<td >36.0</td>
<td >82.4</td>
<td >26.3</td>
<td >50.4</td>
</tr>

<tr>
<td><a href="./transferattack/input_transformation/stm.py" target="_blank" rel="noopener noreferrer">STM</a></td>
<td >100.0</td>
<td >72.9</td>
<td >78.3</td>
<td >96.7</td>
<td >35.0</td>
<td >47.5</td>
<td >62.1</td>
<td >68.3</td>
<td >37.2</td>
<td >70.0</td>
<td >29.6</td>
<td >53.2</td>
</tr>

<tr>
<td><a href="./transferattack/input_transformation/bsr.py" target="_blank" rel="noopener noreferrer">BSR</a></td>
<td >100.0</td>
<td >85.5</td>
<td >90.1</td>
<td >99.2</td>
<td >43.8</td>
<td >61.5</td>
<td >79.3</td>
<td >78.5</td>
<td >36.6 </td>
<td > 81.7</td>
<td > 25.9</td>
<td > 54.5</td>
</tr>

<tr>
<th rowspan="10"><sub><strong>Advanced objective</strong></sub></th>
<td><a href="./transferattack/advanced_objective/tap.py" target="_blank" rel="noopener noreferrer">TAP</a></td>
<td >100.0</td>
<td >36.1</td>
<td >43.4</td>
<td >69.9</td>
<td >13.6</td>
<td >17.3</td>
<td >26.1</td>
<td >33.0</td>
<td >30.8</td>
<td >26.6</td>
<td >19.0</td>
<td >26.8</td>
</tr>

<tr>
<td><a href="./transferattack/advanced_objective/ila.py" target="_blank" rel="noopener noreferrer">ILA</a></td>
<td >100.0</td>
<td >55.9</td>
<td >62.0</td>
<td >85.6</td>
<td >15.5</td>
<td >25.4</td>
<td >42.9</td>
<td >45.2</td>
<td >29.9</td>
<td >38.6</td>
<td >18.5</td>
<td >27.7</td>
</tr>

<tr id="yaila">
<td><a href="./transferattack/advanced_objective/yaila/yaila.py" target="_blank" rel="noopener noreferrer">YAILA</a></td>
<td >47.9</td>
<td >20.9</td>
<td >24.9</td>
<td >46.1</td>
<td >5.9</td>
<td >9.7</td>
<td >13.2</td>
<td >18.7</td>
<td >27.4</td>
<td >12.2</td>
<td >15.7</td>
<td >14.5</td>
</tr>

<tr>
<td><a href="./transferattack/advanced_objective/fia.py" target="_blank" rel="noopener noreferrer">FIA</a></td>
<td >99.8</td>
<td >29.4</td>
<td >32.2</td>
<td >61.6</td>
<td >9.6</td>
<td >16.3</td>
<td >23.5</td>
<td >30.3</td>
<td >29.6</td>
<td >18.9</td>
<td >17.8</td>
<td >27.5</td>
</tr>

<tr>
<td><a href="./transferattack/advanced_objective/trap.py" target="_blank" rel="noopener noreferrer">TRAP</a></td>
<td >97.9</td>
<td >65.1</td>
<td >68.0</td>
<td >87.7</td>
<td >25.9</td>
<td >34.1</td>
<td >52.0</td>
<td >55.0</td>
<td >30.7</td>
<td >58.9</td>
<td >18.3</td>
<td >26.0</td>
</tr> 

<tr>
<td><a href="./transferattack/advanced_objective/naa.py" target="_blank" rel="noopener noreferrer">NAA</a></td>
<td >99.6</td>
<td >53.0</td>
<td >57.6</td>
<td >81.2</td>
<td >22.8</td>
<td >34.2</td>
<td >44.4</td>
<td >52.3</td>
<td >32.0</td>
<td >44.1</td>
<td >21.5</td>
<td >34.1</td>
</tr>

<tr>
<td><a href="./transferattack/advanced_objective/rpa.py" target="_blank" rel="noopener noreferrer">RPA</a></td>
<td >100.0</td>
<td >64.9</td>
<td >68.6</td>
<td >92.5</td>
<td >26.2</td>
<td >35.5</td>
<td >53.0</td>
<td >58.6</td>
<td >34.7</td>
<td >56.8</td>
<td >24.7</td>
<td >44.7</td>
</tr>

<tr>
<td><a href="./transferattack/advanced_objective/taig.py" target="_blank" rel="noopener noreferrer">TAIG</a></td>
<td >100.0</td>
<td >20.3</td>
<td >25.5</td>
<td >56.6</td>
<td >7.3</td>
<td >13.3</td>
<td >18.7</td>
<td >25.5</td>
<td >36.0</td>
<td >14.6</td>
<td >17.4</td>
<td >28.5</td>
</tr>

<tr>
<td><a href="./transferattack/advanced_objective/fmaa.py target="_blank" rel="noopener noreferrer">FMAA</a></td>
<td >100.0</td>
<td >37.0</td>
<td >41.3</td>
<td >76.3</td>
<td >10.5</td>
<td >19.1</td>
<td >28.2</td>
<td >35.2</td>
<td >29.8</td>
<td >24.1</td>
<td >17.9</td>
<td >18.9</td>
</tr>

<tr>
<td><a href="./transferattack/advanced_objective/ilpd.py" target="_blank" rel="noopener noreferrer">ILPD</a></td>
<td >73.1</td>
<td >68.3</td>
<td >70.0</td>
<td >72.7</td>
<td >35.4</td>
<td >49.2</td>
<td >55.8</td>
<td >57.0</td>
<td >47.3</td>
<td >85.2</td>
<td >22.7</td>
<td >48.8</td>
</tr>

<tr>
<th rowspan="10"><sub><strong>Model-related</strong></sub></th>
<td><a href="./transferattack/model_related/ghost.py" target="_blank" rel="noopener noreferrer">Ghost</a></td>
<td >67.2</td>
<td >95.4</td>
<td >71.7</td>
<td >69.3</td>
<td >20.4</td>
<td >36.1</td>
<td >45.4</td>
<td >44.3</td>
<td >30.4</td>
<td >42.8</td>
<td >28.0</td>
<td >35.5</td>
</tr>

<tr>
<td><a href="./transferattack/model_related/sgm.py" target="_blank" rel="noopener noreferrer">SGM</a></td>
<td >100.0</td>
<td >47.2</td>
<td >52.7</td>
<td >81.6</td>
<td >21.1</td>
<td >29.8</td>
<td >42.1</td>
<td >48.7</td>
<td >32.2</td>
<td >41.1</td>
<td >21.6</td>
<td >31.4</td>
</tr>

<tr>
<td><a href="./transferattack/model_related/dsm.py" target="_blank" rel="noopener noreferrer">DSM</a></td>
<td >99.2</td>
<td >62.3</td>
<td >67.6</td>
<td >93.8</td>
<td >42.6</td>
<td >36.9</td>
<td >50.8</td>
<td >56.9</td>
<td >32.5</td>
<td >51.5</td>
<td >21.9</td>
<td >35.2</td>
</tr>

<tr>
<td><a href="./transferattack/model_related/mta.py" target="_blank" rel="noopener noreferrer">MTA</a></td>
<td >84.7</td>
<td >42.4</td>
<td >46.5</td>
<td >73.8</td>
<td >12.9</td>
<td >21.5</td>
<td >32.0</td>
<td >40.0</td>
<td >28.9</td>
<td >36.8</td>
<td >19.3</td>
<td >24.1</td>
</tr>

<tr>
<td><a href="./transferattack/model_related/mup.py" target="_blank" rel="noopener noreferrer">MUP</a></td>
<td >100.0</td>
<td >46.9</td>
<td >54.0</td>
<td >84.6</td>
<td >17.3</td>
<td >26.4</td>
<td >38.3</td>
<td >46.3</td>
<td >30.9</td>
<td >37.2</td>
<td >20.3</td>
<td >29.8</td>
</tr>

<tr>
<td><a href="./transferattack/model_related/bpa.py" target="_blank" rel="noopener noreferrer">BPA</a></td>
<td >100.0</td>
<td >61.4</td>
<td >68.0</td>
<td >92.7</td>
<td >24.1</td>
<td >36.6</td>
<td >52.2</td>
<td >58.9</td>
<td >31.8</td>
<td >52.3</td>
<td >22.4</td>
<td >35.3</td>
</tr>

<tr>
<td><a href="./transferattack/model_related/dhf.py" target="_blank" rel="noopener noreferrer">DHF</a></td>
<td >100</td>
<td >71.8</td>
<td >76.6</td>
<td >94.1</td>
<td >31.3</td>
<td >43.5</td>
<td >61.5</td>
<td >65.2</td>
<td >32.4</td>
<td >62</td>
<td >22.6</td>
<td >40.5</td>
</tr>

<tr id="pna">
<td><a href="./transferattack/model_related/pna_patchout.py" target="_blank" rel="noopener noreferrer">PNA-PatchOut</a></td>
<td >68.0</td>
<td >52.6</td>
<td >56.7</td>
<td >66.9</td>
<td >96.6</td>
<td >63.1</td>
<td >65.7</td>
<td >76.0</td>
<td >32.4</td>
<td >47.4</td>
<td >21.7</td>
<td >34.1</td>
</tr>

<tr id="sapr">
<td><a href="./transferattack/model_related/sapr.py" target="_blank" rel="noopener noreferrer">SAPR</a></td>
<td >67.6</td>
<td >53.1</td>
<td >55.2</td>
<td >66.3</td>
<td >97.2</td>
<td >61.6</td>
<td >65.4</td>
<td >79.1</td>
<td >32.7</td>
<td >47.1</td>
<td >23.3</td>
<td >50.6</td>
</tr>

<tr id="tgr">
<td><a href="./transferattack/model_related/tgr.py" target="_blank" rel="noopener noreferrer">TGR</a></td>
<td >80.0</td>
<td >58.0</td>
<td >63.4</td>
<td >77.8</td>
<td >98.8</td>
<td >69.8</td>
<td >73.8</td>
<td >86.9</td>
<td >36.1</td>
<td >54.0</td>
<td >28.7</td>
<td >41.7</td>
</tr>

</table>

### Targeted Attack

**Note**: We adopt $\epsilon=16/255, \alpha=2/255$ with the number of iterations $T=300$. The defaut surrogate model is ResNet-18. For each image, we randomly set a target label.

<table  style="width:100%" border="1">
<thead>
<tr class="header">
<th rowspan="2"><strong>Category</strong></th>
<th rowspan="2"><strong>Attacks</strong></th>
<th colspan="4"><strong>CNNs</strong></th>
<th colspan="4"><strong>ViTs</strong></th>
<th colspan="4"><strong>Defenses</strong></th>
</tr>
<th> ResNet-18 </th>
<th> ResNet-101 </th>
<th> ResNeXt-50 </th>
<th> DenseNet-101 </th>
<th> ViT </th>
<th> PiT </th>
<th> Visformer </th>
<th> Swin </th>
<th> AT </th>
<th> HGD </th>
<th> RS </th>
<th> NRP </th>
</thead>

<th rowspan="4"><sub><strong>Advanced objective</strong></sub></th>
<td><a href="./transferattack/advanced_objective/potrip.py" target="_blank" rel="noopener noreferrer">PoTrip</a></td>
<td >99.7</td>
<td > 4.8</td>
<td > 5.0</td>
<td >14.2</td>
<td > 0.5</td>
<td > 0.8</td>
<td > 2.5</td>
<td > 0.9</td>
<td > 0.0</td>
<td > 3.2</td>
<td > 0.0</td>
<td > 0.4</td>
</tr>

<td><a href="./transferattack/advanced_objective/logit.py" target="_blank" rel="noopener noreferrer">Logit</a></td>
<td >98.1</td>
<td >12.8</td>
<td >16.4</td>
<td >37.2</td>
<td > 2.8</td>
<td > 3.5</td>
<td > 8.7</td>
<td > 5.5</td>
<td > 0.0</td>
<td >12.9</td>
<td > 0.0</td>
<td > 0.4</td>
</tr>

<td><a href="./transferattack/advanced_objective/logit_margin.py" target="_blank" rel="noopener noreferrer">Logit-Margin</a></td>
<td >100.0</td>
<td >13.9</td>
<td >19.3</td>
<td >42.4</td>
<td > 2.4</td>
<td > 3.0</td>
<td > 8.8</td>
<td > 5.5</td>
<td > 0.0</td>
<td >14.2</td>
<td > 0.0</td>
<td > 0.5</td>
</tr>

<td><a href="./transferattack/advanced_objective/fft.py" target="_blank" rel="noopener noreferrer">FFT</a></td>
<td >99.3</td>
<td > 5.2</td>
<td > 6.3</td>
<td >17.8</td>
<td > 0.3</td>
<td > 1.0</td>
<td > 2.1</td>
<td > 2.0</td>
<td > 0.0</td>
<td > 4.0</td>
<td > 0.0</td>
<td > 0.1</td>
</tr>

</table>

## Contributing to TransferAttack

### Main contributors

<table>
<tr>
    <td align="center" style="word-wrap: break-word; width: 150.0; height: 150.0">
        <a href=https://github.com/xiaosen-wang>
            <img src=https://avatars.githubusercontent.com/u/27060904?v=4 width="100;"  style="border-radius:50%;align-items:center;justify-content:center;overflow:hidden;padding-top:10px" alt=Xiaosen Wang/>
            <br />
            <sub style="font-size:14px"><b>Xiaosen Wang</b></sub>
        </a>
    </td>
    <td align="center" style="word-wrap: break-word; width: 150.0; height: 150.0">
        <a href=https://github.com/zeyuanyin>
            <img src=https://avatars.githubusercontent.com/u/51396847?v=4 width="100;"  style="border-radius:50%;align-items:center;justify-content:center;overflow:hidden;padding-top:10px" alt=Zeyuan Yin/>
            <br />
            <sub style="font-size:14px"><b>Zeyuan Yin</b></sub>
        </a>
    </td>
    <td align="center" style="word-wrap: break-word; width: 150.0; height: 150.0">
        <a href=https://github.com/ZhangAIPI>
            <img src=https://avatars.githubusercontent.com/u/53403225?v=4 width="100;"  style="border-radius:50%;align-items:center;justify-content:center;overflow:hidden;padding-top:10px" alt=Zeliang Zhang/>
            <br />
            <sub style="font-size:14px"><b>Zeliang Zhang</b></sub>
        </a>
    </td>
    <td align="center" style="word-wrap: break-word; width: 150.0; height: 150.0">
        <a href=https://github.com/pipiwky>
            <img src=https://avatars.githubusercontent.com/u/91115026?v=4 width="100;"  style="border-radius:50%;align-items:center;justify-content:center;overflow:hidden;padding-top:10px" alt=Kunyu Wang/>
            <br />
            <sub style="font-size:14px"><b>Kunyu Wang</b></sub>
        </a>
    </td>
    <td align="center" style="word-wrap: break-word; width: 150.0; height: 150.0">
        <a href=https://github.com/Zhijin-Ge>
            <img src=https://avatars.githubusercontent.com/u/42350897?v=4 width="100;"  style="border-radius:50%;align-items:center;justify-content:center;overflow:hidden;padding-top:10px" alt=Zhijin Ge/>
            <br />
            <sub style="font-size:14px"><b>Zhijin Ge</b></sub>
        </a>
    </td>
    <td align="center" style="word-wrap: break-word; width: 150.0; height: 150.0">
        <a href=https://github.com/lyygua>
            <img src=https://avatars.githubusercontent.com/u/56330230?v=4 width="100;"  style="border-radius:50%;align-items:center;justify-content:center;overflow:hidden;padding-top:10px" alt=Yuyang Luo/>
            <br />
            <sub style="font-size:14px"><b>Yuyang Luo</b></sub>
        </a>
    </td>
</tr>
</table>

### Acknowledgement
We thank all the researchers who contribute or check the methods. See [contributors](./contributors.md) for details.

### Welcom more participants
We are trying to include more transfer-based attacks. We welcome suggestions and contributions! Submit an issue or pull request and we will try our best to respond in a timely manner.
