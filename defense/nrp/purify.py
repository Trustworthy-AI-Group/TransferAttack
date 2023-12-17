'''
Purify adversarial images within l_inf <= 16/255
'''

import torch
import os
import argparse
from networks import *
from utils import *
import tqdm

parser = argparse.ArgumentParser(description='Purify Images')
parser.add_argument('--dir', default= 'adv_images/')
parser.add_argument('--purifier', type=str, default= 'NRP',  help ='NPR, NRP_resG')
parser.add_argument('--dynamic', action='store_true', help='Dynamic inferrence (in case of whitebox attack)')
parser.add_argument('--output', type=str, default='purified_imgs', help='GPU to use')
parser.add_argument('--model_pth', type=str, default='./models/NRP.pth', help='pretrained model path')
parser.add_argument('--GPU_ID',default='0',type=str, help='GPU_ID')
args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if args.purifier == 'NRP':
    netG = NRP(3,3,64,23)
    netG.load_state_dict(torch.load(args.model_pth))
if args.purifier == 'NRP_resG':
    netG = NRP_resG(3, 3, 64, 23)
    netG.load_state_dict(torch.load(args.model_pth))
netG = netG.to(device)
netG.eval()
for p in netG.parameters():
    p.requires_grad = False

for p in netG.parameters():
    p.requires_grad = False

print('Parameters (Millions):',sum(p.numel() for p in netG.parameters() if p.requires_grad)/1000000)


dataset = custom_dataset(args.dir)
test_loader = torch.utils.data.DataLoader(dataset,batch_size=1, shuffle=False, num_workers=1, pin_memory=True)


if not os.path.exists(args.output):
    os.makedirs(args.output)
for i, (img, path) in tqdm.tqdm(enumerate(test_loader)):
    img = img.to(device)

    if args.dynamic:
        eps = 16/255
        img_m = img + torch.randn_like(img) * 0.05
        #  Projection
        img_m = torch.min(torch.max(img_m, img - eps), img + eps)
        img_m = torch.clamp(img_m, 0.0, 1.0)
    else:
        img_m = img

    purified = netG(img_m).detach()

    save_img(tensor2img(purified), os.path.join(args.output, path[0]))

