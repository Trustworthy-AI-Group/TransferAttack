import torch

import numpy as np

import os
import argparse
import tqdm

import transferattack
from transferattack.utils import *

def get_parser():
    parser = argparse.ArgumentParser(description='Generating transferable adversaria examples')
    parser.add_argument('--eval', action='store_true', help='attack/evluation')
    parser.add_argument('--attack', default='aifgtm', type=str, help='the attack algorithm',
                        choices=['fgsm', 'ifgsm', 'mifgsm', 'nifgsm', 'vmifgsm', 'vnifgsm', 'emifgsm', 'ifgssm', 'vaifgsm', 'aifgtm', 'pcifgsm', 'dta', 'pgn',
                                'dim', 'tim', 'sim', 'admix', 'dem', 'ssm', 'sia', 'stm', 'bsr',
                                'tap', 'ila', 'fia', 'yaila', 'trap', 'naa', 'rpa', 'taig', 'fmaa', 'ilpd',
                                'sgm', 'dsm', 'mta', 'mup', 'bpa', 'pna_patchout', 'setr', 'sapr', 'tgr'
                        ])
    parser.add_argument('--epoch', default=10, type=int, help='the iterations for updating the adversarial patch')
    parser.add_argument('--batchsize', default=10, type=int, help='the bacth size')
    parser.add_argument('--eps', default=16 / 255, type=float, help='the stepsize to update the perturbation')
    parser.add_argument('--alpha', default=1.6 / 255, type=float, help='the stepsize to update the perturbation')
    parser.add_argument('--momentum', default=0., type=float, help='the decay factor for momentum based attack')
    parser.add_argument('--model', default='resnet18', type=str, help='the source surrogate model',
                        choices=['resnet18', 'resnet101', 'densenet121', 'mobilenet', 'vit', 'swin'])
    parser.add_argument('--input_dir', default='./data', type=str, help='the path for the benign images')
    parser.add_argument('--output_dir', default='./results', type=str, help='the path to store the adversarial patches')
    parser.add_argument('--helper_folder',default='./helper',type=str, help='the path to store the helper models')
    return parser.parse_args()


def main():
    args = get_parser()
    f2l = load_labels(os.path.join(args.input_dir, 'labels.csv'))
    if not args.eval:
            model = EnsembleModel([wrap_model(model_list[name](weights='DEFAULT').eval().cuda()) for name in ['resnet18', 'resnet101', 'densenet121', 'mobilenet', 'vit', 'swin'] if name != args.model])        
            if args.attack in transferattack.attack_zoo:
                if args.attack in ['atta']:
                    from transferattack.input_transformation import atta
                    atta_model = atta.train_ATTA_Model(args.input_dir, args.batchsize,f2l,10,10,eval_model=args.model,device='cuda',path=args.helper_folder)
                    attacker = transferattack.attack_zoo[args.attack.lower()](model,atta_model=atta_model)
                else:
                    attacker = transferattack.attack_zoo[args.attack.lower()](model)
            else:
                raise Exception("Unspported attack algorithm {}".format(args.attack))

            for batch_idx, [filenames, images] in tqdm.tqdm(
                    enumerate(load_images(os.path.join(args.input_dir, 'images'), args.batchsize))):
                labels = get_labels(filenames, f2l)
                perturbations = attacker(images, labels)
                save_images(args.output_dir, images + perturbations.cpu(), filenames)
    else:
        accuracy = dict()
        res = '|'
        for model_name, model_arc in model_list.items():
            model = wrap_model(model_arc(weights='DEFAULT').eval().cuda())
            succ, total = 0, 0
            for batch_idx, [filenames, images] in tqdm.tqdm(enumerate(load_images(args.output_dir, args.batchsize))):
                labels = get_labels(filenames, f2l)
                pred = model(images.cuda())
                succ += (labels.numpy() == pred.argmax(dim=1).detach().cpu().numpy()).sum()
                total += labels.shape[0]
            accuracy[model_name] = (succ / total * 100)
            print(model_name, accuracy[model_name])
            res += ' {:.2f} |'.format(accuracy[model_name])

        print(accuracy)
        print(res)


if __name__ == '__main__':
    main()

