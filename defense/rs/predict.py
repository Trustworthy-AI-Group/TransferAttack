""" This script loads a base classifier and then runs PREDICT on many examples from a dataset.
"""
import argparse
from core import Smooth
import torch
from architectures import get_architecture
from datasets import load_images, get_labels, load_labels
import tqdm
import os

parser = argparse.ArgumentParser(description='Predict on many examples')
parser.add_argument("input", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("label_file", type=str, help="path to the label file")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1, help="batch size")
parser.add_argument("--skip", type=int, default=400, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N", type=int, default=400, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument('--GPU_ID',default='0',type=str, help='GPU_ID')
parser.add_argument("--targeted", action='store_true', help='targeted attack')
args = parser.parse_args()

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID
    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], 'imagenet')
    base_classifier.load_state_dict(checkpoint['state_dict'])

    # create the smoothed classifier g
    f2l = load_labels(args.label_file, args.targeted)
    smoothed_classifier = Smooth(base_classifier, 1000, args.sigma)

    # prepare output file
    succ = total = 0.
    for batch_idx, [filenames, images] in tqdm.tqdm(enumerate(load_images(args.input, args.batch))):
        labels = get_labels(filenames, f2l)
        images = images.cuda()
        prediction = smoothed_classifier.predict(images, args.N, args.alpha, args.batch)
        if prediction == labels + 1:
            succ += 1 # defense succeeded
        total += labels.shape[0]
        if total % 200 == 0:
            if not args.targeted:
                print("Attack Success Rate: {:.2f}%".format(100. * (total - succ) / total))
            else:
                print("Attack Success Rate: {:.2f}%".format(100. * succ / total))

    
    print(args.input)
    if not args.targeted:
        print("=>Final Attack Success Rate: {:.2f}%".format(100. * (total - succ) / total))
    else:
        print("=>Final Attack Success Rate: {:.2f}%".format(100. * succ / total))
    # print("=>Final Attack Success Rate: {:.2f}%".format(100. * (total - succ) / total)) # attack success rate