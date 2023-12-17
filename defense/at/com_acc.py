import numpy as np
import argparse
import pandas as pd

def load_labels(file_name,header=None):
    if header is None:
        dev = pd.read_csv(file_name,header=None)
        f2l = {dev.iloc[i][0]: dev.iloc[i][1] for i in range(len(dev))}
    else:
        dev = pd.read_csv(file_name,header=0)
        f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label'] for i in range(len(dev))}
    return f2l

parser = argparse.ArgumentParser(description='Defence')
parser.add_argument('--f1', metavar='FILE', default='',
                    help='Output file to save labels.')
parser.add_argument('--f2', metavar='FILE', default='',
                    help='Output file to save labels.')
args = parser.parse_args()

f2l_1 = load_labels(args.f1)
f2l_2 = load_labels(args.f2,header=0)
N = 0
M = 0.
# f2l_1.pop('filename')
print('begin computing')
for key in f2l_1.keys():
    print(eval(str(f2l_1[key])))
    print(int(f2l_2[key]))
    if eval(str(f2l_1[key])) != int(f2l_2[key]):
        M+=1
    N+=1
print(M,N)
print("Attack Success Rate: ", M/N)
with open("evaluation_result.txt", "a") as fl:
    fl.write("\n\n")
    fl.write(str(args.f1))
    fl.write("\n")
    fl.write("Attack Success Rate: {}\n".format(M/N))

