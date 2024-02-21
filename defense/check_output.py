import pandas as pd
import argparse


parser = argparse.ArgumentParser(description='Check output')
parser.add_argument('--label_file', metavar='FILE', default='',
                    help='Input file with labels.')
parser.add_argument('--output_file', metavar='FILE', default='',help='Output file with defensed results.')
parser.add_argument('--targeted', action='store_true', help='targeted attack evaluation')

def load_labels(file_name, targeted=False):
    dev = pd.read_csv(file_name, header=None)
    # dev = pd.read_csv(file_name,names=['filename', 'label'])
    if targeted:
        dev = dev.iloc[:,[0,2]]
    else:
        dev = dev.iloc[:,[0,1]]
    dev.columns = ['filename', 'label'] # label or target_label

    if dev.iloc[0]['filename'] == 'filename':
        dev = dev.iloc[1:]

    f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label'] for i in range(len(dev))}
    
    return f2l


def main():
    args = parser.parse_args()
    # previous label: [1-1000], current label (2023): [0-999]
    print("Check output file: ", args.output_file)
    START = 1
    print("Label start gap is: ", START)

    f2l = load_labels(args.label_file, args.targeted)
    check = load_labels(args.output_file)
    assert len(f2l) == len(check) == 1000, f"len(f2l) = {len(f2l)}, len(check) = {len(check)}"

    wrong_num = 0
    for key in f2l.keys():
        if int(f2l[key]) + START != int(check[key]):
            # print(f2l[key], check[key])
            wrong_num +=1
    print("wrong num: ", wrong_num)
    if not args.targeted:
        print('ASR:{:.2f}%'.format((wrong_num/1000 *100)))
    else:
        print('ASR:{:.2f}%'.format(100 - (wrong_num/1000 *100)))


if __name__ == '__main__':
    main()