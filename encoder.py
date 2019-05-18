import argparse
from model import Encoder
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()

parser.add_argument('--model', nargs='?', default='./out/main.tar', help='Path for model checkpoint file')
parser.add_argument('--image', nargs='?', default='./', help='Directory which holds the images to be compressed [default: current dir]')
parser.add_argument('--out', nargs='?', default='./', help='Directory which will hold the compressed images [default: current dir]')
args = parser.parse_args()

f = os.listdir(args.image)
inputs = []
for i in f:
    if '.png' in i:
        inputs.append(i)

encoder = Encoder(args.model)

for i in tqdm(inputs):
    print('converting %s...'%i)
    encoder.encode_and_save(os.path.join(args.image, i), os.path.join(args.out, '%scomp.xfr'%i[:-4]))