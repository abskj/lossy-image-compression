import argparse
from model import Decoder
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()

parser.add_argument('--model', nargs='?', default='./out/main.tar', help='Path for model checkpoint file [default: ./out/main.tar]')
parser.add_argument('--compressed', nargs='?', default='./out/compressed/', help='Directory which holds the compressed files [default: ./out/compressed/]')
parser.add_argument('--out', nargs='?', default='./out/decompressed/', help='Directory which will hold the decompressed images [default: ./out/decompressed/]')
args = parser.parse_args()

f = os.listdir(args.compressed)
inputs = []
for i in f:
    if '.xfr' in i:
        inputs.append(i)

decoder = Decoder(args.model)

for i in inputs:
    print('converting %s...'%i)
    decoder.decompress(os.path.join(args.compressed, i), os.path.join(args.out, '%s.png'%i[:-4]))