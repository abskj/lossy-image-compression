from model import Autoencoder
from utils.training import *
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils.data.bitstring import *
from utils import display
import tqdm

class Decoder():
    def __init__(self,path):
        self.model = Autoencoder().float()
        # self.model.eval()
        checkpoint = load_checkpoint(path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()

    def decompress(self, in_path, out_path):
        dw = dh = y = S2 = S3 = None
        

        with open(in_path, 'rb') as fp:
            dw = int.from_bytes(fp.read(1), byteorder='big', signed=False)
            dh = int.from_bytes(fp.read(1), byteorder='big', signed=False)
            S2 = int.from_bytes(fp.read(2), byteorder='big', signed=False)
            S3 = int.from_bytes(fp.read(2), byteorder='big', signed=False)

            y = np.empty((1,128,S2,S3)).ravel()
            temp = None;
            j = 0

            print('reading matrix')
            byte = fp.read(1)
            while byte != b"":
                temp = BitArray(byte).bin
                for i in range(len(temp)):
                    y[j] = int(temp[i])
                    j += 1
                byte =  fp.read(1)
        

            print(j)
        y = torch.from_numpy(y.reshape(1,128,S2,S3)).float()
        print('dw: %s dh: %s S2: %s S3: %s'%(dw,dh,y.shape[2], y.shape[3]))

        output = self.model.dec(y)
        img = TF.to_pil_image(output.squeeze(0))

        width, height = img.size
        img = img.crop((dw,dh,width,height));
        plt.imshow(img)
        plt.show()
        img.save(out_path, "PNG")
        return y