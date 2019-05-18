from model import Autoencoder
from utils.training import *
import torchvision.transforms.functional as TF
import numpy as np
import pickle
from PIL import Image

class Encoder():
    def __init__(self, path):
        self.model = Autoencoder().float()
        self.model.eval()
        checkpoint = load_checkpoint(path)
        self.model.load_state_dict(checkpoint['model_state'])

    def compress(self,path):
        img = Image.open(path)
        width, height = img.size
        dw = 32 - (width%32)
        dh = 32 - (height%32)
        img = TF.pad(img,(dw,dh,0,0))
        x =  TF.to_tensor(img)
        x = x.unsqueeze(0)
        y = self.model.binarizer(self.model.enc(x))
        y = y.cpu().detach().numpy().astype(np.bool)
        return y,dw,dh
    def encode_and_save(self, in_path, out_path):
        y,dw,dh = self.compress(in_path)
        data = [y,dw,dh]
        with open(out_path, 'wb') as fp:
            pickle.dump(data,fp)
