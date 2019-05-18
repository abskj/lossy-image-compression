import torch
from torch.utils.data import Dataset
import os
import torchvision.transforms.functional as TF
import random
from PIL import Image

class someDataset(Dataset):
    def __init__(self,path = '../input'):
        self.files = []
        for r, d, f in os.walk(path):
            for file in f:
                if '.png' in file:
                    self.files.append(os.path.join(r, file))
        print('Found %s images...'%len(self.files))

    def __getitem__(self,idx):
        img = Image.open(self.files[idx])
        return self.transform(img)
    def __len__(self):
        return len(self.files)
    def transform(self,img):
        if random.random()>0.5:
            angle = random.randint(-60, 60)
            img = TF.rotate(img,angle)
        width, height = img.size
        dw = 32 - (width%32)
        dh = 32 - (height%32)
        img = TF.pad(img,(dw,dh,0,0))
        return TF.to_tensor(img)