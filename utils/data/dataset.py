import torch
from torch.utils.data import Dataset
import os
import torchvision.transforms.functional as TF
import random
from PIL import Image

class imgDataset(Dataset):
    def __init__(self,path = '../input',indices=None):
        self.files = []
        for r, d, f in os.walk(path):
            for file in f:
                if '.png' in file:
                    self.files.append(os.path.join(r, file))
        if indices!=None:
            files2 = self.files
            self.files = []
            for i in range(len(files2)):
                self.files.append(files2[i])

    def __getitem__(self,idx):
        img = Image.open(self.files[idx])
        return self.transform(img)
    def __len__(self):
        return len(self.files)
    def transform(self,img):
        if random.random()>0.3:
            angle = random.randint(-60, 60)
            img = TF.rotate(img,angle)
        width, height = img.size
        dw = 32 - (width%32)
        dh = 32 - (height%32)
        img = TF.pad(img,(dw,dh,0,0))
        return TF.to_tensor(img)