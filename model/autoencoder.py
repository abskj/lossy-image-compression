import torch.nn as nn
from model.components import Binarizer

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder,self).__init__()
        self.enc = nn.Sequential(nn.Conv2d(3,32,8,stride=4,padding=2),
                                nn.ReLU(),
                                nn.BatchNorm2d(32),
                                nn.Conv2d(32,64,2,stride=2),
                                nn.ReLU(),
                                nn.BatchNorm2d(64),
    #                             nn.Conv2d(32,128,6,stride=4,padding=1),
    #                             nn.ReLU(),
    #                             nn.BatchNorm2d(128),
    #                             nn.Conv2d(128,128,3,stride=1,padding=1),
    #                             nn.Sigmoid()
                                )
        self.dec = nn.Sequential(nn.ConvTranspose2d(128,32,8,stride=4, padding=2),
                                nn.BatchNorm2d(32),
                                nn.ReLU(),
                                nn.ConvTranspose2d(32,3,2,2),
                                nn.BatchNorm2d(3),
                                nn.ReLU(),
    #                             nn.ConvTranspose2d(64,32,2,2),
    #                             nn.BatchNorm2d(32),
    #                             nn.Conv2d(32,32,3,stride=1,padding=1),
    #                             nn.BatchNorm2d(32),
    #                             nn.ConvTranspose2d(32,3,2,2),
    #                             nn.Sigmoid()
                                )
        self.binarizer = Binarizer(64,128)
    def forward(self,x):
    
        x = self.enc(x)
        x = self.binarizer(x)
    #     print(x.shape)
        x = self.dec(x)
    #     x = (x+1)*255
    #     x.round_()
        return x