from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F


class SignFunction(Function):
    def __init__(self):
        super(SignFunction,self).__init__()
    @staticmethod
    def forward(ctx,input, is_training=True):
        if is_training:
            prob = input.new(input.size()).uniform_()
            x = input.clone()
            x[(1 - input) / 2 <= prob] = 1
            x[(1 - input) / 2 > prob] = -1
            return x
        else:
            return input.sign()
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None
        
class Sign(nn.Module):
    def __init__(self):
        super(Sign, self).__init__()
    def forward(self,x):
        return SignFunction.apply(x, self.training)
class Binarizer(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Binarizer,self).__init__()
        self.sign = Sign()
        self.conv1 = nn.Conv2d(in_channels, out_channels,kernel_size=1,bias=False)
    def forward(self,x):
        x = self.conv1(x)
        x =  F.tanh(x)
        return self.sign(x)