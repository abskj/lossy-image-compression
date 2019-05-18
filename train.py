from utils.training import train,validate,visualize
from model import *
import torch.nn as nn
import torch
from utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import random
import time

stop_epoch = 31
batch_size = 1
validation_split = 0.1
random_seed= 42
data_root_folder = '../dataset/'



model = Autoencoder().float()
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
exp_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25,40,60], gamma=0.1)
if torch.cuda.is_available():
    model = model.cuda()

ds = Dataset(path=data_root_folder)
shuffle_dataset = True
# Creating data indices for training and validation splits:
dataset_size = len(ds)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]


# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
validation_sampler = SubsetRandomSampler(val_indices)
train_loader = torch.utils.data.DataLoader(ds,batch_size=batch_size,sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size,sampler=validation_sampler)

history = {
    'train_losses':[],
    'val_losses' :[],
    'epoch_data' : []
}
parameters = {
    'stop_epoch': stop_epoch,
    'exp_lr_scheduler' : exp_lr_scheduler,
    'train_indices' : train_indices,
    'val_indices' : val_indices,
    'batch_size' : batch_size
}


start = time.time()
print('Begining training...')

train(parameters,model=model,optimizer=optimizer,criterion=criterion,history=history,train_loader=train_loader,validation_loader=validation_loader)
end = time.time()
print(end-start)