from utils.training import *
from utils.evaluation import evaluate
from model import *
import torch.nn as nn
import torch
from utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import random
import time
import warnings
import sys
import argparse
parser = argparse.ArgumentParser()


warnings.filterwarnings('ignore')
# Parameters for training
batch_size = 1
validation_split = 0.1
random_seed= 42
START_EPOCH = 1

parser.add_argument('--dataset-path', nargs='?', default='../dataset/', help='Root directory of Images')
parser.add_argument('--checkpoint-path', nargs='?', default=None, help='Use to resume training from last checkpoint')
parser.add_argument('--stop-at',nargs='?',default=30,help='Epoch after you want to end training',type=int)
parser.add_argument('--save-at', nargs='?', default='out/', help='Directory where training state will be saved')
args = parser.parse_args()

model = Autoencoder().float()
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
exp_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150,180], gamma=0.1)
if torch.cuda.is_available():
    model = model.cuda()

history = {
    'train_losses':[],
    'val_losses' :[],
    'epoch_data' : []
}

if(args.checkpoint_path):
    checkpoint = load_checkpoint(args.checkpoint_path)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    history = checkpoint['history']
    START_EPOCH = history['epoch_data'][-1]+1

    
ds = Dataset(path=args.dataset_path)
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

parameters = {
    'stop_epoch': args.stop_at,
    'exp_lr_scheduler' : exp_lr_scheduler,
    'train_indices' : train_indices,
    'val_indices' : val_indices,
    'batch_size' : batch_size
}

print('GPU Support Found: %s'%torch.cuda.is_available())

start = time.time()
print('Begining training...')

train(parameters,START_EPOCH,model=model,optimizer=optimizer,criterion=criterion,history=history,train_loader=train_loader,validation_loader=validation_loader)
end = time.time()
print('Finished training in %s seconds'%(end-start))

vds = Dataset(path=args.dataset_path, indices=val_indices)
total_score=0
for i in range(0,len(vds)):
    total_score += evaluate(model,vds,i)
avg_score = '%.5f'%(total_score/len(vds))
print('Average MSSSIM on validation is %s'%avg_score)

save_checkpoint({
    'history' : history,
    'model_state' :model.state_dict(),
    'optimizer_state' : optimizer.state_dict(),
},args.save_at+'train_state_new%s.tar'%avg_score)


visualize(history)