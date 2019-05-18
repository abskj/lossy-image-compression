import matplotlib.pyplot as plt
import torch


def train(train_params,epoch=1,model=None,optimizer=None,criterion=None,history=None,train_loader=None,validation_loader=None):
  while epoch<=train_params['stop_epoch']:
    total_loss = 0
    total_accuracy = 0
    model.train()
    train_params['exp_lr_scheduler'].step()
    print('Epoch: {}\tLR: {:.5f}'.format(epoch,train_params['exp_lr_scheduler'].get_lr()[0]))
    for batch_idx, data in enumerate(train_loader):
      target = data
      if torch.cuda.is_available():
        data = data.cuda()
        target = target.cuda()
      # forward
      output = model(data)
      # backward + optimize
      loss = criterion(output, target)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      # print statistics
      total_loss+=loss
#       print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.5f}'.format(epoch, (batch_idx + 1) * len(data), len(train_params['train_indices']),100*(batch_idx + 1)* len(data) / len(train_params['train_indices']), loss))
    print('Train Loss: \t'+str(total_loss*train_params['batch_size']/len(train_params['train_indices'])))
    vloss, vaccuracy = validate(model,criterion,validation_loader)
    history['train_losses'].append((total_loss*train_params['batch_size'])/len(train_params['train_indices']))
    history['val_losses'].append((vloss*train_params['batch_size'])/len(train_params['val_indices']))
    history['epoch_data'].append(epoch)
#     visualize()
    epoch=1+epoch

def validate(model,criterion,validation_loader):
  total_loss = 0
  total_acc = 0
  model.train()
  for batch_idx, data in enumerate(validation_loader):
    target = data
    if torch.cuda.is_available():
      data = data.cuda()
      target = target.cuda()
    output = model(data)
    loss = criterion(output, target).item()

    total_loss+=loss
    accuracy = 0
    total_acc+=accuracy
  return total_loss,total_acc


def visualize(history):
  plt.figure(figsize=(15,7))
  plt.plot(history['epoch_data'], history['train_losses'],label="Train Loss {:.5f}".format(history['train_losses'][-1]))
  plt.plot(history['epoch_data'], history['val_losses'], label="Validation Loss {:.5f}".format(history['val_losses'][-1]))
#   plt.plot(history['epoch_data'], history['train_accuracy'],label="Train Accuracy {:.5f}".format(history['train_accuracy'][-1]))
#   plt.plot(history['epoch_data'], history['val_accuracy'], label="Validation Accuracy {:.5f}".format(history['val_accuracy'][-1]))
  display.clear_output(wait=False)
  plt.legend()
  plt.show()