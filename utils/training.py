def train():
    while epoch<=stop_epoch:
        total_loss = 0
        total_accuracy = 0
        model.train()
        exp_lr_scheduler.step()
        print('Epoch: {}\tLR: {:.5f}'.format(epoch,exp_lr_scheduler.get_lr()[0]))
        for batch_idx, data in enumerate(train_loader):
          target = data
          if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
          # forward
          output = model(data)
    #       print(output.shape)
    #       print(data.shape)
          # backward + optimize
          loss = criterion(output, target)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          # print statistics
          accuracy = 0
          total_accuracy+=accuracy
          total_loss+=loss
    #       print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.5f}'.format(epoch, (batch_idx + 1) * len(data), len(train_indices),100*(batch_idx + 1)* len(data) / len(train_indices), loss))
        print('Train Loss: \t'+str(total_loss*batch_size/len(train_indices)))
        vloss, vaccuracy = validate()
        train_losses.append((total_loss*batch_size)/len(train_indices))
        val_losses.append((vloss*batch_size)/len(val_indices))
        train_accuracy.append((total_accuracy*batch_size)/len(train_indices))
        val_accuracy.append((vaccuracy*batch_size)/len(val_indices))
        epoch_data.append(epoch)
        visualize()
        epoch=1+epoch