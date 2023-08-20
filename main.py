import os
import numpy as np
from pathlib import Path
from data_prep_bbh import *
from utils import *

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
from torch.nn import init

##############################################################################################################
#
#
#
#
##############################################################################################################


class DatasetGenerator(Dataset):
    def __init__(self, fs=8192, T=1, snr=20,
                 detectors=['H1', 'L1'],
                 nsample_perepoch=100,
                 Nnoise=25, mdist='metric',beta=[0.75,0.95],
                 verbose=True):
        if verbose:
            print('GPU available?', torch.cuda.is_available())
        self.fs = fs     # the sampling frequency (Hz)
        self.T = T       # the observation duration (sec)

        safe = 2         # define the safe multiplication scale for the desired time length
        self.T *= safe

        self.detectors = detectors
        self.snr = snr
        
        self.generate(nsample_perepoch, Nnoise, mdist, beta)  # pre-generate sampels

    def generate(self, Nblock, Nnoise=25, mdist='metric',beta=[0.75,0.95]):
        # Nnoise: # the number of noise realisations per signal
        # Nblock: # the number of training samples per output file
        # mdist:  # mass distribution (astro,gh,metric)

        ts, par = sim_data(self.fs, self.T, self.snr, self.detectors, Nnoise, size=Nblock,mdist=mdist,
                           beta=beta, verbose=False)
        self.strains = np.expand_dims(ts[0], 1)   # (nsample, 1, len(det), fs*T)
        self.labels = ts[1]

    def __len__(self):
        return len(self.strains)

    def __getitem__(self, idx):
        return self.strains[idx], self.labels[idx]

##############################################################################################################
#
#
#
#
##############################################################################################################

def load_model(checkpoint_dir=None):
    Nfilters = [8,16,16,32,64,64,128,128]
    filter_size = [(1,32),] + [(1,16),]*3 + [(1,8),]*2 +[(1,4),]*2
    filter_stride = [(1,1),]*8
    dilation = [(1,1),]*8
    pooling = [1,0,0,0,1,0,0,1]
    pool_size = [[1, 8],] + [(1,1),]*3 + [[1, 6],] + [(1,1),]*2 + [[1, 4]]
    pool_stride = [[1, 8],] + [(1,1),]*3 + [[1, 6],] + [(1,1),]*2 + [[1, 4]]

    net = nn.Sequential()

    for i in range(8):
        net.append(nn.Conv2d(
            in_channels = 1 if i == 0 else Nfilters[i-1],
            out_channels = Nfilters[i],
            kernel_size = filter_size[i],
            stride = filter_stride[i],
            padding = 0,
            dilation = dilation[i],
            groups = 1,
            bias = True,
            padding_mode = 'zeros',
        ))
        net.append(nn.ELU(0.01))
        net.append(nn.BatchNorm2d(num_features=Nfilters[i]))
        if pooling[i]:
            net.append(nn.MaxPool2d(
                kernel_size = pool_size[i],
                stride = pool_stride[i],
                padding = 0,
            ))

    net.append(nn.Flatten())
    net.append(nn.Linear(20224, 64))
    net.append(nn.ELU(0.01))
    net.append(nn.Dropout(0.5))
    net.append(nn.Linear(64, 2))

    if (checkpoint_dir is not None) and (Path(checkpoint_dir).is_dir()):
        p = Path(checkpoint_dir)
        files = [f for f in os.listdir(p) if '.pt' in f]

        # if there is a *.pt model file, load it!
        if (files != []) and (len(files) == 1):
            checkpoint = torch.load(p / files[0])
            net.load_state_dict(checkpoint['model_state_dict'])
        print('Load network from', p / files[0])
        
        epoch = checkpoint['epoch']
        train_loss_history = np.load(p / 'train_loss_history_cnn.npy').tolist()
        return net, epoch, train_loss_history
    else:
        print('Init network!')
        return net, 0, []

def save_model(epoch, model, optimizer, scheduler, checkpoint_dir, filename):
    """Save a model and optimizer to file.
    """
    p = Path(checkpoint_dir)
    p.mkdir(parents=True, exist_ok=True)

    # clear all the *.pt
    assert '.pt' in filename
    for f in [f for f in os.listdir(p) if '.pt' in f]:
        os.remove(p / f)

    # Save loss history
    np.save(p / 'train_loss_history_cnn', train_loss_history)

    output = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }

    if scheduler is not None:
        output['scheduler_state_dict'] = scheduler.state_dict()
    # save the model
    torch.save(output, p / filename)

##############################################################################################################
#
#
#
#
##############################################################################################################

numpy = lambda x, *args, **kwargs: x.detach().numpy(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)
reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)
to = lambda x, *args, **kwargs: x.to(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
transpose = lambda x, *args, **kwargs: x.t(*args, **kwargs)

def accuracy(y_hat, y):
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = argmax(y_hat, dim=1)        
    cmp = astype(y_hat, y.dtype) == y
    return float(reduce_sum(astype(cmp, y.dtype)))

def evaluate_accuracy_gpu(net, data_iter, loss_func, device=None): #@save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量, test_loss
    metric = Accumulator(3)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device).to(torch.float)
            y = y.to(device).to(torch.long)
            y_hat = net(X)
            loss = loss_func(y_hat, y)
            metric.add(accuracy(y_hat, y), y.numel(), loss.sum())
    return metric[0] / metric[1], metric[2] / metric[1]

def train(net, lr, nsample_perepoch, epoch, total_epochs, data_loader, test_iter, notebook=True):
    # Setting for optim.
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=total_epochs,
                    )

    torch.cuda.empty_cache() 
    if notebook:
        animator = Animator(xlabel='epoch', xlim=[1, total_epochs],
                                legend=['train loss', 'test loss', 'train acc', 'test acc'])
    timer, num_batches = Timer(), len(dataset_train)

    # Loop
    for epoch in range(epoch, epoch + total_epochs):
        dataset_train.generate(nsample_perepoch) # pre-generate sampels

        if not notebook:
            print('Learning rate: {}'.format(
                        optimizer.state_dict()['param_groups'][0]['lr']))

        train_loss = 0.0
        total_weight = 0.0
        metric = Accumulator(3)  # 训练损失之和，训练准确率之和，样本数

        net.train()
        for batch_idx, (x, y) in enumerate(data_loader):
            timer.start()
            optimizer.zero_grad()

            data = x.to(device, non_blocking=True).to(torch.float)
            label = y.to(device, non_blocking=True).to(torch.long)

            pred = net(data)
            loss = loss_func(pred, label)

            with torch.no_grad():
                metric.add(loss.sum(), accuracy(pred, label), x.shape[0])
            timer.stop()

            # Optim. (1/2)
            loss.backward()
            optimizer.step()

            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if notebook and (batch_idx + 1) % (num_batches // 5) == 0 or batch_idx == num_batches - 1:
                # plot
                animator.add(epoch + (batch_idx + 1) / num_batches,
                             (train_l, None, train_acc, None))
        # Optim. (2/2)
        scheduler.step()

        # eval test dataset
        test_acc, test_l = evaluate_accuracy_gpu(net, test_iter, loss_func, device)

        # save loss
        train_loss_history.append([epoch+1, train_l, test_l, train_acc, test_acc])

        # plot or print
        if notebook:
            animator.add(epoch + 1, (train_l, test_l, train_acc, test_acc))
        else:
            print(f'Epoch: {epoch+1} \t'
                  f'Train Loss: {train_l:.4f} Test Loss: {test_l:.4f} \t'
                  f'Train Acc: {train_acc} Test Acc: {test_acc}')

        # save the best model
        if (test_l <= min(np.asarray(train_loss_history)[:,1])):
            save_model(epoch, net, optimizer, scheduler, 
                       checkpoint_dir=checkpoint_dir,
                       filename=f'model_e{epoch}.pt',)

    print(f'loss {train_l:.4f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * total_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')

##############################################################################################################
#
#
#
#
##############################################################################################################


if __name__ == "__main__":
    nsample_perepoch = 100
    dataset_train = DatasetGenerator(snr=20, nsample_perepoch=nsample_perepoch)
    dataset_test = DatasetGenerator(snr=20, nsample_perepoch=nsample_perepoch)

    # Create a DataLoader
    data_loader = DataLoader(dataset_train, batch_size=32, shuffle=True,)
    test_iter = DataLoader(dataset_test, batch_size=32, shuffle=True,)

    device = torch.device('cuda')

    # Where we output our model and loss history
    checkpoint_dir = './checkpoints_cnn1/'

    # Creat model    
    net, epoch, train_loss_history = load_model(checkpoint_dir)
    net.to(device);

    # Optim. params
    lr = 0.003
    total_epochs = 100
    total_epochs += epoch
    output_freq = 1

    train(net, lr, nsample_perepoch, epoch, total_epochs, data_loader, test_iter, notebook=False)