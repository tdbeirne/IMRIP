import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

from torch.utils.data import Dataset, DataLoader

import numpy as np
import glob


#https://github.com/hubutui/DiceLoss-PyTorch/blob/master/loss.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#Simple layout for a network in PyTorch
#https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
#Informational article on how pooling and convolution work
#https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/
#Explanatory Gifs
#https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
#PyTorch tutorial on training a network
#https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#Weighted Cross Entropy for class imbalance
#https://datascience.stackexchange.com/questions/48369/what-loss-function-to-use-for-imbalanced-classes-using-pytorch
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #1 3D input image with 4 channels
        self.conv1 = nn.Conv3d(in_channels=4, out_channels=4, kernel_size=(2,2,2), stride=1, padding=0)
        self.conv2 = nn.Conv3d(in_channels=4, out_channels=4, kernel_size=(2,2,2), stride=1, padding=0)
        self.upscale_1 = nn.ConvTranspose3d(in_channels=4, out_channels=4, kernel_size=(2,2,2), stride=1, padding=0)
        self.upscale_2 = nn.ConvTranspose3d(in_channels=4, out_channels=4, kernel_size=(2,2,2), stride=1, padding=0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.upscale_1(x))
        x = F.relu(self.upscale_2(x))

        return x

net = Net()
criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, 10000, 10000, 10000]))
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#load data
data_files = sorted(glob.glob("./data/train/*imgs.npy"))
label_files = sorted(glob.glob("./data/train/*seg.npy"))

data_tensors = []
label_tensors = []

for data_file in data_files:
    data_tensors.append(torch.from_numpy(np.load(data_file)))

for label_file in label_files:
    label_tensors.append(torch.from_numpy(np.load(label_file)))

num_times = 0
for epoch in range(3):
    running_loss = 0.0
    for data, label in zip(data_tensors, label_tensors):
        start_time = time.time()

        optimizer.zero_grad()

        outputs = net(torch.unsqueeze(data, 0))
        loss = criterion(outputs, torch.unsqueeze(label.long(), 0))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        num_times += 1
        print("COMPLETED {} ITEMS IN {} SECONDS".format(num_times, time.time() - start_time))

    print(running_loss / len(data_files))


output_path = './meme3.pth'
torch.save(net.state_dict(), output_path)

print("DONE TRAINING")
