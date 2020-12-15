import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

from torch.utils.data import Dataset, DataLoader, TensorDataset

import numpy as np
import glob

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
        self.batch_norm0 = nn.BatchNorm3d(num_features=4)
        self.conv1 = nn.Conv3d(in_channels=4, out_channels=4, kernel_size=(2,2,2), stride=2, padding=0)
        self.pool1 = nn.MaxPool3d(kernel_size=(2,2,2))
        self.batch_norm1 = nn.BatchNorm3d(num_features=4)
        self.conv2 = nn.Conv3d(in_channels=4, out_channels=4, kernel_size=(2,2,2), stride=2, padding=0)
        self.pool2 = nn.MaxPool3d(kernel_size=(2,2,2))
        self.batch_norm2 = nn.BatchNorm3d(num_features=4)
        self.upscale_1 = nn.Upsample(scale_factor=4, mode="trilinear")
        self.batch_norm3 = nn.BatchNorm3d(num_features=4)
        self.upscale_2 = nn.Upsample(scale_factor=4, mode="trilinear")

    def forward(self, x):
        #print(x.shape)
        x = self.batch_norm0(x)
        #print(x.shape)

        x = self.pool1(F.relu(self.conv1(x)))
        x = self.batch_norm1(x)
        #print(x.shape)

        x = self.pool2(F.relu(self.conv2(x)))
        x = self.batch_norm2(x)

        #print(x.shape)
        x = F.relu(self.upscale_1(x))
        #print(x.shape)
        x = self.batch_norm3(x)
        #print(x.shape)
        x = F.relu(self.upscale_2(x))
        #print(x.shape)
        return x

#check if CUDA is available
if torch.cuda.is_available():
    print("Using CUDA")


net = Net()

#load onto GPU
net = net.cuda()

#[1.0, 131.4, 53.3, 163.1]
criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, 300, 300, 300]).cuda())
optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

#load data
data_files = sorted(glob.glob("./data/train/*imgs.npy"))
label_files = sorted(glob.glob("./data/train/*seg.npy"))

data_tensors = []
label_tensors = []

for data_file in data_files:
    #convert from npy file to PyTorch tensor
    data_tensor = torch.from_numpy(np.load(data_file))

    #convert to 5D tensor of size 1
    data_tensor = torch.unsqueeze(data_tensor, 0)

    #resize tensor
    data_tensors.append(F.interpolate(input=data_tensor, size=(128, 128, 128), mode="trilinear"))

for label_file in label_files:
    #convert from npy file to PyTorch tensor
    label_tensor = torch.from_numpy(np.load(label_file))

    #convert to 5D tensor of size 1
    label_tensor = torch.unsqueeze(label_tensor, 0)
    label_tensor = torch.unsqueeze(label_tensor, 0)

    #resize tensors
    label_tensors.append(F.interpolate(input=label_tensor, size=(128, 128, 128), mode="trilinear"))

#create data stack and label stack
data_stack = torch.cat(data_tensors, dim=0)
label_stack = torch.cat(label_tensors, dim=0)

#squeeze labels back down and change data type for cross entropy
label_stack = torch.squeeze(label_stack, dim=1)
label_stack = label_stack.long()

#create dataset out of stacks
dataset = TensorDataset(data_stack, label_stack)
loader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)

#print(label_stack.shape)
#print(data_stack.shape)

batch_size = 10
num_epochs = 20
losses = []

#Resource I used for building a simple model trainer
#https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    running_loss = 0
    for x_batch, y_batch in loader:
        x_batch,y_batch = x_batch.cuda(), y_batch.cuda()


        #print(x_batch.shape)
        #print(y_batch.shape)

        batch_start_time = time.time()
        #zero out gradients because we are working on a new batch
        optimizer.zero_grad()

        outputs = net(x_batch)
        #print(outputs.shape)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print("COMPLETED EPOCH {} IN {} SECONDS".format(epoch, time.time() -  epoch_start_time))
    print("EPOCH AVERAGE LOSS: {}".format(running_loss / 12.75))


output_path = './model_2.pth'
torch.save(net.state_dict(), output_path)

print("DONE TRAINING")
