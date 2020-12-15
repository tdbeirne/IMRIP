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

        #This is somewhat based off of SegNet: https://arxiv.org/pdf/1511.00561.pdf
        #However, I'm going with a much smaller net to see how it performs

        #encode
        self.conv_1 = nn.Conv3d(in_channels=4, out_channels=8, kernel_size=(2,2,2), stride=2, padding=0)
        self.norm_1 = nn.BatchNorm3d(num_features=8)

        self.conv_2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(2,2,2), stride=2, padding=0)
        self.norm_2 = nn.BatchNorm3d(num_features=16)

        self.conv_3 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(2,2,2), stride=2, padding=0)
        self.norm_3 = nn.BatchNorm3d(num_features=32)

        self.conv_4 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(2,2,2), stride=2, padding=0)
        self.norm_4 = nn.BatchNorm3d(num_features=64)

        # self.conv_5 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(2,2,2), stride=2, padding=0)
        # self.norm_5 = nn.BatchNorm3d(num_features=128)
        #
        # self.conv_6 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(2,2,2), stride=2, padding=0)
        # self.norm_6 = nn.BatchNorm3d(num_features=256)
        #
        # self.unconv_1 = nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=(2,2,2), stride=2, padding=0)
        # self.unnorm_1 = nn.BatchNorm3d(num_features=128)
        #
        # self.unconv_2 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=(2,2,2), stride=2, padding=0)
        # self.unnorm_2 = nn.BatchNorm3d(num_features=64)

        self.unconv_3 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=(2,2,2), stride=2, padding=0)
        self.unnorm_3 = nn.BatchNorm3d(num_features=32)

        self.unconv_4 = nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=(2,2,2), stride=2, padding=0)
        self.unnorm_4 = nn.BatchNorm3d(num_features=16)

        self.unconv_5 = nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=(2,2,2), stride=2, padding=0)
        self.unnorm_5 = nn.BatchNorm3d(num_features=8)

        self.unconv_6 = nn.ConvTranspose3d(in_channels=8, out_channels=4, kernel_size=(2,2,2), stride=2, padding=0)

    def forward(self, x):
        #encode
        x = F.relu(self.norm_1(self.conv_1(x)))
        x = F.relu(self.norm_2(self.conv_2(x)))
        x = F.relu(self.norm_3(self.conv_3(x)))
        x = F.relu(self.norm_4(self.conv_4(x)))
        # x = F.relu(self.norm_5(self.conv_5(x)))
        # x = F.relu(self.norm_6(self.conv_6(x)))
        # x = F.relu(self.unnorm_1(self.unconv_1(x)))
        # x = F.relu(self.unnorm_2(self.unconv_2(x)))
        x = F.relu(self.unnorm_3(self.unconv_3(x)))
        x = F.relu(self.unnorm_4(self.unconv_4(x)))
        x = F.relu(self.unnorm_5(self.unconv_5(x)))
        x = self.unconv_6(x)

        return x

#check if CUDA is available
if torch.cuda.is_available():
    print("Using CUDA")
    torch.cuda.empty_cache()


net = Net()

#load onto GPU
net = net.cuda()
print(torch.cuda.memory_summary(device=0))

#[1.0, 131.4, 53.3, 163.1]
criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, 300.0, 300.0, 300.0]).cuda())
optimizer = optim.Adam(net.parameters(), lr=0.010)

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

batch_size = 30

#create dataset out of stacks
dataset = TensorDataset(data_stack, label_stack)
loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

#print(label_stack.shape)
#print(data_stack.shape)

print("INITIAL MEMORY USAGE: {}".format(torch.cuda.memory_summary(device=0)))


#Resource I used for building a simple model trainer
#https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
num_epochs = 40

print("Number of parameters: {}".format(sum([p.numel() for p in net.parameters()])))

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    running_loss = 0
    batch_num = 0
    for x_batch, y_batch in loader:
        #print(x_batch.size())
        #print(y_batch.size())

        batch_num += 1
        #print("BATCH {} MEMORY USAGE: {}".format(batch_num, torch.cuda.memory_summary(device=0)))
        x_batch,y_batch = x_batch.cuda(), y_batch.cuda()

        batch_start_time = time.time()
        #zero out gradients because we are working on a new batch
        optimizer.zero_grad()

        outputs = net(x_batch)
        #print(outputs.size())
        #print(outputs.get_device())

        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print("COMPLETED EPOCH {} IN {} SECONDS".format(epoch, time.time() -  epoch_start_time))
    print("EPOCH AVERAGE LOSS: {}".format(running_loss / (204 / batch_size)))


output_path = './model_6.pth'
torch.save(net.state_dict(), output_path)

print("DONE TRAINING")
