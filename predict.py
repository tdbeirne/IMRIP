import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import glob

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

net = Net()
net.load_state_dict(torch.load("./model_2.pth"))

data_files = sorted(glob.glob("./data/validation/*imgs.npy"))

data_tensors = []

for data_file in data_files:
    print(data_file)

    id = data_file.split("_")[0][-3:]

    data_mat = np.load(data_file)
    data_tensor = torch.from_numpy(data_mat)
    data_tensor = torch.unsqueeze(data_tensor, 0)

    output = net(data_tensor)
    output = output.squeeze()

    _, prediction = torch.max(output, dim=0, keepdim=False)
    prediction_as_mat = prediction.numpy()
    prediction_as_mat = prediction_as_mat.astype(np.float32)

    #save to file
    np.save("./predictions/{}_seg.npy".format(id), prediction_as_mat)
