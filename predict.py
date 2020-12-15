import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import glob

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

net = Net()
net.load_state_dict(torch.load("./model_6.pth"))
print("Number of parameters {}".format(sum([p.numel() for p in net.parameters()])))

data_files = sorted(glob.glob("./data/validation/*imgs.npy"))

data_tensors = []

for data_file in data_files:
    print(data_file)

    id = data_file[-12:].split("_")[0][-3:]

    data_mat = np.load(data_file)
    data_tensor = torch.from_numpy(data_mat)
    data_tensor = torch.unsqueeze(data_tensor, 0)

    output = net(data_tensor)
    #print(output.size())
    #print(data_tensor.size())
    output = F.interpolate(output, size=data_tensor.size()[2:], mode="trilinear")
    output = output.squeeze()

    #print(output.size())

    _, prediction = torch.max(output, dim=0, keepdim=False)
    prediction_as_mat = prediction.numpy()
    prediction_as_mat = prediction_as_mat.astype(np.float32)

    #save to file
    np.save("./predictions/{}_seg.npy".format(id), prediction_as_mat)
