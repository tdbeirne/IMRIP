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
        self.conv32_1 = nn.Conv3d(in_channels=4, out_channels=32, kernel_size=(2,2,2), stride=1, padding=1)
        self.batch_norm32_1 = nn.BatchNorm3d(num_features=32)
        # self.conv32_2 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(2,2,2), stride=1, padding=1)
        # self.batch_norm32_2 = nn.BatchNorm3d(num_features=32)

        self.pool32 = nn.MaxPool3d(kernel_size=(2,2,2), return_indices=True)

        self.conv64_1 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(2,2,2), stride=1, padding=1)
        self.batch_norm64_1 = nn.BatchNorm3d(num_features=64)
        # self.conv64_2 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(2,2,2), stride=1, padding=1)
        # self.batch_norm64_2 = nn.BatchNorm3d(num_features=64)

        self.pool64 = nn.MaxPool3d(kernel_size=(2,2,2), return_indices=True)

        self.conv128_1 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(2,2,2), stride=1, padding=1)
        self.batch_norm128_1 = nn.BatchNorm3d(num_features=128)
        # self.conv128_2 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(2,2,2), stride=1, padding=1)
        # self.batch_norm128_2 = nn.BatchNorm3d(num_features=128)

        self.pool128 = nn.MaxPool3d(kernel_size=(2,2,2), return_indices=True)

        #decode
        self.unpool128 = nn.MaxUnpool3d(kernel_size=(2,2,2), stride=2)

        self.unconv128_1 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=(1,1,1), stride=1, padding=0)
        self.unbatch_norm128_1 = nn.BatchNorm3d(num_features=64)
        # self.unconv128_2 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=(2,2,2), stride=1, padding=1)
        # self.unbatch_norm128_2 = nn.BatchNorm3d(num_features=64)

        self.unpool64 = nn.MaxUnpool3d(kernel_size=(2,2,2), stride=2)

        self.unconv64_1 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=(1,1,1), stride=1, padding=0)
        self.unbatch_norm64_1 = nn.BatchNorm3d(num_features=32)
        # self.unconv64_2 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=(2,2,2), stride=1, padding=1)
        # self.unbatch_norm64_2 = nn.BatchNorm3d(num_features=32)

        self.unpool32 = nn.MaxUnpool3d(kernel_size=(2,2,2), stride=2)

        self.unconv32_1 = nn.ConvTranspose3d(in_channels=32, out_channels=4, kernel_size=(1,1,1), stride=1, padding=0)
        # self.unbatch_norm32_1 = nn.BatchNorm3d(num_features=32)
        # self.unconv32_2 = nn.ConvTranspose3d(in_channels=32, out_channels=4, kernel_size=(2,2,2), stride=2, padding=1)

    def forward(self, x):
        #encode
        size_0 = x.size()
        #print("SIZE 0: {}".format(size_0))

        x = F.relu(self.batch_norm32_1(self.conv32_1(x)))
        # x = F.relu(self.batch_norm32_2(self.conv32_2(x)))
        x, indices_32 = self.pool32(x)


        #print("X_32: {}".format(x.shape))
        #print("X_32 INDICES: {}".format(indices_32.shape))


        size_1 = x.size()
        #print("SIZE 1: {}".format(size_1))

        x = F.relu(self.batch_norm64_1(self.conv64_1(x)))
        # x = F.relu(self.batch_norm64_2(self.conv64_2(x)))
        x, indices_64 = self.pool64(x)

        #print("X_64: {}".format(x.shape))
        #print("X_64 INDICES: {}".format(indices_64.shape))

        size_2 = x.size()
        #print("SIZE 2: {}".format(size_2))

        x = F.relu(self.batch_norm128_1(self.conv128_1(x)))
        # x = F.relu(self.batch_norm128_2(self.conv128_2(x)))
        x, indices_128 = self.pool128(x)


        #print("X_128: {}".format(x.shape))
        #print("X_128 INDICES: {}".format(indices_128.shape))

        #decode
        x = self.unpool128(x, indices_128, output_size=size_2)

        #print("X_128_AFTER: {}".format(x.shape))
        #print("X_128_AFTER INDICES: {}".format(indices_128.shape))

        x = F.relu(self.unbatch_norm128_1(self.unconv128_1(x)))
        # x = F.relu(self.unbatch_norm128_2(self.unconv128_2(x)))

        #print("X_128_AFTER_RELU: {}".format(x.shape))

        x = self.unpool64(x, indices_64, output_size=size_1)

        #print("X_64_AFTER: {}".format(x.shape))
        #print("X_64_AFTER INDICES: {}".format(indices_64.shape))

        x = F.relu(self.unbatch_norm64_1(self.unconv64_1(x)))
        # x = F.relu(self.unbatch_norm64_2(self.unconv64_2(x)))

        #print("X_64_AFTER_RELU: {}".format(x.shape))

        x = self.unpool32(x, indices_32, output_size=size_0)

        #print("X_32_AFTER: {}".format(x.shape))
        #print("X_32_AFTER INDICES: {}".format(indices_32.shape))

        # x = F.relu(self.unbatch_norm32_1(self.unconv32_1(x)))
        x = self.unconv32_1(x)

        #print("X_32_AFTER_RELU: {}".format(x.shape))

        #print(x.shape)

        return x



net = Net()
net.load_state_dict(torch.load("./model_2.pth"))
print("Number of parameters {}".format(sum([p.numel() for p in net.parameters()])))

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
