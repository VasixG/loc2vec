import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import torch.nn.functional as F


class TripletDataset(Dataset):
    """
    Dataset for triplets where each sample consists of an anchor, positive, and negative example.
    Each example is a tensor with channels corresponding to different tag groups.
    """

    def __init__(self, triplets, tag_group_count=3, image_size=(256, 256)):
        """
        Initialize the dataset.

        :param triplets: List of triplets. Each triplet is a tuple ((lat_a, lon_a), (lat_p, lon_p), (lat_n, lon_n))
        :param tag_group_count: Number of tag groups (number of channels)
        :param image_size: Size to which images will be resized (height, width)
        """
        self.triplets = triplets
        self.tag_group_count = tag_group_count
        self.image_size = image_size

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        anchor_point, positive_point, negative_point = self.triplets[index]

        anchor = self.load_tensor_for_point(anchor_point)
        positive = self.load_tensor_for_point(positive_point)
        negative = self.load_tensor_for_point(negative_point)

        return (anchor, positive, negative)

    def load_tensor_for_point(self, point):
        """
        Load images for a point and create a tensor with channels as different tag groups.

        :param point: Tuple (lat, lon)
        :return: Tensor of shape (C, H, W)
        """
        lat = round(point[0], 6)
        lon = round(point[1], 6)
        dir_name = f"data/loc_{lat}_{lon}"

        if not os.path.exists(dir_name):
            channels = [torch.zeros((3, *self.image_size)) for _ in range(self.tag_group_count)]
            tensor = torch.cat(channels, dim=0)
            return tensor

        channels = []
        for i in range(self.tag_group_count):
            image_name = f"img_{lat}_{lon}_{i}.png"
            image_path = os.path.join(dir_name, image_name)

            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
                image = image.resize(self.image_size)
                image_array = np.array(image).transpose((2, 0, 1))
                image_tensor = torch.from_numpy(image_array).float() / 255.0
                channels.append(image_tensor)
            else:
                channels.append(torch.zeros((3, *self.image_size)))

        tensor = torch.cat(channels, dim=0)
        return tensor


class EmbeddingNet(nn.Module):
    def __init__(self, n_channels):
        super(EmbeddingNet, self).__init__()

        self.conv_net = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.PReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.PReLU()
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        distance_positive = 1 - cos(anchor, positive)
        distance_negative = 1 - cos(anchor, negative)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()
