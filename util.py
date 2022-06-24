import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
import urllib
import glob
import torchvision.transforms as transforms


class component_dataset(torch.utils.data.Dataset):
    def __init__(self, root, csv=None, train_test="train", idx=None, transform=None):
        self.root = root
        if train_test:
            self.csv = pd.read_csv(csv)
            self.files = glob.glob(f"{root}/*.jpg")
        else:
            self.files = glob.glob(f"{root}/*.jpg")
        self.train_test = train_test  # If true, it is train mode
        if self.train_test is True:
            self.train_test = "train"
        self.index = idx
        if self.train_test in ["train", "valid"]:
            if self.index is not None:
                assert idx is not None, "index list is None!"
                self.files = [f"{root}/{self.csv.iloc[i, 0]}.jpg" for i in self.index.tolist()]
                self.csv = self.csv.iloc[self.index, :]
        self.transform = transform
        self.basic_transform = transforms.ToTensor()
    
    def __len__(self):
        ans = len(self.files)
        return ans

    def __getitem__(self, idx):
        image = Image.open(self.files[idx])
        image = Image.open(os.path.join(self.root, str(self.csv.iloc[idx, 0])) + ".jpg")
        label = torch.tensor(self.csv.iloc[idx, 1]).long()  # 2

        if self.train_test == "train":
            image = self.transform(image)
        else:
            image = self.basic_transform(image)

        pack = (image, label)
        return pack

def validation_sampler(root, length_list):
    files = glob.glob(f"{root}/*.jpg")
    #print(sum(length_list), len(files))
    assert sum(length_list) == len(files), "Number of files are not matched to the sum of the number of train/val set"
    randper = torch.randperm(len(files))
    train_idx = randper[:length_list[0]]
    valid_idx = randper[length_list[0]:]
    
    return train_idx, valid_idx

def calc_accuracy(prediction, label):
    with torch.no_grad():
        _, pred2label = prediction.max(dim=1)
        same = (pred2label == label).float()
        accuracy = same.sum().item() / same.numel()
    return accuracy