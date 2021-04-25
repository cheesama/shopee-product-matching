from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from transformers import pipeline

from tqdm import tqdm

import torch
import os, sys
import json
import pandas as pd
import PIL
import argparse
import random

class ProductPairDataset(Dataset):
    def __init__(self, df, root_dir, train_mode=True, transform=None, batch_size=128):
        """
        Args:
            df (DataFrame): part of entire dataframe
            root_dir (str): root image path
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.products_frame = df
        self.root_dir = root_dir
        self.train_mode = train_mode
        self.batch_size = batch_size

        #set gpu based pipeline
        text_feature_extractor = pipeline(task='feature-extraction', model='distilbert-base-uncased', tokenizer='distilbert-base-uncased', device=0)

        self.text_features = None
        for i in tqdm(range(len(self.products_frame) // self.batch_size), desc='preparing text features in advance ...'):
            if self.text_features is None:
                self.text_features = text_feature_extractor(list(self.products_frame['title'])[i:i * self.batch_size])[:,0,:]
            else:
                self.text_features = torch.cat([self.text_features, text_feature_extractor(list(self.products_frame['title'])[i:i * self.batch_size])[:,0,:]])
        
        if transform is not None:
            self.transform = transform
        else:
            if self.train_mode:  # set default image tranform
                self.transform = transforms.Compose(
                    [
                        transforms.Resize([256, 256]),
                        transforms.RandomResizedCrop([224, 224]),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
            else:
                self.transform = transforms.Compose(
                    [
                        transforms.Resize([224, 224]),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )

    def __len__(self):
        return len(self.products_frame)

    def __getitem__(self, index):
        label = int(self.products_frame.iloc[index]["label_group"])

        image_tensor = self.transform(PIL.Image.open(self.root_dir + os.sep + self.products_frame.iloc[index]["image"]))
        text_tensor = self.text_features[index]
        label_tensor = torch.LongTensor([label])

        return image_tensor, text_tensor, label_tensor
