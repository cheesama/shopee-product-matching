from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from tqdm import tqdm

import torch
import os, sys
import json
import pandas as pd
import PIL
import argparse
import random


def positive_pair_augment_collate_fn(samples, max_sampling=128):
    """
    samples : list of (posting_ids, images, labels)
    """
    return (
        torch.stack(posting_ids[:max_sampling]),
        torch.stack(images[:max_sampling]),
        torch.stack(labels[:max_sampling]),
    )


class ProductPairDataset(Dataset):
    def __init__(self, df, root_dir, train_mode=True, transform=None):
        """
        Args:
            df (DataFrame): part of entire dataframe
            root_dir (str): root image path
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.products_frame = df
        self.root_dir = root_dir
        self.train_mode = train_mode

        self.posting_ids = []
        self.images = []
        self.labels = []

        if transform is not None:
            self.transform = transform
        else:
            if self.train_mode:  # set default image tranform
                self.transform = transforms.Compose(
                    [
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
        label_tensor = torch.LongTensor([label])

        #ensure at least one positive pair
        pair_index = random.choice(list(self.products_frame[self.products_frame['label_group']==label].index))
        if pair_index > len(self.products_frame) - 1:
            pair_index = index

        aug_image_tensor = self.transform(PIL.Image.open(self.root_dir + os.sep + self.products_frame.iloc[pair_index]["image"]))
        aug_label_tensor = torch.LongTensor([label])

        return image_tensor, label_tensor, aug_image_tensor, aug_label_tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file")
    parser.add_argument("--root_dir")
    args = parser.parse_args()

    dataset = ProductPairDataset(args.csv_file, args.root_dir)
