from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import torch
import os, sys
import json
import pandas as pd
import PIL
import argparse

def positive_pair_augment_collate_fn(samples, max_sampling=128):
    '''
    samples : list of (posting_ids, images, labels)
    '''

    return torch.stack(posting_ids[:max_sampling]), torch.stack(images[:max_sampling]), torch.stack(labels[:max_sampling])


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

        if transform is not None:
            self.transform = transform
        else:
            if self.train_mode: # set default image tranform
                self.transform = transforms.Compose(
                    [
                        transforms.RandomResizedCrop([224,224]),
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
                        transforms.Resize([224,224]),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )

    def __len__(self):
        return len(self.products_frame)

    def __getitem__(self, index):
        if self.train_mode:
            posting_ids = [torch.LongTensor([int(self.products_frame.iloc[index]['posting_id'].split('_')[1].strip())])]

            image = PIL.Image.open(self.root_dir + os.sep + self.products_frame.iloc[index]["image"])
            images = [self.transform(image)]

            label = self.products_frame.iloc[index]["label_group"]
            labels = [torch.LongTensor([int(label)])]

            # add extra positive pairs for loss balancing
            for row in self.products_frame[self.products_frame['label_group'] == label].iterrows():
                posting_ids.append(torch.LongTensor([int(row[1]['posting_id'].split('_')[1].strip())]))
                images.append(self.transform(PIL.Image.open(self.root_dir + os.sep + row[1]["image"])))
                labels.append(torch.LongTensor([int(row[1]['label_group'])]))

            return posting_ids, images, labels


        else:
            posting_id = torch.LongTensor([int(self.products_frame.iloc[index]['posting_id'].split('_')[1].strip())])

            image = PIL.Image.open(self.root_dir + os.sep + self.products_frame.iloc[index]["image"])
            image = self.transform(image)

            label = self.products_frame.iloc[index]["label_group"]
            label = torch.LongTensor([int(label)])

            return posting_id, image, label

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file')
    parser.add_argument('--root_dir')
    args = parser.parse_args()

    dataset = ProductPairDataset(args.csv_file, args.root_dir)
