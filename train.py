from pytorch_lightning.metrics.functional import f1, accuracy
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.utils import shuffle

from tqdm import tqdm

from models.product_feature_network import ProductFeatureNet, ProductFeatureEncoder
from data_loader.pair_dataset import (
    ProductPairDataset,
)

import pytorch_lightning as pl
import torch
import argparse
import pandas as pd
import multiprocessing
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model parameters
    parser.add_argument("--backbone_net", default="resnet18")
    parser.add_argument("--feature_dim", default=256)

    # training parameters
    parser.add_argument("--epochs", default=30)
    parser.add_argument("--lr", default=1e-3)
    parser.add_argument("--batch", default=16)
    parser.add_argument("--margin", default=0.5)

    # dataset parameters
    parser.add_argument("--train_portion", default=0.9)
    parser.add_argument("--train_csv_file", default="data/train.csv")
    parser.add_argument("--train_root_dir", default="data/train_images")

    args = parser.parse_args()

    # Init model
    embedding_net = ProductFeatureNet(
        backbone_net=args.backbone_net, feature_dim=args.feature_dim
    )
    product_encoder = ProductFeatureEncoder(
        model=embedding_net,
        lr=args.lr,
        margin=args.margin,
    )

    # Init DataLoader from Custom Dataset
    dataset_df = pd.read_csv(args.train_csv_file)
    dataset_df = shuffle(dataset_df)

    # Init sampler for considering data imbalancing
    class_sample_count = np.array([(len(np.where(dataset_df['label_group']==t)[0]), t) for t in np.unique(dataset_df['label_group'])])
    class_sample_count_dict = {}
    for sample_weight in class_sample_count:
        class_sample_count_dict[sample_weight[1]] = 1 / float(sample_weight[0])
    samples_weight = np.array([class_sample_count_dict[label] for label in dataset_df['label_group']])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

    train_dataset = ProductPairDataset(
        df=dataset_df[: int(len(dataset_df) * args.train_portion)],
        root_dir=args.train_root_dir,
        train_mode=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        num_workers=multiprocessing.cpu_count(),
        #collate_fn=positive_pair_augment_collate_fn,
        shuffle=True,
        sampler=sampler
    )

    valid_dataset = ProductPairDataset(
        df=dataset_df[int(len(dataset_df) * args.train_portion) :],
        root_dir=args.train_root_dir,
        train_mode=False,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch, num_workers=multiprocessing.cpu_count()
    )

    early_stopping = EarlyStopping("val_loss")

    # Initialize a trainer
    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        progress_bar_refresh_rate=1,
        accelerator="ddp",
        max_epochs=args.epochs,
        callbacks=[early_stopping],
    )

    # Train the model
    trainer.fit(product_encoder, train_loader, valid_loader)

    # embedding projection using trained model
    valid_dataset = ProductPairDataset(
        df=dataset_df,
        root_dir=args.train_root_dir,
        train_mode=False,
        shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch // 4,
        num_workers=multiprocessing.cpu_count(),
        shuffle=False,
    )

    # store image feature embedding iterating over data
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    product_encoder.model = product_encoder.model.to(device)

    images_tensor = None
    embeddings_tensor = None

    for images, labels in tqdm(valid_loader, desc="storing image features ..."):
        images = images.to(device)
        with torch.no_grad():
            features = product_encoder.model(images)

            if embeddings_tensor is None:
                embeddings_tensor = features.cpu()
            else:
                embeddings_tensor = torch.cat([embeddings_tensor, features.cpu()])

            if images_tensor is None:
                images_tensor = images.cpu()
            else:
                images_tensor = torch.cat([images_tensor, images.cpu()])

    product_encoder.logger.experiment.add_embedding(
        mat=embeddings_tensor, label_img=images_tensor
    )
