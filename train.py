from pytorch_lightning.metrics.functional import f1, accuracy
from torch.utils.data import DataLoader, random_split
from sklearn.utils import shuffle

from models.product_feature_network import ProductFeatureNet, ProductFeatureEncoder
from data_loader.pair_dataset import ProductPairDataset

import pytorch_lightning as pl
import torch
import argparse
import pandas as pd
import multiprocessing

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model parameters
    parser.add_argument("--backbone_net", default="resnet18")
    parser.add_argument("--feature_dim", default=256)

    # training parameters
    parser.add_argument("--lr", default=1e-3)
    parser.add_argument("--batch", default=32)
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
        model=embedding_net, lr=args.lr, margin=args.margin
    )

    # Init DataLoader from Custom Dataset
    dataset_df = pd.read_csv(args.train_csv_file)
    dataset_df = shuffle(dataset_df)

    train_dataset = ProductPairDataset(
        df=dataset_df[: int(len(dataset_df) * args.train_portion)],
        root_dir=args.train_root_dir,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch, num_workers=multiprocessing.cpu_count())

    valid_dataset = ProductPairDataset(
        df=dataset_df[int(len(dataset_df) * args.train_portion) :],
        root_dir=args.train_root_dir,
        train_mode=False,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch, num_workers=multiprocessing.cpu_count())

    # Initialize a trainer
    trainer = pl.Trainer(gpus=torch.cuda.device_count(), progress_bar_refresh_rate=1, accelerator='ddp')

    # Train the model ⚡
    trainer.fit(product_encoder, train_loader, valid_loader)
