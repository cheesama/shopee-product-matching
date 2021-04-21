from pytorch_lightning.metrics.functional import f1, accuracy
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.utils import shuffle

from tqdm import tqdm

from models.product_feature_network import ProductFeatureNet, ProductFeatureEncoder
from data_loader.pair_dataset import ProductPairDataset
from data_loader.custom_batch_sampler import PositivePairAugBatchSampler

import pytorch_lightning as pl
import torch
import argparse
import pandas as pd
import multiprocessing
import numpy as np
import faiss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model parameters
    parser.add_argument("--backbone_net", default="resnet34")
    parser.add_argument("--feature_dim", default=768)

    # training parameters
    parser.add_argument("--epochs", default=30)
    parser.add_argument("--margin", default=0.5)
    parser.add_argument("--lr", default=1e-3)
    parser.add_argument("--lr_patience", default=2)
    parser.add_argument("--early_stop_patience", default=4)
    parser.add_argument("--lr_decay_ratio", default=0.1)
    parser.add_argument("--batch", default=128)

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

    #train_df = dataset_df[: int(len(dataset_df) * args.train_portion)]
    train_df = dataset_df
    train_batch_sampler = PositivePairAugBatchSampler(train_df)
    train_dataset = ProductPairDataset(
        df=train_df,
        root_dir=args.train_root_dir,
        train_mode=True,
    )
    train_loader = DataLoader(
        train_dataset,
        num_workers=multiprocessing.cpu_count(),
        # batch_size=args.batch,
        batch_sampler=train_batch_sampler,
    )

    valid_df = dataset_df[len(train_df) :]
    valid_batch_sampler = PositivePairAugBatchSampler(valid_df)
    valid_dataset = ProductPairDataset(
        df=valid_df,
        root_dir=args.train_root_dir,
        train_mode=False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        num_workers=multiprocessing.cpu_count(),
        batch_sampler=valid_batch_sampler,
    )

    test_loader = DataLoader(
        valid_dataset, num_workers=multiprocessing.cpu_count(), batch_size=args.batch
    )

    early_stopping = EarlyStopping("val_loss", patience=args.early_stop_patience)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")

    # Initialize a trainer
    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        progress_bar_refresh_rate=1,
        accelerator="ddp",
        max_epochs=args.epochs,
        callbacks=[early_stopping, lr_monitor, checkpoint_callback],
        replace_sampler_ddp=False,
    )

    # Train the model
    #trainer.fit(product_encoder, train_loader, valid_loader)
    trainer.fit(product_encoder, train_loader)

    # store image feature embedding iterating over data
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    product_encoder.model = product_encoder.model.to(device)

    images_tensor = None
    embeddings_tensor = None

    for images, labels in tqdm(test_loader, desc="storing image features ..."):
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

    df = pd.read_csv(args.train_csv_file)

    matches_column = []
    for i in tqdm(range(len(df)), desc="matching labels to each posting ..."):
        matches_column.append(
            " ".join(
                list(df[df["label_group"] == df.iloc[i]["label_group"]]["posting_id"])
            )
        )
    df["macthes"] = matches_column

    index = faiss.IndexFlatIP(args.feature_dim)
    index.add(embeddings_tensor.numpy())
    distances, indices = index.search(
        embeddings_tensor.numpy(), k=50
    )  # search max 50 candidates

    print("simliarities")
    print(distances)

    print("\nindices")
    print(indices)

    # find similarity threshold for increasing f1 score about test set(using train set)

    import tensorflow as tf
    import tensorboard as tb

    tf.io.gfile = tb.compat.tensorflow_stub.io.gfile  # avoid tensorboard bug

    # just add part of embeddings for preveting tensorboard pending
    product_encoder.logger.experiment.add_embedding(
        mat=embeddings_tensor[:1000], label_img=images_tensor[:1000]
    )
    print("\nembedding projection was saved !!")
