from torchvision.models import *
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

import PIL
import pandas as pd


class ProductFeatureNet(nn.Module):
    def __init__(self, backbone_net: str, feature_dim=256):
        super(ProductFeatureNet, self).__init__()

        self.backbone_net = eval(backbone_net)(pretrained=True)
        self.feature_layer = nn.Linear(
            self.backbone_net.fc.out_features, feature_dim, bias=False
        )
        nn.init.xavier_uniform_(self.feature_layer.weight)

    def forward(self, images):
        features = self.backbone_net(images)
        features = self.feature_layer(features)
        features = F.normalize(features)

        return features


class ProductFeatureEncoder(pl.LightningModule):
    def __init__(
        self,
        model,
        margin=0.5,
        lr=1e-3,
        lr_patience=2,
        lr_decay_ratio=0.5,
        memory_batch_max_num=2048,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = model
        self.margin = margin
        self.lr = lr
        self.lr_patience = lr_patience
        self.lr_decay_ratio = lr_decay_ratio

        self.memory_batch_max_num = memory_batch_max_num
        self.memory_batch_features = None
        self.memory_batch_labels = None

    def forward(self, images):
        features = self.model(images)

        return features

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            [
                {"params": self.model.backbone_net.parameters(), "lr": self.lr * 0.1},
                {"params": self.model.feature_layer.parameters()},
            ],
            #self.parameters(),
            lr=self.lr,
            weight_decay=0.01
        )

        return {
            "optimizer": optim,
            "lr_scheduler": ReduceLROnPlateau(
                optim,
                patience=self.lr_patience,
                threshold=1e-8,
                factor=self.lr_decay_ratio,
            ),
            "monitor": "val_loss",
        }

    def training_step(self, train_batch, batch_idx):
        self.model.train()

        images, labels = train_batch
        features = self.model(images)

        # update memory batch
        if self.memory_batch_features is not None:
            self.memory_batch_features = torch.cat(
                [self.memory_batch_features, features.detach()]
            )
            self.memory_batch_labels = torch.cat(
                [self.memory_batch_labels, labels.detach()]
            )
        else:
            self.memory_batch_features = features.detach()
            self.memory_batch_labels = labels.detach()

        self.memory_batch_features = self.memory_batch_features[
            -self.memory_batch_max_num :
        ]
        self.memory_batch_labels = self.memory_batch_labels[
            -self.memory_batch_max_num :
        ]

        # cross-batch contrastive loss
        xbm_negative_pairs = (labels != self.memory_batch_labels.t()).float()
        xbm_positive_pairs = (labels == self.memory_batch_labels.t()).float()
        xbm_cosine_similarities = torch.mm(features, self.memory_batch_features.t())

        #back-prop only current batch(not in memory batch)
        xbm_negative_loss = (xbm_cosine_similarities * xbm_negative_pairs).clamp(min=0).sum() / labels.size(0)
        xbm_positive_loss = ((1 - xbm_cosine_similarities) * xbm_positive_pairs).sum() / labels.size(0)
        xbm_loss = xbm_positive_loss + xbm_negative_loss

        self.log("train/neg_loss", xbm_negative_loss, prog_bar=True)
        self.log("train/pos_loss", xbm_positive_loss, prog_bar=True)
        self.log("train/loss", xbm_loss, prog_bar=True)

        return xbm_loss

    def validation_step(self, validation_batch, batch_idx):
        self.model.eval()

        images, labels = validation_batch
        features = self.model(images)

        with torch.no_grad():
            if self.memory_batch_features is not None:
                self.memory_batch_features = torch.cat(
                    [self.memory_batch_features, features.detach()]
                )
                self.memory_batch_labels = torch.cat(
                    [self.memory_batch_labels, labels.detach()]
                )
            else:
                self.memory_batch_features = features.detach()
                self.memory_batch_labels = labels.detach()

            self.memory_batch_features = self.memory_batch_features[
                -self.memory_batch_max_num :
            ]
            self.memory_batch_labels = self.memory_batch_labels[
                -self.memory_batch_max_num :
            ]

            # cross-batch contrastive loss
            xbm_negative_pairs = (labels != self.memory_batch_labels.t()).float()
            xbm_positive_pairs = (labels == self.memory_batch_labels.t()).float()
            xbm_cosine_similarities = torch.mm(features, self.memory_batch_features.t())

            #back-prop only current batch(not in memory batch)
            xbm_negative_loss = (xbm_cosine_similarities * xbm_negative_pairs).clamp(min=0).sum() / labels.size(0)
            xbm_positive_loss = ((1 - xbm_cosine_similarities) * xbm_positive_pairs).sum() / labels.size(0)
            xbm_loss = xbm_positive_loss + xbm_negative_loss
            self.log("val_loss", xbm_loss, prog_bar=True)

        return {
            "features": features,
            "labels": labels,
            "val_loss": xbm_loss,
        }
