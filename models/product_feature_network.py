from torchvision.models import *
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch_metric_learning import miners, losses
from pytorch_metric_learning.distances import CosineSimilarity

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

        self.feature_dim = feature_dim
        self.backbone_net = eval(backbone_net)(pretrained=True)

        self.feature_layer = nn.Linear(self.backbone_net.fc.out_features, feature_dim, bias=False)
        nn.init.kaiming_uniform_(self.feature_layer.weight)

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

        self.loss_func = losses.CrossBatchMemory(
            losses.ContrastiveLoss(pos_margin=1, neg_margin=0, distance=CosineSimilarity()),
            self.model.feature_dim, 
            memory_size=self.memory_batch_max_num, 
            miner=miners.MultiSimilarityMiner()
        )

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

        xbm_loss = self.loss_func(features, labels.squeeze(1))
        self.log("train/loss", xbm_loss, prog_bar=True)

        return xbm_loss

    def validation_step(self, validation_batch, batch_idx):
        self.model.eval()

        images, labels = validation_batch

        with torch.no_grad():
            features = self.model(images)

            xbm_loss = self.loss_func(features, labels.squeeze(1))
            self.log("val_loss", xbm_loss, prog_bar=True)

            return {
                "features": features,
                "labels": labels,
                "val_loss": xbm_loss,
            }
