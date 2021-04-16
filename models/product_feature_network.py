from torchvision.models import *
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

    def forward(self, images):
        features = self.backbone_net(images)
        features = self.feature_layer(features)
        features = F.normalize(features)

        return features


class ProductFeatureEncoder(pl.LightningModule):
    def __init__(self, model, lr=1e-3, margin=0.5):
        super().__init__()

        self.save_hyperparameters()

        self.model = model
        self.lr = lr
        self.margin = margin
        self.csv_file = csv_file

    def forward(self, images):
        features = self.model(images)

        return features

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optim,
            "lr_scheduler": ReduceLROnPlateau(optim, patience=1, threshold=1e-7),
            "monitor": "val_loss",
        }

    def training_step(self, train_batch, batch_idx):
        _, images, labels = train_batch

        features = self.model(images)

        # multi-batch contrastive loss
        positive_pairs = (labels == labels.transpose(1, 0)).float()
        negative_pairs = (labels != labels.transpose(1, 0)).float()
        cosine_similarities = torch.mm(features, features.transpose(1, 0))
        similarity_loss = (
            ((negative_pairs * cosine_similarities) / 2).clamp(min=0.0)
            - (((positive_pairs * cosine_similarities) / 2).clamp(max=0.0))
        ).sum()

        self.log("train_loss", similarity_loss, prog_bar=True)

        return similarity_loss

    def validation_step(self, validation_batch, batch_idx):
        posting_ids, images, labels = validation_batch

        features = self.model(images)

        # multi-batch contrastive loss
        positive_pairs = (labels == labels.transpose(1, 0)).float()
        negative_pairs = (labels != labels.transpose(1, 0)).float()
        cosine_similarities = torch.mm(features, features.transpose(1, 0))
        similarity_loss = (
            ((negative_pairs * cosine_similarities) / 2).clamp(min=0.0)
            - (((positive_pairs * cosine_similarities) / 2).clamp(max=0.0))
        ).sum()

        self.log("val_loss", similarity_loss, prog_bar=True)

        return {
            "posting_ids": posting_ids,
            "features": features,
            "labels": labels,
            "val_loss": similarity_loss,
        }

