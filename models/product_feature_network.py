from torchvision.models import *
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import pipeline

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

import PIL
import pandas as pd


class ProductFeatureNet(nn.Module):
    def __init__(self, backbone_net: str, feature_dim=768):
        super(ProductFeatureNet, self).__init__()

        self.backbone_net = eval(backbone_net)(pretrained=True)
        self.text_encoder = pipeline(
            "feature-extraction",
            model="distilbert-base-uncased",
            tokenizer="distilbert-base-uncased",
            device=0,
        )
        self.feature_layer = nn.Linear(
            self.backbone_net.fc.out_features, feature_dim, bias=False
        )
        nn.init.xavier_uniform_(self.feature_layer.weight)

    def forward(self, images, texts):
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
        memory_batch_max_num=1024,
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
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
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

        # in-batch contrastive loss
        negative_pairs = (labels != labels.transpose(1, 0))
        positive_pairs = ((labels == labels.transpose(1, 0)).float() - torch.eye(labels.size(0)).type_as(labels)).bool()
        cosine_similarities = torch.mm(features, features.transpose(1, 0))

        if negative_pairs.float().sum() > 0:
            negative_loss = torch.masked_select(cosine_similarities, negative_pairs).mean() + self.margin
        else:
            negative_loss = 0.0
        
        if positive_pairs.float().sum() > 0:
            positive_loss = 1 - torch.masked_select(cosine_similarities, positive_pairs).mean()
        else:
            positive_loss = 0.0
        
        similarity_loss = positive_loss + negative_loss

        self.log("train/neg_loss", negative_loss, prog_bar=True)
        self.log("train/pos_loss", positive_loss, prog_bar=True)
        self.log("train/loss", similarity_loss, prog_bar=True)

        if self.memory_batch_features is not None:
            # cross-batch contrastive loss
            xbm_negative_pairs = (labels != self.memory_batch_labels.transpose(1, 0))
            xbm_positive_pairs = (labels == self.memory_batch_labels.transpose(1, 0))
            xbm_cosine_similarities = torch.mm(features, self.memory_batch_features.transpose(1, 0))

            if xbm_negative_pairs.float().sum() > 0:
                xbm_negative_loss = torch.masked_select(xbm_cosine_similarities, xbm_negative_pairs).mean() + self.margin
            else:
                xbm_negative_loss = 0.0
            
            if xbm_positive_pairs.float().sum() > 0:
                xbm_positive_loss = 1 - torch.masked_select(xbm_cosine_similarities, xbm_positive_pairs).mean()
            else:
                xbm_positive_loss = 0.0

            xbm_loss = xbm_positive_loss + xbm_negative_loss

            self.log("train/xbm_neg_loss", xbm_negative_loss, prog_bar=True)
            self.log("train/xbm_pos_loss", xbm_positive_loss, prog_bar=True)
            self.log("train/xbm_loss", xbm_loss, prog_bar=True)

            # update memory batch
            self.memory_batch_features = torch.cat(
                [self.memory_batch_features, features.detach()]
            )
            self.memory_batch_labels = torch.cat(
                [self.memory_batch_labels, labels.detach()]
            )

            self.memory_batch_features = self.memory_batch_features[
                -self.memory_batch_max_num :
            ]
            self.memory_batch_labels = self.memory_batch_labels[
                -self.memory_batch_max_num :
            ]

            return similarity_loss + xbm_loss

        else:
            self.memory_batch_features = features.detach()
            self.memory_batch_labels = labels.detach()

        return similarity_loss

    def training_epoch_end(self, outputs):
        if self.current_epoch == 1:
            sampleInput = torch.rand(1, 3, 224, 224).cuda()
            self.logger.experiment.add_graph(
                ProductFeatureEncoder(model=self.model), sampleInput
            )

    def validation_step(self, validation_batch, batch_idx):
        self.model.eval()

        images, labels = validation_batch

        features = self.model(images)

        # in-batch contrastive loss
        with torch.no_grad():
            negative_pairs = (labels != labels.transpose(1, 0))
            positive_pairs = ((labels == labels.transpose(1, 0)).float() - torch.eye(labels.size(0)).type_as(labels)).bool()
            cosine_similarities = torch.mm(features, features.transpose(1, 0))

            if negative_pairs.float().sum() > 0:
                negative_loss = torch.masked_select(cosine_similarities, negative_pairs).mean() + self.margin
            else:
                negative_loss = 0.0
            
            if positive_pairs.float().sum() > 0:
                positive_loss = 1 - torch.masked_select(cosine_similarities, positive_pairs).mean()
            else:
                positive_loss = 0.0
            
            similarity_loss = positive_loss + negative_loss

            self.log("val/neg_loss", negative_loss, prog_bar=True)
            self.log("val/pos_loss", positive_loss, prog_bar=True)
            self.log("val_loss", similarity_loss, prog_bar=True)

        return {
            "features": features,
            "labels": labels,
            "val_loss": similarity_loss,
        }
