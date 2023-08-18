from __future__ import print_function

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from torchsummary import summary
from tqdm import tqdm
from torch_lr_finder import LRFinder
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from loss import YoloLoss
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples
)



""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""

model_config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)

        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(
                2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1
            ),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )
        

class YOLOv3(pl.LightningModule):
    def __init__(self,config,train_loader,test_loader, in_channels=3, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()
        
        self.config = config
        self.scaled_anchors = (torch.tensor(self.config.ANCHORS)* torch.tensor(self.config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2))
        self.loss_fn = YoloLoss()
        self.train_loader = train_loader
        self.test_loader = test_loader
        

    def forward(self, x):
        outputs = []  # for each scale
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in model_config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats,))

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2),)
                    in_channels = in_channels * 3

        return layers
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-05,weight_decay = 1e-4)
        scheduler = OneCycleLR(optimizer, max_lr=1e-03, steps_per_epoch=len(self.train_loader), epochs=40,div_factor=100,pct_start = 5/40)
        return [optimizer],[scheduler]
    
    def training_step(self, batch, batch_idx):
        x,y = batch
        y0, y1, y2 = (
            y[0],
            y[1],
            y[2]
        )
        
        out = self(x)
        train_loss = (
                loss_fn(out[0], y0, self.scaled_anchors[0])
                + loss_fn(out[1], y1, self.scaled_anchors[1])
                + loss_fn(out[2], y2, self.scaled_anchors[2])
            )
        
        self.log("train_loss", train_loss,prog_bar=True, on_step=True, on_epoch=True)
        
    def validation_step(self,batch,batch_idx):
        pass
        
        
    def on_epoch_start(self):
         plot_couple_examples(self, self.test_loader, 0.6, 0.5, self.scaled_anchors)
        
    def training_epoch_end(self):
        current_epoch = self.current_epoch
        if self.config.SAVE_MODEL:
            save_checkpoint(self, self.optimizers(), filename=f"checkpoint_{current_epoch}.pth.tar")
        check_class_accuracy(self, self.train_loader, threshold=self.config.CONF_THRESHOLD,logger=self.log)
        
    def validation_epoch_end(self):
        current_epoch = self.current_epoch
        if current_epoch % 10 == 0 or current_epoch in (1,2,3):
            check_class_accuracy(self, self.test_loader, threshold=self.config.CONF_THRESHOLD,logger=self.log)
            pred_boxes, true_boxes = get_evaluation_bboxes(
                self.test_loader,
                self,
                iou_threshold=self.config.NMS_IOU_THRESH,
                anchors=self.config.ANCHORS,
                threshold=self.config.CONF_THRESHOLD,
            )
            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=self.config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=self.config.NUM_CLASSES,
            )
            print(mapval)
            self.log("MAP", mapval.item(),prog_bar=True, on_step=True, on_epoch=True)
    
