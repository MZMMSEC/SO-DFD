"""

Author: Andreas Rössler
"""
import os
import argparse
import pdb

import torch
#import pretrainedmodels
import torch.nn as nn
import torch.nn.functional as F
from network.xception import xception, xception_concat
import math
import torchvision


def return_pytorch04_xception(pretrained=False):
    # Raises warning "src not broadcastable to dst" but thats fine
    model = xception(pretrained=False)
    if pretrained:
        # Load model in torch 0.4+
        model.fc = model.last_linear
        del model.last_linear
        state_dict = torch.load(
            # '/public/liuhonggu/.torch/models/xception-b5690688.pth'
            './checkpoints/xception-b5690688.pth'
            # r"D:\CVPR2023\codes\Xception\checkpoints\xception-b5690688.pth"
        )
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        model.load_state_dict(state_dict)
        model.last_linear = model.fc
        del model.fc
    return model


class TransferModel(nn.Module):
    """
    Simple transfer learning model that takes an imagenet pretrained model with
    a fc layer as base model and retrains a new fc layer for num_out_classes
    """
    def __init__(self, modelchoice, num_out_classes=2, dropout=0.5):
        super(TransferModel, self).__init__()
        self.modelchoice = modelchoice
        if modelchoice == 'xception':
            self.model = return_pytorch04_xception(pretrained=True)
            # Replace fc
            num_ftrs = self.model.last_linear.in_features
            if not dropout:
                self.model.last_linear = nn.Linear(num_ftrs, num_out_classes)
            else:
                print('Using dropout', dropout)
                self.model.last_linear = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes)
                )
        elif modelchoice == 'xception_concat':
            self.model = xception_concat()
            num_ftrs = self.model.last_linear.in_features
            if not dropout:
                self.model.last_linear = nn.Linear(num_ftrs, num_out_classes)
            else:
                print('Using dropout', dropout)
                self.model.last_linear = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes)
                )
        elif modelchoice == 'resnet50' or modelchoice == 'resnet18':
            if modelchoice == 'resnet50':
                self.model = torchvision.models.resnet50(pretrained=True)
            if modelchoice == 'resnet18':
                self.model = torchvision.models.resnet18(pretrained=True)
            # Replace fc
            num_ftrs = self.model.fc.in_features
            if not dropout:
                self.model.fc = nn.Linear(num_ftrs, num_out_classes)
            else:
                print('Using dropout', dropout)
                self.model.fc = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes)
                )
        else:
            raise Exception('Choose valid model, e.g. resnet50')

    def set_trainable_up_to(self, boolean, layername="Conv2d_4a_3x3"):
        """
        Freezes all layers below a specific layer and sets the following layers
        to true if boolean else only the fully connected final layer
        :param boolean:
        :param layername: depends on network, for inception e.g. Conv2d_4a_3x3
        :return:
        """
        # Stage-1: freeze all the layers
        if layername is None:
            for i, param in self.model.named_parameters():
                param.requires_grad = True
                return
        else:
            for i, param in self.model.named_parameters():
                param.requires_grad = False
        if boolean:
            # Make all layers following the layername layer trainable
            ct = []
            found = False
            for name, child in self.model.named_children():
                if layername in ct:
                    found = True
                    for params in child.parameters():
                        params.requires_grad = True
                ct.append(name)
            if not found:
                raise Exception('Layer not found, cant finetune!'.format(
                    layername))
        else:
            if self.modelchoice == 'xception':
                # Make fc trainable
                for param in self.model.last_linear.parameters():
                    param.requires_grad = True

            else:
                # Make fc trainable
                for param in self.model.fc.parameters():
                    param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x


import clip
class ViTCLIP_img(nn.Module):
    def __init__(self, name='ViT-B/32', embed_dim=512, out_channels=2):
        super(ViTCLIP_img, self).__init__()

        clip_model, preprocess, _, __ = clip.load(name, device='cuda', jit=False)

        self.vit_visual = clip_model.visual

        self.mlp_head = nn.Linear(embed_dim, out_channels)
        #     nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim//2),
        #     nn.ReLU(),
        #     nn.Linear(embed_dim//2, out_channels)
        # )
        self.preprocess = preprocess

    def forward(self, x):
        fea = self.vit_visual(x)
        x = self.mlp_head(fea)

        return x


def model_selection(modelname, num_out_classes,
                    dropout=None):
    """
    :param modelname:
    :return: model, image size, pretraining<yes/no>, input_list
    """
    if modelname == 'clip-vitb-32':
        return ViTCLIP_img(name='ViT-B/32', out_channels=num_out_classes)

    if modelname == 'xception':
        return TransferModel(modelchoice='xception',
                             num_out_classes=num_out_classes)
    #    , 299, \True, ['image'], None
    elif modelname == 'resnet18':
        return TransferModel(modelchoice='resnet18', dropout=dropout,
                             num_out_classes=num_out_classes)
    #    , \224, True, ['image'], None
    elif modelname == 'xception_concat':
        return TransferModel(modelchoice='xception_concat',
                             num_out_classes=num_out_classes)
    elif modelname == 'resnet50':
        return TransferModel(modelchoice='resnet50',
                             num_out_classes=num_out_classes)
    else:
        raise NotImplementedError(modelname)


if __name__ == '__main__':
    model, image_size, *_ = model_selection('xception', num_out_classes=2)
    print(model)
    model = model.cuda()
    from torchsummary import summary
    input_s = (3, image_size, image_size)
    print(summary(model, input_s))
