from transformers import AutoBackbone
from PIL import Image
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

class And(nn.Module):
    def __init__(self, in_dims=[48, 96, 192, 384], out_dims=64):
        super().__init__()
        self.convs = [nn.Conv2d(in_dim, out_dims, (3, 3), padding=(1, 1), bias=False) for in_dim in in_dims]
        self.convs = nn.ModuleList(self.convs)
        
    def __call__(self, features):
        return [self.convs[i](feature) for i, feature in enumerate(features)]
    
    def load_weights(self, weights):
        new_state_dict = {x: y for x, y in weights.items() if x.startswith('conv')}
        self.load_state_dict(new_state_dict)

class OnwardsLayer(nn.Module):
    def __init__(self, input_dim, output_dim, scale_factor):
        super().__init__()
        self.feature_resizing = nn.Conv2d(input_dim, output_dim, (1, 1), padding=0)
        if scale_factor > 1:
            self.spatial_resizing = nn.ConvTranspose2d(output_dim, 
                                                       output_dim,
                                                       (scale_factor, scale_factor), 
                                                       padding=0, 
                                                       stride=(scale_factor, scale_factor))

        elif scale_factor < 1:
            stride = int(round(1/scale_factor))
            self.spatial_resizing = nn.Conv2d(output_dim, 
                                              output_dim, 
                                              (3, 3), 
                                              padding=1, 
                                              stride=(stride, stride))
            
        else:
            self.spatial_resizing = nn.Identity()

    def forward(self, x, patch_height=None, patch_width=None):
        x = self.feature_resizing(x)
        x = self.spatial_resizing(x)
        return x

    def load_weights(self, state_dict):
        self.load_state_dict(state_dict)

class Onwards(nn.Module):
    def __init__(self, feature_size=384, scale_factors = [4, 2, 1, 0.5]):
        super().__init__()
        self.layers = [OnwardsLayer(feature_size, 
                                    int(feature_size // (2*scale_factor)),
                                    scale_factor) for scale_factor in scale_factors]
        self.layers = nn.ModuleList(self.layers)

    def forward(self, hidden_states, patch_height, patch_width):
        new_hidden_states = []
        for i in range(len(hidden_states)):
            x = hidden_states[i][:, 1:, :]
            b, hw, c = x.shape
            x = x.reshape(b, patch_height, patch_width, c).permute((0, 3, 1, 2))
            hidden_states_post = self.layers[i](x)
            new_hidden_states.append(hidden_states_post)
        return new_hidden_states

    def load_weights(self, state_dict):
        new_state_dict = dict()
        for k, v in state_dict.items():
            new_state_dict[k.replace('resize', 'spatial_resizing').replace('projection', 'feature_resizing')] = v
        self.load_state_dict(new_state_dict)

class UpwardsLayerResidual(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, (3, 3), padding=1)
        
    def forward(self, inp):
        x = F.relu(inp)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        return x + inp

class UpwardsLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.final_conv = nn.Conv2d(hidden_dim, hidden_dim, (1, 1), padding=0)
        self.res1 = UpwardsLayerResidual(hidden_dim)
        self.res2 = UpwardsLayerResidual(hidden_dim)

    def forward(self, x, scale, accumulator=None):
        if accumulator is not None:
            if x.shape != accumulator.shape:
                accumulator = F.interpolate(accumulator, 
                                       size=x.shape[-2:], 
                                       mode='bilinear', 
                                       align_corners=False)
            x = x + self.res1(accumulator)
        x = self.res2(x)
        x = F.interpolate(x, 
                       size=scale, 
                       mode='bilinear', 
                       align_corners=True)
        x = self.final_conv(x)
        return x

    def load_weights(self, state_dict):
        self.load_state_dict(state_dict)

class Upwards(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.layers = [UpwardsLayer(hidden_dim) for _ in range(4)]
        self.layers = nn.ModuleList(self.layers)

    def forward(self, hidden_states):
        hidden_states = hidden_states[::-1]

        new_hidden_states = []
        acc = None
        for i in range(len(hidden_states)):
            if i != len(hidden_states)-1:
                scale = hidden_states[i+1].shape[-2:]
            else:
                scale = [2*x for x in hidden_states[i].shape[-2:]]
            x = hidden_states[i]
            if acc is None:
                x = self.layers[i](x, accumulator=acc, scale=scale)
            else:
                x = self.layers[i](acc, accumulator=x, scale=scale)
            acc = x
            new_hidden_states.append(x)
        return new_hidden_states

    def load_weights(self, state_dict):
        new_state_dict = dict()
        for k, v in state_dict.items():
            new_state_dict[k.replace('projection', 'final_conv').replace('residual_layer', 'res').replace('convolution', 'conv')] = v
        self.load_state_dict(new_state_dict, strict=True)
        
class Head(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, num_classes, (1, 1))

    def forward(self, hidden_states, patch_height=None, patch_width=None):
        x = hidden_states[-1]
        x = self.conv1(x)
        x = F.interpolate(x, (patch_height*14, patch_width*14), align_corners=True, mode='bilinear')
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)        
        return x.squeeze(1)

    def load_weights(self, state_dict):
        self.load_state_dict(state_dict)

class SimpleSegmentationModel2(nn.Module):
    def __init__(self, num_classes):
        super(SimpleSegmentationModel, self).__init__()
        # Use a pre-trained ResNet-18 as the backbone
        self.backbone = AutoBackbone.from_pretrained("facebook/dinov2-small")
        # Add a convolutional layer for segmentation
        self.head = Head(384, num_classes)
        self.patch_size = 14

    def forward(self, x):
        x = self.backbone(x).feature_maps
        h, w = x[0].shape[-2:]
        x = self.head(x, h, w)
        return x
        
class SimpleSegmentationModel(nn.Module):
    def __init__(self, num_classes):
        super(SimpleSegmentationModel, self).__init__()
        # Use a pre-trained ResNet-18 as the backbone
        self.backbone = AutoBackbone.from_pretrained("facebook/dinov2-small")
        # Add a convolutional layer for segmentation
        self.conv1 = nn.Conv2d(384, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = nn.ConvTranspose2d(256, num_classes, kernel_size=14, stride=(14, 14))        
        self.conv2 = nn.Conv2d(32, num_classes, kernel_size=1, stride=(1, 1))        
        # Upsample to the original image size
        self.upsample = nn.Upsample(scale_factor=14, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.backbone.forward_with_filtered_kwargs(x)
        x = x.feature_maps[0]
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x


class DPT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.patch_size = 14
        self.bb = AutoBackbone.from_pretrained('facebook/dinov2-small', 
                                               reshape_hidden_states=False, 
                                               out_features=['stage3', 'stage6', 'stage9', 'stage12'], 
                                               out_indices=[3, 6, 9, 12])
        self.upwards = Upwards()
        self.onwards = Onwards()
        self.and_ = And()
        self.head = Head(64, num_classes)
        self.num_classes = num_classes
        
    def load_weights(self, model):
        self.bb.load_state_dict(model.backbone.state_dict())
        self.upwards.load_weights(model.neck.fusion_stage.state_dict())
        self.and_.load_weights(model.neck.state_dict())
        self.onwards.load_weights(model.neck.reassemble_stage.state_dict())
        self.head.load_weights(model.head.state_dict())
    
    def forward(self, input_data):
        feats = self.bb(input_data).feature_maps
        _, _, height, width = input_data.shape
        patch_size = self.patch_size
        patch_height = height // patch_size
        patch_width = width // patch_size

        x = self.onwards(feats, patch_height, patch_width)
        x = self.and_(x)
        x = self.upwards(x)

        x = self.head(x, patch_height, patch_width)
        return x

def build(model_type, num_classes):
    if model_type == 'simple':
        return SimpleSegmentationModel(num_classes)
    elif model_type == 'dpt':
        return DPT(num_classes)
