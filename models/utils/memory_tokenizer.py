import torch.nn as nn
from einops.layers.torch import Rearrange
import torch

class VerticalTokenizer(nn.Module):
    """
    This class contains the module of the ViT that generates vertical patch embeddings from the input image.
    Attributes:
    projection        object of nn.Sequential class that splits the image into patches
                      and generates projections in the size of the embedding
    positions         object of nn.Parameter class that generates the positional encoding vectors to add to all patch
                      embeddings
    """
    def __init__(self, img_size, width_patches, dim, channels=3):
        super().__init__()
        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=width_patches, p2=img_size),
            nn.Linear(channels * img_size * width_patches, dim),
        )

    def forward(self, x):
        x = self.projection(x)
        return x

    def sequence_length(self, n_channels=3, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Sequential):
            nn.init.kaiming_normal_(m.weight)
    

    
    