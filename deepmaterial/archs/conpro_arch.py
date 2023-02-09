from torch import nn as nn
import torch

from deepmaterial.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class UMAP(nn.Module):
    def __init__(self, dims, inC, outC):
        super(UMAP, self).__init__()
        self.encoder = nn.Sequential([
            nn.Conv2d(
                inC, 64, kernel_size=3, strides=(2, 2), padding="same",
            ),
            nn.ReLU(True),
            nn.Conv2D(
                64, 128, kernel_size=3, strides=(2, 2), padding="same"
            ),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(6272,512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, outC),
        ])
    
    def forward(self, x):
        return self.encoder(x)

@ARCH_REGISTRY.register()
class DenseICNN(nn.Module):
    def __init__(self, input_dim, output_dim,  hidden_layer_sizes, activation='celu', dropout=0.3):
        super(DenseICNN, self).__init__()
        
        
        self.hidden_layer_sizes = hidden_layer_sizes
        self.droput = dropout
        self.activation = activation

        
        self.quadratic_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, output_features, bias=True),
                nn.Dropout(dropout))
            for output_features in hidden_layer_sizes])
        
        sizes = zip(hidden_layer_sizes[:-1], hidden_layer_sizes[1:])
        self.convex_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_features, output_features, bias=False),
                nn.Dropout(dropout))
            for (input_features, output_features) in sizes])
        
        self.final_layer = nn.Linear(hidden_layer_sizes[-1], output_dim, bias=False)
        self.sig = nn.Sigmoid()
        
    def forward(self, input):
        output = self.quadratic_layers[0](input)
        for quadratic_layer, convex_layer in zip(self.quadratic_layers[1:], self.convex_layers):
            output = convex_layer(output) + quadratic_layer(input)
            if self.activation == 'celu':
                output = torch.celu(output)
        output = self.final_layer(output)
        return self.sig(output)
    
    def convexify(self):
        for layer in self.convex_layers:
            for sublayer in layer:
                if (isinstance(sublayer, nn.Linear)):
                    sublayer.weight.data.clamp_(0)