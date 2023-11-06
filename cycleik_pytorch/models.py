import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GenericGenerator(nn.Module):
    def __init__(self, input_size=8, output_size=8, layers=[], activation="GELU", nbr_tanh=2):
        super(GenericGenerator, self).__init__()

        self.activation = None
        if activation == "GELU":
            self.activation = F.gelu
        elif activation == "LeakyReLu":
            self.activation = F.leaky_relu
        elif activation == "CELU":
            self.activation = F.celu
        elif activation == "SELU":
            self.activation = F.selu
        else:
            self.activation = F.gelu

        self.nbr_tanh = nbr_tanh
        self.input_dim = input_size
        self.output_dim = output_size
        self.hidden_dim = layers
        current_dim = input_size
        self.layers = nn.ModuleList()
        for hdim in layers:
            next_layer = nn.Linear(current_dim, hdim)
            torch.nn.init.xavier_normal_(next_layer.weight)
            self.layers.append(next_layer)
            #self.layers.append(nn.BatchNorm1d(hdim, affine=False))
            #torch.nn.init.kaiming_uniform_(self.layers[-1].weight, a=math.sqrt(5))
            #torch.nn.init.normal_(self.layers[-1].weight, mean=0.0, std=1.0)
            #torch.nn.init.zeros_(self.layers[-1].bias)
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, output_size))

    def forward(self, x):
        for e, layer in enumerate(self.layers):
            if e < len(self.layers) - self.nbr_tanh:
                #x = self.activation(torch.nn.functional.dropout(layer(x), p=0.05))
                #print(type(layer))
                #if type(layer) is nn.BatchNorm1d:
                #    #print("--------------")
                #    layer(x)
                #else:
                #    x = self.activation(layer(x))
                x = self.activation(layer(x))
            else:
                x = torch.tanh(layer(x))
        return x

class GenericNoisyGenerator(nn.Module):
    def __init__(self, noise_vector_size=8, input_size=7, output_size=8, layers=[], activation="GELU", nbr_tanh=2):
        super(GenericNoisyGenerator, self).__init__()
        self.activation = None
        if activation == "GELU":
            self.activation = F.gelu
        elif activation == "LeakyReLu":
            self.activation = F.leaky_relu
        elif activation == "CELU":
            self.activation = F.celu
        elif activation == "SELU":
            self.activation = F.selu
        else:
            self.activation = F.gelu

        self.nbr_tanh = nbr_tanh
        self.input_dim = input_size
        self.output_dim = output_size
        self.hidden_dim = layers
        current_dim = noise_vector_size + input_size
        self.layers = nn.ModuleList()
        for hdim in layers:
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, output_size))

    def forward(self, z, x):
        input_tensor = torch.concat((z, x), dim=1)
        for e, layer in enumerate(self.layers):
            if e < len(self.layers) - self.nbr_tanh:
                input_tensor = self.activation(layer(input_tensor))
            else:
                input_tensor = torch.tanh(layer(input_tensor))
        return input_tensor
