import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FineTuneModel(nn.Module):

    def __init__(self, core_model, output_dimension):
        super(FineTuneModel, self).__init__()

        self.prior_activation = F.tanh
        self.posteriori_activation = F.tanh
        self.activation = core_model.activation
        input_size = core_model.input_dim
        output_size = core_model.output_dim

        #for param in core_model.parameters():
        #    param.requires_grad = False

        self.layers = nn.ModuleList()
        #self.layers.append(nn.Linear(input_size, 2048))
        #self.layers.append(nn.Linear(2048, 2048))
        #self.layers.append(nn.Linear(2048, core_model.layers[1].in_features))
        self.layers.append(nn.Linear(input_size, core_model.layers[1].in_features))

        for i in range(len(core_model.layers)):
            if i == 0 or i == len(core_model.layers)- 1: continue
            #print(core_model.layers[i].parameters())
            #core_model.layers[i].requires_grad = False
            #core_model.layers[i].bias = None
            self.layers.append(core_model.layers[i])

        #self.layers.append(nn.Linear(core_model.layers[-2].out_features, 128))
        #self.layers.append(nn.Linear(128, 128))
        #self.layers.append(nn.Linear(128, output_dimension))
        self.layers.append(nn.Linear(core_model.layers[-2].out_features, output_dimension))

        #for layer in self.layers:
        #    print(layer)

    def forward(self, x):
        #print(x.shape)
        for e, layer in enumerate(self.layers):
            #print(e)
            if e <= 2:
                #print("prior")
                x = self.prior_activation(layer(x))
            elif 2 < e < (len(self.layers) - 3):
                #print("core")
                x = self.activation(layer(x))
            else:
                #print("post")
                x = self.posteriori_activation(layer(x))
        return x


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
            next_layer = nn.Linear(current_dim, hdim)
            torch.nn.init.xavier_normal_(next_layer.weight)
            self.layers.append(next_layer)
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



class GenericDiscriminator(nn.Module):

    def __init__(self, input_size=8, output_size=8, layers=[], activation="GELU", nbr_tanh=2):
        super(GenericDiscriminator, self).__init__()

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
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, output_size))

    def forward(self, x):
        for e, layer in enumerate(self.layers):
            if e < len(self.layers) - self.nbr_tanh:
                x = self.activation(layer(x))
            else:
                x = torch.tanh(layer(x))
        return x


class AutoEncoder(nn.Module):
    def __init__(self, size):
        super(AutoEncoder, self).__init__()
        self.to_latent = nn.Sequential(
            # Input
            #nn.Flatten(),
            #nn.BatchNorm1d(size),
            nn.Linear(in_features=size, out_features=6),
            #nn.BatchNorm1d(200),
            nn.ReLU(inplace=True),

            # hidden layers
            nn.Linear(in_features=6, out_features=1),
            #nn.BatchNorm1d(200),
            nn.ReLU(inplace=True),

            #nn.Linear(in_features=200, out_features=3)
        )

        self.to_origin = nn.Sequential(
            #nn.BatchNorm1d(3),
            nn.Linear(in_features=1, out_features=6),
            #nn.BatchNorm1d(200),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=6, out_features=size),
            #nn.BatchNorm1d(200),
            nn.ReLU(inplace=True),

            #nn.Linear(in_features=200, out_features=size)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.to_latent(x.float())
        #x = self.softmax(x)
        x = self.to_origin(x)
        #x = self.softmax(x)
        return x
