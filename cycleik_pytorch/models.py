# Copyright (c) 2023. Jan-Gerrit Habekost. GNU General Public License https://www.gnu.org/licenses/gpl-3.0.html.

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, input_size=8, output_size=8):
        super(Generator, self).__init__()
        #self.main = nn.Sequential(
        #    # Input
        #    #nn.Flatten(),
        #    nn.Linear(in_features=input_size, out_features=512),
        #    nn.CELU(inplace=True),
        #
        #    # hidden layers
        #    nn.Linear(in_features=512, out_features=1024),
        #    #nn.Dropout(p=0.05),
        #    nn.CELU(inplace=True),
        #
        #    nn.Linear(in_features=1024, out_features=1024),
        #    # nn.Dropout(p=0.05),
        #    nn.CELU(inplace=True),
        #
        #    #
        #    nn.Linear(in_features=1024, out_features=512),
        #    #nn.Dropout(p=0.05),
        #    nn.CELU(inplace=True),
        #
        #    #nn.Linear(in_features=512, out_features=512),
        #    # nn.Dropout(p=0.01),
        #    #nn.LeakyReLU(inplace=True),
        #
        #    nn.Linear(in_features=512, out_features=256),
        #    #nn.Dropout(p=0.05),
        #    nn.CELU(inplace=True),
        #
        #    nn.Linear(in_features=256, out_features=128),
        #    #nn.Dropout(p=0.05),
        #    nn.CELU(inplace=True),
        #
        #    nn.Linear(in_features=128, out_features=64),
        #    #nn.Dropout(p=0.05),
        #    nn.CELU(inplace=True),
        #
        #    nn.Linear(in_features=64, out_features=32),
        #    #nn.Dropout(p=0.05),
        #    nn.CELU(inplace=True),
        #
        #    nn.Linear(in_features=32, out_features=20),
        #    # nn.Dropout(p=0.05),
        #    nn.CELU(inplace=True),
        #
        #    # output
        #    nn.Linear(in_features=20, out_features=output_size),
        #    nn.Tanh()
        #)

        self.main = nn.Sequential(
            # Input
            # nn.Flatten(),
            nn.Linear(in_features=input_size, out_features=2700),
            nn.GELU(),

            # hidden layers
            nn.Linear(in_features=2700, out_features=2600),
            # nn.Dropout(p=0.05),
            nn.GELU(),

            nn.Linear(in_features=2600, out_features=2400),
            # nn.Dropout(p=0.05),
            nn.GELU(),

            #
            nn.Linear(in_features=2400, out_features=500),
            # nn.Dropout(p=0.05),
            nn.GELU(),

            # nn.Linear(in_features=512, out_features=512),
            # nn.Dropout(p=0.01),
            # nn.LeakyReLU(inplace=True),

            nn.Linear(in_features=500, out_features=430),
            # nn.Dropout(p=0.05),
            nn.GELU(),

            nn.Linear(in_features=430, out_features=120),
            # nn.Dropout(p=0.05),
            nn.GELU(),

            nn.Linear(in_features=120, out_features=60),
            # nn.Dropout(p=0.05),
            nn.GELU(),

            nn.Linear(in_features=60, out_features=420),
            # nn.Dropout(p=0.05),
            nn.Tanh(),

            nn.Linear(in_features=420, out_features=output_size),
            # nn.Dropout(p=0.05),
            nn.Tanh(),

        )

        #self.softmax = nn.Softmax(dim=output_size)
        #self.hidden_state = torch.zeros(output_size).cuda()
        #self.hidden_state = self.hidden_state.view(1,1,output_size)
        #self.output_size = output_size
        #self.gru =  nn.GRU(input_size=output_size, hidden_size=output_size, num_layers=1, batch_first=True)

    def forward(self, x):
        return self.main(x)


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
            self.layers.append(nn.Linear(current_dim, hdim))
            #torch.nn.init.kaiming_uniform_(self.layers[-1].weight, a=math.sqrt(5))
            #torch.nn.init.normal_(self.layers[-1].weight, mean=0.0, std=1.0)
            #torch.nn.init.zeros_(self.layers[-1].bias)
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, output_size))

    def forward(self, x):
        for e, layer in enumerate(self.layers):
            if e < len(self.layers) - self.nbr_tanh:
                #x = self.activation(torch.nn.functional.dropout(layer(x), p=0.05))
                x = self.activation(layer(x))
            else:
                x = torch.tanh(layer(x))
        return x


class NoisyGenerator(nn.Module):
    def __init__(self, noise_vector_size=3, input_size=8, output_size=8):
        super(NoisyGenerator, self).__init__()
        #self.main = nn.Sequential(
        #    # Input
        #    #nn.Flatten(),
        #    nn.Linear(in_features=(noise_vector_size + input_size), out_features=512),
        #    nn.CELU(),
        #
        #    # hidden layers
        #    nn.Linear(in_features=512, out_features=1024),
        #    #nn.Dropout(p=0.05),
        #    nn.CELU(),
        #
        #    nn.Linear(in_features=1024, out_features=1024),
        #    # nn.Dropout(p=0.05),
        #    nn.CELU(),
        #
        #    #
        #    nn.Linear(in_features=1024, out_features=512),
        #    #nn.Dropout(p=0.05),
        #    nn.CELU(),
        #
        #    #nn.Linear(in_features=512, out_features=512),
        #    # nn.Dropout(p=0.01),
        #    #nn.LeakyReLU(inplace=True),
        #
        #    nn.Linear(in_features=512, out_features=256),
        #    #nn.Dropout(p=0.05),
        #    nn.CELU(),
        #
        #    nn.Linear(in_features=256, out_features=128),
        #    #nn.Dropout(p=0.05),
        #    nn.CELU(),
        #
        #    nn.Linear(in_features=128, out_features=64),
        #    #nn.Dropout(p=0.05),
        #    nn.CELU(),
        #
        #    nn.Linear(in_features=64, out_features=32),
        #    #nn.Dropout(p=0.05),
        #    nn.CELU(),
        #
        #    nn.Linear(in_features=32, out_features=20),
        #    # nn.Dropout(p=0.05),
        #    nn.CELU(),
        #
        #    # output
        #    nn.Linear(in_features=20, out_features=output_size),
        #    nn.Tanh()
        #)

        self.main = nn.Sequential(
            # Input
            # nn.Flatten(),
            nn.Linear(in_features=(noise_vector_size + input_size), out_features=1800),
            nn.GELU(),

            # hidden layers
            nn.Linear(in_features=1800, out_features=800),
            # nn.Dropout(p=0.05),
            nn.GELU(),

            nn.Linear(in_features=800, out_features=900),
            # nn.Dropout(p=0.05),
            nn.GELU(),

            #
            nn.Linear(in_features=900, out_features=800),
            # nn.Dropout(p=0.05),
            nn.GELU(),

            # nn.Linear(in_features=512, out_features=512),
            # nn.Dropout(p=0.01),
            # nn.LeakyReLU(inplace=True),

            nn.Linear(in_features=800, out_features=110),
            # nn.Dropout(p=0.05),
            nn.GELU(),

            nn.Linear(in_features=110, out_features=350),
            # nn.Dropout(p=0.05),
            nn.GELU(),

            nn.Linear(in_features=350, out_features=170),
            # nn.Dropout(p=0.05),
            nn.GELU(),

            nn.Linear(in_features=170, out_features=output_size),
            # nn.Dropout(p=0.05),
            nn.Tanh(),

            #nn.Linear(in_features=420, out_features=output_size),
            # nn.Dropout(p=0.05),
            #nn.Tanh(),

        )

        #self.softmax = nn.Softmax(dim=output_size)
        #self.hidden_state = torch.zeros(output_size).cuda()
        #self.hidden_state = self.hidden_state.view(1,1,output_size)
        #self.output_size = output_size
        #self.gru =  nn.GRU(input_size=output_size, hidden_size=output_size, num_layers=1, batch_first=True)

    def forward(self, z, x):
        return self.main(torch.concat((z, x), dim=1))

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


class Discriminator(nn.Module):
    def __init__(self, input_size=8, output_size=8):
        super(Discriminator, self).__init__()

        #self.main = nn.Sequential(
        #    # Input
        #    #nn.Flatten(),
        #    nn.Linear(in_features=input_size, out_features=512),
        #    nn.GELU(),
        #
        #    # hidden layers
        #    nn.Linear(in_features=512, out_features=1024),
        #    #nn.Dropout(p=0.05),
        #    nn.GELU(),
        #    #
        #    nn.Linear(in_features=1024, out_features=2048),
        #    #nn.Dropout(p=0.05),
        #    nn.GELU(),
        #    #
        #    nn.Linear(in_features=2048, out_features=1024),
        #    #nn.Dropout(p=0.05),
        #    nn.GELU(),
        #    #
        #    nn.Linear(in_features=1024, out_features=512),
        #    #nn.Dropout(p=0.05),
        #    nn.GELU(),
        #
        #    #nn.Linear(in_features=512, out_features=512),
        #    # nn.Dropout(p=0.01),
        #    #nn.LeakyReLU(inplace=True),
        #
        #    nn.Linear(in_features=512, out_features=256),
        #    #nn.Dropout(p=0.05),
        #    nn.GELU(),
        #
        #    nn.Linear(in_features=256, out_features=128),
        #    #nn.Dropout(p=0.05),
        #    nn.GELU(),
        #
        #    nn.Linear(in_features=128, out_features=64),
        #    #nn.Dropout(p=0.05),
        #    nn.GELU(),
        #
        #    nn.Linear(in_features=64, out_features=32),
        #    #nn.Dropout(p=0.05),
        #    nn.GELU(),
        #
        #    nn.Linear(in_features=32, out_features=20),
        #    # nn.Dropout(p=0.05),
        #    #nn.LeakyReLU(inplace=True),
        #    nn.Tanh(),
        #
        #    # output
        #    nn.Linear(in_features=20, out_features=output_size),
        #    nn.Tanh()
        #)

        #self.main = nn.Sequential(
        #    # Input
        #    # nn.Flatten(),
        #    nn.Linear(in_features=input_size, out_features=1900),
        #    nn.GELU(),
        #
        #    # hidden layers
        #    nn.Linear(in_features=1900, out_features=2700),
        #    # nn.Dropout(p=0.05),
        #    nn.GELU(),
        #    #
        #    nn.Linear(in_features=2700, out_features=3000),
        #    # nn.Dropout(p=0.05),
        #    nn.GELU(),
        #    #
        #    nn.Linear(in_features=3000, out_features=2900),
        #    # nn.Dropout(p=0.05),
        #    nn.GELU(),
        #    #
        #    nn.Linear(in_features=2900, out_features=450),
        #    # nn.Dropout(p=0.05),
        #    nn.GELU(),
        #
        #    nn.Linear(in_features=450, out_features=60),
        #    # nn.Dropout(p=0.05),
        #    nn.GELU(),
        #
        #    nn.Linear(in_features=60, out_features=10),
        #    # nn.Dropout(p=0.05),
        #    nn.Tanh(),
        #
        #    nn.Linear(in_features=10, out_features=160),
        #    # nn.Dropout(p=0.05),
        #    nn.Tanh(),
        #
        #    nn.Linear(in_features=160, out_features=output_size),
        #    # nn.Dropout(p=0.05),
        #    nn.Tanh()
        #)

        self.main = nn.Sequential(
            # Input
            # nn.Flatten(),
            nn.Linear(in_features=input_size, out_features=1900),
            nn.GELU(),

            # hidden layers
            nn.Linear(in_features=1900, out_features=2300),
            # nn.Dropout(p=0.05),
            nn.GELU(),
            #
            nn.Linear(in_features=2300, out_features=2400),
            # nn.Dropout(p=0.05),
            nn.GELU(),
            #
            nn.Linear(in_features=2400, out_features=1100),
            # nn.Dropout(p=0.05),
            nn.GELU(),
            #
            nn.Linear(in_features=1100, out_features=440),
            # nn.Dropout(p=0.05),
            nn.GELU(),

            nn.Linear(in_features=440, out_features=460),
            # nn.Dropout(p=0.05),
            nn.GELU(),

            nn.Linear(in_features=460, out_features=460),
            # nn.Dropout(p=0.05),
            nn.GELU(),

            nn.Linear(in_features=460, out_features=output_size),
            # nn.Dropout(p=0.05),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.main(x)


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
