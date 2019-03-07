import utils

import torch.nn as nn
import os

class InfoGANGenerator(nn.Module):
     # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
     # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
     def __init__(self, input_dim=100, output_dim=1, input_size=32):
         super(InfoGANGenerator, self).__init__()
         self.input_dim = input_dim
         self.output_dim = output_dim
         self.input_size = input_size

         self.fc = nn.Sequential(
             nn.Linear(self.input_dim, 1024),
             nn.BatchNorm1d(1024),
             nn.ReLU(),
             nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
             nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
             nn.ReLU(),
         )
         self.deconv = nn.Sequential(
             nn.ConvTranspose2d(128, 64, 4, 2, 1),
             nn.BatchNorm2d(64),
             nn.ReLU(),
             nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
             nn.Tanh(),
         )
         utils.initialize_weights(self)

     def forward(self, input):
         x = self.fc(input)
         x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
         x = self.deconv(x)

         return x

class InfoGANDiscriminator(nn.Module):
     # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
     # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
     def __init__(self, input_dim=1, output_dim=1, input_size=32):
         super(InfoGANDiscriminator, self).__init__()
         self.input_dim = input_dim
         self.output_dim = output_dim
         self.input_size = input_size

         self.conv = nn.Sequential(
             nn.Conv2d(self.input_dim, 64, 4, 2, 1),
             nn.LeakyReLU(0.2),
             nn.Conv2d(64, 128, 4, 2, 1),
             nn.BatchNorm2d(128),
             nn.LeakyReLU(0.2),
         )
         self.fc = nn.Sequential(
             nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
             nn.BatchNorm1d(1024),
             nn.LeakyReLU(0.2),
             nn.Linear(1024, self.output_dim),
             # nn.Sigmoid(),
         )
         utils.initialize_weights(self)

     def forward(self, input):
         x = self.conv(input)
         x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
         x = self.fc(x)

         return x

class InfoGANDiscriminatorClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, input_size, save_dir, model_name):
        super(InfoGANDiscriminatorClassifier, self).__init__()
        self.wrapped = InfoGANDiscriminator(input_dim=input_dim, output_dim=output_dim, input_size=input_size)

        # load weights
        print("loading state dict...")
        state_dict = torch.load(os.path.join(save_dir, model_name + '_G.pkl'))
        self.wrapped.load_state_dict(state_dict)

        # freeze weights
        for param in self.wrapped.parameters():
            param.requires_grad = False

        # add fine tuning layer
        self.wrapped.fc[3] = nn.Linear(1024, 1)

    def forward(self, x):
        return self.wrapped.forward(x)

    def parameters(self):
        ret = []
        for param in self.wrapped.parameters():
            if param.requires_grad == True:
                ret.append(param)

        return ret



