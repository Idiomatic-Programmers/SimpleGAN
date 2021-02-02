import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 784),
            nn.Tanh(),  # make outputs [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(784, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # make outputs [0, 1]
        )

    def forward(self, x):
        return self.disc(x)
