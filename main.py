from models.networks import Generator, Discriminator
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

config = {
    'device': "cuda" if torch.cuda.is_available() else "cpu",
    'lr': 3e-4,
    'epochs': 50,
    'batch_size': 32
}


def load_dataset():
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])
    mnist = datasets.MNIST(root='datasets', train=True,
                           transform=transformations, download=True)
    return DataLoader(mnist, batch_size=config['batch_size'], shuffle=True)


train_data = load_dataset()


gen = Generator().to(config['device'])
disc = Discriminator().to(config['device'])

optimiser_g = optim.Adam(params=gen.parameters(), lr=config['lr'])
optimiser_d = optim.Adam(params=disc.parameters(), lr=config['lr'])

loss_fn = nn.BCELoss()

for epoch in range(config['epoch']):
    for batch_idx, (real, label) in enumerate(train_data):

        noise = torch.randn(config['batch_size'], 100).to(
            config['device'])  # Create a random probability distribution
        fake = gen(noise)  # Generate a fake number
        # pass the real number through the discriminator
        disc_real = disc(real).view(-1)
        # calculate the loss for real image
        lossD_real = loss_fn(disc_real, torch.ones_like(disc_real))
        # pass the fake number through the discriminator
        disc_fake = disc(fake).view(-1)
        lossD_fake = loss_fn(disc_fake, torch.zeros_like(
            disc_fake))  # calculate the loss for fake image
        lossD = (lossD_real + lossD_fake) / 2  # calculate the average loss

        # update the weights for the discriminator
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        optimiser_d.step()

        output = disc(fake).view(-1)
        # calculate the error between the fake image and the true image
        lossG = loss_fn(output, torch.ones_like(output))

        # update the weights for the generator
        gen.zero_grad()
        lossG.backward()
        optimiser_g.step()
