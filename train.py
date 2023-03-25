import torch
from tqdm import tqdm
import os
from time import sleep
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Discriminator
from model import Generator
from data import ImageDataSet
from data import transforms

IMAGE_SIZE = 128
CHANNELS_IMG = 1
LEARNING_RATE = 2e-4
Z_DIM = 100
FEATURES_DISC = 64
FEATURES_GEN = 64

img_dir = input("Folder Containing Image Data: ")
while not os.path.is_dir(img_dir):
  img_dir = input("Type correct Image Data Folder: ")

BATCH_SIZE = int(input("Batch Size (Default 32): ") or 32)
NUM_EPOCHS = int(input("Number of Epochs (Default 32): ") or 32)


dataset = ImageDataSet(img_dir,transform=transforms)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

def initialize_weights(model):
  for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
      nn.init.normal_(m.weight.data,0.0,0.02)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()

step = 0

gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
  for batch_idx, real in enumerate(tqdm(dataloader)):
    real = real.to(device)
    noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
    fake = gen(noise)
    # Train Discriminator
    disc_real = disc(real).reshape(-1)
    loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
    disc_fake = disc(fake.detach()).reshape(-1)
    loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
    loss_disc = (loss_disc_real + loss_disc_fake) / 2
    disc.zero_grad()
    loss_disc.backward(retain_graph=True)
    opt_disc.step()

    ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
    output = disc(fake).reshape(-1)
    loss_gen = criterion(output, torch.ones_like(output))
    gen.zero_grad()
    loss_gen.backward(retain_graph=True)
    opt_gen.step()
    # Print losses occasionally and print to tensorboard
    step += 1
  print(
        f"Epoch [{epoch}/{NUM_EPOCHS}] Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
  )

torch.save(gen.state_dict(), 'generator.pth')
