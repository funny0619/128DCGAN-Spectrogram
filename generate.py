import torch
import torchvision
import os
from model import Generator
from tqdm import trange

state_dict = torch.load('generator.pth',map_location=torch.device('cpu'))
Z_DIM = 100
CHANNELS_IMG = 1
FEATURES_GEN = 64

BATCH_SIZE = int(input("How many images to generate: "))
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN)
gen.load_state_dict(state_dict)
fixed_noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1)
fake = gen(fixed_noise)

if not os.path.isdir("bmp_generated"):
    os.mkdir("bmp_generated")

for i in trange(BATCH_SIZE):
    fp = f"bmp_generated/{i}.bmp"
    img = torchvision.utils.save_image(fake[i],fp=fp,format="bmp",normalize=True)

