import torch
import torch.nn as nn

class Discriminator(nn.Module):
  def __init__(self, channels_img, features_d):
    super(Discriminator, self).__init__()
    # didnt use batchnorm in the first layer of disc and last layer of gen
    self.disc = nn.Sequential(
        # Input: N x channels_img x 128 x 128
        nn.Conv2d(
            channels_img, features_d,kernel_size=4,stride=2,padding=1
        ), # 64x64
        nn.LeakyReLU(0.2),
        self._block(features_d, features_d*2,4,2,1), # 32x32
        self._block(features_d*2, features_d*4,4,2,1), # 16x16
        self._block(features_d*4, features_d*8,4,2,1), # 8x8
        self._block(features_d*8, features_d*16,4,2,1), # 4x4
        nn.Conv2d(features_d*16,1,kernel_size=4,stride=2,padding=0), # 1x1
        nn.Sigmoid(),
    )
  # define a block
  def _block(self, in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2)
    )

  def forward(self,x):
    return self.disc(x)
  
class Generator(nn.Module):
    def __init__(self, z_dim, channels_img,features_g):
        super(Generator,self).__init__()
        self.gen = nn.Sequential(
            # Input : N x z_dim x 1 x 1
            self._block(z_dim,features_g*32,4,1,0), # N x f_g*16 x 4 x 4
            self._block(features_g*32,features_g*16,4,2,1), # 8x8
            self._block(features_g*16,features_g*8,4,2,1), # 16x16
            self._block(features_g*8,features_g*4,4,2,1), # 32x32
            self._block(features_g*4,features_g*2,4,2,1), # 64x64
            nn.ConvTranspose2d(
                features_g*2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(), # [-1,1] output images are normalized to be within this range
        )
        
    def _block(self,in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        
    def forward(self, x):
        return self.gen(x)
