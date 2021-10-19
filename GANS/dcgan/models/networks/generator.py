import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
      super(Generator, self).__init__()
      self.ngpu = ngpu
      self.main = nn.Sequential(
          #input is Z, going into a convolution
          nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
          nn.BatchNorm2d(ngf*8),
          nn.ReLU(True),
          #state size. (ngf*8) * 4 * 4
          nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
          nn.BatchNorm2d(ngf*4),
          nn.ReLU(True),
          #state size. (ngf*4) * 8 * 8
          nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
          nn.BatchNorm2d(ngf*2),
          nn.ReLU(True),
          #state size. (ngf*2) * 16 * 16
          nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
          nn.BatchNorm2d(ngf),
          nn.ReLU(True),
          #state size. (ngf) * 32 * 32
          nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
          nn.Tanh()
          #state size. (nc) * 64 * 64
    )

    def forward(self, input):
        return self.main(input)

