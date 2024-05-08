import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, image_channels=1):
        super(Discriminator, self).__init__()
        self.channels = [32, 64, 128, 128]
        self.convs = nn.ModuleList([
            nn.Conv2d(image_channels, self.channels[0], 3, 1, 1),
            nn.Conv2d(self.channels[0], self.channels[1], 3, 1, 1),
            nn.Conv2d(self.channels[1], self.channels[2], 3, 1, 1),
            nn.Conv2d(self.channels[2], self.channels[3], 3, 1, 1)
        ])
        self.maxpools = nn.ModuleList([
            nn.MaxPool2d(2) for _ in self.channels
        ])
        self.leaky_relus = nn.ModuleList([
            nn.LeakyReLU(0.2) for _ in self.channels
        ])
        self.final_fc = nn.Linear(128 * 8 * 8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for conv, maxpool, leaky_relu in zip(self.convs, self.maxpools, self.leaky_relus):
            x = conv(x)
            x = maxpool(x)
            x = leaky_relu(x)

        x = x.view(x.size(0), -1)
        x = self.final_fc(x)
        x = self.sigmoid(x)
        return x

# encoder = Discriminator()
# dummy_input = torch.randn(1, 1, 128, 128)
# output = encoder(dummy_input)
# print(output)