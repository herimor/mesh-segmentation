# Project: https://github.com/adambielski/siamese-triplet
# Author: Adam Bielski https://github.com/adambielski
# License: BSD 3-Clause


import torch.nn as nn


class EmbeddingNet(nn.Module):
    def __init__(self, input_shape, output_shape=256, layer_width=512, do_rate=0.6):
        super(EmbeddingNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_shape, layer_width),
            nn.PReLU(),
            nn.Dropout(p=do_rate),
            nn.Linear(layer_width, layer_width),
            nn.PReLU(),
            nn.Dropout(p=do_rate),
            nn.Linear(layer_width, output_shape, bias=False)
        )

    def forward(self, x):
        output = self.fc(x.float())
        return output

    def get_embedding(self, x):
        return self.forward(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return [output1, output2, output3]

    def get_embedding(self, x):
        return self.embedding_net(x)
