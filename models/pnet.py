import torch
from torch import nn


class PNet(nn.Module):

    def __init__(self):
        super(PNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 10, (3, 3), stride=1, padding=0),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(),

            nn.MaxPool2d((2, 2), stride=2, padding=0),

            nn.Conv2d(10, 16, (3, 3), stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),

            nn.Conv2d(16, 32, (3, 3), stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )

        self.face_clf = nn.Sequential(
            nn.Conv2d(32, 2, (1, 1), stride=1, padding=0),
            nn.LogSoftmax(dim=1)
        )

        self.bbox_reg = nn.Sequential(
            nn.Conv2d(32, 4, (1, 1), stride=1, padding=0),
            nn.Sigmoid(),
        )

        self.lmark_reg = nn.Sequential(
            nn.Conv2d(32, 10, (1, 1), stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        n = x.size(0)

        x = self.features(x)

        face_logps = self.face_clf(x).view(n, -1)
        bbox_preds = self.bbox_reg(x).view(n, -1)
        lmark_preds = self.lmark_reg(x).view(n, -1)

        return face_logps, bbox_preds, lmark_preds
