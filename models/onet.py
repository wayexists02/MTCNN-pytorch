import torch
from torch import nn


class ONet(nn.Module):

    def __init__(self):
        super(ONet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, (3, 3), stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.MaxPool2d((3, 3), stride=2, padding=0),

            nn.Conv2d(32, 64, (3, 3), stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.MaxPool2d((3, 3), stride=2, padding=0),

            nn.Conv2d(64, 64, (3, 3), stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.MaxPool2d((2, 2), stride=2, padding=0),

            nn.Conv2d(64, 128, (2, 2), stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )

        self.face_clf = nn.Sequential(
            nn.Linear(2*2*128, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 2),
            nn.LogSoftmax(dim=1)
        )

        self.bbox_reg = nn.Sequential(
            nn.Linear(2*2*128, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 4),
            nn.Sigmoid(),
        )

        self.lmark_reg = nn.Sequential(
            nn.Linear(2*2*128, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(256, 10),
            nn.Sigmoid()
        )

    def forward(self, x):
        n = x.size(0)

        x = self.features(x)
        x = x.view(n, -1)

        face_logps = self.face_clf(x)
        bbox_preds = self.bbox_reg(x)
        lmark_preds = self.lmark_reg(x)

        return face_logps, bbox_preds, lmark_preds
