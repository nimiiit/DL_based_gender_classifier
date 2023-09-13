import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, classes=2):
        super(Network, self).__init__()
        self.classes = classes
        # loads pretrained vgg model
        vgg = models.vgg16(pretrained=True)
        self.feature_extractor = vgg.features
        self.numfeat = 512
        self.pool = nn.AdaptiveAvgPool2d(1)

        # changes the classifier based on number of classes
        self.classifier = nn.Sequential(nn.Linear(self.numfeat, 128), nn.Dropout(0.2))
        self.classifier_head = nn.Linear(128, self.classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = x.view(-1, self.numfeat)
        x = F.relu(self.classifier(x))
        x = self.classifier_head(x)
        return x
