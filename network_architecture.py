import torch
import torch.nn as nn
import torch.nn.functional as F

def component_classifier_function():
    import torchvision
    
    model = torchvision.models.mobilenet_v2()
    return model


class component_classifier(nn.Module):
    
    #obsolete class. use component_classifier_function() for preexisting architectures instead.
    
    def __init__(self):
        super(component_classifier, self).__init__()
        
        channel = 16
        
        self.conv1 = nn.Conv2d(3, channel, kernel_size=5, stride=1, padding=2)
        self.bn2d1 = nn.BatchNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=5, stride=1, padding=2)
        self.bn2d2 = nn.BatchNorm2d(channel)
        self.conv3 = nn.Conv2d(channel, channel, kernel_size=5, stride=1, padding=2)
        self.bn2d3 = nn.BatchNorm2d(channel)
        self.conv4 = nn.Conv2d(channel, channel, kernel_size=5, stride=1, padding=2)
        self.bn2d4 = nn.BatchNorm2d(channel)
        self.conv5 = nn.Conv2d(channel, 16, kernel_size=3, stride=1, padding=1)
        self.bn2d5 = nn.BatchNorm2d(16)
        self.linear1 = nn.Linear(16*64*64, 8192)
        self.drop1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(8192, 1024)
        self.drop2 = nn.Dropout(0.4)
        self.linear3 = nn.Linear(1024, 128)
        self.drop3 = nn.Dropout(0.3)
        self.linear4 = nn.Linear(128, 32)
        self.drop4 = nn.Dropout(0.2)
        self.linear5 = nn.Linear(32, 4)
    
    def forward(self, x):
        x = F.relu(self.bn2d1(self.conv1(x)))
        x = F.relu(self.bn2d2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn2d3(self.conv3(x)))
        x = F.relu(self.bn2d4(self.conv4(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn2d5(self.conv5(x)))
        x = x.reshape([-1, 16*64*64])
        x = F.relu(self.drop1(self.linear1(x)))
        x = F.relu(self.drop2(self.linear2(x)))
        x = F.relu(self.drop3(self.linear3(x)))
        x = F.relu(self.drop4(self.linear4(x)))
        x = self.linear5(x)
        return x