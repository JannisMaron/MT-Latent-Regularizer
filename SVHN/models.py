import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize,self).__init__()
        
        self.mean = nn.Parameter(mean.unsqueeze(1).unsqueeze(1), requires_grad=False)
        self.std = nn.Parameter(std.unsqueeze(1).unsqueeze(1), requires_grad=False)
        
    def forward(self, x):
        
        normalized_x = (x - self.mean) / self.std
        return normalized_x 






class Encoder_Simple(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder_Simple, self).__init__()
        
        self.relu = nn.ReLU()
        self.maxPool = nn.MaxPool2d((2,2), stride=(2,2))
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.fc1 = nn.Linear(64*16*16, 100)
        self.fc2 = nn.Linear(100, latent_dim)

        
        

    def forward(self, x):
        
        out = self.conv1(x)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxPool(out)
                
        out = out.view(-1, 64*16*16)
        
        out = self.fc1(out)
        out = self.relu(out)
        
        z = self.fc2(out)

        return z
    

class Decoder(nn.Module):
    def __init__(self, latent_dim, id_init = True):
        super(Decoder, self).__init__()
        
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(latent_dim, 10)
        
        if id_init:
            nn.init.eye_(self.fc1.weight)
            

    def forward(self, x):
        out = self.relu(x)
        out = self.fc1(out)
        return out

    

class SVHN_Simple(nn.Module):
    def __init__(self, latent_dim, identity_init = True):
        super(SVHN_Simple, self).__init__()


        #self.normalize = Normalize(0.5, 0.5) 
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = Encoder_Simple(latent_dim)
        # Decoder
        self.decoder = Decoder(latent_dim, id_init = identity_init)
        

    def encode(self, images):
        return self.encoder(images)

    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
    
        #x = self.normalize(x)    
        z = self.encode(x)
        pred = self.decode(z)
        
        return pred, z
    
    
###############################################################################
# Preact ResNet
###############################################################################
class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out
    
class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
def PreActResNet18(latent_dim):
    return PreActResNet(PreActBlock, [2,2,2,2], latent_dim)


class SVHN_PreAct(nn.Module):
    def __init__(self, latent_dim, identity_init = True):
        super(SVHN_PreAct, self).__init__()


        #self.normalize = Normalize(torch.Tensor([0.43538398, 0.44171983, 0.47072124]).cuda(), 
        #                           torch.Tensor([0.19657154, 0.19998856, 0.1966886]).cuda()) 
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = PreActResNet18(latent_dim)
        # Decoder
        self.decoder = Decoder(latent_dim, id_init = identity_init)
        

    def encode(self, images):
        return self.encoder(images)

    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
    
        #x = self.normalize(x)    
        z = self.encode(x)
        pred = self.decode(z)
        
        return pred, z

    
    
    
    
    
    
    
    
    
    
    