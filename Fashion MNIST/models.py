import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        
        self.relu = nn.ReLU()
        self.maxPool = nn.MaxPool2d((2,2), stride=(2,2))
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.fc1 = nn.Linear(64*14*14, 100)
        self.fc2 = nn.Linear(100, latent_dim)

        
        

    def forward(self, x):
        
        out = self.conv1(x)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxPool(out)
                
        out = out.view(-1, 64*14*14)
        
        out = self.fc1(out)
        out = self.relu(out)
        
        z = self.fc2(out)
        # Maybe ReLu here?
        #z = self.relu(z)

        return z
    

class Decoder(nn.Module):
    def __init__(self, latent_dim, id_init = False):
        super(Decoder, self).__init__()
        
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(latent_dim, 10)
        
        if id_init:
            nn.init.eye_(self.fc1.weight)
            

    def forward(self, x):
        out = self.relu(x)
        out = self.fc1(out)
        return out



class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize,self).__init__()
        
        self.mean = mean
        self.std = std
        
    def forward(self, x):
        normalized_x = (x - self.mean) / self.std
        return normalized_x 
    
    

class MNIST_Model(nn.Module):
    def __init__(self, latent_dim):
        super(MNIST_Model, self).__init__()


        #self.normalize = Normalize(0.5, 0.5) 
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = Encoder(latent_dim)
        # Decoder
        self.decoder = Decoder(latent_dim, id_init = True)
        

    def encode(self, images):
        return self.encoder(images)

    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
    
        #x = self.normalize(x)    
        z = self.encode(x)
        pred = self.decode(z)
        
        return pred, z
    
    
    