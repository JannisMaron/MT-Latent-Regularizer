import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


import numpy as np


import configparser

import util
import models
import train
import losses

#import warnings


def main():
    
    # Get weight Coefficent
    if regularizer != "No Reg" and regularizer != "Classic":
        ks_weight, ks_pair_weight, cv_weight =\
            util.estimate_loss_coefficients(batch_size, gmm_centers, gmm_std, coup, num_samples=batch_size)
  
        
        
    # Select Regularizer and scale weights
    
    if regularizer == "No Reg":
        weights = np.array([1], dtype = np.float32)
        weights *= np.array([pred_scale], dtype = np.float32)
        loss_fn = losses.no_reg_loss
        
    if regularizer == "Classic":
        weights = torch.Tensor([1.,1.])
        weights *= torch.Tensor([pred_scale, adv_pred_scale])
        loss_fn = losses.classic_loss
    
    if regularizer == "Inverse Latent Supervised":
          weights = np.array([1,1,1,ks_pair_weight, cv_weight], dtype = np.float32)
          weights *= np.array([pred_scale,adv_pred_scale,ks_scale, ks_pair_scale, cv_scale], dtype = np.float32)
          loss_fn = losses.inverse_latent_supervised_loss
          
          
          
    # Dataset Transforms
    transform = transforms.Compose([transforms.ToTensor()])
     
    # Dataset
    ds = datasets.SVHN(
       root = data_dir,
       split = "train",                         
       transform = transform, 
       download = True,            
    ) 
    
    # Limited data for fast testing
    #num_train_imgs = 5000
    #num_val_imgs = 1000
    ignore = len(ds) - (num_train_imgs + num_val_imgs)
    
    
    train_ds, val_ds, _ = torch.utils.data.random_split(ds, [num_train_imgs, num_val_imgs, ignore])
        
    
            
    # Dataloaders
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size, shuffle=True)
        
    
    # Model
    model = models.SVHN_PreAct(latent_dim, identity_init=True)
    model = util.to_device(model, device)
    
    
    # Optimizer
    optimizer = optim(model.parameters(), lr=lr)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    
    
    # Train the network
    train.fit(n_epochs, model, optimizer, train_dl, val_dl, loss_fn,
            epsilon, alpha, gmm_centers, gmm_std, weights, coup,
            file_path)
    
    
    pass

if __name__ == "__main__":
     
    
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    experiment = config.get("Experiments", "experiment")
    
    # train settings
    n_epochs = config.getint(experiment, "n_epochs")
    batch_size = config.getint(experiment, "batch_size")
    num_train_imgs = config.getint(experiment, "num_train_imgs")  
    num_val_imgs = config.getint(experiment, "num_val_imgs")  
    
    optim = torch.optim.Adam
    
    # adversarial example settings
    epsilon = config.getfloat(experiment, "epsilon") / 255  
    alpha = config.getfloat(experiment, "alpha") / 255
    
    
    # GMM settings
    latent_dim = config.getint(experiment, "latent_dim")
    num_clusters = config.getint(experiment, "num_clusters")
    coup = config.getfloat(experiment, "coup")
    
    gmm_centers, gmm_std = util.set_gmm_centers(latent_dim, num_clusters)
    
    
    # regularizer settings
    regularizer = config.get("Regularizer", "regularizer")
    pred_scale = config.getfloat(regularizer, "pred_scale")
    adv_pred_scale = config.getfloat(regularizer, "adv_pred_scale")
    ks_scale = config.getfloat(regularizer, "ks_scale")
    ks_pair_scale = config.getfloat(regularizer, "ks_pair_scale")
    cv_scale = config.getfloat(regularizer, "cv_scale")
    
    lr = config.getfloat(regularizer, "lr") 


    # Dataset Location  NEED TO CHANGE
    data_dir = "C:/MT/Datasets/"

    # Save location
    date = "mar_2_"
    file_path =  'models/' + regularizer + "/" + date + experiment + "/Temp"
    
    print()
    print(file_path)
    print()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    else:
        print("CUDA NOT AVAILABLE")
        
    #util.set_rng(-1) 
    
    # Get GPU
    device = util.get_default_device()   
    
    
    
    try:
    
        main()
        
    except KeyboardInterrupt:
        print('\n\nSTOP\n\n')
        torch.cuda.empty_cache()

    torch.cuda.empty_cache()
