import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


import numpy as np


import configparser

import util
import models
import train
import losses

import warnings

def main():
    #warnings.filterwarnings("ignore", category=UserWarning)
    
    if regularizer != "No Reg" or regularizer != "Classic":
        ks_weight, ks_pair_weight, cv_weight =\
            util.estimate_loss_coefficients(batch_size, gmm_centers, gmm_std, coup, num_samples=batch_size)
    
    
    if regularizer == "No Reg":
        weights = np.array([1], dtype = np.float32)
        weights *= np.array([pred_scale], dtype = np.float32)
        loss_fn = losses.no_reg_loss
        
    if regularizer == "Latent":
        weights = np.array([1, ks_weight, ks_pair_weight, cv_weight], dtype = np.float32)
        weights *= np.array([pred_scale, ks_scale, ks_pair_scale, cv_scale], dtype = np.float32)
        loss_fn = losses.latent_loss
        
    if regularizer == "Classic":
        weights = np.array([1,1], dtype = np.float32)
        weights *= np.array([pred_scale, adv_pred_scale], dtype = np.float32)
        loss_fn = losses.classic_loss
        
    if regularizer == "Latent Classic":
        weights = np.array([1, 1, ks_weight, ks_pair_weight, cv_weight], dtype = np.float32)
        weights *= np.array([pred_scale, adv_pred_scale, ks_scale, ks_pair_scale, cv_scale], dtype = np.float32)
        loss_fn = losses.latent_classic_loss
        
    if regularizer == "Supervised":
        weights = np.array([1, 1, ks_weight, ks_pair_weight, cv_weight, 1], dtype = np.float32)
        weights *= np.array([pred_scale, adv_pred_scale, ks_scale, ks_pair_scale, cv_scale, 1], dtype = np.float32)
        loss_fn = losses.supervised_loss
        

    if regularizer == "Latent Supervised":
        weights = np.array([1,1,1,ks_pair_weight, cv_weight], dtype = np.float32)
        weights *= np.array([pred_scale,adv_pred_scale,ks_scale, ks_pair_scale, cv_scale], dtype = np.float32)
        loss_fn = losses.latent_supervised_loss
        
    if regularizer == "Inverse Latent Supervised":
        weights = np.array([1,1,1,ks_pair_weight, cv_weight], dtype = np.float32)
        weights *= np.array([pred_scale,adv_pred_scale,ks_scale, ks_pair_scale, cv_scale], dtype = np.float32)
        loss_fn = losses.inverse_latent_supervised_loss
        
          
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Fashion MNIST Dataset
    ds = datasets.FashionMNIST(
       root = data_dir,
       train = True,                         
       transform = transform, 
       download = True,            
    )
    
    # Split into Train and Validation
    assert num_train_imgs + num_val_imgs == 60000
    train_ds, val_ds = torch.utils.data.random_split(ds, [num_train_imgs, num_val_imgs])
    
    
    # Dataloader
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size, shuffle=True)
    
    
    # Get Model
    model = models.MNIST_Model(latent_dim)
    model = util.to_device(model, device)
    
    # Set Optimizer
    optimizer = optim(model.parameters(), lr=lr)
    
    
    
 
    
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
    adv_method = config.get(experiment, "adv_method")
    alpha = config.getfloat(experiment, "alpha")
    epsilon = config.getfloat(experiment, "epsilon")    
    
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


    
    
    
    # Data location
    date = "feb_2_"
    data_dir = "C:/MT/Datasets/"
    file_path =  'models/' + regularizer + "/" + date + experiment + "/Class PGD-5"
    
    print()
    print(file_path)
    print()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    else:
        print("CUDA NOT AVAILABLE")
        
    util.set_rng(-1) 
    
    # Get GPU
    device = util.get_default_device()   
    
    
    
    try:
    
        main()
        
    except KeyboardInterrupt:
        print('\n\nSTOP\n\n')
        torch.cuda.empty_cache()

    torch.cuda.empty_cache()

    
    
    