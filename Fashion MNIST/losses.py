import regularizer

import torch
from torch.nn import functional as F
import numpy as np




def no_reg_loss(pred, adv_pred, label, z, adv_z, weights,
               gmm_centers, gmm_std, coup):
    
    
    pred_weight = weights[0]
    
    image_loss = F.cross_entropy(pred, label)
    
    weighted_pred_loss = pred_weight * image_loss
   
    # total loss
    loss_mean = weighted_pred_loss.mean().cuda()
    
    # partial_loss
    weighted_partial_losses = np.array([weighted_pred_loss.item()])
    
    return loss_mean, weighted_partial_losses


def latent_loss(pred, adv_pred, label, z, adv_z, weights,
               gmm_centers, gmm_std, coup):
    
    
    pred_weight = weights[0]
    ks_weight = weights[1]
    ks_pair_weight = weights[2] 
    cv_weight = weights[3]
    
    # partial losses
    image_loss = F.cross_entropy(pred, label)
    
    ks = regularizer.ks_loss(z, adv_z, gmm_centers, gmm_std)
    ks_pair = regularizer.ks_pair_loss(z, adv_z, gmm_centers, gmm_std)
    cov = regularizer.covariance_loss(z, adv_z, gmm_centers, gmm_std, coup)
    
    
    
    # weigted partial losses
    weighted_pred_loss = pred_weight * image_loss

    weighted_ks_loss = ks_weight * ks
    weighted_ks_pair_loss = ks_pair_weight * ks_pair
    weighted_cv_loss = cv_weight * cov
    
    
    # total loss
    losses = weighted_pred_loss +  weighted_ks_loss + weighted_ks_pair_loss + weighted_cv_loss 
    loss_mean = losses.mean().cuda()
    
    
    weighted_partial_losses = np.array([weighted_pred_loss.item(), weighted_ks_loss.item(), 
                                        weighted_ks_pair_loss.item(), weighted_cv_loss.item()])
    
    return loss_mean, weighted_partial_losses


def classic_loss(pred, adv_pred, label, z, adv_z, weights,
               gmm_centers, gmm_std, coup):
                 
     pred_weight = weights[0]
     adv_pred_weight = weights[1]
     
     # partial losses
     image_loss = F.cross_entropy(pred, label)
     adv_image_loss = F.cross_entropy(adv_pred, label)
     
     # weigted partial losses
     weighted_pred_loss = pred_weight * image_loss
     weighted_adv_pred_loss = adv_pred_weight * adv_image_loss
         
     # total loss
     losses = weighted_pred_loss + weighted_adv_pred_loss        
     loss_mean = losses.mean().cuda()
     weighted_partial_losses = np.array([weighted_pred_loss.item(), weighted_adv_pred_loss.item()])
                                       
     return loss_mean, weighted_partial_losses
 
    

def latent_classic_loss(pred, adv_pred, label, z, adv_z, weights,
               gmm_centers, gmm_std, coup):
    
    
    pred_weight = weights[0]
    adv_pred_weight = weights[1]
    ks_weight = weights[2]
    ks_pair_weight = weights[3] 
    cv_weight = weights[4]
    
    
    # partial losses
    image_loss = F.cross_entropy(pred, label)
    adv_image_loss = F.cross_entropy(adv_pred, label)
    
    ks = regularizer.ks_loss(z, adv_z, gmm_centers, gmm_std)
    ks_pair = regularizer.ks_pair_loss(z, adv_z, gmm_centers, gmm_std)
    cov = regularizer.covariance_loss(z, adv_z, gmm_centers, gmm_std, coup)
    
    
    
    # weigted partial losses
    weighted_pred_loss = pred_weight * image_loss
    weighted_adv_pred_loss = adv_pred_weight * adv_image_loss

    weighted_ks_loss = ks_weight * ks
    weighted_ks_pair_loss = ks_pair_weight * ks_pair
    weighted_cv_loss = cv_weight * cov
    
    
    # total loss
    losses = weighted_pred_loss + weighted_adv_pred_loss +  weighted_ks_loss + weighted_ks_pair_loss + weighted_cv_loss 
    loss_mean = losses.mean().cuda()
    
    
    weighted_partial_losses = np.array([weighted_pred_loss.item(), weighted_adv_pred_loss.item(),
                            weighted_ks_loss.item(), weighted_ks_pair_loss.item(), weighted_cv_loss.item()])
    
    return loss_mean, weighted_partial_losses


    

def latent_supervised_loss(pred, adv_pred, label, z, z_adv, weights, gmm_centers, gmm_std, coup):
    
    clean_pred_weight = weights[0]
    adv_pred_weight = weights[1]
    supervised_ks_weight = weights[2]
    ks_pair_weight = weights [3]
    cv_weight = weights[4]
    
    
    clean_pred = F.cross_entropy(pred, label)
    adv_pred_loss = F.cross_entropy(adv_pred, label)
    supervised_ks = regularizer.supervised_ks_loss(z, z_adv, label, gmm_centers, gmm_std)
    ks_pair = regularizer.ks_pair_loss(z, z_adv, gmm_centers, gmm_std)
    cv = regularizer.covariance_loss(z, z_adv, gmm_centers, gmm_std, coup)
    
    
    clean_pred_loss = clean_pred_weight * clean_pred
    adv_pred_loss = adv_pred_weight * adv_pred_loss 
    supervised_ks_loss = supervised_ks_weight * supervised_ks
    ks_pair_loss = ks_pair_weight * ks_pair
    cv_loss = cv_weight * cv
    
    total_loss = 0
    total_loss += clean_pred_loss 
    total_loss += adv_pred_loss
    total_loss += supervised_ks_loss 
    total_loss += ks_pair_loss
    total_loss += cv_loss
    total_loss = total_loss.mean().cuda()
    
    partial_losses = []
    partial_losses.append(clean_pred_loss.item())
    partial_losses.append(adv_pred_loss.item())
    partial_losses.append(supervised_ks_loss.item())
    partial_losses.append(ks_pair_loss.item())
    partial_losses.append(cv_loss.item())
    partial_losses = np.array(partial_losses)
        
    
    return total_loss, partial_losses   

def inverse_latent_supervised_loss(pred, adv_pred, label, z, z_adv, weights, gmm_centers, gmm_std, coup):
    
    clean_pred_weight = weights[0]
    adv_pred_weight = weights[1]
    supervised_ks_weight = weights[2]
    ks_pair_weight = weights [3]
    cv_weight = weights[4]
    
    
    clean_pred = F.cross_entropy(pred, label)
    adv_pred_loss = F.cross_entropy(adv_pred, label)
    supervised_ks = regularizer.inverse_supervised_ks_loss(z, z_adv, label, gmm_centers, gmm_std)
    ks_pair = regularizer.ks_pair_loss(z, z_adv, gmm_centers, gmm_std)
    cv = regularizer.covariance_loss(z, z_adv, gmm_centers, gmm_std, coup)
    
    
    clean_pred_loss = clean_pred_weight * clean_pred
    adv_pred_loss = adv_pred_weight * adv_pred_loss 
    supervised_ks_loss = supervised_ks_weight * supervised_ks
    ks_pair_loss = ks_pair_weight * ks_pair
    cv_loss = cv_weight * cv
    
    total_loss = 0
    total_loss += clean_pred_loss 
    total_loss += adv_pred_loss
    total_loss += supervised_ks_loss 
    total_loss += ks_pair_loss
    total_loss += cv_loss
    total_loss = total_loss.mean().cuda()
    
    partial_losses = []
    partial_losses.append(clean_pred_loss.item())
    partial_losses.append(adv_pred_loss.item())
    partial_losses.append(supervised_ks_loss.item())
    partial_losses.append(ks_pair_loss.item())
    partial_losses.append(cv_loss.item())
    partial_losses = np.array(partial_losses)
        
    
    return total_loss, partial_losses   