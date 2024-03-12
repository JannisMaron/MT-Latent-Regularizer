import datetime
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

import util
import load_save


device = util.get_default_device()

train_method = "standart"




def fit(n_epochs, model, optimizer, train_dl, val_dl, loss_fn,
        epsilon, alpha, gmm_centers, gmm_std, weights, coup,
        file_path):
    
    #lambda_lr = lambda epoch: 0.9 ** (epoch // 5)
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
    
    print("Start Training:", datetime.datetime.now().strftime("%X"))
    
    
    tl = []
    tl_partial = []   
    
    train_loss = 0.0
    train_partial_loss = 0.0
    
    vl = {
        'Latent FGSM': [],
        'Latent FGSM Large Eps': [],
        'Classification FGSM': [],
        'Latent PGD': [],
        'Classification PGD': []
        }
    vl_partial = {
        'Latent FGSM': [],
        'Latent FGSM Large Eps': [],
        'Classification FGSM': [],
        'Latent PGD': [],
        'Classification PGD': []
        }
    accuracy = {
        'Clean Acc': [],
        'Latent FGSM': [],
        'Latent FGSM Large Eps': [],
        'Classification FGSM': [],
        'Latent PGD': [],
        'Classification PGD': []
        }
    
    
    for epoch in range(n_epochs):
     
        # Training
        
        if train_method == "standart":
        
            train_loss, train_partial_loss =\
                    train(model, optimizer, train_dl, loss_fn, epsilon, alpha, 
                          gmm_centers, gmm_std, weights, coup)
                    
                    
                    
        tl.append(train_loss)
        tl_partial.append(train_partial_loss)             
                    
                    
        #Validation
        
        vl_total_losses, vl_partial_losses, vl_accuracies =\
            val(model, val_dl, loss_fn, epsilon, alpha, gmm_centers, gmm_std, weights, coup)
        
        
        vl = util.append_to_dict(vl, vl_total_losses)
        vl_partial = util.append_to_dict(vl_partial, vl_partial_losses)
        accuracy = util.append_to_dict(accuracy, vl_accuracies)      
        
        
        # Stats every 10th epoch
        if epoch % 1 == 0:
            print('\n------------------------------')
            print('Time:', datetime.datetime.now().strftime("%X"))
            print("Currend Epoch: ", epoch)
            
            print()
            print(f"Train Loss: {train_loss:.4} / {train_partial_loss}")
            print(f"Latent FGSM Val Loss: {vl_total_losses['Latent FGSM']:.4} / {vl_partial_losses['Latent FGSM']}")
            print(f"Classification PGD Val Loss: {vl_total_losses['Classification PGD']:.4} / {vl_partial_losses['Classification PGD']}")
            
            
            print()
            print(f"Clean Acc: {vl_accuracies['Clean Acc']:.4}")
            print(f"Latent FGSM Acc: {vl_accuracies['Latent FGSM']:.4}")
            print(f"Class. PGD Acc: {vl_accuracies['Classification PGD']:.4}")
            
            print()
            print()
            
            util.plot_acc_progress(epoch, accuracy)
           
           
        # Save Last Epoch    
        load_save.save(model, optimizer, epoch, tl, tl_partial, 
                       vl, vl_partial, accuracy,
                       file_path, "/last.pth")
        
        #scheduler.step() 
        #break          
                    
                    
                    
def train(model, optimizer, train_dl, loss_fn,
          epsilon, alpha, gmm_centers, gmm_std, weights, coup):
    
    
    batch_loss = 0
    batch_partial_losses = 0
    
    model.train()
    
    
    for i,batch in enumerate(train_dl):
        
        batch = util.to_device(batch, device)
        x, label = batch
        

        # Craft Adversarial Samples
        optimizer.zero_grad()
        #adv_x = util.craft_FGSM_adv_samples(model, x, label, 8/255, 10/255) 
        #adv_x = util.craft_output_FGSM_adv_samples(model, x, label, 8/255, 10/255) 
        #adv_x = util.craft_PGD_adv_samples(model, x, label, 8/255, 2/255, 5)
        adv_x = util.craft_output_PGD_adv_samples(model, x, label, 8/255, 2/255, 5)

        optimizer.zero_grad()
        
        # Clean Images
        pred, z = model(x) 
                
        # Adversarial images
        adv_pred, adv_z = model(adv_x)
        
        
        # Loss
        loss, partial_losses =\
            loss_fn(pred, adv_pred, label, z, adv_z, weights, gmm_centers, gmm_std, coup)
            
            
        # Do optimization
        optimizer.zero_grad()
        loss.backward()   
        
        #print()
        #print("----------------------")
        #print("Batch: ", i)
        #print("----------------------")
        #for name, param in model.named_parameters():
        #    if param.grad is not None:
        #        grad_norm = param.grad.norm().item()
        #        print(f'{name} gradient norm: {grad_norm}')

        
        optimizer.step()   
        
        
        
        # Log Losses
        batch_loss += loss.item()
        batch_partial_losses += partial_losses
        
        #if i == 5:
        #    break
        
        
    batch_loss /= len(train_dl)
    batch_partial_losses /= len(train_dl) 
    
    return batch_loss, batch_partial_losses
        
        
        
        
def val(model, val_dl, loss_fn, epsilon, alpha, gmm_centers, gmm_std, weights, coup):
    
    
    # loss
    batch_fgsm_loss = 0
    batch_large_eps_fgsm_loss = 0
    batch_output_fgsm_loss = 0
    batch_pgd_loss = 0
    batch_output_pgd_loss = 0
    
    
    # partial Losses
    batch_fgsm_partial_loss = 0
    batch_large_eps_fgsm_partial_loss = 0
    batch_output_fgsm_partial_loss = 0
    batch_pgd_partial_loss = 0
    batch_output_pgd_partial_loss = 0

    
    # accuracy
    batch_clean_acc = 0
    batch_fgsm_acc = 0
    batch_large_eps_fgsm_acc = 0
    batch_output_fgsm_acc = 0
    batch_pgd_acc = 0
    batch_output_pgd_acc = 0
    
    
    model.eval()
    
    
    
    
    
    for i,batch in enumerate(val_dl):
        
        batch = util.to_device(batch, device)
        x, label = batch
        
        
        # Clean Outputs
        pred, z = model(x)
        
        batch_clean_acc += util.accuracy(pred, label)
        
        
        # Latent FGSM Adversarials
        adv_x = util.craft_FGSM_adv_samples(model, x, label, 8/255, 10/255)
        adv_pred, adv_z = model(adv_x)

        loss, partial_loss =\
            loss_fn(pred, adv_pred, label, z, adv_z, weights, gmm_centers, gmm_std, coup)
        
        batch_fgsm_loss += loss.item()
        batch_fgsm_partial_loss += partial_loss
        
        batch_fgsm_acc += util.accuracy(adv_pred, label)
        
        
        
        # Latent FGSM with larger epsilon
        adv_x = util.craft_FGSM_adv_samples(model, x, label, 12/255, 15/255)
        adv_pred, adv_z = model(adv_x)

        loss, partial_loss =\
            loss_fn(pred, adv_pred, label, z, adv_z, weights, gmm_centers, gmm_std, coup)
        
        batch_large_eps_fgsm_loss += loss.item()
        batch_large_eps_fgsm_partial_loss += partial_loss
        
        batch_large_eps_fgsm_acc += util.accuracy(adv_pred, label)
        
        
        
        # Output FGSM
        adv_x = util.craft_output_FGSM_adv_samples(model, x, label, 8/255, 10/255)
        adv_pred, adv_z = model(adv_x)

        loss, partial_loss =\
            loss_fn(pred, adv_pred, label, z, adv_z, weights, gmm_centers, gmm_std, coup)
        
        batch_output_fgsm_loss += loss.item()
        batch_output_fgsm_partial_loss += partial_loss
        
        batch_output_fgsm_acc += util.accuracy(adv_pred, label)
        
        
        
        
        # Latent PGD
        adv_x = util.craft_PGD_adv_samples(model, x, label, 8/255, 2/255, 10)
        adv_pred, adv_z = model(adv_x)

        loss, partial_loss =\
            loss_fn(pred, adv_pred, label, z, adv_z, weights, gmm_centers, gmm_std, coup)
        
        batch_pgd_loss += loss.item()
        batch_pgd_partial_loss += partial_loss
        
        batch_pgd_acc += util.accuracy(adv_pred, label)
        
        
        # Output PGD
        adv_x = util.craft_output_PGD_adv_samples(model, x, label, 8/255, 2/255, 10)
        adv_pred, adv_z = model(adv_x)

        loss, partial_loss =\
            loss_fn(pred, adv_pred, label, z, adv_z, weights, gmm_centers, gmm_std, coup)
        
        batch_output_pgd_loss += loss.item()
        batch_output_pgd_partial_loss += partial_loss
        
        batch_output_pgd_acc += util.accuracy(adv_pred, label)
        
        
        
        
        
        
    batch_fgsm_loss /= len(val_dl)
    batch_large_eps_fgsm_loss /= len(val_dl)
    batch_output_fgsm_loss /= len(val_dl)
    batch_pgd_loss /= len(val_dl)
    batch_output_pgd_loss /= len(val_dl)

    
    batch_fgsm_partial_loss /= len(val_dl)
    batch_large_eps_fgsm_partial_loss /= len(val_dl)
    batch_output_fgsm_partial_loss /= len(val_dl)
    batch_pgd_partial_loss /= len(val_dl)
    batch_output_pgd_partial_loss /= len(val_dl) 
    
    batch_clean_acc /= len(val_dl)
    batch_fgsm_acc /= len(val_dl)
    batch_large_eps_fgsm_acc /= len(val_dl)
    batch_output_fgsm_acc /= len(val_dl)
    batch_pgd_acc /= len(val_dl)
    batch_output_pgd_acc /= len(val_dl)
    
    
    total_losses = {
        'Latent FGSM': batch_fgsm_loss,
        'Latent FGSM Large Eps': batch_large_eps_fgsm_loss,
        'Classification FGSM': batch_output_fgsm_loss,
        'Latent PGD': batch_pgd_loss,
        'Classification PGD': batch_output_pgd_loss
        }
    
    partial_losses = {
        'Latent FGSM': batch_fgsm_partial_loss,
        'Latent FGSM Large Eps': batch_large_eps_fgsm_partial_loss,
        'Classification FGSM': batch_output_fgsm_partial_loss,
        'Latent PGD': batch_pgd_partial_loss,
        'Classification PGD': batch_output_pgd_partial_loss
        }
        
    accuracies = {
        'Clean Acc': batch_clean_acc, 
        'Latent FGSM': batch_fgsm_acc,
        'Latent FGSM Large Eps': batch_large_eps_fgsm_acc,
        'Classification FGSM': batch_output_fgsm_acc,
        'Latent PGD': batch_pgd_acc,
        'Classification PGD': batch_output_pgd_acc
        }
    
    
    return total_losses, partial_losses, accuracies
        
        
        