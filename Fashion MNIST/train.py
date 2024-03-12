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
    
    
    print("Start Training:", datetime.datetime.now().strftime("%X"))
    
    for epoch in range(n_epochs):
        
        
        
        # Training
        
        if train_method == "standart":
        
            train_loss, train_partial_loss =\
                    train(model, optimizer, train_dl, loss_fn, epsilon, alpha, 
                          gmm_centers, gmm_std, weights, coup)
                    
        elif train_method == "MixedTrain":
            train_loss, train_partial_loss =\
                    mixed_train(model, optimizer, train_dl, loss_fn, epsilon, alpha, 
                          gmm_centers, gmm_std, weights, coup)
                    
        elif train_method == "ValTrain":
            train_loss, train_partial_loss =\
                val_train(model, optimizer, train_dl, val_dl, loss_fn,
                           epsilon, alpha, gmm_centers, gmm_std, weights, coup)
                
        elif train_method == "GradAlign":
            train_loss, train_partial_loss =\
                gardAlign_train(model, optimizer, train_dl, loss_fn, 
                                epsilon, alpha, gmm_centers, gmm_std, weights, coup)
            
            
        tl.append(train_loss)
        tl_partial.append(train_partial_loss)  
            
        
        
        # Validation
        
        vl_total_losses, vl_partial_losses, vl_accuracies =\
            val(model, val_dl, loss_fn, epsilon, alpha, gmm_centers, gmm_std, weights, coup)
        
        
        vl = util.append_to_dict(vl, vl_total_losses)
        vl_partial = util.append_to_dict(vl_partial, vl_partial_losses)
        accuracy = util.append_to_dict(accuracy, vl_accuracies)
        
        
        if epoch % 10 == 0:
            print('\n------------------------------')
            print('Time:', datetime.datetime.now().strftime("%X"))
            print("Currend Epoch: ", epoch)
            
            print()
            print(f"Train Loss: {train_loss:.4} / {train_partial_loss}")
            print(f"Latent FGSM Val Loss: {vl_total_losses['Latent FGSM']:.4} / {vl_partial_losses['Latent FGSM']}")
            
            
            print()
            print(f"Clean Acc: {vl_accuracies['Clean Acc']:.4}")
            print(f"FGSM Acc: {vl_accuracies['Latent FGSM']:.4}")
            
            util.plot_acc_progress(epoch, accuracy)
            
            
        load_save.save(model, optimizer, epoch, tl, tl_partial, 
                       vl, vl_partial, accuracy,
                       file_path, "/last.pth")
            
    pass


def train(model, optimizer, train_dl, loss_fn,
          epsilon, alpha, gmm_centers, gmm_std, weights, coup):
    
    batch_loss = 0
    batch_partial_losses = 0
    
    
    model.train()
    
    for i,batch in enumerate(train_dl):
        
        batch = util.to_device(batch, device)
        x, label = batch

        # craft Adversarial Samples
        optimizer.zero_grad()
        adv_x = util.craft_output_PGD_adv_samples(model, x, label, 0.3, 0.01, 5)
            
        # clean Images
        pred, z = model(x)
                
        # adversarial images
        adv_pred, adv_z = model(adv_x)  
            
        # loss
        loss, partial_losses =\
            loss_fn(pred, adv_pred, label, z, adv_z, weights, gmm_centers, gmm_std, coup)
        
        # do optimization
        optimizer.zero_grad()
        loss.backward()    
        optimizer.step()
        
        
        batch_loss += loss.item()
        batch_partial_losses += partial_losses
        
        
    batch_loss /= len(train_dl)
    batch_partial_losses /= len(train_dl) 
    
    return batch_loss, batch_partial_losses


def mixed_train(model, optimizer, train_dl, loss_fn,
          epsilon, alpha, gmm_centers, gmm_std, weights, coup):
    
    batch_loss = 0
    batch_partial_losses = 0
    
    
    model.train()
    
    for i,batch in enumerate(train_dl):
        
        batch = util.to_device(batch, device)
        x, label = batch
        
        if i%10 != 0:
            # craft Adversarial Samples
            optimizer.zero_grad()
            adv_x = util.craft_FGSM_adv_samples(model, x, label, epsilon, alpha)
                
        else:
            # craft Adversarial Samples
            optimizer.zero_grad()
            adv_x = util.craft_PGD_adv_samples(model, x, label, epsilon, 0.01, 10)
            
            
        # clean Images
        pred, z = model(x)
                
        # adversarial images
        adv_pred, adv_z = model(adv_x)  
            
        # loss
        loss, partial_losses =\
            loss_fn(pred, adv_pred, label, z, adv_z, weights, gmm_centers, gmm_std, coup)
        
        # do optimization
        optimizer.zero_grad()
        loss.backward()    
        optimizer.step()
        
        
        batch_loss += loss.item()
        batch_partial_losses += partial_losses
        
        
    batch_loss /= len(train_dl)
    batch_partial_losses /= len(train_dl) 
    
    return batch_loss, batch_partial_losses


def val_train(model, optimizer, train_dl, val_dl, loss_fn,
          epsilon, alpha, gmm_centers, gmm_std, weights, coup):
    
    batch_loss = 0
    batch_partial_losses = 0
    
    val_robust_acc = 0
    prev_val_acc = 0
    
    trigger = 0
    
    for i,batch in enumerate(train_dl):
        
        model.train()
        
        batch = util.to_device(batch, device)
        x, label = batch
        
        
        if (val_robust_acc+0.1 < prev_val_acc and i>1):
            #print("Catastropic Overfitting Detected:", i)
            trigger += 1
            optimizer.zero_grad()
            adv_x = util.craft_PGD_adv_samples(model, x, label, epsilon, 0.01, 10)
            #adv_x = util.craft_FGSM_adv_samples(model, x, label, 0.4, 1.25*0.4)
        else:
           optimizer.zero_grad()
           adv_x = util.craft_FGSM_adv_samples(model, x, label, epsilon, alpha)
           
           
        # clean Images
        pred, z = model(x)
                
        # adversarial images
        adv_pred, adv_z = model(adv_x)  
            
        # loss
        loss, partial_losses =\
            loss_fn(pred, adv_pred, label, z, adv_z, weights, gmm_centers, gmm_std, coup)
        
        # do optimization
        optimizer.zero_grad()
        loss.backward()    
        optimizer.step()
        
        batch_loss += loss.item()
        batch_partial_losses += partial_losses
        
        if i%40 == 0:
            model.eval()
            prev_val_acc = val_robust_acc
            batch_val_acc = 0
            
            for val_batch in val_dl:
                
                val_batch = util.to_device(val_batch, device)
                x, label = val_batch
        
                val_x = util.craft_PGD_adv_samples(model, x, label, epsilon, 0.01, 10)
                #val_x = util.craft_FGSM_adv_samples(model, x, label, 0.4, 1.25*0.4)
                val_pred, val_z = model(val_x)
                batch_val_acc += util.accuracy(val_pred, label)
                
            batch_val_acc /= len(val_dl)
            val_robust_acc = batch_val_acc
            
            
    
    
    batch_loss /= len(train_dl)
    batch_partial_losses /= len(train_dl) 
    
    print("Triggered",trigger/40, "/", len(train_dl))
    
    return batch_loss, batch_partial_losses


def gardAlign_train(model, optimizer, train_dl, loss_fn,
              epsilon, alpha, gmm_centers, gmm_std, weights, coup):
        
        batch_loss = 0
        batch_partial_losses = 0
        
        
        model.train()
        
        for i,batch in enumerate(train_dl):
            
            batch = util.to_device(batch, device)
            x, label = batch

            # craft Adversarial Samples
            optimizer.zero_grad()
            adv_x = util.craft_FGSM_adv_samples(model, x, label, epsilon, alpha)
                
            # clean Images
            pred, z = model(x)
                    
            # adversarial images
            adv_pred, adv_z = model(adv_x)  
                
            # loss
            loss, partial_losses =\
                loss_fn(pred, adv_pred, label, z, adv_z, weights, gmm_centers, gmm_std, coup)
                
                
            # Grad Align
            grad_align_lambda = 16
            
            grad1 = util.get_input_grad(model, x, label, optimizer, epsilon, delta_init='none', backprop=False)
            grad2 = util.get_input_grad(model, x, label, optimizer, epsilon, delta_init='random_uniform', backprop=True)
            grad1, grad2 = grad1.reshape(len(grad1), -1), grad2.reshape(len(grad2), -1)
            cos = torch.nn.functional.cosine_similarity(grad1, grad2, 1)
            reg = grad_align_lambda * (1.0 - cos.mean())
            


            loss += reg
        
            
            # do optimization
            optimizer.zero_grad()
            loss.backward()    
            optimizer.step()
            
            
            batch_loss += loss.item()
            batch_partial_losses += partial_losses
            
            
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
    batch_acc = 0
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
        
        batch_acc += util.accuracy(pred, label)
        
        
        
        # Standart Training Latent FGSM
        adv_x = util.craft_FGSM_adv_samples(model, x, label, 0.3, 0.375)
        adv_pred, adv_z = model(adv_x)

        loss, partial_loss =\
            loss_fn(pred, adv_pred, label, z, adv_z, weights, gmm_centers, gmm_std, coup)
        
        batch_fgsm_loss += loss.item()
        batch_fgsm_partial_loss += partial_loss
        
        batch_fgsm_acc += util.accuracy(adv_pred, label)
        
        
        
        # Latent FGSM with larger epsilon
        adv_x = util.craft_FGSM_adv_samples(model, x, label, 0.4, 1.25*0.4)
        adv_pred, adv_z = model(adv_x)

        loss, partial_loss =\
            loss_fn(pred, adv_pred, label, z, adv_z, weights, gmm_centers, gmm_std, coup)
        
        batch_large_eps_fgsm_loss += loss.item()
        batch_large_eps_fgsm_partial_loss += partial_loss
        
        batch_large_eps_fgsm_acc += util.accuracy(adv_pred, label)
        
        
        
        # Output FGSM
        adv_x = util.craft_output_FGSM_adv_samples(model, x, label, 0.3, 0.375)
        adv_pred, adv_z = model(adv_x)

        loss, partial_loss =\
            loss_fn(pred, adv_pred, label, z, adv_z, weights, gmm_centers, gmm_std, coup)
        
        batch_output_fgsm_loss += loss.item()
        batch_output_fgsm_partial_loss += partial_loss
        
        batch_output_fgsm_acc += util.accuracy(adv_pred, label)
        
        
        
        
        # Latent PGD
        adv_x = util.craft_PGD_adv_samples(model, x, label, 0.3, 0.01, 10)
        adv_pred, adv_z = model(adv_x)

        loss, partial_loss =\
            loss_fn(pred, adv_pred, label, z, adv_z, weights, gmm_centers, gmm_std, coup)
        
        batch_pgd_loss += loss.item()
        batch_pgd_partial_loss += partial_loss
        
        batch_pgd_acc += util.accuracy(adv_pred, label)
        
        
        # Output PGD
        adv_x = util.craft_output_PGD_adv_samples(model, x, label, 0.3, 0.01, 10)
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
    
    batch_acc /= len(val_dl)
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
        'Clean Acc': batch_acc, 
        'Latent FGSM': batch_fgsm_acc,
        'Latent FGSM Large Eps': batch_large_eps_fgsm_acc,
        'Classification FGSM': batch_output_fgsm_acc,
        'Latent PGD': batch_pgd_acc,
        'Classification PGD': batch_output_pgd_acc
        }
    
    
    return total_losses, partial_losses, accuracies

    
        
 