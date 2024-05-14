import torch
import numpy as np
import os
import common.torch
import common.numpy
import common.regularizer
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import attacks
import attacks.norms
import attacks.projections
import attacks.initializations
import attacks.objectives


class Evaluate:
    
    
    def __init__(self, model, testset, attack, objective, 
                 loss_fn,
                 gmm_centers, gmm_std, coup, 
                 train_info_file,
                 train_progress_file,
                 imgs_folder = "imgs/", cuda=True):
        
        self.model = model
        self.testset = testset
        
        self.attack = attack
        self.objective = objective
        
        train_info = torch.load(train_info_file)
        self.weights = train_info['weights']
        self.loss_fn = loss_fn
        self.gmm_centers = gmm_centers
        self.gmm_std = gmm_std
        self.coup = coup
        
        self.train_info_file = train_info_file
        self.train_progress_file = train_progress_file
        
        self.imgs_folder = imgs_folder
        if not os.path.exists(self.imgs_folder):
            os.makedirs(self.imgs_folder)
        
        self.cuda = cuda
        
        self.model.eval()
        
        
        self.classes = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck") 
     
        
    def train_weights(self):
        
        train_info = torch.load(self.train_info_file)
        
        weight_scale = train_info['weight_scale']
        weights = train_info['weights']
        
        print("Weight Scale: ", weight_scale)
        print("Weights: ", weights)
        pass
        
    def train_progress(self):
        
        train_progress = torch.load(self.train_progress_file)
        
        epochs = train_progress['epoch']
        train_losses = train_progress['train_losses']
        train_partial_losses = train_progress['train_partial_losses']
        test_losses = train_progress['test_losses']
        test_partial_losses = train_progress['test_partial_losses']
        clean_acc = train_progress['clean_acc']
        adv_acc = train_progress['adv_acc']
        
        # Total Loss
        x = np.arange(epochs+1)
        

        
        # Total Loss
        fig,ax1 = plt.subplots(2,figsize=(12,12))
        ax1[0].plot(x, train_losses, label = "Train Loss")
        ax1[1].plot(x, test_losses, label = "Test Loss")
        
        for a in ax1:
            a.grid()
            a.legend()
        
        plt.suptitle('Total Loss - Trainings Loss vs. Test Loss')        
        plt.tight_layout()
        plt.savefig(self.imgs_folder + "/Total Loss.png")
        plt.show()
        
        
        # Partial Losses
        fig,ax2 = plt.subplots(5,2,figsize=(24,12))
        ax2[0,0].plot(x, train_partial_losses[:,0], label = "Train Clean Loss")
        ax2[1,0].plot(x, train_partial_losses[:,1], label = "Train Adv Loss")
        ax2[2,0].plot(x, train_partial_losses[:,2], label = "Train Inv-Sup-KS Loss")
        ax2[3,0].plot(x, train_partial_losses[:,3], label = "Train KS-Pair Loss")
        ax2[4,0].plot(x, train_partial_losses[:,4], label = "Train Cov Loss")
        
        ax2[0,1].plot(x, test_partial_losses[:,0], label = "Test Clean Loss")
        ax2[1,1].plot(x, test_partial_losses[:,1], label = "Test Adv Loss")
        ax2[2,1].plot(x, test_partial_losses[:,2], label = "Test Inv-Sup-KS Loss")
        ax2[3,1].plot(x, test_partial_losses[:,3], label = "Test KS-Pair Loss")
        ax2[4,1].plot(x, test_partial_losses[:,4], label = "Test Cov Loss")
        
        for a in ax2:
            a[0].grid()
            a[1].grid()
            a[0].legend()
            a[1].legend()
            
        plt.suptitle('Partial Loss - Trainings Loss vs. Test Loss')        
        plt.tight_layout()
        plt.savefig(self.imgs_folder + "/Partial Loss.png")
        plt.show()
        
        
        # Accuracy
        fig,ax3 = plt.subplots(2,figsize=(12,12))
        ax3[0].plot(x, clean_acc, label = "Clean Test Acc")
        ax3[1].plot(x, adv_acc, label = "Adv Test Acc")
        
        for a in ax3:
            a.grid()
            a.legend()
        
        plt.suptitle('Test Accuracy - Clean vs. Adv')        
        plt.tight_layout()
        plt.savefig(self.imgs_folder + "/Accuracy.png")
        plt.show()
        
        #print(clean_acc)
        #print(clean_acc.shape)
        print("Last Clean Acc: ", clean_acc[-1])
        print("Last Adv Acc: ", adv_acc[-1])
        

        
        pass
    
    def clean_accuracy(self):
        
        correct = 0
        total = 0
        
        for b, (inputs, targets) in enumerate(self.testset):
            
            inputs = common.torch.as_variable(inputs, self.cuda)
            targets = common.torch.as_variable(targets, self.cuda)
            
            logits = self.model(inputs)
            
            pred = F.softmax(logits, dim=1)
            pred = torch.argmax(pred,1)
            correct += (pred == targets).sum().item()
            
            total += targets.size(0)
            

        accuracy = correct / total
        print("Clean Accuracy: ", accuracy)
        pass
    
    
    def adv_accuracy(self):
        
        correct = 0
        total = 0
        
        for b, (inputs, targets) in enumerate(self.testset):
            
            inputs = common.torch.as_variable(inputs, self.cuda)
            targets = common.torch.as_variable(targets, self.cuda)
            
            self.objective.set(targets)
            perturbations_b, errors_b = self.attack.run(self.model, inputs, self.objective, prefix='')
            inputs = inputs + common.torch.as_variable(perturbations_b, self.cuda)
            
            logits = self.model(inputs)
            
            pred = F.softmax(logits, dim=1)
            pred = torch.argmax(pred,1)
            correct += (pred == targets).sum().item()
            
            total += targets.size(0)
            
        accuracy = correct / total
        print("Adversarial Accuracy: ", accuracy)
        
        pass
    
    
    def check_partial_loss(self):
        
        total_partial_loss = np.array([0,0,0,0,0], dtype=np.float64)
        
        for b, (inputs, targets) in enumerate(self.testset):
            
            inputs = common.torch.as_variable(inputs, self.cuda)
            targets = common.torch.as_variable(targets, self.cuda)
            
            self.objective.set(targets)
            perturbations_b, errors_b = self.attack.run(self.model, inputs, self.objective, prefix='')
            adv_inputs = inputs + common.torch.as_variable(perturbations_b, self.cuda)
            
            logits = self.model(inputs)
            adv_logits = self.model(adv_inputs)
            
            
            loss, partial_loss = self.loss_fn(logits, adv_logits, targets, self.weights,
                                                                 self.gmm_centers, self.gmm_std, self.coup)
            
            partial_loss /= targets.size(0)
            total_partial_loss += partial_loss
            
            
        print("Clean Loss:", total_partial_loss[0])
        print("Adv Loss:", total_partial_loss[1])  
        print("Inv Sup KS:", total_partial_loss[2])
        print("Pair KS:", total_partial_loss[3])
        print("Cov:", total_partial_loss[4])
        
        pass
    
    def clean_t_SNE(self, num_batches = 8):
        
        all_logits = []
        all_targets = []
        
        
        for b, (inputs, targets) in enumerate(self.testset):
            
            inputs = common.torch.as_variable(inputs, self.cuda)
            targets = common.torch.as_variable(targets, self.cuda)
            
            logits = self.model(inputs)
            
            logits = logits.cpu().detach().numpy()
            targets = targets.cpu().numpy()
            
            all_logits.extend(logits)
            all_targets.extend(targets)
            
            if (b == num_batches-1):
                break
            
       
        all_logits = np.array(all_logits)
        all_targets = np.array(all_targets)
        
        perplexity = (num_batches*self.testset.batch_size)/10
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        tsne_model = tsne.fit_transform(all_logits)
        
        plt.figure(figsize=(8, 8))
        plt.scatter(tsne_model[:, 0], tsne_model[:, 1], c=all_targets, cmap='viridis', edgecolors='k')
        plt.title('t-SNE Plot of Clean Data Logits')
        cbar = plt.colorbar()
        custom_ticks = list(range(10))
        cbar.set_ticks(custom_ticks)
        cbar.set_ticklabels(self.classes)     
        
        plt.tight_layout()
        plt.savefig(self.imgs_folder + "tSNE Clean.png")
        plt.show()
            
        return all_logits, all_targets
    
    
    def adv_t_SNE(self, num_batches = 8):
        
        all_logits = []
        all_targets = []
        
        
        for b, (inputs, targets) in enumerate(self.testset):
            
            inputs = common.torch.as_variable(inputs, self.cuda)
            targets = common.torch.as_variable(targets, self.cuda)
            
            self.objective.set(targets)
            perturbations_b, errors_b = self.attack.run(self.model, inputs, self.objective, prefix='')
            inputs = inputs + common.torch.as_variable(perturbations_b, self.cuda)
            
            logits = self.model(inputs)
            
            logits = logits.cpu().detach().numpy()
            targets = targets.cpu().numpy()
            
            all_logits.extend(logits)
            all_targets.extend(targets)
            
            if (b == num_batches-1):
                break
            
       
        all_logits = np.array(all_logits)
        all_targets = np.array(all_targets)
        
        perplexity = (num_batches*self.testset.batch_size)//10
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        tsne_model = tsne.fit_transform(all_logits)
        
        plt.figure(figsize=(8, 8))
        plt.scatter(tsne_model[:, 0], tsne_model[:, 1], c=all_targets, cmap='viridis', edgecolors='k')
        plt.title('t-SNE Plot of Adversarial Data Logits')
        cbar = plt.colorbar()
        custom_ticks = list(range(10))
        cbar.set_ticks(custom_ticks)
        cbar.set_ticklabels(self.classes) 
        
        plt.tight_layout()
        plt.savefig(self.imgs_folder + "tSNE Adv.png")
        plt.show()
        
        return all_logits, all_targets
    
    
    def combined_t_SNE(self, clean_logits, adv_logits, clean_targets, adv_targets):
        
        all_logits = np.concatenate((clean_logits, adv_logits))
        all_targets =  np.concatenate((clean_targets, adv_targets))
        
        perplexity = (8*self.testset.batch_size*2)//10
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        tsne_model = tsne.fit_transform(all_logits)
        
        plt.figure(figsize=(8, 8))
        plt.scatter(tsne_model[:, 0], tsne_model[:, 1], c=all_targets, cmap='viridis', edgecolors='k')
        plt.title('t-SNE Plot of Combined Data Logits')
        cbar = plt.colorbar()
        custom_ticks = list(range(10))
        cbar.set_ticks(custom_ticks)
        cbar.set_ticklabels(self.classes) 
        
        plt.tight_layout()
        plt.savefig(self.imgs_folder + "tSNE Combined.png")
        plt.show()
        
        pass
    
    
    def clean_marginals(self, num_batches = 8, component_1 = 0, component_2 = 1):
        
        all_logits = []
        all_targets = []
        
        
        for b, (inputs, targets) in enumerate(self.testset):
            
            inputs = common.torch.as_variable(inputs, self.cuda)
            targets = common.torch.as_variable(targets, self.cuda)
            
            logits = self.model(inputs)
            
            logits = logits.cpu().detach().numpy()
            targets = targets.cpu().numpy()
            
            all_logits.extend(logits)
            all_targets.extend(targets)
            
            if (b == num_batches-1):
                break
            
       
        all_logits = np.array(all_logits)
        all_targets = np.array(all_targets)
        
        logits_x = all_logits[:,component_1]
        logits_y = all_logits[:,component_2]
        
        plt.scatter(logits_x, logits_y, c=all_targets, cmap='viridis', edgecolors='k')
        cbar = plt.colorbar()
        custom_ticks = list(range(10))
        cbar.set_ticks(custom_ticks)
        cbar.set_ticklabels(self.classes)
        plt.tight_layout()
        plt.savefig(self.imgs_folder + "marginals Clean.png")
        plt.show()

        pass
        
            
    def adv_marginals(self, num_batches = 8, component_1 = 0, component_2 = 1):
        
        all_logits = []
        all_targets = []
        
        
        for b, (inputs, targets) in enumerate(self.testset):
            
            inputs = common.torch.as_variable(inputs, self.cuda)
            targets = common.torch.as_variable(targets, self.cuda)
            
            self.objective.set(targets)
            perturbations_b, errors_b = self.attack.run(self.model, inputs, self.objective, prefix='')
            inputs = inputs + common.torch.as_variable(perturbations_b, self.cuda)
            
            logits = self.model(inputs)
            
            logits = logits.cpu().detach().numpy()
            targets = targets.cpu().numpy()
            
            all_logits.extend(logits)
            all_targets.extend(targets)
            
            if (b == num_batches-1):
                break
            
       
        all_logits = np.array(all_logits)
        all_targets = np.array(all_targets)    
        
        logits_x = all_logits[:,component_1]
        logits_y = all_logits[:,component_2]
        
        plt.scatter(logits_x, logits_y, c=all_targets, cmap='viridis', edgecolors='k')
        cbar = plt.colorbar()
        custom_ticks = list(range(10))
        cbar.set_ticks(custom_ticks)
        cbar.set_ticklabels(self.classes)
        plt.tight_layout()
        plt.savefig(self.imgs_folder + "marginals Adv.png")
        plt.show()
        
    def check_robustness(self, epsilons = np.array([6/255, 8/255, 10/255, 12/255, 16/255])):
        
        for epsilon in epsilons:
            
            # define Attack
            attack = attacks.BatchGradientDescent()
            attack.norm = attacks.norms.LInfNorm()
            attack.initialization = attacks.initializations.LInfUniformNormInitialization(epsilon)
            attack.projection = attacks.projections.SequentialProjections([
                attacks.projections.BoxProjection(),
                attacks.projections.LInfProjection(epsilon),
            ])
            attack.base_lr = 0.007
            attack.lr_factor = 1
            attack.max_iterations = 10
            attack.normalized = True
            attack.backtrack = False
            attack.c = 0
            attack.momentum = 0
            
            objective = attacks.objectives.UntargetedF0Objective()
            
            # calculate accuracy
            correct = 0
            total = 0
            
            for b, (inputs, targets) in enumerate(self.testset):
                
                inputs = common.torch.as_variable(inputs, self.cuda)
                targets = common.torch.as_variable(targets, self.cuda)
                
                objective.set(targets)
                perturbations_b, errors_b = attack.run(self.model, inputs, objective, prefix='')
                inputs = inputs + common.torch.as_variable(perturbations_b, self.cuda)
                
                logits = self.model(inputs)
                
                pred = F.softmax(logits, dim=1)
                pred = torch.argmax(pred,1)
                correct += (pred == targets).sum().item()
                
                total += targets.size(0)
                
            accuracy = correct / total
            
            
            print("Epsilon", np.round(epsilon,3),"=> Accuracy", np.round(accuracy,5))
            
            
        
        pass
    
    def autoattack_accuracy(self):
        
        attack = attacks.BatchAutoAttack()
        attack.epsilon = 0.03
        attack.version = 'standard'
        attack.norm = 'Linf'
        
        objective = attacks.objectives.UntargetedF0Objective()
        
        # calculate accuracy
        correct = 0
        total = 0
        
        
        for b, (inputs, targets) in enumerate(self.testset):
            
            inputs = common.torch.as_variable(inputs, self.cuda)
            targets = common.torch.as_variable(targets, self.cuda)
            
            objective.set(targets)
            perturbations_b, errors_b = attack.run(self.model, inputs, objective, prefix='')
            inputs = inputs + common.torch.as_variable(perturbations_b, self.cuda)
            
            logits = self.model(inputs)
            
            pred = F.softmax(logits, dim=1)
            pred = torch.argmax(pred,1)
            correct += (pred == targets).sum().item()
            
            total += targets.size(0)
            
        accuracy = correct / total
        
        
        print("AutoAttack Accuracy", np.round(accuracy,5))
        
        pass
        
        
    def evaluate(self):
        
        print()
        
        print("--Train Info--")
        self.train_weights()
        self.train_progress()
        print()
        
        
        print("--Test Accuracy--")
        self.clean_accuracy()
        self.adv_accuracy()
        print()
        
        print("--Partial Test Loss--")
        self.check_partial_loss()
        print()
        
        print("--t-SNE Plots--")
        clean_logits, clean_targets = self.clean_t_SNE(num_batches = 8)
        adv_logits, adv_targets = self.adv_t_SNE(num_batches = 8)
        self.combined_t_SNE(clean_logits, adv_logits, clean_targets, adv_targets)
        print()
        
        print("--Marginals--")
        self.clean_marginals(num_batches=8, component_1=0, component_2=1)
        self.adv_marginals(num_batches=8, component_1=0, component_2=1)
        print()
        
        print("--Robustness PGD--")
        self.check_robustness(epsilons = np.array([6/255, 8/255, 10/255, 12/255, 16/255]))
        print()
        
        print("--Auto Attack Accuracy--")
        #self.autoattack_accuracy()
        
        
        pass