import common.torch
import common.summary
import common.numpy
import common.regularizer
import attacks
import numpy as np
import torch.nn.functional as F
from .normal_training import *


class RegAdversarialTraining(NormalTraining):
    
    def __init__(self, model, trainset, testset, optimizer, scheduler, attack, objective, 
                 weights, gmm_centers, gmm_std, coup,
                 loss,
                 grad_clip_value = 1,
                 augmentation=None, cuda=True):
        
        assert isinstance(attack, attacks.Attack)
        assert isinstance(objective, attacks.objectives.Objective)
        assert getattr(attack, 'norm', None) is not None
        
        super(RegAdversarialTraining, self).__init__(model, trainset, testset, optimizer, scheduler, augmentation,loss=common.torch.classification_loss, cuda=cuda)
        
        
        self.attack = attack
        """ (attacks.Attack) Attack. """
 
        self.objective = objective
        """ (attacks.Objective) Objective. """
 
        self.loss = loss
        self.weights = weights
        self.gmm_centers = gmm_centers
        self.gmm_std = gmm_std
        self.coup = coup
        
        self.grad_clip_value = grad_clip_value
 
 
        self.max_batches = 10
        """ (int) Number of batches to test adversarially on. """
 
        
        
    def train(self, epoch):
        
        batch_loss = 0
        batch_partial_losses = 0
        
        largest_grads = []
        
        for b, (inputs, targets) in enumerate(self.trainset):
            
            batch_size = inputs.size(0)
        
            if self.augmentation is not None:
                inputs = self.augmentation.augment_images(inputs.numpy())

            inputs = common.torch.as_variable(inputs, self.cuda)
            targets = common.torch.as_variable(targets, self.cuda)
            
            clean_inputs = inputs.clone()
            adversarial_inputs = inputs.clone()


            self.model.eval()
            self.objective.set(targets)
            adversarial_perturbations, adversarial_objectives = self.attack.run(self.model, adversarial_inputs, self.objective)
            adversarial_perturbations = common.torch.as_variable(adversarial_perturbations, self.cuda)
            adversarial_inputs = adversarial_inputs + adversarial_perturbations
        
            inputs = torch.cat((clean_inputs, adversarial_inputs), dim=0)
            
        
            self.model.train()
            assert self.model.training is True
            self.optimizer.zero_grad()

            logits = self.model(inputs)
            clean_logits = logits[:batch_size]
            adversarial_logits = logits[batch_size:]
            
            
            loss, partial_loss = self.loss(clean_logits, adversarial_logits, targets, 
                                         self.weights, self.gmm_centers, self.gmm_std, self.coup)
        
            loss.backward()
            
            # clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
            largest_grad = max(p.grad.data.abs().max() for p in self.model.parameters() if p.grad is not None)
            largest_grads.append(largest_grad.item())
          
            
            
            self.optimizer.step()
            self.scheduler.step()
        
        
            self.progress('train %d' % epoch, b, len(self.trainset), info='\nClean_Loss=%g Adv_Loss=%g KS_Loss=%g KS_Pair_Loss=%g CV_Loss=%g \nlr=%g' % (
                partial_loss[0].item(),
                partial_loss[1].item(),
                partial_loss[2].item(),
                partial_loss[3].item(),
                partial_loss[4].item(),
                self.scheduler.get_lr()[0],
            ))
            
            batch_loss += loss.item()
            batch_partial_losses += partial_loss
            
        batch_loss /= len(self.trainset)
        batch_partial_losses /= len(self.trainset) 
        
        
        print("Largest Grad:", max(largest_grads))
        
        return batch_loss, batch_partial_losses, max(largest_grads)
        
        
    def test(self, epoch):
        
        #probabilities = super(RegAdversarialTraining, self).test(epoch)

        self.model.eval()
        
        losses = []
        partial_losses = []
        clean_acc = 0
        adv_acc = 0

        clean_correct = 0
        adv_correct = 0
        total = 0

        objectives = None

        for b, (inputs, targets) in enumerate(self.testset):
            
            if b >= self.max_batches:
                break
            
            batch_size = inputs.size(0)
            
            inputs = common.torch.as_variable(inputs, self.cuda)
            targets = common.torch.as_variable(targets, self.cuda)
            
            clean_inputs = inputs.clone()
            
            self.objective.set(targets)
            adversarial_perturbations, adversarial_objectives = self.attack.run(self.model, inputs, self.objective)
            objectives = common.numpy.concatenate(objectives, adversarial_objectives)

            adversarial_perturbations = common.torch.as_variable(adversarial_perturbations, self.cuda)
            adversarial_inputs = inputs + adversarial_perturbations
            
            inputs = torch.cat((clean_inputs, adversarial_inputs), dim=0)
            
            with torch.no_grad():
                logits = self.model(inputs)
                clean_logits = logits[:batch_size]
                adversarial_logits = logits[batch_size:]
            
            
                loss, partial_loss = self.loss(clean_logits, adversarial_logits, targets, 
                                             self.weights, self.gmm_centers, self.gmm_std, self.coup)
                
                
                clean_pred = F.softmax(clean_logits, dim=1)
                clean_pred = torch.argmax(clean_pred,1)
                clean_correct += (clean_pred == targets).sum().item()
                
                adv_pred = F.softmax(adversarial_logits, dim=1)
                adv_pred = torch.argmax(adv_pred,1)
                adv_correct += (adv_pred == targets).sum().item()
                
                total += targets.size(0)
                              
                losses.append(loss.detach().cpu().numpy())
                partial_losses.append(partial_loss)
                
                self.progress('test %d' % epoch, b, self.max_batches, info='loss=%g' % (
                    torch.mean(loss).item()
                ))
                
        losses = np.mean(np.array(losses))
        partial_losses = np.mean(np.array(partial_losses), axis=0)
        
        
        clean_acc = clean_correct / total
        adv_acc = adv_correct / total
                    
        
        return losses, partial_losses, clean_acc, adv_acc
            
            
            
            
            
            