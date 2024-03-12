import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

import regularizer


###############################################################################
# Adversarial Samples
###############################################################################

def craft_FGSM_adv_samples(model, x, y, epsilon, alpha):
 
    
    # Normal images' latent
    _, input_latent_z = model(x)
    
    # FGSM attack to create adversarial samples based on latent loss
    delta = torch.zeros_like(x).uniform_(-epsilon, epsilon).cuda()
    delta.requires_grad = True
    
    # Latent of noisy imgs
    _,latent_z = model(x + delta)
    latentloss = F.mse_loss(input_latent_z, latent_z)
    
    latentloss.backward()
    grad = delta.grad.detach()
    
    # FGSM
    delta.data = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
    delta.data = torch.max(torch.min(1 - x, delta.data), 0 - x)
    delta = delta.detach()
    adv_images = torch.clamp(x + delta, 0, 1)
    
    return adv_images

    


def craft_output_FGSM_adv_samples(model, x, y, epsilon, alpha):
 
    # FGSM attack to create adversarial samples based on latent loss
    delta = torch.zeros_like(x).uniform_(-epsilon, epsilon).cuda()
    delta.requires_grad = True
    
    # Latent of noisy imgs
    output,_ = model(x + delta)
    outputloss = F.cross_entropy(output,y)
    
    outputloss.backward()
    grad = delta.grad.detach()
    
    # FGSM
    delta.data = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
    delta.data = torch.max(torch.min(1 - x, delta.data), 0 - x)
    delta = delta.detach()
    adv_images = torch.clamp(x + delta, 0, 1)
    

    return adv_images




def craft_PGD_adv_samples(model, x, y, epsilon, alpha, iterations = 10):

    x = x.clone().detach()
    
    _,target = model(x)
    target = target.detach()

    adv_x = x.clone().detach()


    adv_x = adv_x + torch.empty_like(adv_x).uniform_(-epsilon, epsilon)
    adv_x = torch.clamp(adv_x, min=0, max=1).detach()

    for i in range(iterations):
        adv_x.requires_grad = True
        
        _,latent_output = model(adv_x)

        # Calculate loss
        cost = F.mse_loss(latent_output, target)

        # Update adversarial images
        grad = torch.autograd.grad(
            cost, adv_x, retain_graph=False, create_graph=False
        )[0]

        adv_x = adv_x.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_x - x, min=-epsilon, max=epsilon)
        adv_x = torch.clamp(x + delta, min=0, max=1).detach()

    return adv_x


def craft_output_PGD_adv_samples(model, x, y, epsilon, alpha, iterations = 10):

    x = x.clone().detach()

    adv_x = x.clone().detach()

    adv_x = adv_x + torch.empty_like(adv_x).uniform_(-epsilon, epsilon)
    adv_x = torch.clamp(adv_x, min=0, max=1).detach()

    for i in range(iterations):
        adv_x.requires_grad = True
        
        outputs,_ = model(adv_x)

        # Calculate loss
        cost = F.cross_entropy(outputs, y)

        # Update adversarial images
        grad = torch.autograd.grad(
            cost, adv_x, retain_graph=False, create_graph=False
        )[0]

        adv_x = adv_x.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_x - x, min=-epsilon, max=epsilon)
        adv_x = torch.clamp(x + delta, min=0, max=1).detach()

    return adv_x


###############################################################################
# Loss weights
###############################################################################

def estimate_loss_coefficients(batch_size, gmm_centers, gmm_std, coup, num_samples=100):
    """Estimate the weights of our multi-modal loss."""
    _, dimension = gmm_centers.shape
    ks_losses, cv_losses, ks_pairlosses = [], [], []

    # Estimate wieghts with gmm samples:
    for i in range(num_samples):
        z, _ = draw_gmm_samples(
            batch_size, gmm_centers, gmm_std
        )
        z1, _ = draw_gmm_samples(
            batch_size, gmm_centers, gmm_std
        )
        ks_loss = regularizer.ks_loss(z, z1, gmm_centers=gmm_centers,gmm_std=gmm_std)
        ks_loss = ks_loss.cpu().detach().numpy()
        
        ks_pairloss = regularizer.ks_pair_loss(z, z1, gmm_centers=gmm_centers, gmm_std=gmm_std)
        ks_pairloss = ks_pairloss.cpu().detach().numpy()
        
        cv_loss = regularizer.covariance_loss(z, z1, gmm_centers=gmm_centers, gmm_std=gmm_std, coup=coup)
        cv_loss = cv_loss.cpu().detach().numpy()

        ks_losses.append(ks_loss)
        ks_pairlosses.append(ks_pairloss)
        cv_losses.append(cv_loss)

    ks_weight = 1 / np.mean(ks_losses)
    ks_pair_weight = 1 / np.mean(ks_pairlosses)
    cv_weight = 1 / np.mean(cv_losses)
    return ks_weight, ks_pair_weight, cv_weight




###############################################################################
# GMM
###############################################################################

def set_gmm_centers(dimension, num_gmm_components):

    gmm_centers = []
    mu = np.zeros(dimension)
    mu[0] = 10
    for i in range(0, num_gmm_components):
        gmm_centers.append(np.roll(mu, i))
    gmm_std = 1
    gmm_centers = torch.tensor(gmm_centers).cuda().float()
    return gmm_centers, gmm_std

def draw_gmm_samples(num_samples, gmm_centers, gmm_std):

    num_gmm_centers, dimension = gmm_centers.shape

    samples = []
    components = []
    for _ in range(num_samples):
        component = np.random.choice(range(num_gmm_centers))

        component_mean = gmm_centers[component, :]
        component_cov = torch.eye(dimension) * gmm_std

        distribution = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=component_mean.cuda(), covariance_matrix=component_cov.cuda()
        )

        sample = distribution.sample((1,))
        samples.append(sample)
        components.append(component)
    samples = torch.vstack(samples)

    return samples, components




###############################################################################
# Utility
###############################################################################
def get_uniform_delta(shape, eps, requires_grad=True):
    delta = torch.zeros(shape).cuda()
    delta.uniform_(-eps, eps)
    delta.requires_grad = requires_grad
    return delta

def get_input_grad(model, X, y, opt, eps, delta_init='none', backprop=False):
    if delta_init == 'none':
        delta = torch.zeros_like(X, requires_grad=True)
    elif delta_init == 'random_uniform':
        delta = get_uniform_delta(X.shape, eps, requires_grad=True)
    elif delta_init == 'random_corner':
        delta = get_uniform_delta(X.shape, eps, requires_grad=True)
        delta = eps * torch.sign(delta)
    else:
        raise ValueError('wrong delta init')
        

    output, z = model(X + delta)
    loss = F.cross_entropy(output, y)
    
    grad = torch.autograd.grad(loss, delta, create_graph=True if backprop else False)[0]
    
    if not backprop:
        grad, delta = grad.detach(), delta.detach()
    return grad



def accuracy(x,y):
    correct = 0
    batch_size = y.size(0)
    
    pred = F.softmax(x, dim=1)
    pred = torch.argmax(pred,1)
    correct += (pred == y).sum().item()
    
    accuracy = correct / batch_size
    
    return accuracy




def set_rng(seed):
    '''Set seed for determenistic behaviour
       If set to -1 to get randomness'''
    
    if seed == -1:
        return None
    else:
        seed = seed
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)      
        pass


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def append_to_dict(list_dict, value_dict):
    for key, value in value_dict.items():
        if key in list_dict:
            list_dict[key].append(value)

    return list_dict

def plot_acc_progress(epoch, accuracies):
    
    x = np.arange(epoch+1)
    
    for key, value in accuracies.items():
        
        acc = np.array(value)
        plt.plot(x, acc, label=key)

    plt.legend()
    plt.show()
    
    
    
    