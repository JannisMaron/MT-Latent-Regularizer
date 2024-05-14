import numpy as np
import torch
from torch.nn import functional as F

###############################################################################
# Loss
###############################################################################


def get_loss_fn(loss_fn):
    
    if loss_fn == "adv_KS_loss":
        return adv_KS_loss
    elif loss_fn == "clean_adv_KS_loss":
        return clean_adv_KS_loss
    elif loss_fn == "clean_adv_sup_KS_loss":
        return clean_adv_sup_KS_loss
    elif loss_fn == "clean_adv_inv_sup_KS_loss":
        return clean_adv_inv_sup_KS_loss
    elif loss_fn == "clean_adv_inv_sup_KS_pair_loss":
        return clean_adv_inv_sup_KS_pair_loss
    
    pass

def adv_KS_loss(clean_logits, adv_logits, targets, weights,
               gmm_centers, gmm_std, coup):
    '''
    Adv Loss
    KS Loss
    KS Pair Loss
    Cov Loss'''
    
    clean_weight = 0
    adv_weight = weights[1]
    ks_weight = weights[2]
    ks_pair_weight = weights [3]
    cov_weight = weights[4]
    
    
    clean = torch.tensor([0.0]).cuda()
    adv = F.cross_entropy(adv_logits, targets)
    ks = ks_loss(clean_logits, adv_logits, targets, gmm_centers, gmm_std)
    ks_pair = ks_pair_loss(clean_logits, adv_logits, gmm_centers, gmm_std)
    cov = covariance_loss(clean_logits, adv_logits, gmm_centers, gmm_std, coup)
    
    clean *= clean_weight
    adv *= adv_weight
    ks *= ks_weight
    ks_pair *= ks_pair_weight
    cov *= cov_weight
    
    total_loss = 0
    total_loss += clean
    total_loss += adv
    total_loss += ks
    total_loss += ks_pair
    total_loss += cov
    total_loss = total_loss.mean().cuda()
    
    partial_losses = []
    partial_losses.append(clean.item())
    partial_losses.append(adv.item())
    partial_losses.append(ks.item())
    partial_losses.append(ks_pair.item())
    partial_losses.append(cov.item())
    partial_losses = np.array(partial_losses)
        
    return total_loss, partial_losses

def clean_adv_KS_loss(clean_logits, adv_logits, targets, weights,
               gmm_centers, gmm_std, coup):
    '''
    Clean Loss,
    Adv Loss
    KS Loss
    KS Pair Loss
    Cov Loss'''
    
    clean_weight = weights[0]
    adv_weight = weights[1]
    ks_weight = weights[2]
    ks_pair_weight = weights [3]
    cov_weight = weights[4]
    
    
    clean = F.cross_entropy(clean_logits, targets)
    adv = F.cross_entropy(adv_logits, targets)
    ks = ks_loss(clean_logits, adv_logits, targets, gmm_centers, gmm_std)
    ks_pair = ks_pair_loss(clean_logits, adv_logits, gmm_centers, gmm_std)
    cov = covariance_loss(clean_logits, adv_logits, gmm_centers, gmm_std, coup)
    
    clean *= clean_weight
    adv *= adv_weight
    ks *= ks_weight
    ks_pair *= ks_pair_weight
    cov *= cov_weight
    
    total_loss = 0
    total_loss += clean
    total_loss += adv
    total_loss += ks
    total_loss += ks_pair
    total_loss += cov
    total_loss = total_loss.mean().cuda()
    
    partial_losses = []
    partial_losses.append(clean.item())
    partial_losses.append(adv.item())
    partial_losses.append(ks.item())
    partial_losses.append(ks_pair.item())
    partial_losses.append(cov.item())
    partial_losses = np.array(partial_losses)
        
    return total_loss, partial_losses


def clean_adv_sup_KS_loss(clean_logits, adv_logits, targets, weights,
               gmm_centers, gmm_std, coup):
    '''
    Clean Loss,
    Adv Loss
    Sup KS Loss
    KS Pair Loss
    Cov Loss'''
    
    clean_weight = weights[0]
    adv_weight = weights[1]
    sup_ks_weight = weights[2]
    ks_pair_weight = weights [3]
    cov_weight = weights[4]
    
    
    clean = F.cross_entropy(clean_logits, targets)
    adv = F.cross_entropy(adv_logits, targets)
    sup_ks = sup_ks_loss(clean_logits, adv_logits, targets, gmm_centers, gmm_std)
    ks_pair = ks_pair_loss(clean_logits, adv_logits, gmm_centers, gmm_std)
    cov = covariance_loss(clean_logits, adv_logits, gmm_centers, gmm_std, coup)
    
    clean *= clean_weight
    adv *= adv_weight
    sup_ks *= sup_ks_weight
    ks_pair *= ks_pair_weight
    cov *= cov_weight
    
    total_loss = 0
    total_loss += clean
    total_loss += adv
    total_loss += sup_ks
    total_loss += ks_pair
    total_loss += cov
    total_loss = total_loss.mean().cuda()
    
    partial_losses = []
    partial_losses.append(clean.item())
    partial_losses.append(adv.item())
    partial_losses.append(sup_ks.item())
    partial_losses.append(ks_pair.item())
    partial_losses.append(cov.item())
    partial_losses = np.array(partial_losses)
        
    return total_loss, partial_losses


def clean_adv_inv_sup_KS_loss(clean_logits, adv_logits, targets, weights,
               gmm_centers, gmm_std, coup):
    '''
    Clean Loss,
    Adv Loss
    Inv Sup KS Loss
    KS Pair Loss
    Cov Loss'''
    
    clean_weight = weights[0]
    adv_weight = weights[1]
    inv_sup_ks_weight = weights[2]
    ks_pair_weight = weights [3]
    cov_weight = weights[4]
    
    
    clean = F.cross_entropy(clean_logits, targets)
    adv = F.cross_entropy(adv_logits, targets)
    inv_sup_ks = inv_sup_ks_loss(clean_logits, adv_logits, targets, gmm_centers, gmm_std)
    ks_pair = ks_pair_loss(clean_logits, adv_logits, gmm_centers, gmm_std)
    cov = covariance_loss(clean_logits, adv_logits, gmm_centers, gmm_std, coup)
    
    clean *= clean_weight
    adv *= adv_weight
    inv_sup_ks *= inv_sup_ks_weight
    ks_pair *= ks_pair_weight
    cov *= cov_weight
    
    total_loss = 0
    total_loss += clean
    total_loss += adv
    total_loss += inv_sup_ks
    total_loss += ks_pair
    total_loss += cov
    total_loss = total_loss.mean().cuda()
    
    partial_losses = []
    partial_losses.append(clean.item())
    partial_losses.append(adv.item())
    partial_losses.append(inv_sup_ks.item())
    partial_losses.append(ks_pair.item())
    partial_losses.append(cov.item())
    partial_losses = np.array(partial_losses)
        
    return total_loss, partial_losses

def clean_adv_inv_sup_KS_pair_loss(clean_logits, adv_logits, targets, weights,
               gmm_centers, gmm_std, coup):
    '''
    Clean Loss,
    Adv Loss
    Inv Sup KS Loss
    Inv KS Pair Loss
    Cov Loss'''
    
    clean_weight = weights[0]
    adv_weight = weights[1]
    inv_sup_ks_weight = weights[2]
    inv_ks_pair_weight = weights [3]
    cov_weight = weights[4]
    
    
    clean = F.cross_entropy(clean_logits, targets)
    adv = F.cross_entropy(adv_logits, targets)
    inv_sup_ks = inv_sup_ks_loss(clean_logits, adv_logits, targets, gmm_centers, gmm_std)
    inv_ks_pair = inv_ks_pair_loss(clean_logits, adv_logits)
    cov = covariance_loss(clean_logits, adv_logits, gmm_centers, gmm_std, coup)
    
    clean *= clean_weight
    adv *= adv_weight
    inv_sup_ks *= inv_sup_ks_weight
    inv_ks_pair *= inv_ks_pair_weight
    cov *= cov_weight
    
    total_loss = 0
    total_loss += clean
    total_loss += adv
    total_loss += inv_sup_ks
    total_loss += inv_ks_pair
    total_loss += cov
    total_loss = total_loss.mean().cuda()
    
    partial_losses = []
    partial_losses.append(clean.item())
    partial_losses.append(adv.item())
    partial_losses.append(inv_sup_ks.item())
    partial_losses.append(inv_ks_pair.item())
    partial_losses.append(cov.item())
    partial_losses = np.array(partial_losses)
        
    return total_loss, partial_losses


    
    
    
###############################################################################
# Regularizer
###############################################################################



###############################################################################
# KS Loss
###############################################################################

def ks_loss(embedding_matrix, adv_embedding_matrix, labels, gmm_centers, gmm_std):

    total_embedding = torch.cat((embedding_matrix, adv_embedding_matrix), dim=0)
    sorted_embeddings = torch.sort(total_embedding, dim=-2).values
    emb_num, emb_dim = sorted_embeddings.shape[-2:]
    num_gmm_centers, _ = gmm_centers.shape
    # For the sorted embeddings, the empirical CDF depends to the "index" of each
    # embedding (the number of embeddings before it).
    # Unsqueeze enables broadcasting
    empirical_cdf = torch.linspace(
        start=1 / emb_num,
        end=1.0,
        steps=emb_num,
        device=embedding_matrix.device,
        dtype=embedding_matrix.dtype,
    ).unsqueeze(-1)

    # compute CDF values for the embeddings using the Error Function
    normalized_embedding_distances_to_centers = (sorted_embeddings[:, None] - gmm_centers[None]) / gmm_std
    normal_cdf_per_center = 0.5 * (1 + torch.erf(normalized_embedding_distances_to_centers * 0.70710678118))
    normal_cdf = normal_cdf_per_center.mean(dim=1)
    
    
    return torch.nn.functional.mse_loss(normal_cdf, empirical_cdf)


def sup_ks_loss(embedding_matrix, adv_embedding_matrix, labels, gmm_centers, gmm_std, num_classes = 10):
    
    loss = 0
    
    total_embedding = torch.cat((embedding_matrix, adv_embedding_matrix), dim=0)
    total_labels = torch.cat((labels,labels))
    
    for i in range(num_classes):
        mean_vec = gmm_centers[i]
        class_embeddings = total_embedding[total_labels == i]
        
        if len(class_embeddings) == 0:
            continue
        
        sorted_embeddings = torch.sort(class_embeddings, dim=-2).values
        emb_num, emb_dim = sorted_embeddings.shape[-2:]
        
        
        # empirical cdf
        empirical_cdf = torch.linspace(
            start=1 / emb_num,
            end=1.0,
            steps=emb_num+1,
            device=embedding_matrix.device,
            dtype=embedding_matrix.dtype,
        ).unsqueeze(-1)
        empirical_cdf = empirical_cdf[:-1]
        empirical_cdf = torch.tile(empirical_cdf,(1,10))
        
        
        
        normalized_embedding_distances_to_centers = (sorted_embeddings[:,None] - mean_vec[None]) / gmm_std
        normal_cdf_per_center = 0.5 * (1 + torch.erf(normalized_embedding_distances_to_centers * 0.70710678118))
        normal_cdf = normal_cdf_per_center.mean(dim=1)
        
        loss += torch.nn.functional.mse_loss(normal_cdf, empirical_cdf)
    
    return loss


def inverse_cdf_values(y, mean, std):
    z = torch.erfinv(2 * y - 1)
    x = mean + std * (2**0.5) * z
    
    return x


def inv_sup_ks_loss(embedding_matrix, adv_embedding_matrix, labels, gmm_centers, gmm_std, num_classes = 10):
    loss = 0
    
    total_embedding = torch.cat((embedding_matrix, adv_embedding_matrix), dim=0)
    total_labels = torch.cat((labels,labels))
    
    for i in range(num_classes):
        mean_vec = gmm_centers[i]
        class_embeddings = total_embedding[total_labels == i]
        
        if len(class_embeddings) == 0:
            continue
        
        sorted_embeddings = torch.sort(class_embeddings, dim=-2).values
        emb_num, emb_dim = sorted_embeddings.shape[-2:]
        
        
        # empirical cdf
        empirical_cdf = torch.linspace(
            start=1 / emb_num,
            end=1.0,
            steps=emb_num+1,
            device=embedding_matrix.device,
            dtype=embedding_matrix.dtype,
        ).unsqueeze(-1)
        empirical_cdf = empirical_cdf[:-1]
        empirical_cdf = torch.tile(empirical_cdf,(1,10))
        
        inv_empirical_cdf = inverse_cdf_values(empirical_cdf, mean_vec, gmm_std)
        
        
        loss += torch.nn.functional.mse_loss(sorted_embeddings, inv_empirical_cdf)
    
    return loss



###############################################################################
# KS Pair Loss
###############################################################################

def ks_pair_loss(embedding_matrix, adv_embedding_matrix, gmm_centers, gmm_std):

    sorted_embeddings = torch.sort(embedding_matrix, dim=-2).values
    sorted_advembeddings = torch.sort(adv_embedding_matrix, dim=-2).values
    num_gmm_centers, _ = gmm_centers.shape

    # compute CDF values for the embeddings using the Error Function
    normalized_embedding_distances_to_centers = (sorted_embeddings[:, None] - gmm_centers[None]) / gmm_std
    normal_cdf_per_center = 0.5 * (1 + torch.erf(normalized_embedding_distances_to_centers * 0.70710678118))
    normal_cdf = normal_cdf_per_center.mean(dim=1)

    normalized_advembedding_distances_to_centers = (sorted_advembeddings[:, None] - gmm_centers[None]) / gmm_std
    normal_advcdf_per_center = 0.5 * (1 + torch.erf(normalized_advembedding_distances_to_centers * 0.70710678118))
    normal_advcdf = normal_advcdf_per_center.mean(dim=1)
    return torch.nn.functional.mse_loss(normal_cdf, normal_advcdf)



def inv_ks_pair_loss(embedding_matrix, adv_embedding_matrix):
    
    sorted_embeddings = torch.sort(embedding_matrix, dim=-2).values
    sorted_advembeddings = torch.sort(adv_embedding_matrix, dim=-2).values
    
    return torch.nn.functional.mse_loss(sorted_embeddings, sorted_advembeddings)
    
    


###############################################################################
# Covariance
###############################################################################



def compute_gmm_covariance(gmm_centers, gmm_std):

    num_gmm_centers, dimension = gmm_centers.shape
    component_cov = torch.eye(dimension) * gmm_std

    # Weighted ceters = mean due to equal weighin
    weighted_gmm_centers = gmm_centers.mean(axis=0)
    gmm_centers = gmm_centers - weighted_gmm_centers

    # Implementing Law of total variance:;
    # Conditional Expectation:
    conditional_expectation = 0
    for component in range(num_gmm_centers):
        center_mean = gmm_centers[component, :].reshape(dimension, 1)
        conditional_expectation += (1 / num_gmm_centers) * torch.mm(center_mean, center_mean.t())
    # Expected conditional variance equals component_cov, since all components are weighted equally,
    # and all component covariances are the same.
    return component_cov.cuda(), component_cov.cuda() + conditional_expectation.cuda()



def compute_empirical_covariance(embedding_matrix):

    m = torch.mean(embedding_matrix, dim=0)
    sigma = (
            torch.mm((embedding_matrix - m).t(), (embedding_matrix - m))
            / embedding_matrix.shape[0]
    )
    return sigma


def covariance_loss(embedding_matrix, adv_embedding_matrix, gmm_centers, gmm_std, coup):

    # Compute empirical covariances:
    comp_covariance, gmm_covariance = compute_gmm_covariance(gmm_centers, gmm_std)
    comp_covariance.to(embedding_matrix.device)
    gmm_covariance.to(embedding_matrix.device)

    # Compute cross covariances: -- DOES NOTHING--
    #cross_covar = torch.eye(gmm_covariance.size(dim=0)).cuda()
    #cross_covar.to(embedding_matrix.device)
    
    gmm_covariance1 = torch.cat((gmm_covariance, coup * gmm_covariance), dim=0)
    gmm_covariance2 = torch.cat((coup * gmm_covariance, gmm_covariance), dim=0)
    combined_gmm_covariance = torch.cat((gmm_covariance1, gmm_covariance2), dim=1)
    
    combined_empirical = torch.cat((embedding_matrix, adv_embedding_matrix), dim=1)
    sigma_combined = compute_empirical_covariance(combined_empirical)
    
    diff = torch.pow(sigma_combined - combined_gmm_covariance, 2)
    mean_cov = torch.mean(diff)
    
    return mean_cov



###############################################################################
# Loss weights
###############################################################################

def get_estimate_loss_fn(loss_fn):
    
    if loss_fn == "adv_KS_loss":
        return estimate_KS_coefficients
    elif loss_fn == "clean_adv_KS_loss":
        return estimate_KS_coefficients
    elif loss_fn == "clean_adv_sup_KS_loss":
        return estimate_sup_KS_coefficients
    elif loss_fn == "clean_adv_inv_sup_KS_loss":
        return estimate_inv_sup_KS_coefficients
    elif loss_fn == "clean_adv_inv_sup_KS_pair_loss":
        return estimate_inv_sup_KS_pair_coefficients
    
    pass


def estimate_KS_coefficients(batch_size, gmm_centers, gmm_std, coup, num_samples=100):
    """Clean and Adv"""
    
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
        
        rand_targets = torch.randint(0,10,size=(batch_size,))
        
        
        ksloss = ks_loss(z, z1, rand_targets, gmm_centers=gmm_centers,gmm_std=gmm_std)
        ksloss = ksloss.cpu().detach().numpy()
        
        ks_pairloss = ks_pair_loss(z, z1, gmm_centers=gmm_centers, gmm_std=gmm_std)
        ks_pairloss = ks_pairloss.cpu().detach().numpy()
        
        cv_loss = covariance_loss(z, z1, gmm_centers=gmm_centers, gmm_std=gmm_std, coup=coup)
        cv_loss = cv_loss.cpu().detach().numpy()

        ks_losses.append(ksloss)
        ks_pairlosses.append(ks_pairloss)
        cv_losses.append(cv_loss)

    ks_weight = 1 / np.mean(ks_losses)
    ks_pair_weight = 1 / np.mean(ks_pairlosses)
    cv_weight = 1 / np.mean(cv_losses)
    return ks_weight, ks_pair_weight, cv_weight


def estimate_sup_KS_coefficients(batch_size, gmm_centers, gmm_std, coup, num_samples=100):
    """Supervised KS Loss"""
    
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
        
        rand_targets = torch.randint(0,10,size=(batch_size,))
        
        
        ksloss = sup_ks_loss(z, z1, rand_targets,gmm_centers=gmm_centers,gmm_std=gmm_std)
        ksloss = ksloss.cpu().detach().numpy()
        
        ks_pairloss = ks_pair_loss(z, z1, gmm_centers=gmm_centers, gmm_std=gmm_std)
        ks_pairloss = ks_pairloss.cpu().detach().numpy()
        
        cv_loss = covariance_loss(z, z1, gmm_centers=gmm_centers, gmm_std=gmm_std, coup=coup)
        cv_loss = cv_loss.cpu().detach().numpy()

        ks_losses.append(ksloss)
        ks_pairlosses.append(ks_pairloss)
        cv_losses.append(cv_loss)

    ks_weight = 1 / np.mean(ks_losses)
    ks_pair_weight = 1 / np.mean(ks_pairlosses)
    cv_weight = 1 / np.mean(cv_losses)
    return ks_weight, ks_pair_weight, cv_weight


def estimate_inv_sup_KS_coefficients(batch_size, gmm_centers, gmm_std, coup, num_samples=100):
    """Inverse Supervised KS Loss"""
    
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
        
        rand_targets = torch.randint(0,10,size=(batch_size,))
        
        
        ksloss = inv_sup_ks_loss(z, z1, rand_targets, gmm_centers=gmm_centers,gmm_std=gmm_std)
        ksloss = ksloss.cpu().detach().numpy()
        
        ks_pairloss = ks_pair_loss(z, z1, gmm_centers=gmm_centers, gmm_std=gmm_std)
        ks_pairloss = ks_pairloss.cpu().detach().numpy()
        
        cv_loss = covariance_loss(z, z1, gmm_centers=gmm_centers, gmm_std=gmm_std, coup=coup)
        cv_loss = cv_loss.cpu().detach().numpy()

        ks_losses.append(ksloss)
        ks_pairlosses.append(ks_pairloss)
        cv_losses.append(cv_loss)

    ks_weight = 1 / np.mean(ks_losses)
    ks_pair_weight = 1 / np.mean(ks_pairlosses)
    cv_weight = 1 / np.mean(cv_losses)
    return ks_weight, ks_pair_weight, cv_weight


def estimate_inv_sup_KS_pair_coefficients(batch_size, gmm_centers, gmm_std, coup, num_samples=100):
    """Inverse Supervised KS Pair Loss"""
    
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
        
        rand_targets = torch.randint(0,10,size=(batch_size,))
        
        
        ksloss = inv_sup_ks_loss(z, z1, rand_targets, gmm_centers=gmm_centers,gmm_std=gmm_std)
        ksloss = ksloss.cpu().detach().numpy()
        
        ks_pairloss = inv_ks_pair_loss(z, z1)
        ks_pairloss = ks_pairloss.cpu().detach().numpy()
        
        cv_loss = covariance_loss(z, z1, gmm_centers=gmm_centers, gmm_std=gmm_std, coup=coup)
        cv_loss = cv_loss.cpu().detach().numpy()

        ks_losses.append(ksloss)
        ks_pairlosses.append(ks_pairloss)
        cv_losses.append(cv_loss)

    ks_weight = 1 / np.mean(ks_losses)
    ks_pair_weight = 1 / np.mean(ks_pairlosses)
    cv_weight = 1 / np.mean(cv_losses)
    return ks_weight, ks_pair_weight, cv_weight
    
    



###############################################################################
# GMM
###############################################################################

def set_gmm_centers(dimension, num_gmm_components, gmm_distance=10):

    gmm_centers = []
    mu = np.zeros(dimension)
    mu[0] = gmm_distance
    for i in range(0, num_gmm_components):
        gmm_centers.append(np.roll(mu, i))
    gmm_std = 1
    gmm_centers = np.array(gmm_centers)
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









