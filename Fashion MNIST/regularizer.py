import torch
import numpy as np
from torch.nn import functional as F
import torch.nn as nn


class L4_Loss(nn.Module):
    def __init__(self):
        super(L4_Loss, self).__init__()
        
    def forward(self, x, y):
        loss = torch.mean(torch.pow(x-y,4))
        return loss




###############################################################################
# Regularizer
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





def ks_loss(embedding_matrix, adv_embedding_matrix, gmm_centers, gmm_std):

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
# Supervised Losses
###############################################################################
def supervised_ks_loss(embedding_matrix, adv_embedding_matrix, labels, gmm_centers, gmm_std, num_classes = 10):
    
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
            steps=emb_num,
            device=embedding_matrix.device,
            dtype=embedding_matrix.dtype,
        ).unsqueeze(-1)
        
        empirical_cdf = torch.tile(empirical_cdf,(1,10))
        
        
        # gmm cdf
        normalized_embeddings = (sorted_embeddings - mean_vec) / gmm_std
        normal_cdf = 0.5 * (1 + torch.erf(normalized_embeddings * 0.70710678118))

        loss += torch.nn.functional.mse_loss(normal_cdf, empirical_cdf)
        
    return loss


def inverse_cdf_values(y, mean, std):
    z = torch.erfinv(2 * y - 1)
    x = mean + std * (2**0.5) * z
    
    return x


def inverse_supervised_ks_loss(embedding_matrix, adv_embedding_matrix, labels, gmm_centers, gmm_std, num_classes = 10):
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
        
        
        # gmm cdf
        #normalized_embeddings = (sorted_embeddings - mean_vec) / gmm_std
        #normal_cdf = 0.5 * (1 + torch.erf(normalized_embeddings * 0.70710678118))
        #loss += torch.nn.functional.mse_loss(normal_cdf, empirical_cdf)
        
        loss += torch.nn.functional.mse_loss(sorted_embeddings, inv_empirical_cdf)
    
    return loss

def supervised_ks_pair_loss(embedding_matrix, adv_embedding_matrix, labels, gmm_centers, gmm_std, num_classes = 10):
    
    loss = 0
    
    for i in range(num_classes):
        
        mean_vec = gmm_centers[i]
        class_embeddings = embedding_matrix[labels == i]
        adv_class_embeddings = adv_embedding_matrix[labels == i]
        
        if len(class_embeddings) == 0 or len(adv_embedding_matrix) == 0:
            continue
        
        sorted_embeddings = torch.sort(class_embeddings, dim=-2).values
        sorted_advembeddings = torch.sort(adv_class_embeddings, dim=-2).values
        
        
        normalized_embeddings = (sorted_embeddings[:, None] - mean_vec[None]) / gmm_std
        normal_cdf = 0.5 * (1 + torch.erf(normalized_embeddings * 0.70710678118))
        normal_cdf = normal_cdf.mean(dim=1)

        normalized_advembedding = (sorted_advembeddings[:, None] - mean_vec[None]) / gmm_std
        normal_advcdf = 0.5 * (1 + torch.erf(normalized_advembedding * 0.70710678118))
        normal_advcdf = normal_advcdf.mean(dim=1)
        
        loss += torch.nn.functional.mse_loss(normal_cdf, normal_advcdf)
                
        
    return loss
