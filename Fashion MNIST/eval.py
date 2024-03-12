import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np

import configparser

import util
import models
import load_save
import eval_tests





def evaluate(model, data, report):
    
    adv_method = "FGSM"
    if adv_method == "FGSM":
        craft_adv = util.craft_FGSM_adv_samples
    if adv_method == "PGD":
        craft_adv = util.craft_PGD_adv_samples
    if adv_method == "Output FGSM":
        craft_adv = util.craft_output_FGSM_adv_samples
    if adv_method == "Output PGD":
        craft_adv = util.craft_output_PGD_adv_samples 
    
    eval_tests.plot_train_report(report, file_path)
    eval_tests.plot_val_report(report, file_path)
    
    eval_tests.accuracy(model, data, craft_adv, epsilon, alpha)
    
    eval_tests.draw_samples(model, data, craft_adv, epsilon, alpha, num_samples = 2, file_path = file_path)
    
    eval_tests.get_confusion_matrix(model, data,  craft_adv, epsilon, alpha, file_path = file_path)
    
    eval_tests.get_tsne(model, data, craft_adv, epsilon, alpha, samples = 1000, file_path = file_path)
    
    eval_tests.get_marginals(model, data, craft_adv, epsilon, alpha,
                                comp1 = 0, comp2 = 1, samples = 1000, file_path = file_path)
    
    eval_tests.marginal_distribution(model, data, craft_adv, epsilon, alpha, 
                                    comp = 0, file_path = file_path)
    
    
    #eval_tests.check_supervision_samples(model, batch_size = 100, batches_per_class = 10)
    eval_tests.check_supervision_data(model, data, craft_adv, epsilon, alpha, file_path)
    


    epsilons = np.array([0.2,0.3,0.35,0.4,0.5])
    alphas = 1.25 * epsilons
    eval_tests.fgsm_robustness(model, data, True, epsilons, alphas)
    
    epsilons = np.array([0.2,0.3,0.35,0.4,0.5])
    alphas = 1.25 * epsilons
    eval_tests.fgsm_robustness(model, data, False, epsilons, alphas)
    
    
    epsilons = np.array([0.2,0.3,0.35,0.4,0.5])
    alphas = 0.01 * np.ones_like(epsilons)
    eval_tests.pgd_robustness(model, data, True, epsilons, alphas, 10)
    
    
    epsilons = np.array([0.2,0.3,0.35,0.4,0.5])
    alphas = 0.01 * np.ones_like(epsilons)
    eval_tests.pgd_robustness(model, data, False, epsilons, alphas, 10)

def main():
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    test_ds = datasets.FashionMNIST(
       root = data_dir,
       train = False,                         
       transform = transform, 
       download = True,            
    )
    
    
    # Dataloader
    test_dl = DataLoader(test_ds, batch_size, shuffle=True)
    
    
    # Get Model
    model = models.MNIST_Model(latent_dim)
    model = util.to_device(model, device)
    
    
    model, tl, vl, acc =\
        load_save.load(file_path ,file_name, model)
        
        
        
    evaluate(model, test_dl, (tl, vl, acc))
    
    
    

if __name__ == "__main__":

    
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    experiment = config.get("Experiments", "experiment")
    
    # train settings
    batch_size = config.getint(experiment, "batch_size")    
    
    # adversarial example settings
    alpha = config.getfloat(experiment, "alpha")
    epsilon = config.getfloat(experiment, "epsilon")
    
    
    # GMM settings
    latent_dim = config.getint(experiment, "latent_dim")
    num_clusters = config.getint(experiment, "num_clusters")
    coup = config.getfloat(experiment, "coup")
    
    gmm_centers, gmm_std = util.set_gmm_centers(latent_dim, num_clusters)
    
    # regularizer settings
    regularizer = config.get("Regularizer", "regularizer")


    # Data location
    date = "feb_2_"
    file_name = "/last.pth"
    data_dir = "C:/MT/Datasets/"
    file_path =  'models/' + regularizer + "/" + date + experiment + "/Class PGD-5"
    
    print(file_path)
    
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    else:
        print("CUDA NOT AVAILABLE")
        
    util.set_rng(-1) 
    
    # Get GPU
    device = util.get_default_device()   
    
    
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')
    
    
    try:
    
        main()
        
    except KeyboardInterrupt:
        print('\n\nSTOP\n\n')
        torch.cuda.empty_cache()

    torch.cuda.empty_cache()