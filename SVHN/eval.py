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
    
    epsilon = 8/255
    alpha = 10/255
    
    adv_method = 'FGSM'
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
    
    
    epsilons = np.array([6, 8, 10, 12, 16]) / 255
    alphas = 1.25 * epsilons
    #eval_tests.fgsm_robustness(model, data, True, epsilons, alphas)
    
    epsilons = np.array([6, 8, 10, 12, 16]) / 255
    alphas = 1.25 * epsilons
    #eval_tests.fgsm_robustness(model, data, False, epsilons, alphas)
    
    
    epsilons =np.array([6, 8, 10, 12, 16]) / 255
    alphas = 2/255 * np.ones_like(epsilons)
    #eval_tests.pgd_robustness(model, data, True, epsilons, alphas, 10)
    
    
    epsilons = np.array([6, 8, 10, 12, 16]) / 255
    alphas = 2/255 * np.ones_like(epsilons)
    #eval_tests.pgd_robustness(model, data, False, epsilons, alphas, 10)
    
    #A = np.random.randint(0, 255, size = (1,3,32,32), dtype=np.uint8)
    #A = torch.Tensor(A).cuda()
    
    #B = np.random.randn(1,3,32,32)
    #B = torch.Tensor(B).cuda()
    
    #imgs, label = next(iter(data))
    #img = imgs[0].unsqueeze(0).cuda()
    #print(imgs.shape)
    #print(img.shape)
    
    #print(model(A)[0])
    #print(model(B)[0])
    #print(model(img)[0])

def main():
    
    # Transforms
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Dataset
    test_ds = datasets.SVHN(
       root = data_dir,
       split = "test",                         
       transform = transform, 
       download = True,            
    ) 
    
    test_ds, _ = torch.utils.data.random_split(test_ds, [26000, 32])
        
  
    
    # Dataloader
    test_dl = DataLoader(test_ds, batch_size, shuffle=True)
    
    
    # Model
    model = models.SVHN_PreAct(latent_dim, identity_init=True)
    model = util.to_device(model, device)
    
    
    model, tl, vl, acc =\
        load_save.load(file_path ,file_name, model)
        
    evaluate(model, test_dl, (tl, vl, acc))
    
    pass

if __name__ == "__main__":

    
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    experiment = config.get("Experiments", "experiment")
    
    # train settings
    batch_size = config.getint(experiment, "batch_size")    
    
    # adversarial example settings
    epsilon = config.getfloat(experiment, "epsilon") / 255
    alpha = config.getfloat(experiment, "alpha") / 255
    
    
    
    # GMM settings
    latent_dim = config.getint(experiment, "latent_dim")
    num_clusters = config.getint(experiment, "num_clusters")
    coup = config.getfloat(experiment, "coup")
    
    gmm_centers, gmm_std = util.set_gmm_centers(latent_dim, num_clusters)
    
    # regularizer settings
    regularizer = config.get("Regularizer", "regularizer")



    # Data location
    date = "mar_1_"
    file_name = "/last.pth"
    data_dir = "C:/MT/Datasets/"
    file_path =  'models/' + regularizer + "/" + date + experiment + "/class_pgd-5/Ex1"
    
    print(file_path)
    
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    else:
        print("CUDA NOT AVAILABLE")
        
    #util.set_rng(-1) 
    
    # Get GPU
    device = util.get_default_device()   
    

    
    
    try:
    
        main()
        
    except KeyboardInterrupt:
        print('\n\nSTOP\n\n')
        torch.cuda.empty_cache()

    torch.cuda.empty_cache()