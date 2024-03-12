import torch
import util


def save(model, optimizer, epoch, tl, tl_partial,  vl, vl_partial, accuracy,
         file_path, file_name):
    
    path = file_path + file_name
    torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch' : epoch,
    'tl': tl, 
    'tl_partial': tl_partial, 
    'vl': vl, 
    'vl_partial': vl_partial, 
    'acc': accuracy, 
    },path)
    
    
    
    
def load(file_path, file_name, model):
    
    path = file_path + file_name
    
    checkpoint = torch.load(path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    tl = checkpoint['tl']
    tl_partial = checkpoint['tl_partial']

    vl = checkpoint['vl']
    vl_partial = checkpoint['vl_partial']
    acc = checkpoint['acc']
    
    
    model.eval()
    
    return model, (tl, tl_partial), (vl, vl_partial), acc
    
    
