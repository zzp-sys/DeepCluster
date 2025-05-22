"""
Authors: Wouter Van Gansbeke
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.config import create_config
from utils.common_config import get_model, get_train_dataset, \
                                get_val_dataset, \
                                get_val_dataloader, \
                                get_val_transformations \
                                
from utils.memory import MemoryBank
from utils.train_utils import simclr_train
from utils.utils import fill_memory_bank
from termcolor import colored

# Parser
parser = argparse.ArgumentParser(description='Eval_nn')
parser.add_argument('--config_env', default='/root/DeepClustermyproject/configs/env.yml',
                    help='Config file for the environment')
parser.add_argument('--config_exp', default='/root/DeepClustermyproject/configs/pretext/simclr_stl10.yml',
                    help='Config file for the experiment')
args = parser.parse_args()

def main():

    # Retrieve config file
    p = create_config(args.config_env, args.config_exp)
    print(colored(p, 'red'))
    
    # Model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p)
    print('Model is {}'.format(model.__class__.__name__))
    print('Model parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print(model)
    model = model.cuda()
   
    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue')) 
    torch.backends.cudnn.benchmark = True
    
    # Dataset
    val_transforms = get_val_transformations(p)
    print('Validation transforms:', val_transforms)
    val_dataset = get_val_dataset(p, val_transforms) 
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Dataset contains {} val samples'.format(len(val_dataset)))
    
    # Memory Bank
    print(colored('Build MemoryBank', 'blue'))
    base_dataset = get_train_dataset(p, val_transforms, split='train') # Dataset w/o augs for knn eval
    base_dataloader = get_val_dataloader(p, base_dataset) 
    memory_bank_base = MemoryBank(len(base_dataset), 
                                p['model_kwargs']['features_dim'],
                                p['num_classes'], p['criterion_kwargs']['temperature'])
    memory_bank_base.cuda()
    memory_bank_val = MemoryBank(len(val_dataset),
                                p['model_kwargs']['features_dim'],
                                p['num_classes'], p['criterion_kwargs']['temperature'])
    memory_bank_val.cuda()

    # Checkpoint
    assert os.path.exists(p['pretext_checkpoint'])
    print(colored('Restart from checkpoint {}'.format(p['pretext_checkpoint']), 'blue'))
    checkpoint = torch.load(p['pretext_checkpoint'], map_location='cpu')
    model.load_state_dict(checkpoint)
    model.cuda()
    
    # Save model
    torch.save(model.state_dict(), p['pretext_model'])

    # Mine the topk nearest neighbors at the very end (Train) 
    # These will be served as input to the SCAN loss.
    print(colored('Fill memory bank for mining the nearest neighbors (train) ...', 'blue'))
    fill_memory_bank(base_dataloader, model, memory_bank_base)
    topk = 20
    print('Mine the nearest neighbors (Top-%d)' %(topk)) 
    indices, acc = memory_bank_base.mine_nearest_neighbors(topk)
    print('Accuracy of top-%d nearest neighbors on train set is %.2f' %(topk, 100*acc))
    np.save(p['topk_neighbors_train_path'], indices)   

    # Mine the topk nearest neighbors at the very end (Val)
    # These will be used for validation.
    print(colored('Fill memory bank for mining the nearest neighbors (val) ...', 'blue'))
    fill_memory_bank(val_dataloader, model, memory_bank_val)
    topk = 5
    print('Mine the nearest neighbors (Top-%d)' %(topk)) 
    indices, acc = memory_bank_val.mine_nearest_neighbors(topk)
    print('Accuracy of top-%d nearest neighbors on val set is %.2f' %(topk, 100*acc))
    np.save(p['topk_neighbors_val_path'], indices)

    #画出近邻图像与自身(stl数据集)
    plt.figure()
    for i in range(1,topk+2):
        ax = plt.subplot(1,6,i)
        plt.imshow(val_dataloader.dataset.data[indices[12][i-1]].transpose(1,2,0))
        s = list(indices[12])
        class_number = val_dataloader.dataset.classes
        class_name = class_number[val_dataloader.dataset.labels[indices[12][i-1]]]
        plt.xlabel(s[i-1],fontsize=6)
        ax.set_title(class_name,fontsize=6)
        plt.xticks([])
        plt.yticks([])
    plt.savefig('/root/DeepClustermyproject/nnpictures/12-tok-5.svg')
    plt.show()   

 
if __name__ == '__main__':
    main()
