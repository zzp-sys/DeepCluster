"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import os
import torch
import numpy as np

from utils.config import create_config
from utils.common_config import get_criterion, get_model, get_decoderModel,get_optimizer1 ,get_train_dataset,\
                                get_val_dataset, get_train_dataloader,\
                                get_val_dataloader, get_train_transformations,\
                                get_val_transformations, get_optimizer,\
                                adjust_learning_rate
from utils.evaluate_utils import contrastive_evaluate
from utils.memory import MemoryBank
from utils.train_utils import simclr_train
from utils.utils import fill_memory_bank
from termcolor import colored
import torchvision
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Parser 
parser = argparse.ArgumentParser(description='SimCLR')
parser.add_argument('--config_env', default='/root/DeepClustermyproject/configs/env.yml',
                    help='Config file for the environment')
parser.add_argument('--config_exp', default='/root/DeepClustermyproject/configs/pretext/simclr_cifar10.yml',
                    help='Config file for the experiment')
args = parser.parse_args()

def main():

    # Retrieve config file
    p = create_config(args.config_env, args.config_exp)
    print(colored(p, 'red'))
    
    # Model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p)
    model1 = get_decoderModel()
    print('Model is {}'.format(model.__class__.__name__))
    print('Model parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print(model)
    model = model.cuda()
    model1 = model1.cuda()
    print(model1)
   
    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue')) 
    torch.backends.cudnn.benchmark = True
    
    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    train_transforms = get_train_transformations(p)
    print('Train transforms:', train_transforms)
    val_transforms = get_val_transformations(p)
    print('Validation transforms:', val_transforms)
    train_dataset = get_train_dataset(p, train_transforms, to_augmented_dataset=True,
                                        split='train+unlabeled') # Split is for stl-10
    val_dataset = get_val_dataset(p, val_transforms) 
    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Dataset contains {}/{} train/val samples'.format(len(train_dataset), len(val_dataset)))
    
    # # 画出数据增强之后的图片（train）
    # for i, batch in enumerate(train_dataloader):
    #     if i == 0:
    #         images = batch['image'][0].numpy().transpose(1,2,0)
    #         indice = batch['meta']['index'][0]
    #         print('标签为：',indice)
    #         images_augmented = batch['image_augmented'][0].numpy().transpose(1,2,0)
    #         plt.imshow(images_augmented)
    #         plt.xticks([])
    #         plt.yticks([])
    #         plt.savefig('/root/Myproject/scan/picture/agumented.png')

    #         plt.imshow(images)
    #         plt.xticks([])
    #         plt.yticks([])
    #         plt.savefig('/root/Myproject/scan/picture/images.png')
            
    #     else:
    #         break
    # plt.show()

    # Memory Bank(只用于knn验证，不参与训练)
    print(colored('Build MemoryBank', 'blue'))
    base_dataset = get_train_dataset(p, val_transforms, split='train') # Dataset w/o augs for knn eval
    base_dataloader = get_val_dataloader(p, base_dataset) #训练集数据

    memory_bank_base = MemoryBank(len(base_dataset), 
                                p['model_kwargs']['features_dim'],
                                p['num_classes'], p['criterion_kwargs']['temperature'])
    memory_bank_base.cuda()
    memory_bank_val = MemoryBank(len(val_dataset),
                                p['model_kwargs']['features_dim'],
                                p['num_classes'], p['criterion_kwargs']['temperature'])
    memory_bank_val.cuda()

    # Criterion
    print(colored('Retrieve criterion', 'blue'))
    criterion = get_criterion(p)
    criterion1 = torch.nn.MSELoss()
    print('Criterion is {}'.format(criterion.__class__.__name__))
    criterion = criterion.cuda()
    criterion1 = criterion1.cuda()

    # Optimizer and scheduler
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
    optimizer1 = get_optimizer1(model1)
    print(optimizer)
 
    # Checkpoint
    if os.path.exists(p['pretext_checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['pretext_checkpoint']), 'blue'))
        checkpoint = torch.load(p['pretext_checkpoint'], map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        model.cuda()
        start_epoch = checkpoint['epoch']

    else:
        print(colored('No checkpoint file at {}'.format(p['pretext_checkpoint']), 'blue'))
        start_epoch = 0
        model = model.cuda()
    
    # Training
    print(colored('Starting main loop', 'blue'))
    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' %(epoch, p['epochs']), 'yellow'))
        print(colored('-'*15, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        lr1 = adjust_learning_rate(p, optimizer1, epoch)

        print('Adjusted learning rate to {:.5f}'.format(lr))
        
        # Train
        print('Train ...')

        simclr_train(train_dataloader, model, model1,criterion,criterion1, optimizer, optimizer1,epoch)

        # Fill memory bank
        print('Fill memory bank for kNN...')
        fill_memory_bank(base_dataloader, model, memory_bank_base)

        # Evaluate (To monitor progress - Not for validation)
        print('Evaluate ...')
        top1 = contrastive_evaluate(val_dataloader, model, memory_bank_base)
        print('Result of kNN evaluation is %.2f' %(top1)) 
        
        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(), 
                    'epoch': epoch + 1}, p['pretext_checkpoint'])

    # Save final model
    torch.save(model.state_dict(), p['pretext_model'])

    # Mine the topk nearest neighbors at the very end (Train) 
    # These will be served as input to the SCAN loss.
    print(colored('Fill memory bank for mining the nearest neighbors (train) ...', 'blue'))
    fill_memory_bank(base_dataloader, model, memory_bank_base)
    # 训练集近邻准确率
    topk = 20
    print('Mine the nearest neighbors (Top-%d)' %(topk)) 
    indices, acc = memory_bank_base.mine_nearest_neighbors(topk)
    print('Accuracy of top-%d nearest neighbors on train set is %.2f' %(topk, 100*acc))
    np.save(p['topk_neighbors_train_path'], indices)   
 
    # Mine the topk nearest neighbors at the very end (Val)
    # These will be used for validation.
    #测试集近邻准确率
    print(colored('Fill memory bank for mining the nearest neighbors (val) ...', 'blue'))
    fill_memory_bank(val_dataloader, model, memory_bank_val)
    topk = 5
    print('Mine the nearest neighbors (Top-%d)' %(topk)) 
    indices, acc = memory_bank_val.mine_nearest_neighbors(topk)
    #画出近邻图像与自身(cifar数据集)
    """plt.figure()
    for i in range(1,topk+2):
        ax = plt.subplot(1,6,i)
        plt.imshow(val_dataloader.dataset.data[indices[0][i-1]])
        s = list(indices[0])
        class_number = val_dataloader.dataset.classes
        class_name = class_number[val_dataloader.dataset.targets[indices[0][i-1]]]
        plt.xlabel(s[i-1],fontsize=6)
        ax.set_title(class_name,fontsize=6)
        plt.xticks([])
        plt.yticks([])
    plt.savefig('/root/DeepClustermyproject/nnpictures/0-tok-5.png')
    plt.show()"""


    """#画出近邻图像与自身(stl数据集)
    plt.figure()
    for i in range(1,topk+2):
        ax = plt.subplot(1,6,i)
        plt.imshow(val_dataloader.dataset.data[indices[6][i-1]].transpose(1,2,0))
        s = list(indices[6])
        class_number = val_dataloader.dataset.classes
        class_name = class_number[val_dataloader.dataset.labels[indices[6][i-1]]]
        plt.xlabel(s[i-1],fontsize=6)
        ax.set_title(class_name,fontsize=6)
        plt.xticks([])
        plt.yticks([])
    plt.savefig('/root/DeepClustermyproject/nnpictures/6-tok-5.png')
    plt.show()"""

    print('Accuracy of top-%d nearest neighbors on val set is %.2f' %(topk, 100*acc))
    np.save(p['topk_neighbors_val_path'], indices)   

 
if __name__ == '__main__':
    main()
