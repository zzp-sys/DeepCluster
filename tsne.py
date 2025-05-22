from tqdm import tqdm

import pandas as pd
import numpy as np
from termcolor import colored
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from PIL import Image
from utils.common_config import get_train_transformations, get_val_transformations,\
								get_train_dataset, get_train_dataloader,\
								get_val_dataset, get_val_dataloader,\
								get_optimizer, get_model, get_criterion,\
								adjust_learning_rate
import argparse
FLAGS = argparse.ArgumentParser(description='SCAN Loss')
FLAGS.add_argument('--config_env', default='/root/DeepClustermyproject/configs/env.yml',help='Location of path config file')
FLAGS.add_argument('--config_exp', default='/root/DeepClustermyproject/configs/scan/scan_cifar10.yml',help='Location of experiments config file')
from utils.config import create_config
# 忽略烦人的红色提示
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from matplotlib.colors import ListedColormap
from torchvision.models.feature_extraction import create_feature_extractor

def main():
	args = FLAGS.parse_args()
	p = create_config(args.config_env, args.config_exp)
	# Data 数据下载在mypath.py中
	print(colored('Get dataset and dataloaders', 'blue'))
	train_transformations = get_train_transformations(p)
	val_transformations = get_val_transformations(p)
	"""train_dataset = get_train_dataset(p, train_transformations, 
											split='train', to_neighbors_dataset = True)"""
	"""train_dataset = get_train_dataset(p, train_transformations, 
											split='train', to_augmented_dataset=True) #simclr"""
	val_dataset = get_val_dataset(p, val_transformations, to_neighbors_dataset = True)
	#val_dataset = get_val_dataset(p, val_transformations) #simclr
	#train_dataloader = get_train_dataloader(p, train_dataset)
	val_dataloader = get_val_dataloader(p, val_dataset)
	print('Train transforms:', train_transformations)
	print('Validation transforms:', val_transformations)
	#print('Train samples %d - Val samples %d' %(len(train_dataset), len(val_dataset)))

	# Model
	print(colored('Get model', 'blue'))
	model = get_model(p, p['pretext_model']) #scan
	#model = get_model(p) #simclr
	print(model)
	model = torch.nn.DataParallel(model)
	model = model.cuda()
	

	# Optimizer
	print(colored('Get optimizer', 'blue'))
	optimizer = get_optimizer(p, model, p['update_cluster_head_only']) #scan
	#optimizer = get_optimizer(p, model) #simclr
	print(optimizer)

	# Checkpoint simclr
	"""if os.path.exists(p['pretext_checkpoint']):
		print(colored('Restart from checkpoint {}'.format(p['pretext_checkpoint']), 'blue'))
		checkpoint = torch.load(p['pretext_checkpoint'], map_location='cpu')
		optimizer.load_state_dict(checkpoint['optimizer'])
		model.load_state_dict(checkpoint['model'], strict=False)
		model.cuda()
		start_epoch = checkpoint['epoch']"""
	# Checkpoint scan
	if os.path.exists(p['scan_checkpoint']):
		print(colored('Restart from checkpoint {}'.format(p['scan_checkpoint']), 'blue'))
		checkpoint = torch.load(p['scan_checkpoint'], map_location='cpu')
		model.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])        
		start_epoch = checkpoint['epoch']
		best_loss = checkpoint['best_loss']
		best_loss_head = checkpoint['best_loss_head']

	features = []
	labels = []
	for i, batch in enumerate(val_dataloader):
		images = batch['anchor']
		#images = batch['image']
		targets = batch['target']
		label = targets.data.cpu().numpy()
		feature = model(images,flag=False,is_main=False).data.cpu().numpy() #scan
		#feature = model(images).data.cpu().numpy() #simclr

		features.extend(feature)
		labels.extend(label)
	features = np.array(features)
	labels = np.array(labels)
	tsne = TSNE(n_components=2, n_iter=20000)
	X_tsne_2d = tsne.fit_transform(features)
	print(X_tsne_2d.shape)

	# 调用函数，绘制图像
	fig = plot_embedding(X_tsne_2d, labels, title="TSNE")
	# 显示图像
	plt.show()

# 对样本进行预处理并画图
def plot_embedding(data, label, title):
	"""
	:param data:数据集
	:param label:样本标签
	:param title:图像标题
	:return:图像
	"""
	classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
	x_min, x_max = np.min(data, 0), np.max(data, 0)
	data = (data - x_min) / (x_max - x_min)		# 对数据进行归一化处理
	fig = plt.figure()		# 创建图形实例
	ax = plt.subplot(111)
	#fig, ax = plt.subplots(figsize=(12, 6))		# 创建子图，经过验证111正合适，尽量不要修改
	#cifar10, stl10
	colors = ['red', 'CornflowerBlue', 'green', 'purple', 'orange', 'gray', 'magenta', 'lime', 'pink', 'brown']
	#cifar100
	"""colors = [
	'red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'lime', 'pink', 'brown',
	'gray', 'olive', 'teal', 'navy', 'gold', 'indigo', 'violet', 'salmon', 'turquoise', 'lavender']"""
	unique_labels = np.unique(label)
	for i, u_label in enumerate(unique_labels):
		indices = np.where(label == u_label)
		plt.scatter(data[indices, 0], data[indices, 1], color=colors[i], marker='.', s=15, label=classes[u_label])
   
	#plt.legend(fontsize=10, markerscale=1, bbox_to_anchor=(1, 1))
	plt.xticks()
	plt.yticks()
	plt.title(title, fontsize=14)
	
	plt.savefig('/root/DeepClustermyproject/t-sne/cifar10_epoch10.pdf', bbox_inches='tight')
	# 返回值
	return fig
	

if __name__ == '__main__':
	main()