# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import time

import faiss
import numpy as np
from PIL import Image
from PIL import ImageFile
from scipy.sparse import csr_matrix, find
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

__all__ = ['PIC', 'Kmeans', 'cluster_assign', 'arrange_clustering']


def pil_loader(path):
    """Loads an image.
    Args:
        path (string): path to image file
    Returns:
        Image
    """
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB') #接受图像文件数据，返回RGB模式；


class ReassignedDataset(data.Dataset): #用于创建一个数据集，其中新的图像标签（伪标签）作为参数提供。
    """A dataset where the new images labels are given in argument.
    Args:
        image_indexes (list): list of data indexes
        pseudolabels (list): list of labels for each data
        dataset (list): list of tuples with paths to images
        transform (callable, optional): a function/transform that takes in
                                        an PIL image and returns a
                                        transformed version
    """

    def __init__(self, image_indexes, pseudolabels, dataset, transform=None):
        self.imgs = self.make_dataset(image_indexes, pseudolabels, dataset)
        self.transform = transform

    def make_dataset(self, image_indexes, pseudolabels, dataset): #创建数据集将图像索引和伪标签结合一起形成一个新的列表。
        label_to_idx = {label: idx for idx, label in enumerate(set(pseudolabels))}
        images = []
        for j, idx in enumerate(image_indexes):
            path = dataset[idx]
            #path = dataset[idx][0]
            pseudolabel = label_to_idx[pseudolabels[j]]
            images.append((path, pseudolabel))
        return images

    def __getitem__(self, index): #对于给定的索引和伪标签，加载图像，应用可选的数据转换，最终返回图像和伪标签。
        """
        Args:
            index (int): index of data
        Returns:
            tuple: (image, pseudolabel) where pseudolabel is the cluster of index datapoint
        """
        path, pseudolabel = self.imgs[index]
        img = pil_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pseudolabel

    def __len__(self):
        return len(self.imgs)


def preprocess_features(npdata, pca=256): #对特征数据预处理，pca降维，白化，L2归一化
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    npdata =  npdata.astype('float32')

    # Apply PCA-whitening with Faiss
    mat = faiss.PCAMatrix (ndim, pca, eigen_power=-0.5)
    mat.train(npdata)
    assert mat.is_trained
    npdata = mat.apply_py(npdata)

    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    return npdata


def make_graph(xb, nnn): #创建最近邻图
    """Builds a graph of nearest neighbors.
    Args:
        xb (np.array): data
        nnn (int): number of nearest neighbors
    Returns:
        list: for each data the list of ids to its nnn nearest neighbors
        list: for each data the list of distances to its nnn NN
    """
    #这个函数的作用是为输入的数据点构建一个最近邻图，以便在后续的操作中可以利用这个图结构，例如在图神经网络中进行节点嵌入。
    N, dim = xb.shape

    # we need only a StandardGpuResources per GPU
    res = faiss.StandardGpuResources() 
    # 通过 Faiss 库，首先在 GPU 上创建一个 L2 距离度量的平坦索引。
    #这个索引用于计算数据点之间的欧几里德距离。

    # L2
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = int(torch.cuda.device_count()) - 1
    index = faiss.GpuIndexFlatL2(res, dim, flat_config)
    index.add(xb) #add方法将数据点添加到索引中
    D, I = index.search(xb, nnn + 1) #I为索引，D为最近邻距离
    return I, D


def cluster_assign(images_lists, dataset):
    """Creates a dataset from clustering, with clusters as labels.
    Args:
        images_lists (list of list): for each cluster, the list of image indexes
                                    belonging to this cluster
        dataset (list): initial dataset
    Returns:
        ReassignedDataset(torch.utils.data.Dataset): a dataset with clusters as
                                                     labels
    """
    #这个函数的目的是根据聚类结果生成一个带有伪标签的数据集，以便在训练过程中将伪标签用作数据的标签，
    #从而进行半监督学习。函数还为图像数据应用了数据增强和标准化，以提升模型的泛化能力。

    assert images_lists is not None #使用assert确保传入的image_list不为空
    pseudolabels = [] #用来存储聚类结果中的伪标签
    image_indexes = [] #用于存储所有图像的索引
    for cluster, images in enumerate(images_lists):
        # 遍历每个簇，将属于每个簇的图像索引加入到 image_indexes 列表中，
        # 并将相应的伪标签（簇的标识）加入到 pseudolabels 列表中。这样，每个图像都被分配到了对应的伪标签。
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    t = transforms.Compose([transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize])

    return ReassignedDataset(image_indexes, pseudolabels, dataset, t)


def run_kmeans(x, nmb_clusters, verbose=False):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    #这个函数的作用是在一个 GPU 上运行 k-means 聚类，并返回每个数据点所属的最近邻簇的索引，
    #以及最终的聚类损失。这个聚类结果可以用于生成伪标签数据集，如前面提到的 cluster_assign 函数中

    n_data, d = x.shape #(512,10)

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)#d聚类维度，nmb_clusters聚类的簇的数量

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    clus.seed = np.random.randint(1234)

    clus.niter = 1000 #k-means迭代次数
    clus.max_points_per_centroid = 10000000 #设置质心的最大数据点数量，控制簇的大小
    res = faiss.StandardGpuResources() #创建 Faiss GPU 资源对象，用于在 GPU 上执行计算。
    flat_config = faiss.GpuIndexFlatConfig() #创建 Faiss GPU 平坦索引配置对象，用于定义 GPU 上的索引。
    flat_config.useFloat16 = False #在 GPU 上使用浮点数计算，而不是浮点16位数（half-precision）。这是为了确保计算精度。
    flat_config.device = 0 #指定要使用的 GPU 设备索引
    index = faiss.GpuIndexFlatL2(res, d, flat_config) #创建 Faiss GPU 平坦 L2 距离索引，用于 K-means 计算。

    # perform the training
    clus.train(x, index)
    #使用输入数据 x 对 K-means 模型进行训练，同时将训练结果存储在 index 中。
    _, I = index.search(x, 1) #I 包含了最近邻搜索的结果，即每个样本的最近邻的索引。
     #使用训练好的 K-means 模型 index 对输入数据 x 进行最近邻搜索，查找每个数据点最近的质心。
    #losses = faiss.vector_to_array(clus.obj) #AttributeError: 'Clustering' object has no attribute 'obj
    stats = clus.iteration_stats
    losses = np.array([stats.at(i).obj for i in range(stats.size())])
    #将 K-means 训练过程中的损失值转换为一个 NumPy 数组。这些损失值可以用于监控训练过程中的收敛情况。
    """if verbose:
        print('k-means loss evolution: {0}'.format(losses))"""

    #返回一个列表，其中包含每个数据点所属的最近质心的索引（聚类结果）以及 K-means 训练的最终损失值。
    return [int(n[0]) for n in I], losses[-1]


def arrange_clustering(images_lists):
    #重新排列聚类结果，以便将图像的伪标签与原始数据集中的图像顺序对应起来。
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))
    indexes = np.argsort(image_indexes)
    return np.asarray(pseudolabels)[indexes]


class Kmeans(object):
    #这个类的作用是执行 k-means 聚类，
    #并提供了一个方法 cluster 用于将数据进行聚类并返回相应的聚类损失。
    #类的属性 images_lists 保存了每个簇的图像索引列表，可以用于生成伪标签数据集。
    def __init__(self, k):
        self.k = k

    def cluster(self, data, verbose=False):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        #xb = preprocess_features(data)
        xb = data

        # cluster the data
        I, loss = run_kmeans(xb, self.k, verbose)
        self.images_lists = [[] for i in range(self.k)]
        for i in range(len(data)):
            self.images_lists[I[i]].append(i)

        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

        return loss


def make_adjacencyW(I, D, sigma):
    #这个函数的作用是使用高斯核为图网络构建邻接矩阵，
    #其中权重由与每个顶点连接的邻居顶点的 L2 距离以及给定的高斯核带宽决定。
    #这种邻接矩阵常用于图神经网络中，用于对图数据进行表示。
    """Create adjacency matrix with a Gaussian kernel.
    Args:
        I (numpy array): for each vertex the ids to its nnn linked vertices
                  + first column of identity.
        D (numpy array): for each data the l2 distances to its nnn linked vertices
                  + first column of zeros.
        sigma (float): Bandwidth of the Gaussian kernel.

    Returns:
        csr_matrix: affinity matrix of the graph.
    """
    V, k = I.shape
    k = k - 1
    indices = np.reshape(np.delete(I, 0, 1), (1, -1))
    indptr = np.multiply(k, np.arange(V + 1))

    def exp_ker(d):
        return np.exp(-d / sigma**2)

    exp_ker = np.vectorize(exp_ker)
    res_D = exp_ker(D)
    data = np.reshape(np.delete(res_D, 0, 1), (1, -1))
    adj_matrix = csr_matrix((data[0], indices[0], indptr), shape=(V, V))
    return adj_matrix


def run_pic(I, D, sigma, alpha):
    #这个函数的作用是运行 PIC 算法，将图数据嵌入到低维向量空间中，并执行聚类操作。最终返回每个节点所属的聚类标识。
    """Run PIC algorithm"""
    a = make_adjacencyW(I, D, sigma)
    graph = a + a.transpose()
    cgraph = graph
    nim = graph.shape[0]

    W = graph
    t0 = time.time()

    v0 = np.ones(nim) / nim

    # power iterations
    v = v0.astype('float32')

    t0 = time.time()
    dt = 0
    for i in range(200):
        vnext = np.zeros(nim, dtype='float32')

        vnext = vnext + W.transpose().dot(v)

        vnext = alpha * vnext + (1 - alpha) / nim
        # L1 normalize
        vnext /= vnext.sum()
        v = vnext

        if i == 200 - 1:
            clust = find_maxima_cluster(W, v)

    return [int(i) for i in clust]


def find_maxima_cluster(W, v):
    #这个函数的作用是在 PIC 算法中找到最大值簇，即根据图的邻接关系和节点向量的差异，
    #将节点分配到合适的簇中。函数返回每个节点所属的簇标识。

    n, m = W.shape
    assert (n == m)
    assign = np.zeros(n)
    # for each node
    pointers = list(range(n))
    for i in range(n):
        best_vi = 0
        l0 = W.indptr[i]
        l1 = W.indptr[i + 1]
        for l in range(l0, l1):
            j = W.indices[l]
            vi = W.data[l] * (v[j] - v[i])
            if vi > best_vi:
                best_vi = vi
                pointers[i] = j
    n_clus = 0
    cluster_ids = -1 * np.ones(n)
    for i in range(n):
        if pointers[i] == i:
            cluster_ids[i] = n_clus
            n_clus = n_clus + 1
    for i in range(n):
        # go from pointers to pointers starting from i until reached a local optim
        current_node = i
        while pointers[current_node] != current_node:
            current_node = pointers[current_node]

        assign[i] = cluster_ids[current_node]
        assert (assign[i] >= 0)
    return assign


class PIC(object):
    """Class to perform Power Iteration Clustering on a graph of nearest neighbors.
        Args:
            args: for consistency with k-means init
            sigma (float): bandwidth of the Gaussian kernel (default 0.2)
            nnn (int): number of nearest neighbors (default 5)
            alpha (float): parameter in PIC (default 0.001)
            distribute_singletons (bool): If True, reassign each singleton to
                                      the cluster of its closest non
                                      singleton nearest neighbors (up to nnn
                                      nearest neighbors).
        Attributes:
            images_lists (list of list): for each cluster, the list of image indexes
                                         belonging to this cluster
    """

    def __init__(self, args=None, sigma=0.2, nnn=5, alpha=0.001, distribute_singletons=True):
        self.sigma = sigma
        self.alpha = alpha
        self.nnn = nnn
        self.distribute_singletons = distribute_singletons

    def cluster(self, data, verbose=False):
        end = time.time()

        # preprocess the data 数据预处理
        xb = preprocess_features(data)

        # construct nnn graph 构建最近邻图
        I, D = make_graph(xb, self.nnn)

        # run PIC
        #运行 PIC 算法，并获取每个数据点的聚类标识 clust。
        clust = run_pic(I, D, self.sigma, self.alpha)

        images_lists = {}
        for h in set(clust):
            images_lists[h] = []
        for data, c in enumerate(clust):
            images_lists[c].append(data)

        # allocate singletons to clusters of their closest NN not singleton
        if self.distribute_singletons:
            clust_NN = {}
            for i in images_lists:
                # if singleton
                if len(images_lists[i]) == 1:
                    s = images_lists[i][0]
                    # for NN
                    for n in I[s, 1:]:
                        # if NN is not a singleton
                        if not len(images_lists[clust[n]]) == 1:
                            clust_NN[s] = n
                            break
            for s in clust_NN:
                del images_lists[clust[s]]
                clust[s] = clust[clust_NN[s]]
                images_lists[clust[s]].append(s)

        self.images_lists = []
        for c in images_lists:
            self.images_lists.append(images_lists[c])

        if verbose:
            print('pic time: {0:.0f} s'.format(time.time() - end))
        return 0
