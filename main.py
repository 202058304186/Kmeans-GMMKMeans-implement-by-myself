# -*- coding: gbk -*-
import sys
import config
from utils import *
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

w4 = readData(config.W4_PATH, "csv")
#plot_scatter(w4[:, 0], w4[:, 1], "Distribution of W4")

datas, centers = KMean(w4, 3, 1e-10, 1)
plot_KMean2DScatter(datas, centers, "1", ['g', 'r', 'b'])

datas, centers = GMM_KMeans(w4, 3, 1e-10, 5, 3)
plot_KMean2DScatter(datas, centers, "5", ['g', 'r', 'b'])

# 生成数据集，包含3个簇
X, y_true = make_blobs(n_samples=1000, n_features=3, centers=5, cluster_std=[1.8, 2.2, 2.0, 2.5, 3.0])
datas, centers = GMM_KMeans(X, 5, 1e-5, 50, 10)
plot_KMean3DScatter(datas, centers, "2", ['g', 'r', 'b', 'black', 'pink'])
