# -*- coding: gbk -*-
import os
import math
import config
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def readData(path, type = None)->np.array:
    """
    read dataset and return the np.array. only can do some csv excel text.
    """
    assert type is not None, "the type shouldn't be the None."
    if type == "csv":
        return np.array(pd.read_csv(path))
    if type == "excel":
        return np.array(pd.read_excel(path))


def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step
 
    return hls_colors



def plot_scatter(x:np.array, y:np.array, color = None, title = ""):
    plt.figure()
    plt.scatter(x, y)
    plt.title(title)
    plt.show()
    plt.close()

def plot_KMean2DScatter(data:list, centers:np.array, title = "", colors = []):
    plt.figure()
    k = len(data)
    if len(colors) == 0:
        colors = get_n_hls_colors(k)
    for i in range(k):
        plt.scatter(data[i][:, 0], data[i][:, 1], c = colors[i], label = f'{i}')
        plt.scatter(centers[i, 0], centers[i, 1], s = 100, c = "yellow", marker='x')
    plt.title(title)
    plt.legend()
    plt.show()
    plt.close()


def plot_KMean3DScatter(data:list, centers:np.array, title = "", colors = []):
    plt.figure()
    ax1 = plt.axes(projection='3d')
    k = len(data)
    if len(colors) == 0:
        colors = get_n_hls_colors(k)
    for i in range(k):
        ax1.scatter(data[i][:, 0], data[i][:, 1], data[i][:, 2], c = colors[i], label = f'{i}')
        ax1.scatter(centers[i, 0], centers[i, 1], centers[i, 2],  s = 100, c = "yellow", marker='x')
    plt.title(title)
    plt.legend()
    plt.show()
    plt.close()


def KMean(data:np.array, k, Threshold = 0.0, MaxTimes = 10000000):
    assert data.shape[0] >= k, "The length of data should bigger than k."

    init_k_index = np.random.randint(0, data.shape[0], k)
    types = np.zeros(data.shape[0], dtype = "int")

    centers = data[init_k_index, :]
    isChange = True

    dists = np.zeros(k, dtype = "float")
    times = 0
    while isChange and times < MaxTimes:
        for mindex, temp in enumerate(data):
            # computing distance between every sample and every center
            dists = np.sqrt(np.sum((temp - centers) ** 2, axis = 1))    #(1 2) - (3 2) -> (3 2) ** 2 -> (3 2) -> sum((3 2) axis = 1) -> (3 1)
            cPos = np.argmin(dists)                                     # divide the sample into the nearest cluster
            types[mindex] = cPos

        # update the center of each cluster
        for cindex in range(k):
            if data[types == cindex, :].shape[0] > 0:
                newCenter = np.mean(data[types == cindex, :], axis = 0) #(n 2) -> (1,2)
                isChange = np.any(np.fabs(newCenter - centers[cindex]) >= Threshold)
                centers[cindex] = newCenter
        times += 1
    # return dataset divided by cluster and their centers
    C = [data[types == i, :] for i in range(k)]
    return C, centers



def GMM_KMeans(data:np.array, k, threshold = 0.0, MaxTimes = 1000000, kmeanstimes = 3):
    """Gaussian mixture distribution clustering, EM ideas"""
    def kernel(x, n, mean, cov):
        vector1D = (x - mean).reshape(1, -1)

        det_cov = np.linalg.det(cov)
        # Prevents the determinant of the covariance matrix from approaching 0, leading to unstable values
        if det_cov == 0:
            det_cov += 1e-6

        first       = 1 / (np.sqrt((2 * math.pi) ** n * det_cov))
        exp_value   = -0.5 * (vector1D @ np.linalg.inv(cov) @ vector1D.T)
        second      = np.exp(exp_value)
        return  first * second
    
    notDone, times = True, 0
    alpha   = np.array([1./k for i in range(k)])    # alpha is the weights of each Gaussian distribution.
    _, mean = KMean(data, k, 0.0, kmeanstimes)      #only need centers from KMean
    cov     = np.array([np.diag([0.1 for i in range(data.shape[1])]) for i in range(k)]) 
    gamma   = np.zeros((data.shape[0], k), dtype="float")   #gamma is the posterior probability matrix. 
                                                            #gamma[i, j] denotes the posterior probability that the sample belongs to the jth Gaussian distribution
    old_mean = np.copy(mean)
    while notDone and times < MaxTimes:
        # E Step
        for sindex, sample in enumerate(data):
            posibilityOfXi = 0.
            for j in range(k):
                posibilityOfXi += alpha[j] * kernel(sample, data.shape[1], mean[j], cov[j])
            for j in range(k):
                gamma[sindex, j] = alpha[j] * kernel(sample, data.shape[1], mean[j], cov[j]) / posibilityOfXi

        # M Step
        for i in range(k):
            gamma_sum, weight_gamma_each_k, cov_top = 0., 0., 0.
            for j in range(data.shape[0]):
                gamma_sum           += gamma[j, i]
                weight_gamma_each_k += gamma[j, i] * data[j]

            mean[i]     = weight_gamma_each_k / gamma_sum
            alpha[i]    = gamma_sum / data.shape[0]

            for j in range(data.shape[0]):
                vector1D = (data[j,:] - mean[i]).reshape(1, -1)
                cov_top += gamma[j, i] * np.dot((vector1D).T, (vector1D))

            cov[i] = cov_top / gamma_sum + np.eye(data.shape[1]) * 1e-6

            # Preventing a cluster from having too little weight
            if alpha[i] < 1e-3:
                mean[i] = data[np.random.randint(0, data.shape[0])]
                cov[i]  = np.eye(data.shape[1]) * 0.1

            # Checking for convergence
            if np.linalg.norm(mean - old_mean) < threshold:
                break
            old_mean = np.copy(mean)

        times += 1
    
    types = np.argmax(gamma, axis = 1)
    centers = np.empty((k, data.shape[1]), dtype = "float")
    
    for cindex in range(k):
        centers[cindex] = np.mean(data[types == cindex, :], axis = 0) #(n 2) -> (1,2)

    return [data[types == i, :] for i in range(k)], centers 

        
