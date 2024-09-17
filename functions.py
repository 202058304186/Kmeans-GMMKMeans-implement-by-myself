# -*- coding: gbk -*-
import math
import numpy as np

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

        
