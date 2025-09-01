import numpy as np
#from numba import njit
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision
import torchvision.transforms.functional as TF

def normalize(data):
    dimension = data.shape[2]
    theta_grid_len1 = data.shape[0]
    samples_per_theta1 = data.shape[1]
    data = data.reshape(theta_grid_len1*samples_per_theta1, dimension)###

    mu = np.zeros(dimension)
    std = np.zeros(dimension)
    for i in range(data.shape[1]):
        mu[i] = np.mean(data[:,i])
        std[i] = np.std(data[:,i])
    for i in range(data.shape[1]):
        data[:,i] = (data[:,i]-mu[i])/std[i]
    data = data.reshape(theta_grid_len1,samples_per_theta1, dimension)###
    return data

def get_LFI(data, delta, cutoff):
    mu = np.mean(data, axis=1)
    dmu = (mu[2]-mu[0])/(2*delta)
    cov = (np.cov(data[0].T)+np.cov(data[2].T))/2
    if data.shape[2] > 1:
        u, s, vh = np.linalg.svd(cov, full_matrices=True)
        count = 0
        s_inv = np.zeros(shape=s.shape[0])
        for i in range(s.shape[0]):
            if np.abs(s[i])<cutoff:
                count += 1
                s[i] = 0.
            else:
                s_inv[i] = 1/s[i]
        if count > 0:
            print("components ignored: ", count)
        lfi = dmu.T@vh.T@np.diag(s_inv)@u.T@dmu  
    elif data.shape[2] == 1:
        lfi = dmu**2/cov
    else:
        print("wrong shape of data")
    return lfi
    
def activation(x):
    return x*(np.heaviside(x, 0)+0.7*np.heaviside(-x, 0))

def transform_data(r_up, batch, noise_factor):
    transformed_data = activation(np.einsum('ij, abj -> abi', r_up, batch))
   # transformed_data = n_normalize(transformed_data)    #######################################
    noise = noise_factor*np.random.uniform(0.,1., size=transformed_data.shape)
    transformed_data += noise
    return transformed_data
def ReLU(x):
    return x*np.heaviside(x, 0)

##################
def get_FI(data, delta, lim, step_size, fluctuation_threshold, constant_threshold, noise_factor, biasedLFI=False):
    print("calculating FI")
    dim = data.shape[2]
    temp = []
    lfi0 = get_LFI(data, delta, cutoff=1e-9)
    for up in range(0,lim,step_size):
        print("up: ", up)
        r_up = np.random.normal(0.,1., size=(dim+up, dim)) # random projection
        transformed_data = activation(np.einsum('ij, abj -> abi', r_up, data)) # non-linear activation
        a = transformed_data.shape[0]
        b = transformed_data.shape[1]
        c = transformed_data.shape[2]
        noise = noise_factor*np.random.normal(0.,1., size=transformed_data.shape) # regularize by adding noise
        transformed_data += noise
        ###############
        lfi = get_LFI(transformed_data, delta, cutoff=1e-9)
        T = transformed_data.shape[1]
        N = transformed_data.shape[2]
        if biasedLFI == True: # compensate for the bias of the estimated LFI
            lfi = lfi - 2*N/(T*(2*delta)**2)
        temp.append(lfi)
        #------------------ convergence criterion: this is quite generic so here should be some options
        if len(temp) == 3:
            first3 = temp[-3:]
            fluctuation = (max(first3) - min(first3))
            mean_lfi = (max(first3) + min(first3))/2
            first_fluctuation = fluctuation
            if np.abs(first_fluctuation/temp[0]) < constant_threshold: # if the LFI doesn't increase initially
                print("LFI didn't increase after the 3rd iteration: ", transformed_data.shape[2])
                return mean_lfi#[0] # , temp
        if len(temp) > 3:
            last3 = temp[-3:]
            fluctuation = (max(last3) - min(last3))
            mean_lfi = (max(last3) + min(last3))/2
            last_fluctuation = fluctuation 
            if np.abs(last_fluctuation/first_fluctuation) < fluctuation_threshold:
                print("finished regularly, max dim: ", transformed_data.shape[2])
                return mean_lfi
                break
    print("Warning: FI calculation did not converge! Increase the parameter lim. Note that large lim might require more data.")
    return max(temp)
###################################################

def get_FI_curve(data, delta, lim, step_size, fluctuation_threshold, constant_threshold, noise_factor, biasedLFI=False):
    dim = data.shape[2]
    temp = []
    lfi0 = get_LFI(data, delta, cutoff=1e-9)
    for up in range(0,lim,step_size):
        r_up = np.random.normal(0.,1., size=(dim+up, dim))
        transformed_data = activation(np.einsum('ij, abj -> abi', r_up, data))
        a = transformed_data.shape[0]
        b = transformed_data.shape[1]
        c = transformed_data.shape[2]
        noise = noise_factor*np.random.normal(0.,1., size=transformed_data.shape)
        transformed_data += noise
        ###############
        lfi = get_LFI(transformed_data, delta, cutoff=1e-9)     
        T = transformed_data.shape[1]
        N = transformed_data.shape[2]
        if biasedLFI == True:
            lfi = lfi - 2*N/(T*(2*delta)**2)
        print("lfi = ", lfi)
        temp.append(lfi)
    return temp