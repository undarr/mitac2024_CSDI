import pickle

import os
import re
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# 1 attributes which contains enough non-values
attributes = ['price']

def linegen(slope=0.5, intecept=100, off=0.1, seed=0, M=20000):
    # Generate 1000 straight lines with slopes between 0.5 and 0.7, and intercepts between 90 and 110
    np.random.seed(seed)  # seed for ground truth choice
    num_lines = M
    num_points = 300
    slopes = np.random.uniform(slope, slope+0.001, num_lines)
    intercepts = np.random.uniform(intecept*(1-off), intecept*(1+off), num_lines)
    x = np.linspace(0, num_points, num_points)
    y = np.zeros((num_lines, num_points))
    for i in range(num_lines):
        y[i] = slopes[i] * x + intercepts[i]
    return(y.T)

def gbm(mu=0.1, n=300, T=1, M=20000, S0=100, sigma=0.3, seed=0):
    np.random.seed(seed)
    n-=1
    dt = T/n
    St = np.exp((mu - sigma ** 2 / 2) * dt+ sigma * np.random.normal(0, np.sqrt(dt), size=(M,n)).T)
    St = np.vstack([np.ones(M), St])
    St = S0 * St.cumprod(axis=0)
    return(St)

def lgbm(mu=0.1, n=300, T=1, M=20000, S0=100, sigma=0.3, seed=0):
    return(np.log(gbm(mu, n, T, M, S0, sigma, seed)))


class Stock_Dataset(Dataset):
    def __init__(self, eval_length=300, use_index_list=None, missing_ratio=0.1, seed=0, stimcount=20000, ransample=False, emptytest=False, drift=0.1, sigma=0.3, line=True):

        self.data=gbm(mu=drift, n=eval_length, T=1, M=stimcount, S0=100, sigma=sigma, seed=seed)
        if (line):
            self.data=linegen(slope=drift, intecept=100, off=0.1, seed=0, M=stimcount)
            self.data=lgbm(mu=drift, n=eval_length, T=1, M=stimcount, S0=100, sigma=sigma, seed=seed)
        self.eval_length = eval_length

        self.observed_values = torch.tensor(self.data.T).unsqueeze(2)
        self.observed_masks = torch.ones_like(self.observed_values)
        self.gt_masks = self.observed_masks.clone()

        if ransample: # randomly set some percentage as ground-truth
            self.masks = self.observed_masks.reshape(-1).clone()
            self.obs_indices = np.where(self.masks)[0].tolist()
            self.miss_indices = np.random.choice(
                self.obs_indices, (int)(len(self.obs_indices) * missing_ratio), replace=False
            )
            self.masks[self.miss_indices] = False
            self.gt_masks = self.masks.reshape(self.observed_masks.shape)
        else:
            self.gt_masks[:, -1*math.floor(eval_length*missing_ratio):] = 0 #set the last missing_ratio% of the data to 0

        if emptytest:
            self.gt_masks = torch.zeros_like(self.observed_values)

        #print(self.observed_values.size()) #(1000, 300, 1)
        #print(self.observed_masks.size()) #(1000, 300, 1) #all ones
        #print(self.gt_masks.size()) #(1000, 300, 1) (1000, 300, 1) first 240 ones, then 60 zeros
        #print(self.observed_masks[0].squeeze())
        #print(ransample, self.gt_masks[0].squeeze())

        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
        }
        return s

    def getdata(self):
        print(self.observed_values[self.use_index_list].size())
        return(self.observed_values[self.use_index_list])


    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(seed=1, nfold=0, batch_size=16, missing_ratio=0.2, timelength=300, stimcount=20000, idea=1, drift=0.1, sigma=0.3, line=False):
    print(missing_ratio,timelength,stimcount,drift,sigma,idea)

    # only to obtain total length of dataset
    dataset = Stock_Dataset(missing_ratio=missing_ratio, seed=seed, eval_length=timelength, stimcount=stimcount)
    indlist = np.arange(len(dataset))

    np.random.seed(seed)
    np.random.shuffle(indlist)

    # 5-fold test
    start = (int)(nfold * 0.2 * len(dataset))
    end = (int)((nfold + 1) * 0.2 * len(dataset))
    test_index = indlist[start:end]
    remain_index = np.delete(indlist, np.arange(start, end))

    np.random.seed(seed)
    np.random.shuffle(remain_index)
    num_train = (int)(len(dataset) * 0.75)
    train_index = remain_index[:num_train]
    valid_index = remain_index[num_train:]

    if idea==1:
        ransample=False
        emptytest=False
    elif idea==2:
        ransample=True
        emptytest=True

    dataset = Stock_Dataset(
        use_index_list=train_index, missing_ratio=missing_ratio, seed=seed, eval_length=timelength, stimcount=stimcount, ransample=ransample, drift=drift, sigma=sigma, line=line
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = Stock_Dataset(
        use_index_list=valid_index, missing_ratio=missing_ratio, seed=seed, eval_length=timelength, stimcount=stimcount, ransample=ransample, drift=drift, sigma=sigma, line=line
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)
    test_dataset = Stock_Dataset(
        use_index_list=test_index, missing_ratio=missing_ratio, seed=seed, eval_length=timelength, stimcount=stimcount, ransample=ransample, emptytest=emptytest, drift=drift, sigma=sigma, line=line
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
    return dataset.getdata(), train_loader, valid_loader, test_loader