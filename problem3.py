## author: Yike Guo

import numpy as np
import copy
import librosa
import os
import re
import random
from num2words import num2words
from mfcc import *
from problem2 import *
import warnings
warnings.filterwarnings('ignore')

num_filter = 40
num_states=5
num_templ=5
epsilon = 0.05
digits = ['zero','one','two','three','four','five','six','seven','eight','nine']

 
## logic: cluster -> state -> hmm

## define cluster as a gaussian
class Cluster():
    def __init__(self,matrix,weight):
        '''
        Parameters
        ----------
        weight: weight of cluster in corresponding state
        mean and cov: mean and covariance of a cluster (gaussian)
        '''
        self.matrix = matrix
        self.weight = weight
        
    @property
    def mean_cov(self):
        mean = np.mean(self.matrix,axis=0)
        cov = np.cov(self.matrix.T)
        
        return mean,cov
    
## define one state as a combination of clusters
class State():
    def __init__(self,clusters):
        '''
        Parameters
        ----------
        clusters: list of cluster object
        num_clusters: number of gaussians in one state
        matrix: matrix of a state (not a gaussian)
        mean and cov: mean and covariance of a state (not a gaussian)
        '''
        self.clusters = clusters
    
    @property
    def num_clusters(self):
        return len(self.clusters)

    ## obtain state_matrix as combination of cluster_matrix
    @property
    def matrix(self):
        state_matrix = np.zeros((1,39))
        for i in range(len(self.clusters)):
            cluster_matrix = self.clusters[i].matrix
            state_matrix = np.vstack((state_matrix,cluster_matrix))
        return state_matrix[1:,:]

    ## obtain mean, cov of state_matrix
    @property
    def mean_cov(self):
        state_matrix = self.matrix
        mean = np.mean(state_matrix,axis=0)
        cov = np.cov(state_matrix.T)
        return mean,cov
    
    ## obtain list of means, covs of clusters of one state
    @property
    def cluster_means(self):
        means = []
         
        for i in range(len(self.clusters)):
            mean,cov = self.clusters[i].mean_cov
            means.append(mean)
             
        return means
    
    @property
    def cluster_covs(self):
         
        covs = []
        for i in range(len(self.clusters)):
            mean,cov = self.clusters[i].mean_cov
             
            covs.append(cov)
       
        return covs
    
    
    ## obtain list of weights of clusters of one state
    @property
    def cluster_weights(self):
        weights = []
        for i in range(len(self.clusters)):
            weight = self.clusters[i].weight
            weights.append(weight)
        return weights
    
    
## define HMM model as a combination of states
## same as problem 2
class HMM():
    def __init__(self,states,start_state=0):
        '''
        Parameters
        ----------
        states: a list of state objects 
        start_state: default is 0
        '''
        #super().__init__()
        self.states = states
        self.start_state = start_state

    @property
    def transition_matrix(self):
        '''
        Return transition matrix from states
        '''
        
        transition_matrix = np.zeros((num_states,num_states))
        for i in range(num_states-1):
            nframes = self.states[i].matrix.shape[0]
            transition_matrix[i][i+1] = num_templ / nframes
            transition_matrix[i][i] = (nframes - num_templ)/nframes
        transition_matrix[num_states-1][num_states-1] = 1 
        return transition_matrix

## key steps in kmeans
## labeling vectors by distance, 
## assign vectors to each cluster and recompute cluster
def kmeans_labeling_assigning_update(state_matrix,means,covs,weights,n):
    '''
    Return a state object as list of updated clusters using kmeans
    
    Parameters
    ----------
    state_matrix: matrix in corresponding state
    means,covs,weights: means,covs,weights of clusters in one state
    n: desired number of gaussians
    '''
    
    labels = []
    ## label each vector by calculating distance
    for i in range(state_matrix.shape[0]):
        dists = []
        for k in range(len(means)):
            dist = get_distance(state_matrix[i],means[k],covs[k],weights[k])
            dists.append(dist)
                
        label = np.argmin(dists) # which cluster state_matrix[i] belongs to
        labels.append(label)
        
    labels = np.array(labels) # list -> array then use np.where!!!
        
    ## find cluster according to label  
    clusters = []
    for j in range(n):
        cluster_matrix = state_matrix[np.where(labels==j)]
        cluster_weight = cluster_matrix.shape[0] / state_matrix.shape[0]
        cluster = Cluster(cluster_matrix,cluster_weight)
        clusters.append(cluster)
        
    ## update state
    state = State(clusters)
    return state,labels

# for one state in hmm, split 1 gaussian to n gaussians
# 1 -> 2; 2->4
def kmeans(state,n):
    '''
    Return state as list of recomputed clusters
    
    Parameters
    ----------
    state: a single state in hmm
    n: number of clusters after kmeans
    '''
    
    ## initialization
    
    ## single gaussian -> 2 gaussians
    if n == 2: 
        mean,cov = state.mean_cov
        means = [mean-epsilon,mean+epsilon]
        covs = [cov] * n   
    ## 2 gaussians -> 4 gaussians
    else:
        means = state.cluster_means
        covs = state.cluster_covs
        means = [means[0]-epsilon,means[0],means[1],means[1]-epsilon]
        covs = [covs[0],covs[0],covs[1],covs[1]]  
    
    weights = [1/n] * n   
    
    ## kmeans
    converge = True
    means_convergence = [np.mean(np.sum(means))]
    cnt = 0
    
    while converge:
        state_matrix = state.matrix
        
        ## update state and means,covs, weights of clusters in a state
        state,labels = kmeans_labeling_assigning_update(state_matrix, means, covs, weights,n)
        means = state.cluster_means
        covs = state.cluster_covs
        weights = state.cluster_weights
        
        
        ## check for convergence
        cnt += 1
        means_convergence.append(np.mean(np.sum(means)))
        if np.abs((means_convergence[cnt] - means_convergence[cnt-1])/means_convergence[cnt-1]) < 0.01:
            converge = False
        
        ## add restriction for n == 4, otherwise it will not converge due to the loss of some cluster
        if n == 4:
            unique_label,count_label = np.unique(labels,return_counts=True)
            min_count = min(count_label)
            if len(unique_label) <= 4:
                converge = False
    
        #print(cnt)
        
    return state        

## calculate distance between vector x and one cluster of mixture gaussians
def get_distance(x,mean,cov,weight):
    '''
    Return distance between vector x and a cluster centroid
    
    Parameters
    ----------
    x: frame vector from mfcc features
    mean,cov,weight: from one cluster
    '''
    
    cov_diag = np.diagonal(cov)
    dist = 0.5 * np.sum(np.log(2*np.pi*cov_diag)) + 0.5 * np.sum(np.square(x-mean)/cov_diag) - np.log(weight)
    return dist

## calculate node cost as negative log likelihood (p.134)
def get_node_cost(x,state):
    '''
    Return node cost
    modified from problem2 to add weight: 
    node cost = -np.log(sum of (weight * likelihood))
    
    Parameters
    ----------
    x: frame vector from mfcc features
    state: combination of clusters
    '''
    node_cost = 0
    for i in range(state.num_clusters):
        mean,cov = state.clusters[i].mean_cov
        weight = state.clusters[i].weight
        cov_diag = np.diagonal(cov)
        node_cost += 0.5 * np.sum(np.log(2*np.pi*cov_diag)) + 0.5 * np.sum(np.square(x-mean)/cov_diag) - np.log(weight)
    
    return node_cost


## calculate edge cost as negative log transition prob
## same as problem 2
def get_edge_cost(transition_matrix):
    zero_index = np.where(transition_matrix == 0)
    transition_matrix[zero_index] = 1/(np.iinfo(np.int32).max)
    edge_cost = -np.log(transition_matrix)
    return edge_cost

## update gaussian mixture model & hmm
def update_gmm_hmm(templ_files,states_ids,hmm,n):
    '''
    Return a updated hmm object with updated gmm after alignment
    
    Parameters
    ----------
    templ_files: list of template file names
    states_ids: list of state_ids
    hmm: current hmm model
    n: desired number of gaussians
    '''
    
    states = []
    for i in range(num_states):
        ## obtain state_matrix from state_ids
        state_matrix = np.zeros((1,39))
        for k in range(len(templ_files)):
            templ = mfcc_features(templ_files[k],num_filter)
            state_ids = np.array(states_ids[k]) #list->array
            state_matrix = np.vstack((state_matrix,templ[np.where(state_ids==i)]))
                
        ## assign, labeling, update
        means= hmm.states[i].cluster_means
        covs = hmm.states[i].cluster_covs
        weights = hmm.states[i].cluster_weights
        state,_ = kmeans_labeling_assigning_update(state_matrix, means, covs, weights,n)
        states.append(state)
        
    hmm = HMM(states)
    return hmm

## same as problem 2
def get_hmm_mean(hmm):
    means = 0
    for i in range(num_states):
        mean,cov = hmm.states[i].mean_cov
        means += mean
    return np.mean(mean)


## split n/2 gaussians to n gaussians and do alignment
def split_align(templ_files,hmm,n):
    '''
    Return hmm model after alignment
    
    Parameters
    ----------
    templ_files: list of template filenames
    hmm: current hmm model
    n: desired number of gaussians in one state
    '''
    
    ## split n/2 gaussin -> n gaussians
    states = []
    for i in range(num_states):
        state = kmeans(hmm.states[i],n)
        states.append(state)
    hmm = HMM(states)
    
    ## alignment until convergence
    converge = True
    mean = get_hmm_mean(hmm)
    means = [mean]
    cnt = 0
    while converge:
        states_ids = []
        for k in range(len(templ_files)):
            test = mfcc_features(templ_files[k],num_filter)
            ## alignment
            state_ids,_ = dtw_alignment(test,hmm)
            states_ids.append(state_ids)
             
        ## update gaussian mixutre model & hmm
        hmm = update_gmm_hmm(templ_files, states_ids, hmm,n)
        
        ## check for converge
        mean = get_hmm_mean(hmm)
        means.append(mean)
        cnt += 1
        if np.abs((means[cnt] - means[cnt-1])/means[cnt-1]) < 0.01:
            converge = False
    
    return hmm

## training
def training(templ_files):
    '''
    Return hmm, with 5 states, each state has four gaussians
    '''
    ## obtain pretrained one gaussian hmm
    hmm = training_single_gaussian(templ_files)
    
    ## split 1 gaussian -> 2 gaussians & alignment
    hmm = split_align(templ_files,hmm,n=2)
    
    ## split 2 gaussians -> 4 gaussians & alignment
    hmm = split_align(templ_files,hmm,n=4)

    return hmm

## use different training function
## modified from problem 2
def get_hmm_digits(filenames):
    '''
    Return a list of 10 hmm models associated with 0-9 digits
    
    Parameters
    ----------
    filenames: all filenames (10 files for 0-9 digits each)
    '''
    ## training hmm model for each digit
    hmm_digits = [] # list to store 10 hmm models associated with 10 digits
    
    # order: 0 -> 9
    for digit in digits:
        
        digit_files = []
        for filename in filenames:
            if digit in filename:
                digit_files.append(filename)
        
        random.shuffle(digit_files)
        
        ## choose 5 files of each digit as templates
        templ_files = digit_files[:10]
        hmm = training(templ_files)
        hmm_digits.append(hmm)
        
        ## save hmm
        
        with open('hmms.txt','a') as f:
            f.write('hmm model for digit ' +str(digit))
            
        np.savetxt('hmms.txt',hmm.transition_matrix)
             
        for i in range(len(hmm.states)):  
            
            np.savetxt('hmms.txt',np.array(hmm.states[i].cluster_weights))
            np.savetxt('hmms.txt',np.array(hmm.states[i].cluster_means))
            with open('hmms.txt','a') as f:
                f.write(str(hmm.states[i].cluster_covs))
                 
        with open('hmms.txt','a') as f:
            f.write('-'*50)
            
    return hmm_digits

## main
def main_gmm():
    pathname = "recds_yg/"
    filenames = []
    for root, dirs, files in os.walk(pathname):
        for name in files:     
            filename = pathname + name
            filenames.append(filename)
            
    hmm_digits = get_hmm_digits(filenames)

    random.shuffle(filenames)
    test_files = filenames[:5]
    recognized_digits = testing(test_files,hmm_digits)
    
    ## accuracy
    cnt = 0
    for i in range(len(test_files)):
        name = test_files[i]
        print("test file: " + name)
        recognized = recognized_digits[i]
        print("recognized digit by hmm: " + recognized)
        if recognized in name:
            cnt += 1
        print("-------------------")
    print("acc: " + str(cnt/len(test_files)))

#main_gmm()

