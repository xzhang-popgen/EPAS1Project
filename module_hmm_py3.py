#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 23:01:53 2018

@author: xinjunzhang
"""
import numpy as np
import pdb
import random
from scipy.optimize import fmin_l_bfgs_b as minimize
import matplotlib.pyplot as plt
import time
import pickle
import copy

class HMM:

    def __init__(self, prior=np.array([.98,.02]), transition=np.array([[.999,.001],[.99,.01]]), emission=np.array([[.999,.001],[.995,.005]]), dist_set=set([1])): 
        self.prior = prior
        self.transition = transition
        self.emission = emission
        self.trans_dict = []
        #self.trans_dict = self.compute_trans_mats(dist_set,self.transition)
        
    def __del__(self):
        del self.prior
        del self.transition
        del self.emission
        del self.trans_dict

    def compute_trans_mats(self,dist_set,A):
        trans_dict = dict()
        for dist in dist_set:
            trans_dict[dist] = np.linalg.matrix_power(A,dist)
        return trans_dict

    def get_dist_set(self,pos_list):
        dist_set = set()
        for positions in pos_list:
            dist = [positions[i+1]-positions[i] for i in range(0,len(positions)-1)]
            dist_set.update(dist)
        return dist_set

    def forward_backward_scaled(self, observed_states, positions, thresh):

        # Initialize
        num_hidden_states = self.transition.shape[0]
        num_sites = len(positions)
        alpha_table = np.zeros([num_hidden_states,num_sites])
        beta_table = np.zeros([num_hidden_states,num_sites])
        gamma_table = np.zeros([num_hidden_states,num_sites])
        path = np.zeros(num_sites,dtype=int)
        probs = np.zeros(num_sites)
        alpha_scales = np.zeros(num_sites)
        beta_scales = np.zeros(num_sites)

        # Alpha pass
        alpha_table[:,0] = self.prior * self.emission[:,observed_states[0]]
        alpha_scales[0] = max(alpha_table[:,0])
        alpha_table[:,0] = alpha_table[:,0]/alpha_scales[0]
        for t in range(1,num_sites): 
            trans_mat = self.trans_dict[positions[t]-positions[t-1]]
            for s in range(0,num_hidden_states):
                alpha_table[s,t] = sum(alpha_table[:,t-1] * trans_mat[:,s]) * self.emission[s,observed_states[t]]
            alpha_scales[t] = max(alpha_table[:,t])
            alpha_table[:,t] = alpha_table[:,t]/alpha_scales[t]
        
        # Beta pass
        beta_table[:,-1] = 1
        beta_scales[-1] = 1
        for t in range(num_sites-2,-1,-1): 
            trans_mat = self.trans_dict[positions[t+1]-positions[t]]
            for s in range(0,num_hidden_states):
                beta_table[s,t] = sum(trans_mat[s,:] * self.emission[:,observed_states[t+1]] * beta_table[:,t+1])
            beta_scales[t] = max(beta_table[:,t])
            beta_table[:,t] = beta_table[:,t]/beta_scales[t]

        # Compute gammas
        for t in range(0,num_sites):
            denom = sum(alpha_table[:,t] * beta_table[:,t])
            gamma_table[:,t] = (alpha_table[:,t] * beta_table[:,t])/denom

        # Reconstruct best path from gamma table
        path[np.where(gamma_table[1,:]>thresh)[0]] = 1
        for i in range(0,len(probs)):
            probs[i] = gamma_table[path[i],i]

        return path, probs

    def compute_log_lkhd_general(self, pi,A,B,positions,observed_states, trans_mats_dict): 
        # Initialize
        num_hidden_states = A.shape[0]
        num_time = len(observed_states)
        prev_alphas = np.zeros(num_hidden_states)
        curr_alphas = np.zeros(num_hidden_states)
        log_scale_sum = 0

        # Initial state
        curr_alphas = pi * B[:,observed_states[0]]
        scale = max(curr_alphas)
        curr_alphas = curr_alphas/scale
        prev_alphas = copy.copy(curr_alphas)
        log_scale_sum += np.log(scale)

        for t in range(1,num_time):
            n_step_trans_mat = trans_mats_dict[positions[t]-positions[t-1]]
            for s in range(0,num_hidden_states):
                curr_alphas[s] = sum(prev_alphas * n_step_trans_mat[:,s]) * B[s,observed_states[t]]
            scale = max(curr_alphas)
            curr_alphas = curr_alphas/scale
            prev_alphas = copy.copy(curr_alphas)
            log_scale_sum += np.log(scale)
        log_lkhd = log_scale_sum + np.log(sum(curr_alphas))
        return log_lkhd

    def compute_log_lkhd_2states(self, pi,A,B,positions,observed_states, trans_mats_dict, num_time): #faster than general case

        curr_alpha_0, curr_alpha_1, prev_alpha_0, prev_alpha_1 = 0,0,0,0
        log_scale_sum = 0

        curr_alpha_0 = pi[0]*B[0][observed_states[0]]
        curr_alpha_1 = pi[1]*B[1][observed_states[0]]
        curr_scale = max(curr_alpha_0,curr_alpha_1)
        curr_alpha_0 = curr_alpha_0/curr_scale
        curr_alpha_1 = curr_alpha_1/curr_scale
        prev_alpha_0 = curr_alpha_0
        prev_alpha_1 = curr_alpha_1
        log_scale_sum+=np.log(curr_scale)

        for t in range(1,num_time):
            n_step_trans_mat = trans_mats_dict[positions[t]-positions[t-1]]
            curr_alpha_0 = (prev_alpha_0*n_step_trans_mat[0][0] + prev_alpha_1*n_step_trans_mat[1][0])*B[0][observed_states[t]]
            curr_alpha_1 = (prev_alpha_0*n_step_trans_mat[0][1] + prev_alpha_1*n_step_trans_mat[1][1])*B[1][observed_states[t]]
            curr_scale = max(curr_alpha_0,curr_alpha_1)
            curr_alpha_0 = curr_alpha_0/curr_scale
            curr_alpha_1 = curr_alpha_1/curr_scale
            prev_alpha_0 = curr_alpha_0
            prev_alpha_1 = curr_alpha_1
            log_scale_sum+=np.log(curr_scale)
        lkhd = log_scale_sum + np.log(curr_alpha_0 + curr_alpha_1)

        return lkhd

    def compute_joint_neg_log_lkhd(self,pi,A,B,trans_mats_dict,pos_list,observed_list): 
        num_chrom = len(pos_list)
        num_indiv = observed_list[0].shape[0]
        joint_log_lkhd = 0

        for chrom in range(0,num_chrom):
            positions = pos_list[chrom]
            num_time = len(positions)
            for indiv in range(0,num_indiv):
                observed_states = observed_list[chrom][indiv]
                log_lkhd = self.compute_log_lkhd_2states(pi,A,B,positions,observed_states,trans_mats_dict, num_time)
                joint_log_lkhd += log_lkhd

        return -joint_log_lkhd

    def estimate_emissions(self, x0, a_00, a_10, pos_list, observed_list):
        pi_0 = a_10/float(1-a_00+a_10)
        A = np.array([[a_00, 1-a_00],
                      [a_10, 1-a_10]])
        pi = np.array([pi_0, 1-pi_0])

        # compute trans_mats_dict
        dist_set = self.get_dist_set(pos_list)
        trans_mats_dict = self.compute_trans_mats(dist_set,A)
        
        bds = [(None,0),(None,0)]
        x,f,d = minimize(self.aux_func_emissions,x0, fprime=None, args=(pi,A,trans_mats_dict,pos_list,observed_list), approx_grad=True, bounds=bds, m=10, factr=1e7, pgtol=1e-08, epsilon=1e-08, iprint=0, maxfun=15000, disp=None)  

        # reparameterize
        e_00 = 1-np.exp(x[0])
        e_10 = 1-np.exp(x[1])
        x = [e_00,e_10]

        return x,f,d

    def aux_func_emissions(self,x,*args): #auxiliary function that returns the joint likelihood for given emissions
        pi,A,trans_mats_dict,pos_list,observed_list = args
        log_e_01, log_e_11 = x
        e_00 = 1-np.exp(log_e_01)
        e_10 = 1-np.exp(log_e_11)
        B = np.array([[e_00, 1-e_00],
                      [e_10, 1-e_10]])

        t0 = time.time()
        joint_neg_log_lkhd = self.compute_joint_neg_log_lkhd(pi,A,B,trans_mats_dict,pos_list,observed_list)
        print (int(time.time()-t0))
        print ([e_00, e_10, joint_neg_log_lkhd])
        
        return joint_neg_log_lkhd
        

    def estimate_transitions(self, x0, e_00, e_10, pos_list, observed_list):
        B = np.array([[e_00, 1-e_00],
                      [e_10, 1-e_10]])

        dist_set = self.get_dist_set(pos_list)
        
        bds = [(None,0),(None,0)]
        x,f,d = minimize(self.aux_func_transitions,x0, fprime=None, args=(B,dist_set,pos_list,observed_list), approx_grad=True, bounds=bds, m=10, factr=1e7, pgtol=1e-08, epsilon=1e-08, iprint=0, maxfun=15000, disp=None)  
      
        # reparameterize
        a_00 = 1-np.exp(x[0])
        a_10 = np.exp(x[1])
        x = [a_00, a_10]

        return x,f,d

    def aux_func_transitions(self,x,*args):
        B,dist_set,pos_list,observed_list = args
        log_a_01,log_a_10 = x 
        a_00 = 1-np.exp(log_a_01)
        a_10 = np.exp(log_a_10)
        pi_0 = a_10/(1-a_00+a_10)
        pi = np.array([pi_0, 1-pi_0])
        A = np.array([[a_00, 1-a_00],
                      [a_10, 1-a_10]])

        # compute trans_mats_dict
        trans_mats_dict = self.compute_trans_mats(dist_set,A)

        t0 = time.time()
        joint_neg_log_lkhd = self.compute_joint_neg_log_lkhd(pi,A,B,trans_mats_dict,pos_list,observed_list)
        print (int(time.time()-t0))
        print ([a_00,a_10,joint_neg_log_lkhd])

        return joint_neg_log_lkhd

    def stitch_tracts(self, path, length_thresh, dist_thresh):
        num_sites = len(path)
        # compute starts and ends
        starts, ends = np.array([]),np.array([])
        prev,curr = 0,0
        for i in range(0,num_sites):
            curr = path[i]
            if curr==1 and prev==0:
                starts = np.append(starts,i)
            elif (curr==0 and prev==1) or (i==num_sites-1 and curr==1):
                ends = np.append(ends,i-1)
            prev = curr

        # stitch short tracts
        stitched_starts, stitched_ends = np.array([]), np.array([])
        stitched_starts = np.append(stitched_starts,starts[0])
        for i in range(1,len(starts)):
            prev_end = ends[i-1]
            curr_start = starts[i]
            if curr_start-prev_end > dist_thresh:
                stitched_ends = np.append(stitched_ends,prev_end)
                stitched_starts = np.append(stitched_starts, curr_start)
        stitched_ends = np.append(stitched_ends, ends[i]) # add the last end

       # remove short tracts
        lengths = stitched_ends-stitched_starts
        bad_ind = np.where(lengths<length_thresh)[0]
        stitched_starts = np.delete(stitched_starts, bad_ind)
        stitched_ends = np.delete(stitched_ends, bad_ind)

        # stitched path
        stitched_path = np.zeros(num_sites,dtype=int)
        for i in range(0,len(stitched_starts)):
            stitched_path[stitched_starts[i]:stitched_ends[i]+1] = 1
        
        return stitched_path

    def get_starts_ends(self,path,pos):
        num_sites = len(path)
        starts,ends = np.array([],dtype=int),np.array([],dtype=int)
        prev,curr = 0,0
        for i in range(0,num_sites):
            curr = path[i]
            if curr==1 and prev==0:
                starts = np.append(starts,pos[i])
            elif (curr==0 and prev==1) or (i==num_sites-1 and curr==1):
                ends = np.append(ends,pos[i-1])
            prev = curr

        return starts,ends
        
        

if __name__=="__main__":

    # Testing posterior decoding
    priors = np.array([.5,.5])
    emission = np.array([[.9, .1],
                         [.2, .8]])
    transition = np.array([[.7,.3],
                           [.3,.7]])

    positions = np.array([1,2,3,4,5])
    observations = np.array([0,0,1,0,0])
    dist_set = set([positions[i+1]-positions[i] for i in range(0,len(positions)-1)])
    thresh=.5

    myhmm = HMM(priors,transition,emission,dist_set)

    # test stitching
    path = np.array([0,0,1,1,0,0,0,1,1,0,1,1,0,1,1,0])
    stitched_path = myhmm.stitch_tracts(path,3,2)
    pdb.set_trace()

    path, probs = myhmm.forward_backward_scaled(observations,positions,thresh)

    # Testing parameter estimation
    observed_mat = np.array([observations])
    pos_list = [positions]
    obs_list = [observed_mat]

    x0 = np.array([np.log(1-0.9999),np.log(1-.98)])
    x,f,d = myhmm.estimate_emissions(x0,transition[0,0],transition[1,0],pos_list,obs_list)

    x0 = np.array([np.log(1-.99),np.log(.1)])
    x,f,d = myhmm.estimate_transitions(x0,emission[0,0],emission[1,0], pos_list,obs_list)

    pdb.set_trace()
