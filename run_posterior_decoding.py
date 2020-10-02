import module_hmm_py3 as hmm
import numpy as np
import pdb
import random
import time
import pickle
import matplotlib.pyplot as plt
from multiprocessing import Lock, Manager, Pool
import format_data as fd

def posterior_decoding_fb(file_name,outfile_name,obs_mat,pos,m,t,r,thresh): # for real data

    # compute transition and emission probs
    a_01 = r*(t-1)*m
    a_10 = r*(t-1)*(1-m)
    e_11 = .050702
    e_01 = 7.405e-4

    # initialize HMM
    myhmm = hmm.HMM()
    myhmm.prior = np.array([1-m,m])
    myhmm.transition = np.array([[1-a_01, a_01],
                                 [a_10, 1-a_10]])
    myhmm.emission = np.array([[1-e_01, e_01],
                               [1-e_11, e_11]])
    dist_set = myhmm.get_dist_set([pos])
    myhmm.trans_dict = myhmm.compute_trans_mats(dist_set,myhmm.transition)
    
    # open outfile & write params, pos
    outfile = open(outfile_name,'w')
    outfile.write('parameters:\tm=%f\tt=%d\tr=%E\tthresh=%f\te_01=%E\te_11=%E\n'%(m,t,r,thresh,e_01,e_11))
    outfile.write('indiv\t')
    for p in pos:
        outfile.write('%d\t' %p)
    outfile.write('\n')
    names = open(file_name).readline().split()
    names = names[7:299]
    print (len(names))

    # run posterior decoding for each individual
    for i in range(0,obs_mat.shape[0]):
        print (i)
        obs = obs_mat[i,:]
        path,probs = myhmm.forward_backward_scaled(obs, pos, thresh)
        outfile.write(names[i]+'\t')
        write_to_file(outfile,path)
    
    # close file
    outfile.close()

def posterior_decoding_return_startEndPos(obs_mat,pos,m,t,r,thresh): # for slim data

    # compute transition and emission probs
    a_01 = r*(t-1)*m
    a_10 = r*(t-1)*(1-m)
    e_11 = .050702
    e_01 = 7.405e-4

    # initialize HMM
    myhmm = hmm.HMM()
    myhmm.prior = np.array([1-m,m])
    myhmm.transition = np.array([[1-a_01, a_01],
                                 [a_10, 1-a_10]])
    myhmm.emission = np.array([[1-e_01, e_01],
                               [1-e_11, e_11]])
    dist_set = myhmm.get_dist_set([pos])
    myhmm.trans_dict = myhmm.compute_trans_mats(dist_set,myhmm.transition)

    # run posterior decoding for each individual
    results = []
    for i in range(0,obs_mat.shape[0]):
        obs = obs_mat[i,:]
        path,probs = myhmm.forward_backward_scaled(obs, pos, thresh)
        starts, ends = myhmm.get_starts_ends(path,pos)
        results.append(zip(starts,ends))

    return results

def write_to_file(outfile,path): # for real data
    for item in path:
        outfile.write('%d\t'%item)
    outfile.write('\n')

if __name__=='__main__':

    #file_name = '/Users/xinjunzhang/Desktop/abc_hoffman/cluster_data_mine/biallelic_tibs_yrionly_deniDefined_epas1core_derived01_jul23_2018.txt'
    #outfile_name = '/Users/xinjunzhang/Desktop/abc_hoffman/cluster_data_mine/real_jul23_2018.txt'

    #obs_mat,pos = fd.compute_obs_pos(file_name,0,318)
    
    #posterior_decoding_fb(file_name,outfile_name,obs_mat,pos,.02,1900,2.3e-8,thresh=.9)
