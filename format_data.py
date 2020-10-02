import numpy as np
import pdb
import random
import time
import pickle
import matplotlib.pyplot as plt
import module_hmm_py3 as hmm

def load_data(filePath): # load emilia's data
    infile = open(filePath,'r')
    #6 header stuff, 2 den, 292 tibs, 216 afr, 318 snps
    infile.readline() # skip header
    geno_mat = np.zeros([1200,510],dtype=int)
    positions = np.array([],dtype=int)
    pos_index = 0
    for line in infile:
        line = line.split('\n')[0].split('\t')
        pos = int(float(line[1]))
        geno = line[6:]
        #print(geno)
        geno = np.array([int(g) for g in geno])
        positions = np.append(positions,pos)
        geno_mat[pos_index] = geno
        pos_index+=1

    geno_mat = geno_mat.T
    hapMat_den = geno_mat[0:2]
    hapMat_nonAfr = geno_mat[2:294]
    hapMat_afr = geno_mat[294:]

    #compute frequency vectors
    af_den = sum(hapMat_den)/float(hapMat_den.shape[0])
    af_nonAfr = sum(hapMat_nonAfr)/float(hapMat_nonAfr.shape[0])
    af_afr = sum(hapMat_afr)/float(hapMat_afr.shape[0])

    # find variable sites among Tibetans
    varInd = np.where(af_nonAfr>0)[0]

    # filter out sites that are not variable/fixed derived in Tibetans
    hapMat_nonAfr = hapMat_nonAfr[:,varInd]
    af_afr = af_afr[varInd]
    af_den = af_den[varInd]
    positions = positions[varInd]

    return hapMat_nonAfr, af_nonAfr, af_afr, af_den, positions, varInd

def load_data_slim(file_path,len_genome,n_snps): # load slim's output
    pos_den, hapMat_den = get_pos_hap(file_path,'p1',len_genome)
    pos_afr, hapMat_afr = get_pos_hap(file_path,'p2',len_genome)
    positions, hapMat_nonAfr = get_pos_hap(file_path,'p3',len_genome,n_snps)
    
    # build hapMat for afr and nea on the european variable positions
    af_den = make_af_vec(pos_den,hapMat_den,positions)
    af_afr = make_af_vec(pos_afr,hapMat_afr,positions)
    af_nonAfr = np.mean(hapMat_nonAfr,0)

    # varInd
    varInd = np.array(range(0,len(positions)))

    return hapMat_nonAfr, af_nonAfr, af_afr, af_den, positions,varInd

def make_af_vec(pos_pop,hapMat_pop,pos_target): # compute af of the given pop on the target pos
    num_indiv = hapMat_pop.shape[0]
    num_pos = len(pos_target)
    pos_pop_set = set(pos_pop)
    af_pop = np.mean(hapMat_pop,0)

    af_target = np.zeros(num_pos)
    for i in range(0,num_pos):
        pos = pos_target[i]
        if pos in pos_pop_set: #if variable site in pop, use that info; otherwise af=0.0
            col = np.where(pos_pop==pos)[0]
            af_target[i] = af_pop[col]

    return af_target

def get_pos_hap(file_path,pop_id,len_genome,n_snps=-1): #get pos and hapMat for a given pop from slim output #n_snps=-1 means don't remove any snps
    infile = open(file_path,'r')
    while True:
        line = infile.readline()
        if line[0:5]=='#OUT:': #output lines
            fields = line.split()
            out_type = fields[2]
            pop = fields[3]
            if out_type=='SM' and pop==pop_id: #ms lines
                num_indiv = int(fields[4])
                infile.readline() #skip //
                infile.readline() #skip segsites
                pos = (np.array(infile.readline().split()[1:]).astype(float) * len_genome).astype(int)
                # find positions with multiple mutations
                mult_mut_pos = find_mult_mut_pos(pos)+1
                # remove repeated pos
                pos = np.delete(pos,mult_mut_pos)
                # get haplotypes
                hapMat = np.zeros((num_indiv,len(pos)),dtype=int)
                for indiv in range(0,num_indiv):
                    hap = np.array(list(infile.readline())[:-1]).astype(int)
                    # remove repeated regions
                    hap = np.delete(hap,mult_mut_pos)
                    hapMat[indiv] = hap
                infile.close()
                #match the number of snps in the real data
                if n_snps!=-1 and len(pos)>n_snps:
                    ind = np.sort(random.sample(range(0,len(pos)),n_snps))
                    pos = pos[ind]
                    hapMat = hapMat[:,ind]
                return pos,hapMat
    return [],[]
    
                
def find_mult_mut_pos(pos):
    dist = np.array([pos[i+1]-pos[i] for i in range(0,len(pos)-1)])
    mult_mut_pos = np.where(dist==0)[0]
    return mult_mut_pos

def compute_obs_pos(file_name,len_genome,n_snps,thresh=0):
    if len_genome>0:
        hapMat_nonAfr, af_nonAfr, af_afr, af_den, positions, varInd = load_data_slim(file_name,len_genome,n_snps)
    else:
        hapMat_nonAfr, af_nonAfr, af_afr, af_den, positions, varInd = load_data(file_name)

    obs_mat = np.zeros(hapMat_nonAfr.shape,dtype=int)
    for indiv in range(0,hapMat_nonAfr.shape[0]):
        for pos in range(0,hapMat_nonAfr.shape[1]):
            obs_mat[indiv,pos] = int(hapMat_nonAfr[indiv,pos]==1 and af_afr[pos]<=thresh and af_den[pos]>0)

    return obs_mat,positions

def path_to_lengths(infile_name,outfile_name): #convert ancestry path file into length file
    myhmm = hmm.HMM()
    infile = open(infile_name)
    outfile = open(outfile_name,'w')
    
    infile.readline() #skip header
    pos = np.array(infile.readline().split()[1:]).astype(int)
    
    outfile.write('> ? ? ?\n')
    counter = 0
    for counter,line in enumerate(infile):
        fields = line.split()
        indiv = fields[0]
        path = np.array(fields[1:]).astype(int)
        starts,ends = myhmm.get_starts_ends(path,pos)
        outfile.write(str(counter)+'\t')
        for i in range(0,len(starts)):
            outfile.write('%d\t%d\t' %(starts[i],ends[i]))
        outfile.write('\n')
    outfile.close()

if __name__=='__main__':
    file_name = '/u/home/x/xinjunzh/epas1/cluster_result_mine/50000.out'
    afr_thresh=0
    n_snps=464
    n_snps=318
    len_genome=100000

    #obs_mat,pos = compute_obs_pos(file_name,len_genome,n_snps,afr_thresh)
    #pdb.set_trace()
    #path_to_lengths('/Users/xinjunzhang/Desktop/abc_hoffman/cluster_data_mine/real_jul23_2018.txt','/Users/xinjunzhang/Desktop/abc_hoffman/cluster_data_mine/real_jul23_2018.out')
    
    file_name = '/Users/xinjunzhang/Desktop/abc_hoffman/cluster_data_mine/181112tib38_real_core.txt'
    outfile_name =  '/Users/xinjunzhang/Desktop/abc_hoffman/cluster_data_mine/real_181112tib38_real_core.out'
    path_to_lengths(file_name,outfile_name)