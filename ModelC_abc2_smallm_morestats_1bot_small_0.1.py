#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 10:18:39 2020

@author: xinjunzhang
"""

import numpy as np
import pdb
import random
import format_data as fd
import run_posterior_decoding as pd
import os
import time,argparse
import itertools
from multiprocessing import Manager, Pool

S_LINE,T_LINE,DUR_LINE,OUT_LINE,PREDETERMINED_MUT, R_LINE = 4,30,16,34,25,9
DIR = '/u/scratch/x/xinjunzh/epas1/cluster_script_mine/' #was home
#DIR='/Users/xinjunzhang/Desktop/abc_hoffman/cluster_script_mine/'

parser = argparse.ArgumentParser(description="A script for computing summary statistics in 50kb windows across given chromosome, between modern human and archaic human.")
parser.add_argument('-r', '--rep', action="store", dest="rep_id",
                        help="which job replicate, default: 0",
                        default=1, type=int)
                        
args = parser.parse_args()

array_id = args.rep_id

#set up ranges of parameter values
def init_abc_pars(m_lower,m_upper,m_step_size,t_lower,t_upper,t_step_size,s_lower,s_upper,s_step_size):
    m_pop = np.arange(m_lower,m_upper+m_step_size,m_step_size)
    t_pop = np.arange(t_lower,t_upper+t_step_size,t_step_size)
    s_pop = np.arange(s_lower,s_upper+s_step_size,s_step_size)
    r_pop = np.arange(r_lower, r_upper+r_step_size, r_step_size)
    return m_pop, t_pop, s_pop, r_pop

def update_par_file_2bot(m,admix_time,s,r,selection_time,old_par_file_name, new_par_file_name):
#m,t_scaled, s_time, s_scaled, r_scaled, old_par_file_name, new_par_file_name
    #m,admix_time,s,r,old_par_file_name, new_par_file_name, outfile_name = m,t_scaled,s_scaled,r_scaled,old_par_file_name_admixture,new_par_file_name_admixture,outfile_name_admixture    
    oldfile = open(old_par_file_name)
    newfile = open(new_par_file_name,'w')
    line_counter=0
    t = 11600 - admix_time    
    S_LINE,T_LINE,DUR_LINE,SEL_LINE,R_LINE = 4,30,16,33,9
    for line_counter, line in enumerate(oldfile):
        fields = line.split()
        if line_counter==S_LINE:
            fields[1] = str(s)+");"
        elif line_counter==R_LINE:
            fields[0] = "initializeRecombinationRate("+str(r)+");"
        elif line_counter==DUR_LINE:
            fields[0] = str(int(t+1))+":11600"
        elif line_counter==T_LINE:
            fields[0] = str(int(t))
            fields[2] = str(m)+");}"
        elif line_counter==T_LINE+1:
            fields[0] = str(int(t+1))
        elif line_counter==SEL_LINE:
            fields[0] = str(int(t))+":"+str(int(11600-selection_time-1))                    
        elif line_counter==SEL_LINE+5:
            fields[0] = str(int(11600-selection_time))+":11600"
        new_line=str()    
        for item in fields:
            new_line = new_line+item+" "
        newfile.write(new_line+'\n')
    newfile.close()




def get_mutation_id(file_name, mut): #returns the id of the given mutation in the slim output file
    infile = open(file_name)
    flag = False
    
    to_return = ''
    for line in infile:
        if line[0:9]=='Mutations':
            flag = True
            continue
        if flag==False:
            continue
        if line[0:7]=='Genomes':
            break

        #good line
        fields = line.split()
        if fields[2]==mut:
            to_return = int(fields[0])
            break

    infile.close()
    return to_return

def replace_mutation(infile_name,outfile_name,old_id,new_id,booster_id): #replace the mutation id of the given population
    infile = open(infile_name)
    outfile = open(outfile_name,'w')
 
    for line in infile:
        if line[0:3]=='p3:': #replace m2 with m4
            fields = line.split()

            if new_id in fields:
                fields.remove(new_id)
            new_fields = [x if x!=old_id else new_id for x in fields] #replace

            for item in new_fields:
                outfile.write(item+' ')
            outfile.write('\n')
        elif line[0:3]=='p1:': #remove booster mutation
            fields = line.split()
            if booster_id in fields:
                fields.remove(booster_id)
            for item in fields:
                outfile.write(item+' ')
            outfile.write('\n')
        else:
            outfile.write(line)

    infile.close()
    outfile.close()


def check_den_fixation(info_name,t_scaled,N,t_scale): # returns true if m2 was fixed in p1 before admixture
    info_file = open(info_name)
    is_fixed = False
    for line in info_file:
        if line[0:5]=='#OUT:':
            fields = line.split()
            #time = int(fields[1])
            pop = fields[3]
            freq = int(fields[-1])
            #if time==t_scaled and pop=='p1': print (time,pop,freq)
            if time==t_scaled and pop=='p1' and freq==(2*N)/t_scale: # fixation
                is_fixed=True
    return is_fixed
#not changed, maybe not used

def check_den_fixation_admixture(file_name,mut): # returns true if m2 was fixed in p1 before admixture
    infile = open(file_name)
    flag = False
    
    to_return = False
    for line in infile:
        if line[0:9]=='Mutations':
            flag = True
            continue
        if flag==False:
            continue
        if line[0:7]=='Genomes':
            break

        #good line
        fields = line.split()
        if fields[2]==mut:
            to_return = (int(fields[8])>=2000)
            break
        
    infile.close()
    return to_return

def getSelectedAlleleFreq(file_name,mut): # returns true if m2 was fixed in p1 before admixture
    infile = open(file_name)
    freq = 0
 
    for line in infile:
        if line[0:5]=='#OUT:':
            fields = line.split()
            outType = fields[2]
            pop = fields[3]
            if outType=='T' and pop=='p3':
                freq = int(fields[11])
                break
    infile.close()

    return freq

def insert_anc_alleles (allpos,pos,hap):
    for site in allpos:
        if site not in pos:
            insertidx = np.searchsorted(pos,site)
                # Add site to posvecdict[pop]
            pos = np.insert(pos,insertidx,site)
                # Add site column with ancestral alleles (zeroes) to genotype matrix
            hap = np.insert(hap, insertidx, 0, axis=1)
    return pos, hap
 
#p2_pos,p2_hap = insert_anc_alleles(all_pos,p2_pos,p2_hap)
    
def calc_derived_freq (pop_hap):
    popfreq = np.sum(pop_hap, axis=0)
    popfreq = popfreq/ float(pop_hap.shape[0])
    return popfreq

def vSumFunc(other_hap, p1_hap,currentArchi):
    current_hap = np.array([p1_hap[currentArchi,]])
    div = np.zeros(other_hap.shape)
    ones = np.ones((other_hap.shape[0],1))
    current_hap = current_hap
    current_hap_extended = np.dot(ones, current_hap)
        #print(current_hap_extended)
        #computes vectorized logical xor between current_hap and the hap being analyzed
    div = np.logical_xor(current_hap_extended == 1, other_hap == 1)

        #reduces the div on each row with sum to count the number of different alleles
    return np.add.reduce(div, 1)   


def calc_freq_info(file_path,len_genome): # load slim's output 
    pos_den, hapMat_den = get_pos_hap(file_path,'p1',len_genome)
    pos_afr, hapMat_afr = get_pos_hap(file_path,'p2',len_genome)
    pos_nonafr, hapMat_nonafr = get_pos_hap(file_path,'p3',len_genome)
    
    p1_pos = pos_den
    p2_pos = pos_afr
    p3_pos = pos_nonafr
    p1_hap = hapMat_den
    p2_hap = hapMat_afr
    p3_hap = hapMat_nonafr
    
    all_pos = np.unique(np.concatenate((p1_pos,p2_pos,p3_pos))) #len=244
    #3. insert non-segregating sites/ancestral alleles 0s to the hap matrices and pos lists  
    p1_pos,p1_hap = insert_anc_alleles(all_pos,p1_pos,p1_hap)
    p2_pos,p2_hap = insert_anc_alleles(all_pos,p2_pos,p2_hap)
    p3_pos,p3_hap = insert_anc_alleles(all_pos,p3_pos,p3_hap)
        
    #4. get derived allele freq   
    p1_freq = calc_derived_freq (p1_hap)
    p2_freq = calc_derived_freq (p2_hap)
    p3_freq = calc_derived_freq (p3_hap)

    abbavec = (1.0 - p2_freq)*p3_freq*p1_freq
    babavec = p2_freq*(1.0 - p3_freq)*p1_freq
    abba = np.sum(abbavec)
    baba = np.sum(babavec)

    checkfd1 = (p3_freq > p1_freq)
    abbafd1 = (1.0 - p2_freq)*p3_freq*p3_freq
    babafd1 = p2_freq*(1.0 - p3_freq)*p3_freq
    checkfd2 = (p3_freq < p1_freq)
    abbafd2 = (1.0 - p2_freq)*p1_freq*p1_freq
    babafd2 = p2_freq*(1.0 - p1_freq)*p1_freq
    abbafd = checkfd1 * abbafd1 + checkfd2 * abbafd2
    babafd = checkfd1 * babafd1 + checkfd2 * babafd2
    abbafd = np.sum(abbafd)
    babafd = np.sum(babafd)
    if (abbafd + babafd > 0):
        fD = (abba - baba) / (abbafd - babafd)
    else:
        fD = float('nan')

    divratio = []

    for archi in range(0, p1_hap.shape[0]): #iterate over 0-99 haps; 100 total
        divarchintro = vSumFunc(p3_hap, p1_hap,archi)
        divarchintro = divarchintro.astype("float")
        divarchnonintro = vSumFunc(p2_hap, p1_hap,archi)        
        divarchnonintro = divarchnonintro.astype("float") #took the inversion here so that the multiplying below is really actually dividing; probably not necessary       
        for comb in itertools.product(divarchintro,divarchnonintro): #pairwise combos of divarchintro/divarchnonintro
            if comb[1] != 0:
                divratio.append(comb[0]/comb[1])
    RD = float(sum(divratio)) / float(len(divratio)) #len(divratio) = 100* (100*100)

    
    ArcDer = (p1_freq ==1)
    NonAdm_1 = (p2_freq < 0.01)
    p3_freq20 = (p3_freq > 0.2)
    p3_freq50 = (p3_freq > 0.5)
    ArcDerNonAdm_1 = (ArcDer & NonAdm_1)
    DerFreqs_NonAdm_1 = p3_freq[np.where(ArcDerNonAdm_1 == True)]
    ArcDerNonAdm_1_20 = (ArcDer & NonAdm_1 & p3_freq20)
    DerFreqs_NonAdm_1_20 = p3_freq[np.where(ArcDerNonAdm_1_20 == True)]
    ArcDerNonAdm_1_50 = (ArcDer & NonAdm_1 & p3_freq50)
    DerFreqs_NonAdm_1_50 = p3_freq[np.where(ArcDerNonAdm_1_50 == True)]

    if DerFreqs_NonAdm_1.size > 0:
        Q_1_100_q95 = np.percentile(DerFreqs_NonAdm_1,95)

    else:
        Q_1_100_q95 = float('nan')

    U_1_20_100 = ( ArcDerNonAdm_1 & (p3_freq > 0.2) )
    U_1_50_100 = ( ArcDerNonAdm_1 & (p3_freq > 0.5) )
    U_1_50_100 = np.sum(U_1_50_100)
    U_1_20_100 = np.sum(U_1_20_100)

    
    return fD, RD, Q_1_100_q95, U_1_20_100, U_1_50_100

def get_pos_hap(file_path,pop_id,len_genome): #get pos and hapMat for a given pop from slim output #n_snps=-1 means don't remove any snps
    infile = open(file_path,'r')
    while True:
        line = infile.readline()
        #print(line)
        if line[0:5]=='#OUT:': #output lines
            fields = line.split()
            out_type = fields[2]
            pop = fields[3]
            gen = fields[1]
            if out_type=='SM' and pop==pop_id and int(gen) == 11600: #ms lines
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
                return pos, hapMat
                
    return [],[]

def find_mult_mut_pos(pos): #find repeating mutations and remove them
    dist = np.array([pos[i+1]-pos[i] for i in range(0,len(pos)-1)])
    mult_mut_pos = np.where(dist==0)[0]
    return mult_mut_pos

 
def run_abc_variable_selection_time(n,q):
    print (n)
    # file names
    old_par_file_name = DIR + 'ModelD_adaptive_intro_scaled_admixture_1bot_small.par'
    #old_par_file_name_admixture = DIR + 'adaptive_intro_scaled_tib_demo_simple.par'
    new_par_file_name = DIR + str(array_id)+'adaptive_intro_scaled_admixture-large_%d.par' %n
    outfile_name_evolve_2 = DIR + str(array_id)+'evolve-large_2_%d.out' %n

    # draw m,s,t
    m = 0.001 #0.1%
    t = random.choice(t_pop)
    s = random.choice(s_pop)
    r = 2.3e-8

    t_scaled = t/t_scale
    s_scaled = s*t_scale
    r_scaled = r*t_scale
    #print (m,t_scaled,s_scaled)
    #this_t = t_scaled

	#times = np.arange(10,120,10)
    #times = [t_scaled-1,35,52] #admixture time or 10,000 or 15,000 generations ago for selection
    times = np.arange(10,t_scaled-1,10)
    s_time = [random.choice(times)]
    #s_time = s_time[0]
    
    for s_time in s_time:

        try:
            
        #evolve until end
            update_par_file_2bot(m,t_scaled, s_scaled, r_scaled, s_time, old_par_file_name, new_par_file_name)
            #m,admix_time,s,r,selection_time,old_par_file_name, new_par_file_name
            os.system('/u/home/x/xinjunzh/slim_build/slim %s > %s' %(new_par_file_name,outfile_name_evolve_2))
        #check the allele frequency of the selected allele is nonzero
            m2Present = False
            freq = getSelectedAlleleFreq(outfile_name_evolve_2, 'm2')
            if(freq > 0): m2Present = True

 
        #print 'Preprocessing data'
        # make obs and pos files from the slim output file
            if(m2Present==True):
                obs_mat, pos = fd.compute_obs_pos(outfile_name_evolve_2,len_genome,n_snps,afr_thresh)
                m2_freq = freq
                fD, RD, Q_1_100_q95, U_1_20_100, U_1_50_100= calc_freq_info(outfile_name_evolve_2,len_genome)
                results = pd.posterior_decoding_return_startEndPos(obs_mat,pos,m,t,hmm_r,hmm_thresh)
                q.put([t,s_time*t_scale,m2_freq,s,m,r,fD, RD, Q_1_100_q95, U_1_20_100, U_1_50_100,results])

        except:
            pass

    #remove files
    os.system('rm '+new_par_file_name)
    os.system('rm '+outfile_name_evolve_2)


def write_to_file(outfile_name,q):
    # open outfile
    outfile = open(outfile_name,'w')
    
    # write things in queue
    while 1:
        q_elem = q.get()

        if q_elem=='kill': # break if end of queue
            print ('END OF SIMULATIONS')
            break

        #[t,s_time,s,m,r,af_den, af_den_tib] = q_elem 
        [t,s_time,m2_freq,s,m,r,fD, RD, Q_1_100_q95, U_1_20_100, U_1_50_100,results] = q_elem
        outfile.write('>%d\t%d\t%f\t%f\t%f\t%.3E\t%f\t%f\t%f\t%f\t%f\n' %(t,s_time,m2_freq,s,m,r,fD, RD, Q_1_100_q95, U_1_20_100, U_1_50_100)) #write abc params
        num_indiv = len(results)
        for i in range(0,num_indiv):
        	outfile.write('%d\t' %i) # write indiv id
        	seg_list = results[i]
        	for start_end in seg_list:
        		outfile.write('%d\t%d\t' %(start_end[0],start_end[1]))
        	outfile.write('\n')
        outfile.flush()
    outfile.close()



if __name__=='__main__':

    ############### CHANGE ME ###############

    # abc parameters
    num_abc_reps=400000 # number of simulations
    m_lower, m_upper, m_step_size = .001, .01, .002 #admixture proportion
    t_lower,t_upper,t_step_size = 500,2400,20 #admixture time in generations (based on EUR demography)
    s_lower,s_upper,s_step_size = .001, .1, .002 #selection strength
    r_lower, r_upper, r_step_size = 1e-8, 3e-8, 5e-9 #recombination rate per bp per generation
    #outfile_name = '/Users/xinjunzhang/Desktop/EPAS1/Simulation/my_result/eurDem.30000reps.out'
    outfile_name = '/u/scratch/x/xinjunzh/epas1/cluster_result_mine/400k_100-1bot-large_f+l'+str(array_id)+'.out' #output file name

    ##########################################


    # slim parameters
    N = 10000
    len_genome = 100000 
    t_scale = 10 # scale time by 1/10
    t_burnin = 100000 #burn in period for den+hum anc pop
    t_div = 16000 #diverence time between denisovan and modern humans
    offset = t_burnin+t_div
    n_snps = 464 #number of snps in the real data; match the number of Tibetan sites in the simulated data to this

    # HMM parameters
    afr_thresh = 0.01
    hmm_thresh = .9
    hmm_r = 2.3e-8

    #range of parameters to be tested
    m_pop, t_pop, s_pop, r_pop = init_abc_pars(m_lower,m_upper,m_step_size,t_lower,t_upper,t_step_size,s_lower,s_upper,s_step_size)

    # Initialize pool
    num_proc = 100
    manager = Manager()
    pool = Pool(processes=num_proc)
    q = manager.Queue()
    watcher = pool.apply_async(write_to_file,(outfile_name,q))
    

    # TESTING
    #run_abc_variable_selection_time(0,q)
 
    # run abc
    reps = range(0,num_abc_reps)
    args_iterable = list(zip(reps,[q]*num_abc_reps))
    

    for i in args_iterable:
        #print(i)
        #print(i[0],i[1])
        run_abc_variable_selection_time(i[0],i[1])
    #pool.map(run_abc_variable_selection_time,args_iterable)

    # close pool
    q.put('kill')
    pool.close()
    pool.join()

