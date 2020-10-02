#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:20:29 2020

@author: xinjunzhang
"""

import os
import numpy as np
import random

os.chdir("/u/scratch/x/xinjunzh/epas1/")

abc_files = "ABC400k.out"
output_txt = "ABC400k.parameter-6stats.txt"

title = ["sim","t","s_time","m2_freq","s","m","r","freq_0-30kb","freq_30-60kb","freq_60-100kb","MEAN","SD","MAX"]

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

total_count = file_len(abc_files)

outfile = open(output_txt,"w")

outfile.writelines(i+"\t" for i in title)
outfile.write("\n")
outfile.flush()


infile = open(abc_files,"r")
line_count = 0
rep_count=0
while line_count < total_count:
    line = infile.readline()
    

    info = line.split()
    if ">" in line:
        info_line = info
        rep_count += 1
        line_count +=1
        i=0
        
    length_all = []
    lengthlong_all = []
    while i <296:
        len_line = infile.readline()
        fields = len_line.split()[1:]
        fields = [int(num) for num in fields]
        
        if len(fields) == 0:
            length = 0
            length_long = 0
        elif len(fields) == 2:
            length = length_long = fields[1]-fields[0]
        elif len(fields) == 4:
            length = (fields[1]-fields[0]) + (fields[3]-fields[2])
            length_long = max([fields[1]-fields[0],fields[3]-fields[2]])
        elif len(fields) ==6:
            length = (fields[1]-fields[0]) + (fields[3]-fields[2])+ (fields[5]-fields[4])
            length_long = max([fields[1]-fields[0],fields[3]-fields[2],fields[5]-fields[4]])
        
        length_all.append(length)
        lengthlong_all.append(length_long)
        i+=1
        line_count +=1
        
    length_all = np.array(length_all)
    lengthlong_all = np.array(lengthlong_all)
        
    random156 = random.sample(list(range(0,296)),156) #sample 156 haplotypes/78 individuals to match observed data 
    length_156 = length_all[random156]
    lengthlong_156 = lengthlong_all[random156]
    
    freq_0_30kb = sum((length_156 >=0) & (length_156 <30000))/156
    freq_30_60kb = sum((length_156 >=30000) & (length_156 <60000))/156
    freq_60_100kb = sum(length_156 >=60000)/156
    #freq_60_80kb = sum((length_156 >=60000) & (length_156 <80000))/156
    #freq_80_100kb = sum(lengthlong_156 >=80000)/156
    mean = np.mean(length_156)
    sd = np.std(length_156)
    maxlen = max(length_156)
        
    info_line=info_line+[freq_0_30kb,freq_30_60kb,freq_60_100kb,mean,sd,maxlen]
      
    info_line[0] = info_line[0][1:]
    this = [rep_count]+info_line
    
    outfile.writelines(str(i)+"\t" for i in this)
    outfile.write("\n")
    outfile.flush()
    
    if rep_count%10000 == 0:
        print(rep_count)
        

    
infile.close()
outfile.close()   
    

            
