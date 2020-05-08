# -*- coding: utf-8 -*-
"""

@author: Dylan Kearns
Date April 2020
Contact : dylan.kearns@ucdconnect.ie

This python script is my implementation of frequency band reduction 
using chromagrams. 

"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# AUTHOR: Delia Fano Yela
# DATE:  February 2019
# CONTACT: d.fanoyela@qmul.ac.uk
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import the libraries that help for this experiment,
# however, none are strictly necessary here
import os
import librosa
from librosa import load, display
import numpy as np
import pandas as pd
# import seaborn as sns
import sklearn.metrics
import time
import sys
import networkx as nx
import scipy
#from tabulate import tabulate
# Import the Divide & Conquer natural visibility graph implementation (necessary)
from visibility_algorithms import nvg_dc
import matplotlib.pyplot as plt  #
import sounddevice as sd
import time
start = time.time()

# Testing parameters:
dir = "AUDIO/synth_dataset"
snr_values = [-24, -12, -6, -3, 0, 3, 6]
metrics = ['euclidean', 'cosine']

# Define the divisions for the audio file corresponding to the following MIDI notes:
# [A2,B2,C3,D3,E3,F3,G3,A3,B3,C4,D4,E4,F4,G4 ]
L = 22000               # length of a note in samples
div = list(range(0,2*L,L)) + list(range(573000,(12*L + 573000), L))


# INITS ------------------------------------------------------------------------
# Max frequency is sampling rate 44100, and the FFT size is 16384; so the frequency increase is 44100/16384 = 2.7 Hz
# The highest note in the dataset is C5, 523.25Hz. If we want to include 10 harmonics we need 523.25 * 10 / 2.7 bins of
# the FFT, 1943.974 bins, rounding up to 2000.
Nf = 16384          # FFT size in samples
# N = 2000            # number of bins of interest in the FFT
N = 500       # number of bins of interest in the FFT
mu, sigma = 0, 1    # mean and standard deviation for gaussian noise


# Pandas DataFrame that will contain the results
df_mrr = pd.DataFrame({'MRR':[], 'ftype':[], 'dtype':[], 'SNR':[]})

for dis in metrics:
    print( "/n Distance Metric: ", dis)

    for snr in snr_values:
        print( "/n SNR:", snr)
        # Start processing--------------------------------------------------------------
    
        A = np.empty([396,0])
        K = np.empty([396,0])
        P = np.empty([396,0])
        


        for subdir, dirs, files in os.walk(dir):
    
            for i, filename in enumerate(files):
                signal, fs = load(os.path.join(subdir, filename), sr=44100, mono=True)

                sys.stdout.write('\r')
                # the exact output you're looking for:
                sys.stdout.write("[%s%s] %d%%" % ('='*i, ' '*((len(files)-1)-i), 100.*i/(len(files)-1)))
                sys.stdout.flush()
                time.sleep(.25)
                


                for index in div:
                    #print(index)
                    # TIME DOMAIN ---------------------------------------------------
                    s = signal[index:index + Nf]
                    n = np.random.normal(mu, sigma, Nf)
                    #sfc = snr_scaling_factor( signal = s, noise = n, SNRdB = snr )
                    sfc = np.sqrt(np.sum(s**2)/np.sum(n**2))*10**(-snr/20.)
                    m = s + sfc*n
         
         

                    # STFT DOMAIN ---------------------------------------------------
                    sa = np.abs(librosa.stft(s))
                    ma = np.abs(librosa.stft(m))
                    
                    #Chromagram construction -----------------------------------------
                    chroma_clean = np.array(librosa.feature.chroma_stft(S=sa, sr=44100))
                    chroma_noisy = np.array(librosa.feature.chroma_stft(S=ma, sr=44100))
                    
                    #Used to construct fig 2.3 & 2.4 in report
# =============================================================================
#                     N = 12
#                     x = chroma_clean[1]
#                    
#                     ind = np.arange(N)    # the x locations for the groups
#                     width = 0.35       # the width of the bars: can also be len(x) sequence
#                     plt.bar(ind, x, width)
#                     plt.ylabel('chroma value')
#                     plt.xlabel('nodes in visibility graph')
#                     plt.show()
# =============================================================================

                    #Visibility graphs-----------------------------------------
                    vis_clean = []
                    for row in chroma_clean.TT:
                       # print(row)
                        sn = nvg_dc(series = row.tolist() , timeLine = range(len(row)), left = 0, right = len(row))
                        vis_clean.append(sn)
                    vis_clean = np.array(vis_clean)
    
                    
                    vis_noisy = []
                    for row in chroma_noisy.T:
                        mn = nvg_dc(series = row.tolist() , timeLine = range(len(row)), left = 0, right = len(row))
                        vis_noisy.append(mn)
                    vis_noisy = np.array(vis_noisy)
                    
                    
                    #Visibility graph construction & degree sequence ------------------------------------------
                    clean_graph = nx.Graph()
                    full_deg_clean = []
                    sp = []
                    for graph in vis_clean:
                        for node in graph:
                            clean_graph.add_node(node[0])
                            for edge in node[1]:
                                clean_graph.add_edge(node[0], edge)
                        
                        degree_sequence_clean = sorted([d for n, d in clean_graph.degree()], reverse=True)# degree sequence
                        full_deg_clean.append(degree_sequence_clean)
                    full_deg_clean = np.array(full_deg_clean)
                    
                    
                    noisy_graph = nx.Graph()
                    full_deg_noisy = []
                    mp = []
                    for graph in vis_noisy:
                        for node in graph:
                            noisy_graph.add_node(node[0])
                            for edge in node[1]:
                                noisy_graph.add_edge(node[0], edge)
                        
                        degree_sequence_noisy = sorted([d for n, d in noisy_graph.degree()], reverse=True)# degree sequence
                        full_deg_noisy.append(degree_sequence_noisy)
                    full_deg_noisy = np.array(full_deg_noisy)
                    
                    #Degree distribution --------------------------------------
                    sp = []
                    for row in full_deg_clean:
                        spp = np.bincount(row.astype(int), minlength = len(row)).astype('float64') / len(row)
                        sp.append(spp)
                    
                    mp = []
                    for row in full_deg_noisy:
                        mpp = np.bincount(row.astype(int), minlength = len(row)).astype('float64') / len(row)
                        mp.append(mpp)
                    
                    
                    chroma_clean = chroma_clean.flatten()
                    chroma_noisy = chroma_noisy.flatten()

                    
                    full_deg_clean = full_deg_clean.flatten()
                    full_deg_noisy = full_deg_noisy.flatten()
                    
                    sp = np.array(sp)
                    mp = np.array(mp)
                    
                    sp = sp.flatten()
                    mp = mp.flatten()

                    #Stack in processing matrix
                    A = np.hstack((A, chroma_clean[:,None], chroma_noisy[:,None]))
                    K = np.hstack((K, full_deg_clean[:,None], full_deg_noisy[:,None]))
                    P = np.hstack((P, sp[:,None], mp[:,None]))
         

        
        # SELF-SIMILARITY MATRIX -----------------------------------
        dA = sklearn.metrics.pairwise_distances(np.transpose(A), metric=dis)
        dK = sklearn.metrics.pairwise_distances(np.transpose(K), metric=dis)
        dP = sklearn.metrics.pairwise_distances(np.transpose(P), metric=dis)
        # Sort -----------------------------------------------------
        dA_s = np.argsort(dA, axis = 1)
        dK_s = np.argsort(dK, axis = 1)
        dP_s = np.argsort(dP, axis = 1)
        # MEAN RECIPROCAL RANK -------------------------------------
        noisy = range(1,dA.shape[0],2)
        clean = range(0,dA.shape[0],2)

        retrieved_rank_A = [np.where(dA_s[clean[i],:] == noisy[i])[0][0] for i in range(0,len(clean)) ]
        retrieved_rank_K = [np.where(dK_s[clean[i],:] == noisy[i])[0][0] for i in range(0,len(clean)) ]
        retrieved_rank_P = [np.where(dP_s[clean[i],:] == noisy[i])[0][0] for i in range(0,len(clean)) ]
        
        mrrA = np.mean(1./np.array(retrieved_rank_A))
        mrrK = np.mean(1./np.array(retrieved_rank_K))
        mrrP = np.mean(1./np.array(retrieved_rank_P))

        print( "\nmean reciprocal rank A: ", np.mean(1./np.array(retrieved_rank_A)))       
        print( "mean reciprocal rank K: ", np.mean(1./np.array(retrieved_rank_K)))


        # Store ----------------------------------------------------
        df_mrr = df_mrr.append(pd.DataFrame([[mrrA, 'spectrum', dis, snr]], columns = ['MRR', 'ftype', 'dtype', 'SNR']), ignore_index=True)#,sort=True)
        df_mrr = df_mrr.append(pd.DataFrame([[mrrK, 'degree', dis, snr]], columns = ['MRR', 'ftype', 'dtype', 'SNR']), ignore_index=True)#,sort=True)
        df_mrr = df_mrr.append(pd.DataFrame([[mrrP, 'degree distribution', dis, snr]], columns = ['MRR', 'ftype', 'dtype', 'SNR']), ignore_index=True)#,sort=True)

#df_mrr.to_csv ('df_mrr_exp01_chroma.csv', index = None, header=True)
print('It took', time.time()-start, 'seconds.')
