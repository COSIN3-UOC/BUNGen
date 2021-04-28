#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 11:07:47 2018

@author: mariapalazzi
"""
import numpy as np
from network_generator import  network_generator
import glob
import sys
#%%
def exploring_transitions(parameters):
    '''
    function to generate synthetic unipartite networks with different planted partitions employing 
    the benchmark model introduced by ASR et al, PRE 2018, and performs structural analysis by means 
    of calculation of nestedness, modularity and in-block nestedness.
    
    The function generate a file containing the generated network in matrix format

    Inputs:
    ----------
       parameters: list of lists
       
       parameters[0]:
           a boolean to indicate if you want to generate bipartite (True) or unipartite (False) networks
       parameters[1]:
           number of row nodes per block
       parameters[2]:
           number of colum nodes per block
       parameters[3]:
           number of blocks 
       parameters[4]:
           shape parameter
       parameters[5]:
          parameter that determines the amount of links outside the perfect nested structure
       parameters[6]:
           parameter that determines the amount of inter blocks links 
    '''
    
    if parameters[0]==False:
        r=int(parameters[1]) # number of rows/cols per block
        print('B',parameters[2])
        B=float(parameters[2])
        xi= float(parameters[3]) #xi parameter
        xi=round(xi,2)
        P_ = float(parameters[4]) # noise parameter outside the perfect nested: this range can be modified as desired
        P_=round(P_,2)
        mu_ = float(parameters[5]) # inter block noise parameter
        mu_=round(mu_,2)
        name="edges_synthetic_unipartite_B"+str(B)+"_xi"+str(xi)+"_P"+str(P_)+"_mu"+str(mu_)+".csv"
        if len(glob.glob(name))==0:
            #generating the synthetic networks (np.fill and np. triu for unipartite)
            M_=network_generator(r,r,B,xi,P_,mu_)
            np.savetxt(name, M_.astype(int), fmt="%d",delimiter=',')
    
    if parameters[0]==True:
        r=int(parameters[1]) # number of rows per block
        cl=int(parameters[2]) # number of cols per block
        print('B',parameters[3])
        B=float(parameters[3])
        xi= float(parameters[4]) #xi parameter
        xi=round(xi,2)
        P_ = float(parameters[5]) # noise parameter outside the perfect nested: this range can be modified as desired
        P_=round(P_,2)
        mu_ = float(parameters[6]) # inter block noise parameter
        mu_=round(mu_,2)
        name="edges_synthetic_bipartite_B"+str(B)+"_xi"+str(xi)+"_P"+str(P_)+"_mu"+str(mu_)+".csv"
        if len(glob.glob(name))==0:
            #generating the synthetic networks (np.fill and np. triu for unipartite)
            M_=network_generator(r,cl,B,xi,P_,mu_)
            np.savetxt(name, M_.astype(int), fmt="%d",delimiter=',')
#%%
if __name__ == '__main__':
    parameters=list()
    if len(sys.argv)==6:
        bipartite=False
        rw=int(sys.argv[1])
        B=float(sys.argv[2])
        xi=float(sys.argv[3])
        P=float(sys.argv[4])
        mu=float(sys.argv[5])
        parameters=[bipartite,rw,B,xi,P,mu] 
#        print('param', parameters)
#        print(type(parameters[0]))
#        print(type(parameters[1]))
#        print(type(parameters[2]))
#        print(type(parameters[3]))
#        print(type(parameters[4]))
#        print(type(parameters[5]))
    if len(sys.argv)==7:
        bipartite=True
        rw=int(sys.argv[1])
        cl=int(sys.argv[2])
        B=float(sys.argv[3])
        xi=float(sys.argv[4])
        P=float(sys.argv[5])
        mu=float(sys.argv[6])
        parameters=[bipartite,rw,cl,B,xi,P,mu] 
#        print('param', parameters)
#        print(type(parameters[0]))
#        print(type(parameters[1]))
#        print(type(parameters[2]))
#        print(type(parameters[3]))
#        print(type(parameters[4]))
#        print(type(parameters[5]))
#        print(type(parameters[6]))
    exploring_transitions(parameters)