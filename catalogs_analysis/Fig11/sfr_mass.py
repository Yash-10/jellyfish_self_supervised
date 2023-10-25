#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Carolina Queiroz"
__maintainer__ = "Carolina Queiroz"
__email__ = "c.queirozabs@gmail.com"
__status__ = "Production"

"""

Analysis
- sfr: Amanda (nov/2022) -> only JClass = 4
- mass: Cigale (10nov) -> IMF Chabrier

Last modification: Nov 22nd
This version: Nov 10th 2022

"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from os import path
import pandas as pd
import sys

#-------------------------------------------------------------------------------

def main():

  import seaborn as sns
  sns.set_context("paper", font_scale = 2)
  sns.set_style('ticks')

  N_filters = 12

  topdir = os.getcwd()

  # Read file with information from NED + S-PLUS photometry + CIGALE fits
  # JClass 4
  info1 = pd.read_csv(topdir + '/jellyfish_jclass4_nov2022_sfr_mass.csv')

  Ncand = len(info1['id'])
  Gname = info1['Galaxy']
  Cluster1 = info1['Cluster']
  Jclass = info1['JClass']
  dist = info1['distance']
  zspec = info1['redshift']
  SFR1 = info1['SFR(Reff)'] #Amanda
  Mstar1 = info1['M_star'] #Cigale

  ind_antlia_j4 = np.where(Cluster1=='Antlia')[0]
  ind_fornax_j4 = np.where(Cluster1=='Fornax')[0]
  ind_hydra_j4 = np.where(Cluster1=='Hydra')[0]

  # JClass 3
  info2 = pd.read_csv(topdir + '/jellyfish_jclass3_nov2022_sfr_mass.csv')

  Ncand = len(info2['id'])
  Gname = info2['Galaxy']
  Cluster2 = info2['Cluster']
  Jclass = info2['JClass']
  dist = info2['distance']
  zspec = info2['redshift']
  SFR2 = info2['SFR(Reff)'] #Amanda
  Mstar2 = info2['M_star'] #Cigale

  ind_antlia_j3 = np.where(Cluster2=='Antlia')[0]
  ind_fornax_j3 = np.where(Cluster2=='Fornax')[0]
  ind_hydra_j3 = np.where(Cluster2=='Hydra')[0]

  # Let's plot!

  markersize = 15

  # ssfr
  plt.figure(figsize=(10,6))
  
  # JClass 4
  plt.plot(Mstar1[ind_antlia_j4[0]],SFR1[ind_antlia_j4[0]]/Mstar1[ind_antlia_j4[0]],markerfacecolor='#7f2704',marker='o',markeredgecolor='none',linestyle='none',zorder=1,label='$\mathrm{JClass} \, 4$', markersize=markersize,alpha=0.9)
  plt.plot(Mstar2[ind_antlia_j3[0]],SFR2[ind_antlia_j3[0]]/Mstar2[ind_antlia_j3[0]],markerfacecolor='#fd8d3c',marker='o',markeredgecolor='none',linestyle='none',zorder=1,label='$\mathrm{JClass} \, 3$', markersize=markersize,alpha=0.9)

  plt.plot(Mstar1[ind_antlia_j4[0]],SFR1[ind_antlia_j4[0]]/Mstar1[ind_antlia_j4[0]],markerfacecolor='none',marker='o',markeredgecolor='k',markeredgewidth=1.3,markersize=markersize,linestyle='none',zorder=1,label='$\mathrm{Antlia}$')
  
  for i in ind_antlia_j4:
    plt.plot(Mstar1[i],SFR1[i]/Mstar1[i],markerfacecolor='#7f2704',marker='o',markeredgecolor='k',markeredgewidth=1.3,markersize=markersize,linestyle='none',zorder=1,alpha=0.9)
  
  plt.plot(Mstar1[ind_fornax_j4[0]],SFR1[ind_fornax_j4[0]]/Mstar1[ind_fornax_j4[0]],markerfacecolor='none',marker='D',markeredgecolor='k',markeredgewidth=1.3,markersize=markersize,linestyle='none',zorder=2,label='$\mathrm{Fornax}$', alpha=0.9) 
  for i in ind_fornax_j4:  
    plt.plot(Mstar1[i],SFR1[i]/Mstar1[i],markerfacecolor='#7f2704',marker='D',markeredgecolor='k',markeredgewidth=1.3,markersize=markersize,linestyle='none',zorder=2,alpha=0.9)
  
  plt.plot(Mstar1[ind_hydra_j4[0]],SFR1[ind_hydra_j4[0]]/Mstar1[ind_hydra_j4[0]],markerfacecolor='none',marker='s',markeredgecolor='k',markeredgewidth=1.3,markersize=markersize,linestyle='none',zorder=3,label='$\mathrm{Hydra}$', alpha=0.9)   
  for i in ind_hydra_j4: 
    plt.plot(Mstar1[i],SFR1[i]/Mstar1[i],markerfacecolor='#7f2704',marker='s',markeredgecolor='k',markeredgewidth=1.3,markersize=markersize,linestyle='none',zorder=3,alpha=0.9)   

  # JClass 3
  for i in ind_antlia_j3:
    plt.plot(Mstar2[i],SFR2[i]/Mstar2[i],markerfacecolor='#fd8d3c',marker='o',markeredgecolor='k',markeredgewidth=1.3,markersize=markersize,linestyle='none',zorder=1,alpha=0.9)
  
  for i in ind_fornax_j3:  
    plt.plot(Mstar2[i],SFR2[i]/Mstar2[i],markerfacecolor='#fd8d3c',marker='D',markeredgecolor='k',markeredgewidth=1.3,markersize=markersize,linestyle='none',zorder=2,alpha=0.9)
  
  for i in ind_hydra_j3: 
    plt.plot(Mstar2[i],SFR2[i]/Mstar2[i],markerfacecolor='#fd8d3c',marker='s',markeredgecolor='k',markeredgewidth=1.3,markersize=markersize,linestyle='none',zorder=3,alpha=0.9)

  plt.xlabel(r'$\mathrm{M_{\star} \ (M_{\odot})}$', fontsize=30)
  plt.ylabel(r'$\mathrm{sSFR \ (yr^{-1})}$', fontsize=30)
  #plt.xlim(10.4,10.6)
  #plt.xticks([10.4,10.45,10.5,10.55,10.6])
  plt.xscale('log')
  plt.yscale('log')
  plt.tick_params(axis="x", labelsize=24)
  plt.tick_params(axis="y", labelsize=24)
  lgnd = plt.legend(loc='upper right', prop={'size':20}, numpoints=1)
  lgnd.legendHandles[0]._sizes = [markersize]
  lgnd.legendHandles[1]._sizes = [markersize]
  plt.grid()
  plt.savefig(topdir+'/sSFR_Mstar_jellyfish_jclass3_4_v2.png',dpi=250,bbox_inches='tight')
  #plt.close('all')
  plt.show()

  sys.exit(-1)

  # ssfr
  plt.figure(figsize=(10,6))
  
  # JClass 4
  plt.plot(Mstar1[ind_antlia_j4[0]],SFR1[ind_antlia_j4[0]]/Mstar1[ind_antlia_j4[0]],markerfacecolor='none',marker='o',markeredgecolor='k',markeredgewidth=1.3,markersize=8,linestyle='none',zorder=1,label='$\mathrm{Antlia}$')
  for i in ind_antlia_j4:
    plt.plot(Mstar1[i],SFR1[i]/Mstar1[i],markerfacecolor='orangered',marker='o',markeredgecolor='k',markeredgewidth=1.3,markersize=8,linestyle='none',zorder=1)
  
  plt.plot(Mstar1[ind_fornax_j4[0]],SFR1[ind_fornax_j4[0]]/Mstar1[ind_fornax_j4[0]],markerfacecolor='none',marker='D',markeredgecolor='k',markeredgewidth=1.3,markersize=8,linestyle='none',zorder=2,label='$\mathrm{Fornax}$') 
  for i in ind_fornax_j4:  
    plt.plot(Mstar1[i],SFR1[i]/Mstar1[i],markerfacecolor='darkorange',marker='D',markeredgecolor='k',markeredgewidth=1.3,markersize=8,linestyle='none',zorder=2)
  
  plt.plot(Mstar1[ind_hydra_j4[0]],SFR1[ind_hydra_j4[0]]/Mstar1[ind_hydra_j4[0]],markerfacecolor='none',marker='s',markeredgecolor='k',markeredgewidth=1.3,markersize=8,linestyle='none',zorder=3,label='$\mathrm{Hydra}$')   
  for i in ind_hydra_j4: 
    plt.plot(Mstar1[i],SFR1[i]/Mstar1[i],markerfacecolor='salmon',marker='s',markeredgecolor='k',markeredgewidth=1.3,markersize=8,linestyle='none',zorder=3)   

  # JClass 3
  for i in ind_antlia_j3:
    plt.plot(Mstar2[i],SFR2[i]/Mstar2[i],markerfacecolor='b',marker='o',markeredgecolor='k',markeredgewidth=1.3,markersize=8,linestyle='none',zorder=1)
  
  for i in ind_fornax_j3:  
    plt.plot(Mstar2[i],SFR2[i]/Mstar2[i],markerfacecolor='skyblue',marker='D',markeredgecolor='k',markeredgewidth=1.3,markersize=8,linestyle='none',zorder=2)
  
  for i in ind_hydra_j3: 
    plt.plot(Mstar2[i],SFR2[i]/Mstar2[i],markerfacecolor='darkblue',marker='s',markeredgecolor='k',markeredgewidth=1.3,markersize=8,linestyle='none',zorder=3)

  plt.xlabel(r'$\mathrm{M_{\star} \ (M_{\odot})}$', fontsize=40)
  plt.ylabel(r'$\mathrm{sSFR \ (yr^{-1})}$', fontsize=40)
  #plt.xlim(10.4,10.6)
  #plt.xticks([10.4,10.45,10.5,10.55,10.6])
  plt.xscale('log')
  plt.yscale('log')
  plt.tick_params(axis="x", labelsize=24)
  plt.tick_params(axis="y", labelsize=24)
  plt.legend(loc='lower right', prop={'size':20}, numpoints=1)
  plt.grid()
  plt.savefig(topdir+'/sSFR_Mstar_jellyfish_jclass3_4.pdf',format='pdf',dpi=1000,bbox_inches='tight')
  #plt.close('all')
  plt.show()

  sys.exit(-1)

  # sfr
  plt.figure(figsize=(10,6))
  
  for i in ind_antlia:
    plt.plot(Mstar[i],SFR[i],markerfacecolor='darkblue',marker='o',markeredgecolor='none',markersize=8,linestyle='none',zorder=1,label='$\mathrm{Antlia}$')
  
  plt.plot(Mstar[ind_fornax[0]],SFR[ind_fornax[0]],markerfacecolor='dodgerblue',marker='D',markeredgecolor='none',markersize=8,linestyle='none',zorder=2,label='$\mathrm{Fornax}$') 
  for i in ind_fornax:  
    plt.plot(Mstar[i],SFR[i],markerfacecolor='dodgerblue',marker='D',markeredgecolor='none',markersize=8,linestyle='none',zorder=2)
  
  for i in ind_hydra: 
    plt.plot(Mstar[i],SFR[i],markerfacecolor='tab:cyan',marker='s',markeredgecolor='none',markersize=8,linestyle='none',zorder=3,label='$\mathrm{Hydra}$')   

  plt.xlabel(r'$\mathrm{M_{\star} \ (M_{\odot})}$', fontsize=40)
  plt.ylabel(r'$\mathrm{SFR \ (M_{\odot} yr^{-1})}$', fontsize=40)
  #plt.xlim(10.4,10.6)
  #plt.xticks([10.4,10.45,10.5,10.55,10.6])
  plt.ylim(0.009,1.1)
  plt.xscale('log')
  plt.yscale('log')
  plt.tick_params(axis="x", labelsize=24)
  plt.tick_params(axis="y", labelsize=24)
  plt.legend(loc='lower right', prop={'size':20}, numpoints=1)
  plt.grid()
  plt.savefig(topdir+'/SFR_Mstar_jellyfish_jclass4.pdf',format='pdf',dpi=1000,bbox_inches='tight')
  #plt.close('all')
  plt.show()

#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------

if __name__ == '__main__':
     main()


