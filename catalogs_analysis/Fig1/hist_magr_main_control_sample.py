#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Carolina Queiroz"
__maintainer__ = "Carolina Queiroz"
__email__ = "c.queirozabs@gmail.com"
__status__ = "Production"

'''

r-band magnitude histogram

Last modification: Nov 25th
This version: Nov 25th 2022

'''

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from os import path
import pandas as pd
from scipy import interpolate
import sys

#-------------------------------------------------------------------------------

def main():

    # Define some constants
    global cLight,z_cl
    cLight = 2.997992458e5  # km/s

    # Define some paths
    topdir = os.getcwd()

    df1 = pd.read_csv(topdir + '/main_magr.csv')
    magr1 = df1['R_AUTO']

    df2 = pd.read_csv(topdir + '/control_magr.csv')
    magr2 = df2['R_AUTO']

    df3 = pd.read_csv(topdir + '/non_jellyfish_magr.csv')
    magr3 = df3['magnitude']

    # Concatenate magr1 and magr3
    magr_main = np.asarray(magr1).tolist() + np.asarray(magr3).tolist()

    # Let's plot!
    print (min(magr1),max(magr1))
    print (min(magr2),max(magr2))
    print (min(magr3),max(magr3))
    print (min(magr_main),max(magr_main))

    mag_bin = np.arange(9,20.5,0.5)

    import seaborn as sns
    sns.set_context("paper", font_scale = 2)
    sns.set_style('whitegrid')
    #sns.set_style('ticks')

    fig, ax = plt.subplots(figsize=(9,5))

    ax.hist(magr_main,bins=mag_bin,color='#9ecae1',edgecolor='None',alpha=0.85,label='Main')
    ax.hist(magr2,bins=mag_bin,color='k',histtype='step',label='Control')

    ax.set_xlabel('$r_{SDSS}$',fontsize=30)
    ax.set_xlim(8,22)
    ax.set_ylim(0,23)
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.legend(loc=0, prop={'size':20}, numpoints=1)

    plt.savefig('Histogram_magr_main_control.png',dpi=250,bbox_inches='tight')
    plt.show()

#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------

if __name__ == '__main__':
     main()
