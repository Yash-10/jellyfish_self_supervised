#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Carolina Queiroz"
__maintainer__ = "Carolina Queiroz"
__email__ = "c.queirozabs@gmail.com"
__status__ = "Production"

'''

Diagram phase for jellyfish galaxies

Last modification: Nov 22nd
This version: Nov 4th 2022

'''

from astropy.coordinates import SkyCoord
from astropy import units as u
from matplotlib import cm
from matplotlib import colors
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import os
from os import path
import pandas as pd
from scipy import interpolate
import sys

#-------------------------------------------------------------------------------

# Calculate distance between two galaxies
def CalcDist1(RA1,Dec1,d1,RA2,Dec2,d2):

    ra1 = RA1*np.pi/180.
    dec1 = Dec1*np.pi/180.
    ra2 = RA2*np.pi/180.
    dec2 = Dec2*np.pi/180. 

    raiz = np.sin((dec2-dec1)/2.)**2+np.cos(dec1)*np.cos(dec2)*np.sin((ra2-ra1)/2.)**2
    theta = 2*np.arcsin(np.sqrt(raiz))

    d12 = np.sqrt(d1**2*np.sin(theta)**2+(d2-d1*np.cos(theta))**2)

    return (d12)

# Cartesian
def CalcDist(RA1,Dec1,d1,RA2,Dec2,d2):

    ra1 = RA1*np.pi/180.
    dec1 = Dec1*np.pi/180.
    ra2 = RA2*np.pi/180.
    dec2 = Dec2*np.pi/180. 

    x1 = d1*np.cos(ra1)*np.cos(dec1)
    y1 = d1*np.sin(ra1)*np.cos(dec1)
    z1 = d1*np.sin(dec1)

    x2 = d2*np.cos(ra2)*np.cos(dec2)
    y2 = d2*np.sin(ra2)*np.cos(dec2)
    z2 = d2*np.sin(dec2)

    d12 = np.sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)

    return (d12)

# Peculiar line-of-sight rest-frame velocity
def Deltav(z):

    D = cLight*(z-z_cl)/(1+z_cl)

    return (D)

def main():

    import seaborn as sns
    sns.set_context("paper", font_scale = 2)
    sns.set_style('ticks')

    # Define some constants
    global cLight,z_cl
    cLight = 2.997992458e5  # km/s

    # Define some paths
    topdir = os.getcwd()

    # Antlia
    root = 'Antlia'
    RA_cl = 157.5125 # deg
    dec_cl = -35.3233 # deg
    coord_cl = SkyCoord(RA_cl*u.degree, dec_cl*u.degree)
    z_cl = 0.0087
    Dl_cl = 40.7 # Mpc (luminosity distance from z)
    R200 = 0.88 # Mpc
    kpc_scale = Dl_cl*1000*np.tan((1./3600.)*np.pi/180.)  # each arcsec is kpc_scale in Antlia
    sigma_v = 591 # km/s

    df = pd.read_csv(topdir + '/SPLUS_new_jellyfish_candidates_velocity_nov2022_Antlia.csv')

    JClass = df['JClass']
    RA_gal = df['RA']
    dec_gal = df['dec']
    dgal = df['distance(Mpc)']
    zgal = df['redshift']
    Ngal = len(dgal)

    Rp = np.zeros(Ngal)
    Dv_los = np.zeros(Ngal)
    ind_gal = []

    for obj in range(Ngal):
        if dgal[obj] != -99. and zgal[obj] != -99.:
            ind_gal.append(obj)
            coord_gal = SkyCoord(RA_gal[obj]*u.degree, dec_gal[obj]*u.degree)
            Rp[obj] = (coord_cl.separation(coord_gal).arcsecond * kpc_scale)/(R200*1000)
            Dv_los[obj] = Deltav(zgal[obj])

    # Let's plot!

    # Color bar
    bounds = [1,2,3,4]

    my_colors = np.array(['#7f2704', '#d94801', '#fd8d3c', '#fdd0a2']) #'#CC79A7':pink,'#F0E442':yellow,'#999999':gray,'#4daf4a':green
    my_colors = my_colors[::-1]
    my_cmap = ListedColormap(my_colors)
    
    # Boundary 1
    x1 = np.arange(0,1.25,0.05)
    y1a = 1.5-(1.5/1.2)*x1
    y1b = -1.5+(1.5/1.2)*x1

    # Boundary 2
    x2 = np.arange(0,0.55,0.05)
    y2a = 2.-(2./0.5)*x2
    y2b = -2.+(2./0.5)*x2

    plt.figure(figsize=(12,5))

    plt.plot(x1,y1a,'k',linestyle='--',linewidth=1.2,zorder=1)
    plt.plot(x1,y1b,'k',linestyle='--',linewidth=1.2,zorder=1)
    plt.plot(x2,y2a,'k',linestyle=':',linewidth=1.2,zorder=1)
    plt.plot(x2,y2b,'k',linestyle=':',linewidth=1.2,zorder=1)

    plt.scatter(Rp[ind_gal]/R200,Dv_los[ind_gal]/sigma_v,c=JClass[ind_gal],cmap=my_cmap,s=250,zorder=2,alpha=0.9,edgecolors='black')
    
    plt.xlabel('$\mathrm{R_{p}/R_{200}}$',fontsize=35)
    plt.ylabel('$\Delta\mathrm{V_{los}/} \sigma_\mathrm{v}}$',fontsize=35)
    #plt.xscale('log')
    plt.xlim(-0.1,1.5)
    plt.ylim(-2, 3)
    plt.tick_params(axis="x", labelsize=24)
    plt.tick_params(axis="y", labelsize=24)

    cbar = plt.colorbar(ticks=bounds)
    cbar.set_label('$\mathrm{JClass}$',size=30)
    cbar.ax.tick_params(labelsize=24)
    plt.grid()
    plt.text(0.85*1.5, 0.85*(-2), 'Antlia', fontsize=23)
    plt.savefig('Phase_diagram_'+root+'.png',dpi=250,bbox_inches='tight')
    plt.show()

#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------

if __name__ == '__main__':
     main()
