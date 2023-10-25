#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Carolina Queiroz"
__maintainer__ = "Carolina Queiroz"
__email__ = "c.queirozabs@gmail.com"
__status__ = "Production"

'''

Diagram phase for jellyfish galaxies

Last modification: Apr 10th
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

    # Fornax
    root = 'Fornax'
    RA_cl = 54.5000 # deg
    dec_cl = -35.4500 # deg
    coord_cl = SkyCoord(RA_cl*u.degree, dec_cl*u.degree)
    z_cl = 0.0046
    Dl_cl = 19.9 # Mpc (luminosity distance from z)
    R200 = 0.7 # Mpc
    kpc_scale = Dl_cl*1000*np.tan((1./3600.)*np.pi/180.)  # each arcsec is kpc_scale in Fornax
    sigma_v = 370. # km/s

    # Main
    df1 = pd.read_csv(topdir + '/SPLUS_new_jellyfish_candidates_velocity_nov2022_'+root+'_main.csv')

    JClass1 = df1['JClass']
    RA_gal1 = df1['RA']
    dec_gal1 = df1['dec']
    dgal1 = df1['distance(Mpc)']
    zgal1 = df1['redshift']
    Ngal1 = len(dgal1)

    Rp1 = np.zeros(Ngal1)
    Dv_los1 = np.zeros(Ngal1)
    ind_gal1 = []
    for obj in range(Ngal1):
        if dgal1[obj] != -99. and zgal1[obj] != -99.:
            ind_gal1.append(obj)
            #Rp1[obj] = CalcDist(RA_gal1[obj],dec_gal1[obj],dgal1[obj],RA_cl,dec_cl,Dl_cl)
            coord_gal1 = SkyCoord(RA_gal1[obj]*u.degree, dec_gal1[obj]*u.degree)
            Rp1[obj] = (coord_cl.separation(coord_gal1).arcsecond * kpc_scale)/(R200*1000)
            Dv_los1[obj] = Deltav(zgal1[obj])

    # Control
    df2 = pd.read_csv(topdir + '/SPLUS_new_jellyfish_candidates_velocity_nov2022_'+root+'_control.csv')

    JClass2 = df2['JClass']
    RA_gal2 = df2['RA']
    dec_gal2 = df2['dec']
    dgal2 = df2['distance(Mpc)']
    zgal2 = df2['redshift']
    Ngal2 = len(dgal2)

    Rp2 = np.zeros(Ngal2)
    Dv_los2 = np.zeros(Ngal2)
    ind_gal2 = []
    for obj in range(Ngal2):
        if dgal2[obj] != -99. and zgal2[obj] != -99.:
            ind_gal2.append(obj)
            #Rp2[obj] = CalcDist(RA_gal2[obj],dec_gal2[obj],dgal2[obj],RA_cl,dec_cl,Dl_cl)
            coord_gal2 = SkyCoord(RA_gal2[obj]*u.degree, dec_gal2[obj]*u.degree)
            Rp2[obj] = (coord_cl.separation(coord_gal2).arcsecond * kpc_scale)/(R200*1000)
            Dv_los2[obj] = Deltav(zgal2[obj])

    ind_outlier_main = np.where(Dv_los1>=5)[0]
    ind_outlier_control = np.where(Dv_los2>=5)[0]

    for i in ind_outlier_main:
        print (i,RA_gal1[i],dec_gal1[i])

    for i in ind_outlier_control:
        print (i,RA_gal2[i],dec_gal2[i])

    # Non-jellyfish galaxies
    nj1 = pd.read_csv(topdir+'/SPLUS_non_jellyfish_Fornax_sample_oct2022.csv')
    nj2 = pd.read_csv(topdir+'/SPLUS_non_jellyfish_Fornax_control_oct2022.csv')

    RA_nj1 = nj1['RA']
    dec_nj1 = nj1['dec']
    RA_nj2 = nj2['RA']
    dec_nj2 = nj2['dec']

    c = SkyCoord(ra="3:38:00", dec="-35:27:00", unit=(u.hourangle, u.degree)) #NGC1399
    RA_center = (float(c.ra.to_string(decimal=True)))
    dec_center = (float(c.dec.to_string(decimal=True)))

    c = SkyCoord(ra="3:22:41.7", dec="-37:12:30", unit=(u.hourangle, u.degree)) #NGC1316
    RA_bcg = (float(c.ra.to_string(decimal=True)))
    dec_bcg = (float(c.dec.to_string(decimal=True)))

    # Groups around Fornax
    RA1, dec1 = 63.0125, -32.8742 #NGC1532
    RA2, dec2 = 52.0583, -20.7444 #Eridanus
    RA3, dec3 = 48.3833, -25.7250 #NGC1255

    plt.figure(figsize=(12,5))

    plt.plot(RA_center,dec_center,color='k',marker='x',linestyle='None',markersize=40,markeredgewidth=1.5,zorder=5)
    plt.plot(RA_bcg,dec_bcg,color='k',marker='*',markeredgecolor='None',linestyle='None',markersize=10,zorder=5)

    plt.plot(RA1,dec1,color='k',marker='D',markeredgecolor='None',linestyle='None',markersize=10,zorder=5)
    plt.plot(RA2,dec2,color='k',marker='D',markeredgecolor='None',linestyle='None',markersize=10,zorder=5)
    plt.plot(RA3,dec3,color='k',marker='D',markeredgecolor='None',linestyle='None',markersize=10,zorder=5)
    plt.text(RA1, dec1, 'NGC1532')
    plt.text(RA2, dec2, 'Eridanus')
    plt.text(RA3, dec3, 'NGC1255')

    circle1 = plt.Circle((RA_center, dec_center), 1.9687, color='dodgerblue', fill=False, linestyle='--', linewidth=1.1, alpha=0.5, zorder=4)
    plt.gca().add_patch(circle1)

    plt.plot(RA_gal1,dec_gal1,color='0.55',alpha=0.6,marker='o',linestyle='None',markersize=10,markeredgecolor='None',zorder=2)
    plt.plot(RA_gal2,dec_gal2,color='darkblue',alpha=0.4,marker='s',linestyle='None',markersize=10,markeredgecolor='None',zorder=2)

    plt.plot(RA_nj1,dec_nj1,color='0.55',alpha=0.6,marker='o',linestyle='None',markersize=10,markeredgecolor='None',zorder=1)
    plt.plot(RA_nj2,dec_nj2,color='darkblue',alpha=0.4,marker='s',linestyle='None',markersize=10,markeredgecolor='None',zorder=1)

    plt.plot(RA_gal1[ind_outlier_main],dec_gal1[ind_outlier_main],color='orangered',marker='o',linestyle='None',markersize=10,markeredgecolor='None',zorder=3)
    plt.plot(RA_gal2[ind_outlier_control],dec_gal2[ind_outlier_control],color='orangered',marker='s',linestyle='None',markersize=10,markeredgecolor='None',zorder=3)

    plt.xlabel(r'$\mathrm{RA}$', fontsize=40)
    plt.ylabel(r'$\mathrm{dec}$', fontsize=40)

    plt.tick_params(axis="x", labelsize=24)
    plt.tick_params(axis="y", labelsize=24)
    plt.grid()

    plt.legend(loc='upper right', prop={'size':20}, numpoints=1)

    plt.savefig('RA_dec_outlier_'+root+'.pdf',format='pdf',dpi=1000,bbox_inches='tight')
    plt.show()

    #sys.exit(-1)

    # Let's plot!
    # Color bar
    bounds = [1,2,3,4]

    my_colors = np.array(['#7f2704', '#d94801', '#fd8d3c', '#fdd0a2']) #'#CC79A7':pink,'#F0E442':yellow,'#999999':gray,'#4daf4a':green
    my_colors = my_colors[::-1]
    my_cmap = ListedColormap(my_colors)

    JClass = JClass1.tolist() + JClass2.tolist()
    min_, max_ = min(JClass), max(JClass)

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

    plt.scatter(Rp2[ind_gal2]/R200,Dv_los2[ind_gal2]/sigma_v,c=JClass2[ind_gal2],cmap=my_cmap,s=250,marker='s',zorder=2,alpha=0.9,edgecolors='black')
    plt.clim(min_, max_)
    plt.scatter(Rp1[ind_gal1]/R200,Dv_los1[ind_gal1]/sigma_v,c=JClass1[ind_gal1],cmap=my_cmap,s=250,marker='o',zorder=3,alpha=0.9,edgecolors='black')
    plt.clim(min_, max_)

    plt.xlabel('$\mathrm{R_{p}/R_{200}}$',fontsize=35)
    plt.ylabel('$\Delta\mathrm{V_{los}/} \sigma_\mathrm{v}}$',fontsize=35)
    #plt.xscale('log')
    plt.xlim(-0.1,5)
    plt.tick_params(axis="x", labelsize=24)
    plt.tick_params(axis="y", labelsize=24)

    cbar = plt.colorbar(ticks=bounds)
    cbar.set_label('$\mathrm{JClass}$',size=30)
    cbar.ax.tick_params(labelsize=24)
    plt.ylim(-5,11)
    plt.xlim(-0.1, 5.1)
    plt.grid()
    plt.text(0.85*5, 0.85*(-4.6), "Fornax", fontsize=23)
    plt.savefig('Phase_diagram_'+root+'.png',dpi=250,bbox_inches='tight')
    plt.show()

    # Center at NGC1316
    RA_cl,dec_cl = 50.6750, -37.2083 #deg
    coord_cl = SkyCoord(RA_cl*u.degree, dec_cl*u.degree)
    z_cl = 0.005911
    Dl_cl = 25.6 # Mpc (luminosity distance from z)
    R200 = 0.35 # Mpc
    kpc_scale = Dl_cl*1000*np.tan((1./3600.)*np.pi/180.)  # each arcsec is kpc_scale in Fornax
    sigma_v = 180. # km/s

    Rp1 = np.zeros(Ngal1)
    Dv_los1 = np.zeros(Ngal1)
    ind_gal1 = []
    for obj in range(Ngal1):
        if dgal1[obj] != -99. and zgal1[obj] != -99.:
            ind_gal1.append(obj)
            coord_gal1 = SkyCoord(RA_gal1[obj]*u.degree, dec_gal1[obj]*u.degree)
            Rp1[obj] = (coord_cl.separation(coord_gal1).arcsecond * kpc_scale)/(R200*1000)
            Dv_los1[obj] = Deltav(zgal1[obj])

    Rp2 = np.zeros(Ngal2)
    Dv_los2 = np.zeros(Ngal2)
    ind_gal2 = []
    for obj in range(Ngal2):
        if dgal2[obj] != -99. and zgal2[obj] != -99.:
            ind_gal2.append(obj)
            coord_gal2 = SkyCoord(RA_gal2[obj]*u.degree, dec_gal2[obj]*u.degree)
            Rp2[obj] = (coord_cl.separation(coord_gal2).arcsecond * kpc_scale)/(R200*1000)
            Dv_los2[obj] = Deltav(zgal2[obj])

    plt.figure(figsize=(12,5))

    plt.plot(x1,y1a,'k',linestyle='--',linewidth=1.2,zorder=1)
    plt.plot(x1,y1b,'k',linestyle='--',linewidth=1.2,zorder=1)
    plt.plot(x2,y2a,'k',linestyle=':',linewidth=1.2,zorder=1)
    plt.plot(x2,y2b,'k',linestyle=':',linewidth=1.2,zorder=1)

    plt.scatter(Rp2[ind_gal2]/R200,Dv_los2[ind_gal2]/sigma_v,c=JClass2[ind_gal2],cmap=my_cmap,s=250,marker='s',zorder=2,alpha=0.9,edgecolors='black')
    plt.clim(min_, max_)
    plt.scatter(Rp1[ind_gal1]/R200,Dv_los1[ind_gal1]/sigma_v,c=JClass1[ind_gal1],cmap=my_cmap,s=250,marker='o',zorder=3,alpha=0.9,edgecolors='black')
    plt.clim(min_, max_)

    plt.xlabel('$\mathrm{R_{p}/R_{200}}$',fontsize=35)
    plt.ylabel('$\Delta\mathrm{V_{los}/} \sigma_\mathrm{v}}$',fontsize=35)
    #plt.xscale('log')
    plt.xlim(-0.5,25)
    plt.tick_params(axis="x", labelsize=24)
    plt.tick_params(axis="y", labelsize=24)

    cbar = plt.colorbar(ticks=bounds)
    cbar.set_label('$\mathrm{JClass}$',size=30)
    cbar.ax.tick_params(labelsize=24)
    plt.ylim(-6,6)
    plt.grid()
    plt.savefig('Phase_diagram_'+root+'_NGC1316.png',dpi=250,bbox_inches='tight')
    plt.show()

#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------

if __name__ == '__main__':
     main()
