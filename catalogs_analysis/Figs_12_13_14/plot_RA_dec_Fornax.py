#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Carolina Queiroz"
__maintainer__ = "Carolina Queiroz"
__email__ = "c.queirozabs@gmail.com"
__status__ = "Production"

'''

Plot RA vs. dec of candidates

Last modification: Nov 22nd
This version: Oct 18th 2022

'''

from astropy import units as u
from astropy.coordinates import SkyCoord
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

#-------------------------------------------------------------------------------
def main():

	import seaborn as sns
	sns.set_context("paper", font_scale = 2)
	sns.set_style('ticks')

	root = 'Fornax'
	samp_type1 = 'sample'
	samp_type2 = 'control'

	# Jelyfish candidates
	cand1 = pd.read_csv('SPLUS_new_jellyfish_candidates_'+root+'_'+samp_type1+'_oct2022.csv')
	cand2 = pd.read_csv('SPLUS_new_jellyfish_candidates_'+root+'_'+samp_type2+'_oct2022.csv')

	RA_cand1 = cand1['RA']
	dec_cand1 = cand1['dec']
	JClass1 = cand1['JClass']
	RA_cand2 = cand2['RA']
	dec_cand2 = cand2['dec']
	JClass2 = cand2['JClass']

	RA_cand = RA_cand1.tolist() + RA_cand2.tolist()
	dec_cand = dec_cand1.tolist() + dec_cand2.tolist()
	JClass = JClass1.tolist() + JClass2.tolist()

	# Non-jellyfish galaxies
	df1 = pd.read_csv('SPLUS_non_jellyfish_'+root+'_'+samp_type1+'_oct2022.csv')
	df2 = pd.read_csv('SPLUS_non_jellyfish_'+root+'_'+samp_type2+'_oct2022.csv')

	RA_1 = df1['RA']
	dec_1 = df1['dec']
	RA_2 = df2['RA']
	dec_2 = df2['dec']

	# Let's plot!

	#3- Fornax
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
	#RA4, dec4 = 35.7708, -21.2339 #NGC908 => too distant

	min_, max_ = min(JClass), max(JClass)
	print (RA_center,dec_center)

	my_colors = np.array(['#7f2704', '#d94801', '#fd8d3c', '#fdd0a2'])
	my_colors = my_colors[::-1]
	my_cmap = ListedColormap(my_colors)

	plt.figure(figsize=(12,5))

	plt.plot(RA_center,dec_center,'k',marker='x',linestyle='None',markersize=10,markeredgewidth=1.5,zorder=3)
	plt.plot(RA_bcg,dec_bcg,'k',marker='*',linestyle='None',markersize=10,markeredgewidth=1.5,zorder=3)

	plt.plot(RA1,dec1,'k',marker='D',linestyle='None',markersize=10,markeredgewidth=1.5,zorder=3)
	plt.plot(RA2,dec2,'k',marker='D',linestyle='None',markersize=10,markeredgewidth=1.5,zorder=3)
	plt.plot(RA3,dec3,'k',marker='D',linestyle='None',markersize=10,markeredgewidth=1.5,zorder=3)

	plt.text(61.8,-34.5,'NGC1532',fontsize=14)
	plt.text(51,-22.5,'Eridanus',fontsize=14)
	plt.text(47.2,-27.5,'NGC1255',fontsize=14)

	circle1 = plt.Circle((RA_center, dec_center), 2.01543104244, color='r', fill=False, linestyle='--', linewidth=1.1, alpha=0.5, zorder=3)
	plt.gca().add_patch(circle1)

	plt.plot(RA_1,dec_1,color='0.55',alpha=0.6,marker='o',linestyle='None',markersize=10,markeredgecolor='None',zorder=2,label='Sample')
	plt.plot(RA_2,dec_2,color='darkblue',alpha=0.4,marker='s',linestyle='None',markersize=10,markeredgecolor='None',zorder=1,label='Control')

	plt.scatter(RA_cand2,dec_cand2,c=JClass2,cmap=my_cmap,s=100,marker='s',zorder=4)
	plt.clim(min_, max_)
	plt.scatter(RA_cand1,dec_cand1,c=JClass1,cmap=my_cmap,s=100,marker='o',zorder=4)
	plt.clim(min_, max_)

	bounds = [1,2,3,4]

	plt.xlabel(r'$\mathrm{RA}$', fontsize=35)
	plt.ylabel(r'$\mathrm{dec}$', fontsize=35)

	#plt.xticks(np.arange(155.5, np.max(RA_cand)+0.5, step=0.5), ('155.5', '156', '156.5', '157', '157.5', '158', '158.5', '159', '159.5'))
	plt.xlim(44.,66.)
	# plt.ylim(-40, -25)
	#plt.ylim(-37.1,-33.4)

	plt.tick_params(axis="x", labelsize=24)
	plt.tick_params(axis="y", labelsize=24)
	cbar = plt.colorbar(ticks=bounds)
	cbar.set_label('$\mathrm{JClass}$',size=35)
	cbar.ax.tick_params(labelsize=24)
	plt.legend(loc='upper right', prop={'size':20}, numpoints=1)
	plt.grid()
	plt.savefig('RA_dec_jellyfish_candidates_'+root+'2.png',dpi=250,bbox_inches='tight')
	plt.show()

	sys.exit(-1)

	#
	ind_high_jclass1 = np.where(JClass1>=3)[0]
	ind_low_jclass1 = np.where(JClass1<3)[0]

	ind_high_jclass2 = np.where(JClass2>=3)[0]
	ind_low_jclass2 = np.where(JClass2<3)[0]

	plt.figure(figsize=(10,5))

	plt.plot(RA_center,dec_center,'k',marker='x',linestyle='None',markersize=10,markeredgewidth=1.5,zorder=3)
	plt.plot(RA_bcg,dec_bcg,'k',marker='*',linestyle='None',markersize=10,markeredgewidth=1.5,zorder=3)
	
	#plt.plot(RA1,dec1,'k',marker='D',linestyle='None',markersize=5,markeredgewidth=1.5,zorder=3)
	#plt.plot(RA2,dec2,'k',marker='D',linestyle='None',markersize=5,markeredgewidth=1.5,zorder=3)
	#plt.plot(RA3,dec3,'k',marker='D',linestyle='None',markersize=5,markeredgewidth=1.5,zorder=3)

	circle1 = plt.Circle((RA_center, dec_center), 2.01543104244, color='deeppink', fill=False, linestyle='--', linewidth=1.1, alpha=0.5, zorder=3)
	plt.gca().add_patch(circle1)

	plt.plot(RA_1,dec_1,color='k',alpha=0.5,marker='o',linestyle='None',markersize=5,markeredgecolor='None',zorder=2,label='$\mathrm{Sample}$')
	plt.plot(RA_2,dec_2,color='0.55',alpha=0.6,marker='s',linestyle='None',markersize=5,markeredgecolor='None',zorder=1,label='$\mathrm{Control}$')

	plt.plot(RA_cand1[ind_low_jclass1],dec_cand1[ind_low_jclass1],color='k',alpha=0.5,marker='o',linestyle='None',markersize=5,markeredgecolor='None',zorder=2)
	plt.plot(RA_cand2[ind_low_jclass2],dec_cand2[ind_low_jclass2],color='0.55',alpha=0.6,marker='s',linestyle='None',markersize=5,markeredgecolor='None',zorder=1)

	#plt.scatter(RA_cand2,dec_cand2,c=JClass2,cmap='viridis',s=50,marker='s',zorder=4)
	#plt.clim(min_, max_)
	#plt.scatter(RA_cand1,dec_cand1,c=JClass1,cmap='viridis',s=50,marker='o',zorder=4)
	#plt.clim(min_, max_)

	for i in ind_high_jclass1:
		if JClass1[i] == 3:
			aux1a = i
			plt.plot(RA_cand1[i],dec_cand1[i],color=my_colors[2],marker='o',linestyle='None',markersize=7,markeredgecolor='k',markeredgewidth=1.2,zorder=4)
		if JClass1[i] == 4:
			aux2a = i
			plt.plot(RA_cand1[i],dec_cand1[i],color=my_colors[3],marker='o',linestyle='None',markersize=7,markeredgecolor='k',markeredgewidth=1.2,zorder=4)

	for i in ind_high_jclass2:
		if JClass2[i] == 3:
			aux1b = i
			plt.plot(RA_cand2[i],dec_cand2[i],color=my_colors[2],marker='s',linestyle='None',markersize=7,markeredgecolor='k',markeredgewidth=1.2,zorder=4)
		if JClass2[i] == 4:
			aux2b = i
			plt.plot(RA_cand2[i],dec_cand2[i],color=my_colors[3],marker='s',linestyle='None',markersize=7,markeredgecolor='k',markeredgewidth=1.2,zorder=4)

	bounds = [1,2,3,4]

	plt.xlabel(r'$\mathrm{RA}$', fontsize=40)
	plt.ylabel(r'$\mathrm{dec}$', fontsize=40)

	plt.xlim(44.,66.)
	plt.ylim(-40,-28)

	plt.tick_params(axis="x", labelsize=24)
	plt.tick_params(axis="y", labelsize=24)
	plt.legend(loc=0, prop={'size':20}, numpoints=1)
	plt.grid()
	plt.savefig('RA_dec_jellyfish_candidates_'+root+'_presentation.pdf',format='pdf',dpi=1000,bbox_inches='tight')
	plt.show()

#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------

if __name__ == '__main__':
     main()
