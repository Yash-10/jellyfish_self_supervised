## Follow the instructions below to setup the data
#  (1) Access the folder here: https://drive.google.com/drive/folders/1HTWDpad8P7trQN_od8FFc6qIdNJ_AfqQ?usp=sharing
# (2) Run these commands:
#       !tar -xvzf splus_prepare_data/splus_galmasked.tar.gz
#       !tar -xvzf splus_prepare_data/splus_original.tar.gz
# (3) Run this script.

import os
import glob
import shutil
import pandas as pd
import numpy as np

from astropy.io import fits
from astropy.stats import SigmaClip
from photutils.background import MedianBackground, Background2D

CLUSTERS = ["Antlia", "Hydra", "Fornax"]
TEST_CLUSTER = "Fornax"
TRAIN_CLUSTERS = ["Antlia", "Hydra"]


if os.path.exists('../train'):
    shutil.rmtree('../train')
if os.path.exists('../test'):
    shutil.rmtree('../test')

os.mkdir('../train')
os.mkdir('../test')
os.mkdir('../train/1')
os.mkdir('../train/0')
os.mkdir('../test/1')
os.mkdir('../test/0')

with open('../splus_prepare_data/need_to_exclude_jelly.list') as f:
    ej = f.read().splitlines()
    exclude_jelly = []
    for i in range(len(ej)):
        exclude_jelly = exclude_jelly + glob.glob('galmask_cut_jelly_candidates/' + ej[i] + '_galmask.fits')
    exclude_jelly = list(set(exclude_jelly))
with open('../splus_prepare_data/need_to_exclude_nonjelly_sample.list') as f:
    enjs = f.read().splitlines()
    exclude_njelly_sample = []
    for i in range(len(enjs)):
        exclude_njelly_sample = exclude_njelly_sample + glob.glob('galmask_cut_non_jelly_sample/' + enjs[i] + '_galmask.fits')
    exclude_njelly_sample = list(set(exclude_njelly_sample))
with open('../splus_prepare_data/need_to_exclude_nonjelly_control.list') as f:
    enjc = f.read().splitlines()
    exclude_njelly_control = []
    for i in range(len(enjc)):
        exclude_njelly_control = exclude_njelly_control + glob.glob('galmask_cut_non_jelly_control/' + enjc[i] + '_galmask.fits')
    exclude_njelly_control = list(set(exclude_njelly_control))


df = pd.read_csv('../splus_prepare_data/SPLUS_jellyfish_candidates.csv')

jclasses = []
rjimg_list = glob.glob('../galmask_cut_jelly_candidates/*_F660_*.fits')

for img in rjimg_list:
    galname = img.split('/')[1].split('_')[0]
    row = df[df['Name'] == galname]
    jclasses.append(int(row['JClass']))

with open("../jclasses.txt", "w") as f:
    for item in jclasses:
        f.write(f"{item}\n")

print(len(exclude_jelly), len(exclude_njelly_control), len(exclude_njelly_sample))

for f in exclude_jelly:
    os.remove(f)
for f in exclude_njelly_control:
    os.remove(f)
for f in exclude_njelly_sample:
    os.remove(f)


nj_control = pd.read_csv('../splus_prepare_data/SPLUS_non_jellyfish_control.csv')
nj_sample = pd.read_csv('../splus_prepare_data/SPLUS_non_jellyfish_sample.csv')
j = pd.read_csv('../splus_prepare_data/SPLUS_jellyfish_candidates.csv')

print(nj_sample['Cluster'].unique(), nj_control['Cluster'].unique(), j['Cluster'].unique())

def get_2D_bkg(data):
    sigma_clip = SigmaClip(sigma=3.)
    bkg_estimator = MedianBackground()
    bkg = Background2D(data, (25, 25), filter_size=(2, 2), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator).background
    zero_pixel_mask = np.ma.masked_array(data, data != 0.).mask.astype(float)
    return np.multiply(bkg, zero_pixel_mask)

def train_setup(TRAIN_CLUSTERS):
  for TRAIN_CLUSTER in TRAIN_CLUSTERS:
    galnames = []
    for f in sorted(glob.glob('../cut_jelly_candidates/*_F660_*.fits')):  # Take any band just for collecting the names.
      galnames.append(f.split('/')[1].split('_')[0])

    test_j_rows = j[j['Cluster'] == TRAIN_CLUSTER]
    for galname in galnames:
      filenames = sorted(glob.glob('../galmask_cut_jelly_candidates/'+ galname + '*.fits'))
      if len(filenames) == 12:
        row = test_j_rows[test_j_rows['Name'] == galname]
        if not row.empty:
          data = np.empty(shape=(12, 350, 350))
          # jj = 0
          for i, f in enumerate(filenames):
            # if i == 0 or i == 1 or i == 2:
            #   continue
            _d = fits.getdata(f)
            _d = _d - get_2D_bkg(_d)
            data[i, :, :] = np.arcsinh(_d)
            # jj += 1
          np.save(f'../train/1/{galname}.npy', data)

    galnames = []
    for f in sorted(glob.glob('../cut_non_jelly_control/*_F660_*.fits')):  # Take any band just for collecting the names.
      galnames.append(f.split('/')[1].split('_')[0])

    test_nj_rows = nj_control[nj_control['Cluster'] == TRAIN_CLUSTER]
    for galname in galnames:
      filenames = sorted(glob.glob('../galmask_cut_non_jelly_control/'+ galname + '*.fits'))
      if len(filenames) == 12:
        row = test_nj_rows[test_nj_rows['Name'] == galname]
        if not row.empty:
          data = np.empty(shape=(12, 350, 350))
          # jj = 0
          for i, f in enumerate(filenames):
            # if i == 0 or i == 1 or i == 2:
            #   continue
            _d = fits.getdata(f)
            _d = _d - get_2D_bkg(_d)
            data[i, :, :] = np.arcsinh(_d)
            # jj += 1
          np.save(f'../train/0/{galname}.npy', data)

    galnames = []
    for f in sorted(glob.glob('../cut_non_jelly_sample/*_F660_*.fits')):  # Take any band just for collecting the names.
      galnames.append(f.split('/')[1].split('_')[0])

    test_nj_rows = nj_sample[nj_sample['Cluster'] == TRAIN_CLUSTER]
    for galname in galnames:
      filenames = sorted(glob.glob('../galmask_cut_non_jelly_sample/'+ galname + '*.fits'))
      if len(filenames) == 12:
        row = test_nj_rows[test_nj_rows['Name'] == galname]
        if not row.empty:
          data = np.empty(shape=(12, 350, 350))
          # jj = 0
          for i, f in enumerate(filenames):
            # if i == 0 or i == 1 or i == 2:
            #   continue
            _d = fits.getdata(f)
            _d = _d - get_2D_bkg(_d)
            data[i, :, :] = np.arcsinh(_d)
            # jj += 1
          np.save(f'../train/0/{galname}.npy', data)

def test_setup(TEST_CLUSTER):
  galnames = []
  for f in sorted(glob.glob('../cut_jelly_candidates/*_F660_*.fits')):  # Take any band just for collecting the names.
    galnames.append(f.split('/')[1].split('_')[0])

  test_j_rows = j[j['Cluster'] == TEST_CLUSTER]
  for galname in galnames:
    filenames = sorted(glob.glob('../galmask_cut_jelly_candidates/'+ galname + '*.fits'))
    if len(filenames) == 12:
      row = test_j_rows[test_j_rows['Name'] == galname]
      if not row.empty:
        data = np.empty(shape=(12, 350, 350))
        # jj = 0
        for i, f in enumerate(filenames):
          # if i == 0 or i == 1 or i == 2:
          #     continue
          _d = fits.getdata(f)
          _d = _d - get_2D_bkg(_d)
          data[i, :, :] = np.arcsinh(_d)
          # jj += 1
        np.save(f'../test/1/{galname}.npy', data)

  galnames = []
  for f in sorted(glob.glob('../cut_non_jelly_control/*_F660_*.fits')):  # Take any band just for collecting the names.
    galnames.append(f.split('/')[1].split('_')[0])

  test_nj_rows = nj_control[nj_control['Cluster'] == TEST_CLUSTER]
  for galname in galnames:
    filenames = sorted(glob.glob('../galmask_cut_non_jelly_control/'+ galname + '*.fits'))
    if len(filenames) == 12:
      row = test_nj_rows[test_nj_rows['Name'] == galname]
      if not row.empty:
        data = np.empty(shape=(12, 350, 350))
        # jj = 0
        for i, f in enumerate(filenames):
          # if i == 0 or i == 1 or i == 2:
          #     continue
          _d = fits.getdata(f)
          _d = _d - get_2D_bkg(_d)
          data[i, :, :] = np.arcsinh(_d)
          # jj += 1
        np.save(f'../test/0/{galname}.npy', data)

  galnames = []
  for f in sorted(glob.glob('../cut_non_jelly_sample/*_F660_*.fits')):  # Take any band just for collecting the names.
    galnames.append(f.split('/')[1].split('_')[0])

  test_nj_rows = nj_sample[nj_sample['Cluster'] == TEST_CLUSTER]
  for galname in galnames:
    filenames = sorted(glob.glob('../galmask_cut_non_jelly_sample/'+ galname + '*.fits'))
    if len(filenames) == 12:
      row = test_nj_rows[test_nj_rows['Name'] == galname]
      if not row.empty:
        data = np.empty(shape=(12, 350, 350))
        # jj = 0
        for i, f in enumerate(filenames):
          # if i == 0 or i == 1 or i == 2:
          #     continue
          _d = fits.getdata(f)
          _d = _d - get_2D_bkg(_d)
          data[i, :, :] = np.arcsinh(_d)
          # jj += 1
        np.save(f'../test/0/{galname}.npy', data)

train_setup(TRAIN_CLUSTERS)
test_setup(TEST_CLUSTER)

# Some quick checks.
if TEST_CLUSTER == "Fornax" and ("Antlia" in TRAIN_CLUSTERS and "Hydra" in TRAIN_CLUSTERS):
    assert len(os.listdir('../train/0')) == 49
    assert len(os.listdir('../train/1')) == 36
    assert len(os.listdir('../test/0')) == 61
    assert len(os.listdir('../test/1')) == 38