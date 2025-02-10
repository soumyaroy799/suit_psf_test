from sunpy.map import Map 
import matplotlib.pyplot as plt 
from glob import glob
import os
from irispy.io import read_files
from psf_estimate_2 import *
from astropy.io import fits
from sunkit_instruments.iris import SJI_to_sequence
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import trange
from matplotlib import colors
from scipy.signal import convolve2d

import sys
sys.path.insert(0,'/Users/soumyaroy/solar_codes')
from register import sun_register

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

suit_files = sorted(glob(f'{project_path}/data/SUIT/*NB03*'))
iris_files = sorted(glob(f'{project_path}/data/IRIS/*.fits'))

suit_map = Map(suit_files[0])

# iris_map = Map(iris_files[0])
# iris_map = read_files(iris_files[0])

iris_map = SJI_to_sequence(iris_files[0])
# iris_map.peek(); plt.show()

idx = 6; b = iris_map[idx]

suit_map = sun_register(suit_map, res=b.meta['cdelt1']*u.arcsec/u.pix)
# blo = SkyCoord(b.bottom_left_coord.Tx, b.bottom_left_coord.Ty, frame=suit_map.coordinate_frame)
# tro = SkyCoord(b.top_right_coord.Tx, b.top_right_coord.Ty, frame=suit_map.coordinate_frame)
blo = SkyCoord(-87*u.arcsec, -190*u.arcsec, frame=suit_map.coordinate_frame)
tro = SkyCoord(77*u.arcsec, -18*u.arcsec, frame=suit_map.coordinate_frame)

# print(b.bottom_left_coord, b.top_right_coord)

c = suit_map.submap(blo, top_right=tro)

blo = SkyCoord(-87*u.arcsec, -190*u.arcsec, frame=b.coordinate_frame)
tro = SkyCoord(77*u.arcsec, -18*u.arcsec, frame=b.coordinate_frame)
b = b.submap(blo, top_right=tro)

# fig = plt.figure()
# ax = fig.add_subplot(121,projection=b)
# b.plot(axes=ax); ax.grid(False)

# ax = fig.add_subplot(122, projection=c)
# c.plot(axes=ax, vmin=0); ax.grid(False)

# plt.show()

# suit_map.plot(vmin=0); plt.show()

# psf_estimated = gaussian_filter(np.zeros(c.data.shape), sigma=3)
# psf_estimated = gaussian_filter(np.zeros((15,15)), sigma=2)
# x, y = psf_estimated.shape
# psf_estimated[x//2, y//2] = 1

im = fits.open('/Users/soumyaroy/Library/CloudStorage/GoogleDrive-soumyaroy799@gmail.com/My Drive/forward_modle_paper_plot/For Raja_09May2023/FF_Dark_Normal_fw1_01_fw2_00_os_254_rd_freq_280k_180sec_002/normal_FF_Dark_Normal_fw1_01_fw2_00_os_254_rd_freq_280k_180sec_002_000003.fits'); im = im[0].data

dark = fits.open('/Users/soumyaroy/Library/CloudStorage/GoogleDrive-soumyaroy799@gmail.com/My Drive/forward_modle_paper_plot/For Raja_09May2023/FF_Dark_Normal_fw1_01_fw2_00_os_254_rd_freq_280k_180sec_002/dark_FF_Dark_Normal_fw1_01_fw2_00_os_254_rd_freq_280k_180sec_002_000002.fits'); dark = dark[0].data

psf = im-dark; psf = np.array(psf)
x_cen = 1440; y_cen = 1425; half_win = 20
psf = psf[x_cen-half_win:x_cen+half_win,y_cen-half_win:y_cen+half_win]
psf2 = np.reshape(psf, (psf.shape[0]*psf.shape[1]))

for i,_ in enumerate(psf2):
    if(psf2[i]<=30 or psf2[i]>4e4):
        psf2[i]=0.


psf_estimated = np.reshape(psf2,(psf.shape[0],psf.shape[1]))
psf_estimated = psf_estimated/np.sum(psf_estimated)#; print(psf_estimated.shape)
plt.imshow(psf_estimated,origin='lower',cmap='jet',norm=colors.LogNorm()); plt.colorbar(); plt.show()

psf_estimated, loss_history = refine_psf(b.data[2:-1,2:-1],c.data, psf_estimated,)
# psf_estimated = refine_psf(b.data[2:-1,2:-1],c.data, psf_estimated,)

fig = plt.figure()
ax = fig.add_subplot(221)
ax.imshow(b.data, origin='lower')

ax = fig.add_subplot(222)
ax.imshow(c.data, origin='lower')

estimated_convolved = convolve2d(b.data[2:-1,2:-1], psf_estimated, mode='same')

ax = fig.add_subplot(223)
ax.imshow(psf_estimated, origin='lower')

ax = fig.add_subplot(224)
ax.imshow(estimated_convolved, origin='lower')

plt.show()