#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample plots for QA.

Created on Wed May 29 11:43:00 2019

@author: yago
"""

from __future__ import print_function, division

from astropy.io import fits, ascii
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors

work_dir = 'SV/20230512_13'
file_name = 'single_3005695'
#rss = fits.open('data/download.was.tng.iac.es/jobs/download/f1cde750-b342-11ed-8852-656428dacc6a/2022-12-12/r2963102.fit')
#rss = fits.open('data/raw/2022-10-26/r2963269.fit') # raw data, not RSS!
rss = fits.open(f'{work_dir}/{file_name}.fit')
# rss = fits.open('./apm3.ast.cam.ac.uk/~dmurphy/ifu_exercise/L1/20190922/single_1004361.fit')
# rss = fits.open('old/single_1004349.fit')
# rss = fits.open('old/single_1004193.fit')
# rss = fits.open('old/single_1004217.fit')
fibtable = rss['FIBTABLE'].data

#tab_lifu = ascii.read('LIFUfibreTable.dat',
#                      names=['x', 'y', 'name', 'dummy', 'id'])
#tab_lifu.sort('id')

# %%

delta = 25.15-2.5*np.log10(fibtable['Meanflux_r'])-fibtable['MAG_R']

'''
plt.plot(fibtable['MAG_R'], delta, 'r.')
plt.xlabel('MAG_R')
plt.xlim(np.nanmin(fibtable['MAG_R'])-.1, 26.5)
plt.ylabel('25.15-2.5*log10(Meanflux_r [ADU]) - MAG_R')
plt.ylim(-.45, .45)
plt.grid()
plt.savefig('delta_m.pdf', bbox_inches='tight')
plt.show()
'''

'''
plt.plot(fibtable['SNR'], delta, 'r.')
plt.xlabel('SNR')
plt.xscale('log')
# plt.xlim(np.nanmin(fibtable['MAG_R'])-.1, 26.5)
plt.ylabel('25.15-2.5*log10(Meanflux_r [ADU]) - MAG_R')
plt.ylim(-.45, .45)
plt.grid()
plt.savefig('delta_SNR.pdf', bbox_inches='tight')
plt.show()
'''

'''
plt.scatter(fibtable['xposition'], fibtable['yposition'],
            s=50, cmap='jet',
            vmin=-.45, vmax=.45,
            c=delta)
plt.title('25.15-2.5*log10(Meanflux_r [ADU]) - MAG_R')
plt.colorbar()
plt.savefig('delta.pdf', bbox_inches='tight')
plt.show()
'''

# %%

'''
plt.scatter(fibtable['xposition'], fibtable['yposition'],
            s=50, cmap='gist_stern_r',
            vmin=16.5, vmax=25.5,
            c=fibtable['MAG_R'])
plt.title('MAG_R')
#plt.xlim(-2.5, 2.5)
#plt.ylim(-2.5, 2.5)
plt.gca().set_aspect('equal')
plt.colorbar()
plt.tight_layout()
plt.savefig('MAG_R.pdf', bbox_inches='tight')
plt.show()
'''

mag_faint = 24.5
mag_bright = 18.5
plt.scatter(fibtable['fibrera'], fibtable['fibredec'],
            s=10, cmap='nipy_spectral_r',
            vmin=mag_bright, vmax=mag_faint,
            c=25.15 - 2.5*np.log10(fibtable['Meanflux_r']))
plt.title('25.15 - 2.5*log10(Meanflux_r [ADU])')
#plt.xlim(-2.5, 2.5)
#plt.ylim(-2.5, 2.5)
plt.gca().set_aspect('equal')
cb = plt.colorbar()
cb.ax.set_ylim(mag_faint, mag_bright)
plt.tight_layout()
plt.savefig(f'{work_dir}/Meanflux_r.pdf', bbox_inches='tight')
plt.show()

plt.scatter(fibtable['fibrera'], fibtable['fibredec'],
            s=10, cmap='nipy_spectral',
            norm=colors.LogNorm(vmin=.5, vmax=50),
            c=fibtable['SNR'])
plt.title('SNR')
#plt.xlim(-2.5, 2.5)
#plt.ylim(-2.5, 2.5)
plt.gca().set_aspect('equal')
plt.colorbar()
plt.tight_layout()
plt.savefig(f'{work_dir}/SNR.pdf', bbox_inches='tight')
plt.show()

# %%

'''
lambda_0 = rss[1].header['CRVAL1']
dlambda = rss[1].header['CD1_1']
wavelength = lambda_0 + dlambda*np.arange(rss[1].shape[1])
mag_sorted = np.argsort(fibtable['MAG_R'])
SED = rss[1].data * rss[5].data
for i in [0, 10, 100]:
    fibre = mag_sorted[i]
    plt.plot(wavelength, SED[fibre],
             label='Fibre {} ({:.2f} mag/arcsec$^2$)'.format(
                     fibre, fibtable['MAG_R'][fibre]))
plt.plot(wavelength, np.sum(SED, axis=0), 'k-', lw=3, label='Total')
plt.yscale('log')
plt.ylim(3e-19, 3e-14)
plt.legend()
plt.savefig('SED.pdf')
plt.show()
'''

# %%
# -----------------------------------------------------------------------------
#                                                    ... Paranoy@ Rulz! ;^D
# -----------------------------------------------------------------------------
