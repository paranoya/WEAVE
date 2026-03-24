#!/usr/bin/env python
# coding: utf-8

# # Quality control of WEAVE RSS files
# 
# Relative wavelength calibration and throughput test for individual fibres
# 
# Based on sky emission lines identified in Row-stacked spectra (RSS)
# 
# ----------------------------------------------------------------------
# # Initialisation
# ----------------------------------------------------------------------

# ## Imports
# ----------------------------------------------------------------------

from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.ticker import AutoMinorLocator

import numpy as np
import os
import glob
import argparse
from time import time
from scipy import ndimage
from scipy import optimize
from scipy import stats

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy import constants as c
from astropy.table import Table

parser = argparse.ArgumentParser(prog='Single RSS QA',
                    description='Perform simple quality control plots',
                    epilog='...Paranoy@ rulz! ;^D')
parser.add_argument('filename', help='Name of RSS file') # positional argument
args = parser.parse_args()

# Plotting functions:

def new_figure(fig_name, figsize=None, nrows=1, ncols=1, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0}, **subplot_kwargs):
    if figsize is None:
        figsize = (9 + ncols, 4 + nrows)
        
    plt.close(fig_name)
    fig = plt.figure(fig_name, figsize=figsize)
    axes = fig.subplots(nrows=nrows, ncols=ncols, squeeze=False,
                        sharex=sharex, sharey=sharey,
                        gridspec_kw=gridspec_kw,
                        **subplot_kwargs
                       )
    fig.set_tight_layout(True)
    for ax in axes.flat:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
        ax.tick_params(which='major', direction='inout', length=8, grid_alpha=.3)
        ax.tick_params(which='minor', direction='in', length=2, grid_alpha=.1)
        ax.grid(True, which='both')

    fig.suptitle(f'{os.path.basename(rss.filename)}: {fig_name}')  # VERY dirty hack :^(
    #fig.suptitle(fig_name)
    
    return fig, axes

default_cmap = plt.get_cmap("gist_earth").copy()
default_cmap.set_bad('gray')


def colour_map(fig, ax, cblabel, data, cmap=default_cmap, norm=None, xlabel=None, x=None, ylabel=None, y=None):
    
    if norm is None:
        percentiles = np.array([1, 16, 50, 84, 99])
        ticks = np.nanpercentile(data, percentiles)
        linthresh = np.median(data[data > 0])
        norm = colors.SymLogNorm(vmin=ticks[0], vmax=ticks[-1], linthresh=linthresh)
    else:
        ticks = None
    if y is None:
        y = np.arange(data.shape[0])
    if x is None:
        x = np.arange(data.shape[1])

    im = ax.imshow(data,
                   extent=(x[0]-(x[1]-x[0])/2, x[-1]+(x[-1]-x[-2])/2, y[0]-(y[1]-y[0])/2, y[-1]+(y[-1]-y[-2])/2),
                   interpolation='nearest', origin='lower',
                   cmap=cmap,
                   norm=norm,
                  )
    ax.set_aspect('auto')
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    cb = fig.colorbar(im, ax=ax, orientation='vertical', shrink=.9)
    cb.ax.set_ylabel(cblabel)
    if ticks is not None:
        cb.ax.set_yticks(ticks=ticks, labels=[f'{value:.3g} ({percent}%)' for value, percent in zip(ticks, percentiles)])
    cb.ax.tick_params(labelsize='small')
    
    return im, cb


# ## Ancillary data
# ----------------------------------------------------------------------

# UVES sky emission atlas: <https://www.eso.org/observing/dfo/quality/UVES/pipeline/sky_spectrum.html>

wave_flux = np.empty((0, 2))
filenames = glob.glob('sky/UVES_sky_emission_atlas/gident_*.dat')
filenames.sort()
for filename in filenames:
    print(filename)
    wave_flux = np.concatenate((wave_flux, np.loadtxt(filename, usecols=(1, 4), skiprows=3, comments=['#', '--------'])), axis=0)
UVES_atlas = Table(wave_flux, names=('wavelength', 'flux'))


# ## RSS files
# ----------------------------------------------------------------------

def find_lower_envelope(x, y, min_separation):
    '''
    Fit lower envelope of a single spectrum:
    1) Find local minima, with a minimum separation `min_separation`.
    2) Interpolate linearly between them.
    '''
    valleys = []
    y[np.isnan(y)] = np.inf
    for i in range(min_separation, y.size-min_separation-1):
        if np.argmin(y[i-min_separation:i+min_separation+1]) == min_separation:
            valleys.append(i)
    y[~np.isfinite(y)] = np.nan
    return np.fmin(y, np.interp(x, x[valleys], y[valleys]))


def find_continuum(x, y, h):
    continuum = find_lower_envelope(x, y, h)

    offset = y - continuum
    offset = offset[offset < 2*np.nanmedian(offset)]
    continuum += np.nanpercentile(offset, 50)
    #continuum = find_lower_envelope(x, continuum, h)

    #return lower_envelope_original + np.nanmedian(offset)
    #continuum = boxcar(lower_envelope_original + np.nanmedian(offset), 10*h)
    #continuum = boxcar(spectrum, h)
    #continuum = find_lower_envelope(x, continuum, h)
    #continuum = boxcar(continuum, 2*h)
    #continuum = np.fmax(lower_envelope_original, continuum)
    
    return continuum


class WEAVE_RSS(object):
    
    def __init__(self, filename):
        '''Read a WEAVE "single exposure" file (i.e. row-stacked spectra for just one arm)'''
        self.filename = filename
        self.hdu = fits.open(filename)
        self.wcs = WCS(self.hdu[1].header)
        pixels = np.arange(self.hdu[1].data.shape[1])
        self.wavelength = self.wcs.spectral.array_index_to_world(pixels).to_value(u.Angstrom)
        self.counts = self.hdu[3].data
        self.counts_error = np.where(self.hdu[4].data > 0, 1/np.sqrt(self.hdu[4].data), np.nan)
        self.sky_counts = self.hdu[3].data - self.hdu[1].data
        self.sensitivity_function = self.hdu[5].data
        self.flux = self.hdu[1].data*self.sensitivity_function
        self.sky_sub_ivar = self.hdu[2].data
        
        bad = np.isnan(self.counts_error).nonzero()
        self.counts[bad] = np.nan
        self.sky_counts[bad] = np.nan
        self.sensitivity_function[bad] = np.nan
        self.flux[bad] = np.nan
        self.sky_sub_ivar[bad] = np.nan
        
        self.fibtable = self.hdu[6].data
        #self.sky_fibres = np.where(self.fibtable['TARGUSE'] != 'Patata')
        self.sky_fibres = np.where(self.fibtable['TARGUSE'] == 'S')
        self.target_fibres = np.where(self.fibtable['TARGUSE'] == 'T')
        self.n_sky_fibres = self.sky_fibres[0].size
        self.n_fibres = self.counts.shape[0]

        heliocentric_correction = (1 + np.nanmean(self.fibtable['Helio_cor']) / c.c.to_value(u.km/u.s))
        wave = UVES_atlas['wavelength'] * heliocentric_correction
        inside = np.where((wave > self.wavelength[0]) & (wave < self.wavelength[-1]))
        self.sky_lines = UVES_atlas[inside].copy()
        self.sky_lines['wavelength'] *= heliocentric_correction
        
        self.intensity = self.counts.copy()
        self.continuum, self.strong_sky_lines = self.find_sky_lines()
        self.f_sky = self.estimate_sky()
        self.line_offset, self.line_throughput = self.trace_sky_lines()

        self.original_offset = self.line_offset.copy()
        self.original_throughput = self.line_throughput.copy()
        self.original_continuum = self.continuum.copy()

        for i in range(0):
            line_weight = self.strong_sky_lines['counts']
            total_weight = np.nansum(line_weight)
            # TODO: flux-conserving interpolation
            wavelength_correction = np.nansum(self.line_offset*line_weight[:, np.newaxis], axis=0) / total_weight
            #wavelength_correction /= 2
            for fibre in range(self.n_fibres):
                self.intensity[fibre] = np.interp(self.wavelength,
                                                         self.wavelength - wavelength_correction[fibre], self.intensity[fibre])        
                self.continuum[fibre] = np.interp(self.wavelength,
                                                         self.wavelength - wavelength_correction[fibre], self.continuum[fibre])        
            fibre_throughput = np.nansum(self.line_throughput*line_weight[:, np.newaxis], axis=0) / total_weight
            self.intensity /= fibre_throughput[:, np.newaxis]
            fluxcal = np.nanmean(self.counts) / np.nanmean(self.intensity)
            print(fluxcal, fibre_throughput[302])
            self.intensity *= fluxcal
            self.continuum *= fluxcal
            self.continuum /= fibre_throughput[:, np.newaxis]
            self.f_sky = self.estimate_sky()
            self.line_offset, self.line_throughput = self.trace_sky_lines()

        for i in range(2):
            # TODO: flux-conserving interpolation
            wavelength_correction = np.nanmedian(self.line_offset, axis=0)
            for fibre in range(self.n_fibres):
                self.intensity[fibre] = np.interp(self.wavelength,
                                                         self.wavelength - wavelength_correction[fibre], self.intensity[fibre])        
                self.continuum[fibre] = np.interp(self.wavelength,
                                                         self.wavelength - wavelength_correction[fibre], self.continuum[fibre])        
            fibre_throughput = np.nanmedian(self.line_throughput, axis=0)
            self.intensity /= fibre_throughput[:, np.newaxis]
            fluxcal = np.nanmean(self.counts) / np.nanmean(self.intensity)
            print(fluxcal, fibre_throughput[302])
            self.intensity *= fluxcal
            self.continuum *= fluxcal
            self.continuum /= fibre_throughput[:, np.newaxis]
            self.f_sky = self.estimate_sky()
            self.line_offset, self.line_throughput = self.trace_sky_lines()


    def find_sky_lines(self, min_separation=5):
        '''Find continuum and identify strong sky emission lines'''
        
        # Find continuum for all fibres:
        print(f"> Find continuum for {self.n_fibres} fibres:")
        t0 = time()
        continuum = np.empty_like(self.counts)
        for i, spectrum in enumerate(self.intensity):
            continuum[i] = find_continuum(self.wavelength, spectrum, min_separation)
        print(f"  Done ({time()-t0:.3g} s)")

        # Identify strong sky emission lines:
        #line_fraction = (self.intensity - continuum) / (self.intensity + continuum)
        line_fraction = 1 - continuum/self.intensity
        p16, p50 = np.nanpercentile(line_fraction, [16, 50])
        line_threshold = p50 + 3*(p50-p16)

        #line_mask = np.all(line_fraction > line_threshold, axis=0)
        line_mask = np.nanmedian(line_fraction, axis=0) > line_threshold
        line_mask[0] = False
        line_mask[-1] = False

        line_left = np.where(~line_mask[:-1] & line_mask[1:])[0]
        line_right = np.where(line_mask[:-1] & ~line_mask[1:])[0]
        line_right += 1
        print(f'  {line_left.size} strong sky lines ({np.count_nonzero(line_mask)} out of {self.wavelength.size} wavelengths with line fraction > {line_threshold:.3f})')
        min_line_width = int(2 * np.nanpercentile(line_right-line_left, 50))
        line_left = np.fmin(line_left, (line_left + line_right - min_line_width + 1) // 2).clip(0, self.wavelength.size)
        line_right = np.fmax(line_right, (line_left + line_right + min_line_width + 1) // 2).clip(0, self.wavelength.size)

        t = Table((line_left, line_right), names=('left', 'right'))
        t.add_column(0., name='wavelength')
        t.add_column(0., name='counts')
        t.add_column(0., name='reference_wavelength')
        t.add_column(0., name='reference_flux')
        for line in t:
            wavelength = self.wavelength[line['left']:line['right']]
            inside = np.where((self.sky_lines['wavelength'] >= wavelength[0])
                              & (self.sky_lines['wavelength'] < wavelength[-1]))
            line['reference_flux'] = np.nansum(self.sky_lines[inside]['flux'])
            line['reference_wavelength'] = np.nansum(self.sky_lines[inside]['flux'] * self.sky_lines[inside]['wavelength'])
            line['reference_wavelength'] /= line['reference_flux']
        return continuum, t[t['reference_flux'] > 0]
        
        
    def estimate_sky(self, max_bins=101):
        '''Linear fit at every wavelength for the spaxels with fainter flux'''
        print(f"> Estimating sky:")
        t0 = time()

        flux = np.nanmean(self.intensity, axis=1)
        #flux -= np.nanmin(flux)
        #faint = np.where(flux < 2*np.nanmedian(flux))[0]
        flux -= np.nanmean(np.nanmedian(self.intensity[self.sky_fibres], axis=0))
        faint = np.where(flux < 2*np.nanmedian(-flux[flux < 0]))[0]

        f_sky = np.zeros_like(self.wavelength)
        for i in range(self.wavelength.size):
            valid = np.isfinite(self.intensity[faint, i])
            if np.count_nonzero(valid):
                slope, f_sky[i] = np.polyfit(flux[faint][valid], self.intensity[faint, i][valid], 1)
                if np.isnan(f_sky[i]) or slope < 0:
                    f_sky[i] = np.nanmean(self.intensity[faint, i])
            else:
                f_sky[i] = np.nan

        print(f"  Done ({time()-t0:.3g} s)")
        return f_sky


    def trace_sky_lines(self, min_separation=5):
        '''Trace the wavelength and intensity of sky emission lines'''

        reference_spectrum = self.f_sky.copy()
        reference_spectrum -= find_continuum(self.wavelength, reference_spectrum, min_separation)
        
        for line in self.strong_sky_lines:
            wavelength = self.wavelength[line['left']:line['right']]
            spectrum = reference_spectrum[line['left']:line['right']]
            weight = spectrum
            #weight = spectrum**2
            #weight = spectrum**3
            line['wavelength'] = np.nansum(weight*wavelength) / np.nansum(weight)
            line['counts'] = np.nanmean(spectrum)
            #line['counts'] = np.nanstd(spectrum)

        #line_weight = self.strong_sky_lines['counts']
        #total_weight = np.nansum(line_weight)
        
        # Trace line wavelengths for every fibre to compare with the reference spectrum:

        line_fibre_wavelength = []
        line_fibre_intensity = []
        for line in self.strong_sky_lines:
            left = line['left']
            right = line['right']
            weight = (self.intensity[:, left:right] - self.continuum[:, left:right])
            #weight = self.intensity[:, left:right]**2
            line_fibre_wavelength.append(np.nansum(weight * self.wavelength[np.newaxis, left:right], axis=1) / np.nansum(weight, axis=1))
            line_fibre_intensity.append(np.nanmean(self.intensity[:, left:right] - self.continuum[:, left:right], axis=1))
            #line_fibre_intensity.append(np.nanstd(self.intensity[:, left:right] - self.continuum[:, left:right], axis=1))
            
        line_offset = np.array(line_fibre_wavelength) - self.strong_sky_lines['wavelength'][:, np.newaxis]
        line_offset -= np.nanmedian(line_offset, axis=1)[:, np.newaxis]
        #line_offset -= np.nanmean(line_offset, axis=1)[:, np.newaxis]
        #line_offset -= (np.nansum(line_offset*line_weight[:, np.newaxis], axis=1)/total_weight)[:, np.newaxis]

        line_throughput = np.array(line_fibre_intensity) / self.strong_sky_lines['counts'][:, np.newaxis]
        line_throughput /= np.nanmedian(line_throughput)
        #line_throughput /= np.nanmedian(line_throughput, axis=1)[:, np.newaxis]
        #line_throughput /= np.nanmean(line_throughput, axis=1)[:, np.newaxis]
        #line_throughput /= (np.nansum(line_throughput*line_weight[:, np.newaxis], axis=1)/total_weight)[:, np.newaxis]
        
        return line_offset, line_throughput

rss = WEAVE_RSS(args.filename)


# ----------------------------------------------------------------------
# # QC tests
# ----------------------------------------------------------------------

# ## Wavelength correction
# ----------------------------------------------------------------------

def test_relative_wavelength_correction(self):
    '''Relative line offset, before and after correction'''
    fig, axes = new_figure('relative_wavelength_correction', nrows=4)
    
    
    ax = axes[0, 0]
    ax.set_ylabel('original $\Delta\lambda$ [$\AA$]')
    ax.set_ylim(-.25, .25)

    p16, p50, p84 = np.nanpercentile(self.original_offset, [16, 50, 84], axis=0)
    ax.plot(p50, 'k-', alpha=.5)
    ax.fill_between(np.arange(self.n_fibres), p16, p84, color='k', alpha=.1)
    
    #for line in self.sky_fibres[0]:
    #    ax.axvline(line, c='b', ls='-', alpha=.2)
    cb = fig.colorbar(None, ax=ax)
    cb.remove()

    
    ax = axes[1, 0]
    ax.set_ylabel('new $\Delta\lambda$ [$\AA$]')
    ax.set_ylim(-.25, .25)
    ax.get_shared_y_axes().join(ax, axes[0, 0])

    p16, p50, p84 = np.nanpercentile(self.line_offset, [16, 50, 84], axis=0)
    ax.plot(p50, 'k-', alpha=.5)
    ax.fill_between(np.arange(self.n_fibres), p16, p84, color='k', alpha=.1)
    cb = fig.colorbar(None, ax=ax)
    cb.remove()

    
    ax = axes[2, 0]
    ax.set_ylim(-1.5, len(self.strong_sky_lines)+.5)
    im, cb = colour_map(fig, ax, 'original $\Delta\lambda$ [$\AA$]', self.original_offset,
                        xlabel='fibre', ylabel='line ID (increasing $\lambda$)', cmap='turbo', norm=colors.Normalize(vmin=-.2, vmax=.2))

    ax = axes[3, 0]
    ax.set_ylim(-1.5, len(self.strong_sky_lines)+.5)
    ax.get_shared_y_axes().join(ax, axes[2, 0])
    im, cb = colour_map(fig, ax, 'new $\Delta\lambda$ [$\AA$]', self.line_offset,
                        xlabel='fibre', ylabel='line ID (increasing $\lambda$)', cmap='turbo', norm=colors.Normalize(vmin=-.2, vmax=.2))

    plt.savefig(f'{self.filename}-{fig.get_label()}.pdf')
    plt.savefig(f'{self.filename}-{fig.get_label()}.png')


test_relative_wavelength_correction(rss)


def test_absolute_wavelength_calibration(self):
    '''Compare measured wavelength of sky lines with UVES atlas'''
    fig, axes = new_figure('absolute_wavelength_calibration', ncols=2)

    
    ax = axes[0, 0]
    ax.set_ylabel(r'$\Delta\lambda\ [\AA]$')
    ax.set_xlabel(r'wavelength [$\AA$]')
    
    ratio = self.strong_sky_lines['wavelength'] - self.strong_sky_lines['reference_wavelength']
    really_strong = np.where(self.strong_sky_lines['reference_flux'] > np.nanmean(self.strong_sky_lines['reference_flux']))
    ax.plot(self.strong_sky_lines['wavelength'][really_strong], ratio[really_strong], 'co')
    ax.plot(self.strong_sky_lines['wavelength'], ratio, 'k+')
    
    #p16_o, p50_o, p84_o = np.nanpercentile(self.line_offset, [16, 50, 84], axis=1)
    #ax.errorbar(self.strong_sky_lines['wavelength'], p50_o, yerr=(np.abs(p16_o), p84_o), fmt='rx')
    #ax.fill_between(self.strong_sky_lines['wavelength'], p16_o, p84_o, color='k', alpha=.1)

    p16, p50, p84 = np.nanpercentile(ratio, [16, 50, 84])
    ax.axhline(p50, c='k', ls=':', alpha=.5)
    ax.fill_between(self.strong_sky_lines['wavelength'], p16, p84, color='k', alpha=.1)

    '''
    heliocentric_correction = np.nanmean(self.fibtable['Helio_cor']) / c.c.to_value(u.km/u.s)
    ratio_ = self.strong_sky_lines['wavelength'] - self.strong_sky_lines['reference_wavelength']/(1+heliocentric_correction)
    p16_, p50_, p84_ = np.nanpercentile(ratio_, [16, 50, 84])
    ax.plot(self.strong_sky_lines['wavelength'], ratio_, 'r.', alpha=.2, label=f'no heliocentric correction: [{p16_:.2f}, {p50_:.2f}, {p84_:.2f}]')
    ax.legend()
    '''    
    
    ax = axes[0, 1]
    ax.set_xlabel(r'counts')
    
    ax.plot(self.strong_sky_lines['counts'][really_strong], ratio[really_strong], 'co')
    ax.plot(self.strong_sky_lines['counts'], ratio, 'k+')
    #ax.errorbar(self.strong_sky_lines['counts'], p50_o, yerr=(np.abs(p16_o), p84_o), fmt='rx')

    ax.axhline(p50, c='k', ls=':', alpha=.5, label=f'{p50:.2f}')
    ax.fill_between(np.sort(self.strong_sky_lines['counts']), p16, p84, color='k', alpha=.1, label=f'{p16:.2f} - {p84:.2f}')
    ax.legend()
    
test_absolute_wavelength_calibration(rss)


# ## Fibre throughput
# ----------------------------------------------------------------------


def test_fibre_throughput(self):
    '''Fibre throughput, before and after correction'''
    fig, axes = new_figure('fibre_throughput', nrows=4)

    
    ax = axes[0, 0]
    ax.set_ylabel('original throughput')
    ax.set_ylim(.91, 1.09)

    p16, p50, p84 = np.nanpercentile(self.original_throughput, [16, 50, 84], axis=0)
    ax.plot(p50, 'k-', alpha=.5)
    ax.fill_between(np.arange(self.n_fibres), p16, p84, color='k', alpha=.1)
    ax.plot(self.sky_fibres[0], p50[self.sky_fibres], 'c.', alpha=1)
    
    #for line in self.sky_fibres[0]:
    #    ax.axvline(line, c='b', ls='-', alpha=.2)
    cb = fig.colorbar(None, ax=ax)
    cb.remove()

    
    ax = axes[1, 0]
    ax.set_ylabel('new throughput')
    #ax.set_ylim(.75, 1.25)
    ax.set_ylim(.91, 1.09)
    ax.get_shared_y_axes().join(ax, axes[0, 0])

    p16, p50, p84 = np.nanpercentile(self.line_throughput, [16, 50, 84], axis=0)
    ax.plot(p50, 'k-', alpha=.5)
    ax.fill_between(np.arange(self.n_fibres), p16, p84, color='k', alpha=.1)
    ax.plot(self.sky_fibres[0], p50[self.sky_fibres], 'c.', alpha=1)
    cb = fig.colorbar(None, ax=ax)
    cb.remove()

    
    ax = axes[2, 0]
    ax.set_ylim(-1.5, len(self.strong_sky_lines)+.5)
    im, cb = colour_map(fig, ax, 'original throughput', self.original_throughput,
                        xlabel='fibre', ylabel='line ID (increasing $\lambda$)', cmap='turbo', norm=colors.Normalize(vmin=.8, vmax=1.2))

    ax = axes[3, 0]
    ax.set_ylim(-1.5, len(self.strong_sky_lines)+.5)
    ax.get_shared_y_axes().join(ax, axes[2, 0])
    im, cb = colour_map(fig, ax, 'new throughput', self.line_throughput,
                        xlabel='fibre', ylabel='line ID (increasing $\lambda$)', cmap='turbo', norm=colors.Normalize(vmin=.8, vmax=1.2))

    plt.savefig(f'{self.filename}-{fig.get_label()}.pdf')
    plt.savefig(f'{self.filename}-{fig.get_label()}.png')


test_fibre_throughput(rss)


def test_throughput_map(self):
    '''Spatial distribution and dependence on total flux'''
    flux = np.nanmean(self.intensity, axis=1)
    fig, axes = new_figure('throughput_map', nrows=2)
    
    ax = axes[0, 0]
    ax.set_ylabel(r'original throughput')
    ax.set_ylim(.65, 1.55)
    p16, p50, p84 = np.nanpercentile(self.original_throughput, [16, 50, 84], axis=0)
    ax.errorbar(flux, p50, yerr=(p50-p16, p84-p50), fmt='k+', alpha=.2)
    ax.errorbar(flux[self.sky_fibres], p50[self.sky_fibres],
                yerr=(p50[self.sky_fibres]-p16[self.sky_fibres], p84[self.sky_fibres]-p50[self.sky_fibres]),
                fmt='c+', alpha=1)
    
    ax = axes[1, 0]
    ax.set_ylabel(r'new throughput')
    ax.set_ylim(.65, 1.55)
    #ax.set_ylim(.8, 1.2)
    #ax.set_yscale('log')
    p16, p50, p84 = np.nanpercentile(self.line_throughput, [16, 50, 84], axis=0)
    ax.errorbar(flux, p50, yerr=(p50-p16, p84-p50), fmt='k+', alpha=.2)
    ax.errorbar(flux[self.sky_fibres], p50[self.sky_fibres],
                yerr=(p50[self.sky_fibres]-p16[self.sky_fibres], p84[self.sky_fibres]-p50[self.sky_fibres]),
                fmt='c+', alpha=1)

    ax.set_xlabel(r'mean intensity')
    ax.set_xscale('log')


test_throughput_map(rss)


# ## Sky subtraction
# ----------------------------------------------------------------------


def test_sky_spectrum(self):
    '''Compare different methods to estimate the sky spectrum. Show average subtracted spectrum, as well as fibre with maximuum signal.'''
    fig, axes = new_figure('sky_subtraction', nrows=3)#, figsize=(10, 8))

    ax = axes[0, 0]
    ax.set_ylabel('sky counts')
    #ax.set_yscale('log')
    ax.plot(self.wavelength, self.sky_counts[0], 'r-', alpha=.5, label='original sky')
    ax.plot(self.wavelength, self.f_sky, 'k-', alpha=1, label='linear fit')
    p16, p50, p84 = np.nanpercentile(self.intensity[self.sky_fibres], [16, 50, 84], axis=0)
    ax.plot(self.wavelength, p50, 'b-', alpha=.5, label='median')
    ax.fill_between(self.wavelength, p16, p84, color='b', alpha=.25)
    ax.legend()

    ax = axes[1, 0]
    ax.set_ylabel('mean spectrum')
    ax.plot(self.wavelength, np.nanmean(self.counts, axis=0) - self.sky_counts[0], 'r-', alpha=.5, label='original mean spectrum')
    ax.plot(self.wavelength, np.nanmean(self.intensity, axis=0) - self.f_sky, 'k-', alpha=1, label='new mean spectrum')
    #ax.plot(self.wavelength, np.nanmedian(self.intensity, axis=0) - self.f_sky, 'b-', alpha=1, label='new median spectrum')
    ax.legend()

    ax = axes[2, 0]
    peak = np.nanargmax(np.nanmean(self.intensity, axis=1))
    ax.set_ylabel(f'fibre {peak} (peak)')
    ax.plot(self.wavelength, self.counts[peak] - self.sky_counts[0], 'r-', alpha=.5, label=f'fibre {peak} (original)')
    ax.plot(self.wavelength, self.intensity[peak] - self.f_sky, 'k-', alpha=1, label=f'fibre {peak} (new)')

    ax.set_xlabel(r'wavelength [$\AA$]')
    '''
    for ax in axes.ravel():
        for line in self.strong_sky_lines:
            ax.axvspan(self.wavelength[line['left']], self.wavelength[line['right']], color='b', alpha=.1)
    '''

    plt.savefig(f'{self.filename}-{fig.get_label()}.pdf')
    plt.savefig(f'{self.filename}-{fig.get_label()}.png')


test_sky_spectrum(rss)


# ## Flux calibration
# ----------------------------------------------------------------------

flux_units = u.erg/u.s/u.cm**2/u.Angstrom
log_fibre_solid_angle_arcsec2 = 2.5 * np.log10(np.pi * 1.305**2)

from pst.observables import load_photometric_filters

filter_names = ["SLOAN_SDSS.r", "SLOAN_SDSS.i"]
filter_colors = ['b', 'g', 'r', 'y']
filter_list = load_photometric_filters(filter_names)
for band in filter_list:
    band.interpolate(rss.wavelength)


synthetic_photo = []
for band in filter_list:
    mag_ab = []
    for spectrum in rss.flux << flux_units:
        mag_ab.append(band.get_ab(spectrum)[0] + log_fibre_solid_angle_arcsec2)
    synthetic_photo.append(np.where(np.isfinite(mag_ab), mag_ab, 30))

synthetic_photo_new = []
for band in filter_list:
    mag_ab = []
    for spectrum in (rss.intensity - rss.f_sky) * rss.sensitivity_function << flux_units:
        mag_ab.append(band.get_ab(spectrum)[0] + log_fibre_solid_angle_arcsec2)
    synthetic_photo_new.append(np.where(np.isfinite(mag_ab), mag_ab, 30))


sky_ab = []
for band in filter_list:
    sky_ab.append(band.get_ab(rss.f_sky * rss.sensitivity_function[0] << u.erg/u.s/u.cm**2/u.Angstrom)[0] + log_fibre_solid_angle_arcsec2)
sky_ab


# synthetic_photo

fibtable_names = ["MAG_R", "MAG_I"]
n_bands = len(fibtable_names)

fig, axes = new_figure('synthetic_photo', nrows=n_bands, ncols=2, figsize=(12, 8))
for ax in axes[-1, :]:
    ax.set_xlabel("observed AB magnitude")

for band in range(n_bands):
    obs_photo = rss.fibtable[fibtable_names[band]] + log_fibre_solid_angle_arcsec2
    
    delta_mag = synthetic_photo[band] - obs_photo
    p16, p50, p84 = np.nanpercentile(delta_mag[synthetic_photo[band] < 24], [16, 50, 84])
    ax = axes[band, 0]
    ax.plot(obs_photo, delta_mag, c=filter_colors[band], marker='o', ls='', alpha=.25)
    ax.axhline(p50, c='k', ls=':', label=f'median={p50:.2f}')
    ax.axhspan(p16, p84, color='k', alpha=.1, label=f'{p16:.2f} - {p84:.2f}')
    ax.axvline(24, c='k', ls=':')
    ax.axvline(sky_ab[band], c='k', ls='--', label=f'm_sky = {sky_ab[band]:.2f}')
    ax.legend(title = filter_names[band])
    ax.set_ylabel("synthetic - observed")
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlim(18.5, 25.5)
        
    delta_mag = synthetic_photo_new[band] - obs_photo
    p16, p50, p84 = np.nanpercentile(delta_mag[synthetic_photo[band] < 24], [16, 50, 84])
    ax = axes[band, 1]
    ax.plot(obs_photo, delta_mag, c=filter_colors[band], marker='o', ls='', alpha=.25)
    ax.axhline(p50, c='k', ls=':', label=f'median={p50:.2f}')
    ax.axhspan(p16, p84, color='k', alpha=.1, label=f'{p16:.2f} - {p84:.2f}')
    ax.axvline(24, c='k', ls=':')
    ax.axvline(sky_ab[band], c='k', ls='--', label=f'm_sky = {sky_ab[band]:.2f}')
    ax.legend(title = filter_names[band])
    ax.set_xlim(18.5, 25.5)

plt.savefig(f'{rss.filename}-{fig.get_label()}.png')


# photo_skymap


def fibre_skymap(fig, ax, rss, data, cblabel='', cmap=default_cmap, norm=None):
    
    if norm is None:
        percentiles = np.array([1, 16, 50, 84, 99])
        ticks = np.nanpercentile(data, percentiles)
        linthresh = np.median(data[data > 0])
        norm = colors.SymLogNorm(vmin=ticks[0], vmax=ticks[-1], linthresh=linthresh)
    else:
        ticks = None

    sc = ax.scatter(rss.fibtable['FIBRERA'], rss.fibtable['FIBREDEC'], c=data, cmap=cmap, norm=norm)
    ax.set_aspect('equal')

    cb = fig.colorbar(sc, ax=ax, orientation='vertical', shrink=.9)
    cb.ax.set_ylabel(cblabel)
    if ticks is not None:
        cb.ax.set_yticks(ticks=ticks, labels=[f'{value:.3g} ({percent}%)' for value, percent in zip(ticks, percentiles)])
    cb.ax.tick_params(labelsize='small')
    
    return sc, cb

fibtable_names = ["MAG_R", "MAG_I"]
n_bands = len(fibtable_names)

fig, axes = new_figure('photo_skymap', nrows=n_bands+1, ncols=3, figsize=(12, 8), sharex=True, sharey=True)
norm = colors.Normalize(vmin=19.5, vmax=25.5)
cmap = "nipy_spectral_r"

for band in range(n_bands):
    fibre_skymap(fig, axes[band, 0], rss, rss.fibtable[fibtable_names[band]] + log_fibre_solid_angle_arcsec2, norm=norm, cmap=cmap, cblabel=f"{filter_names[band]} (observed)")
    fibre_skymap(fig, axes[band, 1], rss, synthetic_photo[band], norm=norm, cmap=cmap, cblabel=f"{filter_names[band]} (original)")
    fibre_skymap(fig, axes[band, 2], rss, synthetic_photo_new[band], norm=norm, cmap=cmap, cblabel=f"{filter_names[band]} (new)")

obs_colour = rss.fibtable[fibtable_names[0]] - rss.fibtable[fibtable_names[1]]
#vmin, vmax = np.nanpercentile(obs_colour, [10, 90])
vmin, vmax = np.nanpercentile(synthetic_photo[0]-synthetic_photo[1], [10, 90])
norm = colors.Normalize(vmin=vmin, vmax=vmax)
cmap = "rainbow"
fibre_skymap(fig, axes[-1, 0], rss, obs_colour, norm=norm, cmap=cmap, cblabel="observed colour")
fibre_skymap(fig, axes[-1, 1], rss, synthetic_photo[0]-synthetic_photo[1], norm=norm, cmap=cmap, cblabel="original colour")
fibre_skymap(fig, axes[-1, 2], rss, synthetic_photo_new[0]-synthetic_photo_new[1], norm=norm, cmap=cmap, cblabel="new colour")

axes[0, 0].set_xlim(np.nanpercentile(rss.fibtable['FIBRERA'][rss.target_fibres], [0, 100]))
axes[0, 0].set_ylim(np.nanpercentile(rss.fibtable['FIBREDEC'][rss.target_fibres], [0, 100]))

plt.savefig(f'{rss.filename}-{fig.get_label()}.png')


# delta_mag_skymap

fibtable_names = ["MAG_R", "MAG_I"]
n_bands = len(fibtable_names)

fig, axes = new_figure('delta_mag_skymap', nrows=n_bands+1, ncols=2, figsize=(10, 8))
norm = colors.Normalize(vmin=-.5, vmax=.5)
cmap = "rainbow"

for band in range(n_bands):
    obs_photo = rss.fibtable[fibtable_names[band]] + log_fibre_solid_angle_arcsec2
    
    ax = axes[band, 0]
    delta_mag = synthetic_photo[band] - obs_photo
    fibre_skymap(fig, ax, rss, delta_mag, norm=norm, cmap=cmap, cblabel=f"{filter_names[band]} (original - observed)")
    
    ax = axes[band, 1]
    delta_mag = synthetic_photo_new[band] - obs_photo
    fibre_skymap(fig, ax, rss, delta_mag, norm=norm, cmap=cmap, cblabel=f"{filter_names[band]} (new - observed)")

norm = colors.Normalize(vmin=-.25, vmax=.25)
cmap = "terrain"
fibre_skymap(fig, axes[-1, 0], rss, (synthetic_photo[0] - synthetic_photo[1]) - obs_colour, norm=norm, cmap=cmap, cblabel="original colour difference")
fibre_skymap(fig, axes[-1, 1], rss, (synthetic_photo_new[0] - synthetic_photo_new[1]) - obs_colour, norm=norm, cmap=cmap, cblabel="new colour difference")

plt.savefig(f'{rss.filename}-{fig.get_label()}.png')


# fibre_photo

fibtable_names = ["MAG_R", "MAG_I"]
n_bands = len(fibtable_names)

fig, axes = new_figure('fibre_photo', nrows=n_bands+1, ncols=3, width_ratios=(1, 1, .05), figsize=(12, 8), sharey=False)
for ax in axes[-1, :]:
    ax.set_xlabel("fibre")

norm_mag = colors.Normalize(vmin=19.5, vmax=25.5)
cmap_mag = "nipy_spectral_r"
cm_mag = plt.cm.ScalarMappable(norm=norm_mag, cmap=cmap_mag)
norm_colour = colors.Normalize(vmin=-.5, vmax=.5)
cmap_colour = "rainbow"

for band in range(n_bands):
    obs_photo = rss.fibtable[fibtable_names[band]] + log_fibre_solid_angle_arcsec2
    
    delta_mag = synthetic_photo[band] - obs_photo
    p16, p50, p84 = np.nanpercentile(delta_mag[synthetic_photo[band] < 24], [16, 50, 84])
    ax = axes[band, 0]
    sc = ax.scatter(np.arange(rss.n_fibres), delta_mag, c=obs_photo, cmap=cmap_mag, norm=norm_mag)
    ax.axhline(p50, c='k', ls=':', label=f'median={p50:.2f}')
    ax.axhspan(p16, p84, color='k', alpha=.1, label=f'{p16:.2f} - {p84:.2f}')
    ax.legend(title = filter_names[band])
    ax.set_ylabel("synthetic - observed")
    ax.set_ylim(-1.15, 1.15)
        
    delta_mag = synthetic_photo_new[band] - obs_photo
    p16, p50, p84 = np.nanpercentile(delta_mag[synthetic_photo[band] < 24], [16, 50, 84])
    ax = axes[band, 1]
    sc = ax.scatter(np.arange(rss.n_fibres), delta_mag, c=obs_photo, cmap=cmap_mag, norm=norm_mag)
    ax.axhline(p50, c='k', ls=':', label=f'median={p50:.2f}')
    ax.axhspan(p16, p84, color='k', alpha=.1, label=f'{p16:.2f} - {p84:.2f}')
    ax.legend(title = filter_names[band])
    ax.set_ylim(-1.15, 1.15)
    ax.yaxis.set_ticklabels('')

    plt.colorbar(cm_mag, cax=axes[band, -1], label=filter_names[band])

ax = axes[-1, 0]
delta_colour = (synthetic_photo[0] - synthetic_photo[1]) - obs_colour
p16, p50, p84 = np.nanpercentile(delta_colour[synthetic_photo[band] < 24], [16, 50, 84])
ax.axhline(p50, c='k', ls=':', label=f'median={p50:.2f}')
ax.axhspan(p16, p84, color='k', alpha=.1, label=f'{p16:.2f} - {p84:.2f}')
sc = ax.scatter(np.arange(rss.n_fibres), delta_colour, c=obs_photo, cmap=cmap_mag, norm=norm_mag)
ax.set_ylim(-1.15, 1.15)
ax.set_ylabel("synthetic - observed colour")
ax.legend(title="colour")

ax = axes[-1, 1]
delta_colour = (synthetic_photo_new[0] - synthetic_photo_new[1]) - obs_colour
p16, p50, p84 = np.nanpercentile(delta_colour[synthetic_photo[band] < 24], [16, 50, 84])
p16, p50, p84 = np.nanpercentile(delta_colour[synthetic_photo[band] < 24], [16, 50, 84])
ax.axhline(p50, c='k', ls=':', label=f'median={p50:.2f}')
ax.axhspan(p16, p84, color='k', alpha=.1, label=f'{p16:.2f} - {p84:.2f}')
sc = ax.scatter(np.arange(rss.n_fibres), delta_colour, c=obs_photo, cmap=cmap_mag, norm=norm_mag)
ax.set_ylim(-1.15, 1.15)
ax.yaxis.set_ticklabels('')
ax.legend(title="colour")

plt.colorbar(sc, cax=axes[-1, -1], label=filter_names[band])

plt.savefig(f'{rss.filename}-{fig.get_label()}.png')
