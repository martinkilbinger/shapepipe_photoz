
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 10:02:58 2020

@author: Xavier Jimenez
"""


#------------------------------------------------------------------#
# # # # # Imports # # # # #
#------------------------------------------------------------------#

import numpy as np
import os
import shutil
import glob
from numpy.compat.py3k import contextlib_nullcontext
import pandas as pd
import importlib
import matplotlib.pyplot as plt
import matplotlib.style as style
from pandas.core.base import NoNewAttributesMixin
import seaborn as sns
# from sklearn import kernel_ridge
# from sklearn.utils import validation
style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot')
from matplotlib.colors import LogNorm

import astropy.units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.stats import mad_std

from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline

from hyperopt import tpe, hp, fmin, STATUS_OK,Trials
from hyperopt.pyll.base import scope

from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVR


#------------------------------------------------------------------#
# # # # # Functions # # # # #
#------------------------------------------------------------------#


class GenerateFiles(object):
    """[summary]
    """

    def __init__(self, survey, bands, temp_path, output_name, output_path=None):
        """[summary]

        Args:
            survey ([type]): [description]
            bands ([type]): [description]
            temp_path ([type]): [description]
        """

        self._path = os.getcwd() +'/'
        self.survey = survey
        self._bands = bands
        self.temp_path = temp_path
        self.output_name = output_name
        if output_path is None:
            self.output_path = self._path
        else:
            self.output_path = output_path


    def make_directory(self, path_to_file):
        """[summary]

        Args:
            filename ([type]): [description]
        """

        try:
            os.mkdir(path_to_file)
        except OSError:
            pass
        else:
            print ("Successfully created the directory %s " % path_to_file)


    def make_directories(self, output=False):
        """[summary]
        """

        if output == False:

            self.make_directory(self.temp_path + self.survey)

            temp_directories = ['matched', 'unmatched', 'vignet', 'spectral_surveys', 'd2d', 'redshift']

            for temp_name in temp_directories:
                self.make_directory(self.temp_path + self.survey + '/' + temp_name)

            for band in self._bands:
                self.make_directory(self.temp_path + self.survey + '/vignet/' + band)

            self.make_directory(self.temp_path + self.survey + '/vignet/array')

        elif output == True:
            self.make_directory(self.output_path + 'output')
            self.make_directory(self.output_path + 'output/' + self.survey)
            self.make_directory(self.output_path + 'output/' + self.survey + '/' + self.output_name)
            self.make_directory(self.output_path + 'output/' + self.survey + '/' + self.output_name + '/files')
            self.make_directory(self.output_path + 'output/' + self.survey + '/' + self.output_name + '/files/ML')
            self.make_directory(self.output_path + 'output/' + self.survey + '/' + self.output_name + '/figures')




    def remove_files_from_directory(self, folder):
        """Removes files for a given directory

        Args:
            folder ([str]): directory path
        """

        for file_name in os.listdir(folder):
            file_path = os.path.join(folder, file_name)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


    def clean_temp_directories(self):
        """[summary]
        """

        if os.path.exists(self.temp_path + self.survey) and os.path.isdir(self.temp_path + self.survey):
            if not os.listdir(self.temp_path + self.survey):
                print("Directory %s is empty"%(self.temp_path + self.survey))
            else:
                temp_directories = ['/d2d/', '/matched/',  '/unmatched/', '/vignet/', '/redshift/', '/spectral_surveys/']
                for directory in temp_directories:
                    self.remove_files_from_directory(self.temp_path + self.survey + directory)
                    print("Successfully removed the directory %s " % (self.temp_path + self.survey + directory))
        else:
            print("Directory %s does not exists"%(self.temp_path + self.survey))


    def is_directory_empty(self, path_to_dir):
        """Checks if given directory is empty or not.

        Args:
            path_to_dir ([str]): path to the directory

        Returns:
            [bool]: True if directory is empty, False if not.
        """

        if os.path.exists(path_to_dir) and os.path.isdir(path_to_dir):
            if not os.listdir(path_to_dir):
                print("Directory %s is empty"%path_to_dir)
                return True
            else:
                print("Directory %s is not empty"%path_to_dir)
                return False
        else:
            print("Directory %s don't exists"%path_to_dir)

class GeneratePlots(object):

    def __init__(self, survey, bands, temp_path, output_name, spectral_names, figure_name='match_cat', output_path=None):

        self._path = os.getcwd() + '/'
        self.survey = survey
        self._bands = bands
        self.temp_path = temp_path
        self.figure_name = figure_name
        self.output_name = output_name
        self.spectral_names = spectral_names
        if output_path is None:
            self.output_path = self._path
        else:
            self.output_path = output_path
        self.figure_path = self.output_path + 'output/' + self.survey + '/' + self.output_name + '/figures/'

        self.df_matched = pd.read_csv(self.output_path + 'output/' + self.survey + '/' + self.output_name + '/files/'  + self.output_name + '.csv')
        self.df_unmatched = pd.read_csv(self.output_path + 'output/' + self.survey + '/' + self.output_name + '/files/' + self.output_name + '_unmatched'+'.csv')
        # self.df_d2d = pd.read_csv(self.output_path + 'output/' + self.survey + '/files/' + self.output_name +'_d2d'+'.csv')


    def plot_d2d(self):
        """
        Distance distribution plot for matched galaxies
        """

        fig =plt.figure(figsize=(8,8), tight_layout=False)

        ax = fig.add_subplot(111)
        ax.set_facecolor('white')
        ax.grid(True, color='grey', lw=0.5)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.set_xlabel(r'$\mathrm{angular\;distance\;\left(arcsec\right)}$', fontsize=20)
        ax.set_ylabel(r'$\mathrm{matched\;galaxies\;in}$'+' band R', fontsize=20)

        ax.hist(np.array(self.df_d2d['d2d'].values)*3600, bins = 50)

        plt.savefig(self.figure_path + self.figure_name + '.pdf', bbox_inches='tight', transparent=True)
        plt.show()
        plt.close()

    def plot_mags(self, df_matched, df_unmatched):
        """
        
        """

        mags = ['MAG_AUTO_R', 'MAG_u', 'i_stk_aper', 'z_stk_aper', 'g_stk_aper']

        fig =plt.figure(figsize=(8,20), tight_layout=False)

        for (i,mag) in enumerate(mags):
            ax = fig.add_subplot(511+i)
            ax.set_facecolor('white')
            ax.grid(True, color='grey', lw=0.5)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            if i == len(mags)-1:
                ax.set_xlabel('Magnitude', fontsize=20)
            ax.set_ylabel(mag, fontsize=20)

            ax.hist(df_unmatched[mag], density=True, bins=50, label='unmatched', log=True, color='orange')
            ax.hist(df_matched[mag], density=True, bins=50, label='matched', log=True, alpha=0.8)
            legend = ax.legend(loc='best', shadow=True, fontsize='x-large')

        plt.savefig(self.figure_path + 'Mags_' + self.figure_name + '.pdf', bbox_inches='tight', transparent=True)
        plt.show()
        plt.close()

    def plot_matched_z_spec_hist(self):
        """
        z_spec hist for matched galaxies
        """

        fig =plt.figure(figsize=(8,8), tight_layout=False)

        ax = fig.add_subplot(111)
        ax.set_facecolor('white')
        ax.grid(True, color='grey', lw=0.5)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.set_xlabel(r'$\mathrm{spectral\;redshift}$', fontsize=20)
        ax.set_ylabel(r'$\mathrm{matched\;galaxies\;in}$'+' band R', fontsize=20)

        ax.axvline(np.median(self.df_matched['Z_SPEC'].values), ls='--', color='k', label='median: {:.2f}'.format(np.median(self.df_matched['Z_SPEC'].values)))
        ax.hist(self.df_matched['Z_SPEC'].values, bins = 50)

        legend = ax.legend(loc='best', shadow=True, fontsize='x-large')
        legend.get_frame().set_facecolor('C0')

        plt.savefig(self.figure_path + self.figure_name + '_z_hist' + '.pdf', bbox_inches='tight', transparent=True)
        plt.show()
        plt.close()


    def plot_matched_r_i_i_z(self):
        """
        R-I vs I-Z 2d histogram plot for matched galaxies
        """

        fig =plt.figure(figsize=(8,8), tight_layout=False)

        ax = fig.add_subplot(111)
        ax.set_facecolor('white')
        ax.grid(True, color='grey', lw=0.5)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.set_ylabel(r'$\mathrm{R-I}$', fontsize=20)
        ax.set_xlabel(r'$\mathrm{I-Z}$', fontsize=20)

        im = ax.hist2d((self.df_matched['MAG_AUTO_I'] - self.df_matched['MAG_AUTO_Z']).values, (self.df_matched['MAG_AUTO_R'] - self.df_matched['MAG_AUTO_I']).values, bins=(50,50), cmap='Blues')
        cbar = fig.colorbar(im[3])
        ax.set_xlim([-5, 5])
        ax.set_ylim([-2, 4])

        plt.suptitle(r'$\mathrm{R-I\; vs \;I-Z \;for\; matched\; galaxies}$', fontsize=20, y=1.0)
        plt.savefig(self.figure_path + 'R-I_I-Z' + '_matched' + '.pdf', bbox_inches='tight', transparent=True)
        plt.show()
        plt.close()


    def plot_matched_u_r_r_i(self):
        """
        U-R vs R-I 2d histogram plot for matched galaxies
        """

        fig =plt.figure(figsize=(8,8), tight_layout=False)

        ax = fig.add_subplot(111)
        ax.set_facecolor('white')
        ax.grid(True, color='grey', lw=0.5)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.set_ylabel(r'$\mathrm{U-R}$', fontsize=20)
        ax.set_xlabel(r'$\mathrm{R-I}$', fontsize=20)

        im = ax.hist2d((self.df_matched['MAG_AUTO_R'] - self.df_matched['MAG_AUTO_I']).values, (self.df_matched['MAG_AUTO_U'] - self.df_matched['MAG_AUTO_R']).values, bins=(50,50), cmap='Blues')
        cbar = fig.colorbar(im[3])
        ax.set_xlim([-2, 4])
        ax.set_ylim([0, 5])

        plt.suptitle(r'$\mathrm{U-R\; vs \;R-I \;for\; matched\; galaxies}$', fontsize=20, y=1.0)
        plt.savefig(self.figure_path + 'U-R_R-I' + '_matched' + '.pdf', bbox_inches='tight', transparent=True)
        plt.show()
        plt.close()


    def plot_unmatched_r_i_i_z(self):
        """
        R-I vs I-Z 2d histogram plot for all galaxies
        """

        fig =plt.figure(figsize=(8,8), tight_layout=False)

        ax = fig.add_subplot(111)
        ax.set_facecolor('white')
        ax.grid(True, color='grey', lw=0.5)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.set_ylabel(r'$\mathrm{R-I}$', fontsize=20)
        ax.set_xlabel(r'$\mathrm{I-Z}$', fontsize=20)

        im = ax.hist2d((self.df_unmatched['MAG_AUTO_I'] - self.df_unmatched['MAG_AUTO_Z']).values, (self.df_unmatched['MAG_AUTO_R'] - self.df_unmatched['MAG_AUTO_I']).values, bins=(50,50), cmap='Blues')
        cbar = fig.colorbar(im[3])
        cbar.formatter.set_powerlimits((0, 0))
        ax.set_xlim([-5, 5])
        ax.set_ylim([-2, 4])

        plt.suptitle(r'$\mathrm{R-I\; vs \;I-Z \;for\; all\; galaxies}$', fontsize=20, y=1.0)
        plt.savefig(self.figure_path + 'R-I_I-Z' + '_unmatched' + '.pdf', bbox_inches='tight', transparent=True)
        plt.show()
        plt.close()


    def plot_unmatched_u_r_r_i(self):
        """
        U-R vs R-I 2d histogram plot for all galaxies
        """

        fig =plt.figure(figsize=(8,8), tight_layout=False)

        ax = fig.add_subplot(111)
        ax.set_facecolor('white')
        ax.grid(True, color='grey', lw=0.5)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.set_ylabel(r'$\mathrm{U-R}$', fontsize=20)
        ax.set_xlabel(r'$\mathrm{R-I}$', fontsize=20)

        im = ax.hist2d((self.df_unmatched['MAG_AUTO_R'] - self.df_unmatched['MAG_AUTO_I']).values, (self.df_unmatched['MAG_AUTO_U'] - self.df_unmatched['MAG_AUTO_R']).values, bins=(50,50), cmap='Blues')
        cbar = fig.colorbar(im[3])
        cbar.formatter.set_powerlimits((0, 0))
        ax.set_xlim([-2, 4])
        ax.set_ylim([0, 5])

        plt.suptitle(r'$\mathrm{U-R\; vs \;R-I \;for\; all\; galaxies}$', fontsize=20, y=1.0)
        plt.savefig(self.figure_path + 'U-R_R-I' + '_unmatched' + '.pdf', bbox_inches='tight', transparent=True)
        plt.show()
        plt.close()


    def plot_unmatched_z_spec_hist(self):
        """
        Z unmatched distribution plot
        """

        df_Z = pd.DataFrame(data={'RA': [], 'DEC': [], 'Z_SPEC': []})
        for spectral_name in self.spectral_names:
            df_Z = pd.concat((df_Z, pd.read_csv(self.temp_path + self.survey + '/spectral_surveys/'+'z_%s'%spectral_name + '.csv')))

        fig =plt.figure(figsize=(8,8), tight_layout=False)

        ax = fig.add_subplot(111)
        ax.set_facecolor('white')
        ax.grid(True, color='grey', lw=0.5)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.set_xlabel(r'$\mathrm{spectral\;redshift}$', fontsize=20)
        ax.set_ylabel(r'$\mathrm{all\;spectral\;galaxies}$', fontsize=20)

        ax.axvline(np.median(df_Z['Z_SPEC'].values), ls='--', color='k', label='median: {:.2f}'.format(np.median(df_Z['Z_SPEC'].values)))
        ax.hist(df_Z['Z_SPEC'].values, bins = 50)

        legend = ax.legend(loc='best', shadow=True, fontsize='x-large')
        legend.get_frame().set_facecolor('C0')


        plt.savefig(self.figure_path + 'cat_' + 'z_hist' + '.pdf', bbox_inches='tight', transparent=True)
        plt.show()
        plt.close()



class MakeCatalogs(object):

    def __init__(self, survey, bands, temp_path, output_name, output_path=None):
        """[summary]

        Args:
            survey ([type]): [description]
            bands ([type]): [description]
            temp_path ([type]): [description]
        """

        self._path = os.getcwd() + '/'
        self.survey = survey
        self._bands = bands
        self.temp_path = temp_path
        self.output_name = output_name
        if output_path is None:
            self.output_path = self._path
        else:
            self.output_path = output_path


    def make_survey_catalog(self, spectral_path, spectral_name):
        """[summary]

        Args:
            spectral_path ([type]): [description]
        """

        # df = pd.DataFrame(data={'RA': [], 'DEC': [], 'Z_SPEC': []})
        # for spectral_name in spectral_names:
        HDU = fits.open(spectral_path + '/' + spectral_name + '.fits')
        RA = HDU[1].data['ra']
        DEC = HDU[1].data['dec']
        Z_SPEC = HDU[1].data['z']
        df = pd.DataFrame(data={'RA': RA.tolist(), 'DEC': DEC.tolist(), 'Z_SPEC': Z_SPEC.tolist()})

        # Keep galaxies with positive redshift only
        df.query("Z_SPEC > 0", inplace=True)

        # Check for duplicate coordinates
        pcatalog = SkyCoord(ra=df['RA'].values, dec=df['DEC'].values, unit='deg')
        isdup = self.check_dup_coord(coords=pcatalog)
        df['isdup'] = isdup
        df.query("isdup == False", inplace=True)
        df.drop(columns=['isdup'], inplace=True)

        df.to_csv(self.temp_path + self.survey + '/spectral_surveys/'+'z_%s'%spectral_name+'.csv', index=False)



    def check_dup_coord(self, coords, tol=0.3*u.arcsecond):
        """ Checks for duplicates in a list or array
        of coordinates, within an angular distance
        tolerance 'tol'.

        Parameters
        ----------
        coords : SkyCoord array or list
            Coordinates to check for duplicates
        tol : Angle, optional
            Angular tolerance

        Returns
        -------
        isdup : boolean array, shape(len(coords),)
            Whether a given coordinate in `coords` is a duplicate
        """

        _, d2d, _ = match_coordinates_sky(coords, coords, nthneighbor=2)
        isdup = d2d < tol

        return isdup


    def make_catalog(self, p, paste_dir, input_path, spectral_name, vignet=True):
        """Creates .csv catalogs with different flux & magnitudes in bands R,I,Z,G,Y
        as well as morphological parameters which are matched with a spectral catalog.
        Creates .npy 4D ndarrays & .png images for vignets

        Args:
            p ([int]): Tile id (0 =< id < len(paste_dir))
            paste_dir ([list]): list containing tile names
            input_path ([str]): directory path for paste_dir
            spectral_name ([str]): [description]
            vignet (bool, optional): [description]. Defaults to True.
        """

        #------------------------------------------------------------------#
        # # # # # Extract data from catalogs # # # # #
        #------------------------------------------------------------------#

        df_Z = pd.read_csv(self.temp_path + self.survey + '/spectral_surveys/'+'z_%s'%spectral_name + '.csv')

        file_name = paste_dir[p]
        HDU_tile = fits.open(input_path + file_name)

        # BANDS = dict(zip(self._bands, np.arange(2,len(self._bands)+2)))
        # BANDS = {'R':2, 'I':3, 'Z':4, 'G':5, 'Y':6}

        band = 'R'

        RA = HDU_tile[band].data['X_WORLD'].tolist()
        DEC = HDU_tile[band].data['Y_WORLD'].tolist()

        SNR = HDU_tile[band].data['SNR_WIN'].tolist()
        FWHM = HDU_tile[band].data['FWHM_WORLD'].tolist()
        ELONGATION = HDU_tile[band].data['ELONGATION'].tolist()

        MAG = HDU_tile[band].data['MAG_AUTO'].tolist()
        MAG_ERR = HDU_tile[band].data['MAGERR_AUTO'].tolist()

        ID = np.arange(0, len(HDU_tile[band].data['X_WORLD'].tolist()), 1)

        df_to_cut = pd.DataFrame(data={'ID': ID, 'RA': RA, 'DEC': DEC, 'SNR': SNR, 'FWHM': FWHM, 'MAG': MAG, 'MAG_ERR': MAG_ERR, 'ELONGATION': ELONGATION})

        #------------------------------------------------------------------#
        # # # # # SNR & FWHM & MAG & MAG_ERR CUTS  # # # # #
        #------------------------------------------------------------------#

        if self.survey == 'unions':
            df_to_cut.query("SNR > 10 & FWHM*3600 > 0.8 & MAG < 30 & MAG_ERR < 0.5", inplace=True)
        if self.survey == 'ps3pi_cfis':
            df_to_cut.query("SNR > 10 & SNR < 500 & FWHM*3600 > 0.8 & MAG < 30 & MAG_ERR < 0.5", inplace=True)
        df_to_cut.drop(columns=['SNR', 'MAG_ERR', 'MAG'], inplace=True)

        #------------------------------------------------------------------#
        # # # # # Check for duplicate coordinates # # # # #
        #------------------------------------------------------------------#

        scatalog = SkyCoord(ra=df_to_cut['RA'].values, dec=df_to_cut['DEC'].values, unit='deg')
        isdup = self.check_dup_coord(coords=scatalog)

        # sub arrays without duplicates for both catalogs
        df_to_cut['isdup'] = isdup
        df_to_cut.query("isdup == False", inplace=True)
        df_to_cut.drop(columns=['isdup'], inplace=True)

        #------------------------------------------------------------------#
        # # # # # Spatial coordinates matching # # # # #
        #------------------------------------------------------------------#
        if self.survey == 'unions':

            scatalog_sub = SkyCoord(ra=df_to_cut['RA'].values, dec=df_to_cut['DEC'].values, unit='deg')
            pcatalog_sub = SkyCoord(ra=df_Z['RA'].values, dec=df_Z['DEC'].values, unit='deg')
            idx, d2d, _ = match_coordinates_sky(scatalog_sub, pcatalog_sub, nthneighbor=1)

            tol = 0.3*u.arcsecond #threshold to consider whether or not two galaxies are the same
            ismatched = d2d < tol

            # try:

            #------------------------------------------------------------------#
            # # # # # Create matched dataframe with redshift # # # # #
            #------------------------------------------------------------------#

            df_d2d = pd.DataFrame(data={'ismatched': ismatched, 'idx': idx, 'd2d': d2d, 'RA': df_to_cut['RA'].values,'DEC':df_to_cut['DEC'].values})
            df_d2d.query("ismatched == True", inplace=True)
            df_d2d.drop(columns=['ismatched'], inplace=True)

        
            try:
                idx_sub = np.array([(i,ide) for (i,ide) in enumerate(idx) if ismatched[i] == True])[:,1]
            except:
                print('0 matched objects wo filters for tile %s'%(file_name[:-5]+'_'+spectral_name))
                return

            z_spec_sub = []
            for i in idx_sub:
                z_spec_sub.append(df_Z['Z_SPEC'].values[i])
            z_spec_sub = np.array(z_spec_sub)

            df_to_cut['ismatched'] = ismatched
            df_to_cut.query("ismatched == True", inplace=True)
            df_to_cut.drop(columns=['ismatched'], inplace=True)
            df_to_cut['Z_SPEC'] = z_spec_sub

        #------------------------------------------------------------------#
        # # # # # Band extraction for dataframe # # # # #
        #------------------------------------------------------------------#

        PARAMS = ['FWHM_WORLD', 'SNR_WIN' ,'MAG_AUTO','MAGERR_AUTO', 'MAG_WIN', 'MAGERR_WIN', 'FLUX_AUTO', 'FLUXERR_AUTO', 'FLUX_WIN', 'FLUXERR_WIN', 'FLUX_APER', 'FLUXERR_APER']
        NAMES = np.array(['%s_%s'%(param, band) for param in PARAMS for band in self._bands])
        MAG_ZEROPOINT_NAMES = np.array(['MAG_AUTO_%s'%band for band in self._bands[1:]])
        MAG_NAMES = ['MAG_AUTO', 'MAG_WIN']
        MAG_COLS = np.array(['%s_%s'%(param, band) for param in MAG_NAMES for band in self._bands[1:]])



        # try:
        try:
            par = [HDU_tile[band].data[param].tolist() for param in PARAMS for band in self._bands]
        except:
            print('Tile %s does not have the right number of bands'%(file_name[:-5]+'_'+spectral_name))
            return
        df_mag = pd.DataFrame(data = dict(zip(NAMES, par)))
        NAME = [file_name[4:-5]+'_%s'%i for i in range(len(df_mag))]
        ide = np.arange(0, len(df_mag), 1)
        df_mag.insert(0, 'ide', ide)
        df_mag.insert(0, 'NAME', NAME)   



        if self.survey == 'ps3pi_cfis':
            df_mag['Z_SPEC'] = HDU_tile['R'].data['Z'].tolist()
            df_mag['id'] = HDU_tile[self._bands[0]].data['NUMBER'].tolist()
            df_mag['ID'] = np.arange(0,len(df_mag),1).tolist()
            ## merge
            df_mag = pd.merge(df_mag, df_to_cut, indicator=True, on='ID', how='outer').query('_merge=="both"').drop('_merge', axis=1)

            MORPHO_NAMES = ['id', 'gal_mag', 'gal_mag_err', 'gal_flux', 'gal_flux_err', 'gal_g1', 'gal_g1_err', 'gal_g2', 'gal_g2_err', 'gal_gini', 'gal_sb', 'gal_rho4', 'gal_sigma', 'gal_resolution', 'psf_sigma']
            morpho_par = [HDU_tile['R_PSF'].data[param].tolist() for param in MORPHO_NAMES]
            df_morph = pd.DataFrame(data = dict(zip(MORPHO_NAMES, morpho_par)))

            mv = LearningAlgorithms.missing_data(self, df_morph)
            print(mv.head(10))
            
            dfbands = pd.merge(df_mag, df_morph, indicator=True, on='id', how='outer').query('_merge=="both"').drop('_merge', axis=1)
            dfbands.drop(columns=['id'], inplace=True)
            
            dfbands_matched = dfbands.copy()
            dfbands_matched = dfbands_matched[dfbands_matched['Z_SPEC'].notna()]
            dfbands_unmatched = df_mag.copy()

        if self.survey == 'unions':
            dfbands = df_mag.copy()
            dfbands['ID'] = np.arange(0,len(dfbands),1).tolist()
            ## merge
            dfbands_matched = pd.merge(dfbands, df_to_cut, indicator=True, on='ID', how='left').query('_merge=="both"').drop('_merge', axis=1)
            df_unmatched = df_to_cut.copy()
            dfbands_unmatched = pd.merge(dfbands, df_unmatched, indicator=True, on='ID', how='left').query('_merge=="both"').drop('_merge', axis=1)




        if self.survey == 'ps3pi_cfis':
            print(np.corrcoef(dfbands_matched['FWHM'], dfbands_matched['gal_sigma']))


        # replace 99 by np.nan in MAG cols (TO DO: replace whatever equivalent there is for FLUX cols)
        for col in MAG_COLS:
            dfbands_matched.loc[(dfbands_matched[col]>98), col] = np.nan
            dfbands_unmatched.loc[(dfbands_unmatched[col]>98), col] = np.nan

        for band in self._bands:
            if self.survey == 'ps3pi_cfis':
                dfbands_matched.query("SNR_WIN_%s > 10 & SNR_WIN_%s < 500 & FWHM_WORLD_%s*3600 > 0.8 & MAGERR_AUTO_%s < 0.5 & gal_mag_err < 0.5" %(band,band,band,band) , inplace=True)
                dfbands_unmatched.query("SNR_WIN_%s > 10 & SNR_WIN_%s < 500 & FWHM_WORLD_%s*3600 > 0.8 & MAGERR_AUTO_%s < 0.5" %(band,band,band,band) , inplace=True)
            else:
                dfbands_matched.query("SNR_WIN_%s > 10 & FWHM_WORLD_%s*3600 > 0.8 & MAGERR_AUTO_%s < 0.5" %(band,band,band) , inplace=True)
                dfbands_unmatched.query("SNR_WIN_%s > 10 & FWHM_WORLD_%s*3600 > 0.8 & MAGERR_AUTO_%s < 0.5" %(band,band,band) , inplace=True)                
        for band in self._bands:
            dfbands_matched.drop(columns=['SNR_WIN_%s'%band, 'FWHM_WORLD_%s'%band], inplace=True)
            dfbands_unmatched.drop(columns=['SNR_WIN_%s'%band, 'FWHM_WORLD_%s'%band], inplace=True)

        # keep only mags with err<0.5
        # for name in MAGERR_NAMES:
            # dfbands_matched.query("%s < 0.5"%name, inplace=True)
            # dfbands_unmatched.query("%s < 0.5"%name, inplace=True)

        # shift -5 mag
        if self.survey == 'ps3pi_cfis':
            for name in MAG_ZEROPOINT_NAMES:
                dfbands_matched[name] = dfbands_matched[name] - 5
                dfbands_unmatched[name] = dfbands_unmatched[name] - 5

        # save png vignets & array
        if vignet == True:
            self.vignet_to_png(file_name, dfbands_matched, HDU_tile)
            self.vignet_to_array(file_name, dfbands_matched, HDU_tile)


        # print(np.corrcoef(dfbands_matched['MAG_AUTO_R'], dfbands_matched['Z_SPEC']))
        print('%i matched objects w filters for tile %s'%(len(dfbands_matched), file_name[:-5]+'_'+spectral_name))

        dfbands_matched.drop(columns=['ID'], inplace=True)
        dfbands_matched.to_csv(self.temp_path + self.survey + '/matched/'+'%s'%file_name[:-5]+'_'+spectral_name+'.csv', index=False)
        dfbands_unmatched.drop(columns=['ID'], inplace=True)
        dfbands_unmatched.to_csv(self.temp_path + self.survey + '/unmatched/'+'%s'%file_name[:-5]+'_'+spectral_name+'.csv', index=False)

        df_Z.to_csv(self.temp_path + self.survey + '/redshift/'+'%s'%file_name[:-5]+'_'+spectral_name+'.csv', index=False)

        dfbands_matched.query("Z_SPEC > 0", inplace=True)

        # df_d2d.to_csv(self.temp_path + self.survey + '/d2d/'+'%s'%file_name[:-5]+'_'+spectral_name+'.csv', index=False)

        #     except:
        #         print('Tile %s does not have the right number of bands'%(file_name[:-5]+'_'+spectral_name))

        # except:
        #     print('0 matched objects for tile %s'%(file_name[:-5]+'_'+spectral_name))



            # print('Tile %s does not have the right number of bands'%(file_name[:-5]+'_'+spectral_name))
        # except:
        #     print('Tile %s did not match any spectral galaxy'%(file_name[:-5]+'_'+spectral_name))



    def vignet_to_png(self, file_name, dfbands_matched, HDU_tile):
        """[summary]

        Args:
            file_name ([type]): [description]
            dfbands_matched ([type]): [description]
            HDU_tile ([type]): [description]
        """

        # BANDS = dict(zip(self._bands, np.arange(2,len(self._bands)+2)))

        for i,id in enumerate(dfbands_matched['ID']):
            for band in self._bands:
                plt.ioff()
                plt.figure(figsize=(3,3), tight_layout=False)
                plt.imshow(HDU_tile[band].data['VIGNET'][int(id)], norm=LogNorm())
                plt.colorbar()
                plt.title('MAG_{} =  {:.1f}'.format(band, dfbands_matched['MAG_AUTO_%s'%band].values[i]))
                plt.savefig(self.temp_path + self.survey + "/vignet/%s/vignet_%s_%s_%s-.png"%(band, band, file_name[:-5], dfbands_matched['ide'].values[i]), bbox_inches='tight', transparent=True)
                plt.close()
        plt.show()


    def vignet_to_array(self, file_name, dfbands_matched, HDU_tile):
        """[summary]

        Args:
            file_name ([type]): [description]
            dfbands_matched ([type]): [description]
            HDU_tile ([type]): [description]
        """

        X = np.ndarray((len(dfbands_matched),51,51,len(self._bands)))

        for i,id in enumerate(dfbands_matched['ID']):
            for j,band in enumerate(self._bands):
                HDU = HDU_tile[band].data['VIGNET'][int(id)]
                X[i,:,:,j] = np.asarray(np.where(HDU > -1, HDU, 0))

        np.save(self.temp_path + self.survey + '/vignet/array/vignet_%s'% file_name[:-5], X)


    def merge_catalogs(self, vignet=True):
        """[summary]

        Args:
            vignet (bool, optional): [description]. Defaults to True.
        """

        #------------------------------------------------------------------#
        # # # # # CSV files # # # # #
        #------------------------------------------------------------------#

        # Import temp files
        path_matched = self.temp_path + self.survey + '/matched/'
        all_files = glob.glob(os.path.join(path_matched, "*.csv"))
        df_matched_short = pd.concat((pd.read_csv(f) for f in all_files), axis=0)
        # Remove duplicates
        scatalog = SkyCoord(ra=df_matched_short['RA'].values, dec=df_matched_short['DEC'].values, unit='deg')
        isdup = self.check_dup_coord(coords=scatalog)
        df_matched_short['isdup'] = isdup
        df_matched_short.query("isdup == False", inplace=True)
        df_matched_short.drop(columns=['isdup'], inplace=True)
        # Save file
        df_matched_short.to_csv(self.output_path + 'output/' + self.survey + '/' + self.output_name + '/files/' + self.output_name + '.csv', index=False)

        if vignet == True:
            # Import temp files
            path_matched_vignet = self.temp_path + self.survey + '/vignet/array/'
            all_files = glob.glob(os.path.join(path_matched_vignet, "*.npy"))
            X = []
            for f in all_files:
                X.append(np.load(f))
            vignet = np.concatenate(X, axis=0)
            # Remove duplicates
            vignet_nodup = np.ndarray((len(df_matched_short),51,51,5))
            k = 0
            for i,dup in enumerate(isdup):
                if dup == False:
                    vignet_nodup[k,:,:,:] = vignet[i,:,:,:]
                    k += 1
            # Save file
            np.save(self.output_path + 'output/' + self.survey + '/' + self.output_name + '/files/' + self.output_name, vignet_nodup)


        # # Import temp files
        # path_matched_d2d = self.temp_path + self.survey + '/d2d/'
        # all_files = glob.glob(os.path.join(path_matched_d2d, "*.csv"))
        # df_matched_d2d = pd.concat((pd.read_csv(f) for f in all_files), axis=0)
        # # Remove duplicates
        # scatalog = SkyCoord(ra=df_matched_d2d['RA'].values, dec=df_matched_d2d['DEC'].values, unit='deg')
        # isdup = self.check_dup_coord(coords=scatalog)
        # df_matched_d2d['isdup'] = isdup
        # df_matched_d2d.query("isdup == False", inplace=True)
        # df_matched_d2d.drop(columns=['isdup', 'RA', 'DEC'], inplace=True)
        # # Save file
        # df_matched_d2d.to_csv(self.output_path + 'output/' + self.survey + '/' + self.output_name + '/files/' + self.output_name + '_d2d' + '.csv', index=False)

        # Import temp files
        path_unmatched_short = self.temp_path + self.survey + '/unmatched/'
        all_files = glob.glob(os.path.join(path_unmatched_short, "*.csv"))
        df_unmatched_short = pd.concat((pd.read_csv(f) for f in all_files), axis=0)
        # Remove duplicates
        scatalog = SkyCoord(ra=df_unmatched_short['RA'].values, dec=df_unmatched_short['DEC'].values, unit='deg')
        isdup = self.check_dup_coord(coords=scatalog)
        df_unmatched_short['isdup'] = isdup
        df_unmatched_short.query("isdup == False", inplace=True)
        df_unmatched_short.drop(columns=['isdup'], inplace=True)
        # Save file
        df_unmatched_short.to_csv(self.output_path + 'output/' + self.survey + '/' + self.output_name + '/files/' + self.output_name + '_unmatched' + '.csv', index=False)

    
    def compute_weights(self, df_matched, column='r'):
        df_unmatched = pd.read_csv(self.output_path + 'output/' + self.survey + '/' + self.output_name + '/files/' + self.output_name + '_unmatched' + '.csv')
        df_unmatched.rename(columns={"MAG_AUTO_R": "r"}, inplace=True)
        n = len(df_matched)//10

        udensity, ubins_edge = np.histogram(df_unmatched[column].values, bins=n, density=True)
        ubin_min = ubins_edge[0]
        ubin_max = ubins_edge[-1]
        ustep = (ubin_max - ubin_min)/n

        density, bins_edge = np.histogram(df_matched[column].values, bins=n, density=True)
        bin_min = bins_edge[0]
        bin_max = bins_edge[-1]
        step = (bin_max - bin_min)/n

        ID = np.arange(0, len(df_matched))
        df_matched.insert(loc=0, value=ID, column='ID')
        df_matched.sort_values(by=column, inplace=True)

        weights, uweights = [], []
        n_min = 0
        n_umin = 0
        for i in range(len(df_matched)):
            for k in range(n_min, n):
                if bin_min+step*(k+1) < df_matched[column].values[i]:
                    continue
                else:
                    weights.append(density[k])
                    n_min = k
                    break 
                

            for k in range(n_umin, n):
                if ubin_min+ustep*(k+1) < df_matched[column].values[i]:
                    continue
                else:
                    uweights.append(udensity[k])
                    n_umin = k
                    break 
                

        w = np.array(uweights) / np.array(weights)

        df_matched['weights'] = w
        df_matched.sort_values(by='ID', inplace=True)
        w = df_matched['weights'].to_numpy()
        df_matched.drop(columns=['weights', 'ID'], inplace=True)

        np.save(self.output_path + 'output/' + self.survey + '/' + self.output_name + '/files/' + 'Weights_' + self.output_name, w)

        return w




class LearningAlgorithms(object):

    def __init__(self, survey, bands, output_name, output_path=None, temp_path=None, path_to_csv=None, dataframe=pd.DataFrame(data={}), sample_weight=None, validation_set=False, cv=4, preprocessing='drop', n_jobs=-1):
        """[summary]

        Args:
            survey ([type]): [description]
            bands ([type]): [description]
            temp_path ([type]): [description]
        """

        self._path = os.getcwd() + '/'
        self.survey = survey
        self._bands = bands
        self.output_name = output_name
        self.temp_path = temp_path
        self.preprocessing = preprocessing
        if path_to_csv is None:
            if dataframe.empty == True:
                raise TypeError('dataframe must be specified if path_to_csv=None')
            else:
                self.df = dataframe
        elif path_to_csv is not None:
            # try:
            self.df = pd.read_csv(path_to_csv)
            # except:
                # raise TypeError('neither path_to_csv nor dataframe were specified')
        if output_path is None:
            self.output_path = self._path + 'output/' + self.survey + '/' + self.output_name + '/'       
        else:
            self.output_path = output_path + 'output/' + self.survey + '/' + self.output_name + '/'
        self.n_jobs = n_jobs
        self.sample_weight = sample_weight
        self.cv = cv
        if sample_weight is None:
            self.train, self.test = train_test_split(self.df, test_size = 0.2, random_state=0)
            self.sample_weight_train = None
        else:
            self.train, self.test, self.sample_weight_train, self.sample_weight_test = train_test_split(self.df, self.sample_weight, test_size = 0.2, random_state=0)

        self.X_train = self.train.iloc[:,:-1]
        self.y_train = self.train.iloc[:,-1]
        self.X_test = self.test.iloc[:,:-1]
        self.y_test = self.test.iloc[:,-1]
        self.val = validation_set
        if validation_set == True:
            if sample_weight is None:
                self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=0)
                self.sample_weight_train = None
            else:
                self.X_train, self.X_val, self.y_train, self.y_val, self.sample_weight_train, self.sample_weight_test = train_test_split(self.X_train, self.y_train, self.sample_weight, test_size=0.2, random_state=0)


    def data(self):
        if self.val == True:
            return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
        else:
            return self.X_train, self.X_test, self.y_train, self.y_test

    def weight_train(self):
        return self.sample_weight_train

    
    def dataframe(self):
        return self.df

    def unmatched_dataframe(self):
        return pd.read_csv(self.output_path + 'output/' + self.survey + '/files/' + self.output_name + '_unmatched' + '.csv')


    def merge_cfis_r_cfht_u_medium_deep_i_g_z(self):

        CFHT = pd.read_csv(self._path+'catalogs/CFHTLens_2021-01-25T12_32_19.tsv', delimiter='\t')
        CFHT = CFHT.replace(-99, 0)
        CFHT = CFHT.replace(99, 0)
        # CFHT['MAG_i'] = CFHT['MAG_i'] + CFHT['MAG_y']
        # CFHT['MAGERR_i'] = CFHT['MAGERR_i'] + CFHT['MAGERR_y']
        # CFHT['EXTINCTION_i'] = CFHT['EXTINCTION_i'] + CFHT['EXTINCTION_y']
        CFHT = CFHT.replace(0, np.nan)
        CFHT.query("MAGERR_u < 0.5", inplace=True)
        CFHT.rename(columns={"ALPHA_J2000": "RA", "DELTA_J2000": "DEC"}, inplace=True)
        CFHT = CFHT[['MAG_u', 'MAG_r', 'RA', 'DEC']].copy()

        mediumdeep = pd.read_csv(self._path+'catalogs/cat_psmd_deep23.rsv', delimiter=',')

        mediumdeep = mediumdeep[['ra', 'dec', 'i_stk_aper', 'z_stk_aper', 'g_stk_aper', 'g_stk_aper_err', 'i_stk_aper_err', 'z_stk_aper_err']].copy()

        for band in ['g', 'i', 'z']:
            for s in ['stk']:
                for a in ['aper']:
                    mediumdeep.query("%s_%s_%s_err < 0.5"%(band,s,a), inplace=True)

        ID = np.arange(0, len(mediumdeep))
        mediumdeep.drop(columns=['g_stk_aper_err', 'i_stk_aper_err', 'z_stk_aper_err'], inplace=True)
        mediumdeep.insert(loc=0, value=ID, column='ID')

        scatalog_sub = SkyCoord(ra=CFHT['RA'].values, dec=CFHT['DEC'].values, unit='deg')
        pcatalog_sub = SkyCoord(ra=mediumdeep['ra'].values, dec=mediumdeep['dec'].values, unit='deg')
        idx, d2d, _ = match_coordinates_sky(scatalog_sub, pcatalog_sub, nthneighbor=1)

        tol = 0.3*u.arcsecond #threshold to consider whether or not two galaxies are the same
        ismatched = d2d < tol

        CFHT['ismatched'], CFHT['ID'] = ismatched, idx

        mediumdeep.drop(columns=['ra', 'dec'], inplace=True)

        df = pd.merge(CFHT, mediumdeep, indicator=True, on='ID', how='outer').query('_merge=="both"').drop('_merge', axis=1)

        df.query("ismatched == True", inplace=True)
        df.drop(columns=['ismatched', 'ID'], inplace=True)

        ID = np.arange(0, len(df))
        df.insert(loc=0, value=ID, column='ID')

        # Save matched

        CFIS_R = pd.read_csv(self.output_path + 'files/' + self.output_name + '.csv')
        CFIS_R = CFIS_R[['MAG_AUTO_R' ,'RA', 'DEC', 'FWHM', 'ELONGATION', 'gal_mag','gal_g1', 'gal_g2', 'gal_gini', 'gal_sb', 'gal_rho4', 'gal_sigma', 'gal_resolution', 'psf_sigma', 'Z_SPEC']].copy()
        CFIS_R = self.gal_g(CFIS_R) #create gal_g = |gal_g1 + gal_g2| and drop gal_g1, gal_g2
        ID = np.arange(0, len(CFIS_R))
        CFIS_R.insert(loc=0, value=ID, column='ID')

        scatalog_sub = SkyCoord(ra=df['RA'].values, dec=df['DEC'].values, unit='deg')
        pcatalog_sub = SkyCoord(ra=CFIS_R['RA'].values, dec=CFIS_R['DEC'].values, unit='deg')
        idx, d2d, _ = match_coordinates_sky(scatalog_sub, pcatalog_sub, nthneighbor=1)

        tol = 0.3*u.arcsecond #threshold to consider whether or not two galaxies are the same
        ismatched = d2d < tol

        df['ismatched'], df['ID'] = ismatched, idx

        CFIS_R.drop(columns=['RA', 'DEC'], inplace=True)

        df_medium = pd.merge(df, CFIS_R, indicator=True, on='ID', how='outer').query('_merge=="both"').drop('_merge', axis=1)



        df_medium.query("ismatched == True", inplace=True)
        df_medium.drop(columns=['ismatched', 'ID'], inplace=True)

        r_df = pd.DataFrame(data={'MAG_r': df_medium['MAG_r'].to_numpy(), 'MAG_AUTO_R': df_medium['MAG_AUTO_R'].to_numpy()})
        r_df.to_csv(self.output_path + 'files/' + 'R_CFHT_vs_CFIS.csv', index=False)
        df_medium.drop(columns=['MAG_r'], inplace=True)


        df_medium = df_medium[['MAG_AUTO_R', 'MAG_u', 'i_stk_aper', 'z_stk_aper', 'g_stk_aper', 'gal_sb', 'gal_rho4', 'gal_sigma', 'gal_gini', 'ELONGATION', 'Z_SPEC']].copy()
        df_medium.rename(columns={"MAG_AUTO_R": "r", "MAG_u": "u", "i_stk_aper": "i", "z_stk_aper": "z", "g_stk_aper": "g", "gal_sb": "Surf. bright.", "gal_rho4": "Rho4", "gal_sigma": "Galaxy size", "gal_gini": "Gini index", "ELONGATION": "Elongation", "Z_SPEC": "Z spec"}, inplace=True)
        
        # df_medium = df_medium[['gal_rho4', 'gal_sigma', 'gal_gini', 'ELONGATION', 'Z_SPEC']].copy()
        # df_medium.rename(columns={"gal_rho4": "Rho4", "gal_sigma": "Galaxy size", "gal_gini": "Gini index", "ELONGATION": "Elongation", "Z_SPEC": "Z spec"}, inplace=True)
        
        # df_medium = df_medium[['MAG_AUTO_R', 'MAG_u', 'i_stk_aper', 'z_stk_aper', 'g_stk_aper', 'Z_SPEC']].copy()
        # df_medium.rename(columns={"MAG_AUTO_R": "r", "MAG_u": "u", "i_stk_aper": "i", "z_stk_aper": "z", "g_stk_aper": "g", "Z_SPEC": "Z spec"}, inplace=True)
        
        df_medium.to_csv(self.output_path + 'files/' + 'MediumDeep_IZG_CFHT_U_CFIS_R_catalog.csv', index=False)

        # Save unmatched 


        CFIS_R_unmatched = pd.read_csv(self.output_path + 'files/' + self.output_name + '_unmatched' + '.csv')
        CFIS_R_unmatched = CFIS_R_unmatched[['MAG_AUTO_R', 'RA', 'DEC']].copy()
        ID = np.arange(0, len(CFIS_R_unmatched))
        CFIS_R_unmatched.insert(loc=0, value=ID, column='ID')

        scatalog_sub = SkyCoord(ra=df['RA'].values, dec=df['DEC'].values, unit='deg')
        pcatalog_sub = SkyCoord(ra=CFIS_R_unmatched['RA'].values, dec=CFIS_R_unmatched['DEC'].values, unit='deg')
        idx, d2d, _ = match_coordinates_sky(scatalog_sub, pcatalog_sub, nthneighbor=1)

        tol = 0.3*u.arcsecond #threshold to consider whether or not two galaxies are the same
        ismatched = d2d < tol

        df['ismatched'], df['ID'] = ismatched, idx

        CFIS_R_unmatched.drop(columns=['RA', 'DEC'], inplace=True)

        df_medium_unmatched = pd.merge(df, CFIS_R_unmatched, indicator=True, on='ID', how='outer').query('_merge=="both"').drop('_merge', axis=1)

        df_medium_unmatched.query("ismatched == True", inplace=True)
        df_medium_unmatched.drop(columns=['ismatched', 'ID'], inplace=True)

        df_medium_unmatched = df_medium_unmatched[['MAG_AUTO_R', 'MAG_u', 'i_stk_aper', 'z_stk_aper', 'g_stk_aper']].copy()
        df_medium_unmatched.rename(columns={"MAG_AUTO_R": "r", "MAG_u": "u", "i_stk_aper": "i", "z_stk_aper": "z", "g_stk_aper": "g"}, inplace=True)
        df_medium_unmatched.to_csv(self.output_path  + 'files/' + 'MediumDeep_IZG_CFHT_U_CFIS_R_catalog_unmatched.csv', index=False)

        return df_medium, df_medium_unmatched


    def gal_g(self, df):
        df.insert(7,'gal_g', np.abs(df['gal_g1'] + df['gal_g2']))
        df.drop(columns=['gal_g1', 'gal_g2'], inplace=True)
        return df


    def sigma_eta(self, y_test, y_pred):
        delta_z = (y_test - y_pred)/(np.ones_like(y_test) + y_test)
        sigma = 1.4826*np.median( np.abs( delta_z - np.median(delta_z) ) )
        outlier = np.abs(delta_z) > 0.15
        outlier_rate = len([x for x in outlier.ravel() if x==True])/len(delta_z.ravel())
        return sigma, outlier_rate


    def missing_data(self, dataset):
        all_data_na = (dataset.isnull().sum() / len(dataset)) * 100
        all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
        missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
        return missing_data


    def plot_corrmat(self, df=pd.DataFrame(data={}), figure_name=None):
        if df.empty == True:
            corrmat = self.df.corr()
            plt.subplots(figsize=(12,9))
            sns.heatmap(corrmat, vmax=1,cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 12})
            if figure_name is None:
                plt.savefig(self.output_path + 'figures/'  + self.output_name + '_corrmat' + '.pdf', bbox_inches='tight', transparent=True)
            else:
                plt.savefig(self.output_path + 'figures/'  + figure_name + '.pdf', bbox_inches='tight', transparent=True)
            plt.show()
            plt.close()
        elif df.empty == False:
            corrmat = df.corr()
            plt.subplots(figsize=(12,9))
            sns.heatmap(corrmat, vmax=1,cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10})
            if figure_name is None:
                plt.savefig(self.output_path + 'figures/'  + self.output_name + '_corrmat' + '.pdf', bbox_inches='tight', transparent=True)
            else:
                plt.savefig(self.output_path + 'figures/'  + figure_name + '.pdf', bbox_inches='tight', transparent=True)
            plt.show()
            plt.close()           


    def plot_zphot_zspec(self, y_pred, y_test, method, lim):
        fig = plt.figure(figsize=(7,6), tight_layout=False)
        ax = fig.add_subplot(111)
        ax.set_facecolor('white')
        ax.grid(True, color='grey', lw=0.5)
        ax.set_xlabel(r'$\mathrm{z}_{\mathrm{spec}}$', fontsize=20)
        ax.set_ylabel(r'$\mathrm{z}_{\mathrm{phot}}$', fontsize=20)

        # ax.set_title('%s'%method, fontsize=15)

        im = ax.hist2d([float(y) for y in y_test], [float(y) for y in y_pred], bins=(50, 50), cmap='gist_yarg')
        ax.set_xlim([np.amin(y_test), lim])
        ax.set_ylim([np.amin(y_pred), np.amax(y_pred)])
        cbar = fig.colorbar(im[3])

        sigma, eta = self.sigma_eta(y_test, y_pred)

        ax.text(0.1,np.amax(y_pred)-0.2,r'$\sigma = {:.3f}$'.format(sigma)+'\n'+r'$\eta = {:.2f}$ %'.format(eta*100), size=15,va="baseline", ha="left", multialignment="left")

        x = np.linspace(0, 3.5, 1000)
        ax.plot(x,x, linewidth=1, color='k', linestyle='--')
        plt.savefig(self.output_path + 'figures/' + self.output_name + '_' + method + '_zphot_zspec' + '.pdf', bbox_inches='tight', transparent=True)
        plt.show()
        plt.close()

    def sigma_eta_score(self, y_test, y_pred):
        y_pred = y_pred.flatten()
        delta_z = (y_test - y_pred)/(np.ones_like(y_test) + y_test)
        sigma = 1.4826*np.median( np.abs( delta_z - np.median(delta_z) ) )
        outlier = np.abs(delta_z) > 0.15
        outlier_rate = len([x for x in outlier.ravel() if x==True])/len(delta_z.ravel())
        return np.sqrt(np.mean(delta_z)**2 + sigma**2)



    def permutation_importance(self, model, params={}, scaler=False, linear=False, ann=False, dataframe=pd.DataFrame(data={})):
        
        if dataframe.empty == True:
            dataframe = self.df
        columns = dataframe.columns
        score, std = self.cross_validation(model=model, params=params, metric=self.sigma_eta_score, scaler=scaler, linear=linear, ann=ann, dataframe=dataframe)  
        df = pd.DataFrame(columns=['Weight', 'Feature', 'score'])  
        print('[Feature importance]')
        print('---------------------------------------------------------------')
        for column in columns[:-1]:
            df_shuffle = dataframe.copy()
            df_shuffle[column] = df_shuffle[column].sample(frac=1).values
            mscore, mstd = self.cross_validation(model=model, params=params, metric=self.sigma_eta_score, scaler=scaler, linear=linear, ann=ann, dataframe=df_shuffle)
            df = pd.concat((df, pd.DataFrame(data={'Weight':['${:.3f}\pm{:.3f}$'.format(mscore-score, np.abs(std+mstd))], 'Feature': ['$' + column + '$'], 'score':score-mscore})))
        df.sort_values(by=['score'], inplace=True)
        score_weight = df['score'].to_numpy()
        feature = df['Feature'].to_numpy()
        df.drop(columns=['score'], inplace=True)
        df.to_csv(self.output_path + 'files/' + 'Feature_Weights_' + self.output_name + '_' + self.method + '.csv', index=False)

        print(df.head(10))
        
        return score_weight, feature
        
 
    def permutation(self, lst): 
        """Python function to make permutations of a given list

        Args:
            lst (list): [description]

        Returns:
            list: [description]
        """
        # If lst is empty then there are no permutations 
        if len(lst) == 0: 
            return [] 
    
        # If there is only one element in lst then, only 
        # one permuatation is possible 
        if len(lst) == 1: 
            return [lst] 
    
        # Find the permutations for lst if there are 
        # more than 1 characters 
    
        l = [] # empty list that will store current permutation 
    
        # Iterate the input(lst) and calculate the permutation 
        for i in range(len(lst)): 
            m = lst[i] 
    
            # Extract lst[i] or m from the list.  remLst is 
            # remaining list 
            remLst = lst[:i] + lst[i+1:] 
        
            # Generating all permutations where m is first 
            # element 
            for p in self.permutation(remLst): 
                l.append([m] + p) 
        return l 

        
    def feature_engineering(self, df, bands=None, color_order=None):
        columns = list(df.columns)[1:len(bands)]
        index_dict = dict(zip(bands, np.arange(len(bands))))
        if bands is not None and color_order is None:
            df_list = [df]
            for p in self.permutation(bands): 
                df_copy = df.copy()
                for i in range(len(bands)-1):
                    df_copy.insert(loc = i+1, column = p[i]+'-'+p[i+1], value = df_copy.iloc[:,index_dict[p[i]]] - df_copy.iloc[:,index_dict[p[i+1]]]) 
                df_copy.drop(columns=columns, inplace=True)
                df_list.append(df_copy)
            return df_list
        elif bands is not None and color_order is not None:
            df_list = []
            df_copy = df.copy()
            p = color_order
            for i in range(len(bands)-1):
                df_copy.insert(loc = i+1, column = p[i]+'-'+p[i+1], value = df_copy.iloc[:,index_dict[p[i]]] - df_copy.iloc[:,index_dict[p[i+1]]]) 
            df_copy.drop(columns=columns, inplace=True)
            df_list.append(df_copy)
            return df_list


    def feature_remover(self, model, dataframe, params={}, metric=False, scaler=False, linear=False, ann=False):
        dscore, dstd = self.cross_validation(model, params=params, metric=self.sigma_eta_score, scaler=scaler, linear=linear, ann=ann, dataframe=dataframe)
        print(list(dataframe.columns))
        print('score = {:.3f} ± {:.3f}'.format(dscore, dstd))
        score_weight, features = self.permutation_importance(model, params=params, scaler=scaler, linear=linear, ann=ann, dataframe=dataframe)
        df_copy = dataframe.copy()
        k = 0
        for i in range(len(score_weight)):

            n = len(score_weight)-1 -k
            feature = df_copy[features[n]].to_numpy()
            df_copy.drop(columns=[features[n]], inplace=True)
            score,std = self.cross_validation(model, params=params, metric=self.sigma_eta_score, scaler=scaler, linear=linear, ann=ann, dataframe=df_copy)
            print(list(df_copy.columns))
            print(list(dataframe.columns))
            print('score = {:.3f} ± {:.3f}'.format(score, std))
            if score < dscore:
                dataframe = df_copy
                dscore = score
                print('Feature [%s] does not improve the score and will be removed'%features[n])
            else:
                print('Feature [%s] improves the score and will be kept'%features[n])
                break
            score_weight, features = self.permutation_importance(model, params=params, scaler=scaler, linear=linear, ann=ann, dataframe=dataframe)
        try:
            dataframe.insert(loc=1, column=features[n], value=feature)
        except:
            pass
        
        return dataframe



    def cross_validation(self, model, params={}, metric=False, scaler=False, linear=False, ann=False, dataframe=pd.DataFrame(data={})):
        """[summary]

        Args:
            metric ([type]): [description]
            model ([type]): [description]
            params ([type]): [description]
            cv (int, optional): [description]. Defaults to 4.

        Returns:
            [type]: [description]
        """
        
        if dataframe.empty == True:
            df = self.df.sample(frac=1).reset_index(drop=True)
        else:
            df = dataframe.sample(frac=1).reset_index(drop=True)

        X = df.iloc[:,:-1]
        y = df.iloc[:,-1]

        df_X_list = []
        df_y_list = []
        weight_list = []
        n = len(X)
        for i in range(0,self.cv):
            if i < self.cv-1:
                df_X_list.append(X.iloc[i*int(n/self.cv):(i+1)*int(n/self.cv), :])
                df_y_list.append(y.iloc[i*int(n/self.cv):(i+1)*int(n/self.cv)])
                if self.sample_weight is not None:
                    weight_list.append(self.sample_weight[i*int(n/self.cv):(i+1)*int(n/self.cv)])
            if i == self.cv-1:
                df_X_list.append(X.iloc[i*int(n/self.cv):, :])
                df_y_list.append(y.iloc[i*int(n/self.cv):])
                if self.sample_weight is not None:
                    weight_list.append(self.sample_weight[i*int(n/self.cv):])

        sigma_list, eta_list, score_list = [], [], []
        for i in tqdm(range(0,self.cv)):

            X_test = df_X_list[i]
            y_test = df_y_list[i]

            X_train = pd.concat(np.array([(j,X) for (j,X) in enumerate(df_X_list) if i != j])[:,1])
            y_train = pd.concat(np.array([(j,y) for (j,y) in enumerate(df_y_list) if i != j])[:,1])
            if self.sample_weight is not None:
                weight_train = np.concatenate(np.array([(j,w) for (j,w) in enumerate(weight_list) if i != j])[:,1])
            elif self.sample_weight is None:
                weight_train = None

            test = pd.concat((X_test, y_test), axis=1)
            train = pd.concat((X_train, y_train), axis=1)
            if ann == True:
                regressor = model
                train, test, weight_train, best_preprocess = self.preprocess(regressor, train, test, weight_train, method=self.preprocessing, scaler=scaler, linear=linear, ann=ann)
            else:
                if linear == True:
                    regressor = make_pipeline(RobustScaler(), model(**params))
                    train, test, weight_train, best_preprocess = self.preprocess(regressor, train, test, weight_train, method=self.preprocessing, scaler=scaler, linear=linear)
                elif linear == False:
                    regressor = model(**params)
                    train, test, weight_train, best_preprocess = self.preprocess(regressor, train, test, weight_train, method=self.preprocessing, scaler=scaler, linear=linear)

            X_train, X_test = train.iloc[:,:-1], test.iloc[:,:-1]
            y_train, y_test = train.iloc[:,-1], test.iloc[:,-1]

            if scaler == True:
                sc_X = StandardScaler()
                X_train = sc_X.fit_transform(X_train)
                X_test = sc_X.transform(X_test)
            if ann == True:
                regressor = model
                regressor.fit(X_train, y_train, batch_size = 32, epochs = 100, verbose=0, sample_weight=weight_train)
            else:
                if linear == True:
                    regressor = make_pipeline(RobustScaler(), model(**params))
                    kwargs = {regressor.steps[-1][0] + '__sample_weight': weight_train}
                    regressor.fit(X_train, y_train, **kwargs)
                elif linear == False:
                    regressor = model(**params)
                    regressor.fit(X_train, y_train, sample_weight=weight_train)

            y_pred = regressor.predict(X_test)
            if metric == False:
                sigma, eta = self.sigma_eta(y_test, y_pred.flatten())
                sigma_list.append(sigma)
                eta_list.append(eta)
                score_list.append(self.sigma_eta_score(y_test, y_pred))
            else:
                score_list.append(metric(y_test, y_pred.flatten())) 
        if metric == False:
            return np.mean(sigma_list), np.std(sigma_list)/np.sqrt(self.cv), np.mean(eta_list), np.std(eta_list)/np.sqrt(self.cv), best_preprocess, np.mean(score_list), np.std(score_list)/np.sqrt(self.cv)
        else: 
            return np.mean(score_list), np.std(score_list)/np.sqrt(self.cv)



    def preprocess(self, regressor, train, test, weight_train=None, weight_test=None, method='drop', scaler=False, linear=False, ann=False):
        if method is None:
            return train, test, weight_train
        else:
            if type(method) == str and method == 'BEST':
                train, test, weight_train, best_method = self.best_missing(regressor, train, test, weight_train, scaler=scaler, linear=linear, ann=ann)
                train['weight'] = weight_train
                train_test = [train, test]
            else:
                
                if weight_train is not None:
                    train['weight'] = weight_train

                train_test = []
                for df in [train, test]:
                    df = df.replace([-1, -10, -99], np.nan)
                    missing_data = self.missing_data(df)
                    for i in range(len(missing_data)):
                        if missing_data['Missing Ratio'][i] > 15:
                            df.drop(columns=[missing_data.index[i]], inplace=True)
                    if type(method) == str:
                        if method == 'drop':
                            df = df.dropna(how='any')
                        elif method == 'mode':
                            missing_data = self.missing_data(df)
                            for i in missing_data.index:
                                df[i] = df[i].fillna(df[i].mode()[0])
                        elif method == 'mean':
                            missing_data = self.missing_data(df)
                            for i in missing_data.index:
                                df[i] = df[i].fillna(df[i].mean())
                        elif method == 'median':
                            missing_data = self.missing_data(df)
                            for i in missing_data.index:
                                df[i] = df[i].fillna(df[i].median())
                        else:
                            raise SyntaxError("preprocess method is not defined. Possible values include 'drop', 'mode', 'mean', 'median' or int/float")
                    elif type(method) == int or type(method) == float:
                        missing_data = self.missing_data(df)
                        for i in missing_data.index:
                            df[i] = df[i].fillna(method)
                    else:
                        raise SyntaxError("preprocess method is not defined. Possible values include 'drop', 'mode', 'mean', 'median' or int/float")
                    train_test.append(df)

            train, test = train_test[0], train_test[1]
            if weight_train is not None:
                weight_train = train['weight'].to_numpy()
                train.drop(columns=['weight'], inplace=True)

            df = pd.concat((train, test), axis=0)
            missing_data = self.missing_data(df)
            if missing_data.empty == True:
                df.to_csv(self.output_path + 'files/' + self.output_name + '_preprocessed.csv', index=False)
                if type(method) == str and method == 'BEST':
                    return train, test, weight_train, best_method
                else:
                    return train, test, weight_train, method
            else:
                print(missing_data.head(10))
                raise ValueError('Dataframe contains NaNs, preprocess function could not handle them.')


    def best_missing(self, regressor, train, test, weight_train, scaler=False, linear=False, ann=False):        
        score_list = []
        df_list = []
        methods = ['drop', 'mean', 'mode', 'median']
        sample_weight_train = weight_train.copy()
        for method in methods:
            train_test = []
            if method == 'drop':
                # print('->'+method)
                for i,df in enumerate([train, test]):
                    if i == 0:
                        df['weight'] = weight_train
                        df = df.dropna(how='any')
                        weight_train = df['weight']
                    else:
                        df = df.dropna(how='any')
                    train_test.append(df)
                # print(train_test[0].head(1))
                train_test[0].drop(columns=['weight'], inplace=True)
                _, y_pred, y_test = self.pmodel(regressor, train=train_test[0], test=train_test[1], weight_train=weight_train, scaler=scaler, linear=linear, ann=ann, preprocess_method='BEST')
                score_list.append(self.sigma_eta_score(y_test, y_pred))
            
            elif method == 'mean':
                # print('->'+method)
                for df in [train, test]:
                    missing_data = self.missing_data(df)
                    for i in missing_data.index:
                        df[i] = df[i].fillna(df[i].mean())
                    train_test.append(df)
                # print(train_test[0].head(1))
                train_test[0].drop(columns=['weight'], inplace=True)
                _, y_pred, y_test = self.pmodel(regressor, train=train_test[0], test=train_test[1], weight_train=sample_weight_train, scaler=scaler, linear=linear, ann=ann, preprocess_method='BEST')
                score_list.append(self.sigma_eta_score(y_test, y_pred))

            elif method == 'mode':
                # print('->'+method)
                for df in [train, test]:
                    missing_data = self.missing_data(df)
                    for i in missing_data.index:
                        df[i] = df[i].fillna(df[i].mode()[0])
                    train_test.append(df)
                # print(train_test[0].head(1))
                _, y_pred, y_test = self.pmodel(regressor, train=train_test[0], test=train_test[1], weight_train=sample_weight_train, scaler=scaler, linear=linear, ann=ann, preprocess_method='BEST')
                score_list.append(self.sigma_eta_score(y_test, y_pred))

            elif method == 'median':
                # print('->'+method)
                for df in [train, test]:
                    missing_data = self.missing_data(df)
                    for i in missing_data.index:
                        df[i] = df[i].fillna(df[i].median())
                    train_test.append(df)
                # print(train_test[0].head(1))
                _, y_pred, y_test = self.pmodel(regressor, train=train_test[0], test=train_test[1], weight_train=sample_weight_train, scaler=scaler, linear=linear, ann=ann, preprocess_method='BEST')
                score_list.append(self.sigma_eta_score(y_test, y_pred))
            df_list.append(train_test)
        
        best_index = np.argmin(np.array(score_list))
        best_method = methods[best_index]
        best_train_test = df_list[best_index]

        if best_method == 'drop':
            w = weight_train
        else:
            w = sample_weight_train

        return best_train_test[0], best_train_test[1], w, best_method


    def find_uncommon_chars(self, str1, str2): 
        MAX_CHAR = 26
        
        # mark presence of each character as 0  
        # in the hash table 'present[]' 
        present = [0] * MAX_CHAR 
        for i in range(0, MAX_CHAR): 
            present[i] = 0
    
        l1 = len(str1) 
        l2 = len(str2) 
        
        # for each character of str1, mark its  
        # presence as 1 in 'present[]' 
        for i in range(0, l1): 
            present[ord(str1[i]) - ord('a')] = 1
            
        # for each character of str2  
        for i in range(0, l2): 
            
            # if a character of str2 is also present  
            # in str1, then mark its presence as -1  
            if(present[ord(str2[i]) - ord('a')] == 1 or 
            present[ord(str2[i]) - ord('a')] == -1): 
                present[ord(str2[i]) - ord('a')] = -1
    
            # else mark its presence as 2  
            else: 
                present[ord(str2[i]) - ord('a')] = 2
    
        # print all the uncommon characters  
        diff = []
        for i in range(0, MAX_CHAR): 
            if(present[i] == 1 or present[i] == 2): 
                diff.append(chr(i + ord('a')))
        return diff

    def best_improvement_morph(self, model, df=None, params={}, scaler=False, linear=False, ann=False):
        import itertools

        if df is None:
            df = self.df
        mags = ['r', 'i', 'g', 'u', 'z']
        df.drop(columns = 'Surf. bright.', inplace=True)
        morph_params = ['Elongation', 'Rho4', 'Galaxy size', 'Gini index'] 
        score_df = pd.DataFrame(columns=['bands', 'score diff.', 'score w. morph', 'score w.o. morph', 'score'])
        magnitudes, score_diff, score_w_morph, score_wo_morph, score = [], [], [], [], []
        print('[Morphological parameters importance]')
        print('---------------------------------------------------------------')
        for L in range(1, len(mags)):
            for subset in itertools.combinations(mags, L):
                df_copy = df.copy()
                df_copy.drop(columns=list(subset), inplace=True)
                df_w_morph = df_copy.copy()
                df_wo_morph = df_copy.copy()
                df_wo_morph.drop(columns=morph_params, inplace=True)

                print(list(df_wo_morph.columns))

                score_w, error_w = self.cross_validation(model, params=params, metric=self.sigma_eta_score, scaler=scaler, linear=linear, ann=ann, dataframe=df_w_morph)
                score_wo, error_wo = self.cross_validation(model, params=params, metric=self.sigma_eta_score, scaler=scaler, linear=linear, ann=ann, dataframe=df_wo_morph)

                bands = self.find_uncommon_chars(''.join(list(subset)), ''.join(mags))
                magnitudes.append('$' + ''.join(bands) + '$')
                score.append(score_w - score_wo)
                score_diff.append('${:.3f}\pm{:.3f}$'.format(score_w - score_wo, np.abs(error_w + error_wo)))
                score_w_morph.append('${:.3f}\pm{:.3f}$'.format(score_w, np.abs(error_w)))
                score_wo_morph.append('${:.3f}\pm{:.3f}$'.format(score_wo, np.abs(error_wo)))
        score_df['bands'] = magnitudes
        score_df['score diff.'] = score_diff
        score_df['score w. morph'] = score_w_morph
        score_df['score w.o. morph'] = score_wo_morph
        score_df['score'] = score
        score_df.sort_values(by='score', inplace=True)
        score_df.drop(columns='score', inplace=True)
        score_df.to_csv(self.output_path + 'files/' + 'Best_improv_morph_' + self.output_name + '_' + self.method + '.csv', index=False)
        print(score_df.head(10))
        



    def pmodel(self, regressor, train=None, test=None, weight_train=None, scaler=False, linear=False, ann=False, preprocess_method=None):

        if test is None and train is None:
            test = pd.concat((self.X_test, self.y_test), axis=1)
            train = pd.concat((self.X_train, self.y_train), axis=1)
        elif test is not None and train is None:           
            raise ValueError('test and train are either both None or both a dataframe') 
        elif test is None and train is not None:
            raise ValueError('test and train are either both None or both a dataframe')

        if weight_train is None:
            sample_weight_train = self.sample_weight_train
        elif weight_train is not None:
            sample_weight_train = weight_train
            try:
                train.drop(columns=['weight'], inplace=True)
            except:
                pass

        if preprocess_method == 'BEST':
            pass
        # elif preprocess_method is not None and preprocess_method != 'BEST':
        else:
            if weight_train is None:
                train, test, sample_weight_train, best_method = self.preprocess(regressor, train, test, self.sample_weight_train, method=self.preprocessing)
            elif weight_train is not None:
                train, test, sample_weight_train, best_method = self.preprocess(regressor, train, test, weight_train, method=self.preprocessing)


        X_train, X_test = train.iloc[:,:-1], test.iloc[:,:-1]
        y_train, y_test = train.iloc[:,-1], test.iloc[:,-1]

        if linear == True:
            kwargs = {regressor.steps[-1][0] + '__sample_weight': sample_weight_train}
            regressor.fit(X_train, y_train, **kwargs)
            y_pred = regressor.predict(X_test)
        else:
            if scaler == False:
                regressor.fit(X_train, y_train, sample_weight=sample_weight_train)
                y_pred = regressor.predict(X_test)
            if scaler == True:
                sc_X = StandardScaler()
                X_train = sc_X.fit_transform(X_train)
                X_test = sc_X.transform(X_test)
                if ann == True:
                    regressor.fit(X_train, y_train, batch_size = 32, epochs = 100, verbose=0, sample_weight=sample_weight_train)
                else:
                    regressor.fit(X_train, y_train, sample_weight=sample_weight_train)
                y_pred = regressor.predict(X_test)

        df = pd.DataFrame(data={'y_pred': y_pred.flatten(), 'y_test': y_test})
        df.to_csv(self.output_path + 'files/ML/' + self.method + '_prediction_' + self.output_name + '.csv', index = False)

        return regressor, y_pred, y_test
        

class RandomForest(LearningAlgorithms):

    def __init__(self, survey, bands, output_name, output_path, temp_path, path_to_csv, dataframe, sample_weight, validation_set, cv, preprocessing, n_jobs):
        super().__init__(survey, bands, output_name, output_path=output_path, temp_path=temp_path, path_to_csv=path_to_csv, dataframe=dataframe, sample_weight=sample_weight, validation_set=validation_set, cv=cv, preprocessing=preprocessing, n_jobs=n_jobs)
        self.method = 'RF'
        self.params = {'n_estimators': 86, 'max_depth': 11, 'n_jobs':-1}


    def model(self):
        """[summary]

        Returns:
            [type]: regressor, y_pred, y_test
        """
        regressor = RandomForestRegressor(**self.params)

        return LearningAlgorithms.pmodel(self, regressor)

    def filter(self):
        df = self.feature_remover(model=RandomForestRegressor, dataframe=self.df, params=self.params, metric=False, scaler=False, linear=False, ann=False)
        return df

    def morph_importance(self, df):
        LearningAlgorithms.best_improvement_morph(self, model=RandomForestRegressor, df=df, params=self.params, scaler=False, linear=False, ann=False)


    def score(self, df):
        """[summary]

        Args:
            cv (int, optional): [description]. Defaults to 4.

        Returns:
            [type]: np.mean(sigma_list), np.std(sigma_list), np.mean(eta_list), np.std(eta_list)
        """
        
        return LearningAlgorithms.cross_validation(self, model=RandomForestRegressor, params=self.params, dataframe=df)


    def plot(self, lim):
        _, y_pred, y_test = self.model()
        LearningAlgorithms.plot_zphot_zspec(self, y_pred, y_test, self.method, lim)


    def permutation(self):
        _ = LearningAlgorithms.permutation_importance(self,  model=RandomForestRegressor, params=self.params, scaler=False, linear=False, ann=False)
        


class SupportVectorRegression(LearningAlgorithms):

    def __init__(self, survey, bands, output_name, output_path, temp_path, path_to_csv, dataframe, sample_weight, validation_set, cv, preprocessing, n_jobs):
        super().__init__(survey, bands, output_name, output_path=output_path, temp_path=temp_path, path_to_csv=path_to_csv, dataframe=dataframe, sample_weight=sample_weight, validation_set=validation_set, cv=cv, preprocessing=preprocessing, n_jobs=n_jobs)
        self.method = 'SVR'
        self.params = {'C':1.0, 'cache_size':200, 'coef0':0.0, 'epsilon':0.1, 'gamma':'scale', 'kernel':'rbf', 'max_iter':-1, 'shrinking':True, 'tol':0.001}


    def model(self):
        regressor = SVR(**self.params)

        return LearningAlgorithms.pmodel(self, regressor, scaler=True)

    def filter(self):
        df = self.feature_remover(model=SVR, dataframe=self.df, params=self.params, metric=False, scaler=True, linear=False, ann=False)
        return df

    def score(self, df):
        """[summary]

        Args:
            cv (int, optional): [description]. Defaults to 4.

        Returns:
            [type]: np.mean(sigma_list), np.std(sigma_list), np.mean(eta_list), np.std(eta_list)
        """
        return LearningAlgorithms.cross_validation(self, model=SVR, params=self.params, dataframe=df)


    def plot(self, lim):
        _, y_pred, y_test = self.model()
        LearningAlgorithms.plot_zphot_zspec(self, y_pred, y_test, self.method, lim)

    def permutation(self):
        _ = LearningAlgorithms.permutation_importance(self,  model=SVR, params=self.params, scaler=True, linear=False, ann=False)


class LightGBM(LearningAlgorithms):

    def __init__(self, survey, bands, output_name, output_path, temp_path, path_to_csv, dataframe, sample_weight, validation_set, cv, preprocessing, n_jobs):
        super().__init__(survey, bands, output_name, output_path=output_path, temp_path=temp_path, path_to_csv=path_to_csv, dataframe=dataframe, sample_weight=sample_weight, validation_set=validation_set, cv=cv, preprocessing=preprocessing, n_jobs=n_jobs)
        self.method = 'LGB'
        self.params = {'objective':'regression','num_leaves':5,
                              'learning_rate':0.05, 'n_estimators':720,
                              'max_bin' : 55, 'bagging_fraction' : 0.8,
                              'bagging_freq' : 5, 'feature_fraction' : 0.2319,
                              'feature_fraction_seed':9, 'bagging_seed':9,
                              'min_data_in_leaf' :6, 'min_sum_hessian_in_leaf' : 11}


    def model(self):
        
        
        regressor = lgb.LGBMRegressor(**self.params)

        return LearningAlgorithms.pmodel(self, regressor, scaler=False)

    def filter(self):
        df = self.feature_remover(model=lgb.LGBMRegressor, dataframe=self.df, params=self.params, metric=False, scaler=False, linear=False, ann=False)
        return df

    def score(self, df):
        """[summary]

        Args:
            cv (int, optional): [description]. Defaults to 4.

        Returns:
            [type]: np.mean(sigma_list), np.std(sigma_list), np.mean(eta_list), np.std(eta_list)
        """
        return LearningAlgorithms.cross_validation(self, model=lgb.LGBMRegressor, params=self.params, dataframe=df)


    def plot(self, lim):
        _, y_pred, y_test = self.model()
        LearningAlgorithms.plot_zphot_zspec(self, y_pred, y_test, self.method, lim)

    def permutation(self):
        _ = LearningAlgorithms.permutation_importance(self,  model=lgb.LGBMRegressor, params=self.params, scaler=False, linear=False, ann=False)

class XGBoost(LearningAlgorithms):

    def __init__(self, survey, bands, output_name, output_path, temp_path, path_to_csv, dataframe, sample_weight, validation_set, cv, preprocessing, n_jobs):
        super().__init__(survey, bands, output_name, output_path=output_path, temp_path=temp_path, path_to_csv=path_to_csv, dataframe=dataframe, sample_weight=sample_weight, validation_set=validation_set, cv=cv, preprocessing=preprocessing, n_jobs=n_jobs)
        self.method = 'XGB'
        self.params = {'max_depth':3, 'n_estimators':2200, 'random_state':0, 'nthread': -1, 'n_jobs':-1}


    def model(self):

        regressor = xgb.XGBRegressor(**self.params)
        return LearningAlgorithms.pmodel(self, regressor, scaler=False)

    def filter(self):
        df = self.feature_remover(model=xgb.XGBRegressor, dataframe=self.df, params=self.params, metric=False, scaler=False, linear=False, ann=False)
        return df


    def score(self, df):
        """[summary]

        Args:
            cv (int, optional): [description]. Defaults to 4.

        Returns:
            [type]: np.mean(sigma_list), np.std(sigma_list), np.mean(eta_list), np.std(eta_list)
        """
        return LearningAlgorithms.cross_validation(self, model=xgb.XGBRegressor, params=self.params, dataframe=df)


    def plot(self, lim):
        _, y_pred, y_test = self.model()
        LearningAlgorithms.plot_zphot_zspec(self, y_pred, y_test, self.method, lim)

    def permutation(self):
        _ = LearningAlgorithms.permutation_importance(self,  model=xgb.XGBRegressor, params=self.params, scaler=False, linear=False, ann=False)


class GradientBoostingRegression(LearningAlgorithms):

    def __init__(self, survey, bands, output_name, output_path, temp_path, path_to_csv, dataframe, sample_weight, validation_set, cv, preprocessing, n_jobs):
        super().__init__(survey, bands, output_name, output_path=output_path, temp_path=temp_path, path_to_csv=path_to_csv, dataframe=dataframe, sample_weight=sample_weight, validation_set=validation_set, cv=cv, preprocessing=preprocessing, n_jobs=n_jobs)
        self.method = 'GBR'
        self.params = {'n_estimators':3000, 'learning_rate':0.05,
                                   'max_depth':4, 'max_features':'sqrt',
                                   'random_state':0}


    def model(self):

        regressor = GradientBoostingRegressor(**self.params)

        return LearningAlgorithms.pmodel(self, regressor, scaler=False)

    def filter(self):
        df = self.feature_remover(model=GradientBoostingRegressor, dataframe=self.df, params=self.params, metric=False, scaler=False, linear=False, ann=False)
        return df


    def score(self, df):
        """[summary]

        Args:
            cv (int, optional): [description]. Defaults to 4.

        Returns:
            [type]: np.mean(sigma_list), np.std(sigma_list), np.mean(eta_list), np.std(eta_list)
        """
        return LearningAlgorithms.cross_validation(self, model=GradientBoostingRegressor, params=self.params, dataframe=df)


    def plot(self, lim):
        _, y_pred, y_test = self.model()
        LearningAlgorithms.plot_zphot_zspec(self, y_pred, y_test, self.method, lim)

    def permutation(self):
        _ = LearningAlgorithms.permutation_importance(self,  model=GradientBoostingRegressor, params=self.params, scaler=False, linear=False, ann=False)


class KernelRidgeRegression(LearningAlgorithms):

    def __init__(self, survey, bands, output_name, output_path, temp_path, path_to_csv, dataframe, sample_weight, validation_set, cv, preprocessing, n_jobs):
        super().__init__(survey, bands, output_name, output_path=output_path, temp_path=temp_path, path_to_csv=path_to_csv, dataframe=dataframe, sample_weight=sample_weight, validation_set=validation_set, cv=cv, preprocessing=preprocessing, n_jobs=n_jobs)
        self.method = 'KRR'
        self.params = {'alpha':0.6, 'kernel':'polynomial', 'degree':2, 'coef0':2.5}

    
    def model(self):

        regressor = KernelRidge(**self.params)

        return LearningAlgorithms.pmodel(self, regressor, scaler=True)

    def filter(self):
        df = self.feature_remover(model=KernelRidge, dataframe=self.df, params=self.params, metric=False, scaler=True, linear=False, ann=False)
        return df

    def score(self, df):
        """[summary]

        Args:
            cv (int, optional): [description]. Defaults to 4.

        Returns:
            [type]: np.mean(sigma_list), np.std(sigma_list), np.mean(eta_list), np.std(eta_list)
        """
        return LearningAlgorithms.cross_validation(self, model=KernelRidge, params=self.params, dataframe=df)


    def plot(self, lim):
        _, y_pred, y_test = self.model()
        LearningAlgorithms.plot_zphot_zspec(self, y_pred, y_test, self.method, lim)

    def permutation(self):
        _ = LearningAlgorithms.permutation_importance(self,  model=KernelRidge, params=self.params, scaler=True, linear=False, ann=False)

class ElasticNetRegression(LearningAlgorithms):

    def __init__(self, survey, bands, output_name, output_path, temp_path, path_to_csv, dataframe, sample_weight, validation_set, cv, preprocessing, n_jobs):
        super().__init__(survey, bands, output_name, output_path=output_path, temp_path=temp_path, path_to_csv=path_to_csv, dataframe=dataframe, sample_weight=sample_weight, validation_set=validation_set, cv=cv, preprocessing=preprocessing, n_jobs=n_jobs)
        self.method = 'ENET'
        self.params={'alpha':0.0005, 'l1_ratio':.9, 'random_state':0}


    def model(self):

        regressor = make_pipeline(RobustScaler(), ElasticNet(**self.params))
        return LearningAlgorithms.pmodel(self, regressor, scaler=False, linear=True)

    def filter(self):
        df = self.feature_remover(model=ElasticNet, dataframe=self.df, params=self.params, metric=False, scaler=False, linear=True, ann=False)
        return df


    def score(self, df):
        """[summary]

        Args:
            cv (int, optional): [description]. Defaults to 4.

        Returns:
            [type]: np.mean(sigma_list), np.std(sigma_list), np.mean(eta_list), np.std(eta_list)
        """
        return LearningAlgorithms.cross_validation(self, model=ElasticNet, params=self.params, linear=True, dataframe=df)


    def plot(self, lim):
        regressor, y_pred, _ = self.model()
        LearningAlgorithms.plot_zphot_zspec(self, y_pred, self.method, lim)

    def permutation(self):
        _ = LearningAlgorithms.permutation_importance(self,  model=ElasticNet, params=self.params, scaler=False, linear=True, ann=False)


class LassoRegression(LearningAlgorithms):

    def __init__(self, survey, bands, output_name, output_path, temp_path, path_to_csv, dataframe, sample_weight, validation_set, cv, preprocessing, n_jobs):
        super().__init__(survey, bands, output_name, output_path=output_path, temp_path=temp_path, path_to_csv=path_to_csv, dataframe=dataframe, sample_weight=sample_weight, validation_set=validation_set, cv=cv, preprocessing=preprocessing, n_jobs=n_jobs)
        self.method = 'LASSO'
        self.params={'alpha': 0.0005, 'random_state':0}




    def model(self):

        regressor = make_pipeline(RobustScaler(), Lasso(**self.params))
        return LearningAlgorithms.pmodel(self, regressor, scaler=False, linear=True)


    def filter(self):
        df = self.feature_remover(model=Lasso, dataframe=self.df, params=self.params, metric=False, scaler=False, linear=True, ann=False)
        return df

    def score(self, df):
        """[summary]

        Args:
            cv (int, optional): [description]. Defaults to 4.

        Returns:
            [type]: np.mean(sigma_list), np.std(sigma_list), np.mean(eta_list), np.std(eta_list)
        """
        return LearningAlgorithms.cross_validation(self, model=Lasso, params=self.params, linear=True, dataframe=df)


    def plot(self, lim):
        _, y_pred, y_test = self.model()
        LearningAlgorithms.plot_zphot_zspec(self, y_pred, y_test, self.method, lim)

    def permutation(self):
        _ = LearningAlgorithms.permutation_importance(self,  model=Lasso, params=self.params, scaler=False, linear=True, ann=False)


class ArtificialNeuralNetwork(LearningAlgorithms):

    def __init__(self, survey, bands, output_name, output_path, temp_path, path_to_csv, dataframe, sample_weight, validation_set, cv, preprocessing, n_jobs):
        super().__init__(survey, bands, output_name, output_path=output_path, temp_path=temp_path, path_to_csv=path_to_csv, dataframe=dataframe, sample_weight=sample_weight, validation_set=validation_set, cv=cv, preprocessing=preprocessing, n_jobs=n_jobs)
        self.method = 'ANN'

    def ann(self):
        import tensorflow as tf
        import tensorflow.keras as tfk

        ann = tf.keras.models.Sequential()
        ann.add(tf.keras.layers.Dense(units=5, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=14, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=9, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=1))
        ann.compile(optimizer = 'adam', loss = 'mean_squared_error')

        return ann

    def model(self):

        sc = StandardScaler()
        X_train = sc.fit_transform(self.X_train)
        X_test = sc.transform(self.X_test)

        ann = self.ann()
        
        ann.fit(X_train, self.y_train, batch_size = 32, epochs = 50, verbose=0, sample_weight=self.sample_weight_train)
        y_pred = ann.predict(X_test)

        y_test = self.y_test
        df = pd.DataFrame(data={'y_pred': y_pred.flatten(), 'y_test': y_test})
        df.to_csv(self.output_path + 'files/ML/' + self.method + '_prediction_' + self.output_name + '.csv', index = False)

        return ann, y_pred, y_test

    def filter(self):
        df = self.feature_remover(model=self.ann(), dataframe=self.df, params={}, metric=False, scaler=True, linear=False, ann=True)
        return df

    def score(self, df):
        """[summary]

        Args:
            cv (int, optional): [description]. Defaults to 4.

        Returns:
            [type]: np.mean(sigma_list), np.std(sigma_list), np.mean(eta_list), np.std(eta_list)
        """
        return LearningAlgorithms.cross_validation(self, model=self.ann(), scaler=True, ann=True, dataframe=df)


    def plot(self, lim):
        _, y_pred, y_test = self.model()
        LearningAlgorithms.plot_zphot_zspec(self, y_pred.flatten(), y_test, self.method, lim)

    def permutation(self):
        _ = LearningAlgorithms.permutation_importance(self, model=self.ann, linear=False, scaler=True, ann=True)


class ConvolutionalNeuralNetwork(LearningAlgorithms):

    def __init__(self, survey, bands, output_name, output_path, path_to_csv, dataframe, sample_weight, validation_set, cv, preprocessing, n_jobs):
        super().__init__(survey, bands, output_name, output_path=output_path, path_to_csv=path_to_csv, dataframe=dataframe, sample_weight=sample_weight, validation_set=validation_set, cv=cv, preprocessing=preprocessing, n_jobs=n_jobs)
        self.X = np.load(self.output_path + 'output/' + self.survey + '/files/' + output_name + '.npy')
        self.X_train, self.X_test = train_test_split(self.X, test_size = 0.2, random_state=0)
        self.X_train, self.X_val = train_test_split(self.X_train, test_size = 0.2, random_state=0)
        self.method = 'CNN'


    def model(self):
        import tensorflow as tf
        import tensorflow.keras as tfk

        model = tfk.models.Sequential()
        model.add(tfk.layers.Conv2D(32, kernel_size=5, padding='same',
                                    input_shape=(51,51,len(self._bands)), activation='elu', strides=2))
        model.add(tfk.layers.BatchNormalization())

        model.add(tfk.layers.Conv2D(64, kernel_size=3, padding='same', 
                                activation='elu'))
        model.add(tfk.layers.BatchNormalization())

        model.add(tfk.layers.Conv2D(128, kernel_size=3, padding='same', strides=2, 
                                    activation='elu'))
        model.add(tfk.layers.BatchNormalization())  

        model.add(tfk.layers.Conv2D(256, kernel_size=3, padding='same', 
                                    activation='elu', strides=2))
        model.add(tfk.layers.BatchNormalization())

        model.add(tfk.layers.Conv2D(512, kernel_size=3, padding='same', 
                                    activation='elu', strides=2))
        model.add(tfk.layers.BatchNormalization())

        model.add(tfk.layers.Flatten())
        model.add(tfk.layers.Dense(512))
        model.add(tfk.layers.Activation('relu'))
        model.add(tfk.layers.Dense(256))
        model.add(tfk.layers.Activation('relu'))
        model.add(tfk.layers.Dense(1))

        model.compile(optimizer='adam', loss=tfk.metrics.mse)
        
        return model

    def cnn_model(self):
        import tensorflow as tf
        import tensorflow.keras as tfk
        scaling_train, scaling_test, scaling_val = [], [], []
        X_train = np.ndarray((len(self.X_train),51,51,len(self._bands)))
        X_val = np.ndarray((len(self.X_val),51,51,len(self._bands)))
        X_test = np.ndarray((len(self.X_test),51,51,len(self._bands)))

        for i in range(len(self._bands)):
            sigma_train = mad_std(self.X_train[...,i].flatten())
            scaling_train.append(sigma_train)
            sigma_val = mad_std(self.X_val[...,i].flatten())
            scaling_val.append(sigma_val)
            sigma_test = mad_std(self.X_test[...,i].flatten())
            scaling_test.append(sigma_test)

        for i in range(len(self.X_train)):
            for j in range(len(self._bands)):
                X_train[i,:,:,j] = np.arcsinh(self.X_train[i,:,:,j]/ scaling_train[j])

        for i in range(len(self.X_val)):
            for j in range(len(self._bands)):
                X_val[i,:,:,j] = np.arcsinh(self.X_val[i,:,:,j]/ scaling_val[j])

        for i in range(len(self.X_test)):
            for j in range(len(self._bands)):
                X_test[i,:,:,j] = np.arcsinh(self.X_test[i,:,:,j]/ scaling_test[j])

            
        # We define the batch size
        BATCH_SIZE = 64

        # Learning rate schedule
        LEARNING_RATE=0.001
        LEARNING_RATE_EXP_DECAY=0.9

        lr_decay = tfk.callbacks.LearningRateScheduler(
            lambda epoch: LEARNING_RATE * LEARNING_RATE_EXP_DECAY**epoch,
            verbose=True)
        
        # We are ready to train our model
        model = self.model()
        model.fit(X_train, self.y_train,
                steps_per_epoch=20000//BATCH_SIZE, 
                epochs=15,
                callbacks=[lr_decay], validation_data=(X_val, self.y_val))
        
        y_pred = model.predict(X_test).flatten()

        y_test = self.y_test
        df = pd.DataFrame(data={'y_pred': y_pred, 'y_test': y_test})
        df.to_csv(self.output_path + 'files/ML/' + self.method + '_prediction_' + self.output_name + '.csv', index = False)

        return model, y_pred



class Optimizer(LearningAlgorithms):

    def __init__(self, survey, bands, output_name, output_path, path_to_csv, dataframe, sample_weight, validation_set, cv, preprocessing, n_jobs):
        super().__init__(survey, bands, output_name, output_path=output_path, path_to_csv=path_to_csv, dataframe=dataframe, sample_weight=sample_weight, validation_set=validation_set, cv=cv, preprocessing=preprocessing, n_jobs=n_jobs)
    
    
    def sigma_eta_score(self, y_test, y_pred):
        delta_z = (y_test - y_pred)/(np.ones_like(y_test) + y_test)
        sigma = 1.4826*np.median( np.abs( delta_z - np.median(delta_z) ) )
        outlier = np.abs(delta_z) > 0.15
        outlier_rate = len([x for x in outlier.ravel() if x==True])/len(delta_z.ravel())
        return np.sqrt(np.mean(delta_z)**2 + sigma**2)


    def cross_validation(self, metric, model, scaler=False, linear=False, ann=False):
        """[summary]

        Args:
            metric ([type]): [description]
            model ([type]): [description]
            params ([type]): [description]
            cv (int, optional): [description]. Defaults to 4.

        Returns:
            [type]: [description]
        """

        df_X_list = []
        df_y_list = []
        weight_list = []
        n = len(self.X_train)
        for i in range(0,self.cv):
            if i < self.cv-1:
                df_X_list.append(self.X_train.iloc[i*int(n/self.cv):(i+1)*int(n/self.cv), :])
                df_y_list.append(self.y_train.iloc[i*int(n/self.cv):(i+1)*int(n/self.cv)])
                if self.sample_weight is not None:
                    weight_list.append(self.sample_weight_train[i*int(n/self.cv):(i+1)*int(n/self.cv)])
            if i == self.cv-1:
                df_X_list.append(self.X_train.iloc[i*int(n/self.cv):, :])
                df_y_list.append(self.y_train.iloc[i*int(n/self.cv):])
                if self.sample_weight is not None:
                    weight_list.append(self.sample_weight_train[i*int(n/self.cv):])

        score_list = []
        for i in range(0,self.cv):

            X_test = df_X_list[i]
            y_test = df_y_list[i]

            X_train = pd.concat(np.array([(j,X) for (j,X) in enumerate(df_X_list) if i != j])[:,1])
            y_train = pd.concat(np.array([(j,y) for (j,y) in enumerate(df_y_list) if i != j])[:,1])
            if self.sample_weight is not None:
                weight_train = np.concatenate(np.array([(j,w) for (j,w) in enumerate(weight_list) if i != j])[:,1])
            elif self.sample_weight is None:
                weight_train = None

            test = pd.concat((X_test, y_test), axis=1)
            train = pd.concat((X_train, y_train), axis=1)
            
            if ann == True:
                regressor = model
                train, test, weight_train, _ = self.preprocess(regressor, train, test, weight_train, method=self.preprocessing, scaler=scaler, linear=linear)
            else:
                if linear == True:
                    regressor = model = make_pipeline(RobustScaler(), model)
                    train, test, weight_train, _ = self.preprocess(regressor, train, test, weight_train, method=self.preprocessing, scaler=scaler, linear=linear)
                elif linear == False:
                    regressor = model
                    train, test, weight_train, _ = self.preprocess(regressor, train, test, weight_train, method=self.preprocessing, scaler=scaler, linear=linear)

            X_train, X_test = train.iloc[:,:-1], test.iloc[:,:-1]
            y_train, y_test = train.iloc[:,-1], test.iloc[:,-1]

            if scaler == True:
                sc_X = StandardScaler()
                X_train = sc_X.fit_transform(X_train)
                X_test = sc_X.transform(X_test)

            if ann == True:
                model.fit(X_train, y_train, batch_size = 32, epochs = 100, verbose=0, sample_weight=weight_train)
            else:
                if linear == True:
                    model = make_pipeline(RobustScaler(), model)
                    kwargs = {model.steps[-1][0] + '__sample_weight': weight_train}
                    model.fit(X_train, y_train, **kwargs)
                    
                elif linear == False:
                    model.fit(X_train, y_train, sample_weight=weight_train)
            y_pred = model.predict(X_test)

            score = metric(y_test, y_pred)
            score_list.append(score)

        return np.array(score)
        

class RandomForestOptimizer(Optimizer):

    def __init__(self, survey, bands, output_name, output_path, path_to_csv, dataframe, sample_weight, validation_set, cv, preprocessing, n_jobs):
        super().__init__(survey, bands, output_name, output_path, path_to_csv, dataframe, sample_weight, validation_set, cv, preprocessing, n_jobs)
        self.method = 'RF_Opt'


    def objective(self, params):

        est=int(params['n_estimators'])
        md=int(params['max_depth'])
        msl=int(params['min_samples_leaf'])
        mss=int(params['min_samples_split'])
        cri=params['criterion']
        params = {'n_estimators':est,'max_depth':md,'min_samples_leaf':msl,'min_samples_split':mss, 'criterion':cri, 'n_jobs':-1}

        # model=RandomForestRegressor(n_estimators=est,max_depth=md,min_samples_leaf=msl,min_samples_split=mss, criterion=cri)
        # model.fit(self.X_train,self.y_train)
        # y_pred = model.predict(self.X_val)
        # score = Optimizer.sigma_eta_score(self, self.y_val, y_pred)

        score =  Optimizer.cross_validation(self, model=RandomForestRegressor(**params), metric=self.sigma_eta_score)
        return score.mean()

    def optimize(self, trial, max_evals = 200):

        params={'n_estimators':hp.uniform('n_estimators',10,500),
                'criterion':hp.choice('criterion', ['mse', 'mae']),
            'max_depth':hp.uniform('max_depth',5,20),
            'min_samples_leaf':hp.uniform('min_samples_leaf',1,5),
            'min_samples_split':hp.uniform('min_samples_split',2,6)}
        best=fmin(fn=self.objective,space=params,algo=tpe.suggest,trials=trial,max_evals=max_evals,rstate=np.random.RandomState(seed=2))

        criterion = ['mse', 'mae']
        best['max_depth'] = int(best['max_depth'])
        best['min_samples_leaf'] = int(best['min_samples_leaf'])
        best['min_samples_split'] = int(best['min_samples_split'])
        best['n_estimators'] = int(best['n_estimators'])
        best['criterion'] = criterion[best['criterion']]
        return best

    def grid_search(self):

        score = make_scorer(self.sigma_eta_score, greater_is_better=False)
        
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 10, stop = 500, num = 4)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(5, 20, num = 4)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}

        # Use the random grid to search for best hyperparameters
        # First create the base model to tune
        rf = RandomForestRegressor()
        # Random search of parameters, using 3 fold cross validation, 
        # search across 100 different combinations, and use all available cores
        
        # Fit the random search model
        train, test, weight_train, method = self.preprocess(regressor=RandomForestRegressor, train=self.train, test=self.test,
            weight_train=self.weight_train, method=self.preprocessing, scaler=False, linear=False, ann=False)
        X_train = train.iloc[:,:-1]
        y_train = train.iloc[:,-1]
        fit_params={'sample_weight': weight_train}
        rf_grid = GridSearchCV(estimator = rf, param_grid = grid, scoring=score, cv = self.cv, verbose=2, n_jobs = self.n_jobs)
        rf_grid.fit(X_train, y_train)

        best = rf_grid.best_params_

        return best


    def random_search(self, max_evals):

        score = make_scorer(self.sigma_eta_score, greater_is_better=False)

        
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 10, stop = 500, num = 10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(5, 20, num = 10)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}

        # Use the random grid to search for best hyperparameters
        # First create the base model to tune
        rf = RandomForestRegressor()
        # Random search of parameters, using 3 fold cross validation, 
        # search across 100 different combinations, and use all available cores
        # Fit the random search model
        train, test, weight_train, method = self.preprocess(regressor=RandomForestRegressor, train=self.train, test=self.test,
            weight_train=self.weight_train, method=self.preprocessing, scaler=False, linear=False, ann=False)
        X_train = train.iloc[:,:-1]
        y_train = train.iloc[:,-1]
        fit_params={'sample_weight': weight_train}

        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, scoring=score, n_iter = max_evals, cv = self.cv, verbose=2, random_state=0, n_jobs = self.n_jobs)
        rf_random.fit(X_train, y_train)

        best = rf_random.best_params_

        return best


    def best_params(self, max_evals=200, method='HyperOpt'):
        trial = Trials()
        if method == 'HyperOpt':
            best = self.optimize(trial, max_evals)
        elif method == 'RandomSearch':
            best = self.random_search(max_evals)
        elif method == 'GridSearch':
            best = self.grid_search()
        _, y_pred, y_test = LearningAlgorithms.pmodel(self, regressor=RandomForestRegressor(**best), scaler=True)
        eta, sigma = LearningAlgorithms.sigma_eta(self, y_test, y_pred)
        score = self.sigma_eta_score(y_test, y_pred)

        LearningAlgorithms.plot_zphot_zspec(self, y_pred.flatten(), y_test, self.method+'_'+method, lim=1.8)

        return y_pred, eta, sigma, score



class SVROptimizer(Optimizer):

    def __init__(self, survey, bands, output_name, output_path, path_to_csv, dataframe, sample_weight, validation_set, cv, preprocessing, n_jobs):
        super().__init__(survey, bands, output_name, output_path, path_to_csv, dataframe, sample_weight, validation_set, cv, preprocessing, n_jobs)
        self.method = 'SVR_Opt'

    def objective(self, params):

        gam = params['gamma']
        cc = params['C']
        params = {'kernel':'rbf', 'gamma':gam, 'C':cc}


        # model = SVR(**params)
        # model.fit(self.X_train,self.y_train)
        # y_pred = model.predict(self.X_val)
        # score = Optimizer.sigma_eta_score(self, self.y_val, y_pred)

        score =  Optimizer.cross_validation(self, model=SVR(**params), metric=self.sigma_eta_score, scaler=True)
        return score.mean()

    def optimize(self, trial, max_evals = 200):

        params = {'C':hp.lognormal('C',0,1),
            'gamma':hp.uniform('gamma',0.001,10000)}

        best=fmin(fn=self.objective,space=params,algo=tpe.suggest,trials=trial,max_evals=max_evals,rstate=np.random.RandomState(seed=2))
        return best

    def best_params(self, max_evals=200, method='HyperOpt'):
        trial = Trials()
        best = self.optimize(trial, max_evals)
        _, y_pred, y_test = LearningAlgorithms.pmodel(self, regressor=SVR(**best), scaler=True)
        eta, sigma = LearningAlgorithms.sigma_eta(self, y_test, y_pred)
        score = self.sigma_eta_score(y_test, y_pred)

        LearningAlgorithms.plot_zphot_zspec(self, y_pred.flatten(), y_test, self.method+'_'+method, lim=1.8)
        return y_pred, eta, sigma, score


class XGBoostOptimizer(Optimizer):

    def __init__(self, survey, bands, output_name, output_path, path_to_csv, dataframe, sample_weight, validation_set, cv, preprocessing, n_jobs):
        super().__init__(survey, bands, output_name, output_path, path_to_csv, dataframe, sample_weight, validation_set, cv, preprocessing, n_jobs)
        self.method = 'XGB_Opt'
    
    def objective(self, params):
        est=int(params['n_estimators'])
        md=int(params['max_depth'])
        learning=params['learning_rate']
        
        params = {'n_estimators': est, 'max_depth':md, 'learning_rate': learning}

        score =  Optimizer.cross_validation(self, model=xgb.XGBRegressor(**params), metric=self.sigma_eta_score)
        return score.mean()

    def optimize(self, trial, max_evals=200):
        params={'n_estimators':hp.uniform('n_estimators',100,500),
            'max_depth':hp.uniform('max_depth',5,20),
            'learning_rate':hp.uniform('learning_rate',0.01,0.1)}
        best=fmin(fn=self.objective,space=params,algo=tpe.suggest,trials=trial,max_evals=max_evals,rstate=np.random.RandomState(seed=2))
        best['n_estimators'] = int(best['n_estimators'])
        best['max_depth'] = int(best['max_depth'])
        return best

    def best_params(self, max_evals=200, method='HyperOpt'):
        trial = Trials()
        best = self.optimize(trial, max_evals)
        _, y_pred, y_test = LearningAlgorithms.pmodel(self, regressor=XGBoost(**best))
        eta, sigma = LearningAlgorithms.sigma_eta(self, y_test, y_pred)
        score = self.sigma_eta_score(y_test, y_pred)

        LearningAlgorithms.plot_zphot_zspec(self, y_pred.flatten(), y_test, self.method+'_'+method, lim=1.8)

        return y_pred, eta, sigma, score


class KRROptimizer(Optimizer):

    def __init__(self, survey, bands, output_name, output_path, path_to_csv, dataframe, sample_weight, validation_set, cv, preprocessing, n_jobs):
        super().__init__(survey, bands, output_name, output_path, path_to_csv, dataframe, sample_weight, validation_set, cv, preprocessing, n_jobs)
        self.method = 'KRR_Opt'

    def objective(self, params):

        gam = params['gamma']
        al= params['alpha']
        deg = params['degree']
        params = {'kernel':'polynomial', 'gamma':gam, 'alpha':al, 'degree':deg}

        # model = SVR(**params)
        # model.fit(self.X_train,self.y_train)
        # y_pred = model.predict(self.X_val)
        # score = Optimizer.sigma_eta_score(self, self.y_val, y_pred)

        score =  Optimizer.cross_validation(self, model=KernelRidge(**params), metric=self.sigma_eta_score, scaler=True)
        return score.mean()

    def optimize(self, trial, max_evals = 200):

        params = {'alpha':hp.lognormal('alpha',0,1),
            'gamma':hp.uniform('gamma',0.001,10000),
            'degree': hp.choice('degree', [1,3,5,7,9])}

        best=fmin(fn=self.objective,space=params,algo=tpe.suggest,trials=trial,max_evals=max_evals,rstate=np.random.RandomState(seed=2))
        return best

    def best_params(self, max_evals=200, method='HyperOpt'):
        trial = Trials()
        best = self.optimize(trial, max_evals)
        _, y_pred, y_test = LearningAlgorithms.pmodel(self, regressor=KernelRidge(**best), scaler=True)
        eta, sigma = LearningAlgorithms.sigma_eta(self, y_test, y_pred)
        score = self.sigma_eta_score(y_test, y_pred)

        LearningAlgorithms.plot_zphot_zspec(self, y_pred.flatten(), y_test, self.method+'_'+method, lim=1.8)

        return y_pred, eta, sigma, score



class ANNOptimizer(Optimizer):

    def __init__(self, survey, bands, output_name, output_path, path_to_csv, dataframe, sample_weight, validation_set, cv, preprocessing, n_jobs):
        super().__init__(survey, bands, output_name, output_path, path_to_csv, dataframe, sample_weight, validation_set, cv, preprocessing, n_jobs)
        self.method = 'ANN_Opt'

    # @staticmethod
    # def data():
    #     import pandas as pd
    #     import numpy as np
    #     from sklearn.model_selection import train_test_split


    #     path = os.getcwd() + '/'
    #     dataset = pd.read_csv(path + 'catalogs/' + 'MediumDeep_CFHT_CFIS_R_matched_catalog_2' + '.csv')
    #     X = dataset.iloc[:, :-1].values
    #     y = dataset.iloc[:, -1].values
    #     y = y.reshape(len(y),1)

    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
    #     X_train, X_val, Y_train, Y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

    #     sc = StandardScaler()
    #     X_train = sc.fit_transform(X_train)
    #     X_test = sc.transform(X_test)

    #     print(len(X_val), len(Y_val))

    #     return X_train, Y_train, X_val, Y_val

    # @staticmethod
    # def model(X_train, Y_train, X_val, Y_val):

    #     model = tf.keras.models.Sequential()
    #     model.add(tf.keras.layers.Dense({{choice([128, 256, 512, 1024])}}, input_shape=(len(X_train[0]),)))
    #     model.add(tf.keras.layers.Activation({{choice(['relu', 'sigmoid'])}}))
    #     model.add(tf.keras.layers.Dropout({{uniform(0, 1)}}))
    #     model.add(tf.keras.layers.Dense({{choice([128, 256, 512, 1024])}}))
    #     model.add(tf.keras.layers.Activation({{choice(['relu', 'sigmoid'])}}))
    #     model.add(tf.keras.layers.Dropout({{uniform(0, 1)}}))


    #     if {{choice(['two', 'three'])}} == 'three':
    #         model.add(tf.keras.layers.Dense({{choice([128, 256, 512, 1024])}}))
    #         model.add(tf.keras.layers.Activation({{choice(['relu', 'sigmoid'])}}))
    #         model.add(tf.keras.layers.Dropout({{uniform(0, 1)}}))
            
    #     model.add(tf.keras.layers.Dense(units=1))
    #     adam = keras.optimizers.Adam(lr={{choice([10**-3, 10**-2, 10**-1])}})
    #     rmsprop = keras.optimizers.RMSprop(lr={{choice([10**-3, 10**-2, 10**-1])}})
    #     sgd = keras.optimizers.SGD(lr={{choice([10**-3, 10**-2, 10**-1])}})

    #     choiceval = {{choice(['adam', 'sgd', 'rmsprop'])}}

    #     if choiceval == 'adam':
    #         optim = adam
    #     elif choiceval == 'rmsprop':
    #         optim = rmsprop
    #     else:
    #         optim = sgd
            
    #     model.compile(loss = 'mean_squared_error', metrics=['mean_squared_error'],optimizer=optim)
    #     model.fit(X_train, Y_train,
    #             batch_size={{choice([128,256,512])}},
    #             epochs=50,
    #             verbose=2,
    #             validation_data=(X_val, Y_val))

    #     y_pred = model.predict(X_val)
    #     y_test = Y_val
    #     delta_z = (y_test - y_pred)/(np.ones_like(y_test) + y_test)
    #     sigma = 1.4826*np.median( np.abs( delta_z - np.median(delta_z) ) )
    #     outlier = np.abs(delta_z) > 0.15
    #     outlier_rate = len([x for x in outlier.ravel() if x==True])/len(delta_z.ravel())
    #     err = outlier_rate + sigma
    #     print('Test mean absolute error:', err)
    #     return {'loss': err, 'status': STATUS_OK, 'model': model}


    def best_params(self, max_evals = 200, method='HyperOpt'):
        from hyperas import optim
        from hyperas.distributions import choice, uniform


        trials = Trials()
        best_run, best_model = optim.minimize(model=model,data=data,algo=tpe.suggest, max_evals=max_evals, trials=trials)
        print(best_run)
        y_pred = best_model.predict(self.X_test, verbose = 0)
        eta, sigma = LearningAlgorithms.sigma_eta(self, self.y_test.values, y_pred)

        LearningAlgorithms.plot_zphot_zspec(self, y_pred.flatten(), y_test, self.method+'_'+method, lim=1.8)

        return y_pred, eta, sigma




def sigma_eta_score(y_test, y_pred):
    delta_z = (y_test - y_pred)/(np.ones_like(y_test) + y_test)
    sigma = 1.4826*np.median( np.abs( delta_z - np.median(delta_z) ) )
    outlier = np.abs(delta_z) > 0.15
    outlier_rate = len([x for x in outlier.ravel() if x==True])/len(delta_z.ravel())
    return sigma + outlier_rate


def data():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split


    path = os.getcwd() + '/'
    dataset = pd.read_csv(path + 'catalogs/' + 'MediumDeep_CFHT_CFIS_R_matched_catalog_2' + '.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    y = y.reshape(len(y),1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    print(len(X_val), len(Y_val))

    return X_train, Y_train, X_val, Y_val


def model(X_train, Y_train, X_val, Y_val):
    import tensorflow as tf
    import tensorflow.keras as tfk
    from hyperas import optim
    from hyperas.distributions import choice, uniform

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=5, activation='relu'))
    model.add(tf.keras.layers.Dense(units=6, activation='relu'))
    model.add(tf.keras.layers.Dense(units=14, activation='relu'))
    model.add(tf.keras.layers.Dense(units=9, activation='relu'))
    model.add(tf.keras.layers.Dense(units=1))

    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    model.fit(X_train, Y_train, batch_size = {{choice([32,64,128,256,512])}}, epochs = 100, verbose=0)
    # score, err = model.evaluate(X_val, Y_val, verbose=0)
    y_pred = model.predict(X_val)
    y_test = Y_val
    delta_z = (y_test - y_pred)/(np.ones_like(y_test) + y_test)
    sigma = 1.4826*np.median( np.abs( delta_z - np.median(delta_z) ) )
    outlier = np.abs(delta_z) > 0.15
    outlier_rate = len([x for x in outlier.ravel() if x==True])/len(delta_z.ravel())
    err = outlier_rate + sigma
    #err = mean_squared_error(Y_val, Y_pred)
    # print('Test mean absolute error:', err)
    return {'loss': err, 'status': STATUS_OK, 'model': model}


def sigma_eta(y_test, y_pred):
    delta_z = (y_test - y_pred)/(np.ones_like(y_test) + y_test)
    sigma = 1.4826*np.median( np.abs( delta_z - np.median(delta_z) ) )
    outlier = np.abs(delta_z) > 0.15
    outlier_rate = len([x for x in outlier.ravel() if x==True])/len(delta_z.ravel())
    return sigma, outlier_rate
