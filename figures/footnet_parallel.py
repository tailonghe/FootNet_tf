# -*- coding: utf-8 -*-
# @Author: tailong
# @Date:   2023-02-13 17:26:01
# @Last Modified by:   nikhildadheech
# @Last Modified time: 2023-02-16 16:55:11


import tensorflow as tf
import keras
from tensorflow.keras.utils import Sequence
from keras.layers import Input, LSTM, Dense, Permute, Reshape, concatenate
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras import backend as K
from keras.metrics import mean_squared_error
from keras.models import Model, load_model, Sequential
#from keras.losses import log_cosh

from datetime import datetime, timedelta
import numpy as np
import os
import glob
from random import shuffle
import pandas as pd
import xarray as xr
# import PseudoNetCDF as pnc
from scipy.spatial import Delaunay
import time
import netCDF4 as nc
# from tqdm import tqdm

# visualization
import proplot as pplt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from joblib import Parallel, delayed


import warnings
warnings.filterwarnings("ignore")

np.random.seed(12345)

# 481, 601 for Bay area
# nSample = 5000
RES = 16
NUNIT = 1024
DIM1 = 400   # 
DIM2 = 400
SHAPE1 = int((DIM1 + 15)//RES*RES  )
SHAPE2 = int((DIM2 + 15)//RES*RES  )
LATNTSHAPE = int(SHAPE2//16*SHAPE1//16)
HRRR_DIR = '/home/disk/hermes2/taihe/HRRR-lite_nc/'
# WRF_DIR = '/home/disk/hermes/data/met_data/BarnettShale_2013/wrf/MYJ_LSM/' # no longer in use
predlist = ['GPR', 'U10M', 'V10M', 'PBLH', 'PRSS', 'P850', 'T850'] # 14
Rd = 2.87053  # hPa·K-1·m3·kg–1
ggg = 9.80665 # m/s*s

HRR_lon_lat_npz = "HRRR_lon_lat.npz"

global hr3lon_full, hr3lat_full
hr3lon_full = np.load(HRR_lon_lat_npz)['lon']
hr3lat_full = np.load(HRR_lon_lat_npz)['lat']
hr3lon_full = (hr3lon_full+180)%360-180  # convert from 0~360 to -180~180


def zstandard(arr):
    _mu = np.nanmean(arr)
    _std = np.nanstd(arr)
    return (arr - _mu)/_std

def cropx(xx, yy):
    # crop input and output to 400 by 400
    dim1 = xx.shape[0]
    dim2 = xx.shape[1]
    dim1res = int((dim1 - 400)/2)
    dim2res = int((dim2 - 400)/2)
    return xx[dim1res:dim1res+400, dim2res:dim2res+400, :], yy[dim1res:dim1res+400, dim2res:dim2res+400]



def build_model(num_var=12):
    inputs = Input( ( 400, 400, 12 ), name='model_input')

    c1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='Block1_Conv1') (inputs)    # 400, 400
    c1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='Block1_Conv2') (c1)   # 400, 400
    p1 = MaxPooling2D((2, 2), name='Block1_MaxPool', padding='same') (c1)   # 200, 200

    c2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='Block2_Conv1') (p1)   # 200, 200
    c2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='Block2_Conv2') (c2)   # 200, 200
    p2 = MaxPooling2D((2, 2), name='Block2_MaxPool', padding='same') (c2)   # 100,100

    c3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='Block3_Conv1') (p2)   # 100, 100
    c3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='Block3_Conv2') (c3)   # 100, 100
    p3 = MaxPooling2D((2, 2), name='Block3_MaxPool', padding='same') (c3)  # 50, 50

    c4 = Conv2D(1024, (3, 3), activation='relu', padding='same', name='Block4_Conv1') (p3)   # 50, 50
    c4 = Conv2D(1024, (3, 3), activation='relu', padding='same', name='Block4_Conv2') (c4)   # 50, 50
    p4 = MaxPooling2D((2, 2), name='Block4_MaxPool', padding='same') (c4)  # 25, 25

    neck1 = Conv2D(1024, (3, 3), activation='relu', padding='same', name='Neck1') (p4)   # 50, 50
    neck2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='Neck2') (neck1)   # 50, 50

    u5 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same', name='Block5_UpConv') (neck2)  # 28, 22
    u5_comb = concatenate([u5, c4])  # 28, 22
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='Block5_Conv1') (u5_comb)  # 28, 22
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='Block5_Conv2') (c5)  # 28, 22

    u6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', name='Block6_UpConv') (c5)  # 56, 44
    u6_comb = concatenate([u6, c3])
    c6 = Conv2D(256, (3, 3), activation='relu', padding='same', name='Block6_Conv1') (u6_comb)  # 56, 44
    c6 = Conv2D(256, (3, 3), activation='relu', padding='same', name='Block6_Conv2') (c6)  # 56, 44

    u7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name='Block7_UpConv') (c6)  # 112, 88
    u7_comb = concatenate([u7, c2])
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same', name='Block7_Conv1') (u7_comb)  # 112, 88
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same', name='Block7_Conv2') (c7)  # 112, 88

    u8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name='Block8_UpConv') (c7)  # 224, 176
    u8_comb = concatenate([u8, c1])
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same', name='Block8_Conv1') (u8_comb)  # 224, 176
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same', name='Block8_Conv2') (c8)  # 224, 176

    outputs = Conv2D(1, (1, 1), activation='relu', name='model_output') (c8)
    
    # prepare model here
    model = Model(inputs=[inputs], outputs=[outputs])
    return model



def get_hrrr_file(yy, mm, dd, hh):
    # 0, 6, 12, 18
    hhh = [0, 6, 12, 18]
    hidx = int(hh//6)
    return HRRR_DIR + '%04d/hysplit.%04d%02d%02d.%02dz.nc'%(yy, yy, mm, dd, hhh[hidx])


def interp_weights(grid_x_in, grid_y_in, grid_x_out, grid_y_out, d=2):
    xy=np.zeros([grid_x_in.shape[0]*grid_x_in.shape[1],2])
    uv=np.zeros([grid_x_out.shape[0]*grid_x_out.shape[1],2])
    xy[:,0] = grid_x_in.flatten('F')
    xy[:,1] = grid_y_in.flatten('F')
    uv[:,0] = grid_x_out.flatten('F')
    uv[:,1] = grid_y_out.flatten('F')
    tri = Delaunay(xy)
    simplex = tri.find_simplex(uv)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uv - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

def interpolate(values, vtx, wts):
    return np.einsum('nj,nj->n', np.take(values, vtx), wts)

def regmet(data, vtx, wts, out_dims):
    '''
    Read and regrid met fields to given longitudes/latitudes
    '''
    data_out = interpolate(data.flatten('F'),vtx,wts).reshape(out_dims, order='F')
    return data_out


def GaussianPlume(lon, lat, fLon, fLat, uu, vv):
    '''
    Function to generate a Gaussian plume using wind fields.
    '''
    # Grid info
    nX,nY = len(lon),len(lat)
    c     = np.zeros([nY,nX],dtype=float)
    # Windspeed and direction
    wspd = np.sqrt(uu**2.+vv**2.)   # 2D
    wdir = np.arctan2(vv, uu)
    # Parameters and stability class
    x0     = 1e3
    aA, wA = 104., 6.
    aB, wB = 213., 2.
    a      = (wspd - wA)/(wB - wA)*(aB - aA) + aA
    if a < aA:
        a = aA
    if a > aB:
        a = aB
    # Flatten the matrices
    # lon,lat = np.meshgrid(lon,lat)
    out_dim = c.shape
    xx      = (lon - fLon)/120.*1e3
    yy      = (lat - fLat)/120.*1e3
    r       = np.sqrt(xx**2.+yy**2.)
    phi     = np.arctan2(yy,xx)-wdir
    lx      = r*np.cos(phi)
    ly      = r*np.sin(phi)
    sig     = a*(lx/x0)**0.894
    c = 1./(sig*wspd) * np.exp(-0.5 * (ly/sig)**2. )
    # c = c.filled(fill_value=0)
    c[np.where(np.isnan(c))] = 0.
    return c


def cropx(xx):
    # crop input and output to 400 by 400
    dim1 = xx.shape[0]
    dim2 = xx.shape[1]
    dim1res = int((dim1 - 400)/2)
    dim2res = int((dim2 - 400)/2)
    return xx[dim1res:dim1res+400, dim2res:dim2res+400, :]



def transform_func_6h(_xx, _6xx, _yy):
    '''
    xx: (400, 400, 14)
    yy: (400, 400)
    predlist: 'GPR', 'U10M', 'V10M', 'PBLH', 'PRSS', 'SHGT', 'T02M', 'ADS',
              'UWND', 'VWND', 'WWND', 'PRES', 'TEMP', 'AD'
    '''
    
    ###
    #          'GPR', 'U10M', 'V10M', 'PBLH', 'PRSS', 'PRES', 'TEMP'
    SCALERS = [1,     1e1,     1e1,    1e-3,   1e-3,    1,      1]
    BIAS =    [  0,     0,       0,       0,      0,     0,     0]
    
    _yy[np.where(_yy <= 1e-12)] = np.nan
    _yy = np.log(_yy) + 20
    _yy[np.where(np.isnan(_yy))] = 0
    _yy = _yy[:, :, np.newaxis]  # 400, 400, 1
    
    __ = transform_x_6h(_xx, _6xx)
    _xx, _6xx = __[:,:,:6], __[:,:,6:12]
    
    return np.concatenate([_xx, _6xx, _yy], axis=-1)




def transform_x_6h(_xx, _6xx):
    '''
    xx: (400, 400, 14)
    predlist: 'GPR', 'U10M', 'V10M', 'PBLH', 'PRSS', 'SHGT', 'T02M', 'ADS',
              'UWND', 'VWND', 'WWND', 'PRES', 'TEMP', 'AD'
    '''
    
    ###
    #          'GPR', 'U10M', 'V10M', 'PBLH', 'PRSS', 'PRES', 'TEMP'
    SCALERS = [1,     1e1,     1e1,    1e-3,   1e-3,    1,      1]
    BIAS =    [  0,     0,       0,       0,      0,     0,     0]
    
    for i in range(7):
        _xx[:, :, i] = _xx[:, :, i]*SCALERS[i]
    _xx[:, :, 0] = zstandard(_xx[:, :, 0])
    _xx[:, :, 5] = _xx[:, :, 5]/Rd/_xx[:, :, 6]
    _xx = np.delete(_xx, [6], axis=-1) # 400, 400, X
    
    for i in range(7):
        _6xx[:, :, i] = _6xx[:, :, i]*SCALERS[i]
    _6xx[:, :, 0] = zstandard(_6xx[:, :, 0])
    _6xx[:, :, 5] = _6xx[:, :, 5]/Rd/_6xx[:, :, 6]
    _6xx = np.delete(_6xx, [6], axis=-1) # 400, 400, X
    
    return np.concatenate([_xx, _6xx], axis=-1)



def get_raw_x_nc(receptor, resolution):
    trimsize=90
    # clon, clat, tstamp, receptor_lon, receptor_lat = receptor

    '''
    clon: longitude of center point
    clat: latitude of center point
    tstamp: yyyymmddhh
    outlat: latitudes of output domain
    outlon: longitudes of output domain
    '''
    gridinfo, tstamp, (receptor_lon, receptor_lat) = receptor
    grid_x_out_1d, grid_y_out_1d = gridinfo[0], gridinfo[1] # define_domain_grid(gridinfo, resolution)
    clon, clat = np.mean(grid_x_out_1d), np.mean(grid_y_out_1d)
    reftime = datetime(1950, 1, 1, 0, 0, 0, 0)
    distances = (hr3lon_full - clon)**2 + (hr3lat_full - clat)**2
    cind = np.argwhere(distances == np.min(distances))[0]
    cxind = cind[0] # lat
    cyind = cind[1] # lon
    hr3lon = hr3lon_full[cxind-trimsize:cxind+trimsize, cyind-trimsize:cyind+trimsize]
    hr3lat = hr3lat_full[cxind-trimsize:cxind+trimsize, cyind-trimsize:cyind+trimsize]
    
    # Regridding weights
    grid_x_in, grid_y_in = hr3lon, hr3lat
    grid_x_out, grid_y_out = np.meshgrid(grid_x_out_1d, grid_y_out_1d)
    vtx, wts = interp_weights(grid_x_in, grid_y_in, grid_x_out, grid_y_out)
    pI = np.argmin(np.abs(receptor_lat - grid_y_out_1d))
    kI = np.argmin(np.abs(receptor_lon - grid_x_out_1d))
    gp_pI = np.argmin(np.abs(receptor_lat - grid_y_out_1d))
    gp_kI = np.argmin(np.abs(receptor_lon - grid_x_out_1d))
    
    predarr = np.empty((400, 400, len(predlist)))
    predarr_6h = np.empty((400, 400, len(predlist)))
    predarr_12h = np.empty((400, 400, len(predlist)))
    
    dtnow = datetime.strptime(tstamp, '%Y%m%d%H')
    
    _yy, _mm, _dd, _hh = int(dtnow.year), int(dtnow.month), int(dtnow.day), int(dtnow.hour)
    # print(_yy, _mm, _dd, _hh, clon, clat, tstamp, receptor_lat, receptor_lon)
    
    # print("Reading starts here: ")
    startTime = time.time()
    h3rfile = get_hrrr_file(_yy, _mm, _dd, _hh)
    fh = nc.Dataset(h3rfile)
    h3r_data = fh.variables
    deltat = h3r_data['time'][:]
    deltat = np.array([reftime + timedelta(hours=int(s)) for s in deltat])
    deltat = deltat - dtnow
    deltat = np.array([s.seconds for s in deltat])
    tidx = np.argmin(abs(deltat))
    # print("Reading ends.")
    # print("Time taken: ", time.time() - startTime)
    _u10m = h3r_data['u10m'][tidx, cxind-trimsize:cxind+trimsize, cyind-trimsize:cyind+trimsize]
    _u10mr = regmet(_u10m, vtx, wts, grid_x_out.shape)
    predarr[:, :, 1] = _u10mr
    _v10m = h3r_data['v10m'][tidx, cxind-trimsize:cxind+trimsize, cyind-trimsize:cyind+trimsize]
    _v10mr = regmet(_v10m, vtx, wts, grid_x_out.shape)
    predarr[:, :, 2] = _v10mr
    gprs = GaussianPlume(grid_x_out, grid_y_out, receptor_lon, receptor_lat, -_u10mr[gp_pI, gp_kI], -_v10mr[gp_pI, gp_kI])
    predarr[:, :, 0] = gprs
    _pblh = h3r_data['pblh'][tidx, cxind-trimsize:cxind+trimsize, cyind-trimsize:cyind+trimsize]
    predarr[:, :, 3] = regmet(_pblh, vtx, wts, grid_x_out.shape)  
    _prss = h3r_data['prss'][tidx, cxind-trimsize:cxind+trimsize, cyind-trimsize:cyind+trimsize]
    predarr[:, :, 4] = regmet(_prss, vtx, wts, grid_x_out.shape)  
    _pres = h3r_data['p850'][tidx, cxind-trimsize:cxind+trimsize, cyind-trimsize:cyind+trimsize]
    _pres = regmet(_pres, vtx, wts, grid_x_out.shape)  
    predarr[:, :, 5] = _pres
    _temp = h3r_data['t850'][tidx, cxind-trimsize:cxind+trimsize, cyind-trimsize:cyind+trimsize]
    _temp = regmet(_temp, vtx, wts, grid_x_out.shape)  
    predarr[:, :, 6] = _temp
    fh.close()

    dtnow = dtnow - timedelta(hours=6)
    _yy, _mm, _dd, _hh = int(dtnow.year), int(dtnow.month), int(dtnow.day), int(dtnow.hour)
    # print(_yy, _mm, _dd, _hh, clon, clat, tstamp, receptor_lat, receptor_lon)
    # print("Reading starts here: ")
    startTime = time.time()
    h3rfile = get_hrrr_file(_yy, _mm, _dd, _hh)
    fh = nc.Dataset(h3rfile)
    h3r_data = fh.variables
    deltat = h3r_data['time'][:]
    deltat = np.array([reftime + timedelta(hours=int(s)) for s in deltat])
    deltat = deltat - dtnow
    deltat = np.array([s.seconds for s in deltat])
    tidx = np.argmin(abs(deltat))
    # print("Reading ends.")
    # print("Time taken: ", time.time() - startTime)
    _u10m = h3r_data['u10m'][tidx, cxind-trimsize:cxind+trimsize, cyind-trimsize:cyind+trimsize]
    _u10mr = regmet(_u10m, vtx, wts, grid_x_out.shape)
    predarr_6h[:, :, 1] = _u10mr
    _v10m = h3r_data['v10m'][tidx, cxind-trimsize:cxind+trimsize, cyind-trimsize:cyind+trimsize]
    _v10mr = regmet(_v10m, vtx, wts, grid_x_out.shape)
    predarr_6h[:, :, 2] = _v10mr
    gprs = GaussianPlume(grid_x_out, grid_y_out, receptor_lon, receptor_lat, -_u10mr[gp_pI, gp_kI], -_v10mr[gp_pI, gp_kI])
    predarr_6h[:, :, 0] = gprs
    _pblh = h3r_data['pblh'][tidx, cxind-trimsize:cxind+trimsize, cyind-trimsize:cyind+trimsize]
    predarr_6h[:, :, 3] = regmet(_pblh, vtx, wts, grid_x_out.shape)  
    _prss = h3r_data['prss'][tidx, cxind-trimsize:cxind+trimsize, cyind-trimsize:cyind+trimsize]
    predarr_6h[:, :, 4] = regmet(_prss, vtx, wts, grid_x_out.shape)  
    _pres = h3r_data['p850'][tidx, cxind-trimsize:cxind+trimsize, cyind-trimsize:cyind+trimsize]
    _pres = regmet(_pres, vtx, wts, grid_x_out.shape)  
    predarr_6h[:, :, 5] = _pres
    _temp = h3r_data['t850'][tidx, cxind-trimsize:cxind+trimsize, cyind-trimsize:cyind+trimsize]
    _temp = regmet(_temp, vtx, wts, grid_x_out.shape)  
    predarr_6h[:, :, 6] = _temp
    fh.close()
    
    return transform_x_6h(predarr, predarr_6h), receptor





class footprint():
    def __init__(self, receptor, footprint, resolution='nasa'):
        self.gridinfo, self.tstamp, (self.receptor_lon, self.receptor_lat) = receptor
        
        self.footprint = footprint
        if resolution not in ['nasa', 'edf']:
            raise ValueError
        self.resolution = resolution
        self.gridlons, self.gridlats = self.gridinfo[0], self.gridinfo[1] # define_domain_grid(self.gridinfo, self.resolution)

    def visualize(self):

        fig = pplt.figure(refwidth=3)
        gs = pplt.GridSpec(ncols=1, nrows=1)

        ax = fig.subplot(gs[0], proj='cyl')
        im = ax.pcolormesh(self.gridlons, self.gridlats, np.log(self.footprint), extend='both')
        ax.scatter(self.receptor_lon, self.receptor_lat, color='red', marker='x')
        ax.add_feature(cfeature.LAKES)
        ax.coastlines(lw=1.5)
        ax.format(lonlim=(np.min(self.gridlons), np.max(self.gridlons)), 
                   latlim=(np.min(self.gridlats), np.max(self.gridlats)), 
                   labels=True,
                  )
    


class footnet_model():

    def __init__(self, num_var=12, resolution='nasa'):
        self.num_var = num_var
        self.model = build_model(num_var=self.num_var)
        if resolution not in ['nasa', 'edf']:
            raise ValueError
        self.resolution = resolution

    def compile(self, optimizer=Adam(), loss='mse', **kwargs):
        self.model.compile(optimizer=optimizer, loss=loss, **kwargs)

    def info(self):
        self.model.summary()

    def load_weights(self, filename):
        self.model.load_weights(filename)

    def form_inputs(self, receptors, mode='serial', n_jobs=1, verbose=0):
        _inputs = np.empty((len(receptors), 400, 400, self.num_var))
        reference_list = []
        OUTPUT = Parallel(n_jobs=n_jobs, verbose=verbose, backend='multiprocessing')(delayed(get_raw_x_nc)(receptor, self.resolution) for receptor in receptors)
        for idx, value in enumerate(OUTPUT):
            _inputs[idx, :, :, :] = value[0]
            reference_list.append(value[1])
        return _inputs, reference_list

    def get_footprint(self, receptors):
        xx, receptor_list = self.form_inputs(receptors)
        self.inputs = xx
        print(xx.shape)
        raw_foot = self.model.predict(xx)
        # change back to normal unit:
        raw_foot[np.where(raw_foot == 0)] = np.nan
        hh = np.exp(raw_foot - 20)
        hh[np.where(np.isnan(hh))] = 0
        hh = np.squeeze(hh)
        if len(hh.shape) == 2:
            hh = hh[np.newaxis, :, :]
        print(hh.shape)

        _footprint = [footprint(receptor_list[idx], hh[idx], resolution=self.resolution) for idx in range(hh.shape[0])]

        return _footprint



