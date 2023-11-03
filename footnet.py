from tensorflow.keras.optimizers import Adam
import numpy as np
from keras import backend as K
from keras.layers import Input, concatenate
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from tensorflow.keras.utils import Sequence
from keras.models import Model

RES = 16
NUNIT = 1024
DIM1 = 400   # 
DIM2 = 400
SHAPE1 = int((DIM1 + 15)//RES*RES  )
SHAPE2 = int((DIM2 + 15)//RES*RES  )
LATNTSHAPE = int(SHAPE2//16*SHAPE1//16)
Rd = 2.87053  # hPa·K-1·m3·kg–1
ggg = 9.80665 # m/s*s


def zstandard(arr):
    _mu = np.nanmean(arr)
    _std = np.nanstd(arr)
    return (arr - _mu)/_std

def r2_keras(y_t, y_p):
    SS_res =  K.sum(K.square(y_t - y_p)) 
    SS_tot = K.sum(K.square(y_t - K.mean(y_t))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


def transform_func_6h(_xx, _6xx, _yy):
    '''
    xx: (400, 400, 14)
    yy: (400, 400)
    predlist: 'GPR', 'U10M', 'V10M', 'PBLH', 'PRSS', 'SHGT', 'T02M', 'ADS',
              'UWND', 'VWND', 'WWND', 'PRES', 'TEMP', 'AD'
    '''
    
    #          'GPR', 'U10M', 'V10M', 'PBLH', 'PRSS', 'PRES', 'TEMP'
    SCALERS = [1,     1e1,     1e1,    1e-3,   1e-3,    1,      1]
    BIAS =    [  0,     0,       0,       0,      0,     0,     0]
    
    _yy[np.where(_yy <= 1e-12)] = np.nan
    _yy = np.log(_yy) + 20
    _yy[np.where(np.isnan(_yy))] = 0
    _yy = _yy[:, :, np.newaxis]  # 400, 400, 1
    
    for i in range(7):
        _xx[:, :, i] = _xx[:, :, i]*SCALERS[i]
    _xx[:, :, 0] = zstandard(_xx[:, :, 0])
    _xx[:, :, 5] = _xx[:, :, 5]/Rd/_xx[:, :, 6] # calculate air density
    _xx = np.delete(_xx, [6], axis=-1) 
    
    for i in range(7):
        _6xx[:, :, i] = _6xx[:, :, i]*SCALERS[i]
    _6xx[:, :, 0] = zstandard(_6xx[:, :, 0])
    _6xx[:, :, 5] = _6xx[:, :, 5]/Rd/_6xx[:, :, 6] # calculate air density
    _6xx = np.delete(_6xx, [6], axis=-1) 

    
    return np.concatenate([_xx, _6xx, _yy], axis=-1)

class data_generator_6h( Sequence ) :
    def __init__(self, fnames, batch_size=12):
        self.fnames = fnames
        self.batch_size = batch_size

    def __len__(self) :
        return np.ceil( len(self.fnames) / float(self.batch_size)).astype(int)

    def __getitem__(self, idx) :
        batch_f = self.fnames[ idx*self.batch_size : idx*self.batch_size + self.batch_size ]
    
        tempxy = np.stack([ transform_func_6h(np.load(s, allow_pickle=True)['pred'], np.load(s, allow_pickle=True)['_6hpred'], np.load(s, allow_pickle=True)['obs']) for s in batch_f])
        tempx, tempy = tempxy[:, :, :, :-1], tempxy[:, :, :, -1]
        tempy = tempy[:, :, :, np.newaxis]
        return tempx, tempy
    
def build_model(num_var=12):
    inputs = Input( ( 400, 400, num_var ), name='model_input')

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
    opt = Adam(lr=1e-5)
    model.compile(optimizer=opt, loss='mse', metrics=[r2_keras])
    return model

