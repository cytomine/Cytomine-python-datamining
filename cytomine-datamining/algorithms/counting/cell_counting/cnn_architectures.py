# -*- coding: utf-8 -*-
from keras.engine import Layer
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import SGD
import numpy as np

__author__ = "Ulysse Rubens <urubens@uliege.be>"
__version__ = "0.1"


def sgd_compile(model, lr=1e-2, momentum=0.9, decay=0., nesterov=False, metrics=None):
    """
    Compile a Keras model using Stochastic Gradient Descent (SGD) as optimizer, using
    the mean squared error as loss function.
    
    Parameters
    ----------
    model : keras.models.Model
        Keras model to compile
    lr : float, optional
        Initial learning rate of SGD (default: 1e-2).
    momentum : float, optional
        Momentum of SGD (default: 0.9)
    decay : float, optional
        Decay of SGD (default: 0.0)
    nesterov : bool, optional
        Apply Nesterov momentum or not (default: False)
    metrics : list of str, optional
        A list of addtional metrics (default: None)

    Returns
    -------
    model : keras.models.Model
        The compiled model
    """
    model.compile(optimizer=SGD(lr=lr,
                                momentum=momentum,
                                decay=decay,
                                nesterov=nesterov),
                  loss='mean_squared_error',
                  metrics=['accuracy', 'mae'] + [] if metrics is None else metrics)

    return model


def FCRN_A(input_shape=(None, None, 3), regularizer=None, initializer='orthogonal', bn=True):
    """
    Build the Keras architecture of the Fully Convolutional Regression Network (FCRN), 
    first version, for the provided input shape. 
    Remark: to obtain the same dimension in output, input width and height have 
    to be multiple of 8 (can be achieved by padding input images).
    
    Parameters
    ----------
    input_shape : tuple (width, height, depth), optional
        Shape of the input images. Providing None for width or height allows to use 
        images of any dimension with the model. (default: (None, None, 3))
    regularizer : str or None, optional
        A valid Keras regularizer name, which allows to apply penalties on 
        layer parameters or layer activity during optimization. These penalties 
        are incorporated in the loss function that the network optimizes (default: None)
    initializer: str, optional
        A valid Keras initializer name, which specifies the way to set the 
        initial random weights (default: 'orthogonal')
    bn: bool, optional
        If batch normalization must be used or not. It ormalizes the activations 
        of the previous layer at each batch, i.e. applies a transformation that maintains 
        the mean activation close to 0 and the activation standard deviation 
        close to 1. (default: True)

    Returns
    -------
    model : keras.models.Model
        The sequential FCRN-A model
        
    References
    ----------
    .. [1] W. Xie, J. A. Noble, and A. Zisserman. “Microscopy Cell Counting with 
           Fully Convolutional Regression Networks”. In: MICCAI 1st Workshop on 
           Deep Learning in Medical Image Analysis. 2015.
    """
    model = Sequential()

    # Layer 1 + 2 - CONV [32 x 3 x 3, pad auto] + (BN) + RELU + POOL [2 x 2]
    # (w, h, d) -> (w/2, h/2, 32)
    model.add(Conv2D(32, (3, 3), padding='same', use_bias=False, kernel_initializer=initializer,
                     kernel_regularizer=regularizer, input_shape=input_shape, name="1-Conv"))
    if bn:
        model.add(BatchNormalization(name="1-BatchNorm"))
    model.add(Activation(activation='relu', name="1-ReLU"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="2-MaxPooling"))

    # Layer 3 + 4 - CONV [64 x 3 x 3, pad auto] + (BN) + RELU + POOL [2 x 2]
    # (w/2, h/2, 32) -> (w/4, h/4, 64)
    model.add(Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_initializer=initializer,
                     kernel_regularizer=regularizer, name="3-Conv"))
    if bn:
        model.add(BatchNormalization(name="3-BatchNorm"))
    model.add(Activation(activation='relu', name="3-ReLU"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="4-MaxPooling"))

    # Layer 5 + 6 - CONV [128 x 3 x 3, pad auto] + (BN) + RELU + POOL [2 x 2]
    # (w/4, h/4, 64) -> (w/8, h/8, 128)
    model.add(Conv2D(128, (3, 3), padding='same', use_bias=False, kernel_initializer=initializer,
                     kernel_regularizer=regularizer, name="5-Conv"))
    if bn:
        model.add(BatchNormalization(name="5-BatchNorm"))
    model.add(Activation(activation='relu', name="5-ReLU"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="6-MaxPooling"))

    # Layer 7 - FC [512 x 3 x 3, pad auto] + (BN) + RELU
    # (w/8, h/8, 128) -> (w/8, h/8, 512)
    model.add(Conv2D(512, (3, 3), padding='same', use_bias=False, kernel_initializer=initializer,
                     kernel_regularizer=regularizer, name="7-Conv"))
    if bn:
        model.add(BatchNormalization(name="7-BatchNorm"))
    model.add(Activation(activation='relu', name="7-ReLU"))

    # Layer 8 + 9 - UPSAMPLING [2 x 2] + CONV [128 x 3 x 3] + (BN) + RELU
    # (w/8, h/8, 512) -> (w/4, h/4, 128)
    model.add(UpSampling2D(size=(2, 2), name="8-UpSampling"))
    model.add(Conv2D(128, (3, 3), padding='same', use_bias=False, kernel_initializer=initializer,
                     kernel_regularizer=regularizer, name="9-Conv"))
    if bn:
        model.add(BatchNormalization(name="9-BatchNorm"))
    model.add(Activation(activation='relu', name="9-ReLU"))

    # Layer 10 + 11 - UPSAMPLING [2 x 2] + CONV [64 x 3 x 3] + (BN) + RELU
    # (w/4, h/4, 128) -> (w/2, h/2, 64)
    model.add(UpSampling2D(size=(2, 2), name="10-UpSampling"))
    model.add(Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_initializer=initializer,
                     kernel_regularizer=regularizer, name="11-Conv"))
    if bn:
        model.add(BatchNormalization(name="11-BatchNorm"))
    model.add(Activation(activation='relu', name="11-ReLU"))

    # Layer 12 + 13 - UPSAMPLING [2 x 2] + CONV [32 x 3 x 3] + (BN) + RELU
    # (w/2, h/2, 64) -> (w, h, 32)
    model.add(UpSampling2D(size=(2, 2), name="12-UpSampling"))
    model.add(Conv2D(32, (3, 3), padding='same', use_bias=False, kernel_initializer=initializer,
                     kernel_regularizer=regularizer, name="13-Conv"))
    if bn:
        model.add(BatchNormalization(name="13-BatchNorm"))
    model.add(Activation(activation='relu', name="13-ReLU"))

    # Layer 14 - CONV [1 x 1 x 1]
    # (w, h, 32) -> (w, h, 1)
    model.add(Conv2D(1, (1, 1), padding='same', activation='linear',
                     use_bias=False, kernel_initializer=initializer,
                     kernel_regularizer=regularizer, name="14-Conv"))

    return model


def FCRN_B(input_shape, regularizer=None, initializer='orthogonal', bn=False):
    """
    Build the Keras architecture of the Fully Convolutional Regression Network (FCRN), 
    second version, for the provided input shape. 
    Remark: to obtain the same dimension in output, input width and height have 
    to be multiple of 4 (can be achieved by padding input images).
    
    Parameters
    ----------
    input_shape : tuple (width, height, depth), optional
        Shape of the input images. Providing None for width or height allows to use 
        images of any dimension with the model. (default: (None, None, 3))
    regularizer : str or None, optional
        A valid Keras regularizer name, which allows to apply penalties on 
        layer parameters or layer activity during optimization. These penalties 
        are incorporated in the loss function that the network optimizes (default: None)
    initializer: str, optional
        A valid Keras initializer name, which specifies the way to set the 
        initial random weights (default: 'orthogonal')
    bn: bool, optional
        If batch normalization must be used or not. It ormalizes the activations 
        of the previous layer at each batch, i.e. applies a transformation that maintains 
        the mean activation close to 0 and the activation standard deviation 
        close to 1. (default: True)

    Returns
    -------
    model : keras.models.Model
        The sequential FCRN-B model
        
    References
    ----------
    .. [1] W. Xie, J. A. Noble, and A. Zisserman. “Microscopy Cell Counting with 
           Fully Convolutional Regression Networks”. In: MICCAI 1st Workshop on 
           Deep Learning in Medical Image Analysis. 2015.
    """

    model = Sequential()

    # Layer 1 - CONV [32 x 3 x 3, pad auto] + (BN) + RELU
    # (w, h, d) -> (w, h, 32)
    model.add(Conv2D(32, (3, 3), padding='same', use_bias=False, kernel_initializer=initializer,
                     kernel_regularizer=regularizer, input_shape=input_shape, name="1-Conv"))
    if bn:
        model.add(BatchNormalization(name="1-BatchNorm"))
    model.add(Activation(activation='relu', name="1-ReLU"))

    # Layer 2 - CONV [64 x 3 x 3, pad auto] + (BN) + RELU
    # (w, h, 32) -> (w, h, 64)
    model.add(Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_initializer=initializer,
                     kernel_regularizer=regularizer, name="2-Conv"))
    if bn:
        model.add(BatchNormalization(name="2-BatchNorm"))
    model.add(Activation(activation='relu', name="2-ReLU"))

    # Layer 3 - POOL [2 x 2]
    # (w, h, 64) -> (w/2, h/2, 64)
    model.add(MaxPooling2D(pool_size=(2, 2), name="2-MaxPooling"))

    # Layer 4 - CONV [128 x 3 x 3, pad auto] + (BN) + RELU
    # (w/2, h/2, 64) -> (w/2, h/2, 128)
    model.add(Conv2D(128, (3, 3), padding='same', use_bias=False, kernel_initializer=initializer,
                     kernel_regularizer=regularizer, name="3-Conv"))
    if bn:
        model.add(BatchNormalization(name="3-BatchNorm"))
    model.add(Activation(activation='relu', name="3-ReLU"))

    # Layer 5 - CONV [256 x 5 x 5, pad auto] + (BN) + RELU
    # (w/2, h/2, 128) -> (w/2, h/2, 256)
    model.add(Conv2D(256, (5, 5), padding='same', use_bias=False, kernel_initializer=initializer,
                     kernel_regularizer=regularizer, name="4-Conv"))
    if bn:
        model.add(BatchNormalization(name="4-BatchNorm"))
    model.add(Activation(activation='relu', name="4-ReLU"))

    # Layer 6 - POOL [2 x 2]
    # (w/2, h/2, 256) -> (w/4, h/4, 256)
    model.add(MaxPooling2D(pool_size=(2, 2), name="4-MaxPooling"))

    # Layer 7 - FC [256 x 3 x 3, pad auto] + (BN) + RELU
    # (w/4, h/4, 256) -> (w/4, h/4, 256)
    model.add(Conv2D(256, (3, 3), padding='same', use_bias=False, kernel_initializer=initializer,
                     kernel_regularizer=regularizer, name="5-Conv"))
    if bn:
        model.add(BatchNormalization(name="5-BatchNorm"))
    model.add(Activation(activation='relu', name="5-ReLU"))

    # Layer 8 - UPSAMPLING [2 x 2]
    # (w/4, h/4, 256) -> (w/2, h/2, 256)
    model.add(UpSampling2D(size=(2, 2), name="6-UpSampling"))

    # Layer 9 - CONV [256 x 5 x 5, pad auto] + (BN) + RELU
    # (w/2, h/2, 256) -> (w/2, h/2, 256)
    model.add(Conv2D(256, (5, 5), padding='same', use_bias=False, kernel_initializer=initializer,
                     kernel_regularizer=regularizer, name="6-Conv"))
    if bn:
        model.add(BatchNormalization(name="6-BatchNorm"))
    model.add(Activation(activation='relu', name="6-ReLU"))

    # Layer 10 - UPSAMPING [2 x 2]
    # (w/2, h/2, 256) -> (w, h, 256)
    model.add(UpSampling2D(size=(2, 2), name="7-UpSampling"))

    # Layer 11 - CONV [1 x 1 x 1, pad auto]
    # (w, h, 256) -> (w, h, 1)
    model.add(Conv2D(1, (1, 1), padding='same', activation='linear', use_bias=False, kernel_initializer=initializer,
                     kernel_regularizer=regularizer, name="7-Conv"))

    return model


#######################################################################################################################
# SC-CNN

class ParameterEstimation(Layer):
    def __init__(self, n_rows, n_cols, M, **kwargs):
        self._M = M
        self._n_rows = n_rows
        self._n_cols = n_cols
        super(ParameterEstimation, self).__init__(**kwargs)

    def call(self, inputs):
        # TODO
        pass

    def compute_output_shape(self, input_shape):
        return input_shape[:3] + (3 * self._M,)


class SpatiallyConstrained(Layer):
    def __init__(self, radius, alpha, **kwargs):
        super(SpatiallyConstrained, self).__init__(**kwargs)

    def call(self, inputs):
        # TODO
        pass

    def compute_output_shape(self, input_shape):
        return input_shape[0], 1, 1


def SCCNN(input_shape, radius, alpha, M=1, regularizer=None, initializer='orthogonal', bn=False, dropout=None):
    model = Sequential()

    # Layer 1 - CONV [36 x 4 x 4, pad auto] + (BN) + RELU
    # (w, h, d) -> (w, h, 36)
    model.add(Conv2D(36, (4, 4), padding='same', use_bias=False, kernel_initializer=initializer,
                     kernel_regularizer=regularizer, input_shape=input_shape, name="1-Conv"))
    if bn:
        model.add(BatchNormalization(name="1-BatchNorm"))
    model.add(Activation(activation='relu', name="1-ReLU"))

    # Layer 2 - POOL [2 x 2]
    # (w, h, 36) -> (w/2, h/2, 36)
    model.add(MaxPooling2D(pool_size=(2, 2), name="1-MaxPooling"))

    # Layer 3 - CONV [48 x 3 x 3, pad auto] + (BN) + RELU
    # (w/2, h/2, 36) -> (w/2, h/2, 48)
    model.add(Conv2D(48, (3, 3), padding='same', use_bias=False, kernel_initializer=initializer,
                     kernel_regularizer=regularizer, name="2-Conv"))
    if bn:
        model.add(BatchNormalization(name="2-BatchNorm"))
    model.add(Activation(activation='relu', name="2-ReLU"))

    # Layer 4 - POOL [2 x 2]
    # (w/2, h/2, 48) -> (w/4, h/4, 48)
    model.add(MaxPooling2D(pool_size=(2, 2), name="2-MaxPooling"))

    # Layer 5 - FC [512 x w/4 x h/4] + (BN) + RELU + (DROPOUT)
    # (w/4, h/4, 48) -> (1, 1024)
    w, h = np.round(input_shape[0] / 4), np.round(input_shape[1] / 4)
    model.add(Conv2D(1024, (w, h), padding='valid', use_bias=False, kernel_initializer=initializer,
                     kernel_regularizer=regularizer, name="3-FC"))
    if bn:
        model.add(BatchNormalization(name="3-BatchNorm"))
    model.add(Activation(activation='relu', name="3-ReLU"))
    if dropout:
        model.add(Dropout(rate=dropout, name="3-Dropout"))

    # Layer 6 - FC [512 x 1 x 1] + (BN) + RELU
    # (1, 1024) -> (1, 512)
    model.add(Conv2D(512, (1, 1), padding='same', use_bias=False, kernel_initializer=initializer,
                     kernel_regularizer=regularizer, name="4-FC"))
    if bn:
        model.add(BatchNormalization(name="4-BatchNorm"))
    model.add(Activation(activation='relu', name="4-ReLU"))
    if dropout:
        model.add(Dropout(rate=dropout, name="4-Dropout"))

    # Layer 7 - S1 [3M x 1 x 1]
    # (1, 512) -> (1, 3M)
    model.add(Conv2D(3 * M, (1, 1), padding='same', use_bias=False, kernel_initializer=initializer,
                     kernel_regularizer=regularizer, name="5-S1-Conv"))
    model.add(Activation(activation='sigmoid', name="5-S1-Sigmoid"))
    model.add(ParameterEstimation(n_rows=input_shape[0], n_cols=input_shape[1], M=M,
                                  name="5-S1-ParameterEstimation"))
    model.add(Flatten(name="5-S1-Flatten"))

    # Layer 8 - S2
    # (1, 3M) -> (w', h')
    model.add(SpatiallyConstrained(radius=radius, alpha=alpha, name="6-S2-SpatiallyConstrained"))

    return model
