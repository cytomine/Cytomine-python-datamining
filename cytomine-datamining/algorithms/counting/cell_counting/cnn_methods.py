# -*- coding: utf-8 -*-

import copy
import itertools
import types

import cv2
import numpy as np
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sldc import DefaultTileBuilder, Image, TileTopologyIterator

from cell_counting.base_method import BaseMethod
from cell_counting.cnn_architectures import FCRN_A, FCRN_B, sgd_compile
from cell_counting.subwindows import mk_subwindows
from cell_counting.utils import open_image

__author__ = "Ulysse Rubens <urubens@uliege.be>"
__version__ = "0.1"


def lr_scheduler(epoch):
    step = 24
    num = epoch % step
    if num == 0 and epoch != 0:
        lr_scheduler.lrate = lr_scheduler.lrate - lr_scheduler.lrate / 2.

    print('Learning rate for epoch {} is {}.'.format(epoch + 1, lr_scheduler.lrate))
    return np.float(lr_scheduler.lrate)


class FCRN(BaseMethod):
    def __init__(self, build_fn=None, callbacks=None, **sk_params):
        super(FCRN, self).__init__(build_fn, **sk_params)
        self.callbacks = callbacks
        self.__model = None

    def check_params(self, params):
        # Just for compatibility with Keras
        pass

    def get_params(self, **params):
        res = super(FCRN, self).get_params(**params)
        res.update({'callbacks': self.callbacks})
        return res

    def fit(self, x, y, **kwargs):
        self.sk_params['callbacks'] = self.callbacks
        self.sk_params['verbose'] = 2

        lr_scheduler.lrate = self.sk_params['learning_rate']

        if self.build_fn is None:
            self.__model = self.build_fcrn(**self.filter_sk_params(self.build_fcrn))
        elif not isinstance(self.build_fn, types.FunctionType) and not isinstance(self.build_fn, types.MethodType):
            self.__model = self.build_fn(**self.filter_sk_params(self.build_fn.__call__))
        else:
            self.__model = self.build_fn(**self.filter_sk_params(self.build_fn))

        fit_args = copy.deepcopy(self.filter_sk_params(Sequential.fit))
        fit_args.update(kwargs)
        del fit_args['batch_size']

        # Make subwindows for training
        _x, _y = mk_subwindows(x, y, None, flatten=False, **self.filter_sk_params(mk_subwindows))
        _y = np.expand_dims(_y, axis=4)

        # Data generator
        seed = np.random.randint(2 ** 32 - 1)
        exceptions_y_datagen = ['featurewise_center',
                                'samplewise_center',
                                'featurewise_std_normalization',
                                'samplewise_std_normalization']
        X_datagen = ImageDataGenerator(**self.filter_sk_params(ImageDataGenerator.__init__))
        y_datagen = ImageDataGenerator(**self.filter_sk_params(ImageDataGenerator.__init__,
                                                               exceptions=exceptions_y_datagen))

        X_datagen.fit(_x, augment=True, seed=seed)
        y_datagen.fit(_y, augment=True, seed=seed)
        X_gen = X_datagen.flow(_x, None, batch_size=self.sk_params['batch_size'], seed=seed)
        y_gen = y_datagen.flow(_y, None, batch_size=self.sk_params['batch_size'], seed=seed)
        datagen = itertools.izip(X_gen, y_gen)

        self.__history = self.__model.fit_generator(
            datagen,
            steps_per_epoch=_x.shape[0] / self.sk_params['batch_size'],
            **fit_args)

        return self.__history

    def predict(self, X, **kwargs):
        kwargs = self.filter_sk_params(Sequential.predict, kwargs)

        div = 8
        max_width = 512
        max_height = 512
        overlap = 30
        dtb = DefaultTileBuilder()
        _X = []
        for x in X:
            x = open_image(x, flag='RGB')  # TODO: get mask
            _x = np.zeros((x.shape[0], x.shape[1]))
            count = np.zeros((x.shape[0], x.shape[1]))

            tile_iterator = TiledImage(x).tile_iterator(dtb, max_width=max_width,
                                                        max_height=max_height, overlap=overlap)
            for tile in tile_iterator:
                height = tile.width
                top = tile.offset_x
                bottom = top + height

                width = tile.height
                left = tile.offset_y
                right = left + width

                __x = np.expand_dims(cv2.copyMakeBorder(x[top:bottom, left:right],
                                                        0, ((height // div * div + div) - height),
                                                        0, ((width // div * div + div) - width),
                                                        borderType=cv2.BORDER_DEFAULT),
                                     axis=0)
                _x[top:bottom, left:right] += np.squeeze(self.model.predict(__x, **kwargs))[:height, :width]
                count[top:bottom, left:right] += 1
                _x[count > 1] = _x[count > 1] / count[count > 1]
                # TODO: remove positions outside mask
            _X.append(_x)

        return np.squeeze(_X)

    @property
    def history(self):
        return self.__history

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, value):
        self.__model = value

    def save(self, filename):
        return self.model.save(filename)

    @staticmethod
    def build_fcrn(architecture='FCRN-test', regularizer=None, initializer='orthogonal',
                   batch_normalization=False, learning_rate=0.01, momentum=0.9, decay=0.,
                   nesterov=False, input_shape=(None, None, 3)):
        if architecture == 'FCRN-A':
            arch = FCRN_A(input_shape, regularizer, initializer,
                          batch_normalization)
        elif architecture == 'FCRN-B':
            arch = FCRN_B(input_shape, regularizer, initializer,
                          batch_normalization)
        else:
            raise ValueError('Unknown method.')

        model = sgd_compile(arch, learning_rate, momentum, decay, nesterov)
        model.summary()
        return model


class TiledImage(Image):
    def __init__(self, np_array):
        self.np_array = np.array(np_array)

    @property
    def width(self):
        return self.np_array.shape[0]

    @property
    def height(self):
        return self.np_array.shape[1]

    @property
    def np_image(self):
        return self.np_array

    @property
    def channels(self):
        return self.np_array.shape[2]

    def tile_iterator(self, builder, max_width=1024, max_height=1024, overlap=0):
        """Build and return a tile iterator that iterates over the image

        Parameters
        ----------
        builder: TileBuilder
            The builder to user for actually constructing the tiles while iterating over the image
        max_width: int (optional, default: 1024)
            The maximum width of the tiles to build
        max_height: int (optional, default: 1024)
            The maximum height of the tiles to build
        overlap: int (optional, default: 0)
            The overlapping between tiles

        Returns
        -------
        iterator: TileTopologyIterator
            An iterator that iterates over a tile topology of the image
        """
        topology = self.tile_topology(builder, max_width=max_width, max_height=max_height, overlap=overlap)
        return TileTopologyIterator(builder, topology)
