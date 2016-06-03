# -*- coding: utf-8 -*-
"""
Copyright 2010-2013 University of LiÃ¨ge, Belgium.

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the
use of this software.

Permission is only granted to use this software for non-commercial purposes.
"""

import numpy as np
import math

__author__ = "Begon Jean-Michel <jm.begon@gmail.com>"
__copyright__ = "Copyright 2010-2013 University of LiÃ¨ge, Belgium"
__version__ = '0.1'


class ColorDeconvoluter:
    """
    =================
    ColorDeconvoluter
    =================
    A :class:`ColorDeconvoluter`
        - performs color deconvolution with the instance kernel
        - may learn the kernel parameters

    The resulting "images" components follow the same order as the row
    of the kernel
    """

    def __init__(self):
        self.kernel = None
        self.kernel_inv = None
        self._log255 = math.log(255.0)

    def set_kernel(self, kernel):
        """
        Set the kernel

        Parameters
        ----------
        kernel : 2D numpy.ndarray with dtype=float
            The kernel to use. It must be 3x3, regular and its value should
            be in the range [0, 255].
        """
        # Normalizing the kernel
        norm = np.sqrt((kernel**2).sum(axis=1))
        tmp = np.zeros((3, 3))
        for i in range(3):
            if norm[i] != 0.0:
                tmp[i, :] = kernel[i, :] / norm[i]
        kernel = tmp
        self.kernel = kernel
        self.kernel_inv = np.linalg.inv(kernel)

    def transform(self, np_image):
        """
        Performs a color deconvolution of the given image

        Note
        ----
        A kernel must be set or this method will fail

        Parameters
        ----------
        np_image : 3D numpy.ndarray
            The image to process. The layout must be [row, column, color].
            The number of colors is expected to be 3 (RGB) or 4 (with a mask
            as the last "color")
        """
        M = self.kernel
        D = self.kernel_inv

        assert M is not None, "Kernel not set"

        # Check for alpha mask
        alpha_mask = None
        if np_image.shape[2] == 4:
            alpha_mask = np_image[:, :, 3].astype(np.uint8)
            np_image = np_image[:, :, 0:3]

        height, width, _ = np_image.shape

        # Normalize the image
        float_img = np_image.astype(np.float) + 1  # +1 to avoid log 0
        log_img = 255 * np.log(255.0 / float_img) / self._log255
        # Compute img : img_{i,j,n} = sum_{k} img_{i,j,k} * D_{k,n}
        C = np.einsum('ijk,kn', log_img, D)

        # Normalize
        c = np.exp((1.0 - C / 255.0) * self._log255)
        c = np.clip(c, 0, 255)

        # only compute first stain
        c_255_minus = 255 - c[:, :, 0]
        first_stain = np.tile(M[0, :], (height, width, 1))
        first_stain[:, :, 0] *= c_255_minus
        first_stain[:, :, 1] *= c_255_minus
        first_stain[:, :, 2] *= c_255_minus
        first_stain = 255 - first_stain

        # Add alpha mask
        to_return = first_stain if alpha_mask is None else np.dstack((first_stain, alpha_mask))
        return to_return.astype("uint8")

    def fit(self, tile_stream1, tile_stream2, tile_stream3):
        """
        Fit the :class:`ColorDeconvoluter` (the kernel parameters) to the given
        data.

        Parameters
        ----------
        tile_streami : :class:`TileStream`
            The ith :class:`TileStream` from whose elements to compute the
            kernel parameters. The tile must be RGB or RGBA images.
            The :class:`TileStream` are mapped to a row
            of the kernel in the given order.
        """
        streams = [tile_stream1, tile_stream2, tile_stream3]
        log255 = math.log(255)
        kernel = np.zeros((3, 3))

        for index, stream in enumerate(streams):

            nb_tile = 0
            mean_tile_components = np.zeros((3, 1))

            for tile in stream:

                img = np.asarray(tile.patch, dtype=np.float64)
                mask = None
                #Cechking for mask
                if img.shape[2] == 4:
                    mask = img[:, :, 3] / 255.
                    mask_area = np.sum(mask)
                    img = img[:, :, 0:3]
                else:
                    mask_area = img.shape[0] * img.shape[1]

                #Normalizing
                tmp = (-((255.0*np.log((img+1)/255.0))/log255))

                if mask is not None:
                    #Overlaying #TODO in one pass
                    img[:, :, 0] = tmp[:, :, 0] * mask
                    img[:, :, 1] = tmp[:, :, 1] * mask
                    img[:, :, 2] = tmp[:, :, 2] * mask

                #Computing the mean value for each "color" component
                mean_tile_components += np.einsum("ijk->k", tmp) / mask_area
                nb_tile += 1

            kernel[index, :] = mean_tile_components / nb_tile

        #Setting the kernel
        self.set_kernel(kernel)
