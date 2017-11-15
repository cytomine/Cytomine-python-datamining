# -*- coding: utf-8 -*-
from abc import abstractmethod, ABCMeta

import numpy as np
import cv2
import scipy.ndimage as snd
# from skimage.feature import structure_tensor, structure_tensor_eigvals, hog

from sldc import Logger, StandardOutputLogger, SilentLogger

__author__ = "Rubens Ulysse <urubens@uliege.be>"
__version__ = '0.1'


def split_list_to_dict(lst, splitter='__'):
    if not isinstance(lst, list):
        lst = [lst]

    d = dict()
    for item in lst:
        c, v = item.split(splitter)
        if c not in d:
            d[c] = [v]
        else:
            d[c].append(v)

    return d


class Colorspace:
    def __init__(self):
        self.image = None
        self.feature_extractors = list()

    @abstractmethod
    def build(self, image):
        pass


class ColorspaceRGB(Colorspace):
    def __init__(self, features):
        Colorspace.__init__(self)

        for feature in features:
            if feature in ('rgb', 'RGB'):
                self.feature_extractors.append(FeatureExtractor_RGB)
            elif feature in ('hsv', 'HSV'):
                self.feature_extractors.append(FeatureExtractor_HSV)
            elif feature in ('luv', 'Luv', 'LUV'):
                self.feature_extractors.append(FeatureExtractor_Luv)

    def build(self, image):
        if len(image.shape) == 3:
            self.image = image
        else:
            self.image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return self


class ColorspaceGrayscale(Colorspace):
    def __init__(self, features):
        Colorspace.__init__(self)

        for feature in features:
            if feature in ('gray', 'grayscale'):
                self.feature_extractors.append(FeatureExtractor_Grayscale)
            elif feature in ('gray_norm', 'normalized'):
                self.feature_extractors.append(FeatureExtractor_GrayscaleNormalized)
            elif feature in ('sobel', 'grad1', 'sobel1'):
                self.feature_extractors.append(FeatureExtractor_Sobel1)
            elif feature in ('grad2', 'sobel2'):
                self.feature_extractors.append(FeatureExtractor_Sobel2)
            elif feature in ('gradmagn', 'grad_magn', 'sobel_gradmagn'):
                self.feature_extractors.append(FeatureExtractor_GradientMagnitude)
            # elif feature in ('HoG', 'hog'):
            #     self.feature_extractors.append(FeatureExtractor_HoG)
            # elif feature in ('gaussian_laplace', 'gauss_laplace'):
            #     self.feature_extractors.append(FeatureExtractor_LaplacianOfGaussian)
            # elif feature in ('structure_tensor_eig', 'ste'):
            #     self.feature_extractors.append(FeatureExtractor_EigenvaluesStructureTensor)

    def build(self, image):
        if len(image.shape) == 2:
            self.image = image
        else:
            self.image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return self


class FeaturesExtractor:
    def __init__(self, colorspaces, dtype=np.float16, logger=StandardOutputLogger(Logger.INFO)):
        if colorspaces is None:
            raise ValueError("Colorspace cannot be None")

        self.logger = logger
        self.dtype = dtype
        self.features = None
        self.colorspaces = list()

        colorspaces_dict = split_list_to_dict(colorspaces, splitter='__')
        for colorspace, features in colorspaces_dict.items():
            if colorspace in ('rgb', 'RGB', 'color'):
                self.colorspaces.append(ColorspaceRGB(features))
            elif colorspace in ('gray', 'grayscale', 'L'):
                self.colorspaces.append(ColorspaceGrayscale(features))

    def build(self, image):
        self.logger.i("[FEATURE EXTRACTOR] Start building features extractor...")

        [c.build(image) for c in self.colorspaces]

        self.features = np.dstack([f(logger=self.logger).extract(c.image)
                                   for c in self.colorspaces for f in c.feature_extractors])

        self.logger.i("[FEATURE_EXTRACTOR] Done.")
        return self

    @property
    def feature_image(self):
        return np.asarray(self.features, dtype=self.dtype)

    @classmethod
    def create_from_parameters(cls, parameters, logger=StandardOutputLogger(Logger.INFO)):
        kwargs = parameters if isinstance(parameters, dict) else vars(parameters)
        kwargs.update({'logger': logger})
        return cls(**kwargs)


class FeatureExtractor(object):
    __metaclass__ = ABCMeta

    def __init__(self, name=None, logger=SilentLogger()):
        self.logger = logger
        self.name = name

    @abstractmethod
    def extract(self, image):
        pass


class FeatureExtractor1(FeatureExtractor):
    def extract(self, image):
        if not len(image.shape) == 2:
            raise ValueError("Cannot extract a grayscale feature on this image")


class FeatureExtractor3(FeatureExtractor):
    def extract(self, image):
        if not len(image.shape) == 3:
            raise ValueError("Cannot extract a color feature on this image")


class FeatureExtractor_RGB(FeatureExtractor3):
    def extract(self, image):
        self.logger.i("[FEATURE EXTRACTOR] Performing RGB extraction.")
        super(FeatureExtractor_RGB, self).extract(image)
        return image


class FeatureExtractor_HSV(FeatureExtractor3):
    def extract(self, image):
        self.logger.i("[FEATURE EXTRACTOR] Performing HSV extraction.")
        super(FeatureExtractor_HSV, self).extract(image)
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)


class FeatureExtractor_Luv(FeatureExtractor3):
    def extract(self, image):
        self.logger.i("[FEATURE EXTRACTOR] Performing Luv extraction.")
        super(FeatureExtractor_Luv, self).extract(image)
        return cv2.cvtColor(image, cv2.COLOR_RGB2Luv)


class FeatureExtractor_Grayscale(FeatureExtractor1):
    def extract(self, image):
        self.logger.i("[FEATURE EXTRACTOR] Performing grayscale extraction.")
        super(FeatureExtractor_Grayscale, self).extract(image)
        return image


class FeatureExtractor_GrayscaleNormalized(FeatureExtractor1):
    def extract(self, image):
        self.logger.i("[FEATURE EXTRACTOR] Performing normalized grayscale extraction.")
        super(FeatureExtractor_GrayscaleNormalized, self).extract(image)
        return cv2.equalizeHist(image.astype(np.uint8))


class FeatureExtractor_Sobel1(FeatureExtractor1):
    def extract(self, image):
        self.logger.i("[FEATURE EXTRACTOR] Performing Sobel at 1st order extraction.")
        super(FeatureExtractor_Sobel1, self).extract(image)
        sob_x = cv2.Sobel(image, cv2.CV_32F, 1, 0)
        sob_y = cv2.Sobel(image, cv2.CV_32F, 0, 1)
        alpha = 1  # 0.25
        return np.dstack((cv2.convertScaleAbs(sob_x, alpha=alpha),
                          cv2.convertScaleAbs(sob_y, alpha=alpha)))


class FeatureExtractor_Sobel2(FeatureExtractor1):
    def extract(self, image):
        self.logger.i("[FEATURE EXTRACTOR] Performing Sobel at 2nd order extraction.")
        super(FeatureExtractor_Sobel2, self).extract(image)
        sob_x = cv2.Sobel(image, cv2.CV_32F, 2, 0)
        sob_y = cv2.Sobel(image, cv2.CV_32F, 0, 2)
        alpha = 1  # 0.25
        return np.dstack((cv2.convertScaleAbs(sob_x, alpha=alpha),
                          cv2.convertScaleAbs(sob_y, alpha=alpha)))


class FeatureExtractor_GradientMagnitude(FeatureExtractor1):
    def extract(self, image):
        self.logger.i("[FEATURE EXTRACTOR] Performing gradient magnitude extraction.")
        super(FeatureExtractor_GradientMagnitude, self).extract(image)
        sob_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, 3)
        sob_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, 3)
        magnitude = cv2.magnitude(sob_x, sob_y)
        alpha = 1  # 0.9
        return cv2.convertScaleAbs(magnitude, alpha=alpha)
        # gray = gray.astype(np.float32)
        # scales = [0.8 * (2 ** n) for n in range(3)]
        # return np.dstack(tuple([snd.filters.gaussian_gradient_magnitude(gray, sigma=s) for s in scales]))


# class FeatureExtractor_HoG(FeatureExtractor1):
#     def extract(self, image):
#         self.logger.i("[FEATURE EXTRACTOR] Performing histogram of gradient (HOG) extraction.")
#         super(FeatureExtractor_HoG, self).extract(image)
#         _, h = hog(image, visualise=True)
#         return h
#
#
# class FeatureExtractor_LaplacianOfGaussian(FeatureExtractor1):
#     def __init__(self, logger=SilentLogger()):
#         super(FeatureExtractor_LaplacianOfGaussian, self).__init__(logger=logger)
#         self.scales = [0.8 * (2 ** n) for n in range(3)]
#
#     def extract(self, image):
#         self.logger.i("[FEATURE EXTRACTOR] Performing Laplacian of Gaussian extraction.")
#         super(FeatureExtractor_LaplacianOfGaussian, self).extract(image)
#         image = image.astype(np.float32)
#         return np.dstack(tuple([snd.filters.gaussian_laplace(image, sigma=s) for s in self.scales]))
#
#
# class FeatureExtractor_EigenvaluesStructureTensor(FeatureExtractor1):
#     def __init__(self, logger=SilentLogger()):
#         super(FeatureExtractor_EigenvaluesStructureTensor, self).__init__(logger=logger)
#         self.scales = [0.8 * (2 ** n) for n in range(3)]
#
#     def extract(self, image):
#         self.logger.i("[FEATURE EXTRACTOR] Performing eigenvalue of structure tensor extraction.")
#         super(FeatureExtractor_EigenvaluesStructureTensor, self).extract(image)
#         image = image.astype(np.float32)
#         image /= image.max()
#         eigs = [structure_tensor_eigvals(*structure_tensor(image, sigma=s)) for s in self.scales]
#         return np.dstack(tuple([a for eig in eigs for a in eig]))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sys
    import os
    import numpy as np

    def flip(m, axis):
        indexer = [slice(None)] * m.ndim
        indexer[axis] = slice(None, None, -1)
        return m[tuple(indexer)]

    filename = sys.argv[1]
    if not os.path.exists(filename):
        raise IOError("File " + filename + " does not exist !")
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    img = img[250:,250:,...]

    colorspaces = ['RGB__rgb', 'RGB__luv', 'RGB__hsv', 'L__normalized',
                   'L__sobel1', 'L__sobel_gradmagn', 'L__sobel2']

    fe = FeaturesExtractor(colorspaces, logger=StandardOutputLogger(Logger.DEBUG))
    fe.build(img)

    titles = ['Red', 'Green', 'Blue', 'L*', 'u*', 'v*', 'Hue', 'Saturation', 'Value',
              'Eq. grayscale', '1st-Sobel x', '1st-Sobel y', 'Sobel grad. magn.',
              '2st-Sobel x', '2st-Sobel y']

    n_rows = 4
    n_cols = 4

    fig = plt.figure()
    a = fig.add_subplot(n_rows, n_cols, 1)
    plt.imshow(flip(img, 2))
    a.set_title('Original RGB image')
    a.set_axis_off()

    for i in range(fe.features.shape[2]):
        a = fig.add_subplot(n_rows, n_cols, i+2)
        plt.imshow(fe.features[:, :, i], cmap='viridis')
        a.set_title(titles[i])
        a.set_axis_off()

    plt.show()
