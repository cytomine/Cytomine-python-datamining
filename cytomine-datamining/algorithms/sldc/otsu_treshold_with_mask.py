import numpy as np
import cv2

#image is a 8-bits grayscale cv2 image. ROI must be brighter, if not replace cv2.THRESH_BINARY by cv2.THRESH_BINARY_INV

def otsu_threshold_with_mask(image, mask, mode):

    mask_indices = np.nonzero(mask)

    temp = np.array([image[mask_indices]])

    temp = temp[temp < 120]

    otsu_threshold,_ = cv2.threshold( temp, 128, 255, cv2.THRESH_OTSU | mode)

    _, image = cv2.threshold(image, otsu_threshold, 255, cv2.THRESH_BINARY_INV)

    return otsu_threshold, image
