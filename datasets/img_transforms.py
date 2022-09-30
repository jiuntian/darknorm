import cv2
import numpy as np


def gamma_intensity_correction(img, gamma):
    """
    :param img: the img of input
    :param gamma:
    :return: a new img
    """
    invGamma = 1.0 / gamma
    LU_table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    gamma_img = cv2.LUT(img, LU_table)
    return gamma_img


def histogram_normalization(img):
    """
    :param img: input image
    """
    r, g, b = cv2.split(img)

    output_r = cv2.equalizeHist(r)
    output_g = cv2.equalizeHist(g)
    output_b = cv2.equalizeHist(b)

    equ = cv2.merge((output_r, output_g, output_b))
    return equ
