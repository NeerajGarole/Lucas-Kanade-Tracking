import numpy as np
import scipy
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import affine_transform
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine


def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """
    
    # put your implementation here
    mask = np.ones(image1.shape, dtype=bool)

    ################### TODO Implement Substract Dominent Motion ###################

    M = InverseCompositionAffine(image1, image2, threshold, num_iters)
    # M = LucasKanadeAffine(image1, image2, threshold, num_iters)

    aff_tr = scipy.ndimage.affine_transform(image1, -M, offset=0.0, output_shape=None)
    binary0 = abs(aff_tr - image2) > tolerance
    mask[binary0] = 0
    # print(mask)
    binary1 = abs(aff_tr - image2) < tolerance
    mask[binary1] = 1
    mask = scipy.ndimage.morphology.binary_erosion(mask)
    mask = scipy.ndimage.morphology.binary_dilation(mask, iterations=2)
    # mask = scipy.ndimage.morphology.binary_dilation(mask, iterations=7)
    return mask.astype(bool)

