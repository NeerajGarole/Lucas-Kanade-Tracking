import numpy as np
from scipy.interpolate import RectBivariateSpline
from numpy import linalg as LNG
from scipy.ndimage import affine_transform
import cv2


def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    ################### TODO Implement Inverse Composition Affine ###################
    spline_It = RectBivariateSpline(np.arange(0, It.shape[0]), np.arange(0, It.shape[1]), It)
    spline_It1 = RectBivariateSpline(np.arange(0, It1.shape[0]), np.arange(0, It1.shape[1]), It1)

    x, y = np.meshgrid(np.arange(It.shape[1]), np.arange(It.shape[0]))
    A = np.zeros((x.flatten().shape[0], 6))
    p = np.zeros(2)
    J = [x.flatten(), y.flatten(), np.ones(x.flatten().shape[0])]

    y_val = y + p[1]
    x_val = x + p[0]
    dpy = spline_It1.ev(y_val, x_val, dx=1).flatten()
    dpx = spline_It1.ev(y_val, x_val, dy=1).flatten()

    for dpx_step in range(3):
        A[:, dpx_step] = dpx * J[dpx_step]
    for dpy_step in range(3, 6, 1):
        A[:, dpy_step] = dpy * J[dpy_step - 3]
    H_mat = np.transpose(A) @ A

    for iters in range(int(num_iters)):
        It_p_temp = spline_It.ev(y, x).flatten()
        var_x = [x, y, np.ones(x.shape)]
        matM1 = [M[1, 0], M[1, 1], M[1, 2]]
        matM2 = [M[0, 0], M[0, 1], M[0, 2]]

        It1_p_temp = spline_It1.ev(matM1[0]*var_x[0] + matM1[1]*var_x[1] + matM1[2]*var_x[2],
                                   matM2[0]*var_x[0] + matM2[1]*var_x[1] + matM2[2]*var_x[2]).flatten()

        dp = LNG.inv(H_mat) @ A.T @ (It1_p_temp - It_p_temp).reshape(-1, 1)
        dp_mat1 = [1, 0, 0]
        for r in range(3):
            dp_mat1[r] = dp_mat1[r] + dp[r, 0]
        dp_mat2 = [0, 1, 0]
        for r1 in range(3, 6, 1):
            dp_mat2[r1-3] = dp_mat2[r1-3] + dp[r1, 0]
        dp_mat3 = [0, 0, 1]

        M = np.concatenate((M, [dp_mat3]), axis=0) @ LNG.inv(np.array([dp_mat1, dp_mat2, dp_mat3]))
        if LNG.norm(dp) < threshold:
            break

    return M[:2, :]
