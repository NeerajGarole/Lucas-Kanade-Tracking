import numpy as np
from scipy.interpolate import RectBivariateSpline
from numpy import linalg as LNG
from scipy.ndimage import affine_transform
import cv2

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    """

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    ################### TODO Implement Lucas Kanade Affine ###################
    spline_It = RectBivariateSpline(np.arange(0, It.shape[0]), np.arange(0, It.shape[1]), It)
    spline_It1 = RectBivariateSpline(np.arange(0, It1.shape[0]), np.arange(0, It1.shape[1]), It1)

    x, y = np.meshgrid(np.arange(It.shape[1]), np.arange(It.shape[0]))
    A = np.zeros((x.flatten().shape[0], 6))
    J = [x.flatten(), y.flatten(), np.ones(x.flatten().shape[0])]

    p = np.zeros(2)
    for iter1 in range(int(num_iters)):
        It_p_temp = spline_It.ev(y, x).flatten()
        var_x = [x, y, np.ones(x.shape)]
        matM1 = [M[1, 0], M[1, 1], M[1, 2]]
        matM2 = [M[0, 0], M[0, 1], M[0, 2]]

        It1_p_temp = spline_It1.ev(matM1[0] * var_x[0] + matM1[1] * var_x[1] + matM1[2] * var_x[2],
                                   matM2[0] * var_x[0] + matM2[1] * var_x[1] + matM2[2] * var_x[2]).flatten()
        y_val = y + p[1]
        x_val = x + p[0]
        dpy = spline_It1.ev(y_val, x_val, dx=1).flatten()
        dpx = spline_It1.ev(y_val, x_val, dy=1).flatten()
        for dpx_step in range(3):
            A[:, dpx_step] = dpx * J[dpx_step]
        for dpy_step in range(3, 6, 1):
            A[:, dpy_step] = dpy * J[dpy_step - 3]
        dp = LNG.inv(np.transpose(A) @ A) @ np.transpose(A) @ (It_p_temp - It1_p_temp).reshape(-1, 1)
        for m in range(2):
            M[m, :] = M[m, :] + dp[m*3:m*3+3, 0]

        if LNG.norm(dp) < threshold:
            break
    return M
