import numpy as np
from scipy.interpolate import RectBivariateSpline
from numpy import linalg as LNG
import cv2

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
    # Put your implementation here
    # set up the threshold
    ################### TODO Implement Lucas Kanade ###################
    x1, y1, x2, y2 = rect
    dim_t = int(x2 - x1) * int(y2 - y1)
    spline_It = RectBivariateSpline(np.arange(0, It.shape[0]), np.arange(0, It.shape[1]), It)
    spline_It1 = RectBivariateSpline(np.arange(0, It1.shape[0]), np.arange(0, It1.shape[1]), It1)
    func_thresh = 1
    iter1 = 1
    x, y = np.meshgrid(np.linspace(x1, x2+1, int(x2 - x1)), np.linspace(y1, y2+1, int(y2 - y1)))
    p = p0

    while func_thresh > threshold and iter1 < num_iters:
        iter1 += 1
        A = np.zeros((dim_t, dim_t + dim_t))
        y_val = y + p[1]
        x_val = x + p[0]
        dpy = spline_It1.ev(y_val, x_val, dx=1).flatten()
        dpx = spline_It1.ev(y_val, x_val, dy=1).flatten()
        for i in range(dim_t):
            j = i + i
            A[i, j + 1] = dpy[i]
            A[i, j] = dpx[i]

        b = np.tile(np.eye(2), (dim_t, 1))
        A = np.dot(A, b)
        It1_p_temp = spline_It1.ev(y + p[1], x + p[0]).flatten()
        It_p_temp = spline_It.ev(y, x).flatten()
        arr_It = np.reshape(It_p_temp - It1_p_temp, (dim_t, 1))
        dp = LNG.pinv(A).dot(arr_It)
        p = (p + np.transpose(dp)).ravel()
        func_thresh = LNG.norm(dp)
        # if func_thresh > threshold:
        #     break
    return p
