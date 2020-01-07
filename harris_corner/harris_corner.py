import numpy as np
import sys
import cv2 as cv
from scipy import ndimage as ndi


# show entire nparray
np.set_printoptions(threshold=sys.maxsize)


class harris_corner(object):
    """a corner detection algorithm"""
    def __init__(self, img):
        self.img = img
    
    def derivative_calculation(self):
        """
        caculate the brightness differential(the dx & dy) by using 
        sobel operator:
        x-component = [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]
        y-component = [
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ]

        then return the M matrix:
        M = [
            [dx, dx*dy],
            [dx*dy, dy]
        ]
        """
        x_mask= np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=np.float)
        y_mask = np.array([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ], dtype=np.float)

        Ix = cv.Sobel(self.img, cv.CV_32F, 1, 0)
        Iy = cv.Sobel(self.img, cv.CV_32F, 0, 1)

        self.Ixx = Ix**2
        self.Iyy = Iy**2
        self.Ixy = Ix * Iy
        return Ix, Iy, self.Ixx, self.Iyy, self.Ixy

    def window_function(self):
        self.Ixx = ndi.gaussian_filter(self.Ixx, sigma=1)
        self.Iyy = ndi.gaussian_filter(self.Iyy, sigma=1)
        self.Ixy = ndi.gaussian_filter(self.Ixy, sigma=1)
        cv.imshow("filter Ixx", self.Ixx)
        cv.imshow("filter Iyy", self.Iyy)
        cv.imshow("filter Ixy", self.Ixy)
        
    def response_function(self, window_size=3, k=0.04):
        height = self.img.shape[0]
        width =  self.img.shape[1]

        h_windows_size = window_size//2
        # (353, 454)
        M_matrix = np.ones((2,2))
        self.R_matrix = np.zeros((height, width))

        for y in range(height-window_size+1):
            for x in range(width-window_size+1):
                M_matrix[0, 0] = np.sum(self.Ixx[y:y+window_size, x:x+window_size])
                M_matrix[1, 1] = np.sum(self.Iyy[y:y+window_size, x:x+window_size])
                M_matrix[1, 0] = M_matrix[0, 1] = np.sum(self.Ixy[y:y+window_size, x:x+window_size])
                
                R = np.linalg.det(M_matrix) - k * (np.trace(M_matrix)**2)
                
                x_moved = x + h_windows_size
                y_moved = y + h_windows_size 
                
                self.R_matrix[y_moved, x_moved] = R
        return self.R_matrix
    
    def plot_res(self, res):
        img = cv.cvtColor(self.img, cv.COLOR_GRAY2RGB)
        edges_img = np.copy(img)
        corner_img = np.copy(img)

        res_max = res.max()
        res_min = res.min()

        for x_index, row in enumerate(res):
            for y_index, response in enumerate(row):
                if response > 0.18 * res_max:
                    corner_img[x_index, y_index] = [0, 255, 0]
                elif response < 0:
                    edges_img[x_index, y_index] = [255, 0, 0]
                print(response)
        return edges_img, corner_img


if __name__ == "__main__":
    img = cv.imread('box.jpg', cv.IMREAD_GRAYSCALE)
    haris_cnr = harris_corner(img)
    Ix, Iy, Ixx, Iyy, Ixy = haris_cnr.derivative_calculation()
    haris_cnr.window_function()
    response = haris_cnr.response_function()
    edge, corner = haris_cnr.plot_res(response)
    
    f_img = np.float32(img)
    cv_harris = cv.cornerHarris(f_img, 3, 3, 0.04)
    
    cv_edge = cv.cvtColor(f_img, cv.COLOR_GRAY2RGB)
    cv_edge[cv_harris < 0] = [0, 0, 255]

    cv_corner = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    cv_corner[cv_harris > 0] = [255, 255, 0]
    
    cv.imshow("Ix", Ix)
    cv.imshow("Iy", Iy)
    cv.imshow("Ixx", Ixx)
    cv.imshow("Iyy", Iyy)
    cv.imshow("Ixy", Ixy)
    cv.imshow("opencv harris edge", cv_edge)
    cv.imshow("opencv harris corner", cv_corner)
    cv.imshow("edge", edge)
    cv.imshow("corner", corner)

    cv.waitKey(0)