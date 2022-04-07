import cv2
import numpy as np
import torch


class GaussianBlur(object):
    def __init__(self, max_kernel=(7, 7)):
        try: self.max_kernel = max_kernel // 2
        except: self.max_kernel = tuple(k // 2 for k in max_kernel)


    def __call__(self, X, Y):
        kernel_size = (
            np.random.randint(0, self.max_kernel[0])*2 + 1,
            np.random.randint(0, self.max_kernel[1])*2 + 1,
        )
        X = cv2.GaussianBlur(X, kernel_size, 0)
        return X, Y

class Sharpen(object):
    def __init__(self, max_center=4):
        self.max_center = max_center
        self.identity = np.array([[0, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 0]])
        self.sharpen = np.array([[  0, -.25,    0],
                                [-.25,    1, -.25],
                                [   0, -.25,    0]])

    def __call__(self, X, Y):

        sharp = self.sharpen * np.random.random() * self.max_center
        kernel = self.identity + sharp

        X = cv2.filter2D(X, -1, kernel)
        return X, Y

class Brightness(object):
    def __init__(self, range_brightness=(-20, 20)):
        self.range_brightness = range_brightness

    def __call__(self, X, Y):
        brightness = np.random.randint(*self.range_brightness)
        X = X + brightness
        return X, Y

class Contrast(object):
    def __init__(self, range_contrast=(-20, 20)):
        self.range_contrast = range_contrast

    def __call__(self, X, Y):
        contrast = np.random.randint(*self.range_contrast)
        X = X * (contrast / 127 + 1) - contrast
        return X, Y

class GaussianNoise(object):
    def __init__(self, center=0, std=10):
        self.center = center
        self.std = std

    def __call__(self, X, Y):
        noise = np.random.normal(self.center, self.std, X.shape)
        X = X + noise
        return X, Y

class LensDistortion(object):

    def __init__(self, d_coef=(0.15, 0.15, 0.1, 0.1, 0.05)):
        self.d_coef = np.array(d_coef)

    def __call__(self, X, Y):

        # get the height and the width of the image
        h, w = X.shape[:2]

        # compute its diagonal
        f = (h ** 2 + w ** 2) ** 0.5

        # set the image projective to carrtesian dimension
        K = np.array([[f, 0, w / 2],
                      [0, f, h / 2],
                      [0, 0,     1]])

        d_coef = self.d_coef * np.random.random(5) # value
        d_coef = d_coef * (2 * (np.random.random(5) < 0.5) - 1) # sign
        # Generate new camera matrix from parameters
        M, _ = cv2.getOptimalNewCameraMatrix(K, d_coef, (w, h), 0)

        # Generate look-up tables for remapping the camera image
        remap = cv2.initUndistortRectifyMap(K, d_coef, None, M, (w, h), 5)

        # Remap the original image to a new image
        X = cv2.remap(X, *remap, cv2.INTER_LINEAR)
        Y = cv2.undistortPoints(np.expand_dims(Y, axis=1), K, d_coef, None, M)[:, 0]
        return X, Y

class Perspective(object):
    def __init__(self,
                 max_ratio_translation=(0.2, 0.2, 0),
                 max_rotation=(5, 5, 180),
                 max_scale=(0.1, 0.1, 0.2),
                 max_shearing=(5, 5, 5)):

        self.max_ratio_translation = np.array(max_ratio_translation)
        self.max_rotation = np.array(max_rotation)
        self.max_scale = np.array(max_scale)
        self.max_shearing = np.array(max_shearing)

    def __call__(self, X, Y):

        # get the height and the width of the image
        h, w = X.shape[:2]
        max_translation = self.max_ratio_translation * np.array([w, h, 1])
        # get the values on each axis
        t_x, t_y, t_z = np.random.uniform(-1, 1, 3) * max_translation
        r_x, r_y, r_z = np.random.uniform(-1, 1, 3) * self.max_rotation
        sc_x, sc_y, sc_z = np.random.uniform(-1, 1, 3) * self.max_scale + 1 + 0.1
        sh_x, sh_y, sh_z = np.random.uniform(-1, 1, 3) * self.max_shearing

        # convert degree angles to rad

        theta_rx = np.deg2rad(r_x)
        theta_ry = np.deg2rad(r_y)
        theta_rz = np.deg2rad(r_z)
        theta_shx = np.deg2rad(sh_x)
        theta_shy = np.deg2rad(sh_y)
        theta_shz = np.deg2rad(sh_z)

        # compute its diagonal
        diag = (h ** 2 + w ** 2) ** 0.5
        # compute the focal length
        f = diag
        if np.sin(theta_rz) != 0:
            f /= 2 * np.sin(theta_rz)

        # set the image from cartesian to projective dimension
        H_M = np.array([[1, 0, -w / 2],
                        [0, 1, -h / 2],
                        [0, 0,      1],
                        [0, 0,      1]])
        # set the image projective to carrtesian dimension
        Hp_M = np.array([[f, 0, w / 2, 0],
                         [0, f, h / 2, 0],
                         [0, 0,     1, 0]])

        # adjust the translation on z
        t_z = (f - t_z) / sc_z ** 2
        # translation matrix to translate the image
        T_M = np.array([[1, 0, 0, t_x],
                        [0, 1, 0, t_y],
                        [0, 0, 1, t_z],
                        [0, 0, 0,  1]])

        # calculate cos and sin of angles
        sin_rx, cos_rx = np.sin(theta_rx), np.cos(theta_rx)
        sin_ry, cos_ry = np.sin(theta_ry), np.cos(theta_ry)
        sin_rz, cos_rz = np.sin(theta_rz), np.cos(theta_rz)
        # get the rotation matrix on x axis
        R_Mx = np.array([[1,      0,       0, 0],
                         [0, cos_rx, -sin_rx, 0],
                         [0, sin_rx,  cos_rx, 0],
                         [0,      0,       0, 1]])
        # get the rotation matrix on y axis
        R_My = np.array([[cos_ry, 0, -sin_ry, 0],
                         [     0, 1,       0, 0],
                         [sin_ry, 0,  cos_ry, 0],
                         [     0, 0,       0, 1]])
        # get the rotation matrix on z axis
        R_Mz = np.array([[cos_rz, -sin_rz, 0, 0],
                         [sin_rz,  cos_rz, 0, 0],
                         [     0,       0, 1, 0],
                         [     0,       0, 0, 1]])
        # compute the full rotation matrix
        R_M = np.dot(np.dot(R_Mx, R_My), R_Mz)

        # get the scaling matrix
        Sc_M = np.array([[sc_x,     0,    0, 0],
                         [   0,  sc_y,    0, 0],
                         [   0,     0, sc_z, 0],
                         [   0,     0,    0, 1]])

        # get the tan of angles
        tan_shx = np.tan(theta_shx)
        tan_shy = np.tan(theta_shy)
        tan_shz = np.tan(theta_shz)
        # get the shearing matrix on x axis
        Sh_Mx = np.array([[      1, 0, 0, 0],
                          [tan_shy, 1, 0, 0],
                          [tan_shz, 0, 1, 0],
                          [      0, 0, 0, 1]])
        # get the shearing matrix on y axis
        Sh_My = np.array([[1, tan_shx, 0, 0],
                          [0,       1, 0, 0],
                          [0, tan_shz, 1, 0],
                          [0,       0, 0, 1]])
        # get the shearing matrix on z axis
        Sh_Mz = np.array([[1, 0, tan_shx, 0],
                          [0, 1, tan_shy, 0],
                          [0, 0,       1, 0],
                          [0, 0,       0, 1]])
        # compute the full shearing matrix
        Sh_M = np.dot(np.dot(Sh_Mx, Sh_My), Sh_Mz)

        Identity = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])

        # compute the full transform matrix
        M = Identity
        M = np.dot(Sh_M, M)
        M = np.dot(R_M,  M)
        M = np.dot(Sc_M, M)
        M = np.dot(T_M,  M)
        M = np.dot(Hp_M, np.dot(M, H_M))
        # apply the transformation
        X = cv2.warpPerspective(X, M, (w, h))

        for i, pt in enumerate(Y):
            homogene_pt = np.array([pt[0], pt[1], 1])
            # Calculate its perspective coordinates
            persp_homogene_pt = M.dot(homogene_pt)
            # Rescale it
            persp_homogene_pt = persp_homogene_pt[:-1] / persp_homogene_pt[-1]
            Y[i] = persp_homogene_pt.astype(np.int)

        return X, Y

class Clip(object):
    def __init__(self, mini, maxi=None):
        if maxi is None:
            self.mini, self.maxi = 0, mini
        else:
            self.mini, self.maxi = mini, maxi

    def __call__(self, X, Y):
        mini_mask = np.where(X < self.mini)
        maxi_mask = np.where(X > self.maxi)
        X[mini_mask] = self.mini
        X[maxi_mask] = self.maxi
        return X, Y


MEDIUM_TRANSFORM = [
    GaussianBlur(),
    Sharpen(),
    Brightness(),
    Contrast(),
    GaussianNoise(),
    LensDistortion(),
    Perspective(),
    Clip(255)
]
