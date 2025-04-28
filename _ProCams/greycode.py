# Author: Bingyao Huang (https://github.com/BingyaoHuang/SPAA/)

import numpy as np
import math
import cv2 as cv
from skimage.filters import threshold_multiotsu
from scipy.interpolate import griddata
from PIL import Image

from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator


def threshold_im(im_in, compensation=False):
    # find the direct light binary mask for SPAA
    if im_in.ndim == 3:
        # get rid of out of range values
        im_in = np.clip(im_in, 0, 1)

        im_in = cv.cvtColor(
            im_in, cv.COLOR_RGB2GRAY
        )  # !!very important, result of COLOR_RGB2GRAY is different from COLOR_BGR2GRAY
        if im_in.dtype == "float32":
            im_in = np.uint8(im_in * 255)
        if compensation:
            # _, im_mask = cv.threshold(cv.GaussianBlur(im_in, (5, 5), 0), 0, 1, cv.THRESH_BINARY + cv.THRESH_OTSU)
            levels = 4
            thresholds = threshold_multiotsu(
                cv.GaussianBlur(im_in, (3, 3), 1.5), levels
            )

            # Quantized image
            im_mask = np.digitize(im_in, bins=thresholds)
            im_mask = im_mask > 2
        else:
            levels = 2
            im_in_smooth = cv.GaussianBlur(im_in, (3, 3), 1.5)
            thresholds = threshold_multiotsu(im_in_smooth, levels)

            # # Quantized image
            im_mask = np.digitize(im_in_smooth, bins=thresholds)
            im_mask = im_mask > 0

    elif im_in.dtype == bool:  # if already a binary image
        im_mask = im_in

    # find the largest contour by area then convert it to convex hull
    contours, hierarchy = cv.findContours(
        np.uint8(im_mask), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )
    if compensation:
        max_contours = max(contours, key=cv.contourArea)
        hulls = cv.convexHull(max(contours, key=cv.contourArea))
    else:
        max_contours = np.concatenate(
            contours
        )  # instead of use the largest area, we use them all
        hulls = cv.convexHull(max_contours)
    im_roi = cv.fillConvexPoly(np.zeros_like(im_mask, dtype=np.uint8), hulls, True) > 0

    # also calculate the bounding box
    # bbox = cv.boundingRect(max(contours, key=cv.contourArea))
    bbox = cv.boundingRect(max_contours)
    corners = [
        [bbox[0], bbox[1]],
        [bbox[0] + bbox[2], bbox[1]],
        [bbox[0] + bbox[2], bbox[1] + bbox[3]],
        [bbox[0], bbox[1] + bbox[3]],
    ]

    # normalize to (-1, 1) following pytorch grid_sample coordinate system
    h = im_in.shape[0]
    w = im_in.shape[1]

    for pt in corners:
        pt[0] = 2 * (pt[0] / w) - 1
        pt[1] = 2 * (pt[1] / h) - 1

    return im_mask, im_roi, corners


def im2double(im, im_type=None):
    """
    Convert image to double precision.
    """
    if im_type is not None:
        assert im_type == "indexed", 'Invalid type. Only "indexed" is supported.'

    if im.dtype == np.float64:
        return im

    elif im.dtype == np.bool_ or im.dtype == np.float32:
        return im.astype(np.float64)

    elif im.dtype == np.uint8:
        if im_type is None:
            return im.astype(np.float64) / 255
        else:
            return im.astype(np.float64) + 1

    elif im.dtype == np.uint16:
        if im_type is None:
            return im.astype(np.float64) / 65535
        else:
            return im.astype(np.float64) + 1

    elif im.dtype == np.int16:
        if im_type is None:
            return (im.astype(np.float64) + 32768) / 65535
        else:
            raise ValueError(
                "Invalid indexed image type. Only uint8, uint16, double, or logical are supported."
            )

    else:
        raise ValueError(
            "Invalid image type. Only double, logical, uint8, uint16, int16, single are supported."
        )


def robust_decode(im, im_cmp, im_d, im_g, m):
    # robust pixel classification (Xu and Aliaga GI'07 see and Moreno & Taubin 3DV'12)
    im_code = np.full(shape=im.shape, fill_value=np.nan)
    im_code[(im >= im_g) & (im_cmp <= im_d)] = 1
    im_code[(im <= im_d) & (im_cmp >= im_g)] = 0
    im_code[(im_d > im_g) & (im <= im_cmp)] = 0
    im_code[(im_d > im_g) & (im > im_cmp)] = 1
    im_code[:, im_d < m] = math.nan
    return im_code


def get_gray_code_bits_and_offset(w, h):
    w, h = int(w), int(h)

    # num of bits for horizontal/vertical patterns
    v_bits, h_bits = np.ceil(np.log2([w, h])).astype(np.uint32)

    # offset the gray pattern to be center symmetric
    v_offset, h_offset = (np.power(2, [v_bits, h_bits]) - [w, h]) // 2

    return v_bits, h_bits, v_offset, h_offset


def bin2gray(im_bin_code):
    # Binary code images to Gray code image
    n, h, w = im_bin_code.shape

    # Per-pixel binary code to Gray code
    im_gray_code = np.copy(im_bin_code)
    for i in range(1, n):
        im_gray_code[i] = np.logical_xor(im_bin_code[i - 1], im_bin_code[i])

    return im_gray_code


def gray2bin(im_gray_code):
    # Gray code images to binary code image
    n, h, w = im_gray_code.shape

    # Per-pixel Gray code to binary code
    im_bin_code = np.copy(im_gray_code)
    for i in range(1, n):
        im_bin_code[i] = np.logical_xor(im_bin_code[i - 1], im_gray_code[i])

    return im_bin_code


def dec2bin(im_dec_code, n_bits):
    # Decimal code images to binary code image
    h, w = im_dec_code.shape

    # Initialize array to store binary code image
    im_bin_code = np.zeros((n_bits, h, w), dtype=int)

    # Per-pixel decimal code to binary code
    for i in range(n_bits):
        im_bin_code[n_bits - i - 1] = (im_dec_code >> i) & 1

    return im_bin_code


def bin2dec(im_bin_code):
    # Gray code images to binary code image
    n, h, w = im_bin_code.shape

    # Per-pixel binary code to decimal code
    im_dec_code = np.zeros((h, w), dtype=int)
    for i in range(n - 1, -1, -1):
        im_dec_code += 2 ** (n - i - 1) * im_bin_code[i].astype(int)

    return im_dec_code


def dec2gray(im_dec_code, n_bits):
    # Gray code images to decimal code image

    # Per-pixel decimal code to binary code
    im_bin_code = dec2bin(im_dec_code, n_bits)

    # Per-pixel binary code to Gray code
    im_gray_code = bin2gray(im_bin_code)

    return im_gray_code


def gray2dec(im_gray_code):
    # Gray code images to decimal code image

    # Per-pixel Gray code to binary code
    im_bin_code = gray2bin(im_gray_code)

    # Per-pixel binary code to decimal code
    im_dec_code = bin2dec(im_bin_code)

    return im_dec_code


def crop_and_resize(imPrj, prjW, prjH, outSize):
    if True:  # Condition for cropAndResize (add your condition here if needed)
        offset = np.floor((prjW - prjH) / 2).astype(int)
        imPrj = np.array(
            Image.fromarray(imPrj[:, offset : prjW - offset - 1, :]).resize(outSize)
        )
        imPrj = np.uint8(imPrj)

    return imPrj


def create_gray_code_pattern(w, h):
    # TODO: slow (compared with MATLAB) and UI may freeze
    w, h = int(w), int(h)
    v_bits, h_bits, v_offset, h_offset = get_gray_code_bits_and_offset(w, h)

    # coordinates in decimal
    im_v_dec, im_h_dec = np.meshgrid(np.arange(w), np.arange(h))

    # decimal to gray code
    im_v_gray = dec2gray(im_v_dec + v_offset, v_bits)
    im_h_gray = dec2gray(im_h_dec + h_offset, h_bits)

    # create projector images
    im_prj = np.zeros((2 * (v_bits + h_bits) + 2, h, w), dtype=np.uint32)
    im_prj[0] = 1  # the 1st image is all white

    # fill gray code patterns
    v_idx = np.arange(2, 2 * (v_bits + 1), 2)
    h_idx = np.arange(2 * (v_bits + 1), 2 * v_bits + 2 * (h_bits + 1), 2)
    im_prj[v_idx] = im_v_gray
    im_prj[h_idx] = im_h_gray

    # fill complementary gray code patterns
    im_prj[v_idx + 1] = 1 - im_v_gray
    im_prj[h_idx + 1] = 1 - im_h_gray

    # convert to 3 channel and float32
    im_prj = np.broadcast_to(memoryview(im_prj[..., None]), (*im_prj.shape, 3)).astype(
        np.float32
    )
    return im_prj


def decode_gray_code_pattern(im_sl, prj_h, prj_w, threshes=None):
    (
        n_imgs,
        cam_h,
        cam_w,
        _,
    ) = im_sl.shape  # N, h, w, c
    v_bits, h_bits, v_offset, h_offset = get_gray_code_bits_and_offset(prj_w, prj_h)

    # convert to grayscale images
    im_sl_gray = np.zeros((n_imgs, cam_h, cam_w), dtype=np.double)

    for i in range(n_imgs):
        im_sl_gray[i] = im2double(cv.cvtColor(im_sl[i], cv.COLOR_RGB2GRAY))

    # Find direct light mask using Nayar TOG'06 method (also see Moreno & Taubin 3DV'12). We need at
    # least two complementary images, but more images are better. We use the 3rd and 2nd highest
    # frequencies (8 images) to estimate direct and global components as suggested by Moreno & Taubin 3DV'12
    high_freq_idx = [
        *range(2 * v_bits - 4, 2 * v_bits),
        *range(2 * v_bits + 2 * h_bits - 4, 2 * v_bits + 2 * h_bits),
    ]
    im_max = im_sl_gray[high_freq_idx].max(axis=0)
    im_min = im_sl_gray[high_freq_idx].min(axis=0)

    if threshes is None:
        t = 0.1  # min max contrast thresh (most significant for eliminating wrong decoded pixels)
        b = 0.7  # projector backlight stength (for mask use a large b, for real direct/indirect separation, use a smaller b)
        m = 0.02  # a threshold set by Xu and Aliaga GI'07 Table 2.
    else:
        t = threshes.t
        b = threshes.b
        m = threshes.m

    # get direct light mask
    im_diff = cv.normalize(
        cv.GaussianBlur(im_max - im_min, (3, 3), sigmaX=1.5),
        None,
        0,
        255,
        cv.NORM_MINMAX,
    ).astype(np.uint8)
    # th, im_mask = cv.threshold(im_diff, 0, 1, cv.THRESH_BINARY + cv.THRESH_OTSU)
    im_mask = (im_diff > threshes.t * 255).astype(np.float32)

    im_d = ((im_max - im_min) / (1 - b)).clip(max=1.0)  # direct light image
    im_g = 2 * (im_min - b * im_max) / (1 - b**2)  # indirect (global) light image

    im_d[im_g < 0] = im_max[im_g < 0]
    im_g = im_g.clip(min=0.0)

    # Robust pixel classification (Xu and Aliaga GI'07 see and Moreno & Taubin 3DV'12)
    v_idx = np.arange(2, 2 * (v_bits + 1), 2)
    h_idx = np.arange(2 * (v_bits + 1), 2 * v_bits + 2 * (h_bits + 1), 2)
    im_v = im_sl_gray[v_idx]  # vertical code
    im_v_cmp = im_sl_gray[v_idx + 1]  # complementary vertical code
    im_h = im_sl_gray[h_idx]  # horizontal code
    im_h_cmp = im_sl_gray[h_idx + 1]  # complementary horizontal code

    im_v_code = robust_decode(im_v, im_v_cmp, im_d, im_g, m)
    im_h_code = robust_decode(im_h, im_h_cmp, im_d, im_g, m)

    im_uncertain_mask = (
        np.any(np.isnan(im_v_code), axis=0)
        | np.any(np.isnan(im_h_code), axis=0)
        | (np.abs(im_max - im_min) < t)
    )
    im_v_code[:, im_uncertain_mask] = 0
    im_h_code[:, im_uncertain_mask] = 0

    # convert gray code to decimal coordinates
    im_prj_x = gray2dec(im_v_code) - v_offset
    im_prj_y = gray2dec(im_h_code) - h_offset

    im_prj_x[im_uncertain_mask] = -1
    im_prj_y[im_uncertain_mask] = -1

    prj_pts = np.vstack((im_prj_x.ravel(), im_prj_y.ravel())).T
    im_cam_x, im_cam_y = np.meshgrid(np.arange(cam_w), np.arange(cam_h))
    cam_pts = np.vstack((im_cam_x.ravel(), im_cam_y.ravel())).T

    im_invalid_idx = (
        (im_d < m)
        | (im_prj_x < 0)
        | (im_prj_x >= prj_w)
        | (im_prj_y < 0)
        | (im_prj_y >= prj_h)
    )
    cam_pts = cam_pts[~im_invalid_idx.ravel(), :].astype(np.float32)
    prj_pts = prj_pts[~im_invalid_idx.ravel(), :].astype(np.float32)

    return cam_pts, prj_pts, im_min, im_max, im_mask


# def warp_imgs(im_src, src_xy, dst_xy, dst_w, dst_h, outSize):
#     # Warp images using matched 2d points in src_xy and dst_xy

#     # interpolate a dense sampling grid using griddata
#     dst_x_dense, dst_y_dense = np.meshgrid(np.arange(dst_w), np.arange(dst_h))
#     src_xy_dense = griddata(dst_xy, src_xy, (dst_x_dense, dst_y_dense), method='linear').astype(np.float32)
#     x_map, y_map = src_xy_dense[..., 0], src_xy_dense[..., 1]

#     if im_src.ndim == 4:
#         n, h, w, c = im_src.shape
#         im_dst = np.empty((n, dst_h, dst_w, c), dtype=im_src.dtype)
#         im_dst_resize = np.empty((n, outSize[0], outSize[1], c), dtype=im_src.dtype)
#         # warp all images
#         for i in range(n):
#             im_dst[i] = cv.remap(im_src[i], x_map, y_map, interpolation=cv.INTER_LINEAR)
#             im_dst_resize[i] = crop_and_resize(im_dst[i], dst_w, dst_h, outSize)


#     else:
#         im_dst = cv.remap(im_src, x_map, y_map, interpolation=cv.INTER_LINEAR)

#     return im_dst_resize


def warp_imgs(im_src, src_xy, dst_xy, dst_w, dst_h, outSize, tri=None):
    dst_x_dense, dst_y_dense = np.meshgrid(np.arange(dst_w), np.arange(dst_h))
    dst_points = np.column_stack((dst_x_dense.ravel(), dst_y_dense.ravel()))

    if tri is None:
        tri = Delaunay(dst_xy)

    interpolator = LinearNDInterpolator(tri, src_xy)
    src_xy_dense = interpolator(dst_points).reshape(dst_h, dst_w, 2).astype(np.float32)

    x_map, y_map = src_xy_dense[..., 0], src_xy_dense[..., 1]

    x_map_umat = cv.UMat(x_map)
    y_map_umat = cv.UMat(y_map)

    if im_src.ndim == 4:
        n, h, w, c = im_src.shape
        im_dst_resize = np.empty((n, outSize[0], outSize[1], c), dtype=im_src.dtype)

        for i in range(n):
            im_src_umat = cv.UMat(im_src[i])

            im_dst_umat = cv.remap(
                im_src_umat, x_map_umat, y_map_umat, interpolation=cv.INTER_LINEAR
            )

            im_dst = im_dst_umat.get()

            im_dst_resize[i] = crop_and_resize(im_dst, dst_w, dst_h, outSize)
    else:
        im_src_umat = cv.UMat(im_src)

        im_dst_umat = cv.remap(
            im_src_umat, x_map_umat, y_map_umat, interpolation=cv.INTER_LINEAR
        )

        im_dst = im_dst_umat.get()

        im_dst_resize = crop_and_resize(im_dst, dst_w, dst_h, outSize)

    return im_dst_resize


class Threshold:
    def __init__(self, t, b, m):
        self.t = t
        self.b = b
        self.m = m
