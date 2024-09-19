
import cv2

import numpy as np
import math

from matplotlib import pyplot as plt

def guided_filter(I,p,win_size,eps):
    '''
    %   - guidance image: I (should be a gray-scale/single channel image)
    %   - filtering input image: p (should be a gray-scale/single channel image)
    %   - local window radius: r
    %   - regularization parameter: eps
    '''
    mean_I = cv2.blur(I,(win_size,win_size))
    mean_p = cv2.blur(p,(win_size,win_size))

    mean_II = cv2.blur(I*I,(win_size,win_size))
    mean_Ip = cv2.blur(I*p,(win_size,win_size))

    var_I = mean_II - mean_I*mean_I
    cov_Ip = mean_Ip - mean_I*mean_p
    #print(np.allclose(var_I, cov_Ip))

    a = cov_Ip/(var_I+eps)
    b = mean_p-a*mean_I

    mean_a = cv2.blur(a,(win_size,win_size))
    mean_b = cv2.blur(b,(win_size,win_size))

    q = mean_a*I + mean_b
    #print(mean_II.dtype, cov_Ip.dtype, b.dtype, mean_a.dtype, I.dtype, q.dtype)
    return q
def side_guided_filter(I,p,kernel,eps=0.02*0.02*255*255):
    '''
    %   - guidance image: I (should be a gray-scale/single channel image)
    %   - filtering input image: p (should be a gray-scale/single channel image)
    %   - local window radius: r
    %   - regularization parameter: eps
    '''

    mean_I = cv2.filter2D(I, -1, kernel) #cv2.blur(I,(win_size,win_size))
    mean_p = cv2.filter2D(p, -1, kernel)

    mean_II = cv2.filter2D(I*I,-1, kernel)
    mean_Ip = cv2.filter2D(I*p,-1, kernel)

    var_I = mean_II - mean_I*mean_I
    cov_Ip = mean_Ip - mean_I*mean_p
    #print(np.allclose(var_I, cov_Ip))

    a = cov_Ip/(var_I+eps)
    b = mean_p-a*mean_I

    mean_a = cv2.filter2D(a,-1, kernel)
    mean_b = cv2.filter2D(b,-1, kernel)

    q = mean_a*I + mean_b
    #print(mean_II.dtype, cov_Ip.dtype, b.dtype, mean_a.dtype, I.dtype, q.dtype)
    return q


def get_side_kernels_3_5():
    side_kernels_3 = np.array([[1, 1, 0, 1, 1, 0, 1, 1, 0],
                              [0, 1, 1, 0, 1, 1, 0, 1, 1],
                              [1, 1, 1, 1, 1, 1, 0, 0, 0],
                              [0, 0, 0, 1, 1, 1, 1, 1, 1],
                              [1, 1, 0, 1, 1, 0, 0, 0, 0],
                              [0, 1, 1, 0, 1, 1, 0, 0, 0],
                              [0, 0, 0, 1, 1, 0, 1, 1, 0],
                              [0, 0, 0, 0, 1, 1, 0, 1, 1],

                              [1, 1, 1, 1, 1, 0, 1, 0, 0],
                              [1, 1, 1, 0, 1, 1, 0, 0, 1],
                              [1, 0, 0, 1, 1, 0, 1, 1, 1],
                              [0, 0, 1, 0, 1, 1, 1, 1, 1]]).astype(np.float32)

    side_kernels_3 = side_kernels_3.reshape(-1, 3, 3)


    k = 5
    side_kernels_5 = []
    for i in range(len(side_kernels_3)):
        tmp = cv2.resize(side_kernels_3[i], dsize=(k, k), interpolation=cv2.INTER_LINEAR)
        tmp[tmp < 0.75] = 0
        tmp[tmp > 0.75] = 1
        side_kernels_5.append(tmp)
        # print(i)
        # print(side_kernels_3[i])
        # print(tmp)
    # # 归一化
    # side_kernels_3 = side_kernels_3 / np.sum(side_kernels_3, axis=(1, 2), keepdims=True)
    # side_kernels_5 = side_kernels_5 / np.sum(side_kernels_5, axis=(1, 2), keepdims=True)
    return side_kernels_3, side_kernels_5



# 根据给定的side kernel中值滤波
def side_median(img, side_kernel):
    h, w, c = img.shape
    r = side_kernel.shape[0] // 2
    #print('r:', r)
    img_ext = np.pad(img, ((r, r), (r, r), (0, 0)))

    imgs = []
    for i in range(2 * r + 1):
        for j in range(2 * r + 1):
            imgs.append(img_ext[i:i+h, j:j + w, :])

    imgs = np.array(imgs) # K, h, w, 3

    imgs_sel = imgs[side_kernel.astype(np.bool_).reshape(-1)]
    # print(imgs.shape, imgs_sel.shape)
    ret = np.median(imgs_sel, axis=0)
    return ret

# kernel size 必须是奇数
# std_dev 标准差
# stride  邻域到中心的间隔距离
# 得到一个高斯2d矩阵
def gauss_kern_raw(kern, std_dev, stride=1):
    """
    Applying Gaussian Filter
    """
    if kern % 2 == 0:
        print("kernel size (kern) cannot be even, setting it as odd value")
        kern = kern + 1

    if kern <= 0:
        print("kernel size (kern) cannot be <= zero, setting it as 3")
        kern = 3

    out_kern = np.zeros((kern, kern), dtype=np.float32)

    for i in range(0, kern):
        for j in range(0, kern):
            # stride is used to adjust the gaussian weights for neighbourhood
            # pixel that are 'stride' spaces apart in a bayer image
            out_kern[i, j] = np.exp(
                -1
                * (
                    (stride * (i - ((kern - 1) / 2))) ** 2
                    + (stride * (j - ((kern - 1) / 2))) ** 2
                )
                / (2 * (std_dev**2))
            )

    sum_kern = np.sum(out_kern)
    out_kern[0:kern:1, 0:kern:1] = out_kern[0:kern:1, 0:kern:1] / sum_kern

    return out_kern

# side双边滤波
def side_bi(in_img, kernel, r=5, sigma_s=20, sigma_r=20):
    in_img = in_img.astype(np.float32)
    spatial_kern = r
    stddev_s = sigma_s
    stddev_r = sigma_r
    stride = 1
    # spawn a NxN gaussian kernel
    s_kern = gauss_kern_raw(spatial_kern, stddev_s, stride)
    # print(kernel)
    # print(s_kern)
    # print('dd')
    # pad the image with half arm length of the kernel;
    # padType='constant' => pad value = 0; 'reflect' is more suitable
    pad_len = int((spatial_kern - 1) / 2)
    in_img_ext = np.pad(in_img, ((pad_len, pad_len), (pad_len, pad_len), (0, 0)), "reflect")

    guide_img = np.mean(in_img, axis=-1) # gray
    guide_img_ext = np.pad(
        guide_img, ((pad_len, pad_len), (pad_len, pad_len)), "reflect"
    )


    norm_fact = np.zeros(in_img.shape[:2], dtype=np.float32)
    sum_filt_out = np.zeros(in_img.shape, dtype=np.float32)

    for i in range(spatial_kern):
        for j in range(spatial_kern):
            # Creating shifted arrays for processing each pixel in the window
            in_img_ext_array = in_img_ext[
                               i: i + in_img.shape[0], j: j + in_img.shape[1], ...
                               ]
            guide_img_ext_array = guide_img_ext[
                                  i: i + in_img.shape[0], j: j + in_img.shape[1]
                                  ]

            # Adding normalization factor for each pixel needed to average out the
            # final result
            t = s_kern[i, j] * np.exp(
                -1 * (guide_img - guide_img_ext_array) ** 2 / (2 * stddev_r ** 2)
            ) * kernel[i, j]
            norm_fact += t  # space weight * value weight * side weight

            # Summing up the final result
            sum_filt_out += t[..., None] * in_img_ext_array

    filt_out = sum_filt_out / (norm_fact[..., None]+1e-8)

    return filt_out

def side_filter_box(img, kernels, iter=1):
    kernels = kernels / np.sum(kernels, axis=(1, 2), keepdims=True)
    img = img.astype(np.float32)
    h, w, c = img.shape

    for k in range(iter):
        img_diff_s = []
        for i in range(len(kernels)):
            img_diff_s.append(cv2.filter2D(img, -1, kernels[i]) - img)

        img_diff_s = np.array(img_diff_s).reshape(len(kernels), -1)
        # print(img_diff_s.shape)
        t = np.argmin(np.abs(img_diff_s), axis=0)
        # print(t.shape)
        img = img_diff_s[t, range(img_diff_s.shape[1])].reshape(h, w, c) + img
    ret = img.clip(0, 255).astype(np.uint8)
    return ret

def side_filter_median(img, kernels, iter=1):
    # kernels = kernels / np.sum(kernels, axis=(1, 2), keepdims=True)
    img = img.astype(np.float32)
    h, w, c = img.shape

    for k in range(iter):
        img_diff_s = []
        for i in range(len(kernels)):
            img_diff_s.append(side_median(img, kernels[i]) - img)

        img_diff_s = np.array(img_diff_s).reshape(len(kernels), -1)
        # print(img_diff_s.shape)
        t = np.argmin(np.abs(img_diff_s), axis=0)
        # print(t.shape)
        img = img_diff_s[t, range(img_diff_s.shape[1])].reshape(h, w, c)  + img
    ret = img.clip(0, 255).astype(np.uint8)
    return ret


def side_filter_bi(img, kernels, iter=1, r=5, sigma_s=20, sigma_r=20):
    # kernels = kernels / np.sum(kernels, axis=(1, 2), keepdims=True)
    img = img.astype(np.float32)
    h, w, c = img.shape

    for k in range(iter):
        img_diff_s = []
        for i in range(len(kernels)):
            img_diff_s.append(side_bi(img, kernels[i], r, sigma_s, sigma_r) - img)

        img_diff_s = np.array(img_diff_s).reshape(len(kernels), -1)
        # print(img_diff_s.shape)
        t = np.argmin(np.abs(img_diff_s), axis=0)
        # print(t.shape)
        img = img_diff_s[t, range(img_diff_s.shape[1])].reshape(h, w, c) + img

    ret = img.clip(0, 255).astype(np.uint8)
    return ret

def side_filter_guide(img, kernels, iter=1, eps=0.02*0.02*255*255):
    kernels = kernels / np.sum(kernels, axis=(1, 2), keepdims=True)
    img = img.astype(np.float32)
    h, w, c = img.shape

    for k in range(iter):
        img_diff_s = []
        for i in range(len(kernels)):
            img_diff_s.append(side_guided_filter(img,img,kernels[i],eps) - img)

        img_diff_s = np.array(img_diff_s).reshape(len(kernels), -1)
        # print(img_diff_s.shape)
        t = np.argmin(np.abs(img_diff_s), axis=0)
        # print(t.shape)
        img = img_diff_s[t, range(img_diff_s.shape[1])].reshape(h, w, c) + img
    ret = img.clip(0, 255).astype(np.uint8)
    return ret

if __name__ == "__main__":

    file = r'D:\dataset\benchmark\Set14\HR\ppt3.png'


    # file = r'D:\code\local_laplacian_filter-master\Snipaste_2024-09-13_17-02-30.png'
    file = r'D:\code\local_laplacian_filter-master\Snipaste_2024-09-13_17-03-06.png'
    # file = r'D:\code\local_laplacian_filter-master\Snipaste_2024-09-14_11-27-19.png'
    # file = r'D:\code\local_laplacian_filter-master\Snipaste_2024-09-14_11-27-07.png'
    img = cv2.imread(file, 1)
    side_kernels_3, side_kernels_5 = get_side_kernels_3_5()
    # img_bi = cv2.bilateralFilter(img, 5, 20, 20)
    # img_bi2 = side_bi(img, side_kernels_5[-1])
    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(img_bi[..., ::-1])
    # plt.subplot(122)
    # plt.imshow(img_bi2.astype(np.uint8)[..., ::-1])
    # plt.show()

    iter_num = 1
    img_box = img.copy()
    img_median = img.copy()
    img_bi = img.copy()
    img_guide = img.copy()

    w_box = 5
    w_median = 5
    w_bi = 5
    sigma_s = 50
    sigma_r = 50

    w_guide = 5
    eps = 0.02*0.02*255*255
    for ii in range(iter_num):
        img_box = cv2.boxFilter(img_box, -1, (w_box, w_box))
        img_median = cv2.medianBlur(img_median, w_median)
        img_bi = cv2.bilateralFilter(img_bi, w_bi, sigma_s, sigma_r)
        img_guide = guided_filter(img_guide, img_guide, w_guide, eps).clip(0, 255).astype(np.uint8)

    out = side_filter_box(img, side_kernels_5, iter_num)
    out_median = side_filter_median(img, side_kernels_5, iter_num)
    out_bi = side_filter_bi(img, side_kernels_5, iter_num, w_bi, sigma_s, sigma_r)
    out_guide = side_filter_guide(img, side_kernels_5, iter_num, eps)


    plt.figure()
    plt.subplot(251)
    plt.imshow(img[..., ::-1])
    plt.subplot(252)
    plt.imshow(img_box[..., ::-1])
    plt.subplot(253)
    plt.imshow(img_median[..., ::-1])
    plt.subplot(254)
    plt.imshow(img_bi[..., ::-1])
    plt.subplot(255)
    plt.imshow(img_guide[..., ::-1])

    plt.subplot(256)
    plt.imshow(img[..., ::-1])
    plt.subplot(257)
    plt.imshow(out[..., ::-1])
    plt.subplot(258)
    plt.imshow(out_median[..., ::-1])
    plt.subplot(259)
    plt.imshow(out_bi[..., ::-1])
    plt.subplot(2, 5, 10)
    plt.imshow(out_guide[..., ::-1])
    plt.show()
