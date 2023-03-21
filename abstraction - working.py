import numpy as np
import numpy.lib.stride_tricks as ns
import imageio
import skimage.feature as feat
import skimage.color as skc
import skimage.filters as skf
import matplotlib.pyplot as plt

def show(images, titles):
    _, ax = plt.subplots(1, len(images), figsize=(15, 10))
    for i, (im, title) in enumerate(zip(images, titles)):
        ax[i].imshow(im, cmap='gray')
        ax[i].set_title(title)


def edge_detection(im):
    im1 = skf.gaussian(image=im, sigma=sigma_e)
    im2 = skf.gaussian(image=im, sigma=(np.sqrt(1.6) * sigma_e))

    im3 = im2 * tau
    im12 = (im1 - im3) * im

    for i in range(0, im12.shape[0]):
        for j in range(0, im12.shape[1]):
            if im12[i, j] > 0:
                im12[i, j] = 1
            else:
                im12[i, j] = 1 + np.tanh(phi_e * im12[i,j])

    # imageio.imsave('edgeWOW.png', im12 * 255)
    return im12



def luminance_quantization(im):
    print(im.shape[0], im.shape[1])
    delta_l = 100 / n_bins + 1
    bins = np.arange(0, 100 + delta_l, delta_l)
    quantized = np.zeros_like(im)
    for i in range(0, im.shape[0]):
        for j in range(0, im.shape[1]):
            bin_index = np.argmin(np.abs(im[i, j] -  bins))
            gx = bins[bin_index] + (delta_l / 2) * np.tanh(phi_q*(im[i, j] - bins[bin_index]))
            quantized[i, j] = gx

    # imageio.imsave('lumWOW.png', quantized * 255)

    return quantized


def bilateral_gaussian(im):
    # Radius of the Gaussian filter
    # Radius of the Gaussian filter
    r = int(2 * sigma_s) + 1
    padded = np.pad(im, ((r, r), (r, r), (0, 0)), 'edge')
    '''
    Implement the bilateral Gaussian filter (Eq. 3).
    Apply it to the padded image.
    '''
    # imageio.imsave('padded.png', np.clip(padded, 0, 1))

    window_size = 2 * r + 1
    img_with_windows = ns.sliding_window_view(padded, (window_size, window_size), (0, 1))
    # x, y  = np.mgrid[-window_size/2:window_size/2,-window_size/2:window_size/2]
    # print(x)


    #always the same since we use indices/offsets from centre pixel
    y,x = np.mgrid[-r:r+1,-r:r+1]
    h = np.exp(-(x**2+y**2)/(2.*(r/2)**2))
    spatial_gaussian = h/h.sum()


    # #imageio.imsave("gauss_kernel.png", h*255)
    # spatial_grid = np.zeros((window_size, window_size))
    # m, n = spatial_grid.shape
    # for i in range(0, m):
    #     for j in range(0, n):
    #         spatial_grid[i, j] = ((i - r) ** 2 + (j - r) ** 2) ** 0.5

    #spatial_distance = spatial_grid

    # spatial_gaussian = h # np.exp(-(spatial_distance ** 2 / (float)(2 * sigma_s ** 2)))


    for i in range(r, padded.shape[0] - r):
        for j in range(r, padded.shape[1] - r):
            window = np.transpose(img_with_windows[i - r, j - r], (1, 2, 0))
            F_p = window[r, r]

            feature_vectors = F_p - window
            feature_distances = np.linalg.norm(feature_vectors, axis=2)

            feature_gaussian = np.exp(-(feature_distances ** 2 / (float)(2 * sigma_r ** 2)))
            gaussian = spatial_gaussian * feature_gaussian


            temp_arr = np.array([gaussian * window[:, :, 0], gaussian * window[:, :, 1], gaussian * window[:, :, 2]])
            U_p = np.array([np.sum(temp_arr[0]), np.sum(temp_arr[1]), np.sum(temp_arr[2])])


            padded[i, j] = U_p / (float)(np.sum(gaussian))

    # ------------tried to implement vectorised bilateral filter but its producing incorrect output ----------------

    # elems = window_size * window_size
    # repeats = np.repeat(img_with_windows[:,:,:,r,r], [elems, elems, elems], axis=2).reshape(img_with_windows.shape)
    # feature_vectors = repeats - img_with_windows
    # test = np.transpose(feature_vectors[235,182], (1,2,0)) * 255
    # imageio.imsave("feature_distance.png", test )
    # feature_distances = np.linalg.norm(feature_vectors, axis=2)

    # feature_gaussian = np.exp(-(feature_distances ** 2 / (float)(2 * sigma_r ** 2)))
    # gaussian = spatial_gaussian * feature_gaussian

    # gauss_repeat = np.repeat(gaussian[:,:], [3]).reshape(img_with_windows.shape)
    # temp_arr = gauss_repeat * img_with_windows
    # U_p = np.sum(temp_arr, axis=(3,4))

    # gaussian_sum = np.sum(gaussian, axis=(2,3))
    # gaussian_sum = np.repeat(gaussian_sum, 3).reshape(U_p.shape)
    # filtered = U_p / gaussian_sum
    #------------------------------------------------------------------------------------------------
    filtered = padded[r:-r, r:-r]


    # imageio.imsave('filtered.png', im*255)
    # imageio.imsave('filtered2.png', filtered*255)

    return filtered


def abstraction(im):
    filtered = skc.rgb2lab(im)

    for _ in range(n_e):
        filtered = bilateral_gaussian(filtered)
    edges = edge_detection(filtered[:, :, 0])
    # print(edges.shape)

    imageio.imsave('edges.png', edges * 255)

    for _ in range(n_b - n_e):
        filtered = bilateral_gaussian(filtered)
    luminance_quantized = luminance_quantization(filtered[:, :, 0])
    print(luminance_quantized.shape)
    imageio.imsave('luminance_quantized.png', luminance_quantized * 255)

    '''Get the final image by merging the channels properly'''
    lab = np.multiply(edges, luminance_quantized)
    # print("LAB: ", lab.shape)
    # print("Filtered: ", filtered[:, :, 1].shape)
    # print()

    abstract = np.dstack((lab, filtered[:, :, 1], filtered[:, :, 2]))
    abstract = skc.lab2rgb(abstract)
    show([im, edges, luminance_quantized, abstract], ['original', 'edges', 'luminance_quantized', 'combined'])
    plt.show()
    return skc.lab2rgb(abstract)


if __name__ == '__main__':
    # Algorithm
    n_e = 2
    n_b = 4
    # Bilateral Filter
    sigma_r = 4.25  # "Range" sigma
    #sigma_r = 2.1  # "Range" sigma
    sigma_s = 1.5  # "Spatial" sigma
    # Edge Detection
    sigma_e = 1
    tau = 0.98
    phi_e = 5 #adjusted from literature value
    # Luminance Quantization
    n_bins = 10
    phi_q = 3


    im = imageio.imread('./mond_20%.jpg') / 255.
    abstracted = abstraction(im)

    imageio.imsave('moon_lowR.png', abstracted)
