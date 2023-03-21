import numpy as np
import imageio
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

import timeit


def mssim(
            x: np.ndarray,
            y: np.ndarray,
    ) -> float:
        # Standard choice for the parameters
        K1 = 0.01
        K2 = 0.03
        sigma = 1.5
        truncate = 3.5
        m = 1
        C1 = (K1 * m) ** 2
        C2 = (K2 * m) ** 2

        x = x.astype(np.float64)
        y = y.astype(np.float64)

        # radius size of the local window (needed for
        # normalizing the standard deviation)
        r = int(truncate * sigma + 0.5)
        win_size = 2 * r + 1
        # use these arguments for the gaussian filtering
        # e.g.
        filter_args = {
            'sigma': sigma,
            'truncate': truncate
        }
        filtered = gaussian_filter(x, **filter_args)

        # Implement Eq. (9) from assignment sheet
        # S should be an "image" of the SSIM evaluated for a window
        # centered around the corresponding pixel in the original input image

        S = np.ones_like(x)

        mu1 = gaussian_filter(x, **filter_args)  # valid
        mu2 = gaussian_filter(y, **filter_args)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = gaussian_filter(x ** 2, **filter_args) - mu1_sq
        sigma2_sq = gaussian_filter(y ** 2, **filter_args) - mu2_sq
        sigma12 = gaussian_filter(x * y, **filter_args) - mu1_mu2

        S = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                         (sigma1_sq + sigma2_sq + C2))
        
        
        
        
        # crop to remove boundary artifacts, return MSSIM
        pad = (win_size - 1) // 2
        return S[pad:-pad, pad:-pad].mean()


def psnr(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    # Implement Eq. (2) without for loops

    diff_array = np.subtract(x,y)
    squared_errors = np.square(diff_array)
    mse = np.sum(squared_errors) / float(x.shape[0] * x.shape[1])
    psnr = 10 * np.log10(1/mse)
    return psnr




def psnr_for(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    # Implement Eq. (2) using for loops

    mse = 0.
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[1]):
            error = x[i][j] - y[i][j]
            error_sq = error * error
            mse += error_sq

    mse /= float(x.shape[0] * x.shape[1])
    psnr = 10 * np.log10(1/mse)
    return psnr


def interpolation_error():
    x = imageio.imread('./girl.png') / 255.
    shape_lower = (x.shape[0] // 2, x.shape[1] // 2)
    # downsample image to half the resolution
    # and successively upsample to the original resolution
    # using no nearest neighbor, linear and cubic interpolation
    nearest, linear, cubic = [
        resize(resize(
            x, shape_lower, order=order, anti_aliasing=False
        ), x.shape, order=order, anti_aliasing=False)
        for order in [0, 1, 3]
    ]

    for label, rescaled in zip(
        ['nearest', 'linear', 'cubic'],
        [nearest, linear, cubic]
    ):
        print(label)
        print(mssim(x, rescaled))
        m = round(float(mssim(x, rescaled)), 2)
        mstr = str(m)

        
        start1 = timeit.default_timer()
        print(psnr(x, rescaled))
        stop1 = timeit.default_timer()
        
        f = round(float(psnr(x, rescaled)), 2)
        fstr = str(f)
                
        start2 = timeit.default_timer()
        print(psnr_for(x, rescaled))
        stop2 = timeit.default_timer()
        
        print('psnr Time: ', stop1 - start1) 
        print('psnr_forTime: ', stop2 - start2) 
        
        	

        
        
        
        
    


#Plotting

    fig, (axNear, axLin, axCub) = plt.subplots(1, 3)
    axNear.imshow(nearest)
    axNear.set_title('nearest')
    axLin.imshow(linear)
    axLin.set_title('linear')
    axCub.imshow(cubic)
    axCub.set_title('cubic')

    plt.savefig("example12s.png")

if __name__ == '__main__':
    interpolation_error()
