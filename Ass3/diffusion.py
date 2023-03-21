import numpy as np
import sys
import imageio
from scipy.ndimage.filters import gaussian_filter
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from skimage import color
import matplotlib.pyplot as plt

import sys

sys.path.insert(0, 'math_tools.py')
import math_tools

from scipy.ndimage import convolve
from time import perf_counter

def diff_tens(
        u: np.ndarray,
        sigma_g: float,
        sigma_u: float,
        alpha: float,
        gamma: float,
        nabla: sp.csr_matrix,
        mode: str,
):
    # global diffusion_tensor
    tic_1 = perf_counter()
    # Initialize an array to store the diffusion tensors
    diffusion_tensors = np.zeros((u.shape[0], u.shape[1], 2, 2))
    u_tilde = gaussian_filter(u, sigma_u)

    u_x_tilde, u_y_tilde = (nabla @ u_tilde.ravel()).reshape(2, *u.shape)

    # Explicit eigenvalue decomposition (non functioning)
    #dx2 = np.gradient(u_x_tilde)[0]
    #dy2 = np.gradient(u_y_tilde)[1]
    #hessian = np.concatenate((dx2, dy2), axis=1)

    t1 = gaussian_filter(u_x_tilde ** 2, sigma_g)
    t2 = gaussian_filter(u_x_tilde * u_y_tilde, sigma_g)
    t3 = gaussian_filter(u_x_tilde * u_y_tilde, sigma_g)
    t4 = gaussian_filter(u_y_tilde ** 2, sigma_g)

    structure_tensor = np.stack(
        [t1, t2, t3, t4], axis=-1)
    structure_tensor_filtered = structure_tensor.reshape(u.shape[0], u.shape[1], 2, 2)

    tac_1 = perf_counter()
    print(f"init and structure tensor in {tac_1 - tic_1:0.4f} seconds")

    tic = perf_counter()
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            eigenvalues, eigenvectors = np.linalg.eigh(structure_tensor_filtered[i, j])
            #Explicit
            #eigenvalues, eigenvectors = np.linalg.eigh(hessian)
            if mode == 'ced':
                lambda_1 = alpha
                g_eig = np.exp(-(eigenvalues[0] - eigenvalues[1]) ** 2 / (2 * (gamma ** 2)))
                lambda_2 = alpha + (1 - alpha) * (1 - g_eig)
                v1 = eigenvectors[:, 0]
                v2 = eigenvectors[:, 1]
                v = np.array([[v1[0],v2[0]], [v1[1],v2[1]]])
                vt = np.array([[v1[0],v1[1]], [v2[0],v2[1]]])
                l = np.diag((lambda_2, lambda_1))
                diffusion_tensor = v @ l @ vt
            elif mode == 'eed':
                eigenvalues = [eigenvalues[1], eigenvalues[0]]
                lambda_1 = (1 + (eigenvalues[0] / (gamma ** 2))) ** (-0.5)
                lambda_2 = 1
                v1 = eigenvectors[:, 0]
                v2 = eigenvectors[:, 1]
                v = np.array([[v1[0],v2[0]], [v1[1],v2[1]]])
                vt = np.array([[v1[0],v1[1]], [v2[0],v2[1]]])
                l = np.diag((lambda_2, lambda_1))
                diffusion_tensor = v @ l @ vt
            else:
                raise ValueError('Invalid Mode chosen')

            diffusion_tensors[i, j] = diffusion_tensor

    D1 = diffusion_tensors[:,:,0,0].ravel()
    D2 = diffusion_tensors[:,:,1,1].ravel()
    D3 = diffusion_tensors[:,:,1,0].ravel()

    diags = [sp.diags(diag) for diag in (D1, D3, D2)]
    DD = sp.bmat([diags[:2], diags[1:]])

    tac = perf_counter()
    print(diffusion_tensors)
    print(diffusion_tensors.shape)
    print(f"calc diffusion tensor for whole timestep in {tac - tic:0.4f} seconds")
    return DD


def nonlinear_anisotropic_diffusion(
        image: np.ndarray,
        sigma_g: float,
        sigma_u: float,
        alpha: float,
        gamma: float,
        tau: float,
        T: float,
        mode: str,
):
    t = 0
    U_t = image.ravel()
    nabla = math_tools.spnabla_hp(*image.shape)
    id = sp.eye(U_t.shape[0], format="csc")
    tic_2 = perf_counter()

    while t < T:
        print(f'{t=}')
        D = diff_tens(
            U_t.reshape(image.shape), sigma_g, sigma_u,
            alpha, gamma, nabla, mode
        )
        U_t = spsolve(id + tau * nabla.T @ D @ nabla, U_t)
        t += tau
        tac_2 = perf_counter()
        print(f"solved in {tac_2 - tic_2:0.4f} seconds for tau= {t}")
    return U_t.reshape(image.shape)


params = {
    'ced': {
        'sigma_g': 1.5,
        'sigma_u': 0.7,
        'alpha': 1e-4,
        'gamma': 1e-4,
        'tau': 5.,
        'T': 100.,
    },
    'eed': {
        'sigma_g': 0.,
        'sigma_u': 2.,
        'alpha': 1e-5,
        'gamma': 1e-6,
        'tau': 2.,
        'T': 20.,
    },
}


inputs = {
    'ced': 'paul.png',
    'eed': 'NC.png',
}

if __name__ == "__main__":
    mode = sys.argv[1]
    input = imageio.imread(inputs[mode]) / 255.
    #input = color.rgba2rgb(input)
    #input = color.rgb2gray(input)
    output = nonlinear_anisotropic_diffusion(
         input, **params[mode], mode=mode
     )
    imageio.imsave(f'./__{mode}_paul_standart.png', (output * 255).astype(np.uint8))
