import numpy as np
import sys
import imageio
from scipy.ndimage.filters import gaussian_filter
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

import sys

sys.path.insert(0, 'math_tools.py')
import math_tools

from scipy.ndimage import convolve
from time import perf_counter


def g(nabla_u, lamda, eigenvalues):
    return np.exp(-(eigenvalues[0] - eigenvalues[1]) ** 2 / (2 * (gamma ** 2)))


def diff_tens(
        u: np.ndarray,
        sigma_g: float,
        sigma_u: float,
        alpha: float,
        gamma: float,
        nabla: sp.csr_matrix,
        mode: str,
):
    global diffusion_tensor
    tic_1 = perf_counter()
    # Initialize an array to store the diffusion tensors
    diffusion_tensors = np.empty((u.shape[0], u.shape[1], 2, 2))
    dx, dy = (nabla @ u.ravel()).reshape(2, *u.shape)
    # calculate the intermediate variables u_x_tilde and u_y_tilde
    u_x_tilde = sigma_u * dx * u
    u_y_tilde = sigma_u * dy * u

    # calculate the structure tensor for this pixel
    structure_tensor = np.stack(
        [u_x_tilde ** 2, u_x_tilde * u_y_tilde, u_x_tilde * u_y_tilde, u_y_tilde ** 2], axis=-1)
    structure_tensor = structure_tensor.reshape(u.shape[0], u.shape[1], 2, 2)
    structure_tensor_filtered = gaussian_filter(structure_tensor, sigma_g)
    print(structure_tensor_filtered)
    tac_1 = perf_counter()
    print(f"init and structure tensor in {tac_1 - tic_1:0.4f} seconds")

    tic = perf_counter()
    # Loop over every pixel in the image
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            # calulate the eigenvalues and eigenvectors respectively
            eigenvalues, eigenvectors = np.linalg.eigh(structure_tensor_filtered[i, j])

            if mode == 'ced':
                lambda_1 = alpha
                g_eig = np.exp(-(eigenvalues[0] - eigenvalues[1]) ** 2 / (2 * (gamma ** 2)))
                lambda_2 = alpha + (1 - alpha) * (1 - g_eig)
                v1 = eigenvectors[:, 0]
                v2 = eigenvectors[:, 1]
                v = np.column_stack((v1, v2))
                l = np.diag((lambda_1, lambda_2))
                diffusion_tensor = v @ l @ np.linalg.inv(v)
                # diag_diff_tens = np.diag(diffusion_tensor)
            elif mode == 'eed':
                lambda_1 = (1 + ((eigenvalues[0] / (gamma ** 2)) ** (-1 / 2)))
                lambda_2 = 1
                v1 = eigenvectors[:, 0]
                v2 = eigenvectors[:, 1]
                v = np.column_stack((v1, v2))
                l = np.diag((lambda_1, lambda_2))
                diffusion_tensor = v @ l @ np.linalg.inv(v)
                # diag_diff_tens = np.diag(diffusion_tensor)
            else:
                raise ValueError('Invalid Mode chosen')


            # add diffusion tensor to all
            diffusion_tensors[i, j] = diffusion_tensor
            #This is fomular 9 from the script without the diag part.
            # i just dont get how i am supposed to output this.
            #the following few lines are my last attempts


    diags = [sp.diags(diag) for diag in
            (diffusion_tensor[0], diffusion_tensor[1])]
    print(diags)
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
        'alpha': 0.0005,
        'gamma': 1e-4,
        'tau': 5.,
        'T': 20.,
    },
    'eed': {
        'sigma_g': 0.,
        'sigma_u': 10,
        'alpha': 0.,
        'gamma': 1e-4,
        'tau': 1.,
        'T': 3.,
    },
}

inputs = {
    'ced': 'starry_night.png',
    'eed': 'fir.png',
}

if __name__ == "__main__":
    mode = sys.argv[1]
    input = imageio.imread(inputs[mode]) / 255.
    output = nonlinear_anisotropic_diffusion(
        input, **params[mode], mode=mode
    )
    imageio.imsave(f'./__{mode}_out_me_Day2.png', (output * 255).astype(np.uint8))
