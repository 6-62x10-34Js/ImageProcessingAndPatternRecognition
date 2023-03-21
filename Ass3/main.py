import numpy as np
import sys
import imageio
from scipy.ndimage.filters import gaussian_filter
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import math_tools
import scipy.signal as sp
import matplotlib.pyplot as plt


def diffusion_tensor(
    u: np.ndarray,
    sigma_g: float,
    sigma_u: float,
    alpha: float,
    gamma: float,
    nabla: sp.csr_matrix,
    mode: str,
):

    u_x, u_y = (nabla @ u.ravel()).reshape(2, *u.shape)

    kx = np.array([
        [0., -1, 1],
    ])
    ky = np.array([
        [0., ],
        [-1, ],
        [1, ],
    ])
    grad_ss = (
        sp.correlate2d(u, kx, mode='same', boundary='symm'),
        sp.correlate2d(u, ky, mode='same', boundary='symm'),
    )

    assert np.allclose(u_x[0], grad_ss[0]) and np.allclose(u_y[1], grad_ss[1])

    plt.figure(2, figsize=(10, 10))
    plt.subplot(121)
    plt.imshow(u_x, cmap="gray")
    plt.subplot(122)
    plt.imshow(u_y, cmap="gray")


    return sp.eye(2 * u.size)


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
    while t < T:
        print(f'{t=}')
        D = diffusion_tensor(
            U_t.reshape(image.shape), sigma_g, sigma_u,
            alpha, gamma, nabla, mode
        )
        U_t = spsolve(id + tau * nabla.T @ D @ nabla, U_t)
        t += tau
    return U_t.reshape(image.shape)


params = {
    'ced': {
        'sigma_g': 1.5,
        'sigma_u': 0.7,
        'alpha': 0.0005,
        'gamma': 1e-4,
        'tau': 5.,
        'T': 100.,
    },
    'eed': {
        'sigma_g': 0.,
        'sigma_u': 10,
        'alpha': 0.,
        'gamma': 1e-4,
        'tau': 1.,
        'T': 10.,
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
    imageio.imsave(f'./__{mode}_out.png', (output * 255).astype(np.uint8))
