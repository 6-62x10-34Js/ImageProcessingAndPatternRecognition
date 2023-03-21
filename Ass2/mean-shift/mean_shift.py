import matplotlib.pyplot as plt
import imageio
import numpy as np
import torch as th


# Define the mean shift function
def mean_shift(features, bandwidth):
    shifted = features.clone()
    to_do = th.arange(features.shape[0])
    while len(to_do):
        chunk = shifted[to_do[:simul]]

        # Shift each sample to the density-weighted average of itself and its neighbors
        for j, sample in enumerate(chunk):
            neighbors = features[th.abs(features - sample) <= bandwidth]
            weighted_sum = th.sum(neighbors, dim=0)
            shifted[to_do[j]] = weighted_sum / th.numel(neighbors)

        # Check for convergence
        diff = th.abs_(chunk - shifted[to_do[:simul]])
        cond = th.any(diff > 1e-6, dim=1)

        to_do = to_do[cond]

    return shifted


# It is highly recommended to set up pytorch to take advantage of CUDA GPUs!
device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')

# Choose the size of the image here. Prototyping is easier with 128.
M = 128
# The reference implementation works in chunks. Set this as high as possible
# while fitting the intermediary calculations in memory.
simul = M ** 2 // 1

im = th.from_numpy(imageio.imread(f'./{M}.png') / 255.).to(device)

# Visualization
_, ax = plt.subplots(1, 2)
ax[0].imshow(im.cpu().numpy())
artist = ax[1].imshow(im.cpu().numpy())

for zeta in [1, 4]:
    y, x = th.meshgrid(
        th.linspace(-zeta, zeta, M, device=device),
        th.linspace(-zeta, zeta, M, device=device),
        indexing='xy'
    )
    features = th.cat((im, y[..., None], x[..., None]), dim=-1).reshape(-1, 5)
    shifted = features.clone()
    for h in [0.1, 0.3]:
        to_do = th.arange(M ** 2, device=device)
        while len(to_do):
            shifted = mean_shift(features, h)
            artist.set_data(shifted.view(M, M, 5).cpu()[:, :, :3])
            plt.pause(0.01)

        # Reference images were saved using this code.
        output = shifted.reshape(M, M, 5)[..., :3].clone().cpu().numpy()
        imageio.imsave(
            f'zeta_{zeta:1.1f}_h_{h:.2f}.png',
            output
        )
