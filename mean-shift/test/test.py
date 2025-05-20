import torch as th
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import imageio
import numpy as np

# Define the Epanechnikov kernel function
def epanechnikov_kernel(x, h):
    C = 2 / (3 * h)
    return C * (1 - (x ** 2 / h ** 2)) * (th.abs_(x) <= h)

# Define the mean shift algorithm
def mean_shift(data, bandwidth=0.5, max_iter=100, tol=1e-6):
    # Initialize the cluster centers using a subset of the data
    cluster_centers = data[th.randperm(data.size(0))[:data.size(0) // 10]]

    # Iterate until convergence or maximum number of iterations
    for _ in range(max_iter):
        # Compute the distances between each point and the cluster centers
        print(data.shape)
        print(cluster_centers.shape)

        dist = data[:, None] - cluster_centers[None, :]

        # Compute the density at each point using the Epanechnikov kernel
        density = epanechnikov_kernel(dist, bandwidth).sum(-1,)

        # HERE IS THE FUCK UP SOMEWHERE.
        # The previous centers are not stored properly and it becomes 0 at the second iteration of this loop.
        # so yea i still cant solve it


        # Save the current value of the cluster centers
        cluster_centers_prev = cluster_centers.clone()

        # Update the cluster centers using the modes of the density function
        cluster_centers = data[density.argmax(0,)]


        # Check for convergence
        if th.abs_(cluster_centers - cluster_centers_prev).max() < tol:
            break

    # Assign each point to the nearest cluster center
    labels = dist.argmin(-1)
    return cluster_centers, labels

# It is highly recommended to set up pytorch to take advantage of CUDA GPUs

# Choose the size of the image here. Prototyping is easier with 128.
M = 128
# The reference implementation works in chunks. Set this as high as possible
# while fitting the intermediary calculations in memory.
simul = M ** 2 // 1

im = th.from_numpy(imageio.imread(f'./{M}.png') / 255.)

# Visualization
_, ax = plt.subplots(1, 2)
ax[0].imshow(im.cpu().numpy())
artist = ax[1].imshow(im.cpu().numpy())

for zeta in [1, 4]:
    y, x = th.meshgrid(
        th.linspace(-zeta, zeta, M),
        th.linspace(-zeta, zeta, M),
        indexing='xy'
    )
    features = th.cat((im, y[..., None], x[..., None]), dim=-1).reshape(-1, 5)
    shifted = features.clone()
    s = shifted.shape
    for h in [0.1, 0.3]:
        to_do = th.arange(M ** 2)
        while len(to_do):
            chunk = shifted[to_do[:simul]]

            # Run the mean shift algorithm on each channel of the image
            segmented_image = th.zeros_like(chunk)
            for c in range(3):
                # Select the current channel of the image
                channel = chunk[:, c]

                # Run the mean shift algorithm on the current channel
                cluster_centers, labels = mean_shift(channel, bandwidth=h)

                # Assign each pixel in the current channel to the nearest cluster center
                segmented_image[:, c] = cluster_centers[labels]

            # Update the shifted image with the segmented image
            shifted[to_do[:simul]] = segmented_image

            # Update the list of pixels to be processed
            to_do = to_do[simul:]

            # Update the visual representation of the shifted image
            artist.set_data(shifted.view(M, M, 5).cpu()[:, :, :3])
            plt.pause(0.01)

    # Save the segmented image to the root folder
    th.save(shifted.view(M, M, 5).cpu()[:, :, :3], 'segmented_image.png')
