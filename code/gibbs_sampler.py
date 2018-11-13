import io_data
import numpy as np
from scipy.stats import norm
import math
import cv2

def sampler(input, output, var, J, n_iter):
    _, noisy_img = io_data.read_data(input, True)
    # Convert to shape [rows, cols]
    noisy_img = np.asarray(noisy_img).reshape(noisy_img.shape[0:2])
    noisy_img[noisy_img < 128] = -1 # Use comparision for float.
    noisy_img[noisy_img > 128] = 1

    norm_1 = norm.pdf(noisy_img, 1, var)
    norm_negative_1 = norm.pdf(noisy_img, -1, var)

    for iter in range(n_iter):
        for i in range(noisy_img.shape[0]):
            for j in range(noisy_img.shape[1]):
                nbr = 0
                if i > 0:
                    nbr += noisy_img[i-1][j]
                if i < noisy_img.shape[0] - 1:
                    nbr += noisy_img[i+1][j]
                if j > 0:
                    nbr += noisy_img[i][j-1]
                if j<noisy_img.shape[1] - 1:
                    nbr += noisy_img[i][j+1]

                potential_1 = math.exp(J * nbr)
                potential_negative_1 = math.exp(-J * nbr)

                prob_1 = norm_1[i][j] * potential_1
                prob_negative_1 = norm_negative_1[i][j] * potential_negative_1

                # Normalize.
                prob_1 /= (prob_1 + prob_negative_1)

                noisy_img[i][j] = (np.random.rand() < prob_1) * 2 - 1

    noisy_img[noisy_img>0] = 255
    noisy_img[noisy_img<0] = 0
    noisy_img = np.expand_dims(noisy_img, 2)

    # Save the data.
    cv2.imwrite(output, noisy_img)

# Need to place this file and io_data.py at code/ folder

input = ["../a1/1_noise.txt", "../a1/2_noise.txt", "../a1/3_noise.txt", "../a1/4_noise.txt"]
output = ["../output/a1/1_denoise_gibbs.png", "../output/a1/2_denoise_gibbs.png", "../output/a1/3_denoise_gibbs.png", "../output/a1/4_denoise_gibbs.png"]

for i in range(len(input)):
    sampler(input[i], output[i], var=2, J=1, n_iter=15)