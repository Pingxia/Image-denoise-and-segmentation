import io_data
import numpy as np
from scipy.stats import norm
import cv2


def sampler(input, output, c, J, lamda, n_iter):
    _, noisy_img = io_data.read_data(input, True)
    # Convert to shape [rows, cols]
    noisy_img = np.asarray(noisy_img).reshape(noisy_img.shape[0:2])
    noisy_img[noisy_img < 128] = -1 # Use comparision for float.
    noisy_img[noisy_img > 128] = 1

    # Initialize mean value.
    mean = np.zeros(noisy_img.shape)

    L_1 = norm.logpdf(noisy_img, 1, c*c)
    L_neg1 = norm.logpdf(noisy_img, -1, c*c)

    for _ in range(n_iter):
        updated_mean = np.zeros(noisy_img.shape)

        for i in range(noisy_img.shape[0]):
            for j in range(noisy_img.shape[1]):
                nbr = 0
                if i > 0:
                    nbr += mean[i - 1][j]
                if i < noisy_img.shape[0] - 1:
                    nbr += mean[i + 1][j]
                if j > 0:
                    nbr += mean[i][j - 1]
                if j < noisy_img.shape[1] - 1:
                    nbr += mean[i][j + 1]

                updated_mean[i][j] = lamda * mean[i][j] + \
                                     (1 - lamda) * np.tanh(nbr * J + 0.5 * (L_1[i][j] - L_neg1[i][j]))

        mean = updated_mean

    # Restore the graph.
    noisy_img[mean>0] = 255
    noisy_img[mean<0] = 0
    noisy_img = np.expand_dims(noisy_img, 2)

    # Save the data.
    cv2.imwrite(output, noisy_img)

# Need to place this file and io_data.py in code/ folder

input = ["../a1/1_noise.txt", "../a1/2_noise.txt", "../a1/3_noise.txt", "../a1/4_noise.txt"]
output = ["../output/a1/1_denoise_variational.png", "../output/a1/2_denoise_variational.png", "../output/a1/3_denoise_variational.png", "../output/a1/4_denoise_variational.png"]

for i in range(len(input)):
    sampler(input[i], output[i], c = 1.414, J = 1, lamda = 0.5, n_iter = 15)