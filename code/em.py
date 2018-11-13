from io_data import read_data, write_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as imageplt
import cv2
import sys
import warnings

'''
EM algorithm
input params: 
pixels - array of values, H - img height, W - img width, k - number of clusters
output:
segments - segments of original image
'''
warnings.filterwarnings("ignore")
def EM(pixels, H, W, k):
    n, c = pixels.shape # n: number of pixels, c: number of channels: 3
    (mu, sigma, pi) = init(pixels, n, c, k)
    R = np.zeros((n, k))
    
    loglikes = []
    loglike_prev = -np.infty

    num_step = 0
    while True:
        num_step = num_step + 1
        ## expectation step
        R, loglike = expectation(R, pixels, mu, sigma, pi, k)
        loglikes.append(loglike)
        ## maximization step
        (mu, sigma, pi) = maximization(R, pixels, mu, sigma, pi, k, n)
        ## check for convergence
        if (meet_convergence(loglike_prev, loglike)):
            break
        loglike_prev = loglike

    # plot_log_likehoods(loglikes)

    # Assign pixels to guassian with higher prob
    mask = np.full((H, W, 3), 0, dtype=np.uint8)
    foreground = np.full((H, W, 3), 0, dtype=np.float32)
    background = np.full((H, W, 3), 0, dtype=np.float32)
    for i in range(H):
        for j in range(W):
            idx = (i-1) * W + j
            pixel = pixels[idx]
            # print (pixel)
            # print (R[idx].shape)
            pixel_segment_id = np.argmax(R[idx])

            if (pixel_segment_id == 0):
                ## assign to foreground
                mask[i,j,] = [255,255,255]
                foreground[i,j,] = pixels[idx]
            else:
                ## assign to background
                mask[i,j,] = [0,0,0]
                background[i,j,] = pixels[idx]

    return (mask, foreground, background)

# Initialize
def init(pixels, n, c, k):
    ## mu
    mu = pixels[np.random.choice(n, k, False), :]
    ## covariance matrices
    sigma= [np.eye(c)] * k
    ## pi
    pi = [1./k] * k
    return (mu, sigma, pi)

'''
E-step: compute responsibility matrix
'''
def expectation(R, pixels, mu, sigma, pi, k):
    # print ("in expectation step.")
    for k in range(k):
        R[:, k] = pi[k] * gaussian(pixels, mu[k], sigma[k])

    # print (R.shape)
    loglike = np.sum(np.log(np.sum(R, axis = 1)))
    # print ("loglike updated: ")
    # print (loglike)
    R = R / np.sum(R, axis = 1)[:,None]
    R[np.isnan(R)]=0
    # print (np.isnan(R).any())
    # print ("R updated: ")
    # print (R)

    return R, loglike

'''
M-step: re-estimate mu, sigma of each gaussian, using the responsibility matrix
'''
def maximization(R, pixels, mu, sigma, pi, k, n):
    # print ("in maximization step")
    # print ("R: ")
    # print (R)
    N_k = np.sum(R, axis = 0)
    # print ("N_k: ")
    # print (N_k)
    for k in range(k):
        ## mu
        mu[k] = 1. / N_k[k] * np.sum(R[:, k] * pixels.T, axis = 1).T
        x_mu = np.matrix(pixels - mu[k])
        ## covariance
        sigma[k] = np.array(1 / N_k[k] * np.dot(np.multiply(x_mu.T,  R[:, k]), x_mu))
        ## pi
        pi[k] = N_k[k] / n 
    return (mu, sigma, pi)

'''
Check whether convergence is met
'''
def meet_convergence(prev, cur):
    # print ("in meet_convergence. prev: " + str(prev) + " cur: " + str(cur))
    return abs(cur - prev) <= CONVERGENCE_THRESHOLD

def gaussian(X, mu, sigma):
    n,c = X.shape
    # prob = np.zeros(n)
    # for i in range(n):
    #   prob[i] = multivariate_normal.pdf(X[i], mean=mu, cov=sigma)

    ## 1/sqrt(2*pi*sigma**2)
    left = np.linalg.det(sigma) ** -.5 ** (2 * np.pi) ** (-X.shape[1]/2.) 
    ## exp((x-mu)/sigma)
    right = np.exp(-.5 * np.einsum('ij, ij -> i', X - mu, np.dot(np.linalg.inv(sigma) , (X - mu).T).T ) )
    prob = left * right

    # print (prob.shape) 
    return prob

def print_params(mu, sigma, pi, R):
    print ("mu:")
    print (mu)
    print ("sigma")
    print (sigma)
    print ("pi:")
    print (pi)
    print ("R:")
    print (R)

def plot_log_likehoods(loglikes):
    # plot convergence of Q as we progress through EM
    plt.figure(1)
    plt.plot(loglikes)
    plt.xlabel("Number of Iterations")
    plt.ylabel("log like at E-step")
    plt.show()
    
'''
Main Function:
Run EM algorithm,
Assign each pixel to the gaussian (foreground VS background) with higher probability
'''
CONVERGENCE_THRESHOLD = 1
K_SEG = 2

# Place this file inside code/ folder
if __name__== "__main__":

    input_path = ["../a2/cow.txt", "../a2/fox.txt", "../a2/owl.txt", "../a2/zebra.txt"]
    output_path = ["../output/a2/cow", "../output/a2/fox", "../output/a2/owl", "../output/a2/zebra"]

    for i in range(len(input_path)):
        data, image = read_data(input_path[i], True)
        height, width, channel = image.shape

        # reshape into pixels, each has 3 channels (RGB)
        pixels = image.reshape((height * width, channel)) 
        mask, foreground, background = EM(pixels, height, width, K_SEG)

        # save result images
        imageplt.imsave(output_path[i] + '_mask.png', mask)
        cv2.imwrite(output_path[i] + '_seg1.png', (cv2.cvtColor(foreground, cv2.COLOR_Lab2BGR) * 255).astype(np.uint8))
        cv2.imwrite(output_path[i] + '_seg2.png', (cv2.cvtColor(background, cv2.COLOR_Lab2BGR) * 255).astype(np.uint8))
