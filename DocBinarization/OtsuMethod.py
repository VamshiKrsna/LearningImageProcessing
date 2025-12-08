import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
import os

def load_grayscale(path):
    # load an image in grayscale 
    img = Image.open(path).convert("L")
    return np.array(img, dtype = np.uint8)

def otsu_binarization(input_img_path):
    """
    this function binarizes an image if you give its path
    """
    img = load_grayscale(input_img_path)
    
    # pixel intensities - histogram
    hist, bin_edges = np.histogram(img.ravel(), bins = 256, range= (0,256))
    
    # total number of pixels
    total_pixels = img.size 
    
    # normalized histogram
    prob = hist / total_pixels

    variances = [] # Track inter class variance

    for k in range(0,256): 
        # first k elements = C0
        # k+1 to rest elements = C1
        bg_class = hist[:k+1]
        fg_class = hist[k+1:]
    
        # Calculate Class Probabilities of bg_class and fg_class 
        w0 = bg_class.sum() / hist.sum()
        w1 = fg_class.sum() / hist.sum() 

        if bg_class.sum() == 0 or fg_class.sum() == 0:
            variances.append(0)
            continue

        # Calculate mean of C0 and C1
        mu0 = np.sum(np.arange(0, k+1) * bg_class) / bg_class.sum()
        mu1 = np.sum(np.arange(k+1, 256) * fg_class) / fg_class.sum()

        # base class variance 
        sigma_b_squared = w0 * w1 * ((mu0 - mu1) ** 2)
        # print(sigma_b_squared)
        variances.append(sigma_b_squared)

    best_k = np.argmax(variances)
    print("Optimal Threshold k for image is : ", best_k)

    # Binarized Image 
    binary = (img > best_k).astype(np.uint8) 
    plt.imshow(binary, cmap = "gray")
    plt.show()




if __name__ == "__main__":
    otsu_binarization("../Datasets/dataset/1.bmp")
