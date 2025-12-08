# GOAL : 
# NO PYTORCH
"""
Goal is to create a CNN from scratch, where there is an input image 
with 3 layers - RGB
We need  3 distinct filters, one for Red, one for Green, one for Blue
these filters differently output a feature map of each color filter scheme
the goal is to also compute a loss function as every prediction and ground truth terms are of derivative type. 
"""

from PIL import Image

img_path = "rgb_balls.webp"

test_img = Image.open(img_path).convert("RGB") 

# print(img)

R,G,B = test_img.split() 

R.save("red_channel.png")
G.save("green_channel.png")
B.save("blue_channel.png")

print("Saved red_channel.png, green_channel.png, blue_channel.png")

import numpy as np 

# lets convert each of the color layer RGB layers to numpy array of numbers : 
R = np.array(R, dtype=np.float32) 
G = np.array(G, dtype=np.float32) 
B = np.array(B, dtype=np.float32) 

print(R, G, B)

# lets plot these number representation of images using plt : 
import matplotlib.pyplot as plt 

# plt.figure(figsize=(12,4)) 

# plt.subplot(1,3,1)
# plt.title("Red Channel Heatmap") 
# plt.imshow(R, cmap = "hot") 
# plt.colorbar() 

# plt.subplot(1,3,2)
# plt.title("Green Channel Heatmap") 
# plt.imshow(G, cmap = "hot") 
# plt.colorbar() 

# plt.subplot(1,3,3)
# plt.title("Blue Channel Heatmap") 
# plt.imshow(B, cmap = "hot") 
# plt.colorbar() 

# plt.tight_layout()
# plt.savefig("heatmaps.png")
# plt.show()

# That's so cool, might as well save this plot

# Now we will be creating three different kernels one for each layer : 
filter_R = np.random.randn(3, 3).astype(np.float32)
filter_G = np.random.randn(3, 3).astype(np.float32)
filter_B = np.random.randn(3, 3).astype(np.float32)

print(filter_R, filter_B, filter_G)

# We have three distinct filters for our CNN, 
# lets get them rolling : 

# implementing convolution : 
def conv2d_single_channel(channel, kernel): 
    H, W = channel.shape
    k = kernel.shape[0] 
    pad = k//2 # 1 for 3x3 
    padded = np.pad(channel, pad, mode = "constant") 
    out = np.zeros((H,W),dtype=np.float32)
    for i in range(H): 
        for j in range(W): 
            region = padded[i:i+k, j:j+k] # Extracting a region of the image
            out[i,j] = np.sum(region * kernel) # dot prod = convolution 
    return out 

feat_R = conv2d_single_channel(R, filter_R)
feat_G = conv2d_single_channel(G, filter_R)
feat_B = conv2d_single_channel(B, filter_R)

