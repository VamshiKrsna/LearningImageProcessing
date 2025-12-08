import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ============================================================
# 1. Utility: load RGB image as tensor
# ============================================================

def load_rgb_image(path, target_size=None, device="cpu"):
    img = Image.open(path).convert("RGB")
    if target_size is not None:
        img = img.resize(target_size, Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0  # (H, W, 3)
    arr = np.transpose(arr, (2, 0, 1))             # (3, H, W)
    tensor = torch.from_numpy(arr).unsqueeze(0)    # (1, 3, H, W)
    return tensor.to(device)


# ============================================================
# 2. Model with MANUAL cross-correlation
# ============================================================

class RGBDerivativeConv(nn.Module):
    def __init__(self):
        super().__init__()

        # 3 filters, each 3x3, one per channel
        # shape: (out_channels=3, in_channels_per_group=1, kH=3, kW=3)
        self.filters = nn.Parameter(torch.zeros(3, 1, 3, 3), requires_grad=True)

        # horizontal derivative kernel [-1 0 1] (stacked as 3x3)
        base_kernel = torch.tensor([
            [-1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0]
        ])  # (3, 3)

        # initialize each channel's filter with the same derivative kernel
        with torch.no_grad():
            self.filters[0, 0, :, :] = base_kernel          # Red derivative
            self.filters[1, 0, :, :] = base_kernel.clone()  # Green derivative
            self.filters[2, 0, :, :] = base_kernel.clone()  # Blue derivative

    def forward(self, x):
        """
        x: (B, 3, H, W)
        returns: (B, 3, H, W)  -- correlation maps
        """
        B, C, H, W = x.shape
        assert C == 3, "Expected 3-channel RGB input"

        pad = 1
        kH, kW = 3, 3

        # zero padding: left,right,top,bottom
        x_padded = F.pad(x, (pad, pad, pad, pad), mode="constant", value=0.0)
        # output tensor
        out = torch.zeros(B, 3, H, W, device=x.device, dtype=x.dtype)

        # ------- manual cross-correlation -------
        # Each out[b, c, i, j] = dot( kernel[c], patch_of_image[b,c] )
        for b in range(B):
            for c in range(3):
                kernel = self.filters[c, 0]        # (3, 3)
                k_vec = kernel.reshape(-1)        # (9,)
                for i in range(H):
                    for j in range(W):
                        patch = x_padded[b, c, i:i+kH, j:j+kW]   # (3, 3)
                        # use reshape instead of view to avoid contiguity issues
                        p_vec = patch.reshape(-1)                # (9,)
                        # this is the correlation between filter and image patch
                        corr = torch.dot(p_vec, k_vec)
                        out[b, c, i, j] = corr

        return out


# ============================================================
# 3. Loss = Squared Error (MSE)
# ============================================================

def squared_error_loss(pred, target):
    return torch.mean((pred - target) ** 2)


# ============================================================
# 4. Training loop with MANUAL optimization
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) Load image
img_path = "/content/rgb_balls.webp"   # <-- change to your image path
img_tensor = load_rgb_image(img_path, target_size=(256, 256), device=device)
# img_tensor: (1, 3, H, W)

# 2) Model
model = RGBDerivativeConv().to(device)

# 3) Target feature maps (dummy: all zeros, same shape as input)
with torch.no_grad():
    target_feats = torch.zeros_like(img_tensor)   # (1, 3, H, W)

lr = 1e-3
num_steps = 10   # keep small just for demo

for step in range(num_steps):
    # Forward: compute correlation feature maps
    pred_feats = model(img_tensor)    # (1, 3, H, W)

    # Loss
    loss = squared_error_loss(pred_feats, target_feats)

    # Backprop to get gradients w.r.t. filters
    loss.backward()

    # Manual gradient descent update: w <- w - lr * grad
    with torch.no_grad():
        model.filters -= lr * model.filters.grad

    # Clear gradients for next iteration
    model.filters.grad.zero_()

    print(f"step={step}, loss={loss.item():.6f}")

print("\nFinal loss:", loss.item())
print("Filter weights:\n", model.filters.data)


# ============================================================
# 5. Visualize feature maps with matplotlib
# ============================================================

# One more forward pass with the trained filters
with torch.no_grad():
    feature_maps = model(img_tensor)          # (1, 3, H, W)
feature_maps = feature_maps.cpu().squeeze(0)  # (3, H, W)

def show_feature_map(fm, title):
    fm = fm.numpy()
    # normalize to [0,1] for display
    fm_min, fm_max = fm.min(), fm.max()
    if fm_max > fm_min:
        fm_norm = (fm - fm_min) / (fm_max - fm_min)
    else:
        fm_norm = fm - fm_min
    plt.imshow(fm_norm, cmap="gray")
    plt.axis("off")
    plt.title(title)

plt.figure(figsize=(10, 3))

plt.subplot(1, 3, 1)
show_feature_map(feature_maps[0], "Red derivative FM")

plt.subplot(1, 3, 2)
show_feature_map(feature_maps[1], "Green derivative FM")

plt.subplot(1, 3, 3)
show_feature_map(feature_maps[2], "Blue derivative FM")

plt.tight_layout()
plt.show()
