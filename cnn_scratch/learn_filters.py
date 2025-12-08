import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load image as tensor

def load_rgb_image(path, target_size=None, device="cpu"):
    img = Image.open(path).convert("RGB")
    if target_size is not None:
        # Use Image.LANCZOS () for high-quality downsampling -> it is used for finding eigen values and vectors clearly.
        img = img.resize(target_size, Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0  # (H, W, 3) --> 3 Channels of HxW
    arr = np.transpose(arr, (2, 0, 1))              # (3, H, W) --> Transpose the channels
    tensor = torch.from_numpy(arr).unsqueeze(0)     # (1, 3, H, W) --> Make Tensor
    return tensor.to(device)


# Independent kernel convolution

class IndependentRGBConv(nn.Module):
    """
    A 2D conv layer that applies a different 3x3 filter
    to each of the 3 input channels (R, G, B) independently.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=3, 
            out_channels=3, 
            kernel_size=3, 
            stride=1, 
            padding=1, # 1-pixel padding to ensure output map size equals input size
            groups=3,  # 3 groups means 3 separate 1-channel convolutions
            bias=False
        )

        # Initialize filters with randoms
        horizontal_deriv_kernel = torch.tensor([
            [-1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0) # (1, 1, 3, 3)

        # Initialize all three independent kernels with the derivative kernel
        initial_weights = torch.cat([horizontal_deriv_kernel] * 3, dim=0) # (3, 1, 3, 3)
        with torch.no_grad():
            self.conv.weight.copy_(initial_weights)

    def forward(self, x):
        """x: (B, 3, H, W) -> returns: (B, 3, H, W) feature maps"""
        return self.conv(x)


# ============================================================
# 3. Loss Function: Derivative-type loss (e.g., L2 on gradients)
# ============================================================

def squared_error_loss(pred, target):
    """Simple Mean Squared Error (MSE) loss."""
    return torch.mean((pred - target) ** 2)

# ============================================================
# 4. Training setup and loop
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
img_path = "rgb_balls.webp" # Make sure this path is correct

# 1) Load and prepare image
# Normalize the input image is already handled in load_rgb_image
img_tensor = load_rgb_image(img_path, target_size=(128, 128), device=device) 

# 2) Model, Target, and Optimizer
model = IndependentRGBConv().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_steps = 100

# The Goal: Learn filters that produce an all-zero output (or a specific derivative)
# Setting the ground truth as all zeros for simplicity (making feature maps disappear)
with torch.no_grad():
    # Target is (1, 3, H, W) of zeros
    target_feats = torch.zeros_like(img_tensor).to(device) 

print(f"Starting training on {device}...")
print(f"Initial Filters (R, G, B):\n{model.conv.weight.data}")

# --- Training Loop ---
for step in range(num_steps):
    # Set gradients to zero
    optimizer.zero_grad() 

    # Forward pass
    pred_feats = model(img_tensor) 

    # Loss calculation
    loss = squared_error_loss(pred_feats, target_feats)

    # Backward pass (Autograd computes all gradients)
    loss.backward()

    # Optimizer updates the weights (filters)
    optimizer.step()

    if (step + 1) % 10 == 0:
        print(f"Step {step+1}/{num_steps}, Loss: {loss.item():.6f}")

print("\n--- Training Complete ---")
final_loss = loss.item()
print(f"Final Loss: {final_loss:.6f}")
print("Final Filter Weights (R, G, B):\n", model.conv.weight.data)


# ============================================================
# 5. Visualize Results
# ============================================================

# One final forward pass with the learned filters
model.eval() # Set model to evaluation mode
with torch.no_grad():
    feature_maps = model(img_tensor).cpu().squeeze(0) # (3, H, W)

def show_feature_map(fm, title, ax):
    """Helper to display a single feature map."""
    fm = fm.numpy()
    # Normalize to [0,1] for display purposes only
    fm_min, fm_max = fm.min(), fm.max()
    if fm_max > fm_min:
        fm_norm = (fm - fm_min) / (fm_max - fm_min)
    else:
        fm_norm = fm - fm_min
    
    ax.imshow(fm_norm, cmap="gray")
    ax.axis("off")
    ax.set_title(title, fontsize=10)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle(f'Feature Maps after {num_steps} Steps (Final Loss: {final_loss:.4f})', fontsize=12)

show_feature_map(feature_maps[0], "Learned Red Filter Output", axes[0])
show_feature_map(feature_maps[1], "Learned Green Filter Output", axes[1])
show_feature_map(feature_maps[2], "Learned Blue Filter Output", axes[2])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()