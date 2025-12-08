import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"
img_path = "rgb_balls.webp" 
target_size = (128, 128)
lr = 1e-2
num_steps = 100

def load_rgb_image(path, target_size=None, device="cpu"):
    # load and resize the rgb image to a (1, 3, H, W) tensor
    try:
        img = Image.open(path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at {path}. Please update img_path.")
        return None
        
    if target_size is not None:
        # Image.LANCZOS for high-quality downsampling (better eigen values and vectors)
        img = img.resize(target_size, Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
        
    arr = np.array(img).astype(np.float32) / 255.0  # (H, W, 3) and normalize to [0, 1]
    arr = np.transpose(arr, (2, 0, 1))              # (3, H, W) --> transposition
    tensor = torch.from_numpy(arr).unsqueeze(0)     # (1, 3, H, W) --> tensor for three channels combined
    return tensor.to(device)


# Standard Multi-Channel Convolution

class MutualInformationConv(nn.Module):
    # standard 2d CNN for entire information extraction 
    def __init__(self):
        super().__init__()
        # Standard convolution:
        # in_channels=3: The filter looks at R, G, and B input simultaneously.
        # out_channels=1: It produces ONE combined feature map.
        # groups=1 (default): Enables full connection between all input channels and output channels.
        self.conv = nn.Conv2d(
            in_channels=3, 
            out_channels=1, 
            kernel_size=3, 
            padding=1, # make sure output map size equals input size
            bias=False
        )

        # A single filter cube (1,3,3,3)
        print(f"Initialized filter shape: {self.conv.weight.shape}")

        # Initialize the 3x3x3 filter cube (1, 3, 3, 3) with zeros
        with torch.no_grad():
            self.conv.weight.zero_()
            base_kernel = torch.tensor([
                [-1.0, 0.0, 1.0],
                [-1.0, 0.0, 1.0],
                [-1.0, 0.0, 1.0]
            ], dtype=torch.float32)
            self.conv.weight[0, 1, :, :].copy_(base_kernel)


    def forward(self, x):
        #x: (B, 3, H, W) -> returns: (B, 1, H, W) combined feature map
        return self.conv(x) 

# Squared Error (MSE)

def squared_error_loss(pred, target):
    """Mean Squared Error (MSE) loss."""
    return torch.mean((pred - target) ** 2)


# Training loop

def train_and_visualize():
    # Runs the training loop and visualizes the results
    # Load image
    img_tensor = load_rgb_image(img_path, target_size=target_size, device=device)
    if img_tensor is None:
        return

    # Model, Target, and Optimizer
    model = MutualInformationConv().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Target feature map (all zeros, single channel)
    H, W = img_tensor.shape[2:]
    with torch.no_grad():
        # Target must match model output shape: (1, 1, H, W)
        target_feats = torch.zeros(1, 1, H, W, device=device) 
    
    print(f"\nStarting training on {device} for {num_steps} steps...")

    #Training Loop 
    for step in range(num_steps):
        optimizer.zero_grad() 

        # Forward pass: Single feature map is produced
        pred_feats = model(img_tensor) 

        # Loss calculation
        loss = squared_error_loss(pred_feats, target_feats)

        # Backpropagation and Update
        loss.backward()
        optimizer.step()

        if (step + 1) % 20 == 0:
            print(f"Step {step+1}/{num_steps}, Loss: {loss.item():.6f}")

    final_loss = loss.item()
    print("\n--- Training Complete ---")
    print(f"Final Loss: {final_loss:.6f}")
    
    # Display final learned filter (weights)
    final_weights = model.conv.weight.data.cpu().squeeze().numpy()
    print(f"\nFinal Learned Filter (1x3x3x3):\n {final_weights}")

    # ============================================================
    # 5. Visualize Feature Maps and Learned Kernels
    # ============================================================
    model.eval() 
    with torch.no_grad():
        feature_map = model(img_tensor).cpu().squeeze().numpy() # (H, W)

    
    # --- Visualization ---
    plt.figure(figsize=(12, 5))

    # 1. Combined Feature Map
    plt.subplot(1, 2, 1)
    fm_min, fm_max = feature_map.min(), feature_map.max()
    if fm_max > fm_min:
        fm_norm = (feature_map - fm_min) / (fm_max - fm_min)
    else:
        fm_norm = feature_map - fm_min
    
    plt.imshow(fm_norm, cmap="gray")
    plt.axis("off")
    plt.title(f"Combined Feature Map (Loss: {final_loss:.4f})")

    # 2. Learned Filter Weights (Visualized as three 3x3 matrices)
    plt.subplot(1, 2, 2)
    
    # Create a subplot for the kernel visualization
    fig_kernel, axes_kernel = plt.subplots(1, 3, figsize=(10, 3))
    fig_kernel.suptitle("Learned 3x3 Kernels (Slices of the 3x3x3 Filter Cube)", fontsize=10)
    
    channel_names = ["Red Slice", "Green Slice", "Blue Slice"]
    
    # Use final_weights: (3, 3, 3) numpy array
    for i in range(3):
        ax = axes_kernel[i]
        kernel_slice = final_weights[i]
        
        # Display the 3x3 kernel slice as a heatmap
        im = ax.imshow(kernel_slice, cmap="viridis") 
        ax.set_title(channel_names[i])
        ax.axis("off")
        
        # Annotate values for clarity
        for r in range(3):
            for c in range(3):
                ax.text(c, r, f"{kernel_slice[r, c]:.2f}", 
                        ha="center", va="center", color="white" if kernel_slice[r,c] < kernel_slice.mean() else "black", fontsize=8)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_and_visualize()