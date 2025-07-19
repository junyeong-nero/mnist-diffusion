import torch
import torch.nn as nn

class MLP(nn.Module):
    """A simple MLP block."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding.
    """
    def __init__(self, img_size=32, patch_size=4, in_channels=1, embed_dim=256):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (self.img_size[1] // self.patch_size[1]) * (self.img_size[0] // self.patch_size[0])
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        if H != self.img_size[0] or W != self.img_size[1]:
            raise ValueError(f"Input image size ({H}x{W}) doesn't match model ({self.img_size[0]}x{self.img_size[1]}).")
        
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2) # (B, embed_dim, num_patches)
        x = x.transpose(1, 2) # (B, num_patches, embed_dim)
        return x


class DiTBlock(nn.Module):
    """
    A block from the Diffusion Transformer (DiT) model, incorporating adaptive LayerNorm.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(in_features=hidden_size, hidden_features=mlp_hidden_dim)
        
        # This single layer projects the conditioning signal to the 6 parameters
        # needed for the adaLN-Zero modulation.
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, D)
            c (torch.Tensor): Conditioning tensor of shape (B, D)
        
        Returns:
            torch.Tensor: Output tensor of shape (B, N, D)
        """
        # Generate scale and shift parameters from conditioning vector c
        # The DiT paper uses adaLN-Zero, which splits the output of a linear layer into 6 parts.
        mod_params = self.adaLN_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod_params.chunk(6, dim=1)

        # Reshape for broadcasting: (B, D) -> (B, 1, D)
        scale_msa = scale_msa.unsqueeze(1)
        shift_msa = shift_msa.unsqueeze(1)
        gate_msa = gate_msa.unsqueeze(1)
        scale_mlp = scale_mlp.unsqueeze(1)
        shift_mlp = shift_mlp.unsqueeze(1)
        gate_mlp = gate_mlp.unsqueeze(1)

        # Attention block with adaLN-Zero
        norm_x = self.norm1(x)
        norm_x_modulated = scale_msa * norm_x + shift_msa
        attn_out, _ = self.attn(norm_x_modulated, norm_x_modulated, norm_x_modulated)
        x = x + gate_msa * attn_out

        # MLP block with adaLN-Zero
        norm_x = self.norm2(x)
        norm_x_modulated = scale_mlp * norm_x + shift_mlp
        mlp_out = self.mlp(norm_x_modulated)
        x = x + gate_mlp * mlp_out

        return x

if __name__ == '__main__':
    # A simple test case for the DiTBlock
    
    # Parameters
    batch_size = 4
    img_size = 32
    patch_size = 4
    in_channels = 1
    hidden_size = 256 # This is the embedding dimension
    num_heads = 8

    # Create a dummy input image tensor
    dummy_image = torch.randn(batch_size, in_channels, img_size, img_size)
    
    # Create a dummy conditioning vector (e.g., from timestep and class embeddings)
    dummy_conditioning_vector = torch.randn(batch_size, hidden_size)

    # 1. Patchify the image
    patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=hidden_size)
    patches = patch_embed(dummy_image)
    print(f"Shape after patchifying: {patches.shape}")

    # 2. Create and apply the DiT block
    dit_block = DiTBlock(hidden_size=hidden_size, num_heads=num_heads)
    output_patches = dit_block(patches, dummy_conditioning_vector)
    print(f"Shape after DiT block: {output_patches.shape}")

    # Verify the output shape is the same as the input shape
    assert patches.shape == output_patches.shape
    print("DiTBlock test passed!")
