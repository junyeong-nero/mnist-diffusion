import torch
import torch.nn as nn

from tiny_DDPM.modules.layer import PositionalEmbedding


class MLP(nn.Module):
    """A simple MLP block."""

    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU
    ):
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
        self.num_patches = (self.img_size[1] // self.patch_size[1]) * (
            self.img_size[0] // self.patch_size[0]
        )

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        if H != self.img_size[0] or W != self.img_size[1]:
            raise ValueError(
                f"Input image size ({H}x{W}) doesn't match model ({self.img_size[0]}x{self.img_size[1]})."
            )

        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
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
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
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
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            mod_params.chunk(6, dim=1)
        )

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


class DiT(nn.Module):
    """
    Diffusion Transformer (DiT) model.
    """

    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=1,
        hidden_size=256,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        n_steps=1000,
        time_emb_dim=256,
        n_classes=10,
        class_emb_dim=64,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, hidden_size)
        self.num_patches = self.patch_embed.num_patches

        # Positional embedding for patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))

        # Time and class embedding
        self.time_embedding = PositionalEmbedding(n_steps, time_emb_dim)
        self.class_embedding = PositionalEmbedding(n_classes, class_emb_dim)

        # Project time and class embeddings to the hidden size
        self.cond_projection = nn.Sequential(
            nn.Linear(time_emb_dim + class_emb_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, patch_size * patch_size * in_channels, bias=True),
        )

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.img_size = img_size

    def unpatchify(self, x):
        """
        x: (B, N, P*P*C)
        return: (B, C, H, W)
        """
        B = x.shape[0]
        P = self.patch_size
        C = self.in_channels
        H = W = self.img_size // P

        x = x.reshape(B, H, W, P, P, C)
        x = x.permute(0, 5, 1, 3, 2, 4)  # (B, C, H, P, W, P)
        x = x.reshape(B, C, H * P, W * P)
        return x

    def forward(self, x, t, c=None):
        x = self.patch_embed(x) + self.pos_embed

        # Flatten t to ensure it's (B,) shape for embedding layer
        t = t.flatten()
        t_emb = self.time_embedding(t)

        if c is not None:
            # Flatten c to ensure it's (B,) shape for embedding layer
            c = c.flatten()
            c_emb = self.class_embedding(c)
            cond = torch.cat([t_emb, c_emb], dim=1)
        else:
            # If class condition is not given, use a zero vector for class embedding
            c_emb = torch.zeros(
                t_emb.shape[0],
                self.class_embedding.time_embed.embedding_dim,
                device=x.device,
            )
            cond = torch.cat([t_emb, c_emb], dim=1)

        cond = self.cond_projection(cond)

        for block in self.blocks:
            x = block(x, cond)

        x = self.final_layer(x)
        x = self.unpatchify(x)
        return x


if __name__ == "__main__":
    # A simple test case for the DiTBlock

    # Parameters
    batch_size = 4
    img_size = 32
    patch_size = 4
    in_channels = 1
    hidden_size = 256  # This is the embedding dimension
    num_heads = 8
    depth = 6  # Number of DiT blocks

    # Create a dummy input image tensor
    dummy_image = torch.randn(batch_size, in_channels, img_size, img_size)

    # Create a dummy conditioning vector (e.g., from timestep and class embeddings)
    dummy_t = torch.randint(0, 1000, (batch_size,))
    dummy_c = torch.randint(0, 10, (batch_size,))

    # 1. Test DiT model
    dit_model = DiT(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
    )

    output = dit_model(dummy_image, dummy_t, dummy_c)
    print(f"Shape after DiT model: {output.shape}")

    # Verify the output shape is the same as the input image shape
    assert dummy_image.shape == output.shape
    print("DiT model test passed!")

    # Test without class conditioning
    output_uncond = dit_model(dummy_image, dummy_t, c=None)
    assert dummy_image.shape == output_uncond.shape
    print("DiT model test (unconditional) passed!")
