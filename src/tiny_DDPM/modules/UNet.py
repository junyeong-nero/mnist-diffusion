import torch
import torch.nn as nn

from tiny_DDPM.modules.layer import (
    SelfAttentionBlock,
    PositionalEmbedding,
    WideResNetBlock,
    MultiHeadAttentionBlock,
)


class UNetDown(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        base_model=WideResNetBlock,
        is_deconv=True,
        is_batchnorm=True,
    ):
        super(UNetDown, self).__init__()
        self.conv = base_model(in_channels, out_channels, is_batchnorm=is_batchnorm)

    def forward(self, input):
        return self.conv(input)


class UNetUp(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        base_model=WideResNetBlock,
        is_deconv=True,
        is_batchnorm=True,
    ):
        super(UNetUp, self).__init__()
        self.conv = base_model(
            out_channels * 2, out_channels, is_batchnorm=is_batchnorm
        )
        if is_deconv:
            self.up = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1
            )
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs0, *input):
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)


class UNetTimeEmbedding(nn.Module):

    def __init__(self, dim_in, dim_out) -> None:
        super(UNetTimeEmbedding, self).__init__()
        self.ln = nn.Linear(dim_in, dim_out)
        self.activation = nn.SiLU()
        self.ln2 = nn.Linear(dim_out, dim_out)

    def forward(self, inputs):
        B = inputs.shape[0]

        x = self.ln(inputs)
        x = self.activation(x)
        x = self.ln2(x)

        return x.reshape(B, -1, 1, 1)


class UNet(nn.Module):

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        n_steps=1000,
        time_emb_dim=256,
        n_classes=10,
        class_emb_dim=64,
        channel_scale=64,
        num_channel_scale=5,
        custom_channel_scale=None,
        cross_attention_layer_indices=[-1],
        self_attention_layer_indices=[-1],
        is_deconv=True,
        is_batchnorm=True,
    ):
        super(UNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_batchnorm = is_batchnorm

        self.self_attention_layer_indices = set(self_attention_layer_indices)
        self.cross_attention_layer_indices = set(cross_attention_layer_indices)

        # time embedding
        self.time_embedding = PositionalEmbedding(n_steps, time_emb_dim)
        self.context_embedding = PositionalEmbedding(n_classes, class_emb_dim)

        if custom_channel_scale is None:
            filters = [channel_scale * (2**i) for i in range(num_channel_scale)]
        else:
            num_channel_scale = len(custom_channel_scale)
            filters = custom_channel_scale

        self.num_layers = num_channel_scale
        filters.insert(0, in_channels)

        # Downsampling
        self.down_convs = nn.ModuleList()
        self.down_tembs = nn.ModuleList()
        self.down_cembs = nn.ModuleList()
        self.down_maxpools = nn.ModuleList()
        self.down_cross_attentions = nn.ModuleList()

        for layer_idx in range(1, self.num_layers):
            base_model = WideResNetBlock
            if (layer_idx - self.num_layers) in self.self_attention_layer_indices:
                base_model = SelfAttentionBlock

            self.down_convs.append(
                UNetDown(
                    in_channels=filters[layer_idx - 1],
                    out_channels=filters[layer_idx],
                    is_batchnorm=self.is_batchnorm,
                    base_model=base_model,
                )
            )
            self.down_tembs.append(UNetTimeEmbedding(time_emb_dim, filters[layer_idx]))
            self.down_cembs.append(UNetTimeEmbedding(class_emb_dim, filters[layer_idx]))
            self.down_maxpools.append(nn.MaxPool2d(kernel_size=2))

            if (layer_idx - self.num_layers) in self.cross_attention_layer_indices:
                self.down_cross_attentions.append(
                    MultiHeadAttentionBlock(
                        in_channels=filters[layer_idx],
                        out_channels=filters[layer_idx],
                        is_batchnorm=False,
                    )
                )
            else:
                self.down_cross_attentions.append(nn.Identity())

        # Bottleneck
        self.center = UNetDown(filters[-2], filters[-1], is_batchnorm=self.is_batchnorm)
        self.temb_center = UNetTimeEmbedding(time_emb_dim, filters[-1])
        self.cemb_center = UNetTimeEmbedding(class_emb_dim, filters[-1])
        self.cross_attention_center = MultiHeadAttentionBlock(
            in_channels=filters[-1], out_channels=filters[-1], is_batchnorm=False
        )

        # Upsampling
        self.up_convs = nn.ModuleList()
        self.up_tembs = nn.ModuleList()
        filters[0] = out_channels

        for layer_idx in range(1, self.num_layers):
            base_model = WideResNetBlock
            if (layer_idx - self.num_layers) in self.self_attention_layer_indices:
                base_model = SelfAttentionBlock
            self.up_convs.append(
                UNetUp(
                    filters[layer_idx + 1],
                    filters[layer_idx],
                    is_deconv=self.is_deconv,
                    is_batchnorm=self.is_batchnorm,
                    base_model=base_model,
                )
            )
            self.up_tembs.append(UNetTimeEmbedding(time_emb_dim, filters[layer_idx]))

        # output
        self.outconv = nn.Conv2d(filters[1], self.out_channels, 3, padding=1)

    def forward(self, inputs, t, c=None):
        t = self.time_embedding(t)
        if c is not None:
            c = self.context_embedding(c)

        x = inputs
        downsampling_result = [None]

        # DOWN-SAMPLING
        for i in range(self.num_layers - 1):
            x = self.down_convs[i](x)
            downsampling_result.append(x)

            if (
                c is not None
                and (i + 1 - self.num_layers) in self.cross_attention_layer_indices
            ):
                context_emb = self.down_cembs[i](c)
                x = self.down_cross_attentions[i](x, context_emb, context_emb)

            x += self.down_tembs[i](t)
            x = self.down_maxpools[i](x)

        # BOTTLENECK
        x = self.center(x)
        if c is not None:
            context_emb = self.cemb_center(c)
            x = self.cross_attention_center(x, context_emb, context_emb)
        x += self.temb_center(t)

        # UP-SAMPLING
        for i in range(self.num_layers - 2, -1, -1):
            x = self.up_convs[i](x, downsampling_result[i + 1])
            x += self.up_tembs[i](t)

        return self.outconv(x)


if __name__ == "__main__":
    unet = UNet(
        in_channels=3,
        out_channels=3,
        n_steps=1000,
        custom_channel_scale=[128, 128, 256, 256, 512, 512],
    )

    B = 1
    t = torch.randint(0, 1000, (B,))
    x = torch.randn(B, 3, 32, 32)
    c = torch.randint(0, 10, (B,))
    output = unet(x, t)
    print(output.shape)
