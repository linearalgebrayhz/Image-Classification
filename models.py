import torch.nn as nn
import torch


class ResNetBlock(nn.Module):
    """A basic residual block with a projection shortcut when needed."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )

    def forward(self, x):
        residual = self.skip(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class ResNet(nn.Module):
    """A compact ResNet suitable for 100–224 pixel fruit images."""

    def __init__(self, num_blocks=2, num_classes=10, base_channels=32):
        super().__init__()
        self.in_channels = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(base_channels, num_blocks, stride=1)
        self.layer2 = self._make_layer(base_channels * 2, num_blocks, stride=2)
        self.layer3 = self._make_layer(base_channels * 4, num_blocks, stride=2)
        self.layer4 = self._make_layer(base_channels * 8, num_blocks, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(base_channels * 8, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        blocks = [ResNetBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        blocks.extend(ResNetBlock(out_channels, out_channels) for _ in range(1, num_blocks))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(self.pool(x).flatten(1))


class VisionTransformer(nn.Module):
    """Small, dependency-free Vision Transformer classifier."""

    def __init__(
        self,
        num_classes,
        image_size=224,
        patch_size=16,
        dim=256,
        depth=4,
        heads=8,
        mlp_dim=512,
        dropout=0.1,
    ):
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        if x.size(1) + 1 != self.position_embedding.size(1):
            raise ValueError("input dimensions do not match the configured image_size")
        class_token = self.class_token.expand(x.size(0), -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = self.dropout(x + self.position_embedding)
        return self.head(self.norm(self.encoder(x)[:, 0]))


class MambaBlock(nn.Module):
    """Input-dependent gated state-space block with a linear-time recurrence."""

    def __init__(self, dim, state_dim=16, expansion=2, conv_kernel=4):
        super().__init__()
        inner_dim = dim * expansion
        self.norm = nn.LayerNorm(dim)
        self.input_projection = nn.Linear(dim, inner_dim * 2)
        self.conv = nn.Conv1d(
            inner_dim,
            inner_dim,
            kernel_size=conv_kernel,
            padding=conv_kernel - 1,
            groups=inner_dim,
        )
        self.state_in = nn.Linear(inner_dim, state_dim)
        self.state_out = nn.Linear(state_dim, inner_dim)
        self.decay = nn.Parameter(torch.zeros(state_dim))
        self.output_projection = nn.Linear(inner_dim, dim)

    def forward(self, x):
        residual = x
        values, gate = self.input_projection(self.norm(x)).chunk(2, dim=-1)
        values = self.conv(values.transpose(1, 2))[..., : x.size(1)].transpose(1, 2)
        values = nn.functional.silu(values)

        decay = torch.sigmoid(self.decay).view(1, -1)
        state = values.new_zeros(values.size(0), self.decay.numel())
        outputs = []
        for token in self.state_in(values).unbind(dim=1):
            state = decay * state + (1.0 - decay) * token
            outputs.append(self.state_out(state))
        x = torch.stack(outputs, dim=1) * nn.functional.silu(gate)
        return residual + self.output_projection(x)


class MambaClassifier(nn.Module):
    """Patch-based image classifier built from selective state-space blocks."""

    def __init__(
        self,
        num_classes,
        image_size=224,
        patch_size=16,
        dim=192,
        depth=4,
        state_dim=16,
    ):
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.blocks = nn.Sequential(
            *(MambaBlock(dim, state_dim=state_dim) for _ in range(depth))
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        return self.head(self.norm(self.blocks(x)).mean(dim=1))


def build_model(name, num_classes, **kwargs):
    """Construct a supported classifier by its CLI name."""
    models = {
        "resnet": ResNet,
        "vit": VisionTransformer,
        "mamba": MambaClassifier,
    }
    try:
        model_class = models[name.lower()]
    except KeyError as error:
        raise ValueError(f"unknown model {name!r}; choose from {', '.join(models)}") from error
    return model_class(num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    for model_name in ("resnet", "vit", "mamba"):
        model = build_model(model_name, num_classes=10)
        output = model(torch.randn(2, 3, 224, 224))
        print(f"{model_name}: {tuple(output.shape)}")
