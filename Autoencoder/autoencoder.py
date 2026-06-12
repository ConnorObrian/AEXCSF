import torch
import torch.nn as nn
import torchmetrics.image
from torchvision.transforms import v2


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attn = self.conv(x_cat)
        return self.sigmoid(attn) * x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_planes, in_planes // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_planes // reduction, in_planes, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _ = x.size()
        avg_out = self.mlp(self.avg_pool(x).view(batch_size, channels))
        max_out = self.mlp(self.max_pool(x).view(batch_size, channels))
        out = avg_out + max_out
        return self.sigmoid(out).view(batch_size, channels, 1, 1) * x


class CBAM(nn.Module):
    def __init__(self, in_planes: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(in_planes, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        return self.spatial_att(x)


class ConvVariationalAutoEncoder(nn.Module):
    """Convolutional variational autoencoder used by the experiment pipelines."""

    def __init__(
        self,
        size_in,
        l1_weight: float,
        l2_weight: float,
        lpips_weight: float,
        max_kld_weight: float,
        latent_dim=5,
        enc_siz=[32, 64, 128, 256],
        dec_siz=[256, 128, 64, 32],
        in_channels=3,
        variable_kld_weight=True,
        kld_warmup_epochs: int = None,
        attn_layers=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.in_size = size_in
        if attn_layers:
            self.attn_layers = {
                "enc": [i for i in range(len(enc_siz)) if i not in [0, 1]],
                "dec": [i for i in range(len(dec_siz)) if i not in [len(dec_siz) - 1, len(dec_siz) - 2]],
            }
        else:
            self.attn_layers = {"enc": [], "dec": []}

        enc_modules = []
        enc_siz = [in_channels] + enc_siz
        for i in range(len(enc_siz) - 1):
            layer = [
                nn.Conv2d(enc_siz[i], enc_siz[i + 1], kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(),
            ]
            if i in self.attn_layers.get("enc", []):
                layer.append(CBAM(enc_siz[i + 1]))
            enc_modules.append(nn.Sequential(*layer))

        self.encoder = nn.Sequential(*enc_modules)

        x = torch.rand(1, in_channels, size_in, size_in)
        out = self.encoder(x)
        self.size = out.shape[2]
        self.dim = enc_siz[-1]
        self.encoder.append(nn.Flatten())

        self.fc_mu = nn.Linear(enc_siz[-1] * self.size * self.size, latent_dim)
        self.fc_var = nn.Sequential(nn.Linear(enc_siz[-1] * self.size * self.size, latent_dim), nn.Softplus())
        self.fc_3 = nn.Linear(latent_dim, enc_siz[-1] * self.size * self.size)

        dec_modules = [nn.Unflatten(dim=1, unflattened_size=out.shape[1:])]
        for i in range(len(dec_siz) - 1):
            layer = [
                nn.ConvTranspose2d(dec_siz[i], dec_siz[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.LeakyReLU(),
            ]
            if i in self.attn_layers.get("dec", []):
                layer.append(CBAM(dec_siz[i + 1]))
            dec_modules.append(nn.Sequential(*layer))

        self.decoder = nn.Sequential(*dec_modules)
        self.decoder.append(
            nn.Sequential(
                nn.ConvTranspose2d(dec_siz[-1], in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.Sigmoid(),
            )
        )

        self.decoder_postprocessing = v2.Compose(
            [
                v2.Resize(size_in, antialias=True),
                v2.CenterCrop(size_in),
            ]
        )

        self.variable_kld_weight = variable_kld_weight
        self.max_kld_weight = 0.1 if max_kld_weight is None else max_kld_weight
        self.kld_weight = 0.0 if self.variable_kld_weight else self.max_kld_weight
        self.l1_weight = 0.0 if l1_weight is None else l1_weight
        self.l2_weight = 1.0 if l2_weight is None else l2_weight
        self.lpips_weight = 0.0 if lpips_weight is None else lpips_weight
        self.kld_warmup_epochs = kld_warmup_epochs

        if self.lpips_weight > 0.0 and in_channels == 3:
            self.use_lpips = True
            self.lpips = torchmetrics.image.LearnedPerceptualImagePatchSimilarity(reduction="sum", net_type="alex")
        else:
            self.use_lpips = False

        print(
            f"ConvVae: Channels: {self.in_channels}, L2 Weight:{self.l2_weight}, L1 Weight: {self.l1_weight}, "
            f"(Max) KLD Weight: {self.max_kld_weight},\n "
            f"KLD Weight at start {self.kld_weight}, Var KLD? {self.variable_kld_weight}, "
            f"KLD Warmup Epochs {self.kld_warmup_epochs}"
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_3(z)
        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def update_kld_weight(self, train_epochs, current_epoch, max_kld_weight=1.2):
        if self.kld_warmup_epochs is None:
            warmup_steps = train_epochs // 4
        else:
            warmup_steps = self.kld_warmup_epochs

        if warmup_steps == 0:
            warmup_steps = 1
        if self.variable_kld_weight:
            self.kld_weight = min(self.max_kld_weight, self.max_kld_weight * (current_epoch / warmup_steps))

    def loss_function(self, x, recon_x, mu, logvar, loss="MSE", kld_weight=1.2):
        if loss != "BCE" and loss != "MSE":
            raise ValueError("Loss must be either 'BCE' or 'MSE'")

        if self.use_lpips:
            lpips_loss = self.lpips(recon_x, x)
        else:
            lpips_loss = 0.0

        l1_loss = torch.nn.functional.l1_loss(recon_x, x, reduction="sum")
        l2_loss = (
            torch.nn.functional.mse_loss(recon_x, x, reduction="sum")
            if loss == "MSE"
            else torch.nn.functional.binary_cross_entropy(recon_x, x, reduction="sum")
        )
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)

        return self.l2_weight * l2_loss + self.lpips_weight * lpips_loss + self.l1_weight * l1_loss + self.kld_weight * kld_loss
