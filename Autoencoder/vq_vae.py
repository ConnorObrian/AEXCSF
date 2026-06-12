import torch
import torch.nn as nn
import torchmetrics
import vector_quantize_pytorch as vq

# --- Helper für robuste Normalisierung ---
def ActNorm(dim):
    """
    Wählt automatisch eine passende GroupNorm Konfiguration.
    Wenn dim < 16, nutzen wir keine Gruppen (Instanz-Norm-Verhalten),
    sonst standardmäßig 8 oder 32 Gruppen, solange es ein Teiler ist.
    """
    if dim < 16:
        return nn.GroupNorm(1, dim)
    
    # Versuche 32 Gruppen (Standard bei großen Modellen), sonst 8, sonst 4
    for groups in [32, 8, 4]:
        if dim % groups == 0:
            return nn.GroupNorm(groups, dim)
    
    # Fallback: Instance Norm (1 Gruppe pro Channel ist witzlos, also 1 Gruppe für alle)
    return nn.GroupNorm(1, dim)


# --- Residual Block mit robuster Norm ---
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            ActNorm(dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            ActNorm(dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        return x + self.block(x)


# --- Self-Attention Block ---
class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = ActNorm(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        
        # Shapes: B, C, H, W
        B, C, H, W = h_.shape
        
        # Berechne Q, K, V
        q = self.q(h_).reshape(B, C, H*W).permute(0, 2, 1) # B, HW, C
        k = self.k(h_).reshape(B, C, H*W)                  # B, C, HW
        v = self.v(h_).reshape(B, C, H*W).permute(0, 2, 1) # B, HW, C
        
        # Attention Score: Q * K
        # Skalierung durch 1/sqrt(C) für stabile Gradienten
        w_ = torch.bmm(q, k) * (int(C)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2) # Attention Map (B, HW, HW)
        
        # Output: Attn * V
        h_ = torch.bmm(w_, v)     # B, HW, C
        h_ = h_.permute(0, 2, 1).reshape(B, C, H, W)
        
        h_ = self.proj_out(h_)
        return x + h_

class Vqvae_Universal(nn.Module):
    """Universal VQ-VAE implementation supporting multiple quantizer types.

    Args:
        in_channels (int, optional): Number of input channels. Defaults to 3.
        layer_dims (list, optional): List of channel dimensions for each layer. Defaults to [32, 64, 128, 256].
        latent_dim (int, optional): Dimension of the latent space. Defaults to 64.
        num_embeddings (int, optional): Number of embeddings in the codebook. Defaults to 256.
        num_res_layers (int, optional): Number of residual layers in encoder/decoder. Defaults to 2.
        use_attention (bool, optional): Whether to use attention blocks. Defaults to True.
        quantizer_type (str, optional): Type of quantizer to use ('vq', 'rvq', 'lfq', 'fsq'). Defaults to "vq".
        commitment_weight (float, optional): Commitment weight for VQ. Defaults to 1.0.
        decay (float, optional): Decay rate for EMA updates. Defaults to 0.8.
        use_ema (bool, optional): Whether to use EMA for codebook updates. Defaults to True.
        num_quantizers (int, optional): Number of quantizers for RVQ. Defaults to 8.
        fsq_levels (list, optional): Levels for FSQ quantizer. Defaults to [8, 5, 5, 5].
        entropy_loss_weight (float, optional): Weight for entropy loss in LFQ. Defaults to 0.1.
    """
    def __init__(
        self,
        in_channels=3,
        layer_dims=[32, 64, 128, 256], 
        latent_dim=64,
        num_embeddings=256,
        num_res_layers=2,
        use_attention=True,
        
        # --- Quantizer Konfiguration ---
        quantizer_type="vq", 
        commitment_weight=1.0,
        decay=0.8,
        use_ema=True,
        num_quantizers=8, # RVQ
        fsq_levels=[8, 5, 5, 5], # FSQ
        entropy_loss_weight=0.1, # LFQ
        l1_weight = None,
        l2_weight = None,
        lpips_weight = None ):
        super().__init__()
        
        self.quantizer_type = quantizer_type.lower()
        self.latent_dim = latent_dim

        if l1_weight is None:
            self.l1_weight = 2.0
        else:
            self.l1_weight = l1_weight

        self.l2_weight = l2_weight
        if l2_weight is None:
            self.l2_weight = 5.0
        else:
            self.l2_weight = l2_weight

        if lpips_weight is None:
            self.lpips_weight = 1.0
        else:
            self.lpips_weight = lpips_weight
        
        # --- Encoder ---
        encoder_layers = []
        self.in_channels = in_channels

        ch_in = in_channels
        # Downsampling
        for ch_out in layer_dims:
            encoder_layers.extend([
                nn.Conv2d(ch_in, ch_out, kernel_size=4, stride=2, padding=1),
                ActNorm(ch_out),
                nn.SiLU(),
            ])
            ch_in = ch_out
            
        # Projektion
        encoder_layers.append(nn.Conv2d(ch_in, latent_dim, kernel_size=3, stride=1, padding=1))
        
        # Bottleneck: ResBlocks + Optional Attention
        for _ in range(num_res_layers):
            encoder_layers.append(ResBlock(latent_dim))
            
        if use_attention:
            encoder_layers.append(AttnBlock(latent_dim))
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        # --- Quantizer (Identisch wie zuvor) ---
        print(f"Initialisiere Quantizer: {self.quantizer_type.upper()}")
        if self.quantizer_type == "vq":
            self.vq = vq.VectorQuantize(
                dim=latent_dim, 
                codebook_size=num_embeddings, 
                decay=decay if use_ema else 0.8, # Faster decay to adapt to new codes quickly
                commitment_weight=1.0, # Stronger commitment to stabilize codebook usage
                use_cosine_sim=True, # Back to Cosine Sim for stability
                kmeans_init=True, 
                kmeans_iters=10,
                threshold_ema_dead_code=2
            )
        elif self.quantizer_type == "rvq":
            self.vq = vq.ResidualVQ(
                dim=latent_dim, num_quantizers=num_quantizers, codebook_size=num_embeddings,
                decay=decay if use_ema else 0.8, commitment_weight=commitment_weight,
                kmeans_init=True, use_cosine_sim=True, threshold_ema_dead_code=2
            )
        elif self.quantizer_type == "lfq":
            self.vq = vq.LFQ(dim=latent_dim, codebook_size=num_embeddings, entropy_loss_weight=entropy_loss_weight, diversity_gamma=1.0, channel_first=False)
        elif self.quantizer_type == "fsq":
            self.vq = vq.FSQ(levels=fsq_levels, dim=latent_dim)
        else:
            raise ValueError(f"Unbekannter Typ: {quantizer_type}")

        # --- Decoder ---
        decoder_layers = []
        
        # Bottleneck: Optional Attention + ResBlocks
        if use_attention:
            decoder_layers.append(AttnBlock(latent_dim))
            
        for _ in range(num_res_layers):
            decoder_layers.append(ResBlock(latent_dim))
            
        # Projektion zurück
        decoder_layers.append(nn.Conv2d(latent_dim, layer_dims[-1], kernel_size=3, stride=1, padding=1))
        decoder_layers.append(ActNorm(layer_dims[-1]))
        decoder_layers.append(nn.SiLU())
        
        # Upsampling
        reversed_layers = layer_dims[::-1]
        for i in range(len(reversed_layers) - 1):
            ch_in_dec = reversed_layers[i]
            ch_out_dec = reversed_layers[i+1]
            decoder_layers.extend([
                nn.Upsample(scale_factor=2.0, mode='nearest'),
                nn.Conv2d(ch_in_dec, ch_out_dec, kernel_size=3, stride=1, padding=1),
                ActNorm(ch_out_dec),
                nn.SiLU(),
            ])
            
        # Output
        decoder_layers.append(
            nn.Upsample(scale_factor=2.0, mode='nearest')
        )
        decoder_layers.append(
            nn.Conv2d(reversed_layers[-1], in_channels, kernel_size=3, stride=1, padding=1)
        )
        decoder_layers.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*decoder_layers)
    

        # LPIPS wird lazy geladen (erst beim ersten Aufruf initialisiert)
        self.perceptual_loss = torchmetrics.image.LearnedPerceptualImagePatchSimilarity(net_type="alex")
        # Deaktiviere Gradienten für LPIPS (spart Speicher)
        for param in self.perceptual_loss.parameters():
            param.requires_grad = False

    def encode(self, x):
        z_e = self.encoder(x)
        z_e_permuted = z_e.permute(0, 2, 3, 1) # B, H, W, C
        B, H, W, C = z_e_permuted.shape
        z_flat = z_e_permuted.reshape(B, H*W, C)
        
        indices = None
        aux_loss = torch.tensor(0.0, device=x.device)
        
        if self.quantizer_type == "fsq":
            quantized, indices = self.vq(z_flat)
        elif self.quantizer_type in ["vq", "rvq", "lfq"]:
            quantized, indices, aux_loss = self.vq(z_flat)
        else: 
            # Fallback
            res = self.vq(z_flat)
            if len(res) == 3:
                quantized, indices, aux_loss = res
            else:
                quantized, indices = res

        # Unflatten quantized
        # quantized is (B, H*W, C) -> (B, H, W, C)
        quantized = quantized.reshape(B, H, W, -1)

        # Unflatten indices
        if indices is not None:
             if indices.ndim == 2: # (B, Seq)
                 indices = indices.reshape(B, H, W)
             elif indices.ndim == 3: # (B, Seq, num_codebooks)
                 indices = indices.reshape(B, H, W, -1)

        z_q = quantized.permute(0, 3, 1, 2)
        return z_e, z_q, aux_loss, indices

    def decode(self, z_q):
        return self.decoder(z_q)

    def forward(self, x):
        _, z_q, aux_loss, _ = self.encode(x)
        recon = self.decode(z_q)
        if aux_loss.ndim > 0: aux_loss = aux_loss.mean()
        return recon, aux_loss
    
    def loss_function(self, x, recon, aux_loss, perceptual_weight=1.0):
        # Balanced Loss: L1 is crucial for sharpness, but 10.0 might be too dominating leading to fast plateau
        # Reducing multipliers to standard range
        l1_loss = torch.nn.functional.l1_loss(recon, x) * self.l1_weight # default weight should be 2.0 
        mse_loss = torch.nn.functional.mse_loss(recon, x) * self.l2_weight # default weight should be 5.0
        
        perceptual = 0.0
        # Default weight for lpips weight should be 1.0
        if self.lpips_weight > 0:
            perceptual = self.perceptual_loss(recon, x).mean()
            
        return l1_loss + mse_loss + aux_loss + self.lpips_weight * perceptual
