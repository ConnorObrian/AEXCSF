from copy import deepcopy
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm_notebook
from tqdm import tqdm
import torch
import os
from torch.utils.data import DataLoader
import warnings
from DatasetMaker.DatasetGenerator import MetricTracker
from Autoencoder.autoencoder import ConvVariationalAutoEncoder
from Autoencoder.vq_vae import Vqvae_Universal
from torchvision.transforms import v2
import dill
import math

def load_autoencoder(autoencoder_type : str, enc_size : list, dec_size : list, latent_size : int, image_shape : tuple[int, int, int], l1_weight: float, l2_weight: float, lpips_weight: float, max_kld_weight: float, kld_warmup_epochs:int=None, 
                     var_kld: bool=True, flatten: bool = True, pretrained : bool = False, codebook_size : int = 1024, histogram : str = None, num_quantizers : int = 8, use_attention = False, device : str = "cuda") -> torch.nn.Module:
    
    input_dim, x_dim, _ = image_shape
    if autoencoder_type == "cae":
        autoencoder = ConvAutoEncoder(in_channels=input_dim, in_dims=x_dim, enc_size=enc_size, dec_size=dec_size, latent_space_size=latent_size, pretrained=pretrained, flatten=flatten)
    elif autoencoder_type == "mae":
        autoencoder = mae_vit_base_patch16_dec512d8b()
    elif autoencoder_type == "vae":
        if pretrained:
            autoencoder = Pretrained_VAE(size_in=x_dim, fc_hidden1=enc_size[0], fc_hidden2=enc_size[1], CNN_embed_dim=latent_size, pretrained=pretrained)
        else:
            autoencoder = VAE(size_in=x_dim, latent_dim=latent_size, in_channels=input_dim, enc_siz=enc_size, dec_siz=dec_size)
    elif autoencoder_type == "cvae":
        if pretrained:
            autoencoder = Pretrained_VAE(size_in=x_dim, fc_hidden1=enc_size[0], fc_hidden2=enc_size[1], CNN_embed_dim=latent_size, pretrained=pretrained)
        else:
            autoencoder = ConvVariationalAutoEncoder(size_in=x_dim, latent_dim=latent_size, in_channels=input_dim, enc_siz=enc_size, dec_siz=dec_size, attn_layers=use_attention, 
                                                     variable_kld_weight=var_kld, max_kld_weight=max_kld_weight, kld_warmup_epochs=kld_warmup_epochs, lpips_weight=lpips_weight, l2_weight=l2_weight, l1_weight=l1_weight)
    elif autoencoder_type == "ae":
        autoencoder = Autoencoder(input_dim=x_dim, in_channels=input_dim, enc_size=enc_size, dec_size=dec_size, latent_dim=latent_size)
    if autoencoder_type == "transformer":
        if pretrained:
            autoencoder = get_vit_autoencoder_pretrained(img_size=x_dim, in_channels=input_dim, latent_dim=latent_size)
        else:
            autoencoder = get_vit_autoencoder(img_size=x_dim, in_channels=input_dim, latent_dim=latent_size)
    # --- VQ-VAE Varianten (vqvae, lfq, fsq, rvq) ---
    elif autoencoder_type in ["vqvae", "lfq", "fsq", "rvq"]:
        # Mapping von autoencoder_type string zu quantizer_type für Vqvae_Universal
        # vqvae -> "vq" (Standard)
        # lfq -> "lfq"
        # fsq -> "fsq"
        # rvq -> "rvq"
        quant_type_map = {
            "vqvae": "vq",
            "lfq": "lfq",
            "fsq": "fsq",
            "rvq": "rvq"
        }
        selected_quantizer = quant_type_map[autoencoder_type]

        # 1. Ziel-Grid berechnen (für RL)
        # latent_size = 64 bedeutet wir wollen 64 Token -> 8x8 Grid
        target_grid_width = int(math.sqrt(latent_size))
        if target_grid_width ** 2 != latent_size:
            print(f"WARNUNG: latent_size {latent_size} ist keine Quadratzahl. Ziel-Grid wird {target_grid_width}x{target_grid_width} = {target_grid_width**2} groß sein.")

        # 2. Layer Tiefe berechnen
        downsample_factor = x_dim / target_grid_width
        
        if not math.log2(downsample_factor).is_integer():
             # Fallback oder Fehler, falls Bildgröße nicht passt
             raise ValueError(f"Bildgröße {x_dim} nicht sauber auf {target_grid_width}x{target_grid_width} skalierbar (Faktor {downsample_factor}). Bitte Bildgröße oder latent_size anpassen (Power of 2).")
        
        num_layers = int(math.log2(downsample_factor))
        
        # 3. Layer Dimensionen generieren
        # Startet bei {enc_size[0]} und verdoppelt sich (z.B. [32, 64, 128, 256])
        # Use more filters to capture details
        start_dim = max(64, enc_size[0])
        generated_layer_dims = [start_dim * (2**i) for i in range(num_layers)]
        
        print(f"--- {selected_quantizer.upper()}-VAE Konfiguration ---")
        print(f"Input: {x_dim}x{x_dim}")
        print(f"Latent Grid: {target_grid_width}x{target_grid_width} (Total {target_grid_width**2} Indizes für RL)")
        print(f"Downsample Faktor: {downsample_factor} -> {num_layers} Layer")
        print(f"Berechnete Layer: {generated_layer_dims}")

        if selected_quantizer == "rvq":
             print(f"RVQ Tiefe: {num_quantizers} Quantizer")
             print(f"Gesamte Latent Codes: {target_grid_width**2 * num_quantizers} (Grid * Tiefe)")

        # Spezielle Konfigurationen für die Quantizer
        fsq_levels = [8, 5, 5, 5] # Default für FSQ (ca. 1000 Codes), kann angepasst werden
        # Sicherstellen dass Dimensionen für FSQ passen (Produkt der Levels <= Latent Dim passt nicht ganz, 
        # FSQ Levels definieren implizit die nötige Projektion, aber latent_dim sollte >= len(levels) sein)
        if selected_quantizer == "fsq" and 64 < len(fsq_levels):
             # Falls der interne Channel-Latent-Dim kleiner als die Anzahl der Levels wäre (unwahrscheinlich bei 64)
             pass 

        autoencoder = Vqvae_Universal(
            in_channels=input_dim,
            layer_dims=generated_layer_dims, 
            latent_dim=latent_size, 
            num_embeddings=codebook_size, 
            quantizer_type=selected_quantizer, 
            use_attention=use_attention, 
            num_res_layers=2,
            commitment_weight=0.25,
            decay=0.9,
            use_ema=True,
            l1_weight=l1_weight,
            l2_weight=l2_weight,
            lpips_weight=lpips_weight,
            
            # Spezifische Parameter weiterreichen
            fsq_levels=fsq_levels if selected_quantizer == "fsq" else None,
            num_quantizers=num_quantizers # Configurable depth
        )
    # -------------------------
    elif autoencoder_type == "ijepa":
        autoencoder = IJepaAutoencoder(pretrained=pretrained, xcs_latent_dim=latent_size, output_channels=input_dim, image_size=x_dim)
    elif autoencoder_type == "dino":
        autoencoder = DINOAutoencoder(pretrained=pretrained, x_size=x_dim, input_dims=input_dim, latent_dim=latent_size)
    elif autoencoder_type == "infovae":
        autoencoder = InfoVAE(img_channels=input_dim, z_dim=latent_size, channels=enc_size, x_size=x_dim)
    if not autoencoder:
        raise ValueError("Autoencoder not found") #future release use a simple compression algorithm here

    # move autoencoder to gpu
    autoencoder.to(device)

    # test the autoencoder
    try:
        x = torch.rand(1, input_dim, x_dim, x_dim).to(device)
        x = autoencoder(x)
        del x
        #raise ValueError("Autoencoder is not compatible with the given input size or is missing the forward function.")
    except:
        raise ValueError("Given autoencoder does not have the right input size or is missing the forward function.")
    return autoencoder


def draw_recon_test_image(data, recon, training_name, test_batch_len, channels, metric_save_dir, epoch:int=None):
    #plt.figure(dpi=250)
    fig, ax = plt.subplots(2, min(7, test_batch_len), figsize=(15, 4))
    for i in range(min(7, test_batch_len)):
        ax[0, i].imshow(data[i].cpu().numpy().transpose((1, 2, 0)), cmap='gray' if channels == 1 else None)
        ax[1, i].imshow(recon[i].cpu().numpy().transpose((1, 2, 0)), cmap='gray' if channels == 1 else None)
        ax[0, i].axis('OFF')
        ax[1, i].axis('OFF')
    file_name = "reconstruction.png" if epoch is None else f"reconstruction_test_epoch_{epoch}.png"
    path = os.path.join(metric_save_dir, training_name)
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(os.path.join(path, file_name))
    plt.clf()
    plt.close(fig)

    
def train_autoencoder(train_loader : DataLoader, test_loader : DataLoader, autoencoder : torch.nn.Module,
                      epochs : int, learning_rate : float, training_name : str, device : str = "cuda", batch_size : int = 32, autoencoder_type : str = "ae",
                      metric_save_dir : str = "./Results/", model_save_dir : str = "./Results/", p_random_transform : float = 0, use_lr_scheduler=True, notebook : bool = False):
    shape = next(iter(train_loader))[0].shape
    transform = v2.Compose([
    v2.ToImage(),
    v2.RandomApply([
        v2.RandomRotation(10),
        v2.RandomHorizontalFlip(p=0.5),
        v2.GaussianBlur(shape[0]),
        v2.GaussianNoise(0.1),
        v2.Lambda(lambda y: v2.functional.adjust_gamma(inpt=y, gamma=2.0)),
        v2.RandomCrop((shape[1], shape[2]), pad_if_needed=True),
    ], p=p_random_transform),
    v2.ToDtype(torch.float32, scale=True)
    ]).to(device)
    main_pbar = tqdm_notebook(range(epochs), desc='Training AE...') if notebook else tqdm(range(epochs), desc='Training AE...')
    metric_tracker = MetricTracker(training_name)

    # Use AdamW only for Transformer/DINO/IJEPA, use standard Adam for VQ/AE variants to avoid codebook decay issues
    use_adamw = autoencoder_type in ["transformer", "dino", "ijepa"]
    optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=learning_rate) if use_adamw else torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_function = torch.nn.MSELoss()
    best_autoencoder = {"loss": float("inf"), "model": None}

    for epoch in range(epochs):
        
        running_params = {
            'train_loss': [],
            "kld_weight" : [],
        }
        
        train_pbar = tqdm_notebook(range(len(train_loader)), desc=f'Epoch {epoch + 1}/{epochs}', leave=False, colour="red") if notebook else tqdm(range(len(train_loader)), desc=f'Epoch {epoch + 1}/{epochs}', leave=False, colour="red") #Training progress bar
        autoencoder.train()
        for i, image in enumerate(train_loader):
            optimizer.zero_grad()
            
            if torch.isnan(image).any():
                warnings.warn("Image contains NaN values")
                print(image)
                continue
            
            image = image.to(device)
            current_input = image if p_random_transform == 0 else transform(image)
            # Reconstruct the transformed image, as spatial transforms (crop/rotate) aren't recoverable from original
            target_image = current_input

            if autoencoder_type == "vae" or autoencoder_type == "cvae" or autoencoder_type == "dino":
                x_hat, mean, log_var = autoencoder(current_input)
                loss = autoencoder.loss_function(target_image, x_hat, mean, log_var)
            elif autoencoder_type in ["vqvae", "lfq", "fsq", "rvq"]:
                x_hat, vq_loss = autoencoder(current_input)
                loss = autoencoder.loss_function(target_image, x_hat, vq_loss)
            elif autoencoder_type == "transformer" or autoencoder_type == "ijepa":
                x = autoencoder(current_input)
                loss = autoencoder.loss_function(target_image, x)
            elif autoencoder_type == "infovae":
                recon, z, mu, logvar = autoencoder(current_input)
                loss = autoencoder.loss_function(target_image, recon, z, mu, logvar)
            else:  
                x = autoencoder(current_input)
                loss = loss_function(x, target_image)
            loss.backward()
            current_metrics = {
                    'train_loss': loss.item() / batch_size,
                    "kld_weight" : autoencoder.kld_weight if autoencoder_type == "vae" or autoencoder_type == "cvae" or autoencoder_type == "dino" else 0,
            }
            optimizer.step()
            if use_lr_scheduler:
                scheduler.step()
            
            for key in running_params:
                    running_params[key].append(current_metrics[key])
            mean_metrics = {key: sum(running_params[key]) / len(running_params[key]) for key in running_params}
            train_pbar.set_postfix(mean_metrics)
            if device == "cuda":
                torch.cuda.empty_cache()
            train_pbar.update()
        mean_train_loss = mean_metrics["train_loss"]
        mean_kld_weight = mean_metrics["kld_weight"]
        # test the autoencoder
        running_params = {
            'test_loss': [],
        }
        test_pbar = tqdm_notebook(range(len(test_loader)), desc=f'Test Epoch {epoch + 1}/{epochs}', leave=False, colour="magenta") if notebook else tqdm(range(len(test_loader)), desc=f'Test Epoch {epoch + 1}/{epochs}', leave=False, colour="magenta") #Test progress bar
        autoencoder.eval()

        if (print_recon_ex:=(epoch % 5 == 0)):
            image_excerpt = None
            recon_excerpt = None

        with torch.no_grad():
            for i, image in enumerate(test_loader):

                image = image.to(device)

                if autoencoder_type == "vae" or autoencoder_type == "cvae" or autoencoder_type == "dino":
                    x_hat, mean, log_var = autoencoder(image)
                    loss = autoencoder.loss_function(image, x_hat, mean, log_var)
                elif autoencoder_type in ["vqvae", "lfq", "fsq", "rvq"]:
                    x_hat, vq_loss = autoencoder(image)
                    loss = autoencoder.loss_function(image, x_hat, vq_loss)
                elif autoencoder_type == "transformer" or autoencoder_type == "ijepa":
                    x_hat = autoencoder(image)
                    loss = autoencoder.loss_function(image, x_hat)
                elif autoencoder_type == "infovae":
                    recon, z, mu, logvar = autoencoder(image if p_random_transform == 0 else transform(image))
                    loss = autoencoder.loss_function(image, recon, z, mu, logvar)
                else:
                    x_hat = autoencoder(image)
                    loss = loss_function(x_hat, image)

                if print_recon_ex and i==0:
                    # NOTE: Think about whether you always want to have the first 7 images of the first batch
                    image_excerpt = image.cpu()
                    recon_excerpt = x_hat.cpu()
                
                current_metrics = {
                    'test_loss': loss.item() / batch_size,
                }
                
                if device == "cuda":
                    torch.cuda.empty_cache()
                for key in running_params:
                    running_params[key].append(current_metrics[key])
                mean_metrics = {key: sum(running_params[key]) / len(running_params[key]) for key in running_params}
                test_pbar.set_postfix(mean_metrics)
                test_pbar.update()
        
        if (print_recon_ex):
            draw_recon_test_image(image_excerpt, recon_excerpt, training_name, 7, channels=autoencoder.in_channels, metric_save_dir=metric_save_dir, epoch=epoch)
        
        metric_tracker.add_metrics({"epoch": epoch, "train_loss": mean_train_loss, "test_loss": mean_metrics["test_loss"], "kld_weight": mean_kld_weight})
        # test_loss = mean_metrics["test_loss"]
        # if test_loss < best_autoencoder["loss"] or epoch == 0:
        #     try:
        #         best_autoencoder["loss"] = test_loss
        #         # NOTE: This should actually deep copy the state dict and decouple it from the internal one
        #         best_autoencoder["model"] = deepcopy(autoencoder.state_dict())
        #         best_epoch = epoch
        #     except:
        #         print("Error saving best model in the variable")
        # update kld weight of vae
        if autoencoder_type == "vae" or autoencoder_type == "cvae" or autoencoder_type == "dino":
            autoencoder.update_kld_weight(train_epochs=epochs, current_epoch=epoch)
        train_pbar.close()
        test_pbar.close()
        main_pbar.update()
    main_pbar.close()

    metric_tracker.save_csv(os.path.join(metric_save_dir, training_name))
    torch.save(autoencoder, os.path.join(model_save_dir, training_name, "autoencoder.pth"), pickle_module=dill)

    # NOTE: In the end, I never used this feature... and it's not desirable for tandem training
    # try:
    #     current_state_dict = deepcopy(autoencoder.state_dict())
    #     assert(epoch == best_epoch or torch.any(current_state_dict != best_autoencoder["model"]))
    #     autoencoder.load_state_dict(best_autoencoder["model"])
    #     torch.save(autoencoder, os.path.join(model_save_dir, training_name, "best_autoencoder.pth"), pickle_module=dill)
    #     print("Autoencoder training finished, model saved in:", os.path.abspath(os.path.join(model_save_dir, training_name, "autoencoder.pth")))
    #     autoencoder.load_state_dict(current_state_dict)
    # except:
    #     print("Error loading best model from the variable")
        
        
