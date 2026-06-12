from collections import deque
import logging
from pprint import pformat
from DatasetMaker import DatasetParser as dp

import torch
from torchinfo import summary

import time

from Autoencoder.train_autoencoder import train_autoencoder, load_autoencoder
from DatasetMaker.DatasetParser import train_val_dataset

import os

import argparse

import pandas as pd
import matplotlib.pyplot as plt

# Autoencoder Params
AUTOENCODER = "cvae" #
PRETRAINED = None # ("vgg", "resnet", "mobile")
FLATTEN = False # True, if you want to flatten before the latent layer inside the autoencoder (no functionality for VAE)

DATASET_PATH = None # Path to the dataset

# image parameter (changing these also changes the behavior of every image generated)
X_DIMS = 64
Y_DIMS = 64
N_CHANNELS = 1 # 1 for grayscale, 3 for rgb

# Autoencoder Training Params
ELIMINATE_DUPLIUCATES = False # Whether to eliminate duplicates from the dataset
BATCH_SIZE = 64
EPOCHS = 150 if ELIMINATE_DUPLIUCATES else 4000
INPUT_DIM = N_CHANNELS 
# for vae, ae use values, that get smaller with each layer, cae use values that get bigger with each layer
ENC_SIZE = [32, 64, 128, 256, 512, 1024] # Encoder layer sizes (caes gets bigger with each layer, vae, ae get smaller)
DEC_SIZE = ENC_SIZE.copy()
DEC_SIZE.reverse()
LATENT_DIM = 16 # size of the latent layer
LEARNING_RATE = 1e-3
NO_VAR_KLD = False
# MAX_KLD_WEIGHT = 1.2
# LPIPS_WEIGHT = 0.1
# L1_WEIGHT = 0.05
# L2_WEIGHT = 1.0

P_RANDOM_TRANSFORM = 0.0

METRIC_SAVE_DIR = "./Results/"
MODEL_SAVE_DIR = "./Results/"

def setup_logging(save_path):
    logging.basicConfig(filename=save_path + "/experiment.log",
                        format="%(levelname)s: %(message)s",
                        level=logging.INFO)

# Argument parser
parser = argparse.ArgumentParser(description='Autoencoder Training Parameters')
parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset')
parser.add_argument('--autoencoder', type=str, default=AUTOENCODER, help='Type of autoencoder to use (ae, vae, cae, mae, cvae)')
parser.add_argument('--pretrained', type=str, default=PRETRAINED, help='Pretrained model to use (vgg, resnet, mobile)')
parser.add_argument('--flatten', action="store_true", default=FLATTEN, help='Whether to flatten before the latent layer')
parser.add_argument('--x_dims', type=int, default=X_DIMS, help='Width of the images')
parser.add_argument('--y_dims', type=int, default=Y_DIMS, help='Height of the images')
parser.add_argument('--n_channels', type=int, default=N_CHANNELS, help='Number of channels in the images')
parser.add_argument('--eliminate_duplicates', action="store_true", default=ELIMINATE_DUPLIUCATES, help='Whether to eliminate duplicates from the dataset')
parser.add_argument('--p_random_transform', type=float, default=P_RANDOM_TRANSFORM, help='Probability of applying a random transformation to the image')
parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs for training')
parser.add_argument('--enc_size', nargs='+', type=int, default=ENC_SIZE, help='Encoder layer sizes (caes gets bigger with each layer, vae, ae get smaller)')
parser.add_argument('--latent_dim', type=int, default=LATENT_DIM, help='Size of the latent layer')
parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE, help='Learning rate for training')
parser.add_argument('--metric_save_dir', type=str, default=METRIC_SAVE_DIR, help='Directory to save metrics')
parser.add_argument('--model_save_dir', type=str, default=MODEL_SAVE_DIR, help='Directory to save models')
parser.add_argument('--num_quantizers', type=int, default=8, help='Number of quantizers for RVQ (default: 8)')
parser.add_argument('--use_attention', action="store_true", default=False, help='Whether to use attention layers in the encoder and decoder (only for CVAE, VQ-VAE, LFQ, FSQ, SimVQ, RVQ)')
parser.add_argument('--codebook_size', type=int, default=2048, help='Size of the codebook for VQ-VAE, LFQ, FSQ, SimVQ (only used for these autoencoders)')
parser.add_argument('--no_lr_scheduler', action="store_true", default=False, help='Activates the lr scheduler')
parser.add_argument('--no_var_kld', action="store_true", default=NO_VAR_KLD, help='Deactivates variable kld weights/kld weight annealing')
parser.add_argument('--kld_warmup_epochs', type=int, default=None, help='Amount of epochs to reach max_kld_weight if var_kld is activated - Default leads to epochs //4 as target')
parser.add_argument('--max_kld_weight',  type=float, default=None, required=False, help='Max KLD Weight in the LOSS of a VAE - constant kld weight without variable/annealed kld')
parser.add_argument('--l1_weight',  type=float, default=None, required=False, help='Weight for the L1 Loss in the LOSS of a VAE.')
parser.add_argument('--l2_weight',  type=float, default=None, required=False, help='Weight for the L2 Loss in the LOSS of a VAE.')
parser.add_argument('--lpips_weight',  type=float, default=None, required=False, help='Weight for the LPIPS loss in the LOSS of a VAE.')
parser.add_argument('--histogram', type=str, default="hard", choices=["hard", "soft", "pos", "graph", "None"],help='Allows to set the \"histogram\" parameter of VQ-VAEs that controls the post-processing applied on the vector-quantized latent space')
parser.add_argument('--name', type=str, default="", help='Adds a custom name to the autoencoder directory')
parser.add_argument('--image_file_type', type=str, default="jpg", help='Allows control which type of images are present in the dataset.')


args = parser.parse_args()
AUTOENCODER = args.autoencoder
PRETRAINED = args.pretrained
FLATTEN = args.flatten
DATASET_PATH = args.dataset_path
X_DIMS = args.x_dims
Y_DIMS = args.y_dims
N_CHANNELS = args.n_channels
ELIMINATE_DUPLIUCATES = args.eliminate_duplicates
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
ENC_SIZE = list(args.enc_size)
DEC_SIZE = ENC_SIZE[::-1]
LATENT_DIM = args.latent_dim
LEARNING_RATE = args.learning_rate
LR_SCHEDULER = not args.no_lr_scheduler
NO_VAR_KLD = args.no_var_kld
MAX_KLD_WEIGHT = args.max_kld_weight
KLD_WARMUP_EPOCHS = args.kld_warmup_epochs
L1_WEIGHT = args.l1_weight
L2_WEIGHT = args.l2_weight
LPIPS_WEIGHT = args.lpips_weight
NAME = args.name
IMAGE_FILE_TYPE: str =  args.image_file_type
METRIC_SAVE_DIR = args.metric_save_dir
MODEL_SAVE_DIR = args.model_save_dir
P_RANDOM_TRANSFORM = args.p_random_transform
CODEBOOK_SIZE = args.codebook_size 
NUM_QUANTIZERS = args.num_quantizers
USE_ATTENTION = args.use_attention
HISTOGRAM = args.histogram

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rest_path = os.path.split(os.path.split(DATASET_PATH)[0])
ENV_NAME = rest_path[1]
METHOD = os.path.split(rest_path[0])[1]

def draw_recon_test_image(data, recon, training_name, test_batch_len, channels, metric_save_dir, batch:int=None):
    #plt.figure(dpi=250)
    fig, ax = plt.subplots(2, min(7, test_batch_len), figsize=(15, 4))
    for i in range(min(7, test_batch_len)):
        ax[0, i].imshow(data[i].cpu().numpy().transpose((1, 2, 0)), cmap='gray' if channels == 1 else None)
        ax[1, i].imshow(recon[i].cpu().numpy().transpose((1, 2, 0)), cmap='gray' if channels == 1 else None)
        ax[0, i].axis('OFF')
        ax[1, i].axis('OFF')
    file_name = "reconstruction.png" if batch is None else f"reconstruction_batch{batch}.png"
    path = os.path.join(metric_save_dir, training_name)
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(os.path.join(path, file_name))
    plt.clf()
    plt.close(fig)

dataset = dp.RLDataset(DATASET_PATH, eliminate_duplicates=ELIMINATE_DUPLIUCATES, image_size=(N_CHANNELS, X_DIMS, Y_DIMS), transform=None, image_file_type=IMAGE_FILE_TYPE)
# split the dataset into train and test

train_ds, test_ds = train_val_dataset(dataset, val_split=0.1)
print(f"Train: {len(train_ds)} Test: {len(test_ds)}, Total: {len(dataset)}")

# create dataloaders
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# load the autoencoder
autoencoder = load_autoencoder(autoencoder_type=AUTOENCODER, enc_size=ENC_SIZE, dec_size=DEC_SIZE, image_shape=(N_CHANNELS, X_DIMS, Y_DIMS), latent_size=LATENT_DIM, pretrained=PRETRAINED, 
                               flatten=FLATTEN, num_quantizers=NUM_QUANTIZERS, codebook_size=CODEBOOK_SIZE, device=DEVICE, l1_weight=L1_WEIGHT, l2_weight=L2_WEIGHT, lpips_weight=LPIPS_WEIGHT, var_kld=not NO_VAR_KLD, 
                               max_kld_weight=MAX_KLD_WEIGHT, kld_warmup_epochs=KLD_WARMUP_EPOCHS, histogram=HISTOGRAM, use_attention=USE_ATTENTION)

training_name = "Autoencoder_training_" + NAME + ENV_NAME + "_" + METHOD + "_" + AUTOENCODER + "_" + time.strftime("%Y%m%d-%H%M%S")
path = os.path.join(METRIC_SAVE_DIR, training_name)
if not os.path.exists(path):
    os.makedirs(path)

setup_logging(save_path=path)
logging.info(f"Commandline arguments passed to the system:\n {pformat(vars(args))}\n")
logging.info(f"Used device: {DEVICE}\n")
logging.info(f"Autoencoder Properties:\n {pformat({k: v for k, v in vars(autoencoder).items() if not k.startswith("_")})}\n")
# Log the autoencoder architecture
logging.info(summary(autoencoder, (1, N_CHANNELS, X_DIMS, Y_DIMS)))

train_autoencoder(train_loader=train_loader, test_loader=test_loader, autoencoder=autoencoder, epochs=EPOCHS, learning_rate=LEARNING_RATE, training_name=training_name,
                  device=DEVICE, batch_size=BATCH_SIZE, autoencoder_type=AUTOENCODER, metric_save_dir=METRIC_SAVE_DIR, model_save_dir=MODEL_SAVE_DIR, notebook=False,
                  p_random_transform=P_RANDOM_TRANSFORM, use_lr_scheduler=LR_SCHEDULER)

# reads csv and generate a diagram of mean loss
parser = pd.read_csv(os.path.join(os.path.join(METRIC_SAVE_DIR, training_name), training_name + ".csv"))
#calc the mean reward over 100 episodes
mean_train_losses = parser["train_loss"]
mean_test_losses = parser["test_loss"]
plt.plot(mean_test_losses)
plt.plot(mean_train_losses)
plt.legend(["Test Loss", "Train Loss"])
plt.xlabel("Episodes")
plt.ylabel("Mean Loss")
plt.savefig(os.path.join(METRIC_SAVE_DIR, training_name, "mean_losses.png"))
plt.clf()

images = []
recons = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        data = data.to(DEVICE)
        recon = autoencoder(data)
        if i < 2:
            images.append(data)
            recons.append(recon)
        # vae returns 3 values
        if AUTOENCODER == "vae" or AUTOENCODER == "cvae" or AUTOENCODER == "infovae" or AUTOENCODER == "dino" or AUTOENCODER == "ijepa":
            recon = recon[0]
        if AUTOENCODER in ["vqvae", "lfq", "fsq", "rvq"]:
            recon = recon[0]
        print(recon.shape)
        test_batch_len = recon.shape[0]
        if i > 1:
            break

# draw_recon_test_image(data, recon, training_name=training_name, test_batch_len=test_batch_len, channels=N_CHANNELS, metric_save_dir=METRIC_SAVE_DIR)
for i, (image, recon) in enumerate(zip(images, recons)):
    draw_recon_test_image(image, recon[0], training_name=training_name, test_batch_len=test_batch_len, channels=N_CHANNELS, metric_save_dir=METRIC_SAVE_DIR, batch=i)


print(f"saved model to {os.path.join(MODEL_SAVE_DIR, training_name)}")
print(f"saved metrics to {os.path.join(METRIC_SAVE_DIR, training_name)}")
print("Finished Training")