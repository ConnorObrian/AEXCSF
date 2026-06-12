from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

import tyro


REPO_ROOT = Path(__file__).resolve().parent


@dataclass
class Args:
    name: str
    dataset_path:str
    ae_type:str
    epochs: int # 20-30 worked well for RVQ - 200-300 for CVAE
    latent_dim:int = 4 # 4 worked well for RVQ -- for CVAE we suggest 8, 12, 24 or 32
    n_channels: int = 3
    image_file_type: str="jpg"

def build_command(args: Args) -> list[str]:
    if args.n_channels == 3:
        c_name = "colour"
    elif args.n_channels == 1:
        c_name = "grey-scale"
    else:
        raise ValueError(f"You passed {args.n_channels} as channels option only 1 or 3 is supported.")

    if args.ae_type == "rvq":
        add_args = ["--num_quantizers=4", "--codebook_size=512", "--enc_size", "32", "64", "128"]
    elif args.ae_type == "cvae":
        add_args = ["--lpips_weight=0.0", "--l2_weight=1.", "--l1_weight=0.0", "--max_kld_weight=0.1", "--no_var_kld", "--no_lr_scheduler", "--p_random_transform=0"]

    command = [
        sys.executable,
        "train_autoencoder.py",
        "--autoencoder",
        args.ae_type,
        "--dataset_path",
        args.dataset_path,
        "--epochs",
        str(args.epochs),
        "--latent_dim", 
        str(args.latent_dim),
        "--image_file_type",
        args.image_file_type,
        "--name",
        args.name + c_name,
        "--learning_rate=1e-4",
        "--batch_size=64",
        "--x_dims=64",
        "--y_dims=64",
        "--n_channels",
        str(args.n_channels),]
    
    return command + add_args


def main() -> None:
    args = tyro.cli(Args)
    subprocess.run(build_command(args), check=True, cwd=REPO_ROOT)

if __name__ == "__main__":
    main()