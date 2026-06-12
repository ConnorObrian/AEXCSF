from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

import tyro


REPO_ROOT = Path(__file__).resolve().parent


@dataclass
class Args:
    name_suffix: str
    env_name: str
    max_steps: int
    n_channels: int = 3
    episodes: int = 2000

def build_command(args: Args) -> list[str]:
    if args.n_channels == 3:
        c_name = "colour"
    elif args.n_channels == 1:
        c_name = "grey-scale"
    else:
        raise ValueError(f"You passed {args.n_channels} as channels option only 1 or 3 is supported.")
    
    if "Vizdoom" in args.env_name:
        add_args = ["--env_flags", "frame_skip=5"]
    elif "FrozenLake" in args.env_name:
        # When collecting datasets for frozen lake you can just pass FrozenLake8 for FrozenLake 8x8
        env_args_str = "is_slippery=False"
        if "8" in args.env_name:
            env_args_str+=",map_name=8x8"
        add_args = ["--use_render_as_state", "--env_flags", env_args_str]
        args.env_name="FrozenLake-v1"

    command = [
        sys.executable,
        "build_dataset.py",
        "--env_name",
        args.env_name,
        "--dataset_episodes",
        str(args.episodes),
        "--max_steps",
        str(args.max_steps),
        "--x_dims=64",
        "--y_dims=64",
        "--n_channels",
        str(args.n_channels),
        "--dataset_path=./Dataset",
        "--dataset_model=ppo"]

    return command + add_args


def main() -> None:
    args = tyro.cli(Args)
    subprocess.run(build_command(args), check=True, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()