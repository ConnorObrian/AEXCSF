#!/bin/bash
# $1 name suffix $2 autoencoder type $3 autoencoder path [optional $4 latent_scaler $5 vq_latent_state]

# Argument validation check
if [ "$#" -lt 3 ] || [ "$#" -gt 5 ]; then
    echo "Usage: <name suffix> <autoencoder type> <autoencoder path> [<latent_scaler> <vq_latent_state>]"
    exit 1
fi

python conduct_aexcsf_experiments.py \
    --runs 30 \
    --name "Empty-N2000-NoSubsume-$1" \
    --p_min 0.01 \
    --beta  0.1 \
    --delta 0.1 \
    --gamma 0.95 \
    --ea_theta 50 \
    --theta_del 50 \
    --env_name MiniGrid-Empty-6x6-v0 \
    --autoencoder_path "$3" \
    --autoencoder "$2" \
    --ignore_sigma \
    --normalize_latent \
    ${4:+"--latent_scaler=$4"} \
    ${5:+"--vq_latent_state=$5"} \
    --axn_episodes 3000 \
    --max_steps 144 \
    --x_dims 64 \
    --y_dims 64 \
    --n_channels 3 \
    --pop_size 2000 \
    --number_threads 16 \
    --max_exploration_trials 200 \
    --condition hyperrectangle_ubr \
    --c_max 1.0 \
    --no-subsumption \
    --p_min 0.01 \
    --seeds 891157510 325477928 219723616  14734832 689496833 439234904 576798521 818803219  87343379 814841174 989958658 290914230 207105079 863720910 693133188 702949794 992560809  53047595 207694158 641173503 284504539 309794006 789710848 927306022 134426938 358001034 458053684 540071881 515077054 449925676