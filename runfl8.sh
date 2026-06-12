#!/bin/bash
# $1 name suffix $2 autoencoder type $3 autoencoder path [optional $4 latent_scaler $5 vq_latent_state]

# Argument validation check
if [ "$#" -lt 3 ] || [ "$#" -gt 5 ]; then
    echo "Usage: <name suffix> <autoencoder type> <autoencoder path> [<latent_scaler> <vq_latent_state>]"
    exit 1
fi

python conduct_aexcsf_experiments.py \
    --runs 30 \
    --name "FL8-N2000-NoSubsume-$1" \
    --p_min 0.01 \
    --beta  0.2 \
    --delta 0.1 \
    --gamma 0.95 \
    --ea_theta 1000 \
    --theta_del 1000 \
    --env_name FrozenLake-v1 \
    --env_flags is_slippery=False,map_name=8x8 \
    --use_render_as_state \
    --autoencoder_path "$3" \
    --autoencoder "$2" \
    --ignore_sigma \
    --normalize_latent \
    ${4:+"--latent_scaler=$4"} \
    ${5:+"--vq_latent_state=$5"} \
    --axn_episodes 4000 \
    --max_steps 100 \
    --x_dims 64 \
    --y_dims 64 \
    --n_channels 3 \
    --pop_size 2000 \
    --number_threads 16 \
    --max_exploration_trials 2000 \
    --condition hyperrectangle_ubr \
    --c_max 1.0 \
    --no-subsumption \
    --p_min 0.01 \
    --seeds 6531387 6361333 1172252 7318222 676103282 391711374 281643764 878724770 221684037 470024247  84732024 675032249 811845518 224703417 880677780  56514707 852866237 939507520 132490397 421295536 893858944 705013051 660154654 222195292 269676435 562248427 601138332 437086370 704598874  43295600
