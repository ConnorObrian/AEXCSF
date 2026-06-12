#!/bin/bash
# $1 name suffix $2 autoencoder type $3 autoencoder path [optional $4 latent_scaler $5 vq_latent_state]

# Argument validation check
if [ "$#" -lt 3 ] || [ "$#" -gt 5 ]; then
    echo "Usage: <name suffix> <autoencoder type> <autoencoder path> [<latent_scaler> <vq_latent_state>]"
    exit 1
fi

python conduct_aexcsf_experiments.py \
    --runs 30 \
    --name "DoomCSR-N5000-NoSubsume-$1" \
    --e0 0.01 \
    --p_min 0.01 \
    --gamma 0.93 \
    --ea_theta 25 \
    --theta_del 25 \
    --env_name VizdoomBasicCustom-v0 \
    --autoencoder_path "$3" \
    --autoencoder "$2" \
    --ignore_sigma \
    --normalize_latent \
    ${4:+"--latent_scaler=$4"} \
    ${5:+"--vq_latent_state=$5"} \
    --axn_episodes 3000 \
    --max_steps 100 \
    --x_dims 64 \
    --y_dims 64 \
    --n_channels 3 \
    --pop_size 5000 \
    --number_threads 16 \
    --max_exploration_trials 1000 \
    --condition hyperrectangle_csr \
    --c_max 3.5 \
    --env_flags frame_skip=5 \
    --no-subsumption \
    --p_min 0.01 \
    --seeds 992977901 333860189 322156938 96647316 194277914 588429196 349325664 817532751 224015233 423723742 79329643 745959810 600036365 245357231 415632686 820935647 66781547 485479248 546194229 177232891 192194810 871654536 827594188 925070057 318965504 358462159 105724622 402385565 674357224 129967652
