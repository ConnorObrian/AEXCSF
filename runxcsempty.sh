#!/bin/bash
# $1 name suffix $2 dim [optional $3 population capacity]

# Argument validation check
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: <name suffix> <dimension M - for images of size MxM> [<population capacity N>]"
    exit 1
fi

python conduct_xcs_experiments.py \
    --runs 30 \
    --name "Empty-N${3:-1000}-NoSubsume-$1" \
    --p_min 0.01 \
    --beta  0.1 \
    --delta 0.1 \
    --gamma 0.95 \
    --ea_theta 50 \
    --theta_del 50 \
    --env_name MiniGrid-Empty-6x6-v0 \
    --use_render_as_state \
    --axn_episodes 3000 \
    --max_steps 144 \
    --x_dims "$2" \
    --y_dims "$2" \
    --n_channels 3 \
    --pop_size "${3:-2000}" \
    --number_threads 16 \
    --max_exploration_trials 200 \
    --condition hyperrectangle_ubr \
    --c_max 1.0 \
    --no-subsumption \
    --p_min 0.01 \
    --seeds 891157510 325477928 219723616  14734832 689496833 439234904 576798521 818803219  87343379 814841174 989958658 290914230 207105079 863720910 693133188 702949794 992560809  53047595 207694158 641173503 284504539 309794006 789710848 927306022 134426938 358001034 458053684 540071881 515077054 449925676