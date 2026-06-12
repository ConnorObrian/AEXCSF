![scale=0.5](Ressources/logo.png)
# AEXCSF and Experiments with XCSF + Downscaling

The autoencoder XCSF hybrid system (AEXCSF) combines deep autoencoders, primarily convolutional VAEs, with the XCSF Classifier System, an evolutionary rule-based Machine Learning method that offers interpretable decision making and an interpetable learned rule-base. It can be applied to grey-scale as well as RGB-based visual multi-step RL environments. We include example experiments for several such environments in this repository. As XCSF implementation, we employ the XCSF of Preen et al. https://github.com/xcsf-dev/xcsf that has been already employed in several publications on evolutionary rule-based machine learning methods. AEXCSF by default is trained in a mixed offline-online training scheme: The dataset is collected employing an arbitrary different RL agent, in our case a simple PPO implementation, then the chosen autoencoder is trained offline. The resulting frozen autoencoder is then used as a dimensionality reduction method to train an XCSF instance online in the targetted RL environment.

In addition to our experiments with AEXCSF, this repository also contains a script to train XCSF on downscaled image states of multi-step RL environments. This serves as a simple, naive and surprisingly good baseline. To our knowledge, there have been no published experiments following this naive approach for visual RL before our two publications.

This repository corresponds to the code that we employed for the experiments of the paper "Latent Representation Learning for Visual Reinforcement Learning with a Classifier System" accepted at PPSN'2026. It is a fork of our original code employed in the experiments of the paper "Dimensionality Reduction for Enabling Visual Reinforcement Learning with a Classifier System" accepted at the IWERL@GECCO'2025. We contributed several improvements compared to this earlier version including, but not limited to: 

- An improved loss function for convolutional VAEs
- The introduction of RQ-VAEs using lucidrains' vector-quantize-pytorch library https://github.com/lucidrains/vector-quantize-pytorch for quantisers
- RQ-VAE-based AEXCSF may learn on the indices of the codebook vectors or the quantised codebook vectors themselves
- Improved scalers for states in the latent space of autoencoders, eliminatation of prior preprocessing, 
- Improved environment support
- Support for RGB image states
- Pre-set hyperparameters for allowing good performance in our targetted environms when learning on RGB images without pre-processing aside of scaling the images to 64x64
- A background writer when generating datasets (ensuring hopefully improved wall clock performance)
- An end-to-end training scheme available on the end-to-end branch
- More command lines arguments
- More user-friendly run scripts
- Additional polishing.

For easier experimentation, we include a non-root docker container and our dependencies are managed using the dependency management tool uv.
For building our docker container with our docker_build_locked.sh script, you should set your own UID, GID and user name in the dockerfile.
You can collect a dataset using our run_data_collector script, train an autoencoder using our run_ae_training script and afterwards train AEXCSF with one of our run scripts for specific environments.


## Installation

Either by 

```sh
uv sync
```
if you want to run it locally on your machine or by building and running our docker container.



## Cite as

PPSN Citation to be added 


You could in addition cite the following for our older version:

@inproceedings{schonbernerDimensionalityReductionEnabling2025a,
  title = {Dimensionality {{Reduction}} for {{Enabling Visual Reinforcement Learning}} with a {{Classifier System}}},
  booktitle = {Proceedings of the {{Genetic}} and {{Evolutionary Computation Conference Companion}}},
  author = {Schönberner, Connor and Mackensen, Armin and Tomforde, Sven},
  date = {2025-08-11},
  series = {{{GECCO}} '25 {{Companion}}},
  pages = {2269--2278},
  publisher = {Association for Computing Machinery},
  location = {New York, NY, USA},
  doi = {10.1145/3712255.3734321},
  abstract = {Originally, the XCS Classifier System, seen as the most popular and most researched Learning Classifier System (LCS), was designed to learn and solve Reinforcement Learning (RL) problems. As has become clear over time, known weaknesses and the curse of dimensionality of Michigan-style LCSs severely limit its applicability to RL problems. As a result, visual RL problems generally appear to be outside the scope of XCS variants. We target this class of RL benchmarks by combining XCSF, XCS with hyperrectangle conditions and computed linear prediction, with dimensionality reduction methods. In particular, we combine deep variational autoencoders (VAE) with XCSF using an automatic dataset collection scheme and offline training of the VAE and online training for the combined system to target several visual RL problems. Our results show that XCSF is able to learn visual RL problems in the latent space of deep VAEs. Further experiments confirm that simple downscaling of images can also enable XCSF to learn several visual RL problems. Our results do not only indicate that XCSF is better at RL than its reputation, but show that significantly more RL problems might be within the scope of XCSF and motivate further investigation of dimensionality reduction methods for LCSs.},
  isbn = {979-8-4007-1464-1},
  venue = {Spain, Malaga}
}