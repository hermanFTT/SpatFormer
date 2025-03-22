
# SpatFormer 
 A simulation-based inference (SBI) method using a transformer-enhanced diffusion model tailored for spatial models with latent GP priors. By leveraging transformers for sequence modeling and probabilistic diffusion for inference, this approach enables efficient, amortized Bayesian inference. Unlike traditional MCMC, it avoids repeated costly computations, allowing rapid exploration of spatial conditional distributions. Simulation experiments on one- and two-dimensional models demonstrate superior scalability compared to GP-based MCMC.

## Installation

If you have conda installed, you should first load a new environment. A minimal environment with
recommended cuda version for JAX is provided in `src/environment.yml`.

```bash
conda env create --file=src/environment.yml
conda activate simformer
pip install -e src/probjax[cuda]
pip install -e src/scoresbibm
```

We recommend installing it on a CUDA capable machine, as the experiments heavily benefit
from GPU acceleration. The above will install the CUDA version of JAX. If you do not have 
a CUDA capable machine, you can install the CPU version by dropping the `[cuda]` flag. 

