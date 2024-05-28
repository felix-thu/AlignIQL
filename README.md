# AlignIQL: Policy Alignment in Implicit Q-Learning through Constrained Optimization
This is the official  Jax implementation of AlignIQL: Policy Alignment in Implicit Q-Learning through Constrained Optimization. Our implementation is based on [IDQL](https://github.com/philippe-eecs/IDQL) and [JAXRL](https://github.com/ikostrikov/jaxrl) repo which uses the JAX framework to implement RL algorithms. 

## Abstract
Implicit Q-learning (IQL) serves as a strong baseline for offline RL, which learns the value function using only dataset actions through quantile regression. However, it is unclear how to recover the implicit policy from the learned implicit Q-function and why IQL can utilize weighted regression for policy extraction. IDQL reinterprets IQL as an actor-critic method and gets weights of implicit policy, however, this weight only holds for the optimal value function. In this work, we introduce a different way to solve the \textit{implicit policy-finding problem} (IPF) by formulating this problem as an optimization problem. Based on this optimization problem, we further propose two practical algorithms AlignIQL and AlignIQL-hard, which inherit the advantages of decoupling actor from critic in IQL and provide insights into why IQL can use weighted regression for policy extraction. Compared with IQL and IDQL, we find our method keeps the simplicity of IQL and solves the implicit policy-finding problem.  Experimental results on D4RL datasets show that our method achieves competitive or superior results compared with other SOTA offline RL methods. Especially in complex sparse reward tasks like Antmaze and Adroit, our method outperforms IQL and IDQL by a significant margin. 
## Requirements
Cuda and cudnn version we used
```
cudnn-linux-x86_64-8.9.7.29_cuda11-archive.tar.xz
cuda_11.8.0_520.61.05_linux.run
```
1. Setting env variables (added to .bashrc or .zshrc)
```
## tensorflow
export CUDNN_PATH="$HOME/.conda/envs/align/lib/python3.9/site-packages/nvidia/cudnn"
export LD_LIBRARY_PATH="$CUDNN_PATH/lib":"/usr/local/cuda-11.8/lib64":"$LD_LIBRARY_PATH":/usr/lib/nvidia
export PATH="$PATH":"/usr/local/cuda/bin"
export PYTHONPATH=~/AlignIQL/:$PYTHONPATH
# solve OOM for jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```
2. [Install MuJoCo](https://www.youtube.com/watch?v=Wnb_fiStFb8&ab_channel=GuyTordjmann)

3. Python Env

```
conda create -n align python==3.10
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt
```

## Quick Start
```
# reproduce our D4RL results 
python launcher/examples/train_align_iql_offline.py --variant=0 --device=0 

# spase tasks
python launcher/examples/train_align_iql_sparse.py --variant=0 --device=0 

# offline2online (O2O) (ongoing work)
python launcher/examples/train_align_iql_finetuning.py --variant=0 --device=0
```

## Important file
We implement our method on 
```
AlignIQL/jaxrl5/agents/align_iql/align_iql_learner.py
```

## Citations
Cite this paper

```

```

## Problems
```
1. Error compiling Cython file:
------------------------------------------------------------
...
    See c_warning_callback, which is the C wrapper to the user defined function
    '''
    global py_error_callback
    global mju_user_error
    py_error_callback = err_callback
    mju_user_error = c_error_callback
                     ^
------------------------------------------------------------
```
Solution
```
pip install Cython==3.0.0a10
```

