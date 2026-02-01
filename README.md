# Attempting to Vectorize Environments in Ray RLlib

In this repository, we explore multiple methods for vectorizing environments using Ray's [RLlib library](https://github.com/ray-project/ray/tree/master/rllib). 

**Environment:** Tested with `Python 3.10.18` on Linux with a `T4` GPU.

We focus specifically on the `vector_entry_point` use case, where the user handles environment vectorization manually before passing it to RLlib.

## Vectorization Approaches

We are currently comparing two methods:

* **`vec`**: The standard, intended way of vectorizing the environment in RLlib.
* **`packed_vec`**: A proposed alternative for specific high-scale scenarios.

### The `packed_vec` Hypothesis
The motivation behind developing `packed_vec` is that processing individual episodes can become computationally expensive in certain cases—specifically when the number of episodes or the batch size reaches six figures. While these settings are uncommon, they occur when users require a more precise estimate of the policy distribution within a batch.

### Implementation Details
The core idea is to create an environment containing `N` sub-environments that RLlib perceives as a **single** environment. 

* **Observation Handling:** We explicitly save `rewards`, `terminateds`, and `truncateds` as part of the observations to be processed later in the learner connector.
* **Batch Size Interpretation:** Due to the nature of `packed_vec`, the `train_batch` parameter deviates from its standard interpretation. The actual number of environment samples is $train\_batch \times num\_envs$ (where `num_envs` is the number of sub-environments within each `vector_entry_point`). 
* **Workaround:** For certain experiments, we manually disabled the [internal check](https://github.com/ray-project/ray/blob/master/rllib/algorithms/ppo/ppo.py#L300) that enforces `self.minibatch_size <= self.train_batch_size`.

---

## Installation

You can install the dependencies using **uv**:
```bash
pip install uv
uv sync
```
or via pip 

```
pip install .
```

## Running the script

If you installed via `uv`, run:
```bash
.venv/bin/python -m vecenv.run_training
```

Otherwise, use
```bash
python -m vecenv.run_training
```
