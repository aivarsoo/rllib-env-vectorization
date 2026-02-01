

import json
import numpy as np
import os
import torch

from ray.tune.registry import register_env
from ray.rllib.connectors.env_to_module import (
    AddObservationsFromEpisodesToBatch,
    BatchIndividualItems,
    AddStatesFromEpisodesToBatch,
    AddTimeDimToBatchAndZeroPad,
    NumpyToTensor,
)
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from vecenv.rllib_module import RLModule
from vecenv.vectorized_pendulum import VecEnvWrapper, PendulaEnv, PackedVecEnvWrapper
from vecenv.connectors import CustomFlattenColumns
from vecenv.custom_ppo_config import CustomPPOCOnfig

def convert_numpy_to_list(obj):
    """
    Recursively converts NumPy arrays in a nested dictionary to lists.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_list(item) for item in obj]
    else:
        return obj

def convert_numpy_to_tensor(obj, device):
    """
    Recursively converts NumPy arrays in a nested dictionary to torch tensors.
    """
    if isinstance(obj, np.ndarray):
        return torch.as_tensor(obj, device=device)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_tensor(value, device=device) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_tensor(item, device=device) for item in obj]
    else:
        return obj


def run_rollout(algo, make_env, env_config, iteration, device, logdir, render):
    """
    Runs a single visual episode using the trained shared_policy and saves a GIF.
    Uses DETERMINISTIC SAMPLING and handles batching correctly.
    """
    import imageio
    print("\n=== Starting Visual Rollout (Deterministic) ===")
    env_config = {
        **env_config,
        "num_envs": 1,
        "render_mode": "rgb_array",
    }
    local_env = make_env(env_config)
    module = algo.get_module("shared_policy")

    dist_class = module.get_inference_action_dist_cls()

    obs_dict, _ = local_env.reset()
    done = False
    step = 0
    returns = 0
    frames = []

    while not done:

        with torch.no_grad():
            out = module.forward_inference({"obs": convert_numpy_to_tensor(obs_dict, device=device)})
            logits = out["action_dist_inputs"]

            dist = dist_class.from_logits(logits, model=module)
            action: torch.Tensor = dist.to_deterministic().sample()


        # 5. STEP
        obs_dict, reward, terminated, truncated, _ = local_env.step(action.squeeze(0).cpu().numpy())
        if render:
            frame = local_env.render()
            if frame is not None:
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                frames.append(frame)


        done = terminated.any() or truncated.any()
        step += 1
        returns += reward


    local_env.close()

    if render and frames:
        gif_filename = os.path.join(logdir, f"iteration_{iteration}_rollout.gif")
        print(f"\nSaving GIF to {gif_filename}...")
        imageio.mimsave(gif_filename, frames, fps=50)
        print(f"GIF saved successfully: {os.path.abspath(gif_filename)}")
    else:
        print("\nNo frames collected for GIF.")
    return returns.item()

def run(num_iterations, version, num_envs, train_batch, minibatch_size, eval_frequency, logdir, render):
    
    env_cfg = {
        "num_envs": num_envs,
        "max_episode_steps": 100,
    }

    config = CustomPPOCOnfig()
    env_multiplier = 1 
    if version == "vec":
        environment = {
            "env": "pendulum_env",
            "env_config": env_cfg,
        }
        env_runners = {
            "gym_env_vectorize_mode": "vector_entry_point",
            "num_envs_per_env_runner": env_cfg["num_envs"],
            }
        training = {
            "train_batch_size": train_batch,
            "minibatch_size": minibatch_size,
        }
        rl_module = {
            # "rl_module_spec": RLModuleSpec(module_class=RLModule),
            "model_config": DefaultModelConfig(fcnet_activation="relu"),
        }
        def make_env(env_cfg):
            env = PendulaEnv(**env_cfg)
            return VecEnvWrapper(env)

        register_env("pendulum_env", make_env)
    elif version == "packed_vec":
        # If we use Packed Environment then every step we get `num_envs` samples,
        # but RLLIB sees it as one sample during the sampling loop
        env_multiplier = num_envs
        environment = {
            "env": "pendulum_env",
            "env_config": env_cfg,
        }

        env_runners = {
            "num_envs_per_env_runner": 1,
            "add_default_connectors_to_env_to_module_pipeline": False,
            "env_to_module_connector": lambda env, spaces, device: [
                AddObservationsFromEpisodesToBatch(),
                AddTimeDimToBatchAndZeroPad(),
                AddStatesFromEpisodesToBatch(),
                BatchIndividualItems(),
                NumpyToTensor(device=device),
                CustomFlattenColumns(),
            ],
        }
        training = {
            "train_batch_size": int(train_batch / env_cfg["num_envs"]),
            "minibatch_size": minibatch_size, # max possible minibatch_size 
            "add_default_connectors_to_learner_pipeline": False,
            # Connectors are defined in the custom config class
        }

        rl_module = {
            "rl_module_spec": RLModuleSpec(module_class=RLModule),
        }

        def make_env(env_cfg):
            return PackedVecEnvWrapper(**env_cfg)

        register_env("pendulum_env", make_env)

    config = config.environment(
        **environment
    ).env_runners(
        **env_runners,
        num_env_runners=1,
        num_gpus_per_env_runner=1e-2,
    ).learners(
        num_learners=1,
        num_gpus_per_learner=1e-2,
    ).training(
        **training,
        gamma=0.95,
        num_epochs=5,
    ).rl_module(
        **rl_module,
    )
    algo = config.build()
    fps = []
    timings = {}
    eval_reward = 0
    for i in range(num_iterations):
        result = algo.train()

        if  i % eval_frequency == 0:
            eval_reward = run_rollout(algo, make_env, env_cfg, iteration=i, device="cpu", logdir=logdir, render=render)

        env_runner_metrics = result.get("env_runners", {})
        total_loss = result['learners']['default_policy']['total_loss']
        episode_return_mean = env_runner_metrics.get("episode_return_mean", 0.0)
        episode_len_mean = env_runner_metrics.get("episode_len_mean", 0.0)
        training_iteration_time_ms = result.get("time_this_iter_s", 0.0)
        num_env_steps_sampled = result["env_runners"]["num_env_steps_sampled"]
        training_time_ms = result.get("time_total_s", 0.0)
        num_total_env_steps_sampled = result["num_env_steps_sampled_lifetime"]
        
        cur_fps = env_multiplier*num_env_steps_sampled/training_iteration_time_ms
        cur_times = {
            "timers": result["timers"],
            "total": training_time_ms,
            "learner_connector": result["learners"]["__all_modules__"]["learner_connector"]["timers"]["connectors"],
            "module_to_env_connector": result["env_runners"]["module_to_env_connector"]["timers"]["connectors"],
            "env_to_module_connector": result["env_runners"]["env_to_module_connector"]["timers"]["connectors"],
            "sample": result["env_runners"]["sample"],
            "fps": cur_fps,
        }

        timings[i] = cur_times
        fps.append(cur_fps)
        print(f"--- Iteration {i+1} ---")
        print(f"  Mean episode return: {episode_return_mean:.2f}")
        print(f"  Last Determinsitic eval return: {eval_reward:.2f}")
        print(f"  Total Loss: {total_loss:.2f}")
        print(f"  Mean episode length: {episode_len_mean:.2f}")
        print(f"  FPS this iter: {cur_fps:.2f}")
        print(f"  Total FPS: {env_multiplier*num_total_env_steps_sampled/training_time_ms:.2f} s")
    return fps, timings


if __name__ == "__main__":
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = os.path.join("./logs", f"run_{timestamp}")
    os.makedirs(logdir, exist_ok=True)

    # # tested on packed vectorized environment 
    # # with these parameters
    # num_iterations = 501
    # eval_frequency = 50
    # num_envs = 1_000
    # train_batch = 10_000
    # minibatch_size = 1_000 #- needs disabling the minibatch check 
    # fps, timings = run(num_iterations, "packed_vec", num_envs, train_batch, minibatch_size, eval_frequency, logdir, render=True)

    num_iterations = 5
    eval_frequency = 50

    num_envss = [10, 100, 1_000, 5_000]
    train_batchs = [10_000, 50_000, 100_000]
    for version in ["packed_vec", "vec"]:
        res = {}
        exp = 0 
        for train_batch in reversed(train_batchs):
            for num_envs in num_envss:
                minibatch_size = train_batch # minibatch is consistent for all versions - needs disabling minibatch check for packed_vec
                fps, timings = run(num_iterations, version, num_envs, train_batch, minibatch_size, eval_frequency, logdir, render=False)
                # Convert all NumPy arrays to lists recursively
                res[exp] = {
                    "times": timings,
                    "num_envs": num_envs,
                    "train_batch_size": train_batch,
                }
                exp += 1
        data_for_json = convert_numpy_to_list(res)

        # Write the dictionary to a JSON file
        with open(f'{logdir}/{version}_data.json', 'w') as f:
            json.dump(data_for_json, f, indent=4)