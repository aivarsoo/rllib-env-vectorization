__credits__ = ["Carlos Luis", "Aivar Sootla"]

from os import path

import numpy as np
import torch
import tree
from typing import Any, Dict

import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled

from gymnasium.envs.registration import register
from gymnasium.core import ActType, ObsType
from gymnasium.vector.vector_env import AutoresetMode, VectorEnv, ArrayType

DEFAULT_X = np.pi
DEFAULT_Y = 1.0

class PendulaEnv(gym.Env):
    """
    ## Description

    A vectorized torch version of the classic control environment Pendulum Swing-up.
 
    The inverted pendulum swingup problem is based on the classic problem in control theory.
    The system consists of a pendulum attached at one end to a fixed point, and the other end being free.
    The pendulum starts in a random position and the goal is to apply torque on the free end to swing it
    into an upright position, with its center of gravity right above the fixed point.

    The diagram below specifies the coordinate system used for the implementation of the pendulum's
    dynamic equations.


    - `x-y`: cartesian coordinates of the pendulum's end in meters.
    - `theta` : angle in radians.
    - `tau`: torque in `N m`. Defined as positive _counter-clockwise_.

    ## Action Space

    The action is a `ndarray` with shape `(1,)` representing the torque applied to free end of the pendulum.

    | Num | Action | Min  | Max |
    |-----|--------|------|-----|
    | 0   | Torque | -2.0 | 2.0 |

    ## Observation Space

    The observation is a `ndarray` with shape `(3,)` representing the x-y coordinates of the pendulum's free
    end and its angular velocity.

    | Num | Observation      | Min  | Max |
    |-----|------------------|------|-----|
    | 0   | x = cos(theta)   | -1.0 | 1.0 |
    | 1   | y = sin(theta)   | -1.0 | 1.0 |
    | 2   | Angular Velocity | -8.0 | 8.0 |

    ## Rewards

    The reward function is defined as:

    *r = -(theta<sup>2</sup> + 0.1 * theta_dt<sup>2</sup> + 0.001 * torque<sup>2</sup>)*

    where `theta` is the pendulum's angle normalized between *[-pi, pi]* (with 0 being in the upright position).
    Based on the above equation, the minimum reward that can be obtained is
    *-(pi<sup>2</sup> + 0.1 * 8<sup>2</sup> + 0.001 * 2<sup>2</sup>) = -16.2736044*,
    while the maximum reward is zero (pendulum is upright with zero velocity and no torque applied).

    ## Starting State

    The starting state is a random angle in *[-pi, pi]* and a random angular velocity in *[-1,1]*.

    ## Episode Truncation

    The episode truncates at 200 time steps.

    ## Arguments

    - `g`: .

    Pendulum has two parameters for `gymnasium.make` with `render_mode` and `g` representing
    the acceleration of gravity measured in *(m s<sup>-2</sup>)* used to calculate the pendulum dynamics.
    The default value is `g = 10.0`.
    On reset, the `options` parameter allows the user to change the bounds used to determine the new random state.

    ```python
    >>> import gymnasium as gym
    >>> env = gym.make("Pendulum-v1", render_mode="rgb_array", g=9.81)  # default g=10.0
    >>> env
    <TimeLimit<OrderEnforcing<PassiveEnvChecker<PendulumEnv<Pendulum-v1>>>>>
    >>> env.reset(seed=123, options={"low": -0.7, "high": 0.5})  # default low=-0.6, high=-0.5
    (array([ 0.4123625 ,  0.91101986, -0.89235795], dtype=float32), {})

    ```

    ## Version History

    * v1: Simplify the math equations, no difference in behavior.
    * v0: Initial versions release
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: str | None = None, g:float=10.0, max_episode_steps:int=200, num_envs:int=1, device:str="cpu"):
        # Number of environments in the vector

        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = g
        self.m = 1.0
        self.l = 1.0

        self.render_mode = render_mode

        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)

        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised as max_torque == 2 by default. Ignoring the issue here as the default settings are too old
        #   to update to follow the gymnasium api
        vec_high = high.reshape(-1, 1).repeat(self.num_envs,1).T
        self.random_action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(self.num_envs, 1), dtype=np.float32
        )
        # self.observation_space = spaces.Box(low=-vec_high, high=vec_high, shape=(self.num_envs, 3), dtype=np.float32)

        self.single_action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.single_observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space

        self.device = device 
        self.step_idx = torch.zeros(size=(self.num_envs,), device=self.device)
        self.active_envs = torch.ones(size=(self.num_envs,), device=self.device, dtype=torch.bool)
        self.last_u = torch.full(
            size=(self.num_envs, 1),
            fill_value=torch.nan,
        )
        self.state = torch.zeros(
            size=(self.num_envs, 2),
        )

    def step(self, u: torch.Tensor):
        self.step_idx[self.active_envs] += 1
        th = self.state[self.active_envs, 0].unsqueeze(1)
        thdot = self.state[self.active_envs, 1].unsqueeze(1)

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u[self.active_envs] = torch.clip(u[self.active_envs], -self.max_torque, self.max_torque)
        self.last_u = u  # for rendering
        costs = torch.zeros((self.num_envs,), device=self.device)
        costs[self.active_envs] = (angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u[self.active_envs]**2)).reshape(-1)

        newthdot = thdot + (3 * g / (2 * l) * torch.sin(th) + 3.0 / (m * l**2) * u[self.active_envs]) * dt
        newthdot = torch.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.state[self.active_envs] = torch.cat([newth, newthdot], dim=1)
        truncated =  self.step_idx >= self.max_episode_steps
        terminated = torch.zeros_like(truncated, dtype=torch.bool)
        self.active_envs = (terminated.logical_or(truncated)).logical_not()
        # if (self.step_idx >= self.max_episode_steps).any():
        #     print("here")
        return self._get_obs(), -costs, terminated, truncated, {}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # TODO: optimize so that we don't have convert random np arrays to tensors
        reset_mask = torch.as_tensor(
            options.get(
                "reset_mask",
                np.ones(shape=(self.num_envs,), dtype=np.bool_)
            ),
            device=self.device,
        )
        num_envs_to_reset = int(reset_mask.sum().item())
        high = np.array([[DEFAULT_X, DEFAULT_Y]])
        vec_high = high.T.repeat(num_envs_to_reset, 1).T
        vec_low = -vec_high  # We enforce symmetric limits.
        # TODO: optimize so that we don't have convert random np arrays to tensors
        self.state[reset_mask] = torch.as_tensor(
            self.np_random.uniform(low=vec_low, high=vec_high), device=self.device, dtype=torch.float32
        )
        self.last_u[reset_mask] = torch.nan
        self.step_idx[reset_mask] = 0
        self.active_envs[reset_mask] = True
        return self._get_obs(), {}

    def _get_obs(self):
        theta = self.state[:, 0].unsqueeze(1)
        thetadot = self.state[:, 1].unsqueeze(1)
        return torch.cat([torch.cos(theta), torch.sin(theta), thetadot], dim=1)

    def render(self, sub_env_idx: int = 0):
        """
        Render one sub environment 
        
        :param subenv_idx: Rendered sub environment index 
        :type subenv_idx: int
        """
        sub_env_state = self.state[sub_env_idx].cpu().numpy()
        sub_env_action = self.last_u[[sub_env_idx]].cpu().numpy()[0] if self.last_u is not None else None
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_dim, self.screen_dim)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

        rod_length = 1 * scale
        rod_width = 0.2 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(sub_env_state[0] + np.pi / 2)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

        gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(
            self.surf, offset, offset, int(rod_width / 2), (204, 77, 77)
        )

        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(sub_env_state[0] + np.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )
        gfxdraw.filled_circle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )

        fname = path.join(path.dirname(__file__), "assets/clockwise.png")
        img = pygame.image.load(fname)
        if sub_env_action is not None:
            scale_img = pygame.transform.smoothscale(
                img,
                (
                    float(scale * np.abs(sub_env_action) / 2),
                    float(scale * np.abs(sub_env_action) / 2),
                ),
            )
            is_flip = bool(sub_env_action > 0)
            scale_img = pygame.transform.flip(scale_img, is_flip, True)
            self.surf.blit(
                scale_img,
                (
                    offset - scale_img.get_rect().centerx,
                    offset - scale_img.get_rect().centery,
                ),
            )

        # drawing axle
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        else:  # mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


class VecEnvWrapper(VectorEnv):
    def __init__(self, env):
        super().__init__()
        self.autoreset_mode = AutoresetMode.NEXT_STEP
        self.metadata["autoreset_mode"] = self.autoreset_mode
        self.wrapped_env = env
        self._autoreset_envs = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
        ) -> tuple[ObsType, dict[str, Any]]:
        options = {} if options is None else options
        obs, info = self.wrapped_env.reset(seed=seed, options=options)
        self._autoreset_envs = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        return self._process_outputs(obs, info)

    def step(self, action_dict: ActType) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict[str, Any]]:
        action_dict = self._process_inputs(action_dict)
        reset_mask = self._autoreset_envs.clone()
        if not reset_mask.all():
            obs, reward, terminated, truncated, info = self.wrapped_env.step(action_dict)
        else:
            reward = torch.zeros(size=reset_mask.shape)
            terminated = torch.zeros_like(reset_mask)
            truncated = torch.zeros_like(reset_mask)
            info = None
        if self.autoreset_mode == AutoresetMode.NEXT_STEP:
            if reset_mask.any():
                # resetting envs that were inactive before the step. Note we don't do
                # a step and we need to do only do a partial reset on a state.
                # Therefore, the returned observation will contain both
                # updated active env state and reset world state
                obs, res_info = self.wrapped_env.reset(options={"reset_mask": reset_mask})

                reward[reset_mask] = 0  # dummy reward - not used in training
                terminated[reset_mask] = False
                truncated[reset_mask] = False
                info = res_info if info is None else info
            self._autoreset_envs = terminated.logical_or(truncated)
        else:
            # TODO Consider implementing AutoresetMode.SAME_STEP, but we will need to
            # unbatch reset observations to fill in "final_obs" field in infos.
            raise ValueError(f"Unexpected autoreset mode, {self.autoreset_mode}")
        return self._process_outputs(obs, reward, terminated, truncated, info)

    def render(self, sub_env_idx:int=0):
        return self.wrapped_env.render(sub_env_idx=sub_env_idx)

    def close(self):
        self.wrapped_env.close()

    def __getattr__(self, name):
        """
        This is called only when 'name' isn't found in Wrapper's own dictionary.
        It redirects the lookup to the 'wrapped' object. 

        NB: If an attribute is defined in `VectorEnv` class, then we this function
        we call it instead of the method in self.wrapped_env
        """
        return getattr(self.wrapped_env, name)

    def _process_outputs(self, *args):
        """Process the outputs."""
        return tuple(self._maybe_to_numpy(arg) for arg in args)

    def _maybe_to_numpy(self, src: Dict[str, Any]) -> Dict[str, Any]:
        return tree.map_structure(
            lambda x: x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x,
            src,
        )

    def _process_inputs(self, src: Dict[str, Any]) -> Dict[str, Any]:
        return tree.map_structure(
            lambda x: torch.as_tensor(x, device=self.wrapped_env.device)
            if not isinstance(x, (dict, list, tuple))
            else x,
            src,
        )


class PackedVecEnvWrapper(gym.Env):
    def __init__(self, num_envs:int=1, render_mode="human", max_episode_steps:int=200):
        super().__init__()

        self.autoreset_mode = AutoresetMode.NEXT_STEP
        self.metadata["autoreset_mode"] = self.autoreset_mode
        self.num_envs = num_envs
        self.wrapped_env = PendulaEnv(num_envs=num_envs, render_mode=render_mode, max_episode_steps=max_episode_steps)
        self._autoreset_envs = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        vec_high = high.reshape(-1, 1).repeat(self.num_envs,1).T
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(self.num_envs, 1), dtype=np.float32
        )
        self.observation_space = spaces.Dict(
            {
                "states": spaces.Box(low=-vec_high, high=vec_high, dtype=np.float32),
                "rewards": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_envs,), dtype=np.float32),
                "terminateds": spaces.Box(low=0, high=1, shape=(self.num_envs,), dtype=np.int64),
                "truncateds": spaces.Box(low=0, high=1, shape=(self.num_envs,), dtype=np.bool),
            }
        )
        # self.single_observation_space = self.observation_space
        # self.single_action_space = self.action_space

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
        ) -> tuple[ObsType, dict[str, Any]]:
        options = {} if options is None else options
        obs, info = self.wrapped_env.reset(seed=seed, options=options)
        self._autoreset_envs = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        rewards = torch.zeros(
            (self.num_envs,),  device=self.device
        )
        terminated = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        truncated = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        return self._process_outputs(self._get_obs(obs, rewards, terminated, truncated), info)

    def step(self, action_dict: ActType) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict[str, Any]]:
        action_dict = self._process_inputs(action_dict)
        reset_mask = self._autoreset_envs.clone()
        if not reset_mask.all():
            obs, reward, terminated, truncated, info = self.wrapped_env.step(action_dict)
        else:
            reward = torch.zeros(size=reset_mask.shape)
            terminated = torch.zeros_like(reset_mask)
            truncated = torch.zeros_like(reset_mask)
            info = None
        if self.autoreset_mode == AutoresetMode.NEXT_STEP:
            if reset_mask.any():
                # resetting envs that were inactive before the step. Note we don't do
                # a step and we need to do only do a partial reset on a state.
                # Therefore, the returned observation will contain both
                # updated active env state and reset world state
                obs, res_info = self.wrapped_env.reset(options={"reset_mask": reset_mask})

                reward[reset_mask] = 0  # dummy reward - not used in training
                terminated[reset_mask] = False
                truncated[reset_mask] = False
                info = res_info if info is None else info
            self._autoreset_envs = terminated.logical_or(truncated)
        else:
            # TODO Consider implementing AutoresetMode.SAME_STEP, but we will need to
            # unbatch reset observations to fill in "final_obs" field in infos.
            raise ValueError(f"Unexpected autoreset mode, {self.autoreset_mode}")
        return self._process_outputs(self._get_obs(obs, reward, terminated, truncated), reward.mean(), terminated.all(), truncated.all(), info)

    def render(self, sub_env_idx:int=0):
        return self.wrapped_env.render(sub_env_idx=sub_env_idx)

    def close(self):
        self.wrapped_env.close()

    def __getattr__(self, name):
        """
        This is called only when 'name' isn't found in Wrapper's own dictionary.
        It redirects the lookup to the 'wrapped' object. 

        NB: If an attribute is defined in `VectorEnv` class, then we this function
        we call it instead of the method in self.wrapped_env
        """
        return getattr(self.wrapped_env, name)

    def _process_outputs(self, *args):
        """Process the outputs."""
        return tuple(self._maybe_to_numpy(arg) for arg in args)

    def _maybe_to_numpy(self, src: Dict[str, Any]) -> Dict[str, Any]:
        return tree.map_structure(
            lambda x: x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x,
            src,
        )

    def _process_inputs(self, src: Dict[str, Any]) -> Dict[str, Any]:
        return tree.map_structure(
            lambda x: torch.as_tensor(x, device=self.wrapped_env.device)
            if not isinstance(x, (dict, list, tuple))
            else x,
            src,
        )

    def _get_obs(self, obs, reward, terminated, truncated):
        return {
            "states": obs,
            "rewards": reward,
            "terminateds": terminated,
            "truncateds": truncated,
        }


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


if __name__ == "__main__":

    register(
        "VecPendulumEnv-v0",
        entry_point=PackedVecEnvWrapper,
        # max_episode_steps=20,
    )
    num_envs = 10
    env = gym.make("VecPendulumEnv-v0", render_mode="human", num_envs=num_envs)
    done = np.zeros(shape=(num_envs, ), dtype=np.bool)
    frames = []
    step_id = 0
    obs, _ = env.reset(options={"reset_mask": done})
    while not done.all():
        if done.any():
            obs, _ = env.reset(options={"reset_mask": done})
        action = env.unwrapped.random_action_space.sample()
        obs, rew, term, trunc, info = env.step(torch.as_tensor(action))
        done = term | trunc
        step_id += 1
        print(step_id, term.any(), trunc.any())
        frames.append(env.render())
    print("done")

