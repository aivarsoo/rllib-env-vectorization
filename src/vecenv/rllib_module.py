import torch
import torch.nn as nn

from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.distribution.torch.torch_distribution import TorchDiagGaussian

LOG_STD_MIN = -20
LOG_STD_MAX = 2

class RLModule(TorchRLModule, ValueFunctionAPI):
    def __init__(
        self,
        observation_space=None,
        action_space=None,
        model_config=None,
        config=None,
        **kwargs,
    ):
        if config is None:
            super().__init__(
                observation_space=observation_space,
                action_space=action_space,
                model_config=model_config or {},
                **kwargs,
            )
        else:
            super().__init__(config)

        # --- 1. CONFIGURATION ---
        self.nesting_key = "av_group"
        self.obs_keys = ["hist_track_pos", "hist_track_hed"]
        self.action_keys = ["pred_traj_pos", "pred_traj_hed"]

        # --- 2. INPUT DIM (Obs) ---
        target_obs_space = self.config.observation_space

        # --- 3. OUTPUT DIM (Action) ---
        target_act_space = self.config.action_space

        # --- 4. MODEL ARCHITECTURE ---
        self.action_shape = target_act_space.shape
        self.num_worlds, self.obs_dim = target_obs_space["states"].shape
        self.action_dim = int(self.action_shape[0]/self.num_worlds)


        self.dims = [256, 128, 64]
        self.encoder = nn.Sequential(
            nn.Linear(self.obs_dim, self.dims[0]),
            nn.ELU(),
            nn.Linear(self.dims[0], self.dims[1]),
            nn.ELU(),
            nn.Linear(self.dims[1], self.dims[2]),
        )

        self.pi_mean = nn.Linear(self.dims[2], self.action_dim)
        self.pi_log_std = nn.Linear(self.dims[2], self.action_dim)
        # self.pi_log_std = nn.Parameter(torch.zeros(self.action_dim))

        self.vf_net = nn.Sequential(
            nn.Linear(self.obs_dim, self.dims[0]),
            nn.ELU(),
            nn.Linear(self.dims[0], self.dims[1]),
            nn.ELU(),
            nn.Linear(self.dims[1], self.dims[2]),
            nn.ELU(),
            nn.Linear(self.dims[2], 1)
        )

    # --- 5. DISTRIBUTION REGISTRATION ---
    @override(TorchRLModule)
    def get_train_action_dist_cls(self):
        return TorchDiagGaussian

    @override(TorchRLModule)
    def get_exploration_action_dist_cls(self):
        return TorchDiagGaussian

    @override(TorchRLModule)
    def get_inference_action_dist_cls(self):
        return TorchDiagGaussian

    @override(ValueFunctionAPI)
    def compute_values(self, batch, embeddings=None):
        return self._value_forward(batch)


    @override(TorchRLModule)
    def _forward_inference(self, batch, **kwargs):
        with torch.no_grad():

            action_dist_inputs =  self._policy_forward(batch)
            # We need the first dimension to be equal to 1,
            # since somewhere along the way the actions get unbatched otherwise
            # and the envronment recieves only individual actions
            action_dist_inputs = action_dist_inputs.unsqueeze(0)

            return {
                "action_dist_inputs": action_dist_inputs,
            }
        
    @override(TorchRLModule)
    def _forward_exploration(self, batch, **kwargs):
        with torch.no_grad():

            action_dist_inputs =  self._policy_forward(batch)
            # We need the first dimension to be equal to 1,
            # since somewhere along the way the actions get unbatched otherwise
            # and the envronment recieves only individual actions
            action_dist_inputs = action_dist_inputs.unsqueeze(0)

            return {
                "action_dist_inputs": action_dist_inputs,
            }
    
    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
            action_dist_inputs =  self._policy_forward(batch)

            return {
                "action_dist_inputs": action_dist_inputs,
            }
    
    def _policy_forward(self, batch):
        obs = batch["obs"]["states"]
        embeddings = self.encoder(obs)

        action_mean = self.pi_mean(embeddings)
        action_log_std = self.pi_log_std(embeddings)    

        action_log_std = torch.clamp(action_log_std, LOG_STD_MIN, LOG_STD_MAX)
        action_std = torch.exp(action_log_std)

        action_dist_inputs = torch.cat([action_mean, action_std], dim=-1)
        return action_dist_inputs

    def _value_forward(self, batch):
        obs = batch["obs"]["states"]
        return self.vf_net(obs).squeeze(-1)