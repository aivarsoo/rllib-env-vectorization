
from typing import List, Any, Dict, Optional

import torch

# --- RLlib Imports ---
from ray.rllib.core import Columns
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import EpisodeType
from ray.rllib.connectors.connector_v2 import ConnectorV2

from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI

# TODO: unite the Flatteners into one connector

class CustomFlattenColumns(ConnectorV2):

    @override(ConnectorV2)
    def __call__(
        self,
        *,
        rl_module: RLModule,
        batch: Dict[str, Any],
        episodes: List[EpisodeType],
        explore: Optional[bool] = None,
        shared_data: Optional[dict] = None,
        **kwargs,
    ) -> Any:

        B, W, _ = batch['obs']['states'].shape
        for col, column_data in batch.copy().items():
            if col == Columns.OBS:
                old_obs = column_data['states']
                column_data = {
                    **column_data,
                    "states": old_obs.reshape((B * W, -1))
                }
            if col in [
                Columns.REWARDS,
                Columns.TERMINATEDS,
                Columns.TRUNCATEDS,
                Columns.VF_PREDS,
                Postprocessing.ADVANTAGES,
                Postprocessing.VALUE_TARGETS,
                Columns.ACTION_LOGP
            ]:
                column_data = column_data.reshape((B * W,))
            if col in [
                Columns.ACTIONS,
                Columns.ACTION_DIST_INPUTS,
            ]:
                column_data = column_data.reshape((B * W, -1))
            if col in [
                "loss_mask",
                "weights_seq_no"
            ]:
                column_data = column_data.repeat_interleave(W)
            batch[col] = column_data
        return batch


class CustomLearnerFlattenColumns(ConnectorV2):

    @override(ConnectorV2)
    def __call__(
        self,
        *,
        rl_module: RLModule,
        batch: Dict[str, Any],
        episodes: List[EpisodeType],
        explore: Optional[bool] = None,
        shared_data: Optional[dict] = None,
        **kwargs,
    ) -> Any:
        for _module, module_data in batch.copy().items():
            B, W, _ = module_data['obs']['states'].shape
            for col, column_data in module_data.items():
                if col == Columns.OBS:
                    old_obs = column_data['states']
                    column_data = {
                        **column_data,
                        "states": old_obs.reshape((B * W, -1))
                    }
                if col in [
                    Columns.REWARDS,
                    Columns.TERMINATEDS,
                    Columns.TRUNCATEDS,
                    Columns.VF_PREDS,
                    Postprocessing.ADVANTAGES,
                    Postprocessing.VALUE_TARGETS,
                    Columns.ACTION_LOGP
                ]:
                    column_data = column_data.reshape((B * W,))
                if col in [
                    Columns.ACTIONS,
                    Columns.ACTION_DIST_INPUTS,
                ]:
                    column_data = column_data.reshape((B * W, -1))
                if col in [
                    "loss_mask",
                    "weights_seq_no"
                ]:
                    column_data = column_data.repeat_interleave(W)
                module_data[col] = column_data
            batch[_module] = module_data
        return batch


class PopulateRewardsTruncated(ConnectorV2):
    @override(ConnectorV2)
    def __call__(
        self,
        *,
        rl_module: RLModule,
        batch: Dict[str, Any],
        episodes: List[EpisodeType],
        explore: Optional[bool] = None,
        shared_data: Optional[dict] = None,
        **kwargs,
    ) -> Any:

        for _module, module_data in batch.copy().items():
            for column, data in module_data.items():
                if column in [
                        Columns.REWARDS,
                        Columns.TERMINATEDS,
                        Columns.TRUNCATEDS,
                    ]:
                    # popping the column for the state observation
                    # sice we do not need it anymore
                    old_col = module_data['obs'].pop(column)
                    if column == Columns.REWARDS:
                        new_col = torch.zeros_like(old_col)
                    else:
                        new_col = torch.ones_like(old_col)
                    new_col[:-1, ...] = old_col[1:, ...]
                    module_data[column] = new_col
        return batch


class VectorisedGAE(ConnectorV2):
    """
    A pure PyTorch implementation of Generalized Advantage Estimation (GAE).
    """

    def __init__(
        self,
        input_observation_space=None,
        input_action_space=None,
        *,
        gamma: float = 0.99,
        lambda_: float = 1.0,
    ):
        super().__init__(input_observation_space, input_action_space)
        self.gamma = gamma
        self.lambda_ = lambda_

    @override(ConnectorV2)
    def __call__(
        self,
        *,
        rl_module: MultiRLModule,
        episodes: List[EpisodeType],
        batch: Dict[str, Any],
        **kwargs,
    ):
        with torch.no_grad():
            vf_preds = rl_module.foreach_module(
                func=lambda mid, module: (
                    module.compute_values(batch[mid])
                    if mid in batch and isinstance(module, ValueFunctionAPI)
                    else None
                ),
                return_dict=True,
            )

        def compute_gae(deltas, terminateds_t):
            gae = 0.0
            advantages = torch.zeros_like(deltas)
            for t in reversed(range(deltas.shape[0])):
                gae = (
                    deltas[t, :]
                    + self.gamma * self.lambda_ * (1.0 - terminateds_t[t, :]) * gae
                )
                advantages[t, :] = gae
            return advantages
            
        for module_id, module_vf_preds in vf_preds.items():
            if module_vf_preds is None:
                continue

            rewards_all = batch[module_id][Columns.REWARDS]
            terminateds_all = batch[module_id][Columns.TERMINATEDS].to(dtype=rewards_all.dtype)
            values_all = module_vf_preds
            # # Handle Batch Truncation/Mismatch (Robustness)
            # min_len = min(values_all.shape[0], rewards_all.shape[0])
            # if values_all.shape[0] != rewards_all.shape[0]:
            #     values_all = values_all[:min_len]
            #     rewards_all = rewards_all[:min_len]
            #     terminateds_all = terminateds_all[:min_len]

            # GAE Computation
            values_t = values_all[:-1, :]
            rewards_t = rewards_all[:-1, :]
            terminateds_t = terminateds_all[:-1, :]
            values_t_plus_1 = values_all[1:, :]

            deltas = (
                rewards_t
                + self.gamma * (1.0 - terminateds_t) * values_t_plus_1
                - values_t
            )

            advantages = compute_gae(deltas, terminateds_t)
            value_targets = (advantages + values_t)

            # Normalize Advantages (Critical for PPO stability)
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

            # Pad back to original length to match batch shape
            # We add 0.0 for the last timestep which was sliced off
            _, W = advantages.shape
            advantages_padded = torch.cat(
                [advantages, torch.zeros((1, W), device=advantages.device)]
            )
            value_targets_padded = torch.cat(
                [value_targets, torch.zeros((1, W), device=value_targets.device)]
            )

            # Ensure these new tensors are detached (Redundant safety)
            batch[module_id][Postprocessing.ADVANTAGES] = advantages_padded.detach()
            batch[module_id][
                Postprocessing.VALUE_TARGETS
            ] = value_targets_padded.detach()

        return batch
