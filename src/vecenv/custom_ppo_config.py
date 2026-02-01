from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.utils.annotations import override

from vecenv.connectors import PopulateRewardsTruncated, VectorisedGAE, CustomLearnerFlattenColumns


class CustomPPOCOnfig(PPOConfig):
    @override(PPOConfig)
    def build_learner_connector(
        self,
        input_observation_space,
        input_action_space,
        device=None,
    ) -> ConnectorV2:
        from ray.rllib.connectors.learner import (
            AddOneTsToEpisodesAndTruncate,
            AddColumnsFromEpisodesToTrainBatch,
            AddObservationsFromEpisodesToBatch,
            AddStatesFromEpisodesToBatch,
            AddTimeDimToBatchAndZeroPad,
            BatchIndividualItems,
            NumpyToTensor,
        )

        pipeline = super().build_learner_connector(
            input_observation_space=input_observation_space,
            input_action_space=input_action_space,
            device=device,
        )
        # if we are not using the default pipeline,
        # then it is empty unless we add the connectors
        if not self.add_default_connectors_to_learner_pipeline:
            # Append OBS handling.
            pipeline.append(AddOneTsToEpisodesAndTruncate())
            pipeline.append(
                AddObservationsFromEpisodesToBatch(as_learner_connector=True)
            )
            # Append all other columns handling.
            pipeline.append(AddColumnsFromEpisodesToTrainBatch())
            # Append time-rank handler.
            pipeline.append(AddTimeDimToBatchAndZeroPad(as_learner_connector=True))
            # Append STATE_IN/STATE_OUT handler.
            pipeline.append(AddStatesFromEpisodesToBatch(as_learner_connector=True))

            pipeline.append(BatchIndividualItems(multi_agent=self.is_multi_agent))
            # Convert to Tensors.
            pipeline.append(NumpyToTensor(as_learner_connector=True, device=device))            
            pipeline.append(PopulateRewardsTruncated())
            pipeline.append(VectorisedGAE(gamma=self.gamma, lambda_=self.lambda_))
            pipeline.append(CustomLearnerFlattenColumns())
        return pipeline
