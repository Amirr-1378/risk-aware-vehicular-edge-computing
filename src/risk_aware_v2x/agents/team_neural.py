"""Team-cost DQN/QR learning with independent local-observation inference.

Shares the tested network, optimizer, risk score and update math with the
individual-return agent. Scope/type/checkpoint boundaries remain explicit.
"""

from .neural import NeuralCostAgent
from .team_replay import TeamReplayBuffer, validate_team_transition
from .team_returns import team_return_contract


class TeamNeuralCostAgent(NeuralCostAgent):
    _replay_type = TeamReplayBuffer
    _transition_validator = staticmethod(validate_team_transition)
    _checkpoint_schema_version = 3

    @classmethod
    def learning_return_contract(cls):
        return team_return_contract()

    def _validate_sample(self, transition):
        super()._validate_sample(transition)
        # Even terminal samples must declare the same interval discount.
        if transition.gamma_per_slot != self.config.gamma_per_slot:
            raise ValueError("team transition gamma differs from agent configuration")
