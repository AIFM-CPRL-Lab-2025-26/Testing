from typing import Type
from stable_baselines3 import A2C, HER, PPO, DQN, DDPG, SAC, TD3  # Import the algorithm you trained on
from sb3_contrib import ARS, MaskablePPO, QRDQN, RecurrentPPO, TQC, TRPO, CrossQ
from gymnasium import Env
from pathlib import Path
from stable_baselines3.common.base_class import BaseAlgorithm

def initialize_agent_policy(agent_path: Path, env: Env, algo_str: str) -> BaseAlgorithm:
    """
        This function initializes the object that serves a policy. We use a policy from stablebaselines3.
        Feel free to replace the policy and tweak hyperparameters if needed.
    """
    sb3_algorithms = {
        "A2C": A2C,
        "DDPG": DDPG,
        "DQN": DQN,
        "HER": HER,
        "PPO": PPO,
        "SAC": SAC,
        "TD3": TD3,
        "ARS": ARS,
        "MaskablePPO": MaskablePPO,
        "QRDQN": QRDQN,
        "QR-DQN": QRDQN,
        "QR_DQN": QRDQN,
        "TRPO": TRPO,
        "TQC": TQC,
        "RPPO": RecurrentPPO,
        "RECURRENTPPO": RecurrentPPO,
        "CROSSQ": CrossQ
    }

    agent_path = check_model_path(agent_path)
    algo_obj = sb3_algorithms[algo_str.upper()]
    model = algo_obj.load(agent_path, env=env)
    return model

def check_model_path(agent_path: Path) -> Path:
    """
    This function searches for the agent path including the upper directories.
    """
    cur_dir = Path(__file__).resolve().parent
    for _ in range(3):
        full_path = cur_dir.joinpath(agent_path)
        if full_path.exists():
            return full_path
        else:
            cur_dir = cur_dir.parent

    raise FileNotFoundError(f"{agent_path} not found.")