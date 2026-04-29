from typing import Type
from stable_baselines3 import A2C, HER, PPO, DQN, DDPG, SAC, TD3  # Import the algorithm you trained on
from sb3_contrib import ARS, MaskablePPO, QRDQN, RecurrentPPO, TQC, TRPO, CrossQ
import gymnasium as gym
from gymnasium import Env
from mo_gymnasium.wrappers import LinearReward
from pathlib import Path
from stable_baselines3.common.base_class import BaseAlgorithm

def initialize_agent_policy(agent_path: Path, env: Env) -> BaseAlgorithm:
    """
        This function initializes the object that serves a policy. We use a policy from stablebaselines3.
        Feel free to replace the policy and tweak hyperparameters if needed.
    """

    agent_path = check_model_path(agent_path)
    model = DQN.load(agent_path, env=env)
    return model

def initialize_agent_environment(env_name: str) -> Env:
    """
    This function initializes the gymnasium environment for your trained policy.
    Feel free to apply different wrappers to your environment
    """
    env = gym.make(env_name, render_mode=None)
    if env_name.startswith("mo-"):
        env = LinearReward(env)
    return env


def check_model_path(agent_str: str) -> Path:
    """
    This function searches for the agent path. It checks for this concrete path up to two directory levels above.
    """
    agent_path = Path(agent_str)
    if agent_path.exists():
        return agent_path
    else:
        raise FileNotFoundError(f"{agent_path} not found. Do you follow a path relative to the main directory?")
