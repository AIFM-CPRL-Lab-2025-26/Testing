"""
This file is automatically executed to sync with the leaderboard.
Do not modify! Modified files cannot be used for the leaderboard.
"""
import numpy as np
import json
import highway_env
import gymnasium as gym
import mo_gymnasium
from pathlib import Path
from mo_gymnasium.wrappers import LinearReward
from tqdm.auto import tqdm
from CPRL.agent import initialize_agent_policy

SEED = 79

def get_reward_score(env_name, reward):
    """
    This function returns a specific combination of the rewards for the hidden_old evaluation
    """
    episode_reward = 0.0
    secret = 1.0
    if env_name.startswith("mo-lunar-lander-v3"):
        # reward[0] = landing, reward[1] = shaping, reward[2] = main engine, reward[3] = side engine
        episode_reward += float(secret * reward[0] + secret * reward[2] + secret * reward[3])
    elif env_name.startswith("mo-highway-v0"):
        # reward[0] = high speed, reward[1] = right lane, reward[2] = collision
        episode_reward += float(secret * reward[0] + secret * reward[1] + secret * reward[2])
    else:
        episode_reward += float(reward)
    return episode_reward

def run_task(agent_path, env, eval_episodes, algo, env_name):
    model = initialize_agent_policy(agent_path, env, algo) # You can define your model in this function.

    episode_rewards = []
    weighted_rewards = []
    totals = []

    for i in range(eval_episodes):
        if env_name.startswith("mo-"):
            weighted_reward = 0.0
            if env_name == "mo-lunar-lander-v3":
                total = np.zeros(4)
            elif env_name == "mo-highway-v0":
                total = np.zeros(3)
        lstm_states = None
        episode_start = np.atleast_1d(True)

        obs, _ = env.reset(seed=SEED+i)
        episode_reward = 0.0
        for _ in range(10000):
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_start,
                deterministic=True
            )
            obs, reward, terminated, truncated, _ = env.step(int(action))
            if isinstance(reward, np.ndarray):
                total += reward
                episode_reward += float(np.sum(reward))
                weighted_reward += get_reward_score(env_name, reward)
            else:
                episode_reward += float(reward)
            episode_start = np.atleast_1d(False)
            if (terminated is True) or (truncated is True):
                episode_rewards.append(episode_reward)
                if isinstance(reward, np.ndarray):
                    totals.append(total)
                    weighted_rewards.append(weighted_reward)
                break

    results = {"main_reward": episode_rewards}

    if len(totals) != 0:
        results.update({"weighted_reward": weighted_rewards})
        total_metrics = {
            f"metric_{i+1}": [val[i] for val in totals]
            for i in range(len(totals[0]))
        }
        results.update(total_metrics)

    return results

def leaderboard_eval(agent_path, eval_episodes, algo, environments):
    out_data = {}

    if isinstance(environments, str):
        environments = [environments]

    for env_name in tqdm(environments, desc="Evaluating"):
        print(f"Evaluating on Task {env_name}.")
        env = gym.make(env_name, render_mode=None)
        #if env_name.startswith("mo-"):
        #    env = LinearReward(env)
        out = run_task(agent_path, env, eval_episodes, algo, env_name)
        out_data[env_name] = out
        print(f"Finished evaluating on Task {env_name}.")
        print(f"Closing {env_name}")
        env.close()

    return out_data

if __name__ == "__main__":
    cur_dir = Path(__file__).parent
    eval_config_file = "eval_config.json"
    eval_config_file = cur_dir / eval_config_file

    print(f"\n=== Evaluating Policy ===\n")
    # Read config
    with open(eval_config_file) as ecf:
        eval_configs = json.load(ecf)

    for eval_config in eval_configs:
        # Run evaluation
        out = leaderboard_eval(
            agent_path=eval_config["path_to_agent"],
            eval_episodes=eval_config["eval_episodes"],
            algo=eval_config['algorithm'],
            environments=eval_config['environments']
        )

        # Write results to file
        eval_save_file = f"eval_data_{eval_config['environments']}.json"
        save_dir = cur_dir / eval_save_file
        with open(save_dir, "w") as eof:
            json.dump(out, eof)

        print("\n=== Evaluation completed ===")