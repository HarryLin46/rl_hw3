import warnings
import gymnasium as gym
import wandb
import numpy as np
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, SubprocVecEnv
from stable_baselines3 import DQN
from stable_baselines3.common.utils import get_device, get_linear_fn
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.evaluation import evaluate_policy
from my_donkey_model import DonkeyKongCNN

warnings.filterwarnings("ignore")

# Set hyper params (configurations) for training
my_config = {
    "run_id": "DQN_DonkeyKong",
    "algorithm": DQN,
    "policy_network": "CnnPolicy",
    "save_path": "./trained_models/donkey_kong_model",
    "epoch_num": 1000,
    "timesteps_per_epoch": 5000,
    "eval_episode_num": 10,
    "learning_rate": 1e-4,
    "buffer_size": 100000,
    "learning_starts": 10000,
    "batch_size": 32,
    "tau": 1.0,
    "gamma": 0.99,
    "train_freq": 4,
    "gradient_steps": 1,
    "target_update_interval": 1000,
    "exploration_fraction": 0.1,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.01,
}

def make_env(render_mode=None):
    env = gym.make('ALE/DonkeyKong-v5', render_mode=render_mode)
    env = AtariWrapper(env)
    return env

def eval(env, model, eval_episode_num):
    """Evaluate the model and return avg_score"""
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=eval_episode_num)
    return mean_reward, std_reward

def train(eval_env, model, config):
    """Train agent using SB3 algorithm and my_config"""
    current_best = float('-inf')
    early_stop_flag = 0
    
    for epoch in range(config["epoch_num"]):
        model.learn(
            total_timesteps=config["timesteps_per_epoch"],
            reset_num_timesteps=False,
            callback=WandbCallback(
                gradient_save_freq=100,
                verbose=2,
            ),
        )

        # Evaluation
        mean_reward, std_reward = eval(eval_env, model, config["eval_episode_num"])
        
        print(f"Epoch: {epoch}, Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        wandb.log({
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "epoch": epoch
        })

        # Save best model
        if current_best < mean_reward:
            print(f"At epoch:{epoch}, Saving Model with mean reward:{mean_reward:.2f}")
            current_best = mean_reward
            save_path = config["save_path"]
            model.save(f"{save_path}/{epoch}")
            early_stop_flag = 0
        else:
            early_stop_flag += 1

        if early_stop_flag > 100:
            print("Early stop: There has been no update for 100 epochs!")
            break

    print(f"Final best mean reward: {current_best:.2f}")

if __name__ == "__main__":
    # Create wandb session
    run = wandb.init(
        project="donkey_kong_dqn",
        config=my_config,
        sync_tensorboard=True,
        id=my_config["run_id"]
    )

    # Run on GPU if available
    device = get_device("cuda")
    print(f"Running on {device}")

    # Create training environment
    num_train_envs = 4
    train_env = SubprocVecEnv([lambda: make_env() for _ in range(num_train_envs)])

    # Create evaluation environment
    eval_env = make_env()

    # Create model
    lr_scheduler = get_linear_fn(
        start=my_config["learning_rate"],
        end=my_config["learning_rate"] * 0.1,
        end_fraction=1.0
    )

    policy_kwargs = dict(
        features_extractor_class=DonkeyKongCNN,
        features_extractor_kwargs=dict(features_dim=512),
    )

    model = my_config["algorithm"](
        my_config["policy_network"],
        train_env,
        verbose=0,
        tensorboard_log=my_config["run_id"],
        learning_rate=lr_scheduler,
        buffer_size=my_config["buffer_size"],
        learning_starts=my_config["learning_starts"],
        batch_size=my_config["batch_size"],
        tau=my_config["tau"],
        gamma=my_config["gamma"],
        train_freq=my_config["train_freq"],
        gradient_steps=my_config["gradient_steps"],
        target_update_interval=my_config["target_update_interval"],
        exploration_fraction=my_config["exploration_fraction"],
        exploration_initial_eps=my_config["exploration_initial_eps"],
        exploration_final_eps=my_config["exploration_final_eps"],
        device=device,
        policy_kwargs=policy_kwargs
    )

    train(eval_env, model, my_config)

    # Close wandb run
    wandb.finish()

    # Run a final evaluation and render the result
    eval_env = make_env(render_mode="human")
    obs, _ = eval_env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = eval_env.step(action)
        eval_env.render()
        if terminated or truncated:
            break
    eval_env.close()