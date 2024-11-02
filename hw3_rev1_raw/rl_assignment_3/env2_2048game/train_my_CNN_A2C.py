import warnings
import gymnasium as gym
from gymnasium.envs.registration import register
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, SubprocVecEnv
from stable_baselines3 import A2C
from stable_baselines3.common.utils import get_device, get_linear_fn
from my_feature_extractor import Custom2048CNN

warnings.filterwarnings("ignore")

register(
    id='2048-v0',
    entry_point='envs:My2048Env'
)

# Set hyper params (configurations) for training
my_config = {
    "run_id": "A2C_2048_improved",
    "algorithm": A2C,
    "policy_network": "CnnPolicy",
    "save_path": "C:/Users/harry/大型資料/RL_HW3_trained_models/3rd_try_my_CNN_diff_alg/improve_model_v1",
    "epoch_num": 1000,
    "timesteps_per_epoch": 10000,
    "eval_episode_num": 15,
    "learning_rate": 1e-3,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "n_steps": 5,
}

def make_env():
    env = gym.make('2048-v0')
    return env

def eval(env, model, eval_episode_num):
    """Evaluate the model and return avg_score and avg_highest"""
    avg_score = 0
    avg_highest = 0
    highest_list = []
    for seed in range(eval_episode_num):
        done = False
        obs, _ = env.reset(seed=seed)
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
        avg_highest += info['highest']
        # print("highest:",info["highest"])
        highest_list.append(info['highest'])
        avg_score += info['score']
        # print("score:",info["score"])
    # print(f"eval_episode_num:{eval_episode_num}")
    avg_highest /= eval_episode_num
    avg_score /= eval_episode_num
    return avg_score, avg_highest, highest_list

# def eval(env, model, eval_episode_num):
#     """Evaluate the model and return avg_score and avg_highest"""
#     avg_score = 0
#     avg_highest = 0
#     highest_list = []
#     for seed in range(eval_episode_num):
#         done = False
#         # Set seed using old Gym API
#         env.seed(seed)
#         obs = env.reset()

#         # Interact with env using old Gym API
#         while not done:
#             action, _state = model.predict(obs, deterministic=True)
#             obs, reward, done, info = env.step(action)
        
#         avg_highest += info[0]['highest']
#         highest_list.append(info[0]['highest'])
#         avg_score   += info[0]['score']

#     avg_highest /= eval_episode_num
#     avg_score /= eval_episode_num
        
#     return avg_score, avg_highest, highest_list

def train(eval_env, model, config):
    """Train agent using SB3 algorithm and my_config"""
    current_best = 0
    current_best_highest = 0
    early_stop_flag = 0
    
    with open(config["save_path"]+"/record.txt", 'w') as file:
        for epoch in range(config["epoch_num"]):
            model.learn(
                total_timesteps=config["timesteps_per_epoch"],
                reset_num_timesteps=False,
                callback=WandbCallback(
                    gradient_save_freq=100,
                    verbose=2,
                ),
            )

            avg_score, avg_highest, highest_list = eval(eval_env, model, config["eval_episode_num"])
            
            print(f"Epoch: {epoch}, Avg_score: {avg_score}, Avg_highest: {avg_highest}")
            wandb.log({
                "avg_highest": avg_highest,
                "avg_score": avg_score,
                "epoch": epoch
            })

            if current_best < avg_score or current_best_highest < max(highest_list):
                print(f"At epoch:{epoch}, Saving Model with score:{avg_score}; highest_block:{max(highest_list)}")
                file.write(f"At epoch:{epoch}, Saving Model with score:{avg_score}; highest_block:{max(highest_list)}\n")
                current_best = max(avg_score, current_best)
                current_best_highest = max(max(highest_list), current_best_highest)
                save_path = config["save_path"]
                model.save(f"{save_path}/{epoch}")
                early_stop_flag = 0
            else:
                early_stop_flag += 1

            if early_stop_flag > 100:
                print("Early stop: There has been no update for 100 epochs!")
                file.write("Early stop: There has been no update for 100 epochs!\n")
                break

        file.write(str(config) + "\n")
        file.write(f"Final_best_Avg_score: {current_best}\n")
        file.write(f"Final_best_highest: {current_best_highest}\n")

    print(f"Final_best_Avg_score: {current_best}")
    print(f"Final_best_highest: {current_best_highest}")

if __name__ == "__main__":
    run = wandb.init(
        project="2048_A2C_improved",
        config=my_config,
        sync_tensorboard=True,
        id=my_config["run_id"]
    )

    device = get_device("cuda")
    print(f"Running on {device}")

    num_train_envs = 12
    train_env = SubprocVecEnv([make_env for _ in range(num_train_envs)])
    eval_env = make_env()

    lr_scheduler = get_linear_fn(
        start=my_config["learning_rate"],
        end=my_config["learning_rate"] * 0.1,
        end_fraction=1.0
    )

    policy_kwargs = dict(
        features_extractor_class=Custom2048CNN,
        features_extractor_kwargs=dict(features_dim=128),
    )

    model = my_config["algorithm"](
        my_config["policy_network"],
        train_env,
        verbose=0,
        tensorboard_log=my_config["run_id"],
        learning_rate=lr_scheduler,
        ent_coef=my_config["ent_coef"],
        vf_coef=my_config["vf_coef"],
        max_grad_norm=my_config["max_grad_norm"],
        n_steps=my_config["n_steps"],
        device=device,
        policy_kwargs=policy_kwargs
    )

    train(eval_env, model, my_config)

    wandb.finish()