import warnings
import gymnasium as gym
from gymnasium.envs.registration import register

import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, SubprocVecEnv
from stable_baselines3 import A2C, DQN, PPO, SAC,DDPG

from stable_baselines3.common.utils import get_device
from stable_baselines3.common.utils import get_schedule_fn,get_linear_fn

### REMEMBER TO EXCLUDE!!!
import os
import torch
from sb3_contrib import QRDQN  # QRDQN是Rainbow DQN的一個變體
### REMEMBER TO EXCLUDE!!!
from my_feature_extractor import Custom2048CNN  # 假設您將上面的代碼保存在custom_cnn.py文件中

warnings.filterwarnings("ignore")
register(
    id='2048-v0',
    entry_point='envs:My2048Env'
)

# Set hyper params (configurations) for training
my_config = {
    "run_id": "A2C_lr8e-3", #DQN_lr8e-3

    "algorithm": A2C, #DQN
    "policy_network": "CnnPolicy",
    "save_path": "C:/Users/harry/大型資料/RL_HW3_trained_models/improve_model_v6",

    "epoch_num": 8000,
    "timesteps_per_epoch": 3000,
    "eval_episode_num": 15,
    "learning_rate": 4e-2, #8e-3 looks good
}


def make_env():
    # env = gym.make('2048-v0', render_mode="ansi")
    env = gym.make('2048-v0')
    return env

def eval(env, model, eval_episode_num):
    """Evaluate the model and return avg_score and avg_highest"""
    avg_score = 0
    avg_highest = 0
    highest_list = []
    for seed in range(eval_episode_num):
        done = False
        # Set seed using old Gym API
        env.seed(seed)
        obs = env.reset()

        # Interact with env using old Gym API
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
        
        avg_highest += info[0]['highest']
        highest_list.append(info[0]['highest'])
        avg_score   += info[0]['score']

    avg_highest /= eval_episode_num
    avg_score /= eval_episode_num
        
    return avg_score, avg_highest, highest_list

def train(eval_env, model, config):
    """Train agent using SB3 algorithm and my_config"""
    current_best = 0
    current_best_highest= 0

    early_stop_flag=0

    #write a txt to record why the model is saved
    with open(config["save_path"]+"/record.txt", 'w') as file:
    
        for epoch in range(config["epoch_num"]):

            # Uncomment to enable wandb logging
            model.learn(
                total_timesteps=config["timesteps_per_epoch"],
                reset_num_timesteps=False,
                # callback=WandbCallback(
                #     gradient_save_freq=100,
                #     verbose=2,
                # ),
            )

            ### Evaluation
            avg_score, avg_highest, highest_list = eval(eval_env, model, config["eval_episode_num"])
            
            # if epoch%100==0:
            #     # print(config["run_id"])
            #     print("Epoch: ", epoch)
            #     print(f"current_best_Avg_score: {current_best}")
            #     print(f"current_best_Avg_highest: {avg_highest}")
            #     print(f"current_episode_best_highest: {max(highest_list)}")
            #     print(f"all_best_highest_block: {current_best_highest}")
            
            # print("Avg_score:  ", avg_score)
            # print("Avg_highest:", avg_highest)
            # print()
            # wandb.log(
            #     {"avg_highest": avg_highest,
            #      "avg_score": avg_score}
            # )
            

            ### Save best model
            if current_best < avg_score or current_best_highest < max(highest_list):
                print(f"At epoch:{epoch}, Saving Model with score:{avg_score}; highest_block:{max(highest_list)}")
                file.write(f"At epoch:{epoch}, Saving Model with score:{avg_score}; highest_block:{max(highest_list)}\n")
                current_best = max(avg_score,current_best)
                current_best_highest = max(max(highest_list),current_best_highest)
                save_path = config["save_path"]
                model.save(f"{save_path}/{epoch}")
                early_stop_flag=0 #reset flag
            else:
                early_stop_flag+=1
            
            if early_stop_flag>2000:
                print("Early stop: There has been no update for 2000 epochs!")
                file.write("Early stop!")
                break

        # print("---------------")
        file.write(str(config))
    print(f"Final_best_Avg_score: {current_best}")
    print(f"Final_best_highest: {current_best_highest}")


if __name__ == "__main__":

    # Create wandb session (Uncomment to enable wandb logging)
    # run = wandb.init(
    #     project="assignment_3",
    #     config=my_config,
    #     sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    #     id=my_config["run_id"]
    # )

    for lr in [2e-2]:#,1e-2,9e-3,8.5e-3,8e-3]: #7.5e-3,7e-3,6e-3 are bad, 8e-2 is badder than 6e-2 in v9, 1e-2 is bad in v10, full:[6e-2,4e-2,2e-2]

        my_config["save_path"] = f"C:/Users/harry/大型資料/RL_HW3_trained_models/3rd_try_my_CNN_diff_alg/improve_model_v1_lr{lr}"
        my_config["learning_rate"] = lr

        os.makedirs(my_config["save_path"],exist_ok=True)

        #run on GPU
        device = get_device("cuda")
        print(f"run on {device}")
        # print(torch.cuda.is_available())
        # print(torch.cuda.device_count())
        # print(torch.__version__)
        # print(torch.version.cuda)

        # Create training environment 
        num_train_envs = 12
        # train_env = DummyVecEnv([make_env for _ in range(num_train_envs)])
        train_env = SubprocVecEnv([make_env for _ in range(num_train_envs)])

        # Create evaluation environment 
        # eval_env = DummyVecEnv([make_env])
        eval_env = SubprocVecEnv([make_env])

        # Create model from loaded config and train
        # Note: Set verbose to 0 if you don't want info messages

        lr_scheduler = get_linear_fn(
            start=my_config["learning_rate"],
            end=my_config["learning_rate"]*0.05, #0.01, 0.002; 0.03 in v9,v10, all converge at 2000th epoch
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
            device="cuda",
            policy_kwargs=policy_kwargs
        )

        train(eval_env, model, my_config)