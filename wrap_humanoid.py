from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3 import A2C

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy

from humanoid import HumanoidBulletEnv

from typing import Callable
import numpy as np

N_PROCESS = 4

class myCallback(BaseCallback):
    def __init__(self,  log_dir: str, check_freq=100):
        super(myCallback, self).__init__()
        self.check_freq = check_freq
        self.log_dir = log_dir
    
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            x, y = ts2xy(load_results(self.locals['tb_log_name']), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                self.logger.record('rollout/mean reward of 100 steps', mean_reward)
        
        return True

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func



if __name__ == "__main__":
    env_max_steps = 1000
    learn_epochs = 5000
    learn_total_steps = env_max_steps * learn_epochs
    learn_verbose = 1

    config = {"n_envs": N_PROCESS, "policy_name":'MlpPolicy',"gamma":.99, "n_steps":16,"vf_coef":.4,"ent_coef":.0,"max_grad_norm": .5, 
            "learning_rate":.001,"epsilon":1e-5, "gae_lambda":.9,"use_rms_prop": True, "normalize_advantage": False, "use_sde": True,
            "policy_hid_dim":256 ,"verbose":learn_verbose,"session_ID":'torLegsPos-FeetPelNeckNegv5'}
            

    train = False
    
    if train:
        env = SubprocVecEnv([lambda: Monitor(HumanoidBulletEnv(animate=False, max_steps=env_max_steps)) for i in range(config["n_envs"])])
        #env = DummyVecEnv([lambda: HumanoidBulletEnv(animate=False, max_steps=env_max_steps)])
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=20.,clip_reward=20.)
        lr = linear_schedule(config["learning_rate"])
        model = A2C(config["policy_name"],
            gae_lambda=config["gae_lambda"],
            env=env,
            gamma=config["gamma"],
            n_steps=config["n_steps"],
            vf_coef=config["vf_coef"],
            ent_coef = config["ent_coef"],
            max_grad_norm=config["max_grad_norm"],
            learning_rate=lr,
            rms_prop_eps=config["epsilon"],
            use_rms_prop=config["use_rms_prop"],
            use_sde=config["use_sde"],
            normalize_advantage=config["normalize_advantage"],
            verbose=config["verbose"],
            tensorboard_log="tb/{}/".format(config["session_ID"]),
            #full_tensorboard_log=config["full_tensorboard_log"],
            policy_kwargs=dict(net_arch=[int(config["policy_hid_dim"]), int(config["policy_hid_dim"])]))
        
        #callback = myCallback("tb/{}/A2C_1/".format(config["session_ID"]))
        #model.is_tb_set = False
        
        model.learn(learn_total_steps)
        model.save("learned/{}".format(config["session_ID"]))
        env.save("learned/{}.pkl".format(config["session_ID"]))

        env.close()
    else:
        model = A2C.load("learned/{}".format(config["session_ID"]))
        env = DummyVecEnv([lambda: HumanoidBulletEnv(animate=True, max_steps=env_max_steps)])
        env = VecNormalize.load(("learned/{}.pkl".format(config["session_ID"])), env)
        env.training = False
        env.norm_reward = False

        #env = HumanoidBulletEnv(animate=True, max_steps=env_max_steps)
        #model = A2C.load("first_try")
        obs = env.reset()
        for i in range(env_max_steps):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            #print(rewards)
            #env.render()
        env.close()
