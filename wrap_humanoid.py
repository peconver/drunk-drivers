from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3 import A2C

from humanoid import HumanoidBulletEnv


if __name__ == "__main__":
    env_max_steps = 1500
    learn_epochs = 100
    learn_total_steps = learn_epochs * env_max_steps
    learn_verbose = 0

    train = False
    
    if train:
        # change the number in range if you see errors, the number depends on power of your computer
        env = SubprocVecEnv([lambda: HumanoidBulletEnv(animate=False, max_steps=env_max_steps) for i in range(17)]) 
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=8., epsilon=1e-8)
        #env = HumanoidBulletEnv(animate=False, max_steps=env_max_steps)

        model = A2C('MlpPolicy', env, tensorboard_log="./a2c_humanoid_tensorboard/", use_sde=True, gae_lambda=0.9, vf_coef=0.4, ent_coef=0.001, n_steps=8, learning_rate=8e-4, policy_kwargs=dict(net_arch=[128, 128], log_std_init=-2, ortho_init=False), verbose=learn_verbose)
        model.learn(learn_total_steps)
        model.save("first_try")

        env.save("first_try.pkl")

        env.close()
    else:
        model = A2C.load("first_try")
        env = DummyVecEnv([lambda: HumanoidBulletEnv(animate=True, max_steps=env_max_steps)])
        env = VecNormalize.load("first_try.pkl", env)
        env.training = False
        env.norm_reward = False

        #env = HumanoidBulletEnv(animate=True, max_steps=env_max_steps)
        #model = A2C.load("first_try")
        obs = env.reset()
        for i in range(env_max_steps):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            print(rewards)
            #env.render()
        env.close()
