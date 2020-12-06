from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import A2C

from humanoid import HumanoidBulletEnv


if __name__ == "__main__":
    env_max_steps = 1500
    learn_epochs = 100
    learn_total_steps = learn_epochs * env_max_steps
    learn_verbose = 0

    train = False
    
    if train:
        env = DummyVecEnv([lambda: HumanoidBulletEnv(animate=False, max_steps=env_max_steps)])
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
        #env = HumanoidBulletEnv(animate=False, max_steps=env_max_steps)

        model = A2C('MlpPolicy', env, policy_kwargs=dict(net_arch=[256, 256]), verbose=learn_verbose)
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

        obs = env.reset()
        for i in range(env_max_steps):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            #print(rewards)
            #env.render()
        env.close()
