import numpy as np
import os
from humanoid import HumanoidBulletEnv

class BlackBoxOpt():
    def __init__(self, obs_dim, act_dim):
        # Initialize your weight parameters of your policy.
        # The simplest policy to implement would be a simple affine mapping y = xw+b, where x is the input,
        # w is the weight matrix and b is the biasget_params
        rng = np.random.default_rng()
        self.policy = rng.uniform(-0.1, 0.1, [98,20])
        self.b = rng.uniform(-0.1,0.1, [98,1])
        # TODO change initialization of weights in main if train

    def set_params(self, w):
       # print('Shape!!!!', len(w))
        self.b = w[1666:1683]
       # print('b!!!', self.b)
        self.policy = np.reshape(w[0:1666], (17,98))
        # This function takes in a list w and maps it to your policy parameters.
        # The simplest way is to probably make an array out of the w vector and reshape it appropriately TODO: policy = res haped(w)
    def get_params(self):
        # This function returns a list w from your policy parameters. You can use numpy's flatten() function TODO: w = flatten(policy)
        w = np.append(self.policy.flatten(), self.b)
        return w
    def forward(self, x):
        return self.policy.dot(x) + self.b
        # Performs the forward pass on your policy. Maps observation input x to action output a TODO: a = policy*x

def f_wrapper(env, policy):
    def f(w):
        reward = 0
        done = False
        obs = env.reset()

        # Map the weight vector to your policy
        policy.set_params(w)

        while not done:

            # Get action from policy
            act = policy.forward(obs)

            # Step environment
            obs, rew, done, _ = env.step(act)

            reward += rew
        return reward

    return f

def my_opt(f, w_init, iters):
    # Your optimization algorithm. Takes in an evaluation function f, and initial solution guess w_init and returns
    # parameters w_best which 'solve' the problem to some degree.
    # vyzkouset f() na ruzna w a vratit tu nejlepsi
    w_best = w_init
    w = w_init.copy()
    r_best = f(w_best)
    policy_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'learned/humanoid_bbx_')
    
    # model will train forever and trained weights are stored every 252 iterations
    print()
    while(1):
        for i in range(252):
            w[i] = w[i] + 0.05
            r = f(w)
            if(r and r > r_best):
                r_best = r
                w_best = w.copy()
            else:
                w = w_best.copy()
                w[i] = w[i] - 0.05
                r = f(w)
                if (r and r > r_best):
                    r_best = r
                    w_best = w.copy()
                else:
                    w = w_best.copy()
            # print('Progress: ', round(i/252*100), '%', end='\r') 
        # print("r_best: {r_best, w_best}")
        print("r_best:", r_best)
        np.save(policy_path + str(round(r_best)) + '.npy', w_best)
        
        
    return w_best, r_best

def test(w_best, max_steps=70, animate=False):
    # Make the environment and your policy
    env = HumanoidBulletEnv(animate=animate, max_steps=max_steps)
    policy = BlackBoxOpt(env.obs_dim, env.act_dim)

    # Make evaluation function
    f = f_wrapper(env, policy)

    # Evaluate
    r_avg = 0
    eval_iters = 10
    for i in range(eval_iters):
        r = f(w_best)
        r_avg += r
        print(r)
    return r_avg / eval_iters

if __name__ == "__main__":
    train = True
    # policy_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'learned/humanoid_bbx.npy')
    max_steps = 1000
    N_training_iters = ...
    w_best = None
    
    if train:
        # Make the environment and your policy
        env = HumanoidBulletEnv(animate=False, max_steps=max_steps)
        policy = BlackBoxOpt(env.obs_dim, env.act_dim)

        # print('obs_dim!!',env.obs_dim)
        # Make evaluation function
        f = f_wrapper(env, policy)

        # Initial guess of solution
        # w_init = np.random.rand(1, env.act_dim*21)e
        rng = np.random.default_rng()
        w_init = rng.uniform(-0.1, 0.1, [98,21])
        w_init = w_init.flatten()
        # Perform optimization
        w_best, r_best = my_opt(f, w_init, N_training_iters)

        print(f"r_best: {r_best}")

        # Save policy
        np.save(policy_path, w_best)
        env.close()

    if not train:
        # WARN: Change policy path, look into dir 'learned' and choose desired model
        policy_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'learned/humanoid_bbx_137.npy')
        w_best = np.load(policy_path)
    print(f"Avg test rew: {test(w_best, max_steps=max_steps, animate=not train)}")




