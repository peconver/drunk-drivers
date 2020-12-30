import numpy as np
import os
from humanoid import HumanoidBulletEnv
total_steps = 0

class BlackBoxOpt():
    def __init__(self, obs_dim, act_dim):
        # Initialize your weight parameters of your policy.
        # The simplest policy to implement would be a simple affine mapping y = xw+b, where x is the input,
        # w is the weight matrix and b is the biasget_params
        rng = np.random.default_rng()
        self.policy = rng.uniform(-0.1, 0.1, [obs_dim,act_dim])
        self.b = rng.uniform(-0.1,0.1, [act_dim,1])
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.total_steps = 0
        # print(act_dim, obs_dim)
        # TODO change initialization of weights in main if train

    def set_params(self, w):
        # print('Shape!!!!', len(w))
        self.b = w[self.act_dim*self.obs_dim:(self.act_dim*self.obs_dim+self.act_dim)]
       # print('b!!!', self.b)
        self.policy = np.reshape(w[0:self.act_dim*self.obs_dim], (self.act_dim,self.obs_dim))
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
            # log total steps count
            policy.total_steps = policy.total_steps+1

            # Get action from policy
            act = policy.forward(obs)

            # Step environment
            obs, rew, done, _ = env.step(act)

            reward += rew
        return reward

    return f

def my_opt(policy, f, w_init, iters):
    # Your optimization algorithm. Takes in an evaluation function f, and initial solution guess w_init and returns
    # parameters w_best which 'solve' the problem to some degree.
    # vyzkouset f() na ruzna w a vratit tu nejlepsi
    w_best = w_init
    w = w_init.copy()
    r_best = f(w_best)
    policy_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'learned/humanoid_bbx_')
    
    # model will train 50 000 000 steps and trained weights are stored every 5000 epoch
    print()
    j = 0
    epoch  = 0
    store = 0
    len_w = len(w)
    while(policy.total_steps<50000000):
        for i in range(len_w):
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
            # every 5000 epoch store best weights
            if(store==5000):
                store = 0
                np.save(policy_path + str(round(r_best)) + '.npy', w_best)
            # after every epoch store reward and number of steps so far
            print(policy.total_steps, r)
            # every 50 epoch store best reward
            # if (i+1)%50 is 0:
            #        print(epoch, w) 
            epoch = epoch + 1
            store = store + 1
        # print('Progress: ', round(i/252*100), '%', end='\r') 
        # print("r_best: {r_best, w_best}")
        # print("r_best:", r_best)
        # np.save(policy_path + str(round(r_best)) + '.npy', w_best)
    print('END')
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
    train = False
    policy_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'learned/humanoid_bbx.npy')
    max_steps = 1000
    N_training_iters =50000
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
        w_init = rng.uniform(-0.1, 0.1, [env.obs_dim+1,env.act_dim])
        w_init = w_init.flatten()
        # Perform optimization
        w_best, r_best = my_opt(policy,f, w_init, N_training_iters)

        print("r_best: {r_best}")

        # Save policy
        np.save(policy_path, w_best)
        env.close()

    if not train:
        # WARN: Change policy path, look into dir 'learned' and choose desired model
        policy_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'learned/humanoid_bbx_174.npy')
        w_best = np.load(policy_path)
    print(f"Avg test rew: {test(w_best, max_steps=max_steps, animate=not train)}")




