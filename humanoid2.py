import os
import numpy as np
from numpy.lib.function_base import angle
import pybullet as p
import pybullet_data
from math import pi
import time
import gym
from gym import spaces


class HumanoidBulletEnv(gym.Env):
    def __init__(self, animate=False, max_steps=100):
        super(HumanoidBulletEnv, self).__init__()
        self.animate = animate
        self.max_steps = max_steps

        if (animate):
            self.client_ID = p.connect(p.GUI)
        else:
            self.client_ID = p.connect(p.DIRECT)
        assert self.client_ID != -1, "Physics client failed to connect"

        p.setGravity(0, 0, -9.8, physicsClientId=self.client_ID)
        p.setRealTimeSimulation(0, physicsClientId=self.client_ID)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_ID)

        # Load actual robot into the world
        shift_x = .8
        shift_z = .17
        self.car = p.loadURDF("polaris.urdf", [-shift_x, 0, (shift_z+0.32)], useFixedBase=True, physicsClientId=self.client_ID, globalScaling=.97)
        self.plane = p.loadURDF("plane.urdf", [-shift_x, 0, shift_z], physicsClientId=self.client_ID)  # Floor
        #time.sleep(1)
        self.robot = p.loadMJCF("humanoid_symmetric_no_ground.xml", physicsClientId=self.client_ID)  # humanoid
        self.robot = self.robot[0]
        p.resetJointState(self.robot, 9, -1.57)
        p.resetJointState(self.robot, 16, -1.57)
        p.resetJointState(self.robot, 7, -.78)
        p.resetJointState(self.robot, 14, -.78)
        p.resetJointState(self.robot, 1, -.5)



        #for i in range (p.getNumJoints(self.robot)):
        #    print(p.getJointInfo(self.robot,i))

        for j in range(p.getNumJoints(self.robot)):
            info = p.getJointInfo(self.robot, j)
            link_name = info[12].decode("ascii")
            if link_name == "left_foot":
                self.left_foot = j
            if link_name == "right_foot":
                self.right_foot = j
            if link_name == "pelvis":
                self.pelvis = j
            if link_name == "link0_21":
                self.right_shoulder = j
            if link_name == "link0_26":
                self.left_shoulder = j


        # Input and output dimensions defined in the environment
        self.obs_dim = 23  # joints + torques + if standing on floor
        self.act_dim = 17

        # Limits of our joints. When using the * (multiply) operation on a list, it repeats the list that many times
        self.joints_deg_low = np.array([-45, -75, -35, -25, -60, -120, -160, -25, -60, -120, -160, -85, -85, -90, -60, -60, -90])
        self.joints_deg_high = np.array([45,  30,  35,   5,  35,   20,   -2,   5,  35,   20,   -2,  60,  60,  50,  85,  85,  50])
        self.joints_deg_diff = self.joints_deg_high - self.joints_deg_low

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.act_dim, ))
        #same as action but + min and max distance of feet
        self.observation_space = spaces.Box(low=-10, high=10, shape=(self.obs_dim, ))

        self.max_joint_force = 1.6
        self.target_vel = 0.4
        self.sim_steps_per_iter = 24  # The amount of simulation steps done every iteration.
        self.lateral_friction = 1.0
        self.torso_target = np.array([0, (1.22+0.15)/2.0 + 0.6, 0])
        self.l_foot_target = np.array([0, (1.22+0.15)/2.0 + 0.6, 0])
        self.r_foot_target = np.array([0, (1.22+0.15)/2.0 + 0.4, 0])

        self.joints_index = np.array([0, 1, 3, 5, 6, 7, 9, 12, 13, 14, 16, 19, 20, 22, 24, 25, 27])

        for i in range(4):
            type(self.lateral_friction), type(3 * i + 2), type(self.robot)
            p.changeDynamics(self.robot, 3 * i + 2, lateralFriction=self.lateral_friction)
        p.changeDynamics(self.robot, -1, lateralFriction=self.lateral_friction)


        """ for i in range(1000):
            time.sleep(0.001)
            p.stepSimulation(physicsClientId=self.client_ID) """

        p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=0, cameraPitch=0, cameraTargetPosition=[0, 0, 2])

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        #p.getCameraImage(320, 200)  # , renderer=p.ER_BULLET_HARDWARE_OPENGL )

        self.step_ctr = 0

    def get_obs(self):
        '''
        Returns a suite of observations of the given state of the robot. Not all of these are required.
        :return: list of lists
        '''
        # Torso
        torso_pos, torso_quat = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.client_ID)  # xyz and quat: x,y,z,w
        torso_vel, torso_angular_vel = p.getBaseVelocity(self.robot, physicsClientId=self.client_ID)

        #distance of each foot to target position
        l_vec_target = p.getLinkState(self.robot, self.left_foot)[0] - self.l_foot_target
        ctct_l = np.linalg.norm(l_vec_target) 
        r_vec_target = p.getLinkState(self.robot, self.right_foot)[0] - self.r_foot_target
        ctct_r = np.linalg.norm(r_vec_target) 

        contacts = [ctct_l, ctct_r]

        # Joints
        obs = p.getJointStates(self.robot, range(self.act_dim), physicsClientId=self.client_ID)  # pos, vel, reaction(6), prev_torque
        joint_angles = []
        joint_velocities = []
        joint_torques = []
        for o in obs:
            joint_angles.append(o[0])
            joint_velocities.append(o[1])
            joint_torques.append(o[3])
        # add positions of links to observation instead of contacts>
        # [l[4] for l in p.getLinkStates(self.robot, range(0, 13+1), physicsClientId=self.client_ID)]
        return torso_pos, torso_quat, torso_vel, torso_angular_vel, joint_angles, joint_velocities, joint_torques, contacts

    def step(self, ctrl):
        '''
        Step function.
        :param ctrl: list or array of target joint angles normalized to [-1,1]
        :return: next_obs, r, done, _  . The last '_' value is left as compatability with gym envs.
        '''

        ctrl_noisy = ctrl + np.random.rand(self.act_dim).astype(np.float32) * 0.1 - 0.05

        # YOU CAN REMOVE THIS CLIP IF YOU WANT
        ctrl_clipped = np.clip(ctrl_noisy, -1, 1)

        # Scale the action to correct range and publish to simulator (this just publishes the action, doesn't step the simulation yet)
        scaled_action = self.norm_to_rads(ctrl_clipped)
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=range(self.act_dim),
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=scaled_action,
                                    forces=[self.max_joint_force] * self.act_dim ,
                                    positionGains=[0.02] * self.act_dim ,
                                    velocityGains=[0.1] * self.act_dim ,
                                    physicsClientId=self.client_ID)

        # Step the simulation.
        for i in range(self.sim_steps_per_iter):
            p.stepSimulation(physicsClientId=self.client_ID)
            if self.animate:
                time.sleep(0.004)

        # Get new observations (Note, not all of these are required and used)
        torso_pos, torso_quat, torso_vel, torso_angular_vel, joint_angles, joint_velocities, joint_torques, contacts = self.get_obs()
        xd, yd, zd = torso_vel  # Unpack for clarity
        roll, pitch, yaw = p.getEulerFromQuaternion(torso_quat)

        self.step_ctr += 1

        """reward
        """
        # Add various rewards here to your liking. For example, np.square(yaw) * 0.5 will penalise the robot facing directions other than the x direction
        # You can use other similar heuristics to 'guide' the agent into doing what you actually want it to do.

        # reward increasing when feet get closer to plane
        r_contact = -(contacts[0] + contacts[1])# / self.max_steps * 100
        # negative reward for wrong orientation, small at beggining and larger at end
        r_angle = (np.square(yaw) * 0.1 + np.square(pitch) * 0.5 + np.square(roll) * 0.1) #/ (self.max_steps - self.step_ctr)

        r = np.clip(r_contact, -3, 3)
        r = r_contact * 100
        """reward
        """

        # Scale joint angles and make the policy observation
        scaled_joint_angles = self.rads_to_norm(joint_angles)
        env_obs = np.concatenate((scaled_joint_angles, torso_quat, contacts)).astype(np.float32)

        # This condition terminates the episode
        done = self.step_ctr > self.max_steps or np.abs(roll) > 1.57 or np.abs(pitch) > 1.57

        env_obs_noisy = env_obs + np.random.rand(self.obs_dim).astype(np.float32) * 0.1 - 0.05

        return env_obs_noisy, r, done, {}

    def reset(self):
        # Reset the robot to initial position and orientation and null the motors
        joint_init_pos_list = self.norm_to_rads([0] * self.act_dim)
        [p.resetJointState(self.robot, i, joint_init_pos_list[i], 0, physicsClientId=self.client_ID) for i in range(self.act_dim)]
        p.resetBasePositionAndOrientation(self.robot, [0, 0, 1.5], [0, 0, 0, 1], physicsClientId=self.client_ID)
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=range(self.act_dim),
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[0] * self.act_dim,
                                    forces=[self.max_joint_force] * self.act_dim,
                                    physicsClientId=self.client_ID)

        p.resetJointState(self.robot, 9, -1.57)
        p.resetJointState(self.robot, 16, -1.57)
        p.resetJointState(self.robot, 7, -.78)
        p.resetJointState(self.robot, 14, -.78)
        p.resetJointState(self.robot, 1, -.5)

        # Step a few times so stuff settles down
        for i in range(20):
            p.stepSimulation(physicsClientId=self.client_ID)

        # Return initial obs
        obs, _, _, _ = self.step(np.zeros(self.act_dim))
        return obs

    def rads_to_norm(self, joints):
        '''
        :param joints: list or array of joint angles in radians
        :return: array of joint angles normalized to [-1,1]
        '''
        sjoints = np.array(joints)
        #sjoints = ((sjoints - self.joints_rads_low) / self.joints_rads_diff) * 2 - 1
        return sjoints

    def norm_to_rads(self, action):
        '''
        :param action: list or array of normalized joint target angles (from your control policy)
        :return: array of target joint angles in radians (to be published to simulator)
        '''
        return np.array(action) #(np.array(action) * 0.5 + 0.5) * self.joints_rads_diff + self.joints_rads_low

    def r_joints_outside(self):
        car_half_wide = (1.22+0.15)/2.0
        reward = 0
        s = p.getLinkStates(self.robot, range(0, 13+1), physicsClientId=self.client_ID)
        for link in s:
            dist = abs(link[4][1]-car_half_wide)
            reward -= dist
        return reward
    
    def r_close_to_target(self):
        l_vec_target = p.getLinkState(self.robot, self.left_foot)[0] - self.l_foot_target
        dist_l = np.linalg.norm(l_vec_target) 
        r_vec_target = p.getLinkState(self.robot, self.right_foot)[0] - self.r_foot_target
        dist_r = np.linalg.norm(r_vec_target)
        t_vec_target = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.client_ID)[0] - self.torso_target
        # ignore z coordinate, should be good when xy are correct and joints are set
        dist_t = np.linalg.norm(t_vec_target[:2])

        return dist_t + (dist_l + dist_r)*0.5

    def r_tumble(self):

        punishment = -100
        
        left_foot = p.getLinkState(self.robot,self.left_foot)[0][2]
        right_foot = p.getLinkState(self.robot,self.right_foot)[0][2]
        stomach = p.getLinkState(self.robot,self.pelvis)[0][2]
        left_shoulder = p.getLinkState(self.robot,self.left_shoulder)[0][2]
        right_shoulder = p.getLinkState(self.robot,self.right_shoulder)[0][2]
        foot = max(left_foot, right_foot)
        shoulder = min(left_shoulder, right_shoulder)

        if foot >= stomach or foot >= shoulder or stomach >= shoulder:
            return punishment
        return 0
        

model = HumanoidBulletEnv(True)
p.setRealTimeSimulation(1)

while(1):
    a = 1
    keys = p.getKeyboardEvents()
    for k in keys:
        if (keys[k] & p.KEY_WAS_TRIGGERED):
            if (k == ord('i')):
                model.reset()
