from operator import neg
import os
import numpy as np
from numpy.lib.function_base import angle
import pybullet as p
import pybullet_data
from math import pi
import time
import gym
from gym import spaces
import math


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
        p.setTimeStep(1/190, self.client_ID)

        # Load actual robot and car into the world
        shift_x = .8
        shift_z = .17
        self.shift_y = -.35
        self.car = p.loadURDF("models/polaris.urdf", [-shift_x, self.shift_y, (shift_z + 0.32)], useFixedBase=True, physicsClientId=self.client_ID, globalScaling=.97)
        self.plane = p.loadURDF("plane.urdf", [-shift_x, 0, shift_z], physicsClientId=self.client_ID)  # Floor
        self.robot = p.loadMJCF("models/humanoid_symmetric_no_ground.xml", physicsClientId=self.client_ID)  # humanoid
        self.robot = self.robot[0]

        #p.changeDynamics(self.robot, 9, jointLowerLimit=-2.792526960372925, jointUpperLimit=-1)
        #p.changeDynamics(self.robot, 16, jointLowerLimit=-2.792526960372925, jointUpperLimit=-1)
        
        
        # set joints, it is not neccessary because it is handled in reset()
        p.resetJointState(self.robot, 9, -1.57)
        p.resetJointState(self.robot, 16, -1.57)
        p.resetJointState(self.robot, 7, -.78)
        p.resetJointState(self.robot, 14, -.78)
        p.resetJointState(self.robot, 1, -.5)
        #print(p.getJointInfo(self.robot, 9)[8:10])

        # get id to often used links
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
        self.obs_dim = 98  # joints + torques + contacts
        self.act_dim = 17

        # Limits of our joints. When using the * (multiply) operation on a list, it repeats the list that many times
        self.joints_deg_low = np.array([-45, -75, -35, -25, -60, -120, -160, -25, -60, -120, -160, -85, -85, -90, -60, -60, -90])
        self.joints_deg_high = np.array([45,  30,  35,   5,  35,   20,  -57,   5,  35,   20,  -57,  60,  60,  50,  85,  85,  50])
        self.joints_deg_diff = self.joints_deg_high - self.joints_deg_low

        self.joints_rad_low = np.deg2rad(self.joints_deg_low)
        self.joints_rad_high = np.deg2rad(self.joints_deg_high)
        self.joints_rad_diff = np.deg2rad(self.joints_deg_diff)

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.act_dim, ))
        #same as action but + min and max distance of feet
        self.observation_space = spaces.Box(low=-10, high=10, shape=(self.obs_dim, ))

        self.lateral_friction = 2.0
        self.torso_target = np.array([-0.001, 0.73, 1.38])
        self.l_foot_target = np.array([-0.05, 0.73, shift_z+0.02])
        self.r_foot_target = np.array([0.049, 0.73, shift_z+0.02])

        # indexes for joints, that aren't fixed
        self.joints_index = np.array([0, 1, 3, 5, 6, 7, 9, 12, 13, 14, 16, 19, 20, 22, 24, 25, 27])
        self.max_forces = np.array([100, 100, 100,100, 100, 300, 300,100, 100, 300, 200, 25, 25, 25,25, 25, 25])

        for i in range(-1,p.getNumJoints(self.robot)):
            type(self.lateral_friction), type(i), type(self.robot)
            p.changeDynamics(self.robot, i, lateralFriction=self.lateral_friction)
        """ for i in range(-1,p.getNumJoints(self.car)):
            type(self.lateral_friction), type(i), type(self.car)
            p.changeDynamics(self.car, i, lateralFriction=1) """
        
        # set camera
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
        """ l_vec_target = p.getLinkState(self.robot, self.left_foot)[0] - self.l_foot_target
        ctct_l = np.linalg.norm(l_vec_target) 
        r_vec_target = p.getLinkState(self.robot, self.right_foot)[0] - self.r_foot_target
        ctct_r = np.linalg.norm(r_vec_target) 

        contacts = [ctct_l, ctct_r] """

        # Joints
        obs = p.getJointStates(self.robot, self.joints_index, physicsClientId=self.client_ID)  # pos, vel, reaction(6), prev_torque
        joint_angles = []
        joint_velocities = []
        joint_torques = []
        for o in obs:
            joint_angles.append(o[0])
            joint_velocities.append(o[1])
            joint_torques.append(o[3])
        # add positions of links to observation instead of contacts>
        # [l[4] for l in p.getLinkStates(self.robot, range(0, 13+1), physicsClientId=self.client_ID)]
        
        # link positions and contacts
        """ obs = p.getLinkStates(self.robot, range(p.getNumJoints(self.robot)), physicsClientId=self.client_ID)
        link_pos_orient = []
        for o in obs:
            link_pos_orient.append(o[1]) """

        a = p.getNumJoints(self.robot)
        # 0/1 contact of each link to car and floor
        contacts = [0] * (a * 2)
        obs = p.getContactPoints(self.robot, self.car, physicsClientId=self.client_ID)
        for o in obs:
            contacts[o[3]] = 1  # for each contact point, store 1 at index of participating robot's link
        obs = p.getContactPoints(self.robot, self.plane, physicsClientId=self.client_ID)
        for o in obs:
            contacts[a + o[3]] = 1  # for each contact point, store 1 at index of participating robot's link
        
        """ l_f = p.getLinkState(self.robot, self.left_foot)[0]
        r_f = p.getLinkState(self.robot, self.right_foot)[0]
        pelvis = p.getLinkState(self.robot, self.pelvis)[0]
        l_r = np.linalg.norm(np.array((l_f[0] - pelvis[0], l_f[1] - pelvis[1])))
        r_r = np.linalg.norm(np.array((r_f[0] - pelvis[0], r_f[1] - pelvis[1])))
        h = (l_f[2]+r_f[2])/2 - pelvis[2]
        print("l_r {:.4f}, r_r {:.4f}, h {:.4f}".format(l_r, r_r, h))
        print("l_f {:.4f}, r_f {:.4f}".format(l_f[1], r_f[1])) """
        #pelvis = p.getLinkState(self.robot, self.pelvis)[0]
        #print("({:.4f}, {:.4f}, {:.4f})".format(pelvis[0],pelvis[1],pelvis[2]))

        """ l = p.getLinkStates(self.robot, [5,9])
        l2 = p.getLinkStates(self.robot, [12, 16])
        torso_z = (p.getLinkStates(self.robot, [0]))[0][4][2]
        h = (l[0][0][2]+l2[0][0][2])/2

        vec1 = (l[1][0][0] - l[0][0][0], l[1][0][1] - l[0][0][1], l[1][0][2] - l[0][0][2])
        vec2 = (l2[1][0][0] - l2[0][0][0], l2[1][0][1] - l2[0][0][1], l2[1][0][2] - l2[0][0][2])

        vec_vychozi = [0,1,h]

        skalarni_soucin1 = vec1[0]*vec_vychozi[0] + vec1[1]*vec_vychozi[1] + vec1[2]*vec_vychozi[2]
        angle1 = math.acos(skalarni_soucin1/math.sqrt(vec1[0]*vec1[0]+vec1[1]*vec1[1]+vec1[2]*vec1[2]))
        skalarni_soucin2 = vec2[0] * vec_vychozi[0] + vec2[1] * vec_vychozi[1] + vec2[2] * vec_vychozi[2]
        print("power {:.4f}".format(vec2[0] * vec2[0] + vec2[1] * vec2[1] + vec2[2] * vec2[2]))
        print("sqrt {:.4f}".format(math.sqrt(vec2[0] * vec2[0] + vec2[1] * vec2[1] + vec2[2] * vec2[2])))
        print("skalar soc {:.4f}".format(skalarni_soucin2))
        angle2 = math.acos(skalarni_soucin2 / math.sqrt(vec2[0] * vec2[0] + vec2[1] * vec2[1] + vec2[2] * vec2[2]))
        print(angle1,angle2, h) """
        
        return torso_pos, torso_quat, torso_vel, torso_angular_vel, joint_angles, joint_velocities, joint_torques, contacts

    def step(self, ctrl):
        '''
        Step function.
        :param ctrl: list or array of target joint angles normalized to [-1,1]
        :return: next_obs, r, done, _  . The last '_' value is left as compatability with gym envs.
        '''

        # YOU CAN REMOVE THIS CLIP IF YOU WANT
        ctrl_clipped = np.clip(ctrl,-1, 1)

        # Scale the action to correct range and publish to simulator (this just publishes the action, doesn't step the simulation yet)
        scaled_action = self.norm_to_rads(ctrl_clipped)
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=self.joints_index,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=scaled_action,
                                    forces=self.max_forces,
                                    positionGains=[0.02] * self.act_dim ,
                                    velocityGains=[0.1] * self.act_dim ,
                                    physicsClientId=self.client_ID)

        # Step the simulation.
        p.stepSimulation(physicsClientId=self.client_ID)
        if self.animate:
            #time.sleep(0.004)
            time.sleep(.005)

        # Get new observations (Note, not all of these are required and used)
        torso_pos, torso_quat, torso_vel, torso_angular_vel, joint_angles, joint_velocities, joint_torques, contacts = self.get_obs()
        xd, yd, zd = torso_vel  # Unpack for clarity
        roll, pitch, yaw = p.getEulerFromQuaternion(torso_quat)

        # get positions of all links
        tmp = p.getLinkStates(self.robot, range(0, 29), physicsClientId=self.client_ID)
        pos = []
        for link in tmp:
            pos += [link[0]]

        joint_angles_n = ((joint_angles - self.joints_rad_low) / self.joints_rad_diff) * 2 - 1

        self.step_ctr += 1

        """reward
        """
        r_foot = np.array(p.getLinkState(self.robot, self.right_foot, physicsClientId=self.client_ID)[0])
        l_foot = np.array(p.getLinkState(self.robot, self.left_foot, physicsClientId=self.client_ID)[0])
        dist_r_foot = np.linalg.norm(r_foot-self.r_foot_target)
        dist_l_foot = np.linalg.norm(l_foot-self.l_foot_target)

        reward_r_foot = 1/(dist_r_foot+0.2) # it gives form 0 to 5
        reward_l_foot = 1/(dist_l_foot+0.2) # it gives from 0 to 5
        reward_r_arm_down = 1/(abs(p.getJointState(self.robot, 20, physicsClientId=self.client_ID)[0])**2+0.2) # it gives from 0 to 5
        reward_l_arm_down = 1 / (abs(p.getJointState(self.robot, 25, physicsClientId=self.client_ID)[0]) ** 2 + 0.2) # it gives from 0 to 5
        reward_r_knee = 1/(abs(p.getJointState(self.robot, 9, physicsClientId=self.client_ID)[0])+0.2) # it gives from 0 to 5
        reward_l_knee = 1 / (abs(p.getJointState(self.robot, 16, physicsClientId=self.client_ID)[0]) + 0.2)  # it gives from 0 to 5

        punish_vel_torso = 1/(np.linalg.norm(torso_vel)**4 + 0.2)-5 # it gives from -5 to 0
        punish_bad_angle_torso = (abs(roll)/pi)*(-2.5) + (abs(pitch)/pi)*(-2.5) # it gives from -5 to 0

        r = (reward_l_foot+reward_r_foot+reward_l_arm_down+reward_r_arm_down+reward_l_knee+reward_r_knee)/6.0 + (punish_vel_torso+punish_bad_angle_torso)/2.0
        """reward
        """

        # get joint_angles into np array and make observation
        env_obs = np.concatenate((torso_pos, [roll, pitch, yaw], joint_angles_n, joint_velocities, contacts)).astype(np.float32)

        # This condition terminates the episode - WARNING - it can cause that the robot will try 
        # terminate the episode as quickly as possible
        #done = self.step_ctr > self.max_steps  or (legs_rot <.2 and abs(yaw-pi/2) < .2)  or (np.abs(roll) > 1.5 or np.abs(pitch) > 1.5) or torso_z < 1
        done = self.step_ctr > self.max_steps or torso_pos[2] < 0.4 or r > 4.2

        # TODO: why random element?
        env_obs_noisy = env_obs# + np.random.rand(self.obs_dim).astype(np.float32) * 0.1 - 0.05

        return env_obs_noisy, r, done, {}

    def r_legs_down(self):
        left_knee = np.array(p.getLinkState(self.robot, 16, physicsClientId=self.client_ID)[0])
        right_knee = np.array(p.getLinkState(self.robot, 9, physicsClientId=self.client_ID)[0])
        left_foot = np.array(p.getLinkState(self.robot, 18, physicsClientId=self.client_ID)[0])
        right_foot = np.array(p.getLinkState(self.robot, 11, physicsClientId=self.client_ID)[0])

        left_leg = left_foot - left_knee
        right_leg = right_foot - right_knee

        size_l = np.linalg.norm(left_leg)
        size_r = np.linalg.norm(right_leg)

        phi_l = math.acos(np.dot(left_leg, [0, 0, -1]) / size_l)
        phi_r = math.acos(np.dot(right_leg, [0, 0, -1]) / size_r)

        r_left = (-2 / pi) * phi_l + 1
        r_right = (-2 / pi) * phi_r + 1

        return 5 * (r_left + r_right) / 2.0

    def r_links_outside(self):
        car_width_half = 1.22/2.0
        right_pos = self.shift_y + car_width_half
        wrong_pos = self.shift_y - car_width_half

        links = p.getLinkStates(self.robot, range(0,29))

        r = 0
        for link in links:
            if link[0][1] >= right_pos:
                r += 1
            elif link[0][1] <= wrong_pos:
                r -= 1
            else:
                r += (2/(right_pos-wrong_pos))*link[0][1] + (right_pos+wrong_pos)/(wrong_pos-right_pos)

        return 5*r/29.0

    def r_arms_down(self):
        # can't get the vector of lower arms, so don't use yet
        left_lower_arm = np.array(p.getLinkState(self.robot, 28, physicsClientId=self.client_ID)[0])
        left_elbow = np.array(p.getLinkState(self.robot, 26, physicsClientId=self.client_ID)[0])
        right_lower_arm = np.array(p.getLinkState(self.robot, 22, physicsClientId=self.client_ID)[0])
        right_elbow = np.array(p.getLinkState(self.robot, 21, physicsClientId=self.client_ID)[0])

        left = np.array(left_lower_arm) - np.array(left_elbow)
        right = np.array(right_lower_arm) - np.array(right_elbow)

        size_l = np.linalg.norm(left)
        size_r = np.linalg.norm(right)

        phi_l = math.acos(np.dot(left, [0, 0, -1])/size_l)
        phi_r = math.acos(np.dot(right, [0, 0, -1])/size_r)

        r_left = (-2/pi)*phi_l + 1
        r_right = (-2/pi)*phi_r + 1

        return 5*(r_left + r_right)/2.0


    def r_knee(self):
        left_knee = p.getJointState(self.robot, 16, self.client_ID)[0]
        right_knee = p.getJointState(self.robot, 9, self.client_ID)[0]

        r_left = (-2/self.joints_rad_low[6])*left_knee + 1
        r_right = (-2/self.joints_rad_low[6])*right_knee + 1

        if r_left > 1:
            r_left = 2 - r_left
        if r_right > 1:
            r_right = 2 - r_right

        return 5*(r_left + r_right)/2.0


    def joints_outside(self):
        states = p.getLinkStates(self.robot, range(0,29))
        r = 0
        for s in states:
            if s[0][1] < 0.3:
                r += max(-1,3.3*s[0][1])
            else:
                r += 1
        r = r/29
        return r
    
    def feet_pelvis_neck_movement(self):
        left_foot = p.getLinkState(self.robot, self.left_foot)[0]
        right_foot = p.getLinkState(self.robot, self.right_foot)[0]
        pelvis = p.getLinkState(self.robot, self.pelvis)[0]

        left_radius = np.linalg.norm(np.array((left_foot[0] - pelvis[0], left_foot[1] - pelvis[1])))
        right_radius = np.linalg.norm(np.array((right_foot[0] - pelvis[0], right_foot[1] - pelvis[1])))
        height = (left_foot[2],right_foot[2])

        l_r = np.clip(abs(left_radius - self.target_radius)*3, 0, 1)
        r_r = np.clip(abs(right_radius - self.target_radius)*3, 0, 1)
        #h = np.clip(.4 - height, 0, .33) *3
        h = np.clip((abs(height[0] - self.target_h) + abs(height[1] - self.target_h))*2,0 , 2)
        #print("h {:.4f} {:.4f} th {:.4f} - {:.4f}".format(height[0],height[1],self.target_h,h))
        #print("lrad {:.4f} rrad {:.4f} height {:.4f}".format(left_radius,right_radius, height))
        #print("lr {:.4f} rr {:.4f} h {:.4f}".format(l_r,r_r, h))

        p_or = self.pelvis_orig
        pelvis_shift = np.linalg.norm(np.array((pelvis[0] - p_or[0], (pelvis[1] - p_or[1])*.5, pelvis[2] - p_or[2])))
        p_r = np.clip(pelvis_shift*5, 0, 1)

        left_shoulder = p.getLinkState(self.robot, self.left_shoulder)[0]
        right_shoulder = p.getLinkState(self.robot, self.right_shoulder)[0]
        neck = ((left_shoulder[0]+right_shoulder[0])/2, (left_shoulder[1]+right_shoulder[1])/2)
        n_r = np.linalg.norm(np.array((pelvis[0] - neck[0], pelvis[1] - neck[1])))
        n_r = np.clip(n_r*5, 0, 1)

        return h, p_r, n_r
        #return (l_r+r_r+h)/3, p_r, n_r

    # def reset(self):
    #     # Reset the robot to initial position and orientation and null the motors
    #     joint_init_pos_list = [0] * 29
    #     joint_init_pos_list[9] = -1.4
    #     joint_init_pos_list[16] = -1.4
    #     joint_init_pos_list[7] = -1
    #     joint_init_pos_list[14] = -1
    #     joint_init_pos_list[1] = -.5
    #     [p.resetJointState(self.robot, i, joint_init_pos_list[i], 0, physicsClientId=self.client_ID) for i in range(29)]
    #     p.resetBasePositionAndOrientation(self.robot, [-0.1, 0, 1.48], [0, 0, 0, 1], physicsClientId=self.client_ID)
    #     p.resetBaseVelocity(self.robot, [0, 0, 0], [0, 0, 0])
    #
    #     p.setJointMotorControlArray(bodyUniqueId=self.robot,
    #                                 jointIndices=range(29),
    #                                 controlMode=p.POSITION_CONTROL,
    #                                 targetPositions=joint_init_pos_list,
    #                                 forces=[100]*29,
    #                                 positionGains=[0] * 29 ,
    #                                 velocityGains=[0] * 29 ,
    #                                 physicsClientId=self.client_ID)
    #
    #     # Step a few times so stuff settles down
    #     for i in range(50):
    #         p.stepSimulation(physicsClientId=self.client_ID)
    #
    #     self.step_ctr = 0
    #
    #     left_foot = p.getLinkState(self.robot, self.left_foot)[0]
    #     right_foot = p.getLinkState(self.robot, self.right_foot)[0]
    #     pelvis = p.getLinkState(self.robot, self.pelvis)[0]
    #     left_radius = np.linalg.norm(np.array((left_foot[0] - pelvis[0], left_foot[1] - pelvis[1])))
    #     right_radius = np.linalg.norm(np.array((right_foot[0] - pelvis[0], right_foot[1] - pelvis[1])))
    #
    #     self.target_radius = (right_radius + left_radius)/2 *.92
    #     self.pelvis_orig = pelvis
    #     self.target_h = (left_foot[2]+right_foot[2])/2
    #     #print(left_foot, right_foot, pelvis)
    #     #print(self.target_radius, self.target_h)
    #     #self.limit_height = (pelvis[2] - (left_foot[2]+right_foot[2])/2)*.5
    #     # Return initial obs
    #     obs, _, _, _ = self.step(np.zeros(self.act_dim))
    #     return obs

    def reset(self):
        # Reset the robot to initial position and orientation and null the motors
        joint_init_pos_list = [0] * 29
        joint_init_pos_list[self.joints_index[0]] = -0.7777035947041838 + np.random.random_sample((1,))*(0.7777035947041838-0.6957214220333279)
        joint_init_pos_list[self.joints_index[1]] = -1.2750854052783716 + np.random.random_sample((1,)) * (1.2750854052783716 - 1.2150521235068843)
        joint_init_pos_list[self.joints_index[2]] = -0.3103475206975532 + np.random.random_sample((1,)) * (0.3103475206975532 + 0.10502250234321431)
        joint_init_pos_list[self.joints_index[3]] = -0.1325084831262215 + np.random.random_sample((1,)) * (0.1325084831262215 + 0.06718491169238455)
        joint_init_pos_list[self.joints_index[4]] = 0.5341795753001188 + np.random.random_sample((1,)) * (0.5785672177544219 - 0.5341795753001188)
        joint_init_pos_list[self.joints_index[5]] = -0.761394347824112 + np.random.random_sample((1,)) * (0.761394347824112 - 0.42427313659889454)
        joint_init_pos_list[self.joints_index[6]] = -1.9746345032203163 + np.random.random_sample((1,)) * (1.9746345032203163 - 1.369853919109763)
        joint_init_pos_list[self.joints_index[7]] = -0.2882251755271585 + np.random.random_sample((1,)) * (0.2882251755271585 - 0.09486357350719085)
        joint_init_pos_list[self.joints_index[8]] = -1.0229275938387836 + np.random.random_sample((1,)) * (1.0229275938387836 - 0.6918956944437685)
        joint_init_pos_list[self.joints_index[9]] = -0.610192621369943 + np.random.random_sample((1,)) * (0.610192621369943 - 0.01911017109254098)
        joint_init_pos_list[self.joints_index[10]] = -1.7118547842009764 + np.random.random_sample((1,)) * (1.7118547842009764 - 1.0308833647970654)
        joint_init_pos_list[self.joints_index[11]] = -1.4718621476983345 + np.random.random_sample((1,)) * (1.4718621476983345 - 1.3746251986672793)
        joint_init_pos_list[self.joints_index[12]] = -1.483488663479734 + np.random.random_sample((1,)) * (1.483488663479734 - 1.3324726008921126)
        joint_init_pos_list[self.joints_index[13]] = -0.49900140746609206 + np.random.random_sample((1,)) * (0.49900140746609206 + 0.8979745330184457)
        joint_init_pos_list[self.joints_index[14]] = -1.0383223159057764 + np.random.random_sample((1,)) * (1.0383223159057764 - 0.0011665145034663524)
        joint_init_pos_list[self.joints_index[15]] = -0.9933693558149945 + np.random.random_sample((1,)) * (0.9933693558149945 - 0.9786463309012039)
        joint_init_pos_list[self.joints_index[16]] = -1.5708091101283777 + np.random.random_sample((1,)) * (1.5708091101283777 - 1.3420558127028452)
        [p.resetJointState(self.robot, i, joint_init_pos_list[i], 0, physicsClientId=self.client_ID) for i in range(29)]
        p.resetBasePositionAndOrientation(self.robot, [-0.06, -0.15, 1.3], [-0.0011780936821453951, 0.06136378079357118, 0.877563835439652, 0.47551531335009795], physicsClientId=self.client_ID)
        p.resetBaseVelocity(self.robot, [0, 0, 0], [0, 0, 0])

        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=range(29),
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=joint_init_pos_list,
                                    forces=[100]*29,
                                    positionGains=[0] * 29 ,
                                    velocityGains=[0] * 29 ,
                                    physicsClientId=self.client_ID)

        # Step a few times so stuff settles down
        for i in range(50):
            p.stepSimulation(physicsClientId=self.client_ID)

        self.step_ctr = 0

        left_foot = p.getLinkState(self.robot, self.left_foot)[0]
        right_foot = p.getLinkState(self.robot, self.right_foot)[0]
        pelvis = p.getLinkState(self.robot, self.pelvis)[0]
        left_radius = np.linalg.norm(np.array((left_foot[0] - pelvis[0], left_foot[1] - pelvis[1])))
        right_radius = np.linalg.norm(np.array((right_foot[0] - pelvis[0], right_foot[1] - pelvis[1])))

        self.target_radius = (right_radius + left_radius)/2 *.92
        self.pelvis_orig = pelvis
        self.target_h = (left_foot[2]+right_foot[2])/2
        #print(left_foot, right_foot, pelvis)
        #print(self.target_radius, self.target_h)
        #self.limit_height = (pelvis[2] - (left_foot[2]+right_foot[2])/2)*.5
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
        return (np.array(action) * 0.5 + 0.5) * self.joints_rad_diff + self.joints_rad_low

    def r_links_outside(self):
        reward = 0
        s = p.getLinkStates(self.robot, range(0, 29), physicsClientId=self.client_ID)
        for link in s:
            dist = link[4][1]/0.26
            dist = min(dist, 1)
            reward += max(-1, dist)
        return reward/29
    
    def r_close_to_target(self):
        l_vec_target = p.getLinkState(self.robot, self.left_foot)[0] - self.l_foot_target
        dist_l = np.linalg.norm(l_vec_target) 
        r_vec_target = p.getLinkState(self.robot, self.right_foot)[0] - self.r_foot_target
        dist_r = np.linalg.norm(r_vec_target)
        t_vec_target = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.client_ID)[0] - self.torso_target
        dist_t = np.linalg.norm(t_vec_target)

        if self.step_ctr == 1:
            self.max_dist = dist_l+dist_r+dist_t
        r = 1-(dist_l+dist_r+dist_t)/self.max_dist
        r = max(r, -1)
        r = min(r, 1)
        #print(r)
        return r

    def r_close_to_target_out5(self):
        l_vec_target = p.getLinkState(self.robot, self.left_foot)[0] - self.l_foot_target
        dist_l = np.linalg.norm(l_vec_target)
        r_vec_target = p.getLinkState(self.robot, self.right_foot)[0] - self.r_foot_target
        dist_r = np.linalg.norm(r_vec_target)
        t_vec_target = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.client_ID)[0] - self.torso_target
        dist_t = np.linalg.norm(t_vec_target)
        ls_vec_target = p.getLinkState(self.robot, 24)[0] - self.l_foot_target
        ls_vec_target[2] -= 1.4
        dist_ls = np.linalg.norm(ls_vec_target)
        rs_vec_target = p.getLinkState(self.robot, 19)[0] - self.r_foot_target
        rs_vec_target[2] -= 1.4
        dist_rs = np.linalg.norm(rs_vec_target)

        if self.step_ctr == 1:
            self.max_distL = dist_l
            self.max_distR = dist_r
            self.max_distT = dist_t
            self.max_distLS = dist_ls
            self.max_distRS = dist_rs

        r = 1-dist_l/self.max_distL
        r += 1 - dist_r/self.max_distR
        r += 1 - dist_t/self.max_distT
        r += 1 - dist_ls/self.max_distLS
        r += 1 - dist_rs/self.max_distRS

        r = max(r, -5)
        r = min(r, 5)
        #print(r)
        return r

    def r_tumble(self):

        punishment = -100
        
        left_foot = p.getLinkState(self.robot,self.left_foot)[0][2]
        right_foot = p.getLinkState(self.robot,self.right_foot)[0][2]
        stomach = p.getLinkState(self.robot,self.pelvis)[0][2]
        left_shoulder = p.getLinkState(self.robot,self.left_shoulder)[0][2]
        right_shoulder = p.getLinkState(self.robot,self.right_shoulder)[0][2]
        foot = max(left_foot, right_foot)
        shoulder = min(left_shoulder, right_shoulder)

        if (foot >= stomach or foot >= shoulder or stomach >= shoulder) or (shoulder < 0.5 and stomach < 0.5):
            return punishment
        return 0
    
    def r_standing(self):
        max_limit = 1.4 # approx height of shoulder when standing straight
        min_limit = 0.7 # approx height of shoulder when sitting straight
        punishment = -100
        left_shoulder = p.getLinkState(self.robot,self.left_shoulder)[0][2]
        right_shoulder = p.getLinkState(self.robot,self.right_shoulder)[0][2]
        
        if (left_shoulder < max_limit or right_shoulder < max_limit):
            reward = min(left_shoulder, right_shoulder) - min_limit
        else:
            reward = max_limit - min_limit
            
        # if shoulder is too low, punish
        if (min(left_shoulder, right_shoulder) < min_limit):
            reward = punishment
        return reward


if __name__ == "__main__":
    model = HumanoidBulletEnv(True)
    p.setRealTimeSimulation(1)
    # model.step_ctr = 1
    while(1):
        #print(model.r_close_to_target_out5())
        # model.step_ctr = 2

        """ l = p.getLinkStates(model.robot, [5, 9])
        l2 = p.getLinkStates(model.robot, [12, 16])
        torso_z = (p.getLinkStates(model.robot, [0]))[0][4][2]

        vec1 = (l[1][4][0] - l[0][4][0], l[1][4][1] - l[0][4][1], l[1][4][2] - l[0][4][2])
        vec2 = (l2[1][4][0] - l2[0][4][0], l2[1][4][1] - l2[0][4][1], l2[1][4][2] - l2[0][4][2])

        vec_vychozi = [0, 1, 0]

        skalarni_soucin1 = vec1[0] * vec_vychozi[0] + vec1[1] * vec_vychozi[1] + vec1[2] * vec_vychozi[2]
        angle1 = math.acos(skalarni_soucin1 / math.sqrt(vec1[0] * vec1[0] + vec1[1] * vec1[1] + vec1[2] * vec1[2]))
        skalarni_soucin2 = vec2[0] * vec_vychozi[0] + vec2[1] * vec_vychozi[1] + vec2[2] * vec_vychozi[2]
        angle2 = math.acos(skalarni_soucin2 / math.sqrt(vec2[0] * vec2[0] + vec2[1] * vec2[1] + vec2[2] * vec2[2]))

        legs_rot = ((angle1) + (angle2)) / (pi)
        print(1-legs_rot) """

        # torso_pos, torso_quat = p.getBasePositionAndOrientation(model.robot, physicsClientId=model.client_ID)
        # roll, pitch, yaw = p.getEulerFromQuaternion(torso_quat)
        # print(((1 - abs(2*roll)/pi) + (1 - abs(2*pitch)/pi)))

        print(p.getJointState(model.robot, 20, physicsClientId=model.client_ID)[0])

        a = 1
        keys = p.getKeyboardEvents()
        #print(p.getLinkState(model.robot, model.left_foot, physicsClientId=model.client_ID)[0])
        for k in keys:
            if (keys[k] & p.KEY_WAS_TRIGGERED):
                if (k == ord('i')):
                    #print(p.getJointStates(model.robot, model.joints_index, physicsClientId=model.client_ID))
                    #print(p.getLinkState(model.robot, model.left_foot, physicsClientId=model.client_ID))
                    model.reset()
            if k == ord('o'):
                    o = model.get_obs()
          
                
                
          
                
                
