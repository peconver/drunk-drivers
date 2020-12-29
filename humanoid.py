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
        p.setTimeStep(1/120, self.client_ID)

        # Load actual robot and car into the world
        shift_x = .8
        shift_z = .17
        shift_y = -.35
        self.car = p.loadURDF("models/polaris.urdf", [-shift_x, shift_y, (shift_z + 0.32)], useFixedBase=True, physicsClientId=self.client_ID, globalScaling=.97)
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
        self.obs_dim = 110  # joints + torques + contacts
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

        self.max_joint_force = 45
        self.lateral_friction = 2.0
        self.torso_target = np.array([-0.05, -0.01, 1.38])
        self.l_foot_target = np.array([-0.17, 0.55, 0.6])
        self.r_foot_target = np.array([-0.22, 0.55, 0.6])

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
        p.resetDebugVisualizerCamera(cameraDistance=8, cameraYaw=0, cameraPitch=0, cameraTargetPosition=[0, 0, 2])

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
            #time.sleep(.05)
        # Get new observations (Note, not all of these are required and used)
        torso_pos, torso_quat, torso_vel, torso_angular_vel, joint_angles, joint_velocities, joint_torques, contacts = self.get_obs()
        xd, yd, zd = torso_vel  # Unpack for clarity
        roll, pitch, yaw = p.getEulerFromQuaternion(torso_quat)

        self.step_ctr += 1

        """reward
        """
        # try to make him turn legs from the car
        l = p.getLinkStates(self.robot, [5,9])
        l2 = p.getLinkStates(self.robot, [12, 16])
        torso_z = (p.getLinkStates(self.robot, [0]))[0][4][2]

        vec1 = (l[1][0][0] - l[0][0][0], l[1][0][1] - l[0][0][1], l[1][0][2] - l[0][0][2])
        vec2 = (l2[1][0][0] - l2[0][0][0], l2[1][0][1] - l2[0][0][1], l2[1][0][2] - l2[0][0][2])

        vec_vychozi = [0,1,0]

        skalarni_soucin1 = vec1[0]*vec_vychozi[0] + vec1[1]*vec_vychozi[1] + vec1[2]*vec_vychozi[2]
        angle1 = math.acos(skalarni_soucin1/math.sqrt(vec1[0]*vec1[0]+vec1[1]*vec1[1]+vec1[2]*vec1[2]))
        skalarni_soucin2 = vec2[0] * vec_vychozi[0] + vec2[1] * vec_vychozi[1] + vec2[2] * vec_vychozi[2]
        angle2 = math.acos(skalarni_soucin2 / math.sqrt(vec2[0] * vec2[0] + vec2[1] * vec2[1] + vec2[2] * vec2[2]))
        # 2<0;pi> -> good<0;2>bad
        legs_rot = ((angle1) + (angle2)) / (pi)
        # <-pi;pi> -> <-3pi/2;pi/2> -> <0;3pi/2> -> good<0;1>bad
        torso_good_rot = abs(yaw - pi / 2)*2 / (3*pi)
        # 2<-pi;pi> -> 2<0;pi> -> good<0;1>bad

        joint_angles_n = ((joint_angles - self.joints_rad_low) / self.joints_rad_diff) * 2 - 1
        yaw = ((yaw + pi) / (2 * pi)) * 2 - 1
        roll = ((roll + pi) / (2 * pi)) * 2 - 1
        pitch = ((pitch + pi) / (2 * pi)) * 2 - 1

        feet, pelvis, neck = self.feet_pelvis_neck_movement()
        r_neg = neck*2 + pelvis*3 + feet
        r_pos = (1-torso_good_rot)*2 + (2 - legs_rot) + (1 - (abs(roll) + abs(pitch))/2)*3
        r = np.clip(r_pos - r_neg, -5, 5)
        """reward
        """
        # horizontal shift
        # left_shoulder = p.getLinkState(self.robot, self.left_shoulder)[0]
        # right_shoulder = p.getLinkState(self.robot, self.right_shoulder)[0]
        # min_limit = 1.2
        # max_limit = 1.5
        # tmp = max((left_shoulder[2] - min_limit) / (max_limit-min_limit), -1)
        # r_left_shoulder = min(tmp, 1)
        # tmp = max((right_shoulder[2] - min_limit) / (max_limit-min_limit), -1)
        # r_right_shoulder = min(tmp, 1)

        # height of shoulders
        left_shoulder = p.getLinkState(self.robot, self.left_shoulder)[0]
        right_shoulder = p.getLinkState(self.robot, self.right_shoulder)[0]
        min_limit = 1.0
        max_limit = 1.4
        tmp = left_shoulder[2] - min_limit
        # shoulder is under limit, punish
        if tmp < 0:
            r_left_shoulder = tmp/min_limit
        # shoulder is higher than limit, reward
        else:
            r_left_shoulder = min(tmp/(max_limit-min_limit),1)
        tmp = right_shoulder[2] - min_limit
        # shoulder is under limit, punish
        if tmp < 0:
            r_right_shoulder = tmp/min_limit
        # shoulder is higher than limit, reward
        else:
            r_right_shoulder = min(tmp/(max_limit-min_limit),1)

        # print('reward shoulders:', r_right_shoulder, r_left_shoulder)
        #  rotation of torso
        # roll_torso, pitch_torso, yaw_torso
        euler_angles = p.getEulerFromQuaternion(torso_quat)
        # here will be rewards for roll and pitch of torso stored
        r_euler_angles = [0, 0]
        
        # if shoulders are under limit, reward for leaning forward
        # if min(left_shoulder[2], right_shoulder[2]) < min_limit:
        #    # TODO - roll should be in boundaries
        #    if euler_angles[1] > 0 and euler_angles[1] < pi/2
        #        r_euler_angles[0] = euler_angles[1]/(pi/4)
        #        if euler_angles[1] > pi/4
        #            r_euler_angles[0]  = ((pi/2) - euler_angles[1])/(pi/4)
        #    # TODO - reward for roll
        if (1):
        # iterate over roll and pitch
            for index, angle in enumerate(euler_angles[0:2]):
                angle = abs(angle)
                # if roll or pitch are in boundaries, reward
                if angle < 0.34:
                    r_euler_angles[index] = min((0.34 - angle)/0.14, 1)
                # if roll or pitch are too big, punish
                else:
                    r_euler_angles[index] = max(-(angle-0.34)/1.23, -1)
                    
        # punishment for jumping
        left_foot = p.getLinkState(self.robot, self.left_foot)[0][2]
        right_foot = p.getLinkState(self.robot, self.right_foot)[0][2]
        upper_limit_of_lower_foot = 0.23
        r_feet = max((upper_limit_of_lower_foot - max(min(left_foot, right_foot), upper_limit_of_lower_foot))/0.4, -1)
       
        # TODO - add reward when only feet ar in contact
        # print('roll pitch yaw', roll_torso, pitch_torso, yaw_torso)

        # TODO - add hint when falling
        # reward
        # TODO - jake rewardy chybi?
        r = r_left_shoulder + r_right_shoulder + sum(r_euler_angles)*1/2 + 2*r_feet

        # TODO - omezit rychlost pohybu?
        # sum([abs(ele) for ele in joint_velocities])


        # print('reward', r)
        # [abs(ele) for ele in test_list]
        # print('joint velocities', sum([abs(ele) for ele in joint_velocities]))
        # get joint_angles into np array and make observation
        env_obs = np.concatenate((torso_pos, [roll, pitch, yaw], joint_angles_n, joint_velocities, contacts,
                                  left_shoulder, right_shoulder, r_euler_angles, [r_right_shoulder, r_left_shoulder], [left_foot, right_foot]),
                                 axis=0).astype(np.float32)

        # env_obs = np.concatenate((torso_pos, [roll, pitch, yaw], joint_angles_n, joint_velocities, contacts),
        #                           axis=0).astype(np.float32)
        # This condition terminates the episode - WARNING - it can cause that the robot will try 
        # terminate the episode as quickly as possible
        # done = self.step_ctr > self.max_steps or (np.abs(roll) > 1.5 or np.abs(pitch) > 1.5) or torso_z < 1 or (legs_rot <.2 and abs(yaw-pi/2) < .2)
        # done = self.step_ctr > self.max_steps or (np.abs(roll) > .5 or np.abs(pitch) > .5) or torso_z < 1 or (angle1 < .05 and angle2 < .05 and abs(yaw-.5) < .1)
        done = (self.step_ctr > self.max_steps) or (min(left_shoulder[2], right_shoulder[2]) < 1.0)

        env_obs_noisy = env_obs# + np.random.rand(self.obs_dim).astype(np.float32) * 0.1 - 0.05
        info = {'rotated':(angle1 < .2 and angle2 < .2 and abs(yaw-.5) < .2)}#,"joints":joint_angles,"pos":torso_pos,"quat":torso_quat

        return env_obs_noisy, r, done, info
    
    def feet_pelvis_neck_movement(self):
        left_foot = p.getLinkState(self.robot, self.left_foot)[0]
        right_foot = p.getLinkState(self.robot, self.right_foot)[0]
        pelvis = p.getLinkState(self.robot, self.pelvis)[0]

        height = (left_foot[2],right_foot[2])
        h = np.clip((abs(height[0] - self.target_h) + abs(height[1] - self.target_h))*2,0 , 2)

        p_or = self.pelvis_orig
        pelvis_shift = np.linalg.norm(np.array((pelvis[0] - p_or[0], (pelvis[1] - p_or[1])*.5, pelvis[2] - p_or[2])))
        p_r = np.clip(pelvis_shift*5, 0, 1)

        left_shoulder = p.getLinkState(self.robot, self.left_shoulder)[0]
        right_shoulder = p.getLinkState(self.robot, self.right_shoulder)[0]
        neck = ((left_shoulder[0]+right_shoulder[0])/2, (left_shoulder[1]+right_shoulder[1])/2)
        n_r = np.linalg.norm(np.array((pelvis[0] - neck[0], pelvis[1] - neck[1])))
        n_r = np.clip(n_r*5, 0, 1)

        return h, p_r, n_r

    def reset(self):
        # # Reset the robot to initial position and orientation and null the motors
        # joint_init_pos_list = [0] * 29
        # joint_init_pos_list[9] = -1.4
        # joint_init_pos_list[16] = -1.4
        # joint_init_pos_list[7] = -1
        # joint_init_pos_list[14] = -1
        # joint_init_pos_list[1] = -.5
        # [p.resetJointState(self.robot, i, joint_init_pos_list[i], 0, physicsClientId=self.client_ID) for i in range(29)]
        # p.resetBasePositionAndOrientation(self.robot, [-0.1, 0, 1.48], [0, 0, 0, 1], physicsClientId=self.client_ID)
        # p.resetBaseVelocity(self.robot, [0, 0, 0], [0, 0, 0])
        #
        # p.setJointMotorControlArray(bodyUniqueId=self.robot,
        #                             jointIndices=range(29),
        #                             controlMode=p.POSITION_CONTROL,
        #                             targetPositions=joint_init_pos_list,
        #                             forces=[100]*29,
        #                             positionGains=[0] * 29 ,
        #                             velocityGains=[0] * 29 ,
        #                             physicsClientId=self.client_ID)
        #
        # # Step a few times so stuff settles down
        # for i in range(50):
        #     p.stepSimulation(physicsClientId=self.client_ID)
        #
        # self.step_ctr = 0
        #
        # left_foot = p.getLinkState(self.robot, self.left_foot)[0]
        # right_foot = p.getLinkState(self.robot, self.right_foot)[0]
        # pelvis = p.getLinkState(self.robot, self.pelvis)[0]
        # left_radius = np.linalg.norm(np.array((left_foot[0] - pelvis[0], left_foot[1] - pelvis[1])))
        # right_radius = np.linalg.norm(np.array((right_foot[0] - pelvis[0], right_foot[1] - pelvis[1])))
        #
        # self.pelvis_orig = pelvis
        # self.target_h = (left_foot[2]+right_foot[2])/2
        # # Return initial obs
        # obs, _, _, _ = self.step(np.zeros(self.act_dim))
        # return obs

        # Reset the robot to initial position and orientation and null the motors
        joint_init_pos_list = [0] * 29
        joint_init_pos_list[9] = 0
        joint_init_pos_list[16] = 0
        joint_init_pos_list[7] = 0
        joint_init_pos_list[14] = 0
        joint_init_pos_list[1] = 0
        [p.resetJointState(self.robot, i, joint_init_pos_list[i], 0, physicsClientId=self.client_ID) for i in range(29)]
        p.resetBasePositionAndOrientation(self.robot, [-0.1, -4, 1.48], [0, 0, 0, 1], physicsClientId=self.client_ID)
        p.resetBaseVelocity(self.robot, [0, 0, 0], [0, 0, 0])

        # p.setJointMotorControlArray(bodyUniqueId=self.robot,
        #                             jointIndices=range(29),
        #                             controlMode=p.POSITION_CONTROL,
        #                             targetPositions=joint_init_pos_list,
        #                             forces=[100] * 29,
        #                             positionGains=[0] * 29,
        #                             velocityGains=[0] * 29,
        #                             physicsClientId=self.client_ID)

        # Step a few times so stuff settles down
        for i in range(80):
             p.stepSimulation(physicsClientId=self.client_ID)

        self.step_ctr = 0

        left_foot = p.getLinkState(self.robot, self.left_foot)[0]
        right_foot = p.getLinkState(self.robot, self.right_foot)[0]
        pelvis = p.getLinkState(self.robot, self.pelvis)[0]
        left_radius = np.linalg.norm(np.array((left_foot[0] - pelvis[0], left_foot[1] - pelvis[1])))
        right_radius = np.linalg.norm(np.array((right_foot[0] - pelvis[0], right_foot[1] - pelvis[1])))

        self.pelvis_orig = pelvis
        self.target_h = (left_foot[2] + right_foot[2]) / 2
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
        car_half_wide = (1.22+0.15)/2.0
        reward = 0
        s = p.getLinkStates(self.robot, range(0, 13+1), physicsClientId=self.client_ID)
        for link in s:
            dist = link[4][1] - car_half_wide
            reward -= max(dist, 0)
        return reward
    
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

    while(1):
        left_foot = p.getLinkState(model.robot, model.left_foot)[0][2]
        right_foot = p.getLinkState(model.robot, model.right_foot)[0][2]
        upper_limit_of_lower_foot = 0.19
        r_feet = max((upper_limit_of_lower_foot - max(min(left_foot, right_foot), upper_limit_of_lower_foot))/0.4, -1)*2
        # print('feet and reward!!!:',left_foot, right_foot, r_feet)
        # height of shoulders
        left_shoulder = p.getLinkState(model.robot, model.left_shoulder)[0]
        right_shoulder = p.getLinkState(model.robot, model.right_shoulder)[0]
        min_limit = 1.0
        max_limit = 1.45
        tmp = left_shoulder[2] - min_limit
        # shoulder is under limit, punish
        if tmp < 0:
            r_left_shoulder = tmp / min_limit
        # shoulder is higher than limit, reward
        else:
            r_left_shoulder = min(tmp / (max_limit - min_limit), 1)
        tmp = right_shoulder[2] - min_limit
        # shoulder is under limit, punish
        if tmp < 0:
            r_right_shoulder = tmp / min_limit
        # shoulder is higher than limit, reward
        else:
            r_right_shoulder = min(tmp / (max_limit - min_limit), 1)

        print('reward shoulders:', r_right_shoulder, r_left_shoulder)
        a = 1
        keys = p.getKeyboardEvents()
        for k in keys:
            if (keys[k] & p.KEY_WAS_TRIGGERED):
                if (k == ord('i')):
                    model.reset()
            if k == ord('o'):
                    o = model.get_obs()
          
                
                
