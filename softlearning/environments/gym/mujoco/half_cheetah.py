import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os
from . import path
from .mirror_utils import convert_to_mirror_list,MIRROR_DICTS
from collections import deque

DEFAULT_CAMERA_CONFIG = {
    'distance': 4.0,
}
"""
0 rootz     slider      C
1 rooty     hinge       C
2 bthigh    hinge       L
3 bshin     hinge       L
4 bfoot     hinge       L
5 fthigh    hinge       R
6 fshin     hinge       R
7 ffoot     hinge       R
8 rootx     slider      C
9 rootz     slider      C
10 rooty     hinge      C
11 bthigh    hinge      L
12 bshin     hinge      L
13 bfoot     hinge      L
14 fthigh    hinge      R
15 fshin     hinge      R
16 ffoot     hinge      R
"""

class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='half_cheetah.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 energy_weights=0.):

        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self.joint_list=['bthigh','bshin','bfoot','fthigh','fshin','ffoot']

        self._ctrl_cost_weight = ctrl_cost_weight
        self.energy_weights=energy_weights
        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        self.mirror_inds=MIRROR_DICTS["HalfCheetah"]
        self.mirror_lists=convert_to_mirror_list(self.mirror_inds)

        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), 5)
        #mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        states_angle = []
        for j in self.joint_list:
            states_angle.append(self.sim.data.get_joint_qpos(j))
        #states=self._get_obs()
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]

        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        #next_states=observation
        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        reward = forward_reward - ctrl_cost
        done = False

        energy = 0
        for i in range(6):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(action[i]) * delta_theta

        '''delta_theta_bt = np.abs(next_states[2] - states[2])
        delta_theta_bs = np.abs(next_states[3] - states[3])
        delta_theta_bf = np.abs(next_states[4] - states[4])
        delta_theta_ft = np.abs(next_states[5] - states[5])
        delta_theta_fs = np.abs(next_states[6] - states[6])
        delta_theta_ff = np.abs(next_states[7] - states[7])

        energy_bt = np.abs(action[0]) * delta_theta_bt
        energy_bs = np.abs(action[1]) * delta_theta_bs
        energy_bf = np.abs(action[2]) * delta_theta_bf
        energy_ft = np.abs(action[3]) * delta_theta_ft
        energy_fs = np.abs(action[4]) * delta_theta_fs
        energy_ff = np.abs(action[5]) * delta_theta_ff

        energy = energy_bt + energy_bs + energy_bf + energy_ft + energy_fs + energy_ff'''

        reward -= self.energy_weights*energy
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            'energy'    : energy,
            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'ori_reward':forward_reward-ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

class HalfCheetahHeavyEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='half_cheetah_heavy.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 energy_weights=0.):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self.joint_list=['bthigh','bshin','bfoot','fthigh','fshin','ffoot']

        self._ctrl_cost_weight = ctrl_cost_weight
        self.energy_weights=energy_weights
        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)
        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), 5)
        #mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        states_angle = []
        for j in self.joint_list:
            states_angle.append(self.sim.data.get_joint_qpos(j))
        #states=self._get_obs()
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]

        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        #next_states=observation
        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        reward = forward_reward - ctrl_cost
        done = False

        energy = 0
        for i in range(6):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(action[i]) * delta_theta

        '''delta_theta_bt = np.abs(next_states[2] - states[2])
        delta_theta_bs = np.abs(next_states[3] - states[3])
        delta_theta_bf = np.abs(next_states[4] - states[4])
        delta_theta_ft = np.abs(next_states[5] - states[5])
        delta_theta_fs = np.abs(next_states[6] - states[6])
        delta_theta_ff = np.abs(next_states[7] - states[7])

        energy_bt = np.abs(action[0]) * delta_theta_bt
        energy_bs = np.abs(action[1]) * delta_theta_bs
        energy_bf = np.abs(action[2]) * delta_theta_bf
        energy_ft = np.abs(action[3]) * delta_theta_ft
        energy_fs = np.abs(action[4]) * delta_theta_fs
        energy_ff = np.abs(action[5]) * delta_theta_ff

        energy = energy_bt + energy_bs + energy_bf + energy_ft + energy_fs + energy_ff'''

        reward -= self.energy_weights*energy
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            'energy'    : energy,
            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'ori_reward':forward_reward-ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

class FullCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='full_cheetah.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 contact_cost_weight=0,  ##5e-4
                 contact_force_range=(-1.0, 1.0),
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 walkstyle="",#"gallop" "trot"
                 energy_weights=0.,
                 phase_delay=0,
                 soft_gait_penalty_weight=0,
                 soft_gait_target=0,
                 soft_gait_target2=0,
                 speed=0):
        utils.EzPickle.__init__(**locals())

        self.speed=speed
        self._forward_reward_weight = forward_reward_weight
        self.soft_gait_penalty_weight=soft_gait_penalty_weight
        self.soft_gait_target=soft_gait_target
        self.soft_gait_target2=soft_gait_target2
        self.joint_list=['bthighL','bshinL','bfootL','fthighL','fshinL','ffootL',
                         'bthighR', 'bshinR', 'bfootR', 'fthighR', 'fshinR', 'ffootR']

        self._ctrl_cost_weight = ctrl_cost_weight
        self.energy_weights=energy_weights
        self._reset_noise_scale = reset_noise_scale
        self._contact_cost_weight = contact_cost_weight
        self._contact_force_range = contact_force_range
        self.phase_delay=phase_delay
        if self.phase_delay!=0:
            self.delay_deque=deque(maxlen=self.phase_delay)

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        self.walkstyle=walkstyle
        mykey="FullCheetah_"+walkstyle
        self.agent=mykey

        self.mirror_lr_inds=MIRROR_DICTS["FullCheetah_lr"]
        self.mirror_lr_lists=convert_to_mirror_list(self.mirror_lr_inds)

        self.mirror_fb_inds=MIRROR_DICTS["FullCheetah_fb"]
        self.mirror_fb_lists=convert_to_mirror_list(self.mirror_fb_inds)

        self.contact={
            "BL":[],
            "FL": [],
            "BR": [],
            "FR": [],
        }


        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), 5)

    def contact_forces(self):
        raw_contact_forces = self.sim.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    def contact_cost(self):

        contact_cost = np.sum(np.square(self.contact_forces()))
        return self._contact_cost_weight *contact_cost

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def gait_cost(self,action):
        gait_cost=0

        if self.walkstyle=="gallop":
            gait_cost=np.abs(np.sum(np.abs(action[6:12] - action[0:6]))-self.soft_gait_target)
        elif self.walkstyle == "gallop2":
            gait_cost=np.abs(np.sum(np.abs(action[6:9] - action[0:3]))-self.soft_gait_target)+\
                      np.abs(np.sum(np.abs(action[9:12] - action[3:6]))-self.soft_gait_target2)

        elif self.walkstyle == "trot":
            gait_cost=np.abs(np.sum(np.abs(action[6:9] +action[0:3]))
                             +np.sum(np.abs(action[9:12] +action[3:6]))-self.soft_gait_target)

        return self.soft_gait_penalty_weight *gait_cost

    def step(self, action):
        if self.soft_gait_penalty_weight==0:
            if self._contact_cost_weight==0 and self.walkstyle=="gallop":

                if self.phase_delay!=0 and len(self.delay_deque)==self.phase_delay:
                    action[6:12] = self.delay_deque[0]
                else:
                    action[6:12] = action[0:6]  # Right equals left

                if self.phase_delay != 0:
                    self.delay_deque.append((action[0:6]))

            elif self.walkstyle=="trot":
                action[6:9] = -1*action[0:3]#bR equals negative bL
                action[9:12] = -1 * action[3:6]  # fR equals negative fL
            elif self.walkstyle == "trotv2":
                action[9:12] =  action[0:3]  # fR equals  bL
                action[6:9] =  action[3:6]  # bR equals  fL

        # self.contact["BL"].append(0)
        # self.contact["FL"].append(0)
        # self.contact["BR"].append(0)
        # self.contact["FR"].append(0)
        #
        # for contactnum in range(self.data.ncon):
        #     c=self.data.contact[contactnum]
        #     if c.dist!=0:
        #         cur_g = c.geom2
        #         if cur_g == 8:
        #             self.contact["BL"][-1]=1
        #             #print("BL")
        #         elif cur_g == 11:
        #             self.contact["FL"][-1] = 1
        #             #print("FL")
        #         elif cur_g == 14:
        #             self.contact["BR"][-1] = 1
        #             #print("BR")
        #         elif cur_g == 17:
        #             #print("FR")
        #             self.contact["FR"][-1] = 1
        #
        # if len(self.contact["BL"])>500:
        #     print("SAVE gait!")
        #     #name="gallop_200_s5r5"
        #     name = "RealFCT_v1r1"
        #     np.save('../gait_info/' + name, self.contact)
        #     quit()

        #print(self.data.contact[0].geom2)
        states_angle = []
        for j in self.joint_list:
            states_angle.append(self.sim.data.get_joint_qpos(j))
        #states=self._get_obs()
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]

        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        if self.speed != 0:
            delta_speed=-abs(x_velocity-self.speed) #negative as we want to minimize this delta
            x_velocity=delta_speed

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost()
        gait_cost = self.gait_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        #next_states=observation
        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        reward = forward_reward - ctrl_cost -  contact_cost- gait_cost
        done = False

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(action[i]) * delta_theta

        '''delta_theta_bt = np.abs(next_states[2] - states[2])
        delta_theta_bs = np.abs(next_states[3] - states[3])
        delta_theta_bf = np.abs(next_states[4] - states[4])
        delta_theta_ft = np.abs(next_states[5] - states[5])
        delta_theta_fs = np.abs(next_states[6] - states[6])
        delta_theta_ff = np.abs(next_states[7] - states[7])

        energy_bt = np.abs(action[0]) * delta_theta_bt
        energy_bs = np.abs(action[1]) * delta_theta_bs
        energy_bf = np.abs(action[2]) * delta_theta_bf
        energy_ft = np.abs(action[3]) * delta_theta_ft
        energy_fs = np.abs(action[4]) * delta_theta_fs
        energy_ff = np.abs(action[5]) * delta_theta_ff

        energy = energy_bt + energy_bs + energy_bf + energy_ft + energy_fs + energy_ff'''

        reward -= self.energy_weights*energy
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            'energy'    : energy,
            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'reward_contact': -contact_cost,
            'reward_gait': -gait_cost,
            'ori_reward':forward_reward-ctrl_cost -contact_cost-gait_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

class HalfCheetahEnv_5dof(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='half_cheetah_5dof.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 energy_weights=0.):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self.joint_list=['bthigh','bshin','bfoot','fthigh','fshin']

        self._ctrl_cost_weight = ctrl_cost_weight
        self.energy_weights=energy_weights
        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), 5)

        #mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        states_angle = []
        for j in self.joint_list:
            states_angle.append(self.sim.data.get_joint_qpos(j))
        #states=self._get_obs()
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]

        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        #next_states=observation
        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        reward = forward_reward - ctrl_cost
        done = False

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(action[i]) * delta_theta

        '''delta_theta_bt = np.abs(next_states[2] - states[2])
        delta_theta_bs = np.abs(next_states[3] - states[3])
        delta_theta_bf = np.abs(next_states[4] - states[4])
        delta_theta_ft = np.abs(next_states[5] - states[5])
        delta_theta_fs = np.abs(next_states[6] - states[6])
        delta_theta_ff = np.abs(next_states[7] - states[7])

        energy_bt = np.abs(action[0]) * delta_theta_bt
        energy_bs = np.abs(action[1]) * delta_theta_bs
        energy_bf = np.abs(action[2]) * delta_theta_bf
        energy_ft = np.abs(action[3]) * delta_theta_ft
        energy_fs = np.abs(action[4]) * delta_theta_fs
        energy_ff = np.abs(action[5]) * delta_theta_ff

        energy = energy_bt + energy_bs + energy_bf + energy_ft + energy_fs + energy_ff'''

        reward -= self.energy_weights*energy
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            'energy'    : energy,
            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'ori_reward':forward_reward-ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

class HalfCheetahEnv_5dofv2(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='half_cheetah_5dofv2.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 energy_weights=0.):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight
        #self.joint_list = ['bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot']
        self.joint_list=['bthigh','bshin','bfoot','fthigh', 'ffoot']

        self._ctrl_cost_weight = ctrl_cost_weight
        self.energy_weights=energy_weights
        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), 5)

        #mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        states_angle = []
        for j in self.joint_list:
            states_angle.append(self.sim.data.get_joint_qpos(j))
        #states=self._get_obs()
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]

        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        #next_states=observation
        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        reward = forward_reward - ctrl_cost
        done = False

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(action[i]) * delta_theta


        reward -= self.energy_weights*energy
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            'energy'    : energy,
            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'ori_reward':forward_reward-ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

class HalfCheetahEnv_5dofv3(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='half_cheetah_5dofv3.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 energy_weights=0.):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight
        #self.joint_list = ['bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot']
        self.joint_list=['bthigh','bshin','bfoot', 'fshin', 'ffoot']

        self._ctrl_cost_weight = ctrl_cost_weight
        self.energy_weights=energy_weights
        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), 5)

        #mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        states_angle = []
        for j in self.joint_list:
            states_angle.append(self.sim.data.get_joint_qpos(j))
        #states=self._get_obs()
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]

        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        #next_states=observation
        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        reward = forward_reward - ctrl_cost
        done = False

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(action[i]) * delta_theta


        reward -= self.energy_weights*energy
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            'energy'    : energy,
            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'ori_reward':forward_reward-ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

class HalfCheetahEnv_5dofv4(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='half_cheetah_5dofv4.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 energy_weights=0.):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight
        #self.joint_list = ['bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot']
        self.joint_list=['bshin','bfoot',  'fthigh','fshin', 'ffoot']

        self._ctrl_cost_weight = ctrl_cost_weight
        self.energy_weights=energy_weights
        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), 5)

        #mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        states_angle = []
        for j in self.joint_list:
            states_angle.append(self.sim.data.get_joint_qpos(j))
        #states=self._get_obs()
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]

        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        #next_states=observation
        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        reward = forward_reward - ctrl_cost
        done = False

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(action[i]) * delta_theta


        reward -= self.energy_weights*energy
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            'energy'    : energy,
            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'ori_reward':forward_reward-ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

class HalfCheetahEnv_5dofv5(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='half_cheetah_5dofv5.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 energy_weights=0.):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight
        #self.joint_list = ['bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot']
        self.joint_list=['bthigh','bfoot',  'fthigh','fshin', 'ffoot']

        self._ctrl_cost_weight = ctrl_cost_weight
        self.energy_weights=energy_weights
        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), 5)

        #mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        states_angle = []
        for j in self.joint_list:
            states_angle.append(self.sim.data.get_joint_qpos(j))
        #states=self._get_obs()
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]

        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        #next_states=observation
        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        reward = forward_reward - ctrl_cost
        done = False

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(action[i]) * delta_theta


        reward -= self.energy_weights*energy
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            'energy'    : energy,
            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'ori_reward':forward_reward-ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

class HalfCheetahEnv_5dofv6(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='half_cheetah_5dofv6.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 energy_weights=0.):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight
        #self.joint_list = ['bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot']
        self.joint_list=['bthigh', 'bshin',  'fthigh','fshin', 'ffoot']

        self._ctrl_cost_weight = ctrl_cost_weight
        self.energy_weights=energy_weights
        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), 5)

        #mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        states_angle = []
        for j in self.joint_list:
            states_angle.append(self.sim.data.get_joint_qpos(j))
        #states=self._get_obs()
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]

        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        #next_states=observation
        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        reward = forward_reward - ctrl_cost
        done = False

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(action[i]) * delta_theta


        reward -= self.energy_weights*energy
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            'energy'    : energy,
            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'ori_reward':forward_reward-ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

class HalfCheetahEnv_4dof(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='half_cheetah_4dof.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 energy_weights=0.):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self.joint_list=['bthigh','bshin','fthigh','fshin']

        self._ctrl_cost_weight = ctrl_cost_weight
        self.energy_weights=energy_weights
        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), 5)

        #mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        states_angle = []
        for j in self.joint_list:
            states_angle.append(self.sim.data.get_joint_qpos(j))
        #states=self._get_obs()
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]

        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        #next_states=observation
        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        reward = forward_reward - ctrl_cost
        done = False

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(action[i]) * delta_theta

        '''delta_theta_bt = np.abs(next_states[2] - states[2])
        delta_theta_bs = np.abs(next_states[3] - states[3])
        delta_theta_bf = np.abs(next_states[4] - states[4])
        delta_theta_ft = np.abs(next_states[5] - states[5])
        delta_theta_fs = np.abs(next_states[6] - states[6])
        delta_theta_ff = np.abs(next_states[7] - states[7])

        energy_bt = np.abs(action[0]) * delta_theta_bt
        energy_bs = np.abs(action[1]) * delta_theta_bs
        energy_bf = np.abs(action[2]) * delta_theta_bf
        energy_ft = np.abs(action[3]) * delta_theta_ft
        energy_fs = np.abs(action[4]) * delta_theta_fs
        energy_ff = np.abs(action[5]) * delta_theta_ff

        energy = energy_bt + energy_bs + energy_bf + energy_ft + energy_fs + energy_ff'''

        reward -= self.energy_weights*energy
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            'energy'    : energy,
            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'ori_reward':forward_reward-ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

class HalfCheetahEnv_4dofv2(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='half_cheetah_4dofv2.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 energy_weights=0.):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight
        #self.joint_list = ['bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot']
        self.joint_list=['bshin','bfoot','fthigh','ffoot']

        self._ctrl_cost_weight = ctrl_cost_weight
        self.energy_weights=energy_weights
        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), 5)

        #mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        states_angle = []
        for j in self.joint_list:
            states_angle.append(self.sim.data.get_joint_qpos(j))
        #states=self._get_obs()
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]

        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        #next_states=observation
        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        reward = forward_reward - ctrl_cost
        done = False

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(action[i]) * delta_theta

        '''delta_theta_bt = np.abs(next_states[2] - states[2])
        delta_theta_bs = np.abs(next_states[3] - states[3])
        delta_theta_bf = np.abs(next_states[4] - states[4])
        delta_theta_ft = np.abs(next_states[5] - states[5])
        delta_theta_fs = np.abs(next_states[6] - states[6])
        delta_theta_ff = np.abs(next_states[7] - states[7])

        energy_bt = np.abs(action[0]) * delta_theta_bt
        energy_bs = np.abs(action[1]) * delta_theta_bs
        energy_bf = np.abs(action[2]) * delta_theta_bf
        energy_ft = np.abs(action[3]) * delta_theta_ft
        energy_fs = np.abs(action[4]) * delta_theta_fs
        energy_ff = np.abs(action[5]) * delta_theta_ff

        energy = energy_bt + energy_bs + energy_bf + energy_ft + energy_fs + energy_ff'''

        reward -= self.energy_weights*energy
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            'energy'    : energy,
            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'ori_reward':forward_reward-ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

class HalfCheetahEnv_4dofv3(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='half_cheetah_4dofv3.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 energy_weights=0.):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight
        #self.joint_list = ['bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot']
        self.joint_list=['bshin','bfoot','fshin','ffoot']

        self._ctrl_cost_weight = ctrl_cost_weight
        self.energy_weights=energy_weights
        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), 5)

        #mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        states_angle = []
        for j in self.joint_list:
            states_angle.append(self.sim.data.get_joint_qpos(j))
        #states=self._get_obs()
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]

        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        #next_states=observation
        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        reward = forward_reward - ctrl_cost
        done = False

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(action[i]) * delta_theta

        '''delta_theta_bt = np.abs(next_states[2] - states[2])
        delta_theta_bs = np.abs(next_states[3] - states[3])
        delta_theta_bf = np.abs(next_states[4] - states[4])
        delta_theta_ft = np.abs(next_states[5] - states[5])
        delta_theta_fs = np.abs(next_states[6] - states[6])
        delta_theta_ff = np.abs(next_states[7] - states[7])

        energy_bt = np.abs(action[0]) * delta_theta_bt
        energy_bs = np.abs(action[1]) * delta_theta_bs
        energy_bf = np.abs(action[2]) * delta_theta_bf
        energy_ft = np.abs(action[3]) * delta_theta_ft
        energy_fs = np.abs(action[4]) * delta_theta_fs
        energy_ff = np.abs(action[5]) * delta_theta_ff

        energy = energy_bt + energy_bs + energy_bf + energy_ft + energy_fs + energy_ff'''

        reward -= self.energy_weights*energy
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            'energy'    : energy,
            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'ori_reward':forward_reward-ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

class HalfCheetahEnv_4dofv4(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='half_cheetah_4dofv4.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 energy_weights=0.):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight
        #self.joint_list = ['bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot']
        self.joint_list=['bshin','bfoot','fthigh','fshin']

        self._ctrl_cost_weight = ctrl_cost_weight
        self.energy_weights=energy_weights
        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), 5)

        #mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        states_angle = []
        for j in self.joint_list:
            states_angle.append(self.sim.data.get_joint_qpos(j))
        #states=self._get_obs()
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]

        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        #next_states=observation
        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        reward = forward_reward - ctrl_cost
        done = False

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(action[i]) * delta_theta

        '''delta_theta_bt = np.abs(next_states[2] - states[2])
        delta_theta_bs = np.abs(next_states[3] - states[3])
        delta_theta_bf = np.abs(next_states[4] - states[4])
        delta_theta_ft = np.abs(next_states[5] - states[5])
        delta_theta_fs = np.abs(next_states[6] - states[6])
        delta_theta_ff = np.abs(next_states[7] - states[7])

        energy_bt = np.abs(action[0]) * delta_theta_bt
        energy_bs = np.abs(action[1]) * delta_theta_bs
        energy_bf = np.abs(action[2]) * delta_theta_bf
        energy_ft = np.abs(action[3]) * delta_theta_ft
        energy_fs = np.abs(action[4]) * delta_theta_fs
        energy_ff = np.abs(action[5]) * delta_theta_ff

        energy = energy_bt + energy_bs + energy_bf + energy_ft + energy_fs + energy_ff'''

        reward -= self.energy_weights*energy
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            'energy'    : energy,
            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'ori_reward':forward_reward-ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

class HalfCheetahEnv_4dofv5(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='half_cheetah_4dofv5.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 energy_weights=0.):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight
        #self.joint_list = ['bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot']
        self.joint_list=['bthigh','bfoot','fshin','ffoot']

        self._ctrl_cost_weight = ctrl_cost_weight
        self.energy_weights=energy_weights
        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), 5)

        #mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        states_angle = []
        for j in self.joint_list:
            states_angle.append(self.sim.data.get_joint_qpos(j))
        #states=self._get_obs()
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]

        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        #next_states=observation
        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        reward = forward_reward - ctrl_cost
        done = False

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(action[i]) * delta_theta

        '''delta_theta_bt = np.abs(next_states[2] - states[2])
        delta_theta_bs = np.abs(next_states[3] - states[3])
        delta_theta_bf = np.abs(next_states[4] - states[4])
        delta_theta_ft = np.abs(next_states[5] - states[5])
        delta_theta_fs = np.abs(next_states[6] - states[6])
        delta_theta_ff = np.abs(next_states[7] - states[7])

        energy_bt = np.abs(action[0]) * delta_theta_bt
        energy_bs = np.abs(action[1]) * delta_theta_bs
        energy_bf = np.abs(action[2]) * delta_theta_bf
        energy_ft = np.abs(action[3]) * delta_theta_ft
        energy_fs = np.abs(action[4]) * delta_theta_fs
        energy_ff = np.abs(action[5]) * delta_theta_ff

        energy = energy_bt + energy_bs + energy_bf + energy_ft + energy_fs + energy_ff'''

        reward -= self.energy_weights*energy
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            'energy'    : energy,
            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'ori_reward':forward_reward-ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

class HalfCheetahEnv_4dofv6(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='half_cheetah_4dofv6.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 energy_weights=0.):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight
        #self.joint_list = ['bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot']
        self.joint_list=['bthigh','bshin','fshin','ffoot']

        self._ctrl_cost_weight = ctrl_cost_weight
        self.energy_weights=energy_weights
        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), 5)

        #mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        states_angle = []
        for j in self.joint_list:
            states_angle.append(self.sim.data.get_joint_qpos(j))
        #states=self._get_obs()
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]

        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        #next_states=observation
        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        reward = forward_reward - ctrl_cost
        done = False

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(action[i]) * delta_theta

        '''delta_theta_bt = np.abs(next_states[2] - states[2])
        delta_theta_bs = np.abs(next_states[3] - states[3])
        delta_theta_bf = np.abs(next_states[4] - states[4])
        delta_theta_ft = np.abs(next_states[5] - states[5])
        delta_theta_fs = np.abs(next_states[6] - states[6])
        delta_theta_ff = np.abs(next_states[7] - states[7])

        energy_bt = np.abs(action[0]) * delta_theta_bt
        energy_bs = np.abs(action[1]) * delta_theta_bs
        energy_bf = np.abs(action[2]) * delta_theta_bf
        energy_ft = np.abs(action[3]) * delta_theta_ft
        energy_fs = np.abs(action[4]) * delta_theta_fs
        energy_ff = np.abs(action[5]) * delta_theta_ff

        energy = energy_bt + energy_bs + energy_bf + energy_ft + energy_fs + energy_ff'''

        reward -= self.energy_weights*energy
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            'energy'    : energy,
            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'ori_reward':forward_reward-ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

class HalfCheetahEnv_3doff(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='half_cheetah_3dof_front.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 energy_weights=0.):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self.joint_list=['bthigh','fthigh','fshin']

        self._ctrl_cost_weight = ctrl_cost_weight
        self.energy_weights=energy_weights
        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), 5)

        #mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        states_angle = []
        for j in self.joint_list:
            states_angle.append(self.sim.data.get_joint_qpos(j))
        #states=self._get_obs()
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]

        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        #next_states=observation
        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        reward = forward_reward - ctrl_cost
        done = False

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(action[i]) * delta_theta

        '''delta_theta_bt = np.abs(next_states[2] - states[2])
        delta_theta_bs = np.abs(next_states[3] - states[3])
        delta_theta_bf = np.abs(next_states[4] - states[4])
        delta_theta_ft = np.abs(next_states[5] - states[5])
        delta_theta_fs = np.abs(next_states[6] - states[6])
        delta_theta_ff = np.abs(next_states[7] - states[7])

        energy_bt = np.abs(action[0]) * delta_theta_bt
        energy_bs = np.abs(action[1]) * delta_theta_bs
        energy_bf = np.abs(action[2]) * delta_theta_bf
        energy_ft = np.abs(action[3]) * delta_theta_ft
        energy_fs = np.abs(action[4]) * delta_theta_fs
        energy_ff = np.abs(action[5]) * delta_theta_ff

        energy = energy_bt + energy_bs + energy_bf + energy_ft + energy_fs + energy_ff'''

        reward -= self.energy_weights*energy
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            'energy'    : energy,
            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'ori_reward':forward_reward-ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

class HalfCheetahEnv_3dofb(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='half_cheetah_3dof_back.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 energy_weights=0.):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self.joint_list=['bthigh','bshin','fthigh']

        self._ctrl_cost_weight = ctrl_cost_weight
        self.energy_weights=energy_weights
        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), 5)

        #mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        states_angle = []
        for j in self.joint_list:
            states_angle.append(self.sim.data.get_joint_qpos(j))
        #states=self._get_obs()
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]

        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        #next_states=observation
        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        reward = forward_reward - ctrl_cost
        done = False

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(action[i]) * delta_theta

        '''delta_theta_bt = np.abs(next_states[2] - states[2])
        delta_theta_bs = np.abs(next_states[3] - states[3])
        delta_theta_bf = np.abs(next_states[4] - states[4])
        delta_theta_ft = np.abs(next_states[5] - states[5])
        delta_theta_fs = np.abs(next_states[6] - states[6])
        delta_theta_ff = np.abs(next_states[7] - states[7])

        energy_bt = np.abs(action[0]) * delta_theta_bt
        energy_bs = np.abs(action[1]) * delta_theta_bs
        energy_bf = np.abs(action[2]) * delta_theta_bf
        energy_ft = np.abs(action[3]) * delta_theta_ft
        energy_fs = np.abs(action[4]) * delta_theta_fs
        energy_ff = np.abs(action[5]) * delta_theta_ff

        energy = energy_bt + energy_bs + energy_bf + energy_ft + energy_fs + energy_ff'''

        reward -= self.energy_weights*energy
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            'energy'    : energy,
            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'ori_reward':forward_reward-ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

class HalfCheetahEnv_3dofv3(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='half_cheetah_3dofv3.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 energy_weights=0.):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight
        #self.joint_list = ['bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot']
        self.joint_list=['bshin','bfoot','fthigh']

        self._ctrl_cost_weight = ctrl_cost_weight
        self.energy_weights=energy_weights
        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), 5)

        #mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        states_angle = []
        for j in self.joint_list:
            states_angle.append(self.sim.data.get_joint_qpos(j))
        #states=self._get_obs()
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]

        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        #next_states=observation
        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        reward = forward_reward - ctrl_cost
        done = False

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(action[i]) * delta_theta

        '''delta_theta_bt = np.abs(next_states[2] - states[2])
        delta_theta_bs = np.abs(next_states[3] - states[3])
        delta_theta_bf = np.abs(next_states[4] - states[4])
        delta_theta_ft = np.abs(next_states[5] - states[5])
        delta_theta_fs = np.abs(next_states[6] - states[6])
        delta_theta_ff = np.abs(next_states[7] - states[7])

        energy_bt = np.abs(action[0]) * delta_theta_bt
        energy_bs = np.abs(action[1]) * delta_theta_bs
        energy_bf = np.abs(action[2]) * delta_theta_bf
        energy_ft = np.abs(action[3]) * delta_theta_ft
        energy_fs = np.abs(action[4]) * delta_theta_fs
        energy_ff = np.abs(action[5]) * delta_theta_ff

        energy = energy_bt + energy_bs + energy_bf + energy_ft + energy_fs + energy_ff'''

        reward -= self.energy_weights*energy
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            'energy'    : energy,
            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'ori_reward':forward_reward-ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

class HalfCheetahEnv_3dofv4(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='half_cheetah_3dofv4.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 energy_weights=0.):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight
        #self.joint_list = ['bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot']
        self.joint_list=['bthigh','fshin','ffoot']

        self._ctrl_cost_weight = ctrl_cost_weight
        self.energy_weights=energy_weights
        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), 5)

        #mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        states_angle = []
        for j in self.joint_list:
            states_angle.append(self.sim.data.get_joint_qpos(j))
        #states=self._get_obs()
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]

        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        #next_states=observation
        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        reward = forward_reward - ctrl_cost
        done = False

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(action[i]) * delta_theta

        '''delta_theta_bt = np.abs(next_states[2] - states[2])
        delta_theta_bs = np.abs(next_states[3] - states[3])
        delta_theta_bf = np.abs(next_states[4] - states[4])
        delta_theta_ft = np.abs(next_states[5] - states[5])
        delta_theta_fs = np.abs(next_states[6] - states[6])
        delta_theta_ff = np.abs(next_states[7] - states[7])

        energy_bt = np.abs(action[0]) * delta_theta_bt
        energy_bs = np.abs(action[1]) * delta_theta_bs
        energy_bf = np.abs(action[2]) * delta_theta_bf
        energy_ft = np.abs(action[3]) * delta_theta_ft
        energy_fs = np.abs(action[4]) * delta_theta_fs
        energy_ff = np.abs(action[5]) * delta_theta_ff

        energy = energy_bt + energy_bs + energy_bf + energy_ft + energy_fs + energy_ff'''

        reward -= self.energy_weights*energy
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            'energy'    : energy,
            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'ori_reward':forward_reward-ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

class HalfCheetahEnv_2dof(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='half_cheetah_2dof.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 energy_weights=0.):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self.joint_list=['bthigh','fthigh']

        self._ctrl_cost_weight = ctrl_cost_weight
        self.energy_weights=energy_weights
        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        states_angle = []
        for j in self.joint_list:
            states_angle.append(self.sim.data.get_joint_qpos(j))
        #states=self._get_obs()
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]

        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        #next_states=observation
        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        reward = forward_reward - ctrl_cost
        done = False

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(action[i]) * delta_theta

        '''delta_theta_bt = np.abs(next_states[2] - states[2])
        delta_theta_bs = np.abs(next_states[3] - states[3])
        delta_theta_bf = np.abs(next_states[4] - states[4])
        delta_theta_ft = np.abs(next_states[5] - states[5])
        delta_theta_fs = np.abs(next_states[6] - states[6])
        delta_theta_ff = np.abs(next_states[7] - states[7])

        energy_bt = np.abs(action[0]) * delta_theta_bt
        energy_bs = np.abs(action[1]) * delta_theta_bs
        energy_bf = np.abs(action[2]) * delta_theta_bf
        energy_ft = np.abs(action[3]) * delta_theta_ft
        energy_fs = np.abs(action[4]) * delta_theta_fs
        energy_ff = np.abs(action[5]) * delta_theta_ff

        energy = energy_bt + energy_bs + energy_bf + energy_ft + energy_fs + energy_ff'''

        reward -= self.energy_weights*energy
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            'energy'    : energy,
            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'ori_reward':forward_reward-ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

class HalfCheetahEnv_2dofv2(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='half_cheetah_2dofv2.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 energy_weights=0.):

        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self.joint_list=['bfoot','ffoot']

        self._ctrl_cost_weight = ctrl_cost_weight
        self.energy_weights=energy_weights
        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)
        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), 5)
        #mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        states_angle = []
        for j in self.joint_list:
            states_angle.append(self.sim.data.get_joint_qpos(j))
        #states=self._get_obs()
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]

        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        #next_states=observation
        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        reward = forward_reward - ctrl_cost
        done = False

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(action[i]) * delta_theta

        '''delta_theta_bt = np.abs(next_states[2] - states[2])
        delta_theta_bs = np.abs(next_states[3] - states[3])
        delta_theta_bf = np.abs(next_states[4] - states[4])
        delta_theta_ft = np.abs(next_states[5] - states[5])
        delta_theta_fs = np.abs(next_states[6] - states[6])
        delta_theta_ff = np.abs(next_states[7] - states[7])

        energy_bt = np.abs(action[0]) * delta_theta_bt
        energy_bs = np.abs(action[1]) * delta_theta_bs
        energy_bf = np.abs(action[2]) * delta_theta_bf
        energy_ft = np.abs(action[3]) * delta_theta_ft
        energy_fs = np.abs(action[4]) * delta_theta_fs
        energy_ff = np.abs(action[5]) * delta_theta_ff

        energy = energy_bt + energy_bs + energy_bf + energy_ft + energy_fs + energy_ff'''

        reward -= self.energy_weights*energy
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            'energy'    : energy,
            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'ori_reward':forward_reward-ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

class HalfCheetahEnv_2dofv3(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='half_cheetah_2dofv3.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 energy_weights=0.):

        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self.joint_list=['bshin','fshin']

        self._ctrl_cost_weight = ctrl_cost_weight
        self.energy_weights=energy_weights
        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)
        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), 5)
        #mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        states_angle = []
        for j in self.joint_list:
            states_angle.append(self.sim.data.get_joint_qpos(j))
        #states=self._get_obs()
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]

        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        #next_states=observation
        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        reward = forward_reward - ctrl_cost
        done = False

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(action[i]) * delta_theta

        '''delta_theta_bt = np.abs(next_states[2] - states[2])
        delta_theta_bs = np.abs(next_states[3] - states[3])
        delta_theta_bf = np.abs(next_states[4] - states[4])
        delta_theta_ft = np.abs(next_states[5] - states[5])
        delta_theta_fs = np.abs(next_states[6] - states[6])
        delta_theta_ff = np.abs(next_states[7] - states[7])

        energy_bt = np.abs(action[0]) * delta_theta_bt
        energy_bs = np.abs(action[1]) * delta_theta_bs
        energy_bf = np.abs(action[2]) * delta_theta_bf
        energy_ft = np.abs(action[3]) * delta_theta_ft
        energy_fs = np.abs(action[4]) * delta_theta_fs
        energy_ff = np.abs(action[5]) * delta_theta_ff

        energy = energy_bt + energy_bs + energy_bf + energy_ft + energy_fs + energy_ff'''

        reward -= self.energy_weights*energy
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            'energy'    : energy,
            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'ori_reward':forward_reward-ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

class HalfCheetahEnv_2dofv4(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='half_cheetah_2dofv4.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 energy_weights=0.):

        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self.joint_list=['bshin','fthigh']

        self._ctrl_cost_weight = ctrl_cost_weight
        self.energy_weights=energy_weights
        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)
        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), 5)
        #mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        states_angle = []
        for j in self.joint_list:
            states_angle.append(self.sim.data.get_joint_qpos(j))
        #states=self._get_obs()
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]

        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        #next_states=observation
        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        reward = forward_reward - ctrl_cost
        done = False

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(action[i]) * delta_theta

        '''delta_theta_bt = np.abs(next_states[2] - states[2])
        delta_theta_bs = np.abs(next_states[3] - states[3])
        delta_theta_bf = np.abs(next_states[4] - states[4])
        delta_theta_ft = np.abs(next_states[5] - states[5])
        delta_theta_fs = np.abs(next_states[6] - states[6])
        delta_theta_ff = np.abs(next_states[7] - states[7])

        energy_bt = np.abs(action[0]) * delta_theta_bt
        energy_bs = np.abs(action[1]) * delta_theta_bs
        energy_bf = np.abs(action[2]) * delta_theta_bf
        energy_ft = np.abs(action[3]) * delta_theta_ft
        energy_fs = np.abs(action[4]) * delta_theta_fs
        energy_ff = np.abs(action[5]) * delta_theta_ff

        energy = energy_bt + energy_bs + energy_bf + energy_ft + energy_fs + energy_ff'''

        reward -= self.energy_weights*energy
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            'energy'    : energy,
            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'ori_reward':forward_reward-ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

class HalfCheetahEnv_2dofv5(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='half_cheetah_2dofv5.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 energy_weights=0.):

        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self.joint_list=['bthigh','fshin']

        self._ctrl_cost_weight = ctrl_cost_weight
        self.energy_weights=energy_weights
        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)
        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), 5)
        #mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        states_angle = []
        for j in self.joint_list:
            states_angle.append(self.sim.data.get_joint_qpos(j))
        #states=self._get_obs()
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]

        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        #next_states=observation
        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        reward = forward_reward - ctrl_cost
        done = False

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(action[i]) * delta_theta

        '''delta_theta_bt = np.abs(next_states[2] - states[2])
        delta_theta_bs = np.abs(next_states[3] - states[3])
        delta_theta_bf = np.abs(next_states[4] - states[4])
        delta_theta_ft = np.abs(next_states[5] - states[5])
        delta_theta_fs = np.abs(next_states[6] - states[6])
        delta_theta_ff = np.abs(next_states[7] - states[7])

        energy_bt = np.abs(action[0]) * delta_theta_bt
        energy_bs = np.abs(action[1]) * delta_theta_bs
        energy_bf = np.abs(action[2]) * delta_theta_bf
        energy_ft = np.abs(action[3]) * delta_theta_ft
        energy_fs = np.abs(action[4]) * delta_theta_fs
        energy_ff = np.abs(action[5]) * delta_theta_ff

        energy = energy_bt + energy_bs + energy_bf + energy_ft + energy_fs + energy_ff'''

        reward -= self.energy_weights*energy
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            'energy'    : energy,
            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'ori_reward':forward_reward-ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)



class HalfCheetahSquat2dof(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='half_cheetah_2dofsquat.xml',
                 distance_weigth=5.0,
                 health_weight=1,
                 horizontal_weight=0.1,  ##

                 ctrl_cost_weight=0.,
                 reset_noise_scale=0.1,
                 healthy_z_range=(0.2, 1),
                 terminate_when_unhealthy=False,  ##

                 exclude_current_positions_from_observation=True,
                 energy_weights=0.):
        utils.EzPickle.__init__(**locals())
        if terminate_when_unhealthy:
            healthy_reward=1.0
        else:
            healthy_reward=0
        self._ctrl_cost_weight = ctrl_cost_weight
        self.energy_weights=energy_weights
        self._healthy_reward = healthy_reward
        self._health_weight=health_weight
        self._distance_weigth = distance_weigth
        self._horizontal_weight=horizontal_weight
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self.target_low=0.25#0.45
        self.target_high=0.7#0.6
        self.joint_list=['bshin','fshin']

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        self.target_pos = np.asarray([0, 0, 0])
        self.target_site=0
        self.flipstep=75
        self.timestep=0

        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def is_healthy(self):
        # state = self.state_vector()
        # state = self.get_body_com("torso")
        state = self.sim.data.get_geom_xpos('head')
        min_z, max_z = self._healthy_z_range

        is_healthy = (np.isfinite(state).all() and min_z <= state[2] <= max_z)
        return is_healthy

    @property
    def healthy_reward(self):
        return float(
            self.is_healthy
            or self._terminate_when_unhealthy
        ) * self._healthy_reward

    @property
    def done(self):
        done = (not self.is_healthy
                if self._terminate_when_unhealthy
                else False)
        return done

    def step(self, action):
        states_angle = []
        for j in self.joint_list:
            states_angle.append(self.sim.data.get_joint_qpos(j))
        #states=self._get_obs()
        #x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        self.timestep += 1

        #x_position_after = self.sim.data.qpos[0]

        #x_velocity = ((x_position_after - x_position_before)
        #              / self.dt)

        ctrl_cost = np.sum(np.square(action))#self.control_cost(action)


        healthy_reward = self.healthy_reward

        vec = self.get_body_com("torso")[2] - self.target_pos[2]
        reward_dist = np.linalg.norm(vec)

        horizontal_penalty=np.abs(sum([0, 0, 1] - self.sim.data.geom_xmat[1][6:9]))

        #forward_reward =  x_velocity#self._forward_reward_weight *

        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        #reward = forward_reward - ctrl_cost
        ori_reward = - self._distance_weigth * reward_dist \
                     - self._ctrl_cost_weight * ctrl_cost \
                     - self._horizontal_weight * horizontal_penalty \
                     + self._health_weight * healthy_reward


        done = self.done

        if self.timestep % self.flipstep == 0 and self.timestep != 0:  # 2
            if self.target_pos[2] <= self.target_low:
                self.target_pos[2] = self.target_high
            elif self.target_pos[2] >= self.target_high:
                self.target_pos[2] = self.target_low

            self.target_pos[0] = self.get_body_com("torso")[0]
            self.target_pos[1] = self.get_body_com("torso")[1]

        self.sim.data.site_xpos[self.target_site]= self.target_pos

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            # energy = energy + np.abs(action[i]) * delta_theta
            energy = energy + np.sum(np.square(delta_theta))

        if not self._terminate_when_unhealthy:
            unhealty_penalty = -5
            if not self.is_healthy:
                ori_reward += unhealty_penalty

        final_reward =ori_reward- self.energy_weights*energy

        observation = self._get_obs()

        info = {
            'energy'    : energy,
            'reward_dist': -reward_dist,
            'reward_ctrl': -ctrl_cost,
            'horizontal_penalty': -horizontal_penalty,
            'reward_survive': healthy_reward,
            'ori_reward': ori_reward,
        }

        return observation, final_reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        body_orientation=self.sim.data.geom_xmat[1][6:9].flat.copy()
        #distance=list(self.get_body_com("torso")[2] - self.target_pos[2] )
        distance = [self.get_body_com("torso")[2] - self.target_pos[2]]
        if self._exclude_current_positions_from_observation:
            position = position[1:]

        #observation = np.concatenate((position, velocity)).ravel()

        if  self._horizontal_weight != 0:
            observation = np.concatenate(
                (position, velocity,body_orientation, distance)).ravel()
        else:
            observation = np.concatenate(
                (position, velocity, distance)).ravel()

        return observation

    def reset_model(self):
        self.timestep=0

        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)
        self.target_pos = np.asarray([self.get_body_com("torso")[0], self.get_body_com("torso")[1], self.target_low])

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        '''for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)'''

        self.viewer.cam.trackbodyid = 0  # id of the body to track ()
        self.viewer.cam.distance = self.model.stat.extent * 1.0  # how much you "zoom in", model.stat.extent is the max limits of the arena
        self.viewer.cam.lookat[0] += 0.5  # x,y,z offset from the object (works if trackbodyid=-1)
        self.viewer.cam.lookat[1] += 0.5
        self.viewer.cam.lookat[2] += 0
        self.viewer.cam.elevation = 0  # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
        self.viewer.cam.azimuth = 90  # camera rotation around the camera's vertical axis

class HalfCheetahSquat4dof(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='half_cheetah_4dofsquat.xml',
                 distance_weigth=5.0,
                 health_weight=1,
                 horizontal_weight=0.1,  ##

                 ctrl_cost_weight=0.,
                 reset_noise_scale=0.1,
                 healthy_z_range=(0.2, 1),
                 terminate_when_unhealthy=False,  ##

                 exclude_current_positions_from_observation=True,
                 energy_weights=0.):
        utils.EzPickle.__init__(**locals())
        if terminate_when_unhealthy:
            healthy_reward=1.0
        else:
            healthy_reward=0
        self._ctrl_cost_weight = ctrl_cost_weight
        self.energy_weights=energy_weights
        self._healthy_reward = healthy_reward
        self._health_weight=health_weight
        self._distance_weigth = distance_weigth
        self._horizontal_weight=horizontal_weight
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self.target_low=0.25
        self.target_high=0.7#0.6
        self.joint_list=['bthigh','bshin','fthigh','fshin']

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        self.target_pos = np.asarray([0, 0, 0])
        self.target_site=0
        self.flipstep=75
        self.timestep=0

        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def is_healthy(self):
        # state = self.state_vector()
        #state = self.get_body_com("torso")
        state=self.sim.data.get_geom_xpos('head')
        min_z, max_z = self._healthy_z_range

        is_healthy = (np.isfinite(state).all() and min_z <= state[2] <= max_z)
        return is_healthy

    @property
    def healthy_reward(self):
        return float(
            self.is_healthy
            or self._terminate_when_unhealthy
        ) * self._healthy_reward

    @property
    def done(self):
        done = (not self.is_healthy
                if self._terminate_when_unhealthy
                else False)
        return done

    def step(self, action):
        states_angle = []
        for j in self.joint_list:
            states_angle.append(self.sim.data.get_joint_qpos(j))
        #states=self._get_obs()
        #x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        self.timestep += 1

        #x_position_after = self.sim.data.qpos[0]

        #x_velocity = ((x_position_after - x_position_before)
        #              / self.dt)

        ctrl_cost = np.sum(np.square(action))#self.control_cost(action)



        healthy_reward = self.healthy_reward

        vec = self.get_body_com("torso")[2] - self.target_pos[2]
        reward_dist = np.linalg.norm(vec)

        horizontal_penalty=np.abs(sum([0, 0, 1] - self.sim.data.geom_xmat[1][6:9]))
        #print(horizontal_penalty)

        #forward_reward =  x_velocity#self._forward_reward_weight *

        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        #reward = forward_reward - ctrl_cost
        ori_reward = - self._distance_weigth * reward_dist \
                     - self._ctrl_cost_weight * ctrl_cost \
                     - self._horizontal_weight * horizontal_penalty \
                     + self._health_weight * healthy_reward


        done = self.done

        if self.timestep % self.flipstep == 0 and self.timestep != 0:  # 2
            if self.target_pos[2] <= self.target_low:
                self.target_pos[2] = self.target_high
            elif self.target_pos[2] >= self.target_high:
                self.target_pos[2] = self.target_low

            self.target_pos[0] = self.get_body_com("torso")[0]
            self.target_pos[1] = self.get_body_com("torso")[1]

        self.sim.data.site_xpos[self.target_site]= self.target_pos

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            # energy = energy + np.abs(action[i]) * delta_theta
            energy = energy + np.sum(np.square(delta_theta))

        if not self._terminate_when_unhealthy:
            unhealty_penalty = -5
            if not self.is_healthy:
                ori_reward += unhealty_penalty

        final_reward =ori_reward- self.energy_weights*energy

        observation = self._get_obs()

        info = {
            'energy'    : energy,
            'reward_dist': -reward_dist,
            'reward_ctrl': -ctrl_cost,
            'horizontal_penalty': -horizontal_penalty,
            'reward_survive': healthy_reward,
            'ori_reward': ori_reward,
        }

        return observation, final_reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        body_orientation=self.sim.data.geom_xmat[1][6:9].flat.copy()
        #distance=list(self.get_body_com("torso")[2] - self.target_pos[2] )
        distance = [self.get_body_com("torso")[2] - self.target_pos[2]]
        if self._exclude_current_positions_from_observation:
            position = position[1:]

        #observation = np.concatenate((position, velocity)).ravel()

        if  self._horizontal_weight != 0:
            observation = np.concatenate(
                (position, velocity,body_orientation, distance)).ravel()
        else:
            observation = np.concatenate(
                (position, velocity, distance)).ravel()

        return observation

    def reset_model(self):
        self.timestep=0

        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)
        self.target_pos = np.asarray([self.get_body_com("torso")[0], self.get_body_com("torso")[1], self.target_low])

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        '''for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)'''

        self.viewer.cam.trackbodyid = 0  # id of the body to track ()
        self.viewer.cam.distance = self.model.stat.extent * 1.0  # how much you "zoom in", model.stat.extent is the max limits of the arena
        self.viewer.cam.lookat[0] += 0.5  # x,y,z offset from the object (works if trackbodyid=-1)
        self.viewer.cam.lookat[1] += 0.5
        self.viewer.cam.lookat[2] += 0
        self.viewer.cam.elevation = 0  # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
        self.viewer.cam.azimuth = 90  # camera rotation around the camera's vertical axis

class HalfCheetahSquat6dof(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='half_cheetah_squat.xml',
                 distance_weigth=5.0,
                 health_weight=1,
                 horizontal_weight=0.1,  ##

                 ctrl_cost_weight=0.,
                 reset_noise_scale=0.1,
                 healthy_z_range=(0.2, 1),
                 terminate_when_unhealthy=False,  ##

                 exclude_current_positions_from_observation=True,
                 energy_weights=0.):
        utils.EzPickle.__init__(**locals())
        if terminate_when_unhealthy:
            healthy_reward=1.0
        else:
            healthy_reward=0
        self._ctrl_cost_weight = ctrl_cost_weight
        self.energy_weights=energy_weights
        self._healthy_reward = healthy_reward
        self._health_weight=health_weight
        self._distance_weigth = distance_weigth
        self._horizontal_weight=horizontal_weight
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self.target_low=0.25
        self.target_high=0.7#0.65
        self.joint_list=['bthigh','bshin','bfoot','fthigh','fshin','ffoot']

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        self.target_pos = np.asarray([0, 0, 0])
        self.target_site=0
        self.flipstep=75
        self.timestep=0

        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def is_healthy(self):
        # state = self.state_vector()
        # state = self.get_body_com("torso")
        state = self.sim.data.get_geom_xpos('head')
        min_z, max_z = self._healthy_z_range

        is_healthy = (np.isfinite(state).all() and min_z <= state[2] <= max_z)
        return is_healthy

    @property
    def healthy_reward(self):
        return float(
            self.is_healthy
            or self._terminate_when_unhealthy
        ) * self._healthy_reward

    @property
    def done(self):
        done = (not self.is_healthy
                if self._terminate_when_unhealthy
                else False)
        return done

    def step(self, action):
        states_angle = []
        for j in self.joint_list:
            states_angle.append(self.sim.data.get_joint_qpos(j))
        #states=self._get_obs()
        #x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        self.timestep += 1

        #x_position_after = self.sim.data.qpos[0]

        #x_velocity = ((x_position_after - x_position_before)
        #              / self.dt)

        ctrl_cost = np.sum(np.square(action))#self.control_cost(action)


        healthy_reward = self.healthy_reward

        vec = self.get_body_com("torso")[2] - self.target_pos[2]
        reward_dist = np.linalg.norm(vec)

        horizontal_penalty=np.abs(sum([0, 0, 1] - self.sim.data.geom_xmat[1][6:9]))

        #forward_reward =  x_velocity#self._forward_reward_weight *

        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        #reward = forward_reward - ctrl_cost
        ori_reward = - self._distance_weigth * reward_dist \
                     - self._ctrl_cost_weight * ctrl_cost \
                     - self._horizontal_weight * horizontal_penalty \
                     + self._health_weight * healthy_reward


        done = self.done

        if self.timestep % self.flipstep == 0 and self.timestep != 0:  # 2
            if self.target_pos[2] <= self.target_low:
                self.target_pos[2] = self.target_high
            elif self.target_pos[2] >= self.target_high:
                self.target_pos[2] = self.target_low

            self.target_pos[0] = self.get_body_com("torso")[0]
            self.target_pos[1] = self.get_body_com("torso")[1]

        self.sim.data.site_xpos[self.target_site]= self.target_pos

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            # energy = energy + np.abs(action[i]) * delta_theta
            energy = energy + np.sum(np.square(delta_theta))

        if not self._terminate_when_unhealthy:
            unhealty_penalty = -5
            if not self.is_healthy:
                ori_reward += unhealty_penalty

        final_reward =ori_reward- self.energy_weights*energy

        observation = self._get_obs()

        info = {
            'energy'    : energy,
            'reward_dist': -reward_dist,
            'reward_ctrl': -ctrl_cost,
            'horizontal_penalty': -horizontal_penalty,
            'reward_survive': healthy_reward,
            'ori_reward': ori_reward,
        }

        return observation, final_reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        body_orientation=self.sim.data.geom_xmat[1][6:9].flat.copy()
        #distance=list(self.get_body_com("torso")[2] - self.target_pos[2] )
        distance = [self.get_body_com("torso")[2] - self.target_pos[2]]
        if self._exclude_current_positions_from_observation:
            position = position[1:]

        #observation = np.concatenate((position, velocity)).ravel()

        if  self._horizontal_weight != 0:
            observation = np.concatenate(
                (position, velocity,body_orientation, distance)).ravel()
        else:
            observation = np.concatenate(
                (position, velocity, distance)).ravel()

        return observation

    def reset_model(self):
        self.timestep=0

        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)
        self.target_pos = np.asarray([self.get_body_com("torso")[0], self.get_body_com("torso")[1], self.target_low])

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        '''for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)'''

        self.viewer.cam.trackbodyid = 0  # id of the body to track ()
        self.viewer.cam.distance = self.model.stat.extent * 1.0  # how much you "zoom in", model.stat.extent is the max limits of the arena
        self.viewer.cam.lookat[0] += 0.5  # x,y,z offset from the object (works if trackbodyid=-1)
        self.viewer.cam.lookat[1] += 0.5
        self.viewer.cam.lookat[2] += 0
        self.viewer.cam.elevation = 0  # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
        self.viewer.cam.azimuth = 90  # camera rotation around the camera's vertical axis




