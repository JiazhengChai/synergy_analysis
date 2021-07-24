import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from . import path
import os
DEFAULT_CAMERA_CONFIG = {
    'distance': 4.0,
}


class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='ant.xml',
                 ctrl_cost_weight=0.5,
                 contact_cost_weight=5e-4,
                 healthy_reward=1.0,
                 terminate_when_unhealthy=True,
                 healthy_z_range=(0.2, 1.0),
                 contact_force_range=(-1.0, 1.0),
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 energy_weights=0.):
        utils.EzPickle.__init__(**locals())

        self.joint_list=['hip_4','ankle_4','hip_1','ankle_1','hip_2','ankle_2','hip_3','ankle_3']

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self.energy_weights = energy_weights
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), 5)
        #mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    @property
    def healthy_reward(self):
        return float(
            self.is_healthy
            or self._terminate_when_unhealthy
        ) * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.sim.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces))
        return contact_cost

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = (np.isfinite(state).all() and min_z <= state[2] <= max_z)
        return is_healthy

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
        x_position_before = self.get_body_com("torso")[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.get_body_com("torso")[0]
        x_velocity = (x_position_after - x_position_before) / self.dt
        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        forward_reward = x_velocity
        healthy_reward = self.healthy_reward

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(action[i]) * delta_theta

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost + contact_cost
        ori_reward=rewards - costs
        reward = ori_reward-self.energy_weights*energy
        done = self.done
        observation = self._get_obs()

        info = {
            'reward_forward': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'reward_contact': -contact_cost,
            'reward_survive': healthy_reward,
            'ori_reward': ori_reward,
            'energy': energy,
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        contact_force = self.contact_forces.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        observations = np.concatenate((position, velocity, contact_force))

        return observations

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

    '''def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)'''

    def viewer_setup(self):

        self.viewer.cam.trackbodyid = 0  # id of the body to track ()
        self.viewer.cam.distance = self.model.stat.extent * 1.0  # how much you "zoom in", model.stat.extent is the max limits of the arena
        self.viewer.cam.lookat[0] += 0.5  # x,y,z offset from the object (works if trackbodyid=-1)
        self.viewer.cam.lookat[1] += 0.5
        self.viewer.cam.lookat[2] += 0.5
        self.viewer.cam.elevation = -90  # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
        self.viewer.cam.azimuth = 0  # camera rotation around the camera's vertical axis

class AntHeavyEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='ant_heavy.xml',
                 ctrl_cost_weight=0.5,
                 contact_cost_weight=5e-4,
                 healthy_reward=1.0,
                 terminate_when_unhealthy=True,
                 healthy_z_range=(0.2, 1.0),
                 contact_force_range=(-1.0, 1.0),
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 energy_weights=0.):
        utils.EzPickle.__init__(**locals())

        self.joint_list=['hip_4','ankle_4','hip_1','ankle_1','hip_2','ankle_2','hip_3','ankle_3']

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self.energy_weights = energy_weights
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), 5)
        #mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    @property
    def healthy_reward(self):
        return float(
            self.is_healthy
            or self._terminate_when_unhealthy
        ) * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.sim.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces))
        return contact_cost

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = (np.isfinite(state).all() and min_z <= state[2] <= max_z)
        return is_healthy

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
        x_position_before = self.get_body_com("torso")[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.get_body_com("torso")[0]
        x_velocity = (x_position_after - x_position_before) / self.dt
        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        forward_reward = x_velocity
        healthy_reward = self.healthy_reward

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(action[i]) * delta_theta

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost + contact_cost
        ori_reward=rewards - costs
        reward = ori_reward-self.energy_weights*energy
        done = self.done
        observation = self._get_obs()

        info = {
            'reward_forward': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'reward_contact': -contact_cost,
            'reward_survive': healthy_reward,
            'ori_reward': ori_reward,
            'energy': energy,
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        contact_force = self.contact_forces.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        observations = np.concatenate((position, velocity, contact_force))

        return observations

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

    '''def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)'''

    def viewer_setup(self):

        self.viewer.cam.trackbodyid = 0  # id of the body to track ()
        self.viewer.cam.distance = self.model.stat.extent * 1.0  # how much you "zoom in", model.stat.extent is the max limits of the arena
        self.viewer.cam.lookat[0] += 0.5  # x,y,z offset from the object (works if trackbodyid=-1)
        self.viewer.cam.lookat[1] += 0.5
        self.viewer.cam.lookat[2] += 0.5
        self.viewer.cam.elevation = -90  # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
        self.viewer.cam.azimuth = 0  # camera rotation around the camera's vertical axis

class AntSquaTEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='ant_squatv2.xml',
                 ctrl_cost_weight=0.1,##0.1
                 horizontal_weight=0.1,##
                 contact_cost_weight=0,##5e-4
                 health_weight=1,
                 distance_weigth=5,
                 terminate_when_unhealthy=True,##
                 healthy_z_range=(0.26, 2),
                 contact_force_range=(-1.0, 1.0),
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,##
                 energy_weights=0.,
                 target_position_type='FWT',#FWT F Z
                 ):
        utils.EzPickle.__init__(**locals())

        if xml_file=='ant_squatv2.xml':
            self.joint_list=['hip_1','ankle_1','hip_2','ankle_2','hip_3','ankle_3','hip_4','ankle_4']
        elif xml_file=='ant_squat4.xml':
            self.joint_list=['ankle_1','ankle_2','ankle_3','ankle_4']
        if terminate_when_unhealthy:
            healthy_reward=1.0
        else:
            healthy_reward=0
        self.target_position_type=target_position_type
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self.energy_weights = energy_weights
        self._healthy_reward = healthy_reward
        self._health_weight=health_weight
        self._distance_weight=distance_weigth
        self._horizontal_weight=horizontal_weight
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        self.frame_skip=5
        self.target_pos = np.asarray([0, 0, 0])
        self.target_site=0
        self.timestep=0
        self.flipstep=75
        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), self.frame_skip)


    @property
    def healthy_reward(self):
        return float(
            self.is_healthy
            or self._terminate_when_unhealthy
        ) * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.sim.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        #contact_cost = self._contact_cost_weight * np.sum(
        #    np.square(self.contact_forces)) self._contact_cost_weight *
        contact_cost = np.sum(
            np.count_nonzero(self.contact_forces))
        return contact_cost

    @property
    def is_healthy(self):
        #state = self.state_vector()
        state=self.get_body_com("torso")
        min_z, max_z = self._healthy_z_range

        is_healthy = (np.isfinite(state).all() and min_z <= state[2] <= max_z)
        return is_healthy

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

        self.do_simulation(action, self.frame_skip)
        self.timestep += 1
        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        ctrl_cost = np.sum(np.square(action))# self.control_cost(action)

        #contact_cost =  np.sum(np.square(self.contact_forces))#self.contact_cost
        contact_cost=self.contact_cost

        healthy_reward = self.healthy_reward

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(action[i]) * delta_theta

        vec = self.get_body_com("torso") - self.target_pos

        reward_dist =  np.linalg.norm(vec)

        horizontal_penalty=np.abs(sum([0, 0, 1] - self.sim.data.geom_xmat[1][6:9]))

        ori_reward=- self._distance_weight*reward_dist \
                   - self._ctrl_cost_weight * ctrl_cost \
                   + self._contact_cost_weight * contact_cost \
                   - self._horizontal_weight * horizontal_penalty \
                   + self._health_weight*healthy_reward
        #print("distance")
        #print(reward_dist)
        #print("ctrl_cost")
        #print(ctrl_cost)
        #print("contact_cost")
        #print(contact_cost)
        #print("horizontal_penalty")
        #print(horizontal_penalty)
        #print("healthy_reward")
        #print(healthy_reward)
        #print(" ")


        if not self._terminate_when_unhealthy:
            unhealty_penalty=-5
            if not self.is_healthy:
                ori_reward+=unhealty_penalty

        reward = ori_reward-self.energy_weights*energy
        done = self.done

        if  self.timestep%self.flipstep==0 and  self.timestep!=0:#2
            if  self.target_pos[2]<=0.3:
                self.target_pos[2]=1.2
            elif  self.target_pos[2]>=1.2:
                self.target_pos[2]=0.3
            if self.target_position_type=="FWT":
                self.target_pos[0] = self.get_body_com("torso")[0]
                self.target_pos[1] = self.get_body_com("torso")[1]

        if self.target_position_type=="F":
            self.target_pos[0] = self.get_body_com("torso")[0]
            self.target_pos[1] = self.get_body_com("torso")[1]

        self.sim.data.site_xpos[self.target_site]= self.target_pos

        observation = self._get_obs()

        info = {
            'reward_distance': -reward_dist,
            'reward_ctrl': -ctrl_cost,
            'reward_contact': contact_cost,
            'horizontal_penalty': -horizontal_penalty,
            'reward_survive': healthy_reward,
            'ori_reward': ori_reward,
            'energy': energy,
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        contact_force = self.contact_forces.flat.copy()
        body_orientation=self.sim.data.geom_xmat[1][6:9].flat.copy()
        distance=list(self.get_body_com("torso") - self.target_pos)
        if self._exclude_current_positions_from_observation:
            position = position[2:]

        if self._contact_cost_weight!=0 and self._horizontal_weight!=0:
            observations = np.concatenate((position, velocity,body_orientation,contact_force, distance))
        elif self._contact_cost_weight!=0 and self._horizontal_weight==0:
            observations = np.concatenate((position, velocity,contact_force, distance))
        elif self._contact_cost_weight == 0 and self._horizontal_weight != 0:
            observations = np.concatenate(
                (position, velocity,body_orientation, distance))
        elif self._contact_cost_weight == 0 and self._horizontal_weight == 0:
            observations = np.concatenate(
                (position, velocity, distance))

        return observations

    def reset_model(self):
        self.timestep=0
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)
        self.set_state(qpos, qvel)
        if self.target_position_type=='F' or self.target_position_type=='FWT':
            self.target_pos = np.asarray([self.get_body_com("torso")[0], self.get_body_com("torso")[1], 1.2])
        else:
            self.target_pos = np.asarray([0,0, 1.2])

        observation = self._get_obs()

        return observation

    def viewer_setup(self):

        self.viewer.cam.trackbodyid = 0  # id of the body to track ()
        self.viewer.cam.distance = self.model.stat.extent * 1.0  # how much you "zoom in", model.stat.extent is the max limits of the arena
        self.viewer.cam.lookat[0] += 0.5  # x,y,z offset from the object (works if trackbodyid=-1)
        self.viewer.cam.lookat[1] += 0.5
        self.viewer.cam.lookat[2] += 0
        self.viewer.cam.elevation = 0  # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
        self.viewer.cam.azimuth = 0  # camera rotation around the camera's vertical axis

class AntSquaTRedundantEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='ant_squatv3.xml',
                 ctrl_cost_weight=0.1,##0.1
                 horizontal_weight=0.1,##
                 contact_cost_weight=0,##5e-4
                 health_weight=1,
                 distance_weigth=5,
                 terminate_when_unhealthy=True,##
                 healthy_z_range=(0.2, 2),
                 contact_force_range=(-1.0, 1.0),
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,##
                 energy_weights=0.,
                 target_position_type='FWT',#FWT F Z
                 ):
        utils.EzPickle.__init__(**locals())


        self.joint_list=['hip_1','ankle_1','hip_2','ankle_2','hip_3','ankle_3','hip_4','ankle_4']

        if terminate_when_unhealthy:
            healthy_reward=1.0
        else:
            healthy_reward=0
        self.target_position_type=target_position_type
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self.energy_weights = energy_weights
        self._healthy_reward = healthy_reward
        self._health_weight=health_weight
        self._distance_weight=distance_weigth
        self._horizontal_weight=horizontal_weight
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        self.frame_skip=5
        self.target_pos = np.asarray([0, 0, 0])
        self.targethigh=1.3
        self.target_site=0
        self.timestep = 0
        self.flipstep = 75
        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), self.frame_skip)

    @property
    def healthy_reward(self):
        return float(
            self.is_healthy
            or self._terminate_when_unhealthy
        ) * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.sim.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        # contact_cost = self._contact_cost_weight * np.sum(
        #    np.square(self.contact_forces)) self._contact_cost_weight *
        contact_cost =  np.sum(
            np.count_nonzero(self.contact_forces))
        return contact_cost

    @property
    def is_healthy(self):
        #state = self.state_vector()
        state=self.get_body_com("torso")
        min_z, max_z = self._healthy_z_range

        is_healthy = (np.isfinite(state).all() and min_z <= state[2] <= max_z)
        return is_healthy

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

        self.do_simulation(action, self.frame_skip)
        self.timestep+=1
        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        ctrl_cost = np.sum(np.square(action))# self.control_cost(action)

        contact_cost =  self.contact_cost

        healthy_reward = self.healthy_reward

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(action[i]) * delta_theta

        vec = self.get_body_com("torso") - self.target_pos

        reward_dist =  np.linalg.norm(vec)

        horizontal_penalty=np.abs(sum([0, 0, 1] - self.sim.data.geom_xmat[1][6:9]))

        ori_reward=- self._distance_weight*reward_dist \
                   - self._ctrl_cost_weight * ctrl_cost \
                   + self._contact_cost_weight * contact_cost \
                   - self._horizontal_weight * horizontal_penalty \
                   + self._health_weight*healthy_reward

        if not self._terminate_when_unhealthy:
            unhealty_penalty=-5
            if not self.is_healthy:
                ori_reward+=unhealty_penalty



        reward = ori_reward-self.energy_weights*energy
        done = self.done

        if  self.timestep%self.flipstep==0 and  self.timestep!=0:#2
            if  self.target_pos[2]<=0.3:
                self.target_pos[2]=self.targethigh
            elif  self.target_pos[2]>=self.targethigh:
                self.target_pos[2]=0.3
            if self.target_position_type=="FWT":
                self.target_pos[0] = self.get_body_com("torso")[0]
                self.target_pos[1] = self.get_body_com("torso")[1]

        if self.target_position_type=="F":
            self.target_pos[0] = self.get_body_com("torso")[0]
            self.target_pos[1] = self.get_body_com("torso")[1]

        self.sim.data.site_xpos[self.target_site]= self.target_pos

        observation = self._get_obs()

        info = {
            'reward_distance': -reward_dist,
            'reward_ctrl': -ctrl_cost,
            'reward_contact': contact_cost,
            'horizontal_penalty': -horizontal_penalty,
            'reward_survive': healthy_reward,
            'ori_reward': ori_reward,
            'energy': energy,
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        contact_force = self.contact_forces.flat.copy()
        body_orientation=self.sim.data.geom_xmat[1][6:9].flat.copy()
        distance=list(self.get_body_com("torso") - self.target_pos)
        if self._exclude_current_positions_from_observation:
            position = position[2:]

        if self._contact_cost_weight!=0 and self._horizontal_weight!=0:
            observations = np.concatenate((position, velocity,body_orientation,contact_force, distance))
        elif self._contact_cost_weight!=0 and self._horizontal_weight==0:
            observations = np.concatenate((position, velocity,contact_force, distance))
        elif self._contact_cost_weight == 0 and self._horizontal_weight != 0:
            observations = np.concatenate(
                (position, velocity,body_orientation, distance))
        elif self._contact_cost_weight == 0 and self._horizontal_weight == 0:
            observations = np.concatenate(
                (position, velocity, distance))

        return observations

    def reset_model(self):
        self.timestep=0
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)
        self.set_state(qpos, qvel)
        if self.target_position_type=='F' or self.target_position_type=='FWT':
            self.target_pos = np.asarray([self.get_body_com("torso")[0], self.get_body_com("torso")[1], self.targethigh])
        else:
            self.target_pos = np.asarray([0,0, self.targethigh])

        observation = self._get_obs()

        return observation

    def viewer_setup(self):

        self.viewer.cam.trackbodyid = 0  # id of the body to track ()
        self.viewer.cam.distance = self.model.stat.extent * 1.0  # how much you "zoom in", model.stat.extent is the max limits of the arena
        self.viewer.cam.lookat[0] += 0.5  # x,y,z offset from the object (works if trackbodyid=-1)
        self.viewer.cam.lookat[1] += 0.5
        self.viewer.cam.lookat[2] += 0
        self.viewer.cam.elevation = 0  # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
        self.viewer.cam.azimuth = 0  # camera rotation around the camera's vertical axis

class AntRunEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='ant_run.xml',
                 ctrl_cost_weight=0.5,##0.1
                 contact_cost_weight=5e-4,##5e-4
                 health_weight=1,
                 speed_weigth=1,
                 terminate_when_unhealthy=False,##
                 healthy_z_range=(0.26, 1.5),
                 contact_force_range=(-1.0, 1.0),
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,##
                 energy_weights=0.,
                 ):
        utils.EzPickle.__init__(**locals())

        self.joint_list=['hip_1','ankle_1','hip_2','ankle_2','hip_3','ankle_3','hip_4','ankle_4']

        if terminate_when_unhealthy:
            healthy_reward=1.0
        else:
            healthy_reward=0
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self.energy_weights = energy_weights
        self._healthy_reward = healthy_reward
        self._health_weight=health_weight
        self._speed_weigth=speed_weigth

        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        self.frame_skip=5
        self.timestep=0

        global path
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), self.frame_skip)


    @property
    def healthy_reward(self):
        return float(
            self.is_healthy
            or self._terminate_when_unhealthy
        ) * self._healthy_reward

    def control_cost(self, action):
        control_cost =  np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.sim.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):

        contact_cost = np.sum(np.square(self.contact_forces))
        return contact_cost

    @property
    def is_healthy(self):
        state=self.get_body_com("torso")
        min_z, max_z = self._healthy_z_range

        is_healthy = (np.isfinite(state).all() and min_z <= state[2] <= max_z)
        return is_healthy

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
        x_position_before = self.get_body_com("torso")[0]

        self.do_simulation(action, self.frame_skip)
        x_position_after = self.get_body_com("torso")[0]
        x_velocity = (x_position_after - x_position_before) / self.dt
        forward_reward = x_velocity

        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        ctrl_cost = self.control_cost(action)

        contact_cost=self.contact_cost

        healthy_reward = self.healthy_reward

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(action[i]) * delta_theta


        ori_reward= self._speed_weigth*forward_reward \
                   - self._ctrl_cost_weight * ctrl_cost \
                   - self._contact_cost_weight * contact_cost \
                   + self._health_weight*healthy_reward
        #print("distance")
        #print(reward_dist)
        #print("ctrl_cost")
        #print(ctrl_cost)
        #print("contact_cost")
        #print(contact_cost)
        #print("horizontal_penalty")
        #print(horizontal_penalty)
        #print("healthy_reward")
        #print(healthy_reward)
        #print(" ")


        if not self._terminate_when_unhealthy:
            unhealty_penalty=-5
            if not self.is_healthy:
                ori_reward+=unhealty_penalty

        reward = ori_reward-self.energy_weights*energy
        done = self.done

        observation = self._get_obs()

        info = {
            'reward_forward': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'reward_contact': -contact_cost,
            'reward_survive': healthy_reward,
            'ori_reward': ori_reward,
            'energy': energy,
        }
        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        contact_force = self.contact_forces.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        observations = np.concatenate(
                (position, velocity, contact_force))

        return observations

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

        self.viewer.cam.trackbodyid = 1  # id of the body to track ()
        self.viewer.cam.distance = self.model.stat.extent * 3  # how much you "zoom in", model.stat.extent is the max limits of the arena
        #self.viewer.cam.lookat[0] += 0.5  # x,y,z offset from the object (works if trackbodyid=-1)
        #self.viewer.cam.lookat[1] += 0.5
        #self.viewer.cam.lookat[2] += 0.5
        self.viewer.cam.elevation = -20  # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
        self.viewer.cam.azimuth = 135  # camera rotation around the camera's vertical axis