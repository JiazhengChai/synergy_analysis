from ray import tune
import numpy as np

from softlearning.misc.utils import  deep_update

M = 256
N = 128#256
#N=45#human
REPARAMETERIZE = True

NUM_COUPLING_LAYERS = 2

GAUSSIAN_POLICY_PARAMS_BASE = {
    'type': 'GaussianPolicy',
    'kwargs': {
        'hidden_layer_sizes': (M, M),
        'squash': True,
    }
}

DETERMINISTICS_POLICY_PARAMS_BASE = {
    'type': 'DeterministicsPolicy',
    'kwargs': {
        'hidden_layer_sizes': (M, M),
        'squash': True
    }
}

GAUSSIAN_POLICY_PARAMS_FOR_DOMAIN = {}

DETERMINISTICS_POLICY_PARAMS_FOR_DOMAIN = {}

POLICY_PARAMS_BASE = {
    'GaussianPolicy': GAUSSIAN_POLICY_PARAMS_BASE,
    'DeterministicsPolicy': DETERMINISTICS_POLICY_PARAMS_BASE,
}

POLICY_PARAMS_BASE.update({
    'gaussian': POLICY_PARAMS_BASE['GaussianPolicy'],
    'deterministicsPolicy': POLICY_PARAMS_BASE['DeterministicsPolicy'],
})

POLICY_PARAMS_FOR_DOMAIN = {
    'GaussianPolicy': GAUSSIAN_POLICY_PARAMS_FOR_DOMAIN,

    'DeterministicsPolicy': DETERMINISTICS_POLICY_PARAMS_FOR_DOMAIN,
}

POLICY_PARAMS_FOR_DOMAIN.update({
    'gaussian': POLICY_PARAMS_FOR_DOMAIN['GaussianPolicy'],

    'deterministicsPolicy': POLICY_PARAMS_FOR_DOMAIN['DeterministicsPolicy'],

})

DEFAULT_MAX_PATH_LENGTH = 1000
MAX_PATH_LENGTH_PER_DOMAIN = {
    'Point2DEnv': 50,
    'Pendulum': 200,
}

ALGORITHM_PARAMS_BASE = {
    'type': 'SAC',

    'kwargs': {
        'epoch_length': 1000,
        'train_every_n_steps': 1,
        'n_train_repeat': 1,
        'eval_render_mode': None,
        'eval_n_episodes': 3,
        'eval_deterministic': True,

        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,
    }
}


ALGORITHM_PARAMS_ADDITIONAL = {
    'SAC': {
        'type': 'SAC',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            'store_extra_policy_info': False,
            'action_prior': 'uniform',
            'n_initial_exploration_steps': int(1e3),
        }
    },

    'TD3': {
        'type': 'TD3',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 1e-3,
            'target_update_interval': 2,
            'tau': 5e-3,
            'store_extra_policy_info': False,
            'action_prior': 'uniform',
            'n_initial_exploration_steps': int(1e4),
        }
    },

}

DEFAULT_NUM_EPOCHS = 200

NUM_EPOCHS_PER_DOMAIN = {
    'Swimmer': int(3e2),
    'Hopper': int(1e3),
    'HalfCheetah': int(3e3),
    'Giraffe': int(2e3),
    'HalfCheetahHeavy':int(3e3),
    'HalfCheetah5dof': int(3e3),
    'HalfCheetah5dofv2': int(3e3),
    'HalfCheetah5dofv3': int(3e3),
    'HalfCheetah5dofv4': int(3e3),
    'HalfCheetah5dofv5': int(3e3),
    'HalfCheetah5dofv6': int(3e3),
    'HalfCheetah4dof':int(3e3),
    'HalfCheetah4dofv2': int(3e3),
    'HalfCheetah4dofv3': int(3e3),
    'HalfCheetah4dofv4': int(3e3),
    'HalfCheetah4dofv5': int(3e3),
    'HalfCheetah4dofv6': int(3e3),
    'HalfCheetah2dof':int(3e3),
    'HalfCheetah2dofv2': int(3e3),
    'HalfCheetah2dofv3': int(3e3),
    'HalfCheetah2dofv4': int(3e3),
    'HalfCheetah2dofv5': int(3e3),
    'HalfCheetah3doff': int(3e3),
    'HalfCheetah3dofb': int(3e3),
    'HalfCheetah3dofv3': int(3e3),
    'HalfCheetah3dofv4': int(3e3),
    'HalfCheetahSquat2dof': int(60),
    'HalfCheetahSquat4dof': int(90),
    'HalfCheetahSquat6dof': int(120),
    'FullCheetah':int(3e3),
    'FullCheetahHeavy': int(3e3),
    'Centripede':int(2e3),
    'Walker2d': int(1e3),
    'Bipedal2d':int(300),
    'Ant': int(2e3),
    'AntSquaTRedundant': int(500),
    'AntSquaT': int(500),
    'AntRun': int(300),
    'VA': int(30),
    'VA4dof': int(30),
    'VA6dof': int(30),
    'VA8dof': int(100),
    'RealArm7dof':int(90),
    'RealArm6dof': int(90),
    'RealArm5dof': int(60),
    'RealArm4dof': int(60),
    'RealArm5dofLT': int(60),
    'RealArm4dofLT': int(60),
    'RealArm5dofMinE': int(60),
    'RealArm4dofMinE': int(60),
    'RealArm3dof': int(30),
    'AntHeavy': int(2e3),
    'Humanoid': int(5e3),#int(1e4),
    'Humanoidrllab': int(3e3),#int(1e4),
    'Pusher2d': int(2e3),
    'HandManipulatePen': int(1e4),
    'HandManipulateEgg': int(1e4),
    'HandManipulateBlock': int(1e4),
    'HandReach': int(1e4),
    'Point2DEnv': int(200),
    'Reacher': int(200),
    'Pendulum': 10,
    'VMP': 50,
}

ALGORITHM_PARAMS_PER_DOMAIN = {
    **{
        domain: {
            'kwargs': {
                'n_epochs': NUM_EPOCHS_PER_DOMAIN.get(
                    domain, DEFAULT_NUM_EPOCHS),
                'n_initial_exploration_steps': (
                    MAX_PATH_LENGTH_PER_DOMAIN.get(
                        domain, DEFAULT_MAX_PATH_LENGTH
                    ) * 10),
            }
        } for domain in NUM_EPOCHS_PER_DOMAIN
    }
}

ENV_PARAMS = {
    'Bipedal2d': {  # 6 DoF
        'Energy0-v0': {
            'target_energy':3
        },

    },
    'VA': {  # 6 DoF
        'Energyz-v0': {
            'distance_reward_weight': 5.0,
            'ctrl_cost_weight': 0.0,
        },
        'Energy0-v0': {
            'distance_reward_weight':5.0,
            'ctrl_cost_weight':0.05,
        },
        'EnergyPoint5-v0': {
            'distance_reward_weight': 5.0,
            'ctrl_cost_weight': 0.5,
        },
        'EnergyOne-v0': {
            'distance_reward_weight': 5.0,
            'ctrl_cost_weight':1,
        },

        'smallRange-v0': {
            'distance_reward_weight': 5.0,
            'ctrl_cost_weight': 0.05,
            'xml_file':'vertical_arm_smallRange.xml'
        },

        'bigRange-v0': {
            'distance_reward_weight': 5.0,
            'ctrl_cost_weight': 0.05,
            'xml_file':'vertical_arm_bigRange.xml'
        },
    },
    'VA4dof': {  # 6 DoF
        'Energyz-v0': {
            'distance_reward_weight': 5.0,
            'ctrl_cost_weight': 0.0,
        },
        'Energy0-v0': {
            'distance_reward_weight': 5.0,
            'ctrl_cost_weight': 0.05,
        },
        'EnergyPoint5-v0': {
            'distance_reward_weight': 5.0,
            'ctrl_cost_weight': 0.5,
        },
        'EnergyOne-v0': {
            'distance_reward_weight': 5.0,
            'ctrl_cost_weight': 1,
        },
        'smallRange-v0': {
            'distance_reward_weight': 5.0,
            'ctrl_cost_weight': 0.05,
            'xml_file': 'vertical_arm4dof_smallRange.xml'
        },

        'bigRange-v0': {
            'distance_reward_weight': 5.0,
            'ctrl_cost_weight': 0.05,
            'xml_file': 'vertical_arm4dof_bigRange.xml'
        },
    },
    'VA6dof': {  # 6 DoF
        'Energyz-v0': {
            'distance_reward_weight': 5.0,
            'ctrl_cost_weight': 0.0,
        },
        'Energy0-v0': {
            'distance_reward_weight': 5.0,
            'ctrl_cost_weight': 0.05,
        },
        'EnergyPoint5-v0': {
            'distance_reward_weight': 5.0,
            'ctrl_cost_weight': 0.5,
        },
        'EnergyOne-v0': {
            'distance_reward_weight': 5.0,
            'ctrl_cost_weight': 1,
        },
        'smallRange-v0': {
            'distance_reward_weight': 5.0,
            'ctrl_cost_weight': 0.05,
            'xml_file': 'vertical_arm6dof_smallRange.xml'
        },

        'bigRange-v0': {
            'distance_reward_weight': 5.0,
            'ctrl_cost_weight': 0.05,
            'xml_file': 'vertical_arm6dof_bigRange.xml'
        },
    },
    'VA8dof': {  # 6 DoF
        'Energyz-v0': {
            'distance_reward_weight': 5.0,
            'ctrl_cost_weight': 0.0,
        },
        'Energy0-v0': {
            'distance_reward_weight': 5.0,
            'ctrl_cost_weight': 0.05,#0.05
        },
        'EnergyPoint5-v0': {
            'distance_reward_weight': 5.0,
            'ctrl_cost_weight': 0.5,
        },
        'EnergyOne-v0': {
            'distance_reward_weight': 5.0,
            'ctrl_cost_weight': 1,
        },
        'smallRange-v0': {
            'distance_reward_weight': 5.0,
            'ctrl_cost_weight': 0.05,
            'xml_file': 'vertical_arm8dof_smallRange.xml'
        },

        'bigRange-v0': {
            'distance_reward_weight': 5.0,
            'ctrl_cost_weight': 0.05,
            'xml_file': 'vertical_arm8dof_bigRange.xml'
        },
    },
    'RealArm7dof': {  # 6 DoF
        'Energy0-v0': {
            'distance_reward_weight':5,
            'shoulder_cost_weight':1,
            'wrist_cost_weight':0
        },
        'Energy0-v1': {
            'distance_reward_weight': 5,
            'shoulder_cost_weight': 1,
            'wrist_cost_weight': 0,
            'pcx':0.01
        },
        'Energy0-v2': {
            'distance_reward_weight': 5,
            'shoulder_cost_weight': 1,
            'wrist_cost_weight': 0,
            'pcx': 0.1
        },
        'Energy0-v3': {
            'distance_reward_weight': 5,
            'shoulder_cost_weight': 1,
            'wrist_cost_weight': 0,
            'pcx': 0.2
        },
        'Energy0-v9': {
            'distance_reward_weight': 5,
            'shoulder_cost_weight': 1,
            'wrist_cost_weight': 0,
            'pcx': 0.15
        },

        'Energy0-v4': {
            'xml_file': 'real_arm7dofLessTorque.xml',
            'distance_reward_weight': 5,
            'shoulder_cost_weight': 1,
            'wrist_cost_weight': 0
        },
        'Energy0-v5': {
            'xml_file': 'real_arm7dofMinTorque.xml',
            'distance_reward_weight': 5,
            'shoulder_cost_weight': 1,
            'wrist_cost_weight': 0
        },
        'Energy0-v6': {
            'xml_file': 'real_arm7dofMoreWeight.xml',
            'distance_reward_weight': 5,
            'shoulder_cost_weight': 1,
            'wrist_cost_weight': 0
        },
        'Energy0-v7': {
            'xml_file': 'real_arm7dof1p5Weight.xml',
            'distance_reward_weight': 5,
            'shoulder_cost_weight': 1,
            'wrist_cost_weight': 0
        },
        'Energy0-v8': {
            'xml_file': 'real_arm7dof2p5Weight.xml',
            'distance_reward_weight': 5,
            'shoulder_cost_weight': 1,
            'wrist_cost_weight': 0
        },

    },

    'RealArm6dof': {  # 6 DoF
        'Energy0-v0': {
            'distance_reward_weight': 5,
            'shoulder_cost_weight': 1,
            'wrist_cost_weight': 0
        },
        'Energy0-v1': {
            'distance_reward_weight': 5,
            'shoulder_cost_weight': 1,
            'wrist_cost_weight': 0,
            'pcx': 0.01
        },
        'Energy0-v2': {
            'distance_reward_weight': 5,
            'shoulder_cost_weight': 1,
            'wrist_cost_weight': 0,
            'pcx': 0.1
        },
    },
    'RealArm5dof': {  # 6 DoF
        'Energy0-v0': {
            'distance_reward_weight': 5,
            'shoulder_cost_weight': 1,
            'wrist_cost_weight': 0
        },
        'Energy0-v1': {
            'distance_reward_weight': 5,
            'shoulder_cost_weight': 1,
            'wrist_cost_weight': 0,
            'pcx': 0.01
        },
        'Energy0-v2': {
            'distance_reward_weight': 5,
            'shoulder_cost_weight': 1,
            'wrist_cost_weight': 0,
            'pcx': 0.1
        },
    },
    'RealArm5dofMinE': {  # 6 DoF
        'Energy0-v0': {
            'distance_reward_weight': 5,
            'shoulder_cost_weight': 1,
            'wrist_cost_weight': 0,
            'MinE_cost_weight': 0.5,
        },
    },
    'RealArm4dofMinE': {  # 6 DoF
        'Energy0-v0': {
            'distance_reward_weight': 5,
            'shoulder_cost_weight': 1,
            'wrist_cost_weight': 0,
            'MinE_cost_weight': 0.5,
        },
    },
    'RealArm5dofLT': {  # 6 DoF
        'Energy0-v0': {
            'xml_file': 'real_arm5dofLessTorque.xml',
            'distance_reward_weight': 5,
            'shoulder_cost_weight': 1,
            'wrist_cost_weight': 0
        },
    },
    'RealArm4dofLT': {  # 6 DoF
        'Energy0-v0': {
            'xml_file': 'real_arm4dofLessTorque.xml',
            'distance_reward_weight': 5,
            'shoulder_cost_weight': 1,
            'wrist_cost_weight': 0
        },
    },
    'RealArm4dof': {  # 6 DoF
        'Energy0-v0': {
            'distance_reward_weight': 5,
            'shoulder_cost_weight': 1,
            'wrist_cost_weight': 0
        },
        'Energy0-v1': {
            'distance_reward_weight': 5,
            'shoulder_cost_weight': 1,
            'wrist_cost_weight': 0,
            'pcx': 0.01
        },
        'Energy0-v2': {
            'distance_reward_weight': 5,
            'shoulder_cost_weight': 1,
            'wrist_cost_weight': 0,
            'pcx': 0.1
        },
    },
    'RealArm3dof': {  # 6 DoF
        'Energy0-v0': {
            'distance_reward_weight': 5,
            'shoulder_cost_weight': 1,
            'wrist_cost_weight': 0
        },
        'Energy0-v1': {
            'distance_reward_weight': 5,
            'shoulder_cost_weight': 1,
            'wrist_cost_weight': 0,
            'pcx': 0.01
        },
        'Energy0-v2': {
            'distance_reward_weight': 5,
            'shoulder_cost_weight': 1,
            'wrist_cost_weight': 0,
            'pcx': 0.1
        },
    },
    'Walker2d': {  # 6 DoF
        'Symloss-v0': {
        },
        'Symdup-v0': {
        },
    },
    'HalfCheetahHeavy': {  # 6 DoF
        'Energy0-v0': {
            'forward_reward_weight':1.0,
            'ctrl_cost_weight':0.1,
            'energy_weights':0,
        },

    },

    'HalfCheetah5dof': {  # 6 DoF
        'Energy0-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
        },
        'Energyz-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0,
            'energy_weights': 0,
        },
        'EnergyOne-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 1.,
        },

    },
    'HalfCheetah5dofv2': {  # 6 DoF
        'Energy0-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
        },
    },
    'HalfCheetah5dofv3': {  # 6 DoF
        'Energy0-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
        },
    },
    'HalfCheetah5dofv4': {  # 6 DoF
        'Energy0-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
        },
    },
    'HalfCheetah5dofv5': {  # 6 DoF
        'Energy0-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
        },
    },
    'HalfCheetah5dofv6': {  # 6 DoF
        'Energy0-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
        },
    },
    'HalfCheetah4dof': {  # 6 DoF
        'Energy0-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
        },
        'Energyz-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0,
            'energy_weights': 0,
        },
        'EnergyOne-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 1.,
        },

    },
    'HalfCheetah4dofv2': {  # 6 DoF
        'Energy0-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
        },
        'Energyz-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0,
            'energy_weights': 0,
        },
        'EnergyOne-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 1.,
        },

    },
    'HalfCheetah4dofv3': {  # 6 DoF
        'Energy0-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
        },
        'Energyz-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0,
            'energy_weights': 0,
        },
        'EnergyOne-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 1.,
        },

    },
    'HalfCheetah4dofv4': {  # 6 DoF
        'Energy0-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
        },
        'Energyz-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0,
            'energy_weights': 0,
        },
        'EnergyOne-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 1.,
        },

    },
    'HalfCheetah4dofv5': {  # 6 DoF
        'Energy0-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
        },
        'Energyz-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0,
            'energy_weights': 0,
        },
        'EnergyOne-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 1.,
        },

    },
    'HalfCheetah4dofv6': {  # 6 DoF
        'Energy0-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
        },
        'Energyz-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0,
            'energy_weights': 0,
        },
        'EnergyOne-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 1.,
        },

    },
    'HalfCheetah3doff': {  # 6 DoF
        'Energy0-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
        },
        'Energyz-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0,
            'energy_weights': 0,
        },
        'EnergyOne-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 1.,
        },

    },
    'HalfCheetah3dofb': {  # 6 DoF
        'Energy0-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
        },
        'Energyz-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0,
            'energy_weights': 0,
        },
        'EnergyOne-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 1.,
        },

    },
    'HalfCheetah3dofv3': {  # 6 DoF
        'Energy0-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
        },
        'Energyz-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0,
            'energy_weights': 0,
        },
        'EnergyOne-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 1.,
        },

    },
    'HalfCheetah3dofv4': {  # 6 DoF
        'Energy0-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
        },
        'Energyz-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0,
            'energy_weights': 0,
        },
        'EnergyOne-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 1.,
        },

    },
    'HalfCheetahSquat2dof': {  # 6 DoF
        'Energyz-v0': {
            'distance_weigth': 5.0,
            'ctrl_cost_weight': 0,
            'horizontal_weight': 0.1,
            'energy_weights': 0,
        },
        'Energy0-v0': {
            'distance_weigth': 5.0,
            'ctrl_cost_weight': 0.1,
            'horizontal_weight': 0.1,
            'energy_weights': 0,
        },
        'EnergyPoint25-v0': {
            'distance_weigth': 5.0,
            'ctrl_cost_weight': 0.25,
            'horizontal_weight': 0.1,
            'energy_weights': 0,
        },
        'EnergyAlt-v0': {
            'distance_weigth': 5.0,
            'ctrl_cost_weight': 0.,
            'horizontal_weight': 0.1,
            'energy_weights': 1.5,
        },

    },
    'HalfCheetahSquat4dof': {  # 6 DoF
        'Energyz-v0': {
            'distance_weigth': 5.0,
            'ctrl_cost_weight': 0,
            'horizontal_weight': 0.1,
            'energy_weights': 0,
        },
        'Energy0-v0': {
            'distance_weigth': 5.0,
            'ctrl_cost_weight': 0.25,
            'horizontal_weight': 0.1,
            'energy_weights': 0,
        },
        'EnergyPoint1-v0': {
            'distance_weigth': 5.0,
            'ctrl_cost_weight': 0.1,
            'horizontal_weight': 0.1,
            'energy_weights': 0,
        },
        'EnergyPoint25-v0': {
            'distance_weigth': 5.0,
            'ctrl_cost_weight': 0.25,
            'horizontal_weight': 0.1,
            'energy_weights': 0,
        },
        'EnergyAlt-v0': {
            'distance_weigth': 5.0,
            'ctrl_cost_weight': 0.,
            'horizontal_weight': 0.1,
            'energy_weights': 1.5,
        },

    },
    'HalfCheetahSquat6dof': {  # 6 DoF
        'Energyz-v0': {
            'distance_weigth': 5.0,
            'ctrl_cost_weight': 0,
            'horizontal_weight': 0.1,
            'energy_weights': 0,
        },
        'Energy0-v0': {
            'distance_weigth': 5.0,
            'ctrl_cost_weight': 0.25,
            'horizontal_weight': 0.1,
            'energy_weights': 0,
        },
        'EnergyPoint1-v0': {
            'distance_weigth': 5.0,
            'ctrl_cost_weight': 0.1,
            'horizontal_weight': 0.1,
            'energy_weights': 0,
        },
        'EnergyPoint25-v0': {
            'distance_weigth': 5.0,
            'ctrl_cost_weight': 0.25,
            'horizontal_weight': 0.1,
            'energy_weights': 0,
        },
        'EnergyAlt-v0': {
            'distance_weigth': 5.0,
            'ctrl_cost_weight': 0.,
            'horizontal_weight': 0.1,
            'energy_weights': 1.5,
        },

    },
    'HalfCheetah2dof': {  # 6 DoF
        'Energy0-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
        },
        'Energyz-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0,
            'energy_weights': 0,
        },
        'EnergyOne-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 1.,
        },

    },
    'HalfCheetah2dofv2': {  # 6 DoF
        'Energy0-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
        },
        'Energyz-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0,
            'energy_weights': 0,
        },
        'EnergyOne-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 1.,
        },

    },
    'HalfCheetah2dofv3': {  # 6 DoF
        'Energy0-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
        },
        'Energyz-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0,
            'energy_weights': 0,
        },
        'EnergyOne-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 1.,
        },

    },
    'HalfCheetah2dofv4': {  # 6 DoF
        'Energy0-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
        },
        'Energyz-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0,
            'energy_weights': 0,
        },
        'EnergyOne-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 1.,
        },

    },
    'HalfCheetah2dofv5': {  # 6 DoF
        'Energy0-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
        },
        'Energyz-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0,
            'energy_weights': 0,
        },
        'EnergyOne-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 1.,
        },

    },
    'HalfCheetah': {  # 6 DoF
        'EnergySix-v0': {
            'forward_reward_weight':1.0,
            'ctrl_cost_weight':0.1,
            'energy_weights':6.0,

        },
        'EnergyFour-v0': {
            'forward_reward_weight':1.0,
            'ctrl_cost_weight':0.1,
            'energy_weights':4.0,

        },
        'EnergyTwo-v0': {
            'forward_reward_weight':1.0,
            'ctrl_cost_weight':0.1,
            'energy_weights':2.0,
        },
        'EnergyOnePoint5-v0': {
            'forward_reward_weight':1.0,
            'ctrl_cost_weight':0.1,
            'energy_weights':1.5,
        },
        'EnergyOne-v0': {
            'forward_reward_weight':1.0,
            'ctrl_cost_weight':0.1,
            'energy_weights':1.,
        },
        'EnergyPoint5-v0': {
            'forward_reward_weight':1.0,
            'ctrl_cost_weight':0.1,
            'energy_weights':0.5,
        },
        'EnergyPoint1-v0': {
            'forward_reward_weight':1.0,
            'ctrl_cost_weight':0.1,
            'energy_weights':0.1,
        },
        'Energy0-v0': {
            'forward_reward_weight':1.0,
            'ctrl_cost_weight':0.1,
            'energy_weights':0,
        },

        'Symloss-v0': {
            'forward_reward_weight':1.0,
            'ctrl_cost_weight':0.1,
            'energy_weights':0,
        },

        'Symdup-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
        },
        'Energy0-v1': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'half_cheetah_v2.xml',
        },
        'Symloss-v1': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'half_cheetah_v2.xml',

        },

        'Symdup-v1': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'half_cheetah_v2.xml',

        },
        'Energyz-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0,
            'energy_weights': 0,
        },

    },

    'FullCheetahHeavy': {  # 6 DoF
        'MinSpring-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv6.xml',

        },
        'MinSpring-v00': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv6.xml',

        },
        'MinSpring-v2': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv6.xml',
            "speed": 1
        },
        'MinSpring-v25': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.25,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv6.xml',
            "speed": 1
        },
        'MinSpring-v4': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv6.xml',
            "speed": 3
        },
        'MinSpring-v45': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.25,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv6.xml',
            "speed": 3
        },
        'MinSpring-v6': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv6.xml',
            "speed": 5
        },
        'MinSpring-v65': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.25,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv6.xml',
            "speed": 5
        },

        'LessSpring-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv4.xml',

        },
        'LessSpring-v2': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv4.xml',
            "speed":1
        },
        'LessSpring-v4': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv4.xml',
            "speed": 3
        },
        'LessSpring-v6': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv4.xml',
            "speed": 5
        },
        'MoreSpring-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv5.xml',

        },
        'MoreSpring-v2': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv5.xml',
            "speed":1
        },
        'MoreSpring-v4': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv5.xml',
            "speed": 3
        },
        'MoreSpring-v6': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv5.xml',
            "speed": 5
        },
        'ExSpring-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv7.xml',

        },
        'ExSpring-v00': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv7.xml',

        },
        'ExSpring-v2': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv7.xml',
            "speed": 1
        },
        'ExSpring-v25': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.25,
            'energy_weights': 0.,
            'xml_file': 'full_cheetah_heavyv7.xml',
            "speed": 1
        },
        'ExSpring-v210': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv7.xml',
            "speed": 1
        },
        'ExSpring-v4': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv7.xml',
            "speed": 3
        },
        'ExSpring-v45': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.25,
            'energy_weights': 0.,
            'xml_file': 'full_cheetah_heavyv7.xml',
            "speed": 3
        },
        'ExSpring-v410': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv7.xml',
            "speed": 3
        },
        'ExSpring-v6': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv7.xml',
            "speed": 5
        },
        'ExSpring-v65': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.25,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv7.xml',
            "speed": 5
        },
        'ExSpring-v610': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv7.xml',
            "speed": 5
        },
        'Energy0-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv3.xml',

        },
        'Energy0-v00': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv3.xml',

        },
        'Energy0-v1': {
                'forward_reward_weight': 1.0,
                'ctrl_cost_weight': 0.1,
                'energy_weights': 0,
                'xml_file': 'full_cheetah_heavyv3.xml',
                "speed":0.5
            },
        'Energy0-v2': {
                'forward_reward_weight': 1.0,
                'ctrl_cost_weight': 0.1,
                'energy_weights': 0,
                'xml_file': 'full_cheetah_heavyv3.xml',
                "speed":1
            },
        'Energy0-v25': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.25,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv3.xml',
            "speed": 1
        },
        'Energy0-v3': {
                'forward_reward_weight': 1.0,
                'ctrl_cost_weight': 0.1,
                'energy_weights': 0,
                'xml_file': 'full_cheetah_heavyv3.xml',
                "speed":2
        },
        'Energy0-v4': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv3.xml',
            "speed": 3
        },
        'Energy0-v45': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.25,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv3.xml',
            "speed": 3
        },
        'Energy0-v5': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv3.xml',
            "speed": 4
        },
        'Energy0-v6': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv3.xml',
            "speed": 5
        },
        'Energy0-v65': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.25,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv3.xml',
            "speed": 5
        },


        'RealFC-v1': {
            'forward_reward_weight': 5.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyReal.xml',
            "speed": 0.25   #  5m/s:0.25  10m/s:0.5  15m/s:0.75  20m/s:1   25m/s:1.25  30m/s: 1.5
        },
        'RealFC-v2': {
            'forward_reward_weight': 5.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyReal.xml',
            "speed": 0.5
        },
        'RealFC-v3': {
            'forward_reward_weight': 5.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyReal.xml',
            "speed": 0.75
        },
        'RealFC-v4': {
            'forward_reward_weight': 5.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyReal.xml',
            "speed": 1
        },
        'RealFC-v5': {
            'forward_reward_weight': 5.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyReal.xml',
            "speed": 1.25
        },
        'RealFC-v6': {
            'forward_reward_weight': 5.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyReal.xml',
            "speed": 1.5
        },

        'RealFCT-v1': {
            'forward_reward_weight': 10.0,
            'ctrl_cost_weight': 0.,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyReal.xml',
            "speed": 0.25,
            "walkstyle": "trot",  # 5m/s:0.25  10m/s:0.5  15m/s:0.75  20m/s:1   25m/s:1.25  30m/s: 1.5
        },
        'RealFCT-v2': {
            'forward_reward_weight': 5.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyReal.xml',
            "speed": 0.5,
            "walkstyle": "trot",
        },
        'RealFCT-v3': {
            'forward_reward_weight': 5.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyReal.xml',
            "speed": 0.75,
            "walkstyle": "trot",
        },
        'RealFCT-v4': {
            'forward_reward_weight': 5.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyReal.xml',
            "speed": 1,
            "walkstyle": "trot",
        },
        'RealFCT-v5': {
            'forward_reward_weight': 5.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyReal.xml',
            "speed": 1.25,
            "walkstyle": "trot",
        },
        'RealFCT-v6': {
            'forward_reward_weight': 5.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyReal.xml',
            "speed": 1.5,
            "walkstyle": "trot",
        },

        'RealFCG-v1': {
            'forward_reward_weight': 5.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyReal.xml',
            "speed": 0.25,
            "walkstyle": "gallop",  # 5m/s:0.25  10m/s:0.5  15m/s:0.75  20m/s:1   25m/s:1.25  30m/s: 1.5
        },
        'RealFCG-v2': {
            'forward_reward_weight': 5.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyReal.xml',
            "speed": 0.5,
            "walkstyle": "gallop",
        },
        'RealFCG-v3': {
            'forward_reward_weight': 5.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyReal.xml',
            "speed": 0.75,
            "walkstyle": "gallop",
        },
        'RealFCG-v4': {
            'forward_reward_weight': 5.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyReal.xml',
            "speed": 1,
            "walkstyle": "gallop",
        },
        'RealFCG-v5': {
            'forward_reward_weight': 5.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyReal.xml',
            "speed": 1.25,
            "walkstyle": "gallop",
        },
        'RealFCG-v6': {
            'forward_reward_weight': 5.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyReal.xml',
            "speed": 1.5,
            "walkstyle": "gallop",
        },


        'MinSpringGc-v0': {
                    'forward_reward_weight': 1.0,
                    'ctrl_cost_weight': 0.1,
                    'energy_weights': 0,
                    'contact_cost_weight':8e-4,
                    'xml_file': 'full_cheetah_heavyv6.xml',
                    "walkstyle": "gallop",
                },
        'MinSpringGc-v2': {
                            'forward_reward_weight': 1.0,
                            'ctrl_cost_weight': 0.1,
                            'energy_weights': 0,
                            'contact_cost_weight':8e-4,
                            'xml_file': 'full_cheetah_heavyv6.xml',
                            "walkstyle": "gallop",
                            "speed": 1
                        },
        'MinSpringGc-v4': {
                            'forward_reward_weight': 1.0,
                            'ctrl_cost_weight': 0.1,
                            'energy_weights': 0,
                            'contact_cost_weight':8e-4,
                            'xml_file': 'full_cheetah_heavyv6.xml',
                            "walkstyle": "gallop",
                            "speed": 3
                        },
        'MinSpringGc-v6': {
                            'forward_reward_weight': 1.0,
                            'ctrl_cost_weight': 0.1,
                            'energy_weights': 0,
                            'contact_cost_weight':8e-4,
                            'xml_file': 'full_cheetah_heavyv6.xml',
                            "walkstyle": "gallop",
                            "speed": 5
                        },

        'ExSpringGc-v0': {
                    'forward_reward_weight': 1.0,
                    'ctrl_cost_weight': 0.1,
                    'energy_weights': 0,
                    'contact_cost_weight':8e-4,
                    'xml_file': 'full_cheetah_heavyv7.xml',
                    "walkstyle": "gallop",
                },
        'ExSpringGc-v2': {
                            'forward_reward_weight': 1.0,
                            'ctrl_cost_weight': 0.1,
                            'energy_weights': 0,
                            'contact_cost_weight':8e-4,
                            'xml_file': 'full_cheetah_heavyv7.xml',
                            "walkstyle": "gallop",
                            "speed": 1
                        },
        'ExSpringGc-v4': {
                            'forward_reward_weight': 1.0,
                            'ctrl_cost_weight': 0.1,
                            'energy_weights': 0,
                            'contact_cost_weight':8e-4,
                            'xml_file': 'full_cheetah_heavyv7.xml',
                            "walkstyle": "gallop",
                            "speed": 3
                        },
        'ExSpringGc-v6': {
                            'forward_reward_weight': 1.0,
                            'ctrl_cost_weight': 0.1,
                            'energy_weights': 0,
                            'contact_cost_weight':8e-4,
                            'xml_file': 'full_cheetah_heavyv7.xml',
                            "walkstyle": "gallop",
                            "speed": 5
                        },
        'SymlossGc-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'contact_cost_weight': 8e-4,
            "walkstyle": "gallop",
            'xml_file': 'full_cheetah_heavyv3.xml',
        },
        'SymlossGc-v2': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'contact_cost_weight': 8e-4,
            "walkstyle": "gallop",
            'xml_file': 'full_cheetah_heavyv3.xml',
            "speed": 1
        },
        'SymlossGc-v4': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'contact_cost_weight': 8e-4,
            "walkstyle": "gallop",
            'xml_file': 'full_cheetah_heavyv3.xml',
            "speed": 3
        },
        'SymlossGc-v6': {
                    'forward_reward_weight': 1.0,
                    'ctrl_cost_weight': 0.1,
                    'energy_weights': 0,
                    'contact_cost_weight':8e-4,
                    "walkstyle": "gallop",
                    'xml_file': 'full_cheetah_heavyv3.xml',
                    "speed": 5
                },

        'SymlossGphase-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            "walkstyle": "gallop",
            'xml_file': 'full_cheetah_heavyv3.xml',
            "phase_delay": 15,
        },
        'SymlossGphase-v4': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            "walkstyle": "gallop",
            'xml_file': 'full_cheetah_heavyv3.xml',
            "speed": 3,
            "phase_delay":15,
        },

        'ExSpringGphase-v4': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv7.xml',
            "walkstyle": "gallop",
            "speed": 3,
            "phase_delay":15,
        },

        'MinSpringGphase-v4': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv6.xml',
            "walkstyle": "gallop",
            "speed": 3,
            "phase_delay":15,
        },

        'MinSpringG-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv6.xml',
            "walkstyle": "gallop",
        },
        'MinSpringG-v2': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv6.xml',
            "walkstyle": "gallop",
            "speed": 1
        },
        'MinSpringG-v4': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv6.xml',
            "walkstyle": "gallop",
            "speed": 3
        },
        'MinSpringG-v6': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv6.xml',
            "walkstyle": "gallop",
            "speed": 5
        },
        'ExSpringG-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv7.xml',
            "walkstyle": "gallop",
        },
        'ExSpringG-v2': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv7.xml',
            "walkstyle": "gallop",
            "speed": 1
        },
        'ExSpringG-v4': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv7.xml',
            "walkstyle": "gallop",
            "speed": 3
        },
        'ExSpringG-v6': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv7.xml',
            "walkstyle": "gallop",
            "speed": 5
        },

        'SymlossG-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            "walkstyle": "gallop",
            'xml_file': 'full_cheetah_heavyv3.xml',

        },
        'SymlossG-v1': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            "walkstyle": "gallop",
            'xml_file': 'full_cheetah_heavyv3.xml',
            "speed": 0.5

        },
        'SymlossG-v2': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv3.xml',
            "walkstyle": "gallop",
            "speed": 1
        },
        'SymlossG-v3': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            "walkstyle": "gallop",
            'xml_file': 'full_cheetah_heavyv3.xml',
            "speed": 2
        },
        'SymlossG-v4': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            "walkstyle": "gallop",
            'xml_file': 'full_cheetah_heavyv3.xml',
            "speed": 3
        },
        'SymlossG-v5': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            "walkstyle": "gallop",
            'xml_file': 'full_cheetah_heavyv3.xml',
            "speed": 4
        },
        'SymlossG-v6': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            "walkstyle": "gallop",
            'xml_file': 'full_cheetah_heavyv3.xml',
            "speed": 5
        },

        'MinSpringT-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv6.xml',
            "walkstyle": "trot",
        },
        'MinSpringT-v2': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv6.xml',
            "walkstyle": "trot",
            "speed": 1
        },
        'MinSpringT-v4': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv6.xml',
            "walkstyle": "trot",
            "speed": 3
        },
        'MinSpringT-v6': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv6.xml',
            "walkstyle": "trot",
            "speed": 5
        },
        'ExSpringT-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv7.xml',
            "walkstyle": "trot",
        },
        'ExSpringT-v2': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv7.xml',
            "walkstyle": "trot",
            "speed": 1
        },
        'ExSpringT-v4': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv7.xml',
            "walkstyle": "trot",
            "speed": 3
        },
        'ExSpringT-v6': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            'xml_file': 'full_cheetah_heavyv7.xml',
            "walkstyle": "trot",
            "speed": 5
        },

        'SymlossT-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            "walkstyle": "trot",
            'xml_file': 'full_cheetah_heavyv3.xml',
        },
        'SymlossT-v1': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            "walkstyle": "trot",
            'xml_file': 'full_cheetah_heavyv3.xml',
            "speed":0.5
        },
        'SymlossT-v2': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            "walkstyle": "trot",
            'xml_file': 'full_cheetah_heavyv3.xml',
            "speed":1
        },
        'SymlossT-v3': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            "walkstyle": "trot",
            'xml_file': 'full_cheetah_heavyv3.xml',
            "speed": 2
        },
        'SymlossT-v4': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            "walkstyle": "trot",
            'xml_file': 'full_cheetah_heavyv3.xml',
            "speed": 3
        },
        'SymlossT-v5': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            "walkstyle": "trot",
            'xml_file': 'full_cheetah_heavyv3.xml',
            "speed": 4
        },
        'SymlossT-v6': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            "walkstyle": "trot",
            'xml_file': 'full_cheetah_heavyv3.xml',
            "speed": 5
        },
        'SymlossT-v7': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            "walkstyle": "trotv2",
            'xml_file': 'full_cheetah_heavyv3.xml',
            "speed": 3
        },
        'SymlossT-v8': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            "walkstyle": "trotv2",
            'xml_file': 'full_cheetah_heavyv3.xml',
            "speed": 5
        },
        'SymlossGfblr-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            "walkstyle": "gallopFBLR",
            'xml_file': 'full_cheetah_heavyv3.xml',
        },

        'SymPenG-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            "walkstyle": "gallop",
            'xml_file': 'full_cheetah_heavyv3.xml',
            "soft_gait_penalty_weight" : 0.05,
            "soft_gait_target" : 0,
            "speed":3,
        },
        'SymPenG-v1': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            "walkstyle": "gallop",
            'xml_file': 'full_cheetah_heavyv3.xml',
            "soft_gait_penalty_weight" : 0.05,#0.25#0.5
            "soft_gait_target" : 0,
            "speed":5,
        },


        'SymPenT-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            "walkstyle": "trot",
            'xml_file': 'full_cheetah_heavyv3.xml',
            "soft_gait_penalty_weight": 0.05,
            "soft_gait_target": 0,
            "speed":3,
        },
        'SymPenT-v1': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            "walkstyle": "trot",
            'xml_file': 'full_cheetah_heavyv3.xml',
            "soft_gait_penalty_weight": 0.05,
            "soft_gait_target": 0,
            "speed": 5,
        },


        'SymPenG-v2': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            "walkstyle": "gallop",
            'xml_file': 'full_cheetah_heavyv3.xml',
            "soft_gait_penalty_weight": 1,
            "soft_gait_target": 2,
            "speed": 3,
        },
        'SymPenG-v3': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            "walkstyle": "gallop",
            'xml_file': 'full_cheetah_heavyv3.xml',
            "soft_gait_penalty_weight": 1,
            "soft_gait_target": 2.5,
            "speed": 3,
        },
        'SymPenG-v4': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            "walkstyle": "gallop",
            'xml_file': 'full_cheetah_heavyv3.xml',
            "soft_gait_penalty_weight": 1,
            "soft_gait_target": 3,
            "speed": 3,
        },
        'SymPenG-v5': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            "walkstyle": "gallop",
            'xml_file': 'full_cheetah_heavyv3.xml',
            "soft_gait_penalty_weight": 1,
            "soft_gait_target": 2,
            "speed": 5,
        },
        'SymPenG-v6': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            "walkstyle": "gallop",
            'xml_file': 'full_cheetah_heavyv3.xml',
            "soft_gait_penalty_weight": 1,
            "soft_gait_target": 2.5,
            "speed": 5,
        },
        'SymPenG-v7': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            "walkstyle": "gallop",
            'xml_file': 'full_cheetah_heavyv3.xml',
            "soft_gait_penalty_weight": 1,
            "soft_gait_target": 3,
            "speed": 5,
        },
        # 'SymPenG-v5': {
        #     'forward_reward_weight': 1.0,
        #     'ctrl_cost_weight': 0.1,
        #     'energy_weights': 0,
        #     "walkstyle": "gallop2",
        #     'xml_file': 'full_cheetah_heavyv3.xml',
        #     "soft_gait_penalty_weight": 1,
        #     "soft_gait_target": 0.25,
        #     "soft_gait_target2": 0.5,
        # },
        # 'SymPenG-v6': {
        #     'forward_reward_weight': 1.0,
        #     'ctrl_cost_weight': 0.1,
        #     'energy_weights': 0,
        #     "walkstyle": "gallop2",
        #     'xml_file': 'full_cheetah_heavyv3.xml',
        #     "soft_gait_penalty_weight": 1,
        #     "soft_gait_target": 0.5,
        #     "soft_gait_target2": 1,
        # },
        # 'SymPenG-v7': {
        #     'forward_reward_weight': 1.0,
        #     'ctrl_cost_weight': 0.1,
        #     'energy_weights': 0,
        #     "walkstyle": "gallop2",
        #     'xml_file': 'full_cheetah_heavyv3.xml',
        #     "soft_gait_penalty_weight": 1,
        #     "soft_gait_target": 0.25,
        #     "soft_gait_target2": 0.75,
        # },

    },

    'FullCheetah': {  # 6 DoF
        'Energy0-v0': {
            'forward_reward_weight':1.0,
            'ctrl_cost_weight':0.1,
            'energy_weights':0,
        },
        'SymlossG-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            "walkstyle":"gallop"
        },

        'SymdupG-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            "walkstyle": "gallop"

        },
        'SymlossT-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            "walkstyle": "trot"
        },

        'SymdupT-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            "walkstyle": "trot"

        },
        'SymlossGfblr-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            "walkstyle": "gallopFBLR"
        },

        'SymdupGfblr-v0': {
            'forward_reward_weight': 1.0,
            'ctrl_cost_weight': 0.1,
            'energy_weights': 0,
            "walkstyle": "gallopFBLR"

        },

    },
    'AntRun': {  # 6 DoF
        'Energy0-v0': {
            'terminate_when_unhealthy': False,  ##
            'energy_weights': 0,
        },

    },
    'AntSquaT': {  # 6 DoF
        'Energy0-v0': {
            'ctrl_cost_weight' : 0.1,  ##0.1
            'horizontal_weight' : 1,  ##
            'contact_cost_weight' : 0.005,  ##5e-4
            'distance_weigth' : 5,
            'terminate_when_unhealthy':False,##
            'energy_weights': 0,
        },

    },
    'AntSquaTRedundant': {  # 6 DoF
        'Energy0-v0': {
            'ctrl_cost_weight' : 0.1,  ##0.1
            'horizontal_weight' : 10,  ##
            'contact_cost_weight' : 0.1,  ##5e-4
            'distance_weigth' : 10,
            'terminate_when_unhealthy':False,##
            'energy_weights': 0,
        },

    },


}

NUM_CHECKPOINTS = 10


def get_variant_spec_base(universe, domain, task, policy, algorithm,epoch_length,num_epoch,actor_size=256,critic_size=256,
                          n_layer=2):
    if num_epoch is not None:
        ALGORITHM_PARAMS_PER_DOMAIN[domain]['kwargs']['n_epochs']=num_epoch

    algorithm_params = deep_update(
        ALGORITHM_PARAMS_BASE,
        ALGORITHM_PARAMS_PER_DOMAIN.get(domain, {})
    )

    ALGORITHM_PARAMS_ADDITIONAL[algorithm]['kwargs']['epoch_length']=epoch_length

    nl=[]
    for i in range(n_layer):
        nl.append(actor_size)
    POLICY_PARAMS_BASE[policy]['kwargs']['hidden_layer_sizes']=tuple(nl)
    #POLICY_PARAMS_BASE[policy]['kwargs']['hidden_layer_sizes']=(actor_size,actor_size)

    algorithm_params = deep_update(
        algorithm_params,
        ALGORITHM_PARAMS_ADDITIONAL.get(algorithm, {})
    )

    print(algorithm_params)

    env_param = ENV_PARAMS.get(domain, {}).get(task, {})

    variant_spec = {
        'domain': domain,
        'task': task,
        'universe': universe,

        'env_params':env_param ,
        'policy_params': deep_update(
            POLICY_PARAMS_BASE[policy],
            POLICY_PARAMS_FOR_DOMAIN[policy].get(domain, {})
        ),
        'Q_params': {
            'type': 'double_feedforward_Q_function',
            'kwargs': {
                'hidden_layer_sizes': (critic_size,critic_size),#256
            }
        },
        'algorithm_params': algorithm_params,
        'replay_pool_params': {
            'type': 'SimpleReplayPool',
            'kwargs': {
                'max_size': tune.sample_from(lambda spec: (
                    {
                        'SimpleReplayPool': int(1e6),
                        'TrajectoryReplayPool': int(1e4),
                    }.get(
                        spec.get('config', spec)
                        ['replay_pool_params']
                        ['type'],
                        int(1e6))
                )),
            }
        },
        'sampler_params': {
            'type': 'SimpleSampler',
            'kwargs': {
                'max_path_length': MAX_PATH_LENGTH_PER_DOMAIN.get(
                    domain, epoch_length),#DEFAULT_MAX_PATH_LENGTH
                'min_pool_size': MAX_PATH_LENGTH_PER_DOMAIN.get(
                    domain, DEFAULT_MAX_PATH_LENGTH),
                'batch_size': 256,
            }
        },
        'run_params': {
            'seed': tune.sample_from(
                lambda spec: np.random.randint(0, 10000)),
            'checkpoint_at_end': True,
            'checkpoint_frequency': NUM_EPOCHS_PER_DOMAIN.get(
                domain, DEFAULT_NUM_EPOCHS) // NUM_CHECKPOINTS,
            'checkpoint_replay_pool': False,
        },
    }

    return variant_spec


def get_variant_spec_image(universe,
                           domain,
                           task,
                           policy,
                           algorithm,
                           *args,
                           **kwargs):
    variant_spec = get_variant_spec_base(
        universe, domain, task, policy, algorithm, *args, **kwargs)

    if 'image' in task.lower() or 'image' in domain.lower():
        preprocessor_params = {
            'type': 'convnet_preprocessor',
            'kwargs': {
                'image_shape': variant_spec['env_params']['image_shape'],
                'output_size': M,
                'conv_filters': (4, 4),
                'conv_kernel_sizes': ((3, 3), (3, 3)),
                'pool_type': 'MaxPool2D',
                'pool_sizes': ((2, 2), (2, 2)),
                'pool_strides': (2, 2),
                'dense_hidden_layer_sizes': (),
            },
        }
        variant_spec['policy_params']['kwargs']['preprocessor_params'] = (
            preprocessor_params.copy())
        variant_spec['Q_params']['kwargs']['preprocessor_params'] = (
            preprocessor_params.copy())

    return variant_spec


def get_variant_spec(args):
    universe, domain, task = args.universe, args.domain, args.task

    if ('image' in task.lower()
        or 'blind' in task.lower()
        or 'image' in domain.lower()):
        variant_spec = get_variant_spec_image(
            universe, domain, task, args.policy, args.algorithm)
    else:
        variant_spec = get_variant_spec_base(
            universe, domain, task, args.policy, args.algorithm,args.epoch_length,args.total_epoch,args.actor_size,
            args.critic_size,n_layer=args.n_layer)

    if args.checkpoint_replay_pool is not None:
        variant_spec['run_params']['checkpoint_replay_pool'] = (
            args.checkpoint_replay_pool)

    return variant_spec
