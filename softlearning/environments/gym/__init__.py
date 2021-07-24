"""Custom Gym environments.

Every class inside this module should extend a gym.Env class. The file
structure should be similar to gym.envs file structure, e.g. if you're
implementing a mujoco env, you would implement it under gym.mujoco submodule.
"""

import gym


CUSTOM_GYM_ENVIRONMENTS_PATH = __package__
MUJOCO_ENVIRONMENTS_PATH = f'{CUSTOM_GYM_ENVIRONMENTS_PATH}.mujoco'

MUJOCO_ENVIRONMENT_SPECS = (
    {
        'id': 'Walker2d-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.walker2d:Walker2dEnv'),
    },
    {
        'id': 'Walker2d-EnergyPoint5-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.walker2d:Walker2dEnv'),
    },
    {
        'id': 'Walker2d-EnergyOne-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.walker2d:Walker2dEnv'),
    },
    {
        'id': 'Walker2d-EnergyOnePoint5-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.walker2d:Walker2dEnv'),
    },
    {
        'id': 'Walker2d-EnergyTwo-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.walker2d:Walker2dEnv'),
    },
    {
        'id': 'Walker2d-EnergyFour-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.walker2d:Walker2dEnv'),
    },
    {
        'id': 'Walker2d-EnergySix-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.walker2d:Walker2dEnv'),
    },
    {
        'id': 'HalfCheetah-EnergySix-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv'),
    },
    {
        'id': 'HalfCheetah-EnergyFour-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv'),
    },
    {
        'id': 'HalfCheetah-EnergyTwo-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv'),
    },
    {
        'id': 'HalfCheetah-EnergyOnePoint5-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv'),
    },
    {
        'id': 'HalfCheetah-EnergyOne-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv'),
    },
    {
        'id': 'HalfCheetah-EnergyPoint5-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv'),
    },
    {
        'id': 'HalfCheetah-EnergyPoint1-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv'),
    },
    {
        'id': 'HalfCheetah-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv'),
    },
    {
        'id': 'HalfCheetah-Energy0-v1',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv'),
    },
    {
        'id': 'HalfCheetah-Energyz-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv'),
    },
    {
        'id': 'HalfCheetah5dof-EnergyOne-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv_5dof'),
    },
    {
        'id': 'HalfCheetah5dof-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv_5dof'),
    },
    {
        'id': 'HalfCheetah5dof-Energyz-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv_5dof'),
    },
    {
        'id': 'HalfCheetah5dofv2-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv_5dofv2'),
    },
    {
        'id': 'HalfCheetah5dofv3-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv_5dofv3'),
    },
    {
        'id': 'HalfCheetah5dofv4-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv_5dofv4'),
    },
    {
        'id': 'HalfCheetah5dofv5-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv_5dofv5'),
    },
    {
        'id': 'HalfCheetah5dofv6-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv_5dofv6'),
    },
    {
        'id': 'HalfCheetah4dof-EnergyOne-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv_4dof'),
    },
    {
        'id': 'HalfCheetah4dof-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv_4dof'),
    },
    {
        'id': 'HalfCheetah4dof-Energyz-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv_4dof'),
    },
    {
        'id': 'HalfCheetah4dofv2-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv_4dofv2'),
    },
    {
        'id': 'HalfCheetah4dofv3-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv_4dofv3'),
    },
    {
        'id': 'HalfCheetah4dofv4-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv_4dofv4'),
    },
    {
        'id': 'HalfCheetah4dofv5-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv_4dofv5'),
    },
    {
        'id': 'HalfCheetah4dofv6-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv_4dofv6'),
    },
    {
        'id': 'HalfCheetah3doff-EnergyOne-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv_3doff'),
    },
    {
        'id': 'HalfCheetah3doff-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv_3doff'),
    },
    {
        'id': 'HalfCheetah3dofv3-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv_3dofv3'),
    },
    {
        'id': 'HalfCheetah3dofv4-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv_3dofv4'),
    },
    {
        'id': 'HalfCheetah3doff-Energyz-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv_3doff'),
    },
    {
        'id': 'HalfCheetah3dofb-EnergyOne-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv_3dofb'),
    },
    {
        'id': 'HalfCheetah3dofb-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv_3dofb'),
    },
    {
        'id': 'HalfCheetah3dofb-Energyz-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv_3dofb'),
    },
    {
        'id': 'HalfCheetah2dof-EnergyOne-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv_2dof'),
    },
    {
        'id': 'HalfCheetah2dof-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv_2dof'),
    },
    {
        'id': 'HalfCheetah2dof-Energyz-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv_2dof'),
    },
    {
        'id': 'HalfCheetah2dofv2-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv_2dofv2'),
    },
    {
        'id': 'HalfCheetah2dofv3-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv_2dofv3'),
    },
    {
        'id': 'HalfCheetah2dofv4-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv_2dofv4'),
    },
    {
        'id': 'HalfCheetah2dofv5-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv_2dofv5'),
    },

    {
        'id': 'Giraffe-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.giraffe:GiraffeEnv'),
    },
    {
        'id': 'HalfCheetahHeavy-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahHeavyEnv'),
    },
    {
        'id': 'VA-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:VA'),
    },
    {
        'id': 'VA-smallRange-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:VA'),
    },
    {
        'id': 'VA-bigRange-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:VA'),
    },

    {
        'id': 'VA4dof-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:VA4dof'),
    },
    {
        'id': 'VA4dof-smallRange-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:VA4dof'),
    },
    {
        'id': 'VA4dof-bigRange-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:VA4dof'),
    },
    {
        'id': 'VA6dof-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:VA6dof'),
    },
    {
        'id': 'VA6dof-smallRange-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:VA6dof'),
    },
    {
        'id': 'VA6dof-bigRange-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:VA6dof'),
    },
    {
        'id': 'VA8dof-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:VA8dof'),
    },
    {
        'id': 'VA8dof-smallRange-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:VA8dof'),
    },
    {
        'id': 'VA8dof-bigRange-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:VA8dof'),
    },
    {
        'id': 'VA-Energyz-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:VA'),
    },
    {
        'id': 'VA4dof-Energyz-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:VA4dof'),
    },
    {
        'id': 'VA6dof-Energyz-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:VA6dof'),
    },
    {
        'id': 'VA8dof-Energyz-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:VA8dof'),
    },
    {
        'id': 'VA-EnergyOne-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:VA'),
    },
    {
        'id': 'VA4dof-EnergyOne-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:VA4dof'),
    },
    {
        'id': 'VA6dof-EnergyOne-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:VA6dof'),
    },
    {
        'id': 'VA8dof-EnergyOne-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:VA8dof'),
    },

    {
        'id': 'VA-EnergyPoint5-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:VA'),
    },
    {
        'id': 'VA4dof-EnergyPoint5-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:VA4dof'),
    },
    {
        'id': 'VA6dof-EnergyPoint5-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:VA6dof'),
    },
    {
        'id': 'VA8dof-EnergyPoint5-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:VA8dof'),
    },
    {
    'id': 'Centripede-Energy0-v0',
    'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                    '.centripede:CentripedeEnv'),
    },
    {
        'id': 'FullCheetahHeavy-MinSpringGc-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-MinSpringGc-v2',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-MinSpringGc-v4',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-MinSpringGc-v6',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-ExSpringGc-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-ExSpringGc-v2',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-ExSpringGc-v4',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-ExSpringGc-v6',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-SymlossGc-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-SymlossGc-v2',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-SymlossGc-v4',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-SymlossGc-v6',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-SymlossGphase-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },


    {
    'id': 'FullCheetahHeavy-MinSpringGphase-v4',
    'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                    '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-SymlossGphase-v4',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-ExSpringGphase-v4',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-MinSpringG-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-MinSpringG-v2',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-MinSpringG-v4',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-MinSpringG-v6',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-MinSpringT-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-MinSpringT-v2',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-MinSpringT-v4',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-MinSpringT-v6',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-MinSpring-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-MinSpring-v00',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-MinSpring-v2',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-MinSpring-v25',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-MinSpring-v4',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-MinSpring-v45',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-MinSpring-v6',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-MinSpring-v65',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-LessSpring-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-LessSpring-v2',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-LessSpring-v4',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-LessSpring-v6',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-MoreSpring-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-MoreSpring-v2',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-MoreSpring-v4',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-MoreSpring-v6',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-ExSpringG-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-ExSpringG-v2',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-ExSpringG-v4',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-ExSpringG-v6',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-ExSpringT-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-ExSpringT-v2',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-ExSpringT-v4',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-ExSpringT-v6',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-ExSpring-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-ExSpring-v00',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-ExSpring-v2',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
{
        'id': 'FullCheetahHeavy-ExSpring-v21',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
{
        'id': 'FullCheetahHeavy-ExSpring-v25',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
{
        'id': 'FullCheetahHeavy-ExSpring-v210',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-ExSpring-v4',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-ExSpring-v41',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-ExSpring-v45',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-ExSpring-v410',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-ExSpring-v6',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-ExSpring-v61',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-ExSpring-v65',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-ExSpring-v610',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-Energy0-v00',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },

    {
        'id': 'FullCheetahHeavy-Energy0-v1',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-Energy0-v2',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-Energy0-v25',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-Energy0-v3',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-Energy0-v4',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-Energy0-v45',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-Energy0-v5',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-Energy0-v6',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-Energy0-v65',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-RealFC-v1',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-RealFC-v2',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-RealFC-v3',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-RealFC-v4',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-RealFC-v5',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-RealFC-v6',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-RealFCT-v1',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-RealFCT-v2',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-RealFCT-v3',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-RealFCT-v4',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-RealFCT-v5',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-RealFCT-v6',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-RealFCG-v1',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-RealFCG-v2',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-RealFCG-v3',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-RealFCG-v4',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-RealFCG-v5',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-RealFCG-v6',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-SymlossGfblr-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-SymlossG-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-SymlossT-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-SymPenG-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-SymPenT-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-SymPenG-v1',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-SymPenG-v2',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-SymPenG-v3',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-SymPenG-v4',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-SymPenG-v5',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-SymPenG-v6',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-SymPenG-v7',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-SymPenT-v1',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-SymlossG-v1',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-SymlossT-v1',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-SymlossG-v2',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-SymlossT-v2',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-SymlossG-v3',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-SymlossT-v3',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-SymlossG-v4',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-SymlossT-v4',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-SymlossG-v5',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-SymlossT-v5',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-SymlossG-v6',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-SymlossT-v6',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-SymlossT-v7',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetahHeavy-SymlossT-v8',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'FullCheetah-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:FullCheetahEnv'),
    },
    {
        'id': 'HalfCheetah-PerfIndex-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
        '.half_cheetah:HalfCheetahEnv2'),
    },
    {
        'id': 'HalfCheetah-InvPerfIndex-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahEnv3'),
    },
    {
        'id': 'Ant-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.ant:AntEnv'),
    },
    {
        'id': 'HalfCheetahSquat2dof-Energyz-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahSquat2dof'),
    },
    {
        'id': 'HalfCheetahSquat2dof-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahSquat2dof'),
    },
    {
        'id': 'HalfCheetahSquat2dof-EnergyPoint25-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahSquat2dof'),
    },
    {
        'id': 'HalfCheetahSquat2dof-EnergyAlt-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahSquat2dof'),
    },
    {
        'id': 'HalfCheetahSquat4dof-Energyz-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahSquat4dof'),
    },
    {
        'id': 'HalfCheetahSquat4dof-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahSquat4dof'),
    },
    {
        'id': 'HalfCheetahSquat4dof-EnergyAlt-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahSquat4dof'),
    },
    {
        'id': 'HalfCheetahSquat6dof-EnergyAlt-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahSquat6dof'),
    },  #
    {
        'id': 'HalfCheetahSquat6dof-Energyz-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahSquat6dof'),
    },  #
    {
        'id': 'HalfCheetahSquat6dof-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahSquat6dof'),
    },#
    {
        'id': 'HalfCheetahSquat4dof-EnergyPoint1-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahSquat4dof'),
    },
    {
        'id': 'HalfCheetahSquat4dof-EnergyPoint25-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahSquat4dof'),
    },
    {
        'id': 'HalfCheetahSquat6dof-EnergyPoint1-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahSquat6dof'),
    },
    {
        'id': 'HalfCheetahSquat6dof-EnergyPoint25-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.half_cheetah:HalfCheetahSquat6dof'),
    },
    {
        'id': 'RealArm7dof-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:RealArm7dof'),
    },
    {
        'id': 'RealArm7dof-Energy0-v1',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:RealArm7dof'),
    },
    {
        'id': 'RealArm7dof-Energy0-v2',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:RealArm7dof'),
    },
    {
        'id': 'RealArm7dof-Energy0-v3',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:RealArm7dof'),
    },
    {
        'id': 'RealArm7dof-Energy0-v4',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:RealArm7dof'),
    },
    {
        'id': 'RealArm7dof-Energy0-v5',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:RealArm7dof'),
    },
    {
        'id': 'RealArm7dof-Energy0-v6',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:RealArm7dof'),
    },
    {
        'id': 'RealArm7dof-Energy0-v7',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:RealArm7dof'),
    },
    {
        'id': 'RealArm7dof-Energy0-v8',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:RealArm7dof'),
    },
    {
        'id': 'RealArm7dof-Energy0-v9',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:RealArm7dof'),
    },
    {
        'id': 'RealArm6dof-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:RealArm6dof'),
    },
    {
        'id': 'RealArm6dof-Energy0-v1',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:RealArm6dof'),
    },
    {
        'id': 'RealArm6dof-Energy0-v2',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:RealArm6dof'),
    },
    {
        'id': 'RealArm5dof-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:RealArm5dof'),
    },
    {
        'id': 'RealArm5dof-Energy0-v1',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:RealArm5dof'),
    },
    {
        'id': 'RealArm5dof-Energy0-v2',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:RealArm5dof'),
    },
    {
        'id': 'RealArm5dofMinE-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:RealArm5dof'),
    },
    {
        'id': 'RealArm4dofMinE-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:RealArm4dof'),
    },
    {
        'id': 'RealArm5dofLT-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:RealArm5dof'),
    },
    {
        'id': 'RealArm4dofLT-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:RealArm4dof'),
    },
    {
        'id': 'RealArm4dof-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:RealArm4dof'),
    },
    {
        'id': 'RealArm4dof-Energy0-v1',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:RealArm4dof'),
    },
    {
        'id': 'RealArm4dof-Energy0-v2',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:RealArm4dof'),
    },
    {
        'id': 'RealArm3dof-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:RealArm3dof'),
    },
    {
        'id': 'RealArm3dof-Energy0-v1',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:RealArm3dof'),
    },
    {
        'id': 'RealArm3dof-Energy0-v2',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.vertical_arm:RealArm3dof'),
    },
    {
        'id': 'AntSquaT-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.ant:AntSquaTEnv'),
    },
    {
        'id': 'AntSquaTRedundant-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.ant:AntSquaTRedundantEnv'),
    },
    {
        'id': 'AntRun-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.ant:AntRunEnv'),
    },
    {
        'id': 'AntHeavy-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.ant:AntHeavyEnv'),
    },
    {
        'id': 'Ant-EnergyPoint5-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.ant:AntEnv'),
    },
    {
        'id': 'Ant-EnergyOne-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.ant:AntEnv'),
    },
    {
        'id': 'Ant-EnergyOnePoint5-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.ant:AntEnv'),
    },
    {
        'id': 'Ant-EnergyTwo-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.ant:AntEnv'),
    },
    {
        'id': 'Humanoid-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.humanoid:HumanoidEnv'),
    },
    {
        'id': 'Humanoid-EnergyOne-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.humanoid:HumanoidEnv'),
    },
    {
        'id': 'Humanoid-EnergyPoint5-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.humanoid:HumanoidEnv'),
    },
    {
        'id': 'Humanoid-EnergyPoint1-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.humanoid:HumanoidEnv'),
    },
    {
        'id': 'Humanoid-EnergyPz5-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.humanoid:HumanoidEnv'),
    },
    {
        'id': 'Humanoidrllab-Energy0-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.humanoid_rllab:HumanoidEnv'),
    },
    {
        'id': 'Humanoidrllab-EnergyOne-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.humanoid_rllab:HumanoidEnv'),
    },
    {
        'id': 'Humanoidrllab-EnergyP5-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.humanoid_rllab:HumanoidEnv'),
    },
    {
        'id': 'Humanoidrllab-EnergyP1-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.humanoid_rllab:HumanoidEnv'),
    },
    {
        'id': 'Humanoidrllab-EnergyPz5-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.humanoid_rllab:HumanoidEnv'),
    },
    {
        'id': 'Pusher2d-Default-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.pusher_2d:Pusher2dEnv'),
    },
    {
        'id': 'Pusher2d-DefaultReach-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.pusher_2d:ForkReacherEnv'),
    },
    {
        'id': 'Pusher2d-ImageDefault-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.image_pusher_2d:ImagePusher2dEnv'),
    },
    {
        'id': 'Pusher2d-ImageReach-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.image_pusher_2d:ImageForkReacher2dEnv'),
    },
    {
        'id': 'Pusher2d-BlindReach-v0',
        'entry_point': (f'{MUJOCO_ENVIRONMENTS_PATH}'
                        '.image_pusher_2d:BlindForkReacher2dEnv'),
    },
)

GENERAL_ENVIRONMENT_SPECS = (
    {
        'id': 'MultiGoal-Default-v0',
        'entry_point': (f'{CUSTOM_GYM_ENVIRONMENTS_PATH}'
                        '.multi_goal:MultiGoalEnv')
    },
)

MULTIWORLD_ENVIRONMENT_SPECS = (
    {
        'id': 'Point2DEnv-Default-v0',
        'entry_point': 'multiworld.envs.pygame.point2d:Point2DWallEnv'
    },
    {
        'id': 'Point2DEnv-Wall-v0',
        'entry_point': 'multiworld.envs.pygame.point2d:Point2DWallEnv'
    },
)

MUJOCO_ENVIRONMENTS = tuple(
    environment_spec['id']
    for environment_spec in MUJOCO_ENVIRONMENT_SPECS)


GENERAL_ENVIRONMENTS = tuple(
    environment_spec['id']
    for environment_spec in GENERAL_ENVIRONMENT_SPECS)


MULTIWORLD_ENVIRONMENTS = tuple(
    environment_spec['id']
    for environment_spec in MULTIWORLD_ENVIRONMENT_SPECS)

GYM_ENVIRONMENTS = (
    *MUJOCO_ENVIRONMENTS,
    *GENERAL_ENVIRONMENTS,
    *MULTIWORLD_ENVIRONMENTS,
)


def register_mujoco_environments():
    """Register softlearning mujoco environments."""
    for mujoco_environment in MUJOCO_ENVIRONMENT_SPECS:
        gym.register(**mujoco_environment)

    gym_ids = tuple(
        environment_spec['id']
        for environment_spec in  MUJOCO_ENVIRONMENT_SPECS)

    return gym_ids


def register_general_environments():
    """Register gym environments that don't fall under a specific category."""
    for general_environment in GENERAL_ENVIRONMENT_SPECS:
        gym.register(**general_environment)

    gym_ids = tuple(
        environment_spec['id']
        for environment_spec in  GENERAL_ENVIRONMENT_SPECS)

    return gym_ids


def register_multiworld_environments():
    """Register custom environments from multiworld package."""
    for multiworld_environment in MULTIWORLD_ENVIRONMENT_SPECS:
        gym.register(**multiworld_environment)

    gym_ids = tuple(
        environment_spec['id']
        for environment_spec in  MULTIWORLD_ENVIRONMENT_SPECS)

    return gym_ids


def register_environments():
    registered_mujoco_environments = register_mujoco_environments()
    registered_general_environments = register_general_environments()
    registered_multiworld_environments = register_multiworld_environments()

    return (
        *registered_mujoco_environments,
        *registered_general_environments,
        *registered_multiworld_environments,
    )
