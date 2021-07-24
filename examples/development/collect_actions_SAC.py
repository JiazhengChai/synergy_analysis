import argparse
from distutils.util import strtobool
import json
import os
import pickle
from collections import OrderedDict
import tensorflow as tf
import numpy as np

from softlearning.policies.utils import get_policy_from_variant
from softlearning.samplers import rollouts,my_rollouts

speed_dict={
    "v1":0.5,
    "v2":1,
    "v25": 1,
    "v3":2,
    "v4":3,
    "v45": 3,
    "v5":4,
    "v6":5,
    "v65": 5,
}
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--agent',
                        type=str)

    parser.add_argument('--path',
                        type=str, default=None)

    parser.add_argument('--base_path',
                        type=str, default=None)

    parser.add_argument('--tr',
                        default='all',
                        type=str,
                        nargs='+')

    parser.add_argument('--energy',
                        type=str, default='Energy0-v0',
                        choices=['Energy0-v0','EnergyOne-v0','Energyz-v0',
                                'EnergyPoint5-v0','EnergyPoint1-v0',
                                'EnergyPoint25-v0','EnergyAlt-v0','smallRange-v0',
                                'bigRange-v0','Symloss-v0','Symdup-v0','SymlossGfblr-v0',
                                'SymdupGfblr-v0','SymlossG-v0','SymdupG-v0',
                                'SymlossT-v0','SymdupT-v0',
                                'Energy0-v1','Energy0-v2','Energy0-v3','Energy0-v4','Energy0-v5','Energy0-v6','Energy0-v7','Energy0-v8',
                                 'Energy0-v9',
                                'SymlossT-v1','SymlossT-v2','SymlossT-v3','SymlossT-v4','SymlossT-v5','SymlossT-v6',
                                'SymlossG-v1','SymlossG-v2','SymlossG-v3','SymlossG-v4','SymlossG-v5','SymlossG-v6',
                                'LessSpring-v0','LessSpring-v2','LessSpring-v4','LessSpring-v6',
                                'MoreSpring-v0','MoreSpring-v2','MoreSpring-v4','MoreSpring-v6',
                                'ExSpring-v0', 'ExSpring-v2', 'ExSpring-v4','ExSpring-v6',
                                'MinSpring-v0', 'MinSpring-v2', 'MinSpring-v4', 'MinSpring-v6',
                                'ExSpring-v00', 'ExSpring-v45','ExSpring-v65',
                                'MinSpring-v00',  'MinSpring-v45', 'MinSpring-v65',
                                'Energy0-v00','Energy0-v45','Energy0-v65',
                                'MinSpringG-v0','MinSpringG-v2', 'MinSpringG-v4', 'MinSpringG-v6',
                                'MinSpringT-v0', 'MinSpringT-v2', 'MinSpringT-v4','MinSpringT-v6',
                                 'ExSpringT-v0', 'ExSpringT-v2', 'ExSpringT-v4', 'ExSpringT-v6',
                                 'ExSpringG-v0', 'ExSpringG-v2', 'ExSpringG-v4', 'ExSpringG-v6',
                                ])

    parser.add_argument('--start', '-s', type=int,default=100)
    parser.add_argument('--final', '-f', type=int,default=3000)
    parser.add_argument('--step', '-st', type=int,default=100)

    parser.add_argument('--gpu_choice', type=int, default=0)

    parser.add_argument('--max-path-length', '-l', type=int, default=1000)
    parser.add_argument('--num-rollouts', '-n', type=int, default=10)
    parser.add_argument('--render-mode', '-r',
                        type=str,
                        default=None,
                        choices=('human', 'rgb_array', None),
                        help="Mode to render the rollouts in.")
    parser.add_argument('--deterministic', '-d',
                        type=lambda x: bool(strtobool(x)),
                        nargs='?',
                        const=True,
                        default=True,
                        help="Evaluate policy deterministically.")
    parser.add_argument('--name',
                        type=str,
                        help='Experiment name')

    args = parser.parse_args()

    return args


def simulate_policy(args):
    if args.gpu_choice is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_choice)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    checkpoint_path = args.checkpoint_path.rstrip('/')
    experiment_path = os.path.dirname(checkpoint_path)

    variant_path = os.path.join(experiment_path, 'params.json')
    with open(variant_path, 'r') as f:
        variant = json.load(f)

    pickle_path = os.path.join(checkpoint_path, 'checkpoint.pkl')
    print(pickle_path)
    print(os.getcwd())
    with open(pickle_path, 'rb') as f:
        picklable = pickle.load(f)

    env = picklable['env']
    policy = (
        get_policy_from_variant(variant, env))
    policy.set_weights(picklable['policy_weights'])

    with policy.set_deterministic	(args.deterministic):
        paths = my_rollouts(env=env,
                         policy=policy,
                         path_length=args.max_path_length,
                         n_paths=args.num_rollouts,
                         render_mode=args.render_mode)


    return paths


if __name__ == '__main__':
    args = parse_args()

    if not args.path:
        agent=args.agent
        energy=args.energy
        if args.base_path:
            top_path=os.path.join(args.base_path,agent+'/'+energy)
        else:
            top_path='./experiments_results/gym/'+agent+'/'+energy

        print(top_path)
        if 'Energy0' in energy:
            ene_sub='_E0'
        elif 'EnergyOne' in energy:
            ene_sub = '_E1'
        elif 'EnergyPoint5' in energy:
            ene_sub = '_Ep5'
        elif 'EnergyPoint1' in energy:
            ene_sub = '_Ep1'
        elif 'Energyz' in energy:
            ene_sub = '_Ez'
        elif 'EnergyPoint25' in energy:
            ene_sub = '_Ep25'
        elif 'EnergyAlt' in energy:
            ene_sub = '_Ealt'
        elif 'smallRange' in energy:
            ene_sub = '_sR'
        elif 'bigRange' in energy:
            ene_sub = '_bR'
        elif 'LessSpring' in energy:
            ene_sub = '_lS'
        elif 'MoreSpring' in energy:
            ene_sub = '_mS'
        elif 'ExSpring' in energy:
            if "T" in energy:
                ene_sub = '_maxST'
            elif "G" in energy:
                ene_sub = '_maxSG'
            else:
                ene_sub = '_maxS'
        elif 'MinSpring' in energy:
            if "T" in energy:
                ene_sub = '_minST'
            elif "G" in energy:
                ene_sub = '_minSG'
            else:
                ene_sub = '_minS'

        elif 'Symloss'in energy:
            if "SymlossGfblr" in energy:
                ene_sub = '_sLGfblr'
            elif 'SymlossG' in energy:
                ene_sub = '_sLG'
            elif 'SymlossT' in energy:
                ene_sub = '_sLT'
            else:
                ene_sub = '_sL'

        elif 'Symdup' in energy:
            if "SymdupGfblr" in energy  :
                ene_sub = '_sDGfblr'
            elif'SymdupG' in energy :
                ene_sub = '_sDG'
            elif 'SymdupT' in  energy:
                ene_sub = '_sDT'
            else:
                ene_sub = '_sD'

        version = energy.split("-")[1]
        if version!="v0":
            ene_sub = ene_sub+version

        if agent=='HalfCheetah':
            abrv='HC'
        elif 'HalfCheetahSquat' in agent and 'dof' in  agent:
            abrv='HCsquat'
            tmp_agent_name=agent.replace('HalfCheetahSquat', '')
            abrv+=tmp_agent_name
        elif 'HalfCheetah' in agent and 'dof' in  agent:
            abrv='HC'
            tmp_agent_name=agent.replace('HalfCheetah', '')
            abrv+=tmp_agent_name
        elif agent=='HalfCheetahHeavy':
            abrv = 'HCheavy'
        elif agent=='FullCheetah':
            abrv = 'FC'
        elif agent == 'FullCheetahHeavy':
            abrv = 'FCheavy'
        else:
            abrv = agent

        for experiment in os.listdir(top_path):
            exp_path=os.path.join(top_path,experiment)

            if 'TD3' not in experiment:
                base_name=abrv+ene_sub

                trial='_'+experiment.split('_')[-1]

                extract_b=True
                if 'all' not in args.tr and trial not in args.tr:
                    extract_b=False

                if extract_b:

                    for folder in os.listdir(exp_path):
                        if 'ExperimentRunner' in folder:
                            base_path=os.path.join(exp_path,folder)

                    start=args.start
                    step=args.step
                    final=args.final

                    all_checkpoint = []
                    all_name =[]
                    for ch in range(start,final+1,step):
                        specific='checkpoint_'+str(ch)
                        all_checkpoint.append(os.path.join(base_path,specific))
                        namee = base_name + '_C' + str(ch) + trial

                        all_name.append(namee)

                    for ind,chk in enumerate(all_checkpoint):
                        args.checkpoint_path=chk
                        args.name=all_name[ind]

                        paths=simulate_policy(args)

                        total_ori_reward = []
                        total_energy = []
                        total_pure_reward = []

                        action_list=[]
                        states_list = []
                        for path in paths:
                            try:
                                tmp = 0
                                tmpe=0
                                tmpPure=0
                                for i in range(len(path['infos'])):
                                    tmp = tmp + path['infos'][i]['ori_reward']
                                    tmpe = tmpe + path['infos'][i]['energy']
                                    if "HalfCheetah" in agent or "FullCheetah" in agent :
                                        if "Squat" in agent:
                                            tmpPure=tmpPure + path['infos'][i]['reward_dist']
                                        else:
                                            tmpPure=tmpPure + path['infos'][i]['reward_run']
                                    elif "Ant" in agent:
                                        if "SquaT" in agent:
                                            tmpPure = tmpPure + path['infos'][i]['reward_distance']
                                        elif "Run" in agent:
                                            tmpPure = tmpPure + path['infos'][i]['reward_forward']
                                    elif "VA" in agent:
                                        tmpPure = tmpPure + path['infos'][i]['reward_dist']/5
                                    elif "RealArm" in agent:
                                        tmpPure = tmpPure + path['infos'][i]['reward_dist']/5

                                if agent=="FullCheetahHeavy" and "v0" not in energy:
                                    speed=speed_dict[version]
                                    print("Speed: ",speed)
                                    tmp=tmp+len(path['infos'])*speed
                                    tmpPure=tmpPure+len(path['infos'])*speed

                                total_ori_reward.append(tmp)
                                total_energy.append(tmpe)
                                total_pure_reward.append(tmpPure)
                            except:
                                pass

                            path_action=path['actions']
                            if agent == 'FullCheetahHeavy' :
                                if  'T' in ene_sub:
                                    print("trot")
                                    path_action=np.asarray(path_action)
                                    path_action[:,6:9] = -1 * path_action[:,0:3]
                                    path_action[:,9:12] = -1 * path_action[:,3:6]

                                elif  'G' in ene_sub :
                                    print("gallop")
                                    path_action=np.asarray(path_action)
                                    path_action[:,6:12] = path_action[:,0:6]

                            action_list.append(path_action)
                            states_list.append(path['states'])

                        action_list=np.asarray(action_list)
                        states_list = np.asarray(states_list)
                        name = args.name
                        print(name)

                        total_energy = np.asarray(total_energy)
                        if not os.path.exists('./experiments_results/collected_actions/trajectory_npy/reward_energy_dict'):
                            os.makedirs('./experiments_results/collected_actions/trajectory_npy/reward_energy_dict',exist_ok=True)

                        if not os.path.exists('./experiments_results/collected_actions/trajectory_npy/actions_npy'):
                            os.makedirs('./experiments_results/collected_actions/trajectory_npy/actions_npy',exist_ok=True)

                        if not os.path.exists('./experiments_results/collected_actions/trajectory_npy/states_npy'):
                            os.makedirs('./experiments_results/collected_actions/trajectory_npy/states_npy',exist_ok=True)

                        if not os.path.exists(
                                './experiments_results/collected_actions/trajectory_npy/pure_reward_dict'):
                            os.makedirs('./experiments_results/collected_actions/trajectory_npy/pure_reward_dict',
                                        exist_ok=True)
                        try:
                            diagnostics = OrderedDict((
                                ('ori-return-average', np.mean(total_ori_reward)),
                                ('ori-return-min', np.min(total_ori_reward)),
                                ('ori-return-max', np.max(total_ori_reward)),
                                ('ori-return-std', np.std(total_ori_reward)),

                                ('total-energy-average', np.mean(total_energy)),
                                ('total-energy-min', np.min(total_energy)),
                                ('total-energy-max', np.max(total_energy)),
                                ('total-energy-std', np.std(total_energy)),

                            ))

                            np.save('./experiments_results/collected_actions/trajectory_npy/reward_energy_dict/' + name, diagnostics)

                        except:
                            print("error in reward energy dict saving")
                            pass

                        try:
                            diagnostics = OrderedDict((
                                ('pure-return-average', np.mean(total_pure_reward)),
                                ('pure-return-std', np.std(total_pure_reward)),
                            ))

                            np.save('./experiments_results/collected_actions/trajectory_npy/pure_reward_dict/' + name, diagnostics)

                        except:
                            pass

                        np.save('./experiments_results/collected_actions/trajectory_npy/actions_npy/' + name, action_list)
                        np.save('./experiments_results/collected_actions/trajectory_npy/states_npy/' + name, states_list)

    else:
        base_path = args.path
        trial = '_' + base_path.split('/')[-2].split('-')[-1].split('_')[-1]
        base_name = base_path.split('/')[-2].split('-')[-1].replace(trial, '')

        start = args.start
        step = args.step
        final = args.final

        all_checkpoint = []
        all_name = []
        for ch in range(start, final + 1, step):
            specific = 'checkpoint_' + str(ch)
            all_checkpoint.append(os.path.join(base_path, specific))
            namee = base_name + '_C' + str(ch) + trial

            all_name.append(namee)
        # print(all_checkpoint)
        for ind, chk in enumerate(all_checkpoint):
            args.checkpoint_path = chk
            args.name = all_name[ind]

            paths = simulate_policy(args)

            total_ori_reward = []
            total_energy = []

            action_list = []
            states_list = []
            for path in paths:
                try:
                    tmp = 0
                    tmpe = 0
                    for i in range(len(path['infos'])):
                        tmp = tmp + path['infos'][i]['ori_reward']
                        tmpe = tmpe + path['infos'][i]['energy']
                    total_ori_reward.append(tmp)
                    total_energy.append(tmpe)
                except:
                    pass

                action_list.append(path['actions'])
                states_list.append(path['states'])

            action_list = np.asarray(action_list)
            states_list = np.asarray(states_list)
            name = args.name
            print(name)

            total_energy = np.asarray(total_energy)
            if not os.path.exists('./experiments_results/collected_actions/trajectory_npy/reward_energy_dict'):
                os.makedirs('./experiments_results/collected_actions/trajectory_npy/reward_energy_dict', exist_ok=True)

            if not os.path.exists('./experiments_results/collected_actions/trajectory_npy/actions_npy'):
                os.makedirs('./experiments_results/collected_actions/trajectory_npy/actions_npy', exist_ok=True)

            if not os.path.exists('./experiments_results/collected_actions/trajectory_npy/states_npy'):
                os.makedirs('./experiments_results/collected_actions/trajectory_npy/states_npy', exist_ok=True)

            try:
                diagnostics = OrderedDict((
                    ('ori-return-average', np.mean(total_ori_reward)),
                    ('ori-return-min', np.min(total_ori_reward)),
                    ('ori-return-max', np.max(total_ori_reward)),
                    ('ori-return-std', np.std(total_ori_reward)),

                    ('total-energy-average', np.mean(total_energy)),
                    ('total-energy-min', np.min(total_energy)),
                    ('total-energy-max', np.max(total_energy)),
                    ('total-energy-std', np.std(total_energy)),
                ))

                np.save('./experiments_results/collected_actions/trajectory_npy/reward_energy_dict/' + name,
                        diagnostics)

            except:
                pass

            np.save('./experiments_results/collected_actions/trajectory_npy/actions_npy/' + name, action_list)
            np.save('./experiments_results/collected_actions/trajectory_npy/states_npy/' + name, states_list)




