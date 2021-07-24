import os
import numpy as np
import platform
from scipy.signal import butter, lfilter, freqz

file_path=os.path.abspath(os.getcwd())
if platform.system()=='Windows':
    path_list = file_path.split('\\')
else:
    path_list=file_path.split('/')

while path_list[-1] !="synergy_analysis":
    path_list.pop(-1)

cwd="/".join(path_list)
file_path=cwd

sac_color = "green"#'#7f6d5f'
td3_color = '#7f6d5f'#557f2d'
errWidth=3.5
Y_label="SEA"
TD3_color_offset=0.4
SAC_color_offset=0.4
ASA_color_scale=6

color_list=['tab:blue', 'tab:orange', 'tab:green',
             'tab:purple', 'tab:brown','tab:olive',#'tab:red',
              'tab:gray','tab:pink',
            'tab:cyan']

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=3):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def ma_filter(data, window=5):
    y=[]
    for ind,v in enumerate(data):
        if ind-window>=0:
            y.append(np.mean(data[ind-window:ind+1]))
        else:
            y.append(np.mean(data[0: ind+1]))

    return np.asarray(y)

def gauss(x, mu, a = 1, sigma = 1/6):
    return a * np.exp(-(x - mu)**2 / (2*sigma**2))

def R2():

    return r'R^{{{e:d}}}'.format(e=int(2))
def my_as_si(x, ndp):
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    m, e = s.split('e')
    return r'\times 10^{{{e:d}}}'.format(e=int(e))

def reshape_into_spatial_shape(X_input):
    assert len(X_input.shape) == 3
    trial_num, sample_l, action_dim = X_input.shape

    X_loc = X_input[0, :, :]
    X_loc=np.transpose(X_loc)

    for l in range(1, trial_num, 1):
        X_loc = np.hstack((X_loc,np.transpose(X_input[l, :, :])))

    return X_loc

def reshape_into_spt_shape(X_input):
    assert len(X_input.shape) == 3
    trial_num, sample_l, action_dim = X_input.shape

    X_loc = np.expand_dims(X_input[0, :, 0], 0)

    for i in range(1, action_dim, 1):
        X_loc = np.concatenate((X_loc, np.expand_dims(X_input[0, :, i], 0)), axis=1)

    for l in range(1, trial_num, 1):
        X_inner = np.expand_dims(X_input[l, :, 0], 0)
        for i in range(1, action_dim, 1):
            X_inner = np.concatenate((X_inner, np.expand_dims(X_input[l, :, i], 0)), axis=1)

        X_loc = np.concatenate((X_loc, X_inner), axis=0)

    return X_loc

def reshape_into_ori_shape(X_input,total_vec):
    assert len(X_input.shape) == 2
    trial_num = X_input.shape[0]
    sample_l = int(X_input.shape[1] / total_vec)
    action_dim = total_vec

    X_loc = np.expand_dims(X_input[0, 0:sample_l], 1)
    for i in range(1, action_dim, 1):
        X_loc = np.concatenate(
            (X_loc, np.expand_dims(X_input[0, sample_l * i:sample_l * (i + 1)], 1)), axis=1)
    X_loc = np.expand_dims(X_loc, 0)

    for l in range(1, trial_num, 1):
        X_inner = np.expand_dims(X_input[l, 0:sample_l], 1)
        for i in range(1, action_dim, 1):
            X_inner = np.concatenate(
                (X_inner, np.expand_dims(X_input[l, sample_l * i:sample_l * (i + 1)], 1)),
                axis=1)
        X_inner = np.expand_dims(X_inner, 0)

        X_loc = np.concatenate((X_loc, X_inner), axis=0)

    assert X_loc.shape[0] == trial_num
    assert X_loc.shape[1] == sample_l
    assert X_loc.shape[2] == action_dim

    return X_loc

all_agent_list=['HCheavy','HC','FC','HC4dof','HC5dof','HC3doff',
                'HC3dofb','HC2dof','VA','VA4dof','VA6dof','VA8dof','AntRun','AntSquaT',
                'AntSquaTRedundant','VA_bR','VA_sR','VA8dof_bR','VA8dof_sR','VA_Ez','VA4dof_Ez','VA6dof_Ez','VA8dof_Ez',
                'VA4dof_bR','VA4dof_sR','VA6dof_bR','VA6dof_sR',
                'VA_Ep5','VA4dof_Ep5','VA6dof_Ep5','VA8dof_Ep5',
                'VA_E1', 'VA4dof_E1', 'VA6dof_E1', 'VA8dof_E1',
                'HC_Ez','HC4dof_Ez','HC5dof_Ez','HC3doff_Ez','HC3dofb_Ez','HC2dof_Ez',
                'HC_E1','HC4dof_E1','HC5dof_E1','HC3doff_E1','HC3dofb_E1','HC2dof_E1',
                'HC5dofv2', 'HC5dofv3', 'HC5dofv4', 'HC5dofv5', 'HC5dofv6',
                'HC2dofv2', 'HC2dofv3', 'HC2dofv4', 'HC2dofv5',
                'HCsquat2dof', 'HCsquat4dof', 'HCsquat6dof',
                'HCsquat4dof_Ep1', 'HCsquat6dof_Ep1',
                'HCsquat2dof_Ep25', 'HCsquat4dof_Ep25', 'HCsquat6dof_Ep25',
                'HCsquat2dof_Ez', 'HCsquat4dof_Ez', 'HCsquat6dof_Ez',
                'HCsquat2dof_Ealt', 'HCsquat4dof_Ealt', 'HCsquat6dof_Ealt',
                'RealArm3dof', 'RealArm4dof', 'RealArm5dof', 'RealArm6dof', 'RealArm7dof',
                'RealArm4dofMinE', 'RealArm5dofMinE', 'RealArm4dofLT', 'RealArm5dofLT','Walker2d'
                ,'Walker2d_sL','HC_sL','Walker2d_sD','FCheavy','FC_sLT','FC_sLG','FC_sLGfblr','FCheavy_sLT','FCheavy_sLG','FCheavy_lS','FCheavy_mS',
                'FCheavy_minS','FCheavy_maxS','RealArm7dofE0v1','RealArm7dofE0v2','RealArm7dofE0v3',
                'RealArm7dofE0v4','RealArm7dofE0v5','RealArm7dofE0v6','RealArm7dofE0v7','RealArm7dofE0v8','RealArm7dofE0v9',
                'RealArm6dofE0v1','RealArm6dofE0v2',
                'RealArm5dofE0v1','RealArm5dofE0v2',
                'RealArm4dofE0v1','RealArm4dofE0v2',
                'RealArm3dofE0v1','RealArm3dofE0v2',
                'RealArm7dofE0_TD3v9'

                ]#'HC_Ez','HC4dof_Ez','HC5dof_Ez','HC3doff_Ez','HC3dofb_Ez','HC2dof_Ez','A','Antheavy','Ctp','G'

spt_agent_list=['HCheavy','HC','A','Antheavy','FC','Ctp','G','HC4dof','HC5dof','HC3doff','HC3dofb',
                'HC2dof','VA','VA4dof','fVA4dof','VA6dof','VA8dof','AntRun','AntSquaT','AntSquaTRedundant',
                'HC5dofv2', 'HC5dofv3', 'HC5dofv4', 'HC5dofv5', 'HC5dofv6',
                'HC2dofv2', 'HC2dofv3', 'HC2dofv4', 'HC2dofv5',
                'HCsquat2dof', 'HCsquat4dof', 'HCsquat6dof',
                'RealArm3dof', 'RealArm4dof', 'RealArm5dof', 'RealArm6dof', 'RealArm7dof',
            'RealArm4dofMinE', 'RealArm5dofMinE', 'RealArm4dofLT', 'RealArm5dofLT','Walker2d',
                'Walker2d_sL','Walker2d_sD','FC_sLT','FC_sLG','FC_sLGfblr','FCheavy','FCheavy_sLT','FCheavy_sLG','FCheavy_lS','FCheavy_mS',
                'FCheavy_minS','FCheavy_maxS', 'FCheavy_minSTv0','FCheavy_maxSTv0', 'FCheavy_minSGv0','FCheavy_maxSGv0'
                ]

spt_energy_list=['E0','E0_TD3','E1','E1_TD3','E0_spt_alphap1','E0_spt_alpha0',
                'E0_spt_alpha1','E0_spt_alpha2', 'E0_spt_alpha3', 'E0_spt_alpha5',
                'E0_spt_alpha10','E0_TD3_spt_alphap1', 'E0_TD3_spt_alpha0',
                'E0_TD3_spt_alpha1','E0_TD3_spt_alpha5','Ez','Ez_TD3','Ep5','Ep1','Ep1_TD3',
                'Ep25','Ep25_TD3','Ealt','Ealt_TD3','bR','sR','bR_TD3','sR_TD3','sL','sD','sLT','sLG','sLGfblr',
                'sDT','sDG','sDGfblr',
                 'E0v1','E0v2','E0v3','E0v4','E0v5','E0v6','E0v00','E0v45','E0v65','E0v7','E0v8','E0v9',
                'E0_TD3v1','E0_TD3v2','E0_TD3v3','E0_TD3v4','E0_TD3v5','E0_TD3v6','E0_TD3v7','E0_TD3v8',
                 'E0_TD3v9',
                 'sLTv1','sLTv2','sLTv3','sLTv4','sLTv5','sLTv6',
                 'sLGv1','sLGv2','sLGv3','sLGv4','sLGv5','sLGv6',
                 'lS', 'lSv2', 'lSv4', 'lSv6',
                 'mS', 'mSv2', 'mSv4', 'mSv6',
                'maxS', 'maxSv2', 'maxSv4', 'maxSv6','maxSv00', 'maxSv45','maxSv65',
                'minS', 'minSv2', 'minSv4', 'minSv6','minSv00', 'minSv45','minSv65',
                'minSTv0', 'minSTv2', 'minSTv4', 'minSTv6',
                'minSGv0', 'minSGv2', 'minSGv4', 'minSGv6',
                'maxSTv0', 'maxSTv2', 'maxSTv4', 'maxSTv6',
                'maxSGv0', 'maxSGv2', 'maxSGv4', 'maxSGv6',
                ]

agent_info_dict={
    'HC':{'total_vec' : 6,'total_chk':30,
          'ori_final' : 3000,'ori_begin' : 100, 'ori_step' : 100,
          'x_speed_index':8,'desired_dist':500},
    'HCsquat2dof':{'total_vec' : 2,'total_chk':30,
              'ori_final' : 120,'ori_begin' : 4, 'ori_step' : 4,
              'x_speed_index':None,'desired_dist':1000,
                   'dll': 550, 'truncated_start': 250,
                   'agentt_folder': 'HCsquat2dof'},
    'HCsquat2dof_Ep25':{'total_vec' : 2,'total_chk':30,
              'ori_final' : 120,'ori_begin' : 4, 'ori_step' : 4,
              'x_speed_index':None,'desired_dist':1000,
                   'dll': 550, 'truncated_start': 250,
                   'agentt_folder': 'HCsquat2dof_Ep25'},
    'HCsquat2dof_Ez':{'total_vec' : 2,'total_chk':30,
              'ori_final' : 120,'ori_begin' : 4, 'ori_step' : 4,
              'x_speed_index':None,'desired_dist':1000,
                   'dll': 550, 'truncated_start': 250,
                   'agentt_folder': 'HCsquat2dof_Ez'},
    'HCsquat2dof_Ealt': {'total_vec': 2, 'total_chk': 30,
                       'ori_final': 120, 'ori_begin': 4, 'ori_step': 4,
                       'x_speed_index': None, 'desired_dist': 1000,
                       'dll': 550, 'truncated_start': 250,
                       'agentt_folder': 'HCsquat2dof_Ealt'},

    'HCsquat4dof':{'total_vec' : 4,'total_chk':30,
              'ori_final' : 120,'ori_begin' : 4, 'ori_step' : 4,
              'x_speed_index':None,'desired_dist':1000,
                   'dll': 550, 'truncated_start': 250,
                   'agentt_folder': 'HCsquat4dof'},

    'HCsquat6dof':{'total_vec' : 6,'total_chk':30,
              'ori_final' : 120,'ori_begin' : 4, 'ori_step' : 4,
              'x_speed_index':None,'desired_dist':1000,
                   'dll': 550, 'truncated_start': 250,
                   'agentt_folder': 'HCsquat6dof'},
    'HCsquat4dof_Ep1': {'total_vec': 4, 'total_chk': 30,
                    'ori_final': 120, 'ori_begin': 4, 'ori_step': 4,
                    'x_speed_index': None, 'desired_dist': 1000,
                    'dll': 550, 'truncated_start': 250,
                    'agentt_folder': 'HCsquat4dof_Ep1'},
    'HCsquat4dof_Ez': {'total_vec': 4, 'total_chk': 30,
                    'ori_final': 120, 'ori_begin': 4, 'ori_step': 4,
                    'x_speed_index': None, 'desired_dist': 1000,
                    'dll': 550, 'truncated_start': 250,
                    'agentt_folder': 'HCsquat4dof_Ez'},
    'HCsquat4dof_Ep25': {'total_vec': 4, 'total_chk': 30,
                        'ori_final': 120, 'ori_begin': 4, 'ori_step': 4,
                        'x_speed_index': None, 'desired_dist': 1000,
                        'dll': 550, 'truncated_start': 250,
                        'agentt_folder': 'HCsquat4dof_Ep25'},
    'HCsquat4dof_Ealt': {'total_vec': 4, 'total_chk': 30,
                         'ori_final': 120, 'ori_begin': 4, 'ori_step': 4,
                         'x_speed_index': None, 'desired_dist': 1000,
                         'dll': 550, 'truncated_start': 250,
                         'agentt_folder': 'HCsquat4dof_Ealt'},

    'HCsquat6dof_Ep1': {'total_vec': 6, 'total_chk': 30,
                    'ori_final': 120, 'ori_begin': 4, 'ori_step': 4,
                    'x_speed_index': None, 'desired_dist': 1000,
                    'dll': 550, 'truncated_start': 250,
                    'agentt_folder': 'HCsquat6dof_Ep1'},
    'HCsquat6dof_Ep25': {'total_vec': 6, 'total_chk': 30,
                        'ori_final': 120, 'ori_begin': 4, 'ori_step': 4,
                        'x_speed_index': None, 'desired_dist': 1000,
                        'dll': 550, 'truncated_start': 250,
                        'agentt_folder': 'HCsquat6dof_Ep25'},
    'HCsquat6dof_Ez': {'total_vec': 6, 'total_chk': 30,
                    'ori_final': 120, 'ori_begin': 4, 'ori_step': 4,
                    'x_speed_index': None, 'desired_dist': 1000,
                    'dll': 550, 'truncated_start': 250,
                    'agentt_folder': 'HCsquat6dof_Ez'},
    'HCsquat6dof_Ealt': {'total_vec': 6, 'total_chk': 30,
                       'ori_final': 120, 'ori_begin': 4, 'ori_step': 4,
                       'x_speed_index': None, 'desired_dist': 1000,
                       'dll': 550, 'truncated_start': 250,
                       'agentt_folder': 'HCsquat6dof_Ealt'},

    'FC': {'total_vec': 12, 'total_chk': 30,
           'ori_final': 3000, 'ori_begin': 100, 'ori_step': 100,
           'x_speed_index': 14, 'desired_dist': 500},
    'FC_sLT': {'total_vec': 12, 'total_chk': 30,
               'ori_final': 1500, 'ori_begin': 100, 'ori_step': 100,
               'x_speed_index': 14, 'desired_dist': 500},

    'FCheavy': {'total_vec': 12, 'total_chk': 20,
                'ori_final': 400, 'ori_begin': 20, 'ori_step': 20,
                'x_speed_index': 14, 'desired_dist': 500},

    'HC5dof': {'total_vec': 5, 'total_chk': 30,
           'ori_final': 3000, 'ori_begin': 100, 'ori_step': 100,
           'x_speed_index': 7, 'desired_dist': 500,
            'agentt_folder' : 'HC5dof'},
    'HC5dofv2': {'total_vec': 5, 'total_chk': 30,
               'ori_final': 3000, 'ori_begin': 100, 'ori_step': 100,
               'x_speed_index': 7, 'desired_dist': 500,
               'agentt_folder': 'HC5dofv2'},
    'HC5dofv3': {'total_vec': 5, 'total_chk': 30,
               'ori_final': 3000, 'ori_begin': 100, 'ori_step': 100,
               'x_speed_index': 7, 'desired_dist': 500,
               'agentt_folder': 'HC5dofv3'},
    'HC5dofv4': {'total_vec': 5, 'total_chk': 30,
               'ori_final': 3000, 'ori_begin': 100, 'ori_step': 100,
               'x_speed_index': 7, 'desired_dist': 500,
               'agentt_folder': 'HC5dofv4'},
    'HC5dofv5': {'total_vec': 5, 'total_chk': 30,
               'ori_final': 3000, 'ori_begin': 100, 'ori_step': 100,
               'x_speed_index': 7, 'desired_dist': 500,
               'agentt_folder': 'HC5dofv5'},
    'HC5dofv6': {'total_vec': 5, 'total_chk': 30,
               'ori_final': 3000, 'ori_begin': 100, 'ori_step': 100,
               'x_speed_index': 7, 'desired_dist': 500,
               'agentt_folder': 'HC5dofv6'},
    'HC4dof': {'total_vec': 4, 'total_chk': 30,
               'ori_final': 3000, 'ori_begin': 100, 'ori_step': 100,
               'x_speed_index': 6, 'desired_dist': 500, 'agentt_folder': 'HC5dof'},
    'HC3doff': {'total_vec': 3, 'total_chk': 30,
               'ori_final': 3000, 'ori_begin': 100, 'ori_step': 100,
               'x_speed_index': 5, 'desired_dist': 500, 'agentt_folder': 'HC3doff'},
    'HC3dofb': {'total_vec': 3, 'total_chk': 30,
                'ori_final': 3000, 'ori_begin': 100, 'ori_step': 100,
                'x_speed_index': 5, 'desired_dist': 500, 'agentt_folder': 'HC3dofb'},
    'HC2dof': {'total_vec': 2, 'total_chk': 30,
               'ori_final': 3000, 'ori_begin': 100, 'ori_step': 100,
               'x_speed_index': 4, 'desired_dist': 500, 'agentt_folder': 'HC2dof'},
    'HC2dofv2': {'total_vec': 2, 'total_chk': 30,
               'ori_final': 3000, 'ori_begin': 100, 'ori_step': 100,
               'x_speed_index': 4, 'desired_dist': 500, 'agentt_folder': 'HC2dofv2'},
    'HC2dofv3': {'total_vec': 2, 'total_chk': 30,
               'ori_final': 3000, 'ori_begin': 100, 'ori_step': 100,
               'x_speed_index': 4, 'desired_dist': 500, 'agentt_folder': 'HC2dofv3'},
    'HC2dofv4': {'total_vec': 2, 'total_chk': 30,
               'ori_final': 3000, 'ori_begin': 100, 'ori_step': 100,
               'x_speed_index': 4, 'desired_dist': 500, 'agentt_folder': 'HC2dofv4'},
    'HC2dofv5': {'total_vec': 2, 'total_chk': 30,
               'ori_final': 3000, 'ori_begin': 100, 'ori_step': 100,
               'x_speed_index': 4, 'desired_dist': 500, 'agentt_folder': 'HC2dofv5'},

    'HC5dof_Ez': {'total_vec': 5, 'total_chk': 30,
               'ori_final': 3000, 'ori_begin': 100, 'ori_step': 100,
               'x_speed_index': 7, 'desired_dist': 500,
               'agentt_folder': 'HC5dof'},
    'HC4dof_Ez': {'total_vec': 4, 'total_chk': 30,
               'ori_final': 3000, 'ori_begin': 100, 'ori_step': 100,
               'x_speed_index': 6, 'desired_dist': 500, 'agentt_folder': 'HC4dof'},
    'HC3doff_Ez': {'total_vec': 3, 'total_chk': 30,
                'ori_final': 3000, 'ori_begin': 100, 'ori_step': 100,
                'x_speed_index': 5, 'desired_dist': 500, 'agentt_folder': 'HC3doff'},
    'HC3dofb_Ez': {'total_vec': 3, 'total_chk': 30,
                'ori_final': 3000, 'ori_begin': 100, 'ori_step': 100,
                'x_speed_index': 5, 'desired_dist': 500, 'agentt_folder': 'HC3dofb'},
    'HC2dof_Ez': {'total_vec': 2, 'total_chk': 30,
               'ori_final': 3000, 'ori_begin': 100, 'ori_step': 100,
               'x_speed_index': 4, 'desired_dist': 500, 'agentt_folder': 'HC2dof'},

    'HC5dof_E1': {'total_vec': 5, 'total_chk': 30,
                  'ori_final': 3000, 'ori_begin': 100, 'ori_step': 100,
                  'x_speed_index': 7, 'desired_dist': 500,
                  'agentt_folder': 'HC5dof'},
    'HC4dof_E1': {'total_vec': 4, 'total_chk': 30,
                  'ori_final': 3000, 'ori_begin': 100, 'ori_step': 100,
                  'x_speed_index': 6, 'desired_dist': 500, 'agentt_folder': 'HC5dof'},
    'HC3doff_E1': {'total_vec': 3, 'total_chk': 30,
                   'ori_final': 3000, 'ori_begin': 100, 'ori_step': 100,
                   'x_speed_index': 5, 'desired_dist': 500, 'agentt_folder': 'HC3doff'},
    'HC3dofb_E1': {'total_vec': 3, 'total_chk': 30,
                   'ori_final': 3000, 'ori_begin': 100, 'ori_step': 100,
                   'x_speed_index': 5, 'desired_dist': 500, 'agentt_folder': 'HC3dofb'},
    'HC2dof_E1': {'total_vec': 2, 'total_chk': 30,
                  'ori_final': 3000, 'ori_begin': 100, 'ori_step': 100,
                  'x_speed_index': 4, 'desired_dist': 500, 'agentt_folder': 'HC2dof'},

    'RealArm3dof': {'total_vec': 3, 'total_chk': 30,
                    'ori_final': 90, 'ori_begin': 3, 'ori_step': 3,
                    'x_speed_index': None, 'desired_dist': 500, 'agentt_folder': 'RealArm3dof',
                    'joint_list': ['s_flexion', 's_rotation', 'e_flexion'], 'dll': 400, 'truncated_start': 300,
                    'C_y_lim': [-3, 3], 'W_y_lim': [-0.2, 0.2], 'recon_y_lim': [-0.5, 0.5]},
    'RealArm4dof': {'total_vec': 4, 'total_chk': 30,
                    'ori_final': 90, 'ori_begin': 3, 'ori_step': 3,
                    'x_speed_index': None, 'desired_dist': 500, 'agentt_folder': 'RealArm4dof',
                    'joint_list': ['s_abduction', 's_flexion', 's_rotation', 'e_flexion'], 'dll': 400,
                    'truncated_start': 300,
                    'C_y_lim': [-3, 3], 'W_y_lim': [-0.2, 0.2], 'recon_y_lim': [-0.5, 0.5]},
    'RealArm5dof': {'total_vec': 5, 'total_chk': 30,
                    'ori_final': 90, 'ori_begin': 3, 'ori_step': 3,
                    'x_speed_index': None, 'desired_dist': 500, 'agentt_folder': 'RealArm5dof',
                    'joint_list': ['s_abduction', 's_flexion', 's_rotation', 'e_flexion',
                                   'w_flexion'], 'dll': 400, 'truncated_start': 300,
                    'C_y_lim': [-3, 3], 'W_y_lim': [-0.2, 0.2], 'recon_y_lim': [-0.5, 0.5]},

    'RealArm4dofMinE': {'total_vec': 4, 'total_chk': 30,
                    'ori_final': 90, 'ori_begin': 3, 'ori_step': 3,
                    'x_speed_index': None, 'desired_dist': 500, 'agentt_folder': 'RealArm4dofMinE',
                    'joint_list': ['s_abduction', 's_flexion', 's_rotation', 'e_flexion'], 'dll': 400,
                    'truncated_start': 300,
                    'C_y_lim': [-3, 3], 'W_y_lim': [-0.2, 0.2], 'recon_y_lim': [-0.5, 0.5]},
    'RealArm5dofMinE': {'total_vec': 5, 'total_chk': 30,
                    'ori_final': 90, 'ori_begin': 3, 'ori_step': 3,
                    'x_speed_index': None, 'desired_dist': 500, 'agentt_folder': 'RealArm5dofMinE',
                    'joint_list': ['s_abduction', 's_flexion', 's_rotation', 'e_flexion',
                                   'w_flexion'], 'dll': 400, 'truncated_start': 300,
                    'C_y_lim': [-3, 3], 'W_y_lim': [-0.2, 0.2], 'recon_y_lim': [-0.5, 0.5]},
    'RealArm4dofLT': {'total_vec': 4, 'total_chk': 30,
                    'ori_final': 90, 'ori_begin': 3, 'ori_step': 3,
                    'x_speed_index': None, 'desired_dist': 500, 'agentt_folder': 'RealArm4dofLT',
                    'joint_list': ['s_abduction', 's_flexion', 's_rotation', 'e_flexion'], 'dll': 400,
                    'truncated_start': 300,
                    'C_y_lim': [-3, 3], 'W_y_lim': [-0.2, 0.2], 'recon_y_lim': [-0.5, 0.5]},
    'RealArm5dofLT': {'total_vec': 5, 'total_chk': 30,
                    'ori_final': 90, 'ori_begin': 3, 'ori_step': 3,
                    'x_speed_index': None, 'desired_dist': 500, 'agentt_folder': 'RealArm5dofLT',
                    'joint_list': ['s_abduction', 's_flexion', 's_rotation', 'e_flexion',
                                   'w_flexion'], 'dll': 400, 'truncated_start': 300,
                    'C_y_lim': [-3, 3], 'W_y_lim': [-0.2, 0.2], 'recon_y_lim': [-0.5, 0.5]},
    'RealArm6dof': {'total_vec': 6, 'total_chk': 30,
                    'ori_final': 90, 'ori_begin': 3, 'ori_step': 3,
                    'x_speed_index': None, 'desired_dist': 500, 'agentt_folder': 'RealArm6dof',
                    'joint_list':['s_abduction','s_flexion', 's_rotation','e_flexion', 'w_abduction','w_flexion'],
                    'dll': 400, 'truncated_start': 300,
                    'C_y_lim': [-3, 3], 'W_y_lim': [-0.2, 0.2], 'recon_y_lim': [-0.5, 0.5]},
    'RealArm7dof': {'total_vec': 7, 'total_chk': 30,
                    'ori_final': 90, 'ori_begin': 3, 'ori_step': 3,
                    'x_speed_index': None, 'desired_dist': 500, 'agentt_folder': 'RealArm7dof',
                    'joint_list':['s_abduction','s_flexion', 's_rotation','e_flexion','e_pronation', 'w_abduction','w_flexion'],
                    'dll': 400, 'truncated_start': 300,
                    'C_y_lim': [-3, 3], 'W_y_lim': [-0.2, 0.2], 'recon_y_lim': [-0.5, 0.5]},

    'VA': {'total_vec': 2, 'total_chk': 30,
           'ori_final': 30, 'ori_begin': 1, 'ori_step': 1,
           'x_speed_index': None, 'desired_dist': 500, 'agentt_folder': 'VA',
           'joint_list': ['shoulder', 'elbow'], 'dll': 400, 'truncated_start': 300,
           'C_y_lim': [-3, 3], 'W_y_lim': [-0.2, 0.2], 'recon_y_lim': [-0.5, 0.5]},
    'VA_bR': {'total_vec': 2, 'total_chk': 30,
           'ori_final': 30, 'ori_begin': 1, 'ori_step': 1,
           'x_speed_index': None, 'desired_dist': 500, 'agentt_folder': 'VA_bR',
           'joint_list': ['shoulder', 'elbow'], 'dll': 400, 'truncated_start': 300,
           'C_y_lim': [-3, 3], 'W_y_lim': [-0.2, 0.2], 'recon_y_lim': [-0.5, 0.5]},
    'VA_sR': {'total_vec': 2, 'total_chk': 30,
           'ori_final': 30, 'ori_begin': 1, 'ori_step': 1,
           'x_speed_index': None, 'desired_dist': 500, 'agentt_folder': 'VA_sR',
           'joint_list': ['shoulder', 'elbow'], 'dll': 400, 'truncated_start': 300,
           'C_y_lim': [-3, 3], 'W_y_lim': [-0.2, 0.2], 'recon_y_lim': [-0.5, 0.5]},

    'VA4dof': {'total_vec': 4, 'total_chk': 30,
               'ori_final': 30, 'ori_begin': 1, 'ori_step': 1,
               'x_speed_index': None, 'desired_dist': 500, 'agentt_folder': 'VA4dof',
               'joint_list': ['shoulder', 'shoulder2', 'elbow', 'elbow2'], 'dll': 400, 'truncated_start': 300,
               'C_y_lim': [-3, 3], 'W_y_lim': [-0.2, 0.2], 'recon_y_lim': [-0.5, 0.5]},
    'VA4dof_bR': {'total_vec': 4, 'total_chk': 30,
                   'ori_final': 30, 'ori_begin': 1, 'ori_step': 1,
                   'x_speed_index': None, 'desired_dist': 500, 'agentt_folder': 'VA4dof_bR',
                   'joint_list': ['shoulder', 'shoulder2', 'elbow', 'elbow2'], 'dll': 400, 'truncated_start': 300,
                   'C_y_lim': [-3, 3], 'W_y_lim': [-0.2, 0.2], 'recon_y_lim': [-0.5, 0.5]},
    'VA4dof_sR': {'total_vec': 4, 'total_chk': 30,
                   'ori_final': 30, 'ori_begin': 1, 'ori_step': 1,
                   'x_speed_index': None, 'desired_dist': 500, 'agentt_folder': 'VA4dof_sR',
                   'joint_list': ['shoulder', 'shoulder2', 'elbow', 'elbow2'], 'dll': 400, 'truncated_start': 300,
                   'C_y_lim': [-3, 3], 'W_y_lim': [-0.2, 0.2], 'recon_y_lim': [-0.5, 0.5]},
    'VA6dof': {'total_vec': 6, 'total_chk': 30,
               'ori_final': 390, 'ori_begin': 13, 'ori_step': 13,
               'x_speed_index': None, 'desired_dist': 500, 'agentt_folder': 'VA6dof',
               'joint_list': ['shoulder','shoulder2', 'elbow','elbow2','elbow3','elbow4'],
               'dll': 400, 'truncated_start': 300,
               'C_y_lim': [-3, 3], 'W_y_lim': [-0.2, 0.2], 'recon_y_lim': [-0.5, 0.5]},
    'VA6dof_bR': {'total_vec': 6, 'total_chk': 30,
               'ori_final': 390, 'ori_begin': 13, 'ori_step': 13,
               'x_speed_index': None, 'desired_dist': 500, 'agentt_folder': 'VA6dof_bR',
               'joint_list': ['shoulder', 'shoulder2', 'elbow', 'elbow2', 'elbow3', 'elbow4'],
               'dll': 400, 'truncated_start': 300,
               'C_y_lim': [-3, 3], 'W_y_lim': [-0.2, 0.2], 'recon_y_lim': [-0.5, 0.5]},
    'VA6dof_sR': {'total_vec': 6, 'total_chk': 30,
               'ori_final': 390, 'ori_begin': 13, 'ori_step': 13,
               'x_speed_index': None, 'desired_dist': 500, 'agentt_folder': 'VA6dof_sR',
               'joint_list': ['shoulder', 'shoulder2', 'elbow', 'elbow2', 'elbow3', 'elbow4'],
               'dll': 400, 'truncated_start': 300,
               'C_y_lim': [-3, 3], 'W_y_lim': [-0.2, 0.2], 'recon_y_lim': [-0.5, 0.5]},
    'VA8dof': {'total_vec': 8, 'total_chk': 30,
               'ori_final': 390, 'ori_begin': 13, 'ori_step': 13,
               'x_speed_index': None, 'desired_dist': 500, 'agentt_folder': 'VA8dof',
               'joint_list': ['shoulder','shoulder2','shoulder3','shoulder4', 'elbow','elbow2','elbow3','elbow4'],
               'dll': 400, 'truncated_start': 300,
               'C_y_lim': [-3, 3], 'W_y_lim': [-0.2, 0.2], 'recon_y_lim': [-0.5, 0.5]},
    'VA8dof_sR': {'total_vec': 8, 'total_chk': 30,
               'ori_final': 390, 'ori_begin': 13, 'ori_step': 13,
               'x_speed_index': None, 'desired_dist': 500, 'agentt_folder': 'VA8dof_bR',
               'joint_list': ['shoulder', 'shoulder2', 'shoulder3', 'shoulder4', 'elbow', 'elbow2', 'elbow3', 'elbow4'],
               'dll': 400, 'truncated_start': 300,
               'C_y_lim': [-3, 3], 'W_y_lim': [-0.2, 0.2], 'recon_y_lim': [-0.5, 0.5]},
    'VA8dof_bR': {'total_vec': 8, 'total_chk': 30,
               'ori_final': 390, 'ori_begin': 13, 'ori_step': 13,
               'x_speed_index': None, 'desired_dist': 500, 'agentt_folder': 'VA8dof_sR',
               'joint_list': ['shoulder', 'shoulder2', 'shoulder3', 'shoulder4', 'elbow', 'elbow2', 'elbow3', 'elbow4'],
               'dll': 400, 'truncated_start': 300,
               'C_y_lim': [-3, 3], 'W_y_lim': [-0.2, 0.2], 'recon_y_lim': [-0.5, 0.5]},
    'VA_Ez': {'total_vec': 2, 'total_chk': 30,
           'ori_final': 30, 'ori_begin': 1, 'ori_step': 1,
           'x_speed_index': None, 'desired_dist': 500, 'agentt_folder': 'VA',
           'joint_list': ['shoulder', 'elbow'], 'dll': 400, 'truncated_start': 300,
           'C_y_lim': [-3, 3], 'W_y_lim': [-0.2, 0.2], 'recon_y_lim': [-0.5, 0.5]},
    'VA4dof_Ez': {'total_vec': 4, 'total_chk': 30,
               'ori_final': 30, 'ori_begin': 1, 'ori_step': 1,
               'x_speed_index': None, 'desired_dist': 500, 'agentt_folder': 'VA4dof',
               'joint_list': ['shoulder', 'shoulder2', 'elbow', 'elbow2'], 'dll': 400, 'truncated_start': 300,
               'C_y_lim': [-3, 3], 'W_y_lim': [-0.2, 0.2], 'recon_y_lim': [-0.5, 0.5]},
    'VA6dof_Ez': {'total_vec': 6, 'total_chk': 30,
               'ori_final': 390, 'ori_begin': 13, 'ori_step': 13,
               'x_speed_index': None, 'desired_dist': 500, 'agentt_folder': 'VA6dof',
               'joint_list': ['shoulder', 'shoulder2', 'elbow', 'elbow2', 'elbow3', 'elbow4'],
               'dll': 400, 'truncated_start': 300,
               'C_y_lim': [-3, 3], 'W_y_lim': [-0.2, 0.2], 'recon_y_lim': [-0.5, 0.5]},
    'VA8dof_Ez': {'total_vec': 8, 'total_chk': 30,
               'ori_final': 390, 'ori_begin': 13, 'ori_step': 13,
               'x_speed_index': None, 'desired_dist': 500, 'agentt_folder': 'VA8dof',
               'joint_list': ['shoulder', 'shoulder2', 'shoulder3', 'shoulder4', 'elbow', 'elbow2', 'elbow3', 'elbow4'],
               'dll': 400, 'truncated_start': 300,
               'C_y_lim': [-3, 3], 'W_y_lim': [-0.2, 0.2], 'recon_y_lim': [-0.5, 0.5]},
    'VA_E1': {'total_vec': 2, 'total_chk': 30,
              'ori_final': 30, 'ori_begin': 1, 'ori_step': 1,
              'x_speed_index': None, 'desired_dist': 500, 'agentt_folder': 'VA',
              'joint_list': ['shoulder', 'elbow'], 'dll': 400, 'truncated_start': 300,
              'C_y_lim': [-3, 3], 'W_y_lim': [-0.2, 0.2], 'recon_y_lim': [-0.5, 0.5]},
    'VA4dof_E1': {'total_vec': 4, 'total_chk': 30,
                  'ori_final': 30, 'ori_begin': 1, 'ori_step': 1,
                  'x_speed_index': None, 'desired_dist': 500, 'agentt_folder': 'VA4dof',
                  'joint_list': ['shoulder', 'shoulder2', 'elbow', 'elbow2'], 'dll': 400, 'truncated_start': 300,
                  'C_y_lim': [-3, 3], 'W_y_lim': [-0.2, 0.2], 'recon_y_lim': [-0.5, 0.5]},
    'VA6dof_E1': {'total_vec': 6, 'total_chk': 30,
                  'ori_final': 390, 'ori_begin': 13, 'ori_step': 13,
                  'x_speed_index': None, 'desired_dist': 500, 'agentt_folder': 'VA6dof',
                  'joint_list': ['shoulder', 'shoulder2', 'elbow', 'elbow2', 'elbow3', 'elbow4'],
                  'dll': 400, 'truncated_start': 300,
                  'C_y_lim': [-3, 3], 'W_y_lim': [-0.2, 0.2], 'recon_y_lim': [-0.5, 0.5]},
    'VA8dof_E1': {'total_vec': 8, 'total_chk': 30,
                  'ori_final': 390, 'ori_begin': 13, 'ori_step': 13,
                  'x_speed_index': None, 'desired_dist': 500, 'agentt_folder': 'VA8dof',
                  'joint_list': ['shoulder', 'shoulder2', 'shoulder3', 'shoulder4', 'elbow', 'elbow2', 'elbow3',
                                 'elbow4'],
                  'dll': 400, 'truncated_start': 300,
                  'C_y_lim': [-3, 3], 'W_y_lim': [-0.2, 0.2], 'recon_y_lim': [-0.5, 0.5]},
    'VA_Ep5': {'total_vec': 2, 'total_chk': 30,
              'ori_final': 30, 'ori_begin': 1, 'ori_step': 1,
              'x_speed_index': None, 'desired_dist': 500, 'agentt_folder': 'VA',
              'joint_list': ['shoulder', 'elbow'], 'dll': 400, 'truncated_start': 300,
              'C_y_lim': [-3, 3], 'W_y_lim': [-0.2, 0.2], 'recon_y_lim': [-0.5, 0.5]},
    'VA4dof_Ep5': {'total_vec': 4, 'total_chk': 30,
                  'ori_final': 30, 'ori_begin': 1, 'ori_step': 1,
                  'x_speed_index': None, 'desired_dist': 500, 'agentt_folder': 'VA4dof',
                  'joint_list': ['shoulder', 'shoulder2', 'elbow', 'elbow2'], 'dll': 400, 'truncated_start': 300,
                  'C_y_lim': [-3, 3], 'W_y_lim': [-0.2, 0.2], 'recon_y_lim': [-0.5, 0.5]},
    'VA6dof_Ep5': {'total_vec': 6, 'total_chk': 30,
                  'ori_final': 390, 'ori_begin': 13, 'ori_step': 13,
                  'x_speed_index': None, 'desired_dist': 500, 'agentt_folder': 'VA6dof',
                  'joint_list': ['shoulder', 'shoulder2', 'elbow', 'elbow2', 'elbow3', 'elbow4'],
                  'dll': 400, 'truncated_start': 300,
                  'C_y_lim': [-3, 3], 'W_y_lim': [-0.2, 0.2], 'recon_y_lim': [-0.5, 0.5]},
    'VA8dof_Ep5': {'total_vec': 8, 'total_chk': 30,
                  'ori_final': 390, 'ori_begin': 13, 'ori_step': 13,
                  'x_speed_index': None, 'desired_dist': 500, 'agentt_folder': 'VA8dof',
                  'joint_list': ['shoulder', 'shoulder2', 'shoulder3', 'shoulder4', 'elbow', 'elbow2', 'elbow3',
                                 'elbow4'],
                  'dll': 400, 'truncated_start': 300,
                  'C_y_lim': [-3, 3], 'W_y_lim': [-0.2, 0.2], 'recon_y_lim': [-0.5, 0.5]},
    'AntRun': {'total_vec': 8, 'total_chk': 20,
               'ori_final': 300, 'ori_begin': 15, 'ori_step': 15,
               'x_speed_index': 13, 'desired_dist': 1000, 'agentt_folder': 'AntRun',
               'joint_list': ['hip_1','ankle_1','hip_2','ankle_2','hip_3','ankle_3','hip_4','ankle_4'],
               'dll': 400, 'truncated_start': 300},
    'AntSquaTRedundant': {'total_vec': 8, 'total_chk': 25,
               'ori_final': 500, 'ori_begin': 20, 'ori_step': 20,
               'x_speed_index': None, 'desired_dist': 1000, 'agentt_folder': 'AntSquaTRedundant',
               'joint_list': ['hip_1', 'ankle_1', 'hip_2', 'ankle_2', 'hip_3', 'ankle_3', 'hip_4', 'ankle_4'],
               'dll': 550, 'truncated_start': 250},
    'AntSquaT': {'total_vec': 8, 'total_chk': 25,
                  'ori_final': 500, 'ori_begin': 20, 'ori_step': 20,
                  'x_speed_index': None, 'desired_dist': 1000, 'agentt_folder': 'AntSquaT',
                  'joint_list': ['hip_1', 'ankle_1', 'hip_2', 'ankle_2', 'hip_3', 'ankle_3', 'hip_4',
                                 'ankle_4'],
                  'dll': 550, 'truncated_start': 250},
    'Walker2d': {'total_vec': 6, 'total_chk': 10,
           'ori_final': 1000, 'ori_begin': 100, 'ori_step': 100,
           'x_speed_index': None, 'desired_dist': 1000,'dll': 240},#8

}



agent_info_dict['RealArm7dofE0v1']=agent_info_dict['RealArm7dof']
agent_info_dict['RealArm7dofE0v2']=agent_info_dict['RealArm7dof']
agent_info_dict['RealArm7dofE0v3']=agent_info_dict['RealArm7dof']
agent_info_dict['RealArm7dofE0v4']=agent_info_dict['RealArm7dof']
agent_info_dict['RealArm7dofE0v5']=agent_info_dict['RealArm7dof']
agent_info_dict['RealArm7dofE0v6']=agent_info_dict['RealArm7dof']
agent_info_dict['RealArm7dofE0v7']=agent_info_dict['RealArm7dof']
agent_info_dict['RealArm7dofE0v8']=agent_info_dict['RealArm7dof']
agent_info_dict['RealArm7dofE0v9']=agent_info_dict['RealArm7dof']

agent_info_dict['RealArm6dofE0v1']=agent_info_dict['RealArm6dof']
agent_info_dict['RealArm6dofE0v2']=agent_info_dict['RealArm6dof']

agent_info_dict['RealArm5dofE0v1']=agent_info_dict['RealArm5dof']
agent_info_dict['RealArm5dofE0v2']=agent_info_dict['RealArm5dof']

agent_info_dict['RealArm4dofE0v1']=agent_info_dict['RealArm4dof']
agent_info_dict['RealArm4dofE0v2']=agent_info_dict['RealArm4dof']

agent_info_dict['RealArm3dofE0v1']=agent_info_dict['RealArm3dof']
agent_info_dict['RealArm3dofE0v2']=agent_info_dict['RealArm3dof']

agent_info_dict['HCheavy']=agent_info_dict['HC']
agent_info_dict['HC_Ez']=agent_info_dict['HC']
agent_info_dict['HC_sL']=agent_info_dict['HC']

agent_info_dict['HC_E1']=agent_info_dict['HC']

agent_info_dict['FC_sLG']=agent_info_dict['FC_sLT']
agent_info_dict['FC_sLGfblr']=agent_info_dict['FC_sLT']
agent_info_dict['FCheavy_sLT']=agent_info_dict['FCheavy']
agent_info_dict['FCheavy_sLG']=agent_info_dict['FCheavy']



agent_info_dict['FCheavy_lS']=agent_info_dict['FCheavy']
agent_info_dict['FCheavy_mS']=agent_info_dict['FCheavy']
agent_info_dict['FCheavy_minS']=agent_info_dict['FCheavy']
agent_info_dict['FCheavy_maxS']=agent_info_dict['FCheavy']
agent_info_dict['FCheavy_maxSGv0']=agent_info_dict['FCheavy']
agent_info_dict['FCheavy_maxSTv0']=agent_info_dict['FCheavy']
agent_info_dict['FCheavy_minSGv0']=agent_info_dict['FCheavy']
agent_info_dict['FCheavy_minSTv0']=agent_info_dict['FCheavy']

agent_info_dict['Walker2d_sL']=agent_info_dict['Walker2d']
agent_info_dict['Walker2d_sD']=agent_info_dict['Walker2d']


'''elif 'A' in agentt and 'VA' not in agentt:
    total_vec = 8
    total_chk=20
    ori_final = 2000
    ori_begin = 100#
    ori_step = 100
    x_speed_index=13
    desired_dist=500
    ori_total_vec_rsq=total_vec'''

base_folder=cwd+'/experiments_results/Synergy/all_csv/raw_csv'

HC_folder=base_folder+'/HC'
HeavyHC_folder=base_folder+'/HCheavy'
FC_folder=base_folder+'/FC'

HC4dof_folder=base_folder+'/HC4dof'
HC2dof_folder=base_folder+'/HC2dof'
HC3dofb_folder=base_folder+'/HC3dofb'
HC3doff_folder=base_folder+'/HC3doff'
HC5dof_folder=base_folder+'/HC5dof'

HC5dofv2_folder=base_folder+'/HC5dofv2'
HC5dofv3_folder=base_folder+'/HC5dofv3'
HC5dofv4_folder=base_folder+'/HC5dofv4'
HC5dofv5_folder=base_folder+'/HC5dofv5'
HC5dofv6_folder=base_folder+'/HC5dofv6'

HCsquat2dof_folder=base_folder+'/HCsquat2dof'
HCsquat4dof_folder=base_folder+'/HCsquat4dof'
HCsquat6dof_folder=base_folder+'/HCsquat6dof'

HCsquat4dofEp1_folder=base_folder+'/HCsquat4dof_Ep1'
HCsquat6dofEp1_folder=base_folder+'/HCsquat6dof_Ep1'

HCsquat2dofEp25_folder=base_folder+'/HCsquat2dof_Ep25'
HCsquat4dofEp25_folder=base_folder+'/HCsquat4dof_Ep25'
HCsquat6dofEp25_folder=base_folder+'/HCsquat6dof_Ep25'

HCsquat2dofez_folder=base_folder+'/HCsquat2dof_Ez'
HCsquat4dofez_folder=base_folder+'/HCsquat4dof_Ez'
HCsquat6dofez_folder=base_folder+'/HCsquat6dof_Ez'

HCsquat2dofEalt_folder=base_folder+'/HCsquat2dof_Ealt'
HCsquat4dofEalt_folder=base_folder+'/HCsquat4dof_Ealt'
HCsquat6dofEalt_folder=base_folder+'/HCsquat6dof_Ealt'

HC2dofv2_folder=base_folder+'/HC2dofv2'
HC2dofv3_folder=base_folder+'/HC2dofv3'
HC2dofv4_folder=base_folder+'/HC2dofv4'
HC2dofv5_folder=base_folder+'/HC2dofv5'

HCez_folder=base_folder+'/HC_Ez'
HC4dofez_folder=base_folder+'/HC4dof_Ez'
HC2dofez_folder=base_folder+'/HC2dof_Ez'
HC3dofbez_folder=base_folder+'/HC3dofb_Ez'
HC3doffez_folder=base_folder+'/HC3doff_Ez'
HC5dofez_folder=base_folder+'/HC5dof_Ez'

HCE1_folder=base_folder+'/HC_E1'
HC4dofE1_folder=base_folder+'/HC4dof_E1'
HC2dofE1_folder=base_folder+'/HC2dof_E1'
HC3dofbE1_folder=base_folder+'/HC3dofb_E1'
HC3doffE1_folder=base_folder+'/HC3doff_E1'
HC5dofE1_folder=base_folder+'/HC5dof_E1'

VA_folder=base_folder+'/VA'
VA4dof_folder=base_folder+'/VA4dof'
VA6dof_folder=base_folder+'/VA6dof'
VA8dof_folder=base_folder+'/VA8dof'

VAsR_folder=base_folder+'/VA_sR'
VAbR_folder=base_folder+'/VA_bR'
VA4dofsR_folder=base_folder+'/VA4dof_sR'
VA4dofbR_folder=base_folder+'/VA4dof_bR'
VA6dofsR_folder=base_folder+'/VA6dof_sR'
VA6dofbR_folder=base_folder+'/VA6dof_bR'
VA8dofsR_folder=base_folder+'/VA8dof_sR'
VA8dofbR_folder=base_folder+'/VA8dof_bR'

RealArm3dof_folder=base_folder+'/RealArm3dof'
RealArm4dof_folder=base_folder+'/RealArm4dof'
RealArm5dof_folder=base_folder+'/RealArm5dof'
RealArm6dof_folder=base_folder+'/RealArm6dof'
RealArm7dof_folder=base_folder+'/RealArm7dof'
RealArm7dofE0v1_folder=base_folder+'/RealArm7dofE0v1'
RealArm7dofE0v2_folder=base_folder+'/RealArm7dofE0v2'
RealArm7dofE0v3_folder=base_folder+'/RealArm7dofE0v3'
RealArm7dofE0v4_folder=base_folder+'/RealArm7dofE0v4'
RealArm7dofE0v5_folder=base_folder+'/RealArm7dofE0v5'
RealArm7dofE0v6_folder=base_folder+'/RealArm7dofE0v6'
RealArm7dofE0v7_folder=base_folder+'/RealArm7dofE0v7'
RealArm7dofE0v8_folder=base_folder+'/RealArm7dofE0v8'
RealArm7dofE0v9_folder=base_folder+'/RealArm7dofE0v9'

RealArm6dofE0v1_folder=base_folder+'/RealArm6dofE0v1'
RealArm6dofE0v2_folder=base_folder+'/RealArm6dofE0v2'
RealArm5dofE0v1_folder=base_folder+'/RealArm5dofE0v1'
RealArm5dofE0v2_folder=base_folder+'/RealArm5dofE0v2'
RealArm4dofE0v1_folder=base_folder+'/RealArm4dofE0v1'
RealArm4dofE0v2_folder=base_folder+'/RealArm4dofE0v2'
RealArm3dofE0v1_folder=base_folder+'/RealArm3dofE0v1'
RealArm3dofE0v2_folder=base_folder+'/RealArm3dofE0v2'


RealArm4dofMinE_folder=base_folder+'/RealArm4dofMinE'
RealArm5dofMinE_folder=base_folder+'/RealArm5dofMinE'
RealArm4dofLT_folder=base_folder+'/RealArm4dofLT'
RealArm5dofLT_folder=base_folder+'/RealArm5dofLT'


VAez_folder=base_folder+'/VA_Ez'
VA4dofez_folder=base_folder+'/VA4dof_Ez'
VA6dofez_folder=base_folder+'/VA6dof_Ez'
VA8dofez_folder=base_folder+'/VA8dof_Ez'

VAE1_folder=base_folder+'/VA_E1'
VA4dofE1_folder=base_folder+'/VA4dof_E1'
VA6dofE1_folder=base_folder+'/VA6dof_E1'
VA8dofE1_folder=base_folder+'/VA8dof_E1'

VAp5_folder=base_folder+'/VA_Ep5'
VA4dofp5_folder=base_folder+'/VA4dof_Ep5'
VA6dofp5_folder=base_folder+'/VA6dof_Ep5'
VA8dofp5_folder=base_folder+'/VA8dof_Ep5'

AntRun_folder=base_folder+'/AntRun'
AntSquaT_folder=base_folder+'/AntSquaT'
AntSquaTRedundant_folder=base_folder+'/AntSquaTRedundant'

HCsL_folder=base_folder+'/HC_sL'
Walker2dsL_folder=base_folder+'/Walker2d_sL'
Walker2dsD_folder=base_folder+'/Walker2d_sD'
Waler2d_folder=base_folder+'/Walker2d'
FCsLT_folder=base_folder+'/FC_sLT'
FCsLG_folder=base_folder+'/FC_sLG'
FCsLGfblr_folder=base_folder+'/FC_sLGfblr'

FCheavy_folder=base_folder+'/FCheavy'
FCHeavysLT_folder=base_folder+'/FCheavy_sLT'
FCHeavysLG_folder=base_folder+'/FCheavy_sLG'
FCHeavylS_folder=base_folder+'/FCheavy_lS'
FCHeavymS_folder=base_folder+'/FCheavy_mS'
FCHeavyminS_folder=base_folder+'/FCheavy_minS'
FCHeavymaxS_folder=base_folder+'/FCheavy_maxS'
FCHeavyminSG_folder=base_folder+'/FCheavyminSGv0'
FCHeavymaxSG_folder=base_folder+'/FCheavymaxSGv0'
FCHeavyminST_folder=base_folder+'/FCheavyminSTv0'
FCHeavymaxST_folder=base_folder+'/FCheavymaxSTv0'

FCheavyE0v00_folder=base_folder+'/FCheavyE0v00'
FCheavyE0v45_folder=base_folder+'/FCheavyE0v45'
FCheavyE0v65_folder=base_folder+'/FCheavyE0v65'
FCheavyE0v1_folder=base_folder+'/FCheavyE0v1'
FCheavyE0v2_folder=base_folder+'/FCheavyE0v2'
FCheavyE0v3_folder=base_folder+'/FCheavyE0v3'
FCheavyE0v4_folder=base_folder+'/FCheavyE0v4'
FCheavyE0v5_folder=base_folder+'/FCheavyE0v5'
FCheavyE0v6_folder=base_folder+'/FCheavyE0v6'

FCheavylSv2_folder=base_folder+'/FCheavylSv2'
FCheavylSv4_folder=base_folder+'/FCheavylSv4'
FCheavylSv6_folder=base_folder+'/FCheavylSv6'

FCheavymSv2_folder=base_folder+'/FCheavymSv2'
FCheavymSv4_folder=base_folder+'/FCheavymSv4'
FCheavymSv6_folder=base_folder+'/FCheavymSv6'

FCheavyminSv2_folder=base_folder+'/FCheavyminSv2'
FCheavyminSv4_folder=base_folder+'/FCheavyminSv4'
FCheavyminSv6_folder=base_folder+'/FCheavyminSv6'
FCheavyminSv00_folder=base_folder+'/FCheavyminSv00'
FCheavyminSv45_folder=base_folder+'/FCheavyminSv45'
FCheavyminSv65_folder=base_folder+'/FCheavyminSv65'

FCheavymaxSv2_folder=base_folder+'/FCheavymaxSv2'
FCheavymaxSv4_folder=base_folder+'/FCheavymaxSv4'
FCheavymaxSv6_folder=base_folder+'/FCheavymaxSv6'
FCheavymaxSv00_folder=base_folder+'/FCheavymaxSv00'
FCheavymaxSv45_folder=base_folder+'/FCheavymaxSv45'
FCheavymaxSv65_folder=base_folder+'/FCheavymaxSv65'


FCheavyminSGv2_folder=base_folder+'/FCheavyminSGv2'
FCheavyminSGv4_folder=base_folder+'/FCheavyminSGv4'
FCheavyminSGv6_folder=base_folder+'/FCheavyminSGv6'

FCheavymaxSGv2_folder=base_folder+'/FCheavymaxSGv2'
FCheavymaxSGv4_folder=base_folder+'/FCheavymaxSGv4'
FCheavymaxSGv6_folder=base_folder+'/FCheavymaxSGv6'

FCheavyminSTv2_folder=base_folder+'/FCheavyminSTv2'
FCheavyminSTv4_folder=base_folder+'/FCheavyminSTv4'
FCheavyminSTv6_folder=base_folder+'/FCheavyminSTv6'

FCheavymaxSTv2_folder=base_folder+'/FCheavymaxSTv2'
FCheavymaxSTv4_folder=base_folder+'/FCheavymaxSTv4'
FCheavymaxSTv6_folder=base_folder+'/FCheavymaxSTv6'


FCheavysLGv1_folder=base_folder+'/FCheavysLGv1'
FCheavysLGv2_folder=base_folder+'/FCheavysLGv2'
FCheavysLGv3_folder=base_folder+'/FCheavysLGv3'
FCheavysLGv4_folder=base_folder+'/FCheavysLGv4'
FCheavysLGv5_folder=base_folder+'/FCheavysLGv5'
FCheavysLGv6_folder=base_folder+'/FCheavysLGv6'

FCheavysLTv1_folder=base_folder+'/FCheavysLTv1'
FCheavysLTv2_folder=base_folder+'/FCheavysLTv2'
FCheavysLTv3_folder=base_folder+'/FCheavysLTv3'
FCheavysLTv4_folder=base_folder+'/FCheavysLTv4'
FCheavysLTv5_folder=base_folder+'/FCheavysLTv5'
FCheavysLTv6_folder=base_folder+'/FCheavysLTv6'

P_axis_range={
    'HCheavy':[0, 8000],'HC':[0, 20000],'FC':[0, 25000],
    'HC4dof':[0, 20000], 'HC5dof':[0, 20000],'HC3doff':[0, 20000],'HC3dofb':[0, 20000],'HC2dof':[0, 20000],
    'HC4dof_Ez': [0, 20000], 'HC5dof_Ez': [0, 20000], 'HC3doff_Ez': [0, 20000], 'HC3dofb_Ez': [0, 20000],
    'HC2dof_Ez': [0, 20000],'HC_Ez':[0, 20000],
    'HC4dof_E1': [0, 20000], 'HC5dof_E1': [0, 20000], 'HC3doff_E1': [0, 20000], 'HC3dofb_E1': [0, 20000],
    'HC2dof_E1': [0, 20000], 'HC_E1': [0, 20000],
    'HC5dofv2':[0, 20000],'HC5dofv3':[0, 20000],'HC5dofv4':[0, 20000],'HC5dofv5':[0, 20000],'HC5dofv6':[0, 20000],
    'HC2dofv2': [0, 20000], 'HC2dofv3': [0, 20000], 'HC2dofv4': [0, 20000], 'HC2dofv5': [0, 20000],
    'HCsquat2dof':[-1200, -600],'HCsquat4dof':[-1200, -600],'HCsquat6dof':[-1200, -600],
    'HCsquat4dof_Ep1':[-1200, -600],'HCsquat6dof_Ep1':[-1200, -600],'HCsquatep1':[-1200, 0],
    'HCsquat2dof_Ep25':[-1200, -600],'HCsquat4dof_Ep25':[-1200, -600],'HCsquat6dof_Ep25':[-1200, -600],
    'HCsquat2dof_Ez':[-1200, -600],'HCsquat4dof_Ez':[-1200, -600],'HCsquat6dof_Ez':[-1200, -600],
    'HCsquatep25':[-1200, 0],'HCsquat': [-1200, -400],'HCsquatez':[-1200, 0],

    'HCsquat2dof_Ealt':[-1000, -200],'HCsquat4dof_Ealt':[-1000, -200],'HCsquat6dof_Ealt':[-1000, -200],
    'HCsquatEalt':[-250, 0],'HCthree':[4000, 15500],

    'VA':[-200, 0],'VA4dof':[-600, 0],'VA6dof':[-600, 0],'VA8dof':[-600, 0],#[-1000, 300]
    'VA_Ez': [-600, 0], 'VA4dof_Ez': [-600, 0], 'VA6dof_Ez': [-600, 0], 'VA8dof_Ez': [-600, 0],
    'VA_Ep5': [-600, 0], 'VA4dof_Ep5': [-600, 0], 'VA6dof_Ep5': [-600, 0], 'VA8dof_Ep5': [-600, 0],
    'VA_E1': [-600, 0], 'VA4dof_E1': [-600, 0], 'VA6dof_E1': [-600, 0], 'VA8dof_E1': [-600, 0],

    'RealArm3dof':[-3000, 0],'RealArm4dof':[-3000, 0],'RealArm5dof':[-3000, 0],'RealArm6dof':[-3000, 0],'RealArm7dof':[-3000, 0],
    'RealArmAll':[-350, 0],
    'RealArm4dofMinE':[-3000, 0],'RealArm5dofMinE':[-3000, 0],'RealArm4dofLT':[-3000, 0],'RealArm5dofLT':[-3000, 0],
    'RealArmCompareMultiOb':[-3000, 0],

    'AntRun':[0, 10000],'AntSquaT':[-500, 0],'AntSquaTRedundant':[-5000, -500],'Walker2dsl': [0, 7000],
    'Walker2dAll' :[0, 7000],

    'FC_sLGfblr': [0, 20000],  'FC_sLG': [0, 20000], 'FC_sLT': [0, 20000],"FCHeavyActionManip": [3000, 5000],
    "FCHeavyOri": [0, 5000],"FCHeavyTrot": [0, 5000],"FCHeavyGallop": [0, 5000],"FCHeavySpeedp5": [400,600],"FCHeavySpeed1":[750, 1250],
    "FCHeavySpeed2": [1500, 2500],"FCHeavySpeed3": [2000, 3500],"FCHeavySpeed4": [2000, 4500],"FCHeavySpeed5":[2000, 5000],"FCHeavySpeed6": [2000, 5000],
    "FCHeavyminSpring": [2000, 5000],"FCHeavymaxSpring": [2000, 5000],
}

PI_axis_range = {
    'HCheavy': [0, 5], 'HC': [0, 15], 'FC': [0, 6], 'HC4dof': [0, 15],
    'HC5dof': [0, 15], 'HC3doff': [0, 15], 'HC3dofb': [0, 15], 'HC2dof': [0, 15],
    'HC_Ez': [0, 15], 'HC4dof_Ez': [0, 15],'HC5dof_Ez': [0, 15],
    'HC3doff_Ez': [0, 15], 'HC3dofb_Ez': [0, 15], 'HC2dof_Ez': [0, 15],
    'HC_E1': [0, 15], 'HC4dof_E1': [0, 15], 'HC5dof_E1': [0, 15],
    'HC3doff_E1': [0, 15], 'HC3dofb_E1': [0, 15], 'HC2dof_E1': [0, 15],
    'HC5dofv2': [0, 15], 'HC5dofv3': [0, 15], 'HC5dofv4': [0, 15], 'HC5dofv5': [0, 15],
    'HC5dofv6': [0, 15],
    'HC2dofv2': [0, 15], 'HC2dofv3': [0, 15], 'HC2dofv4': [0, 15], 'HC2dofv5': [0, 15],
    'HCsquat2dof': [-200, 0], 'HCsquat4dof': [-200, 0], 'HCsquat6dof': [-200, 0],
    'HCsquat4dof_Ep1': [-200, 0], 'HCsquat6dof_Ep1': [-200, 0],'HCsquatep1':[-200, 0],
    'HCsquat2dof_Ep25': [-200, 0], 'HCsquat4dof_Ep25': [-200, 0], 'HCsquat6dof_Ep25': [-200, 0],
    'HCsquat2dof_Ez': [-200, 0], 'HCsquat4dof_Ez': [-200, 0], 'HCsquat6dof_Ez': [-200, 0],
    'HCsquatep25': [-200, 0],'HCsquat': [-200, 0],'HCsquatez': [-200, 0],

    'HCsquat2dof_Ealt': [-200, 0], 'HCsquat4dof_Ealt': [-200, 0], 'HCsquat6dof_Ealt': [-200, 0],
    'HCsquatEalt':[-30, -0],

    'VA': [-120, 0], 'VA4dof': [-100, 0], 'VA6dof': [-100, 0], 'VA8dof': [-100, 0],
    'VA_Ez': [-100, 0], 'VA4dof_Ez': [-100, 0], 'VA6dof_Ez': [-100, 0], 'VA8dof_Ez': [-100, 0],
    'VA_Ep5': [-100, 0], 'VA4dof_Ep5': [-100, 0], 'VA6dof_Ep5': [-100, 0], 'VA8dof_Ep5': [-100, 0],
    'VA_E1': [-100, 0], 'VA4dof_E1': [-100, 0], 'VA6dof_E1': [-100, 0], 'VA8dof_E1': [-100, 0],

    'RealArm3dof': [-100, 0], 'RealArm4dof': [-100, 0], 'RealArm5dof': [-100, 0], 'RealArm6dof': [-100, 0],
    'RealArmAll': [-20, 0],

    'RealArm4dofMinE': [-100, 0], 'RealArm5dofMinE': [-100, 0], 'RealArm4dofLT': [-100, 0],
    'RealArm5dofLT': [-100, 0],'RealArmCompareMultiOb':[-100, 0],

    'AntRun': [0, 30], 'AntSquaT': [-120, 0], 'AntSquaTRedundant': [-160, -30],

    'Walker2dsl': [0, 70],'Walker2dAll' :[0, 70],
    'FC_sLGfblr': [0, 6], 'FC_sLG': [0, 6], 'FC_sLT': [0, 6], "FCHeavyActionManip": [1, 2],"FCHeavyOri": [0, 3.5],
    "FCHeavyVarySpeed": [-10, 0],"FCHeavyTrot": [0, 3.5],"FCHeavyGallop": [0, 3.5],"FCHeavySpeedp5": [0,3.5],"FCHeavySpeed1":[0, 3.5],
    "FCHeavySpeed2": [0, 3.5],"FCHeavySpeed3": [0, 3.5],"FCHeavySpeed4": [0, 3.5],"FCHeavySpeed5": [0, 3.5],"FCHeavySpeed6": [0, 3.5],
    "FCHeavyminSpring": [0, 3.5],"FCHeavymaxSpring": [0, 3.5],
}

SA_axis_range = {
    'HCheavy': [4, 8], 'HC': [3, 8], 'FC': [3, 8], 'HC4dof': [3, 8],
    'HC5dof': [3, 8], 'HC3doff': [3, 8], 'HC3dofb': [3, 8], 'HC2dof': [3, 8],
    'HC_Ez': [3, 8],  'HC4dof_Ez': [3, 8],'HC5dof_Ez': [3, 8],
    'HC3doff_Ez': [3, 8], 'HC3dofb_Ez': [3, 8], 'HC2dof_Ez': [3, 8],
    'HC_E1': [3, 8], 'HC4dof_E1': [3, 8], 'HC5dof_E1': [3, 8],
    'HC3doff_E1': [3, 8], 'HC3dofb_E1': [3, 8], 'HC2dof_E1': [3, 8],
    'HC5dofv2':  [3, 8], 'HC5dofv3':  [3, 8], 'HC5dofv4':  [3, 8], 'HC5dofv5':  [3, 8],
    'HC5dofv6':  [3, 8],
    'HC2dofv2': [3, 8], 'HC2dofv3': [3, 8], 'HC2dofv4': [3, 8], 'HC2dofv5': [3, 8],
    'HCsquat2dof': [2, 9], 'HCsquat4dof':[2, 9], 'HCsquat6dof':[2, 9],
    'HCsquat4dof_Ep1': [2, 9], 'HCsquat6dof_Ep1': [2, 9],'HCsquatep1':[2, 9],
    'HCsquat2dof_Ep25': [2, 9], 'HCsquat4dof_Ep25': [2, 9], 'HCsquat6dof_Ep25': [2, 9],
    'HCsquat2dof_Ez': [2, 9], 'HCsquat4dof_Ez': [2, 9], 'HCsquat6dof_Ez': [2, 9],
    'HCsquatep25': [2, 9],'HCsquat': [2, 9],'HCsquatez': [2, 9],

    'HCsquat2dof_Ealt': [2, 9], 'HCsquat4dof_Ealt': [2, 9], 'HCsquat6dof_Ealt': [2, 9],
    'HCsquatEalt': [2, 9],

    'VA': [3, 9], 'VA4dof':[3, 8], 'VA6dof': [3, 8], 'VA8dof': [3, 8],
    'VA_Ez':[3, 8], 'VA4dof_Ez': [3, 8], 'VA6dof_Ez':[3, 8], 'VA8dof_Ez': [3, 8],
    'VA_Ep5':[3, 8], 'VA4dof_Ep5': [3, 8], 'VA6dof_Ep5': [3, 8], 'VA8dof_Ep5': [3, 8],
    'VA_E1': [3, 8], 'VA4dof_E1': [3, 8], 'VA6dof_E1':[3, 8], 'VA8dof_E1': [3, 8],'VAsR': [3, 9],
    'VAbR': [3, 9],

    'RealArm3dof':[2, 9], 'RealArm4dof':[2, 9], 'RealArm5dof': [2, 9], 'RealArm6dof': [2, 9],
    'RealArmAll':[2, 9],
    'RealArm4dofMinE': [2, 9], 'RealArm5dofMinE': [2, 9], 'RealArm4dofLT': [2, 9],
    'RealArm5dofLT': [2, 9],'RealArmCompareMultiOb':[2, 9],

    'AntRun': [2.5, 7], 'AntSquaT': [2.5, 7], 'AntSquaTRedundant':[2.5, 7],
    'Walker2d': [2, 9],'Walker2dsl': [2, 9],'Walker2dAll' :[2, 9],
    'FC_sLGfblr': [2, 9], 'FC_sLG':[2, 9], 'FC_sLT': [2, 9],"FCHeavyActionManip": [5, 9],"FCHeavyOri": [2, 9],
    "FCHeavyTrot": [2,9],"FCHeavyGallop": [2,9],"FCHeavySpeedp5": [2, 9],"FCHeavySpeed1": [2, 9],
    "FCHeavySpeed2": [2, 9],"FCHeavySpeed3": [2, 9],"FCHeavySpeed4": [2, 9],"FCHeavySpeed6": [2, 9],
    "FCHeavyUnlimited":[2, 9],'FCHeavySpeed5':[2, 9],"FCHeavyminSpring": [2, 9],"FCHeavymaxSpring": [2, 9],
    "VarySpeed_maxSG":[2, 9],"VarySpeed_minSG":[2, 9],"VarySpeed_defS":[2, 9],"VarySpeed_defmaxS":[2, 9],
    "VarySpeed_defminS":[2, 9],"VarySpeed_maxST":[2, 9],"VarySpeed_minST":[2, 9],"VarySpring_v4":[2, 9],"VarySpring_Gv4":[2, 9],
    "VarySpring_Tv4":[2, 9],"VarySpring_UNL":[2, 9],"VarySpring_GUNL":[2, 9],"VarySpring_TUNL":[2, 9],
    "VaryGait_minSv4":[2, 9],"VaryGait_maxSv4":[2, 9],"VaryGait_minSv6":[2, 9],"VaryGait_maxSv6":[2, 9],
    "VaryGait_minSUNL":[2, 9],"VaryGait_maxSUNL":[2, 9],
}

E_axis_range = {
    'HCheavy': [0, 2500], 'HC': [0, 2500], 'FC': [0, 7000], 'HC4dof': [0, 2500],
    'HC5dof':[0, 2500], 'HC3doff': [0, 2500], 'HC3dofb':[0, 2500], 'HC2dof': [0, 2500],
    'HC_Ez': [0, 2500],'HC4dof_Ez': [0, 2500],'HC5dof_Ez': [0, 2500],
    'HC3doff_Ez': [0, 2500], 'HC3dofb_Ez': [0, 2500], 'HC2dof_Ez': [0, 2500],
    'HC_E1': [0, 2500], 'HC4dof_E1': [0, 2500], 'HC5dof_E1': [0, 2500],
    'HC3doff_E1': [0, 2500], 'HC3dofb_E1': [0, 2500], 'HC2dof_E1': [0, 2500],
    'HC5dofv2': [0, 2500], 'HC5dofv3': [0, 2500], 'HC5dofv4': [0, 2500], 'HC5dofv5': [0, 2500],
    'HC5dofv6': [0, 2500],
    'HC2dofv2': [0, 2500], 'HC2dofv3': [0, 2500], 'HC2dofv4': [0, 2500], 'HC2dofv5': [0, 2500],
    'HCsquat2dof': [0, 500], 'HCsquat4dof': [0, 500], 'HCsquat6dof': [0, 500],
    'HCsquat4dof_Ep1': [0, 500], 'HCsquat6dof_Ep1': [0, 500],'HCsquatep1':[0, 500],
    'HCsquat2dof_Ep25': [0, 500], 'HCsquat4dof_Ep25': [0, 500], 'HCsquat6dof_Ep25': [0, 500],
    'HCsquat2dof_Ez': [0, 500], 'HCsquat4dof_Ez': [0, 500], 'HCsquat6dof_Ez': [0, 500],
    'HCsquatep25': [0, 500],'HCsquatez': [0, 500],

    'HCsquat2dof_Ealt': [0, 500], 'HCsquat4dof_Ealt': [0, 500], 'HCsquat6dof_Ealt': [0, 500],
    'HCsquatEalt': [0, 500],

    'VA': [0, 20], 'VA4dof':[0, 20], 'VA6dof': [0, 20], 'VA8dof': [0, 20],
    'VA_Ez':[0, 20], 'VA4dof_Ez': [0, 20], 'VA6dof_Ez':[0, 20], 'VA8dof_Ez': [0, 20],
    'VA_Ep5':[0, 20], 'VA4dof_Ep5': [0, 20], 'VA6dof_Ep5': [0, 20], 'VA8dof_Ep5': [0, 20],
    'VA_E1': [0, 20], 'VA4dof_E1': [0, 20], 'VA6dof_E1':[0, 20], 'VA8dof_E1': [0, 20],

    'RealArm3dof': [0, 100], 'RealArm4dof': [0, 100], 'RealArm5dof':[0, 100], 'RealArm6dof': [0, 100],
    'RealArmAll':[0, 100],
    'RealArm4dofMinE': [0, 100], 'RealArm5dofMinE': [0, 100], 'RealArm4dofLT': [0, 100],
    'RealArm5dofLT': [0, 100],'RealArmCompareMultiOb':[0, 100],

    'AntRun': [0, 450], 'AntSquaT': [0, 50], 'AntSquaTRedundant':[0, 50],
    'FC_sLGfblr': [0, 7000], 'FC_sLG': [0, 7000], 'FC_sLT': [0, 7000]

}

folder_algo_dict={
    HCsquat2dof_folder: [('HCsquat2dof', 'SAC'), ('HCsquat2dof', 'TD3')],
    HCsquat4dof_folder: [('HCsquat4dof', 'SAC'), ('HCsquat4dof', 'TD3')],
    HCsquat6dof_folder: [('HCsquat6dof', 'SAC'), ('HCsquat6dof', 'TD3')],
    HCsquat4dofEp1_folder: [('HCsquat4dof_Ep1', 'SAC'), ('HCsquat4dof_Ep1', 'TD3')],
    HCsquat6dofEp1_folder: [('HCsquat6dof_Ep1', 'SAC'), ('HCsquat6dof_Ep1', 'TD3')],

    HCsquat2dofEp25_folder: [('HCsquat2dof_Ep25', 'SAC'), ('HCsquat2dof_Ep25', 'TD3')],
    HCsquat4dofEp25_folder: [('HCsquat4dof_Ep25', 'SAC'), ('HCsquat4dof_Ep25', 'TD3')],
    HCsquat6dofEp25_folder: [('HCsquat6dof_Ep25', 'SAC'), ('HCsquat6dof_Ep25', 'TD3')],

    HCsquat2dofEalt_folder: [('HCsquat2dof_Ealt', 'SAC'), ('HCsquat2dof_Ealt', 'TD3')],
    HCsquat4dofEalt_folder: [('HCsquat4dof_Ealt', 'SAC'), ('HCsquat4dof_Ealt', 'TD3')],
    HCsquat6dofEalt_folder: [('HCsquat6dof_Ealt', 'SAC'), ('HCsquat6dof_Ealt', 'TD3')],

    HCsquat2dofez_folder: [('HCsquat2dof_Ez', 'SAC'), ('HCsquat2dof_Ez', 'TD3')],
    HCsquat4dofez_folder: [('HCsquat4dof_Ez', 'SAC'), ('HCsquat4dof_Ez', 'TD3')],
    HCsquat6dofez_folder: [('HCsquat6dof_Ez', 'SAC'), ('HCsquat6dof_Ez', 'TD3')],

    HC_folder:[('HC', 'SAC'),('HC', 'TD3')],
    HeavyHC_folder:[('HeavyHC', 'SAC'), ('HeavyHC', 'TD3')],
    FC_folder:[('FC', 'SAC'),('FC', 'TD3')],
    HC2dof_folder:[('HC2dof', 'SAC'), ('HC2dof', 'TD3')],
    HC4dof_folder: [('HC4dof', 'SAC'), ('HC4dof', 'TD3')],
    HC5dof_folder: [('HC5dof', 'SAC'), ('HC5dof', 'TD3')],
    HC5dofv2_folder: [('HC5dofv2', 'SAC')],
    HC5dofv3_folder: [('HC5dofv3', 'SAC')],
    HC5dofv4_folder: [('HC5dofv4', 'SAC')],
    HC5dofv5_folder: [('HC5dofv5', 'SAC')],
    HC5dofv6_folder: [('HC5dofv6', 'SAC')],

    HC2dofv2_folder: [('HC2dofv2', 'SAC')],
    HC2dofv3_folder: [('HC2dofv3', 'SAC')],
    HC2dofv4_folder: [('HC2dofv4', 'SAC')],
    HC2dofv5_folder: [('HC2dofv5', 'SAC')],

    HC3doff_folder: [('HC3doff', 'SAC'), ('HC3doff', 'TD3')],
    HC3dofb_folder:[('HC3dofb', 'SAC'), ('HC3dofb', 'TD3')],
    HCez_folder: [('HC_Ez', 'SAC')],
    HC2dofez_folder: [('HC2dof_Ez', 'SAC')],
    HC4dofez_folder: [('HC4dof_Ez', 'SAC')],
    HC5dofez_folder: [('HC5dof_Ez', 'SAC')],
    HC3doffez_folder: [('HC3doff_Ez', 'SAC')],
    HC3dofbez_folder: [('HC3dofb_Ez', 'SAC')],
    HCE1_folder: [('HC_E1', 'SAC')],
    HC2dofE1_folder: [('HC2dof_E1', 'SAC')],
    HC4dofE1_folder: [('HC4dof_E1', 'SAC')],
    HC5dofE1_folder: [('HC5dof_E1', 'SAC')],
    HC3doffE1_folder: [('HC3doff_E1', 'SAC')],
    HC3dofbE1_folder: [('HC3dofb_E1', 'SAC')],
    VA_folder:[('VA', 'SAC', ('VA', 'TD3'))],#, ('VA', 'TD3')
    VA4dof_folder:[('VA4dof', 'SAC'), ('VA4dof', 'TD3')],
    VA6dof_folder: [('VA6dof', 'SAC'), ('VA6dof', 'TD3')],
    VA8dof_folder:[('VA8dof', 'SAC'), ('VA8dof', 'TD3')],
    VAez_folder:[('VA_Ez', 'SAC')],
    VA4dofez_folder:[('VA4dof_Ez', 'SAC')],
    VA6dofez_folder:[('VA6dof_Ez', 'SAC')],
    VA8dofez_folder: [('VA8dof_Ez', 'SAC')],
    VAp5_folder:[('VA_Ep5', 'SAC')],
    VA4dofp5_folder: [('VA4dof_Ep5', 'SAC')],
    VA6dofp5_folder: [('VA6dof_Ep5', 'SAC')],
    VA8dofp5_folder: [('VA8dof_Ep5', 'SAC')],
    VAE1_folder: [('VA_E1', 'SAC')],
    VA4dofE1_folder:[('VA4dof_E1', 'SAC')],
    VA6dofE1_folder: [('VA6dof_E1', 'SAC')],
    VA8dofE1_folder:[('VA8dof_E1', 'SAC')],

    RealArm3dof_folder:[('RealArm3dof', 'SAC'),('RealArm3dof', 'TD3')],
    RealArm4dof_folder:[('RealArm4dof', 'SAC'),('RealArm4dof', 'TD3')],
    RealArm5dof_folder:[('RealArm5dof', 'SAC'),('RealArm5dof', 'TD3')],
    RealArm6dof_folder:[('RealArm6dof', 'SAC'),('RealArm6dof', 'TD3')],
    RealArm7dof_folder:[('RealArm7dof', 'SAC'),('RealArm7dof', 'TD3')],

    RealArm4dofMinE_folder: [('RealArm4dofMinE', 'SAC')],
    RealArm5dofMinE_folder: [('RealArm5dofMinE', 'SAC')],
    RealArm4dofLT_folder: [('RealArm4dofLT', 'SAC')],
    RealArm5dofLT_folder: [('RealArm5dofLT', 'SAC')],

    AntRun_folder: [('AntRun', 'SAC'), ('AntRun', 'TD3')],
    AntSquaT_folder:[('AntSquaT', 'SAC'), ('AntSquaT', 'TD3')],
    AntSquaTRedundant_folder:[('AntSquaTRedundant', 'SAC'), ('AntSquaTRedundant', 'TD3')],

    HCsL_folder: [('HC_sL', 'SAC')],

    FCsLT_folder: [('FC_sLT', 'SAC')],
    FCsLG_folder: [('FC_sLG', 'SAC')],
    FCsLGfblr_folder: [('FC_sLGfblr', 'SAC')],

    FCheavy_folder: [('FCheavy', 'SAC')],
    FCHeavysLT_folder: [('FCheavy_sLT', 'SAC')],
    FCHeavysLG_folder: [('FCheavy_sLG', 'SAC')],

    #Walker2d_folder: [('AntRun', 'SAC'), ('AntRun', 'TD3')],
}

title_name_dict={
    'HC2dof':'2-DOF Running Half-Cheetah',
    'HC4dof':'4-DOF Running Half-Cheetah',
    'HC':'6-DOF Running Half-Cheetah',

    'HCsquat2dof':'2-DOF Squatting Half-Cheetah',
    'HCsquat4dof':'4-DOF Squatting Half-Cheetah',
    'HCsquat6dof':'6-DOF Squatting Half-Cheetah',

    'RealArm3dof':'3-DOF Arm3D',
    'RealArm4dof':'4-DOF Arm3D',
    'RealArm5dof':'5-DOF Arm3D',
    'RealArm6dof':'6-DOF Arm3D',
    'RealArm7dof':'7-DOF Arm3D',

    'VA':'Arm2D',#2-DOF
    'VA4dof':'4-DOF Arm2D',
    'VA6dof':'6-DOF Arm2D',
    'VA8dof':'8-DOF Arm2D',

    'AntRun':'Running Ant',
    'AntSquaT':'Squatting Ant',
    'AntSquaTRedundant':'Squatting Redundant-Ant',

    "HCsquatEalt":'Squatting Half-Cheetah',
    "HCthree":'Running Half-Cheetah',
    "RealArmAll":'Arm3D',

    "HC_sL":'HC symmetry loss vs normal',

}
skip = 7
VA_timestep = [str(x) for x in range(6500, 195000, 6500 * 7)]
VA_xticks=range(1, len(VA_timestep)*7 + 1,7)

AntSquaT_timestep = [str(x) for x in range(2, 51, 2 * 4)]  # *10 4
AntSquaT_xticks=range(1, len(AntSquaT_timestep)*4 + 1,4)
AntSquaT_xlabel=my_as_si(1e4, 2)

AntRun_timestep = [str(x) for x in range(15, 301, 15 * 3)]  # 10 3
AntRun_xticks=range(1, len(AntRun_timestep)*3 + 1,3)
AntRun_xlabel=my_as_si(1e3, 2)

common_xtime=my_as_si(1e5, 2)






