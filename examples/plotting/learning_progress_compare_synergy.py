import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import argparse
try:
    from examples.plotting.commons import *
except:
    from commons import *

plt.rcParams['figure.figsize'] = [15, 12]
plt.rcParams['axes.linewidth'] = 2.
plt.rcParams['font.size'] = 35
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['font.family'] = 'Times New Roman'

LW=5
side=160
LGS=30
DPI=350
trans_rate=0.3

parser = argparse.ArgumentParser()
parser.add_argument('agentt_type',type=str,choices=['AntRun','AntSquaT','HC','HCez','HCAll','HC5dofAll','VA','AntAll','VAez',
                                                    'VAp5','VAall','VAE1','HCsquatep1','HCsquatep25','HCsquat','HCsquatez',
                                                    'HCsquatEalt','RealArmAll','HCthree','RealArmCompareMultiOb','VArange',
                                                    'VAsR','VAbR','HCsL','Walker2dsl','Walker2dAll',"FCHeavyActionManip",
                                                    "FCHeavyOri","FCHeavyTrot","FCHeavyGallop", "FCHeavySpeedp5","FCHeavySpeed1","FCHeavySpeed2","FCHeavySpeed3",
                                                    "FCHeavySpeed4","FCHeavySpeed5","FCHeavySpeed6","FCHeavyUnlimited",
                                                    "FCHeavylessSpring","FCHeavymoreSpring", "FCHeavyminSpring","FCHeavymaxSpring","VarySpeed_maxSG",
                                                    "VarySpeed_minSG","VarySpeed_defS","VarySpeed_defmaxS","VarySpeed_defminS",
                                                    "VarySpeed_maxST","VarySpeed_minST","VarySpring_v4","VarySpring_Gv4",
                                                    "VarySpring_Tv4","VarySpring_UNL","VarySpring_GUNL","VarySpring_TUNL",
                                                    "VaryGait_minSv4","VaryGait_maxSv4","VaryGait_minSv6","VaryGait_maxSv6",
                                                    "VaryGait_minSUNL","VaryGait_maxSUNL",
                                                    ])
parser.add_argument('--div_rate',type=float, default=0.4)
parser.add_argument('--no_div',action='store_true')
parser.add_argument('--no_fixed_scale',action='store_false')
parser.add_argument('--plot_range',type=int, default=0)
parser.add_argument('--spatial_synergy',action='store_true')
args = parser.parse_args()

get_rid_div=args.no_div
div_rate = args.div_rate
fixed_scale=args.no_fixed_scale
agentt_type=args.agentt_type
plot_range=args.plot_range

plot_list= [ 'SA' ,'PI', 'P', 'PxPI']

algo_list=['TD3', 'SAC']
#algo_list=['SAC']
available_list=[]
if agentt_type=='HC':
    available_label_list=['2DOF','3DOFb','3DOFf','4DOF','5DOF','6DOF']

    folder_list=[   HC2dof_folder,
                    HC3dofb_folder,
                    HC3doff_folder,
                    HC4dof_folder,
                    HC5dof_folder,
                    HC_folder
                    ]
elif agentt_type=='HCthree':
    available_label_list=['2DOF','4DOF','6DOF']
    algo_list = ['SAC']
    folder_list=[   HC2dof_folder,
                    HC4dof_folder,
                    HC_folder
                    ]
elif agentt_type=='HCAll':
    available_label_list=['2DOF','3DOFb','3DOFf','4DOF','5DOF','6DOF',
                            '2DOFez','3DOFbez','3DOFfez','4DOFez','5DOFez','6DOFez']
    algo_list = ['SAC']
    folder_list=[ HC2dof_folder,
                    HC3dofb_folder,
                    HC3doff_folder,
                    HC4dof_folder,
                    HC5dof_folder,
                    HC_folder,
                    HC2dofez_folder,
                    HC3dofbez_folder,
                    HC3doffez_folder,
                    HC4dofez_folder,
                    HC5dofez_folder,
                    HCez_folder,
                    ]
elif agentt_type=='HC5dofAll':
    available_label_list=['5DOF','5DOFv2','5DOFv3','5DOFv4','5DOFv5','5DOFv6','6DOF']
    algo_list = ['SAC']
    folder_list=[   HC5dof_folder,
                    HC5dofv2_folder,
                    HC5dofv3_folder,
                    HC5dofv4_folder,
                    HC5dofv5_folder,
                    HC5dofv6_folder,
                    HC_folder,

                    ]
elif agentt_type=='HCez':
    available_label_list=['2DOF','3DOFb','3DOFf','4DOF','5DOF','6DOF']
    algo_list = ['SAC']
    folder_list=[   HC2dofez_folder,
                    HC3dofbez_folder,
                    HC3doffez_folder,
                    HC4dofez_folder,
                    HC5dofez_folder,
                    HCez_folder,
                    ]
elif agentt_type == 'HCsquat':
    available_label_list = ['2', '4', '6']
    algo_list = ['SAC','TD3']
    agent_list = ['HCsquat2dof', 'HCsquat4dof', 'HCsquat6dof']
    folder_list = [HCsquat2dof_folder,
                   HCsquat4dofEp1_folder,
                   HCsquat6dofEp1_folder
                   ]
elif agentt_type == 'HCsquatep1':
    available_label_list = ['2', '4', '6']
    algo_list = ['SAC','TD3']
    agent_list = ['HCsquat2dof', 'HCsquat4dof_Ep1', 'HCsquat6dof_Ep1']
    folder_list = [HCsquat2dof_folder,
                   HCsquat4dofEp1_folder,
                   HCsquat6dofEp1_folder
                   ]
elif agentt_type == 'HCsquatep25':
    available_label_list = ['2', '4', '6']
    algo_list = ['SAC','TD3']
    agent_list = ['HCsquat2dof_Ep25', 'HCsquat4dof_Ep25', 'HCsquat6dof_Ep25']
    folder_list = [HCsquat2dofEp25_folder,
                   HCsquat4dofEp25_folder,
                   HCsquat6dofEp25_folder
                   ]
elif agentt_type == 'HCsquatez':
    available_label_list = ['2', '4', '6']
    algo_list = ['SAC','TD3']
    agent_list = ['HCsquat2dof_Ez', 'HCsquat4dof_Ez', 'HCsquat6dof_Ez']
    folder_list = [HCsquat2dofez_folder,
                   HCsquat4dofez_folder,
                   HCsquat6dofez_folder
                   ]
elif agentt_type == 'HCsquatEalt':
    available_label_list = ['2', '4', '6']
    algo_list = ['SAC','TD3']
    agent_list = ['HCsquat2dof_Ealt', 'HCsquat4dof_Ealt', 'HCsquat6dof_Ealt']
    folder_list = [HCsquat2dofEalt_folder,
                   HCsquat4dofEalt_folder,
                   HCsquat6dofEalt_folder
                   ]
elif agentt_type == 'Walker2dAll':
    available_label_list = ['Waler2d_sL','Waler2d_sD',  'Walker2d_normal']
    algo_list = ['SAC']
    folder_list = [Walker2dsL_folder,
                   Walker2dsD_folder,
                   Waler2d_folder
                   ]
elif agentt_type == 'Walker2dsl':
    available_label_list = ['Waler2d_sL', 'Walker2d_normal']
    algo_list = ['SAC']
    folder_list = [Walker2dsL_folder,
                   Waler2d_folder
                   ]
elif agentt_type=='HCsL':
    available_label_list=['HC_sL','HC_normal']
    algo_list = ['SAC']
    folder_list=[   HCsL_folder,
                    HC_folder
                    ]


elif agentt_type=='FCHeavyActionManip':
    available_label_list=['FCheavy','FCheavy_sLT','FCheavy_sLG']
    algo_list = ['SAC']
    folder_list=[   FCheavy_folder,
                    FCHeavysLT_folder,
                    FCHeavysLG_folder
                    ]
elif agentt_type=='FCHeavyTrot':
    available_label_list=['sp5','s1','s2','s3','s4','s5','unlimited']
    algo_list = ['SAC']
    folder_list=[
                    FCheavysLTv1_folder,
                    FCheavysLTv2_folder,
                    FCheavysLTv3_folder,
                    FCheavysLTv4_folder,
                    FCheavysLTv5_folder,
                    FCheavysLTv6_folder,
                    FCHeavysLT_folder
                ]
elif agentt_type=='FCHeavyGallop':
    available_label_list=['sp5','s1','s2','s3','s4','s5','unlimited']
    algo_list = ['SAC']
    folder_list=[
                    FCheavysLGv1_folder,
                    FCheavysLGv2_folder,
                    FCheavysLGv3_folder,
                    FCheavysLGv4_folder,
                    FCheavysLGv5_folder,
                    FCheavysLGv6_folder,
                    FCHeavysLG_folder
                ]
elif agentt_type == 'FCHeavySpeedp5':

    available_label_list = ['normal_sp5','gallop_sp5','trot_sp5']
    algo_list = ['SAC']
    folder_list = [
        FCheavyE0v1_folder,
        FCheavysLGv1_folder,
        FCheavysLTv1_folder,
    ]
elif agentt_type == 'FCHeavySpeed2':
    available_label_list = ['normal_s2','gallop_s2','trot_s2']
    algo_list = ['SAC']
    folder_list = [
        FCheavyE0v3_folder,
        FCheavysLGv3_folder,
        FCheavysLTv3_folder,
    ]
elif agentt_type == 'FCHeavySpeed4':
    available_label_list = ['normal_s4','gallop_s4','trot_s4']
    algo_list = ['SAC']
    folder_list = [
        FCheavyE0v5_folder,
        FCheavysLGv5_folder,
        FCheavysLTv5_folder,
    ]

elif agentt_type=="VaryGait_minSUNL":
    available_label_list = ['normal', 'gallop', 'trot']  # 's1',

    algo_list = ['SAC']
    folder_list = [
        FCHeavyminS_folder,
        FCHeavyminSG_folder,
        FCHeavyminST_folder
    ]
elif agentt_type=="VaryGait_maxSUNL":
    available_label_list = ['normal', 'gallop', 'trot']  # 's1',

    algo_list = ['SAC']
    folder_list = [
        FCHeavymaxS_folder,
        FCHeavymaxSG_folder,
        FCHeavymaxST_folder
    ]
elif agentt_type=="VaryGait_minSv6":
    available_label_list = ['normal', 'gallop', 'trot']  # 's1',

    algo_list = ['SAC']
    folder_list = [
        FCheavyminSv6_folder,
        FCheavyminSGv6_folder,
        FCheavyminSTv6_folder
    ]
elif agentt_type=="VaryGait_maxSv6":
    available_label_list = ['normal', 'gallop', 'trot']  # 's1',

    algo_list = ['SAC']
    folder_list = [
        FCheavymaxSv6_folder,
        FCheavymaxSGv6_folder,
        FCheavymaxSTv6_folder
    ]
elif agentt_type=="VaryGait_minSv4":
    available_label_list = ['normal', 'gallop', 'trot']  # 's1',

    algo_list = ['SAC']
    folder_list = [
        FCheavyminSv4_folder,
        FCheavyminSGv4_folder,
        FCheavyminSTv4_folder
    ]
elif agentt_type=="VaryGait_maxSv4":
    available_label_list = ['normal', 'gallop', 'trot']  # 's1',

    algo_list = ['SAC']
    folder_list = [
        FCheavymaxSv4_folder,
        FCheavymaxSGv4_folder,
        FCheavymaxSTv4_folder
    ]
elif agentt_type=="VarySpring_UNL":
    available_label_list = ['defaultSpring', 'maxSpring', 'minSpring']  # 's1',

    algo_list = ['SAC']
    folder_list = [
        FCheavy_folder,
        FCHeavymaxS_folder,
        FCHeavyminS_folder,
    ]
elif agentt_type=="VarySpring_GUNL":
    available_label_list = ['maxSpring', 'minSpring']  # 's1',

    algo_list = ['SAC']
    folder_list = [
        FCHeavymaxSG_folder,
        FCHeavyminSG_folder,
    ]
elif agentt_type=="VarySpring_TUNL":
    available_label_list = ['maxSpring', 'minSpring']  # 's1',

    algo_list = ['SAC']
    folder_list = [
        FCHeavymaxST_folder,
        FCHeavyminST_folder,
    ]
elif agentt_type=="VarySpring_Tv4":
    available_label_list = ['maxSpring', 'minSpring']  # 's1',

    algo_list = ['SAC']
    folder_list = [
        FCheavymaxSTv4_folder,
        FCheavyminSTv4_folder,
    ]
elif agentt_type=="VarySpring_Gv4":
    available_label_list = ['maxSpring', 'minSpring']  # 's1',

    algo_list = ['SAC']
    folder_list = [
        FCheavymaxSGv4_folder,
        FCheavyminSGv4_folder,
    ]
elif agentt_type=="VarySpring_v4":
    available_label_list = ['defaultSpring', 'maxSpring', 'minSpring']  # 's1',

    algo_list = ['SAC']
    folder_list = [
        FCheavyE0v4_folder,
        FCheavymaxSv4_folder,
        FCheavyminSv4_folder,
    ]
elif agentt_type == 'VarySpeed_defmaxS':
    #available_label_list = [ 's3', 's5', 'unlimited','unlimited_noE', 's3_largeE', 's5_largeE']#'s1',
    available_label_list = [  's4', 's6', 'UNL']#'s1',

    algo_list = ['SAC']
    folder_list = [
        FCheavymaxSv4_folder,
        FCheavymaxSv6_folder,
        FCHeavymaxS_folder,
    ]
elif agentt_type == 'VarySpeed_defminS':
    #available_label_list = [ 's3', 's5', 'unlimited','unlimited_noE', 's3_largeE', 's5_largeE']#'s1',
    available_label_list = [  's4', 's6', 'UNL']#'s1',

    algo_list = ['SAC']
    folder_list = [
        FCheavyminSv4_folder,
        FCheavyminSv6_folder,
        FCHeavyminS_folder,
    ]
elif agentt_type == 'VarySpeed_defS':
    #available_label_list = [ 's3', 's5', 'unlimited','unlimited_noE', 's3_largeE', 's5_largeE']#'s1',
    available_label_list = [  's4', 's6', 'UNL']#'s1',

    algo_list = ['SAC']
    folder_list = [
        FCheavyE0v4_folder,
        FCheavyE0v6_folder,
        FCheavy_folder,
    ]
elif agentt_type == 'VarySpeed_maxSG':
    #available_label_list = [ 's3', 's5', 'unlimited','unlimited_noE', 's3_largeE', 's5_largeE']#'s1',
    available_label_list = [ 's2', 's4', 's6', 'UNL']#'s1',

    algo_list = ['SAC']
    folder_list = [
        FCheavymaxSGv2_folder,
        FCheavymaxSGv4_folder,
        FCheavymaxSGv6_folder,
        FCHeavymaxSG_folder,
    ]
elif agentt_type == 'VarySpeed_minSG':
    #available_label_list = [ 's3', 's5', 'unlimited','unlimited_noE', 's3_largeE', 's5_largeE']#'s1',
    available_label_list = [ 's2', 's4', 's6', 'UNL']#'s1',

    algo_list = ['SAC']
    folder_list = [
        FCheavyminSGv2_folder,
        FCheavyminSGv4_folder,
        FCheavyminSGv6_folder,
        FCHeavyminSG_folder,
    ]
elif agentt_type == 'VarySpeed_maxST':
    #available_label_list = [ 's3', 's5', 'unlimited','unlimited_noE', 's3_largeE', 's5_largeE']#'s1',
    available_label_list = [ 's2', 's4', 's6', 'UNL']#'s1',

    algo_list = ['SAC']
    folder_list = [
        FCheavymaxSTv2_folder,
        FCheavymaxSTv4_folder,
        FCheavymaxSTv6_folder,
        FCHeavymaxST_folder,
    ]
elif agentt_type == 'VarySpeed_minST':
    #available_label_list = [ 's3', 's5', 'unlimited','unlimited_noE', 's3_largeE', 's5_largeE']#'s1',
    available_label_list = [ 's2', 's4', 's6', 'UNL']#'s1',

    algo_list = ['SAC']
    folder_list = [
        FCheavyminSTv2_folder,
        FCheavyminSTv4_folder,
        FCheavyminSTv6_folder,
        FCHeavyminST_folder,
    ]

elif agentt_type == 'FCHeavyOri':
    available_label_list = [ 's3',  's5','unlimited','unlimited_noE', 's3_largeE', 's5_largeE']#'s1',
    algo_list = ['SAC']
    folder_list = [
        #FCheavyE0v2_folder,
        FCheavyE0v4_folder,
        FCheavyE0v6_folder,
        FCheavy_folder,
        FCheavyE0v00_folder,
        FCheavyE0v45_folder,
        FCheavyE0v65_folder,
    ]
elif agentt_type == 'FCHeavylessSpring':
    available_label_list = [ 's1',  's3',  's5', 'unlimited']

    algo_list = ['SAC']
    folder_list = [
        FCheavylSv2_folder,
        FCheavylSv4_folder,
        FCheavylSv6_folder,
        FCHeavylS_folder,
    ]
elif agentt_type == 'FCHeavymoreSpring':
    available_label_list = ['s1',  's3', 's5', 'unlimited']

    algo_list = ['SAC']
    folder_list = [
        FCheavymSv2_folder,
        FCheavymSv4_folder,
        FCheavymSv6_folder,
        FCHeavymS_folder,
    ]
elif agentt_type == 'FCHeavyminSpring':
    available_label_list = [  's3', 's5', 'unlimited','unlimited_noE', 's3_largeE']#'s1',, 's5_largeE'

    algo_list = ['SAC']
    folder_list = [
        #FCheavyminSv2_folder,
        FCheavyminSv4_folder,
        FCheavyminSv6_folder,
        FCHeavyminS_folder,
        FCheavyminSv00_folder,
        FCheavyminSv45_folder,
        #FCheavyminSv65_folder,
    ]
elif agentt_type == 'FCHeavymaxSpring':
    #available_label_list = [ 's3', 's5', 'unlimited','unlimited_noE', 's3_largeE', 's5_largeE']#'s1',
    available_label_list = [ 's3', 's5', 'unlimited','unlimited_noE', 's3_largeE', 's5_largeE']#'s1',

    algo_list = ['SAC']
    folder_list = [
        #FCheavymaxSv2_folder,
        FCheavymaxSv4_folder,
        FCheavymaxSv6_folder,
        FCHeavymaxS_folder,
        FCheavymaxSv00_folder,
        FCheavymaxSv45_folder,
        FCheavymaxSv65_folder,
    ]
elif agentt_type == 'FCHeavySpeed1':
    available_label_list = ['minSpring_s1','lesSpring_s1','normal_s1','moreSpring_s1','maxSpring_s1']
    algo_list = ['SAC']
    folder_list = [
        FCheavyminSv2_folder,
        FCheavylSv2_folder,
        FCheavyE0v2_folder,
        FCheavymSv2_folder,
        FCheavymaxSv2_folder,
    ]
elif agentt_type == 'FCHeavySpeed3':
    #available_label_list = ['minSpring_s3','lesSpring_s3','normal_s3','moreSpring_s3','maxSpring_s3']
    available_label_list = ['minSpring_s3','minSpring_s3_largeE','normal_s3','normal_s3_largeE','maxSpring_s3', 'maxSpring_s3_largeE']

    algo_list = ['SAC']
    folder_list = [
        FCheavyminSv4_folder,
        FCheavyminSv45_folder,
        #FCheavylSv4_folder,
        FCheavyE0v4_folder,
        FCheavyE0v45_folder,
        #FCheavymSv4_folder,
        FCheavymaxSv4_folder,
        FCheavymaxSv45_folder,
    ]
elif agentt_type == 'FCHeavySpeed5':
    #available_label_list = ['minSpring_s5','lesSpring_s5','normal_s5','moreSpring_s5','maxSpring_s5']
    available_label_list = ['minSpring_s5','normal_s5','normal_s5_largeE','maxSpring_s5', 'maxSpring_s5_largeE']
    available_label_list = ['minSpring_s5','normal_s5','maxSpring_s5']

    algo_list = ['SAC']
    folder_list = [
        FCheavyminSv6_folder,

        #FCheavylSv6_folder,
        FCheavyE0v6_folder,
        #FCheavyE0v65_folder,

        #FCheavymSv6_folder,
        FCheavymaxSv6_folder,
        #FCheavymaxSv65_folder,

    ]
elif agentt_type == 'FCHeavyUnlimited':
    #available_label_list = ['minSpring','lesSpring','normal','moreSpring','maxSpring']
    available_label_list = ['minSpring','normal','maxSpring']
    available_label_list = ['minSpring','minSpring_noE','normal','normal_noE','maxSpring','maxSpring_noE']

    algo_list = ['SAC']
    folder_list = [
        FCHeavyminS_folder,
        FCheavyminSv00_folder,

        #FCHeavylS_folder,
        FCheavy_folder,
        FCheavyE0v00_folder,

        #FCHeavymS_folder,
        FCHeavymaxS_folder,
        FCheavymaxSv00_folder,

    ]

elif agentt_type=='RealArmAll':#'RealArmAllv2'
    available_label_list=['3DOF','4DOF','5DOF','6DOF','7DOF']

    folder_list=[   RealArm3dof_folder,
                    RealArm4dof_folder,
                    RealArm5dof_folder,
                    RealArm6dof_folder,
                    RealArm7dof_folder]
elif agentt_type=='RealArmCompareMultiOb':#'RealArmAllv2'
    available_label_list=['4DOF','5DOF','4DOFMinE','5DOFMinE','4DOFLT','5DOFLT']

    folder_list=[
                    RealArm4dof_folder,
                    RealArm5dof_folder,
                    RealArm4dofMinE_folder,
                    RealArm5dofMinE_folder,
                    RealArm4dofLT_folder,
                    RealArm5dofLT_folder,
                   ]
elif agentt_type == 'VArange':
    available_label_list=['2DOFsR','2DOF','2DOFbR']

    folder_list=[   VAsR_folder,
                    VA_folder,
                    VAbR_folder
                    ]
elif agentt_type == 'VA':
    available_label_list=['2DOF','4DOF','6DOF','8DOF']

    folder_list=[   VA_folder,
                    VA4dof_folder,
                    VA6dof_folder,
                    VA8dof_folder]
elif agentt_type=='VAsR':
    available_label_list = ['2sR','4sR','6sR','8sR']
    folder_list=[   VAsR_folder,
                    VA4dofsR_folder,
                    VA6dofsR_folder,
                    VA8dofsR_folder]
elif agentt_type=='VAbR':
    available_label_list = ['2bR','4bR','6bR','8bR']
    folder_list=[   VAbR_folder,
                    VA4dofbR_folder,
                    VA6dofbR_folder,
                    VA8dofbR_folder]
elif agentt_type == 'VAez':
    available_label_list=['2DOF','4DOF','6DOF','8DOF']
    algo_list = [ 'SAC']
    folder_list=[   VAez_folder,
                    VA4dofez_folder,
                    VA6dofez_folder,
                    VA8dofez_folder]
elif agentt_type == 'VAp5':
    available_label_list=['2DOF','4DOF','6DOF','8DOF']
    algo_list = ['SAC']
    folder_list=[   VAp5_folder,
                    VA4dofp5_folder,
                    VA6dofp5_folder,
                    VA8dofp5_folder]
elif agentt_type == 'VAE1':
    available_label_list=['2DOF','4DOF','6DOF','8DOF']
    algo_list = ['SAC']
    folder_list=[   VAE1_folder,
                    VA4dofE1_folder,
                    VA6dofE1_folder,
                    VA8dofE1_folder]
elif agentt_type == 'VAall':
    available_label_list=['2DOF','4DOF','6DOF','8DOF',
                          '2DOFez','4DOFez','6DOFez','8DOFez',
                          '2DOFp5','4DOFp5','6DOFp5','8DOFp5',
                          '2DOFe1', '4DOFe1', '6DOFe1', '8DOFe1'
                          ]
    algo_list = ['SAC']

    folder_list=[   VA_folder,
                    VA4dof_folder,
                    VA6dof_folder,
                    VA8dof_folder,
                    VAez_folder,
                    VA4dofez_folder,
                    VA6dofez_folder,
                    VA8dofez_folder,
                    VAp5_folder,
                    VA4dofp5_folder,
                    VA6dofp5_folder,
                    VA8dofp5_folder,
                    VAE1_folder,
                    VA4dofE1_folder,
                    VA6dofE1_folder,
                    VA8dofE1_folder]

elif agentt_type == 'AntRun':
    available_label_list = ['AntRun']
    folder_list=[ AntRun_folder]
elif agentt_type == 'AntSquaT':
    available_label_list = ['AntSquat', 'AntSquatRedundant']
    folder_list=[  AntSquaT_folder,
                AntSquaTRedundant_folder]

elif agentt_type == 'AntAll':
    available_label_list = ['AntRun','AntSquat', 'AntSquatRedundant']
    folder_list=[  AntRun_folder,
                   AntSquaT_folder,
                AntSquaTRedundant_folder]
    plot_list = ['SA']

for folder in folder_list:
    if os.path.exists(folder):
        available_list.append(folder)

print(available_list)

def my_as_si(x, ndp):
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    m, e = s.split('e')
    return r'\times 10^{{{e:d}}}'.format(e=int(e))

c_SA='C4'
c_P='C2'
c_PI='C1'


save_path=cwd+'/experiments_results/Synergy/compare_synergy_graphs/'+agentt_type
if not os._exists(save_path):
    os.makedirs(save_path,exist_ok=True)

for algo in algo_list:
    for plot_type in plot_list:#, 'PxPI'
        for with_STD in [True, False]:
            if plot_type == 'PxPI':
                Perf_list = []
                PerfI_list = []
            compare_synergy_SAC, compare_synergy_SAC_ax = plt.subplots(1, 1)

            for ind_top,choice in enumerate(available_list):
                counter_sac=0
                counter_TD3=0
                for f in os.listdir(choice):
                    file_path=os.path.join(choice,f)
                    if 'no_div' in file_path:
                        continue
                    current_file = pd.read_csv(file_path)
                    if 'TD3' not in f:
                        counter_sac=counter_sac+1
                        if counter_sac==1:
                            if args.spatial_synergy:
                                current_SAC_surface_list = np.asarray(current_file['Spatial area'])
                            else:
                                current_SAC_surface_list=np.asarray(current_file['Surface Area'])

                            if 'PP' in current_file:
                                current_SAC_P_list = np.asarray(current_file['PP'])#P
                                current_SAC_PI_list = np.asarray(current_file['PPI'])
                                if(np.sum(current_SAC_P_list)==0):
                                    print("No PP list.")
                                    current_SAC_P_list = np.asarray(current_file['P'])  # P
                                    current_SAC_PI_list = np.asarray(current_file['PI'])
                            else:
                                print("No PP list.")
                                current_SAC_P_list = np.asarray(current_file['P'])  # P
                                current_SAC_PI_list = np.asarray(current_file['PI'])

                            current_SAC_E_list =np.asarray( current_file['E'])
                        else:
                            if args.spatial_synergy:
                                current_SAC_surface_list = np.vstack(
                                    (current_SAC_surface_list, np.asarray(current_file['Spatial area'])))
                            else:
                                current_SAC_surface_list=np.vstack((current_SAC_surface_list,np.asarray(current_file['Surface Area'])))

                            if 'PP' in current_file:
                                if (np.sum(np.asarray(current_file['PP'])) == 0):
                                    current_SAC_P_list = np.vstack(
                                        (current_SAC_P_list, np.asarray(current_file['P'])))  # P
                                    current_SAC_PI_list = np.vstack(
                                        (current_SAC_PI_list, np.asarray(current_file['PI'])))
                                else:
                                    current_SAC_P_list=np.vstack((current_SAC_P_list,np.asarray(current_file['PP'])))#P
                                    current_SAC_PI_list=np.vstack((current_SAC_PI_list,np.asarray(current_file['PPI'])))
                            else:
                                current_SAC_P_list = np.vstack(
                                    (current_SAC_P_list, np.asarray(current_file['P'])))  # P
                                current_SAC_PI_list = np.vstack((current_SAC_PI_list, np.asarray(current_file['PI'])))
                            current_SAC_E_list=np.vstack((current_SAC_E_list,np.asarray(current_file['E'])))

                    else:
                        counter_TD3 = counter_TD3 + 1
                        if counter_TD3 == 1:
                            if args.spatial_synergy:
                                current_TD3_surface_list = np.asarray(current_file['Surface Area'])
                            else:
                                current_TD3_surface_list = np.asarray(current_file['Spatial area'])

                            if 'PP' in current_file:
                                current_TD3_P_list = np.asarray(current_file['PP'])#P
                                current_TD3_PI_list = np.asarray(current_file['PPI'])
                                if (np.sum(current_TD3_P_list) == 0):
                                    current_TD3_P_list = np.asarray(current_file['P'])  # P
                                    current_TD3_PI_list = np.asarray(current_file['PI'])
                            else:
                                current_TD3_P_list = np.asarray(current_file['P'])  # P
                                current_TD3_PI_list = np.asarray(current_file['PI'])
                            current_TD3_E_list = np.asarray(current_file['E'])
                        else:
                            if args.spatial_synergy:
                                current_TD3_surface_list = np.vstack(
                                    (current_TD3_surface_list, np.asarray(current_file['Spatial Area'])))
                            else:
                                current_TD3_surface_list = np.vstack((current_TD3_surface_list, np.asarray(current_file['Surface Area'])))

                            if 'PP' in current_file:
                                if (np.sum(np.asarray(current_file['PP'])) == 0):
                                    current_TD3_P_list = np.vstack(
                                        (current_TD3_P_list, np.asarray(current_file['P'])))  # P
                                    current_TD3_PI_list = np.vstack(
                                        (current_TD3_PI_list, np.asarray(current_file['PI'])))
                                else:
                                    current_TD3_P_list = np.vstack((current_TD3_P_list, np.asarray(current_file['PP'])))#P
                                    current_TD3_PI_list = np.vstack((current_TD3_PI_list, np.asarray(current_file['PPI'])))
                            else:
                                current_TD3_P_list = np.vstack(
                                    (current_TD3_P_list, np.asarray(current_file['P'])))  # P
                                current_TD3_PI_list = np.vstack((current_TD3_PI_list, np.asarray(current_file['PI'])))
                            current_TD3_E_list = np.vstack((current_TD3_E_list, np.asarray(current_file['E'])))


                try:
                    mean_P_SAC=np.flip(np.mean(current_SAC_P_list,axis=0),axis=0)
                    mean_PI_SAC=np.flip(np.mean(current_SAC_PI_list,axis=0),axis=0)
                    mean_surface_SAC=np.flip(np.mean(current_SAC_surface_list,axis=0),axis=0)
                    std_surface_SAC=np.flip(np.std(current_SAC_surface_list,axis=0),axis=0)
                    std_P_SAC=np.flip(np.std(current_SAC_P_list,axis=0),axis=0)
                    std_PI_SAC=np.flip(np.std(current_SAC_PI_list,axis=0),axis=0)


                except:
                    mean_P_SAC = np.flip(current_SAC_P_list, axis=0)
                    mean_PI_SAC = np.flip(current_SAC_PI_list, axis=0)
                    mean_surface_SAC = np.flip(current_SAC_surface_list, axis=0)
                    std_surface_SAC=None
                    std_P_SAC=None
                    std_PI_SAC=None
                try:
                    mean_P_TD3=np.flip(np.mean(current_TD3_P_list,axis=0),axis=0)
                    mean_PI_TD3=np.flip(np.mean(current_TD3_PI_list,axis=0),axis=0)
                    mean_surface_TD3=np.flip(np.mean(current_TD3_surface_list,axis=0),axis=0)
                    std_P_TD3=np.flip(np.std(current_TD3_P_list,axis=0),axis=0)
                    std_PI_TD3=np.flip(np.std(current_TD3_PI_list,axis=0),axis=0)
                    std_surface_TD3=np.flip(np.std(current_TD3_surface_list,axis=0),axis=0)

                except:
                    try:
                        mean_P_TD3=np.flip(np.mean(current_TD3_P_list,axis=0),axis=0)
                        mean_PI_TD3=np.flip(np.mean(current_TD3_PI_list,axis=0),axis=0)
                        mean_surface_TD3=np.flip(np.mean(current_TD3_surface_list,axis=0),axis=0)
                        std_P_TD3 = None
                        std_PI_TD3 = None
                        std_surface_TD3 = None
                    except:
                        print('No TD3')


                try:
                    if get_rid_div:
                        bad_ind_list = []
                        for ind, p in enumerate(mean_PI_SAC):
                            if ind > 0 and ind < (len(mean_PI_SAC) - 1):
                                if abs(p - mean_PI_SAC[ind - 1]) / abs(p) > div_rate and abs(
                                        p - mean_PI_SAC[ind + 1]) / abs(
                                    p) > div_rate:
                                    bad_ind_list.append(ind)

                        if len(bad_ind_list) > 0:
                            print('DIV')

                            '''mean_P_SAC = np.delete(mean_P_SAC, bad_ind_list, 0)
                            mean_PI_SAC = np.delete(mean_PI_SAC, bad_ind_list, 0)
                            mean_surface_SAC = np.delete(mean_surface_SAC, bad_ind_list, 0)
                            try:
                                std_surface_SAC = np.delete(std_surface_SAC, bad_ind_list, 0)
                                std_P_SAC = np.delete(std_P_SAC, bad_ind_list, 0)
                                std_PI_SAC = np.delete(std_PI_SAC, bad_ind_list, 0)
                            except:
                                pass'''
                            for ind in bad_ind_list:
                                replacement=ind-1
                                while replacement in bad_ind_list:
                                    replacement=replacement-1

                                mean_P_SAC[ind]=mean_P_SAC[replacement]
                                mean_PI_SAC[ind] = mean_PI_SAC[replacement]
                                mean_surface_SAC[ind] =mean_surface_SAC[replacement]
                                try:
                                    std_surface_SAC[ind] = std_surface_SAC[replacement]
                                    std_P_SAC[ind] = std_P_SAC[replacement]
                                    std_PI_SAC[ind] = std_PI_SAC[replacement]
                                except:
                                    pass

                except:
                    print('exception in get_rid_div for SAC')

                try:
                    if get_rid_div:
                        bad_ind_list = []
                        for ind, p in enumerate(mean_PI_TD3):
                            if ind > 0 and ind < (len(mean_PI_TD3) - 1):
                                if abs(p - mean_PI_TD3[ind - 1]) / abs(p) > div_rate and abs(
                                        p - mean_PI_TD3[ind + 1]) / abs(
                                    p) > div_rate:
                                    bad_ind_list.append(ind)

                        if len(bad_ind_list) > 0:
                            print('DIV')
                            for ind in bad_ind_list:
                                replacement = ind - 1
                                while replacement in bad_ind_list:
                                    replacement = replacement - 1

                                mean_P_TD3[ind] = mean_P_TD3[replacement]
                                mean_PI_TD3[ind] = mean_PI_TD3[replacement]
                                mean_surface_TD3[ind] = mean_surface_TD3[replacement]
                                try:
                                    std_P_TD3[ind] = std_P_TD3[replacement]
                                    std_PI_TD3[ind] = std_PI_TD3[replacement]
                                    std_surface_TD3[ind] = std_surface_TD3[replacement]
                                except:
                                    pass

                            '''mean_P_TD3 = np.delete(mean_P_TD3, bad_ind_list, 0)
                            mean_PI_TD3 = np.delete(mean_PI_TD3, bad_ind_list, 0)
                            mean_surface_TD3 = np.delete(mean_surface_TD3, bad_ind_list, 0)
                        try:
                            std_P_TD3 = np.delete(std_P_TD3, bad_ind_list, 0)
                            std_PI_TD3 = np.delete(std_PI_TD3, bad_ind_list, 0)
                            std_surface_TD3 = np.delete(std_surface_TD3, bad_ind_list, 0)
                        except:
                            pass'''
                except:
                    print('exception in get_rid_div for TD3')



                if algo=='SAC':
                    if plot_type=='SA':
                        plot_metrtic=mean_surface_SAC
                        try:
                            plot_metrtic_std=std_surface_SAC
                        except:
                            pass
                    elif plot_type=='P':
                        plot_metrtic=mean_P_SAC
                        try:
                            plot_metrtic_std=std_P_SAC
                        except:
                            pass
                    elif plot_type == 'PI':
                        plot_metrtic=mean_PI_SAC
                        try:
                            plot_metrtic_std=std_PI_SAC
                        except:
                            pass
                    elif plot_type == 'PxPI':
                        #plot_metrtic=mean_PI_SAC*mean_P_SAC
                        #plot_metrtic_std=std_PI_SAC
                        Perf_list.append(mean_P_SAC)
                        PerfI_list.append(mean_PI_SAC)
                elif algo=='TD3':
                    if plot_type=='SA':
                        plot_metrtic=mean_surface_TD3
                        plot_metrtic_std=std_surface_TD3
                    elif plot_type=='P':
                        plot_metrtic=mean_P_TD3
                        plot_metrtic_std=std_P_TD3
                    elif plot_type == 'PI':
                        plot_metrtic=mean_PI_TD3
                        plot_metrtic_std=std_PI_TD3
                    elif plot_type == 'PxPI':
                        #plot_metrtic = mean_PI_TD3 * mean_P_TD3
                        Perf_list.append(mean_P_TD3)
                        PerfI_list.append(mean_PI_TD3)




                if plot_type != 'PxPI':
                    if plot_range!=0:
                        compare_synergy_SAC_ax.plot(range(1, len(plot_metrtic[0:plot_range]) + 1), plot_metrtic[0:plot_range],label=available_label_list[ind_top], linewidth=LW, color=color_list[ind_top])#, color=color_list[s]
                        if with_STD :
                            try:
                                compare_synergy_SAC_ax.fill_between(range(1, len(plot_metrtic[0:plot_range]) + 1), plot_metrtic[0:plot_range] + plot_metrtic_std[0:plot_range],
                                                                plot_metrtic[0:plot_range] - plot_metrtic_std[0:plot_range], alpha = trans_rate, facecolor=color_list[ind_top])#, facecolor = c_SA
                            except:
                                pass
                    else:
                        compare_synergy_SAC_ax.plot(range(1, len(plot_metrtic) + 1), plot_metrtic,label=available_label_list[ind_top], linewidth=LW, color=color_list[ind_top])#, color=color_list[s]
                        if with_STD :
                            try:
                                compare_synergy_SAC_ax.fill_between(range(1, len(plot_metrtic) + 1), plot_metrtic + plot_metrtic_std,
                                                                plot_metrtic - plot_metrtic_std, alpha = trans_rate, facecolor=color_list[ind_top])#, facecolor = c_SA
                            except:
                                pass
            if plot_type == 'PxPI' and not with_STD:
                max_P=np.max(Perf_list)
                max_PI=np.max(PerfI_list)
                Perf_list=Perf_list/max_P
                PerfI_list=PerfI_list/max_PI
                plot_metrtic=0.6*Perf_list+0.4*PerfI_list

                for ind_top,pm in enumerate(plot_metrtic):
                    if plot_range != 0:
                        compare_synergy_SAC_ax.plot(range(1, len(pm) + 1), pm, label=available_label_list[ind_top],
                                                    linewidth=LW)  # , color=color_list[s]
                    else:
                        compare_synergy_SAC_ax.plot(range(1, len(pm[0:plot_range]) + 1), pm[0:plot_range], label=available_label_list[ind_top],
                                                    linewidth=LW)  # , color=color_list[s]

            if plot_type == 'SA':
                compare_synergy_SAC_ax.set_title('Synergy level of ' + agentt_type + ' with ' + algo)
                if fixed_scale:
                    try:
                        compare_synergy_SAC_ax.set_ylim(SA_axis_range[agentt_type])
                    except:
                        compare_synergy_SAC_ax.set_ylim([3, 8])
                    # if 'VA' in agentt_type:
                    #     compare_synergy_SAC_ax.set_ylim([3, 9])
                    # elif 'Ant' in agentt_type:
                    #     compare_synergy_SAC_ax.set_ylim([2.5, 8])
                if args.spatial_synergy:
                    compare_synergy_SAC_ax.set_ylabel('Spatial synergy Surface Area')#, color=c_SA

                else:
                    compare_synergy_SAC_ax.set_ylabel('Surface Area')#, color=c_SA
            elif plot_type == 'P':
                compare_synergy_SAC_ax.set_title('Performance of ' + agentt_type + ' with ' + algo)
                if fixed_scale:

                    try:
                        compare_synergy_SAC_ax.set_ylim(P_axis_range[agentt_type])
                    except:
                        compare_synergy_SAC_ax.set_ylim([0, 20000])

                    # if 'VA' in agentt_type:
                    #     compare_synergy_SAC_ax.set_ylim([-1000, 200])
                    # elif 'AntSquaT' == agentt_type:  # agentt_type == 'VA' or agentt_type == 'VA4dof':
                    #     compare_synergy_SAC_ax.set_ylim([-5000, -300])
                    # elif 'AntRun' == agentt_type:  # agentt_type == 'VA' or agentt_type == 'VA4dof':
                    #     compare_synergy_SAC_ax.set_ylim([-4000, 10000])

                compare_synergy_SAC_ax.set_ylabel('Performance')  # , color=c_SA


            elif plot_type == 'PI':
                compare_synergy_SAC_ax.set_title('Performance energy of ' + agentt_type + ' with ' + algo)
                if fixed_scale:
                    try:
                        compare_synergy_SAC_ax.set_ylim(PI_axis_range[agentt_type])
                    except:
                        compare_synergy_SAC_ax.set_ylim([0, 15])

                    '''if 'VA' in  agentt_type :
                        compare_synergy_SAC_ax.set_ylim([-400, 100])
                    elif 'AntSquaT' == agentt_type:  # agentt_type == 'VA' or agentt_type == 'VA4dof':
                        compare_synergy_SAC_ax.set_ylim([-160, 0])
                    elif 'AntRun' == agentt_type:  # agentt_type == 'VA' or agentt_type == 'VA4dof':
                        compare_synergy_SAC_ax.set_ylim([-10, 30])
                    else:
                        compare_synergy_SAC_ax.set_ylim([0, 15])'''

                compare_synergy_SAC_ax.set_ylabel('Performance energy')  # , color=c_SA
            elif plot_type == 'PxPI':
                compare_synergy_SAC_ax.set_title('P x PI of ' + agentt_type + ' with ' + algo)
                compare_synergy_SAC_ax.set_ylabel('P x PI')  # , color=c_SA
                if fixed_scale:
                    if  'VA' in agentt_type :
                        compare_synergy_SAC_ax.set_ylim([0, 22])
                    elif 'AntSquaT' == agentt_type:  # agentt_type == 'VA' or agentt_type == 'VA4dof':
                        compare_synergy_SAC_ax.set_ylim([0, 6])
                    elif 'AntRun' == agentt_type:  # agentt_type == 'VA' or agentt_type == 'VA4dof':
                        compare_synergy_SAC_ax.set_ylim([-1, 2])

            compare_synergy_SAC_ax.legend(loc=0, prop={'size': LGS})
            if 'VA' in agentt_type:

                #skip = 7
                #VA_timestep = [str(x) for x in range(6500, 195000, 6500 * skip)]

                #compare_synergy_SAC_ax.set_xticks(VA_xticks)
                #compare_synergy_SAC_ax.set_xticklabels(VA_timestep)
                #compare_synergy_SAC_ax.set_xlabel("timesteps")
                compare_synergy_SAC_ax.set_xlabel("training checkpoints")

            elif 'AntSquaT' in agentt_type:

                #skip = 4
                #AntSquaT_timestep = [str(x) for x in range(2, 51, 2 * skip)]  # *10 4
                compare_synergy_SAC_ax.set_xticks(AntSquaT_xticks)
                compare_synergy_SAC_ax.set_xticklabels(AntSquaT_timestep)
                # surface_plot_w_ax.set_xlabel("timesteps")
                compare_synergy_SAC_ax.set_xlabel(r"${0:s}$ timesteps".format(AntSquaT_xlabel))

            elif 'AntRun' in agentt_type:

                #skip = 3
                #AntSquaT_timestep = [str(x) for x in range(15, 301, 15 * skip)]  # 10 3
                compare_synergy_SAC_ax.set_xticks(AntRun_xticks)
                compare_synergy_SAC_ax.set_xticklabels(AntRun_timestep)
                # surface_plot_w_ax.set_xlabel("timesteps")
                compare_synergy_SAC_ax.set_xlabel(r"${0:s}$ timesteps".format(AntRun_xlabel))
            elif 'HCsquat' in agentt_type or 'RealArm' in agentt_type:

                # skip = 3
                # AntSquaT_timestep = [str(x) for x in range(15, 301, 15 * skip)]  # 10 3

                compare_synergy_SAC_ax.set_xlabel("Training checkpoints")
            else:
                compare_synergy_SAC_ax.set_xlabel(r"${0:s}$ timesteps".format(common_xtime))

            compare_synergy_SAC.tight_layout()
            if get_rid_div:
                if fixed_scale:
                    if not with_STD or plot_type == 'PxPI':
                        compare_synergy_SAC.savefig(os.path.join(save_path,
                                                                 'Compare_' + plot_type + '_synergy_' + agentt_type + '_' + algo + '_fixed_scale_noStd_noDiv' + '.png'),
                                                    format='png')
                    else:
                        compare_synergy_SAC.savefig(
                            os.path.join(save_path, 'Compare_' + plot_type + '_synergy_' + agentt_type + '_' + algo + '_fixed_scale_noDiv.png'),
                            format='png')
                else:
                    if not with_STD or plot_type == 'PxPI':
                        compare_synergy_SAC.savefig(os.path.join(save_path,
                                                                 'Compare_' + plot_type + '_synergy_' + agentt_type + '_' + algo + '_noStd_noDiv' + '.png'),
                                                    format='png')
                    else:
                        compare_synergy_SAC.savefig(
                            os.path.join(save_path, 'Compare_' + plot_type + '_synergy_' + agentt_type + '_' + algo + '_noDiv.png'),
                            format='png')
            else:
                if fixed_scale:
                    if not with_STD or plot_type == 'PxPI':
                        compare_synergy_SAC.savefig(os.path.join(save_path,
                                                                 'Compare_' + plot_type + '_synergy_' + agentt_type + '_' + algo + '_fixed_scale_noStd' + '.png'),
                                                    format='png')
                    else:
                        compare_synergy_SAC.savefig(
                            os.path.join(save_path, 'Compare_' + plot_type + '_synergy_' + agentt_type + '_' + algo + '_fixed_scale.png'),
                            format='png')
                else:
                    if not with_STD or plot_type == 'PxPI':
                        compare_synergy_SAC.savefig(os.path.join(save_path,
                                                                 'Compare_' + plot_type + '_synergy_' + agentt_type + '_' + algo + '_noStd' + '.png'),
                                                    format='png')
                    else:
                        compare_synergy_SAC.savefig(
                            os.path.join(save_path, 'Compare_' + plot_type + '_synergy_' + agentt_type + '_' + algo + '.png'),
                            format='png')


