import numpy as np
from matplotlib import pyplot as plt
import os
import argparse
import pandas as pd

try:
    from examples.plotting.commons import *
except:
    from commons import *

cmap = plt.cm.viridis
cmaplist = [cmap(i) for i in range(cmap.N)]
cmaplen=len(cmaplist)

color_list=['b','r','g','c','m','y','k','#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']

plt.rcParams['figure.figsize'] = [15, 12]
plt.rcParams['axes.linewidth'] = 2.
plt.rcParams['font.size'] = 35
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['font.family'] = 'Times New Roman'
LW=5
side=160
LGS=30
DPI=350
barwidth=0.4
trans_rate=0.2

path_to_folder=cwd+'/experiments_results/Synergy/all_csv/process_SA_final_summary'

parser = argparse.ArgumentParser()
parser.add_argument('agentt_type',type=str,choices=['Paper1',"Paper2_Arm2D",'Paper2_Arm3D','Ant','HCallv6','HCez','HCE1','HCAll','VA','VAez',
                                                    'VAp5','VAall','VAE1','HC5dofAll','HC5dofAllo',
                                                    'HC5dofo','HCdofAllv2','HCAllfinal1','HC2dofAllo',
                                                    'HCsquat','HCsquatep1','HCsquatep25','HCsquatez',
                                                    'HCsquatEalt','RealArmAll','HCthree','RealArmCompareMultiOb','AntSquaT'
                                                    ,'HCsL','Walker2dsl','Walker2dAll','FCHeavyActionManip',
                                                    "FCHeavyVarySpeed","FCHeavyOri","FCHeavyTrot","FCHeavyGallop",
                                                    "FCHeavySpeedp5","FCHeavySpeed1","FCHeavySpeed2","FCHeavySpeed3",
                                                    "FCHeavySpeed4","FCHeavySpeed5",
                                                    "FCHeavylessSpring","FCHeavymoreSpring","FCHeavyUnlimited","FCHeavyminSpring","FCHeavymaxSpring"
                                                    ])
parser.add_argument('--plot_type',type=str,default='bar',choices=['bar','line'])
parser.add_argument('--no_div',action='store_true')
parser.add_argument('--no_fixed_scale',action='store_false')
parser.add_argument('--double_bars',action='store_true')

args = parser.parse_args()

name_tag=args.agentt_type
get_rid_div=args.no_div
fixed_scale=args.no_fixed_scale
plot_type=args.plot_type
double_bars=args.double_bars

if double_bars:
    plot_type='bar'

if name_tag=='Ant':
    Dof_list = ['AntRun', 'AntSquat', 'AntSquatRedundant']
    agent_list = ['AntRun', 'AntSquaT', 'AntSquaTRedundant']
elif name_tag == 'Paper1':
    Dof_list = ['HC', 'HCheavy','FC']
    agent_list = ['HC','HCheavy',  'FC']
elif name_tag == 'Paper2_Arm2D':
    Dof_list = ['2', '4', '6', '8']
    agent_list=['VA',   'VA4dof',   'VA6dof',   'VA8dof']
elif name_tag == 'Paper2_Arm3D':
    Dof_list = ['3', '4', '5', '6', '7']
    agent_list=['RealArm3dof',   'RealArm4dof',   'RealArm5dof',   'RealArm6dof',   'RealArm7dof']

elif name_tag=='AntSquaT':
    Dof_list = ['AntSquat', 'AntSquatRedundant']
    agent_list = [ 'AntSquaT', 'AntSquaTRedundant']
elif name_tag == 'HCsquat':
    Dof_list = ['2', '4', '6']
    agent_list = ['HCsquat2dof', 'HCsquat4dof', 'HCsquat6dof']
elif name_tag == 'HCsquatep1':
    Dof_list = ['2', '4', '6']
    agent_list = ['HCsquat2dof', 'HCsquat4dof_Ep1', 'HCsquat6dof_Ep1']
elif name_tag == 'HCsquatep25':
    Dof_list = ['2', '4', '6']
    agent_list = ['HCsquat2dof_Ep25', 'HCsquat4dof_Ep25', 'HCsquat6dof_Ep25']
elif name_tag == 'HCsquatez':
    Dof_list = ['2', '4', '6']
    agent_list = ['HCsquat2dof_Ez', 'HCsquat4dof_Ez', 'HCsquat6dof_Ez']
elif name_tag == 'HCsquatEalt':
    Dof_list = ['2', '4', '6']
    agent_list = ['HCsquat2dof_Ealt', 'HCsquat4dof_Ealt', 'HCsquat6dof_Ealt']
elif name_tag=='HCallv6':
    Dof_list = ['2', '3a', '3b', '4', '5', '6']
    agent_list = ['HC2dof', 'HC3dofb', 'HC3doff', 'HC4dof', 'HC5dof', 'HC']
elif name_tag == 'HCthree':
    Dof_list = ['2', '4', '6']
    agent_list = ['HC2dof', 'HC4dof',  'HC']

elif name_tag=='HCez':
    Dof_list = ['2', '3a', '3b', '4', '5', '6']
    agent_list = ['HC2dof_Ez', 'HC3dofb_Ez', 'HC3doff_Ez', 'HC4dof_Ez', 'HC5dof_Ez', 'HC_Ez']
elif name_tag=='HCE1':
    Dof_list = ['2', '3a', '3b', '4', '5', '6']
    agent_list = ['HC2dof_E1', 'HC3dofb_E1', 'HC3doff_E1', 'HC4dof_E1', 'HC5dof_E1', 'HC_E1']
elif name_tag == 'HCAll':
    Dof_list = ['2z','2','2E1', '3az','3a','3aE1', '3bz','3b','3bE1', '4z', '4', '4E1', '5z', '5', '5E1', '6z', '6', '6E1']
    agent_list = ['HC2dof_Ez','HC2dof','HC2dof_E1',
                  'HC3dofb_Ez','HC3dofb','HC3dofb_E1',
                  'HC3doff_Ez', 'HC3doff','HC3doff_E1',
                  'HC4dof_Ez','HC4dof','HC4dof_E1',
                  'HC5dof_Ez', 'HC5dof','HC5dof_E1',
                  'HC_Ez','HC','HC_E1']
elif name_tag == 'HC5dofAll':
    Dof_list = ['6', '5', '5v2', '5v3', '5v4', '5v5', '5v6']
    agent_list = ['HC', 'HC5dof', 'HC5dofv2', 'HC5dofv3', 'HC5dofv4', 'HC5dofv5', 'HC5dofv6']

elif name_tag == 'HC5dofo':
    Dof_list = [ '5', '5v2', '5v3', '5v4', '5v5', '5v6']
    agent_list = [ 'HC5dof', 'HC5dofv2', 'HC5dofv3', 'HC5dofv4', 'HC5dofv5', 'HC5dofv6']

elif name_tag == 'HC5dofAllo':

    Dof_list = ['2', '3a', '3b', '4', '5', '5v2', '5v3', '5v4', '5v5', '5v6' ,'6']
    agent_list = ['HC2dof', 'HC3dofb', 'HC3doff', 'HC4dof', 'HC5dof',
                  'HC5dofv2','HC5dofv3','HC5dofv4','HC5dofv5','HC5dofv6', 'HC']
elif name_tag == 'HCdofAllv2':

    Dof_list = ['2f','2s','2a','2b',
                '2t', '3a', '3b', '4', '5', '5v2', '5v3', '5v4', '5v5', '5v6' ,'6']
    agent_list = ['HC2dofv2', 'HC2dofv3', 'HC2dofv4', 'HC2dofv5',
                  'HC2dof', 'HC3dofb', 'HC3doff', 'HC4dof', 'HC5dof',
                  'HC5dofv2','HC5dofv3','HC5dofv4','HC5dofv5','HC5dofv6', 'HC']
elif name_tag == 'HC2dofAllo':

    Dof_list = ['2f','2s','2a','2b',
                '2t', '3a', '3b', '4', '5' ,'6']
    agent_list = ['HC2dofv2', 'HC2dofv3', 'HC2dofv4', 'HC2dofv5',
                  'HC2dof', 'HC3dofb', 'HC3doff', 'HC4dof', 'HC5dof', 'HC']

elif name_tag == 'HCAllfinal1':

    Dof_list = ['2f','2a','2b',
                '2t', '3a', '3b', '4', '5' ,'6']
    agent_list = ['HC2dofv2', 'HC2dofv4', 'HC2dofv5',
                  'HC2dof', 'HC3dofb', 'HC3doff', 'HC4dof', 'HC5dof', 'HC']
elif name_tag == 'Walker2dsl':
    Dof_list = ['Walker2d_sL', 'Walker2d_normal']
    agent_list = ['Walker2d_sL', 'Walker2d'
                   ]
elif name_tag == 'Walker2dAll':
    Dof_list = ['Walker2d_sL', 'Walker2d_sD','Walker2d_normal']
    agent_list = ['Walker2d_sL','Walker2d_sD', 'Walker2d'
                  ]
elif name_tag == 'FCHeavyActionManip':
    Dof_list = ['FC_Heavy', 'FC_Heavy_Trot','FC_Heavy_Gallop']
    agent_list = ['FCheavy', 'FCheavy_sLT','FCheavy_sLG'
                  ]
elif name_tag=='FCHeavyOri':
    Dof_list=['s1','s3', 's3_BE','s5', 's5_BE','UNL_noE','UNL']
    agent_list = [ 'FCheavyE0v2',  "FCheavyE0v4",  "FCheavyE0v45",'FCheavyE0v6','FCheavyE0v65','FCheavyE0v00','FCheavy' ,]

elif name_tag=='FCHeavylessSpring':
    Dof_list=['s1','s3','s5','unlimited']
    agent_list = [ 'FCheavylSv2',  "FCheavylSv4",'FCheavylSv6','FCheavy_lS' ]

elif name_tag=='FCHeavymoreSpring':
    Dof_list=['s1','s3','s5','unlimited']
    agent_list = [ 'FCheavymSv2',  "FCheavymSv4",'FCheavymSv6','FCheavy_mS' ]
elif name_tag=='FCHeavyminSpring':
    Dof_list=['s1','s3', 's3_BE','s5','UNL_noE','UNL']
    agent_list = [ 'FCheavyminSv2',  "FCheavyminSv4",  "FCheavyminSv45",'FCheavyminSv6','FCheavyminSv00','FCheavy_minS' ]

elif name_tag=='FCHeavymaxSpring':
    Dof_list=['s1','s3', 's3_BE','s5','s5_BE','UNL_noE','UNL']
    agent_list = [ 'FCheavymaxSv2',  "FCheavymaxSv4",  "FCheavymaxSv45",'FCheavymaxSv6','FCheavymaxSv65','FCheavymaxSv00','FCheavy_maxS' ]

elif name_tag == 'FCHeavyGallop':
    Dof_list = ['sp5', 's1', 's2', 's3', 's4', 's5', 'unlimited']
    agent_list = ["FCheavysLGv1", 'FCheavysLGv2', 'FCheavysLGv3',
                  "FCheavysLGv4", 'FCheavysLGv5', 'FCheavysLGv6', 'FCheavy_sLG'
                  ]
elif name_tag == 'FCHeavyTrot':
    Dof_list = ['sp5', 's1', 's2', 's3', 's4', 's5', 'unlimited']
    agent_list = ["FCheavysLTv1", 'FCheavysLTv2', 'FCheavysLTv3',
                  "FCheavysLTv4", 'FCheavysLTv5', 'FCheavysLTv6', 'FCheavy_sLT'
                  ]
elif name_tag == 'FCHeavySpeedp5':
    Dof_list = ['normal_sp5','gallop_sp5','trot_sp5']
    agent_list = ["FCheavyE0v1","FCheavysLGv1", 'FCheavysLTv1'
                  ]
elif name_tag == 'FCHeavySpeed1':
    Dof_list = ['minS','lessS','N','moreS','maxS']
    agent_list = ['FCheavyminSv2',"FCheavylSv2","FCheavyE0v2", 'FCheavymSv2' ,'FCheavymaxSv2'
                  ]
elif name_tag == 'FCHeavySpeed2':
    Dof_list = ['normal_s2','gallop_s2','trot_s2']
    agent_list = ["FCheavyE0v3","FCheavysLGv3", 'FCheavysLTv3'
                  ]
elif name_tag == 'FCHeavySpeed3':
    # Dof_list = ['minS','minSBE','N','NBE','maxS','maxSBE']#,'lS','mS'
    # agent_list = ['FCheavyminSv4','FCheavyminSv45',"FCheavyE0v4","FCheavyE0v45", 'FCheavymaxSv4',  "FCheavymaxSv45"
    #               ]#"FCheavylSv4",, 'FCheavymSv4'
    Dof_list = ['minS','lessS','N','moreS','maxS']#,'lS','mS'
    agent_list = ['FCheavyminSv4',"FCheavylSv4","FCheavyE0v4",  'FCheavymSv4','FCheavymaxSv4'
                  ]#"FCheavylSv4",, 'FCheavymSv4'
elif name_tag == 'FCHeavySpeed4':
    Dof_list = ['normal_s4','gallop_s4','trot_s4']
    agent_list = ["FCheavyE0v5","FCheavysLGv5", 'FCheavysLTv5'
                  ]
elif name_tag == 'FCHeavySpeed5':
    #Dof_list = ['minSpring','lessSpring','normal','moreSpring','maxSpring']
    #Dof_list = ['minS','N','NBE','maxS','maxSBE']#,'lS','mS'
    Dof_list = ['minS', 'lessS', 'N', 'moreS', 'maxS']

    # agent_list = [ 'FCheavyminSv6',"FCheavylSv6","FCheavyE0v6", 'FCheavymSv6', 'FCheavymaxSv6'
    #               ]
    # agent_list = ['FCheavyminSv6',"FCheavyE0v6","FCheavyE0v65", 'FCheavymaxSv6',  "FCheavymaxSv65"
    #               ]#"FCheavylSv4",, 'FCheavymSv4'
    agent_list = ['FCheavyminSv6',"FCheavylSv6","FCheavyE0v6" ,"FCheavymSv6",'FCheavymaxSv6'
                  ]#"FCheavylSv4",, 'FCheavymSv4'
elif name_tag == 'FCHeavyUnlimited':
    # Dof_list = ['minSpring','lessSpring','normal','moreSpring','maxSpring']
    # agent_list = [ 'FCheavy_minS',"FCheavy_lS","FCheavy", 'FCheavy_mS', 'FCheavy_maxS'
    #               ]
    # Dof_list = ['minS','minS_noE','N','N_noE','maxS','maxS_noE']
    # agent_list = [ 'FCheavy_minS','FCheavyminSv00',"FCheavy","FCheavyE0v00", 'FCheavy_maxS','FCheavymaxSv00'
    #               ]
    Dof_list =['minS', 'lessS', 'N', 'moreS', 'maxS']
    agent_list = [ 'FCheavy_minS',"FCheavy_lS","FCheavy", 'FCheavy_mS', 'FCheavy_maxS'
                  ]
elif name_tag == 'HCsL':
    Dof_list = ['HC_sL', 'HC_normal']
    agent_list = ['HC_sL', 'HC'
                   ]

elif name_tag=='RealArmAll':
    Dof_list = ['3', '4', '5', '6', '7']
    agent_list=['RealArm3dof',   'RealArm4dof',   'RealArm5dof',   'RealArm6dof',   'RealArm7dof']

elif name_tag == 'RealArmCompareMultiOb':  # 'RealArmAllv2'
    Dof_list = ['4', '5', '4MinE', '5MinE', '4LT', '5LT']
    agent_list = ['RealArm4dof', 'RealArm5dof', 'RealArm4dofMinE', 'RealArm5dofMinE',
                  'RealArm4dofLT', 'RealArm5dofLT']

elif name_tag=='VA':
    Dof_list = ['2', '4', '6', '8']
    agent_list=['VA',   'VA4dof',   'VA6dof',   'VA8dof']
elif name_tag=='VAez':
    Dof_list = ['2', '4', '6', '8']
    agent_list=['VA_Ez',   'VA4dof_Ez',   'VA6dof_Ez',   'VA8dof_Ez']
elif name_tag=='VAE1':
    Dof_list = ['2', '4', '6', '8']
    agent_list=['VA_E1',   'VA4dof_E1',   'VA6dof_E1',   'VA8dof_E1']
elif name_tag=='VAp5':
    Dof_list = ['2', '4', '6', '8']
    agent_list=['VA_Ep5',   'VA4dof_Ep5',   'VA6dof_Ep5',   'VA8dof_Ep5']
elif name_tag=='VAall':
    Dof_list = ['2z', '2', '2p5','2e1', '4z','4', '4p5','4e1', '6z', '6','6p5','6e1',
                '8z', '8', '8p5','8e1']

    agent_list=['VA_Ez', 'VA', 'VA_Ep5','VA_E1',  'VA4dof_Ez',   'VA4dof',   'VA4dof_Ep5',
                 'VA4dof_E1', 'VA6dof_Ez',    'VA6dof',    'VA6dof_Ep5', 'VA6dof_E1',
                'VA8dof_Ez','VA8dof',  'VA8dof_Ep5'  ,'VA8dof_E1' ]

FSA_SAC = []
FSA_SAC_std = []
FSA_TD3 = []
FSA_TD3_std = []

DSA_SAC = []
DSA_SAC_std = []
DSA_TD3 = []
DSA_TD3_std = []

ASA_SAC = []
ASA_SAC_std = []
ASA_TD3 = []
ASA_TD3_std = []


for agent in agent_list:
    path_to_agent=os.path.join(path_to_folder,agent)
    for csv in os.listdir(path_to_agent):
        if get_rid_div:
            if 'no_div' in csv:
                path_to_csv = os.path.join(path_to_agent, csv)
                csv_file = pd.read_csv(path_to_csv)
            else:
                continue
        else:
            if 'no_div' not in csv:
                path_to_csv = os.path.join(path_to_agent, csv)
                csv_file = pd.read_csv(path_to_csv)
            else:
                continue
        try:
            FSA_mean = csv_file['FPP mean']
            FSA_std = csv_file['FPP std']

            DSA_mean = csv_file['FPPI mean']
            DSA_std = csv_file['FPPI std']
            if(np.sum(FSA_mean)==0):
                FSA_mean = csv_file['FP mean']
                FSA_std = csv_file['FP std']

                DSA_mean = csv_file['FPI mean']
                DSA_std = csv_file['FPI std']

        except:
            FSA_mean = csv_file['FP mean']
            FSA_std = csv_file['FP std']

            DSA_mean = csv_file['FPI mean']
            DSA_std = csv_file['FPI std']

        ASA_mean = csv_file['FE mean']
        ASA_std = csv_file['FE std']

    FSA_SAC.append(FSA_mean[0])
    FSA_SAC_std.append(FSA_std[0])

    FSA_TD3.append(FSA_mean[1])
    FSA_TD3_std.append(FSA_std[1])

    DSA_SAC.append(DSA_mean[0])
    DSA_SAC_std.append(DSA_std[0])

    DSA_TD3.append(DSA_mean[1])
    DSA_TD3_std.append(DSA_std[1])

    ASA_SAC.append(ASA_mean[0])
    ASA_SAC_std.append(ASA_std[0])

    ASA_TD3.append(ASA_mean[1])
    ASA_TD3_std.append(ASA_std[1])

FSA_SAC=np.asarray(FSA_SAC)
FSA_SAC_std=np.asarray(FSA_SAC_std)
FSA_TD3=np.asarray(FSA_TD3)
FSA_TD3_std=np.asarray(FSA_TD3_std)

DSA_SAC=np.asarray(DSA_SAC)
DSA_SAC_std=np.asarray(DSA_SAC_std)
DSA_TD3=np.asarray(DSA_TD3)
DSA_TD3_std=np.asarray(DSA_TD3_std)

ASA_SAC=np.asarray(ASA_SAC)
ASA_SAC_std=np.asarray(ASA_SAC_std)
ASA_TD3=np.asarray(ASA_TD3)
ASA_TD3_std=np.asarray(ASA_TD3_std)

if not args.double_bars:
    FSA_fig, FSA_fig_ax = plt.subplots(1, 1)

    if plot_type=='line':
        FSA_fig_ax.plot(range(1, len(FSA_SAC) + 1), FSA_SAC, color='b', linewidth=LW)
        FSA_fig_ax.fill_between(range(1, len(FSA_SAC) + 1), FSA_SAC + FSA_SAC_std,
                                       FSA_SAC - FSA_SAC_std, facecolor='b', alpha=trans_rate)
    else:
        FSA_fig_ax.bar(range(1, len(FSA_SAC) + 1), FSA_SAC, yerr=FSA_SAC_std, color='b', width=barwidth, edgecolor='white')

    FSA_fig_ax.set_xticks(range(1,len(Dof_list)+1), )
    FSA_fig_ax.set_xticklabels(Dof_list)

    FSA_fig_ax.set_title('FP vs DOF '+name_tag+' SAC')
    FSA_fig_ax.set_ylabel('FP')
    if name_tag == 'Ant'or "FCHeavy" in name_tag:
        FSA_fig_ax.set_xlabel('Type of agent')
    else:
        FSA_fig_ax.set_xlabel('DOF')

    if not os.path.exists(cwd + '/experiments_results/Synergy/dof_P_'+plot_type+'_plot/'+name_tag):
        os.makedirs(cwd + '/experiments_results/Synergy/dof_P_'+plot_type+'_plot/'+name_tag)
    '''if fixed_scale:
        FSA_fig_ax.set_ylim([3, 8.5])
        if name_tag=='VA':
            FSA_fig_ax.set_ylim([3, 9])
        elif name_tag == 'Ant':
            FSA_fig_ax.set_ylim([3,7])'''
    if get_rid_div:
        if fixed_scale:

            plt.savefig(cwd + '/experiments_results/Synergy/dof_P_'+plot_type+'_plot/'+name_tag+'/FP_'+plot_type+'_SAC_fixed_scale_'+name_tag+'_no_div.png')
        else:
            plt.savefig(cwd + '/experiments_results/Synergy/dof_P_'+plot_type+'_plot/'+name_tag+'/FP_'+plot_type+'_SAC_'+name_tag+'_no_div.png')
    else:
        if fixed_scale:

            plt.savefig(
                cwd + '/experiments_results/Synergy/dof_P_'+plot_type+'_plot/' + name_tag + '/FP_'+plot_type+'_SAC_fixed_scale_' + name_tag + '.png')
        else:
            plt.savefig(
                cwd + '/experiments_results/Synergy/dof_P_'+plot_type+'_plot/' + name_tag + '/FP_'+plot_type+'_SAC_' + name_tag + '.png')


    FSA_fig, FSA_fig_ax = plt.subplots(1, 1)


    if plot_type=='line':
        FSA_fig_ax.plot(range(1, len(FSA_TD3) + 1), FSA_TD3, color='b', linewidth=LW)
        FSA_fig_ax.fill_between(range(1, len(FSA_TD3) + 1), FSA_TD3 + FSA_TD3_std,
                                FSA_TD3 - FSA_TD3_std, facecolor='b', alpha=trans_rate)
    else:
        FSA_fig_ax.bar(range(1, len(FSA_TD3) + 1), FSA_TD3, yerr=FSA_TD3_std, color='b', width=barwidth, edgecolor='white')



    FSA_fig_ax.set_xticks(range(1,len(Dof_list)+1), )
    FSA_fig_ax.set_xticklabels(Dof_list)

    FSA_fig_ax.set_title('FP vs DOF '+name_tag+' TD3')
    FSA_fig_ax.set_ylabel('FP')
    if name_tag == 'Ant'or "FCHeavy" in name_tag:
        FSA_fig_ax.set_xlabel('Type of agent')
    else:
        FSA_fig_ax.set_xlabel('DOF')


    '''if fixed_scale:
        FSA_fig_ax.set_ylim([3, 8.5])
        if name_tag=='VA':
            FSA_fig_ax.set_ylim([3, 9])
        elif name_tag == 'Ant':
            FSA_fig_ax.set_ylim([3,7])'''
    if get_rid_div:
        if fixed_scale:

            plt.savefig(cwd + '/experiments_results/Synergy/dof_P_'+plot_type+'_plot/'+name_tag+'/FP_'+plot_type+'_TD3_fixed_scale_'+name_tag+'_no_div.png')
        else:
            plt.savefig(cwd + '/experiments_results/Synergy/dof_P_'+plot_type+'_plot/'+name_tag+'/FP_'+plot_type+'_TD3_'+name_tag+'_no_div.png')
    else:
        if fixed_scale:

            plt.savefig(
                cwd + '/experiments_results/Synergy/dof_P_'+plot_type+'_plot/' + name_tag + '/FP_'+plot_type+'_TD3_fixed_scale_' + name_tag + '.png')
        else:
            plt.savefig(
                cwd + '/experiments_results/Synergy/dof_P_'+plot_type+'_plot/' + name_tag + '/FP_'+plot_type+'_TD3_' + name_tag + '.png')


    DSA_fig, DSA_fig_ax = plt.subplots(1, 1)


    if plot_type=='line':
        DSA_fig_ax.plot(range(1, len(DSA_SAC) + 1), DSA_SAC, color='r', linewidth=LW)
        DSA_fig_ax.fill_between(range(1, len(DSA_SAC) + 1), DSA_SAC + DSA_SAC_std,
                                DSA_SAC - DSA_SAC_std, facecolor='r', alpha=trans_rate)
    else:
        DSA_fig_ax.bar(range(1, len(DSA_SAC) + 1), DSA_SAC, yerr=DSA_SAC_std, color='r', width=barwidth, edgecolor='white')




    DSA_fig_ax.set_xticks(range(1,len(Dof_list)+1), )
    DSA_fig_ax.set_xticklabels(Dof_list)

    DSA_fig_ax.set_title('FPI vs DOF '+name_tag+' SAC')
    DSA_fig_ax.set_ylabel('FPI')
    if name_tag == 'Ant' or "FCHeavy" in name_tag:
        DSA_fig_ax.set_xlabel('Type of agent')
    else:
        DSA_fig_ax.set_xlabel('DOF')

    '''if fixed_scale:
        DSA_fig_ax.set_ylim([-1, 5])
        if name_tag=='VA':
            DSA_fig_ax.set_ylim([-3, 6])
        elif name_tag == 'Ant':
            DSA_fig_ax.set_ylim([0,4])'''
    if get_rid_div:
        if fixed_scale:

            plt.savefig(cwd + '/experiments_results/Synergy/dof_P_'+plot_type+'_plot/'+name_tag+'/FPI_'+plot_type+'_SAC_fixed_scale_'+name_tag+'_no_div.png')
        else:
            plt.savefig(cwd + '/experiments_results/Synergy/dof_P_'+plot_type+'_plot/'+name_tag+'/FPI_'+plot_type+'_SAC_'+name_tag+'_no_div.png')
    else:
        if fixed_scale:

            plt.savefig(cwd + '/experiments_results/Synergy/dof_P_'+plot_type+'_plot/'+name_tag+'/FPI_'+plot_type+'_SAC_fixed_scale_'+name_tag+'.png')
        else:
            plt.savefig(cwd + '/experiments_results/Synergy/dof_P_'+plot_type+'_plot/'+name_tag+'/FPI_'+plot_type+'_SAC_'+name_tag+'.png')

    #plt.show()

    DSA_fig, DSA_fig_ax = plt.subplots(1, 1)


    if plot_type=='line':
        DSA_fig_ax.plot(range(1, len(DSA_TD3) + 1), DSA_TD3, color='r', linewidth=LW)
        DSA_fig_ax.fill_between(range(1, len(DSA_TD3) + 1), DSA_TD3 + DSA_TD3_std,
                                DSA_TD3 - DSA_TD3_std, facecolor='r', alpha=trans_rate)
    else:
        DSA_fig_ax.bar(range(1, len(DSA_TD3) + 1), DSA_TD3, yerr=DSA_TD3_std, color='r', width=barwidth, edgecolor='white')



    DSA_fig_ax.set_xticks(range(1,len(Dof_list)+1), )
    DSA_fig_ax.set_xticklabels(Dof_list)

    DSA_fig_ax.set_title('FPI vs DOF '+name_tag+' TD3')
    DSA_fig_ax.set_ylabel('FPI')
    if name_tag == 'Ant'or "FCHeavy" in name_tag:
        DSA_fig_ax.set_xlabel('Type of agent')
    else:
        DSA_fig_ax.set_xlabel('DOF')
    '''if fixed_scale:
        DSA_fig_ax.set_ylim([-1, 5])
        if name_tag=='VA':
            DSA_fig_ax.set_ylim([-3, 6])
        elif name_tag == 'Ant':
            DSA_fig_ax.set_ylim([0,4])'''
    if get_rid_div:
        if fixed_scale:

            plt.savefig(cwd + '/experiments_results/Synergy/dof_P_'+plot_type+'_plot/'+name_tag+'/FPI_'+plot_type+'_TD3_fixed_scale_'+name_tag+'_no_div.png')
        else:
            plt.savefig(cwd + '/experiments_results/Synergy/dof_P_'+plot_type+'_plot/'+name_tag+'/FPI_'+plot_type+'_TD3_'+name_tag+'_no_div.png')
    else:
        if fixed_scale:

            plt.savefig(cwd + '/experiments_results/Synergy/dof_P_'+plot_type+'_plot/'+name_tag+'/FPI_'+plot_type+'_TD3_fixed_scale_'+name_tag+'.png')
        else:
            plt.savefig(cwd + '/experiments_results/Synergy/dof_P_'+plot_type+'_plot/'+name_tag+'/FPI_'+plot_type+'TD3_'+name_tag+'.png')

    ASA_fig, ASA_fig_ax = plt.subplots(1, 1)


    if plot_type=='line':
        ASA_fig_ax.plot(range(1, len(ASA_SAC) + 1), ASA_SAC, color='g', linewidth=LW)
        ASA_fig_ax.fill_between(range(1, len(ASA_SAC) + 1), ASA_SAC + ASA_SAC_std,
                                ASA_SAC - ASA_SAC_std, facecolor='g', alpha=trans_rate)
    else:
        ASA_fig_ax.bar(range(1, len(ASA_SAC) + 1), ASA_SAC, yerr=ASA_SAC_std, color='g', width=barwidth, edgecolor='white')



    ASA_fig_ax.set_xticks(range(1,len(Dof_list)+1), )
    ASA_fig_ax.set_xticklabels(Dof_list)

    ASA_fig_ax.set_title('FE vs DOF '+name_tag+' SAC')
    ASA_fig_ax.set_ylabel('FE')
    if name_tag == 'Ant'or "FCHeavy" in name_tag:
        ASA_fig_ax.set_xlabel('Type of agent')
    else:
        ASA_fig_ax.set_xlabel('DOF')
    '''if fixed_scale:
        if name_tag=='VA':
            ASA_fig_ax.set_ylim([-0.25, 6])
        elif name_tag=='Ant':
            ASA_fig_ax.set_ylim([1, 4.5])
        elif name_tag=='HC':
            ASA_fig_ax.set_ylim([0, 5])'''
    if get_rid_div:
        if fixed_scale:
            plt.savefig(cwd + '/experiments_results/Synergy/dof_P_'+plot_type+'_plot/'+name_tag+'/FE_'+plot_type+'_SAC_fixed_scale_'+name_tag+'_no_div.png')
        else:
            plt.savefig(cwd + '/experiments_results/Synergy/dof_P_'+plot_type+'_plot/'+name_tag+'/FE_'+plot_type+'_SAC_'+name_tag+'_no_div.png')
    else:
        if fixed_scale:
            plt.savefig(
                cwd + '/experiments_results/Synergy/dof_P_'+plot_type+'_plot/' + name_tag + '/FE_'+plot_type+'_SAC_fixed_scale_' + name_tag + '.png')
        else:
            plt.savefig(
                cwd + '/experiments_results/Synergy/dof_P_'+plot_type+'_plot/' + name_tag + '/FE_'+plot_type+'_SAC_' + name_tag + '.png')

    ASA_fig, ASA_fig_ax = plt.subplots(1, 1)


    if plot_type=='line':
        ASA_fig_ax.plot(range(1, len(ASA_TD3) + 1), ASA_TD3, color='g', linewidth=LW)
        ASA_fig_ax.fill_between(range(1, len(ASA_TD3) + 1), ASA_TD3 + ASA_TD3_std,
                                ASA_TD3 - ASA_TD3_std, facecolor='g', alpha=trans_rate)
    else:
        ASA_fig_ax.bar(range(1, len(ASA_TD3) + 1), ASA_TD3, yerr=ASA_TD3_std, color='g', width=barwidth, edgecolor='white')



    ASA_fig_ax.set_xticks(range(1,len(Dof_list)+1), )
    ASA_fig_ax.set_xticklabels(Dof_list)

    ASA_fig_ax.set_title('FE vs DOF '+name_tag+' TD3')
    ASA_fig_ax.set_ylabel('FE')
    if name_tag == 'Ant'or "FCHeavy" in name_tag:
        ASA_fig_ax.set_xlabel('Type of agent')
    else:
        ASA_fig_ax.set_xlabel('DOF')

    '''if fixed_scale:
        if name_tag=='VA':
            ASA_fig_ax.set_ylim([-0.25, 6])
        elif name_tag=='Ant':
            ASA_fig_ax.set_ylim([1, 4.5])
        elif name_tag=='HC':
            ASA_fig_ax.set_ylim([0, 5])'''
    if get_rid_div:
        if fixed_scale:

            plt.savefig(cwd + '/experiments_results/Synergy/dof_P_'+plot_type+'_plot/'+name_tag+'/FE_'+plot_type+'_TD3_fixed_scale_'+name_tag+'_no_div.png')
        else:
            plt.savefig(cwd + '/experiments_results/Synergy/dof_P_'+plot_type+'_plot/'+name_tag+'/FE_'+plot_type+'_TD3_'+name_tag+'_no_div.png')#svg
    else:
        if fixed_scale:

            plt.savefig(
                cwd + '/experiments_results/Synergy/dof_P_'+plot_type+'_plot/' + name_tag + '/FE_'+plot_type+'_TD3_fixed_scale_' + name_tag + '.png')
        else:
            plt.savefig(
                cwd + '/experiments_results/Synergy/dof_P_'+plot_type+'_plot/' + name_tag + '/FE_'+plot_type+'_TD3_' + name_tag + '.png')  # svg
else:

    sac_color='#7f6d5f'
    td3_color='#557f2d'

    FSA_fig, FSA_fig_ax = plt.subplots(1, 1)

    r1 = range(1, len(FSA_SAC) + 1)
    r2 = [x + barwidth for x in r1]

    FSA_fig_ax.bar(r1, FSA_SAC, yerr=FSA_SAC_std, color=sac_color, width=barwidth,
                   edgecolor='white', label='SAC')
    FSA_fig_ax.bar(r2, FSA_TD3, yerr=FSA_TD3_std, color=td3_color, width=barwidth,
                   edgecolor='white', label='TD3')

    FSA_fig_ax.set_xticks(range(1, len(Dof_list) + 1), )
    FSA_fig_ax.set_xticklabels(Dof_list)

    FSA_fig_ax.set_title('FP vs DOF ' + name_tag )
    FSA_fig_ax.set_ylabel('FP')
    FSA_fig_ax.legend(loc=2)
    if name_tag == 'Ant'or "FCHeavy" in name_tag:
        FSA_fig_ax.set_xlabel('Type of agent',fontweight='bold')
    else:
        FSA_fig_ax.set_xlabel('DOF')

    if not os.path.exists(cwd + '/experiments_results/Synergy/dof_P_double_bars_plot/' + name_tag):
        os.makedirs(cwd + '/experiments_results/Synergy/dof_P_double_bars_plot/' + name_tag)

    '''if fixed_scale:
        FSA_fig_ax.set_ylim([3, 8.5])
        if name_tag == 'VA':
            FSA_fig_ax.set_ylim([3, 9])
        elif name_tag == 'Ant':
            FSA_fig_ax.set_ylim([3, 7])'''

    if get_rid_div:
        if fixed_scale:

            plt.savefig(
                cwd + '/experiments_results/Synergy/dof_P_double_bars_plot/' + name_tag + '/FP_double_bars_fixed_scale_' + name_tag + '_no_div.png')
        else:
            plt.savefig(
                cwd + '/experiments_results/Synergy/dof_P_double_bars_plot/' + name_tag + '/FP_double_bars_' + name_tag + '_no_div.png')
    else:
        if fixed_scale:

            plt.savefig(
                cwd + '/experiments_results/Synergy/dof_P_double_bars_plot/' + name_tag + '/FP_double_bars_fixed_scale_' + name_tag + '.png')
        else:
            plt.savefig(
                cwd + '/experiments_results/Synergy/dof_P_double_bars_plot/' + name_tag + '/FP_double_bars_' + name_tag + '.png')



    DSA_fig, DSA_fig_ax = plt.subplots(1, 1)

    DSA_fig_ax.bar(r1, DSA_SAC, yerr=DSA_SAC_std, color=sac_color, width=barwidth,
                       edgecolor='white', label='SAC')
    DSA_fig_ax.bar(r2, DSA_TD3, yerr=DSA_TD3_std, color=td3_color, width=barwidth,
                   edgecolor='white', label='TD3')

    DSA_fig_ax.set_xticks(range(1, len(Dof_list) + 1), )
    DSA_fig_ax.set_xticklabels(Dof_list)

    DSA_fig_ax.set_title('FPI vs DOF ' + name_tag )
    DSA_fig_ax.set_ylabel('FPI')
    DSA_fig_ax.legend(loc=2)
    if name_tag == 'Ant'or "FCHeavy" in name_tag:
        DSA_fig_ax.set_xlabel('Type of agent',fontweight='bold')
    else:
        DSA_fig_ax.set_xlabel('DOF')

    if fixed_scale:

        if name_tag == 'HC':
            DSA_fig_ax.set_ylim([0, 16])

    if get_rid_div:
        if fixed_scale:

            plt.savefig(
                cwd + '/experiments_results/Synergy/dof_P_double_bars_plot/' + name_tag + '/FPI_double_bars_fixed_scale_' + name_tag + '_no_div.png')
        else:
            plt.savefig(
                cwd + '/experiments_results/Synergy/dof_P_double_bars_plot/' + name_tag + '/FPI_double_bars_' + name_tag + '_no_div.png')
    else:
        if fixed_scale:

            plt.savefig(
                cwd + '/experiments_results/Synergy/dof_P_double_bars_plot/' + name_tag + '/FPI_' + plot_type + '_fixed_scale_' + name_tag + '.png')
        else:
            plt.savefig(
                cwd + '/experiments_results/Synergy/dof_P_double_bars_plot/' + name_tag + '/FPI_' + plot_type + '_' + name_tag + '.png')

    # plt.show()


    ASA_fig, ASA_fig_ax = plt.subplots(1, 1)

    ASA_fig_ax.bar(r1, ASA_SAC, yerr=ASA_SAC_std, color=sac_color, width=barwidth,
                       edgecolor='white', label='SAC')
    ASA_fig_ax.bar(r2, ASA_TD3, yerr=ASA_TD3_std, color=td3_color, width=barwidth,
                   edgecolor='white', label='TD3')

    ASA_fig_ax.set_xticks(range(1, len(Dof_list) + 1), )
    ASA_fig_ax.set_xticklabels(Dof_list)

    ASA_fig_ax.set_title('FE vs DOF ' + name_tag )
    ASA_fig_ax.set_ylabel('FE')

    ASA_fig_ax.legend(loc=2)
    if name_tag == 'Ant'or "FCHeavy" in name_tag:
        ASA_fig_ax.set_xlabel('Type of agent',fontweight='bold')
    else:
        ASA_fig_ax.set_xlabel('DOF')
    '''if fixed_scale:
        if name_tag == 'VA':
            ASA_fig_ax.set_ylim([-0.25, 6])
        elif name_tag == 'Ant':
            ASA_fig_ax.set_ylim([1, 4.5])
        elif name_tag == 'HC':
            ASA_fig_ax.set_ylim([0, 5])'''
    if get_rid_div:
        if fixed_scale:
            plt.savefig(
                cwd + '/experiments_results/Synergy/dof_P_double_bars_plot/' + name_tag + '/FE_double_bars_fixed_scale_' + name_tag + '_no_div.png')
        else:
            plt.savefig(
                cwd + '/experiments_results/Synergy/dof_P_double_bars_plot/' + name_tag + '/FE_double_bars_' + name_tag + '_no_div.png')
    else:
        if fixed_scale:
            plt.savefig(
                cwd + '/experiments_results/Synergy/dof_P_double_bars_plot/' + name_tag + '/FE_double_bars_fixed_scale_' + name_tag + '.png')
        else:
            plt.savefig(
                cwd + '/experiments_results/Synergy/dof_P_double_bars_plot/' + name_tag + '/FE_double_bars_' + name_tag + '.png')

