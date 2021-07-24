import numpy as np
from matplotlib import pyplot as plt
import os
import argparse
import pandas as pd

try:
    from examples.plotting.commons import *
except:
    from commons import *


import scipy.stats as ss

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
DPI=350
barwidth=0.4
trans_rate=0.2

path_to_folder=cwd+'/experiments_results/Synergy/all_csv/process_SA_final_summary'

parser = argparse.ArgumentParser()

#Ant VA HCsquatEalt RealArmAllv4 HCthreev3

parser.add_argument('agentt_type',type=str,choices=['Ant','Paper2_Arm2D','Paper2_Arm3D','AntSquaT','HCallv6','HCez','HCE1','HCAll','VA','VAez',
                                                    'VAp5','VAall','VAE1','HC5dofAll','HC5dofAllo',
                                                    'HC5dofo','HCdofAllv2','HCAllfinal1','HC2dofAllo',
                                                    'HCsquat','HCsquatep1','HCsquatep25','HCsquatez',
                                                    'HCsquatEalt','RealArmAll','HCthree'])

parser.add_argument('--plot_type',type=str,default='line',choices=['bar','line'])
parser.add_argument('--no_div',action='store_true')
parser.add_argument('--with_std',action='store_true')
parser.add_argument('--no_fixed_scale',action='store_false')
parser.add_argument('--double_bars',action='store_true')
parser.add_argument('--save_svg',action='store_true')

args = parser.parse_args()

name_tag=args.agentt_type
get_rid_div=args.no_div
fixed_scale=args.no_fixed_scale
plot_type=args.plot_type
double_bars=args.double_bars
save_svg=args.save_svg
with_std=args.with_std
if double_bars:
    plot_type='bar'

if name_tag=='Ant':
    Dof_list = ['Run 8', 'Squat 4', 'Squat 8']
    agent_list = ['AntRun', 'AntSquaT', 'AntSquaTRedundant']
elif name_tag=='Paper2_Arm2D':
    Dof_list = ['2', '4', '6', '8']
    agent_list=['VA',   'VA4dof',   'VA6dof',   'VA8dof']
elif name_tag == 'Paper2_Arm3D':
    Dof_list = ['3', '4', '5', '6', '7']
    agent_list=['RealArm3dof',   'RealArm4dof',   'RealArm5dof',   'RealArm6dof',   'RealArm7dof']

elif name_tag=='AntSquaT':
    Dof_list = [ 'Squat 4', 'Squat 8']
    agent_list = [ 'AntSquaT', 'AntSquaTRedundant']
elif name_tag == 'HCsquat':
    Dof_list = ['2', '4', '6']
    agent_list = ['HCsquat2dof', 'HCsquat4dof', 'HCsquat6dof']
elif name_tag == 'HCsquatep1':
    Dof_list = ['2', '4', '6']
    agent_list = ['HCsquat2dof', 'HCsquat4dof_Ep1', 'HCsquat6dof_Ep1']
elif name_tag == 'HCsquatez':
    Dof_list = ['2', '4', '6']
    agent_list = ['HCsquat2dof_Ez', 'HCsquat4dof_Ez', 'HCsquat6dof_Ez']
elif name_tag == 'HCsquatEalt':
    Dof_list = ['2', '4', '6']
    agent_list = ['HCsquat2dof_Ealt', 'HCsquat4dof_Ealt', 'HCsquat6dof_Ealt']
elif name_tag == 'HCsquatep25':
    Dof_list = ['2', '4', '6']
    agent_list = ['HCsquat2dof_Ep25', 'HCsquat4dof_Ep25', 'HCsquat6dof_Ep25']
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
elif name_tag=='RealArmAll':
    Dof_list = ['3', '4', '5', '6', '7']
    agent_list=['RealArm3dof',   'RealArm4dof',   'RealArm5dof',   'RealArm6dof',   'RealArm7dof']
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

ASA_SAC = []
ASA_SAC_std = []
ASA_TD3 = []
ASA_TD3_std = []

FP_SAC = []
FP_SAC_std = []
FP_TD3 = []
FP_TD3_std = []

FPI_SAC = []
FPI_SAC_std = []
FPI_TD3 = []
FPI_TD3_std = []

FE_SAC = []
FE_SAC_std = []
FE_TD3 = []
FE_TD3_std = []

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


        ASA_mean = csv_file['ASA mean']
        ASA_std = csv_file['ASA std']

        try:
            FP_mean = csv_file['FPP mean']
            FP_std = csv_file['FPP std']
        except:
            FP_mean = csv_file['FP mean']
            FP_std = csv_file['FP std']

        try:
            FPI_mean = csv_file['FPPI mean']
            FPI_std = csv_file['FPPI std']
        except:
            FPI_mean = csv_file['FPI mean']
            FPI_std = csv_file['FPI std']

        FE_mean = csv_file['FE mean']
        FE_std = csv_file['FE std']

    ASA_SAC.append(ASA_mean[0])
    ASA_SAC_std.append(ASA_std[0])

    ASA_TD3.append(ASA_mean[1])
    ASA_TD3_std.append(ASA_std[1])

    FP_SAC.append(FP_mean[0])
    FP_SAC_std.append(FP_std[0])

    FP_TD3.append(FP_mean[1])
    FP_TD3_std.append(FP_std[1])

    FPI_SAC.append(FPI_mean[0])
    FPI_SAC_std.append(FPI_std[0])

    FPI_TD3.append(FPI_mean[1])
    FPI_TD3_std.append(FPI_std[1])

    FE_SAC.append(FE_mean[0])
    FE_SAC_std.append(FE_std[0])

    FE_TD3.append(FE_mean[1])
    FE_TD3_std.append(FE_std[1])


ASA_SAC=np.asarray(ASA_SAC)
ASA_SAC_std=np.asarray(ASA_SAC_std)
ASA_TD3=np.asarray(ASA_TD3)
ASA_TD3_std=np.asarray(ASA_TD3_std)

FP_SAC=np.asarray(FP_SAC)
FP_SAC_std=np.asarray(FP_SAC_std)
FP_TD3=np.asarray(FP_TD3)
FP_TD3_std=np.asarray(FP_TD3_std)

FPI_SAC=np.asarray(FPI_SAC)
FPI_SAC_std=np.asarray(FPI_SAC_std)
FPI_TD3=np.asarray(FPI_TD3)
FPI_TD3_std=np.asarray(FPI_TD3_std)

FE_SAC=np.asarray(FE_SAC)
FE_SAC_std=np.asarray(FE_SAC_std)
FE_TD3=np.asarray(FE_TD3)
FE_TD3_std=np.asarray(FE_TD3_std)

sac_color = '#7f6d5f'
td3_color = '#557f2d'
if not args.double_bars:

#################################################################################################################
    ASA_fig, ASA_fig_ax = plt.subplots(1, 1)

    if plot_type=='line':

        ASA_fig_ax.errorbar(ASA_SAC, FP_SAC, color=sac_color, linewidth=LW, marker="x", markersize=20, markeredgewidth=5,
                        label='SAC', yerr=FP_SAC_std)

        for index,d in enumerate(Dof_list):
            ASA_fig_ax.text(ASA_SAC[index]*0.95,FP_SAC[index]*1.01,d+ ' DOF',va='center',  ha='left')
    else:
        ASA_fig_ax.bar(FP_SAC, ASA_SAC, yerr=ASA_SAC_std, color='g', width=barwidth, edgecolor='white')

    ASA_fig_ax.set_title('Performance vs ASA '+name_tag+' SAC')
    ASA_fig_ax.set_xlabel('ASA')
    ASA_fig_ax.set_ylabel('Performance')
    ASA_fig_ax.legend()
    if fixed_scale:
        if name_tag=='VA':
            ASA_fig_ax.set_xlim([-0.25, 6])
        elif name_tag=='Ant':
            ASA_fig_ax.set_xlim([1, 4.5])
        elif name_tag=='HC':
            ASA_fig_ax.set_xlim([0, 5])
        elif name_tag == 'HCsquatEalt':
            ASA_fig_ax.set_xlim([0, 5])
        else:
            ASA_fig_ax.set_xlim([0, 5])

    if not save_svg:
        # if not os.path.exists(cwd + '/experiments_results/Synergy/ASA_vs_P_' + plot_type + '_plot/' + name_tag):
        #     os.makedirs(cwd + '/experiments_results/Synergy/ASA_vs_P_' + plot_type + '_plot/' + name_tag)

        if not os.path.exists(cwd + '/experiments_results/Synergy/ASA_vs_P_inv_' + plot_type + '_plot/' + name_tag):
            os.makedirs(cwd + '/experiments_results/Synergy/ASA_vs_P_inv_' + plot_type + '_plot/' + name_tag)
    else:
        # if not os.path.exists(cwd + '/experiments_results/Synergy/ASA_vs_P_' + plot_type + '_SVGplot/' + name_tag):
        #     os.makedirs(cwd + '/experiments_results/Synergy/ASA_vs_P_' + plot_type + '_SVGplot/' + name_tag)

        if not os.path.exists(cwd + '/experiments_results/Synergy/ASA_vs_P_inv_' + plot_type + '_SVGplot/' + name_tag):
            os.makedirs(cwd + '/experiments_results/Synergy/ASA_vs_P_inv_' + plot_type + '_SVGplot/' + name_tag)

    if get_rid_div:
        if fixed_scale:
            if not save_svg:
                plt.savefig(cwd + '/experiments_results/Synergy/ASA_vs_P_inv_'+plot_type+'_plot/'+name_tag+'/ASA_'+plot_type+'_SAC_fixed_scale_'+name_tag+'_no_div.png')
            else:
                plt.savefig(cwd + '/experiments_results/Synergy/ASA_vs_P_inv_'+plot_type+'_SVGplot/'+name_tag+'/ASA_'+plot_type+'_SAC_fixed_scale_'+name_tag+'_no_div.eps', format='eps')

        else:
            plt.savefig(cwd + '/experiments_results/Synergy/ASA_vs_P_inv_'+plot_type+'_plot/'+name_tag+'/ASA_'+plot_type+'_SAC_'+name_tag+'_no_div.png')
    else:
        if fixed_scale:
            plt.savefig(
                cwd + '/experiments_results/Synergy/ASA_vs_P_inv_'+plot_type+'_plot/' + name_tag + '/ASA_'+plot_type+'_SAC_fixed_scale_' + name_tag + '.png')
        else:
            plt.savefig(
                cwd + '/experiments_results/Synergy/ASA_vs_P_inv_'+plot_type+'_plot/' + name_tag + '/ASA_'+plot_type+'_SAC_' + name_tag + '.png')

    ASA_fig, ASA_fig_ax = plt.subplots(1, 1)

    if plot_type=='line':

        ASA_fig_ax.errorbar(ASA_TD3, FP_TD3, color=td3_color, linewidth=LW, marker=".", markersize=20, markeredgewidth=5,
                        label='TD3', yerr = FP_TD3_std)


        for index,d in enumerate(Dof_list):
            ASA_fig_ax.text(ASA_TD3[index]*0.9,FP_TD3[index]*1.01,d+ ' DOF',va='center',  ha='left')

    else:
        ASA_fig_ax.bar(range(1, len(ASA_TD3) + 1), ASA_TD3, yerr=ASA_TD3_std, color='g', width=barwidth, edgecolor='white')



    ASA_fig_ax.set_title('ASA vs Performance '+name_tag+' TD3')
    ASA_fig_ax.set_ylabel('ASA')
    ASA_fig_ax.legend()

    ASA_fig_ax.set_xlabel('Performance')

    if fixed_scale:
        if name_tag=='VA':
            ASA_fig_ax.set_xlim([-0.25, 6])
        elif name_tag=='Ant':
            ASA_fig_ax.set_xlim([1, 4.5])
        elif name_tag=='HC':
            ASA_fig_ax.set_xlim([0, 5])
        elif name_tag=='HCsquatEalt':
            ASA_fig_ax.set_xlim([0, 5])
        else:
            ASA_fig_ax.set_xlim([0, 5])
    if get_rid_div:
        if fixed_scale:
            if not save_svg:
                plt.savefig(cwd + '/experiments_results/Synergy/ASA_vs_P_inv_'+plot_type+'_plot/'+name_tag+'/ASA_'+plot_type+'_TD3_fixed_scale_'+name_tag+'_no_div.png')
            else:
                plt.savefig(cwd + '/experiments_results/Synergy/ASA_vs_P_inv_'+plot_type+'_SVGplot/'+name_tag+'/ASA_'+plot_type+'_TD3_fixed_scale_'+name_tag+'_no_div.eps', format='eps')


        else:
            plt.savefig(cwd + '/experiments_results/Synergy/ASA_vs_P_'+plot_type+'_plot/'+name_tag+'/ASA_'+plot_type+'_TD3_'+name_tag+'_no_div.png')#svg
    else:
        if fixed_scale:

            plt.savefig(
                cwd + '/experiments_results/Synergy/ASA_vs_P_'+plot_type+'_plot/' + name_tag + '/ASA_'+plot_type+'_TD3_fixed_scale_' + name_tag + '.png')
        else:
            plt.savefig(
                cwd + '/experiments_results/Synergy/ASA_vs_P_'+plot_type+'_plot/' + name_tag + '/ASA_'+plot_type+'_TD3_' + name_tag + '.png')  # svg



    ASA_fig, ASA_fig_ax = plt.subplots(1, 1)

    if plot_type == 'line':

        ASA_fig_ax.errorbar(ASA_SAC, FP_SAC, color=sac_color, linewidth=LW, marker="x", markersize=20, markeredgewidth=5,
                        label='SAC', yerr = FP_SAC_std)


        for index, d in enumerate(Dof_list):
            ASA_fig_ax.text( ASA_SAC[index] * 0.95,FP_SAC[index] * 1.01, d + ' DOF', va='center', ha='left')

        ASA_fig_ax.errorbar(ASA_TD3, FP_TD3, color=td3_color, linewidth=LW, marker=".", markersize=20, markeredgewidth=5,
                        label='TD3', yerr =  FP_TD3_std)

        for index, d in enumerate(Dof_list):
            ASA_fig_ax.text( ASA_TD3[index] * 0.9,FP_TD3[index] * 1.01, d + ' DOF', va='center', ha='left')

    else:
        ASA_fig_ax.bar(range(1, len(ASA_TD3) + 1), ASA_TD3, yerr=ASA_TD3_std, color='g', width=barwidth,
                       edgecolor='white')

    ASA_fig_ax.set_title('Performance vs ASA ' + name_tag)
    ASA_fig_ax.set_xlabel('ASA')
    ASA_fig_ax.legend()
    ASA_fig_ax.set_ylabel('Performance')

    if fixed_scale:
        if name_tag == 'VA':
            ASA_fig_ax.set_xlim([-0.25, 6])
        elif name_tag == 'Ant':
            ASA_fig_ax.set_xlim([1, 4.5])
        elif name_tag == 'HC':
            ASA_fig_ax.set_xlim([0, 5])
        elif name_tag == 'HCsquatEalt':
            ASA_fig_ax.set_xlim([0, 5])
        else:
            ASA_fig_ax.set_xlim([0, 5])
    if get_rid_div:
        if fixed_scale:

            if not save_svg:
                plt.savefig(
                    cwd + '/experiments_results/Synergy/ASA_vs_P_inv_' + plot_type + '_plot/' + name_tag + '/ASA_' + plot_type + '_fixed_scale_' + name_tag + '_no_div.png')
            else:
                plt.savefig(
                    cwd + '/experiments_results/Synergy/ASA_vs_P_inv_' + plot_type + '_SVGplot/' + name_tag + '/ASA_' + plot_type + '_fixed_scale_' + name_tag + '_no_div.eps', format='eps')

        else:
            plt.savefig(
                cwd + '/experiments_results/Synergy/ASA_vs_P_' + plot_type + '_plot/' + name_tag + '/ASA_' + plot_type + '_' + name_tag + '_no_div.png')  # svg
    else:
        if fixed_scale:

            plt.savefig(
                cwd + '/experiments_results/Synergy/ASA_vs_P_' + plot_type + '_plot/' + name_tag + '/ASA_' + plot_type + '_fixed_scale_' + name_tag + '.png')
        else:
            plt.savefig(
                cwd + '/experiments_results/Synergy/ASA_vs_P_' + plot_type + '_plot/' + name_tag + '/ASA_' + plot_type + '_' + name_tag + '.png')  # svg

#################################################################################################################

#################################################################################################################
    ASA_fig, ASA_fig_ax = plt.subplots(1, 1)

    if plot_type=='line':

        ASA_fig_ax.errorbar( ASA_SAC,FPI_SAC, color=sac_color, linewidth=LW,marker="x", markersize=20,markeredgewidth=5,
                         label='SAC', yerr = FPI_SAC_std)


        for index,d in enumerate(Dof_list):
            ASA_fig_ax.text(ASA_SAC[index]*0.95,FPI_SAC[index]*1.01,d+ ' DOF',va='center',  ha='left')

    ASA_fig_ax.set_title('Performance-Energy vs ASA  '+name_tag+' SAC')
    ASA_fig_ax.set_xlabel('ASA')
    ASA_fig_ax.legend()
    ASA_fig_ax.set_ylabel('Performance-Energy')

    if fixed_scale:
        if name_tag=='VA':
            ASA_fig_ax.set_xlim([-0.25, 6])
        elif name_tag=='Ant':
            ASA_fig_ax.set_xlim([1, 4.5])
        elif name_tag=='HC':
            ASA_fig_ax.set_xlim([0, 5])
        elif name_tag == 'HCsquatEalt':
            ASA_fig_ax.set_xlim([0, 5])
        else:
            ASA_fig_ax.set_xlim([0, 5])


    if get_rid_div:
        if fixed_scale:
            if not save_svg:
                plt.savefig(cwd + '/experiments_results/Synergy/ASA_vs_P_inv_'+plot_type+'_plot/'+name_tag+'/ASA_PI_'+plot_type+'_SAC_fixed_scale_'+name_tag+'_no_div.png')
            else:
                plt.savefig(cwd + '/experiments_results/Synergy/ASA_vs_P_inv_'+plot_type+'_SVGplot/'+name_tag+'/ASA_PI_'+plot_type+'_SAC_fixed_scale_'+name_tag+'_no_div.eps', format='eps')


        else:
            plt.savefig(cwd + '/experiments_results/Synergy/ASA_vs_P_'+plot_type+'_plot/'+name_tag+'/ASA_PI_'+plot_type+'_SAC_'+name_tag+'_no_div.png')
    else:
        if fixed_scale:
            plt.savefig(
                cwd + '/experiments_results/Synergy/ASA_vs_P_'+plot_type+'_plot/' + name_tag + '/ASA_PI_'+plot_type+'_SAC_fixed_scale_' + name_tag + '.png')
        else:
            plt.savefig(
                cwd + '/experiments_results/Synergy/ASA_vs_P_'+plot_type+'_plot/' + name_tag + '/ASA_PI_'+plot_type+'_SAC_' + name_tag + '.png')

    ASA_fig, ASA_fig_ax = plt.subplots(1, 1)

    if plot_type=='line':

        ASA_fig_ax.errorbar( ASA_TD3, FPI_TD3,color=td3_color, linewidth=LW,marker=".", markersize=20,
                         markeredgewidth=5,label='TD3', yerr = FPI_TD3_std)


        for index,d in enumerate(Dof_list):
            ASA_fig_ax.text(ASA_TD3[index]*0.9,FPI_TD3[index]*1.01,d+ ' DOF',va='center',  ha='left')



    ASA_fig_ax.set_title('Performance-Energy vs ASA '+name_tag+' TD3')
    ASA_fig_ax.set_xlabel('ASA')
    ASA_fig_ax.legend()
    ASA_fig_ax.set_ylabel('Performance-Energy')

    if fixed_scale:
        if name_tag=='VA':
            ASA_fig_ax.set_xlim([-0.25, 6])
        elif name_tag=='Ant':
            ASA_fig_ax.set_xlim([1, 4.5])
        elif name_tag=='HC':
            ASA_fig_ax.set_xlim([0, 5])
        elif name_tag=='HCsquatEalt':
            ASA_fig_ax.set_xlim([0, 5])
        else:
            ASA_fig_ax.set_xlim([0, 5])
    if get_rid_div:
        if fixed_scale:

            if not save_svg:
                plt.savefig(cwd + '/experiments_results/Synergy/ASA_vs_P_inv_'+plot_type+'_plot/'+name_tag+'/ASA_PI_'+plot_type+'_TD3_fixed_scale_'+name_tag+'_no_div.png')
            else:
                plt.savefig(cwd + '/experiments_results/Synergy/ASA_vs_P_inv_'+plot_type+'_SVGplot/'+name_tag+'/ASA_PI_'+plot_type+'_TD3_fixed_scale_'+name_tag+'_no_div.eps', format='eps')

        else:
            plt.savefig(cwd + '/experiments_results/Synergy/ASA_vs_P_'+plot_type+'_plot/'+name_tag+'/ASA_PI_'+plot_type+'_TD3_'+name_tag+'_no_div.png')#svg
    else:
        if fixed_scale:

            plt.savefig(
                cwd + '/experiments_results/Synergy/ASA_vs_P_'+plot_type+'_plot/' + name_tag + '/ASA_PI_'+plot_type+'_TD3_fixed_scale_' + name_tag + '.png')
        else:
            plt.savefig(
                cwd + '/experiments_results/Synergy/ASA_vs_P_'+plot_type+'_plot/' + name_tag + '/ASA_PI_'+plot_type+'_TD3_' + name_tag + '.png')  # svg



    ASA_fig, ASA_fig_ax = plt.subplots(1, 1)

    if plot_type == 'line':

        ASA_fig_ax.errorbar( ASA_SAC,FPI_SAC, color=sac_color, linewidth=LW, marker="x", markersize=20,
                         markeredgewidth=5,label='SAC', yerr = FPI_SAC_std)


        for index, d in enumerate(Dof_list):
            ASA_fig_ax.text( ASA_SAC[index] * 0.95,FPI_SAC[index] * 1.01, d + ' DOF', va='center', ha='left')

        ASA_fig_ax.errorbar(ASA_TD3, FPI_TD3, color=td3_color, linewidth=LW, marker=".", markersize=20, markeredgewidth=5,
                        label='TD3', yerr = FPI_TD3_std)

        for index, d in enumerate(Dof_list):
            ASA_fig_ax.text( ASA_TD3[index] * 0.9,FPI_TD3[index] * 1.01, d + ' DOF', va='center', ha='left')



    ASA_fig_ax.set_title('Performance-Energy vs ASA ' + name_tag)
    ASA_fig_ax.set_xlabel('ASA')
    ASA_fig_ax.legend()
    ASA_fig_ax.set_ylabel('Performance-Energy')

    if fixed_scale:
        if name_tag == 'VA':
            ASA_fig_ax.set_xlim([-0.25, 6])
        elif name_tag == 'Ant':
            ASA_fig_ax.set_xlim([1, 4.5])
        elif name_tag == 'HC':
            ASA_fig_ax.set_xlim([0, 5])
        elif name_tag == 'HCsquatEalt':
            ASA_fig_ax.set_xlim([0, 5])
        else:
            ASA_fig_ax.set_xlim([0, 5])
    if get_rid_div:
        if fixed_scale:
            if not save_svg:
                plt.savefig(
                    cwd + '/experiments_results/Synergy/ASA_vs_P_inv_' + plot_type + '_plot/' + name_tag + '/ASA_PI_' + plot_type + '_fixed_scale_' + name_tag + '_no_div.png')
            else:
                plt.savefig(
                    cwd + '/experiments_results/Synergy/ASA_vs_P_inv_' + plot_type + '_SVGplot/' + name_tag + '/ASA_PI_' + plot_type + '_fixed_scale_' + name_tag + '_no_div.eps', format='eps')

        else:
            plt.savefig(
                cwd + '/experiments_results/Synergy/ASA_vs_P_' + plot_type + '_plot/' + name_tag + '/ASA_PI_' + plot_type + '_' + name_tag + '_no_div.png')  # svg
    else:
        if fixed_scale:

            plt.savefig(
                cwd + '/experiments_results/Synergy/ASA_vs_P_' + plot_type + '_plot/' + name_tag + '/ASA_PI_' + plot_type + '_fixed_scale_' + name_tag + '.png')
        else:
            plt.savefig(
                cwd + '/experiments_results/Synergy/ASA_vs_P_' + plot_type + '_plot/' + name_tag + '/ASA_PI_' + plot_type + '_' + name_tag + '.png')  # svg

#################################################################################################################



#################################################################################################################
    ASA_fig, ASA_fig_ax = plt.subplots(1, 1)

    if plot_type=='line':

        ASA_fig_ax.errorbar(FE_SAC, ASA_SAC, color=sac_color, linewidth=LW,marker="x", markersize=20,
                        markeredgewidth=5,label='SAC', yerr =  ASA_SAC_std)



        for index,d in enumerate(Dof_list):
            ASA_fig_ax.text(FE_SAC[index]*1.01,ASA_SAC[index]*0.95,d+ ' DOF',va='center',  ha='left')


    ASA_fig_ax.set_title('ASA vs Energy '+name_tag+' SAC')
    ASA_fig_ax.set_ylabel('ASA')
    ASA_fig_ax.legend()
    ASA_fig_ax.set_xlabel('Energy')

    if fixed_scale:
        if name_tag=='VA':
            ASA_fig_ax.set_ylim([-0.25, 6])
        elif name_tag=='Ant':
            ASA_fig_ax.set_ylim([1, 4.5])
        elif name_tag=='HC':
            ASA_fig_ax.set_ylim([0, 5])
        elif name_tag == 'HCsquatEalt':
            ASA_fig_ax.set_ylim([0, 5])
        else:
            ASA_fig_ax.set_ylim([0, 5])


    if get_rid_div:
        if fixed_scale:
            if not save_svg:
                plt.savefig(cwd + '/experiments_results/Synergy/ASA_vs_P_inv_'+plot_type+'_plot/'+name_tag+'/ASA_E_'+plot_type+'_SAC_fixed_scale_'+name_tag+'_no_div.png')
            else:

                plt.savefig(cwd + '/experiments_results/Synergy/ASA_vs_P_inv_'+plot_type+'_SVGplot/'+name_tag+'/ASA_E_'+plot_type+'_SAC_fixed_scale_'+name_tag+'_no_div.eps', format='eps')

        else:
            plt.savefig(cwd + '/experiments_results/Synergy/ASA_vs_P_'+plot_type+'_plot/'+name_tag+'/ASA_E_'+plot_type+'_SAC_'+name_tag+'_no_div.png')
    else:
        if fixed_scale:
            plt.savefig(
                cwd + '/experiments_results/Synergy/ASA_vs_P_'+plot_type+'_plot/' + name_tag + '/ASA_E_'+plot_type+'_SAC_fixed_scale_' + name_tag + '.png')
        else:
            plt.savefig(
                cwd + '/experiments_results/Synergy/ASA_vs_P_'+plot_type+'_plot/' + name_tag + '/ASA_E_'+plot_type+'_SAC_' + name_tag + '.png')

    ASA_fig, ASA_fig_ax = plt.subplots(1, 1)

    if plot_type=='line':

        ASA_fig_ax.errorbar(FE_TD3, ASA_TD3, color=td3_color, linewidth=LW,marker=".", markersize=20,
                        markeredgewidth=5,label='TD3' , yerr = ASA_TD3_std)


        for index,d in enumerate(Dof_list):
            ASA_fig_ax.text(FE_TD3[index]*1.01,ASA_TD3[index]*0.9,d+ ' DOF',va='center',  ha='left')



    ASA_fig_ax.set_title('ASA vs Energy '+name_tag+' TD3')
    ASA_fig_ax.set_ylabel('ASA')
    ASA_fig_ax.legend()
    ASA_fig_ax.set_xlabel('Energy')

    if fixed_scale:
        if name_tag=='VA':
            ASA_fig_ax.set_ylim([-0.25, 6])
        elif name_tag=='Ant':
            ASA_fig_ax.set_ylim([1, 4.5])
        elif name_tag=='HC':
            ASA_fig_ax.set_ylim([0, 5])
        elif name_tag=='HCsquatEalt':
            ASA_fig_ax.set_ylim([0, 5])
        else:
            ASA_fig_ax.set_ylim([0, 5])
    if get_rid_div:
        if fixed_scale:
            if not save_svg:
                plt.savefig(cwd + '/experiments_results/Synergy/ASA_vs_P_inv_'+plot_type+'_plot/'+name_tag+'/ASA_E_'+plot_type+'_TD3_fixed_scale_'+name_tag+'_no_div.png')
            else:
                plt.savefig(cwd + '/experiments_results/Synergy/ASA_vs_P_inv_'+plot_type+'_SVGplot/'+name_tag+'/ASA_E_'+plot_type+'_TD3_fixed_scale_'+name_tag+'_no_div.eps', format='eps')


        else:
            plt.savefig(cwd + '/experiments_results/Synergy/ASA_vs_P_'+plot_type+'_plot/'+name_tag+'/ASA_E_'+plot_type+'_TD3_'+name_tag+'_no_div.png')#svg
    else:
        if fixed_scale:

            plt.savefig(
                cwd + '/experiments_results/Synergy/ASA_vs_P_'+plot_type+'_plot/' + name_tag + '/ASA_E_'+plot_type+'_TD3_fixed_scale_' + name_tag + '.png')
        else:
            plt.savefig(
                cwd + '/experiments_results/Synergy/ASA_vs_P_'+plot_type+'_plot/' + name_tag + '/ASA_E_'+plot_type+'_TD3_' + name_tag + '.png')  # svg



    ASA_fig, ASA_fig_ax = plt.subplots(1, 1)

    if plot_type == 'line':
        ASA_fig_ax.errorbar(FE_SAC, ASA_SAC, color=sac_color, linewidth=LW, marker="x", markersize=20,
                        markeredgewidth=5,label='SAC' , yerr =  ASA_SAC_std)


        for index, d in enumerate(Dof_list):
            ASA_fig_ax.text(FE_SAC[index] * 1.01, ASA_SAC[index] * 0.95, d + ' DOF', va='center', ha='left')


        ASA_fig_ax.errorbar(FE_TD3, ASA_TD3, color=td3_color, linewidth=LW, marker=".", markersize=20,
                        markeredgewidth=5,
                        label='TD3', yerr =  ASA_TD3_std)


        for index, d in enumerate(Dof_list):
            ASA_fig_ax.text(FE_TD3[index] * 1.01, ASA_TD3[index] * 0.9, d + ' DOF', va='center', ha='left')


    ASA_fig_ax.set_title('ASA vs Energy ' + name_tag)
    ASA_fig_ax.set_ylabel('ASA')
    ASA_fig_ax.legend()
    ASA_fig_ax.set_xlabel('Energy')

    if fixed_scale:
        if name_tag == 'VA':
            ASA_fig_ax.set_ylim([-0.25, 6])
        elif name_tag == 'Ant':
            ASA_fig_ax.set_ylim([1, 4.5])
        elif name_tag == 'HC':
            ASA_fig_ax.set_ylim([0, 5])
        elif name_tag == 'HCsquatEalt':
            ASA_fig_ax.set_ylim([0, 5])
        else:
            ASA_fig_ax.set_ylim([0, 5])
    if get_rid_div:
        if fixed_scale:
            if not save_svg:
                plt.savefig(
                    cwd + '/experiments_results/Synergy/ASA_vs_P_inv_' + plot_type + '_plot/' + name_tag + '/ASA_E_' + plot_type + '_fixed_scale_' + name_tag + '_no_div.png')
            else:

                plt.savefig(
                    cwd + '/experiments_results/Synergy/ASA_vs_P_inv_' + plot_type + '_SVGplot/' + name_tag + '/ASA_E_' + plot_type + '_fixed_scale_' + name_tag + '_no_div.eps', format='eps')

        else:
            plt.savefig(
                cwd + '/experiments_results/Synergy/ASA_vs_P_' + plot_type + '_plot/' + name_tag + '/ASA_E_' + plot_type + '_' + name_tag + '_no_div.png')  # svg
    else:
        if fixed_scale:

            plt.savefig(
                cwd + '/experiments_results/Synergy/ASA_vs_P_' + plot_type + '_plot/' + name_tag + '/ASA_E_' + plot_type + '_fixed_scale_' + name_tag + '.png')
        else:
            plt.savefig(
                cwd + '/experiments_results/Synergy/ASA_vs_P_' + plot_type + '_plot/' + name_tag + '/ASA_E_' + plot_type + '_' + name_tag + '.png')  # svg

#################################################################################################################