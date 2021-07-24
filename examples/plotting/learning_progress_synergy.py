import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import argparse

from examples.plotting.commons import *

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

cwd=os.getcwd()
cwd_list=cwd.split('/')
while cwd_list[-1]!='synergy_analysis':
    cwd_list.pop()

cwd='/'.join(cwd_list)


parser = argparse.ArgumentParser()
parser.add_argument('--div_rate',type=float, default=0.4)
parser.add_argument('--no_div',action='store_true')
parser.add_argument('--only_synergy',action='store_true')
parser.add_argument('--no_fixed_scale',action='store_false')

args = parser.parse_args()
agentt_type=args.agentt_type
div_rate = args.div_rate

get_rid_div=args.no_div
only_synergy=args.only_synergy
fixed_scale=args.no_fixed_scale

available_list=[]

for folder in [
                # AntRun_folder,
                # AntSquaT_folder,
                # AntSquaTRedundant_folder,
                #VA_folder,
                #VA4dof_folder,
                # VA6dof_folder,
                # VA8dof_folder,
                # HC2dof_folder,
                # HC3dofb_folder,
                # HC3doff_folder,
                # HC4dof_folder,
                # HC5dof_folder,
                # HC5dofv2_folder,
                # HC5dofv3_folder,
                # HC5dofv4_folder,
                # HC5dofv5_folder,
                # HC5dofv6_folder,
                HC_folder,
                # HC5dof_folder,
                HC2dof_folder,
                # HC3doff_folder,
                # HC3dofb_folder,
                HC4dof_folder,
                # HC2dofv2_folder,
                # HC2dofv3_folder,
                # HC2dofv4_folder,
                # HC2dofv5_folder,
                # HCsquat2dof_folder,
                # HCsquat4dof_folder,
                # HCsquat6dof_folder,
                # HCsquat4dofEp1_folder,
                # HCsquat6dofEp1_folder,
                # HCsquat2dofEp25_folder,
                # HCsquat4dofEp25_folder,
                # HCsquat6dofEp25_folder,

                #FCsLT_folder,
                #FCsLG_folder,
                #FCsLGfblr_folder,

                # HeavyHC_folder,
                # FC_folder,
                #HCE1_folder,
                # VAez_folder,
                # VA4dofez_folder,
                # VA6dofez_folder,
                # VA8dofez_folder,

                # VAp5_folder,
                # VA4dofp5_folder,
                # VA6dofp5_folder,
                # VA8dofp5_folder,

                # HCez_folder,
                # HC4dofez_folder,
                # HC2dofez_folder,
                # HC3dofbez_folder,
                # HC3doffez_folder,
                # HC5dofez_folder,

                #VAE1_folder,
                #VA4dofE1_folder,
                #VA6dofE1_folder,
                #VA8dofE1_folder,

                ]:
    if os.path.exists(folder):
        available_list.append(folder)

print(available_list)



c_SA='C4'
c_P='C2'
c_PI='C1'


for choice in available_list:
    save_path = cwd + '/experiments_results/Synergy/learning_progress_graphs/' + choice.split('/')[-1]

    llist=folder_algo_dict[choice]
    if not os._exists(save_path):
        os.makedirs(save_path, exist_ok=True)
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
                current_SAC_surface_list=np.asarray(current_file['Surface Area'])
                current_SAC_P_list = np.asarray(current_file['P'])
                current_SAC_PI_list = np.asarray(current_file['PI'])
                current_SAC_E_list =np.asarray( current_file['E'])
            else:
                current_SAC_surface_list=np.vstack((current_SAC_surface_list,np.asarray(current_file['Surface Area'])))
                current_SAC_P_list=np.vstack((current_SAC_P_list,np.asarray(current_file['P'])))
                current_SAC_PI_list=np.vstack((current_SAC_PI_list,np.asarray(current_file['PI'])))
                current_SAC_E_list=np.vstack((current_SAC_E_list,np.asarray(current_file['E'])))
        else:
            counter_TD3 = counter_TD3 + 1
            if counter_TD3 == 1:
                current_TD3_surface_list = np.asarray(current_file['Surface Area'])
                current_TD3_P_list = np.asarray(current_file['P'])
                current_TD3_PI_list = np.asarray(current_file['PI'])
                current_TD3_E_list = np.asarray(current_file['E'])
            else:
                current_TD3_surface_list = np.vstack((current_TD3_surface_list, np.asarray(current_file['Surface Area'])))
                current_TD3_P_list = np.vstack((current_TD3_P_list, np.asarray(current_file['P'])))
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

    try:
        mean_P_TD3=np.flip(np.mean(current_TD3_P_list,axis=0),axis=0)
        mean_PI_TD3=np.flip(np.mean(current_TD3_PI_list,axis=0),axis=0)
        mean_surface_TD3=np.flip(np.mean(current_TD3_surface_list,axis=0),axis=0)
        std_P_TD3=np.flip(np.std(current_TD3_P_list,axis=0),axis=0)
        std_PI_TD3=np.flip(np.std(current_TD3_PI_list,axis=0),axis=0)
        std_surface_TD3=np.flip(np.std(current_TD3_surface_list,axis=0),axis=0)

        P_all = np.vstack((current_SAC_P_list, current_TD3_P_list))
        PI_all = np.vstack((current_SAC_PI_list, current_TD3_PI_list))
        SA_all = np.vstack((current_SAC_surface_list, current_TD3_surface_list))
    except:

        P_all=current_SAC_P_list
        PI_all=current_SAC_PI_list
        SA_all=current_SAC_surface_list

    try:
        mean_P_all=np.flip(np.mean(P_all,axis=0),axis=0)
        mean_PI_all=np.flip(np.mean(PI_all,axis=0),axis=0)
        mean_surface_all=np.flip(np.mean(SA_all,axis=0),axis=0)
        std_P_all=np.flip(np.std(P_all,axis=0),axis=0)
        std_PI_all=np.flip(np.std(PI_all,axis=0),axis=0)
        std_surface_all=np.flip(np.std(SA_all,axis=0),axis=0)
    except:
        mean_P_all = np.flip(P_all, axis=0)
        mean_PI_all = np.flip(PI_all, axis=0)
        mean_surface_all = np.flip(SA_all, axis=0)

    for agentt_type, algo in llist:
        if algo=='SAC':
            surface_area_w=mean_surface_SAC
            P_list=mean_P_SAC
            PI_list=mean_PI_SAC

            try:
                std_surface_area_w=std_surface_SAC
                std_P_list=std_P_SAC
                std_PI_list=std_PI_SAC
            except:
                pass

        elif algo=='TD3':
            surface_area_w=mean_surface_TD3
            P_list=mean_P_TD3
            PI_list=mean_PI_TD3

            std_surface_area_w=std_surface_TD3
            std_P_list=std_P_TD3
            std_PI_list=std_PI_TD3

        surface_plot_w_ax = host_subplot(111, axes_class=AA.Axes)
        plt.subplots_adjust(right=0.75)

        if not only_synergy:
            ax2_w = surface_plot_w_ax.twinx()
            ax3_w = surface_plot_w_ax.twinx()

            new_fixed_axis = ax2_w.get_grid_helper().new_fixed_axis
            ax2_w.axis["right"] = new_fixed_axis(loc="right",
                                                 axes=ax2_w,
                                                 offset=(0, 0))

            ax2_w.axis["right"].toggle(all=True)

            new_fixed_axis = ax3_w.get_grid_helper().new_fixed_axis
            ax3_w.axis["right"] = new_fixed_axis(loc="right",
                                                 axes=ax3_w,
                                                 offset=(side, 0))

            ax3_w.axis["right"].toggle(all=True)

            ax2_w.set_ylabel('Performance', color=c_P)
            ax3_w.set_ylabel('Performance-energy', color=c_PI)


            if fixed_scale:
                ax2_w.set_ylim(P_axis_range[agentt_type])
                ax3_w.set_ylim(PI_axis_range[agentt_type])

        surface_plot_w_ax.set_ylabel('Surface Area', color=c_SA)
        if 'VA' in agentt_type:
            surface_plot_w_ax.set_xticks(VA_xticks)
            surface_plot_w_ax.set_xticklabels(VA_timestep)
            surface_plot_w_ax.set_xlabel("timesteps")
        elif 'AntSquaT' in agentt_type:

            surface_plot_w_ax.set_xticks(AntSquaT_xticks)
            surface_plot_w_ax.set_xticklabels(AntSquaT_timestep)
            surface_plot_w_ax.set_xlabel(r"${0:s}$ timesteps".format(AntSquaT_xlabel))

        elif 'AntRun' in agentt_type:


            surface_plot_w_ax.set_xticks(AntRun_xticks)
            surface_plot_w_ax.set_xticklabels(AntRun_timestep)
            surface_plot_w_ax.set_xlabel(r"${0:s}$ timesteps".format(AntRun_xlabel))
        elif 'HCsquat' in agentt_type:
            surface_plot_w_ax.set_xlabel("Training checkpoints")
        else:
            surface_plot_w_ax.set_xlabel(r"${0:s}$ timesteps".format(common_xtime))

        surface_plot_w_ax.set_ylim(SA_axis_range[agentt_type])


        if get_rid_div:
            bad_ind_list = []
            for ind, p in enumerate(PI_list):
                if ind > 0 and ind < (len(PI_list) - 1):
                    if abs(p - PI_list[ind - 1]) / abs(p) > div_rate and abs(p - PI_list[ind + 1]) / abs(
                            p) > div_rate:
                        bad_ind_list.append(ind)

            if len(bad_ind_list) > 0:
                print('DIV')

                P_list = np.delete(P_list, bad_ind_list, 0)
                PI_list = np.delete(PI_list, bad_ind_list, 0)
                try:
                    std_P_list = np.delete(std_P_list, bad_ind_list, 0)
                    std_PI_list = np.delete(std_PI_list, bad_ind_list, 0)
                    std_surface_area_w = np.delete(std_surface_area_w, bad_ind_list, 0)

                except:
                    pass
                surface_area_w = np.delete(surface_area_w, bad_ind_list, 0)

        surface_plot_w_ax.plot(range(1, len(surface_area_w) + 1), surface_area_w, color=c_SA,
                                                       label='Surface area',linewidth=LW)
        try:
            surface_plot_w_ax.fill_between(range(1, len(surface_area_w) + 1),surface_area_w+std_surface_area_w,
                                       surface_area_w-std_surface_area_w, facecolor=c_SA, alpha=trans_rate)
        except:
            pass

        if not only_synergy:
            ax2_w.plot(range(1, len(surface_area_w) + 1), P_list, color=c_P,
                       label='Performance',linewidth=LW)
            try:
                ax2_w.fill_between(range(1, len(surface_area_w) + 1),P_list+std_P_list,
                               P_list-std_P_list, facecolor=c_P, alpha=trans_rate)
            except:
                pass

            ax3_w.plot(range(1, len(surface_area_w) + 1), PI_list, color=c_PI,
                       label='Performance-energy',linewidth=LW)
            try:
                ax3_w.fill_between(range(1, len(surface_area_w) + 1),PI_list+std_PI_list,
                               PI_list-std_PI_list, facecolor=c_PI, alpha=trans_rate)
            except:
                pass

        surface_plot_w_ax.legend(loc=2,prop={'size': LGS})

        title_name=algo+' + '
        title_name = title_name + agentt_type

        surface_plot_w_ax.set_title(title_name)

        plt.tight_layout()

        if not get_rid_div:
            if not only_synergy:
                plt.savefig(os.path.join(save_path, 'Learning_progress_' + agentt_type  + '_' +algo  + '.png'),format = 'png')
            else:
                plt.savefig(os.path.join(save_path, 'Synergy_progress_' + agentt_type  + '_' +algo  + '.png'),format = 'png')
        else:

            if not only_synergy:
                plt.savefig(os.path.join(save_path, 'Learning_progress_no_div_' + agentt_type + '_' + algo + '.png'),
                            format='png')
            else:
                plt.savefig(os.path.join(save_path, 'Synergy_progress_no_div_' + agentt_type + '_' + algo + '.png'),
                            format='png')
        plt.close('all')








