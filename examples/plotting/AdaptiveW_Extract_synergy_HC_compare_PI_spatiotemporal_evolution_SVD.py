import os
import argparse
import numpy as np
from matplotlib.lines import Line2D
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from examples.plotting.exp_variant_class import exp_variant
from sklearn.decomposition import TruncatedSVD
from examples.plotting.commons import *

cmap = plt.cm.viridis
cmaplist = [cmap(i) for i in range(cmap.N)]
cmaplen = len(cmaplist)

plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['font.size'] = 25
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['font.family'] = 'Times New Roman'
LW = 3

parser = argparse.ArgumentParser()

parser.add_argument('--tr', nargs='+', required=True)

parser.add_argument('--ee',type=str, nargs='+',choices=spt_energy_list)

parser.add_argument('--agentt',
                    type=str,choices=spt_agent_list)
#parser.add_argument('--high_res',
#                    type=bool, default=False)
parser.add_argument('--high_res',action="store_true")
args = parser.parse_args()
high_res = args.high_res
agentt = args.agentt

svg = True
save = True
sortt = False
avg_plot = True
precheck = False
truncated = True
plot_r_sq = True
manual_rsq = True
standscale = True
get_rid_div=True
named_label = False
energy_penalty = False
plot_norm_single = True

min_rsq = 0
num_epi = 10
div_rate=0.4
recon_num = 0
LW_action = 4
ori_num_vec = 3
desired_length = 28#28  # 50#28#7
truncated_start = 200
ori_total_vec_rsq = 9
per_episode_max_length=1000

C_y_lim=[-5, 5]
X_fig_size=(5, 10)
W_y_lim=[-0.6, 0.6]
recon_y_lim=[-1.2, 1.2]

type_ = 'P'
action_path=None
state_path=None
reward_path=None
version = '3_components_truncated'

top_folder=agentt+'_spatiotemporal_evolution'

total_vec=agent_info_dict[agentt]['total_vec']
total_chk=agent_info_dict[agentt]['total_chk']
ori_final=agent_info_dict[agentt]['ori_final']
ori_begin=agent_info_dict[agentt]['ori_begin']
ori_step=agent_info_dict[agentt]['ori_step']
x_speed_index=agent_info_dict[agentt]['x_speed_index']
x_speed_index=None
desired_dist=agent_info_dict[agentt]['desired_dist']

try:
    dll=agent_info_dict[agentt]['dll']
    desired_length=dll
    truncated_start = agent_info_dict[agentt]['truncated_start']
except:
    pass
try:
    W_y_lim = agent_info_dict[agentt]['W_y_lim']
except:
    pass
try:
    recon_y_lim = agent_info_dict[agentt]['recon_y_lim']
except:
    pass

try:
    C_y_lim = agent_info_dict[agentt]['C_y_lim']
except:
    pass


if energy_penalty:
    top_folder=top_folder+'_EP'

'''if args.agentt=='A' and 'E0' in args.ee:
    args.tr=['']+args.tr'''

for tr in args.tr:
    for ee in args.ee:
        final = ori_final
        begin = ori_begin
        step = ori_step
        trial = tr
        subfolder=ee

        if ee not in agentt:
            base = agentt + '_' + ee
        else:
            base = agentt

        all_names=[]
        tmp=[]
        for cc in range(begin, final + step, step):
            tmp.append(base + '_C' + str(cc) + trial)
        '''if final == 3000 and 'HC' in agentt and 'dof' not in agentt:
            for cc in range(begin, final + step, step):
                if cc != final:
                    tmp.append(base + '_C' + str(cc) + trial)
                else:
                    tmp.append(base + trial)
        else:
            for cc in range(begin, final + step, step):
                tmp.append(base + '_C' + str(cc) + trial)'''

        all_names.append(tmp)
        #all_names.reverse()
        print(all_names)
        if precheck:
            top_tmp=[]
            for all_name in all_names:
                tmpp = []
                for n in all_name:
                    if action_path != None:
                        complete = exp_variant(n, action_path=action_path, reward_path=reward_path,
                                               state_path=state_path).check_complete_data()
                    else:
                        complete=exp_variant(n).check_complete_data()
                    if complete:
                        tmpp.append(n)
                top_tmp.append(tmpp)

            all_names=top_tmp



        rsq_all_list = []
        r_sq_all_TOP, r_sq_all_TOP_ax = plt.subplots(1, 1)
        for all_name in all_names:
            agent = all_name[0] + '_spatiotemporal_evolution'

            folder_name=agent+'_Compare_'+type_+'_'+str(desired_length)
            if manual_rsq:
                folder_name=folder_name+'_manual_rsq'
            if standscale:
                folder_name = folder_name + '_SS'

            step=cmaplen//len(all_name)
            color_list = []
            c = cmaplen - 1
            for l in range(len(all_name)):
                color_list.append(cmaplist[c])
                c -= step
            all_label=[]
            exp_variant_list=[]
            for ind,n in enumerate(all_name):
                if action_path != None:
                    exp_variant_list.append(
                        exp_variant(n, action_path=action_path, reward_path=reward_path, state_path=state_path))

                else:
                    exp_variant_list.append(exp_variant(n))

                all_label.append(exp_variant_list[ind].eval(type_))
            ############################################################################################################
            if sortt:
                new_index=sorted(range(len(all_label)), key=lambda k: all_label[k],reverse=True)
                all_label=sorted(all_label,reverse=True)

                tmp=[]
                tmp_2=[]
                for ni in new_index:
                    tmp.append(exp_variant_list[ni])
                    tmp_2.append(all_name[ni])
                exp_variant_list=tmp
                all_name=tmp_2

            if named_label:
                all_label=[]
                for ind,n in enumerate(all_name):
                    prefix_list=exp_variant_list[ind].name.split('_')
                    for pp in prefix_list:
                        if 'C' in pp and 'H' not in pp:
                            prefix=pp
                            break
                    all_label.append(prefix + ': ' + '{:.2f}'.format(exp_variant_list[ind].eval(type_),2))
            else:
                all_label = []
                for ind, n in enumerate(all_name):
                    all_label.append('{:.2f}'.format(exp_variant_list[ind].eval(type_), 2))
            ############################################################################################################

            r_sq_all_combare, r_sq_all_combare_ax = plt.subplots(1, 1)
            if energy_penalty:
                energy_all_list = []
                best_perf_energy_ind_tuple=(0,0,0)

                for n_ind, name in enumerate(all_name):
                    exp_variant_obj = exp_variant_list[n_ind]
                    current_energy=exp_variant_obj.eval('E')
                    current_P = exp_variant_obj.eval('P')
                    energy_all_list.append(current_energy)

                    if current_P>best_perf_energy_ind_tuple[0]:
                        best_perf_energy_ind_tuple=(current_P,current_energy,n_ind)

                energy_all_list = np.asarray(energy_all_list)
                E_max = np.max(energy_all_list)
                energy_all_list = energy_all_list / E_max
                best_perf_energy_ind_tuple = (best_perf_energy_ind_tuple[0], energy_all_list[best_perf_energy_ind_tuple[2]], best_perf_energy_ind_tuple[2])
                energy_all_list=gauss(energy_all_list,best_perf_energy_ind_tuple[1])

            P_list=[]
            for n_ind,name in enumerate(all_name):
                exp_variant_obj=exp_variant_list[n_ind]
                P_list.append(exp_variant_obj.eval('P'))

                num_vec_to_keep=ori_num_vec
                X=np.load(exp_variant_obj.action_npy, allow_pickle=True)
                state_ = np.load(exp_variant_obj.state_npy, allow_pickle=True)

                mini = per_episode_max_length
                if X.shape == (num_epi,):
                    for i in range(num_epi):
                        amin = np.asarray(X[i]).shape[0]
                        if amin < mini:
                            mini = amin
                    print(mini)

                    #tmp = np.expand_dims(np.asarray(X[0])[0:mini, :], 0)
                    tmp = np.expand_dims(np.asarray(X[0])[-mini::, :], 0)

                    for i in range(num_epi-1):
                        #tmp = np.vstack((tmp, np.expand_dims(np.asarray(X[i + 1])[0:mini, :], 0)))
                        tmp = np.vstack((tmp, np.expand_dims(np.asarray(X[i + 1])[-mini::, :], 0)))
                    print(tmp.shape)
                    X = tmp

                    #tmp2 = np.expand_dims(np.asarray(state_[0])[0:mini, :], 0)
                    tmp2 = np.expand_dims(np.asarray(state_[0])[-mini::, :], 0)

                    for i in range(num_epi-1):
                        #tmp2 = np.vstack((tmp2, np.expand_dims(np.asarray(state_[i + 1])[0:mini, :], 0)))
                        tmp2 = np.vstack((tmp2, np.expand_dims(np.asarray(state_[i + 1])[-mini::, :], 0)))

                    state_ = tmp2

                X=X[0:num_epi,:,:]
                state_ = state_[0:num_epi, :, :]
                distance = []
                if x_speed_index:

                    speed_record = state_[0, :, x_speed_index]
                    for i in range(len(speed_record)):
                        if i == 0:
                            distance.append(speed_record[0])
                        else:
                            distance.append(np.sum(speed_record[0:i]))

                    distance = np.asarray(distance)

                if truncated:
                    total_vec_rsq = ori_total_vec_rsq

                    if x_speed_index:
                        if mini == per_episode_max_length or mini>300:
                            current_dist = distance[truncated_start]
                            end_dist_index = truncated_start
                            tmp_dist = 0

                            while tmp_dist < desired_dist and end_dist_index < len(distance) - 1:
                                end_dist_index += 1
                                tmp_dist = distance[end_dist_index] - current_dist

                            remaining_index = end_dist_index - truncated_start
                            desired_length=remaining_index
                        elif mini - desired_length >= 0:
                            remaining_index=desired_length
                            desired_length = remaining_index


                    if mini == per_episode_max_length:

                        X_truncated = X[:, truncated_start:truncated_start + desired_length, :]
                    else:
                        if mini >= (truncated_start + desired_length):
                            X_truncated = X[:, truncated_start:truncated_start + desired_length, :]

                        elif mini - desired_length >= 0:

                            X_truncated = X[:, mini - desired_length:mini, :]

                        else:
                            X_truncated = X[:, 0:mini, :]
                            if mini>=ori_total_vec_rsq:
                                total_vec_rsq=ori_total_vec_rsq
                            else:
                                total_vec_rsq=mini
                    X = X_truncated

                ori_shape = X.shape
                X_unnorm = np.copy(X)

                if avg_plot:
                    max_list = []
                    max_ind = np.argmax(X[0, :, 0])
                    max_list.append(max_ind)
                    X_temp = np.concatenate(
                        (np.expand_dims(X[0, max_ind::, :], 0), np.expand_dims(X[0, 0:max_ind, :], 0)), axis=1)
                    for l in range(1, X.shape[0], 1):
                        max_ind = np.argmax(X[l, :, 0])
                        max_list.append(max_ind)
                        X_temp = np.concatenate((X_temp, np.concatenate(
                            (np.expand_dims(X[l, max_ind::, :], 0), np.expand_dims(X[l, 0:max_ind, :], 0)),
                            axis=1)), axis=0)
                    X = X_temp


                if standscale:
                    mx = np.mean(X, axis=1)

                    for k in range(X.shape[1]):
                        X[:, k, :] = X[:, k, :] - mx

                X_mean = np.mean(X, axis=0)
                X_std = np.std(X, axis=0)

                a_list = []
                a_list_std = []
                if avg_plot:
                    for iterrr in range(total_vec):
                        a_list.append(X_mean[:, iterrr])
                        a_list_std.append(X_std[:, iterrr])
                else:
                    if plot_norm_single:
                        for iterrr in range(total_vec):
                            a_list.append(X[recon_num, :, iterrr])

                    else:
                        for iterrr in range(total_vec):
                            a_list.append(X_unnorm[recon_num, :, iterrr])

                gg, ax = plt.subplots(total_vec, 1, figsize=X_fig_size)
                for ii in range(len(a_list)):
                    if len(a_list[ii]) >= desired_length:
                        ax[ii].plot(range(desired_length), a_list[ii][0:desired_length], linewidth=LW_action)

                        if avg_plot:
                            ax[ii].fill_between(range(desired_length),
                                                a_list[ii] + a_list_std[ii],
                                                a_list[ii] - a_list_std[ii],
                                                alpha=0.2)
                    else:
                        ax[ii].plot(range(mini), a_list[ii][0:mini])

                    ax[ii].get_xaxis().set_visible(False)
                    if ii == len(a_list) - 1:
                        ax[ii].get_xaxis().set_visible(True)
                        ax[ii].xaxis.set_major_locator(MaxNLocator(integer=True))

                X = reshape_into_spt_shape(X)

                s_by_a=X
                a_by_s=X.T


                pca = TruncatedSVD(n_components=num_vec_to_keep)
                pca.fit(X)
                eig_vecs=pca.components_
                eig_vals=pca.singular_values_
                eig_pairs = [(eig_vals[i], eig_vecs[i, :]) for i in range(len(eig_vals))]
                percentage=sum(pca.explained_variance_ratio_)


                print(num_vec_to_keep)
                print(percentage)

                num_features = X.shape[1]
                proj_mat = eig_pairs[0][1].reshape(num_features,1)
                for eig_vec_idx in range(1, num_vec_to_keep):
                  proj_mat = np.hstack((proj_mat, eig_pairs[eig_vec_idx][1].reshape(num_features,1)))

                W=proj_mat
                C = X.dot(W)
                X_prime=C.dot(W.T)

                X_prime_norm = np.copy(X_prime)
                X_prime_norm = reshape_into_ori_shape(X_prime_norm,total_vec)

                X_prime = reshape_into_ori_shape(X_prime,total_vec)

                #C = C.reshape([num_vec_to_keep,-1])
                if avg_plot:
                    Xp_mean = np.mean(X_prime, axis=0)
                    Xp_std = np.std(X_prime, axis=0)

                a_list=[]
                a_list_std = []
                if avg_plot:
                    for iterrr in range(total_vec):
                        a_list.append(Xp_mean[:, iterrr])
                        a_list_std.append(Xp_std[:, iterrr])
                else:
                    if plot_norm_single:
                        for iterrr in range(total_vec):
                            a_list.append(X_prime_norm[recon_num, :, iterrr])
                    else:
                        for iterrr in range(total_vec):
                            a_list.append(X_prime[recon_num, :, iterrr])

                for ii in range(len(a_list)):

                    if len(a_list[ii])>=desired_length:
                        ax[ii].plot(range(desired_length), a_list[ii][0:desired_length], color='r')
                    else:
                        ax[ii].plot(range(mini), a_list[ii][0:mini], color='r')

                    if avg_plot:
                        if len(a_list[ii]) >= desired_length:
                            ax[ii].fill_between(range(desired_length),
                                                a_list[ii] + a_list_std[ii],
                                                a_list[ii] - a_list_std[ii],
                                                alpha=0.1, color='r')
                        else:
                            ax[ii].fill_between(range(mini),
                                                a_list[ii] + a_list_std[ii],
                                                a_list[ii] - a_list_std[ii],
                                                alpha=0.1, color='r')

                    ax[ii].set_ylabel('Joint ' + str(ii + 1))
                    if ii == len(a_list) - 1:
                        ax[ii].set_xlabel('Time steps')
                    ax[ii].set_ylim(recon_y_lim)

                if save==False:
                    plt.show()
                else:
                    path=file_path+'/experiments_results/Synergy/synergy_development_'+agentt+'/'+top_folder+'/'+subfolder+'/'+folder_name+'/Reconstructions'

                    os.makedirs(path, exist_ok=True)
                    gg.tight_layout()

                    if svg:
                        gg.savefig(os.path.join(path, 'Reconstructions' + exp_variant_obj.name + '.pdf'),format='pdf')
                    else:
                        if not high_res:
                            gg.savefig(os.path.join(path, 'Reconstructions' + exp_variant_obj.name + '.png'))#eps
                        else:
                            #gg.savefig(os.path.join(path, 'Reconstructions' + exp_variant_obj.name + '.png'),
                            #           dpi=300)
                            gg.savefig(os.path.join(path, 'Reconstructions' + exp_variant_obj.name + '.pdf'),format='pdf')
                gg, ax = plt.subplots(num_vec_to_keep, 1)


                c_list = []
                for iterr in range(num_vec_to_keep):
                    c_list.append(C[0:X.shape[0], iterr])
                c_list = np.asarray(c_list)
                for ii in range(num_vec_to_keep):
                    ax[ii].bar(range(c_list.shape[1]), c_list[ii, :], 0.8)
                    ax[ii].set_ylabel('C '+str(ii))
                    if ii==len(c_list)-1:
                        ax[ii].set_xlabel('Number of trials')
                    ax[ii].set_ylim(C_y_lim)
                if save == False:
                    plt.show()
                else:
                    path = file_path+'/experiments_results/Synergy/synergy_development_'+agentt+'/'+top_folder+'/'+subfolder+'/'+folder_name+'/'+ '/Synergy_plot_' + version

                    os.makedirs(path, exist_ok=True)
                    gg.tight_layout()
                    if svg:
                        gg.savefig(os.path.join(path, 'C-matrix-' + exp_variant_obj.name + '.pdf'),format='pdf')
                    else:
                        if not high_res:
                            gg.savefig(os.path.join(path, 'C-matrix-' + exp_variant_obj.name + '.jpg'),
                                       format='jpg')
                        else:
                            #gg.savefig(os.path.join(path, 'C-matrix-' + exp_variant_obj.name + '.png'),
                            #           format='png', dpi=300)
                            gg.savefig(os.path.join(path, 'C-matrix-' + exp_variant_obj.name + '.pdf'),
                                       format='pdf')

                gb, bx = plt.subplots(total_vec, num_vec_to_keep, figsize=(8, 10))
                sample_length = int(W.shape[0] / total_vec)
                for ii in range(num_vec_to_keep):

                    for j in range(num_vec_to_keep):
                        for k in range(total_vec):
                            bx[k, j].plot(W[sample_length * k:sample_length * (k + 1), j],
                                          linewidth=LW)  # ,c='r',alpha=0.5
                            bx[k, j].axhline(0, color='black', alpha=0.5, linestyle='--')

                            bx[k, j].get_xaxis().set_visible(False)
                            bx[k, j].get_yaxis().set_visible(False)

                            bx[k, j].set_ylim(W_y_lim)

                            if k == 0:
                                bx[k, j].set_title('W' + '$_{}$'.format(j + 1))

                            if k == total_vec - 1:
                                bx[k, j].get_xaxis().set_visible(True)
                                bx[k, j].xaxis.set_major_locator(MaxNLocator(integer=True))
                                if j == 1:
                                    bx[k, j].set_xlabel('Time steps')
                            if j == 0:
                                bx[k, j].get_yaxis().set_visible(True)
                                bx[k, j].set_ylabel('Joint {}'.format(k + 1))

                if save==False:
                    plt.show()
                else:
                    path = file_path+'/experiments_results/Synergy/synergy_development_'+agentt+'/'+top_folder+'/'+subfolder+'/'+folder_name+'/' + '/PCA_components_'+version

                    os.makedirs(path, exist_ok=True)
                    gb.tight_layout()
                    if svg:
                        gb.savefig(path+'/Line_PCA_components_' + exp_variant_obj.name + '.pdf',format='pdf')
                    else:
                        gb.savefig(path + '/Line_PCA_components_' + exp_variant_obj.name+'.png')

                if plot_r_sq:



                    # X=np.load(exp_variant_obj.action_npy, allow_pickle=True)
                    #
                    # state_ = np.load(exp_variant_obj.state_npy, allow_pickle=True)
                    #
                    # mini = per_episode_max_length
                    # if X.shape == (num_epi,):
                    #
                    #     for i in range(num_epi):
                    #         amin = np.asarray(X[i]).shape[0]
                    #         if amin < mini:
                    #             mini = amin
                    #     print(mini)
                    #
                    #     tmp = np.expand_dims(np.asarray(X[0])[0:mini, :], 0)
                    #     for i in range(num_epi-1):
                    #         tmp = np.vstack((tmp, np.expand_dims(np.asarray(X[i + 1])[0:mini, :], 0)))
                    #     print(tmp.shape)
                    #     X = tmp
                    #
                    #     tmp2 = np.expand_dims(np.asarray(state_[0])[0:mini, :], 0)
                    #     for i in range(num_epi-1):
                    #         tmp2 = np.vstack((tmp2, np.expand_dims(np.asarray(state_[i + 1])[0:mini, :], 0)))
                    #
                    #     state_ = tmp2
                    # X=X[0:num_epi,:,:]
                    # state_ = state_[0:num_epi, :, :]
                    #
                    # distance = []
                    # if x_speed_index:
                    #     speed_record = state_[0, :, x_speed_index]
                    #     for i in range(len(speed_record)):
                    #         if i == 0:
                    #             distance.append(speed_record[0])
                    #         else:
                    #             distance.append(np.sum(speed_record[0:i]))
                    #
                    #     distance = np.asarray(distance)
                    # if truncated:
                    #     total_vec_rsq = ori_total_vec_rsq
                    #     if x_speed_index:
                    #         if mini == per_episode_max_length or mini>300:
                    #             current_dist = distance[truncated_start]
                    #             end_dist_index = truncated_start
                    #             tmp_dist = 0
                    #
                    #             while tmp_dist < desired_dist and end_dist_index < len(distance) - 1:
                    #                 end_dist_index += 1
                    #                 tmp_dist = distance[end_dist_index] - current_dist
                    #
                    #             remaining_index = end_dist_index - truncated_start
                    #             desired_length = remaining_index
                    #
                    #         elif mini - desired_length >= 0:
                    #             remaining_index=desired_length
                    #             desired_length = remaining_index
                    #     else:
                    #         desired_length=dll
                    #
                    #     if mini == per_episode_max_length:
                    #         X_truncated = X[:, truncated_start:truncated_start + desired_length, :]
                    #     else:
                    #         if mini >= (truncated_start + desired_length):
                    #             X_truncated = X[:, truncated_start:truncated_start + desired_length, :]
                    #
                    #         elif mini - desired_length >= 0:
                    #
                    #             X_truncated = X[:, mini - desired_length:mini, :]
                    #
                    #         else:
                    #             X_truncated = X[:, 0:mini, :]
                    #
                    #             if mini>=ori_total_vec_rsq:
                    #                 total_vec_rsq=ori_total_vec_rsq
                    #             else:
                    #                 total_vec_rsq=mini
                    #
                    #     X = X_truncated
                    X = reshape_into_ori_shape(X, total_vec)

                    rsq_label = []
                    rsq_single_list=[]

                    max_list = []
                    max_ind = np.argmax(X[0, :, 0])
                    max_list.append(max_ind)
                    X_temp = np.concatenate(
                        (np.expand_dims(X[0, max_ind::, :], 0), np.expand_dims(X[0, 0:max_ind, :], 0)), axis=1)
                    for l in range(1, X.shape[0], 1):
                        max_ind = np.argmax(X[l, :, 0])
                        max_list.append(max_ind)
                        X_temp = np.concatenate((X_temp, np.concatenate(
                            (np.expand_dims(X[l, max_ind::, :], 0), np.expand_dims(X[l, 0:max_ind, :], 0)),
                            axis=1)), axis=0)
                    X = X_temp

                    # if standscale:
                    #     mx = np.mean(X, axis=1)
                    #
                    #     for k in range(X.shape[1]):
                    #         X[:, k, :] = X[:, k, :] - mx

                    X = reshape_into_spt_shape(X)
                    for num_vec_to_keep_ in range(1,total_vec_rsq+1):
                        pca = TruncatedSVD(n_components=num_vec_to_keep_)
                        pca.fit(X)
                        eig_vecs = pca.components_
                        eig_vals = pca.singular_values_
                        eig_pairs = [(eig_vals[i], eig_vecs[i, :]) for i in range(len(eig_vals))]

                        num_features = X.shape[1]
                        percentage = sum(pca.explained_variance_ratio_)
                        proj_mat = eig_pairs[0][1].reshape(num_features, 1)

                        for eig_vec_idx in range(1, num_vec_to_keep_):
                            proj_mat = np.hstack((proj_mat, eig_pairs[eig_vec_idx][1].reshape(num_features, 1)))

                        W = proj_mat

                        C = X.dot(W)
                        X_prime = C.dot(W.T)

                        if manual_rsq:
                            Vm = np.mean(X, axis=0, keepdims=True)
                            resid = X - np.dot(Vm, np.ones((X.shape[1], 1)))
                            resid2 = X - X_prime
                            SST = np.linalg.norm(resid)** 2
                            SSE = np.linalg.norm(resid2)** 2
                            rsq = 1 - SSE / SST
                        else:
                            rsq = r2_score(X, X_prime)

                        rsq_single_list.append(rsq)


                    rsq_label.append('Rsq_' + exp_variant_obj.name)

                    rsq_all_list.append(rsq_single_list)
                    r_sq_single, r_sq_single_ax = plt.subplots(1, 1)

                    r_sq_single_ax.plot(range(1,total_vec_rsq+1),rsq_single_list)
                    r_sq_single_ax.set_ylim([0, 1.05])
                    r_sq_single_ax.set_ylabel(r"${0:s}$".format(R2()))
                    r_sq_single_ax.set_xlabel('Number of PCA components')

                    path = file_path+'/experiments_results/Synergy/synergy_development_'+agentt+'/'+top_folder+'/'+subfolder+'/'+folder_name+'/' + '/Rsq'#+ exp_variant_obj.name+'_synergy'

                    os.makedirs(path, exist_ok=True)
                    r_sq_single.tight_layout()
                    if svg:
                        r_sq_single.savefig(os.path.join(path, 'Rsq_' + exp_variant_obj.name + '.pdf'),format='pdf')
                    else:
                        if not high_res:
                            r_sq_single.savefig(os.path.join(path,
                                                             'Rsq_' + exp_variant_obj.name + '.jpg'))
                        else:
                            r_sq_single.savefig(os.path.join(path, 'Rsq_' + exp_variant_obj.name + '.pdf'),format='pdf')


                    plt.close('all')

            P_list = np.asarray(P_list)
            rsq_all_list = np.asarray(rsq_all_list)
            if get_rid_div:
                bad_ind_list = []
                for ind, p in enumerate(P_list):
                    if ind > 5 and ind < (len(P_list) - 1):
                        if abs(p - P_list[ind - 1]) / abs(p) > div_rate and abs(p - P_list[ind + 1]) / abs(
                                p) > div_rate:
                            bad_ind_list.append(ind)
                print(len(bad_ind_list))
                print(bad_ind_list)
                if len(bad_ind_list) > 0:
                    P_list = np.delete(P_list, bad_ind_list, 0)
                    rsq_all_list = np.delete(rsq_all_list, bad_ind_list, 0)

                step = cmaplen // len(rsq_all_list)
                color_list = []
                c = cmaplen - 1
                for l in range(len(rsq_all_list)):
                    color_list.append(cmaplist[c])
                    c -= step

                if named_label:
                    all_label = []
                    for ind, n in enumerate(np.delete(all_name, bad_ind_list, 0)):
                        prefix_list = exp_variant_list[ind].name.split('_')
                        for pp in prefix_list:
                            if 'C' in pp and 'H' not in pp:
                                prefix = pp
                                break
                        all_label.append(prefix + ': ' + '{:.2f}'.format(exp_variant_list[ind].eval(type_), 2))
                else:
                    all_label = []
                    for ind, n in enumerate(np.delete(all_name, bad_ind_list, 0)):
                        all_label.append('{:.2f}'.format(exp_variant_list[ind].eval(type_), 2))

            for index,ral in enumerate(rsq_all_list):

                if energy_penalty:
                    rsq_single_list = np.asarray(ral)
                    A = 0.5
                    B = 0.5
                    rsq_single_list = (A * rsq_single_list + B * energy_all_list[index]) / (A + B)
                    r_sq_all_combare_ax.plot(range(1, total_vec_rsq + 1), rsq_single_list, color=color_list[index],
                                             label=all_label[index])

                else:
                    r_sq_all_combare_ax.plot(range(1, total_vec_rsq + 1), ral, color=color_list[index],
                                             label=all_label[index])

                r_sq_all_combare_ax.set_ylim([min_rsq, 1])

                if index == 0:
                    r_sq_all_combare_ax.set_ylabel(r"${0:s}$".format(R2()))
                    r_sq_all_combare_ax.set_xlabel('Number of PCA components')

            path = file_path+'/experiments_results/Synergy/synergy_development_'+agentt+'/'+top_folder+'/'+subfolder+'/'+folder_name

            os.makedirs(path, exist_ok=True)
            ex=''


            if named_label:
                ex='_named'
                r_sq_all_combare_ax.legend(fontsize=15, ncol=2)

            r_sq_all_combare.tight_layout()
            r_sq_all_combare_ax.set_ylim([0, 1.05])
            if energy_penalty:
                if svg:
                    r_sq_all_combare.savefig(
                        os.path.join(path, 'Rsq_all_' + type_ + ex + '_' + str(min_rsq) + '_EP' + '.pdf'),format='pdf')
                else:
                    r_sq_all_combare.savefig(
                        os.path.join(path, 'Rsq_all_' + type_ + ex + '_' + str(min_rsq) + '_EP.jpg'))

            else:
                if svg:
                    r_sq_all_combare.savefig(os.path.join(path, 'Rsq_all_' + type_ + ex + '_' + str(min_rsq) + '.pdf'),format='pdf')
                else:
                    if not high_res:
                        r_sq_all_combare.savefig(os.path.join(path, 'Rsq_all_' + type_ + ex + '.jpg'))
                    else:
                        r_sq_all_combare.savefig(os.path.join(path, 'Rsq_all_' + type_ + ex + '.pdf'),format='pdf')