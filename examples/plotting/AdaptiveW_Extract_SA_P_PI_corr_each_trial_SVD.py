import os
import csv
import argparse
import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from exp_variant_class import exp_variant
from sklearn.decomposition import TruncatedSVD,PCA
from examples.plotting.commons import *
from copy import deepcopy

cmap = plt.cm.viridis
cmaplist = [cmap(i) for i in range(cmap.N)]
cmaplen=len(cmaplist)


plt.rcParams["figure.figsize"] = (10,8)

parser = argparse.ArgumentParser()

parser.add_argument('--tr', nargs='+', required=True)

parser.add_argument('--ee',type=str, nargs='+',choices=spt_energy_list)

parser.add_argument('--no_div',action='store_true')

parser.add_argument('--agentt',
                    type=str,choices=spt_agent_list)

args = parser.parse_args()
agentt=args.agentt

svg = False
save = True
sortt = False
avg_plot = True
precheck = False
truncated = True
plot_r_sq = True
manual_rsq = True
standscale = True
if args.no_div:
    get_rid_div=True
else:
    get_rid_div=False
named_label = False
energy_penalty = False
plot_norm_single = True


min_rsq = 0
num_epi = 10
div_rate=0.4
recon_num = 8
LW_action = 4
ori_num_vec = 3
desired_length = 50  # 50#28#7
truncated_start = 200
ori_total_vec_rsq = 9
per_episode_max_length=1000

type_ = 'P'
action_path=None
reward_path=None
state_path=None

total_vec=agent_info_dict[agentt]['total_vec']
total_chk=agent_info_dict[agentt]['total_chk']
ori_final=agent_info_dict[agentt]['ori_final']
ori_begin=agent_info_dict[agentt]['ori_begin']
ori_step=agent_info_dict[agentt]['ori_step']
x_speed_index=agent_info_dict[agentt]['x_speed_index']
desired_dist=agent_info_dict[agentt]['desired_dist']

try:
    dll=agent_info_dict[agentt]['dll']
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

if 'E1' in args.ee[0]:
    top_folder=agentt+'_E1'
elif 'Ep5' in args.ee[0]:
    top_folder=agentt+'_Ep5'
elif 'Ep1' in args.ee[0]:
    top_folder=agentt+'_Ep1'
elif 'Ep25' in args.ee[0]:
    top_folder=agentt+'_Ep25'
elif 'Ez' in args.ee[0]:
    top_folder=agentt+'_Ez'
elif 'Ealt' in args.ee[0]:
    top_folder=agentt+'_Ealt'
elif 'sR' in args.ee[0]:
    top_folder=agentt+'_sR'
elif 'bR' in args.ee[0]:
    top_folder=agentt+'_bR'
elif  args.ee[0]=='sL':
    top_folder=agentt+'_sL'
elif  args.ee[0]=='sD':
    top_folder=agentt+'_sD'
elif  "v" in args.ee[0]:
    top_folder=agentt+args.ee[0]
else:
    top_folder=agentt

if not os.path.exists(file_path+'/experiments_results/Synergy/all_csv/raw_csv/'+top_folder):
    os.makedirs(file_path+'/experiments_results/Synergy/all_csv/raw_csv/'+top_folder, exist_ok=True)

if args.agentt=='A' and 'E0' in args.ee:
    args.tr=['']+args.tr

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

        if not get_rid_div:
            surface_csv = open(
                file_path+'/experiments_results/Synergy/all_csv/raw_csv/' + top_folder  + '/' + base +tr+'_all_surface.csv', 'w')
        else:
            surface_csv = open(
                file_path+'/experiments_results/Synergy/all_csv/raw_csv/' + top_folder  + '/' + base +tr+'_no_div_all_surface.csv', 'w')

        writer = csv.writer(surface_csv, lineterminator='\n')

        writer.writerow(['Checkpoint', 'Surface Area','P', 'PI', 'E','PP','PPI','Spatial area'])

        all_names=[]

        tmp=[]
        for cc in range(final, begin - step, -step):
            tmp.append(base + '_C' + str(cc) + trial)
            '''if cc==3000 and 'HC' in agentt and 'dof' not in agentt and ee=='E0':
                tmp.append(base + trial)
                try:
                    dummy = exp_variant(tmp[-1],action_path=action_path,reward_path=reward_path,state_path=state_path)
                except:
                    tmp.pop()
                    tmp.append(base + '_C' + str(cc) + trial)
            else:
                tmp.append(base + '_C' + str(cc) + trial)'''


        all_names.append(tmp)
        all_names.reverse()
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
                        complete = exp_variant(n).check_complete_data()
                    if complete:
                        tmpp.append(n)
                top_tmp.append(tmpp)

            all_names=top_tmp

        for ind_all_name,all_name in enumerate(all_names):
            agent = all_name[0] + '_spatiotemporal_evolution'

            folder_name=agent+'_Compare_'+type_+'_'+str(desired_length)
            if manual_rsq:
                folder_name=folder_name+'_manual_rsq'
            if standscale:
                folder_name = folder_name + '_SS'

            '''step = cmaplen // 30
            color_list=[]
            c=0
            for l in range(30):
                color_list.append(cmaplist[c])
                c+=step

            color_list=[color_list[0],color_list[-1]]'''
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

            P_list=[]
            PI_list=[]
            E_list=[]
            SA_list=[]
            PP_list = []
            PPI_list=[]
            SpatialArea_list=[]
            current_checkpoint_list=[]
            for n_ind,name in enumerate(all_name):

                exp_variant_obj=exp_variant_list[n_ind]

                name_list=name.split('_')
                current_checkpoint=0
                for nn in name_list:
                    if nn[0]=='C':
                        current_checkpoint=nn
                if current_checkpoint==0:
                    current_checkpoint='C'+str(ori_final)

                rsq_label = []

                X=np.load(exp_variant_obj.action_npy,allow_pickle=True)
                state_ = np.load(exp_variant_obj.state_npy,allow_pickle=True)


                mini = per_episode_max_length
                if X.shape == (num_epi,):
                    # print('a')
                    for i in range(num_epi):
                        amin = np.asarray(X[i]).shape[0]
                        if amin < mini:
                            mini = amin
                    print(mini)

                    tmp = np.expand_dims(np.asarray(X[0])[-mini::, :], 0)
                    for i in range(num_epi-1):
                        tmp = np.vstack((tmp, np.expand_dims(np.asarray(X[i + 1])[-mini::, :], 0)))
                    print(tmp.shape)
                    X = tmp

                    tmp2 = np.expand_dims(np.asarray(state_[0])[-mini::, :], 0)
                    for i in range(num_epi-1):
                        tmp2 = np.vstack((tmp2, np.expand_dims(np.asarray(state_[i + 1])[-mini::, :], 0)))
                    state_ = tmp2
                X=X[0:num_epi,:,:]#10,1000,12
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

                            desired_length = remaining_index
                            print(desired_length)
                        elif mini - desired_length >= 0:
                            remaining_index=desired_length
                            desired_length = remaining_index
                    else:
                        desired_length = dll

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

                    X = X_truncated#10,167,12


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
                X = X_temp#10,167,12

                X_before_norm=deepcopy(X)
                ##########################SPT START###################################
                if standscale:
                    mx = np.mean(X, axis=1)
                    for k in range(X.shape[1]):
                        X[:, k, :] = X[:, k, :] - mx

                X = reshape_into_spt_shape(X)
                rsq_single_list = []
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
                        SST = np.linalg.norm(resid)
                        SSE = np.linalg.norm(resid2)
                        rsq = 1 - SSE / SST

                    else:
                        rsq = r2_score(X, X_prime)

                    rsq_single_list.append(rsq)


                surface_area=integrate.simps(rsq_single_list,range(1,total_vec_rsq+1))
                SA_list.append(surface_area)
                ##########################SPT END###################################

                ##########################SPATIAL START###################################
                if standscale:
                    ori_shape = X_before_norm.shape

                    X_spatial = reshape_into_spatial_shape(X_before_norm)
                    mx = np.mean(X_spatial, axis=1)

                    X_spatial = X_spatial - np.expand_dims(mx, 1)
                    X_spatial = X_spatial.T  # 1200,8

                    num_features = X_spatial.shape[1]
                    num_vec_to_keep_max = X_spatial.shape[1] + 1

                rsq_spatial_list = []
                for num_vec_to_keep_ in range(1, total_vec + 1):

                    pca = PCA(n_components=num_vec_to_keep_)
                    pca.fit(X_spatial)
                    eig_vecs = pca.components_
                    eig_vals = pca.singular_values_
                    eig_pairs = [(eig_vals[i], eig_vecs[i, :]) for i in range(len(eig_vals))]

                    num_features = X_spatial.shape[1]
                    percentage = sum(pca.explained_variance_ratio_)
                    proj_mat = eig_pairs[0][1].reshape(num_features, 1)

                    for eig_vec_idx in range(1, num_vec_to_keep_):
                        proj_mat = np.hstack((proj_mat, eig_pairs[eig_vec_idx][1].reshape(num_features, 1)))

                    W = proj_mat

                    C = X_spatial.dot(W)
                    X_spatial_prime = C.dot(W.T)

                    if manual_rsq:
                        Vm = np.mean(X_spatial, axis=0, keepdims=True)
                        # resid = X - np.dot(Vm, np.ones((X.shape[1],1 )))
                        resid = X_spatial - np.dot(np.ones((X_spatial.shape[0], 1)), Vm)
                        resid2 = X_spatial - X_spatial_prime
                        SST = np.linalg.norm(resid) ** 2
                        SSE = np.linalg.norm(resid2) ** 2
                        rsq = 1 - SSE / SST
                    else:
                        rsq = r2_score(X_spatial, X_spatial_prime)

                    rsq_spatial_list.append(rsq)

                surface_area_spatial = integrate.simps(rsq_spatial_list, range(1, total_vec + 1))

                SpatialArea_list.append(surface_area_spatial)
                ##########################SPATIAL END###################################

                rsq_label.append('Rsq_' + exp_variant_obj.name)

                P=exp_variant_obj.eval('P')
                PI=exp_variant_obj.eval('PI')
                E = exp_variant_obj.eval('E')
                P_list.append(P)
                PI_list.append(PI)
                E_list.append(E)
                current_checkpoint_list.append(current_checkpoint)

                try:
                    PP = exp_variant_obj.eval('PP')
                    PP_list.append(PP)
                    PPI = exp_variant_obj.eval('PPI')
                    PPI_list.append(PPI)
                except:
                    print("PP list load failed in AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py line 390.")

                if np.isnan(surface_area):
                    surface_area=SA_list[n_ind-1]


                if np.isnan(surface_area_spatial):
                    surface_area_spatial=SpatialArea_list[n_ind-1]

        if get_rid_div:
            SpatialArea_list = np.flip(np.asarray(SpatialArea_list))
            SA_list = np.flip(np.asarray(SA_list))
            P_list = np.flip(np.asarray(P_list))
            PI_list = np.flip(np.asarray(PI_list))
            E_list = np.flip(np.asarray(E_list))
            current_checkpoint_list = np.flip(np.asarray(current_checkpoint_list))
            try:
                PP_list = np.flip(np.asarray(PP_list))
                PPI_list = np.flip(np.asarray(PPI_list))

            except:
                print("PP list failed in AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py line 406.")


            bad_ind_list = []
            for ind, p in enumerate(P_list):
                if ind >5 and ind < (len(P_list) - 1):
                    if abs(p - P_list[ind - 1]) / abs(p) > div_rate and abs(p - P_list[ind + 1]) / abs(
                            p) > div_rate:
                        bad_ind_list.append(ind)
            print(len(bad_ind_list))
            print(bad_ind_list)
            if len(bad_ind_list) > 0:
                P_list = np.delete(P_list, bad_ind_list, 0)
                PI_list = np.delete(PI_list, bad_ind_list, 0)

                E_list = np.delete(E_list, bad_ind_list, 0)
                current_checkpoint_list=np.delete(current_checkpoint_list, bad_ind_list, 0)
                SA_list = np.delete(SA_list, bad_ind_list, 0)
                SpatialArea_list = np.delete(SpatialArea_list, bad_ind_list, 0)

                try:
                    PP_list = np.delete(PP_list, bad_ind_list, 0)
                    PPI_list = np.delete(PPI_list, bad_ind_list, 0)

                except:
                    print("PP list failed in AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py line 427.")

            SpatialArea_list = np.flip(SpatialArea_list)
            SA_list = np.flip(SA_list)
            P_list = np.flip(P_list)
            PI_list = np.flip(PI_list)
            E_list = np.flip(E_list)
            current_checkpoint_list = np.flip(current_checkpoint_list)
            try:
                PP_list = np.flip(PP_list)
                PPI_list = np.flip(PPI_list)

            except:
                print("PP list failed in AdaptiveW_Extract_SA_P_PI_corr_each_trial_SVD.py line 437.")

        if len(PP_list)==0:
            PP_list=[0]*len(P_list)
            PPI_list=[0]*len(P_list)

        for ind in range(len(P_list)):
            writer.writerow([current_checkpoint_list[ind], SA_list[ind], P_list[ind], PI_list[ind], E_list[ind],PP_list[ind],PPI_list[ind],SpatialArea_list[ind]])


surface_csv.close()




