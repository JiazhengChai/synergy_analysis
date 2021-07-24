import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from commons import cwd
"""
File to plot the learning progress for trained quadrupeds.
Need to update 'all_list' below to add path to your own experiment results folder.
Figures will be in synergy_analysis/experiments_results/perf_plot
"""
choice="Exp1" #Exp2 Exp3

base_folder_name="/home/jzchai/PycharmProjects/synergy_analysis/experiments_results/gym"

all_list={
    "No_prior_speed_3":[
        base_folder_name + "/FullCheetahHeavy/Energy0-v4/2021-02-04T10-59-22-FCHeavy_E0_s3r1/ExperimentRunner_0_max_size=1000000,seed=7798_2021-02-04_10-59-24wprc6xd3",
        base_folder_name + "/FullCheetahHeavy/Energy0-v4/2021-02-04T13-54-35-FCHeavy_E0_s3r2/ExperimentRunner_0_max_size=1000000,seed=4973_2021-02-04_13-54-36o2shg03w",
        base_folder_name + "/FullCheetahHeavy/Energy0-v4/2021-02-04T16-51-41-FCHeavy_E0_s3r3/ExperimentRunner_0_max_size=1000000,seed=8542_2021-02-04_16-51-42sc44cvpt",
    ],
    "Trot_speed_3":[
        base_folder_name + "/FullCheetahHeavy/SymlossT-v4/2021-01-23T09-45-45-FCHeavy_SymlossT_s3r1/ExperimentRunner_0_max_size=1000000,seed=2957_2021-01-23_09-45-46hme__ejl",
        base_folder_name + "/FullCheetahHeavy/SymlossT-v4/2021-01-23T13-22-49-FCHeavy_SymlossT_s3r2/ExperimentRunner_0_max_size=1000000,seed=5221_2021-01-23_13-22-50w6ipuicb",
        base_folder_name + "/FullCheetahHeavy/SymlossT-v4/2021-01-23T17-00-37-FCHeavy_SymlossT_s3r3/ExperimentRunner_0_max_size=1000000,seed=5434_2021-01-23_17-00-3967ydnksa",
    ],
    "Trot_speed_5":[
        base_folder_name + "/FullCheetahHeavy/SymlossT-v6/2021-01-24T07-29-52-FCHeavy_SymlossT_s5r1/ExperimentRunner_0_max_size=1000000,seed=199_2021-01-24_07-29-52rr4o4j62",
        base_folder_name + "/FullCheetahHeavy/SymlossT-v6/2021-01-24T11-03-39-FCHeavy_SymlossT_s5r2/ExperimentRunner_0_max_size=1000000,seed=7935_2021-01-24_11-03-40we84gphe",
        base_folder_name + "/FullCheetahHeavy/SymlossT-v6/2021-01-24T07-30-30-FCHeavy_SymlossT_s5r3/ExperimentRunner_0_max_size=1000000,seed=3956_2021-01-24_07-30-316gd1ju8t",
    ],
    "Gallop_speed_3":[
        base_folder_name + "/FullCheetahHeavy/SymlossG-v4/2021-01-23T09-38-42-FCHeavy_SymlossG_s3r1/ExperimentRunner_0_max_size=1000000,seed=6205_2021-01-23_09-38-43njmnp5qt",
        base_folder_name + "/FullCheetahHeavy/SymlossG-v4/2021-01-23T13-12-32-FCHeavy_SymlossG_s3r2/ExperimentRunner_0_max_size=1000000,seed=5361_2021-01-23_13-12-33edrdvhm6",
        base_folder_name + "/FullCheetahHeavy/SymlossG-v4/2021-01-23T16-48-14-FCHeavy_SymlossG_s3r3/ExperimentRunner_0_max_size=1000000,seed=3583_2021-01-23_16-48-15vsaqc779",
    ],
    "Gallop_speed_5":[
        base_folder_name + "/FullCheetahHeavy/SymlossG-v6/2021-01-24T07-13-13-FCHeavy_SymlossG_s5r1/ExperimentRunner_0_max_size=1000000,seed=7744_2021-01-24_07-13-15jcucoqhf",
        base_folder_name + "/FullCheetahHeavy/SymlossG-v6/2021-01-24T10-47-24-FCHeavy_SymlossG_s5r2/ExperimentRunner_0_max_size=1000000,seed=8605_2021-01-24_10-47-25sfqs026j",
        base_folder_name + "/FullCheetahHeavy/SymlossG-v6/2021-01-24T07-20-39-FCHeavy_SymlossG_s5r3/ExperimentRunner_0_max_size=1000000,seed=4691_2021-01-24_07-20-40hrv_a48x",
    ],
    "Gallop_minSpring_speed_3":[
        base_folder_name + "/FullCheetahHeavy/MinSpringG-v4/2021-02-11T04-15-34-FCHeavy_MinSpringG_s3r1/ExperimentRunner_0_max_size=1000000,seed=9510_2021-02-11_04-15-351i_cqg0l",
        base_folder_name + "/FullCheetahHeavy/MinSpringG-v4/2021-02-11T05-25-56-FCHeavy_MinSpringG_s3r2/ExperimentRunner_0_max_size=1000000,seed=6416_2021-02-11_05-25-570kccez67",
        base_folder_name + "/FullCheetahHeavy/MinSpringG-v4/2021-02-11T06-36-17-FCHeavy_MinSpringG_s3r3/ExperimentRunner_0_max_size=1000000,seed=1494_2021-02-11_06-36-173wdfc3dt",
    ],
    "Gallop_maxSpring_speed_3":[
        base_folder_name + "/FullCheetahHeavy/ExSpringG-v4/2021-02-10T23-34-47-FCHeavy_ExSpringG_s3r1/ExperimentRunner_0_max_size=1000000,seed=8248_2021-02-10_23-34-499pgi94yw",
        base_folder_name + "/FullCheetahHeavy/ExSpringG-v4/2021-02-11T00-45-32-FCHeavy_ExSpringG_s3r2/ExperimentRunner_0_max_size=1000000,seed=689_2021-02-11_00-45-33ogchg63p",
        base_folder_name + "/FullCheetahHeavy/ExSpringG-v4/2021-02-11T01-56-10-FCHeavy_ExSpringG_s3r3/ExperimentRunner_0_max_size=1000000,seed=2353_2021-02-11_01-56-115_zdeu37",
    ],

}

####EMBC pictures
#Gait specification effect
if choice=="Exp1":
    title_list =[["No_prior_speed_3", ], ["Trot_speed_3", ],["Gallop_speed_3", ]]
    ori_fig_name="gait_compare"
elif choice=="Exp2":
    #Vary gait and speed
    title_list =[ ["Trot_speed_3", ],["Trot_speed_5", ],["Gallop_speed_3", ],["Gallop_speed_5", ],]
    ori_fig_name="galop_trot_speed_compare"
elif choice=="Exp3":
    #Vary passive spring
    title_list =[["Gallop_minSpring_speed_3", ],["Gallop_speed_3", ],["Gallop_maxSpring_speed_3", ]]
    ori_fig_name="gallop_spring_compare"

folder_path=os.path.join(cwd,"experiments_results","perf_plot")

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

plt.rcParams['figure.figsize'] = [14, 12]#, figsize=(8, 6) 15,12
plt.rcParams['axes.linewidth'] = 2.
plt.rcParams['font.size'] = 35#45#35#50#35
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['lines.linewidth'] = 3.

ctrl_coef=0.1
max_len=400
PI_scale=[-0.5,3.5]

PI_scale_b=False

pic_format="pdf"#"pdf"
legen_loc=0


ori_fig_name=ori_fig_name+"_"+title_list[0][0]

plot_list=[
    all_list[exp_name[0]] for exp_name in title_list
]

for pea,pe,spring in [[True,True,True]]:
    fig, fig_ax = plt.subplots(1, 1)

    plt_energy_action=pea
    plot_energy=pe
    plot_spring_energy=spring

    plot_symloss=False
    plot_PI=False

    if plot_PI:
        fig_name=ori_fig_name+"_PI"
    elif plot_energy:
        fig_name=ori_fig_name+"_energyv2"
    elif plt_energy_action:
        fig_name=ori_fig_name+"_PIactionMag"
    elif plot_spring_energy:
        fig_name = ori_fig_name + "springEnergy"
    else:
        fig_name=ori_fig_name+"_performance"

    for ind,folder in enumerate(plot_list):
        if isinstance(folder,list):
            for indj,subj in enumerate(folder):

                data = pd.read_csv(subj + "/progress.csv")
                epoch_length=data["evaluation/episode-length-avg"][0]


                target_speed=None
                cur_name= title_list[ind][0]
                if "speed" in cur_name:
                    speed=cur_name.split("_")[-1]
                    if "p" in speed:
                        num=speed.replace("p","")
                        target_speed=int(num)/(10*len(num))
                    else:
                        target_speed=int(speed)
                print(target_speed)

                if plot_symloss:
                    y_list = np.expand_dims(np.asarray(data["Sym_loss"][0:max_len]),0)
                elif plot_spring_energy:
                    A = np.expand_dims(np.asarray(data["evaluation/env_infos/energy-sum-mean"][0:max_len]) , 0)
                    B = np.expand_dims(
                        -np.asarray(data["evaluation/env_infos/reward_ctrl-mean-mean"][0:max_len]) * epoch_length * (
                                    1 / ctrl_coef), 0)
                    y_list=A-B
                    plt.ylabel("Spring Energy")

                elif plot_energy:
                    y_list = np.expand_dims(
                        -np.asarray(data["evaluation/env_infos/reward_ctrl-mean-mean"][0:max_len]) * epoch_length * (1/ctrl_coef), 0)

                    plt.ylabel("Energy")
                elif plt_energy_action:
                    energy=np.expand_dims(-np.asarray(data["evaluation/env_infos/reward_ctrl-mean-mean"][0:max_len])*epoch_length*(1/ctrl_coef),0)
                    performance=np.expand_dims(np.asarray(data["evaluation/env_infos/reward_run-mean-mean"][0:max_len])*epoch_length,0)

                    if target_speed:
                        performance=performance+target_speed*epoch_length

                    y_list = performance/energy

                    plt.ylabel("Performance-energy")
                    if PI_scale_b:
                        plt.ylim(PI_scale)

                elif plot_PI:
                    energy = np.expand_dims(np.asarray(data["evaluation/env_infos/energy-sum-mean"][0:max_len]), 0)
                    performance=np.expand_dims(np.asarray(data["evaluation/env_infos/reward_run-mean-mean"][0:max_len])*epoch_length,0)

                    if target_speed:
                        performance=performance+target_speed*epoch_length
                    y_list = performance/energy


                    plt.ylabel("PI")
                    if PI_scale_b:
                        plt.ylim(PI_scale)

                else:
                    y_list=np.expand_dims(np.asarray(data["evaluation/env_infos/reward_run-mean-mean"][0:max_len])*epoch_length,0)

                    if target_speed:
                        y_list=y_list+target_speed*epoch_length
                    plt.ylabel('Performance')

                if indj==0:
                    tmp=y_list
                else:
                    print(y_list.shape)
                    tmp=np.concatenate([tmp,y_list],axis=0)

            tmpMean=np.mean(tmp,axis=0)
            tmpStd=np.std(tmp,axis=0)
            y_list=tmpMean
            fig_ax.plot(range(len(y_list)), y_list, label=title_list[ind][0])
            fig_ax.fill_between(range(len(y_list)), y_list+tmpStd,y_list-tmpStd,alpha=0.2)
        else:
            data = pd.read_csv(folder + "/progress.csv")
            y_list=data["evaluation/ori-return-average"][0:max_len]#.plot(label=title_list[ind])
            fig_ax.plot(range(len(y_list)),y_list,label=title_list[ind])

    fig_ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 3), useOffset=True)
    plt.xlabel("Time steps")
    if not plot_energy:
        plt.legend(loc=legen_loc)

    plt.tight_layout()
    if fig_name=="":
        plt.show()
    else:
        plt.savefig(os.path.join(folder_path,fig_name+"."+pic_format),format=pic_format)

    plt.close()

