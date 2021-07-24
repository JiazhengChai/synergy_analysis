from matplotlib import pyplot as plt
import argparse
import csv
import pandas as pd
try:
    from examples.plotting.commons import *
except:
    from commons import *
cmap = plt.cm.viridis
cmaplist = [cmap(i) for i in range(cmap.N)]
cmaplen=len(cmaplist)

color_list=['b','r','g','c','m','y','k','#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']

plt.rcParams["figure.figsize"] = (10,8)

parser = argparse.ArgumentParser()

parser.add_argument('--agentt',
                    type=str,choices=all_agent_list)
parser.add_argument('--ee',
                    type=str,choices=spt_energy_list,default="")
args = parser.parse_args()

tif=False
sortt=False
standscale=True
temporal=True
manual_pca=False
recon_num=8

ori_total_vec_rsq=9
truncated_start=200
dll=50

std=True

agentt=args.agentt

precheck=False

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

if args.ee:
    agentt=agentt+args.ee

#cwd=os.getcwd()
path_to_folder=cwd+'/experiments_results/Synergy/all_csv/process_SA_intermediate'

final = ori_final
begin = ori_begin
step = ori_step

path_to_csv=path_to_folder+'/'+agentt

output_folder=cwd+'/experiments_results/Synergy/all_csv/process_SA_final_summary/'+agentt
if not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)

process_csv = open(output_folder+ '/' + agentt +'_final_summary.csv', 'w')

writer = csv.writer(process_csv, lineterminator='\n')

writer.writerow(['Algorithms',

                 'corr SA_P mean', 'corr SA_P std',
                 'corr SA_PI mean', 'corr_SA_PI std',
                 'corr SA_E mean', 'corr SA_E std',

                'FP mean','FP std',
                'FPI mean','FPI std',
                'FE mean','FE std',

                'FSA mean','FSA std',
                'DSA mean','DSA std',
                'ASA mean','ASA std',


                'Corr FSA_FP','Corr FSA_FPI', 'Corr FSA_FE',
                'Corr DSA_FP','Corr DSA_FPI', 'Corr DSA_FE',
                'Corr ASA_FP','Corr ASA_FPI', 'Corr ASA_FE',

                 'FPP mean', 'FPP std',
                 'corr SA_PP mean', 'corr SA_PP std',
                 'Corr FSA_FPP',
                 'Corr DSA_FPP',
                 'Corr ASA_FPP',

                 'FPPI mean', 'FPPI std',
                 'corr SA_PPI mean', 'corr_SA_PPI std',
                 'Corr FSA_FPPI',
                 'Corr DSA_FPPI',
                 'Corr ASA_FPPI'
                 ])

process_csv_no_div =open(output_folder+ '/' + agentt +'_final_summary_no_div.csv', 'w')

writer_no_div = csv.writer(process_csv_no_div, lineterminator='\n')

writer_no_div.writerow(['Algorithms',

                 'corr SA_P mean', 'corr SA_P std',
                 'corr SA_PI mean', 'corr_SA_PI std',
                 'corr SA_E mean', 'corr SA_E std',

                'FP mean','FP std',
                'FPI mean','FPI std',
                'FE mean','FE std',

                'FSA mean','FSA std',
                'DSA mean','DSA std',
                'ASA mean','ASA std',


                'Corr FSA_FP','Corr FSA_FPI', 'Corr FSA_FE',
                'Corr DSA_FP','Corr DSA_FPI', 'Corr DSA_FE',
                'Corr ASA_FP','Corr ASA_FPI', 'Corr ASA_FE',

                'FPP mean', 'FPP std',
                'corr SA_PP mean', 'corr SA_PP std',
                'Corr FSA_FPP',
                'Corr DSA_FPP',
                'Corr ASA_FPP',

                'FPPI mean', 'FPPI std',
                'corr SA_PPI mean', 'corr_SA_PPI std',
                'Corr FSA_FPPI',
                'Corr DSA_FPPI',
                'Corr ASA_FPPI'
                 ])

TD3_data=[]
TD3_no_div_data=[]

for csv_ in os.listdir(path_to_csv):
    current_csv = pd.read_csv(path_to_csv + '/' + csv_)

    trial_name_list=current_csv['Trials']
    counter=0
    for name in trial_name_list:
        if 'TD3' not in name:
            counter+=1


    corr_SAP_list = current_csv['Corr SA_P'][0:counter]
    corr_SAPI_list = current_csv['Corr SA_PI'][0:counter]
    corr_SAE_list = current_csv['Corr SA_E'][0:counter]

    FSA_list=current_csv['FSA'][0:counter]
    DSA_list = current_csv['DSA'][0:counter]
    ASA_list = current_csv['ASA'][0:counter]

    FP_list=current_csv['FP'][0:counter]
    FPI_list = current_csv['FPI'][0:counter]
    FE_list = current_csv['FE'][0:counter]

    corr_SAP_list = np.asarray(corr_SAP_list)
    corr_SAPI_list = np.asarray(corr_SAPI_list)
    corr_SAE_list = np.asarray(corr_SAE_list)

    FSA_list = np.asarray(FSA_list)
    DSA_list = np.asarray(DSA_list)
    ASA_list = np.asarray(ASA_list)

    FP_list = np.asarray(FP_list)
    FPI_list = np.asarray(FPI_list)
    FE_list = np.asarray(FE_list)

    try:
        FPP_list = current_csv['FPP'][0:counter]
        FPP_list = np.asarray(FPP_list)

        corr_SAPP_list = current_csv['Corr SA_PP'][0:counter]
        corr_SAPP_list = np.asarray(corr_SAPP_list)

        FPPI_list = current_csv['FPPI'][0:counter]
        FPPI_list = np.asarray(FPPI_list)

        corr_SAPPI_list = current_csv['Corr SA_PPI'][0:counter]
        corr_SAPPI_list = np.asarray(corr_SAPPI_list)

    except:
        print("Exception in AdativeW_SA_summary.py line 162.")

    trial_name_list= np.asarray(trial_name_list)

    try:
        corr_SAP_mean = np.mean(corr_SAP_list)
        corr_SAP_std = np.std(corr_SAP_list)
        corr_SAPI_mean = np.mean(corr_SAPI_list)
        corr_SAPI_std = np.std(corr_SAPI_list)
        corr_SAE_mean = np.mean(corr_SAE_list)
        corr_SAE_std = np.std(corr_SAE_list)

        FP_mean=np.mean(FP_list)
        FP_std=np.std(FP_list)
        FPI_mean=np.mean(FPI_list)
        FPI_std=np.std(FPI_list)
        FE_mean=np.mean(FE_list)
        FE_std=np.std(FE_list)

        FSA_mean = np.mean(FSA_list)
        FSA_std = np.std(FSA_list)
        DSA_mean = np.mean(DSA_list)
        DSA_std = np.std(DSA_list)
        ASA_mean = np.mean(ASA_list)
        ASA_std = np.std(ASA_list)

        FPP_mean=np.mean(FPP_list)
        FPP_std=np.std(FPP_list)
        corr_SAPP_mean = np.mean(corr_SAPP_list)
        corr_SAPP_std = np.std(corr_SAPP_list)

        FPPI_mean=np.mean(FPPI_list)
        FPPI_std=np.std(FPPI_list)
        corr_SAPPI_mean = np.mean(corr_SAPPI_list)
        corr_SAPPI_std = np.std(corr_SAPPI_list)
    except:
        print("Exception in AdativeW_SA_summary.py line 197.")
        continue


    corr_FSA_FP = np.corrcoef(FSA_list, FP_list)[0, 1]
    corr_FSA_FPI = np.corrcoef(FSA_list, FPI_list)[0, 1]
    corr_FSA_FE = np.corrcoef(FSA_list, FE_list)[0, 1]
    corr_DSA_FP = np.corrcoef(DSA_list, FP_list)[0, 1]
    corr_DSA_FPI = np.corrcoef(DSA_list, FPI_list)[0, 1]
    corr_DSA_FE = np.corrcoef(DSA_list, FE_list)[0, 1]
    corr_ASA_FP = np.corrcoef(ASA_list, FP_list)[0, 1]
    corr_ASA_FPI = np.corrcoef(ASA_list, FPI_list)[0, 1]
    corr_ASA_FE = np.corrcoef(ASA_list, FE_list)[0, 1]

    try:
        corr_FSA_FPP = np.corrcoef(FSA_list, FPP_list)[0, 1]
        corr_DSA_FPP = np.corrcoef(DSA_list, FPP_list)[0, 1]
        corr_ASA_FPP = np.corrcoef(ASA_list, FPP_list)[0, 1]

        corr_FSA_FPPI = np.corrcoef(FSA_list, FPPI_list)[0, 1]
        corr_DSA_FPPI = np.corrcoef(DSA_list, FPPI_list)[0, 1]
        corr_ASA_FPPI = np.corrcoef(ASA_list, FPPI_list)[0, 1]
    except:
        print("Exception in AdativeW_SA_summary.py line 221.")
        continue

    if 'no_div' not in name:
        writer.writerow(['SAC',corr_SAP_mean,corr_SAP_std,corr_SAPI_mean,corr_SAPI_std,corr_SAE_mean,corr_SAE_std,
                         FP_mean,FP_std,FPI_mean,FPI_std,FE_mean,FE_std,
                         FSA_mean,FSA_std,DSA_mean,DSA_std,ASA_mean,ASA_std,
                         corr_FSA_FP,corr_FSA_FPI,corr_FSA_FE,
                         corr_DSA_FP,corr_DSA_FPI,corr_DSA_FE,
                         corr_ASA_FP,corr_ASA_FPI,corr_ASA_FE,
                         FPP_mean,FPP_std,
                         corr_SAPP_mean, corr_SAPP_std,
                         corr_FSA_FPP,
                         corr_DSA_FPP,
                         corr_ASA_FPP,

                         FPPI_mean, FPPI_std,
                         corr_SAPPI_mean, corr_SAPPI_std,
                         corr_FSA_FPPI,
                         corr_DSA_FPPI,
                         corr_ASA_FPPI

                         ])
    else:
        writer_no_div.writerow(['SAC',corr_SAP_mean,corr_SAP_std,corr_SAPI_mean,corr_SAPI_std,corr_SAE_mean,corr_SAE_std,
                         FP_mean,FP_std,FPI_mean,FPI_std,FE_mean,FE_std,
                         FSA_mean,FSA_std,DSA_mean,DSA_std,ASA_mean,ASA_std,
                         corr_FSA_FP,corr_FSA_FPI,corr_FSA_FE,
                         corr_DSA_FP,corr_DSA_FPI,corr_DSA_FE,
                         corr_ASA_FP,corr_ASA_FPI,corr_ASA_FE,
                            FPP_mean, FPP_std,
                            corr_SAPP_mean, corr_SAPP_std,
                            corr_FSA_FPP,
                            corr_DSA_FPP,
                            corr_ASA_FPP,

                            FPPI_mean, FPPI_std,
                            corr_SAPPI_mean, corr_SAPPI_std,
                            corr_FSA_FPPI,
                            corr_DSA_FPPI,
                            corr_ASA_FPPI
                         ])
###################################################TD3##########################################################################
    corr_SAP_list = current_csv['Corr SA_P'][counter::]
    corr_SAPI_list = current_csv['Corr SA_PI'][counter::]
    corr_SAE_list = current_csv['Corr SA_E'][counter::]

    FSA_list = current_csv['FSA'][counter::]
    DSA_list = current_csv['DSA'][counter::]
    ASA_list = current_csv['ASA'][counter::]

    FP_list = current_csv['FP'][counter::]
    FPI_list = current_csv['FPI'][counter::]
    FE_list = current_csv['FE'][counter::]

    corr_SAP_list = np.asarray(corr_SAP_list)
    corr_SAPI_list = np.asarray(corr_SAPI_list)
    corr_SAE_list = np.asarray(corr_SAE_list)

    FSA_list = np.asarray(FSA_list)
    DSA_list = np.asarray(DSA_list)
    ASA_list = np.asarray(ASA_list)

    FP_list = np.asarray(FP_list)
    FPI_list = np.asarray(FPI_list)
    FE_list = np.asarray(FE_list)
    try:
        FPP_list = current_csv['FPP'][counter::]
        FPP_list = np.asarray(FPP_list)

        corr_SAPP_list = current_csv['Corr SA_PP'][counter::]
        corr_SAPP_list = np.asarray(corr_SAPP_list)

        FPPI_list = current_csv['FPPI'][counter::]
        FPPI_list = np.asarray(FPPI_list)

        corr_SAPPI_list = current_csv['Corr SA_PPI'][counter::]
        corr_SAPPI_list = np.asarray(corr_SAPPI_list)

    except:
        print("Exception in AdativeW_SA_summary.py line 283.")

    trial_name_list = np.asarray(trial_name_list)

    try:
        corr_SAP_mean = np.mean(corr_SAP_list)
        corr_SAP_std = np.std(corr_SAP_list)
        corr_SAPI_mean = np.mean(corr_SAPI_list)
        corr_SAPI_std = np.std(corr_SAPI_list)
        corr_SAE_mean = np.mean(corr_SAE_list)
        corr_SAE_std = np.std(corr_SAE_list)

        FP_mean = np.mean(FP_list)
        FP_std = np.std(FP_list)
        FPI_mean = np.mean(FPI_list)
        FPI_std = np.std(FPI_list)
        FE_mean = np.mean(FE_list)
        FE_std = np.std(FE_list)

        FSA_mean = np.mean(FSA_list)
        FSA_std = np.std(FSA_list)
        DSA_mean = np.mean(DSA_list)
        DSA_std = np.std(DSA_list)
        ASA_mean = np.mean(ASA_list)
        ASA_std = np.std(ASA_list)

        FPP_mean = np.mean(FPP_list)
        FPP_std = np.std(FPP_list)
        corr_SAPP_mean = np.mean(corr_SAPP_list)
        corr_SAPP_std = np.std(corr_SAPP_list)

        FPPI_mean = np.mean(FPPI_list)
        FPPI_std = np.std(FPPI_list)
        corr_SAPPI_mean = np.mean(corr_SAPPI_list)
        corr_SAPPI_std = np.std(corr_SAPPI_list)
    except:
        print("Exception in AdativeW_SA_summary.py line 314.")
        continue

    corr_FSA_FP = np.corrcoef(FSA_list, FP_list)[0, 1]
    corr_FSA_FPI = np.corrcoef(FSA_list, FPI_list)[0, 1]
    corr_FSA_FE = np.corrcoef(FSA_list, FE_list)[0, 1]
    corr_DSA_FP = np.corrcoef(DSA_list, FP_list)[0, 1]
    corr_DSA_FPI = np.corrcoef(DSA_list, FPI_list)[0, 1]
    corr_DSA_FE = np.corrcoef(DSA_list, FE_list)[0, 1]
    corr_ASA_FP = np.corrcoef(ASA_list, FP_list)[0, 1]
    corr_ASA_FPI = np.corrcoef(ASA_list, FPI_list)[0, 1]
    corr_ASA_FE = np.corrcoef(ASA_list, FE_list)[0, 1]

    try:
        corr_FSA_FPP = np.corrcoef(FSA_list, FPP_list)[0, 1]
        corr_DSA_FPP = np.corrcoef(DSA_list, FPP_list)[0, 1]
        corr_ASA_FPP = np.corrcoef(ASA_list, FPP_list)[0, 1]

        corr_FSA_FPPI = np.corrcoef(FSA_list, FPPI_list)[0, 1]
        corr_DSA_FPPI = np.corrcoef(DSA_list, FPPI_list)[0, 1]
        corr_ASA_FPPI = np.corrcoef(ASA_list, FPPI_list)[0, 1]
    except:
        print("Exception in AdativeW_SA_summary.py line 332.")
        continue

    if 'no_div' not in name:
        writer.writerow(['TD3', corr_SAP_mean, corr_SAP_std, corr_SAPI_mean, corr_SAPI_std, corr_SAE_mean, corr_SAE_std,
                         FP_mean, FP_std, FPI_mean, FPI_std, FE_mean, FE_std,
                         FSA_mean, FSA_std, DSA_mean, DSA_std, ASA_mean, ASA_std,
                         corr_FSA_FP, corr_FSA_FPI, corr_FSA_FE,
                         corr_DSA_FP, corr_DSA_FPI, corr_DSA_FE,
                         corr_ASA_FP, corr_ASA_FPI, corr_ASA_FE,
                         FPP_mean, FPP_std,
                         corr_SAPP_mean, corr_SAPP_std,
                         corr_FSA_FPP,
                         corr_DSA_FPP,
                         corr_ASA_FPP,
                         FPPI_mean, FPPI_std,
                         corr_SAPPI_mean, corr_SAPPI_std,
                         corr_FSA_FPPI,
                         corr_DSA_FPPI,
                         corr_ASA_FPPI
                         ])
    else:
        writer_no_div.writerow(['TD3',
                                corr_SAP_mean, corr_SAP_std, corr_SAPI_mean, corr_SAPI_std, corr_SAE_mean, corr_SAE_std,
                         FP_mean, FP_std, FPI_mean, FPI_std, FE_mean, FE_std,
                         FSA_mean, FSA_std, DSA_mean, DSA_std, ASA_mean, ASA_std,
                         corr_FSA_FP, corr_FSA_FPI, corr_FSA_FE,
                         corr_DSA_FP, corr_DSA_FPI, corr_DSA_FE,
                         corr_ASA_FP, corr_ASA_FPI, corr_ASA_FE,
                                FPP_mean, FPP_std,
                                corr_SAPP_mean, corr_SAPP_std,
                                corr_FSA_FPP,
                                corr_DSA_FPP,
                                corr_ASA_FPP,
                                FPPI_mean, FPPI_std,
                                corr_SAPPI_mean, corr_SAPPI_std,
                                corr_FSA_FPPI,
                                corr_DSA_FPPI,
                                corr_ASA_FPPI
                         ])

process_csv.close()

"""
'Algorithms',

                 'corr SA_P mean', 'corr SA_P std',
                 'corr SA_PI mean', 'corr_SA_PI std',
                 'corr SA_E mean', 'corr SA_E std',

                'FP mean','FP std',
                'FPI mean','FPI std',
                'FE mean','FE std',

                'FSA mean','FSA std',
                'DSA mean','DSA std',
                'ASA mean','ASA std',


                'Corr FSA_FP','Corr FSA_FPI', 'Corr FSA_FE',
                'Corr DSA_FP','Corr DSA_FPI', 'Corr DSA_FE',
                'Corr ASA_FP','Corr ASA_FPI', 'Corr ASA_FE',

                'FPP mean', 'FPP std',
                'corr SA_PP mean', 'corr SA_PP std',
                'Corr FSA_FPP',
                'Corr DSA_FPP',
                'Corr ASA_FPP',

                'corr SA_PPI mean', 'corr_SA_PPI std',
                'FPPI mean', 'FPPI std',
                'Corr FSA_FPPI',
                'Corr DSA_FPPI',
                'Corr ASA_FPPI'
"""
