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

agentt_folder = agentt
top_folder=agentt

#cwd=os.getcwd()
path_to_folder=cwd+'/experiments_results/Synergy/all_csv/raw_csv'
#path_to_folder=os.path.join(cwd,'experiments_results','Synergy','all_csv','raw_csv')

final = ori_final
begin = ori_begin
step = ori_step
print(agentt_folder)
#path_to_csv=path_to_folder+'/'+agentt_folder
path_to_csv=os.path.join(path_to_folder,agentt_folder)
output_folder=cwd+'/experiments_results/Synergy/all_csv/process_SA_intermediate/'+agentt
if not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)

process_csv = open(output_folder+ '/' + agentt +'_process_all_surface.csv', 'w')

writer = csv.writer(process_csv, lineterminator='\n')

writer.writerow(['Trials', 'Corr SA_P','Corr SA_PI', 'Corr SA_E','FSA', 'DSA', 'ASA','FP', 'FPI', 'FE','FPP','Corr SA_PP', 'FPPI','Corr SA_PPI' ])

process_csv_no_div = open(output_folder+ '/' + agentt +'_process_all_surface_no_div.csv', 'w')

writer_no_div = csv.writer(process_csv_no_div, lineterminator='\n')

writer_no_div.writerow(['Trials', 'Corr SA_P','Corr SA_PI', 'Corr SA_E','FSA', 'DSA', 'ASA','FP', 'FPI', 'FE','FPP','Corr SA_PP', 'FPPI','Corr SA_PPI' ])

TD3_data=[]
TD3_no_div_data=[]
for csv_ in os.listdir(path_to_csv):
    current_csv = pd.read_csv(path_to_csv + '/' + csv_)

    current_name_list=csv_.split('_')
    current_name_list=current_name_list[0:-2]
    name=''
    for cn in current_name_list:
        name=name+cn+'_'
    name=name[0:-1]

    P_list = current_csv['P']
    PI_list = current_csv['PI']
    E_list = current_csv['E']
    try:
        PP_list = current_csv['PP']
        PPI_list = current_csv['PPI']

    except:
        print("Exception in AdaptiveW_process_SA.py line 102")

    SA_list = current_csv['Surface Area']
    Checkpoint_list = current_csv['Checkpoint']
    P_list = np.asarray(P_list)
    try:
        PP_list =  np.asarray(PP_list)
        PPI_list =  np.asarray(PPI_list)

    except:
        print("Exception in AdaptiveW_process_SA.py line 110")
    PI_list = np.asarray(PI_list)
    E_list = np.asarray(E_list)
    SA_list = np.asarray(SA_list)
    Checkpoint_list = np.asarray(Checkpoint_list)

    corr_SA_P = np.corrcoef(SA_list, P_list)[0, 1]
    corr_SA_PI = np.corrcoef(SA_list, PI_list)[0, 1]
    corr_SA_E = np.corrcoef(SA_list, E_list)[0, 1]
    try:
        corr_SA_PP = np.corrcoef(SA_list, PP_list)[0, 1]
        corr_SA_PPI = np.corrcoef(SA_list, PPI_list)[0, 1]

    except:
        print("Exception in AdaptiveW_process_SA.py line 122")


    FP = P_list[0]
    FPI = PI_list[0]
    FE = E_list[0]
    try:
        FPP = PP_list[0]
        FPPI = PPI_list[0]

    except:
        print("Exception in AdaptiveW_process_SA.py line 131")

    FSA = SA_list[0]
    DSA = SA_list[0] - SA_list[-1]

    SA_list2 = sorted(SA_list)
    ASA=SA_list2[-1]-SA_list2[0]

    try:
        if 'TD3' not in name:
            if 'no_div' not in name:
                writer.writerow([name,corr_SA_P,corr_SA_PI,corr_SA_E,FSA,DSA,ASA,FP,FPI,FE,FPP,corr_SA_PP,FPPI,corr_SA_PPI])
            else:
                writer_no_div.writerow([name,corr_SA_P,corr_SA_PI,corr_SA_E,FSA,DSA,ASA,FP,FPI,FE,FPP,corr_SA_PP,FPPI,corr_SA_PPI])

        else:
            if 'no_div' not in name:
                TD3_data.append([name, corr_SA_P, corr_SA_PI, corr_SA_E, FSA, DSA, ASA, FP, FPI, FE,FPP,corr_SA_PP,FPPI,corr_SA_PPI])
            else:
                TD3_no_div_data.append([name, corr_SA_P, corr_SA_PI, corr_SA_E, FSA, DSA, ASA, FP, FPI, FE,FPP,corr_SA_PP,FPPI,corr_SA_PPI])
    except:
        print("Exception in AdaptiveW_process_SA.py line 152")

        if 'TD3' not in name:
            if 'no_div' not in name:
                writer.writerow([name, corr_SA_P, corr_SA_PI, corr_SA_E, FSA, DSA, ASA, FP, FPI, FE])
            else:
                writer_no_div.writerow(
                    [name, corr_SA_P, corr_SA_PI, corr_SA_E, FSA, DSA, ASA, FP, FPI, FE])

        else:
            if 'no_div' not in name:
                TD3_data.append([name, corr_SA_P, corr_SA_PI, corr_SA_E, FSA, DSA, ASA, FP, FPI, FE])
            else:
                TD3_no_div_data.append(
                    [name, corr_SA_P, corr_SA_PI, corr_SA_E, FSA, DSA, ASA, FP, FPI, FE])

for row in TD3_data:
    writer.writerow(row)

for row in TD3_no_div_data:
    writer_no_div.writerow(row)

process_csv.close()


