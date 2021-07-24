from matplotlib import pyplot as plt
import argparse
import pandas as pd
from examples.plotting.commons import *
"""
File to plot the synergy surface area development throughout the learning phase.
Need to update the all_agent_list in common.py in order to plot the results for a folder.
"""

parser = argparse.ArgumentParser()

parser.add_argument('--agentt',
                    type=str,choices=all_agent_list)
parser.add_argument('--fixed_scale',action='store_true')
parser.add_argument('--not_avg_only',action='store_false')

args = parser.parse_args()


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
LGS=20
DPI=350
trans_rate=0.3

avg_only=args.not_avg_only

FSA_color='b'
DSA_color='r'
ASA_color='g'

agentt=args.agentt
fixed_scale=args.fixed_scale

agentt_folder = agentt
top_folder=agentt

cwd=os.getcwd()
cwd_list=cwd.split('/')
while cwd_list[-1]!='synergy_analysis':
    cwd_list.pop()

cwd='/'.join(cwd_list)
path_to_folder=cwd+'/experiments_results/Synergy/all_csv/raw_csv'

print(agentt_folder)
path_to_csv=path_to_folder+'/'+agentt_folder

output_folder=cwd+'/experiments_results/Synergy/SA_evolution/'+agentt
if not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)

no_div_path=output_folder+'/no_div'
div_path=output_folder + '/include_div'
total_dict={}

for csv_ in os.listdir(path_to_csv):

    current_csv = pd.read_csv(path_to_csv + '/' + csv_)

    current_name_list=csv_.split('_')
    current_name_list=current_name_list[0:-2]
    name=''
    for cn in current_name_list:
        name=name+cn+'_'
    name=name[0:-1]

    total_dict[name]={}

    P_list = current_csv['P']
    PI_list = current_csv['PI']
    E_list = current_csv['E']
    SA_list = current_csv['Surface Area']
    Checkpoint_list = current_csv['Checkpoint']
    P_list = np.asarray(P_list)
    PI_list = np.asarray(PI_list)
    E_list = np.asarray(E_list)
    SA_list = np.asarray(SA_list)
    SA_list = np.flip(SA_list,0)
    Checkpoint_list = np.asarray(Checkpoint_list)

    FSA_list=[]
    DSA_list = []
    ASA_list = []

    FSA_list.append(SA_list[0])
    DSA_list.append(0)
    ASA_list.append(0)

    for i in range(1,len(P_list),1):
        tmp_SA_list=SA_list[0:i+1]

        cur_FSA = tmp_SA_list[-1]
        cur_DSA = tmp_SA_list[-1] - tmp_SA_list[0]
        tmp_SA_list2 = sorted(tmp_SA_list)
        cur_ASA=tmp_SA_list2[-1]-tmp_SA_list2[0]

        FSA_list.append(cur_FSA)
        DSA_list.append(cur_DSA)
        ASA_list.append(cur_ASA)

    FSA_list=np.asarray(FSA_list)
    DSA_list=np.asarray(DSA_list)
    ASA_list=np.asarray(ASA_list)

    total_dict[name]['FSA_evolution']=FSA_list
    total_dict[name]['DSA_evolution']=DSA_list
    total_dict[name]['ASA_evolution']=ASA_list

    if not avg_only:
        cur_agent_FSA_fig,cur_FSA_ax=plt.subplots(1, 1)
        cur_agent_DSA_fig,cur_DSA_ax=plt.subplots(1, 1)
        cur_agent_ASA_fig,cur_ASA_ax=plt.subplots(1, 1)
        cur_agent_fig,cur_ax=plt.subplots(1, 1)

        cur_agent_3fig, cur_3ax = plt.subplots(3, 1)
        cur_FSA_ax.plot(FSA_list,c=FSA_color ,label='FSA',linewidth=LW)
        cur_DSA_ax.plot(DSA_list,c=DSA_color ,label='DSA',linewidth=LW)
        cur_ASA_ax.plot(ASA_list,c=ASA_color ,label='ASA',linewidth=LW)
        cur_ax.plot(FSA_list,c=FSA_color ,label='FSA',linewidth=LW)
        cur_ax.plot(DSA_list,c=DSA_color ,label='DSA',linewidth=LW)
        cur_ax.plot(ASA_list,c=ASA_color ,label='ASA',linewidth=LW)

        cur_3ax[0].plot(FSA_list, c=FSA_color,label='FSA',linewidth=LW)
        cur_3ax[1].plot(DSA_list, c=DSA_color,label='DSA',linewidth=LW)
        cur_3ax[2].plot(ASA_list, c=ASA_color,label='ASA',linewidth=LW)

        cur_FSA_ax.set_title('FSA evolution ' + name)
        cur_FSA_ax.set_ylabel('FSA')
        cur_FSA_ax.set_xlabel('Traning checkpoints')
        cur_FSA_ax.legend(loc=2, prop={'size': LGS})

        cur_DSA_ax.set_title('DSA evolution ' + name)
        cur_DSA_ax.set_ylabel('DSA')
        cur_DSA_ax.set_xlabel('Traning checkpoints')
        cur_FSA_ax.legend(loc=2, prop={'size': LGS})

        cur_ASA_ax.set_title('ASA evolution ' + name)
        cur_ASA_ax.set_ylabel('ASA')
        cur_ASA_ax.set_xlabel('Traning checkpoints')
        cur_FSA_ax.legend(loc=2, prop={'size': LGS})

        cur_ax.set_title('SA evolution ' + name)
        cur_ax.set_ylabel('SA')
        cur_ax.set_xlabel('Traning checkpoints')
        cur_ax.legend(loc=2, prop={'size': LGS})

        cur_3ax[0].set_title('SA evolution ' + name)
        cur_3ax[0].set_ylabel('FSA')
        cur_3ax[1].set_ylabel('DSA')
        cur_3ax[2].set_ylabel('ASA')
        cur_3ax[0].legend(loc=2, prop={'size': LGS})
        cur_3ax[1].legend(loc=2, prop={'size': LGS})
        cur_3ax[2].legend(loc=2, prop={'size': LGS})
        cur_3ax[2].set_xlabel('Traning checkpoints')

        if fixed_scale:
            cur_FSA_ax.set_ylim([3, 8.5])
            cur_DSA_ax.set_ylim([3, 8.5])
            cur_ASA_ax.set_ylim([3, 8.5])

        plt.tight_layout()
        if 'no_div' not in name:

            if not os.path.exists(no_div_path):
                os.makedirs(no_div_path, exist_ok=True)
            cur_agent_FSA_fig.savefig(no_div_path+'/FSA_evolution_'+name+'_no_div.png')
            cur_agent_DSA_fig.savefig(no_div_path+'/DSA_evolution_'+name+'_no_div.png')
            cur_agent_ASA_fig.savefig(no_div_path+'/ASA_evolution_'+name+'_no_div.png')
            cur_agent_fig.savefig(no_div_path+'/All_evolution_'+name+'_no_div.png')

            cur_agent_3fig.savefig(no_div_path+ '/3_evolution_' + name + '_no_div.png')

        else:

            if not os.path.exists(div_path):
                os.makedirs(div_path, exist_ok=True)
            cur_agent_FSA_fig.savefig(div_path + '/FSA_evolution_' + name + '.png')
            cur_agent_DSA_fig.savefig(div_path+ '/DSA_evolution_' + name + '.png')
            cur_agent_ASA_fig.savefig(div_path+ '/ASA_evolution_' + name + '.png')
            cur_agent_fig.savefig(div_path + '/All_evolution_' + name + '.png')

            cur_agent_3fig.savefig(div_path+ '/3_evolution_' + name + '.png')


    plt.close('all')

avg_FSA_SAC=[]
avg_FSA_TD3=[]

avg_DSA_SAC=[]
avg_DSA_TD3=[]

avg_ASA_SAC=[]
avg_ASA_TD3=[]


avg_FSA_SAC_no_div=[]
avg_FSA_TD3_no_div=[]

avg_DSA_SAC_no_div=[]
avg_DSA_TD3_no_div=[]

avg_ASA_SAC_no_div=[]
avg_ASA_TD3_no_div=[]
have_td3=False
have_sac=False
max_div_len=0
for indn,name in enumerate(total_dict.keys()):
    if 'no_div' in name:
        if 'TD3' in name:

            avg_FSA_TD3_no_div.append( total_dict[name]['FSA_evolution'])
            avg_DSA_TD3_no_div.append(total_dict[name]['DSA_evolution'])
            avg_ASA_TD3_no_div.append(total_dict[name]['ASA_evolution'])
            have_td3=True
        else:
            have_sac=True
            avg_FSA_SAC_no_div.append(total_dict[name]['FSA_evolution'])
            avg_DSA_SAC_no_div.append(total_dict[name]['DSA_evolution'])
            avg_ASA_SAC_no_div.append(total_dict[name]['ASA_evolution'])
    else:
        if 'TD3' in name:
            avg_FSA_TD3.append(total_dict[name]['FSA_evolution'])
            avg_DSA_TD3.append(total_dict[name]['DSA_evolution'])
            avg_ASA_TD3.append(total_dict[name]['ASA_evolution'])
            have_td3=True

        else:
            have_sac = True
            avg_FSA_SAC.append(total_dict[name]['FSA_evolution'])
            avg_DSA_SAC.append(total_dict[name]['DSA_evolution'])
            avg_ASA_SAC.append(total_dict[name]['ASA_evolution'])


if have_sac:
    avg_FSA_SAC=np.asarray(avg_FSA_SAC)
    avg_DSA_SAC=np.asarray(avg_DSA_SAC)
    avg_ASA_SAC=np.asarray(avg_ASA_SAC)
    FSA_mean=np.mean(avg_FSA_SAC,axis=0)
    DSA_mean=np.mean(avg_DSA_SAC,axis=0)
    ASA_mean=np.mean(avg_ASA_SAC,axis=0)
    FSA_std=np.std(avg_FSA_SAC,axis=0)
    DSA_std=np.std(avg_DSA_SAC,axis=0)
    ASA_std=np.std(avg_ASA_SAC,axis=0)

    avg_FSA_fig,avg_FSA_ax=plt.subplots(1, 1)
    avg_DSA_fig,avg_DSA_ax=plt.subplots(1, 1)
    avg_ASA_fig,avg_ASA_ax=plt.subplots(1, 1)

    avg_FSA_ax.plot( FSA_mean, color=FSA_color,
                           label='FSA', linewidth=LW)
    avg_FSA_ax.fill_between(range( len(FSA_mean)), FSA_mean + FSA_std,
                            FSA_mean - FSA_std, facecolor=FSA_color, alpha=trans_rate)

    #agent_type=name.replace('_'+name.split('_')[-1],'')
    agent_type=name.split('_')[0]+'_'+name.split('_')[1]

    avg_FSA_ax.set_title('FSA evolution '+agent_type+' SAC' )
    avg_FSA_ax.set_ylabel('FSA')
    avg_FSA_ax.set_xlabel('Traning checkpoints')
    avg_FSA_ax.legend(loc=2, prop={'size': LGS})

    if 'HC' in name:
        avg_FSA_ax.set_ylim([0, 8.5])
    elif 'VA' in name:
        avg_FSA_ax.set_ylim([0, 9])

    if not os.path.exists(div_path+'/avg'):
        os.makedirs(div_path+'/avg', exist_ok=True)
    avg_FSA_fig.savefig(div_path+'/avg' + '/avg_FSA_evolution_' + name + '.png')


    avg_DSA_ax.plot( DSA_mean, color=DSA_color,
                           label='DSA', linewidth=LW)
    avg_DSA_ax.fill_between(range( len(DSA_mean)), DSA_mean + DSA_std,
                            DSA_mean - DSA_std, facecolor=DSA_color, alpha=trans_rate)

    avg_DSA_ax.set_title('DSA evolution '+agent_type+' SAC')
    avg_DSA_ax.set_ylabel('DSA')
    avg_DSA_ax.set_xlabel('Traning checkpoints')
    avg_DSA_ax.legend(loc=2, prop={'size': LGS})
    if 'HC' in name:
        avg_DSA_ax.set_ylim([-1, 5])
    elif 'VA' in name:
        avg_DSA_ax.set_ylim([-3, 6])

    avg_DSA_fig.savefig(div_path+'/avg' + '/avg_DSA_evolution_' + name + '.png')

    avg_ASA_ax.plot( ASA_mean, color=ASA_color,
                           label='ASA', linewidth=LW)
    avg_ASA_ax.fill_between(range( len(ASA_mean)), ASA_mean + ASA_std,
                            ASA_mean - ASA_std, facecolor=ASA_color, alpha=trans_rate)

    avg_ASA_ax.set_title('ASA evolution '+agent_type+' SAC')
    avg_ASA_ax.set_ylabel('ASA')
    avg_ASA_ax.set_xlabel('Traning checkpoints')
    avg_ASA_ax.legend(loc=2, prop={'size': LGS})
    if 'HC' in name:
        avg_ASA_ax.set_ylim([0, 5])
    elif 'VA' in name:
        avg_ASA_ax.set_ylim([-0.25, 6])

    avg_ASA_fig.savefig(div_path+'/avg' + '/avg_ASA_evolution_' + name + '.png')

plt.close('all')

if have_td3:
    avg_FSA_TD3=np.asarray(avg_FSA_TD3)
    avg_DSA_TD3=np.asarray(avg_DSA_TD3)
    avg_ASA_TD3=np.asarray(avg_ASA_TD3)
    FSA_mean_TD3=np.mean(avg_FSA_TD3,axis=0)
    DSA_mean_TD3=np.mean(avg_DSA_TD3,axis=0)
    ASA_mean_TD3=np.mean(avg_ASA_TD3,axis=0)
    FSA_std_TD3=np.std(avg_FSA_TD3,axis=0)
    DSA_std_TD3=np.std(avg_DSA_TD3,axis=0)
    ASA_std_TD3=np.std(avg_ASA_TD3,axis=0)

    avg_FSA_TD3_fig,avg_FSA_TD3_ax=plt.subplots(1, 1)
    avg_DSA_TD3_fig,avg_DSA_TD3_ax=plt.subplots(1, 1)
    avg_ASA_TD3_fig,avg_ASA_TD3_ax=plt.subplots(1, 1)
    cur_agent_TD3_fig,cur_TD3_ax=plt.subplots(1, 1)

    avg_FSA_fig,avg_FSA_ax=plt.subplots(1, 1)
    avg_DSA_fig,avg_DSA_ax=plt.subplots(1, 1)
    avg_ASA_fig,avg_ASA_ax=plt.subplots(1, 1)

    avg_FSA_ax.plot( FSA_mean_TD3, color=FSA_color,
                           label='FSA', linewidth=LW)
    avg_FSA_ax.fill_between(range( len(FSA_mean_TD3)), FSA_mean_TD3 + FSA_std_TD3,
                            FSA_mean_TD3 - FSA_std_TD3, facecolor=FSA_color, alpha=trans_rate)

    agent_type=name.replace('_'+name.split('_')[-1],'')
    agent_type=name.split('_')[0]+'_'+name.split('_')[1]

    avg_FSA_ax.set_title('FSA evolution '+agent_type+' TD3' )
    avg_FSA_ax.set_ylabel('FSA')
    avg_FSA_ax.set_xlabel('Traning checkpoints')
    avg_FSA_ax.legend(loc=2, prop={'size': LGS})

    if 'HC' in name:
        avg_FSA_ax.set_ylim([0, 8.5])
    elif 'VA' in name:
        avg_FSA_ax.set_ylim([0, 9])
    if not os.path.exists(div_path+'/avg'):
        os.makedirs(div_path+'/avg', exist_ok=True)
    avg_FSA_fig.savefig(div_path+'/avg' + '/avg_FSA_evolution_' + name + '_TD3.png')


    avg_DSA_ax.plot( DSA_mean_TD3, color=DSA_color,
                           label='DSA', linewidth=LW)
    avg_DSA_ax.fill_between(range( len(DSA_mean_TD3)), DSA_mean_TD3 + DSA_std_TD3,
                            DSA_mean_TD3 - DSA_std_TD3, facecolor=DSA_color, alpha=trans_rate)

    avg_DSA_ax.set_title('DSA evolution '+agent_type+' TD3')
    avg_DSA_ax.set_ylabel('DSA')
    avg_DSA_ax.set_xlabel('Traning checkpoints')
    avg_DSA_ax.legend(loc=2, prop={'size': LGS})
    if 'HC' in name:
        avg_DSA_ax.set_ylim([-1, 5])
    elif 'VA' in name:
        avg_DSA_ax.set_ylim([-3, 6])

    avg_DSA_fig.savefig(div_path+'/avg' + '/avg_DSA_evolution_' + name + '_TD3.png')

    avg_ASA_ax.plot( ASA_mean_TD3, color=ASA_color,
                           label='ASA', linewidth=LW)
    avg_ASA_ax.fill_between(range( len(ASA_mean_TD3)), ASA_mean_TD3 + ASA_std_TD3,
                            ASA_mean_TD3 - ASA_std_TD3, facecolor=ASA_color, alpha=trans_rate)

    avg_ASA_ax.set_title('ASA evolution '+agent_type+' TD3')
    avg_ASA_ax.set_ylabel('ASA')
    avg_ASA_ax.set_xlabel('Traning checkpoints')
    avg_ASA_ax.legend(loc=2, prop={'size': LGS})
    if 'HC' in name:
        avg_ASA_ax.set_ylim([0, 5])
    elif 'VA' in name:
        avg_ASA_ax.set_ylim([-0.25, 6])

    avg_ASA_fig.savefig(div_path+'/avg' + '/avg_ASA_evolution_' + name + '_TD3.png')

plt.close('all')
