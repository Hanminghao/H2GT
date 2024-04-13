import os
import pickle
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt

# 假设 model_lis 和 dataset_lis 是你的模型和数据集列表
dataset_lis = ['BRCA', 'GBMLGG', 'BLCA']
# dataset_lis = ['BLCA']
model_lis = ['AMIL', 'PatchGCN', 'HEAT', 'H2GT']
# model_lis = ['HEAT', 'H2GT']
fig, axs = plt.subplots(4, 3, figsize=(15, 20))  # 创建一个2行3列的大图
# fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # 创建一个2行3列的大图

for i, model in enumerate(model_lis):
    for j, dataset_name in enumerate(dataset_lis):
        exp_path = 'plots/{}/{}/'.format(dataset_name, model)
        risk_scores = {}
        for fold in range(1, 6):
            file_path = os.path.join(exp_path, str(fold), 'risk_scores.pkl')
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
            risk_scores.update(data)
        
        risk = list(risk_scores.keys())
        if model in ['HEAT', 'H2GT']:
            risk = sorted(risk)
        else:
            risk = sorted(risk, reverse=True)
        low_risk = risk[len(risk)//2:]
        low_time = [risk_scores[i][0] for i in low_risk]
        low_censor_bool = [True if risk_scores[i][1] == 1 else False for i in low_risk]
        high_risk = risk[:len(risk)//2]
        high_time = [risk_scores[i][0] for i in high_risk]
        high_censor_bool = [True if risk_scores[i][1] == 1 else False for i in high_risk]
        
        kmf_high = KaplanMeierFitter()  
        kmf_high.fit(durations=high_time, event_observed=high_censor_bool, label='Low Risk')
        kmf_low = KaplanMeierFitter()
        kmf_low.fit(durations=low_time, event_observed=low_censor_bool, label='High Risk')
        
        ax = kmf_high.plot(ylabel='Survival Probability', ci_show='log', show_censors=True, legend=True, ax=axs[i, j])
        kmf_low.plot(ax=ax, ci_show='log', show_censors=True, legend=True)
        
        p_value = logrank_test(high_time, low_time, high_censor_bool, low_censor_bool).p_value

        axs[i, j].set_title('TCGA-{}\n (P-value={:.2e})'.format(dataset_name, p_value))
        axs[i, j].set_ylabel('Cumulative proportions surviving', fontsize=12)
        axs[i, j].set_xlabel('Timeline (months)')

# for i, label in enumerate(['HEAT', 'Ours']):
for i, label in enumerate(['AMIL', 'PatchGCN', 'HEAT', 'H2GT']):
    axs[i, 0].annotate(label, (0, 0.5), xytext=(-axs[i, 0].yaxis.labelpad - 5, 0),
                        xycoords=axs[i, 0].yaxis.label, textcoords='offset points',
                        fontsize=20, ha='right', va='center', rotation='vertical')
for j, label in enumerate(dataset_lis):
    axs[0, j].annotate(label, (0.5, 1.05), xytext=(0, axs[0, j].yaxis.labelpad + 20),
                        xycoords='axes fraction', textcoords='offset points',
                        fontsize=20, ha='center', va='bottom')

plt.tight_layout()  # 调整子图之间的间距
# plt.savefig(os.path.join('./fig', 'KM.svg'), format='svg')
plt.savefig(os.path.join('./fig', 'all_KM.png'), format='png')
plt.savefig(os.path.join('./fig', 'all_KM.pdf'), format='pdf')
