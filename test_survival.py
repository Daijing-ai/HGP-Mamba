import os
import torch
import argparse
from dataset.dataset_survival import Generic_MIL_Survival_Dataset
from utils.survival_utils import *
import pandas as pd
import numpy as np
import tqdm
from sksurv.metrics import concordance_index_censored
from torch.utils.data import ConcatDataset
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts
import matplotlib.pyplot as plt

def validate_survival(model, mode, loader, n_classes=4, early_stopping=None, monitor_cindex=None, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0, results_dir=None, k_fold=False):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    info = {}

    for batch_idx, (data_WSI, data_mIF, label, event_time, c, slide_id) in enumerate(loader):
        data_WSI, data_mIF = data_WSI.to(device), data_mIF.to(device)
        label = label.to(device)
        c = c.to(device)

        with torch.no_grad():
            if mode == 'path':
                hazards, S, Y_hat, _, _ = model(data_WSI)
            elif mode == 'multi-modal':
                hazards, S, Y_hat, _, _ = model(x=data_WSI, y=data_mIF)
            else:
                raise NotImplementedError('{} is not implemented ...'.format(mode))
        
        risk = -torch.sum(S, dim=1).cpu().numpy()
        info[slide_id[0]] = {
							'risk': risk.item(),
							'fold': args.fold_save
        }
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.cpu().numpy()
        all_event_times[batch_idx] = event_time
        
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    return {c_index: c_index}, info

def clam_val_loop(model, loader, retrun_WSI_feature = False, return_WSI_attn=False, return_KM=False, writer=None, early_stopping=None, monitor_cindex=None, results_dir=None, k_fold=False):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss_surv, val_loss = 0., 0.
    WSI_features = []
    WSI_attns = []
    info = {}
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    all_risk_scores = np.zeros((len(loader)))
    with torch.no_grad():
        for batch_idx, (data_he, data_ihc, label, event_time, c, slide_id) in enumerate(loader):
            data_he, data_ihc = data_he.to(device), data_ihc.to(device)
            label =label.to(device)
            c = c.to(device).float()
            event_time = event_time
            if retrun_WSI_feature:
                WSI_feature = model(data_he,label = label, return_WSI_feature=True)['WSI_feature']
                WSI_features.append(WSI_feature)
                continue
            if return_WSI_attn:
                WSI_attn = model(data_he,label = label, return_WSI_attn=True)['WSI_attn']
                WSI_attns.append(WSI_attn)
                continue
            output = model(data_he, label=label)
            risk = -torch.sum(output['S'], dim=1).detach().cpu().numpy()
            info[slide_id[0]] = {
							'risk': risk.item(),
							'fold': args.fold_save
            }
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.item()
            all_event_times[batch_idx] = event_time

    c_index = concordance_index_censored((1-all_censorships).astype(bool),
                                            all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    return {c_index: c_index}, info


parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default='/home/guestdj/sdc/TCGA_Pancancer/CONCH/TCGA_KIRC', 
                    help='Data directory to WSI features (extracted via CLAM)')
parser.add_argument('--results_dir', default='./results13', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default='/home/guestdj/nvme3n1/DJ/Survival_baseline/splits/TCGA_KIRC', 
                    help='manually specify the set of splits to use')
parser.add_argument('--csv_path', type=str, default='/home/guestdj/nvme3n1/DJ/Survival_baseline/dataset_csv/TCGA_KIRC_processed.csv', help='csv file containing the dataset')
parser.add_argument('--k_fold', type=bool, default=True, help='k fold for cross validation')
parser.add_argument('--k', type=int, default=5, help='number of folds (default: 10)')
parser.add_argument('--model_type', type=str, default='HGIMamba', help='type of model')
parser.add_argument('--mode', type = str, choices=['path', 'mIF', 'multi-modal', 'CMTA', 'ours'], default='multi-modal', help='which modalities to use')
parser.add_argument('--fusion', type=str, choices=['None', 'concat', 'IFBlock'], default='None', help='Type of fusion. (Default: None).')
parser.add_argument('--exp_code', type=str, default='HGIMamba', help='experiment code for saving results')
parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
parser.add_argument('--task', default='TCGA_KIRC', type=str, help='which dataset to use (default: TCGA_KIRC)')

args = parser.parse_args()

args.results_dir = os.path.join(args.results_dir, args.task, f'{args.exp_code}_s{args.seed}')

dataset = Generic_MIL_Survival_Dataset(csv_path = args.csv_path,
                                        mode = args.mode,
                                        apply_sig = False,
                                        data_dir= args.data_root_dir,
                                        shuffle = False, 
                                        seed = 42, 
                                        print_info = True,
                                        patient_strat= False,
                                        n_bins=4,
                                        label_col = 'survival_months',
                                        ignore=[])
args.backbone = 'conch'
args.patch_size = 512
for i in range(args.k):
    train_dataset, val_dataset, test_dataset = dataset.return_splits(args.backbone, args.patch_size, from_id=False, 
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
    combined_val_dataset = ConcatDataset([train_dataset, val_dataset])
    combined_loader = get_split_loader(combined_val_dataset, testing=False, mode=args.mode, batch_size=1)
    print("Data loaders created.")
    args.fold_save = os.path.join(args.results_dir, f'{i}')
    os.makedirs(args.fold_save, exist_ok=True)
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    if args.model_type == 'MEAN_MIL':
        from models.Mean_Max_MIL import MeanMIL
        model = MeanMIL(in_dim=512, n_classes=4)
    elif args.model_type == 'MAX_MIL':
        from models.Mean_Max_MIL import MaxMIL
        model = MaxMIL(in_dim=512, n_classes=4)
    elif args.model_type == 'CLAM_SB':
        from models.clam_sb import CLAM_SB_MIL
        model = CLAM_SB_MIL(gate=True, size_arg='small', dropout=0.25, k_sample=8, num_classes=4, subtyping=True, in_dim=512, act=nn.ReLU(), instance_eval=True)
    elif args.model_type == 'CLAM_MB':
        from models.clam_mb import CLAM_MB_MIL
        model = CLAM_MB_MIL(gate=True, size_arg='small', dropout=0.25, k_sample=8, num_classes=4, subtyping=True, embed_dim=512, act=nn.ReLU(), instance_eval=True)
    elif args.model_type == 'AB_MIL':
        from models.ABMIL import DAttention
        model = DAttention(512, 4, dropout = 0.25, act='relu')
    elif args.model_type == 'TRANS_MIL':
        from models.TransMIL import TransMIL
        model = TransMIL(512, 4, dropout = 0.25, act='relu')
    elif args.model_type == 'MAMBA_MIL_raw':
        from models.MambaMIL import MambaMIL
        model = MambaMIL(in_dim = 512, n_classes=4, dropout=0.25, act='gelu', layer = 2, rate = 10, type = 'Mamba')
    elif args.model_type == 'MAMBA_MIL_bi':
        from models.MambaMIL import MambaMIL
        model = MambaMIL(in_dim = 512, n_classes=4, dropout=0.25, act='gelu', layer = 2, rate = 10, type = 'BiMamba')
    elif args.model_type == 'MAMBA_MIL_sr':
        from models.MambaMIL import MambaMIL
        model = MambaMIL(in_dim = 512, n_classes=4, dropout=0.25, act='gelu', layer = 2, rate = 10, type = 'SRMamba')
    ### our proposed model
    elif args.model_type == 'HGP-Mamba':
        from models.HGPMamba import HGPMamba
        model = HGPMamba(in_dim=512, d_state=16, n_classes=4, dropout=0.25, act='relu', mamba_type='BiMamba', fusion='IFBlock')
    else:
        raise NotImplementedError(f'{args.model_type} is not implemented ...')
    model = model.to(device)
    print(f'Model {args.model_type} initialized.')
    model.load_state_dict(torch.load(os.path.join(args.results_dir, f's_{i}_checkpoint.pt')))
    if args.model_type in ['CLAM_SB', 'CLAM_MB']:
        best_test_result, info = clam_val_loop(model, combined_loader)
    else:
        best_test_result, info = validate_survival(model, args.mode, combined_loader)
    pd.DataFrame.from_dict(info, orient='index').to_excel(os.path.join(args.fold_save, f'{i}_fold_all.xlsx'))
    print(f'Fold {i} completed.')
    df_risk = pd.read_excel(f"{args.results_dir}/{i}/{i}_fold_all.xlsx")
    df_risk.rename(columns={'Unnamed: 0': 'slide_id'}, inplace=True)
    df_risk.reset_index(drop=True, inplace=True)

    df_data = pd.read_csv(args.csv_path)
    df_data['censorship'] = df_data['censorship'].map({0: 1, 1: 0})

    # ============================================================
    # 2. 合并
    # ============================================================
    df = pd.merge(df_risk, df_data, on='slide_id', how='inner')
    df.rename(columns={'censorship': 'event', 'survival_months': 'time'}, inplace=True)

    # ============================================================
    # 3. 按 case_id 聚合 risk（核心）
    # ============================================================
    # df_patient = df.groupby('case_id').agg({
    #     'risk': 'mean',
    #     'time': 'first',
    #     'event': 'first'
    # }).reset_index()

    # ============================================================
    # 4. 根据中位数分组 High / Low
    # ============================================================
    median_risk = df['risk'].median()
    df['risk_group'] = np.where(df['risk'] >= median_risk, 'High', 'Low')

    high_risk_group = df[df['risk_group'] == 'High']
    low_risk_group  = df[df['risk_group'] == 'Low']

    # ============================================================
    # 5. Kaplan-Meier 绘图
    # ============================================================
    kmf = KaplanMeierFitter()

    plt.figure(figsize=(8, 6))

    # Low risk
    kmf.fit(
        low_risk_group['time'],
        event_observed=low_risk_group['event'],
        label=f'Low Risk'
    )
    kmf.plot_survival_function(
        ci_show=True,
        show_censors=True,
        censor_styles={'ms': 5}
    )

    # High risk
    kmf.fit(
        high_risk_group['time'],
        event_observed=high_risk_group['event'],
        label=f'High Risk'
    )
    kmf.plot_survival_function(
        ci_show=True,
        show_censors=True,
        censor_styles={'ms': 5}
    )

    # ============================================================
    # 6. Log-rank test
    # ============================================================
    results = logrank_test(
        high_risk_group['time'], low_risk_group['time'],
        event_observed_A=high_risk_group['event'],
        event_observed_B=low_risk_group['event']
    )

    p_value = results.p_value

    # 显示 p-value
    plt.text(
        0.02, 0.02,
        f'p-value: {p_value:.2e}',
        fontsize=16,
        transform=plt.gca().transAxes,
        horizontalalignment='left',
        verticalalignment='bottom'
    )

    # ============================================================
    # 7. 美化
    # ============================================================
    name = args.task.replace('_', '-')
    plt.title(name, fontsize=24)
    plt.xlabel('Time (months)', fontsize=20)
    plt.ylabel('Survival Probability', fontsize=20)
    # 调整刻度字体和图例字体大小
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=14)   # x/y 主刻度字体
    # 放大图例字体
    legend = ax.legend(fontsize=16)
 
    # 添加 number at risk 表格
    # kmf_low = KaplanMeierFitter().fit(low_risk_group['time'], low_risk_group['event'])
    # kmf_high = KaplanMeierFitter().fit(high_risk_group['time'], high_risk_group['event'])
    # add_at_risk_counts(kmf_low, kmf_high)

    plt.tight_layout()
    plt.savefig(f'{args.results_dir}/{i}/survival_analysis{i}_fold_all.png', dpi=300, bbox_inches='tight')