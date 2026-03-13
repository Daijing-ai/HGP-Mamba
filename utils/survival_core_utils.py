from argparse import Namespace
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from .regularization import Regularization

from lifelines.utils import concordance_index
import numpy as np
from sksurv.metrics import concordance_index_censored

import torch

from dataset.dataset_generic import save_splits
from utils.survival_utils import *
import wandb

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, warmup=5, patience=15, stop_epoch=20, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = val_loss
        # score = -val_loss

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score <= self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

class EarlyStopping_cindex:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, warmup=5, patience=15, stop_epoch=20, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = val_loss
        # score = -val_loss

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score <= self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

class Monitor_CIndex:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.best_score = None

    def __call__(self, val_cindex, model, ckpt_name:str='checkpoint.pt'):

        score = val_cindex

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        else:
            pass

    def save_checkpoint(self, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), ckpt_name)


def train(datasets: tuple, cur: int, args: Namespace):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from torch.utils.tensorboard.writer import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None
    if (args.k_fold):
        print('K-fold cross validation')
        train_split, val_split = datasets

    else:
        print('\nInit train/val/test splits...', end=' ')
        train_split, val_split, test_split = datasets
        save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
        print('Done!')
        print("Training on {} samples".format(len(train_split)))
        print("Validating on {} samples".format(len(val_split)))
        print("Testing on {} samples".format(len(test_split)))

        print('\nInit loss function...', end=' ')
    
    if args.task_type == 'survival':
        if args.bag_loss == 'ce_surv':
            loss_fn = CrossEntropySurvLoss(alpha=args.alpha_surv)
        elif args.bag_loss == 'nll_surv':
            loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
        elif args.bag_loss == 'cox_surv':
            loss_fn = CoxSurvLoss()
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    
    if args.inst_loss == 'ce':
        instance_loss_fn = torch.nn.CrossEntropyLoss()
    elif args.inst_loss == 'svm':
        instance_loss_fn = torch.nn.MultiMarginLoss()
    else:
        raise NotImplementedError

    print('Done!')
    
    print('\nInit Model...', end=' ')
    args.fusion = None if args.fusion == 'None' else args.fusion

    ### single-modal models
    if args.model_type == 'MEAN_MIL':
        from models.Mean_Max_MIL import MeanMIL
        model = MeanMIL(in_dim=args.in_dim, n_classes=args.n_classes)
    elif args.model_type == 'MAX_MIL':
        from models.Mean_Max_MIL import MaxMIL
        model = MaxMIL(in_dim=args.in_dim, n_classes=args.n_classes)
    elif args.model_type == 'CLAM_SB':
        from models.clam_sb import CLAM_SB_MIL
        model = CLAM_SB_MIL(gate=True, size_arg='small', dropout=args.drop_out, k_sample=args.B, num_classes=args.n_classes, instance_loss_fn=instance_loss_fn, subtyping=args.subtyping, in_dim=args.in_dim, act=nn.ReLU(), instance_eval=True)
    elif args.model_type == 'CLAM_MB':
        from models.clam_mb import CLAM_MB_MIL
        model = CLAM_MB_MIL(gate=True, size_arg='small', dropout=args.drop_out, k_sample=args.B, num_classes=args.n_classes, instance_loss_fn=instance_loss_fn, subtyping=args.subtyping, embed_dim=args.in_dim, act=nn.ReLU(), instance_eval=True)
    elif args.model_type == 'AB_MIL':
        from models.ABMIL import DAttention
        model = DAttention(args.in_dim, args.n_classes, dropout = args.drop_out, act='relu')
    elif args.model_type == 'TRANS_MIL':
        from models.TransMIL import TransMIL
        model = TransMIL(args.in_dim, args.n_classes, dropout = args.drop_out, act='relu')
    elif args.model_type == 'MAMBA_MIL':
        from models.MambaMIL import MambaMIL
        model = MambaMIL(in_dim = args.in_dim, n_classes=args.n_classes, dropout=args.drop_out, act='gelu', layer = args.mambamil_layer, rate = args.mambamil_rate, type = args.mambamil_type)

    ### multi-modal models
    elif args.model_type == 'MCAT':
        from models.MCAT import MCAT_Surv
        model = MCAT_Surv(fusion='concat', n_classes=args.n_classes, dropout=args.drop_out, in_dim=args.in_dim)
    elif args.model_type == 'PORPOISE':
        from models.porpoise import PORPOISE
        model = PORPOISE(fusion='bilinear', n_classes=args.n_classes, dropout=args.drop_out, in_dim=args.in_dim)

    ### our proposed model
    elif args.model_type == 'HGP-Mamba':
        from models.HGPMamba import HGPMamba
        model = HGPMamba(in_dim=args.in_dim, d_state=args.d_state, n_classes=args.n_classes, dropout=args.drop_out, act='relu', mamba_type=args.mambamil_type, fusion=args.fusion)
    else:
        raise NotImplementedError(f'{args.model_type} is not implemented ...')
    
    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(torch.device('cuda'))
    
    print('Done!')
    print_network(model)

    if args.reg_type == 'L1':
        reg_fn = Regularization(model, args.lambda_reg, p=1)
    else:
        reg_fn = None

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, 
        weighted = args.weighted_sample, mode=args.mode, batch_size=args.batch_size)
    val_loader = get_split_loader(val_split, testing = args.testing, mode=args.mode, batch_size=args.batch_size)
    if not args.k_fold:
        test_loader = get_split_loader(test_split, testing = args.testing, mode=args.mode, batch_size=args.batch_size)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        if args.k_fold:
            early_stopping = EarlyStopping_cindex(warmup=0, patience=10, stop_epoch=20, verbose = True)
        else:
            early_stopping = EarlyStopping(warmup=0, patience=20, stop_epoch=40, verbose = True)
    else:
        early_stopping = None

    print('\nSetup Validation C-Index Monitor...', end=' ')
    monitor_cindex = Monitor_CIndex()
    print('Done!')

    for epoch in range(args.max_epochs):
        if args.task_type == 'survival':
            if args.model_type in ['CLAM_SB', 'CLAM_MB']:
                clam_train_loop(epoch=epoch, model=model, loader=train_loader, criterion=loss_fn, optimizer=optimizer, writer=writer, bag_weight=args.bag_weight, gc=args.gc)
                stop = clam_val_loop(cur, epoch=epoch, num_classes=args.n_classes, model=model, loader=val_loader, criterion=loss_fn, bag_weight=args.bag_weight, gc=args.gc, return_KM=False, writer=writer, early_stopping=early_stopping, monitor_cindex=monitor_cindex, results_dir=args.results_dir, k_fold=args.k_fold)
            else:
                train_loop_survival(epoch, model, args.mode, train_loader, optimizer, args.n_classes, writer, loss_fn, reg_fn, args.lambda_reg, args.gc)
                stop = validate_survival(cur, epoch, model, args.mode, val_loader, args.n_classes, early_stopping, monitor_cindex, writer, loss_fn, reg_fn, args.lambda_reg, args.results_dir, args.k_fold)

        if stop:
            break

    print('Done!')
    # if not use early stopping, save the final model
    # torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    _, val_cindex = summary_survival(cur, model, args.mode, val_loader, args.n_classes, args.model_type)
    print('Val c-Index: {:.4f}'.format(val_cindex))
    if (not args.k_fold):
        results_dict, test_cindex = summary_survival(model, test_loader, args.n_classes)
        print('Test c-Index: {:.4f}'.format(test_cindex))
        writer.close()
        return results_dict, test_cindex, val_cindex
    writer.close()
    return val_cindex

def train_loop_survival(epoch, model, mode, loader, optimizer, n_classes, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., gc=16):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    train_loss_surv, train_loss = 0., 0.

    print('\n')
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, batch in enumerate(loader):

        data_WSI, data_mIF, label, event_time, c, _= batch
        # print(f"Processing case_id: {case_id}")
        data_WSI, data_mIF = data_WSI.to(device, non_blocking = True), data_mIF.to(device, non_blocking = True)
        label = label.to(device, non_blocking = True)
        c = c.to(device, non_blocking=True)
        if mode == 'path':
            hazards, S, Y_hat, _, _ = model(data_WSI)
        elif mode == 'mIF':
            hazards, S, Y_hat, _, _ = model(data_mIF)
        elif mode == 'multi-modal':
            hazards, S, Y_hat, _, _ = model(x=data_WSI, y=data_mIF)
        else:
            raise NotImplementedError('{} is not implemented ...'.format(mode))
        
        loss_surv = loss_fn(hazards=hazards, S=S, Y=label, c=c)
        loss_value = loss_surv.item()
        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model)

        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.item()
        all_event_times[batch_idx] = event_time

        train_loss_surv += loss_value
        train_loss += loss_value + loss_reg

        if (batch_idx + 1) % 100 == 0:
            print('batch {}, loss: {:.4f}, loss_surv: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}, bag_size: {}'.format(batch_idx, train_loss, train_loss_surv, label.item(), float(event_time), float(risk), data_WSI.size(0)))
        
        # backward pass
        loss = loss_surv
        loss = loss / gc + loss_reg
        loss.backward()

        if (batch_idx + 1) % gc == 0: 
            optimizer.step()
            optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss_surv /= len(loader)
    train_loss /= len(loader)

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    print('Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss_surv, train_loss, c_index))

    if writer:
        writer.add_scalar('train/loss_surv', train_loss_surv, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index, epoch)

def validate_survival(cur, epoch, model, mode, loader, n_classes, early_stopping=None, monitor_cindex=None, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0, results_dir=None, k_fold=False):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, (data_WSI, data_mIF, label, event_time, c, _) in enumerate(loader):
        data_WSI, data_mIF = data_WSI.to(device), data_mIF.to(device)
        label = label.to(device)
        c = c.to(device)

        with torch.no_grad():
            if mode == 'path':
                hazards, S, Y_hat, _, _ = model(data_WSI)
            elif mode == 'mIF':
                hazards, S, Y_hat, _, _ = model(data_mIF)
            elif mode == 'multi-modal':
                hazards, S, Y_hat, _, _ = model(x=data_WSI, y=data_mIF)
            else:
                raise NotImplementedError('{} is not implemented ...'.format(mode))
        loss_surv = loss_fn(hazards=hazards, S=S, Y=label, c=c, alpha=0)
        loss_value = loss_surv.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model)

        risk = -torch.sum(S, dim=1).cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.cpu().numpy()
        all_event_times[batch_idx] = event_time

        val_loss_surv += loss_value
        val_loss += loss_value + loss_reg

    val_loss_surv /= len(loader)
    val_loss /= len(loader)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    print('Epoch: {}, val_loss_surv: {:.4f}, val_loss: {:.4f}, val_c_index: {:.4f}'.format(epoch, val_loss_surv, val_loss, c_index))
    if writer:
        writer.add_scalar('val/loss_surv', val_loss_surv, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c-index', c_index, epoch)

    if early_stopping:
        assert results_dir
        if k_fold:
            early_stopping(epoch, c_index, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        else:
            early_stopping(epoch, c_index, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def clam_train_loop(epoch, model, loader, criterion, optimizer, writer, bag_weight, gc):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    train_loss_surv, train_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    for batch_idx, batch in enumerate(loader):
        data_he, data_ihc, label, event_time, c, _ = batch
        data_he, data_ihc = data_he.to(device, non_blocking = True), data_ihc.to(device, non_blocking = True)
        label = label.to(device, non_blocking = True)
        c = c.to(device, non_blocking=True)

        output = model(data_he, label=label)
        instance_loss = output['instance_loss']
        hazards = output['hazards']
        S = output['S']

        loss = criterion(hazards=hazards, S=S, Y=label, c=c)
        total_loss = loss * bag_weight + instance_loss * (1-bag_weight)
        loss_value = total_loss.item()

        # train_loss_log += total_loss.item()
        
        risk = -torch.sum(output['S'], dim=1).detach().cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.item()
        all_event_times[batch_idx] = event_time

        train_loss_surv += loss_value
        train_loss += loss_value
        if (batch_idx + 1) % 100 == 0:
            print('batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}, bag_size: {}'.format(batch_idx, loss_value, label.item(), float(event_time), float(risk), data_he.size(0)))

        total_loss = total_loss / gc
        total_loss.backward()

        if (batch_idx + 1) % gc == 0:
        # backward pass
            optimizer.step()
            optimizer.zero_grad()

    train_loss_surv /= len(loader)
    train_loss /= len(loader)
    c_index = concordance_index_censored((1-all_censorships).astype(bool),
                                            all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    print('Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss_surv, train_loss, c_index))

    if writer:
        writer.add_scalar('train/loss_surv', train_loss_surv, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index, epoch)

def clam_val_loop(cur, epoch, num_classes, model, loader, criterion, bag_weight, gc, retrun_WSI_feature = False, return_WSI_attn=False, return_KM=False, writer=None, early_stopping=None, monitor_cindex=None, results_dir=None, k_fold=False):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss_surv, val_loss = 0., 0.
    WSI_features = []
    WSI_attns = []
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    all_risk_scores = np.zeros((len(loader)))
    with torch.no_grad():
        for batch_idx, (data_he, data_ihc, label, event_time, c, _) in enumerate(loader):
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
            instance_loss = output['instance_loss']
            loss = criterion(hazards=output['hazards'], S=output['S'], Y=label, c=c, alpha=0)
            total_loss = loss * bag_weight + instance_loss * (1-bag_weight)
            loss_value = total_loss.item()


            risk = -torch.sum(output['S'], dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.item()
            all_event_times[batch_idx] = event_time

            val_loss_surv += loss_value
            val_loss += loss_value
    
    val_loss_surv /= len(loader)
    val_loss /= len(loader)

    c_index = concordance_index_censored((1-all_censorships).astype(bool),
                                            all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    print('Epoch: {}, val_loss_surv: {:.4f}, val_loss: {:.4f}, val_c_index: {:.4f}'.format(epoch, val_loss_surv, val_loss, c_index))
    if writer:
        writer.add_scalar('val/loss_surv', val_loss_surv, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c-index', c_index, epoch)

    if retrun_WSI_feature:
        WSI_features = torch.cat(WSI_features, dim=0).cpu().numpy()
        return WSI_features
    if return_WSI_attn:
        return WSI_attns
    if return_KM:
        return all_risk_scores, all_censorships, all_event_times, val_loss_surv, c_index

    
    if early_stopping:
        assert results_dir
        if k_fold:
            early_stopping(epoch, c_index, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        else:
            early_stopping(epoch, c_index, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary_survival(cur, model, mode, loader, n_classes, model_type=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0.

    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data_WSI, data_mIF, label, event_time, c, _) in enumerate(loader):
        data_WSI, data_mIF = data_WSI.to(device), data_mIF.to(device)
        label = label.to(device)
        
        slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            if model_type in ['CLAM_SB', 'CLAM_MB']:
                output = model(data_WSI, label=label)
                survival = output['S']
            else:
                if mode == 'path':
                    hazards, survival, Y_hat, _, _ = model(data_WSI)
                elif mode == 'mIF':
                    hazards, survival, Y_hat, group_outputs, _ = model(data_mIF)
                elif mode == 'multi-modal':
                    hazards, survival, Y_hat, _, _ = model(x=data_WSI, y=data_mIF)
                else:
                    raise NotImplementedError('{} is not implemented ...'.format(mode))

        risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
        event_time = np.ndarray.item(event_time)
        c = np.ndarray.item(c.cpu().numpy())
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c
        all_event_times[batch_idx] = event_time
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'risk': risk, 'disc_label': label.item(), 'survival': event_time, 'censorship': c}})

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    return patient_results, c_index