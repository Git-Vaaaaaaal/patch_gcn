from argparse import Namespace
from collections import OrderedDict
import os
import pickle 

from lifelines.utils import concordance_index
import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw

import torch

from datasets.dataset_generic import save_splits
from models.model_set_mil import *
from models.model_graph_mil import *
from utils.utils import *


def to_structured_survival(censorships, event_times):
    """Construit le tableau structure (event, time) attendu par concordance_index_ipcw.
    event = True si l'evenement a ete observe (censorship == 0)."""
    censorships = np.asarray(censorships)
    event_times = np.asarray(event_times, dtype=float)
    event_indicator = (1 - censorships).astype(bool)
    return np.array(list(zip(event_indicator, event_times)), dtype=[('event', bool), ('time', float)])


def compute_cindex_ipcw(train_survival, censorships, event_times, risk_scores):
    """Calcule le c-index IPCW de maniere robuste. Renvoie nan si le c-index n'est
    pas calculable : aucun evenement observe (train ou set evalue), temps hors
    plage de la distribution de censure d'entrainement, etc."""
    survival_test = to_structured_survival(censorships, event_times)
    if not train_survival['event'].any() or not survival_test['event'].any():
        print("Attention: aucun evenement observe (train ou set evalue), c-index = nan.")
        return float('nan')
    # tau : plus grand temps d'evenement observe dans le train, pour eviter
    # l'erreur "time must be smaller than largest observed time point".
    tau = train_survival['time'][train_survival['event']].max()
    try:
        return concordance_index_ipcw(train_survival, survival_test, risk_scores, tau=tau, tied_tol=1e-08)[0]
    except ValueError as e:
        print("Attention: c-index IPCW non calculable ({}), c-index = nan.".format(e))
        return float('nan')


def save_loss_curves(loss_history, out_path):
    """Trace les courbes train/val (loss et c-index) et enregistre un PNG."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib non disponible, courbes non generees.")
        return

    epochs = loss_history['epoch']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, loss_history['train_loss'], label='train_loss')
    ax1.plot(epochs, loss_history['val_loss'], label='val_loss')
    ax1.set_xlabel('epoch'); ax1.set_ylabel('loss'); ax1.set_title('Loss')
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, loss_history['train_c_index'], label='train_c_index')
    ax2.plot(epochs, loss_history['val_c_index'], label='val_c_index')
    ax2.set_xlabel('epoch'); ax2.set_ylabel('c-index'); ax2.set_title('C-index')
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("Courbes sauvegardees dans {}".format(out_path))


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

        score = -val_loss

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
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
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')

    # Distribution de survie du train set : sert de reference (censoring KM) a
    # concordance_index_ipcw, aussi bien pour le c-index train, val que test.
    train_survival = to_structured_survival(
        train_split.slide_data['censorship'].values,
        train_split.slide_data[train_split.label_col].values,
    )
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

    reg_fn = None

    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    if args.model_type == 'deepset':
        model_dict = {'n_classes': args.n_classes}
        model = MIL_Sum_FC_surv(**model_dict)
    elif args.model_type =='amil':
        model_dict = {'n_classes': args.n_classes}
        model = MIL_Attention_FC_surv(**model_dict)
    elif args.model_type == 'mifcn':
        model_dict = {'num_clusters': 10, 'n_classes': args.n_classes}
        model = MIL_Cluster_FC_surv(**model_dict)
    elif args.model_type == 'dgc':
        model_dict = {'edge_agg': args.edge_agg, 'resample': args.resample, 'n_classes': args.n_classes}
        model = DeepGraphConv_Surv(**model_dict)
    elif args.model_type == 'patchgcn':
        model_dict = {'num_layers': args.num_gcn_layers, 'edge_agg': args.edge_agg, 'resample': args.resample, 'n_classes': args.n_classes, 'num_features': args.encoding_size}
        model = PatchGCN_Surv(**model_dict)
    else:
        raise NotImplementedError
    
    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(torch.device('cuda'))
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing,
                                    weighted = args.weighted_sample, mode=args.mode, batch_size=args.batch_size)
    val_loader = get_split_loader(val_split,  testing = args.testing, mode=args.mode, batch_size=args.batch_size)
    test_loader = get_split_loader(test_split,  testing = args.testing, mode=args.mode, batch_size=args.batch_size)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(warmup=0, patience=10, stop_epoch=20, verbose = True)
    else:
        early_stopping = None

    print('\nSetup Validation C-Index Monitor...', end=' ')
    monitor_cindex = Monitor_CIndex()
    print('Done!')

    loss_history = {
        'epoch': [], 'train_loss_surv': [], 'train_loss': [], 'train_c_index': [],
        'val_loss_surv': [], 'val_loss': [], 'val_c_index': [],
    }

    for epoch in range(args.max_epochs):
        if args.task_type == 'survival':
            if args.mode == 'cluster':
                train_loss_surv, train_loss, train_c_index = train_loop_survival_cluster(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn, reg_fn, args.lambda_reg, args.gc, VAE, train_survival=train_survival)
                stop, val_loss_surv, val_loss, val_c_index = validate_survival_cluster(cur, epoch, model, val_loader, args.n_classes, early_stopping, monitor_cindex, writer, loss_fn, reg_fn, args.lambda_reg, args.results_dir, VAE, train_survival=train_survival)
            else:
                train_loss_surv, train_loss, train_c_index = train_loop_survival(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn, reg_fn, args.lambda_reg, args.gc, train_survival=train_survival)
                stop, val_loss_surv, val_loss, val_c_index = validate_survival(cur, epoch, model, val_loader, args.n_classes, early_stopping, monitor_cindex, writer, loss_fn, reg_fn, args.lambda_reg, args.results_dir, train_survival=train_survival)

        loss_history['epoch'].append(epoch)
        loss_history['train_loss_surv'].append(train_loss_surv)
        loss_history['train_loss'].append(train_loss)
        loss_history['train_c_index'].append(train_c_index)
        loss_history['val_loss_surv'].append(val_loss_surv)
        loss_history['val_loss'].append(val_loss)
        loss_history['val_c_index'].append(val_c_index)

        if stop:
            break

    pd.DataFrame(loss_history).to_csv(os.path.join(args.results_dir, 'loss_curve_{}.csv'.format(cur)), index=False)
    save_loss_curves(loss_history, os.path.join(args.results_dir, 'loss_curve_{}.png'.format(cur)))

    torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    results_test_dict, test_cindex = summary_survival(model, test_loader, args.n_classes, train_survival=train_survival)
    _, val_cindex = summary_survival(model, val_loader, args.n_classes, train_survival=train_survival)
    print('Val c-Index: {:.4f}, Test c-Index: {:.4f}'.format(val_cindex, test_cindex))

    with open(os.path.join(args.results_dir, 'test_cindex_{}.txt'.format(cur)), 'w') as f:
        f.write('val_cindex: {:.6f}\n'.format(val_cindex))
        f.write('test_cindex: {:.6f}\n'.format(test_cindex))

    writer.close()
    return results_test_dict, val_cindex, test_cindex


def train_loop_survival(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., gc=16, train_survival=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    train_loss_surv, train_loss = 0., 0.

    print('\n')
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, (data_WSI, label, event_time, c) in enumerate(loader):

        if isinstance(data_WSI, torch_geometric.data.Batch):
            if data_WSI.x.shape[0] > 100_000:
                continue

        data_WSI = data_WSI.to(device)
        label = label.to(device)
        c = c.to(device)

        hazards, S, Y_hat, _, _ = model(x_path=data_WSI) # return hazards, S, Y_hat, A_raw, results_dict
        
        loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = (-torch.sum(S, dim=1).detach().cpu().numpy()).item()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.item()
        all_event_times[batch_idx] = event_time

        train_loss_surv += loss_value
        train_loss += loss_value + loss_reg

        if (batch_idx + 1) % 100 == 0:
            print('batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}, bag_size: {}'.format(batch_idx, loss_value + loss_reg, label.item(), float(event_time), float(risk), data_WSI.size(0)))
        # backward pass
        loss = loss / gc + loss_reg
        loss.backward()

        if (batch_idx + 1) % gc == 0:
            optimizer.step()
            optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss_surv /= len(loader)
    train_loss /= len(loader)

    # c_index = concordance_index(all_event_times, all_risk_scores, event_observed=1-all_censorships)
    c_index = compute_cindex_ipcw(train_survival, all_censorships, all_event_times, all_risk_scores)

    print('Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss_surv, train_loss, c_index))

    if writer:
        writer.add_scalar('train/loss_surv', train_loss_surv, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index, epoch)

    return train_loss_surv, train_loss, c_index


def validate_survival(cur, epoch, model, loader, n_classes, early_stopping=None, monitor_cindex=None, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None, train_survival=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, (data_WSI, label, event_time, c) in enumerate(loader):
        
        if isinstance(data_WSI, torch_geometric.data.Batch):
            if data_WSI.x.shape[0] > 100_000:
                continue

        data_WSI = data_WSI.to(device)
        label = label.to(device)
        c = c.to(device)

        with torch.no_grad():
            hazards, S, Y_hat, _, _ = model(x_path=data_WSI) # return hazards, S, Y_hat, A_raw, results_dict

        loss = loss_fn(hazards=hazards, S=S, Y=label, c=c, alpha=0)
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = -torch.sum(S, dim=1).cpu().numpy()
        all_risk_scores[batch_idx] = risk.item()
        all_censorships[batch_idx] = c.item()
        all_event_times[batch_idx] = event_time

        val_loss_surv += loss_value
        val_loss += loss_value + loss_reg

    val_loss_surv /= len(loader)
    val_loss /= len(loader)
    c_index = compute_cindex_ipcw(train_survival, all_censorships, all_event_times, all_risk_scores)

    if writer:
        writer.add_scalar('val/loss_surv', val_loss_surv, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c-index', c_index, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss_surv, model, ckpt_name=os.path.join(results_dir, "s_{}_minloss_checkpoint.pt".format(cur)))

        if early_stopping.early_stop:
            print("Early stopping")
            return True, val_loss_surv, val_loss, c_index

    return False, val_loss_surv, val_loss, c_index


def summary_survival(model, loader, n_classes, train_survival=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0.

    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data_WSI, label, event_time, c) in enumerate(loader):
        
        if isinstance(data_WSI, torch_geometric.data.Batch):
            if data_WSI.x.shape[0] > 100_000:
                continue

        data_WSI = data_WSI.to(device)
        label = label.to(device)
        
        slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            hazards, survival, Y_hat, _, _ = model(x_path=data_WSI)

        risk = (-torch.sum(survival, dim=1).cpu()).item()
        event_time = event_time.item() if torch.is_tensor(event_time) else float(event_time)
        c = c.item() if torch.is_tensor(c) else float(c)
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c
        all_event_times[batch_idx] = event_time
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'risk': risk, 'disc_label': label.item(), 'survival': event_time, 'censorship': c}})

    c_index = compute_cindex_ipcw(train_survival, all_censorships, all_event_times, all_risk_scores)
    return patient_results, c_index


def train_loop_survival_cluster(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., gc=16, train_survival=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    train_loss_surv, train_loss = 0., 0.

    print('\n')
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, (data_WSI, cluster_id, label, event_time, c) in enumerate(loader):
        data_WSI = data_WSI.to(device)
        label = label.to(device)
        c = c.to(device)

        hazards, S, Y_hat, _, _ = model(x_path=data_WSI, cluster_id=cluster_id) # return hazards, S, Y_hat, A_raw, results_dict
        
        loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = (-torch.sum(S, dim=1).detach().cpu().numpy()).item()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.item()
        all_event_times[batch_idx] = event_time

        train_loss_surv += loss_value
        train_loss += loss_value + loss_reg

        if (batch_idx + 1) % 100 == 0:
            print('batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}, bag_size: {}'.format(batch_idx, loss_value + loss_reg, label.item(), float(event_time), float(risk), data_WSI.size(0)))
        # backward pass
        loss = loss / gc + loss_reg
        loss.backward()

        if (batch_idx + 1) % gc == 0:
            optimizer.step()
            optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss_surv /= len(loader)
    train_loss /= len(loader)

    # c_index = concordance_index(all_event_times, all_risk_scores, event_observed=1-all_censorships)
    c_index = compute_cindex_ipcw(train_survival, all_censorships, all_event_times, all_risk_scores)

    print('Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss_surv, train_loss, c_index))

    if writer:
        writer.add_scalar('train/loss_surv', train_loss_surv, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index, epoch)

    return train_loss_surv, train_loss, c_index


def validate_survival_cluster(cur, epoch, model, loader, n_classes, early_stopping=None, monitor_cindex=None, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None, train_survival=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, (data_WSI, cluster_id, label, event_time, c) in enumerate(loader):
        data_WSI = data_WSI.to(device)
        label = label.to(device)
        c = c.to(device)

        with torch.no_grad():
            hazards, S, Y_hat, _, _ = model(x_path=data_WSI, cluster_id=cluster_id) # return hazards, S, Y_hat, A_raw, results_dict

        loss = loss_fn(hazards=hazards, S=S, Y=label, c=c, alpha=0)
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = -torch.sum(S, dim=1).cpu().numpy()
        all_risk_scores[batch_idx] = risk.item()
        all_censorships[batch_idx] = c.item()
        all_event_times[batch_idx] = event_time

        val_loss_surv += loss_value
        val_loss += loss_value + loss_reg

    val_loss_surv /= len(loader)
    val_loss /= len(loader)
    c_index = compute_cindex_ipcw(train_survival, all_censorships, all_event_times, all_risk_scores)

    if writer:
        writer.add_scalar('val/loss_surv', val_loss_surv, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c-index', c_index, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss_surv, model, ckpt_name=os.path.join(results_dir, "s_{}_minloss_checkpoint.pt".format(cur)))

        if early_stopping.early_stop:
            print("Early stopping")
            return True, val_loss_surv, val_loss, c_index

    return False, val_loss_surv, val_loss, c_index


def summary_survival_cluster(model, loader, n_classes, train_survival=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0.

    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data_WSI, cluster_id, label, event_time, c) in enumerate(loader):
        data_WSI = data_WSI.to(device)
        label = label.to(device)
        
        slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            hazards, survival, Y_hat, _, _ = model(x_path=data_WSI, cluster_id=cluster_id)

        risk = (-torch.sum(survival, dim=1).cpu()).item()
        event_time = event_time.item() if torch.is_tensor(event_time) else float(event_time)
        c = c.item() if torch.is_tensor(c) else float(c)
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c
        all_event_times[batch_idx] = event_time
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'risk': risk, 'disc_label': label.item(), 'survival': event_time, 'censorship': c}})

    c_index = compute_cindex_ipcw(train_survival, all_censorships, all_event_times, all_risk_scores)
    return patient_results, c_index