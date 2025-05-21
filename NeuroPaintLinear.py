import numpy as np
import torch
import torch.nn as nn
from torch import optim
from utils import *

# K: number of trials
# T: number of timesteps
# N: number of neurons 
# train_data:
#  {eid: {X: (train_X, val_X),
#         y: (train_y, val_y)},
#         setup: {acronym_X: (N_X),
#                 acronym_y: (N_X)}
#   }
#  where train_X, val_X is of shape (K, T, N_X)
#  and train_y, val_y is of shape (K, T, N_y)
class NeuroPaint_Linear():
    def __init__(self, train_data, n_emb, lat_dict, n_comp, dt=0.01):
        self.n_emb = n_emb
        # self.n_lat = To set later
        self.n_comp = n_comp
        self.eids = list(train_data.keys())
        self.dt = dt

        self.areas_X = np.unique(np.concatenate([train_data[eid]['setup']['acronym_X'] for eid in train_data]))
        self.lat_dict = lat_dict
        self.areas_y = list(self.lat_dict.keys())
        self.n_lats = np.array([self.lat_dict[a] for a in self.areas_y])
        self.lat_area = np.concatenate([[a]*self.lat_dict[a] for a in self.areas_y])
        self.N_areas_X = len(self.areas_X)
        self.N_areas_y = len(self.areas_y)

        np.random.seed(0)
        self.N = 0; self.model = {}; 
        for eid in train_data:
            _X = train_data[eid]['X'][0] # (K,T,N_X) 
            _y = train_data[eid]['y'][0] # (K,T,N_Y)
            K, T, NX = _X.shape
            K, T, Ny = _y.shape
            for a in np.unique(train_data[eid]['setup']['copy']['acronym_X']):
                _Na = np.sum(train_data[eid]['setup']['copy']['acronym_X'] == a)
                self.model[f"{eid}_enc_{a}"] = nn.Linear(_Na, n_emb)
            for a in np.unique(train_data[eid]['setup']['copy']['acronym_y']):
                _Na = np.sum(train_data[eid]['setup']['copy']['acronym_y'] == a)
                self.model[f"{eid}_dec_{a}"] = nn.Linear(self.lat_dict[a], _Na)
        U = np.random.normal(size=(n_emb*self.N_areas_X, n_comp))/np.sqrt(n_emb*self.N_areas_X*n_comp)
        V = np.random.normal(size=(np.sum(self.n_lats), n_comp))/np.sqrt(np.sum(self.n_lats)*n_comp)
        self.model['U'] = np2param(U) # shared across sessions
        self.model['V'] = np2param(V) # shared across sessions
        self.model = nn.ParameterDict(self.model)

    def train(self):
        self.model.train()
    def eval(self):
        self.model.eval()
    
    def to(self, device):
        self.model.to(device)

    def state_dict(self):
        checkpoint = {"model": {k: v.cpu() for k, v in self.model.state_dict().items()},
                      "eids": self.eids,
                      "areas_X": self.areas_X,
                      "areas_y": self.areas_y,
                      "n_comp": self.n_comp,
                      "n_emb": self.n_emb,
                      "n_lats": self.n_lats,}
        return checkpoint
    
    def load_state_dict(self, f):
        self.model.load_state_dict(f)

    """
    - input nparray 
    - output tensor
    """
    def predict_y(self, X, y, acronym_X, acronym_y, eid):
        # X: (K, T, Nx)
        # y: (K, T, Ny)
        device = get_device()
        K,T,NX = X.shape
        K,T,Ny = y.shape
        
        X = np2tensor(X).to(device).reshape((K*T, NX))
        y = np2tensor(y).to(device).reshape((K*T, Ny))
    
        emb_X = torch.zeros((K*T, self.n_emb*self.N_areas_X)).to(device)
        y_pred = torch.zeros((K*T, Ny)).to(device)

        for ai, a in enumerate(self.areas_X):
            if a in acronym_X:
                emb_X[:, ai*self.n_emb:(ai+1)*self.n_emb] = self.model[f"{eid}_enc_{a}"](X[:, acronym_X == a])

        lat_y = emb_X @ self.model['U'] @ self.model['V'].T

        for ai, a in enumerate(self.areas_y):
            if a in acronym_y:
                y_pred[:, acronym_y == a] = self.model[f"{eid}_dec_{a}"](lat_y[:, self.lat_area==a])
        return X, y, y_pred, lat_y
    
    """
    - input nparray 
    - output tensor
    """
    def predict_y_fr(self, data, eid, k, exp=False):
        X_np_3d = data[eid]['X'][k]
        y_np_3d = data[eid]['y'][k]
        X, y, ypred, lat_y = self.predict_y(X_np_3d, y_np_3d, data[eid]['setup']['acronym_X'], data[eid]['setup']['acronym_y'], eid)
        X = torch.reshape(X, X_np_3d.shape)
        y = torch.reshape(y, y_np_3d.shape)
        ypred = torch.reshape(ypred, y_np_3d.shape)
        lat_y = torch.reshape(lat_y, (y_np_3d.shape[0], y_np_3d.shape[1], -1))
        if exp:
            ypred = torch.exp(ypred)*self.dt
        return X, y, ypred, lat_y
    
    """
    - input and output nparray by default
    """
    def compute_MSE_RRRGD(self, data, k):
        mses_all = {}
        eids = self.eids
        for eid in eids:
            _, y, ypred, _ = self.predict_y(data[eid]['X'][k], data[eid]['y'][k], 
                                         data[eid]['setup']['acronym_X'], data[eid]['setup']['acronym_y'], 
                                         eid)
            mses_all[eid] = torch.sum((ypred - y) ** 2, axis=(0))
        return mses_all
    
    """
    - input and output nparray by default
    """
    def compute_LL_RRRGD(self, data, k, eps=1e-6):
        mses_all = {}
        eids = self.eids
        for eid in eids:
            _, y, ypred, _ = self.predict_y_fr(data, eid, k, exp=True)
            mses_all[eid] = poisson_nll_loss(ypred, y, eps=eps)
        return mses_all


def mask_neurons(data, N_areas_mask=1):
    print(f"masking {N_areas_mask} areas")
    for eid in data:
        ## randomly mask neurons
        unique_areas = np.unique(data[eid]['setup']['copy']['acronym_y'])
        masked_areas = np.random.choice(unique_areas, size=N_areas_mask, replace=False)
        X_mask = ~np.isin(data[eid]['setup']['copy']['acronym_X'], masked_areas)
        y_mask = np.isin(data[eid]['setup']['copy']['acronym_y'], masked_areas)
        data[eid]['X'] = [data[eid]['setup']['copy']['X'][0][:, :, X_mask],
                            data[eid]['setup']['copy']['X'][1][:, :, X_mask]]
        data[eid]['y'] = [data[eid]['setup']['copy']['y'][0][:, :, y_mask],
                            data[eid]['setup']['copy']['y'][1][:, :, y_mask]]
        data[eid]['setup']['acronym_X'] = data[eid]['setup']['copy']['acronym_X'][X_mask]
        data[eid]['setup']['acronym_y'] = data[eid]['setup']['copy']['acronym_y'][y_mask]
    return data

def unmask_neurons(data):
    for eid in data:
        data[eid]['X'] = data[eid]['setup']['copy']['X']
        data[eid]['y'] = data[eid]['setup']['copy']['y']
        data[eid]['setup']['acronym_X'] = data[eid]['setup']['copy']['acronym_X']
        data[eid]['setup']['acronym_y'] = data[eid]['setup']['copy']['acronym_y']
    return data

"""
train the 
    model: NeuroPaint_Linear
given the
# train_data:
#  {eid: {X: (train_X, val_X),
#         y: (train_y, val_y)},
#         setup: {acronym_X: (N_X),
#                 acronym_y: (N_X),
#                 copy: {
#                     "X": (K, T, N_X),
#                     "y": (K, T, N_y),
#                     "acronym_X": (N_X),
#                     "acronym_y": (N_y)
#                 }
#   }
#  where train_X, val_X is of shape (K, T, N_X)
#  and train_y, val_y is of shape (K, T, N_y)
#  and train_data[eid]['setup']['copy'] is the original data; used for masking and unmasking
"""
def train_model_main(train_data, n_emb, lat_dict, n_comp, dt, model_fname,
                     loss_type="mse",
                     remask = 5,
                     max_iter=20, lr=1):
    def closure():
        optimizer.zero_grad()
        NP_linear.train()
        total_loss = 0.0;
        if loss_type == "mse":
            train_mses_all = NP_linear.compute_MSE_RRRGD(train_data, 0)
        elif loss_type == "poisson":
            train_mses_all = NP_linear.compute_LL_RRRGD(train_data, 0)
        else:
            raise ValueError("loss should be either mse or poisson")
        for eid in train_mses_all:
            total_loss += train_mses_all[eid].sum()
        total_loss.backward()
        closure_calls[0] += 1
        print(f"step: {closure_calls[0]} total_loss: {total_loss.item()}")
        return total_loss
    
    NP_linear = NeuroPaint_Linear(train_data, n_emb, lat_dict, n_comp, dt=dt)
    
    device = get_device()
    NP_linear.to(device)

    best_tot_loss = float("inf"); best_tot_loss_epoch = 0
    for _ in range(remask):
        print(f"epoch {_}: randomly selecting one area to mask")
        # Randomly mask neurons
        mask_neurons(train_data, N_areas_mask=1)

        closure_calls = [0]  
        optimizer = optim.LBFGS(NP_linear.model.parameters(),
                            max_iter=max_iter,
                            lr=lr)
        tot_loss = optimizer.step(closure)

        if tot_loss < best_tot_loss:
            best_tot_loss = tot_loss.item()
            best_tot_loss_epoch = _
            
            if model_fname is not None:
                # Save the best model parameters
                checkpoint = {"model": NP_linear.state_dict(),
                            "optimizer": optimizer.state_dict()}
                torch.save(checkpoint, model_fname)
        
        print(f"best loss: {best_tot_loss} at epoch: {best_tot_loss_epoch}")
            
    unmask_neurons(train_data)

    if model_fname is not None:
        NP_linear.load_state_dict(torch.load(model_fname, weights_only=False)['model']['model'])
         
    return NP_linear