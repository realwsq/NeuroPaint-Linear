import numpy as np
import os, pdb, pickle, h5py
import matplotlib.pyplot as plt
from NeuroPaintLinear import train_model_main
from utils import preprocess_X_and_y

dataset = "syn"; 
smooth_w = 5; dt=0.01; 
halfbin_X = 1 if dataset != "syn" else 5
n_comp = 20 if dataset != "syn" else 50
remask = 50 if dataset != "syn" else 100
loss = "mse" if dataset != "syn" else "poisson"
max_iter = 20 if dataset != "syn" else 40
lr = 1 if dataset != "syn" else 3e-2



train_data = {}
sessneu_area_map = np.load(f"/home/shuqi/EPFL/research/multi_area_inference/{dataset}_data/recorded_area_inds.npy", allow_pickle=True)
sessneu_area_map = sessneu_area_map.item()
data_folder = f"/home/shuqi/EPFL/research/multi_area_inference/{dataset}_data"
data_files = os.listdir(data_folder)
data_files = [f for f in data_files if f.endswith(".h5")]
for data_f in data_files:
    eid = data_f.split(".")[0].split("_")[-1]
    with h5py.File(os.path.join(data_folder, data_f), 'r') as f:
        train_enc_data = f['train_encod_data'][:] # (K, T, N)
        train_rec_data = f['train_recon_data'][:]
        val_enc_data = f['valid_encod_data'][:] # (K, T, N)
        val_rec_data = f['valid_recon_data'][:]
        train_enc_data = np.concatenate((train_enc_data, val_enc_data), axis=0)
        train_rec_data = np.concatenate((train_rec_data, val_rec_data), axis=0)
        test_enc_data = f['test_encod_data'][:] # (K, T, N)
        test_rec_data = f['test_recon_data'][:]
    
    train_data[eid] = {"X": [], "y": [], "setup": {}}
    X, y = preprocess_X_and_y(train_enc_data, train_rec_data, smooth_w, halfbin_X)
    X_test, y_test= preprocess_X_and_y(test_enc_data, test_rec_data, smooth_w, halfbin_X)
    train_data[eid]['X'] = [X, X_test]
    train_data[eid]['y'] = [y, y_test]

    if dataset == "ibl":
        train_data[eid]['setup']['acronym_y'] = sessneu_area_map[int(eid)][0].numpy() # small bug in ibl data
        train_data[eid]['setup']['acronym_X'] = np.repeat(sessneu_area_map[int(eid)][0].numpy(),(1+2*halfbin_X))
    else:
        train_data[eid]['setup']['acronym_y'] = sessneu_area_map[int(eid)].numpy()
        train_data[eid]['setup']['acronym_X'] = np.repeat(sessneu_area_map[int(eid)].numpy(),(1+2*halfbin_X))

    if dataset == "syn":
        lat_dict = {a: 24 for a in range(5)}
    else:
        lat_dict = pickle.load(open(f"./{dataset}_data/pr_max_dict.pkl", "rb"))
        lat_dict = {a: int(lat_dict[a])+10 for a in lat_dict} # as done for NeuroPaint

    # save the original data
    # since masking and unmasking, done in the model, will rewrite the data
    train_data[eid]['setup']['copy'] = {
        "X": train_data[eid]['X'],
        "y": train_data[eid]['y'],
        "acronym_X": train_data[eid]['setup']['acronym_X'],
        "acronym_y": train_data[eid]['setup']['acronym_y'],
    }


res_folder = f"{dataset}_{n_comp}_{smooth_w}_{halfbin_X}_{loss}_{remask}_{max_iter}_{lr}_res"
os.makedirs(res_folder, exist_ok=True)
model = train_model_main(train_data, 
                        48, lat_dict, 
                        n_comp, 
                        dt,
                        f"{res_folder}/model",
                        loss_type=loss,
                        remask=remask,
                        max_iter=max_iter,lr=lr)

preds = {}; r2s = {}
for eid in train_data:
    X_all, y_all, y_all_pred, lat_y = model.predict_y_fr(train_data, eid, 1, exp=loss=='poisson')
    y_all = y_all.cpu().detach().numpy()
    y_all_pred = y_all_pred.cpu().detach().numpy()
    lat_y = lat_y.cpu().detach().numpy()
    preds[eid] = lat_y
    r2 = [np.corrcoef(y_all[:,:,ni].flatten(), y_all_pred[:,:,ni].flatten())[0,1] for ni in range(y_all.shape[2])]
    r2s[eid] = r2
pickle.dump(preds, open(f"{res_folder}/preds_lat.pkl", "wb"))
np.save(f"{res_folder}/preds_lat_area.npy", model.lat_area, )

# ### viz
plt.hist(r2, bins=50)
plt.xlabel("R2")
plt.ylabel("Count")
plt.title("R2 distribution")
plt.savefig(f"{res_folder}/r2.png", dpi=300); plt.close()


X_all, y_all, y_all_pred, lat_y = model.predict_y_fr(train_data, eid, 1, exp=loss=='poisson')
y_all = y_all.cpu().detach().numpy()
y_all_pred = y_all_pred.cpu().detach().numpy()


def _plot(ni):
    plt.figure(figsize=(8,8))
    im_kwargs = dict(aspect='auto', interpolation='nearest', 
                    vmax=np.percentile(y_all_pred[:,:,ni], 95), 
                    vmin=np.percentile(y_all_pred[:,:,ni], 5), 
                    # vmin=0,
                    cmap='binary')
    plt.subplot(2,1,1)
    plt.imshow(y_all[:,:,ni], **im_kwargs)
    plt.subplot(2,1,2)
    plt.imshow(y_all_pred[:,:,ni], **im_kwargs)
    plt.tight_layout()
    plt.savefig(f"{res_folder}/res_{ni}.png", dpi=300); plt.close()

[_plot(ni) for ni in np.where(np.array(r2) > 0.4)[0]]
