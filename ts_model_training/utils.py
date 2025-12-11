import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import os
import random
import pickle
import torch.backends.cudnn as cudnn
from transformers import set_seed as transformers_set_seed 
from pathlib import Path
from config.constants import PROJECT_ROOT

# UTILS for PREPROCESSING

def compute_lab_frequency(data):
    return (data.groupby(["hadm_id", "itemid"])
                .size()
                .unstack()
                .mean(axis=0)
                .sort_values(ascending=False)
                .reset_index()
                .rename(columns={0: "num_ts"}))


def remove_features_not_in_train(data, train_ids, logger=None):
    
    train_variables = np.array(data.loc[data.hadm_id.isin(train_ids), 'itemid'].unique(), dtype=str)
    all_variables = np.array(data['itemid'].unique(), dtype=str)
    delete_variables = np.setdiff1d(all_variables, train_variables)

    if logger is not None:
        logger.write('\nRemoving variables not in training set: ' + str(delete_variables))

    return data.loc[data['itemid'].isin(train_variables)]


def ids_in_data(data, ids, logger=None):

    # extract ids from dictionary, defaulting to empty arrays if None
    train_ids = np.array(ids["train"] if ids["train"] is not None else [], dtype=int)
    val_ids   = np.array(ids["val"] if ids["val"] is not None else [], dtype=int)
    test_ids  = np.array(ids["test"] if ids["test"] is not None else [], dtype=int)

    # get ids in current data
    all_ids = np.unique(np.concatenate([train_ids, val_ids, test_ids]))
    curr_ids = data.hadm_id.unique()
    missing_ids = np.setdiff1d(all_ids, curr_ids)

    # for each element that is not na, get intersections 
    train_ids = np.intersect1d(train_ids, curr_ids)
    val_ids   = np.intersect1d(val_ids, curr_ids)
    test_ids  = np.intersect1d(test_ids, curr_ids)

    # concatenate not na
    sup_ts_ids = np.concatenate((train_ids, val_ids, test_ids))

    if logger is not None:
        logger.write(f"\nTotal ids removed (not in data): {len(missing_ids)}")
        logger.write(f"\n# train, val, test TS FILTERED: {len(train_ids)}, {len(val_ids)}, {len(test_ids)}")

    # keep only relevant ids in data
    data = data.loc[data.hadm_id.isin(sup_ts_ids)]
    
    # update ids dictionary
    ids["train"] = train_ids
    ids["val"]   = val_ids
    ids["test"]  = test_ids
    ids["sup_ids"] = sup_ts_ids

    return data, ids


def set_splits(ids):
    ts_id_to_ind = {ts_id: i for i, ts_id in enumerate(ids["sup_ids"])}
    splits = {
        'train': [ts_id_to_ind[i] for i in ids["train"]],
        'val':   [ts_id_to_ind[i] for i in ids["val"]],
        'test':  [ts_id_to_ind[i] for i in ids["test"]],
    }
    splits['eval_train'] = splits['train'][:1000]
    return ts_id_to_ind, splits


def compute_class_weight(pos_freq, pos_class_weight=None, stratify_batch=0, train_batch_size=32):
    # if class weight is given - use predefined value
    if pos_class_weight != 0:
        return pos_class_weight
    # if stratified batch is applied - adapt weight accordingly
    if stratify_batch > 0:
        pos_freq = stratify_batch / train_batch_size
    # compute class weight
    return (1 - pos_freq) / pos_freq


def safe_pos_freq(splits):
    if len(splits) == 0:
        return np.nan 
    return splits.sum() / len(splits)


### UTILS for MODEL-SPECIFIC DATA PREPARATION

def discrete_tensors(data, N, T, V):# data has columns int (time), itemid (variable), value
    # define last, obs, delta, counts
    last = np.zeros((N,T,V))
    sums = np.zeros((N,T,V))
    obs = np.zeros((N,T,V))
    counts = np.zeros((N,T,V))
    avgs = np.zeros((N,T,V))
    for row in data.itertuples():
        last[row.ts_ind, row.int, row.var_ind] = row.value
        sums[row.ts_ind, row.int, row.var_ind] += row.value
        obs[row.ts_ind, row.int, row.var_ind] = 1 # dataset M
        counts[row.ts_ind, row.int, row.var_ind] += 1
    np.divide(sums, counts, out=avgs, where=counts > 0)

    # compute delta
    delta = np.zeros((N,T,V))
    delta[:,0,:] = 1-obs[:,0,:]
    for t in range(1,T):
        delta[:,t,:] = obs[:,t,:]*0 + (1-obs[:,t,:])*(1+delta[:,t-1,:])
    delta = delta/T
    return last, avgs, obs, delta, sums, counts 

def fill_impute(values, obs):  
    # values and obs are numpy arrays of shape (N, T, V)
    N, T, V = values.shape
    values_imputed = values.copy()
    values_imputed[obs == 0] = np.nan 
    # to dataframe
    values_reshaped = values_imputed.transpose(0, 2, 1).reshape(N * V, T)
    df = pd.DataFrame(values_reshaped)
    # forward/backward will
    df = df.ffill(axis=1).bfill(axis=1)
    df = df.fillna(0)
    return df.values.reshape(N, V, T).transpose(0, 2, 1)

def fill_mean(values, obs, train_ind):
    _, _, V = values.shape
    means = (values[train_ind] * obs[train_ind]).sum(axis=(0, 1)) \
            / obs[train_ind].sum(axis=(0, 1))
    return values * obs + (1 - obs) * means.reshape((1, 1, V))

def compute_means_stds_df(data, train_ids):
    means_stds = data.loc[data.ts_ind.isin(train_ids)].groupby('itemid').agg({'value':['mean', 'std']})
    means_stds.columns = means_stds.columns.droplevel(0)
    means_stds.loc[means_stds['std'] == 0, 'std'] = 1 
    return means_stds

def compute_deltas(curr_times, curr_mask, max_time):
    T = len(curr_times)
    curr_delta = np.zeros_like(curr_mask, dtype=float)
    
    for t in range(1, T):
        curr_delta[t, :] = (curr_times[t] - curr_times[t - 1]) + \
                           (1 - curr_mask[t - 1]) * curr_delta[t - 1, :]
    
    return curr_delta / max_time

def compute_holdout(curr_mask):
    
    hmask = np.copy(curr_mask)
    V = curr_mask.shape[1]

    for j in range(V):
        obs_time_indices = np.argwhere(curr_mask[:, j]).reshape(-1)
        num_to_mask = int(0.2 * len(obs_time_indices))
        
        if num_to_mask > 0:
            to_mask = np.random.choice(obs_time_indices, num_to_mask, replace=False)
            hmask[to_mask, j] = 0

    return hmask    

### UTILS for TRAINING

def set_all_seeds(seed: int, fast: bool):

    random.seed(seed)                       # Python random seed
    np.random.seed(seed)                    # NumPy random seed
    torch.manual_seed(seed)                 # PyTorch CPU seed

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)    # PyTorch CUDA seed

    transformers_set_seed(seed)             # Transformer
    if fast:
        cudnn.deterministic = False
        cudnn.benchmark = True
    else:
        cudnn.deterministic = True              # for deterministic behavior  
        cudnn.benchmark = False
    #torch.use_deterministic_algorithms(True)


def focal_loss_with_logits(logits, labels, gamma=2, alpha=0.95, pos_weight=None):

    BCE_loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight, reduction='none')
    probs = torch.sigmoid(logits)
    pt = probs * labels + (1 - probs) * (1 - labels)  # pt = p if y == 1 else 1 - p
    focal_term = (1 - pt) ** gamma

    if alpha is not None:
        alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
        BCE_loss *= alpha_t

    loss = focal_term * BCE_loss
    return loss.mean()

### UTILS for LOADING and SAVING

def set_all_paths(args, out=True):
    dataset = args.dataset
    cohort = args.cohort
    days = args.days_before_discharge

    # set paths for dataset, folds, features, and cohorts
    input_features = f"{cohort}_admissions_labs_{days}_days"
    saved_data_path = Path(PROJECT_ROOT) / dataset / "saved_data"

    feature_path = saved_data_path / "top_features" # / cohort
    data_path = saved_data_path / "processed_admission_features_for_ts" / cohort /f"{input_features}_to_ts.csv.gz"
    cohort_path = saved_data_path / "cohorts" / f"{cohort}.csv.gz"

    if args.split_seed is None:
        CV_folds_path = saved_data_path / "folds" / cohort
    else:
        CV_folds_path = saved_data_path / "multi_seed_training" / "folds" / cohort / f"seed_{args.split_seed}"
        
    # save this as dict
    path_dict = {
        "saved_data_path": saved_data_path,
        "CV_folds_path": CV_folds_path,
        "feature_path": feature_path,
        "data_path": data_path,
        "cohort_path": cohort_path
    }

    if out:
        path_dict["output_path"] = get_output_dir(args)

    return path_dict

    
def get_output_dir(args):

    if args.split_seed is None:
        output_dir = os.path.join(
            args.dataset, "saved_data", "results",
            args.cohort, "time_series",
            args.train_mode,
            args.model_type,
            args.prefix,
            "fold_" + str(args.fold),
            "grid_" + str(args.grid)
        )
    else:
        output_dir = os.path.join(
            args.dataset, "saved_data", "multi_seed_training", "results",
            args.cohort, "time_series",
            args.train_mode,
            args.model_type,
            f"seed_{args.split_seed}",
            args.prefix,
            "fold_" + str(args.fold),
            "grid_" + str(args.grid)
        )        
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def load_fold_file(args):
    fold_file = args.paths["CV_folds_path"] / f"fold_{args.fold}.pkl"
    with open(fold_file, "rb") as f:
        train_ids, val_ids, test_ids = pickle.load(f)
    return train_ids[:, 1], val_ids[:, 1], test_ids[:, 1]


def format_dict(d, precision=4):
    #Return a copy of dictionary d with all float values rounded to given precision.
    return {k: round(v, precision) if isinstance(v, float) else v for k, v in d.items()}

def count_parameters(logger, model):

    info_dict = {}

    # Total and trainable parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = 100 * trainable / total if total > 0 else 0.0

    info_dict['total_parameters'] = total
    info_dict['trainable_parameters'] = trainable

    logger.write('\nModel details:')
    logger.write(f'# parameters: {total}')
    logger.write(f'# trainable parameters: {trainable}, {trainable_pct:.2f}%')

    # Parameters by dtype
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = str(p.dtype)
        dtypes[dtype] = dtypes.get(dtype, 0) + p.numel()

    info_dict['parameters_by_dtype'] = dtypes

    logger.write('# parameters by dtype:')
    for k, v in dtypes.items():
        pct = 100 * v / total if total > 0 else 0.0
        logger.write(f'{k}: {v}, {pct:.2f}%')

    return info_dict