import ast
import math
import pickle
import random
import json
import warnings
from collections import Counter, deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import median_abs_deviation
from sklearn.metrics import auc

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score, classification_report,
    precision_recall_curve
)
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

from iterstrat.ml_stratifiers import (
    MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
)




def process_GO_data(file_path, embeddings):
    """
    Loads and filters GO annotation data, aligning it with a corresponding embeddings array.
    """

    GO_df = pd.read_csv(file_path, sep='\t')
    GO_df['Raw propagated GO terms'] = GO_df['Raw propagated GO terms'].apply(ast.literal_eval)
    GO_df = GO_df.reset_index(drop=True)
    indices_to_remove = GO_df[GO_df['Raw propagated GO terms'].apply(lambda x: len(x) == 0)].index
    embeddings = np.delete(embeddings, indices_to_remove, axis=0)
    GO_df = GO_df[GO_df['Raw propagated GO terms'].apply(lambda x: len(x) != 0)]
    GO_df = GO_df.reset_index(drop=True)
    assert len(embeddings) == len(GO_df), "Embeddings and GO data lengths do not match."
    GO_list = GO_df['Raw propagated GO terms'].tolist()
    GO_annotated = GO_df['Raw GO terms'].tolist()
    
    return GO_df, embeddings, GO_list, GO_annotated

class PFP(nn.Module):
    """
    Flexible feedforward neural network model for multilabel GO term prediction.
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.5):
        super(PFP, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LeakyReLU())
        
        if dims[-1] != 2:
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def mc_dropout_inference(model, data_loader, beta=10, device='cpu'):
    """
    Performs Monte Carlo Dropout inference to estimate uncertainty in predictions.
    """
    model.train()  
    dropout_probs = []
    with torch.no_grad():
        for x_batch in data_loader:
            x_batch = x_batch[0].to(device)
            batch_probs = []
            for _ in range(beta):
                model.train()  
                logits = model(x_batch)  
                softmax_probs = torch.sigmoid(logits).cpu().numpy()
                batch_probs.append(softmax_probs)
            dropout_probs.append(np.stack(batch_probs, axis=0))
    return np.concatenate(dropout_probs, axis=1)

def store_predicted_terms(median_probs, mad_probs, confidence_thresholds, function_mlb):
    """
    Generates GO term predictions based on MC dropout-derived uncertainty bounds.
    """
    predicted_terms_dict = {}

    for threshold in confidence_thresholds:
        rounded_threshold = np.round(threshold, 2)
        upper_bounds = np.clip(median_probs + mad_probs, 0, 1)
        lower_bounds = np.clip(median_probs - mad_probs, 0, 1)
        binary_predictions = ((lower_bounds >= threshold) & (upper_bounds >= threshold)).astype(int)
        predicted_go_terms = function_mlb.inverse_transform(binary_predictions)
        predicted_terms_dict[rounded_threshold] = predicted_go_terms
    
    return predicted_terms_dict


def calculate_sharpness_coverage_and_fdr(predicted_terms_dict, y_true, confidence_thresholds):
    """
    Computes evaluation metrics (sharpness, coverage, FDR, precision, recall) for each threshold.
    """

    sharpness_dict = {}
    coverage_dict = {}
    fdr_dict = {}
    precision_micro_dict = {}
    precision_macro_dict = {}
    recall_micro_dict = {}
    recall_macro_dict = {}

    for threshold in confidence_thresholds:
        rounded_threshold = np.round(threshold, 2)
        predicted_terms = predicted_terms_dict[rounded_threshold]
        prediction_set_sizes = []
        correct_predictions = 0
        total_samples = len(predicted_terms)
        total_false_positives = 0
        total_true_positives = 0
        total_false_negatives = 0

        precision_list = []
        recall_list = []

        for pred_set, true_label in zip(predicted_terms, y_true):
            prediction_set_sizes.append(len(pred_set))

            true_positives = len(set(pred_set).intersection(set(true_label)))
            total_true_positives += true_positives

            false_positives = len(set(pred_set) - set(true_label))
            total_false_positives += false_positives

            false_negatives = len(set(true_label) - set(pred_set))
            total_false_negatives += false_negatives

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

            precision_list.append(precision)
            recall_list.append(recall)

            if true_positives > 0:
                correct_predictions += 1

        avg_set_size = np.mean(prediction_set_sizes)
        coverage = correct_predictions / total_samples

        fdr = total_false_positives / (total_false_positives + total_true_positives) if (total_false_positives + total_true_positives) > 0 else 0

        precision_micro = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
        recall_micro = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0

        precision_macro = np.mean(precision_list) if precision_list else 0
        recall_macro = np.mean(recall_list) if recall_list else 0
        sharpness_dict[rounded_threshold] = avg_set_size
        coverage_dict[rounded_threshold] = coverage
        fdr_dict[rounded_threshold] = fdr
        precision_micro_dict[rounded_threshold] = precision_micro
        recall_micro_dict[rounded_threshold] = recall_micro
        precision_macro_dict[rounded_threshold] = precision_macro
        recall_macro_dict[rounded_threshold] = recall_macro

    rounded_thresholds = [np.round(t, 2) for t in confidence_thresholds]
    
    sharpness_list = [sharpness_dict[t] for t in rounded_thresholds]
    coverage_list = [coverage_dict[t] for t in rounded_thresholds]
    fdr_list = [fdr_dict[t] for t in rounded_thresholds]
    precision_micro_list = [precision_micro_dict[t] for t in rounded_thresholds]
    precision_macro_list = [precision_macro_dict[t] for t in rounded_thresholds]
    recall_micro_list = [recall_micro_dict[t] for t in rounded_thresholds]
    recall_macro_list = [recall_macro_dict[t] for t in rounded_thresholds]
    
    return (sharpness_list, coverage_list, fdr_list, 
            precision_micro_list, precision_macro_list, 
            recall_micro_list, recall_macro_list)

def display_metrics(
    sharpness_dict, coverage_dict, fdr_dict, 
    precision_micro_dict, precision_macro_dict, 
    recall_micro_dict, recall_macro_dict
):
    """
    Display sharpness, coverage, FDR, precision (micro/macro), and recall (micro/macro) for each confidence threshold.
    """
    print(
        f"{'Threshold':<10} {'Sharpness (Avg Set Size)':<30} {'Coverage':<10} "
        f"{'FDR':<10} {'Prec (Micro)':<15} {'Prec (Macro)':<15} "
        f"{'Recall (Micro)':<15} {'Recall (Macro)':<15}"
    )
    print("=" * 120)
    
    for threshold in sharpness_dict:
        print(
            f"{threshold:<10.2f} {sharpness_dict[threshold]:<30.2f} "
            f"{coverage_dict[threshold]:<10.2%} {fdr_dict[threshold]:<10.2%} "
            f"{precision_micro_dict[threshold]:<15.2%} {precision_macro_dict[threshold]:<15.2%} "
            f"{recall_micro_dict[threshold]:<15.2%} {recall_macro_dict[threshold]:<15.2%}"
        )

def predict_without_MCD(model, data_loader, device='cpu'):
    """
    Performs standard deterministic inference without MC Dropout.
    """
    model.eval()  
    all_probs = []

    with torch.no_grad():
        for x_batch in data_loader:
            x_batch = x_batch[0].to(device)  
            logits = model(x_batch)  
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)

def find_closest_term_from_sorted(term, valid_terms, sorted_ancestors_dict):
    """
    For a given term, return the first ancestor that is also in the valid set.
    """
    ancestors = sorted_ancestors_dict.get(term, [])
    for anc in ancestors:
        if anc in valid_terms:
            return anc
    return None


def compute_effective_threshold(effective_scores, ground_truth, candidate_thresholds, target_fdr, smoothing=1e-6):
    """
    Compute the smallest candidate threshold for which the observed FDR is <= target_fdr.
    """
    for threshold in candidate_thresholds:
        preds = (effective_scores >= threshold).astype(int)
        TP = np.sum((preds == 1) & (ground_truth == 1))
        FP = np.sum((preds == 1) & (ground_truth == 0))
        total_positive = TP + FP
        if total_positive == 0:
            continue
        fdr = (FP + smoothing) / (total_positive + 2 * smoothing)
        if fdr <= target_fdr:
            return threshold
    return None

def compute_thresholds_for_term(effective_scores, ground_truth, candidate_thresholds, target_fdrs, smoothing=1e-6):
    """
    Compute effective thresholds for a single term across multiple target FDR values.
    """
    thresholds = {}
    for target_fdr in target_fdrs:
        threshold = compute_effective_threshold(effective_scores, ground_truth, candidate_thresholds, target_fdr, smoothing)
        thresholds[target_fdr] = threshold
    return thresholds

def compute_all_effective_thresholds(balanced_median_probs, balanced_MAD_probs, test_labels, candidate_thresholds, target_fdrs, lambda_val=1, smoothing=1e-6):
    """
    Compute effective thresholds for each term using the given candidate thresholds and target FDRs.
    """
    num_terms = test_labels.shape[1]
    effective_thresholds = {}
    for j in tqdm(range(num_terms), desc="Computing effective thresholds"):
        effective_scores = balanced_median_probs[:, j] - lambda_val * balanced_MAD_probs[:, j]
        ground_truth_term = test_labels[:, j]
        thresholds_for_term = compute_thresholds_for_term(effective_scores, ground_truth_term, candidate_thresholds, target_fdrs, smoothing)
        effective_thresholds[j] = thresholds_for_term
    return effective_thresholds

def count_valid_terms(lambda_val, balanced_median_probs, balanced_MAD_probs, val_labels, candidate_thresholds, target_fdrs, smoothing=1e-6):
    """
    For a given lambda, compute effective thresholds for all terms and count how many terms have at least one valid threshold.
    """
    num_terms = val_labels.shape[1]
    valid_count = 0
    thresholds_dict = {}
    for j in tqdm(range(num_terms), desc="Processing terms", leave=False):
        effective_scores = balanced_median_probs[:, j] - lambda_val * balanced_MAD_probs[:, j]
        ground_truth_term = val_labels[:, j]
        thresholds_for_term = {}
        valid = False
        for target_fdr in target_fdrs:
            threshold = compute_effective_threshold(effective_scores, ground_truth_term, candidate_thresholds, target_fdr, smoothing)
            thresholds_for_term[target_fdr] = threshold
            if threshold is not None:
                valid = True
        if valid:
            valid_count += 1
        thresholds_dict[j] = thresholds_for_term
    return valid_count, thresholds_dict
def find_global_threshold_for_fdr(df, target_fdr):
    """
    Given a DataFrame with columns ['Threshold', 'FDR'], return the threshold whose FDR is closest to target_fdr.
    """
    df['FDR_diff'] = np.abs(df['FDR'] - target_fdr)
    best_idx = df['FDR_diff'].idxmin()
    best_threshold = df.loc[best_idx, 'Threshold']
    return best_threshold
import numpy as np

def compute_effective_scores(median_probs, mad_probs, lambda_val=1.0):
    """
    Compute effective scores as median_probs - lambda_val * mad_probs.
    """
    return median_probs - lambda_val * mad_probs

def get_term_threshold(term, effective_thresholds, target_fdr=0.05):
    """
    For a given term, return the best available effective threshold for target_fdr using the closest lower FDR approach.
    """
    term_thresholds = effective_thresholds.get(term, {})
    
    term_thresholds_float = {}
    for fdr_key_str, threshold in term_thresholds.items():
        try:
            fdr_key_float = float(fdr_key_str)
            term_thresholds_float[fdr_key_float] = threshold
        except (ValueError, TypeError):
            if isinstance(fdr_key_str, (int, float)):
                term_thresholds_float[float(fdr_key_str)] = threshold
            continue
    
    if target_fdr in term_thresholds_float and term_thresholds_float[target_fdr] is not None:
        return term_thresholds_float[target_fdr]
    lower_fdr_levels = [
        fdr_key_float for fdr_key_float, threshold in term_thresholds_float.items() 
        if fdr_key_float < target_fdr and threshold is not None
    ]
    
    if lower_fdr_levels:
        closest_lower_fdr = max(lower_fdr_levels)
        return term_thresholds_float[closest_lower_fdr]
    else:
        return None

def build_pred_annots_dict(test_median_probs, test_MAD_probs, effective_thresholds, test_entries, terms, lambda_val=1.0, target_fdr=0.05):
    """
    Create a predictions dictionary for the test set using per-term effective thresholds.
    """
    effective_scores = compute_effective_scores(test_median_probs, test_MAD_probs, lambda_val)
    
    pred_annots_dict = {}
    num_entries, num_terms = effective_scores.shape
    for i in tqdm(range(num_entries), desc="Building predictions"):
        entry = test_entries[i]
        entry_preds = {}
        for j in range(num_terms):
            term = terms[j]
            threshold = get_term_threshold(term, effective_thresholds, target_fdr)
            if threshold is not None:
                score = effective_scores[i, j]
                if score >= threshold:
                    entry_preds[term] = score
        if entry_preds:
            pred_annots_dict[entry] = entry_preds
    return pred_annots_dict

def convert_predictions_to_set(pred_annots_dict):
    """
    Convert a predictions dictionary into a dictionary mapping each entry to a set of predicted GO terms.
    """
    return {entry: set(go_scores.keys()) for entry, go_scores in pred_annots_dict.items()}

def evaluate_annotations(ic_dict, real_annots_dict, pred_annots_dict):
    """
    Evaluates precision, recall, F1-score, remaining uncertainty (ru), and misinformation (mi) using sets of GO terms.
    """
    total = 0
    p_sum = 0.0
    r_sum = 0.0
    p_total = 0
    ru = 0.0
    mi = 0.0
    fps = []
    fns = []
    tp_global, fp_global, fn_global = 0, 0, 0

    common_entries = set(real_annots_dict.keys()).intersection(pred_annots_dict.keys())

    for entry in common_entries:
        real_annots = real_annots_dict[entry]
        pred_annots = pred_annots_dict[entry]

        tp = real_annots.intersection(pred_annots)
        fp = pred_annots - tp
        fn = real_annots - tp

        tp_global += len(tp)
        fp_global += len(fp)
        fn_global += len(fn)

        for go_id in fp:
            if go_id in ic_dict:
                mi += ic_dict[go_id]

        for go_id in fn:
            if go_id in ic_dict:
                ru += ic_dict[go_id]

        fps.append(fp)
        fns.append(fn)
        total += 1

        recall = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0
        precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0

        r_sum += recall
        if len(pred_annots) > 0:
            p_total += 1
            p_sum += precision

    r = r_sum / total if total > 0 else 0
    p = p_sum / p_total if p_total > 0 else 0

    p_micro = tp_global / (tp_global + fp_global) if (tp_global + fp_global) > 0 else 0
    r_micro = tp_global / (tp_global + fn_global) if (tp_global + fn_global) > 0 else 0

    f = 2 * p * r / (p + r) if (p + r) > 0 else 0
    f_micro = 2 * p_micro * r_micro / (p_micro + r_micro) if (p_micro + r_micro) > 0 else 0

    ru /= total
    mi /= total

    s = math.sqrt(ru * ru + mi * mi)

    return f, p, r, s, ru, mi, f_micro, p_micro, r_micro, tp_global, fp_global, fn_global

def _calculate_metrics_at_threshold(ic_dict, real_annots_dict, pred_annots_dict_with_scores, threshold):
    """
    Helper function to calculate metrics at a specific threshold.
    """
    filtered_pred_annots_dict = {
        entry: {go_id for go_id, score in go_scores.items() if score >= threshold}
        for entry, go_scores in pred_annots_dict_with_scores.items()
    }

    f, p, r, s, ru, mi, f_micro, p_micro, r_micro, tp_global, fp_global, fn_global = \
        evaluate_annotations(ic_dict, real_annots_dict, filtered_pred_annots_dict)
    cov = len([1 for preds in filtered_pred_annots_dict.values() if len(preds) > 0]) / len(real_annots_dict)

    return {
        'n': threshold,
        'tp': tp_global,
        'fp': fp_global,
        'fn': fn_global,
        'pr': p,
        'rc': r,
        'cov': cov,
        'mi': mi,
        'ru': ru,
        'f': f,
        's': s,
        'pr_micro': p_micro,
        'rc_micro': r_micro,
        'f_micro': f_micro,
        'cov_max': cov
    }


def threshold_performance_metrics(ic_dict, real_annots_dict, pred_annots_dict_with_scores, threshold_range=None, set_threshold=None):
    """
    Calculates S-min and F-max over a range of thresholds or at a set threshold.
    """
    if threshold_range is None and set_threshold is None:
        raise ValueError("Either threshold_range or set_threshold must be provided.")

    smin = float('inf')
    fmax = 0
    best_threshold_s = None
    best_threshold_f = None
    s_at_fmax = None
    results = []

    if set_threshold is not None:
        result = _calculate_metrics_at_threshold(ic_dict, real_annots_dict, pred_annots_dict_with_scores, set_threshold)
        results.append(result)
    else:
        for threshold in tqdm(threshold_range, desc='Calculating Smin & Fmax'):
            result = _calculate_metrics_at_threshold(ic_dict, real_annots_dict, pred_annots_dict_with_scores, threshold)
            results.append(result)

            if result['s'] < smin:
                smin = result['s']
                best_threshold_s = threshold
            if result['f'] > fmax:
                fmax = result['f']
                best_threshold_f = threshold
                s_at_fmax = result['s']

    results_df = pd.DataFrame(results)

    print(f"F-max @ Best Threshold ({best_threshold_f}): {fmax}")
    print(f"S-min @ Best Threshold ({best_threshold_s}): {smin}")
    print(f"S-min @ F-max Threshold ({best_threshold_f}): {s_at_fmax}")

    return smin, fmax, best_threshold_s, best_threshold_f, s_at_fmax, results_df


def calculate_aupr_micro(real_annots_dict, pred_annots_dict_with_scores):
    """
    Calculate AUPR Micro for the entire dataset.
    """
    y_true_flat = []
    y_scores_flat = []

    for entry, go_scores in pred_annots_dict_with_scores.items():
        real_annots = real_annots_dict.get(entry, set())
        
        for go_id, score in go_scores.items():
            y_true_flat.append(1 if go_id in real_annots else 0)
            y_scores_flat.append(score)

    precision, recall, _ = precision_recall_curve(y_true_flat, y_scores_flat)
    return auc(recall, precision)

def filter_predictions_by_fdr(entry_details, target_fdr_cutoff):
    """
    Given the entry_prediction_details dictionary, return a set of GO terms predicted at a minimum FDR target less than or equal to target_fdr_cutoff.
    """
    predicted_terms = {term for term, (score, fdr_target) in entry_details.items() if fdr_target <= target_fdr_cutoff}
    return predicted_terms


import os
import torch
from typing import Dict, List, Tuple

def load_kfold_ensembles(
    kf_root: str,
    fold_options: List[int],
    model_cls: type,
    model_kwargs: Dict,
    device: torch.device,
    ckpt_suffix: str = ".pt"
) -> Dict[int, List[Tuple[int, torch.nn.Module]]]:
    """
    Scans each subfolder under kf_root, loads checkpoints into model instances, and returns dict mapping k to models.
    """
    ensembles: Dict[int, List[Tuple[int, torch.nn.Module]]] = {}
    for k in fold_options:
        k_dir = os.path.join(kf_root, str(k))
        models: List[Tuple[int, torch.nn.Module]] = []

        for fn in os.listdir(k_dir):
            if not fn.endswith(ckpt_suffix):
                continue

            name = fn[:-len(ckpt_suffix)]
            parts = name.split("_")
            try:
                fold_idx = int(parts[-1].lstrip("fold"))
            except ValueError:
                raise ValueError(f"Cannot parse fold index from '{fn}'")

            path = os.path.join(k_dir, fn)
            state = torch.load(path, map_location=device)

            m = model_cls(**model_kwargs).to(device)
            m.load_state_dict(state)
            m.eval()

            models.append((fold_idx, m))

        models.sort(key=lambda x: x[0])
        ensembles[k] = models

    return ensembles

import numpy as np
import torch
from typing import Callable, Dict, Any


def compute_ensemble_probs(
    kf_models: Dict[int, List[Tuple[int, nn.Module]]],
    loader: DataLoader,
    device: torch.device,
    predict_fn: Callable[[nn.Module, DataLoader, torch.device], np.ndarray]
) -> Dict[int, np.ndarray]:
    """
    Given a dict of k-fold models, runs each model on loader and stacks the outputs.
    """
    ensemble_probs: Dict[int, np.ndarray] = {}
    for k, models_list in kf_models.items():
        runs = []
        for fold_idx, model in models_list:
            p = predict_fn(model, loader, device)
            runs.append(p)
        ensemble_probs[k] = np.stack(runs, axis=0)
    return ensemble_probs


def chunked_mad_over_runs(arr, chunk_size=100):
    """
    Compute the MAD over axis=0 of arr in class-chunks of size chunk_size to save memory.
    """
    runs, N, C = arr.shape
    mad = np.empty((N, C), dtype=arr.dtype)

    for start in range(0, C, chunk_size):
        end = min(start + chunk_size, C)
        block = arr[:, :, start:end]
        
        med_block = np.median(block, axis=0)
        abs_dev = np.abs(block - med_block[None, :, :])
        mad[:, start:end] = np.median(abs_dev, axis=0)
        
        del block, med_block, abs_dev

    return mad


def chunked_median_over_runs(arr, chunk_size=100):
    """
    Compute the median over axis=0 of arr in class-chunks of size chunk_size to save memory.
    """
    runs, N, C = arr.shape
    med = np.empty((N, C), dtype=arr.dtype)

    for start in range(0, C, chunk_size):
        end = min(start + chunk_size, C)
        block = arr[:, :, start:end]
        med[:, start:end] = np.median(block, axis=0)
        del block

    return med
