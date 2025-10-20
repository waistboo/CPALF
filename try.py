import copy
import os
import pickle
from collections import defaultdict, Counter
from itertools import zip_longest, cycle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances, f1_score
from sklearn.preprocessing import StandardScaler
from torch.nn import Sequential
from torch.utils.tensorboard import SummaryWriter
from scipy.spatial import distance
import scipy.stats as stats
from torch.utils.data import DataLoader, TensorDataset
import random
import tensorflow as tf
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.graphgym import GCNConv
from pyclustering.cluster.kmedoids import kmedoids
from collections import defaultdict


class PatternClassificationNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(PatternClassificationNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.model(x)


class LayerConfig:
    def __init__(self, dim_in, dim_out, has_bias=True):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.has_bias = has_bias


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_channels, output_dim):
        super(GCN, self).__init__()
        layer_config_1 = LayerConfig(dim_in=input_dim, dim_out=hidden_channels)
        layer_config_2 = LayerConfig(dim_in=hidden_channels, dim_out=output_dim)
        self.conv1 = GCNConv(layer_config_1)
        self.conv2 = GCNConv(layer_config_2)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x


def train_GCN(model, data, lbl_labels, epochs=50, optimizer=None):
    """
    Train a GCN on nodes with labels in lbl_labels.
    lbl_labels is a dict {node_id: label}.
    """
    criterion_CE = nn.CrossEntropyLoss()
    criterion_KL = nn.KLDivLoss(reduction='batchmean')

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    num_nodes = data.num_nodes
    labels_tensor = -1 * torch.ones(num_nodes, dtype=torch.long)
    for node_idx, label in lbl_labels.items():
        labels_tensor[node_idx] = label

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        train_mask = labels_tensor >= 0
        loss = criterion_CE(out[train_mask], labels_tensor[train_mask])
        loss.backward()
        optimizer.step()

    return model, optimizer


def predict_GCN(model, data, unlbl_ids):
    """
    Predict labels and confidences for unlabeled node ids using a trained GCN.
    Returns dicts of predictions and confidences keyed by node id.
    """
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_attr)
        probs = F.softmax(out, dim=1)
        preds = probs.argmax(dim=1)
        confidence_gcn = probs.max(dim=1)[0]

    unlbl_preds = preds[unlbl_ids]
    gcn_results, gcn_conf_results = {}, {}

    for id, label, confidence in zip(unlbl_ids, unlbl_preds.flatten(), confidence_gcn[unlbl_ids]):
        gcn_results[int(id)] = label.item()
        gcn_conf_results[int(id)] = confidence.item()

    return gcn_results, gcn_conf_results


def train_model_semi_sup(X, y, un_X, un_y, model, epochs=50, optimizer=None):
    """
    Semi-supervised training: supervised CE on labeled data + KL loss on selected pseudo-labels.
    """
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    labeled_dataset = TensorDataset(X_tensor, y_tensor)
    dataloader_labeled = DataLoader(labeled_dataset, batch_size=64, shuffle=True, drop_last=False)

    if un_X is not None and len(un_X) > 0:
        un_X_tensor = torch.tensor(un_X, dtype=torch.float32)
        un_y_tensor = un_y
        unlabeled_dataset = TensorDataset(un_X_tensor, un_y_tensor)
        dataloader_unlabeled = DataLoader(unlabeled_dataset, batch_size=64*7, shuffle=True, drop_last=False)
        has_unlabeled = True
    else:
        has_unlabeled = False

    criterion_CE = nn.CrossEntropyLoss()
    criterion_KL = nn.KLDivLoss(reduction='batchmean')

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        for (batch_X, batch_y) in dataloader_labeled:
            optimizer.zero_grad()
            if has_unlabeled:
                try:
                    batch_un_X, batch_un_y = next(iter(dataloader_unlabeled))
                except StopIteration:
                    continue
                predictions = model(batch_X)
                loss_labeled = criterion_CE(predictions, batch_y)
                predictions_un = model(batch_un_X)
                smoothed_un_labels = label_smoothing(batch_un_y, epsilon=0.1)
                loss_unlabeled = criterion_KL(F.log_softmax(predictions_un, dim=1), smoothed_un_labels)
                total_loss = loss_labeled + loss_unlabeled
            else:
                predictions = model(batch_X)
                total_loss = criterion_CE(predictions, batch_y)
            total_loss.backward()
            optimizer.step()

    return model, optimizer


def label_smoothing(labels, epsilon=0.1):
    num_classes = labels.size(1)
    smooth_labels = labels * (1 - epsilon) + (epsilon / num_classes)
    return smooth_labels


def human_annotation_stim(data, cluster, m, reduced_data):
    """
    Simulate human annotation by clustering and mapping clusters to m classes.
    Returns mapping real_labels {index: class} and representative first_sample_indices list.
    """
    kmeans = KMeans(n_clusters=cluster, random_state=42)
    kmeans.fit(data)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    sorted_dict = {}
    first_samples = {}
    for unique_val in sorted(set(labels)):
        indices = np.where(labels == unique_val)[0]
        distances = np.linalg.norm(data[indices] - centers[unique_val], axis=1)
        sorted_indices = indices[np.argsort(distances)]
        sorted_dict[unique_val] = sorted_indices.tolist()

    real_rank = {}
    favorite = 0
    for i in range(len(sorted_dict)):
        for j in range(len(sorted_dict[i])):
            real_rank[sorted_dict[i][j]] = favorite
            favorite += 1
    rank_keys = list(real_rank.keys())

    n = len(real_rank)
    part_size = n // m
    real_labels = {}
    for i in range(m):
        start_index = i * part_size
        if i == m - 1:
            end_index = n
        else:
            end_index = start_index + part_size
        for j in range(start_index, end_index):
            mode_id = rank_keys[j]
            real_labels[mode_id] = i
            if i not in first_samples:
                first_samples[i] = mode_id

    first_sample_indices = [first_samples[i] for i in range(m)]
    return real_labels, first_sample_indices


def labels_annotation(ids, real_labels):
    """
    Return a dict of {id: label} for the provided ids, using real_labels mapping.
    """
    labeled_samples = {}
    for current_id in ids:
        labeled_samples[current_id] = real_labels.get(current_id, None)
    return labeled_samples


def labeled_X_y_and_unlabeled_X_y(data, train_lbl_ids, train_unlbl_ids, train_lbl_labels, pseudo_y):
    """
    Build X, y for labeled samples and un_X, un_y for selected pseudo-labeled samples.
    Returns X, y, un_X, un_y_one_hot (for KL loss)
    """
    lbl_y = []
    lbl_x = []
    unlbl_y = []
    unlbl_x = []
    for i in train_lbl_ids:
        label = train_lbl_labels.get(i)
        if label is not None:
            lbl_y.append(label)
            lbl_x.append(data[i])
    X = np.array(lbl_x, dtype=np.float32)
    y = np.array(lbl_y, dtype=np.int64)

    for j in train_unlbl_ids:
        label = pseudo_y.get(j)
        if label is not None:
            unlbl_y.append(label)
            unlbl_x.append(data[j])
    un_X = np.array(unlbl_x, dtype=np.float32)
    un_y = np.array(unlbl_y, dtype=np.int64)
    y_one_hot = torch.nn.functional.one_hot(torch.tensor(y, dtype=torch.long), num_classes=5)
    un_y_one_hot = torch.nn.functional.one_hot(torch.tensor(un_y, dtype=torch.long), num_classes=5)
    return X, y, un_X, un_y_one_hot


def pred_k_get(dictionary, k):
    keys = list(dictionary.keys())
    selected_keys = keys[:k]
    k_pred = [(key, dictionary[key]) for key in selected_keys]
    return k_pred


def truth_k_get(real_labels, k_pred):
    k_pred_ids = [index for index, _ in k_pred]
    k_true = [(index, real_labels[index]) for index in k_pred_ids if index in real_labels]
    return k_true


def top_k_accuracy(true, predict, k):
    true_dict = {key: value for key, value in sorted(true)}
    predict_dict = {key: value for key, value in sorted(predict)}
    corr_ids = []
    correct = 0
    for index in true_dict:
        value_true = true_dict[index]
        if index in predict_dict:
            value_pred = predict_dict[index]
            if value_true == value_pred:
                correct += 1
                corr_ids.append(index)
    accuracy = correct / k
    return accuracy


def normalize(dict_scores):
    """
    Normalize dictionary values to [0,1].
    """
    if len(dict_scores) == 0:
        return {}
    scores = list(dict_scores.values())
    min_score = np.min(scores)
    max_score = np.max(scores)
    if max_score - min_score == 0:
        return {id_: 0.0 for id_ in dict_scores.keys()}
    return {id_: (score - min_score) / (max_score - min_score) for id_, score in dict_scores.items()}


def flexmatch_filter_samples(model, data, unlbl_ids, prior_pl, thresholds, n_class):
    """
    FlexMatch-style pseudo label selection with GCN consistency check.
    Returns selected ids, corresponding probs, and updated thresholds.
    """
    unlabeled_X = data[unlbl_ids]
    with torch.no_grad():
        logits_u = model(torch.tensor(unlabeled_X, dtype=torch.float32))
        probs_u = torch.softmax(logits_u, dim=1)
        max_probs, model_labels = probs_u.max(dim=1)

    id_to_model_pred = {id_: (int(model_labels[i]), float(max_probs[i]), probs_u[i].tolist()) for i, id_ in enumerate(unlbl_ids)}

    use_condition1 = True
    use_condition2 = True
    use_condition3 = True

    selected_ids, selected_labels, selected_probs, confidences = [], [], [], []
    pseudo_counts = {cls: 0 for cls in range(n_class)}
    max_per_cls = n_class
    for id_ in unlbl_ids:
        if id_ in prior_pl:
            gcn_label = prior_pl[id_]
            model_label, model_conf, probs_un = id_to_model_pred[id_]

            condition1_met = True
            condition2_met = True
            condition3_met = True

            if use_condition1:
                condition1_met = (gcn_label == model_label)
            if use_condition2:
                condition2_met = (model_conf > thresholds[model_label])
            if use_condition3:
                condition3_met = (pseudo_counts[model_label] < max_per_cls)

            if condition1_met and condition2_met and condition3_met:
                selected_ids.append(id_)
                selected_labels.append(model_label)
                selected_probs.append(probs_un)
                confidences.append(model_conf)
                pseudo_counts[model_label] += 1

    if len(selected_ids) > 0:
        confidences = torch.tensor(confidences)
        pseudo_labels = torch.tensor(selected_labels)
        class_confidence = {}
        for cls in range(n_class):
            cls_mask = (pseudo_labels == cls)
            if cls_mask.sum() > 0:
                class_confidence[cls] = confidences[cls_mask].mean().item()

        global_avg_conf = confidences.mean().item()
        updated_thresholds = thresholds.clone()

        for cls in range(n_class):
            if cls in class_confidence:
                beta_t = class_confidence[cls] / (global_avg_conf + 1e-9)
                mapped_beta_t = beta_t / (2 - beta_t)
                updated_thresholds[cls] = 0.999 * thresholds[cls] + 0.001 * mapped_beta_t
                updated_thresholds[cls] = torch.clamp(updated_thresholds[cls], 0.7, 0.9999)
    else:
        updated_thresholds = thresholds.clone()
    print(f"Next round thresholds: {updated_thresholds}")

    return selected_ids, selected_probs, updated_thresholds


def active_learning_query(n_sortedCluster, results_uncert, alpha, n_query):
    """
    Combine uncertainty and representativeness to select top samples for query.
    """
    normalized_uncertainty = normalize(results_uncert)
    normalized_representativeness = normalize(n_sortedCluster)
    combined_scores = {}
    for id_ in normalized_uncertainty.keys():
        combined_scores[id_] = alpha * normalized_uncertainty[id_] + (1 - alpha) * normalized_representativeness[id_]
    sort_combine = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
    top_combine = [item[0] for item in sort_combine[:n_query]]
    return top_combine


def select_history_samples_by_miss_labels(labeled_kind1, labeled_kind, budget, n_class):
    """
    Select historical samples to cover missing labels.
    Older samples get priority; selected samples are moved to the end of labeled_kind.
    """
    n_reap = 5

    all_labels = set(range(n_class))
    current_labels = set(labeled_kind1.values())
    missing_labels = all_labels - current_labels

    additional_samples = []
    labels_of_selected_samples = {}
    if missing_labels:
        label_to_samples = defaultdict(list)
        for idx, (sample_id, label) in enumerate(labeled_kind.items()):
            if label is not None:
                label_to_samples[label].append((idx, sample_id))
        for label in missing_labels:
            if label in label_to_samples:
                samples_with_indices = label_to_samples[label]
                samples_with_indices.sort()
                for index, sample_id in samples_with_indices[:n_reap]:
                    additional_samples.append(sample_id)
                    labels_of_selected_samples[sample_id] = labeled_kind[sample_id]
            else:
                print(f"No historical sample with label {label}")

    for sample_id in additional_samples:
        if sample_id in labeled_kind:
            del labeled_kind[sample_id]

    for sample_id in additional_samples:
        labeled_kind[sample_id] = labels_of_selected_samples[sample_id]
    labeled_kind.update(labeled_kind1)

    return additional_samples, labeled_kind, labels_of_selected_samples


def calculate_acc(test_ids, true_labels, results_labels):
    """
    Calculate accuracy on test_ids using results_labels mapping.
    Uses top-k style comparison where k = len(test_ids).
    """
    test_labels = {key: value for key, value in true_labels.items() if key in test_ids}
    top_k = int(1 * len(test_ids))
    top_k_predict = pred_k_get(results_labels, top_k)
    top_k_true = truth_k_get(test_labels, top_k_predict)
    accuracy = top_k_accuracy(top_k_true, top_k_predict, top_k)
    return accuracy


def interactive_training_with_mixed_selection(data, data1, test_ids, model, optimizer, selected_samples, unlbl_data_ids,
                n_sortedCluster, true_labels, labeled_kind1, labeled_kind, iteration, train_results, results_uncert,
                thresholds, budget, m, alpha, reduced_data, model_gcn, optimizer_gcn):
    """
    Main interactive loop per round: select queries, collect labels, update training sets, train model.
    """
    print(f"Thresholds this round: {thresholds}")
    n_sortedCluster_unlbl = {id: repre for id, repre in n_sortedCluster.items() if id in unlbl_data_ids}

    prior_pl, model_gcn, optimizer_gcn = assign_ppl(data1, model_gcn, optimizer_gcn, unlbl_data_ids, labeled_kind, true_labels, reduced_data)
    pseudo_X_ids, pseudo_y, thresholds = flexmatch_filter_samples(model, data, unlbl_data_ids, prior_pl, thresholds, m)

    selected_miss_samples = []
    if iteration == 20:
        print(f"Iteration {iteration}")
        existing_labels = set(labeled_kind.values())
        all_labels = set(range(m))
        missing_labels = all_labels - existing_labels
        for label in missing_labels:
            for sample_id in unlbl_data_ids:
                if true_labels[sample_id] == label:
                    selected_miss_samples.append(sample_id)
                    break
    n = len(selected_miss_samples)

    top_query_ids = active_learning_query(n_sortedCluster_unlbl, results_uncert, alpha, budget-n)
    top_query_ids.extend(selected_miss_samples)
    selected_all = selected_samples + top_query_ids
    labeled_kind1 = labels_annotation(top_query_ids, true_labels)
    repeated_ids, labeled_kind, repeated_kind = select_history_samples_by_miss_labels(labeled_kind1, labeled_kind, budget, m)

    unlbl_ids = [id for id in unlbl_data_ids if id not in top_query_ids]
    unlabeled_kind = {index: category for index, category in train_results.items() if index in pseudo_X_ids}
    unlabeled_kind_real = {index: category for index, category in true_labels.items() if index in pseudo_X_ids}
    common_keys = set(unlabeled_kind.keys()) & set(unlabeled_kind_real.keys())
    matched_ids = [k for k in common_keys if unlabeled_kind[k] == unlabeled_kind_real[k]]
    ppl_match_count = len(matched_ids)

    labeled_ids = top_query_ids + repeated_ids
    labeled_kind1.update(repeated_kind)

    X, y, un_X, un_y = labeled_X_y_and_unlabeled_X_y(data, labeled_ids, pseudo_X_ids, labeled_kind1, unlabeled_kind)
    print(f"Training samples this round: {labeled_kind1}")
    print(f"Total labeled: {len(labeled_kind)}; new: {len(top_query_ids)}; repeated: {len(repeated_ids)}; pseudo labels: {len(unlabeled_kind)}; correct pseudo: {ppl_match_count}")

    model, optimizer = train_model_semi_sup(X, y, un_X, un_y, model, optimizer=optimizer)

    train_results1 = {}
    results_uncert1 = {}
    if len(unlbl_ids) > 0:
        model_predictions_train = model(torch.tensor(data[unlbl_ids], dtype=torch.float32))
        pred_pseudo_labels = model_predictions_train.argmax(dim=1)
        uncertainty_scores = -torch.sum(F.softmax(model_predictions_train, dim=1) * F.log_softmax(model_predictions_train, dim=1), dim=1)
        for id, label, uncertainty in zip(unlbl_ids, pred_pseudo_labels.flatten(), uncertainty_scores):
            train_results1[int(id)] = label.item()
            results_uncert1[int(id)] = uncertainty.item()
    else:
        train_results1 = {}
        results_uncert1 = {}

    results1 = {}
    model_predictions = model(torch.tensor(data[test_ids], dtype=torch.float32))
    pseudo_labels = model_predictions.argmax(dim=1)
    for id, label in zip(test_ids, pseudo_labels.flatten()):
        results1[id] = label.item()
    real_labels_unlabeled = np.array([true_labels[id] for id in test_ids if id in true_labels])
    class_avg_f1 = f1_score(real_labels_unlabeled, pseudo_labels.numpy(), average='macro') if len(real_labels_unlabeled) > 0 else 0.0
    pred_labels = pseudo_labels.numpy()
    true_labels_binary = (pred_labels == real_labels_unlabeled).astype(np.float32) if len(real_labels_unlabeled) > 0 else np.array([])
    max_probs = F.softmax(model_predictions.detach(), dim=1).max(dim=1).values.numpy()
    ece = compute_ece(true_labels_binary, max_probs) if len(true_labels_binary) > 0 else 0.0
    print(f"Class Avg F1: {class_avg_f1:.4f}, ECE: {ece:.4f}")

    accuracy_test = calculate_acc(test_ids, true_labels, results1)
    print("Test accuracy:", accuracy_test)

    return model, optimizer, selected_all, unlbl_ids, labeled_kind1, labeled_kind, accuracy_test, train_results1, results_uncert1, thresholds, model_gcn, optimizer_gcn


def compute_ece(true_labels, probs, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).
    true_labels: binary indicator array where 1 means correct prediction.
    probs: max predicted probabilities.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    for bin_idx in range(n_bins):
        mask = (bin_indices == bin_idx)
        if np.sum(mask) > 0:
            acc = np.mean(true_labels[mask])
            conf = np.mean(probs[mask])
            ece += np.abs(acc - conf) * np.sum(mask) / len(probs)
    return ece


def random_test(cluster, budget, ndata, sortedCluster, frequent_patterns, num_iterations, n_class, alpha, graph, tau):
    """
    Run incremental training and testing loop.
    """
    n_budget = budget
    data = ndata.copy()
    n_sortedCluster = copy.deepcopy(sortedCluster)
    m = n_class

    data_scaled = StandardScaler().fit_transform(data)
    reduced_data = TSNE(n_components=2).fit_transform(data_scaled)

    real_labels, first_sample_indices = human_annotation_stim(data, cluster=cluster, m=m, reduced_data=reduced_data)

    split_index = int(len(n_sortedCluster) * 0.7)
    train_ids = list(n_sortedCluster.keys())[:split_index]
    test_ids = list(n_sortedCluster.keys())[split_index:]
    train_sortedrepre = {k: n_sortedCluster[k] for k in n_sortedCluster if k in train_ids}

    lbl_ids = list(train_sortedrepre.keys())[:budget]
    unlbl_ids = list(train_sortedrepre.keys())[budget:]
    lbl_labels = labels_annotation(lbl_ids, real_labels)

    data1 = copy.deepcopy(graph)
    model_gcn = GCN(input_dim=data1.x.shape[1], hidden_channels=16, output_dim=n_class)
    model_gcn, optimizer_gcn = train_GCN(model_gcn, data1, lbl_labels)
    pre_pseudo_labels, gcn_conf_results = predict_GCN(model_gcn, data1, unlbl_ids)
    accuracy_train = calculate_acc(unlbl_ids, real_labels, pre_pseudo_labels)
    print("PPL accuracy:", accuracy_train)

    labeled_kind = {}
    labeled_kind.update(lbl_labels)
    print(f"Initial labeled samples: {labeled_kind}")

    print("Round 1 supervised training:")
    pseudo_X_ids = []
    unlabeled_kind = {}
    X, y, un_X, un_y = labeled_X_y_and_unlabeled_X_y(data, lbl_ids, pseudo_X_ids, lbl_labels, unlabeled_kind)

    model = PatternClassificationNN(input_dim=data.shape[1], num_classes=n_class)
    model, optimizer = train_model_semi_sup(X, y, un_X, un_y, model)

    train_results = {}
    results_uncert = {}
    if len(unlbl_ids) > 0:
        model_predictions_train = model(torch.tensor(data[unlbl_ids], dtype=torch.float32))
        post_pseudo_labels = model_predictions_train.argmax(dim=1)
        uncertainty_scores = -torch.sum(F.softmax(model_predictions_train, dim=1) * F.log_softmax(model_predictions_train, dim=1), dim=1)
        for id, label, uncertainty in zip(unlbl_ids, post_pseudo_labels.flatten(), uncertainty_scores):
            train_results[int(id)] = label.item()
            results_uncert[int(id)] = uncertainty.item()

    results = {}
    model_predictions = model(torch.tensor(data[test_ids], dtype=torch.float32))
    pseudo_labels = model_predictions.argmax(dim=1)
    for id, label in zip(test_ids, pseudo_labels.flatten()):
        results[id] = label.item()
    real_labels_unlabeled = np.array([real_labels[id] for id in test_ids if id in real_labels])
    class_avg_f1 = f1_score(real_labels_unlabeled, pseudo_labels.numpy(), average='macro') if len(real_labels_unlabeled) > 0 else 0.0
    pred_labels = pseudo_labels.numpy()
    true_labels_binary = (pred_labels == real_labels_unlabeled).astype(np.float32) if len(real_labels_unlabeled) > 0 else np.array([])
    max_probs = F.softmax(model_predictions.detach(), dim=1).max(dim=1).values.numpy()
    ece = compute_ece(true_labels_binary, max_probs) if len(true_labels_binary) > 0 else 0.0
    print(f"Class Avg F1: {class_avg_f1:.4f}, ECE: {ece:.4f}")

    print(f"{len(lbl_ids)} samples labeled")
    accuracy_test = calculate_acc(test_ids, real_labels, results)
    print("Test accuracy:", accuracy_test)
    accuracy_results = []
    accuracy_results.append(accuracy_test)

    thresh_per_class = torch.ones(n_class) * tau

    for iteration in range(2, num_iterations+1):
        print(f"Iteration {iteration}:")
        model, optimizer, lbl_ids, unlbl_ids, lbl_labels, labeled_kind, accuracy_test, train_results, results_uncert, \
            thresh_per_class, model_gcn, optimizer_gcn = interactive_training_with_mixed_selection(data, data1, test_ids,
            model, optimizer, lbl_ids, unlbl_ids, train_sortedrepre, real_labels, lbl_labels, labeled_kind, iteration,
            train_results, results_uncert, thresh_per_class, n_budget, m, alpha, reduced_data, model_gcn, optimizer_gcn)

        accuracy_results.append(accuracy_test)

    return accuracy_test, accuracy_results


def assign_ppl(data, model, optimizer, unlbl_ids, lbl_labels, real_labels, reduced_data):
    """
    Train GCN and use label propagation (nearest labeled example) as pseudo labels (PPL).
    Returns {node: pseudo_label}, updated model and optimizer.
    """
    lbl_ids = list(lbl_labels.keys())
    model, optimizer = train_GCN(model, data, lbl_labels=lbl_labels, optimizer=optimizer)
    pre_pseudo_labels, gcn_conf_results = predict_GCN(model, data, unlbl_ids)
    accuracy_train = calculate_acc(unlbl_ids, real_labels, pre_pseudo_labels)
    print("PPL accuracy:", accuracy_train)
    return pre_pseudo_labels, model, optimizer


def load_data(filepath):
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    return data


def main():
    datasets = ['Skitter']

    accuracy_all_results = []
    for dataset in datasets:
        info_graph = torch.load(f"datasets/{dataset}/{dataset}.pt")
        filepath = f"datasets/{dataset}/{dataset}.pkl"
        frequent_patterns = f"datasets/{dataset}.txt"
        print(f"Processing file: {filepath}")

        data, sortedCluster = load_data(filepath)
        n_sortedCluster = dict(sortedCluster)

        num_iterations = 20
        n_class = 5
        alpha = 0.6
        batch_size = 5
        thresholds = [round(0.941 + i * 0.001, 3) for i in range(20)]
        # thresholds = [0.954]

        for n_clusters in range(1, 16):
            print(f"Number of clusters: {n_clusters}")
            for threshold in thresholds:
                experiment_seed = 0
                np.random.seed(experiment_seed)
                torch.manual_seed(experiment_seed)

                accuracy, accuracy_results = random_test(n_clusters, batch_size, data, n_sortedCluster,
                                                         frequent_patterns,
                                                         num_iterations=num_iterations, n_class=n_class,
                                                         alpha=alpha,
                                                         graph=info_graph, tau=threshold)
                print(f"Final Top-k accuracy with threshold {threshold}, alpha {alpha}: {accuracy}")
                accuracy_all_results.extend(accuracy_results)


if __name__ == "__main__":
    main()