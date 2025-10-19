import copy
import os
import pickle
from collections import defaultdict, Counter
from itertools import zip_longest, cycle
from re import M
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
# from keras.src.utils import to_categorical
from sklearn.cluster import KMeans
from sklearn.cluster._hdbscan import hdbscan
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

# from pattern_size import pattern_sizes_dict

# # 设定随机种子,全局设定有问题
# np.random.seed(0)
# torch.manual_seed(0)

class PatternClassificationNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(PatternClassificationNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),  # 第一个隐藏层，256个神经元
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),  # 输出类别数量
            # nn.Dropout(0.1)  # 防止过拟合
        )

    def forward(self, x):
        return self.model(x)


class ConvNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ConvNet, self).__init__()

        # 假设输入有800个特征，使用 1D 卷积
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5)  # 第一个卷积层
        self.pool = nn.MaxPool1d(kernel_size=2)  # 池化层
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5)  # 第二个卷积层

        # 计算 flatten 层的输入
        # 计算每一层的输出大小
        conv1_output_size = ((input_dim - 5 + 1) // 2)  # Conv1 输出大小经过池化后
        conv2_output_size = (conv1_output_size - 5 + 1) // 2  # Conv2 输出大小经过池化后

        self.flatten = nn.Flatten()  # 将特征展平
        self.fc1 = nn.Linear(32 * conv2_output_size, 128)  # 第一个全连接层
        self.fc2 = nn.Linear(128, num_classes)  # 第二个全连接层，最终输出 num_classes

    def forward(self, x):  # (32,800)
        # 输入的 x 需要形状为 (batch_size, 1, 800)
        x = x.unsqueeze(1)  # (32,1,800)
        x = self.conv1(x)  # 第一层卷积(32,16,796)
        x = F.relu(x)  # 使用 ReLU 激活
        x = self.pool(x)  # 池化(32,16,398)

        x = self.conv2(x)  # 第二层卷积(32,32,394)
        x = F.relu(x)  # 使用 ReLU 激活
        x = self.pool(x)  # 池化(32,32,197)

        x = self.flatten(x)  # 展平(32,6304)
        x = F.relu(self.fc1(x))  # 第一个全连接层
        x = self.fc2(x)  # 第二个全连接层(32,4)
        x = F.softmax(x, dim=1)
        return x


class LayerConfig:
    def __init__(self, dim_in, dim_out, has_bias=True):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.has_bias = has_bias


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_channels, output_dim):
        super(GCN, self).__init__()
        # 创建 LayerConfig 对象
        layer_config_1 = LayerConfig(dim_in=input_dim, dim_out=hidden_channels)
        layer_config_2 = LayerConfig(dim_in=hidden_channels, dim_out=output_dim)
        self.conv1 = GCNConv(layer_config_1)
        self.conv2 = GCNConv(layer_config_2)

        # self.conv1 = GCNConv(input_dim, hidden_channels)
        # self.conv2 = GCNConv(hidden_channels, output_dim)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x


def train_GCN(model, data, lbl_labels, epochs=50, optimizer=None):
    criterion_CE = nn.CrossEntropyLoss()  # 使用交叉熵损失
    criterion_KL = nn.KLDivLoss(reduction='batchmean')

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()

    num_nodes = data.num_nodes
    labels_tensor = -1 * torch.ones(num_nodes, dtype=torch.long)  # 默认未标记节点为 -1
    for node_idx, label in lbl_labels.items():
        labels_tensor[node_idx] = label  # 填入有标签的节点

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)

        # 只计算有标签的节点的损失
        train_mask = labels_tensor >= 0  # 训练样本掩码
        loss = criterion_CE(out[train_mask], labels_tensor[train_mask])
        # num_classes = out.shape[1]
        # one_hot_labels = F.one_hot(labels_tensor[train_mask], num_classes=num_classes).float()
        # smoothed_labels = label_smoothing(one_hot_labels, epsilon=0.1)
        # loss = criterion_KL(F.log_softmax(out[train_mask], dim=1), smoothed_labels)

        loss.backward()
        optimizer.step()

    return model, optimizer


def predict_GCN(model, data, unlbl_ids):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_attr)  # 原始输出 [n_nodes, n_classes]
        probs = F.softmax(out, dim=1)  # 计算概率
        preds = probs.argmax(dim=1)  # 预测类别
        confidence_gcn = probs.max(dim=1)[0]  # 置信度

    unlbl_preds = preds[unlbl_ids]  # 无标签节点的预测
    gcn_results, gcn_conf_results = {}, {}

    for id, label, confidence in zip(unlbl_ids, unlbl_preds.flatten(), confidence_gcn[unlbl_ids]):
        gcn_results[int(id)] = label.item()
        gcn_conf_results[int(id)] = confidence.item()

    return gcn_results, gcn_conf_results


def train_model_semi_sup(X, y, un_X, un_y, model, epochs=50, optimizer=None):
    """支持部分无标签数据的半监督训练"""
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # 处理有标签数据
    labeled_dataset = TensorDataset(X_tensor, y_tensor)
    dataloader_labeled = DataLoader(labeled_dataset, batch_size=64, shuffle=True, drop_last=False)

    # 处理无标签数据（若非空）
    if un_X is not None and len(un_X) > 0:
        un_X_tensor = torch.tensor(un_X, dtype=torch.float32)
        # un_y_tensor = torch.tensor(un_y, dtype=torch.long)
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
        total_loss = 0
        for (batch_X, batch_y) in dataloader_labeled:
            optimizer.zero_grad()

            # 有无无标签数据两个分支
            if has_unlabeled:
                try:
                    batch_un_X, batch_un_y = next(iter(dataloader_unlabeled))
                except StopIteration:
                    # 如果用完了unlabeled就重新开始（可选）
                    continue

                # 计算有标签损失
                predictions = model(batch_X)
                loss_labeled = criterion_CE(predictions, batch_y)

                # 计算无标签损失
                predictions_un = model(batch_un_X)
                # loss_unlabeled = criterion_CE(predictions_un, batch_un_y)  # 对无标签样本进行交叉熵损失
                smoothed_un_labels = label_smoothing(batch_un_y, epsilon=0.1)
                loss_unlabeled = criterion_KL(F.log_softmax(predictions_un, dim=1), smoothed_un_labels)
                # loss_unlabeled = criterion_KL(F.log_softmax(predictions_un, dim=1), batch_un_y)  # 不标签平滑直接模型预测

                total_loss = loss_labeled + 1 * loss_unlabeled
            else:
                # 只用有标签数据
                predictions = model(batch_X)
                loss_labeled = criterion_CE(predictions, batch_y)
                total_loss = loss_labeled

            total_loss.backward()
            optimizer.step()

    return model, optimizer


def distill_train(teacher_model, student_model, X, y, un_X, un_y, optimizer=None, epochs=50, temperature=2.0, beta=0.6):
    """
    蒸馏训练函数：学生模型同时学习
    1. 教师模型的软化输出（KL散度损失）
    2. 有标签数据的硬标签（交叉熵损失）
    3. 无标签数据的伪标签（半监督逻辑，保留原有）
    """
    # 1. 数据预处理（与原有逻辑一致）
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    labeled_dataset = TensorDataset(X_tensor, y_tensor)
    dataloader_labeled = DataLoader(labeled_dataset, batch_size=64, shuffle=True, drop_last=False)

    # 处理无标签数据（保留原有半监督逻辑）
    has_unlabeled = False
    dataloader_unlabeled = None
    if un_X is not None and len(un_X) > 0:
        un_X_tensor = torch.tensor(un_X, dtype=torch.float32)
        un_y_tensor = un_y  # 伪标签，来自原有逻辑
        unlabeled_dataset = TensorDataset(un_X_tensor, un_y_tensor)
        dataloader_unlabeled = DataLoader(unlabeled_dataset, batch_size=64 * 7, shuffle=True, drop_last=False)
        has_unlabeled = True

    # 2. 损失函数定义（新增蒸馏损失+保留原有损失）
    criterion_CE = nn.CrossEntropyLoss()  # 硬标签/伪标签损失
    criterion_KL = nn.KLDivLoss(reduction='batchmean')  # 蒸馏损失（KL散度）

    if optimizer is None:
        optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)

    # 3. 蒸馏训练流程
    teacher_model.eval()  # 教师模型固定，仅提供知识
    student_model.train()  # 学生模型作为训练主体
    for epoch in range(epochs):
        total_loss = 0.0
        # 初始化无标签数据迭代器（避免每轮重复创建）
        unlabeled_iter = iter(dataloader_unlabeled) if has_unlabeled else None

        for (batch_X, batch_y) in dataloader_labeled:
            optimizer.zero_grad()

            # -------------------------- 原有半监督逻辑：计算有标签+无标签损失 --------------------------
            # 有标签数据损失（硬标签）
            student_pred_l = student_model(batch_X)
            student_soft_pred_l = F.log_softmax(student_pred_l / temperature, dim=1)
            loss_labeled = criterion_CE(student_pred_l, batch_y)

            # 无标签数据损失（伪标签+标签平滑，保留原有逻辑）
            loss_unlabeled = 0.0
            if has_unlabeled:
                try:
                    batch_un_X, batch_un_y = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(dataloader_unlabeled)
                    batch_un_X, batch_un_y = next(unlabeled_iter)

                student_pred_unl = student_model(batch_un_X)
                smoothed_un_labels = label_smoothing(batch_un_y, epsilon=0.1)
                loss_unlabeled = criterion_KL(F.log_softmax(student_pred_unl, dim=1), smoothed_un_labels)
                student_soft_pred_unl = F.log_softmax(student_pred_unl / temperature, dim=1)

            # -------------------------- 新增蒸馏逻辑：计算教师指导损失 --------------------------
            with torch.no_grad():  # 教师模型不更新，禁用梯度
                # 1. 计算有标签数据的蒸馏损失（必选，无数据依赖）
                teacher_pred_l = teacher_model(batch_X)
                teacher_soft_l = F.softmax(teacher_pred_l / temperature, dim=1)
                loss_distill_l = criterion_KL(student_soft_pred_l, teacher_soft_l) * (temperature ** 2)

                # 2. 计算伪标签数据的蒸馏损失（可选，仅当有伪标签时计算）
                loss_distill_un = 0.0  # 默认值：无伪标签时贡献0
                if has_unlabeled:
                    teacher_pred_unl = teacher_model(batch_un_X)
                    teacher_soft_unl = F.softmax(teacher_pred_unl / temperature, dim=1)
                    loss_distill_un = criterion_KL(student_soft_pred_unl, teacher_soft_unl) * (temperature ** 2)

            # 3. 总蒸馏损失：融合有标签+伪标签（无伪标签时自动仅用loss_distill_l）
            loss_distill = loss_distill_l + loss_distill_un

            # -------------------------- 总损失：融合蒸馏+原有半监督损失 --------------------------
            # beta：蒸馏损失权重（0.5~0.7为宜，平衡历史知识与新数据）
            total_loss_step = beta * loss_distill + (1 - beta) * (loss_labeled + loss_unlabeled)
            total_loss_step.backward()
            optimizer.step()

            total_loss += total_loss_step.item()

    return student_model, optimizer


# 新增：记录每个参数的重要性（用于正则化损失）
class EWCRegularizer:
    def __init__(self, model, dataset, device='cpu'):
        """初始化EWC正则化器，计算当前模型在旧数据集上的参数重要性"""
        self.model = model
        self.device = device
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {n: p.data.clone() for n, p in self.params.items()}  # 记录旧参数（均值）
        self._importance = self._compute_importance(dataset)  # 计算参数重要性（Fisher信息矩阵对角线）

    def _compute_importance(self, dataset):
        """通过在旧数据集上计算梯度，估计参数重要性（简化版：用梯度平方均值）"""
        importance = {n: torch.zeros_like(p) for n, p in self.params.items()}
        self.model.eval()

        # 用旧数据集的样本计算梯度，累积参数重要性
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            self.model.zero_grad()
            outputs = self.model(batch_X)
            loss = nn.CrossEntropyLoss()(outputs, batch_y)
            loss.backward()  # 计算当前参数在旧任务上的梯度

            # 用梯度平方累积重要性（Fisher信息矩阵的近似）
            for n, p in self.params.items():
                if p.grad is not None:
                    importance[n] += p.grad.data ** 2 / len(dataloader)  # 平均梯度平方

        return importance

    def penalty(self, model):
        """计算当前模型参数与旧参数的正则化损失（惩罚偏离）"""
        loss = 0.0
        for n, p in model.named_parameters():
            if n in self._importance:
                # 正则化损失：参数重要性 × (当前参数 - 旧参数)^2
                loss += (self._importance[n] * (p - self._means[n]) ** 2).sum()
        return loss


def regularized_train(model, optimizer, X, y, un_X, un_y, ewc_regularizer=None, ewc_lambda=1000, epochs=50):
    """
    正则化增量训练函数：
    1. 保留原有半监督逻辑（有标签+伪标签损失）
    2. 新增EWC正则化损失，惩罚参数偏离旧任务重要参数
    """
    # 1. 数据预处理（与原有逻辑一致）
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    labeled_dataset = TensorDataset(X_tensor, y_tensor)
    dataloader_labeled = DataLoader(labeled_dataset, batch_size=64, shuffle=True)

    # 处理无标签数据（保留原有半监督逻辑）
    has_unlabeled = False
    dataloader_unlabeled = None
    if un_X is not None and len(un_X) > 0:
        un_X_tensor = torch.tensor(un_X, dtype=torch.float32)
        un_y_tensor = torch.tensor(un_y, dtype=torch.long)  # 伪标签
        unlabeled_dataset = TensorDataset(un_X_tensor, un_y_tensor)
        dataloader_unlabeled = DataLoader(unlabeled_dataset, batch_size=64 * 7, shuffle=True)
        has_unlabeled = True

    # 2. 损失函数定义（保留原有+新增正则化损失）
    criterion_CE = nn.CrossEntropyLoss()  # 有标签损失
    criterion_KL = nn.KLDivLoss(reduction='batchmean')  # 伪标签损失（标签平滑）

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        unlabeled_iter = iter(dataloader_unlabeled) if has_unlabeled else None

        for (batch_X, batch_y) in dataloader_labeled:
            optimizer.zero_grad()

            # -------------------------- 原有半监督损失 --------------------------
            # 有标签数据损失
            pred_l = model(batch_X)
            loss_labeled = criterion_CE(pred_l, batch_y)

            # 无标签数据损失（伪标签+标签平滑）
            loss_unlabeled = 0.0
            if has_unlabeled:
                try:
                    batch_un_X, batch_un_y = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(dataloader_unlabeled)
                    batch_un_X, batch_un_y = next(unlabeled_iter)

                pred_unl = model(batch_un_X)
                smoothed_un_labels = torch.softmax(batch_un_y.float(), dim=1)  # 标签平滑（简化版）
                loss_unlabeled = criterion_KL(torch.log_softmax(pred_unl, dim=1), smoothed_un_labels)

            # -------------------------- 新增：正则化损失 --------------------------
            loss_regularization = 0.0
            if ewc_regularizer is not None:
                # 若存在旧任务正则化器，添加参数偏离惩罚
                loss_regularization = ewc_regularizer.penalty(model)

            # -------------------------- 总损失 = 任务损失 + 正则化损失 --------------------------
            total_loss_step = loss_labeled + loss_unlabeled + ewc_lambda * loss_regularization
            total_loss_step.backward()
            optimizer.step()

            total_loss += total_loss_step.item()

        # # 打印每轮损失
        # if (epoch + 1) % 10 == 0:
        #     print(f"Epoch [{epoch + 1}/{epochs}], Total Loss: {total_loss / len(dataloader_labeled):.4f}")

    return model, optimizer


def label_smoothing(labels, epsilon=0.1):
    num_classes = labels.size(1)
    smooth_labels = labels * (1 - epsilon) + (epsilon / num_classes)
    return smooth_labels


def generate_pseudo_labels(model, unlabeled_data):
    """生成伪标签"""
    model.eval()
    with torch.no_grad():
        # model_predictions = model(torch.tensor(data[unlabeled_ids], dtype=torch.float32))
        # pseudo_labels = model_predictions.argmax(dim=1) + 1
        unlabeled_tensor = torch.tensor(unlabeled_data, dtype=torch.float32)
        model_predictions = model(unlabeled_tensor)
        pseudo_labels = model_predictions.argmax(dim=0)
    return pseudo_labels.item()  # 返回每个样本的预测类别，要确保是整数类型


def human_annotation_stim(data, cluster, m, reduced_data):
    """人工标注，使用KMeans聚类并根据中心选择最具代表性样本，用户真实喜好参照"""
    kmeans = KMeans(n_clusters=cluster, random_state=42)
    kmeans.fit(data)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    sorted_dict = {}
    real_labels = {}
    first_sample_indices = []
    for unique_val in sorted(set(labels)):
        indices = np.where(labels == unique_val)[0]
        distances = np.linalg.norm(data[indices] - centers[unique_val], axis=1)
        sorted_indices = indices[np.argsort(distances)]
        sorted_dict[unique_val] = sorted_indices.tolist()

    #     first_sample_indices.append(sorted_indices[0])  # 直接添加第一个样本的索引
    #
    # # 簇号为标签 类别不均衡
    # for cluster_id, sample_indices in sorted_dict.items():
    #     for sample_idx in sample_indices:
    #         real_labels[sample_idx] = cluster_id
    #
    # from collections import Counter
    # cluster_counts = Counter(real_labels.values())

    real_rank = {}
    first_samples = {}  # 用于记录每个标签的第一个样本
    favorite = 0
    for i in range(len(sorted_dict)):
        for j in range(len(sorted_dict[i])):
            real_rank[sorted_dict[i][j]] = favorite
            favorite += 1
    rank_keys = list(real_rank.keys())

    # 将标签转换为m个类别
    n = len(real_rank)
    part_size = n // m
    real_labels = {}
    for i in range(m):
        # 计算当前部分的起始和结束索引
        start_index = i * part_size
        # 处理最后一部分
        if i == m - 1:
            end_index = n
        else:
            end_index = start_index + part_size
        for j in range(start_index, end_index):
            mode_id = rank_keys[j]
            real_labels[mode_id] = i

            if i not in first_samples:  # 如果标签还没有记录
                first_samples[i] = mode_id  # 记录第一个样本的索引

    first_sample_indices = [first_samples[i] for i in range(m)]

    # real_ids = list(real_labels.keys())
    # all_classes = list(range(m))
    # true_label_values = np.array([real_labels[idx] for idx in real_ids])
    # color_map = plt.get_cmap('Spectral', m)
    # # color_map = ['red', 'yellow', 'blue', 'pink', 'purple']
    # for i, label in enumerate(all_classes):
    #     indices = np.where(true_label_values == label)[0]
    #     if len(indices) > 0:
    #         plt.scatter(reduced_data[indices, 0], reduced_data[indices, 1],
    #                     color=color_map(i), label=f'Class {label}', alpha=0.6, s=50)
    #     else:
    #         plt.scatter([], [], color=color_map(i), label=f'Class {label}', alpha=0.6, s=50)
    #
    # plt.title('Data Visualization with Pseudo Labels Comparison')
    # plt.xlabel('PCA Component 1')
    # plt.ylabel('PCA Component 2')
    # plt.legend()
    # plt.show()

    return real_labels, first_sample_indices


def labels_annotation(ids, real_labels):
    """对选择的样本获得用户真实喜好"""
    labeled_samples = {}
    for current_id in ids:
        labeled_samples[current_id] = real_labels.get(current_id, None)
        # labeled_samples[current_id] = real_labels[current_id]
    # print(labeled_samples)
    return labeled_samples


def labeled_X_y_and_unlabeled_X_y(data, train_lbl_ids, train_unlbl_ids, train_lbl_labels, pseudo_y):
    """根据选中的样本和排序标签生成训练数据和标签"""
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

    # # # 处理软标签
    # for idx, j in enumerate(train_unlbl_ids):
    #     label = pseudo_y[idx]
    #     if label is not None:
    #         unlbl_y.append(label)
    #         unlbl_x.append(data[j])
    # un_X = np.array(unlbl_x, dtype=np.float32)
    # un_y = torch.tensor(unlbl_y, dtype=torch.float32)
    #
    # return X, y, un_X, un_y

    # # 处理硬标签
    for j in train_unlbl_ids:
        label = pseudo_y.get(j)
        if label is not None:
            unlbl_y.append(label)
            unlbl_x.append(data[j])
    un_X = np.array(unlbl_x, dtype=np.float32)
    un_y = np.array(unlbl_y, dtype=np.int64)  # CE
    y_one_hot = torch.nn.functional.one_hot(torch.tensor(y, dtype=torch.long), num_classes=5)
    un_y_one_hot = torch.nn.functional.one_hot(torch.tensor(un_y, dtype=torch.long), num_classes=5)  # for KL

    return X, y, un_X, un_y_one_hot


def pred_k_get(dictionary, k):
    """从预测结果中按顺序选择k个键，获取预测结果"""
    # random_keys = random.sample(list(dictionary.keys()), k)  # 随机选取 k 个键
    keys = list(dictionary.keys())
    selected_keys = keys[:k]
    k_pred = [(key, dictionary[key]) for key in selected_keys]
    return k_pred


def truth_k_get(real_labels, k_pred):
    """先从pred_k_get中获取k个索引，然后在real_labels中找到这些索引及其对应的真实值"""
    k_pred_ids = [index for index, _ in k_pred]
    k_true = [(index, real_labels[index]) for index in k_pred_ids if index in real_labels]
    return k_true


def top_k_accuracy(true, predict, k):
    """计算Top-k准确率"""
    true_dict = {key: value for key, value in sorted(true)}
    predict_dict = {key: value for key, value in sorted(predict)}

    corr_ids = []
    correct = 0
    for index in true_dict:
        value_true = true_dict[index]
        if index in predict_dict:  # 只有当预测字典中也有这个索引时才进行计算
            value_pred = predict_dict[index]
            if value_true == value_pred:
                correct += 1
                corr_ids.append(index)
    # print(f"预测对的样本有{corr_ids}")
    accuracy = correct / k
    return accuracy


def normalize(dict_scores):
    """对输入的分数进行归一化"""
    if len(dict_scores) == 0:
        return {}
    scores = list(dict_scores.values())
    min_score = np.min(scores)
    max_score = np.max(scores)
    # 防止分母为0的情况
    if max_score - min_score == 0:
        return {id_: 0.0 for id_ in dict_scores.keys()}
    return {id_: (score - min_score) / (max_score - min_score) for id_, score in dict_scores.items()}


def flexmatch_filter_samples(model, data, unlbl_ids, prior_pl, thresholds, n_class):
    """
    FlexMatch 伪标签筛选函数（增加GCN一致性验证）
    Args:
        model: 当前模型
        data: 所有数据（包含有标签和无标签）
        unlbl_ids: 无标签样本的ID列表
        prior_pl: {样本id: GCN伪标签} 字典
        thresholds: 各类别阈值 [n_class]
        n_class: 类别数
    Returns:
        selected_ids: 筛选后的样本ID列表
        selected_probs: 预测分布
        updated_thresholds: 更新后的阈值
    """
    # 1. 准备无标签数据
    unlabeled_X = data[unlbl_ids]

    # 2. 模型预测无标签数据
    with torch.no_grad():
        logits_u = model(torch.tensor(unlabeled_X, dtype=torch.float32))
        probs_u = torch.softmax(logits_u, dim=1)
        max_probs, model_labels = probs_u.max(dim=1)  # 模型预测的伪标签

    # 3. 构建样本ID到预测结果的映射 id:(预测类别,置信度分数,概率分布)
    id_to_model_pred = {id_: (int(model_labels[i]), float(max_probs[i]), probs_u[i].tolist())for i, id_ in enumerate(unlbl_ids)}

    # 4. 一致性比对和阈值筛选

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

            # 初始化条件检测
            condition1_met = True
            condition2_met = True
            condition3_met = True

            # 条件1：GCN和模型预测一致
            if use_condition1:
                condition1_met = (gcn_label == model_label)

            # 条件2：置信度超过当前类别阈值
            if use_condition2:
                condition2_met = (model_conf > thresholds[model_label])

            # 条件3：类别样本数量限制
            if use_condition3:
                condition3_met = (pseudo_counts[model_label] < max_per_cls)

            # 只筛选满足全部启用的条件
            if condition1_met and condition2_met and condition3_met:
                selected_ids.append(id_)
                selected_labels.append(model_label)
                selected_probs.append(probs_un)
                confidences.append(model_conf)
                pseudo_counts[model_label] += 1

    # 5. 动态阈值更新
    if len(selected_ids) > 0:
        # 准备阈值更新所需数据
        confidences = torch.tensor(confidences)
        pseudo_labels = torch.tensor(selected_labels)

        # 重新计算各类别平均置信度
        class_confidence = {}
        for cls in range(n_class):
            cls_mask = (pseudo_labels == cls)
            if cls_mask.sum() > 0:
                class_confidence[cls] = confidences[cls_mask].mean().item()

        # 更新阈值
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
    print(f"下一轮阈值为{updated_thresholds}")

    return selected_ids, selected_probs, updated_thresholds


def active_learning_query(n_sortedCluster, results_uncert, alpha, n_query):
    """根据不确定性和代表性综合选出信息量最大的样本进行标注"""
    normalized_uncertainty = normalize(results_uncert)
    normalized_representativeness = normalize(n_sortedCluster)
    combined_scores = {}
    for id_ in normalized_uncertainty.keys():
        combined_scores[id_] = alpha * normalized_uncertainty[id_] + (1 - alpha) * normalized_representativeness[id_]
    sort_combine = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
    top_combine = [item[0] for item in sort_combine[:n_query]]

    return top_combine


def select_history_samples_by_miss_labels(labeled_kind1, labeled_kind, budget, n_class):
    """选择一部分缺失标签的历史样本一起学习，被训练的越早优先权越高，被选择过的样本优先权变低"""
    # if budget <= n_class:
    #     n_reap = 5
    # else:
    #     n_reap = int(budget / n_class)

    n_reap = 5

    # 1. 检查缺失的标签
    all_labels = set(range(n_class))
    current_labels = set(labeled_kind1.values())
    missing_labels = all_labels - current_labels

    # 2. 从历史样本中选择缺失标签的样本
    additional_samples = []
    labels_of_selected_samples = {}  # 用于保存选择样本的标签
    if missing_labels:
        label_to_samples = defaultdict(list)
        for idx, (sample_id, label) in enumerate(labeled_kind.items()):
            if label is not None:
                label_to_samples[label].append((idx, sample_id))  # 存储样本索引和ID
        # 根据缺失标签选择样本
        for label in missing_labels:
            if label in label_to_samples:
                samples_with_indices = label_to_samples[label]
                # 按照索引升序排序，确保选择早期的样本
                samples_with_indices.sort()  # 按时间升序排序
                # 选择早期的样本，不超过n_reap个
                for index, sample_id in samples_with_indices[:n_reap]:
                    additional_samples.append(sample_id)
                    labels_of_selected_samples[sample_id] = labeled_kind[sample_id]  # 保存标签信息
            else:
                print(f"历史样本中也没有标签为{label}样本")

    # 3. 让被选择的样本优先权变低
    # 从 labeled_kind 中删除选择的样本
    for sample_id in additional_samples:
        if sample_id in labeled_kind:
            del labeled_kind[sample_id]

    # 将选择的样本添加到 labeled_kind 的末尾
    for sample_id in additional_samples:
        labeled_kind[sample_id] = labels_of_selected_samples[sample_id]
    labeled_kind.update(labeled_kind1)  # 更新历史训练样本

    return additional_samples, labeled_kind, labels_of_selected_samples


def dely(selected_samples, n_sortedCluster):
    """删除已选样本，返回剩余样本"""
    remaining_samples = {key: value for key, value in n_sortedCluster.items() if key not in selected_samples}
    # remaining_samples = [item for item in n_sortedCluster if item[0] not in selected_samples]
    # remaining_samples = [item for item in n_sortedCluster if item not in selected_samples]
    return remaining_samples


def calculate_acc(test_ids, true_labels, results_labels):
    """准确率的计算"""
    test_labels = {key: value for key, value in true_labels.items() if key in test_ids}
    # print("测试集个数：", len(test_labels))
    top_k = int(1 * len(test_ids))
    top_k_predict = pred_k_get(results_labels, top_k)
    top_k_true = truth_k_get(test_labels, top_k_predict)
    accuracy = top_k_accuracy(top_k_true, top_k_predict, top_k)
    return accuracy


def interactive_training_with_mixed_selection(data, data1, test_ids, model, optimizer, selected_samples, unlbl_data_ids,
                n_sortedCluster, true_labels, labeled_kind1, labeled_kind, iteration, train_results, results_uncert,
                thresholds, budget, m, alpha, reduced_data, model_gcn, optimizer_gcn):

    print(f"当轮阈值111为{thresholds}")
    n_sortedCluster_unlbl = {id: repre for id, repre in n_sortedCluster.items() if id in unlbl_data_ids}

    prior_pl, model_gcn, optimizer_gcn = assign_ppl(data1, model_gcn, optimizer_gcn, unlbl_data_ids, labeled_kind, true_labels, reduced_data)
    # prior_pl, acc_ppl = assign_ppl(data, selected_samples, test_data_ids, labeled_kind, true_labels, reduced_data)
    # pseudo_X, thresholds = select_pseudo_labels(prior_pl, train_results, results_conf, thresholds, n_class)
    pseudo_X_ids, pseudo_y, thresholds = flexmatch_filter_samples(model, data, unlbl_data_ids, prior_pl, thresholds, m)

    selected_miss_samples = []
    if iteration == 20:
        print(f"当前轮次为{iteration}")
        # 检查labeled_kind是否包含所有m个类别
        existing_labels = set(labeled_kind.values())
        all_labels = set(range(m))
        missing_labels = all_labels - existing_labels
        # 如果缺少某个标签类别，根据unlbl_ids在true_labels中找到缺少类别
        for label in missing_labels:
            for sample_id in unlbl_data_ids:
                if true_labels[sample_id] == label:
                    selected_miss_samples.append(sample_id)
                    break
    n = len(selected_miss_samples)

    top_query_ids = active_learning_query(n_sortedCluster_unlbl, results_uncert, alpha, budget-n)
    # random.seed(42)  # 设置随机种子
    # top_query_ids = random.sample(list(n_sortedCluster_unlbl.keys()), budget - n)
    top_query_ids.extend(selected_miss_samples)
    selected_all = selected_samples + top_query_ids
    labeled_kind1 = labels_annotation(top_query_ids, true_labels)
    repeated_ids, labeled_kind, repeated_kind = select_history_samples_by_miss_labels(labeled_kind1, labeled_kind, budget, m)
    # 加随机选择历史样本
    # repeated_ids = random.sample(list(labeled_kind.keys()), budget)
    # repeated_kind = {id: labeled_kind[id] for id in repeated_ids}

    unlbl_ids = [id for id in unlbl_data_ids if id not in top_query_ids]
    unlabeled_kind = {index: category for index, category in train_results.items() if index in pseudo_X_ids}
    unlabeled_kind_real = {index: category for index, category in true_labels.items() if index in pseudo_X_ids}
    common_keys = set(unlabeled_kind.keys()) & set(unlabeled_kind_real.keys())
    matched_ids = [k for k in common_keys if unlabeled_kind[k] == unlabeled_kind_real[k]]
    ppl_match_count = len(matched_ids)

    # labeled_ids = selected_all  # 全量
    # labeled_kind1 = labeled_kind
    labeled_ids = top_query_ids + repeated_ids  # his
    labeled_kind1.update(repeated_kind)  # 更新当轮模型学习样本  # his
    # labeled_ids = top_query_ids  # no his
    # pseudo_X_ids = []  # supp
    # unlabeled_kind = {}  # supp

    # X, y, un_X, un_y = labeled_X_y_and_unlabeled_X_y(data, labeled_ids, pseudo_X_ids, labeled_kind1, pseudo_y)
    X, y, un_X, un_y = labeled_X_y_and_unlabeled_X_y(data, labeled_ids, pseudo_X_ids, labeled_kind1, unlabeled_kind)
    # X, y, un_X, un_y = labeled_X_y_and_unlabeled_X_y(data, labeled_ids, pseudo_X_ids, labeled_kind1, unlabeled_kind_real)
    print(f"当轮学习样本为{labeled_kind1}")
    print(f"总共{len(labeled_kind)} 个样本有标签，其中本轮{len(top_query_ids)}个新增样本，{len(repeated_ids)}个历史样本，"
          f"{len(unlabeled_kind)}个伪标签，其中伪标签正确的个数为{ppl_match_count}")

    model, optimizer = train_model_semi_sup(X, y, un_X, un_y, model, optimizer=optimizer)
    # EWC 用正则化训练函数更新模型（在原有模型基础上训练）
    # model, optimizer = regularized_train(model=model, optimizer=optimizer, X=X, y=y, un_X=un_X, un_y=un_y,
    #                                      ewc_regularizer=prev_ewc_regularizer, ewc_lambda=ewc_lambda, epochs=50)
    # 调用蒸馏函数：教师模型指导学生模型，保留原有半监督逻辑（有标签+无标签）
    # model, optimizer = distill_train(teacher_model=teacher_model, student_model=model, X=X, y=y, un_X=un_X,
    # un_y=un_y, optimizer=optimizer, epochs=50, temperature=2.0, beta=0.6)

    train_results1 = {}
    results_uncert1 = {}
    model_predictions_train = model(torch.tensor(data[unlbl_ids], dtype=torch.float32))
    pred_pseudo_labels = model_predictions_train.argmax(dim=1)
    uncertainty_scores = -torch.sum(F.softmax(model_predictions_train, dim=1) *
                                    F.log_softmax(model_predictions_train, dim=1), dim=1)
    for id, label, uncertainty in zip(unlbl_ids, pred_pseudo_labels.flatten(), uncertainty_scores):
        train_results1[int(id)] = label.item()
        results_uncert1[int(id)] = uncertainty.item()

    # 对测试集进行预测
    results1 = {}
    model_predictions = model(torch.tensor(data[test_ids], dtype=torch.float32))
    pseudo_labels = model_predictions.argmax(dim=1)
    for id, label in zip(test_ids, pseudo_labels.flatten()):
        results1[id] = label.item()
    # 计算Class Avg F1
    real_labels_unlabeled = np.array([true_labels[id] for id in test_ids if id in true_labels])
    class_avg_f1 = f1_score(real_labels_unlabeled, pseudo_labels.numpy(), average='macro')
    # 计算ECE
    pred_labels = pseudo_labels.numpy()
    true_labels_binary = (pred_labels == real_labels_unlabeled).astype(np.float32)
    max_probs = F.softmax(model_predictions.detach(), dim=1).max(dim=1).values.numpy()
    ece = compute_ece(true_labels_binary, max_probs)
    print(f"Class Avg F1: {class_avg_f1:.4f}, ECE: {ece:.4f}")

    accuracy_test = calculate_acc(test_ids, true_labels, results1)
    print("测试集准确率:", accuracy_test)
    # print("模型预测结果:", results1)

    return model, optimizer, selected_all, unlbl_ids, labeled_kind1, labeled_kind, accuracy_test, train_results1,\
        results_uncert1, thresholds, model_gcn, optimizer_gcn


def compute_itr_scores(pattern_sizes_dict, pattern_supports_dict, alpha=2):
    """计算Itr分数"""
    itr_scores = {}
    for pid in pattern_sizes_dict:
        size = pattern_sizes_dict[pid]
        support = pattern_supports_dict[pid]
        decay = 1 / (1 + alpha ** (-size))
        itr = decay * support
        itr_scores[pid] = itr
    return itr_scores


def compute_pattern_metrics(filename):
    """
    计算模式的支持度和大小

    参数:
        filename: 模式文件路径

    返回:
        support_dict: 键为模式索引，值为支持度的字典
        size_dict: 键为模式索引，值为模式大小的字典
    """
    support_dict = defaultdict(int)
    size_dict = defaultdict(int)
    current_index = -1  # 模式索引，从0开始
    current_size = 0  # 当前模式的大小计数器

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            # 检测支持度行（纯数字）
            if line.isdigit():
                # 如果是新的支持度行，说明上一个模式已结束
                if current_index >= 0:
                    size_dict[current_index] = current_size

                # 开始处理新模式
                current_index += 1
                support_dict[current_index] = int(line)
                current_size = 0  # 重置大小计数器
            else:
                # 统计顶点和边的行数，累加模式大小
                current_size += 1

    # 处理最后一个模式
    if current_index >= 0:
        size_dict[current_index] = current_size

    return dict(support_dict), dict(size_dict)


def calculate_pairwise_accuracy_optimized(test_labels, ranked_ids):
    """客观指标实验，相对次序与用户真实次序相比"""
    # 统计每个类别的样本在ranked_ids中的位置索引
    class_positions = defaultdict(list)
    for idx, id_ in enumerate(ranked_ids):
        class_positions[test_labels[id_]].append(idx)

    # 初始化正确对数
    correct_pairs = 0
    total_pairs = 0

    # 遍历所有类别组合
    classes = sorted(class_positions.keys())
    for i in range(len(classes)):
        for j in range(i, len(classes)):
            class_i = classes[i]
            class_j = classes[j]

            # 同类情况：顺序默认正确
            if class_i == class_j:
                n = len(class_positions[class_i])
                correct_pairs += n * (n - 1) // 2  # 组合数 C(n,2)
            # 不同类且 class_i < class_j：所有 class_i 样本在 class_j 前正确
            else:
                for pos_i in class_positions[class_i]:
                    for pos_j in class_positions[class_j]:
                        if pos_i < pos_j:
                            correct_pairs += 1

            # 更新总对数
            if class_i == class_j:
                total_pairs += n * (n - 1) // 2
            else:
                total_pairs += len(class_positions[class_i]) * len(class_positions[class_j])

    accuracy = correct_pairs / total_pairs
    return accuracy


def calculate_segment_accuracy(test_labels, ranked_ids):

    print(f"test_labels is {test_labels}")
    class_distribution = Counter(test_labels.values())
    print("测试集类别分布:")
    for class_id, count in class_distribution.items():
        print(f"类别 {class_id}: {count} 个样本")

    correct_total = 0
    start_idx = 0
    results = {}

    # 按类别顺序分段处理
    for class_id, count in class_distribution.items():
        true_class_ids = {id_ for id_, label in test_labels.items() if label == class_id}

        end_idx = start_idx + count
        segment_ids = set(ranked_ids[start_idx:end_idx])

        correct = len(segment_ids & true_class_ids)
        correct_total += correct

        results[class_id] = correct / count
        start_idx = end_idx  # 更新下一段的起始索引

    overall_accuracy = correct_total / len(ranked_ids)
    return overall_accuracy


def compute_ece(true_labels, probs, n_bins=10):
    """
    计算预期校准误差（ECE）
    Args:
        true_labels: 真实标签（整数，如 [0, 1, 0, ...]）
        probs: 模型预测的最大概率值（shape=[n_samples]）
        n_bins: 分桶数量
    Returns:
        ece: 预期校准误差
    """
    # 将预测概率分桶并计算实际准确率
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # 计算每个桶的acc和conf
    ece = 0.0
    for bin_idx in range(n_bins):
        mask = (bin_indices == bin_idx)
        if np.sum(mask) > 0:
            acc = np.mean(true_labels[mask])
            conf = np.mean(probs[mask])
            ece += np.abs(acc - conf) * np.sum(mask) / len(probs)
    return ece


def random_test(cluster, budget, ndata, sortedCluster, frequent_patterns, num_iterations, n_class, alpha, graph, tau):
    """执行增量训练并测试模型准确度"""
    n_budget = budget
    data = ndata.copy()
    n_sortedCluster = copy.deepcopy(sortedCluster)
    m = n_class

    data_scaled = StandardScaler().fit_transform(data)
    reduced_data = TSNE(n_components=2).fit_transform(data_scaled)
    # reduced_data = PCA(n_components=2).fit_transform(data_scaled)

    real_labels, first_sample_indices = human_annotation_stim(data, cluster=cluster, m=m, reduced_data=reduced_data)

    # # 按代表性排序划分数据集
    sorted_ids = list(n_sortedCluster.keys())
    # train_count = int(len(n_sortedCluster) * 0.4)
    # last_part_start = int(len(n_sortedCluster) * 0.7)
    # train_ids = sorted_ids[:train_count] + sorted_ids[last_part_start:]
    # test_ids = sorted_ids[train_count:last_part_start]
    split_index = int(len(n_sortedCluster) * 0.7)
    train_ids = sorted_ids[:split_index]  # 训练集 ID
    test_ids = sorted_ids[split_index:]  # 测试集 ID
    train_sortedrepre = {k: n_sortedCluster[k] for k in n_sortedCluster if k in train_ids}

    # 与客观指标进行对比
    test_labels = {key: value for key, value in real_labels.items() if key in test_ids}
    support_dict, pattern_sizes_dict = compute_pattern_metrics(frequent_patterns)
    # 1支持度
    sorted_test_labels = {k: test_labels[k] for k in sorted(test_labels, key=lambda x: int(x))}
    # sorted_test_ids = list(sorted_test_labels.keys())
    # 2模式大小
    # pattern_sizes_sorted = {k: v for k, v in sorted(pattern_sizes_dict.items(), key=lambda item: item[1], reverse=True)
    #                         if k in test_ids}
    # sorted_test_ids = list(pattern_sizes_sorted.keys())
    # 3 itr指标
    # itr_scores = compute_itr_scores(pattern_sizes_dict, support_dict)
    # itr_scores_sorted = {k: v for k, v in sorted(itr_scores.items(), key=lambda item: item[1], reverse=True)
    #                         if k in test_ids}
    # sorted_test_ids = list(itr_scores_sorted.keys())

    # acc_seta_size_results = calculate_segment_accuracy(test_labels, sorted_test_ids)
    # acc_seta_size_results = calculate_pairwise_accuracy_optimized(test_labels, sorted_test_ids)
    # accuracy_results1 = []  # 记录每轮的精确度
    # accuracy_results1.append(acc_seta_size_results)

    # 随机打乱数据顺序以确保划分的随机性
    # items = list(n_sortedCluster.items())
    # random.shuffle(items)
    # shuffled_dict = dict(items)
    # random_ids = list(shuffled_dict.keys())
    # split_index = int(len(shuffled_dict) * 0.7)
    # train_ids = random_ids[:split_index]
    # test_ids = random_ids[split_index:]
    # train_sortedrepre = {k: n_sortedCluster[k] for k in n_sortedCluster if k in train_ids}

    # lbl_ids = first_sample_indices  # 覆盖所有类别
    # unlbl_ids = [x for x in train_sortedrepre.keys() if x not in lbl_ids]
    lbl_ids = list(train_sortedrepre.keys())[:budget]
    unlbl_ids = list(train_sortedrepre.keys())[budget:]
    lbl_labels = labels_annotation(lbl_ids, real_labels)

    # GCN
    data1 = copy.deepcopy(graph)
    model_gcn = GCN(input_dim=data1.x.shape[1], hidden_channels=16, output_dim=n_class)
    model_gcn, optimizer_gcn = train_GCN(model_gcn, data1, lbl_labels)
    pre_pseudo_labels, gcn_conf_results = predict_GCN(model_gcn, data1, unlbl_ids)
    accuracy_train = calculate_acc(unlbl_ids, real_labels, pre_pseudo_labels)
    print("ppl准确率:", accuracy_train)

    # # pam聚类
    # initial_medoids = np.random.choice(range(len(data)), size=n_class, replace=False)
    # kmedoids_instance = kmedoids(data, initial_medoids)
    # kmedoids_instance.process()
    # cluster_labels = kmedoids_instance.get_clusters()
    # medoids = kmedoids_instance.get_medoids()
    # cluster_samples = []
    # lbl_ids = []
    # for cluster_id, indices in enumerate(cluster_labels):
    #     medoid_index = medoids[cluster_id]
    #     cluster_samples.append((cluster_id, medoid_index, indices))
    #     lbl_ids.append(medoid_index)
    # lbl_labels = labels_annotation(lbl_ids, real_labels)
    # unlbl_ids = dely(lbl_ids, list(train_sortedrepre.keys()))
    # pseudo_labels, accuracy = assign_ppl(data, lbl_ids, unlbl_ids, lbl_labels, real_labels, reduced_data)

    labeled_kind = {}
    labeled_kind.update(lbl_labels)
    print(f"当轮学习样本为{labeled_kind}")

    # 开始训练
    print(f"第 1 轮监督学习：")
    pseudo_X_ids = []
    unlabeled_kind = {}
    X, y, un_X, un_y = labeled_X_y_and_unlabeled_X_y(data, lbl_ids, pseudo_X_ids, lbl_labels, unlabeled_kind)

    model = PatternClassificationNN(input_dim=data.shape[1], num_classes=n_class)
    model, optimizer = train_model_semi_sup(X, y, un_X, un_y, model)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # EWC
    # # 无蒸馏beta=0
    # initial_optimizer = optim.Adam(model.parameters(), lr=0.001)
    # model, optimizer = distill_train(teacher_model=model, student_model=model, X=X, y=y, un_X=un_X,
    #                    un_y=torch.tensor([]) if un_y is None else un_y, optimizer=initial_optimizer, beta=0.0)
    # EWC
    # model, optimizer = regularized_train(model=model, optimizer=optimizer, X=X, y=y, un_X=un_X, un_y=un_y,
    #     ewc_regularizer=None, ewc_lambda=0, epochs=50)
    # 第1轮结束：生成第一个正则化器（记录第1轮的重要参数）
    # current_dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    # prev_ewc_regularizer = EWCRegularizer(model, current_dataset)

    train_results = {}
    results_uncert = {}
    model_predictions_train = model(torch.tensor(data[unlbl_ids], dtype=torch.float32))
    post_pseudo_labels = model_predictions_train.argmax(dim=1)
    uncertainty_scores = -torch.sum(F.softmax(model_predictions_train, dim=1) * F.log_softmax(model_predictions_train, dim=1), dim=1)
    for id, label,  uncertainty in zip(unlbl_ids, post_pseudo_labels.flatten(), uncertainty_scores):
        train_results[int(id)] = label.item()
        results_uncert[int(id)] = uncertainty.item()

    # 对测试集进行预测
    results = {}
    model_predictions = model(torch.tensor(data[test_ids], dtype=torch.float32))
    pseudo_labels = model_predictions.argmax(dim=1)
    for id, label in zip(test_ids, pseudo_labels.flatten()):
        results[id] = label.item()
    # 计算 Class Avg F1
    real_labels_unlabeled = np.array([real_labels[id] for id in test_ids if id in real_labels])
    class_avg_f1 = f1_score(real_labels_unlabeled, pseudo_labels.numpy(), average='macro')
    # 计算 ECE
    pred_labels = pseudo_labels.numpy()
    true_labels_binary = (pred_labels == real_labels_unlabeled).astype(np.float32)
    max_probs = F.softmax(model_predictions.detach(), dim=1).max(dim=1).values.numpy()
    ece = compute_ece(true_labels_binary, max_probs)
    print(f"Class Avg F1: {class_avg_f1:.4f}, ECE: {ece:.4f}")

    print(f" {len(lbl_ids)} 个样本有标签")
    accuracy_test = calculate_acc(test_ids, real_labels, results)
    print("测试集准确率:", accuracy_test)
    # print("模型预测结果:", results)
    accuracy_results = []  # 记录每轮的精确度
    accuracy_results.append(accuracy_test)

    thresh_per_class = torch.ones(n_class) * tau  # 初始阈值

    # 多轮交互训练，多轮由主动学习选择的标签和由上一轮得到的PPL一起训练

    # current_teacher = copy.deepcopy(model)  # 蒸馏
    for iteration in range(2, num_iterations+1):
        print(f"第 {iteration} 轮交互训练：")

        accuracy_results = []  # 只记录最后一轮的
        model, optimizer, lbl_ids, unlbl_ids, lbl_labels, labeled_kind, accuracy_test, train_results, results_uncert, \
            thresh_per_class, model_gcn, optimizer_gcn = interactive_training_with_mixed_selection(data, data1, test_ids,
            model, optimizer, lbl_ids, unlbl_ids, train_sortedrepre, real_labels, lbl_labels, labeled_kind, iteration,
            train_results, results_uncert, thresh_per_class, n_budget, m, alpha, reduced_data, model_gcn, optimizer_gcn)

        # model, optimizer, lbl_ids, unlbl_ids, lbl_labels, labeled_kind, accuracy_p, train_results, results_conf,\
        # results_uncert = interactive_training_with_mixed_selection(data, test_ids, model, optimizer, lbl_ids, unlbl_ids,
        # train_sortedrepre, real_labels, lbl_labels, labeled_kind, train_results, results_conf, results_uncert, n_budget,
        # m, alpha, reduced_data)

        # EWC
        # model, optimizer, lbl_ids, unlbl_ids, lbl_labels, labeled_kind, accuracy_test, train_results, results_uncert,\
        #     thresh_per_class, model_gcn, optimizer_gcn = interactive_training_with_mixed_selection(data, data1, test_ids,
        #      model=model, optimizer=optimizer, prev_ewc_regularizer=prev_ewc_regularizer, ewc_lambda=1000,
        #      selected_samples=lbl_ids, unlbl_data_ids=unlbl_ids, n_sortedCluster=train_sortedrepre,
        #     true_labels=real_labels, labeled_kind1=lbl_labels, labeled_kind=labeled_kind, iteration=iteration,
        #     train_results=train_results, results_uncert=results_uncert, thresholds=thresh_per_class, budget=n_budget,
        #     m=m, alpha=alpha, reduced_data=reduced_data, model_gcn=model_gcn, optimizer_gcn=optimizer_gcn)
        # X_current = data[lbl_ids]  # 累积到当前轮的所有有标签数据
        # y_current = [real_labels[id] for id in lbl_ids]
        # current_dataset = TensorDataset(torch.tensor(X_current, dtype=torch.float32),
        #                                 torch.tensor(y_current, dtype=torch.long))
        # prev_ewc_regularizer = EWCRegularizer(model, current_dataset)

        # # 知识蒸馏
        # current_student = PatternClassificationNN(input_dim=data.shape[1], num_classes=n_class)
        # current_student.load_state_dict(current_teacher.state_dict())
        # # 初始化学生模型优化器
        # current_optimizer = optim.Adam(current_student.parameters(), lr=0.001)
        # # 调用修改后的交互训练函数：传入教师+学生模型，返回训练好的学生模型（下一轮教师）
        # current_teacher, current_optimizer, lbl_ids, unlbl_ids, lbl_labels, labeled_kind, accuracy_test, train_results, results_uncert, \
        #     thresh_per_class, model_gcn, optimizer_gcn = interactive_training_with_mixed_selection(
        #     data=data, data1=data1, test_ids=test_ids, teacher_model=current_teacher, model=current_student,
        #     optimizer=current_optimizer, selected_samples=lbl_ids, unlbl_data_ids=unlbl_ids, n_sortedCluster=train_sortedrepre,
        #     true_labels=real_labels, labeled_kind1=lbl_labels, labeled_kind=labeled_kind, iteration=iteration,
        #     train_results=train_results, results_uncert=results_uncert, thresholds=thresh_per_class,
        #     budget=n_budget, m=m, alpha=alpha, reduced_data=reduced_data, model_gcn=model_gcn, optimizer_gcn=optimizer_gcn)

        accuracy_results.append(accuracy_test)
        # accuracy_results.append(class_avg_f1)

    return accuracy_test, accuracy_results


# def assign_ppl(data, lbl_ids, unlbl_ids, lbl_labels, real_labels, reduced_data):
#     lbl_data = data[lbl_ids]
#     unlbl_data = data[unlbl_ids]
#
#     distance_matrix = pairwise_distances(unlbl_data, lbl_data)  # 计算距离矩阵
#     min_dist_center_dict = {}
#     # 找到每个无标签样本最近中心样本对应的ID 无标签样本id：中心样本id
#     for i in range(distance_matrix.shape[0]):
#         min_dist_center_index = np.argmin(distance_matrix[i])
#         min_dist_center_dict[unlbl_ids[i]] = lbl_ids[min_dist_center_index]
#
#     # 标签传播
#     pseudo_labels = {}
#     for unlbl_id in min_dist_center_dict:
#         pseudo_labels[unlbl_id] = lbl_labels[min_dist_center_dict[unlbl_id]]
#
#     # ppl准确率
#     top_k = len(pseudo_labels)
#     top_k_predict = pred_k_get(pseudo_labels, top_k)
#     top_k_true = truth_k_get(real_labels, top_k_predict)
#     accuracy_p = top_k_accuracy(top_k_true, top_k_predict, top_k)
#     print("ppl准确率:", accuracy_p)
#
#     # # 1. 绘制伪标签的样本
#     # unlabeled_indices = np.asarray(list(pseudo_labels.keys()))
#     # plt.scatter(reduced_data[unlabeled_indices, 0],
#     #             reduced_data[unlabeled_indices, 1],
#     #             color='gray', alpha=0.5, s=50, label='Unlabeled Samples')
#     # # 2. 添加伪标签与真实标签的比较
#     # for index in unlabeled_indices:
#     #     pseudo_label = pseudo_labels[index]
#     #     true_label = real_labels[index]
#     #     # 根据伪标签和真实标签来决定标记
#     #     if pseudo_label == true_label:
#     #         plt.scatter(reduced_data[index, 0], reduced_data[index, 1],
#     #                     marker='+', color='green', s=25)
#     #     else:
#     #         plt.scatter(reduced_data[index, 0], reduced_data[index, 1],
#     #                     marker='x', color='black', s=25)
#     # # 3. 绘制有标签样本
#     # all_classes = list(range(m))
#     # true_label_values = np.array([lbl_labels[idx] for idx in lbl_ids])
#     # color_map = plt.get_cmap('Spectral', len(all_classes))
#     # # color_map = ['red', 'yellow', 'blue', 'pink', 'purple']
#     # for i, label in enumerate(all_classes):
#     #     indices = np.where(true_label_values == label)[0]
#     #     if len(indices) > 0:
#     #         plt.scatter(reduced_data[indices, 0], reduced_data[indices, 1],
#     #                     color=color_map(i), label=f'Class {label}', alpha=0.6, s=50)
#     #     else:
#     #         plt.scatter([], [], color=color_map(i), label=f'Class {label}', alpha=0.6, s=50)
#     #
#     # plt.title('Data Visualization with Pseudo Labels Comparison')
#     # plt.xlabel('PCA Component 1')
#     # plt.ylabel('PCA Component 2')
#     # plt.legend()
#     # plt.show()
#
#     return pseudo_labels, accuracy_p


def assign_ppl(data, model, optimizer, unlbl_ids, lbl_labels, real_labels, reduced_data):

    lbl_ids = list(lbl_labels.keys())
    model, optimizer = train_GCN(model, data, lbl_labels=lbl_labels, optimizer=optimizer)
    # 标签传播
    pre_pseudo_labels, gcn_conf_results = predict_GCN(model, data, unlbl_ids)

    # ppl准确率
    accuracy_train = calculate_acc(unlbl_ids, real_labels, pre_pseudo_labels)
    print("ppl准确率:", accuracy_train)

    # # 1. 绘制伪标签的样本
    # unlabeled_indices = np.asarray(list(pre_pseudo_labels.keys()))
    # plt.scatter(reduced_data[unlabeled_indices, 0],
    #             reduced_data[unlabeled_indices, 1],
    #             color='gray', alpha=0.5, s=50, label='Unlabeled Samples')
    # # 2. 添加伪标签与真实标签的比较
    # for index in unlabeled_indices:
    #     pseudo_label = pre_pseudo_labels[index]
    #     true_label = real_labels[index]
    #     # 根据伪标签和真实标签来决定标记
    #     if pseudo_label == true_label:
    #         plt.scatter(reduced_data[index, 0], reduced_data[index, 1],
    #                     marker='+', color='green', s=25)
    #     else:
    #         plt.scatter(reduced_data[index, 0], reduced_data[index, 1],
    #                     marker='x', color='black', s=25)
    # # 3. 绘制有标签样本
    # all_classes = list(range(5))
    # true_label_values = np.array([lbl_labels[idx] for idx in lbl_ids])
    # color_map = plt.get_cmap('Spectral', len(all_classes))
    # # color_map = ['red', 'yellow', 'blue', 'pink', 'purple']
    # for i, label in enumerate(all_classes):
    #     indices = np.where(true_label_values == label)[0]
    #     if len(indices) > 0:
    #         plt.scatter(reduced_data[indices, 0], reduced_data[indices, 1],
    #                     color=color_map(i), label=f'Class {label}', alpha=0.6, s=50)
    #     else:
    #         plt.scatter([], [], color=color_map(i), label=f'Class {label}', alpha=0.6, s=50)
    #
    # plt.title('Data Visualization with Pseudo Labels Comparison')
    # plt.xlabel('PCA Component 1')
    # plt.ylabel('PCA Component 2')
    # plt.legend()
    # plt.show()

    return pre_pseudo_labels, model, optimizer


# def assign_ppl(data, lbl_ids, unlbl_ids, lbl_labels, real_labels, reduced_data):
#     """使用 kmedoids (PAM算法) 进行标签传播"""
#
#     # 为无标签样本分配伪标签（基于最近 medoid）
#     pseudo_labels = {}
#     distance_matrix = pairwise_distances(data[unlbl_ids], data[lbl_ids])  # 计算所有样本到 medoids 的距离
#     for i in range(len(unlbl_ids)):
#         closest_medoid_idx = np.argmin(distance_matrix[i])
#         closest_medoid = lbl_ids[closest_medoid_idx]
#         pseudo_labels[i] = lbl_labels.get(closest_medoid, None)
#
#     top_k = len(pseudo_labels)
#     top_k_predict = pred_k_get(pseudo_labels, top_k)
#     top_k_true = truth_k_get(real_labels, top_k_predict)
#     accuracy_p = top_k_accuracy(top_k_true, top_k_predict, top_k)
#     print("PPL 准确率:", accuracy_p)
#
#     # 1. 绘制伪标签的样本
#     unlabeled_indices = np.asarray(list(pseudo_labels.keys()))
#     plt.scatter(reduced_data[unlabeled_indices, 0],
#                 reduced_data[unlabeled_indices, 1],
#                 color='gray', alpha=0.5, s=50, label='Unlabeled Samples')
#     # 2. 添加伪标签与真实标签的比较
#     for index in unlabeled_indices:
#         pseudo_label = pseudo_labels[index]
#         true_label = real_labels[index]
#         # 根据伪标签和真实标签来决定标记
#         if pseudo_label == true_label:
#             plt.scatter(reduced_data[index, 0], reduced_data[index, 1],
#                         marker='+', color='green', s=25)
#         else:
#             plt.scatter(reduced_data[index, 0], reduced_data[index, 1],
#                         marker='x', color='black', s=25)
#     # 3. 绘制有标签样本
#     all_classes = list(range(5))
#     true_label_values = np.array([lbl_labels[idx] for idx in lbl_ids])
#     color_map = plt.get_cmap('Spectral', len(all_classes))
#     # color_map = ['red', 'yellow', 'blue', 'pink', 'purple']
#     for i, label in enumerate(all_classes):
#         indices = np.where(true_label_values == label)[0]
#         if len(indices) > 0:
#             plt.scatter(reduced_data[indices, 0], reduced_data[indices, 1],
#                         color=color_map(i), label=f'Class {label}', alpha=0.6, s=50)
#         else:
#             plt.scatter([], [], color=color_map(i), label=f'Class {label}', alpha=0.6, s=50)
#
#     plt.title('Data Visualization with Pseudo Labels Comparison')
#     plt.xlabel('PCA Component 1')
#     plt.ylabel('PCA Component 2')
#     plt.legend()
#     plt.show()
#
#     return pseudo_labels, accuracy_p


def load_data(filepath):
    """加载数据"""
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    return data


def main():
    """主函数"""
    # datasets = ['twitter', 'twitch', 'skitter', 'mico', 'dblp']
    # datasets = ['Aviation']
    datasets = ['dblp']
    # kfiles = ['1000', '1500', '2000', '2500', '3000']
    # kfiles = ['1000', '1500', '2000', '2500']
    kfiles = ['3000']
    # kfiles = ['1000']

    # 定义每个数据集对应的簇数k=3000
    dataset_info = {
        'twitter': {'n_clusters': 8, 'threshold': 0.947},
        'twitch': {'n_clusters': 10, 'threshold': 0.95},
        'skitter': {'n_clusters': 6, 'threshold': 0.954},
        'mico': {'n_clusters': 7, 'threshold': 0.959},
        'dblp': {'n_clusters': 8, 'threshold': 0.95},
        'Aviation': {'n_clusters': 2, 'threshold': 0.955},
        'facebook': {'n_clusters': 5, 'threshold': 0.954}
    }
    # k=1000
    # dataset_info = {
    #     'twitter': {'n_clusters': 10},
    #     'twitch': {'n_clusters': 14},
    #     'skitter': {'n_clusters': 12},
    #     'mico': {'n_clusters': 6},
    #     'dblp': {'n_clusters': 9},
    #     'Aviation': {'n_clusters': 5}
    # }

    accuracy_all_results = []
    # 迭代每个数据集
    for kfile in kfiles:
        for dataset in datasets:
            info_graph = torch.load(f"D:/Python/learn_pytorch/pythonProject/my_test_data/{kfile}/{dataset}_1.pt")
            filepath = f"D:/Python/learn_pytorch/pythonProject/my_test_data/{kfile}/{dataset}_1.pkl"
            # frequent_patterns = f"my_test_data/{kfile}/{dataset}_1.txt"
            frequent_patterns = f"my_test_data/{kfile}/{dataset}_supp_{kfile}.txt"
            print(f"正在处理文件: {filepath}")

            data, sortedCluster = load_data(filepath)
            n_sortedCluster = dict(sortedCluster)

            # 可调参数
            num_iterations = 20
            n_class = 5
            alpha = 0.6

            n_clusters = dataset_info[dataset]['n_clusters']
            print(f"处理簇数: {n_clusters}")
            threshold = dataset_info[dataset]['threshold']
            # thresholds = [round(0.941 + i * 0.001, 3) for i in range(20)]
            # alphas = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            # batchsizes = [10, 15, 20]
            batchsizes = [5]
            for batch_size in batchsizes:
                # for n_clusters in range(1, 16):  # n_clusters 从 1 到 15
                #     print(f"处理簇数: {n_clusters}")
                # for threshold in thresholds:
                print(f"batch_size: {batch_size}")
                experiment_seed = 0  # 固定种子
                np.random.seed(experiment_seed)
                torch.manual_seed(experiment_seed)

                accuracy, accuracy_results = random_test(n_clusters, batch_size, data, n_sortedCluster,
                                                         frequent_patterns,
                                                         num_iterations=num_iterations, n_class=n_class,
                                                         alpha=alpha,
                                                         graph=info_graph, tau=threshold)
                print(f"固定阈值为 {threshold}，固定alpha {alpha} 的最终Top-k准确率: {accuracy}")
                accuracy_all_results.extend(accuracy_results)
                # accuracy_all_results.append(accuracy)

    # 文件名和目录
    # filename = f'EWCtau1.xlsx'
    # directory = 'data_results'
    # # 确保目录存在
    # os.makedirs(directory, exist_ok=True)
    # # 路径组合
    # output_path = os.path.join(directory, filename)
    # df = pd.DataFrame(accuracy_all_results)
    # df.to_excel(output_path, index=False)
    #
    # print("所有数据集处理完毕！")


if __name__ == "__main__":
    main()



