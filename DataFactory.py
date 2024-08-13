import ipaddress

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import os
from torch.utils.data import DataLoader, TensorDataset, random_split

def ip_to_int(ip):
    try:
        return int(ipaddress.ip_address(ip))
    except ValueError:
        return np.nan


def sliding_window(data, labels, window_size, step_size):
    segments = []
    segment_labels = []
    for start in range(0, len(data) - window_size + 1, step_size):
        end = start + window_size
        segment = data[start:end]
        segments.append(segment)
        segment_labels.append(labels[end - 1])
    segments = np.array(segments)
    segment_labels = np.array(segment_labels)

    # 验证所有样本是否都被覆盖
    covered_samples = len(segment_labels) * step_size + window_size - step_size
    if covered_samples == len(data):
        print("All samples are covered.")
    else:
        print(f"Covered samples: {covered_samples}, Total samples: {len(data)}")
    return segments, segment_labels

# 数据预处理函数
def preprocess_data(data, labels, window_size, step_size):
    segments, segment_labels = sliding_window(data, labels, window_size, step_size)
    return torch.tensor(segments, dtype=torch.float32), torch.tensor(segment_labels, dtype=torch.float32)

def LoadDarkNetByDate():
    # 读取原始数据文件
    df = pd.read_csv('DataSet/DarkNet/Darknet.CSV')

    # 过滤非Tor和NonTor的数据
    df = df[df['Label'].isin(['Tor', 'Non-Tor'])]

    # 将字符串标签替换为数值标签
    df['Label'] = df['Label'].apply(lambda x: 0 if x == 'Non-Tor' else 1)

    # 处理缺失值（用列的平均值填补）
    df = df.fillna(df.mean())

    # 替换无穷大和极大值
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # 对TimeStamp列进行排序
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Date'] = df['Timestamp'].dt.date
    # df = df.sort_values(by='Timestamp')

    # 选择特定日期的数据作为pretrain_dataset
    pretrain_date = pd.to_datetime('2015-07-13').date()
    pretrain_df = df[df['Date'] == pretrain_date]

    # 选择某日的数据作为测试集
    test_date = pd.to_datetime('2016-02-25').date()
    test_df = df[df['Date'] == test_date]

    # 剩余的数据用作训练集
    train_df = df.drop(pretrain_df.index).drop(test_df.index)

    # 去掉不必要的字符串列
    pretrain_df = pretrain_df.drop(columns=['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Date', 'Label.1'])
    train_df = train_df.drop(columns=['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Date', 'Label.1'])
    test_df = test_df.drop(columns=['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Date', 'Label.1'])

    # 数据标准化
    scaler = MinMaxScaler()

    # 处理预训练集数据
    pretrain_labels = pretrain_df['Label'].values
    pretrain_features = pretrain_df.drop(columns=['Label']).values
    pretrain_features_scaled = scaler.fit_transform(pretrain_features)

    # 处理训练集数据
    train_labels = train_df['Label'].values
    train_features = train_df.drop(columns=['Label']).values
    train_features_scaled = scaler.fit_transform(train_features)

    # 处理测试集数据
    test_labels = test_df['Label'].values
    test_features = test_df.drop(columns=['Label']).values
    test_features_scaled = scaler.fit_transform(test_features)

    # 使用滑动窗口生成片段
    window_size = 30
    step_size = 1

    pretrain_data, pretrain_labels = preprocess_data(pretrain_features_scaled, pretrain_labels, window_size, step_size)
    train_data, train_labels = preprocess_data(train_features_scaled, train_labels, window_size, step_size)
    test_data, test_labels = preprocess_data(test_features_scaled, test_labels, window_size, step_size)

    # 创建 TensorDataset
    pretrain_dataset = TensorDataset(pretrain_data, pretrain_labels)
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    # 创建 DataLoader
    batch_size = 6400
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    torch.save(pretrain_loader, 'DataLoader/DarkNet_pretrain_loader.pth')
    torch.save(train_loader, 'DataLoader/DarkNet_train_loader.pth')
    torch.save(test_loader, 'DataLoader/DarkNet_test_loader.pth')

    # 保存 remaining_data.shape[2]
    with open('DataLoader/DarkNet_shape', 'w') as f:
        f.write(str(test_data.shape[2]))

    return pretrain_loader, train_loader, test_loader, test_data.shape[2]


def LoadLoader(file_path_prefix):
    # 加载 DataLoader
    pretrain_loader = torch.load(file_path_prefix + '_pretrain_loader.pth')
    train_loader = torch.load(file_path_prefix + '_train_loader.pth')
    test_loader = torch.load(file_path_prefix + '_test_loader.pth')

    # 加载 remaining_data.shape[2]
    with open(file_path_prefix + '_shape', 'r') as f:
        remaining_data_shape = int(f.read().strip())

    print("Len of pretrain_loader", len(pretrain_loader))
    print("Len of train_loader", len(train_loader))
    print("Len of test_loader", len(test_loader))
    return pretrain_loader, train_loader, test_loader, remaining_data_shape

if __name__ == '__main__':
    LoadDarkNetByDate()