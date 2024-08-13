from sklearn.ensemble import RandomForestClassifier
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

from model import LSTMVAE
from LossDatabase import LossDatabase
from DataFactory import LoadDoHBrwDataNewest, LoadDoHBrw, LoadDarkNet, LoadDarkNetNew, loadDroid, loadIOV, LoadLoader, LoadDarkNetByDate, LoadIomtData
from utils import *

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# 初始化LossDatabase
loss_db = LossDatabase()

# 初始化有监督模型
rf = RandomForestClassifier(n_estimators=100, random_state=42)

pretrain_loader, train_loader, test_loader, input_shape = LoadDarkNetByDate()

def compute_losses(model, loss_db, data_loader):
    model.eval()
    all_losses = []
    with torch.no_grad():
        for batch in data_loader:
            data, _ = batch
            data = data.to(model.device)
            x_hat, mu, logvar = model(data)
            for i in range(data.size(0)):
                sample_loss = model.loss_function(x_hat[i], data[i], mu[i], logvar[i])
                all_losses.append(sample_loss.item())
                # 存入Loss database
                loss_db.add_loss(sample_loss, 0)
    return all_losses

def train_model_with_early_stop(model, train_loader, val_loader, epochs=50, lr=0.001, patience=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            data, _ = batch
            data = data.to(model.device)
            optimizer.zero_grad()
            x_hat, mu, logvar = model(data)
            loss = model.loss_function(x_hat, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                data, _ = batch
                data = data.to(model.device)
                x_hat, mu, logvar = model(data)
                loss = model.loss_function(x_hat, data, mu, logvar)

                val_loss += loss.item()

        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader.dataset)}, Validation Loss: {val_loss}")

        # Early stopping check
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')  # 保存最好的模型
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

def train_model_with_supervised_labels(model, train_loader, val_loader, rf, loss_db, epochs=50, lr=0.001, patience=5):
    # 初始化数据池
    supervised_data_pool = []
    supervised_labels_pool = []

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        remaining_epochs = epochs - epoch
        print(f"Remaining Epochs: {remaining_epochs}")

        model.train()
        train_loss = 0
        total_batches = len(train_loader)

        for batch_idx, batch in enumerate(train_loader):
            remaining_batches = total_batches - batch_idx
            print(f"Remaining Batches: {remaining_batches}")

            # batch label for plot
            data, batch_label = batch
            data = data.to(model.device)
            optimizer.zero_grad()
            x_hat, mu, logvar = model(data)
            loss = model.loss_function(x_hat, data, mu, logvar)

            train_loss += loss.item()

            plot_normal_loss = []
            plot_abnormal_loss = []
            # 每个batch计算T1、T2
            batch_losses = []
            for i in range(data.size(0)):
                sample_loss = model.loss_function(x_hat[i], data[i], mu[i], logvar[i])
                batch_losses.append(sample_loss)
                if batch_label[i] == 0:
                    plot_normal_loss.append(sample_loss.item())
                else:
                    plot_abnormal_loss.append(sample_loss.item())

            # 使用loss_db.find_intersections() 方法获取T1, T2
            print("Len of loss_db:")
            print(f"Normal Loss : {len(loss_db.normal_losses)}")
            print(f"Abnormal Loss : {len(loss_db.abnormal_losses)}")

            # 如果abnormal loss数量过少，则使用单个阈值
            if len(loss_db.abnormal_losses) <= 500:
                T1 = loss_db.get_threshold_normal_losses()

                labels = []
                for loss in batch_losses:
                    if loss <= T1:
                        labels.append(0)  # Normal
                        loss_db.add_loss(loss, 0)
                    else:
                        labels.append(1)  # Abnormal
                        loss_db.add_loss(loss, 1)
            else:
                # fix pad 修改处
                T1, T2 = loss_db.find_intersections()
                current_loss_dict = {}
                current_normal_loss = plot_normal_loss
                current_abnormal_loss = plot_abnormal_loss
                current_loss_dict["T1"] = T1
                current_loss_dict["T2"] = T2


                # 对Loss<=T1和Loss>=T2的样本进行标记
                labels = []

                for loss in batch_losses:
                    if loss <= T1:
                        labels.append(0)  # Normal
                        loss_db.add_loss(loss, 0)
                    elif loss >= T2:
                        labels.append(1)  # Abnormal
                        loss_db.add_loss(loss, 1)
                    else:
                        labels.append(-1)  # 不参与训练

                current_loss_dict["normal_loss"] = current_normal_loss
                current_loss_dict["abnormal_loss"] = current_abnormal_loss
                loss_db.time_loss.append(current_loss_dict)


            if supervised_data and supervised_labels:
                rf.fit(supervised_data, supervised_labels)  # 更新有监督模型

        zero_label_data = [data[i] for i in range(len(labels)) if labels[i] == 0]

        if zero_label_data:
            zero_label_data = torch.stack(zero_label_data)
            x_hat, mu, logvar = model(zero_label_data)
            zero_label_loss = model.loss_function(x_hat, zero_label_data, mu, logvar)
            zero_label_loss.backward()
            optimizer.step()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                data, _ = batch
                data = data.to(model.device)
                x_hat, mu, logvar = model(data)
                loss = model.loss_function(x_hat, data, mu, logvar)
                val_loss += loss.item()

        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader.dataset)}, Validation Loss: {val_loss}")

        # Early stopping check
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')  # 保存最好的模型
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

# 初始化模型
input_size = input_shape
hidden_size = 64
latent_size = 32
num_layers = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMVAE(input_size, hidden_size, latent_size, num_layers, device=device).to(device)

# 预训练模型
train_model_with_early_stop(model, pretrain_loader, pretrain_loader, epochs=5000, lr=0.001, patience=3)

# 加载最好的预训练模型
model.load_state_dict(torch.load('best_model.pth'))

compute_losses(model, loss_db, pretrain_loader)

# 继续训练并使用打标签后的样本进行训练
train_model_with_supervised_labels(model, train_loader, train_loader, rf, loss_db, epochs=1, lr=0.001, patience=10)

def compute_losses_and_last_data(model, data_loader):
    model.eval()
    all_losses = []
    ground_truth_labels = []
    last_data_points = []
    with torch.no_grad():
        for batch in data_loader:
            data, labels = batch
            data = data.to(model.device)
            x_hat, mu, logvar = model(data)
            for i in range(data.size(0)):
                sample_loss = model.loss_function(x_hat[i], data[i], mu[i], logvar[i])
                all_losses.append(sample_loss.item())
                ground_truth_labels.append(labels[i].item())
                last_data_points.append(data[i, -1, :].cpu().numpy())  # 提取时间序列的最后一个数据点
    return all_losses, ground_truth_labels, last_data_points


# 计算训练集的Loss和最后一个数据点
train_losses, _, train_last_data_points = compute_losses_and_last_data(model, train_loader)

# 设定阈值
threshold = loss_db.get_threshold_normal_losses()

# 计算训练集的Loss并打标签
def label_train_data(train_losses, threshold):
    train_labels_pred = ['normal' if loss <= threshold else 'abnormal' for loss in train_losses]
    train_labels_pred = [0 if label == 'normal' else 1 for label in train_labels_pred]
    return train_labels_pred


# 计算测试集的Loss和最后一个数据点
test_losses, test_labels, test_last_data_points = compute_losses_and_last_data(model, test_loader)
test_last_data_points = np.array(test_last_data_points)

# 使用测试集进行预测并评估有监督模型
test_labels_pred = rf.predict(test_last_data_points)

# 计算评价指标
print("RF Result: ")
evaluate_and_log_results(test_labels, test_labels_pred)

def evaluate_model(test_losses, test_labels, threshold):
    # 使用阈值进行分类
    test_predictions = ['normal' if loss <= threshold else 'abnormal' for loss in test_losses]
    ground_truth_labels = ['normal' if label == 0 else 'abnormal' for label in test_labels]

    test_predictions_num = [0 if loss <= threshold else 1 for loss in test_losses]
    ground_truth_labels_num = [0 if label == 0 else 1 for label in test_labels]

    print("LSTM-VAE Result: ")
    evaluate_and_log_results(ground_truth_labels_num, test_predictions_num)

    return ground_truth_labels, test_predictions

# 计算评价指标
ground_truth_labels, test_predictions = evaluate_model(test_losses, test_labels, threshold)

# 分离不同ground truth的Loss
normal_losses = [loss for loss, label in zip(test_losses, ground_truth_labels) if label == 'normal']
abnormal_losses = [loss for loss, label in zip(test_losses, ground_truth_labels) if label == 'abnormal']

# 计算测试集的Loss和最后一个数据点
test_losses, test_labels, test_last_data_points = compute_losses_and_last_data(model, test_loader)
test_last_data_points = np.array(test_last_data_points)

# 使用loss_db.find_intersections() 方法获取T1, T2
T1, T2 = loss_db.find_intersections()
print(f"T1: {T1}, T2: {T2}")

# 使用测试集进行预测
test_predictions = []
for i in range(len(test_losses)):
    if test_losses[i] <= T1:
        test_predictions.append(0)  # normal
    elif test_losses[i] >= T2:
        test_predictions.append(1)  # abnormal
    else:
        # 使用有监督模型预测
        test_predictions.append(rf.predict(test_last_data_points[i].reshape(1, -1))[0])

evaluate_and_log_results(test_labels, test_predictions)
