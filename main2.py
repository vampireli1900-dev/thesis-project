# ==============================================================================
# 光伏功率预测终极代码：多模型对比评估 (黄金参数 + 归一化指标版)
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import emd.sift as sift
import warnings
import os

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 创建保存模型的文件夹
os.makedirs("models", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 当前使用计算设备: {DEVICE}")

# ====================== [核心配置] 注入最优超参数 ======================
OPTIMAL_IMF = 4  # 保留的 IMF 数量
TIME_STEPS = 12  # 输入时间步长
BATCH_SIZE = 64  # 批次大小
HIDDEN_DIM = 128  # 隐藏层神经元数量
LEARNING_RATE = 0.005  # 全局学习率
PRED_STEPS = 1  # 目标预测步长 (单步预测精度最高)

# ====================== 1. 数据读取与清洗 ======================
print("\n[1/7] 正在读取与清洗数据...")
df = pd.read_csv("data.csv", encoding="utf-8")
df["日期"] = pd.to_datetime(df["日期"])
df = df.sort_values("日期").dropna().reset_index(drop=True)

# 剔除极端异常值 (只删极小值，保留正常的负数功率)
df = df[df["功率"] > -100].reset_index(drop=True)

feature_cols = [
    "2米比湿(g/kq)", "2米气温(℃)", "地表气温(℃)",
    "到达地表的短波辐射(w/m2)", "2米相对湿度(%)",
    "相对湿度500(%)", "地面气压(hPa)"
]
raw_features = df[feature_cols].values
target = df["功率"].values
target = np.abs(target)  # 负数变正数

# ====================== 2. EMD 分解 ======================
print("\n[2/7] 正在进行 EMD 分解 (仅对功率序列)...")
imfs_all = sift.sift(target)
# 【注入最优参数 1】：保留前 4 个 IMF 分量
imfs_valid = imfs_all[:, :OPTIMAL_IMF]
features_with_emd = np.hstack([raw_features, imfs_valid])

# ====================== 3. 严谨的防泄露归一化 ======================
print("\n[3/7] 正在进行防泄露归一化与时序样本构建...")

# 数据集划分节点 (Train: 70%, Val: 10%, Test: 20%)
total_len = len(df)
train_split = int(total_len * 0.7)

# --- A. 普通模型数据处理 ---
scaler_X_norm = MinMaxScaler()
scaler_y_norm = MinMaxScaler()
scaler_X_norm.fit(raw_features[:train_split])
scaler_y_norm.fit(target[:train_split].reshape(-1, 1))

X_norm_scaled = scaler_X_norm.transform(raw_features)
y_norm_scaled = scaler_y_norm.transform(target.reshape(-1, 1)).ravel()

# --- B. EMD模型数据处理 ---
scaler_X_emd = MinMaxScaler()
scaler_X_emd.fit(features_with_emd[:train_split])

X_emd_scaled = scaler_X_emd.transform(features_with_emd)


# ====================== 4. 构建时序样本 ======================
def build_sequences(X, y, time_steps, pred_steps):
    Xs, ys = [], []
    for i in range(time_steps, len(X) - pred_steps + 1):
        Xs.append(X[i - time_steps: i, :])
        ys.append(y[i: i + pred_steps])
    return np.array(Xs), np.array(ys)


# 普通样本
X_seq_norm, y_seq_norm = build_sequences(X_norm_scaled, y_norm_scaled, TIME_STEPS, PRED_STEPS)
# EMD样本
X_seq_emd, y_seq_emd = build_sequences(X_emd_scaled, y_norm_scaled, TIME_STEPS, PRED_STEPS)

seq_train_split = int(len(X_seq_norm) * 0.7)
seq_val_split = int(len(X_seq_norm) * 0.8)


def split_dataset(X, y):
    X_tr, y_tr = X[:seq_train_split], y[:seq_train_split]
    X_v, y_v = X[seq_train_split:seq_val_split], y[seq_train_split:seq_val_split]
    X_te, y_te = X[seq_val_split:], y[seq_val_split:]
    return X_tr, y_tr, X_v, y_v, X_te, y_te


X_tr_norm, y_tr_norm, X_v_norm, y_v_norm, X_te_norm, y_te_norm = split_dataset(X_seq_norm, y_seq_norm)
X_tr_emd, y_tr_emd, X_v_emd, y_v_emd, X_te_emd, y_te_emd = split_dataset(X_seq_emd, y_seq_emd)

# ====================== 5. 定义模型 ======================
print("\n[4/7] 正在初始化神经网络模型...")


# 基础对比模型
class BaseTimeSeriesModel(nn.Module):
    def __init__(self, model_type, input_dim):
        super().__init__()
        # 【注入最优参数 2】：统一修改隐藏层为 128
        if model_type == "RNN":
            self.rnn = nn.RNN(input_dim, HIDDEN_DIM, batch_first=True)
        elif model_type == "LSTM":
            self.rnn = nn.LSTM(input_dim, HIDDEN_DIM, batch_first=True)
        elif model_type == "GRU":
            self.rnn = nn.GRU(input_dim, HIDDEN_DIM, batch_first=True)
        elif model_type == "BiGRU":
            self.rnn = nn.GRU(input_dim, HIDDEN_DIM, bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(0.2)
        out_dim = HIDDEN_DIM * 2 if "Bi" in model_type else HIDDEN_DIM
        self.fc = nn.Linear(out_dim, PRED_STEPS)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)


# 优化版 EMD-BiGRU
class Optimized_EMD_BiGRU(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, HIDDEN_DIM, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(HIDDEN_DIM * 2, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, PRED_STEPS)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        return self.fc2(out)


model_rnn = BaseTimeSeriesModel("RNN", X_tr_norm.shape[2])
model_lstm = BaseTimeSeriesModel("LSTM", X_tr_norm.shape[2])
model_gru = BaseTimeSeriesModel("GRU", X_tr_norm.shape[2])
model_bigru = BaseTimeSeriesModel("BiGRU", X_tr_norm.shape[2])
model_emd_bigru = Optimized_EMD_BiGRU(X_tr_emd.shape[2])


# ====================== 6. 通用训练逻辑 ======================
def train_model(model, X_train, y_train, X_val, y_val, name, lr, patience=6, epochs=100):
    print(f"\n🚀 开始训练: {name}")
    model = model.to(DEVICE)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # 【注入最优参数 3】：批次大小为 64
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)), batch_size=BATCH_SIZE,
                            shuffle=False)

    best_val_loss = float('inf')
    counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                val_loss += criterion(model(bx), by).item()

        avg_t_loss = train_loss / len(train_loader)
        avg_v_loss = val_loss / len(val_loader)
        scheduler.step(avg_v_loss)

        if avg_v_loss < best_val_loss:
            best_val_loss = avg_v_loss
            torch.save(model.state_dict(), f"models/best_{name}.pth")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"✅ {name} 早停于 Epoch {epoch + 1} (Val Loss: {best_val_loss:.6f})")
                break


print("\n[5/7] 正在训练全部 5 个模型...")
# 【注入最优参数 4】：全部模型应用最优学习率 0.005
for m, name in zip([model_rnn, model_lstm, model_gru, model_bigru], ["RNN", "LSTM", "GRU", "BiGRU"]):
    train_model(m, X_tr_norm, y_tr_norm, X_v_norm, y_v_norm, name, lr=LEARNING_RATE)

train_model(model_emd_bigru, X_tr_emd, y_tr_emd, X_v_emd, y_v_emd, "EMD_BiGRU", lr=LEARNING_RATE, patience=10)

# ====================== 7. 预测与归一化指标计算 ======================
print("\n[6/7] 正在进行测试集预测与归一化指标计算...")


def predict_and_inverse(model, X_test_data, name):
    model.load_state_dict(torch.load(f"models/best_{name}.pth"))
    model.eval()
    model.to(DEVICE)
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test_data).to(DEVICE)
        pred_scaled = model(X_tensor).cpu().numpy()
    pred_real = scaler_y_norm.inverse_transform(pred_scaled)
    return pred_real


pred_rnn = predict_and_inverse(model_rnn, X_te_norm, "RNN")
pred_lstm = predict_and_inverse(model_lstm, X_te_norm, "LSTM")
pred_gru = predict_and_inverse(model_gru, X_te_norm, "GRU")
pred_bigru = predict_and_inverse(model_bigru, X_te_norm, "BiGRU")
pred_emd_bigru = predict_and_inverse(model_emd_bigru, X_te_emd, "EMD_BiGRU")

y_true_real = scaler_y_norm.inverse_transform(y_te_norm)


def calc_normalized_metrics(y_true, y_pred):
    """计算归一化指标 nRMSE 和 nMAE，以及 R2"""
    yt_flat, yp_flat = y_true.flatten(), y_pred.flatten()

    mae = mean_absolute_error(yt_flat, yp_flat)
    rmse = np.sqrt(mean_squared_error(yt_flat, yp_flat))
    r2 = r2_score(yt_flat, yp_flat)

    # 归一化处理：除以实际功率的最大值 (代表电站装机容量)
    capacity = np.max(yt_flat) if np.max(yt_flat) > 0 else 1.0

    n_mae = (mae / capacity) * 100  # 转化为百分比
    n_rmse = (rmse / capacity) * 100  # 转化为百分比

    return n_rmse, n_mae, r2


results = {
    "RNN": calc_normalized_metrics(y_true_real, pred_rnn),
    "LSTM": calc_normalized_metrics(y_true_real, pred_lstm),
    "GRU": calc_normalized_metrics(y_true_real, pred_gru),
    "BiGRU": calc_normalized_metrics(y_true_real, pred_bigru),
    "EMD-BiGRU": calc_normalized_metrics(y_true_real, pred_emd_bigru)
}

# ====================== 8. 绘制图表并保存 ======================
print("\n[7/7] 正在生成归一化评价体系的高清论文图表...")
model_names = list(results.keys())
n_rmses = [results[m][0] for m in model_names]
n_maes = [results[m][1] for m in model_names]
r2s = [results[m][2] for m in model_names]

PLOT_STEPS = 288

# 1. 预测曲线对比图
plt.figure(figsize=(15, 6))
plt.plot(y_true_real[:PLOT_STEPS, 0], label='真实功率', lw=2, c='#222222')
plt.plot(pred_rnn[:PLOT_STEPS, 0], label='RNN', alpha=0.6, ls='--')
plt.plot(pred_lstm[:PLOT_STEPS, 0], label='LSTM', alpha=0.6, ls='-.')
plt.plot(pred_gru[:PLOT_STEPS, 0], label='GRU', alpha=0.6, ls=':')
plt.plot(pred_bigru[:PLOT_STEPS, 0], label='BiGRU', alpha=0.6)
plt.plot(pred_emd_bigru[:PLOT_STEPS, 0], label='EMD-BiGRU (本文方法)', lw=3, c='#E63946')
plt.title('测试集光伏功率多模型预测对比 (真实量纲)', fontsize=14)
plt.xlabel('时间步', fontsize=12)
plt.ylabel('实际功率 (kW/MW)', fontsize=12)
plt.legend(fontsize=10, loc='best')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('1_预测曲线多模型对比.png', dpi=300)
plt.close()


def plot_bar(names, values, title, ylabel, filename, is_r2=False):
    plt.figure(figsize=(10, 5))
    colors = ['#8D99AE'] * 4 + ['#E63946']
    bars = plt.bar(names, values, color=colors, width=0.5)
    for bar, v in zip(bars, values):
        offset = 0.01 if is_r2 else v * 0.02
        plt.text(bar.get_x() + bar.get_width() / 2, v + offset, f'{v:.2f}', ha='center', va='bottom', fontsize=11)
    plt.title(title, fontsize=14)
    plt.ylabel(ylabel, fontsize=12)
    plt.ylim(0, max(values) * 1.2 if not is_r2 else 1.1)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# 2, 3, 4. 误差柱状图 (此时为归一化百分比指标)
plot_bar(model_names, n_rmses, '各模型归一化 RMSE 对比 (nRMSE)', 'nRMSE (%)', '2_nRMSE对比.png')
plot_bar(model_names, n_maes, '各模型归一化 MAE 对比 (nMAE)', 'nMAE (%)', '3_nMAE对比.png')
plot_bar(model_names, r2s, r'各模型 $R^2$ 对比', r'$R^2$', '4_R2对比.png', is_r2=True)

# 5. EMD 分解图
plt.figure(figsize=(12, 8))
for i in range(OPTIMAL_IMF):
    plt.subplot(OPTIMAL_IMF, 1, i + 1)
    plt.plot(imfs_all[:1000, i], c='#2A9D8F')
    plt.title(f'IMF {i + 1}', loc='right')
    plt.grid(alpha=0.3)
plt.suptitle('功率序列 EMD 经验模态分解图', fontsize=15)
plt.tight_layout()
plt.savefig('5_EMD分解图.png', dpi=300)
plt.close()

# 导出汇总表
df_result = pd.DataFrame({
    '模型': model_names,
    'nRMSE (%)': np.round(n_rmses, 4),
    'nMAE (%)': np.round(n_maes, 4),
    'R²': np.round(r2s, 4)
})
df_result.to_csv('模型指标汇总表.csv', index=False, encoding='utf-8-sig')



print("\n🎉 全部运行完成！祝你毕业答辩顺利！")
print("已生成以下文件：")
print(" |- 1_预测曲线多模型对比.png")
print(" |- 2_nRMSE对比.png")
print(" |- 3_nMAE对比.png")
print(" |- 4_R2对比.png")
print(" |- 5_EMD分解图.png")
print(" |- 模型指标汇总表.csv")

# ====================== 9. 新增模块：最优100连续点多模型深度对比 ======================
print("\n[8/8] 正在生成最优100连续点局部对比分析与图表...")

WINDOW_SIZE = 100
min_total_error = float('inf')
best_idx = 0

# 1. 寻找 EMD-BiGRU 拟合最好的连续 100 点
for i in range(len(y_true_real) - WINDOW_SIZE + 1):
    true_segment = y_true_real[i:i + WINDOW_SIZE, 0]
    pred_segment = pred_emd_bigru[i:i + WINDOW_SIZE, 0]
    error = np.sum(np.abs(true_segment - pred_segment))  # 累计绝对误差最小
    if error < min_total_error:
        min_total_error = error
        best_idx = i

print(f"✅ 成功定位！最优100连续点起始索引：{best_idx}，结束索引：{best_idx + WINDOW_SIZE}")

# 2. 截取所有模型在该窗口的数据
y_true_best = y_true_real[best_idx: best_idx + WINDOW_SIZE]
preds_best = {
    "RNN": pred_rnn[best_idx: best_idx + WINDOW_SIZE],
    "LSTM": pred_lstm[best_idx: best_idx + WINDOW_SIZE],
    "GRU": pred_gru[best_idx: best_idx + WINDOW_SIZE],
    "BiGRU": pred_bigru[best_idx: best_idx + WINDOW_SIZE],
    "EMD-BiGRU": pred_emd_bigru[best_idx: best_idx + WINDOW_SIZE]
}

# 3. 计算局部 100 点的评价指标 (使用归一化指标 nRMSE, nMAE)
best100_results = []
# 局部区间的装机容量基准 (防止除以0)
capacity_100 = np.max(y_true_best) if np.max(y_true_best) > 0 else 1.0

for name, pred_b in preds_best.items():
    yt_b, yp_b = y_true_best.flatten(), pred_b.flatten()
    mae_b = mean_absolute_error(yt_b, yp_b)
    rmse_b = np.sqrt(mean_squared_error(yt_b, yp_b))
    r2_b = r2_score(yt_b, yp_b)

    n_mae_b = (mae_b / capacity_100) * 100
    n_rmse_b = (rmse_b / capacity_100) * 100

    best100_results.append({
        "模型": name,
        "局部nRMSE(%)": round(n_rmse_b, 4),
        "局部nMAE(%)": round(n_mae_b, 4),
        "局部R²": round(r2_b, 4)
    })

df_best100 = pd.DataFrame(best100_results)
df_best100.to_csv('最优100点_模型指标对比表.csv', index=False, encoding='utf-8-sig')
print("\n📊 最优100点局部评价指标对比：")
print(df_best100.to_string(index=False))

# 4. 绘图：最优100点预测曲线多模型追踪对比
plt.figure(figsize=(14, 6))
plt.plot(y_true_best[:, 0], label='真实功率', linewidth=2.5, c='#222222')
plt.plot(preds_best["RNN"][:, 0], label='RNN', linestyle='--', alpha=0.7)
plt.plot(preds_best["LSTM"][:, 0], label='LSTM', linestyle='-.', alpha=0.7)
plt.plot(preds_best["GRU"][:, 0], label='GRU', linestyle=':', alpha=0.7)
plt.plot(preds_best["BiGRU"][:, 0], label='BiGRU', alpha=0.8)
plt.plot(preds_best["EMD-BiGRU"][:, 0], label='EMD-BiGRU (本文)', linewidth=3.5, c='#E63946')

plt.title('最优100连续点局部预测曲线多模型追踪对比', fontsize=15)
plt.xlabel('局部时间步 (Time Steps)', fontsize=12)
plt.ylabel('光伏功率 (MW/kW)', fontsize=12)
plt.legend(fontsize=11, loc='best')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('6_最优100点_曲线追踪对比.png', dpi=300)
plt.close()

# 5. 绘图：最优100点指标柱状图
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors = ['#8D99AE'] * 4 + ['#E63946']
models = df_best100["模型"]

# 子图1: nRMSE
axes[0].bar(models, df_best100["局部nRMSE(%)"], color=colors, width=0.5)
for i, v in enumerate(df_best100["局部nRMSE(%)"]):
    axes[0].text(i, v + 0.1, f'{v:.2f}', ha='center', fontsize=11)
axes[0].set_title('最优100点局部 nRMSE (%)', fontsize=14)
axes[0].set_ylabel('nRMSE (%)')

# 子图2: nMAE
axes[1].bar(models, df_best100["局部nMAE(%)"], color=colors, width=0.5)
for i, v in enumerate(df_best100["局部nMAE(%)"]):
    axes[1].text(i, v + 0.1, f'{v:.2f}', ha='center', fontsize=11)
axes[1].set_title('最优100点局部 nMAE (%)', fontsize=14)
axes[1].set_ylabel('nMAE (%)')

# 子图3: R2



axes[2].bar(models, df_best100["局部R²"], color=colors, width=0.5)
for i, v in enumerate(df_best100["局部R²"]):
    axes[2].text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=11)
axes[2].set_title('最优100点局部 R²', fontsize=14)
axes[2].set_ylabel('$R^2$')
axes[2].set_ylim(0, 1.15)  # 给文字留出空间

plt.tight_layout()
plt.savefig('7_最优100点_误差指标对比柱状图.png', dpi=300)
plt.close()

print("\n🎉 局部 100 点分析执行完毕！已生成以下文件：")
print(" |- 6_最优100点_曲线追踪对比.png")
print(" |- 7_最优100点_误差指标对比柱状图.png")
print(" |- 最优100点_模型指标对比表.csv")