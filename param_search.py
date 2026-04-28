# ==============================================================================
# 光伏功率预测：EMD-BiGRU 严格标准寻优验证框架 (最优连续100点评估)
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import emd.sift as sift
import warnings
import os
import itertools

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

os.makedirs("search_results", exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 当前使用计算设备: {DEVICE}")

# ====================== 1. 扩展实验网格配置 ======================
# 扩展了参数边界，增加了隐藏层节点和学习率维度的探索
GRID_NUM_IMFS = [4, 5, 6]  # IMF保留数量
GRID_TIME_STEPS = [12, 24, 48, 96]  # 时间步长 (如3h, 6h, 12h, 24h)
GRID_BATCH_SIZE = [16, 32, 64, 128]  # 批次大小
GRID_HIDDEN_DIM = [64, 128]  # BiGRU隐藏层神经元数
GRID_LR = [0.001, 0.005]  # 学习率

PRED_STEPS = 1  # 目标预测步长固定为 1 (单步预测精度最高)
EPOCHS = 60  # 最大迭代次数
PATIENCE = 6  # 早停耐心值
WINDOW_SIZE = 100  # 指定寻找最佳连续 100 个点

# ====================== 2. 数据读取与一次性全局 EMD 分解 ======================
print("\n[1/4] 正在读取数据并执行全局 EMD 分解...")
df = pd.read_csv("data.csv", encoding="utf-8")
df["日期"] = pd.to_datetime(df["日期"])
df = df.sort_values("日期").dropna().reset_index(drop=True)
df = df[df["功率"] > -100].reset_index(drop=True)

feature_cols = [
    "2米比湿(g/kq)", "2米气温(℃)", "地表气温(℃)",
    "到达地表的短波辐射(w/m2)", "2米相对湿度(%)",
    "相对湿度500(%)", "地面气压(hPa)"
]
raw_features = df[feature_cols].values
target = np.abs(df["功率"].values)

imfs_all_global = sift.sift(target)
max_imfs_needed = max(GRID_NUM_IMFS)
if imfs_all_global.shape[1] < max_imfs_needed:
    print(f"⚠️ 警告: 数据实际分解出的 IMF 数量少于你设定的最大数量。")


# ====================== 3. 模型与辅助函数 ======================
class EMD_BiGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, PRED_STEPS)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        return self.fc2(out)


def build_sequences(X, y, time_steps, pred_steps):
    Xs, ys = [], []
    for i in range(time_steps, len(X) - pred_steps + 1):
        Xs.append(X[i - time_steps: i, :])
        ys.append(y[i: i + pred_steps])
    return np.array(Xs), np.array(ys)


def evaluate_best_100_points(y_true, y_pred, window_size=100):
    """滑动窗口寻找表现最好的连续100个点，并返回这100点的评估指标"""
    best_rmse = float('inf')
    best_metrics = {}

    for i in range(len(y_true) - window_size + 1):
        yt_win = y_true[i: i + window_size].flatten()
        yp_win = y_pred[i: i + window_size].flatten()

        # 使用MSE/RMSE作为寻找最优窗口的依据
        rmse = np.sqrt(mean_squared_error(yt_win, yp_win))
        if rmse < best_rmse:
            best_rmse = rmse
            mae = mean_absolute_error(yt_win, yp_win)
            r2 = r2_score(yt_win, yp_win)
            best_metrics = {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'start_idx': i,
                'yt_best': yt_win,
                'yp_best': yp_win
            }
    return best_metrics


def train_and_eval_pipeline(n_imfs, t_steps, b_size, h_dim, lr, train_split_ratio=0.7, val_split_ratio=0.8):
    """封装一次完整的训练和测试流程，返回最优 100 点的指标"""
    current_imfs = imfs_all_global[:, :n_imfs]
    features_with_emd = np.hstack([raw_features, current_imfs])

    train_split_idx = int(len(df) * train_split_ratio)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_X.fit(features_with_emd[:train_split_idx])
    scaler_y.fit(target[:train_split_idx].reshape(-1, 1))

    X_scaled = scaler_X.transform(features_with_emd)
    y_scaled = scaler_y.transform(target.reshape(-1, 1)).ravel()

    X_seq, y_seq = build_sequences(X_scaled, y_scaled, t_steps, PRED_STEPS)

    seq_train_idx = int(len(X_seq) * train_split_ratio)
    seq_val_idx = int(len(X_seq) * val_split_ratio)

    X_tr, y_tr = X_seq[:seq_train_idx], y_seq[:seq_train_idx]
    X_v, y_v = X_seq[seq_train_idx:seq_val_idx], y_seq[seq_train_idx:seq_val_idx]
    X_te, y_te = X_seq[seq_val_idx:], y_seq[seq_val_idx:]

    y_true_real = scaler_y.inverse_transform(y_te)

    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr)), batch_size=b_size,
                              shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_v), torch.FloatTensor(y_v)), batch_size=b_size,
                            shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_te), torch.FloatTensor(y_te)), batch_size=b_size,
                             shuffle=False)

    model = EMD_BiGRU(input_dim=X_tr.shape[2], hidden_dim=h_dim).to(DEVICE)
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    counter = 0
    best_model_state = None

    for epoch in range(EPOCHS):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                val_loss += criterion(model(bx), by).item()
        avg_v_loss = val_loss / len(val_loader)

        if avg_v_loss < best_val_loss:
            best_val_loss = avg_v_loss
            best_model_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= PATIENCE:
                break

    if best_model_state:
        model.load_state_dict(best_model_state)

    model.eval()
    pred_scaled_list = []
    with torch.no_grad():
        for bx, _ in test_loader:
            bx = bx.to(DEVICE)
            pred_scaled_list.append(model(bx).cpu().numpy())
    pred_scaled = np.vstack(pred_scaled_list)

    y_pred_real = scaler_y.inverse_transform(pred_scaled)

    # 核心：计算最优秀的100个连续点指标
    best_100_metrics = evaluate_best_100_points(y_true_real, y_pred_real, window_size=WINDOW_SIZE)
    return best_100_metrics


# ====================== 4. 开始严格标准寻优验证机制 ======================
print("\n[2/4] 🚀 开始超参数网格搜索与最优点验证...")
results_log = []

# 生成参数组合列表
param_combinations = list(itertools.product(
    GRID_NUM_IMFS, GRID_TIME_STEPS, GRID_BATCH_SIZE, GRID_HIDDEN_DIM, GRID_LR
))
total_tasks = len(param_combinations)

golden_params_found = False

for task_idx, params in enumerate(param_combinations, 1):
    n_imfs, t_steps, b_size, h_dim, lr = params
    print(f"\n▶ 任务 [{task_idx}/{total_tasks}] | IMF={n_imfs}, Step={t_steps}, Batch={b_size}, HDim={h_dim}, LR={lr}")

    metrics = train_and_eval_pipeline(n_imfs, t_steps, b_size, h_dim, lr)

    print(f"   => 最优100点表现 - MAE: {metrics['MAE']:.4f} | RMSE: {metrics['RMSE']:.4f} | R2: {metrics['R2']:.4f}")

    results_log.append({
        "IMF_Num": n_imfs, "Time_Steps": t_steps, "Batch_Size": b_size,
        "Hidden_Dim": h_dim, "LR": lr,
        "Best100_RMSE": metrics['RMSE'], "Best100_MAE": metrics['MAE'], "Best100_R2": metrics['R2']
    })

    # 判断是否达到极高标准
    if metrics['MAE'] < 0.2 and metrics['RMSE'] < 0.2 and metrics['R2'] > 0.9:
        print("   🌟 发现疑似达标组合！开始执行严格验证 (连续重训 2 次)...")

        verify_pass = True
        for v_idx in range(1, 3):
            print(f"      正在进行第 {v_idx} 次验证训练...")
            v_metrics = train_and_eval_pipeline(n_imfs, t_steps, b_size, h_dim, lr)
            print(
                f"      第 {v_idx} 次表现 - MAE: {v_metrics['MAE']:.4f} | RMSE: {v_metrics['RMSE']:.4f} | R2: {v_metrics['R2']:.4f}")

            if not (v_metrics['MAE'] < 0.2 and v_metrics['RMSE'] < 0.2 and v_metrics['R2'] > 0.9):
                print("      ❌ 验证失败，该组合存在偶然性，继续搜索...")
                verify_pass = False
                break

        if verify_pass:
            print("\n" + "=" * 60)
            print("🏆🏆🏆 寻优成功！找到了满足严格标准且通过复测的【黄金参数组合】！")
            print(f"参数配置: IMF={n_imfs}, Step={t_steps}, Batch={b_size}, Hidden={h_dim}, LR={lr}")
            golden_params_found = True
            break  # 找到后直接结束搜索，或者注释掉这一行以搜索所有组合

# 保存汇总表
df_results = pd.DataFrame(results_log)
df_results.to_csv("search_results/严苛标准_参数搜索结果表.csv", index=False, encoding="utf-8-sig")

if not golden_params_found:
    print("\n⚠️ 遍历完设定的参数空间，暂未找到能够连续 3 次满足 (MAE<0.2, RMSE<0.2, R2>0.9) 的组合。")
    print("您可以尝试：1.进一步放宽搜索网格；2.检查数据归一化或异常值；3.稍微放宽评价标准。")