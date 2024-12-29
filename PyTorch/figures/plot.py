import json
import matplotlib.pyplot as plt

# 读取 JSON 文件中的 log_history 数据
with open("../checkpoints/cb-glm-prefix/trainer_state.json", "r") as file:
    data = json.load(file)

# 提取 log_history
log_history = data.get("log_history", [])

# 初始化存储数据的列表
epochs = []
losses = []
accuracies = []

# 遍历 log_history 提取 eval_loss 和 eval_accuracy
for log in log_history:
    if "epoch" in log and ("eval_loss" in log or "eval_accuracy" in log):
        epochs.append(log["epoch"])
        losses.append(log.get("eval_loss", None))  # 如果没有 eval_loss，值为 None
        accuracies.append(log.get("eval_accuracy", None))  # 如果没有 eval_accuracy，值为 None

# 绘制曲线
fig, ax1 = plt.subplots()

# 绘制 loss 曲线
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss", color="tab:red")
ax1.plot(epochs, losses, label="Loss", color="tab:red")
ax1.tick_params(axis="y", labelcolor="tab:red")

# 在同一图上添加 accuracy 曲线
ax2 = ax1.twinx()
ax2.set_ylabel("Accuracy", color="tab:blue")
ax2.plot(epochs, accuracies, label="Accuracy", color="tab:blue")
ax2.tick_params(axis="y", labelcolor="tab:blue")

# 添加标题和图例
plt.title("Training Metrics")
fig.tight_layout()
plt.show()