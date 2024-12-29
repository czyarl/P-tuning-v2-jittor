import os
import json
import random

with open("alpaca_data.json", "r") as f:
    data = json.load(f)

random.shuffle(data)
split_point = int(len(data)*0.8)
eval_data = data[split_point:]
train_data = data[:split_point]
# eval_data = data[-200:]
# train_data = data[:20]
print(f"eval_data len = {len(eval_data)}")
print(f"train_data len = {len(train_data)}")

with open("alpaca_data_train.json", "w") as f:
    json.dump(train_data, f)
with open("alpaca_data_eval.json", "w") as f:
    json.dump(eval_data, f)