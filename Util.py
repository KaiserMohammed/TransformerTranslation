import torch
import torch.nn as nn
import torch.optim as optim
import json
import random
from sklearn.model_selection import train_test_split

def split_json_data(input_json_path, train_path, val_path, test_path, seed=42):
    """
    划分JSON格式的中英双语数据为8:1:1
    Args:
        input_json_path: 原始JSON文件路径（每行一个JSON对象）
        train_path/val_path/test_path: 划分后的数据保存路径
        seed: 随机种子（确保划分结果可复现）
    """
    # 1. 读取原始JSON数据（每行一个对象）
    data = []
    with open(input_json_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)  # 解析单个JSON对象
            # 检查是否包含中英文键（避免数据损坏）
            if "english" in item and "chinese" in item:
                data.append(item)

    # 2. 按6:2:2划分（先分训练集和临时集，再分验证集和测试集）
    random.seed(seed)
    # 第一步：划分训练集（80%）和 验证+测试集（20%）
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=seed)
    # 第二步：划分验证集（10%总数据）和测试集（10%总数据）
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=seed)

    # 3. 保存为JSON数组格式（模型读取需标准数组）
    def save_json(data, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    save_json(train_data, train_path)
    save_json(val_data, val_path)
    save_json(test_data, test_path)

    # 打印划分结果
    total = len(data)
    print(f"数据划分完成！")
    print(f"总数据量：{total} 条")
    print(f"训练集：{len(train_data)} 条（{len(train_data)/total*100:.1f}%）")
    print(f"验证集：{len(val_data)} 条（{len(val_data)/total*100:.1f}%）")
    print(f"测试集：{len(test_data)} 条（{len(test_data)/total*100:.1f}%）")

input_json = "./data/translation2019zh_train.json"  # 原始JSON文件（每行一个对象）
train_json = "./data/train_data.json"          # 划分后的训练集
val_json = "./data/val_data.json"              # 划分后的验证集
test_json = "./data/test_data.json"            # 划分后的测试集

split_json_data(input_json, train_json, val_json, test_json)