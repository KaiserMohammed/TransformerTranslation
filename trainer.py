import torch
import os
import json
import torch.nn as nn
from numba.core.cgutils import sizeof
from torch.utils.data import Dataset, DataLoader
from sentencepiece import SentencePieceProcessor  # 用于构建共享词汇表（需安装：pip install sentencepiece）
from TransformerTranslationModel import BidirectionalTransformer
from tqdm import tqdm
import re
import jieba
from collections import Counter

def clean_text(text, is_chinese=False):
    if is_chinese:
        text = re.sub(r'[^\u4e00-\u9fff，。、；：？！,.!?]', '', text)
        return text.strip()
    else:
        text = re.sub(r'[^a-zA-Z0-9,.!?;:\'\" ]', ' ', text)
        return ' '.join(text.strip().split())

def preprocess_chinese(text):
    if not text:
        return ""
    return ' '.join(jieba.lcut(text))

# 构建词汇表
def build_shared_vocab(data_paths, vocab_path, vocab_size=10000):
    """
    用SentencePiece构建中英共享词汇表
    data_paths: [train_json, val_json, test_json]（所有数据路径）
    vocab_path: 词汇表保存路径（.model文件）
    """
    all_text = []
    en_words = []  # 用于统计英文高频词
    zh_words = []  # 用于统计中文高频词

    for path in data_paths:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                # 处理英文
                en_clean = clean_text(item["english"], is_chinese=False)
                if en_clean:
                    all_text.append(en_clean)
                    en_words.extend(en_clean.split())

                # 处理中文（清洗+分词）
                zh_clean = clean_text(item["chinese"], is_chinese=True)
                zh_processed = preprocess_chinese(zh_clean)
                if zh_processed:
                    all_text.append(zh_processed)
                    zh_words.extend(zh_processed.split())

    # 统计高频词并加入自定义符号
    top_en = [f"▁{word}" for word, _ in Counter(en_words).most_common(200)]  # 英文加前缀▁
    top_zh = [word for word, _ in Counter(zh_words).most_common(200)]
    user_symbols = ["<en2zh>", "<zh2en>"] + top_en + top_zh

    # 保存临时文本
    temp_text_path = "temp_all_text.txt"
    with open(temp_text_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_text))

    # 3. 训练SentencePiece模型（生成共享词汇表）
    import sentencepiece as spm
    spm.SentencePieceTrainer.Train(
        input=temp_text_path,
        model_prefix=vocab_path,  # 输出：vocab_path.model 和 vocab_path.vocab
        vocab_size=vocab_size,
        model_type="word",  #
        pad_id=0, bos_id=1, eos_id=2, unk_id=3,  # 特殊token：<pad>=0, <bos>=1, <eos>=2, <unk>=3
        user_defined_symbols=["<en2zh>", "<zh2en>"], # 自定义翻译方向标记（可选）
        split_by_unicode_script=True,#分开中英文来分词
        split_by_whitespace = True,  # 先按空格拆分单词（确保“the”先作为整体统计频率）
        character_coverage = 1.0,  # 确保覆盖所有中英文字符（避免遗漏导致拆分异常）
        train_extremely_large_corpus = True
    )
    print(f"共享词汇表已保存到 {vocab_path}.model")
    return f"{vocab_path}.model"


# 数据构建，包含了中英互译的词汇表
class BiTextDataset(Dataset):
    def __init__(self, json_path, sp_model_path, max_seq_len=128):
        """
        Args:
            json_path: 划分后的JSON数据路径（train/val/test）
            sp_model_path: SentencePiece共享词汇表路径
            max_seq_len: 最大序列长度（超过截断，不足填充）
        """
        self.data = json.load(open(json_path, 'r', encoding='utf-8'))
        self.sp = SentencePieceProcessor()
        self.sp.Load(sp_model_path)  # 加载共享词汇表
        self.max_seq_len = max_seq_len
        self.pad_id = self.sp.pad_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        en_text = item["english"]
        zh_text = item["chinese"]

        # 1. 分词并添加BOS/EOS（英文→中文：en为src，zh为tgt；中文→英文反之）
        # 英文序列（添加BOS/EOS）
        en_ids = [self.bos_id] + self.sp.EncodeAsIds(en_text) + [self.eos_id]
        # 中文序列（添加BOS/EOS）
        zh_ids = [self.bos_id] + self.sp.EncodeAsIds(zh_text) + [self.eos_id]

        # 2. 截断或填充到max_seq_len
        en_ids = self._pad_truncate(en_ids)
        zh_ids = self._pad_truncate(zh_ids)

        return {
            "en_ids": torch.tensor(en_ids, dtype=torch.long),  # 英文序列（用于en2zh的src，zh2en的tgt）
            "zh_ids": torch.tensor(zh_ids, dtype=torch.long)  # 中文序列（用于en2zh的tgt，zh2en的src）
        }

    def _pad_truncate(self, ids):
        """截断或填充序列到max_seq_len"""
        if len(ids) > self.max_seq_len:
            return ids[:self.max_seq_len]
        else:
            return ids + [self.pad_id] * (self.max_seq_len - len(ids))


# 生成掩码
def collate_fn(batch, pad_id=0):
    """
    批量处理函数：生成模型所需的src、tgt和掩码
    batch: 从Dataset获取的批量数据
    """
    # 1. 提取英文和中文序列（batch_size, max_seq_len）
    en_ids = torch.stack([item["en_ids"] for item in batch])  # en2zh的src，zh2en的tgt
    zh_ids = torch.stack([item["zh_ids"] for item in batch])  # en2zh的tgt，zh2en的src

    # 2. 生成两种翻译方向的输入（en2zh 和 zh2en）
    # 方向1：英文→中文（src=en_ids，tgt=zh_ids）
    src_en2zh = en_ids
    tgt_en2zh = zh_ids[:, :-1]  # tgt输入：去掉最后一个token（避免泄露EOS）
    tgt_label_en2zh = zh_ids[:, 1:]  # tgt标签：去掉第一个token（与输入错位）

    # 方向2：中文→英文（src=zh_ids，tgt=en_ids）
    src_zh2en = zh_ids
    tgt_zh2en = en_ids[:, :-1]
    tgt_label_zh2en = en_ids[:, 1:]

    # 3. 生成掩码（调用模型的掩码函数，也可在此提前生成）
    # 注：模型forward时会自动生成掩码，此处仅返回基础数据
    return {
        # 英文→中文数据
        "en2zh": {
            "src": src_en2zh,
            "tgt": tgt_en2zh,
            "label": tgt_label_en2zh
        },
        # 中文→英文数据
        "zh2en": {
            "src": src_zh2en,
            "tgt": tgt_zh2en,
            "label": tgt_label_zh2en
        },
        "pad_id": pad_id
    }


# 模型训练
def train_model(args):
    # 1. 构建共享词汇表
    data_paths = [args.train_json, args.val_json, args.test_json]
    if os.path.exists("./shared_vocab.model"):
        sp_model_path = "./shared_vocab.model"
    else:
        sp_model_path = build_shared_vocab(data_paths, args.vocab_prefix, args.vocab_size)

    # 2. 加载数据集和DataLoader
    train_dataset = BiTextDataset(args.train_json, sp_model_path, args.max_seq_len)
    val_dataset = BiTextDataset(args.val_json, sp_model_path, args.max_seq_len)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda x: collate_fn(x, pad_id=train_dataset.pad_id)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda x: collate_fn(x, pad_id=val_dataset.pad_id)
    )

    # 3. 初始化双向Transformer模型
    sp = SentencePieceProcessor()
    sp.Load(sp_model_path)
    model = BidirectionalTransformer(
        shared_vocab_size=sp.GetPieceSize(),  # 共享词汇表大小
        dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len
    ).to(args.device)

    # 4. 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.pad_id)  # 忽略pad的损失

    # 5. 训练循环（同时训练en2zh和zh2en两个方向）
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        print(epoch)
        train_pbar = tqdm(train_loader, desc="Training", unit="batch")#添加训练进度条
        for batch in train_pbar:
            optimizer.zero_grad()
            #print(batch)
            # 初始化处理后的batch
            processed_batch = {}

            # 遍历batch中的所有键值对
            for key, value in batch.items():
                # 处理翻译方向的嵌套字典（en2zh/zh2en）
                if key in ['en2zh', 'zh2en']:
                    # 对每个方向的src/tgt/label张量移动设备
                    processed_batch[key] = {
                        subkey: subvalue.to(args.device)
                        for subkey, subvalue in value.items()
                    }
                # 处理pad_id（直接保留整数，无需移动设备）
                elif key == 'pad_id':
                    processed_batch[key] = value
                # 其他可能的键（如有）
                else:
                    # 如需处理其他键可在此扩展
                    processed_batch[key] = value

            # 用处理后的batch替换原batch
            batch = processed_batch

            # （1）训练英文→中文
            en2zh_data = batch["en2zh"]
            output_en2zh = model(
                src=en2zh_data["src"],
                tgt=en2zh_data["tgt"],
                translate_dir="en2zh"
            )
            # 计算损失（output: (batch, seq_len, vocab_size) → 展平为 (batch*seq_len, vocab_size)）
            loss_en2zh = criterion(
                output_en2zh.reshape(-1, output_en2zh.size(-1)),
                en2zh_data["label"].reshape(-1)
            )

            # （2）训练中文→英文
            zh2en_data = batch["zh2en"]
            output_zh2en = model(
                src=zh2en_data["src"],
                tgt=zh2en_data["tgt"],
                translate_dir="zh2en"
            )
            loss_zh2en = criterion(
                output_zh2en.reshape(-1, output_zh2en.size(-1)),
                zh2en_data["label"].reshape(-1)
            )

            # （3）总损失（两个方向损失平均）
            total_loss = (loss_en2zh + loss_zh2en) / 2
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item() * en2zh_data["src"].size(0)  # 累计批次损失
            train_pbar.set_postfix({"batch_loss": f"{total_loss.item():.4f}"})#当前损失
        # 计算epoch训练损失
        train_loss_avg = train_loss / len(train_loader)

        # 6. 验证循环
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation", unit="batch")
            for batch in val_pbar:
                processed_batch = {}

                # 遍历batch中的所有键值对
                for key, value in batch.items():
                    # 处理翻译方向的嵌套字典（en2zh/zh2en）
                    if key in ['en2zh', 'zh2en']:
                        # 对每个方向的src/tgt/label张量移动设备
                        processed_batch[key] = {
                            subkey: subvalue.to(args.device)
                            for subkey, subvalue in value.items()
                        }
                    # 处理pad_id（直接保留整数，无需移动设备）
                    elif key == 'pad_id':
                        processed_batch[key] = value
                    # 其他可能的键（如有）
                    else:
                        # 如需处理其他键可在此扩展
                        processed_batch[key] = value

                # 用处理后的batch替换原batch
                batch = processed_batch
                # 验证英文→中文
                en2zh_data = batch["en2zh"]
                output_en2zh = model(src=en2zh_data["src"], tgt=en2zh_data["tgt"], translate_dir="en2zh")
                loss_en2zh = criterion(output_en2zh.reshape(-1, output_en2zh.size(-1)), en2zh_data["label"].reshape(-1))

                # 验证中文→英文
                zh2en_data = batch["zh2en"]
                output_zh2en = model(src=zh2en_data["src"], tgt=zh2en_data["tgt"], translate_dir="zh2en")
                loss_zh2en = criterion(output_zh2en.reshape(-1, output_zh2en.size(-1)), zh2en_data["label"].reshape(-1))

                val_loss += ((loss_en2zh + loss_zh2en) / 2).item() * en2zh_data["src"].size(0)
                val_pbar.set_postfix({"batch_loss": f"{((loss_en2zh + loss_zh2en) / 2).item():.4f}"})
        val_loss_avg = val_loss / len(val_loader)

        # 打印训练信息
        print(f"Epoch [{epoch + 1}/{args.epochs}]")
        print(f"Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f}")

        # 保存最优模型
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(model.state_dict(), args.save_model_path)
            print(f"Best model saved to {args.save_model_path} (Val Loss: {best_val_loss:.4f})")


#使用参数化方法初始化模型并训练
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="双向Transformer翻译模型训练")
    # 数据路径参数
    parser.add_argument("--train_json", default="./data/train_data.json", help="训练集JSON路径")
    parser.add_argument("--val_json", default="./data/val_data.json", help="验证集JSON路径")
    parser.add_argument("--test_json", default="./data/test_data.json", help="测试集JSON路径")
    parser.add_argument("--vocab_prefix", default="shared_vocab", help="共享词汇表前缀（输出shared_vocab.model）")
    # 模型参数
    parser.add_argument("--model_dim", type=int, default=256,help="模型维度")
    parser.add_argument("--num_heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--num_layers", type=int, default=6, help="编码器/解码器层数")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout概率")
    parser.add_argument("--vocab_size", type=int, default=10000, help="共享词汇表大小")
    parser.add_argument("--max_seq_len", type=int, default=128, help="最大序列长度")
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=128, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮次")
    parser.add_argument("--save_model_path", default="best_bidirectional_model.pth", help="最优模型保存路径")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="训练设备")

    args = parser.parse_args()
    train_model(args)