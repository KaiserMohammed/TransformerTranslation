from sentencepiece import SentencePieceProcessor
import torch
from trainer import BiTextDataset
from trainer import BidirectionalTransformer
from torch.utils.data import Dataset, DataLoader

def test_model(test_json, sp_model_path, model_path, device="cuda"):
    # 加载词汇表和模型
    sp = SentencePieceProcessor()
    sp.Load(sp_model_path)
    print("分词器加载成功，词汇表大小：", sp.GetPieceSize())

    # 2. 加载测试数据（以单句为例，假设你从test_json中提取了英文句子）
    test_en_text = "At the break of each day, the Arch-mage of Menzoberranzan went out to Narbondel and infused the pillar with a magical, lingering heat that would work its way up, then back down."
    print("\n原始英文输入：", test_en_text)

    # 3. 转换为token id，查看映射是否正常
    en_ids = sp.EncodeAsIds(test_en_text)
    print("英文对应的token id：", en_ids)  # 正常应输出一串整数（如[123, 45, 678, ...]）
    print("解码回英文（验证分词器）：", sp.DecodeIds(en_ids))  # 正常应还原为原英文句子
    model = BidirectionalTransformer(
        shared_vocab_size=sp.GetPieceSize(),
        dim=512, num_heads=8, num_layers=6
    ).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 加载测试集
    test_dataset = BiTextDataset(test_json, sp_model_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 推理示例（英文→中文）
    with torch.no_grad():
        for batch in test_loader:
            en_ids = batch["en_ids"].to(device)  # (1, max_seq_len)
            # 生成中文翻译（调用模型的generate方法）
            generated_zh_ids = model.generate(
                src=en_ids, translate_dir="en2zh",
                max_len=128, bos_token=sp.bos_id(), eos_token=sp.eos_id()
            )
            # 解码为文本
            en_ids_np = en_ids[0].cpu().numpy()
            en_text = sp.DecodeIds(en_ids_np.tolist())

            generated_zh_ids = generated_zh_ids[0].cpu().numpy().tolist()  # 转为Python列表
            eos_id = sp.PieceToId("</s>")
            if eos_id in generated_zh_ids:
                generated_zh_ids = generated_zh_ids[:generated_zh_ids.index(eos_id)]  # 截断到结束标记
            else:
                generated_zh_ids = generated_zh_ids  # 若无eos_id，取完整序列

            # 解码为中文
            generated_zh_text = sp.DecodeIds(generated_zh_ids)

            # 打印结果
            print(f"英文原文：{en_text.strip('<bos>').strip('<eos>')}")
            print(f"中文翻译：{generated_zh_text.strip('<bos>').strip('<eos>')}")
            print("-" * 50)
              # 仅打印第一条示例

# 执行测试
test_model("./data/test_data.json", "shared_vocab.model", "best_bidirectional_model.pth")