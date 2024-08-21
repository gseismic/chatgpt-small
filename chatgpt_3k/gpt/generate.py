from ..config import torch, F


def generate_sequence(model, start_token, max_len, vocab_size, device,
                      temperature=1.0, top_k=None, top_p=None):
    """
    生成文本序列。

    Parameters:
    - model: GPT模型
    - start_token: 起始token的索引
    - max_len: 生成序列的最大长度
    - vocab_size: 词汇表大小
    - device: 设备（如 'cuda' 或 'cpu'）
    - temperature: 温度调整参数 (默认为1.0)
    - top_k: Top-k采样的k值，如果为None则不使用Top-k采样
    - top_p: Top-p采样的p值，如果为None则不使用Top-p采样

    Returns:
    - 生成的token序列

    使用示例
    ```
        embed_dim = 512
        num_heads = 8
        num_layers = 6
        ff_dim = 2048
        vocab_size = 10000
        dropout = 0.1
        max_len = 20
        start_token = 0  # 起始token索引

        # 假设您已经定义并加载了 GPT 模型
        gpt = GPT(num_layers, embed_dim, num_heads, ff_dim, vocab_size, dropout)

        # 温度调整生成
        generated_sequence = generate_sequence(gpt, start_token, max_len, vocab_size, device='cpu', temperature=0.7)
        print("Generated sequence with temperature:", generated_sequence)

        # Top-k 采样生成
        generated_sequence = generate_sequence(gpt, start_token, max_len, vocab_size, device='cpu', top_k=50)
        print("Generated sequence with top-k sampling:", generated_sequence)

        # Top-p 采样生成
        generated_sequence = generate_sequence(gpt, start_token, max_len, vocab_size, device='cpu', top_p=0.9)
        print("Generated sequence with top-p sampling:", generated_sequence)
    """
    assert not (top_k is True and top_p is True)
    assert top_p is None or 0 < top_p < 1
    model.eval()
    generated = [start_token]
    input_seq = torch.tensor([start_token], device=device).unsqueeze(0)  # [1, 1]

    with torch.no_grad():
        for _ in range(max_len - 1):
            output = model(input_seq)  # [1, seq_len, vocab_size]
            logits = output[0, -1, :]  # [vocab_size]
            
            # 应用温度调整
            if temperature != 1.0:
                logits /= temperature

            probs = F.softmax(logits, dim=-1)
            
            if top_k is not None:
                # Top-k采样
                # logits: [vocab_size]
                # top_k_values: [50]
                top_k_values, top_k_indices = logits.topk(top_k)
                top_k_probs = F.softmax(top_k_values, dim=-1)
                # 采样1个
                next_token = top_k_indices[torch.multinomial(top_k_probs, 1).item()]
            elif top_p is not None:
                # Top-p采样
                # print(f'{probs.shape=}')
                # [vocab_size]
                # 概率累积求和，因为是一次只预测一个token
                # https://docs.cohere.com/docs/controlling-generation-with-top-k-top-p#2-pick-from-amongst-the-top-tokens-top-k
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                # print(f'{cumulative_probs=}')
                sorted_indices_to_keep = cumulative_probs <= top_p
                # print(f'{sorted_indices_to_keep=}, {sorted_indices_to_keep.shape=}')
                top_p_indices = sorted_indices[sorted_indices_to_keep]
                top_p_probs = sorted_probs[sorted_indices_to_keep]
                # Ensure at least one token is kept
                # 如果为空，会导致RuntimeError
                if len(top_p_probs) > 0:
                    next_token = top_p_indices[torch.multinomial(top_p_probs, 1).item()]
                else:
                    next_token = sorted_indices[0]
            else:
                # 基于温度调整的采样
                next_token = torch.multinomial(probs, 1).item()

            generated.append(next_token)
            input_seq = torch.cat([input_seq, torch.tensor([[next_token]], device=device)], dim=1)

    return generated