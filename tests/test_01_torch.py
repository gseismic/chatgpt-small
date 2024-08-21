from chatgpt_3k.config import torch


def test_torch_basic1():
    # 示例数据
    n_batch = 4
    d_embed = 2
    n_head = 3
    d_key = 5

    # 创建示例张量
    X = torch.randn(n_batch, d_embed)  # Shape: [n_batch, d_embed]
    W_q = torch.randn(n_head, d_embed, d_key)  # Shape: [n_head, d_embed, d_key]

    # 计算 Y 的步骤
    # 1. 扩展 X 的维度以匹配 W_q 的维度
    Y_list = [torch.matmul(X, W_q[i]) for i in range(n_head)]
    Y = torch.stack(Y_list, dim=0)  # Shape: [n_head, n_batch, d_key]
    print('X', X)
    print('W_q', W_q)
    print("Y:", Y)
    # X   的shape是 [n_batch, d_embed]
    # W_q   的shape是：[n_head, d_embed, d_key]
    # 我需要计算Y，它的shape是：[n_head, n_batch, d_key]，
    print(f'{X.shape=}, {[n_batch, d_embed]=}')
    print(f'{W_q.shape=}, {[n_head, d_embed, d_key]=}')
    print(f'{Y.shape=}, {[n_head, n_batch, d_key]=}')
    print("Shape of Y:", Y.shape)


if __name__ == '__main__':
    if 1:
        test_torch_basic1()
