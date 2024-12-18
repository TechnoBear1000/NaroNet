import torch
from torch_geometric.nn import SplineConv

def test_spline_conv():
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    x = torch.randn((3, 16))  # 3 nodes with 16 features each
    conv = SplineConv(16, 32, dim=2, kernel_size=5)
    out = conv(x, edge_index)
    print("SplineConv output shape:", out.shape)

if __name__ == "__main__":
    test_spline_conv()
