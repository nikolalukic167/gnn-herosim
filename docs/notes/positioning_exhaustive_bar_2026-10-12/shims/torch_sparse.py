import torch
class SparseTensor:
    def __init__(self, row, col, sparse_sizes, is_sorted=False, trust_data=False, value=None):
        self.row, self.col, self.sizes = row.long(), col.long(), tuple(sparse_sizes)
    def t(self):
        return SparseTensor(self.col, self.row, (self.sizes[1], self.sizes[0]))
    def to_dense_coo(self, dtype, device):
        idx = torch.stack([self.row, self.col]).to(device)
        val = torch.ones(idx.shape[1], dtype=dtype, device=device)
        return torch.sparse_coo_tensor(idx, val, self.sizes).coalesce()
def matmul(adj, dense):
    return torch.sparse.mm(adj.to_dense_coo(dense.dtype, dense.device), dense)
