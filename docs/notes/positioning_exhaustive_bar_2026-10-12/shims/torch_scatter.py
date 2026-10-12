import torch
def segment_csr(src, indptr, reduce='sum'):
    indptr = indptr.to(src.device).long()
    counts = indptr[1:] - indptr[:-1]
    seg = torch.repeat_interleave(torch.arange(len(counts), device=src.device), counts)
    out_shape = (len(counts),) + tuple(src.shape[1:])
    if reduce == 'sum':
        out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
        return out.index_add(0, seg, src)
    if reduce == 'mean':
        out = torch.zeros(out_shape, dtype=src.dtype, device=src.device).index_add(0, seg, src)
        return out / counts.clamp(min=1).view(-1, *([1]*(src.dim()-1))).to(src.dtype)
    raise NotImplementedError(reduce)
