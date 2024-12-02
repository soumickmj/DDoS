import numpy as np
import torch
import torch.nn as nn

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2022, Faculty of Computer Science, Otto von Guericke University Magdeburg, Germany"
__credits__ = ["Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Soumick Chatterjee"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Under Testing"

class BrokenConvNd(nn.Module):
    def __init__(self, div_dim, learn_alpha=False, conv_layer=nn.Conv2d, **kwargs):
        super(BrokenConvNd, self).__init__()

        self.div_dim = div_dim
        self.n_conv = np.multiply(*div_dim)
        self.convs = nn.ModuleList()
        for _ in range(self.n_conv):
            self.convs.append(conv_layer(**kwargs))

        if learn_alpha:
            self.alphas = nn.Parameter(data=torch.rand(self.n_conv))
        else:
            self.alphas = [1]*self.n_conv

    def _split_tensorlist(self, tensor_list, split_size_or_sections, dim):
        split_tensor_list = []
        for t in tensor_list:
            split_tensor_list += list(torch.split(t, split_size_or_sections=split_size_or_sections, dim=dim+2))
        return split_tensor_list

    def _chunk_tensorlist(self, tensor_list, n_chunks, dim):
        split_tensor_list = []
        for t in tensor_list:
            split_tensor_list += list(torch.chunk(t, chunks=n_chunks, dim=dim+2))
        return split_tensor_list

    def _cat_tensorlist(self, split_tensor_list, n_split, dim):
        return [
            torch.cat(split_tensor_list[i : i + n_split], dim=dim + 2)
            for i in range(0, len(split_tensor_list), n_split)
        ]

    def forward(self, x):
        # dim = x.shape[2:]
        # dim_size = np.divide(dim, self.div_dim).astype(np.int)
        x = [x]
        for d in range(len(self.div_dim)):
            # x = self._split_tensorlist(x, split_size_or_sections=int(dim_size[d]), dim=d)
            x = self._chunk_tensorlist(x, n_chunks=int(self.div_dim[d]), dim=d)
        res = [self.alphas[i] * self.convs[i](x[i]) for i in range(self.n_conv)]
        for d in range(len(self.div_dim)-1,-1,-1):
            res = self._cat_tensorlist(res, n_split=int(self.div_dim[d]), dim=d)
        return res[0]
