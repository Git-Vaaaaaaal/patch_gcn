import torch_geometric
from torch_geometric.data import Data, Batch

# 'edge_latent' est un tenseur [2, E] comme 'edge_index' (indices de nœuds pour le
# graphe KNN en espace latent), mais PyG ne le détecte pas automatiquement car son
# nom ne contient pas "index". On force le même comportement de concaténation/
# incrément que pour edge_index lors du batching.
_orig_cat_dim = Data.__cat_dim__
def _wsi_cat_dim(self, key, value, *args, **kwargs):
    if key == 'edge_latent':
        return -1
    return _orig_cat_dim(self, key, value, *args, **kwargs)
Data.__cat_dim__ = _wsi_cat_dim

_orig_inc = Data.__inc__
def _wsi_inc(self, key, value, *args, **kwargs):
    if key == 'edge_latent':
        return self.num_nodes
    return _orig_inc(self, key, value, *args, **kwargs)
Data.__inc__ = _wsi_inc


class BatchWSI(torch_geometric.data.Batch):
    @classmethod
    def from_data_list(cls, data_list, follow_batch=None, exclude_keys=None, update_cat_dims=None):
        return Batch.from_data_list(
            data_list,
            follow_batch=follow_batch or [],
            exclude_keys=exclude_keys or [],
        )
