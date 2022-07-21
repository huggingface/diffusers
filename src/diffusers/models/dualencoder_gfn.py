# Model adapted from GeoDiff https://github.com/MinkaiXu/GeoDiff
from typing import Callable, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Embedding, Linear, Module, ModuleList, Sequential

from rdkit.Chem.rdchem import BondType as BT
from torch_geometric.nn import MessagePassing, radius, radius_graph
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from torch_scatter import scatter_add, scatter_mean
from torch_sparse import SparseTensor, coalesce

from ..configuration_utils import ConfigMixin
from ..modeling_utils import ModelMixin


BOND_TYPES = {t: i for i, t in enumerate(BT.names.values())}


class MultiLayerPerceptron(nn.Module):
    """
    Parameters:
    Multi-layer Perceptron. Note there is no activation or dropout in the last layer.
        input_dim (int): input dimension hidden_dim (list of int): hidden dimensions activation (str or function,
        optional): activation function dropout (float, optional): dropout rate
    """

    def __init__(self, input_dim, hidden_dims, activation="relu", dropout=0):
        super(MultiLayerPerceptron, self).__init__()

        self.dims = [input_dim] + hidden_dims
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))

    def forward(self, input):
        """"""
        x = input
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.activation:
                    x = self.activation(x)
                if self.dropout:
                    x = self.dropout(x)
        return x


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class CFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_filters, nn, cutoff, smooth):
        super(CFConv, self).__init__(aggr="add")
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff
        self.smooth = smooth

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_length, edge_attr):
        if self.smooth:
            C = 0.5 * (torch.cos(edge_length * np.pi / self.cutoff) + 1.0)
            C = C * (edge_length <= self.cutoff) * (edge_length >= 0.0)  # Modification: cutoff
        else:
            C = (edge_length <= self.cutoff).float()
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j, W):
        return x_j * W


class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff, smooth):
        super(InteractionBlock, self).__init__()
        mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters, mlp, cutoff, smooth)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, edge_length, edge_attr):
        x = self.conv(x, edge_index, edge_length, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class SchNetEncoder(Module):
    def __init__(
        self, hidden_channels=128, num_filters=128, num_interactions=6, edge_channels=100, cutoff=10.0, smooth=False
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.cutoff = cutoff

        self.embedding = Embedding(100, hidden_channels, max_norm=10.0)

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, edge_channels, num_filters, cutoff, smooth)
            self.interactions.append(block)

    def forward(self, z, edge_index, edge_length, edge_attr, embed_node=True):
        if embed_node:
            assert z.dim() == 1 and z.dtype == torch.long
            h = self.embedding(z)
        else:
            h = z
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_length, edge_attr)

        return h


class GINEConv(MessagePassing):
    def __init__(self, nn: Callable, eps: float = 0.0, train_eps: bool = False, activation="softplus", **kwargs):
        super(GINEConv, self).__init__(aggr="add", **kwargs)
        self.nn = nn
        self.initial_eps = eps

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None

        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))

    def forward(
        self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr: OptTensor = None, size: Size = None
    ) -> torch.Tensor:
        """"""
        if isinstance(x, torch.Tensor):
            x: OptPairTensor = (x, x)

        # Node and edge feature dimensionalites need to match.
        if isinstance(edge_index, torch.Tensor):
            assert edge_attr is not None
            assert x[0].size(-1) == edge_attr.size(-1)
        elif isinstance(edge_index, SparseTensor):
            assert x[0].size(-1) == edge_index.size(-1)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        if self.activation:
            return self.activation(x_j + edge_attr)
        else:
            return x_j + edge_attr

    def __repr__(self):
        return "{}(nn={})".format(self.__class__.__name__, self.nn)


class GINEncoder(torch.nn.Module):
    def __init__(self, hidden_dim, num_convs=3, activation="relu", short_cut=True, concat_hidden=False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_convs = num_convs
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.node_emb = nn.Embedding(100, hidden_dim)

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            self.convs.append(
                GINEConv(
                    MultiLayerPerceptron(hidden_dim, [hidden_dim, hidden_dim], activation=activation),
                    activation=activation,
                )
            )

    def forward(self, z, edge_index, edge_attr):
        """
        Input:
            data: (torch_geometric.data.Data): batched graph node_attr: node feature tensor with shape (num_node,
            hidden) edge_attr: edge feature tensor with shape (num_edge, hidden)
        Output:
            node_attr graph feature
        """

        node_attr = self.node_emb(z)  # (num_node, hidden)

        hiddens = []
        conv_input = node_attr  # (num_node, hidden)

        for conv_idx, conv in enumerate(self.convs):
            hidden = conv(conv_input, edge_index, edge_attr)
            if conv_idx < len(self.convs) - 1 and self.activation is not None:
                hidden = self.activation(hidden)
            assert hidden.shape == conv_input.shape
            if self.short_cut and hidden.shape == conv_input.shape:
                hidden += conv_input

            hiddens.append(hidden)
            conv_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]

        return node_feature


class MLPEdgeEncoder(Module):
    def __init__(self, hidden_dim=100, activation="relu"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bond_emb = Embedding(100, embedding_dim=self.hidden_dim)
        self.mlp = MultiLayerPerceptron(1, [self.hidden_dim, self.hidden_dim], activation=activation)

    @property
    def out_channels(self):
        return self.hidden_dim

    def forward(self, edge_length, edge_type):
        """
        Input:
            edge_length: The length of edges, shape=(E, 1). edge_type: The type pf edges, shape=(E,)
        Returns:
            edge_attr: The representation of edges. (E, 2 * num_gaussians)
        """
        d_emb = self.mlp(edge_length)  # (num_edge, hidden_dim)
        edge_attr = self.bond_emb(edge_type)  # (num_edge, hidden_dim)
        return d_emb * edge_attr  # (num_edge, hidden)


def assemble_atom_pair_feature(node_attr, edge_index, edge_attr):
    h_row, h_col = node_attr[edge_index[0]], node_attr[edge_index[1]]
    h_pair = torch.cat([h_row * h_col, edge_attr], dim=-1)  # (E, 2H)
    return h_pair


def generate_symmetric_edge_noise(num_nodes_per_graph, edge_index, edge2graph, device):
    num_cum_nodes = num_nodes_per_graph.cumsum(0)  # (G, )
    node_offset = num_cum_nodes - num_nodes_per_graph  # (G, )
    edge_offset = node_offset[edge2graph]  # (E, )

    num_nodes_square = num_nodes_per_graph**2  # (G, )
    num_nodes_square_cumsum = num_nodes_square.cumsum(-1)  # (G, )
    edge_start = num_nodes_square_cumsum - num_nodes_square  # (G, )
    edge_start = edge_start[edge2graph]

    all_len = num_nodes_square_cumsum[-1]

    node_index = edge_index.t() - edge_offset.unsqueeze(-1)
    node_large = node_index.max(dim=-1)[0]
    node_small = node_index.min(dim=-1)[0]
    undirected_edge_id = node_large * (node_large + 1) + node_small + edge_start

    symm_noise = torch.zeros(size=[all_len.item()], device=device)
    symm_noise.normal_()
    d_noise = symm_noise[undirected_edge_id].unsqueeze(-1)  # (E, 1)
    return d_noise


def _extend_graph_order(num_nodes, edge_index, edge_type, order=3):
    """
    Args:
        num_nodes:  Number of atoms.
        edge_index: Bond indices of the original graph.
        edge_type:  Bond types of the original graph.
        order:  Extension order.
    Returns:
        new_edge_index: Extended edge indices. new_edge_type: Extended edge types.
    """

    def binarize(x):
        return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

    def get_higher_order_adj_matrix(adj, order):
        """
        Args:
            adj:        (N, N)
            type_mat:   (N, N)
        Returns:
            Following attributes will be updated:
              - edge_index
              - edge_type
            Following attributes will be added to the data object:
              - bond_edge_index: Original edge_index.
        """
        adj_mats = [
            torch.eye(adj.size(0), dtype=torch.long, device=adj.device),
            binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device)),
        ]

        for i in range(2, order + 1):
            adj_mats.append(binarize(adj_mats[i - 1] @ adj_mats[1]))
        order_mat = torch.zeros_like(adj)

        for i in range(1, order + 1):
            order_mat += (adj_mats[i] - adj_mats[i - 1]) * i

        return order_mat

    num_types = len(BOND_TYPES)

    N = num_nodes
    adj = to_dense_adj(edge_index).squeeze(0)
    adj_order = get_higher_order_adj_matrix(adj, order)  # (N, N)

    type_mat = to_dense_adj(edge_index, edge_attr=edge_type).squeeze(0)  # (N, N)
    type_highorder = torch.where(adj_order > 1, num_types + adj_order - 1, torch.zeros_like(adj_order))
    assert (type_mat * type_highorder == 0).all()
    type_new = type_mat + type_highorder

    new_edge_index, new_edge_type = dense_to_sparse(type_new)
    _, edge_order = dense_to_sparse(adj_order)

    # data.bond_edge_index = data.edge_index  # Save original edges
    new_edge_index, new_edge_type = coalesce(new_edge_index, new_edge_type.long(), N, N)  # modify data

    # [Note] This is not necessary
    # data.is_bond = (data.edge_type < num_types)

    # [Note] In earlier versions, `edge_order` attribute will be added.
    #         However, it doesn't seem to be necessary anymore so I removed it.
    # edge_index_1, data.edge_order = coalesce(new_edge_index, edge_order.long(), N, N) # modify data
    # assert (data.edge_index == edge_index_1).all()

    return new_edge_index, new_edge_type


def _extend_to_radius_graph(pos, edge_index, edge_type, cutoff, batch, unspecified_type_number=0, is_sidechain=None):
    assert edge_type.dim() == 1
    N = pos.size(0)

    bgraph_adj = torch.sparse.LongTensor(edge_index, edge_type, torch.Size([N, N]))

    if is_sidechain is None:
        rgraph_edge_index = radius_graph(pos, r=cutoff, batch=batch)  # (2, E_r)
    else:
        # fetch sidechain and its batch index
        is_sidechain = is_sidechain.bool()
        dummy_index = torch.arange(pos.size(0), device=pos.device)
        sidechain_pos = pos[is_sidechain]
        sidechain_index = dummy_index[is_sidechain]
        sidechain_batch = batch[is_sidechain]

        assign_index = radius(x=pos, y=sidechain_pos, r=cutoff, batch_x=batch, batch_y=sidechain_batch)
        r_edge_index_x = assign_index[1]
        r_edge_index_y = assign_index[0]
        r_edge_index_y = sidechain_index[r_edge_index_y]

        rgraph_edge_index1 = torch.stack((r_edge_index_x, r_edge_index_y))  # (2, E)
        rgraph_edge_index2 = torch.stack((r_edge_index_y, r_edge_index_x))  # (2, E)
        rgraph_edge_index = torch.cat((rgraph_edge_index1, rgraph_edge_index2), dim=-1)  # (2, 2E)
        # delete self loop
        rgraph_edge_index = rgraph_edge_index[:, (rgraph_edge_index[0] != rgraph_edge_index[1])]

    rgraph_adj = torch.sparse.LongTensor(
        rgraph_edge_index,
        torch.ones(rgraph_edge_index.size(1)).long().to(pos.device) * unspecified_type_number,
        torch.Size([N, N]),
    )

    composed_adj = (bgraph_adj + rgraph_adj).coalesce()  # Sparse (N, N, T)
    # edge_index = composed_adj.indices()
    # dist = (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)

    new_edge_index = composed_adj.indices()
    new_edge_type = composed_adj.values().long()

    return new_edge_index, new_edge_type


def extend_graph_order_radius(
    num_nodes,
    pos,
    edge_index,
    edge_type,
    batch,
    order=3,
    cutoff=10.0,
    extend_order=True,
    extend_radius=True,
    is_sidechain=None,
):
    if extend_order:
        edge_index, edge_type = _extend_graph_order(
            num_nodes=num_nodes, edge_index=edge_index, edge_type=edge_type, order=order
        )
        # edge_index_order = edge_index
        # edge_type_order = edge_type

    if extend_radius:
        edge_index, edge_type = _extend_to_radius_graph(
            pos=pos, edge_index=edge_index, edge_type=edge_type, cutoff=cutoff, batch=batch, is_sidechain=is_sidechain
        )

    return edge_index, edge_type


def get_distance(pos, edge_index):
    return (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)


def eq_transform(score_d, pos, edge_index, edge_length):
    N = pos.size(0)
    dd_dr = (1.0 / edge_length) * (pos[edge_index[0]] - pos[edge_index[1]])  # (E, 3)
    score_pos = scatter_add(dd_dr * score_d, edge_index[0], dim=0, dim_size=N) + scatter_add(
        -dd_dr * score_d, edge_index[1], dim=0, dim_size=N
    )  # (N, 3)
    return score_pos


class DualEncoderEpsNetwork(ModelMixin, ConfigMixin):
    def __init__(
        self,
        hidden_dim,
        num_convs,
        num_convs_local,
        cutoff,
        mlp_act,
        edge_order,
        edge_encoder,
        smooth_conv,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.edge_encoder = edge_encoder
        self.edge_order = edge_order

        """
        edge_encoder: Takes both edge type and edge length as input and outputs a vector [Note]: node embedding is done
        in SchNetEncoder
        """
        self.edge_encoder_global = MLPEdgeEncoder(hidden_dim, mlp_act)  # get_edge_encoder(config)
        self.edge_encoder_local = MLPEdgeEncoder(hidden_dim, mlp_act)  # get_edge_encoder(config)

        """
        The graph neural network that extracts node-wise features.
        """
        self.encoder_global = SchNetEncoder(
            hidden_channels=hidden_dim,
            num_filters=hidden_dim,
            num_interactions=num_convs,
            edge_channels=self.edge_encoder_global.out_channels,
            cutoff=cutoff,
            smooth=smooth_conv,
        )
        self.encoder_local = GINEncoder(
            hidden_dim=hidden_dim,
            num_convs=num_convs_local,
        )

        """
        `output_mlp` takes a mixture of two nodewise features and edge features as input and outputs
            gradients w.r.t. edge_length (out_dim = 1).
        """
        self.grad_global_dist_mlp = MultiLayerPerceptron(
            2 * hidden_dim, [hidden_dim, hidden_dim // 2, 1], activation=mlp_act
        )

        self.grad_local_dist_mlp = MultiLayerPerceptron(
            2 * hidden_dim, [hidden_dim, hidden_dim // 2, 1], activation=mlp_act
        )

        """
        Incorporate parameters together
        """
        self.model_global = nn.ModuleList([self.edge_encoder_global, self.encoder_global, self.grad_global_dist_mlp])
        self.model_local = nn.ModuleList([self.edge_encoder_local, self.encoder_local, self.grad_local_dist_mlp])

    def forward(
        self,
        atom_type,
        pos,
        bond_index,
        bond_type,
        batch,
        time_step,      # NOTE, model trained without timestep performed best
        edge_index=None,
        edge_type=None,
        edge_length=None,
        return_edges=False,
        extend_order=True,
        extend_radius=True,
        is_sidechain=None,
    ):
        """
        Args:
            atom_type:  Types of atoms, (N, ).
            bond_index: Indices of bonds (not extended, not radius-graph), (2, E).
            bond_type:  Bond types, (E, ).
            batch:      Node index to graph index, (N, ).
        """
        N = atom_type.size(0)
        if edge_index is None or edge_type is None or edge_length is None:
            edge_index, edge_type = extend_graph_order_radius(
                num_nodes=N,
                pos=pos,
                edge_index=bond_index,
                edge_type=bond_type,
                batch=batch,
                order=self.edge_order,
                cutoff=self.cutoff,
                extend_order=extend_order,
                extend_radius=extend_radius,
                is_sidechain=is_sidechain,
            )
            edge_length = get_distance(pos, edge_index).unsqueeze(-1)  # (E, 1)
        local_edge_mask = is_local_edge(edge_type)  # (E, )

        # with the parameterization of NCSNv2
        # DDPM loss implicit handle the noise variance scale conditioning
        sigma_edge = torch.ones(size=(edge_index.size(1), 1), device=pos.device)  # (E, 1)

        # Encoding global
        edge_attr_global = self.edge_encoder_global(edge_length=edge_length, edge_type=edge_type)  # Embed edges

        # Global
        node_attr_global = self.encoder_global(
            z=atom_type,
            edge_index=edge_index,
            edge_length=edge_length,
            edge_attr=edge_attr_global,
        )
        ## Assemble pairwise features
        h_pair_global = assemble_atom_pair_feature(
            node_attr=node_attr_global,
            edge_index=edge_index,
            edge_attr=edge_attr_global,
        )  # (E_global, 2H)
        ## Invariant features of edges (radius graph, global)
        edge_inv_global = self.grad_global_dist_mlp(h_pair_global) * (1.0 / sigma_edge)  # (E_global, 1)

        # Encoding local
        edge_attr_local = self.edge_encoder_global(edge_length=edge_length, edge_type=edge_type)  # Embed edges
        # edge_attr += temb_edge

        # Local
        node_attr_local = self.encoder_local(
            z=atom_type,
            edge_index=edge_index[:, local_edge_mask],
            edge_attr=edge_attr_local[local_edge_mask],
        )
        ## Assemble pairwise features
        h_pair_local = assemble_atom_pair_feature(
            node_attr=node_attr_local,
            edge_index=edge_index[:, local_edge_mask],
            edge_attr=edge_attr_local[local_edge_mask],
        )  # (E_local, 2H)

        ## Invariant features of edges (bond graph, local)
        if isinstance(sigma_edge, torch.Tensor):
            edge_inv_local = self.grad_local_dist_mlp(h_pair_local) * (
                1.0 / sigma_edge[local_edge_mask]
            )  # (E_local, 1)
        else:
            edge_inv_local = self.grad_local_dist_mlp(h_pair_local) * (1.0 / sigma_edge)  # (E_local, 1)

        if return_edges:
            return edge_inv_global, edge_inv_local, edge_index, edge_type, edge_length, local_edge_mask
        else:
            return edge_inv_global, edge_inv_local

    def get_residual_params(
        self,
        t,
        batch,
        extend_order=False,
        extend_radius=True,
        clip_local=None,
    ):
        atom_type = batch.atom_type
        bond_index = batch.edge_index
        bond_type = batch.edge_type
        num_graphs = batch.num_graphs
        pos = batch.pos

        timesteps = torch.full(size=(num_graphs,), fill_value=t, dtype=torch.long, device=pos.device)

        edge_inv_global, edge_inv_local, edge_index, edge_type, edge_length, local_edge_mask = self.forward(
            atom_type=atom_type,
            pos=batch.pos,
            bond_index=bond_index,
            bond_type=bond_type,
            batch=batch.batch,
            time_step=timesteps,
            return_edges=True,
            extend_order=extend_order,
            extend_radius=extend_radius,
        )  # (E_global, 1), (E_local, 1)

        # Important equation in the paper for equivariant features - eqns 5-7 of GeoDiff
        node_eq_local = eq_transform(edge_inv_local, pos, edge_index[:, local_edge_mask], edge_length[local_edge_mask])
        if clip_local is not None:
            node_eq_local = clip_norm(node_eq_local, limit=clip_local)

        return edge_inv_global, edge_inv_local, edge_index, edge_type, edge_length, local_edge_mask, node_eq_local

    def get_residual(
        self,
        pos,
        sigma,
        model_outputs,
        global_start_sigma=0.5,
        w_global=1.0,
        clip=1000.0,
    ):
        (
            edge_inv_global,
            edge_inv_local,
            edge_index,
            edge_type,
            edge_length,
            local_edge_mask,
            node_eq_local,
        ) = model_outputs

        # Global
        if sigma < global_start_sigma:
            edge_inv_global = edge_inv_global * (1 - local_edge_mask.view(-1, 1).float())
            node_eq_global = eq_transform(edge_inv_global, pos, edge_index, edge_length)
            node_eq_global = clip_norm(node_eq_global, limit=clip)
        else:
            node_eq_global = 0

        # Sum
        eps_pos = node_eq_local + node_eq_global * w_global
        return -eps_pos


def clip_norm(vec, limit, p=2):
    norm = torch.norm(vec, dim=-1, p=2, keepdim=True)
    denom = torch.where(norm > limit, limit / norm, torch.ones_like(norm))
    return vec * denom


def is_local_edge(edge_type):
    return edge_type > 0


def is_train_edge(edge_index, is_sidechain):
    if is_sidechain is None:
        return torch.ones(edge_index.size(1), device=edge_index.device).bool()
    else:
        is_sidechain = is_sidechain.bool()
        return torch.logical_or(is_sidechain[edge_index[0]], is_sidechain[edge_index[1]])
