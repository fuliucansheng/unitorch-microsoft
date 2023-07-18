# -*- coding: UTF-8 -*-

import sys
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as Func


# utils part
def global_avg_pool(x, mask):
    x = torch.sum(x, 2)
    length = torch.sum(torch.max(mask, 1)[0], 1, keepdim=True)  # n x 1
    len_mask = torch.where(
        length > 0, torch.zeros_like(length), torch.ones_like(length)
    ).float()
    length = length.float()
    length += len_mask * 1e-12
    length = length.repeat(1, x.size()[1])
    x /= length
    return x


def global_max_pool(x, mask):
    mask = 1 - mask

    mask *= -(2**32) + 1
    x += mask
    x = torch.max(x, 2)[0]  # the result tuple of two output tensors (max, max_indices)
    return x


def get_length(mask):
    length = torch.sum(torch.max(mask, 1)[0], 1)
    length = length.long()
    return length


## ops part
class MaskOpt(nn.Module):
    def __init__(self, is_cuda=False):
        super(MaskOpt, self).__init__()
        self.is_cuda = is_cuda

    def forward(self, seq, mask):
        # move following ops into pre-processing to save running time
        # seq_mask = torch.unsqueeze(mask, 2)
        # seq_mask = torch.transpose(seq_mask.repeat(1, 1, seq.size()[1]), 1, 2)
        """
        seq_mask = mask
        if self.is_cuda:
          seq = seq.where(torch.eq(seq_mask, 1), torch.zeros(seq.shape, device='cuda'))
        else:
          seq = seq.where(torch.eq(seq_mask, 1), torch.zeros(seq.shape))
        return seq
        """
        return seq * mask.to(torch.float)


class NormFactory(object):
    def __init__(self, **kwargs):
        assert isinstance(kwargs["norm_type"], str)
        assert isinstance(kwargs["num_features"], int)
        assert isinstance(kwargs["pre_mask"], bool)
        assert isinstance(kwargs["post_mask"], bool)
        assert isinstance(kwargs["is_cuda"], bool)
        self.params = kwargs

    def create_norm_func(self):
        if self.params["norm_type"] == "LN":
            return LayerNorm(
                self.params["num_features"],
                self.params["pre_mask"],
                self.params["post_mask"],
                is_cuda=self.params["is_cuda"],
            )
        elif self.params["norm_type"] == "BN":
            return BatchNorm(
                self.params["num_features"],
                self.params["pre_mask"],
                self.params["post_mask"],
                is_cuda=self.params["is_cuda"],
            )
        else:
            raise ValueError("Unsupported norm type.")


class BatchNorm(nn.Module):
    def __init__(
        self,
        num_features,
        pre_mask,
        post_mask,
        eps=1e-5,
        decay=0.9,
        affine=True,
        is_cuda=False,
    ):
        super(BatchNorm, self).__init__()
        self.mask_opt = MaskOpt(is_cuda=is_cuda)
        self.pre_mask = pre_mask
        self.post_mask = post_mask
        self.bn = nn.BatchNorm1d(
            num_features, eps=eps, momentum=1.0 - decay, affine=affine
        )
        # self.bn.weight.data.fill_(1.0)
        self.is_cuda = is_cuda

    def forward(self, seq, mask):
        if self.pre_mask:
            seq = self.mask_opt(seq, mask)
        seq = self.bn(seq)
        if self.post_mask:
            seq = self.mask_opt(seq, mask)
        return seq


class LayerNorm(nn.Module):
    def __init__(
        self,
        num_features,
        pre_mask,
        post_mask,
        eps=1e-5,
        decay=0.9,
        affine=True,
        is_cuda=False,
    ):
        super(LayerNorm, self).__init__()
        self.mask_opt = MaskOpt(is_cuda=is_cuda)
        self.pre_mask = pre_mask
        self.post_mask = post_mask
        self.is_cuda = is_cuda
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.variance_epsilon = eps

    def forward(self, seq, mask):
        if self.pre_mask:
            seq = self.mask_opt(seq, mask)
        seq = torch.transpose(seq, 1, 2)
        u = seq.mean(-1, keepdim=True)
        s = (seq - u).pow(2).mean(-1, keepdim=True)
        seq = (seq - u) / torch.sqrt(s + self.variance_epsilon)
        seq = self.weight * seq + self.bias
        seq = torch.transpose(seq, 1, 2)
        if self.post_mask:
            seq = self.mask_opt(seq, mask)
        return seq


class ConvOpt(nn.Module):
    """_conv_opt + layer_norm/batch_norm"""

    def __init__(
        self,
        kernel_size,
        in_channels,
        out_channels,
        cnn_keep_prob,
        pre_mask,
        post_mask,
        norm_type="LN",
        with_relu=True,
        is_cuda=False,
    ):
        super(ConvOpt, self).__init__()
        self.mask_opt = MaskOpt(is_cuda=is_cuda)
        self.pre_mask = pre_mask
        self.post_mask = post_mask
        self.norm_type = norm_type
        self.with_relu = with_relu
        self.is_cuda = is_cuda
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            1,
            bias=True,
            padding=(kernel_size - 1) // 2,
        )
        self.dropout = nn.Dropout(p=(1 - cnn_keep_prob))

        if norm_type:
            self.norm_func = NormFactory(
                num_features=out_channels,
                pre_mask=not post_mask,
                post_mask=True,
                is_cuda=is_cuda,
                norm_type=norm_type,
            ).create_norm_func()
        if with_relu:
            self.relu = nn.ReLU()

    def forward(self, seq, mask):
        if self.pre_mask:
            seq = self.mask_opt(seq, mask)
        seq = self.conv(seq)
        if self.post_mask:
            seq = self.mask_opt(seq, mask)
        if self.norm_type:
            seq = self.norm_func(seq, mask)
        if self.with_relu:
            seq = self.relu(seq)
        seq = self.dropout(seq)
        return seq


class AvgPoolOpt(nn.Module):
    def __init__(self, kernel_size, pre_mask, post_mask, is_cuda=False):
        super(AvgPoolOpt, self).__init__()
        self.avg_pool = nn.AvgPool1d(kernel_size, 1, padding=(kernel_size - 1) // 2)
        self.pre_mask = pre_mask
        self.post_mask = post_mask
        self.mask_opt = MaskOpt(is_cuda=is_cuda)
        self.is_cuda = is_cuda

    def forward(self, seq, mask):
        if self.pre_mask:
            seq = self.mask_opt(seq, mask)
        seq = self.avg_pool(seq)
        if self.post_mask:
            seq = self.mask_opt(seq, mask)
        return seq


class MaxPoolOpt(nn.Module):
    def __init__(self, kernel_size, pre_mask, post_mask, is_cuda=False):
        super(MaxPoolOpt, self).__init__()
        self.max_pool = nn.MaxPool1d(kernel_size, 1, padding=(kernel_size - 1) // 2)
        self.pre_mask = pre_mask
        self.post_mask = post_mask
        self.mask_opt = MaskOpt(is_cuda=is_cuda)
        self.is_cuda = is_cuda

    def forward(self, seq, mask):
        if self.pre_mask:
            seq = self.mask_opt(seq, mask)
        seq = self.max_pool(seq)
        if self.post_mask:
            seq = self.mask_opt(seq, mask)
        return seq


class AttentionOpt(nn.Module):
    def __init__(
        self, num_units, num_heads, keep_prob, is_mask, is_cuda=False, norm_type=None
    ):
        super(AttentionOpt, self).__init__()
        self.num_heads = num_heads
        self.keep_prob = keep_prob

        self.linear_q = nn.Linear(num_units, num_units)
        self.linear_k = nn.Linear(num_units, num_units)
        self.linear_v = nn.Linear(num_units, num_units)

        self.norm_func = NormFactory(
            num_features=num_units,
            pre_mask=True,
            post_mask=is_mask,
            is_cuda=is_cuda,
            norm_type=norm_type,
        ).create_norm_func()
        self.dropout = nn.Dropout(p=1 - self.keep_prob)
        self.is_cuda = is_cuda

    def transpose_for_scores(self, x, in_c):
        new_x_shape = x.size()[:-1] + (
            self.num_heads,
            in_c // self.num_heads,
        )  # x: N, len, C  -> N, Len, H, C/H
        x = x.view(*new_x_shape)  # -> N, L, H, C/H
        return x.permute(0, 2, 1, 3)  # N, H, L, C/H

    def forward(self, seq, mask):
        in_c = seq.size()[1]
        seq = torch.transpose(seq, 1, 2)  # (N, L, C)
        queries = seq
        keys = seq
        num_heads = self.num_heads
        L = seq.size()[1]

        Q = Func.relu(self.linear_q(seq))  # (N, T_q, C)
        K = Func.relu(self.linear_k(seq))  # (N, T_k, C)
        V = Func.relu(self.linear_v(seq))  # (N, T_k, C)
        """ original implementation

    # Split and concat
    Q_ = torch.cat(torch.split(Q, in_c // num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
    K_ = torch.cat(torch.split(K, in_c // num_heads, dim=2), dim=0)  # (h*N, T_k, C/h)
    V_ = torch.cat(torch.split(V, in_c // num_heads, dim=2), dim=0)  # (h*N, T_k, C/h)

    # Multiplication
    outputs = torch.matmul(Q_, K_.transpose(1, 2))  # (h*N, T_q, T_k)
    # Scale
    outputs = outputs / (K_.size()[-1] ** 0.5)
    # Key Masking
    key_masks = mask.repeat(num_heads, 1)  # (h*N, T_k)
    key_masks = torch.unsqueeze(key_masks, 1)  # (h*N, 1, T_k)
    key_masks = key_masks.repeat(1, queries.size()[1], 1)  # (h*N, T_q, T_k)

    paddings = torch.ones_like(outputs) * (-2 ** 32 + 1)  # extremely small value
    outputs = torch.where(torch.eq(key_masks, 0), paddings, outputs)

    query_masks = mask.repeat(num_heads, 1)  # (h*N, T_q)
    query_masks = torch.unsqueeze(query_masks, -1)  # (h*N, T_q, 1)
    query_masks = query_masks.repeat(1, 1, keys.size()[1]).float()  # (h*N, T_q, T_k)

    att_scores = Func.softmax(outputs, dim=-1) * query_masks  # (h*N, T_q, T_k)
    att_scores = self.dropout(att_scores)

    # Weighted sum
    x_outputs = torch.matmul(att_scores, V_)  # (h*N, T_q, C/h)  

    # Restore shape
    x_outputs = torch.cat(
            torch.split(x_outputs, x_outputs.size()[0] // num_heads, dim=0),
            dim=2)  # (N, T_q, C)
    """

        Q_ = self.transpose_for_scores(Q, in_c)  # (N, h, T_q, C/h)
        K_ = self.transpose_for_scores(K, in_c)  # (N, h, T_k, C/h)
        V_ = self.transpose_for_scores(V, in_c)  # (N, h, T_k, C/h)

        # Multiplication
        outputs = torch.matmul(Q_, K_.transpose(-2, -1))  # (N, h, T_q, T_k)

        # Scale
        outputs = outputs / (K_.size()[-1] ** 0.5)  # (N, h, T_q, T_k)

        # N, 1, L, 1
        # mask, N, C, L
        attn_masks = torch.max(mask, 1, keepdim=True)[0]  # N, 1, L
        attn_masks = attn_masks.view(*(attn_masks.size() + (1,)))  # N, 1, L, 1
        attn_masks = attn_masks.repeat(
            1, num_heads, 1, L
        )  # (N, h, T_k)  # (N, h, T_k, T_k)

        paddings = torch.ones_like(outputs) * (-(2**32) + 1)  # extremely small value
        outputs = torch.where(
            torch.transpose(attn_masks.byte(), 2, 3), outputs, paddings
        )

        att_scores = Func.softmax(outputs, dim=-1) * attn_masks.to(
            torch.float
        )  # (N, h, T_q, T_k)
        att_scores = self.dropout(att_scores)

        # Weighted sum
        x_outputs = torch.matmul(att_scores, V_)  # (N, h, T_q, C/h)
        x_outputs = x_outputs.permute(0, 2, 1, 3).contiguous()  # (N, T_q, h, C/h)
        # Restore shape
        new_shape = x_outputs.size()[:-2] + (in_c,)  # keep N, T_q; merge h and C/h to C
        x_outputs = x_outputs.view(*new_shape)  # (N, T_q, C)

        x = torch.transpose(x_outputs, 1, 2)  # (N, C, L)
        x = self.norm_func(x, mask)

        return x


class RnnOpt(nn.Module):
    def __init__(self, hidden_size, output_keep_prob, is_cuda=False):
        super(RnnOpt, self).__init__()
        self.hidden_size = hidden_size
        self.bid_rnn = nn.GRU(
            hidden_size, hidden_size, batch_first=True, bidirectional=True
        )
        self.output_keep_prob = output_keep_prob

        self.out_dropout = nn.Dropout(p=(1 - self.output_keep_prob))
        self.is_cuda = is_cuda

    def forward(self, seq, mask):
        # seq: (N, C, L)
        # mask: (N, L)
        max_len = seq.size()[2]
        length = get_length(mask)
        seq = torch.transpose(seq, 1, 2)  # to (N, L, C)
        packed_seq = nn.utils.rnn.pack_padded_sequence(
            seq, length, batch_first=True, enforce_sorted=False
        )
        outputs, state = self.bid_rnn(packed_seq)
        outputs = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True, total_length=max_len
        )[0]
        outputs = outputs.view(-1, max_len, 2, self.hidden_size).sum(2)  # (N, L, C)
        outputs = self.out_dropout(outputs)  # output dropout
        return torch.transpose(outputs, 1, 2)  # back to: (N, C, L)


class LinearCombine(nn.Module):
    def __init__(
        self,
        layers_num,
        trainable=True,
        input_aware=False,
        word_level=False,
        is_cuda=False,
    ):
        super(LinearCombine, self).__init__()
        self.input_aware = input_aware
        self.word_level = word_level
        self.is_cuda = is_cuda

        if input_aware:
            raise ValueError("input_aware Not supported")
        else:
            if self.is_cuda:
                self.w = torch.full(
                    (layers_num, 1, 1, 1), 1.0 / layers_num, device="cuda"
                )
            else:
                self.w = torch.full((layers_num, 1, 1, 1), 1.0 / layers_num)
            if trainable:
                self.w = nn.Parameter(self.w)

    def forward(self, seq):
        nw = Func.softmax(self.w, dim=0)
        seq = torch.mul(seq, nw)
        seq = torch.sum(seq, dim=0)
        return seq


# emb parts
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512, padding_idx=0):
        super().__init__(vocab_size, embed_size, padding_idx=padding_idx)


class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512, padding_idx=0):
        super().__init__(3, embed_size, padding_idx=padding_idx)


# sub model
class Model(nn.Module):
    def __init__(
        self,
        batch_size=32,
        eval_batch_size=100,
        clip_mode=None,
        grad_bound=None,
        l2_reg=1e-4,
        lr_init=0.1,
        lr_dec_start=0,
        lr_dec_every=100,
        lr_dec_rate=0.1,
        cnn_keep_prob=1.0,
        final_output_keep_prob=1.0,
        embed_keep_prob=1.0,
        lstm_out_keep_prob=1.0,
        attention_keep_prob=1.0,
        var_rec=False,
        optim_algo=None,
        sync_replicas=False,
        num_aggregate=None,
        num_replicas=None,
        dataset="sst",
        name="generic_model",
        seed=None,
    ):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.clip_mode = clip_mode
        self.grad_bound = grad_bound
        self.l2_reg = l2_reg
        self.lr_init = lr_init
        self.lr_dec_start = lr_dec_start
        self.lr_dec_rate = lr_dec_rate
        self.cnn_keep_prob = cnn_keep_prob
        self.final_output_keep_prob = final_output_keep_prob
        self.embed_keep_prob = embed_keep_prob
        self.lstm_out_keep_prob = lstm_out_keep_prob
        self.attention_keep_prob = attention_keep_prob
        self.var_rec = var_rec
        self.optim_algo = optim_algo
        self.sync_replicas = sync_replicas
        self.num_aggregate = num_aggregate
        self.num_replicas = num_replicas
        self.dataset = dataset
        self.name = name
        # self.seed = seed
        self.seed = None

        if self.seed is not None:
            print("set random seed for model building: {}".format(self.seed))
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)


class GeneralChild(Model):
    def __init__(
        self,
        vocab_size,
        fixed_arc=None,
        out_filters_scale=1,
        num_layers=2,
        num_branches=6,
        out_filters=24,
        cnn_keep_prob=1.0,
        final_output_keep_prob=1.0,
        lstm_out_keep_prob=1.0,
        embed_keep_prob=1.0,
        attention_keep_prob=1.0,
        multi_path=False,
        input_positional_encoding=False,
        is_sinusolid=False,
        all_layer_output=False,
        num_last_layer_output=0,
        is_mask=False,
        output_type="avg_pool",
        norm_type=None,
        max_doc_length=32,
        name="child",
        pool_step=3,
        class_num=5,
        seed=None,
        downscaled_size=128,
        is_cuda=True,
        *args,
        **kwargs,
    ):
        super(self.__class__, self).__init__(
            cnn_keep_prob=cnn_keep_prob,
            final_output_keep_prob=final_output_keep_prob,
            embed_keep_prob=embed_keep_prob,
            lstm_out_keep_prob=lstm_out_keep_prob,
            attention_keep_prob=attention_keep_prob,
            name=name,
            seed=seed,
        )

        self.fixed_arc = fixed_arc
        self.vocab_size = vocab_size
        self.max_doc_length = max_doc_length
        self.all_layer_output = all_layer_output
        self.num_last_layer_output = max(num_last_layer_output, 0)
        self.is_mask = is_mask
        self.output_type = output_type
        self.multi_path = multi_path
        self.input_positional_encoding = input_positional_encoding
        self.is_sinusolid = is_sinusolid
        self.out_filters = out_filters * out_filters_scale
        self.num_layers = num_layers
        self.pool_step = pool_step
        self.class_num = class_num
        self.downscaled_size = downscaled_size
        self.num_branches = num_branches
        self.out_filters_scale = out_filters_scale
        self.is_cuda = is_cuda

        pool_distance = self.num_layers // self.pool_step
        self.pool_layers = []
        for i in range(1, self.pool_step):  # pool_step == 1
            self.pool_layers.append(i * pool_distance - 1)

        self.fixed_flag = False
        if self.fixed_arc is not None:
            fixed_arc = np.array([int(x) for x in self.fixed_arc.split(" ") if x])
            self.sample_arc = fixed_arc
            self.fixed_flag = True

        layers = []

        out_filters = self.out_filters
        # _conv_opt + layer_norm/batch_norm
        self.init_conv = ConvOpt(
            1,
            out_filters,
            out_filters,
            cnn_keep_prob,
            False,
            True,
            norm_type=norm_type,
            is_cuda=self.is_cuda,
        )

        for layer_id in range(self.num_layers):
            layers.append(
                self.make_fixed_layer(layer_id, out_filters, norm_type)
            )  # construct all branches
            if layer_id in self.pool_layers:  # not supported yet
                pass

        self.layers = nn.ModuleList(layers)
        if self.downscaled_size:
            self.downscale = nn.Linear(out_filters, downscaled_size)

    def forward(self, sequence_ids, sequence_mask, is_training):
        """is_training maybe unuseful and can delete"""
        seq = torch.transpose(sequence_ids, 1, 2)  # from (N, L, C) -> (N, C, L)
        inp_c, inp_l = sequence_ids.shape[1:]

        x = self.init_conv(seq, sequence_mask)

        start_idx = 0
        prev_layers = []
        final_flags = []

        for layer_id in range(self.num_layers):  # run layers
            layer = self.layers[layer_id]
            x = self.run_fixed_layer(
                x,
                sequence_mask,
                prev_layers,
                layer,
                layer_id,
                start_idx,
                final_flags=final_flags,
            )  # run needed brunches
            prev_layers.append(x)
            final_flags.append(1)

            start_idx += 1 + layer_id
            if self.multi_path:
                start_idx += 1

        final_layers = []
        final_layers_idx = []
        for i in range(0, len(prev_layers)):
            if self.all_layer_output:
                if self.num_last_layer_output == 0:
                    final_layers.append(prev_layers[i])
                    final_layers_idx.append(i)
                elif i >= max((len(prev_layers) - self.num_last_layer_output), 0):
                    final_layers.append(prev_layers[i])
                    final_layers_idx.append(i)
            else:
                final_layers.append(final_flags[i] * prev_layers[i])

        x = sum(final_layers)
        if not self.all_layer_output:
            x /= sum(final_flags)
        else:
            x /= len(final_layers)

        class_num = self.class_num
        inp_c = x.size()[1]
        inp_l = x.size()[2]
        if self.output_type == "cls":
            x = x[:, :, :, 0]  # NCHW, H=1, W=1)
            x = torch.reshape(x, [-1, inp_c])
        elif self.output_type == "avg_pool":
            x = global_avg_pool(x, sequence_mask)  # NC
        elif self.output_type == "max_pool":
            x = global_max_pool(x, sequence_mask)  # NC
        else:
            raise ValueError("Unsupported output type.")
        if self.downscaled_size:
            x = self.downscale(x)
        return x

    def make_fixed_layer(self, layer_id, out_filters, norm_type):
        size = [1, 3, 5, 7]
        separables = [False, False, False, False]

        branches = []

        if self.fixed_flag:
            branch_id = (layer_id + 1) * (layer_id + 2) // 2
            whether_add_norm = False
            for i in range(layer_id):
                if self.sample_arc[branch_id + 1 + i] == 1:
                    whether_add_norm = True
            branch_id = self.sample_arc[branch_id]

            for operation_id in [0, 1, 2, 3]:  # conv_opt
                if branch_id == operation_id:
                    filter_size = size[operation_id]
                    separable = separables[operation_id]
                    op = ConvOpt(
                        filter_size,
                        out_filters,
                        out_filters,
                        self.cnn_keep_prob,
                        False,
                        True,
                        norm_type=norm_type,
                        is_cuda=self.is_cuda,
                    )
                    branches.append(op)
            if branch_id == 4:
                branches.append(AvgPoolOpt(3, False, True, is_cuda=self.is_cuda))
            elif branch_id == 5:
                branches.append(MaxPoolOpt(3, False, True, is_cuda=self.is_cuda))
            elif branch_id == 6:
                branches.append(
                    RnnOpt(out_filters, self.lstm_out_keep_prob, is_cuda=self.is_cuda)
                )
            elif branch_id == 7:
                branches.append(
                    AttentionOpt(
                        out_filters,
                        4,
                        self.attention_keep_prob,
                        self.is_mask,
                        is_cuda=self.is_cuda,
                        norm_type=norm_type,
                    )
                )
            branches = nn.ModuleList(branches)
            norm_func = None
            if whether_add_norm:
                norm_func = NormFactory(
                    num_features=self.out_filters,
                    pre_mask=False,
                    post_mask=True,
                    is_cuda=self.is_cuda,
                    norm_type=norm_type,
                ).create_norm_func()
        else:
            for operation_id in [0, 1, 2, 3]:  # conv_opt
                filter_size = size[operation_id]
                separable = separables[operation_id]
                op = ConvOpt(
                    filter_size,
                    out_filters,
                    out_filters,
                    self.cnn_keep_prob,
                    False,
                    True,
                    is_cuda=self.is_cuda,
                )
                branches.append(op)
            branches.append(AvgPoolOpt(3, False, True, is_cuda=self.is_cuda))
            branches.append(MaxPoolOpt(3, False, True, is_cuda=self.is_cuda))
            branches.append(
                RnnOpt(out_filters, self.lstm_out_keep_prob, is_cuda=self.is_cuda)
            )
            branches.append(
                AttentionOpt(
                    out_filters,
                    4,
                    self.attention_keep_prob,
                    self.is_mask,
                    is_cuda=self.is_cuda,
                    norm_type=norm_type,
                )
            )
            branches = nn.ModuleList(branches)
            norm_func = NormFactory(
                num_features=self.out_filters,
                pre_mask=False,
                post_mask=True,
                is_cuda=self.is_cuda,
                norm_type=norm_type,
            ).create_norm_func()

        return nn.ModuleList([branches, norm_func])

    def run_fixed_layer(
        self, x, mask, prev_layers, layers, layer_id, start_idx, final_flags
    ):
        layer = layers[0]
        norm_func = layers[1]

        if len(prev_layers) > 0:
            if self.multi_path:
                pre_layer_id = self.sample_arc[start_idx]
                num_pre_layers = len(prev_layers)
                if num_pre_layers > 5:
                    num_pre_layers = 5
                if pre_layer_id >= num_pre_layers:
                    final_flags[-1] = 0
                    inputs = prev_layers[-1]
                else:
                    layer_idx = len(prev_layers) - 1 - pre_layer_id
                    final_flags[layer_idx] = 0
                    inputs = prev_layers[layer_idx]
            else:
                inputs = prev_layers[-1]
                final_flags[-1] = 0
        else:
            inputs = x

        if self.multi_path:
            start_idx += 1

        branches = []
        # run branch op
        if self.fixed_flag:
            branch_id = 0
        else:
            branch_id = self.sample_arc[start_idx]
        branches.append(layer[branch_id](inputs, mask))

        if layer_id == 0:
            out = sum(branches)
        else:
            skip_start = start_idx + 1
            skip = self.sample_arc[skip_start : skip_start + layer_id]

            res_layers = []
            for i in range(layer_id):
                if skip[i] == 1:
                    res_layers.append(prev_layers[i])
                    final_flags[i] = 0
            prev = branches + res_layers
            out = sum(prev)  # tensor sum
            if len(prev) > 1:
                out = norm_func(out, mask)

        return out


# final model
class NASADRModel(nn.Module):
    def __init__(
        self,
        final_combination_type="concat",
        share_model=False,
        vocab_size=49293,
        num_layers=6,
        num_branches=8,
        fixed_arc="0 3 0 4 0 1 7 1 0 0 0 1 1 0 0 4 0 0 1 0 4 2 1 1 1 0 1",
        out_filters_scale=1,
        out_filters=512,
        cnn_keep_prob=1.0,
        final_output_keep_prob=1.0,
        embed_keep_prob=1.0,
        lstm_out_keep_prob=1.0,
        attention_keep_prob=1.0,
        class_num=5,
        pool_step=3,
        multi_path=True,
        input_positional_encoding=True,
        is_sinusolid=False,
        all_layer_output=False,
        num_last_layer_output=0,
        is_mask=True,
        output_type="avg_pool",
        norm_type="BN",
        max_doc_length=20,
        max_query_length=20,
        max_ad_length=20,
        downscaled_size=64,
        max_n_letter=20,
        is_wordpiece=False,
        is_extra_features=False,
        seed=1234,
        is_cuda=False,
    ):
        super(NASADRModel, self).__init__()

        self.final_combination_type = final_combination_type
        self.share_model = share_model
        self.input_positional_encoding = input_positional_encoding
        self.is_sinusolid = is_sinusolid
        self.out_filters = out_filters
        self.downscaled_size = downscaled_size
        self.max_n_letter = max_n_letter
        self.is_wordpiece = is_wordpiece
        self.is_extra_features = is_extra_features

        self.model_query = GeneralChild(
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_branches=num_branches,
            fixed_arc=fixed_arc,
            out_filters_scale=out_filters_scale,
            out_filters=out_filters,
            cnn_keep_prob=cnn_keep_prob,
            final_output_keep_prob=final_output_keep_prob,
            embed_keep_prob=embed_keep_prob,
            lstm_out_keep_prob=lstm_out_keep_prob,
            attention_keep_prob=attention_keep_prob,
            class_num=class_num,
            pool_step=pool_step,
            multi_path=multi_path,
            input_positional_encoding=input_positional_encoding,
            is_sinusolid=is_sinusolid,
            all_layer_output=all_layer_output,
            num_last_layer_output=num_last_layer_output,
            is_mask=is_mask,
            output_type=output_type,
            norm_type=norm_type,
            max_doc_length=max_doc_length,
            downscaled_size=downscaled_size,
            seed=seed,
            is_cuda=is_cuda,
        )
        if self.share_model:
            self.model_doc = self.model_query
            self.model_ad = self.model_query
        else:
            self.model_doc = GeneralChild(
                vocab_size=vocab_size,
                num_layers=num_layers,
                num_branches=num_branches,
                fixed_arc=fixed_arc,
                out_filters_scale=out_filters_scale,
                out_filters=out_filters,
                cnn_keep_prob=cnn_keep_prob,
                final_output_keep_prob=final_output_keep_prob,
                embed_keep_prob=embed_keep_prob,
                lstm_out_keep_prob=lstm_out_keep_prob,
                attention_keep_prob=attention_keep_prob,
                class_num=class_num,
                pool_step=pool_step,
                multi_path=multi_path,
                input_positional_encoding=input_positional_encoding,
                is_sinusolid=is_sinusolid,
                all_layer_output=all_layer_output,
                num_last_layer_output=num_last_layer_output,
                is_mask=is_mask,
                output_type=output_type,
                norm_type=norm_type,
                max_doc_length=max_doc_length,
                downscaled_size=downscaled_size,
                seed=seed,
                is_cuda=is_cuda,
            )
            self.model_ad = GeneralChild(
                vocab_size=vocab_size,
                num_layers=num_layers,
                num_branches=num_branches,
                fixed_arc=fixed_arc,
                out_filters_scale=out_filters_scale,
                out_filters=out_filters,
                cnn_keep_prob=cnn_keep_prob,
                final_output_keep_prob=final_output_keep_prob,
                embed_keep_prob=embed_keep_prob,
                lstm_out_keep_prob=lstm_out_keep_prob,
                attention_keep_prob=attention_keep_prob,
                class_num=class_num,
                pool_step=pool_step,
                multi_path=multi_path,
                input_positional_encoding=input_positional_encoding,
                is_sinusolid=is_sinusolid,
                all_layer_output=all_layer_output,
                num_last_layer_output=num_last_layer_output,
                is_mask=is_mask,
                output_type=output_type,
                norm_type=norm_type,
                max_doc_length=max_ad_length,
                downscaled_size=downscaled_size,
                seed=seed,
                is_cuda=is_cuda,
            )
        updated_filters = out_filters
        if self.downscaled_size:
            updated_filters = self.downscaled_size
        if self.final_combination_type == "sum":
            self.linear_out = nn.Linear(updated_filters, 1)
        elif self.final_combination_type == "concat":
            self.linear_out = nn.Linear(3 * updated_filters, 1)
            self.dense = nn.Linear(3 * updated_filters, 3 * updated_filters)
        elif self.final_combination_type == "cross_concat":
            # TODO:need confirmation on how to implement "cross_concat" for three vectors
            raise Exception(
                "Unimplemented condition when final_combination_type == cross_concat"
            )
        elif self.final_combination_type == "max_concat_residual_skip_layer":
            self.linear_out = nn.Linear(updated_filters, 1)
            self.dense1 = nn.Linear(updated_filters, updated_filters)
            self.dense2 = nn.Linear(updated_filters, updated_filters)
        self.embed_dropout = nn.Dropout(p=(1 - embed_keep_prob))
        self.output_dropout = nn.Dropout(p=(1 - final_output_keep_prob))

        if fixed_arc is not None:
            fixed_arc = np.array([int(x) for x in fixed_arc.split(" ") if x])
            self.sample_arc = fixed_arc

        max_position_embeddings = 128
        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size, embed_size=out_filters, padding_idx=0
        )
        if self.input_positional_encoding:
            if self.is_sinusolid:
                self.query_position_embedding = PositionalEmbedding(
                    out_filters, max_position_embeddings
                )
                self.doc_position_embedding = PositionalEmbedding(
                    out_filters, max_position_embeddings
                )
                self.ad_position_embedding = PositionalEmbedding(
                    out_filters, max_position_embeddings
                )
            else:
                self.query_position_embedding = nn.Embedding(
                    max_position_embeddings, out_filters
                )
                self.doc_position_embedding = nn.Embedding(
                    max_position_embeddings, out_filters
                )
                self.ad_position_embedding = nn.Embedding(
                    max_position_embeddings, out_filters
                )

        if self.is_extra_features:
            self.dense3 = nn.Linear(4, 1)

    def forward(
        self,
        q_ids,
        d_ids,
        ad_ids,
        q_pos_ids,
        d_pos_ids,
        ad_pos_ids,
        q_mask,
        d_mask,
        ad_mask,
        q_count,
        d_count,
        ad_count,
        extra_features=None,
        is_training=False,
    ):
        query_output = None
        if q_ids is not None:
            if self.is_wordpiece:
                query = self.token_embedding(q_ids)
            else:
                # The reason why the token embedding is more complicated for triletter is that:
                # The q_ids are the triletter ids of the query input, and each word has 20 triltters
                # as suggested by self.max_n_letter. Sequence length is the number of words in the query
                # for triletter encoding in our implementation. We will sum the embedding for all triletters
                # for a given word. Document side is the same as query side.
                q_embed_mask = (q_ids > 0).float()
                query = self.token_embedding(q_ids)
                N, L, C = query.size()
                # N is the batch size
                # L is 12(max number of words) * 20 (max number of letters in a word)
                # C is the embedding size
                query = query * q_embed_mask.unsqueeze(2).repeat(1, 1, C)
                query = query.view(N, L // self.max_n_letter, self.max_n_letter, C)
                query = torch.sum(query, -2)
            if self.input_positional_encoding:
                q_position_embedding = self.query_position_embedding(q_pos_ids)
                mask = (
                    torch.unsqueeze(q_mask, dim=2)
                    .repeat(1, 1, self.out_filters)
                    .float()
                )
                q_position_embedding = q_position_embedding * mask
                query = query + q_position_embedding
            query = self.embed_dropout(query)
            # in mask_opt:
            # seq_mask = torch.unsqueeze(mask, 2) #bsz, l -> bsz, l, 1
            # seq_mask = torch.transpose(seq_mask.repeat(1, 1, seq.size()[1]), 1, 2)	#bsz, l, dim -> bsz, dim, l
            q_mask = torch.unsqueeze(q_mask, dim=1).repeat(
                1, self.out_filters, 1
            )  # bsz, dim, l
            query_output = self.model_query(query, q_mask, is_training)
            query_output = self.output_dropout(query_output)

        if d_ids is None and ad_ids is None:
            return None, query_output, None, None

        doc_output = None
        if d_ids is not None:
            if self.is_wordpiece:
                doc = self.token_embedding(d_ids)
            else:
                d_embed_mask = (d_ids > 0).float()
                doc = self.token_embedding(d_ids)
                N, L, C = doc.size()
                doc = doc * d_embed_mask.unsqueeze(2).repeat(1, 1, C)
                doc = doc.view(N, L // self.max_n_letter, self.max_n_letter, C)
                doc = torch.sum(doc, -2)
                d_count = d_count.float()
                d_count += torch.eq(d_count, 0.0).float() * 1e-12
                d_count = d_count.unsqueeze(-1).repeat(1, 1, C)
                doc = torch.div(doc, d_count)
            if self.input_positional_encoding:
                d_position_embedding = self.doc_position_embedding(d_pos_ids)
                mask = (
                    torch.unsqueeze(d_mask, dim=2)
                    .repeat(1, 1, self.out_filters)
                    .float()
                )
                d_position_embedding = d_position_embedding * mask
                doc = doc + d_position_embedding
            doc = self.embed_dropout(doc)
            # update to adapt new mask format: bsz, dim, l
            d_mask = torch.unsqueeze(d_mask, dim=1).repeat(1, self.out_filters, 1)

            doc_output = self.model_doc(doc, d_mask, is_training)
            doc_output = self.output_dropout(doc_output)

        ad_output = None
        if ad_ids is not None:
            if self.is_wordpiece:
                ad = self.token_embedding(ad_ids)
            else:
                ad_embed_mask = (ad_ids > 0).float()
                ad = self.token_embedding(ad_ids)
                N, L, C = ad.size()
                ad = ad * ad_embed_mask.unsqueeze(2).repeat(1, 1, C)
                ad = ad.view(N, L // self.max_n_letter, self.max_n_letter, C)
                ad = torch.sum(ad, -2)
                # TODO: ask for difference here from query side
                ad_count = ad_count.float()
                ad_count += torch.eq(ad_count, 0.0).float() * 1e-12
                ad_count = ad_count.unsqueeze(-1).repeat(1, 1, C)
                ad = torch.div(ad, ad_count)
            if self.input_positional_encoding:
                ad_position_embedding = self.ad_position_embedding(ad_pos_ids)
                mask = (
                    torch.unsqueeze(ad_mask, dim=2)
                    .repeat(1, 1, self.out_filters)
                    .float()
                )
                ad_position_embedding = ad_position_embedding * mask
                ad = ad + ad_position_embedding
            ad = self.embed_dropout(ad)
            # update to adapt new mask format: bsz, dim, l
            ad_mask = torch.unsqueeze(ad_mask, dim=1).repeat(1, self.out_filters, 1)

            ad_output = self.model_ad(ad, ad_mask, is_training)
            ad_output = self.output_dropout(ad_output)

        output = None
        if (
            doc_output is not None
            and query_output is not None
            and ad_output is not None
        ):
            if self.final_combination_type == "sum":
                output = query_output + doc_output + ad_output
            elif self.final_combination_type == "concat":
                output = torch.cat((query_output, doc_output, ad_output), dim=-1)
                output = torch.relu(self.dense(output) + output)
            elif self.final_combination_type == "cross_concat":
                # TODO:need confirmation on how to implement "cross_concat" for three vectors
                raise Exception(
                    "Unimplemented condition when final_combination_type == cross_concat"
                )

            elif self.final_combination_type == "max_concat_residual_skip_layer":
                output = torch.max(query_output, doc_output)
                output = torch.max(output, ad_output)
                output = torch.relu(
                    self.dense2(torch.relu(self.dense1(output))) + output
                )
            elif self.final_combination_type == "cosine":
                raise Exception(
                    "Unimplemented condition when final_combination_type == cosine"
                )

            output = self.linear_out(output).squeeze(-1)
            if self.is_extra_features:
                extra_features = extra_features.float()
                lusweight = extra_features[:, 0].squeeze(-1)
                adjustment = self.dense3(extra_features[:, 1:]).squeeze(-1)
                output = output + lusweight + adjustment

        return output, query_output, doc_output, ad_output
