from typing import Sequence

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ReferenceEncoder(nn.Module):
    def __init__(
        self,
        in_dim=80,
        conv_layers: int = 6,
        conv_chans_list: Sequence[int] = (32, 32, 64, 64, 128, 128),
        conv_kernel_size: int = 3,
        conv_stride: int = 2,
        gru_layers: int = 1,
        gru_units: int = 128,
        enforce_sorted=True,
    ):
        super(ReferenceEncoder, self).__init__()
        self.conv_stride = conv_stride
        self.conv_layers = conv_layers
        self.enforce_sorted = enforce_sorted

        # check hyperparameters are valid
        assert conv_kernel_size % 2 == 1, "kernel size must be odd."
        assert (
            len(conv_chans_list) == conv_layers
        ), "the number of conv layers and length of channels list must be the same."

        convs = []
        padding = (conv_kernel_size - 1) // 2
        for i in range(conv_layers):
            conv_in_chans = 1 if i == 0 else conv_chans_list[i - 1]
            conv_out_chans = conv_chans_list[i]
            convs += [
                torch.nn.Conv2d(
                    conv_in_chans,
                    conv_out_chans,
                    kernel_size=conv_kernel_size,
                    stride=conv_stride,
                    padding=padding,
                    # Do not use bias due to the following batch norm
                    bias=False,
                ),
                torch.nn.BatchNorm2d(conv_out_chans),
                torch.nn.ReLU(inplace=True),
            ]
        self.convs = torch.nn.Sequential(*convs)

        self.conv_layers = conv_layers
        self.kernel_size = conv_kernel_size
        self.stride = conv_stride
        self.padding = padding

        # get the number of GRU input units
        gru_in_units = in_dim
        for _ in range(conv_layers):
            gru_in_units = (
                gru_in_units - conv_kernel_size + 2 * padding
            ) // conv_stride + 1
        gru_in_units *= conv_out_chans
        self.gru = torch.nn.GRU(gru_in_units, gru_units, gru_layers, batch_first=True)

    def forward(self, speech: torch.Tensor, in_lens=None) -> torch.Tensor:
        """Forward step

        Args:
            speech (Tensor): Batch of padded target features (B, Lmax, in_dim).

        Returns:
            tuple
                - GRU states (B, Lmax', gru_units)
                - Reference embedding (B, gru_units)
        """
        in_lens = (
            torch.tensor(in_lens, dtype=torch.long, device=speech.device)
            if isinstance(in_lens, list)
            else in_lens
        )

        batch_size = speech.size(0)
        xs = speech.unsqueeze(1)  # (B, 1, Lmax, in_dim)
        hs = self.convs(xs).transpose(1, 2)  # (B, Lmax', conv_out_chans, in_dim')
        time_length = hs.size(1)
        hs = hs.contiguous().view(batch_size, time_length, -1)  # (B, Lmax', gru_units)
        self.gru.flatten_parameters()
        if in_lens is None:
            out, ref_embs = self.gru(hs)  # (gru_layers, batch_size, gru_units)
        else:
            # NOTE: compute effective lengths for sub-sampled features
            hs_lens = torch.ceil(
                in_lens.float() / (self.conv_stride ** self.conv_layers)
            ).long()
            # safe guard
            hs_lens = torch.clamp(hs_lens, 1)
            hs = pack_padded_sequence(
                hs,
                hs_lens.to("cpu"),
                batch_first=True,
                enforce_sorted=self.enforce_sorted,
            )
            out, ref_embs = self.gru(hs)  # (gru_layers, batch_size, gru_units)
            out, _ = pad_packed_sequence(out, batch_first=True)

        ref_embs = ref_embs[-1]  # (batch_size, gru_units)

        return out, ref_embs


class StyleTokenLayer(torch.nn.Module):
    def __init__(
        self,
        ref_embed_dim: int = 128,
        gst_tokens: int = 10,
        gst_token_dim: int = 256,
        gst_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        gst_embs = torch.randn(gst_tokens, gst_token_dim // gst_heads)
        self.register_parameter("gst_embs", torch.nn.Parameter(gst_embs))
        self.mha = nn.MultiheadAttention(
            ref_embed_dim,
            gst_heads,
            kdim=gst_token_dim // gst_heads,
            vdim=gst_token_dim // gst_heads,
            batch_first=True,
            dropout=dropout,
        )

    def forward(self, ref_embs: torch.Tensor) -> torch.Tensor:
        """Forward step

        Args:
            ref_embs (Tensor): Reference embeddings (B, ref_embed_dim).

        Returns:
            Tensor: Style token embeddings (B, gst_token_dim).
        """
        batch_size = ref_embs.size(0)
        # (B, 1, ref_embed_dim)
        # NOTE(kan-bayashi): Should we apply Tanh?
        ref_embs = ref_embs.unsqueeze(1) if ref_embs.dim() == 2 else ref_embs

        # (num_tokens, token_dim) -> (batch_size, num_tokens, token_dim)
        gst_embs = torch.tanh(self.gst_embs).unsqueeze(0).expand(batch_size, -1, -1)
        style_embs, _ = self.mha(query=ref_embs, key=gst_embs, value=gst_embs)

        return style_embs.squeeze(1)


class TimeVaryingStyleStateEncoder(torch.nn.Module):
    def __init__(
        self,
        in_dim: int = 80,
        num_heads: int = 4,
        conv_layers: int = 6,
        conv_chans_list: Sequence[int] = (32, 32, 64, 64, 128, 128),
        conv_kernel_size: int = 3,
        conv_stride: int = 2,
        gru_layers: int = 1,
        gru_units: int = 128,
        query_dim=None,
    ):
        super().__init__()
        if query_dim is None:
            raise ValueError("query_dim must be specified explicitly.")
        self.query_proj = torch.nn.Linear(query_dim, gru_units)

        self.encoder = ReferenceEncoder(
            in_dim=in_dim,
            conv_layers=conv_layers,
            conv_chans_list=conv_chans_list,
            conv_kernel_size=conv_kernel_size,
            conv_stride=conv_stride,
            gru_layers=gru_layers,
            gru_units=gru_units,
        )
        self.mha = nn.MultiheadAttention(
            gru_units,
            num_heads=num_heads,
            kdim=gru_units,
            vdim=gru_units,
            batch_first=True,
        )

    def forward(self, feats, query, in_lens=None) -> torch.Tensor:
        """Forward step

        Args:
            feats (Tensor): Batch of padded target features (B, Lmax, odim).
            query (Tensor): Batch of padded query features (B, Lmax, idim).

        Returns:
            tuple:
                - Global style token embeddings (B, 1, token_dim).
                - Time-varying style embeddings (B, Lmax', gru_units)
        """
        encoder_states, ref_embs = self.encoder(feats, in_lens)

        # Use provided query to retrieve time-varying style token
        ref_embs = ref_embs.unsqueeze(1) if ref_embs.dim() == 2 else ref_embs
        ref_embs_expanded = ref_embs.expand(-1, query.size(1), -1)
        query = torch.cat([query, ref_embs_expanded], dim=-1)
        query = self.query_proj(query)
        time_varying_embs, _ = self.mha(
            query=query, key=encoder_states, value=encoder_states
        )

        return ref_embs, time_varying_embs


class TimeVaryingStyleTokenEncoder(torch.nn.Module):
    def __init__(
        self,
        in_dim: int = 80,
        gst_tokens: int = 10,
        gst_token_dim: int = 256,
        gst_heads: int = 4,
        conv_layers: int = 6,
        conv_chans_list: Sequence[int] = (32, 32, 64, 64, 128, 128),
        conv_kernel_size: int = 3,
        conv_stride: int = 2,
        gru_layers: int = 1,
        gru_units: int = 128,
        query_dim=None,
        query_concat_global_emb: bool = True,
        gst: bool = False,
    ):
        super().__init__()
        if query_dim is None:
            raise ValueError("query_dim must be specified explicitly.")
        self.query_proj = torch.nn.Linear(query_dim, gru_units)
        self.query_concat_global_emb = query_concat_global_emb
        self.gst = gst

        self.encoder = ReferenceEncoder(
            in_dim=in_dim,
            conv_layers=conv_layers,
            conv_chans_list=conv_chans_list,
            conv_kernel_size=conv_kernel_size,
            conv_stride=conv_stride,
            gru_layers=gru_layers,
            gru_units=gru_units,
        )

        # Style token layer for time-varying style token
        self.style_token_layer = StyleTokenLayer(
            ref_embed_dim=gru_units,
            gst_tokens=gst_tokens,
            gst_token_dim=gst_token_dim,
            gst_heads=gst_heads,
        )

        # Style token layer for global style token
        if self.gst:
            self.global_style_token_layer = StyleTokenLayer(
                ref_embed_dim=gru_units,
                gst_tokens=gst_tokens,
                gst_token_dim=gst_token_dim,
                gst_heads=gst_heads,
            )

    def forward(self, feats, query, in_lens) -> torch.Tensor:
        """Forward step

        Args:
            speech (Tensor): Batch of padded target features (B, Lmax, odim).

        Returns:
            tuple:
                - Global style token embeddings (B, 1, token_dim).
                - Time-varying style embeddings (B, Lmax', gru_units)
        """
        _, ref_embs = self.encoder(feats, in_lens)

        # Use provided query to retrieve time-varying style token
        ref_embs = ref_embs.unsqueeze(1) if ref_embs.dim() == 2 else ref_embs
        if self.query_concat_global_emb:
            ref_embs_expanded = ref_embs.expand(-1, query.size(1), -1)
            query = torch.cat([query, ref_embs_expanded], dim=-1)
        query = self.query_proj(query)
        time_varying_embs = self.style_token_layer(query)

        # Convert reference embedding into global style token
        if self.gst:
            ref_embs = self.global_style_token_layer(ref_embs)
            ref_embs = ref_embs.unsqueeze(1) if ref_embs.dim() == 2 else ref_embs

        return ref_embs, time_varying_embs


class GlobalStyleTokenEncoder(torch.nn.Module):
    def __init__(
        self,
        in_dim: int = 80,
        gst_tokens: int = 10,
        gst_token_dim: int = 256,
        gst_heads: int = 4,
        conv_layers: int = 6,
        conv_chans_list: Sequence[int] = (32, 32, 64, 64, 128, 128),
        conv_kernel_size: int = 3,
        conv_stride: int = 2,
        gru_layers: int = 1,
        gru_units: int = 128,
        enforce_sorted=True,
    ):
        super().__init__()
        self.encoder = ReferenceEncoder(
            in_dim=in_dim,
            conv_layers=conv_layers,
            conv_chans_list=conv_chans_list,
            conv_kernel_size=conv_kernel_size,
            conv_stride=conv_stride,
            gru_layers=gru_layers,
            gru_units=gru_units,
            enforce_sorted=enforce_sorted,
        )
        self.global_style_token_layer = StyleTokenLayer(
            ref_embed_dim=gru_units,
            gst_tokens=gst_tokens,
            gst_token_dim=gst_token_dim,
            gst_heads=gst_heads,
        )

    def forward(self, feats, query, in_lens) -> torch.Tensor:
        _, ref_embs = self.encoder(feats, in_lens)

        style_embs = self.global_style_token_layer(ref_embs)
        style_embs = style_embs.unsqueeze(1) if style_embs.dim() == 2 else style_embs

        return style_embs
