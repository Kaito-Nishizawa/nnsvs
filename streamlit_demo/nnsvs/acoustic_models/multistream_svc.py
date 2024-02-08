"""Multi-stream acoustic model implementation designed for singing voice conversion (SVC)"""

from typing import Sequence

import torch
from nnsvs.acoustic_models.util import get_voiced_segment, pad_inference
from nnsvs.base import BaseModel, PredictionType
from nnsvs.multistream import split_streams
from nnsvs.pitch import quantize_f0_midi
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

__all__ = [
    "MultistreamMelF0Model",
    "MultistreamSpkMelF0Model",
]


# DEPRECATEDD: see stle_tokens.py for the new implementation
class ReferenceEncoder(nn.Module):
    def __init__(
        self,
        idim=80,
        conv_layers: int = 6,
        conv_chans_list: Sequence[int] = (32, 32, 64, 64, 128, 128),
        conv_kernel_size: int = 3,
        conv_stride: int = 2,
        gru_layers: int = 1,
        gru_units: int = 128,
    ):
        super(ReferenceEncoder, self).__init__()
        self.conv_stride = conv_stride
        self.conv_layers = conv_layers

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
        gru_in_units = idim
        for _ in range(conv_layers):
            gru_in_units = (
                gru_in_units - conv_kernel_size + 2 * padding
            ) // conv_stride + 1
        gru_in_units *= conv_out_chans
        self.gru = torch.nn.GRU(gru_in_units, gru_units, gru_layers, batch_first=True)

    def forward(self, speech: torch.Tensor, in_lens=None) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            speech (Tensor): Batch of padded target features (B, Lmax, idim).

        Returns:
            Tensor: Reference embedding (B, gru_units)
        """
        in_lens = (
            torch.tensor(in_lens, dtype=torch.long, device=speech.device)
            if isinstance(in_lens, list)
            else in_lens
        )

        batch_size = speech.size(0)
        xs = speech.unsqueeze(1)  # (B, 1, Lmax, idim)
        hs = self.convs(xs).transpose(1, 2)  # (B, Lmax', conv_out_chans, idim')
        # NOTE(kan-bayashi): We need to care the length?
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
            hs = pack_padded_sequence(hs, hs_lens.to("cpu"), batch_first=True)
            out, ref_embs = self.gru(hs)  # (gru_layers, batch_size, gru_units)
            out, _ = pad_packed_sequence(out, batch_first=True)

        return out


# DEPRECATEDD: see stle_tokens.py for the new implementation


class LinguisticInstNormEncoder(BaseModel):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        stream_sizes: list,
        model=None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.stream_sizes = stream_sizes
        self.norm_layer = nn.InstanceNorm1d(stream_sizes[0], affine=False)
        self.model = model

    def forward(self, x, lengths=None, y=None, spk=None, spk_enc_inp=None):
        ling, lf0, vuv, erg = split_streams(x, self.stream_sizes)

        # Perform instance normalization to linguistic features
        # to remove speaker information effectively
        # https://arxiv.org/abs/1904.05742
        # NOTE: channel must be in the second axis
        # (B, T, C) -> (B, C, T)
        ling = ling.transpose(1, 2)
        ling = self.norm_layer(ling)
        # (B, C, T) -> (B, T, C)
        ling = ling.transpose(1, 2)

        out = torch.cat([ling, lf0, vuv, erg], dim=-1)

        return out if self.model is None else self.model(out, lengths, y)


class SpeakerLinguisticInstNormEncoder(BaseModel):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        stream_sizes: list,
        model: nn.Module,
        gru_units: int = 128,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.stream_sizes = stream_sizes
        self.norm_layer = nn.InstanceNorm1d(stream_sizes[0], affine=False)
        self.model = model

        # TODO: remove hardcode
        self.speaker_encoder = ReferenceEncoder(idim=80, gru_units=gru_units)
        self.mha = nn.MultiheadAttention(
            in_dim, num_heads=1, kdim=gru_units, vdim=gru_units, batch_first=True
        )

    def forward(self, x, lengths=None, y=None, spk=None, spk_enc_inp=None):
        ling, lf0, vuv, erg = split_streams(x, self.stream_sizes)

        # Perform instance normalization to linguistic features
        # to remove speaker information effectively
        # https://arxiv.org/abs/1904.05742
        # NOTE: channel must be in the second axis
        # (B, T, C) -> (B, C, T)
        ling = ling.transpose(1, 2)
        ling = self.norm_layer(ling)
        # (B, C, T) -> (B, T, C)
        ling = ling.transpose(1, 2)

        out = torch.cat([ling, lf0, vuv, erg], dim=-1)

        # Time-varying speaker embeddings
        # (B, T//down_sample_factor, C)
        assert spk_enc_inp is not None
        if y is not None:
            time_varying_spk_emb = self.speaker_encoder(spk_enc_inp, lengths)
        else:
            # infernece: lengths can be different
            time_varying_spk_emb = self.speaker_encoder(spk_enc_inp)

        # Use abstracted content information as query and blend information with
        # time-varying speaker embeddings
        time_varying_spk_emb, _ = self.mha(
            query=out, key=time_varying_spk_emb, value=time_varying_spk_emb
        )
        out = out + time_varying_spk_emb

        return self.model(out, lengths, y)


class MultistreamMelF0Model(BaseModel):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        stream_sizes: list,
        reduction_factor: int,
        mel_model: nn.Module,
        lf0_model: nn.Module,
        vuv_model: nn.Module,
        freeze_lf0_model=True,
        freeze_vuv_model=True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.stream_sizes = stream_sizes
        self.reduction_factor = reduction_factor

        assert len(stream_sizes) in [3]

        self.mel_model = mel_model
        self.lf0_model = lf0_model
        self.vuv_model = vuv_model
        self.freeze_lf0_model = freeze_lf0_model
        self.freeze_vuv_model = freeze_vuv_model
        if freeze_lf0_model:
            for p in lf0_model.parameters():
                p.requires_grad = False
        if freeze_vuv_model:
            for p in vuv_model.parameters():
                p.requires_grad = False

    def _set_lf0_params(self):
        # Special care for residual F0 prediction models
        # NOTE: don't overwrite out_lf0_idx and in_lf0_idx
        if hasattr(self.lf0_model, "out_lf0_mean"):
            self.lf0_model.in_lf0_min = self.in_lf0_min
            self.lf0_model.in_lf0_max = self.in_lf0_max
            self.lf0_model.out_lf0_mean = self.out_lf0_mean
            self.lf0_model.out_lf0_scale = self.out_lf0_scale

    def prediction_type(self):
        return PredictionType.MULTISTREAM_HYBRID

    def is_autoregressive(self):
        return (
            self.mel_model.is_autoregressive()
            or self.lf0_model.is_autoregressive()
            or self.vuv_model.is_autoregressive()
        )

    def has_residual_lf0_prediction(self):
        return True

    def requires_spk(self):
        return self.mel_model.requires_spk()

    # TODO: more carefull treatment for spk
    def forward(
        self,
        x,
        lengths=None,
        y=None,
        spk=None,
        spk_enc_inp=None,
        wave1=None,
        wave2=None,
        wave_lens=None,
    ):
        self._set_lf0_params()
        assert x.shape[-1] == self.in_dim
        is_inference = y is None

        if y is not None:
            # Teacher-forcing
            outs = split_streams(y, self.stream_sizes)
            y_mel, y_lf0, y_vuv = outs
        else:
            # Inference
            y_mel, y_lf0, y_vuv = (
                None,
                None,
                None,
            )

        # TODO: in the future, we may want to consider additional F0 model
        # Predict continuous log-F0 first
        if is_inference:
            lf0, lf0_residual = self.lf0_model.inference(x, lengths), None
        else:
            if self.freeze_lf0_model:
                # dummy
                lf0 = y_lf0
                lf0_residual = torch.zeros_like(lf0)
            else:
                if self.lf0_model.has_residual_lf0_prediction():
                    lf0, lf0_residual = self.lf0_model(x, lengths, y_lf0)
                else:
                    lf0, lf0_residual = self.lf0_model(x, lengths, y_lf0), None

        # Predict mel
        if is_inference:
            mel_inp = x
            mel = self.mel_model.inference(
                mel_inp, lengths, spk=spk, spk_enc_inp=spk_enc_inp
            )
        else:
            mel_inp = x
            mel = self.mel_model(
                mel_inp, lengths, y_mel, spk=spk, spk_enc_inp=spk_enc_inp
            )

        # Predict V/UV
        if is_inference:
            vuv_inp = x
            vuv = self.vuv_model.inference(vuv_inp, lengths)
        else:
            vuv_inp = x
            vuv = self.vuv_model(vuv_inp, lengths, y_vuv)

        if is_inference:
            if self.lf0_model.prediction_type() == PredictionType.PROBABILISTIC:
                lf0_ = lf0[0]
            else:
                lf0_ = lf0
            if self.mel_model.prediction_type() == PredictionType.PROBABILISTIC:
                mel_ = mel[0]
            else:
                mel_ = mel
            out = torch.cat([mel_, lf0_, vuv], dim=-1)
            assert out.shape[-1] == self.out_dim
            # TODO: better design
            return out, out
        else:
            return (mel, lf0, vuv), lf0_residual

    def inference(self, x, lengths=None, spk=None, spk_enc_inp=None):
        return pad_inference(
            model=self,
            x=x,
            lengths=lengths,
            spk=spk,
            spk_enc_inp=spk_enc_inp,
            reduction_factor=self.reduction_factor,
            mdn=True,
        )


class MultistreamSpkMelF0Model(BaseModel):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        stream_sizes: list,
        reduction_factor: int,
        preprocess_encoder: nn.Module,
        timbre_encoder: nn.Module,
        spk_proj: nn.Module,
        mel_model: nn.Module,
        lf0_model: nn.Module,
        vuv_model: nn.Module,
        freeze_lf0_model=True,
        freeze_vuv_model=True,
        spk_embed_blend_mode=0,
        normalize_spk_embed=False,
        concat_spk_embed_as_timbre_query=False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.stream_sizes = stream_sizes
        self.reduction_factor = reduction_factor
        self.spk_embed_blend_mode = spk_embed_blend_mode
        self.normalize_spk_embed = normalize_spk_embed
        self.concat_spk_embed_as_timbre_query = concat_spk_embed_as_timbre_query

        assert len(stream_sizes) in [3]

        self.preprocess_encoder = preprocess_encoder
        self.spk_proj = spk_proj
        self.timbre_encoder = timbre_encoder
        self.mel_model = mel_model
        self.lf0_model = lf0_model
        self.vuv_model = vuv_model
        self.freeze_lf0_model = freeze_lf0_model
        self.freeze_vuv_model = freeze_vuv_model
        if freeze_lf0_model:
            for p in lf0_model.parameters():
                p.requires_grad = False
        if freeze_vuv_model:
            for p in vuv_model.parameters():
                p.requires_grad = False

    def _set_lf0_params(self):
        # Special care for residual F0 prediction models
        # NOTE: don't overwrite out_lf0_idx and in_lf0_idx
        if hasattr(self.lf0_model, "out_lf0_mean"):
            self.lf0_model.in_lf0_min = self.in_lf0_min
            self.lf0_model.in_lf0_max = self.in_lf0_max
            self.lf0_model.out_lf0_mean = self.out_lf0_mean
            self.lf0_model.out_lf0_scale = self.out_lf0_scale

    def prediction_type(self):
        return PredictionType.MULTISTREAM_HYBRID

    def is_autoregressive(self):
        return (
            self.mel_model.is_autoregressive()
            or self.lf0_model.is_autoregressive()
            or self.vuv_model.is_autoregressive()
        )

    def has_residual_lf0_prediction(self):
        return True

    def requires_spk(self):
        return self.mel_model.requires_spk()

    def forward(
        self,
        x,
        lengths=None,
        y=None,
        spk=None,
        spk_enc_inp=None,
        wave1=None,
        wave2=None,
        wave_lens=None,
    ):
        self._set_lf0_params()
        assert x.shape[-1] == self.in_dim
        is_inference = y is None

        if y is not None:
            # Teacher-forcing
            outs = split_streams(y, self.stream_sizes)
            y_mel, y_lf0, y_vuv = outs
        else:
            # Inference
            y_mel, y_lf0, y_vuv = (
                None,
                None,
                None,
            )

        x = self.preprocess_encoder(x, lengths)

        # Pre-extracted speaker embedding with projection
        spk_proj = self.spk_proj(spk)

        # Timbre encoder
        in_lens = None if is_inference else lengths
        if self.concat_spk_embed_as_timbre_query:
            assert spk.dim() == 3  # (B, 1, C)
            spk_proj_expanded = spk_proj.expand(-1, x.shape[1], -1)
            query = torch.cat([x, spk_proj_expanded], dim=-1)
        else:
            query = x
        g_timbre_emb, c_timbre_emb = self.timbre_encoder(
            feats=spk_enc_inp, query=query, in_lens=in_lens
        )
        assert x.shape[1] == c_timbre_emb.shape[1]
        assert g_timbre_emb.shape[-1] == c_timbre_emb.shape[-1]

        # Blend speaker embeddings
        if self.spk_embed_blend_mode == 0:
            spk = spk_proj + g_timbre_emb + c_timbre_emb
        elif self.spk_embed_blend_mode == 1:
            spk = spk_proj + g_timbre_emb
        elif self.spk_embed_blend_mode == 2:
            spk = spk_proj + c_timbre_emb
        elif self.spk_embed_blend_mode == 3:
            spk = g_timbre_emb + c_timbre_emb
        elif self.spk_embed_blend_mode == 4:
            spk = spk_proj
        elif self.spk_embed_blend_mode == 5:
            spk = g_timbre_emb
        elif self.spk_embed_blend_mode == 6:
            spk = c_timbre_emb

        if self.normalize_spk_embed:
            spk = F.normalize(spk, dim=-1)

        # TODO: in the future, we may want to consider additional F0 model
        # Predict continuous log-F0 first
        if is_inference:
            lf0, lf0_residual = self.lf0_model.inference(x, lengths), None
        else:
            if self.freeze_lf0_model:
                # dummy
                lf0 = y_lf0
                lf0_residual = torch.zeros_like(lf0)
            else:
                if self.lf0_model.has_residual_lf0_prediction():
                    lf0, lf0_residual = self.lf0_model(x, lengths, y_lf0)
                else:
                    lf0, lf0_residual = self.lf0_model(x, lengths, y_lf0), None

        # Predict mel
        if is_inference:
            mel_inp = x
            mel = self.mel_model.inference(
                mel_inp, lengths, spk=spk, spk_enc_inp=spk_enc_inp
            )
        else:
            mel_inp = x
            mel = self.mel_model(
                mel_inp, lengths, y_mel, spk=spk, spk_enc_inp=spk_enc_inp
            )

        # Predict V/UV
        if is_inference:
            vuv_inp = x
            vuv = self.vuv_model.inference(vuv_inp, lengths)
        else:
            vuv_inp = x
            vuv = self.vuv_model(vuv_inp, lengths, y_vuv)

        if is_inference:
            if self.lf0_model.prediction_type() == PredictionType.PROBABILISTIC:
                lf0_ = lf0[0]
            else:
                lf0_ = lf0
            if self.mel_model.prediction_type() == PredictionType.PROBABILISTIC:
                mel_ = mel[0]
            else:
                mel_ = mel
            out = torch.cat([mel_, lf0_, vuv], dim=-1)
            assert out.shape[-1] == self.out_dim
            # TODO: better design
            return out, out
        else:
            return (mel, lf0, vuv), lf0_residual

    def inference(self, x, lengths=None, spk=None, spk_enc_inp=None):
        return pad_inference(
            model=self,
            x=x,
            lengths=lengths,
            spk=spk,
            spk_enc_inp=spk_enc_inp,
            reduction_factor=self.reduction_factor,
            mdn=True,
        )


class LinguisticNormWrapper(BaseModel):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        model=None,
    ):
        super().__init__()
        self.norm_layer = nn.InstanceNorm1d(in_dim, affine=False)
        self.model = model

    def forward(self, x, lengths=None, y=None):
        ling = x

        # Perform instance normalization to linguistic features
        # to remove speaker information effectively
        # https://arxiv.org/abs/1904.05742
        # NOTE: channel must be in the second axis
        # (B, T, C) -> (B, C, T)
        ling = ling.transpose(1, 2)
        ling = self.norm_layer(ling)
        # (B, C, T) -> (B, T, C)
        ling = ling.transpose(1, 2)

        out = ling if self.model is None else self.model(ling, lengths, y)

        # (B, T, C)
        out = F.normalize(out, dim=-1)

        return out


class MultistreamLingSpkMelF0Model(BaseModel):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        in_stream_sizes: list,
        out_stream_sizes: list,
        reduction_factor: int,
        timbre_encoder: nn.Module,
        spk_proj: nn.Module,
        mel_model: nn.Module,
        lf0_model: nn.Module,
        vuv_model: nn.Module,
        ling_encoder=None,
        ling_post_extractor=None,
        freeze_ling_model=False,
        freeze_spk_proj=False,
        freeze_timbre_encoder=False,
        freeze_lf0_model=True,
        freeze_vuv_model=True,
        freeze_mel_model=False,
        spk_embed_blend_mode=0,
        normalize_spk_embed=False,
        concat_spk_embed_as_timbre_query=False,
        ling_only_timbre_query=False,
        in_lf0_idx=None,
        in_lf0_mean=None,
        in_lf0_scale=None,
        out_lf0_idx=80,
        out_lf0_mean=None,
        out_lf0_scale=None,
        quantize_in_lf0=False,
        quantize_smoothing_kernel_size=15,
        quantize_bins_per_octave=12,
        smoothing_filt=None,
        spk_embed_blend_mode_lf0=0,
        train_mode=None,
        lf0_timbre_encoder=None,
        mel_model_vuv_conditioning=True,
        vuv_model_vuv_conditioning=True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_stream_sizes = in_stream_sizes
        self.out_stream_sizes = out_stream_sizes
        self.reduction_factor = reduction_factor
        self.spk_embed_blend_mode = spk_embed_blend_mode
        self.spk_embed_blend_mode_lf0 = spk_embed_blend_mode_lf0
        self.normalize_spk_embed = normalize_spk_embed
        self.concat_spk_embed_as_timbre_query = concat_spk_embed_as_timbre_query
        self.ling_only_timbre_query = ling_only_timbre_query
        self.mel_model_vuv_conditioning = mel_model_vuv_conditioning
        self.vuv_model_vuv_conditioning = vuv_model_vuv_conditioning

        assert len(out_stream_sizes) in [3]

        self.ling_encoder = ling_encoder
        self.ling_post_extractor = ling_post_extractor
        self.spk_proj = spk_proj
        self.timbre_encoder = timbre_encoder
        self.mel_model = mel_model
        self.lf0_model = lf0_model
        self.vuv_model = vuv_model

        self.lf0_timbre_encoder = lf0_timbre_encoder

        if train_mode is not None:
            if train_mode == "lf0_only":
                freeze_ling_model = True
                freeze_spk_proj = True
                freeze_timbre_encoder = True
                freeze_lf0_model = False
                freeze_mel_model = True
                freeze_vuv_model = False
            elif train_mode == "spk_cond_only":
                freeze_ling_model = True
                freeze_spk_proj = True
                freeze_timbre_encoder = False
                freeze_mel_model = False
                for p in mel_model.parameters():
                    p.requires_grad = False
                # assuems mel_model is a diffusion model and also has
                # speaker conditioning layers
                for layer in mel_model.denoise_fn.residual_layers:
                    for p in layer.norm.parameters():
                        p.requires_grad = True

        self.freeze_ling_model = freeze_ling_model
        self.freeze_spk_proj = freeze_spk_proj
        self.freeze_timbre_encoder = freeze_timbre_encoder
        self.freeze_lf0_model = freeze_lf0_model
        self.freeze_mel_model = freeze_mel_model
        self.freeze_vuv_model = freeze_vuv_model
        if freeze_ling_model:
            for p in ling_encoder.parameters():
                p.requires_grad = False
            for p in ling_post_extractor.parameters():
                p.requires_grad = False
        if freeze_spk_proj:
            for p in spk_proj.parameters():
                p.requires_grad = False
        if freeze_timbre_encoder:
            for p in timbre_encoder.parameters():
                p.requires_grad = False
        if freeze_lf0_model:
            for p in lf0_model.parameters():
                p.requires_grad = False
        if freeze_mel_model:
            for p in mel_model.parameters():
                p.requires_grad = False
        if freeze_vuv_model:
            for p in vuv_model.parameters():
                p.requires_grad = False

        self.in_lf0_idx = in_lf0_idx
        self.in_lf0_mean = in_lf0_mean
        self.in_lf0_scale = in_lf0_scale
        self.out_lf0_idx = out_lf0_idx
        self.out_lf0_mean = out_lf0_mean
        self.out_lf0_scale = out_lf0_scale

        self.quantize_in_lf0 = quantize_in_lf0
        self.quantize_bins_per_octave = quantize_bins_per_octave
        if self.quantize_in_lf0:
            if smoothing_filt is None:
                self.smoothing_filt = nn.Conv1d(
                    1,
                    1,
                    quantize_smoothing_kernel_size,
                    bias=False,
                    padding="same",
                    padding_mode="replicate",
                )
                self.smoothing_filt.requires_grad = False
                self.smoothing_filt.weight.data.fill_(
                    1 / quantize_smoothing_kernel_size
                )
            else:
                self.smoothing_filt = smoothing_filt

    def _set_lf0_params(self):
        # Special care for residual F0 prediction models
        # NOTE: don't overwrite out_lf0_idx and in_lf0_idx
        if hasattr(self.lf0_model, "out_lf0_mean"):
            self.lf0_model.in_lf0_mean = self.in_lf0_mean
            self.lf0_model.in_lf0_scale = self.in_lf0_scale
            self.lf0_model.out_lf0_mean = self.out_lf0_mean
            self.lf0_model.out_lf0_scale = self.out_lf0_scale

    def prediction_type(self):
        return PredictionType.MULTISTREAM_HYBRID

    def is_autoregressive(self):
        return (
            self.mel_model.is_autoregressive()
            or self.lf0_model.is_autoregressive()
            or self.vuv_model.is_autoregressive()
        )

    def has_residual_lf0_prediction(self):
        return True

    def requires_spk(self):
        return self.mel_model.requires_spk()

    def forward(
        self,
        x,
        lengths=None,
        y=None,
        spk=None,
        spk_enc_inp=None,
        wave1=None,
        wave2=None,
        wave_lens=None,
    ):
        self._set_lf0_params()
        assert x.shape[-1] == self.in_dim
        is_inference = y is None

        if y is not None:
            # Teacher-forcing
            outs = split_streams(y, self.out_stream_sizes)
            y_mel, y_lf0, y_vuv = outs
        else:
            # Inference
            y_mel, y_lf0, y_vuv = (
                None,
                None,
                None,
            )
        if self.in_lf0_idx is not None:
            in_lf0 = x[:, :, self.in_lf0_idx].unsqueeze(-1)

        # Split input streams
        ling, lf0, vuv, erg = split_streams(x, self.in_stream_sizes)

        # Re-compute linguistic features
        if wave1 is not None and (not self.freeze_mel_model):
            assert wave2 is not None
            assert self.ling_post_extractor is not None
            ling_org = ling

            with torch.no_grad():
                ling = self.ling_encoder(wave1)
                if (wave1 == wave2).all():
                    ling_aug = ling
                else:
                    ling_aug = self.ling_encoder(wave2)
                if ling.shape[1] < x.shape[1]:
                    ling = F.pad(
                        ling, (0, 0, 0, x.shape[1] - ling.shape[1]), mode="replicate"
                    )
                    ling_aug = F.pad(
                        ling_aug,
                        (0, 0, 0, x.shape[1] - ling_aug.shape[1]),
                        mode="replicate",
                    )
                else:
                    ling = ling[:, : x.shape[1]]
                    ling_aug = ling_aug[:, : x.shape[1]]

            # Extract linguistic features from possibly entangled features
            ling = self.ling_post_extractor(ling)
            ling_aug = self.ling_post_extractor(ling_aug)
            # (2, B, T, C)
            ling_s = torch.stack([ling, ling_aug])

            # NOTE: use the input (clean) ling features for F0 prediction
            # since perturbuted (e.g., pitch-shifted) features may not be suitable
            ling_lf0 = self.ling_post_extractor(ling_org)
        else:
            ling = self.ling_post_extractor(ling)
            ling_lf0 = ling
            ling_s = None

        # Combine streams with new linguistic features
        x = torch.cat([ling, lf0, vuv, erg], dim=-1)

        # Pre-extracted speaker embedding with projection
        spk_proj = self.spk_proj(spk)

        # Timbre encoder
        in_lens = None if is_inference else lengths
        if self.ling_only_timbre_query:
            # TODO: ling or ling_lf0?
            query = ling
        elif self.concat_spk_embed_as_timbre_query:
            assert spk.dim() == 3  # (B, 1, C)
            spk_proj_expanded = spk_proj.expand(-1, x.shape[1], -1)
            query = torch.cat([x, spk_proj_expanded], dim=-1)
        else:
            query = x
        g_timbre_emb, c_timbre_emb = self.timbre_encoder(
            feats=spk_enc_inp, query=query, in_lens=in_lens
        )
        assert x.shape[1] == c_timbre_emb.shape[1]
        assert g_timbre_emb.shape[-1] == c_timbre_emb.shape[-1]

        # Blend speaker embeddings
        if self.spk_embed_blend_mode == 0:
            spk = spk_proj + g_timbre_emb + c_timbre_emb
        elif self.spk_embed_blend_mode == 1:
            spk = spk_proj + g_timbre_emb
        elif self.spk_embed_blend_mode == 2:
            spk = spk_proj + c_timbre_emb
        elif self.spk_embed_blend_mode == 3:
            spk = g_timbre_emb + c_timbre_emb
        elif self.spk_embed_blend_mode == 4:
            spk = spk_proj
        elif self.spk_embed_blend_mode == 5:
            spk = g_timbre_emb
        elif self.spk_embed_blend_mode == 6:
            spk = c_timbre_emb

        if self.normalize_spk_embed:
            spk = F.normalize(spk, dim=-1)

        # Make input for log-F0 model
        if self.quantize_in_lf0:
            assert self.in_lf0_idx is not None
            in_lf0_denorm = in_lf0 * self.in_lf0_scale + self.in_lf0_mean
            # NOTE: VUV is normalized to N(0, 1)
            vuv_binary = (vuv > 0.0).float()
            with torch.no_grad():
                in_lf0_denorm = quantize_f0_midi(
                    in_lf0_denorm.exp(),
                    smoothing_filt=self.smoothing_filt,
                    bins_per_octave=self.quantize_bins_per_octave,
                    vuv=vuv_binary,
                ).log()
            in_lf0_q = (in_lf0_denorm - self.in_lf0_mean) / self.in_lf0_scale
            lf0_inp = torch.cat([ling_lf0, in_lf0_q, vuv, erg], dim=-1)
        else:
            # (ling, lf0, vuv, erg)
            lf0_inp = torch.cat([ling_lf0, lf0, vuv, erg], dim=-1)
            lf0_inp = x

        # Predict continuous log-F0 first
        if self.lf0_model.requires_spk():
            if self.spk_embed_blend_mode_lf0 == 0:
                spk_lf0 = spk
            elif self.spk_embed_blend_mode_lf0 == 1:
                spk_lf0 = F.normalize(g_timbre_emb, dim=-1)
            elif self.spk_embed_blend_mode_lf0 == 2:
                # Nansy-TTS like
                if is_inference:
                    spk_enc_inp_seg = spk_enc_inp
                    seg_lens = None
                else:
                    spk_enc_inp_seg, seg_lens = get_voiced_segment(spk_enc_inp, vuv)
                spk_lf0 = self.lf0_timbre_encoder(
                    spk_enc_inp_seg, query=None, in_lens=seg_lens
                )
                spk_lf0 = F.normalize(g_timbre_emb, dim=-1)

            # TODO: may want to consider GST as alternative
            lf0_extra_kwargs = {"spk": spk_lf0, "spk_enc_inp": spk_enc_inp}
        else:
            lf0_extra_kwargs = {}
        if is_inference:
            lf0, lf0_residual = (
                self.lf0_model.inference(lf0_inp, lengths, **lf0_extra_kwargs),
                None,
            )
            if self.lf0_model.prediction_type() == PredictionType.PROBABILISTIC:
                lf0_cond = lf0[0]
            else:
                lf0_cond = lf0
        else:
            if self.freeze_lf0_model:
                # dummy
                lf0 = y_lf0
                lf0_residual = torch.zeros_like(lf0)
            else:
                if self.lf0_model.has_residual_lf0_prediction():
                    lf0, lf0_residual = self.lf0_model(
                        lf0_inp,
                        lengths,
                        y_lf0,
                        **lf0_extra_kwargs,
                    )
                else:
                    lf0, lf0_residual = (
                        self.lf0_model(
                            lf0_inp,
                            lengths,
                            y_lf0,
                            **lf0_extra_kwargs,
                        ),
                        None,
                    )

        # Predict mel
        if self.mel_model.requires_spk():
            mel_extra_kwargs = {"spk": spk, "spk_enc_inp": spk_enc_inp}
        else:
            mel_extra_kwargs = {}
        if is_inference:
            if self.lf0_model.prediction_type() == PredictionType.PROBABILISTIC:
                lf0_ = lf0[0]
            else:
                lf0_ = lf0
            mel_inp = [ling, lf0_]
            if self.mel_model_vuv_conditioning:
                mel_inp.append(vuv)
            mel_inp.append(erg)

            mel_inp = torch.cat(mel_inp, dim=-1)
            mel = self.mel_model.inference(mel_inp, lengths, **mel_extra_kwargs)
        else:
            if self.freeze_mel_model:
                # dummy
                mel = y_mel
            else:
                mel_inp = [ling, y_lf0]
                if self.mel_model_vuv_conditioning:
                    mel_inp.append(y_vuv)
                mel_inp.append(erg)
                mel_inp = torch.cat(mel_inp, dim=-1)
                mel = self.mel_model(mel_inp, lengths, y_mel, **mel_extra_kwargs)

        # Predict V/UV
        if is_inference:
            if self.mel_model.prediction_type() == PredictionType.PROBABILISTIC:
                mel_cond = mel[0]
            else:
                mel_cond = mel
            # full cond: (ling, lf0, (vuv), erg, mel)
            vuv_inp = [ling_lf0, lf0_cond]
            if self.vuv_model_vuv_conditioning:
                vuv_inp.append(vuv)
            vuv_inp.append(erg)
            vuv_inp.append(mel_cond)
            vuv_inp = torch.cat(vuv_inp, dim=-1)
            vuv = self.vuv_model.inference(vuv_inp, lengths)
        else:
            vuv_inp = [ling_lf0, y_lf0]
            if self.vuv_model_vuv_conditioning:
                vuv_inp.append(y_vuv)
            vuv_inp.append(erg)
            vuv_inp.append(y_mel)
            vuv_inp = torch.cat(vuv_inp, dim=-1)
            vuv = self.vuv_model(vuv_inp, lengths, y_vuv)

        if is_inference:
            if self.mel_model.prediction_type() == PredictionType.PROBABILISTIC:
                mel_ = mel[0]
            else:
                mel_ = mel
            out = torch.cat([mel_, lf0_, vuv], dim=-1)
            assert out.shape[-1] == self.out_dim
            # TODO: better design
            return out, out
        else:
            return (mel, lf0, vuv), lf0_residual, ling_s

    def inference(self, x, lengths=None, spk=None, spk_enc_inp=None):
        return pad_inference(
            model=self,
            x=x,
            lengths=lengths,
            spk=spk,
            spk_enc_inp=spk_enc_inp,
            reduction_factor=self.reduction_factor,
            mdn=True,
        )
