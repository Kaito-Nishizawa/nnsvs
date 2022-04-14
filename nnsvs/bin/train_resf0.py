from pathlib import Path

import hydra
import mlflow
import numpy as np
import torch
from hydra.utils import to_absolute_path
from nnmnkwii import metrics
from nnsvs.base import PredictionType
from nnsvs.mdn import mdn_get_most_probable_sigma_and_mu, mdn_loss
from nnsvs.multistream import get_static_features
from nnsvs.pitch import nonzero_segments
from nnsvs.train_util import (
    log_params_from_omegaconf_dict,
    save_checkpoint,
    save_configs,
    setup,
)
from nnsvs.util import PyTorchStandardScaler, make_non_pad_mask
from omegaconf import DictConfig
from torch import nn
from tqdm import tqdm


def note_segments(lf0_score_denorm):
    """Compute note segments (start and end indices) from log-F0

    Note that unvoiced frames must be set to 0 in advance.

    Args:
        lf0_score_denorm (Tensor): (B, T)

    Returns:
        list: list of note (start, end) indices
    """
    segments = []
    for s, e in nonzero_segments(lf0_score_denorm):
        out = torch.sign(torch.abs(torch.diff(lf0_score_denorm[s : e + 1])))
        transitions = torch.where(out > 0)[0]
        note_start, note_end = s, -1
        for pos in transitions:
            note_end = int(s + pos)
            segments.append((note_start, note_end))
            note_start = note_end

    return segments


def compute_pitch_regularization_weight(segments, N, decay_size=25, max_w=0.5):
    """Compute pitch regularization weight given note segments

    Args:
        segments (list): list of note (start, end) indices
        N (int): number of frames
        decay_size (int): size of the decay window
        max_w (float): maximum weight

    Returns:
        Tensor: weights of shape (N,)
    """
    w = torch.zeros(N)

    for s, e in segments:
        L = e - s
        w[s:e] = max_w
        if L > decay_size * 2:
            w[s : s + decay_size] *= torch.arange(decay_size) / decay_size
            w[e - decay_size : e] *= torch.arange(decay_size - 1, -1, -1) / decay_size

    return w


def compute_batch_pitch_regularization_weight(lf0_score_denorm):
    """Batch version of computing pitch regularization weight

    Args:
        lf0_score_denorm (Tensor): (B, T)

    Returns:
        Tensor: weights of shape (B, N, 1)
    """
    B, T = lf0_score_denorm.shape
    w = torch.zeros_like(lf0_score_denorm)
    for idx in range(len(lf0_score_denorm)):
        segments = note_segments(lf0_score_denorm[idx])
        w[idx, :] = compute_pitch_regularization_weight(segments, T).to(w.device)

    return w.unsqueeze(-1)


@torch.no_grad()
def compute_distortions(pred_out_feats, out_feats, lengths, out_scaler, model_config):
    out_feats = out_scaler.inverse_transform(out_feats)
    pred_out_feats = out_scaler.inverse_transform(pred_out_feats)
    out_streams = get_static_features(
        out_feats,
        model_config.num_windows,
        model_config.stream_sizes,
        model_config.has_dynamic_features,
    )
    pred_out_streams = get_static_features(
        pred_out_feats,
        model_config.num_windows,
        model_config.stream_sizes,
        model_config.has_dynamic_features,
    )

    assert len(out_streams) >= 4
    mgc, lf0, vuv, bap = out_streams[0], out_streams[1], out_streams[2], out_streams[3]
    pred_mgc, pred_lf0, pred_vuv, pred_bap = (
        pred_out_streams[0],
        pred_out_streams[1],
        pred_out_streams[2],
        pred_out_streams[3],
    )

    # binarize vuv
    vuv, pred_vuv = (vuv > 0.5).float(), (pred_vuv > 0.5).float()

    dist = {
        "mcd": metrics.melcd(mgc[:, :, 1:], pred_mgc[:, :, 1:], lengths=lengths),
        "bap_mcd": metrics.melcd(bap, pred_bap, lengths=lengths) / 10.0,
        "vuv_err": metrics.vuv_error(vuv, pred_vuv, lengths=lengths),
    }

    try:
        f0_mse = metrics.lf0_mean_squared_error(
            lf0, vuv, pred_lf0, pred_vuv, lengths=lengths, linear_domain=True
        )
        dist["f0_rmse"] = np.sqrt(f0_mse)
    except ZeroDivisionError:
        pass

    return dist


def train_step(
    model,
    model_config,
    optimizer,
    train,
    in_feats,
    out_feats,
    lengths,
    out_scaler,
    pitch_reg_dyn_ws,
    pitch_reg_weight=1.0,
):
    optimizer.zero_grad()

    criterion = nn.MSELoss(reduction="none")
    prediction_type = (
        model.module.prediction_type()
        if isinstance(model, nn.DataParallel)
        else model.prediction_type()
    )

    # Apply preprocess if required (e.g., FIR filter for shallow AR)
    # defaults to no-op
    if isinstance(model, nn.DataParallel):
        out_feats = model.module.preprocess_target(out_feats)
    else:
        out_feats = model.preprocess_target(out_feats)

    # Run forward
    pred_out_feats, lf0_residual = model(in_feats, lengths)

    # Mask (B, T, 1)
    mask = make_non_pad_mask(lengths).unsqueeze(-1).to(in_feats.device)

    # Compute loss
    if prediction_type == PredictionType.PROBABILISTIC:
        pi, sigma, mu = pred_out_feats

        # (B, max(T)) or (B, max(T), D_out)
        mask_ = mask if len(pi.shape) == 4 else mask.squeeze(-1)
        # Compute loss and apply mask
        loss = mdn_loss(pi, sigma, mu, out_feats, reduce=False)
        loss = loss.masked_select(mask_).mean()
    else:
        loss = criterion(
            pred_out_feats.masked_select(mask), out_feats.masked_select(mask)
        ).mean()

    # Pitch regularization
    # NOTE: l1 loss seems to be better than mse loss in my experiments
    # we could use l2 loss as suggested in the sinsy's paper
    loss += (
        pitch_reg_weight
        * (pitch_reg_dyn_ws * lf0_residual.abs()).masked_select(mask).mean()
    )

    if prediction_type == PredictionType.PROBABILISTIC:
        with torch.no_grad():
            pred_out_feats_ = mdn_get_most_probable_sigma_and_mu(pi, sigma, mu)[1]
    else:
        pred_out_feats_ = pred_out_feats
    distortions = compute_distortions(
        pred_out_feats_, out_feats, lengths, out_scaler, model_config
    )

    if train:
        loss.backward()
        optimizer.step()

    return loss, distortions


def train_loop(
    config,
    logger,
    device,
    model,
    optimizer,
    lr_scheduler,
    data_loaders,
    writer,
    in_scaler,
    out_scaler,
    use_mlflow,
):
    out_dir = Path(to_absolute_path(config.train.out_dir))
    best_dev_loss = torch.finfo(torch.float32).max
    last_dev_loss = torch.finfo(torch.float32).max

    in_lf0_idx = config.data.in_lf0_idx
    in_rest_idx = config.data.in_rest_idx
    if in_lf0_idx is None or in_rest_idx is None:
        raise ValueError("in_lf0_idx and in_rest_idx must be specified")
    pitch_reg_weight = config.train.pitch_reg_weight

    for epoch in tqdm(range(1, config.train.nepochs + 1)):
        for phase in data_loaders.keys():
            train = phase.startswith("train")
            model.train() if train else model.eval()
            running_loss = 0
            running_metrics = {}
            for in_feats, out_feats, lengths in data_loaders[phase]:
                # NOTE: This is needed for pytorch's PackedSequence
                lengths, indices = torch.sort(lengths, dim=0, descending=True)
                in_feats, out_feats = (
                    in_feats[indices].to(device),
                    out_feats[indices].to(device),
                )
                # Compute denormalized log-F0 in the musical scores
                # test - s.min_[in_lf0_idx]) / s.scale_[in_lf0_idx]
                lf0_score_denorm = (
                    in_feats[:, :, in_lf0_idx] - in_scaler.min_[in_lf0_idx]
                ) / in_scaler.scale_[in_lf0_idx]
                # Fill zeros for rest and padded frames
                lf0_score_denorm *= (in_feats[:, :, in_rest_idx] <= 0).float()
                for idx, length in enumerate(lengths):
                    lf0_score_denorm[idx, length:] = 0
                # Compute time-variant pitch regularization weight vector
                pitch_reg_dyn_ws = compute_batch_pitch_regularization_weight(
                    lf0_score_denorm
                )

                loss, distortions = train_step(
                    model,
                    config.model,
                    optimizer,
                    train,
                    in_feats,
                    out_feats,
                    lengths,
                    out_scaler,
                    pitch_reg_dyn_ws,
                    pitch_reg_weight,
                )
                running_loss += loss.item()
                for k, v in distortions.items():
                    try:
                        running_metrics[k] += float(v)
                    except KeyError:
                        running_metrics[k] = float(v)

            ave_loss = running_loss / len(data_loaders[phase])
            logger.info("[%s] [Epoch %s]: loss %s", phase, epoch, ave_loss)
            if writer is not None:
                writer.add_scalar(f"Loss/{phase}", ave_loss, epoch)
            if use_mlflow:
                mlflow.log_metric(f"{phase}_loss", ave_loss, step=epoch)

            for k, v in running_metrics.items():
                ave_v = v / len(data_loaders[phase])
                if writer is not None:
                    writer.add_scalar(f"{k}/{phase}", ave_v, epoch)
                if use_mlflow:
                    mlflow.log_metric(f"{phase}_{k}", ave_v, step=epoch)

            if not train:
                last_dev_loss = ave_loss
            if not train and ave_loss < best_dev_loss:
                best_dev_loss = ave_loss
                save_checkpoint(
                    logger, out_dir, model, optimizer, lr_scheduler, epoch, is_best=True
                )

        lr_scheduler.step()
        if epoch % config.train.checkpoint_epoch_interval == 0:
            save_checkpoint(
                logger, out_dir, model, optimizer, lr_scheduler, epoch, is_best=False
            )

    save_checkpoint(
        logger, out_dir, model, optimizer, lr_scheduler, config.train.nepochs
    )
    logger.info("The best loss was %s", best_dev_loss)
    if use_mlflow:
        mlflow.log_metric("best_dev_loss", best_dev_loss, step=epoch)
        mlflow.log_artifacts(out_dir)

    return last_dev_loss


def check_resf0_config(logger, model, config, in_scaler, out_scaler):
    logger.info("Checking model configs for residual F0 prediction")
    if in_scaler is None or out_scaler is None:
        raise ValueError("in_scaler and out_scaler must be specified")

    if isinstance(model, nn.DataParallel):
        model = model.module

    in_lf0_idx = config.data.in_lf0_idx
    in_rest_idx = config.data.in_rest_idx
    out_lf0_idx = config.data.out_lf0_idx
    if in_lf0_idx is None or in_rest_idx is None or out_lf0_idx is None:
        raise ValueError("in_lf0_idx, in_rest_idx and out_lf0_idx must be specified")

    logger.info("in_lf0_idx: %s", in_lf0_idx)
    logger.info("in_rest_idx: %s", in_rest_idx)
    logger.info("out_lf0_idx: %s", out_lf0_idx)

    ok = True
    if hasattr(model, "in_lf0_idx"):
        if model.in_lf0_idx != in_lf0_idx:
            logger.warn(
                "in_lf0_idx in model and data config must be same",
                model.in_lf0_idx,
                in_lf0_idx,
            )
            ok = False
    if hasattr(model, "out_lf0_idx"):
        if model.out_lf0_idx != out_lf0_idx:
            logger.warn(
                "out_lf0_idx in model and data config must be same",
                model.out_lf0_idx,
                out_lf0_idx,
            )
            ok = False

    if hasattr(model, "in_lf0_min") and hasattr(model, "in_lf0_max"):
        # Inject values from the input scaler
        if model.in_lf0_min is None or model.in_lf0_max is None:
            model.in_lf0_min = in_scaler.data_min_[in_lf0_idx]
            model.in_lf0_max = in_scaler.data_max_[in_lf0_idx]

        logger.info("in_lf0_min: %s", model.in_lf0_min)
        logger.info("in_lf0_max: %s", model.in_lf0_max)
        if not np.allclose(model.in_lf0_min, in_scaler.data_min_[model.in_lf0_idx]):
            logger.warn(
                f"in_lf0_min is set to {model.in_lf0_min}, "
                f"but should be {in_scaler.data_min_[model.in_lf0_idx]}"
            )
            ok = False
        if not np.allclose(model.in_lf0_max, in_scaler.data_max_[model.in_lf0_idx]):
            logger.warn(
                f"in_lf0_max is set to {model.in_lf0_max}, "
                f"but should be {in_scaler.data_max_[model.in_lf0_idx]}"
            )
            ok = False

    if hasattr(model, "out_lf0_mean") and hasattr(model, "out_lf0_scale"):
        # Inject values from the output scaler
        if model.out_lf0_mean is None or model.out_lf0_scale is None:
            model.out_lf0_mean = out_scaler.mean_[out_lf0_idx]
            model.out_lf0_scale = out_scaler.scale_[out_lf0_idx]

        logger.info("model.out_lf0_mean: %s", model.out_lf0_mean)
        logger.info("model.out_lf0_scale: %s", model.out_lf0_scale)
        if not np.allclose(model.out_lf0_mean, out_scaler.mean_[model.out_lf0_idx]):
            logger.warn(
                f"out_lf0_mean is set to {model.out_lf0_mean}, "
                f"but should be {out_scaler.mean_[model.out_lf0_idx]}"
            )
            ok = False
        if not np.allclose(model.out_lf0_scale, out_scaler.scale_[model.out_lf0_idx]):
            logger.warn(
                f"out_lf0_scale is set to {model.out_lf0_scale}, "
                f"but should be {out_scaler.scale_[model.out_lf0_idx]}"
            )
            ok = False

    if not ok:
        if (
            model.in_lf0_idx == in_lf0_idx
            and hasattr(model, "in_lf0_min")
            and hasattr(model, "out_lf0_mean")
        ):
            logger.info(
                f"""
If you are 100% sure that you set model.in_lf0_idx and model.out_lf0_idx correctly,
Please consider the following parameters in your model config:

    in_lf0_idx: {model.in_lf0_idx}
    out_lf0_idx: {model.out_lf0_idx}
    in_lf0_min: {in_scaler.data_min_[model.in_lf0_idx]}
    in_lf0_max: {in_scaler.data_max_[model.in_lf0_idx]}
    out_lf0_mean: {out_scaler.mean_[model.out_lf0_idx]}
    out_lf0_scale: {out_scaler.scale_[model.out_lf0_idx]}
"""
            )
        raise ValueError("The model config has wrong configurations.")

    # Overwrite the parameters to the config
    for key in ["in_lf0_min", "in_lf0_max", "out_lf0_mean", "out_lf0_scale"]:
        config.model.netG[key] = float(getattr(model, key))


@hydra.main(config_path="conf/train_resf0", config_name="config")
def my_app(config: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        model,
        optimizer,
        lr_scheduler,
        data_loaders,
        writer,
        logger,
        in_scaler,
        out_scaler,
    ) = setup(config, device)

    check_resf0_config(logger, model, config, in_scaler, out_scaler)

    out_scaler = PyTorchStandardScaler(
        torch.from_numpy(out_scaler.mean_), torch.from_numpy(out_scaler.scale_)
    ).to(device)
    use_mlflow = config.mlflow.enabled

    if use_mlflow:
        with mlflow.start_run() as run:
            # NOTE: modify out_dir when running with mlflow
            config.train.out_dir = f"{config.train.out_dir}/{run.info.run_id}"
            save_configs(config)
            log_params_from_omegaconf_dict(config)
            last_dev_loss = train_loop(
                config,
                logger,
                device,
                model,
                optimizer,
                lr_scheduler,
                data_loaders,
                writer,
                in_scaler,
                out_scaler,
                use_mlflow,
            )
    else:
        save_configs(config)
        last_dev_loss = train_loop(
            config,
            logger,
            device,
            model,
            optimizer,
            lr_scheduler,
            data_loaders,
            writer,
            in_scaler,
            out_scaler,
            use_mlflow,
        )

    return last_dev_loss


def entry():
    my_app()


if __name__ == "__main__":
    my_app()
