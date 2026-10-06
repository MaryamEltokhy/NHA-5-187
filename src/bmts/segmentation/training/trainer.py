"""Train a segmentation model and evaluate it on the validation patients, everything logged in MLflow (M2-T03).

Built for free Colab: checkpoints go to Drive after every epoch, a rerun resumes from the last one (same MLflow
run), and --max-hours stops cleanly before the session ends. Locked test patients are never loaded.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import sqlite3
import subprocess
import time
import warnings
from pathlib import Path

import mlflow
import pandas as pd
import torch
import yaml
from monai.data import SmartCacheDataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceLoss
from monai.utils import set_determinism

from bmts.common.constants import MODALITIES, REGIONS
from bmts.data.datasets.brats import build_train_loader, build_val_loader, load_config
from bmts.segmentation.models.unet3d import build_unet3d
from bmts.segmentation.training.metrics import binarize, region_scores, summarize

REPO = Path(__file__).resolve().parents[4]


def load_experiment(path, **overrides):
    """Experiment config (configs/training/EXP-*.yaml) + its data and model configs, with command-line overrides."""
    exp = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    data = load_config(REPO / exp["data_config"])
    data.update(exp.get("data_overrides", {}))
    model = yaml.safe_load((REPO / exp["model_config"]).read_text(encoding="utf-8"))
    for key in ("processed_root", "split_csv", "num_workers", "smart_cache_num"):
        if overrides.get(key) is not None:
            data[key] = overrides[key]
    for key in ("epochs", "iters_per_epoch", "val_every", "val_cases_during_training", "train_cases", "val_cases"):
        if overrides.get(key) is not None:
            exp["training"][key] = overrides[key]
    if overrides.get("output_dir"):
        exp["output_dir"] = overrides["output_dir"]
    if overrides.get("tracking_uri"):
        exp["mlflow"]["tracking_uri"] = overrides["tracking_uri"]
    if overrides.get("model"):
        model.update(overrides["model"])
    return {"exp": exp, "data": data, "model": model}


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v if isinstance(v, (int, float, str, bool)) else json.dumps(v)
    return out


def _sha256(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return "unavailable"


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _lr_lambda(epochs, warmup, final_fraction=0.01):
    """Linear warm-up, then cosine decay to final_fraction of the starting learning rate (stepped once per epoch)."""
    def f(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return final_fraction + (1 - final_fraction) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return f


def _predict(model, image, cfg, device, amp):
    t = cfg["exp"]["training"]
    with torch.no_grad(), torch.autocast(device.type, enabled=amp):
        logits = sliding_window_inference(image.to(device), cfg["data"]["roi_size"], t.get("sw_batch_size", 2), model,
                                          overlap=t.get("sw_overlap", 0.5), mode="gaussian")
    return binarize(logits)


def validate(model, loader, cfg, device, amp, with_hd95=False, per_case_rows=None):
    """Score every case of the loader; returns summarize() of region_scores."""
    model.eval()
    results = []
    for batch in loader:
        pred = _predict(model, batch["image"], cfg, device, amp)[0].cpu()
        scores = region_scores(pred, batch["label"][0], with_hd95=with_hd95)
        results.append(scores)
        if per_case_rows is not None:
            case_id = batch["id"][0] if "id" in batch else str(len(results))
            per_case_rows.append({"case": case_id, **{f"{r}_{m}": v for r, s in scores.items() for m, v in s.items()}})
    return summarize(results)


def _sqlite_path(uri):
    return Path(uri[len("sqlite:///"):]) if uri.startswith("sqlite:///") else None


def _tracking_uri(cfg):
    """SQLite cannot lock files reliably on Google Drive (random "disk I/O error"). With mlflow.local_copy set, MLflow
    works on a copy on the local disk, refreshed from the Drive file when that is newer, and mirror_tracking_db()
    copies it back after every epoch."""
    m = cfg["exp"]["mlflow"]
    remote, local = _sqlite_path(m["tracking_uri"]), m.get("local_copy")
    if not remote or not local:
        return m["tracking_uri"]
    local = Path(local)
    if remote.exists() and (not local.exists() or remote.stat().st_mtime > local.stat().st_mtime):
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(remote, local)
    return f"sqlite:///{local.as_posix()}"


def mirror_tracking_db(cfg):
    """Copy the local MLflow database back to its home (e.g. Drive) as one consistent snapshot."""
    m = cfg["exp"]["mlflow"]
    remote, local = _sqlite_path(m["tracking_uri"]), m.get("local_copy")
    if not remote or not local or not Path(local).exists():
        return
    snapshot = Path(local).with_name(Path(local).name + ".snapshot")
    snapshot.unlink(missing_ok=True)
    src, dst = sqlite3.connect(local), sqlite3.connect(snapshot)
    try:
        src.backup(dst)
    finally:
        src.close()
        dst.close()
    try:
        remote.parent.mkdir(parents=True, exist_ok=True)
        tmp = remote.with_name(remote.name + ".tmp")
        shutil.copyfile(snapshot, tmp)
        try:
            os.replace(tmp, remote)
        except PermissionError:  # Windows: the target is open elsewhere; overwrite it in place instead
            shutil.copyfile(snapshot, remote)
            tmp.unlink(missing_ok=True)
    except OSError as e:  # never stop training over the mirror; the next epoch tries again
        warnings.warn(f"could not copy the MLflow database to {remote}: {e}", stacklevel=2)


def _open_mlflow_run(cfg, state, out):
    m = cfg["exp"]["mlflow"]
    mlflow.set_tracking_uri(_tracking_uri(cfg))
    experiment = mlflow.get_experiment_by_name(m["experiment"])
    if experiment is None:  # keep artifacts on Drive next to the runs, not on the temporary Colab disk
        mlflow.create_experiment(m["experiment"], artifact_location=(out.parent / "mlflow_artifacts").resolve().as_uri())
    mlflow.set_experiment(m["experiment"])
    if state.get("run_id"):
        return mlflow.start_run(run_id=state["run_id"]), False
    return mlflow.start_run(run_name=cfg["exp"]["experiment_id"]), True


def _log_setup(cfg, device, n_train, n_val, model):
    exp, data = cfg["exp"], cfg["data"]
    params = {"experiment_id": exp["experiment_id"], "train_cases": n_train, "val_cases_during_training": n_val,
              "n_parameters": sum(p.numel() for p in model.parameters())}
    params.update(_flatten(exp["training"], "train."))
    params.update(_flatten(cfg["model"], "model."))
    params.update(_flatten({k: v for k, v in data.items() if k not in ("split_csv", "processed_root")}, "data."))
    mlflow.log_params(params)
    split_versions = "unknown"
    try:
        split_versions = ",".join(sorted(pd.read_csv(data["split_csv"], dtype=str).get("split_version", pd.Series(["unknown"])).unique()))
    except Exception:  # noqa: BLE001 - the tag is informative only
        pass
    mlflow.set_tags({
        "task": exp.get("task", ""), "description": exp.get("description", ""), "device": str(device),
        "git_commit": _git("rev-parse", "HEAD") or "unknown", "git_dirty": str(bool(_git("status", "--porcelain"))),
        "split_csv": str(data["split_csv"]), "split_csv_sha256": _sha256(data["split_csv"]), "split_version": split_versions,
        "processed_root": str(data["processed_root"]), "modalities": ",".join(MODALITIES), "regions": ",".join(REGIONS),
        "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest(),
        "output_dir": exp["output_dir"],
    })


def forward_loss(model, loss_fn, image, label, device, amp):
    """Loss in mixed precision; if half precision overflows (inf/NaN), the same step again in full precision.

    Returns (loss, retried_in_fp32). Half precision tops out at about 65,000, which large activations can exceed
    as training goes on; redoing only those rare steps in float32 keeps the speed of mixed precision.
    """
    with torch.autocast(device.type, enabled=amp):
        loss = loss_fn(model(image), label)
    if amp and not torch.isfinite(loss):
        with torch.autocast(device.type, enabled=False):
            return loss_fn(model(image.float()), label.float()), True
    return loss, False


def _save(path, payload):
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def train(cfg, max_hours=None, resume=True, log=print):
    """Train until cfg epochs are done (or max_hours is reached), then evaluate the best checkpoint on all validation patients."""
    exp, t, data = cfg["exp"], cfg["exp"]["training"], cfg["data"]
    out = Path(exp["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    set_determinism(t.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(t.get("amp", True)) and device.type == "cuda"

    model = build_unet3d(cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t.get("weight_decay", 1e-5))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda(t["epochs"], t.get("warmup_epochs", 0)))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loss_fn = (DiceCELoss(sigmoid=True, smooth_nr=1e-5, smooth_dr=1e-5) if t.get("loss", "dice_bce") == "dice_bce"
               else DiceLoss(sigmoid=True, smooth_nr=1e-5, smooth_dr=1e-5))

    state = {"epoch": 0, "best_mean_dice": -1.0, "best_epoch": 0, "run_id": None, "history": []}
    last, best = out / "last.pt", out / "best.pt"
    if resume and last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        state = ck["state"]
        log(f"resuming after epoch {state['epoch']} (best mean Dice {state['best_mean_dice']:.4f} at epoch {state['best_epoch']})")

    train_loader = build_train_loader(data, limit=t.get("train_cases"))
    val_loader = build_val_loader(data, limit=t.get("val_cases_during_training"))
    n_train = len(train_loader.dataset.data)  # all training patients (a cache may hold only some of them at a time)
    run, new_run = _open_mlflow_run(cfg, state, out)
    state["run_id"] = run.info.run_id
    if new_run:
        _log_setup(cfg, device, n_train, len(val_loader.dataset), model)
    else:
        logged = mlflow.get_run(run.info.run_id).data.params.get("train.epochs")
        if logged and int(logged) != t["epochs"]:  # parameters cannot change; record the new total as a tag
            mlflow.set_tag("epochs_actual", f"{t['epochs']} (the run started with {logged})")
    mlflow.set_tag("state", "training")
    log(f"device {device}{' (' + torch.cuda.get_device_name(0) + ')' if amp else ''} | MLflow run {run.info.run_id} | "
        f"{n_train} training cases, {len(val_loader.dataset)} validation cases during training")

    dataset = train_loader.dataset
    smart = isinstance(dataset, SmartCacheDataset)
    if smart:
        dataset.start()
    started, finished = time.time(), state["epoch"] >= t["epochs"]
    try:
        for epoch in range(state["epoch"] + 1, t["epochs"] + 1):
            model.train()
            tick, losses, skipped, fp32_retries = time.time(), [], [], 0
            batches = iter(train_loader)
            for _ in range(t["iters_per_epoch"]):
                try:
                    batch = next(batches)
                except StopIteration:
                    batches = iter(train_loader)
                    batch = next(batches)
                image, label = batch["image"].to(device), batch["label"].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss, retried = forward_loss(model, loss_fn, image, label, device, amp)
                fp32_retries += retried
                if not torch.isfinite(loss):  # even full precision fails: skip the step; stop only if it keeps happening
                    skipped.extend(sorted(set(batch.get("id", ["?"]))))
                    if len(skipped) > max(3, 0.1 * t["iters_per_epoch"]):
                        raise FloatingPointError(f"{len(skipped)} non-finite losses in epoch {epoch} (cases {skipped[:5]}): "
                                                 "the model has probably diverged; lower lr or resume from best.pt")
                    log(f"  skipped a step with a non-finite loss ({loss.item()}) in epoch {epoch}, cases {sorted(set(batch.get('id', ['?'])))}")
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), t.get("grad_clip", 12.0))
                scaler.step(optimizer)
                scaler.update()
                losses.append(loss.item())
            metrics = {"train_loss": sum(losses) / max(1, len(losses)), "lr": optimizer.param_groups[0]["lr"],
                       "epoch_seconds": time.time() - tick, "skipped_steps": len(skipped), "fp32_retry_steps": fp32_retries}
            if skipped:
                state.setdefault("skipped_cases", []).extend(skipped)
                mlflow.set_tag("skipped_step_cases", ", ".join(sorted(set(state["skipped_cases"])))[:4900])
            scheduler.step()
            if smart:
                dataset.update_cache()

            if epoch % t.get("val_every", 1) == 0 or epoch == t["epochs"]:
                tick = time.time()
                summary = validate(model, val_loader, cfg, device, amp)
                metrics.update({f"val_dice_{r}": summary[r]["dice"] for r in REGIONS})
                metrics.update(val_mean_dice=summary["mean_dice"], val_seconds=time.time() - tick)
                if summary["mean_dice"] > state["best_mean_dice"]:
                    state["best_mean_dice"], state["best_epoch"] = summary["mean_dice"], epoch
                    _save(best, {"model": model.state_dict(), "epoch": epoch, "val": summary, "config": cfg})
                    metrics["best_val_mean_dice"] = summary["mean_dice"]
            if amp:
                metrics["peak_gpu_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
            mlflow.log_metrics(metrics, step=epoch)
            state["epoch"] = epoch
            state["history"].append({"epoch": epoch, **metrics})
            _save(last, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "state": state})
            mirror_tracking_db(cfg)
            val_text = f", val Dice {[round(metrics[f'val_dice_{r}'], 3) for r in REGIONS]}" if "val_mean_dice" in metrics else ""
            log(f"epoch {epoch}/{t['epochs']}: loss {metrics['train_loss']:.4f}, lr {metrics['lr']:.2e}, "
                f"{metrics['epoch_seconds']:.0f}s{val_text}")
            if max_hours and time.time() - started > max_hours * 3600 and epoch < t["epochs"]:
                log(f"time budget of {max_hours} h reached after epoch {epoch}: run the same command again to resume")
                break
        finished = state["epoch"] >= t["epochs"]
    finally:
        if smart:
            dataset.shutdown()

    if not finished:
        mlflow.set_tag("state", f"paused after epoch {state['epoch']} (rerun to resume)")
        mlflow.end_run(status="KILLED")
        mirror_tracking_db(cfg)
        return {"finished": False, "epoch": state["epoch"], "best_mean_dice": state["best_mean_dice"]}
    result = final_evaluation(cfg, device, amp, log)
    mlflow.set_tag("state", "finished")
    mlflow.end_run()
    mirror_tracking_db(cfg)
    return {"finished": True, **result}


def final_evaluation(cfg, device=None, amp=None, log=print):
    """Best checkpoint on all validation patients: Dice, IoU, sensitivity, precision and HD95 per region, logged as final_*."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (device.type == "cuda") if amp is None else amp
    out = Path(cfg["exp"]["output_dir"])
    ck = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model = build_unet3d(cfg["model"]).to(device)
    model.load_state_dict(ck["model"])
    loader = build_val_loader(cfg["data"], limit=cfg["exp"]["training"].get("val_cases"))
    log(f"final evaluation: best checkpoint (epoch {ck['epoch']}) on {len(loader.dataset)} validation cases")
    rows, tick = [], time.time()
    summary = validate(model, loader, cfg, device, amp, with_hd95=True, per_case_rows=rows)
    result = {"best_epoch": ck["epoch"], "val_cases": len(loader.dataset), "summary": summary,
              "seconds": round(time.time() - tick, 1)}
    (out / "final_val_metrics.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    with open(out / "final_val_per_case.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = {"final_val_mean_dice": summary["mean_dice"], "final_best_epoch": ck["epoch"]}
    for r in REGIONS:
        for m in ("dice", "iou", "sensitivity", "precision", "hd95"):
            if m in summary[r] and not math.isnan(summary[r][m]):
                metrics[f"final_val_{m}_{r}"] = summary[r][m]
    if mlflow.active_run() is None:
        mlflow.set_tracking_uri(_tracking_uri(cfg))
        mlflow.start_run(run_id=ck.get("run_id") or _run_id_from_last(out))
    mlflow.log_metrics(metrics)
    for name in ("final_val_metrics.json", "final_val_per_case.csv", "config_resolved.yaml"):
        try:
            mlflow.log_artifact(str(out / name), artifact_path="evaluation")
        except Exception as e:  # noqa: BLE001 - artifacts are a convenience copy; the files stay in output_dir
            log(f"could not copy {name} to the MLflow artifact store ({type(e).__name__}); it is in {out}")
    mlflow.set_tag("best_checkpoint", str(out / "best.pt"))
    log("final validation: " + ", ".join(f"{r} Dice {summary[r]['dice']:.3f} HD95 {summary[r].get('hd95', float('nan')):.1f} mm"
                                         for r in REGIONS))
    return result


def _run_id_from_last(out):
    last = Path(out) / "last.pt"
    return torch.load(last, map_location="cpu", weights_only=False)["state"]["run_id"] if last.exists() else None
