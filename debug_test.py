import argparse
import logging
import os
import random
import shutil
import time
from pathlib import Path
from warnings import warn

import math
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test  # import test.py to get mAP after each epoch
from models.yolo import Model
from utils.datasets import create_dataloader
from utils.general import (
    torch_distributed_zero_first,
    labels_to_class_weights,
    plot_labels,
    check_anchors,
    labels_to_image_weights,
    compute_loss,
    plot_images,
    fitness,
    strip_optimizer,
    plot_results,
    get_latest_run,
    check_dataset,
    check_file,
    check_git_status,
    check_img_size,
    increment_dir,
    print_mutation,
    plot_evolution,
    set_logging,
    init_seeds,
)
from utils.google_utils import attempt_download
from utils.torch_utils import ModelEMA, select_device, intersect_dicts

logger = logging.getLogger(__name__)


def train(hyp, opt, device, tb_writer=None):
    logger.info(f"Hyperparameters {hyp}")
    log_dir = (
        Path(tb_writer.log_dir) if tb_writer else Path(opt.logdir) / "evolve"
    )  # logging directory
    wdir = log_dir / "weights"  # weights directory
    os.makedirs(wdir, exist_ok=True)
    last = wdir / "last.pt"
    best = wdir / "best.pt"
    results_file = str(log_dir / "results.txt")
    epochs, batch_size, total_batch_size, weights, rank = (
        opt.epochs,
        opt.batch_size,
        opt.total_batch_size,
        opt.weights,
        opt.global_rank,
    )

    # Save run settings
    with open(log_dir / "hyp.yaml", "w") as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(log_dir / "opt.yaml", "w") as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Configure
    cuda = device.type != "cpu"
    init_seeds(2 + rank)
    with open(opt.data, encoding="utf8") as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # check
    train_path = data_dict["train"]
    test_path = data_dict["val"]
    nc, names = (
        (1, ["item"]) if opt.single_cls else (int(data_dict["nc"]), data_dict["names"])
    )  # number classes, names
    assert len(names) == nc, "%g names found for nc=%g dataset in %s" % (
        len(names),
        nc,
        opt.data,
    )  # check

    # Model
    model = Model(opt.cfg, ch=3, nc=nc).to(device)  # create

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(
        round(nbs / total_batch_size), 1
    )  # accumulate loss before optimizing
    hyp["weight_decay"] *= total_batch_size * accumulate / nbs  # scale weight_decay

    # Resume
    start_epoch, best_fitness = 0, 0.0

    # Image sizes
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz, imgsz_test = [
        check_img_size(x, gs) for x in opt.img_size
    ]  # verify imgsz are gs-multiples

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Exponential moving average
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # DDP mode
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank)

    # Trainloader
    dataloader, dataset = create_dataloader(
        train_path,
        imgsz,
        batch_size,
        gs,
        opt,
        hyp=hyp,
        augment=True,
        cache=opt.cache_images,
        rect=opt.rect,
        rank=rank,
        world_size=opt.world_size,
        workers=1,
        do_crop=opt.do_crop,
    )
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, (
        "Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g"
        % (mlc, nc, opt.data, nc - 1)
    )

    # Model parameters
    hyp["cls"] *= nc / 80.0  # scale coco-tuned hyp['cls'] to current dataset
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(
        device
    )  # attach class weights
    model.names = names

    logger.info(
        "Image sizes %g train, %g test\n"
        "Using %g dataloader workers\nLogging results to %s\n"
        "Starting training for %g epochs..."
        % (imgsz, imgsz_test, dataloader.num_workers, log_dir, epochs)
    )

    # epoch ------------------------------------------------------------------
    for epoch in range(start_epoch, epochs):
        model.train()
        pbar = enumerate(dataloader)
        logger.info(
            ("\n" + "%10s" * 8)
            % ("Epoch", "gpu_mem", "box", "obj", "cls", "total", "targets", "img_size")
        )
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar

        # batch -------------------------------------------------------------
        for (i, (imgs, targets, paths, _),) in pbar:
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = (
                imgs.to(device, non_blocking=True).float() / 255.0
            )  # uint8 to float32, 0-255 to 0.0-1.0

            print("imgs.shape: ", imgs.shape)
            print("targets.shape: ", targets.shape)
            print("targets: ", targets)
            print("\n")

            # # Forward
            # with amp.autocast(enabled=cuda):
            #     pred = model(imgs)  # forward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", type=str, default="yolov5s.pt", help="initial weights path"
    )
    parser.add_argument("--cfg", type=str, default="", help="model.yaml path")
    parser.add_argument(
        "--data", type=str, default="data/coco128.yaml", help="data.yaml path"
    )
    parser.add_argument(
        "--hyp", type=str, default="data/hyp.scratch.yaml", help="hyperparameters path"
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--batch-size", type=int, default=16, help="total batch size for all GPUs"
    )
    parser.add_argument(
        "--img-size",
        nargs="+",
        type=int,
        default=[640, 640],
        help="[train, test] image sizes",
    )
    parser.add_argument("--rect", action="store_true", help="rectangular training")
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        help="resume most recent training",
    )
    parser.add_argument(
        "--nosave", action="store_true", help="only save final checkpoint"
    )
    parser.add_argument("--notest", action="store_true", help="only test final epoch")
    parser.add_argument(
        "--noautoanchor", action="store_true", help="disable autoanchor check"
    )
    parser.add_argument("--evolve", action="store_true", help="evolve hyperparameters")
    parser.add_argument("--bucket", type=str, default="", help="gsutil bucket")
    parser.add_argument(
        "--cache-images", action="store_true", help="cache images for faster training"
    )
    parser.add_argument(
        "--do-crop",
        action="store_true",
        default=False,
        help="randomly crop image for training",
    )
    parser.add_argument(
        "--image-weights",
        action="store_true",
        help="use weighted image selection for training",
    )
    parser.add_argument(
        "--name",
        default="",
        help="renames experiment folder exp{N} to exp{N}_{name} if supplied",
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--multi-scale",
        action="store_true",
        default=False,
        help="vary img-size +/- 50%%",
    )
    parser.add_argument(
        "--single-cls", action="store_true", help="train as single-class dataset"
    )
    parser.add_argument(
        "--adam", action="store_true", help="use torch.optim.Adam() optimizer"
    )
    parser.add_argument(
        "--sync-bn",
        action="store_true",
        help="use SyncBatchNorm, only available in DDP mode",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="DDP parameter, do not modify"
    )
    parser.add_argument("--logdir", type=str, default="runs/", help="logging directory")
    parser.add_argument(
        "--workers", type=int, default=8, help="maximum number of dataloader workers"
    )
    opt = parser.parse_args()

    # Set DDP variables
    opt.total_batch_size = opt.batch_size
    opt.world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    opt.global_rank = int(os.environ["RANK"]) if "RANK" in os.environ else -1
    set_logging(opt.global_rank)
    if opt.global_rank in [-1, 0]:
        check_git_status()

    # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
    opt.data, opt.cfg, opt.hyp = (
        check_file(opt.data),
        check_file(opt.cfg),
        check_file(opt.hyp),
    )  # check files
    assert len(opt.cfg) or len(
        opt.weights
    ), "either --cfg or --weights must be specified"
    opt.img_size.extend(
        [opt.img_size[-1]] * (2 - len(opt.img_size))
    )  # extend to 2 sizes (train, test)
    log_dir = increment_dir(Path(opt.logdir) / "exp", opt.name)  # runs/exp1

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device("cuda", opt.local_rank)
        dist.init_process_group(
            backend="nccl", init_method="env://"
        )  # distributed backend
        assert (
            opt.batch_size % opt.world_size == 0
        ), "--batch-size must be multiple of CUDA device count"
        opt.batch_size = opt.total_batch_size // opt.world_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
        if "box" not in hyp:
            warn(
                'Compatibility: %s missing "box" which was renamed from "giou" in %s'
                % (opt.hyp, "https://github.com/ultralytics/yolov5/pull/1120")
            )
            hyp["box"] = hyp.pop("giou")

    # Train
    logger.info(opt)
    tb_writer = None
    train(hyp, opt, device, tb_writer)

