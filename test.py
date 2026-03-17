"""
NTIRE 2026 Image Super-Resolution (×4) Challenge
=================================================
Official test script format.

Usage:
    CUDA_VISIBLE_DEVICES=0 python test.py --test_dir [path] --save_dir [path] --model_id 29
"""

import os.path
import logging
import torch
import argparse
import json
import glob

from pprint import pprint
from utils import utils_logger
from utils import utils_image as util


def select_model(args, device):
    """
    Model selector — each team adds their model here.
    Model ID is assigned according to the team registration order.
    """
    model_id = args.model_id

    if model_id == 0:
        # DAT baseline, ICCV 2023
        from models.team00_DAT import main as DAT
        name = f"{model_id:02}_DAT_baseline"
        model_path = os.path.join('model_zoo', 'team00_dat.pth')
        model_func = DAT

    elif model_id == 29:
        # ── Team 29: Anant_SVNIT — FreqFusionSR ─────────────────────
        # Multi-Expert Frequency-Guided Fusion for Image SR
        # Experts: DRCT-L + GRL-B + NAFNet-SIDD + MambaIR
        # Fusion: 7-Phase DCT+DWT+FFT frequency-guided network
        from models.team29_FreqFusionSR import main as FreqFusionSR
        name = f"{model_id:02}_FreqFusionSR"
        model_path = os.path.join('model_zoo', 'team29_FreqFusionSR')
        model_func = FreqFusionSR

    else:
        raise NotImplementedError(f"Model {model_id} is not implemented.")

    return model_func, model_path, name


def run(model_func, model_name, model_path, device, args, mode="test"):
    """Run inference on a dataset split."""
    if mode == "valid":
        data_path = args.valid_dir
    elif mode == "test":
        data_path = args.test_dir
    assert data_path is not None, "Please specify the dataset path for validation or test."

    save_path = os.path.join(args.save_dir, model_name, mode)
    util.mkdir(save_path)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    model_func(model_dir=model_path, input_path=data_path, output_path=save_path, device=device)
    end.record()
    torch.cuda.synchronize()
    print(f"Model {model_name} runtime (Including I/O): {start.elapsed_time(end):.1f} ms")


def main(args):
    utils_logger.logger_info("NTIRE2026-ImageSRx4", log_path="NTIRE2026-ImageSRx4.log")
    logger = logging.getLogger("NTIRE2026-ImageSRx4")

    # Basic settings
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    json_dir = os.path.join(os.getcwd(), "results.json")
    if not os.path.exists(json_dir):
        results = dict()
    else:
        with open(json_dir, "r") as f:
            results = json.load(f)

    # Load model
    model_func, model_path, model_name = select_model(args, device)
    logger.info(model_name)

    if args.valid_dir is not None:
        run(model_func, model_name, model_path, device, args, mode="valid")

    if args.test_dir is not None:
        run(model_func, model_name, model_path, device, args, mode="test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("NTIRE2026-ImageSRx4")
    parser.add_argument("--valid_dir", default=None, type=str, help="Path to the validation set (LQ images)")
    parser.add_argument("--test_dir", default=None, type=str, help="Path to the test set (LQ images)")
    parser.add_argument("--save_dir", default="NTIRE2026-ImageSRx4/results", type=str)
    parser.add_argument("--model_id", default=29, type=int)

    args = parser.parse_args()
    pprint(args)

    main(args)
