"""
NTIRE 2026 Image Super-Resolution (×4) — IQA Evaluation Script
===============================================================
Based on the official NTIRE2026_ImageSR_x4 baseline by Zheng Chen.

Metrics computed:
  - PSNR, SSIM (Full-Reference)
  - LPIPS, DISTS (Full-Reference, perceptual)
  - NIQE, MUSIQ, MANIQA, CLIP-IQA (No-Reference)

Usage:
    python eval.py \
        --output_folder "/path/to/your/output_dir" \
        --target_folder "/path/to/test_dir/HR" \
        --metrics_save_path "./IQA_results" \
        --gpu_ids 0
"""

import os
import torch
from PIL import Image
from tqdm import tqdm
import torch.multiprocessing as mp
from torchvision import transforms
import torchvision.transforms.functional as F
import csv
import numpy as np
from einops import rearrange

from utils import utils_image as util


def read_csv_to_dict(filename):
    data = {}
    with open(filename, mode='r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            key = row[csv_reader.fieldnames[0]]
            data[key] = {
                field: (float(value) if _is_number(value) else value)
                for field, value in row.items() if field != csv_reader.fieldnames[0]
            }
    return data


def _is_number(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def rgb_to_ycrcb(tensor):
    tensor_np = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    ycrcb_np = cv2.cvtColor(tensor_np, cv2.COLOR_RGB2YCrCb)
    ycrcb_tensor = torch.tensor(ycrcb_np).permute(2, 0, 1).unsqueeze(0).float()
    return ycrcb_tensor


class IQA:
    """Image Quality Assessment using pyiqa metrics."""

    def __init__(self, device=None):
        import pyiqa
        self.device = device if device else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.iqa_metrics = {
            'lpips': pyiqa.create_metric('lpips', device=self.device),
            'dists': pyiqa.create_metric('dists', device=self.device),
            'niqe': pyiqa.create_metric('niqe', device=self.device),
            'musiq': pyiqa.create_metric('musiq', device=self.device),
            'maniqa': pyiqa.create_metric('maniqa', device=self.device),
            'clipiqa': pyiqa.create_metric('clipiqa', device=self.device),
        }

    def calculate_values(self, output_image, target_image):
        if target_image is not None:
            assert type(output_image) == type(target_image)

        if isinstance(output_image, (torch.Tensor, np.ndarray)):
            if isinstance(output_image, np.ndarray):
                output_image = torch.tensor(output_image).contiguous().float()
                if target_image is not None:
                    target_image = torch.tensor(target_image).contiguous().float()

            if len(output_image.shape) == 3:
                output_image = output_image.unsqueeze(0)
                if target_image is not None:
                    target_image = target_image.unsqueeze(0)

            if output_image.shape[-1] == 3:
                output_image = rearrange(output_image, "b h w c -> b c h w").contiguous().float()
                if target_image is not None:
                    target_image = rearrange(target_image, "b h w c -> b c h w").contiguous().float()

            output_tensor = output_image.to(self.device)
            target_tensor = target_image.to(self.device) if target_image is not None else None
        else:
            output_tensor = F.to_tensor(output_image).unsqueeze(0).to(self.device)
            target_tensor = F.to_tensor(target_image).unsqueeze(0).to(self.device) if target_image is not None else None

        if target_tensor is not None and output_tensor.shape != target_tensor.shape:
            min_height = min(output_tensor.shape[2], target_tensor.shape[2])
            min_width = min(output_tensor.shape[3], target_tensor.shape[3])
            resize_transform = transforms.Resize((min_height, min_width))
            output_tensor = resize_transform(output_tensor)
            target_tensor = resize_transform(target_tensor)

        try:
            result = {}
            if target_tensor is not None:
                result['LPIPS'] = self.iqa_metrics['lpips'](output_tensor, target_tensor).item()
                result['DISTS'] = self.iqa_metrics['dists'](output_tensor, target_tensor).item()

            result['NIQE'] = self.iqa_metrics['niqe'](output_tensor).item()
            result['MUSIQ'] = self.iqa_metrics['musiq'](output_tensor).item()
            result['MANIQA'] = self.iqa_metrics['maniqa'](output_tensor).item()
            result['CLIP-IQA'] = self.iqa_metrics['clipiqa'](output_tensor).item()
        except Exception as e:
            print(f"Error: {e}")
            return None

        return result


def calculate_iqa_for_partition(output_folder, target_folder, output_files, device, rank):
    iqa = IQA(device=device)
    local_results = {}
    for output_file in tqdm(output_files, total=len(output_files), desc=f"Processing images on GPU {rank}"):
        output_image_path = os.path.join(output_folder, output_file)
        output_image = Image.open(output_image_path)

        if target_folder is not None:
            target_file = output_file.replace('x4', '')
            target_image_path = os.path.join(target_folder, target_file)
            assert os.path.exists(target_image_path), f"No such path: {target_image_path}"
            target_image = Image.open(target_image_path)
        else:
            target_image = None

        values = iqa.calculate_values(output_image, target_image)
        if target_folder is not None:
            target_file = output_file.replace('x4', '')
            target_image_path = os.path.join(target_folder, target_file)
            values["psnr"], values["ssim"] = util.cal_psnr_ssim(output_image_path, target_image_path)
        if values is not None:
            local_results[output_file] = values

    return local_results


def main_worker(rank, gpu_id, output_folder, target_folder, output_files, return_dict, num_gpus):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    partition_size = len(output_files) // num_gpus
    start_idx = rank * partition_size
    end_idx = (rank + 1) * partition_size if rank != num_gpus - 1 else len(output_files)

    output_files_partition = output_files[start_idx:end_idx]

    local_results = calculate_iqa_for_partition(
        output_folder, target_folder, output_files_partition, device, rank
    )
    return_dict[rank] = local_results


import argparse
import cv2

if __name__ == "__main__":
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=str, default="output_dir")
    parser.add_argument("--target_folder", type=str, default=None)
    parser.add_argument("--metrics_save_path", type=str, default="./IQA_results")
    parser.add_argument("--gpu_ids", type=str, default="0")
    args = parser.parse_args()

    output_files = sorted([f for f in os.listdir(args.output_folder) if f.endswith('.png')])
    if args.target_folder is not None:
        target_files = sorted([f for f in os.listdir(args.target_folder) if f.endswith('.png')])
        assert len(output_files) == len(target_files), \
            f"Output/target count mismatch: {len(output_files)} != {len(target_files)}"

    manager = mp.Manager()
    return_dict = manager.dict()

    args.gpu_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(',')]
    print(f"Using GPU: {args.gpu_ids}")
    num_gpus = len(args.gpu_ids)

    processes = []
    for rank, gpu_id in enumerate(args.gpu_ids):
        p = mp.Process(target=main_worker, args=(
            rank, gpu_id, args.output_folder, args.target_folder,
            output_files, return_dict, num_gpus
        ))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    results = {}
    for rank in return_dict.keys():
        results.update(return_dict[rank])

    folder_name = os.path.basename(args.output_folder)
    parent_folder = os.path.dirname(args.output_folder)
    next_level_folder = os.path.basename(parent_folder)

    os.makedirs(args.metrics_save_path, exist_ok=True)
    average_results_filename = f"{args.metrics_save_path}/{next_level_folder}--{folder_name}.txt"
    results_filename = f"{args.metrics_save_path}/{next_level_folder}--{folder_name}.csv"

    if results:
        all_keys = set()
        for values in results.values():
            try:
                all_keys.update(values.keys())
            except Exception as e:
                print(f"Error: {e}")

        all_keys = sorted(all_keys)

        average_results = {}
        for key in all_keys:
            average_results[key] = np.mean([values.get(key, 0) for values in results.values()])

        # Weighted score for Perception Quality Track
        average_results['Total Score'] = 0
        for metric, value in average_results.items():
            if metric in ('psnr', 'ssim', 'Total Score'):
                continue
            if metric == 'DISTS':
                average_results['Total Score'] += (1 - value)
                print(f"DISTS Score: {1 - value}")
            elif metric == 'LPIPS':
                average_results['Total Score'] += (1 - value)
                print(f"LPIPS Score: {1 - value}")
            elif metric == 'NIQE':
                average_results['Total Score'] += max(0, (10 - value) / 10)
                print(f"NIQE Score: {max(0, (10 - value) / 10)}")
            elif metric == 'CLIP-IQA':
                average_results['Total Score'] += value
                print(f"CLIP-IQA Score: {value}")
            elif metric == 'MANIQA':
                average_results['Total Score'] += value
                print(f"MANIQA Score: {value}")
            elif metric == 'MUSIQ':
                average_results['Total Score'] += value / 100
                print(f"MUSIQ Score: {value / 100}")

        print("Average:")
        print(average_results)

        with open(results_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Filename'] + list(all_keys))
            for filename, values in results.items():
                row = [filename] + [values.get(key, '') for key in all_keys]
                writer.writerow(row)
            print(f"IQA results saved to {results_filename}")

        with open(average_results_filename, 'w') as f:
            for key, value in average_results.items():
                f.write(f"{key}: {value}\n")
            print(f"Average IQA results saved to {average_results_filename}")
