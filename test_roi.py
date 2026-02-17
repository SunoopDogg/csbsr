##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## ROI-based Inference for CSBSR
## Based on original test.py by Yuki Kondo (Toyota Technological Institute)
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import argparse
import os
import re

import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image

from model.config import cfg
from model.modeling.build_model import JointModel, JointInvModel
from model.data.transforms.data_preprocess import TestTransforms
from model.data.samplers.patch_sampler import SplitPatch, JointPatch
from model.utils.misc import fix_model_state_dict
from model.utils.save_output import save_img, save_mask
from torch.multiprocessing import set_start_method


def select_roi(image_path, display_scale):
    """Show image in GUI and let user select ROI via mouse drag.

    Returns (x, y, w, h) in original image coordinates, or None if skipped.
    """
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    disp_w, disp_h = int(w * display_scale), int(h * display_scale)
    display_img = cv2.resize(img, (disp_w, disp_h))

    window_name = f"Select ROI: {image_path.name} (ENTER=confirm, ESC=skip)"
    roi = cv2.selectROI(window_name, display_img, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    if roi[2] == 0 or roi[3] == 0:
        return None

    # Convert display coordinates back to original resolution
    ox = int(roi[0] / display_scale)
    oy = int(roi[1] / display_scale)
    ow = int(roi[2] / display_scale)
    oh = int(roi[3] / display_scale)

    return (ox, oy, ow, oh)


def snap_roi_to_patch_grid(roi, patch_h, patch_w, img_h, img_w):
    """Snap ROI dimensions to patch-size multiples, keeping within image bounds."""
    x, y, w, h = roi

    # Round up to patch multiples (at least one patch)
    w = max(patch_w, ((w + patch_w - 1) // patch_w) * patch_w)
    h = max(patch_h, ((h + patch_h - 1) // patch_h) * patch_h)

    # Shift position if ROI exceeds image bounds
    if x + w > img_w:
        x = max(0, img_w - w)
    if y + h > img_h:
        y = max(0, img_h - h)

    # Final clamp to image bounds and ensure patch multiples
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    w = (w // patch_w) * patch_w
    h = (h // patch_h) * patch_h

    if w == 0 or h == 0:
        return None

    return (x, y, w, h)


def run_inference(args, cfg):
    device = torch.device(cfg.DEVICE)

    # Load model
    print("Loading model...")
    if cfg.MODEL.SR_SEG_INV:
        model = JointInvModel(cfg).to(device)
    else:
        model = JointModel(cfg).to(device)

    model.load_state_dict(fix_model_state_dict(
        torch.load(args.trained_model, map_location=lambda storage, loc: storage)
    ))
    model.eval()

    # Setup transforms and patch tools
    test_transforms = TestTransforms(cfg)
    patch_h, patch_w = cfg.INPUT.IMAGE_SIZE
    split_patch = SplitPatch(1, 3, patch_h, patch_w)
    joint_patch = JointPatch()
    scale_factor = cfg.MODEL.SCALE_FACTOR

    # Threshold settings for segmentation binarization
    thresholds = [i * 0.01 for i in range(1, 100)]
    threshold_map = torch.Tensor(thresholds).view(len(thresholds), 1, 1).to(device)
    zero = torch.Tensor([0]).to(device)
    save_thresholds_idx = [0] + [9 + i * 10 for i in range(9)] + [98]

    # Load image list
    image_dir = cfg.DATASET.TEST_IMAGE_DIR
    image_paths = sorted(Path(image_dir).glob("*.png"))
    if not image_paths:
        image_paths = sorted(Path(image_dir).glob("*.jpg"))
    assert len(image_paths) > 0, f"No images found in {image_dir}"

    os.makedirs(os.path.join(args.output_dirname, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dirname, "masks"), exist_ok=True)

    print(f"Found {len(image_paths)} images in {image_dir}")
    print(f"Patch size: {patch_h}x{patch_w}, Scale factor: {scale_factor}")
    print(f"Patches per batch: {args.patches_per_batch}")
    print(f"Output directory: {args.output_dirname}")
    print("=" * 60)

    with torch.no_grad():
        for img_idx, img_path in enumerate(image_paths):
            fname = img_path.name
            print(f"\n[{img_idx + 1}/{len(image_paths)}] {fname}")

            # Select ROI via GUI
            roi = select_roi(img_path, args.display_scale)
            if roi is None:
                print("  Skipped (no ROI selected)")
                continue

            # Read image as RGB numpy array (same as original pipeline uses PIL)
            img_np = np.array(Image.open(str(img_path)))
            img_h, img_w = img_np.shape[:2]

            # Snap ROI to patch grid
            roi = snap_roi_to_patch_grid(roi, patch_h, patch_w, img_h, img_w)
            if roi is None:
                print("  Skipped (ROI too small for patch size)")
                continue
            x, y, w, h = roi
            n_patches_h = h // patch_h
            n_patches_w = w // patch_w
            num_patches = n_patches_h * n_patches_w
            print(f"  ROI: x={x}, y={y}, w={w}, h={h} ({n_patches_w}x{n_patches_h} = {num_patches} patches)")

            # Crop ROI region
            cropped = img_np[y:y + h, x:x + w]

            # Apply test transforms (ConvertFromInts + ToTensor → /255)
            img_tensor, _ = test_transforms(cropped, None)

            # Split into patches
            patches, unfold_shape = split_patch(img_tensor)
            assert len(patches) == num_patches, f"Patch count mismatch: {len(patches)} vs {num_patches}"

            # Prepare unfold shapes for SR and segmentation reassembly
            sr_unfold_shape = unfold_shape.copy()
            sr_unfold_shape[[5, 6]] = sr_unfold_shape[[5, 6]] * scale_factor

            seg_unfold_shape = sr_unfold_shape.copy()
            seg_unfold_shape[[1, 4]] = cfg.MODEL.NUM_CLASSES

            # Process patches in small batches to avoid OOM
            sr_preds_list = []
            seg_preds_list = []

            for batch_start in range(0, num_patches, args.patches_per_batch):
                batch_end = min(batch_start + args.patches_per_batch, num_patches)
                batch_patches = patches[batch_start:batch_end]
                batch_size = len(batch_patches)

                # Correctly-sized damy_kernel (fixes the OOM bug)
                damy_kernel = torch.zeros(
                    (batch_size, 1, cfg.BLUR.KERNEL_SIZE, cfg.BLUR.KERNEL_SIZE)
                )

                # JointInvModel.forward takes only x; JointModel.forward takes (x, damy_kernel)
                if cfg.MODEL.SR_SEG_INV:
                    sr_pred, seg_pred, _ = model(batch_patches)
                else:
                    sr_pred, seg_pred, _ = model(batch_patches, damy_kernel)
                sr_preds_list.append(sr_pred.cpu())
                seg_preds_list.append(seg_pred.cpu())

            # Concatenate all batch results
            all_sr_preds = torch.cat(sr_preds_list, dim=0).to(device)
            all_seg_preds = torch.cat(seg_preds_list, dim=0).to(device)

            # Reassemble patches into full ROI image
            sr_result = joint_patch(all_sr_preds, sr_unfold_shape)
            seg_result = joint_patch(all_seg_preds, seg_unfold_shape)

            # Clip SR output to [0, 1]
            sr_result = sr_result.clamp(0, 1)

            # Save SR image
            save_img(args.output_dirname, sr_result, (fname,))

            # Binarize segmentation at multiple thresholds and save
            # seg_result: (1, 1, H, W), threshold_map: (99, 1, 1)
            # broadcast result: (1, 99, H, W) — [:, idx] selects one threshold
            seg_result_bi = (seg_result - threshold_map > zero).float()
            for idx in save_thresholds_idx:
                save_mask(args, seg_result_bi[:, idx], (fname,), thresholds[idx])

            # Save raw (continuous) segmentation output
            save_mask(args, seg_result, (fname,), -1)

            print(f"  Saved SR image and segmentation masks")

            # Free GPU memory
            del all_sr_preds, all_seg_preds, sr_result, seg_result, seg_result_bi, patches
            torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("All images processed!")


def main():
    parser = argparse.ArgumentParser(description="CSBSR ROI-based Inference")
    parser.add_argument("test_dir", type=str, help="Weight directory (e.g. weights/CSBSR_w_PSPNet_beta03/)")
    parser.add_argument("iter_or_weight_name", type=str, help="Iteration number or weight name (e.g. latest)")
    parser.add_argument("--patches_per_batch", type=int, default=6,
                        help="Number of patches per GPU batch (default: 6, reduce if OOM)")
    parser.add_argument("--display_scale", type=float, default=0.25,
                        help="Scale factor for image display in ROI selection GUI (default: 0.25)")
    parser.add_argument("--patch_size", type=int, default=64,
                        help="Patch size for splitting (default: 64 for TTI crack)")
    parser.add_argument("--image_dir", type=str, default="datasets/tti_crack/blured_image/",
                        help="Input image directory (default: datasets/tti_crack/blured_image/)")
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument("--trained_model", type=str, default=None)
    parser.add_argument("--output_dirname", type=str, default=None)
    args = parser.parse_args()

    # Resolve paths (same logic as test.py)
    if bool(re.search(r"[^0-9]", args.iter_or_weight_name)):
        _out_dir = args.iter_or_weight_name
        model_fname = args.iter_or_weight_name
    else:
        _out_dir = f"iter_{args.iter_or_weight_name}"
        model_fname = f"iteration_{args.iter_or_weight_name}"

    if args.config_file is None:
        args.config_file = f"{args.test_dir}config.yaml"
    if args.output_dirname is None:
        args.output_dirname = f"{args.test_dir}eval_roi/{_out_dir}"
    if args.trained_model is None:
        args.trained_model = f"{args.test_dir}model/{model_fname}.pth"

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print(f"Configuration file is loaded from {args.config_file}")
    cfg.merge_from_file(args.config_file)

    # Override dataset settings for ROI inference
    cfg.DATASET.TEST_IMAGE_DIR = args.image_dir
    cfg.INPUT.IMAGE_SIZE = [args.patch_size, args.patch_size]
    cfg.OUTPUT_DIR = args.output_dirname
    cfg.freeze()

    print(f"Running with config:\n{cfg}")
    run_inference(args, cfg)


if __name__ == "__main__":
    set_start_method("spawn")
    main()
