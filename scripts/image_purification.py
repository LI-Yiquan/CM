"""
Purification on diffusion model
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
from cm import dist_util, logger
from cm.image_datasets import load_data
from cm.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from cm.random_util import get_generator
from cm.karras_diffusion import karras_sample


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    if "consistency" in args.training_mode:
        distillation = True
    else:
        distillation = False

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
        distillation=distillation,
    )
                                      
    if args.model_path.endswith("pkl"):
        model = dist_util.load_state_dict(args.model_path, map_location="cpu")
    else:
        model.load_state_dict(
                    dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    if args.sampler == "multistep":
        assert len(args.ts) > 0
        ts = tuple(int(x) for x in args.ts.split(","))
    else:
        ts = None

    all_images = []
    all_images_corrupted = []
    all_labels = []
    generator = None

    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )
    
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        
        images, _ = next(data)
        images = (images).to('cuda')
        
        noise = th.randn_like(images, device='cuda') * args.noise_sigma
        
        corrupted_images = images + noise
        sample = karras_sample(
            diffusion,
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            steps=args.steps,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
            clip_denoised=args.clip_denoised,
            sampler=args.sampler,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            s_churn=args.s_churn,
            s_tmin=args.s_tmin,
            s_tmax=args.s_tmax,
            s_noise=args.s_noise,
            generator=generator,
            ts=ts,
            purification=True,
            corrupted_images=corrupted_images,
            noise_sigma=args.noise_sigma
        )
        sample = ((sample+1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        corrupted_images = ((images+1) * 127.5).clamp(0, 255).to(th.uint8)
        corrupted_images = corrupted_images.permute(0, 2, 3, 1)
        corrupted_images = corrupted_images.contiguous()
        
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        
        gathered_corrupted_images = [th.zeros_like(corrupted_images) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_corrupted_images, corrupted_images)  # gather not supported with NCCL
        
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        all_images_corrupted.extend([corrupted_images.cpu().numpy() for corrupted_images in gathered_corrupted_images])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    arr_corrupted = np.concatenate(all_images_corrupted, axis=0)
    arr_corrupted = arr_corrupted[: args.num_samples]
    print("arr shape: ",arr.shape)
    print("arr corrupted: ",arr_corrupted.shape)
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        out_path_corrupted = os.path.join(logger.get_dir(), f"corrupted_samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)
            np.savez(out_path_corrupted, arr_corrupted)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        training_mode="edm",
        generator="determ",
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        sampler="heun",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        steps=40,
        model_path="",
        seed=42,
        ts="",
        noise_sigma=0.5,
        data_dir="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
