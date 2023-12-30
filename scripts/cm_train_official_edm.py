"""
Train a diffusion model on images.
"""
import argparse
from cm import dist_util, logger
from cm.image_datasets import load_data
from cm.resample import create_named_schedule_sampler
from cm.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    cm_train_defaults,
    args_to_dict,
    add_dict_to_argparser,
    create_ema_and_scales_fn,
)
from cm.train_util import CMTrainLoop
import torch.distributed as dist
import copy
import torch
import dnnlib
import pickle

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    ema_scale_fn = create_ema_and_scales_fn(
        target_ema_mode=args.target_ema_mode,
        start_ema=args.start_ema,
        scale_mode=args.scale_mode,
        start_scales=args.start_scales,
        end_scales=args.end_scales,
        total_steps=args.total_training_steps,
        distill_steps_per_iter=args.distill_steps_per_iter,
    )
    if args.training_mode == "progdist":
        distillation = False
    elif "consistency" in args.training_mode:
        distillation = True
    else:
        raise ValueError(f"unknown training mode {args.training_mode}")
    model_and_diffusion_kwargs = args_to_dict(
        args, model_and_diffusion_defaults().keys()
    )
    model_and_diffusion_kwargs["distillation"] = distillation
    _, diffusion = create_model_and_diffusion(**model_and_diffusion_kwargs)
 
    edm_network_kwargs = {'model_type': 'SongUNet',
                          'embedding_type': 'positional', 
                          'encoder_type': 'standard', 
                          'decoder_type': 'standard', 
                          'channel_mult_noise': 1, 
                          'resample_filter': [1, 1],
                          'model_channels': 128, 
                          'channel_mult': [2, 2, 2], 
                          'class_name': 'training.networks.EDMPrecond', 
                          'augment_dim': 9, 
                          'dropout': 0.13, 
                          'use_fp16': False}
    edm_interface_kwargs = {'img_resolution': 32, 
                            'img_channels': 3, 
                            'label_dim': 0}
    model = dnnlib.util.construct_class_by_name(**edm_network_kwargs, **edm_interface_kwargs) # subclass of torch.nn.Module
    # model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained'
    # with dnnlib.util.open_url(f'{model_root}/edm-cifar10-32x32-uncond-vp.pkl') as f:
    #     model = pickle.load(f)['ema']    
    model.train().requires_grad_(True).to(dist_util.dev())
    #
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0], broadcast_buffers=False,find_unused_parameters=True)
    #
    model.module.model.map_augment.weight.requires_grad = False
    
    
    """
    TODO: Add convert_to_fp16 method
    if args.use_fp16:
        model.convert_to_fp16()
    """

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion) # diffusion in this class will not be used
    
    logger.log("creating data loader...")
    if args.batch_size == -1:
        batch_size = args.global_batch_size // dist.get_world_size()
        if args.global_batch_size % dist.get_world_size() != 0:
            logger.log(
                f"warning, using smaller global_batch_size of {dist.get_world_size()*batch_size} instead of {args.global_batch_size}"
            )
    else:
        batch_size = args.batch_size

    data = load_data(
        data_dir=args.data_dir,
        batch_size=batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    if len(args.teacher_model_path) > 0:  # path to the teacher score model.
        logger.log(f"loading the teacher model from {args.teacher_model_path}")
    
        print("Loading model")
        model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained'
        with dnnlib.util.open_url(f'{model_root}/edm-cifar10-32x32-uncond-vp.pkl') as f:
            teacher_model = pickle.load(f)['ema']
        
        teacher_model.to(dist_util.dev())
        teacher_model.eval()
        for dst, src in zip(model.parameters(), teacher_model.parameters()):
            dst.data.copy_(src.data)

        # if args.use_fp16:
        #     teacher_model.convert_to_fp16()

    else:
        teacher_model = None
        teacher_diffusion = None

    # load the target model for distillation, if path specified.

    logger.log("creating the target model")
    # target_model, _ = create_model_and_diffusion(
    #     **model_and_diffusion_kwargs,
    # )
    with dnnlib.util.open_url(f'{model_root}/edm-cifar10-32x32-uncond-vp.pkl') as f:
            target_model = pickle.load(f)['ema']
        
    target_model.to(dist_util.dev())
    target_model.train()
    
    dist_util.sync_params(target_model.parameters())
    dist_util.sync_params(target_model.buffers())

    # target model: same as teacher model
    for dst, src in zip(target_model.parameters(), model.parameters()):
        dst.data.copy_(src.data)

    # if args.use_fp16:
    #     target_model.convert_to_fp16()

    logger.log("training...")
    CMTrainLoop(
        model=model,
        target_model=target_model,
        teacher_model=teacher_model,
        teacher_diffusion=teacher_model,
        training_mode=args.training_mode,
        ema_scale_fn=ema_scale_fn,
        total_training_steps=args.total_training_steps,
        diffusion=diffusion,
        data=data,
        batch_size=batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        use_official_edm=True,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        global_batch_size=2048,
        batch_size=-1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=2000,
        resume_checkpoint="",
        use_fp16=True,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(cm_train_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
