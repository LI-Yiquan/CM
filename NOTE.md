# Consistency distillation on official unconditional edm model

The file structure should look like this:  
```
/home/username  
        |--consistency_models  
        |--edm  
```
### Consistency Distillation
To run consistency distillation, use following command:
```
~/consistency_models/scripts$ mpiexec -n 1 python cm_train_official_edm.py --training_mode consistency_distillation --target_ema_mode fixed --start_ema 0.95 --scale_mode fixed --start_scales 18 --total_training_steps 600000 --loss_norm lpips --lr_anneal_steps 0 --teacher_model_path PATH_TO_EDM_OFFICIAL_MODEL --attention_resolutions 16 --class_cond False --use_scale_shift_norm True --dropout 0.0 --teacher_dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 32 --image_size 32 --lr 0.000008 --num_channels 64 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --schedule_sampler uniform --use_fp16 True --weight_decay 0.0 --weight_schedule uniform --data_dir PATH_TO_CIFAR10_DATA
```

In my repository, PATH_TO_EDM_OFFICIAL_MODEL is 
```
~/consistency_models/pretrained/edm-cifar10-32x32-uncond-vp.pkl
```
PATH_TO_CIFAR10_DATA is 
```
~/consistency_models/data/cifar10/cifar_train
```
Add CUDA_VISIBLE_DEVICES=GPU_ID to set gpu devices, for example, run the following on gpu7
```
~/consistency_models/scripts$ CUDA_VISIBLE_DEVICES=7 python *.py
```
May have to add "export PYTHONPATH=$PYTHONPATH:path" for edm and consistency_models directory

### Calculating FID
To evaluate the result, use code in ~/edm, first generate images by generate_selfmodel.py, then calculate fid by fid.py
