####################################################################
# Training EDM models on class-unconditional cifar10
####################################################################

mpiexec -n 4 python edm_train.py --attention_resolutions 16 --class_cond False --use_scale_shift_norm True --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 1024 --image_size 32 --lr 0.0002 --num_channels 128 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --schedule_sampler lognormal --use_fp16 True --weight_decay 0.0 --weight_schedule karras --data_dir /home/yiquan/consistency_models/data/cifar10/cifar_train

####################################################################
# Sampling EDM models on class-unconditional cifar10
####################################################################

mpiexec -n 4 python image_sample.py --batch_size 2048 --sigma_max 80 --sigma_min 0.002 --s_churn 0 --steps 40 --sampler heun --model_path /home/yiquan/consistency_models/results/train_cifar10/openai-2023-12-29-15-56-16-663466/ema_0.999_050000.pt --attention_resolutions 16 --class_cond False --dropout 0.1 --image_size 32 --num_channels 128 --num_head_channels 64 --num_res_blocks 3 --num_samples 50000 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --weight_schedule karras
 
####################################################################
# Training CM models on class-unconditional cifar10
####################################################################

CUDA_VISIBLE_DEVICES=6 mpiexec -n 1 python cm_train.py --training_mode consistency_distillation --target_ema_mode fixed --start_ema 0.95 --scale_mode fixed --start_scales 40 --total_training_steps 600000 --loss_norm lpips --lr_anneal_steps 0 --teacher_model_path /home/yiquan/consistency_models/results/train_cifar10/openai-2023-06-27-21-00-56-216763/ema_0.9999_100000.pt --attention_resolutions 16 --class_cond False --use_scale_shift_norm True --dropout 0.0 --teacher_dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 64 --image_size 32 --lr 0.000008 --num_channels 128 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --schedule_sampler uniform --use_fp16 True --weight_decay 0.0 --weight_schedule uniform --data_dir /home/yiquan/consistency_models/data/cifar10/cifar_train
 
####################################################################
# Sampling CM models on class-unconditional cifar10
####################################################################

mpiexec -n 4 python image_sample.py --batch_size 64 --sigma_max 80 --sigma_min 0.002 --s_churn 0 --steps 1 --sampler heun --model_path /home/yiquan/consistency_models/results/train_cifar10/openai-2023-12-30-01-11-04-896370/model034000.pt --attention_resolutions 16 --class_cond False --dropout 0.1 --image_size 32 --num_channels 128 --num_head_channels 64 --num_res_blocks 3 --num_samples 500 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --weight_schedule karras


####################################################################
# Calculating sampled images FID
####################################################################

CUDA_VISIBLE_DEVICES=5 python fid_npzs.py --ref=/home/yiquan/consistency_models/data/cifar10/cifar_ref/cifar10_legacy_pytorch_train_32.npz --num_samples=50000 --images=/home/yiquan/consistency_models/results/train_cifar10/openai-2023-12-30-12-42-05-583630/samples_50000x32x32x3.npz

