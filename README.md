# yf-lora-train

确保 yf-lora-train 目录跟 webui_lib, stable-diffusion-webui, sd-scripts 在同一层目录即可

第一次运行或者sd-scripts的训练脚本发生了更新时，可以先运行 python prepare.py 来生成一个修改过的训练脚本train_network_yf.py

运行训练脚本：

..\sd-scripts\venv\Scripts\Activate

accelerate launch --num_cpu_threads_per_process=2 .\train_network_yf.py  --enable_bucket   --pretrained_model_name_or_path=D:\stable-diffusion-webui\models\Stable-diffusion\meinamix_meinaV9.safetensors   --train_data_dir=D:\train\train\xuantu   --output_dir="D:\train\output"    --logging_dir="./logs"    --resolution="512,512"    --network_module=networks.lora    --max_train_epochs=1    --learning_rate="1e-4"   --unet_lr="1e-4"   --text_encoder_lr="1e-5"    --lr_scheduler="cosine_with_restarts"    --lr_warmup_steps=0    --network_dim=4   --network_alpha=4    --output_name=xuantu   --train_batch_size=1   --save_every_n_epochs=1   --mixed_precision="fp16"   --save_precision="fp16"    --seed="1337"    --cache_latents    --clip_skip=1   --prior_loss_weight=1    --max_token_length=225    --caption_extension=".txt"   --save_model_as="safetensors"   --min_bucket_reso=256     --max_bucket_reso=1024   --xformers --shuffle_caption --use_8bit_adam  


或者使用powershell运行 train.ps1