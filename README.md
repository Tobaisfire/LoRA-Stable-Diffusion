# Low-Rank Adaptation of Stable Diffusion (LoRA)
Low-Rank Adaptation of Large Language Models (LoRA) is a training method that accelerates the training of large models while consuming less memory. It adds pairs of rank-decomposition weight matrices (called update matrices) to existing weights, and only trains those newly added weights. This has a couple of advantages:

1. Previous pretrained weights are kept frozen so the model is not as prone to catastrophic forgetting.
2. Rank-decomposition matrices have significantly fewer parameters than the original model, which means that trained LoRA weights are easily portable.
LoRA matrices are generally added to the attention layers of the original model.
3. Diffusers provides the load_attn_procs() method to load the LoRA weights into a model’s attention layers. You can control the extent to which the model is adapted toward new training images via a scale parameter.
4. The greater memory-efficiency allows you to run fine-tuning on consumer GPUs and readily accessible in Kaggle or Google Colab notebooks.

```
I have fine tuned stable diffusion model on my dataset of 50 high quality Images of Interior Design.

Dataset structure :

- Datasset--|
          Train--|
              Img-caption-folder--|
                              Images & csv

csv should contain img name and image caption fo each images.
```
**For more details on dataset goto :  https://huggingface.co/blog/lora**

# Training Stable Diffusion using LoRA

**Installing dependencies**


```
# !git clone https://github.com/huggingface/diffusers
```

To get Pretrained Satble diffusion model V1-4 and a Text-to-image model


*   Installing dependecies.

```
!pip install accelerate
!pip install datasets
!pip install transformers
!pip install git+https://github.com/huggingface/diffusers
```

# Getting Started

# 1. Fine-tuning Stable diffusion with LoRA CLI

**If you have over 12 GB of memory, it is recommended to use Pivotal Tuning Inversion CLI provided with lora implementation. 
They have the best performance, and will be updated many times in the future as well. 
These are the parameters that worked for various dataset.


**Travelling to diffusers directory to training Script in diffures**
![image](https://github.com/Tobaisfire/LoRA-Stable-Diffusion/assets/67000746/9f264028-d6ee-4374-a230-1b21ada04065)

**Trained parameters**

```
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./data/data_disney"
export OUTPUT_DIR="./exps/output_dsn"

lora_pti \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --scale_lr \
  --learning_rate_unet=1e-4 \
  --learning_rate_text=1e-5 \
  --learning_rate_ti=5e-4 \
  --color_jitter \
  --lr_scheduler="linear" \
  --lr_warmup_steps=0 \
  --placeholder_tokens="<s1>|<s2>" \
  --use_template="style"\
  --save_steps=100 \
  --max_train_steps_ti=1000 \
  --max_train_steps_tuning=1000 \
  --perform_inversion=True \
  --clip_ti_decay \
  --weight_decay_ti=0.000 \
  --weight_decay_lora=0.001\
  --continue_inversion \
  --continue_inversion_lr=1e-4 \
  --device="cuda:0" \
  --lora_rank=1 \
#  --use_face_segmentation_condition\

```

**Parameter I used during Fine tuning on my dataset**
```
! python  train_text_to_image_lora.py
--mixed_precision='fp16'
--pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4'
--dataset_name='/content/interior_dataset'
 --caption_column='text'
--resolution=512
--random_flip
--train_batch_size=1
--max_train_steps=1000
--checkpointing_steps=100
--learning_rate=1e-04
--lr_scheduler="constant"
--lr_warmup_steps=0
--seed=42
--output_dir="/content/output_LoRa"
--validation_prompt="any prompt to evaluate "

```

# Load Trained model 

```
from diffusers import StableDiffusionPipeline
import torch

model_path = "/content/drive/MyDrive/LORA/Trained_Lora/Trained_model_using_LoRA.bin"
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe.unet.load_attn_procs(model_path)
# pipe.to("cuda")

```
# Function to get output img in given number of grid
```
from PIL import Image

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

```

# Final End 

```
#trained
number_images = 5
prompt1 = ["A Study Room in Black color theme with wooden furnitures."] * number_images
images1 = pipe(prompt1, num_inference_steps=30, guidance_scale=8.5).images
grid1 = image_grid(images1, rows=1, cols=5)
# image.show()

```

# References

This work was heavily influenced by  @cloneofsimo (https://github.com/cloneofsimo/lora.git), and originated from these awesome researches. I'm just applying them here on my dataset.

```
@article{roich2022pivotal,
  title={Pivotal tuning for latent-based editing of real images},
  author={Roich, Daniel and Mokady, Ron and Bermano, Amit H and Cohen-Or, Daniel},
  journal={ACM Transactions on Graphics (TOG)},
  volume={42},
  number={1},
  pages={1--13},
  year={2022},
  publisher={ACM New York, NY}
}
```

```
@article{ruiz2022dreambooth,
  title={Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation},
  author={Ruiz, Nataniel and Li, Yuanzhen and Jampani, Varun and Pritch, Yael and Rubinstein, Michael and Aberman, Kfir},
  journal={arXiv preprint arXiv:2208.12242},
  year={2022}
}
```

```
@article{gal2022image,
  title={An image is worth one word: Personalizing text-to-image generation using textual inversion},
  author={Gal, Rinon and Alaluf, Yuval and Atzmon, Yuval and Patashnik, Or and Bermano, Amit H and Chechik, Gal and Cohen-Or, Daniel},
  journal={arXiv preprint arXiv:2208.01618},
  year={2022}
}
```

```
@article{hu2021lora,
  title={Lora: Low-rank adaptation of large language models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  journal={arXiv preprint arXiv:2106.09685},
  year={2021}
}

```



