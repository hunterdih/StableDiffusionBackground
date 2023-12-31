import torch
import diffusers
import requests
from PIL import Image
from io import BytesIO
import torch
from pathlib import Path
import os
import numpy as np
from removeSeam import *
import click

@click.command()
@click.option('--prompt', help='Image prompt.') # this is the only required parameter
@click.option('--width', default=640, help='Initial width of the image.')
@click.option('--height', default=360, help='Initial height of the image.')
@click.option('--desired_width', default=2560, help='Desired width after increasing resolution.')
@click.option('--desired_height', default=1440, help='Desired height after increasing resolution.')
@click.option('--overlap', default=30, help='Overlap size for image parts.')
@click.option('--blend_width', default=120, help='Width for blending image parts.')
@click.option('--output', default="", help='Optional output directory suffix.')
@click.option('--increase_res', default=True, help='Increase resolution.')
@click.option('--save_images', default=True, help='Save images.')
def main(prompt, width, height, desired_width, desired_height, overlap, blend_width, output, increase_res, save_images):

    desired_width_before_merge = desired_width + blend_width
    desired_height_before_merge = desired_height + blend_width
    
    output_dir = r'outputs/' + prompt if output == "" else r'outputs/' + output

    if not os.path.exists(output_dir) and save_images:
        os.makedirs(output_dir)
        os.makedirs(output_dir + '/original')
        os.makedirs(output_dir + '/parts')
        os.makedirs(output_dir + '/parts_super_resolution')
        os.makedirs(output_dir + '/super_resolution')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id_sd = "Lykon/dreamshaper-7"
    #model_id_sd = "SimianLuo/LCM_Dreamshaper_v7"
    model_id_sr = "CompVis/ldm-super-resolution-4x-openimages"

    adapter_id = "latent-consistency/lcm-lora-sdv1-5"

    pipe_sd = diffusers.DiffusionPipeline.from_pretrained(model_id_sd, torch_dtype=torch.float16, variant="fp16", cache_dir='Model',safety_checker=None)
    pipe_sd.scheduler = diffusers.LCMScheduler.from_config(pipe_sd.scheduler.config)
    pipe_sd.to(device)

    # load and fuse lcm lora
    pipe_sd.load_lora_weights(adapter_id)
    pipe_sd.fuse_lora()

    pipe_sd.enable_vae_tiling()
    pipe_sd.enable_xformers_memory_efficient_attention()
    pipe_sd.enable_sequential_cpu_offload()

    choice = 'placeholder'
    while True:
        # disable guidance_scale by passing 0
        image = pipe_sd(batch_size = 16, num_inference_steps=5, prompt=prompt, width=width, height=height, guidance_scale=0).images[0]
        #image = pipe_sd(prompt=prompt, width = width, height = height, num_inference_steps=50, guidance_scale=8.0, lcm_origin_steps=50, output_type="pil").images[0]
        image.show()
        choice = input("Satisfied with image? [y]")
        if choice == 'y':
            break
        elif choice == 'exit':
            exit()

    width, height = image.size

    # Setting the points for cropped image
    top_left = (0, 0, (width // 2)+overlap//2, (height // 2)+overlap//2)

    top_right = ((width // 2)-overlap//2, 0, width, (height // 2)+overlap//2)

    bottom_left = (0, (height // 2)-overlap//2, (width // 2)+overlap//2, height)

    bottom_right = ((width // 2)-overlap//2, (height // 2)-overlap//2, width, height)

    print(top_left)
    print(top_right)
    print(bottom_left)
    print(bottom_right)

    # Cropped image of above dimension
    # (It will not change cat_with_hat image)
    im_tl = image.crop(top_left)
    im_tr = image.crop(top_right)
    im_bl = image.crop(bottom_left)
    im_br = image.crop(bottom_right)

    im_tl.save(output_dir + r'/parts/im_tl.png')
    im_tr.save(output_dir + r'/parts/im_tr.png')
    im_bl.save(output_dir + r'/parts/im_bl.png')
    im_br.save(output_dir + r'/parts/im_br.png')
    """im_tl = im_tl.resize((desired_width_before_merge // 2, desired_height_before_merge // 2))
    im_tr = im_tr.resize((desired_width_before_merge // 2, desired_height_before_merge // 2))
    im_bl = im_bl.resize((desired_width_before_merge // 2, desired_height_before_merge // 2))
    im_br = im_tl.resize((desired_width_before_merge // 2, desired_height_before_merge // 2))"""


    if increase_res:
        pipe_sr = diffusers.LDMSuperResolutionPipeline.from_pretrained(model_id_sr, variant="fp16", cache_dir='Model', safety_checker=None)
        pipe_sr = pipe_sr.to(device)

        # run pipe_sr in inference (sample random noise and denoise)
        print("upscaling top left quadrant...")
        upscaled_im_tl = pipe_sr(im_tl, num_inference_steps=20, eta=1).images[0]
        print("upscaling top right quadrant...")
        upscaled_im_tr = pipe_sr(im_tr, num_inference_steps=20, eta=1).images[0]
        print("upscaling bottom left quadrant...")
        upscaled_im_bl = pipe_sr(im_bl, num_inference_steps=20, eta=1).images[0]
        print("upscaling bottom right quadrant...")
        upscaled_im_br = pipe_sr(im_br, num_inference_steps=20, eta=1).images[0]


        upscaled_im_tl.save(output_dir + r'/parts_super_resolution/im_tl.png')
        upscaled_im_tr.save(output_dir + r'/parts_super_resolution/im_tr.png')
        upscaled_im_bl.save(output_dir + r'/parts_super_resolution/im_bl.png')
        upscaled_im_br.save(output_dir + r'/parts_super_resolution/im_br.png')

        upscaled_im_tl = upscaled_im_tl.resize((desired_width_before_merge // 2, desired_height_before_merge // 2))
        upscaled_im_tr = upscaled_im_tr.resize((desired_width_before_merge // 2, desired_height_before_merge // 2))
        upscaled_im_bl = upscaled_im_bl.resize((desired_width_before_merge // 2, desired_height_before_merge // 2))
        upscaled_im_br = upscaled_im_br.resize((desired_width_before_merge // 2, desired_height_before_merge // 2))

        ###########Seam not removed################
        upscaled_image = Image.new('RGB', (desired_width_before_merge, desired_height_before_merge), 'white')

        upscaled_image.paste(upscaled_im_tl, (0, 0))
        upscaled_image.paste(upscaled_im_tr, (desired_width_before_merge//2, 0))
        upscaled_image.paste(upscaled_im_bl, (0, desired_height_before_merge//2))
        upscaled_image.paste(upscaled_im_br, (desired_width_before_merge // 2, desired_height_before_merge // 2))

        upscaled_image = upscaled_image.resize((desired_width_before_merge, desired_height_before_merge))

        upscaled_image_npy = np.array(upscaled_image)

        if save_images:
            upscaled_image.save(output_dir + r'/super_resolution/super_resolution.png')
            image.save(output_dir + '/original/original_image.png')
    
        #######################

    print("Blending images to remove the seam...")
    blend_images(output_dir + r'/parts_super_resolution/im_tl.png', output_dir + r'/parts_super_resolution/im_tr.png', output_dir + r'/parts_super_resolution/top.png', resize_width=desired_width_before_merge, resize_height=desired_height_before_merge, blend_width=blend_width, direction='horizontal')
    blend_images(output_dir + r'/parts_super_resolution/im_bl.png', output_dir + r'/parts_super_resolution/im_br.png', output_dir + r'/parts_super_resolution/bottom.png', resize_width=desired_width_before_merge, resize_height=desired_height_before_merge, blend_width=blend_width, direction='horizontal')
    blend_images(output_dir + r'/parts_super_resolution/top.png', output_dir + r'/parts_super_resolution/bottom.png', output_dir + r'/super_resolution/seam_removed.png', blend_width=blend_width, direction='vertical')
    
if __name__ == '__main__':
    main()