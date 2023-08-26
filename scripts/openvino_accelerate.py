# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: AGPL-3.0

from logging.config import valid_ident
from multiprocessing import Value
from pdb import run
import cv2
import os
import torch
import time
import hashlib
import functools
import gradio as gr
import numpy as np
import sys

import yaml

import modules
import modules.paths as paths


from modules import images, devices, extra_networks, masking, shared, sd_models_config
from modules.processing import (
    StableDiffusionProcessing, Processed, apply_overlay, apply_color_correction,
    get_fixed_seed, create_infotext, setup_color_correction,
    process_images, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img
)
from modules.sd_models import CheckpointInfo, get_checkpoint_state_dict
from modules.shared import opts, state
from modules.ui_common import create_refresh_button
from modules.timer import Timer

from PIL import Image, ImageOps
from pathlib import Path

from openvino.frontend.pytorch.torchdynamo import backend, compile # noqa: F401
from openvino.frontend.pytorch.torchdynamo.execute import execute, partitioned_modules, compiled_cache # noqa: F401
from openvino.frontend.pytorch.torchdynamo.partition import Partitioner
from openvino.runtime import Core, Type, PartialShape

import modules.scripts as scripts
import scripts.ov_model_state as ovms
model_state = ovms.model_state



from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    AutoencoderKL,
)

from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt

from diffusers.utils import (
    DIFFUSERS_CACHE,
    HF_HUB_OFFLINE,
    #is_safetensors_available,
)

def openvino_clear_caches():
    global partitioned_modules
    global compiled_cache

    compiled_cache.clear()
    partitioned_modules.clear()

def from_single_file(self, pretrained_model_link_or_path, **kwargs):

    cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
    resume_download = kwargs.pop("resume_download", False)
    force_download = kwargs.pop("force_download", False)
    proxies = kwargs.pop("proxies", None)
    local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
    use_auth_token = kwargs.pop("use_auth_token", None)
    revision = kwargs.pop("revision", None)
    extract_ema = kwargs.pop("extract_ema", False)
    image_size = kwargs.pop("image_size", None)
    scheduler_type = kwargs.pop("scheduler_type", "pndm")
    num_in_channels = kwargs.pop("num_in_channels", None)
    upcast_attention = kwargs.pop("upcast_attention", None)
    load_safety_checker = kwargs.pop("load_safety_checker", True)
    prediction_type = kwargs.pop("prediction_type", None)
    text_encoder = kwargs.pop("text_encoder", None)
    tokenizer = kwargs.pop("tokenizer", None)
    local_config_file = kwargs.pop("local_config_file", None)

    torch_dtype = kwargs.pop("torch_dtype", None)

    #use_safetensors = kwargs.pop("use_safetensors", None if is_safetensors_available() else False)

    pipeline_name = self.__name__
    file_extension = pretrained_model_link_or_path.rsplit(".", 1)[-1]
    from_safetensors = file_extension == "safetensors"

    #if from_safetensors and use_safetensors is False:
       # raise ValueError("Make sure to install `safetensors` with `pip install safetensors`.")

    # TODO: For now we only support stable diffusion
    stable_unclip = None
    model_type = None
    controlnet = False

    if pipeline_name == "StableDiffusionControlNetPipeline":
        # Model type will be inferred from the checkpoint.
        controlnet = True
    elif "StableDiffusion" in pipeline_name:
        # Model type will be inferred from the checkpoint.
        pass
    elif pipeline_name == "StableUnCLIPPipeline":
        model_type = "FrozenOpenCLIPEmbedder"
        stable_unclip = "txt2img"
    elif pipeline_name == "StableUnCLIPImg2ImgPipeline":
        model_type = "FrozenOpenCLIPEmbedder"
        stable_unclip = "img2img"
    elif pipeline_name == "PaintByExamplePipeline":
        model_type = "PaintByExample"
    elif pipeline_name == "LDMTextToImagePipeline":
        model_type = "LDMTextToImage"
    else:
        raise ValueError(f"Unhandled pipeline class: {pipeline_name}")

    # remove huggingface url
    for prefix in ["https://huggingface.co/", "huggingface.co/", "hf.co/", "https://hf.co/"]:
        if pretrained_model_link_or_path.startswith(prefix):
            pretrained_model_link_or_path = pretrained_model_link_or_path[len(prefix) :]
    # Code based on diffusers.pipelines.pipeline_utils.DiffusionPipeline.from_pretrained
    ckpt_path = Path(pretrained_model_link_or_path)
    if not ckpt_path.is_file():
        # get repo_id and (potentially nested) file path of ckpt in repo
        repo_id = "/".join(ckpt_path.parts[:2])
        file_path = "/".join(ckpt_path.parts[2:])

        if file_path.startswith("blob/"):
            file_path = file_path[len("blob/") :]

        if file_path.startswith("main/"):
            file_path = file_path[len("main/") :]

        from huggingface_hub import hf_hub_download
        pretrained_model_link_or_path = hf_hub_download(
            repo_id,
            filename=file_path,
            cache_dir=cache_dir,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            use_auth_token=use_auth_token,
            revision=revision,
            force_download=force_download,
        )

    pipe = download_from_original_stable_diffusion_ckpt(
        pretrained_model_link_or_path,
        original_config_file=local_config_file,
        pipeline_class=self,
        model_type=model_type,
        stable_unclip=stable_unclip,
        controlnet=controlnet,
        from_safetensors=from_safetensors,
        extract_ema=extract_ema,
        image_size=image_size,
        scheduler_type=scheduler_type,
        num_in_channels=num_in_channels,
        upcast_attention=upcast_attention,
        load_safety_checker=load_safety_checker,
        prediction_type=prediction_type,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )

    if torch_dtype is not None:
        pipe.to(torch_dtype=torch_dtype)

    return pipe

StableDiffusionPipeline.from_single_file = functools.partial(from_single_file, StableDiffusionPipeline)


def sd_diffusers_model(self):
    import modules.sd_models
    return modules.sd_models.model_data.get_sd_model()

def cond_stage_key(self):
    return None

shared.sd_diffusers_model = sd_diffusers_model
#refiner model
shared.sd_refiner_model = None

def set_scheduler(sd_model, sampler_name):
    if (sampler_name == "Euler a"):
        sd_model.scheduler = EulerAncestralDiscreteScheduler.from_config(sd_model.scheduler.config)
    elif (sampler_name == "Euler"):
        sd_model.scheduler = EulerDiscreteScheduler.from_config(sd_model.scheduler.config)
    elif (sampler_name == "LMS"):
        sd_model.scheduler = LMSDiscreteScheduler.from_config(sd_model.scheduler.config)
    elif (sampler_name == "Heun"):
        sd_model.scheduler = HeunDiscreteScheduler.from_config(sd_model.scheduler.config)
    elif (sampler_name == "DPM++ 2M"):
        sd_model.scheduler = DPMSolverMultistepScheduler.from_config(sd_model.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=False)
    elif (sampler_name == "LMS Karras"):
        sd_model.scheduler = LMSDiscreteScheduler.from_config(sd_model.scheduler.config, use_karras_sigmas=True)
    elif (sampler_name == "DPM++ 2M Karras"):
        sd_model.scheduler = DPMSolverMultistepScheduler.from_config(sd_model.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True)
    elif (sampler_name == "DDIM"):
        sd_model.scheduler = DDIMScheduler.from_config(sd_model.scheduler.config)
    elif (sampler_name == "PLMS"):
        sd_model.scheduler = PNDMScheduler.from_config(sd_model.scheduler.config)
    else:
        sd_model.scheduler = EulerAncestralDiscreteScheduler.from_config(sd_model.scheduler.config)

    return sd_model.scheduler

def get_diffusers_sd_model(model_config, vae_ckpt, sampler_name, enable_caching, openvino_device, mode, is_xl_ckpt, refiner_ckpt, refiner_steps):
    if (model_state.recompile == 1):
        os.environ["INFERENCE_PRECISION_HINT"] = "None"
        torch._dynamo.reset()
        openvino_clear_caches()
        curr_dir_path = os.getcwd()
        checkpoint_name = shared.opts.sd_model_checkpoint.split(" ")[0]
        checkpoint_path = os.path.join(curr_dir_path, 'models', 'Stable-diffusion', checkpoint_name)
        checkpoint_info = CheckpointInfo(checkpoint_path)
        timer = Timer()
        state_dict = get_checkpoint_state_dict(checkpoint_info, timer)
        checkpoint_config = sd_models_config.find_checkpoint_config(state_dict, checkpoint_info)
        print("OpenVINO Script:  created model from config : " + checkpoint_config)

        if(is_xl_ckpt):
            if model_config != "None":
                local_config_file = os.path.join(curr_dir_path, 'configs', model_config)
                sd_model = StableDiffusionXLPipeline.from_single_file(checkpoint_path, local_config_file=local_config_file, use_safetensors=True, dtype=torch.float32)
            else:
                sd_model = StableDiffusionXLPipeline.from_single_file(checkpoint_path, local_config_file=checkpoint_config, use_safetensors=True, dtype=torch.float32)
            if (mode == 1):
                sd_model = StableDiffusionXLImg2ImgPipeline.from_single_file(checkpoint_path, local_config_file=checkpoint_config, use_safetensors=True, dtype=torch.float32)
            elif (mode == 2):
                sd_model = StableDiffusionXLInpaintPipeline.from_single_file(checkpoint_path, local_config_file=checkpoint_config, use_safetensors=True, dtype=torch.float32)
        else:
            if model_config != "None":
                local_config_file = os.path.join(curr_dir_path, 'configs', model_config)
                sd_model = StableDiffusionPipeline.from_single_file(checkpoint_path, local_config_file=local_config_file, load_safety_checker=False, use_safetensors=True, dtype=torch.float32)
            else:
                sd_model = StableDiffusionPipeline.from_single_file(checkpoint_path, local_config_file=checkpoint_config, load_safety_checker=False, use_safetensors=True, dtype=torch.float32)
            if (mode == 1):
                sd_model = StableDiffusionImg2ImgPipeline(**sd_model.components)
            elif (mode == 2):
                sd_model = StableDiffusionInpaintPipeline(**sd_model.components)
        sd_model.sd_checkpoint_info = checkpoint_info
        sd_model.sd_model_hash = checkpoint_info.calculate_shorthash()
        sd_model.safety_checker = None
        sd_model.cond_stage_key = functools.partial(cond_stage_key, shared.sd_model)
        sd_model.scheduler = set_scheduler(sd_model, sampler_name)
        ## UNET
        sd_model.unet = torch.compile(sd_model.unet,  backend="openvino_fx")
        print("UNET COMPILED")
        ## VAE
        if vae_ckpt == "Disable-VAE-Acceleration":
            sd_model.vae.decode = sd_model.vae.decode
        elif vae_ckpt == "None":
            #os.environ["INFERENCE_PRECISION_HINT"] = "f32"
            sd_model.vae.decode = torch.compile(sd_model.vae.decode, backend="openvino_fx")
            print("VAE Compiled")
        else:
            vae_path = os.path.join(curr_dir_path, 'models', 'VAE', vae_ckpt)
            print("OpenVINO Script:  loading vae from : " + vae_path)
            sd_model.vae = AutoencoderKL.from_single_file(vae_path, local_files_only=True)
            #os.environ["INFERENCE_PRECISION_HINT"] = "f32"
            sd_model.vae = torch.compile(sd_model.vae,  backend="openvino_fx")
        shared.sd_diffusers_model = sd_model
        del sd_model
    return shared.sd_diffusers_model

##get refiner model

def get_diffusers_sd_refiner_model(model_config, vae_ckpt, sampler_name, enable_caching, openvino_device, mode, is_xl_ckpt, refiner_ckpt, refiner_steps):
    if (model_state.recompile == 1):
        curr_dir_path = os.getcwd()
        if refiner_ckpt != "None":
            refiner_checkpoint_path= os.path.join(curr_dir_path, 'models', 'Stable-diffusion', refiner_ckpt)
            refiner_checkpoint_info = CheckpointInfo(refiner_checkpoint_path)
            refiner_model = StableDiffusionXLImg2ImgPipeline.from_single_file(refiner_checkpoint_path, load_safety_checker=False, use_safetensors=True)
            print("OpenVINO Script: refiner model loaded from" + refiner_checkpoint_path)
            refiner_model.sd_checkpoint_info = refiner_checkpoint_info
            refiner_model.sd_model_hash = refiner_checkpoint_info.calculate_shorthash()
            ## UNET
            refiner_model.unet = torch.compile(refiner_model.unet,  backend="openvino_fx")
            print("OpenVINO Script: refiner model compiled")
        shared.sd_refiner_model = refiner_model
        del refiner_model
    return shared.sd_refiner_model


def init_new(self, all_prompts, all_seeds, all_subseeds):
    crop_region = None

    image_mask = self.image_mask

    if image_mask is not None:
        image_mask = image_mask.convert('L')

        if self.inpainting_mask_invert:
            image_mask = ImageOps.invert(image_mask)

        if self.mask_blur_x > 0:
            np_mask = np.array(image_mask)
            kernel_size = 2 * int(4 * self.mask_blur_x + 0.5) + 1
            np_mask = cv2.GaussianBlur(np_mask, (kernel_size, 1), self.mask_blur_x)
            image_mask = Image.fromarray(np_mask)

        if self.mask_blur_y > 0:
            np_mask = np.array(image_mask)
            kernel_size = 2 * int(4 * self.mask_blur_y + 0.5) + 1
            np_mask = cv2.GaussianBlur(np_mask, (1, kernel_size), self.mask_blur_y)
            image_mask = Image.fromarray(np_mask)

        if self.inpaint_full_res:
            self.mask_for_overlay = image_mask
            mask = image_mask.convert('L')
            crop_region = masking.get_crop_region(np.array(mask), self.inpaint_full_res_padding)
            crop_region = masking.expand_crop_region(crop_region, self.width, self.height, mask.width, mask.height)
            x1, y1, x2, y2 = crop_region

            mask = mask.crop(crop_region)
            image_mask = images.resize_image(2, mask, self.width, self.height)
            self.paste_to = (x1, y1, x2-x1, y2-y1)
        else:
            image_mask = images.resize_image(self.resize_mode, image_mask, self.width, self.height)
            np_mask = np.array(image_mask)
            np_mask = np.clip((np_mask.astype(np.float32)) * 2, 0, 255).astype(np.uint8)
            self.mask_for_overlay = Image.fromarray(np_mask)

        self.overlay_images = []

    latent_mask = self.latent_mask if self.latent_mask is not None else image_mask

    add_color_corrections = opts.img2img_color_correction and self.color_corrections is None
    if add_color_corrections:
        self.color_corrections = []
    imgs = []
    for img in self.init_images:
        # Save init image
        if opts.save_init_img:
            self.init_img_hash = hashlib.md5(img.tobytes()).hexdigest()
            images.save_image(img, path=opts.outdir_init_images, basename=None, forced_filename=self.init_img_hash, save_to_dirs=False)

        image = images.flatten(img, opts.img2img_background_color)

        if crop_region is None and self.resize_mode != 3:
            image = images.resize_image(self.resize_mode, image, self.width, self.height)

        if image_mask is not None:
            image_masked = Image.new('RGBa', (image.width, image.height))
            image_masked.paste(image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(self.mask_for_overlay.convert('L')))
            self.mask = image_mask
            self.overlay_images.append(image_masked.convert('RGBA'))

        # crop_region is not None if we are doing inpaint full res
        if crop_region is not None:
            image = image.crop(crop_region)
            image = images.resize_image(2, image, self.width, self.height)

        self.init_images = image
        if image_mask is not None:
            if self.inpainting_fill != 1:
                image = masking.fill(image, latent_mask)

        if add_color_corrections:
            self.color_corrections.append(setup_color_correction(image))

        image = np.array(image).astype(np.float32) / 255.0
        image = np.moveaxis(image, 2, 0)

        imgs.append(image)

    if len(imgs) == 1:
        if self.overlay_images is not None:
            self.overlay_images = self.overlay_images * self.batch_size

        if self.color_corrections is not None and len(self.color_corrections) == 1:
            self.color_corrections = self.color_corrections * self.batch_size

    elif len(imgs) <= self.batch_size:
        self.batch_size = len(imgs)
    else:
        raise RuntimeError(f"bad number of images passed: {len(imgs)}; expecting {self.batch_size} or less")

def process_images_openvino(p: StableDiffusionProcessing, model_config, vae_ckpt, sampler_name, enable_caching, openvino_device, mode, is_xl_ckpt, refiner_ckpt, refiner_steps) -> Processed:
    """this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""

    print(os.getenv("OPENVINO_TORCH_BACKEND_DEVICE"))
    
    if type(p.prompt) == list:
        assert(len(p.prompt) > 0)
    else:
        assert p.prompt is not None

    devices.torch_gc()

    seed = get_fixed_seed(p.seed)
    subseed = get_fixed_seed(p.subseed)

    comments = {}
    custom_inputs = {}

    p.setup_prompts()

    if type(seed) == list:
        p.all_seeds = seed
    else:
        p.all_seeds = [int(seed) + (x if p.subseed_strength == 0 else 0) for x in range(len(p.all_prompts))]

    if type(subseed) == list:
        p.all_subseeds = subseed
    else:
        p.all_subseeds = [int(subseed) + x for x in range(len(p.all_prompts))]

    def infotext(iteration=0, position_in_batch=0):
        return create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, comments, iteration, position_in_batch)

    if p.scripts is not None:
        p.scripts.process(p)

    if 'ControlNet' in p.extra_generation_params:
        return process_images(p)

    infotexts = []
    output_images = []

    with torch.no_grad():
        with devices.autocast():
            p.init(p.all_prompts, p.all_seeds, p.all_subseeds)

        if state.job_count == -1:
            state.job_count = p.n_iter

        extra_network_data = None
        for n in range(p.n_iter):
            p.iteration = n

            if state.skipped:
                state.skipped = False

            if state.interrupted:
                break

            p.prompts = p.all_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            p.negative_prompts = p.all_negative_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            p.seeds = p.all_seeds[n * p.batch_size:(n + 1) * p.batch_size]
            p.subseeds = p.all_subseeds[n * p.batch_size:(n + 1) * p.batch_size]

            if p.scripts is not None:
                p.scripts.before_process_batch(p, batch_number=n, prompts=p.prompts, seeds=p.seeds, subseeds=p.subseeds)

            if len(p.prompts) == 0:
                break

            if (model_state.height != p.height or model_state.width != p.width or model_state.batch_size != p.batch_size
                    or model_state.mode != mode or model_state.model_hash != shared.sd_model.sd_model_hash):
                model_state.recompile = 1
                model_state.height = p.height
                model_state.width = p.width
                model_state.batch_size = p.batch_size
                model_state.mode = mode
                model_state.model_hash = shared.sd_model.sd_model_hash

            shared.sd_diffusers_model = get_diffusers_sd_model(model_config, vae_ckpt, sampler_name, enable_caching, openvino_device, mode, is_xl_ckpt, refiner_ckpt, refiner_steps)
            shared.sd_diffusers_model.scheduler = set_scheduler(shared.sd_diffusers_model, sampler_name)

            if refiner_ckpt != "None":
                shared.sd_refiner_model = get_diffusers_sd_refiner_model(model_config, vae_ckpt, sampler_name, enable_caching, openvino_device, mode, is_xl_ckpt, refiner_ckpt, refiner_steps)
                shared.sd_refiner_model.scheduler = set_scheduler(shared.sd_refiner_model, sampler_name)
                print("refiner used: " + refiner_ckpt)

            extra_network_data = p.parse_extra_network_prompts()

            if not p.disable_extra_networks:
                with devices.autocast():
                    extra_networks.activate(p, p.extra_network_data)

            if ('lora' in modules.extra_networks.extra_network_registry):
                import lora
                # TODO: multiple Loras aren't supported for Diffusers now, needs to add warning
                if lora.loaded_loras:
                    lora_model = lora.loaded_loras[0]
                    shared.sd_diffusers_model.load_lora_weights(os.path.join(os.getcwd(), "models", "Lora"), weight_name=lora_model.name + ".safetensors")
                    custom_inputs.update(cross_attention_kwargs={"scale" : lora_model.te_multiplier})

            if p.scripts is not None:
                p.scripts.process_batch(p, batch_number=n, prompts=p.prompts, seeds=p.seeds, subseeds=p.subseeds)

            # params.txt should be saved after scripts.process_batch, since the
            # infotext could be modified by that callback
            # Example: a wildcard processed by process_batch sets an extra model
            # strength, which is saved as "Model Strength: 1.0" in the infotext
            if n == 0:
                with open(os.path.join(paths.data_path, "params.txt"), "w", encoding="utf8") as file:
                    file.write(create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, comments=[], position_in_batch=0 % p.batch_size, iteration=0 // p.batch_size))

            if p.n_iter > 1:
                shared.state.job = f"Batch {n+1} out of {p.n_iter}"

            generator = [torch.Generator(device="cpu").manual_seed(s) for s in p.seeds]

            time_stamps = []

            def callback(iter, t, latents):
                time_stamps.append(time.time()) # noqa: B023

            time_stamps.append(time.time())

            if (mode == 0):
                custom_inputs.update({
                    'width': p.width,
                    'height': p.height,
                })
            elif (mode == 1):
                custom_inputs.update({
                    'image': p.init_images,
                    'strength':p.denoising_strength,
                })
            else:
                custom_inputs.update({
                    'image': p.init_images,
                    'strength':p.denoising_strength,
                    'mask_image': p.mask,
                })

            #
            if refiner_ckpt != "None":
                base_output_type = "latent"
            else:
                base_output_type = "np"

            print(p.prompts)

            output = shared.sd_diffusers_model(
                    prompt=p.prompts,
                    negative_prompt=p.negative_prompts,
                    num_inference_steps=p.steps,
                    guidance_scale=p.cfg_scale,
                    generator=generator,
                    output_type=base_output_type,
                    callback = callback,
                    callback_steps = 1,
                    **custom_inputs
            )


            if refiner_ckpt != "None":
                refiner_output = shared.sd_refiner_model(
                        prompt=p.prompts,
                        negative_prompt=p.negative_prompts,
                        num_inference_steps=refiner_steps,
                        image=output.images[0][None, :],
                        output_type="np"
                )
                print("refiner steps " + str(refiner_steps))


            model_state.recompile = 0

            warmup_duration = time_stamps[1] - time_stamps[0]
            generation_rate = (p.steps - 1) / (time_stamps[-1] - time_stamps[1])
            
            if refiner_ckpt != "None":
                x_samples_ddim = refiner_output.images
            else:
                x_samples_ddim = output.images

            for i, x_sample in enumerate(x_samples_ddim):
                p.batch_index = i

                x_sample = (255. * x_sample).astype(np.uint8)

                if p.restore_faces:
                    if opts.save and not p.do_not_save_samples and opts.save_images_before_face_restoration:
                        images.save_image(Image.fromarray(x_sample), p.outpath_samples, "", p.seeds[i], p.prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-before-face-restoration")

                    devices.torch_gc()

                    x_sample = modules.face_restoration.restore_faces(x_sample)
                    devices.torch_gc()

                image = Image.fromarray(x_sample)

                if p.scripts is not None:
                    pp = scripts.PostprocessImageArgs(image)
                    p.scripts.postprocess_image(p, pp)
                    image = pp.image

                if p.color_corrections is not None and i < len(p.color_corrections):
                    if opts.save and not p.do_not_save_samples and opts.save_images_before_color_correction:
                        image_without_cc = apply_overlay(image, p.paste_to, i, p.overlay_images)
                        images.save_image(image_without_cc, p.outpath_samples, "", p.seeds[i], p.prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-before-color-correction")
                    image = apply_color_correction(p.color_corrections[i], image)

                image = apply_overlay(image, p.paste_to, i, p.overlay_images)

                if opts.samples_save and not p.do_not_save_samples:
                    images.save_image(image, p.outpath_samples, "", p.seeds[i], p.prompts[i], opts.samples_format, info=infotext(n, i), p=p)

                text = infotext(n, i)
                infotexts.append(text)
                if opts.enable_pnginfo:
                    image.info["parameters"] = text
                output_images.append(image)

                if hasattr(p, 'mask_for_overlay') and p.mask_for_overlay and any([opts.save_mask, opts.save_mask_composite, opts.return_mask, opts.return_mask_composite]):
                    image_mask = p.mask_for_overlay.convert('RGB')
                    image_mask_composite = Image.composite(image.convert('RGBA').convert('RGBa'), Image.new('RGBa', image.size), images.resize_image(2, p.mask_for_overlay, image.width, image.height).convert('L')).convert('RGBA')

                    if opts.save_mask:
                        images.save_image(image_mask, p.outpath_samples, "", p.seeds[i], p.prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-mask")

                    if opts.save_mask_composite:
                        images.save_image(image_mask_composite, p.outpath_samples, "", p.seeds[i], p.prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-mask-composite")

                    if opts.return_mask:
                        output_images.append(image_mask)

                    if opts.return_mask_composite:
                        output_images.append(image_mask_composite)

            del x_samples_ddim

            devices.torch_gc()

            state.nextjob()

        p.color_corrections = None

        index_of_first_image = 0
        unwanted_grid_because_of_img_count = len(output_images) < 2 and opts.grid_only_if_multiple
        if (opts.return_grid or opts.grid_save) and not p.do_not_save_grid and not unwanted_grid_because_of_img_count:

            grid = images.image_grid(output_images, p.batch_size)

            if opts.return_grid:
                text = infotext()
                infotexts.insert(0, text)
                if opts.enable_pnginfo:
                    grid.info["parameters"] = text
                output_images.insert(0, grid)
                index_of_first_image = 1

            if opts.grid_save:
                images.save_image(grid, p.outpath_grids, "grid", p.all_seeds[0], p.all_prompts[0], opts.grid_format, info=infotext(), short_filename=not opts.grid_extended_filename, p=p, grid=True)

    if not p.disable_extra_networks and extra_network_data:
        extra_networks.deactivate(p, p.extra_network_data)

    devices.torch_gc()

    res = Processed(
        p,
        images_list=output_images,
        seed=p.all_seeds[0],
        info=infotext(),
        comments="".join(f"{comment}\n" for comment in comments),
        subseed=p.all_subseeds[0],
        index_of_first_image=index_of_first_image,
        infotexts=infotexts,
    )

    res.info = res.info + ", Warm up time: " + str(round(warmup_duration, 2)) + " secs "

    if (generation_rate >= 1.0):
        res.info = res.info + ", Performance: " + str(round(generation_rate, 2)) + " it/s "
    else:
        res.info = res.info + ", Performance: " + str(round(1/generation_rate, 2)) + " s/it "


    if p.scripts is not None:
        p.scripts.postprocess(p, res)

    return res

def ov_process_images_inner_wrapper(p, model_config, vae_ckpt, openvino_device, override_sampler, sampler_name, enable_caching, is_xl_ckpt, refiner_ckpt, refiner_steps):


        mode = 0
        if p.__class__ == StableDiffusionProcessingTxt2Img:
            mode = 0
            processed = process_images_openvino(p, model_config, vae_ckpt, p.sampler_name, enable_caching, openvino_device, mode, is_xl_ckpt, refiner_ckpt, refiner_steps)
        else:
            if p.image_mask is None:
                mode = 1
            else:
                mode = 2
            p.init = functools.partial(init_new, p)
            processed = process_images_openvino(p, model_config, vae_ckpt, p.sampler_name, enable_caching, openvino_device, mode, is_xl_ckpt, refiner_ckpt, refiner_steps)

        """
        # mode can be 0, 1, 2 corresponding to txt2img, img2img, inpaint respectively
        if mode == 0:
            mode = 0
            processed = process_images_openvino(p, model_config, vae_ckpt, p.sampler_name, enable_caching, openvino_device, mode, is_xl_ckpt, refiner_ckpt, refiner_steps)
        else:
            if p.image_mask is None:
                mode = 1
            else:
                mode = 2
            p.init = functools.partial(init_new, p)
            processed = process_images_openvino(p, model_config, vae_ckpt, p.sampler_name, enable_caching, openvino_device, mode, is_xl_ckpt, refiner_ckpt, refiner_steps)
        """
        return processed



import modules.processing
import modules.sd_models as sd_models
import modules.sd_vae as sd_vae

def modified_process_images(p: StableDiffusionProcessing) -> Processed:
    if p.scripts is not None:
        p.scripts.before_process(p)

    stored_opts = {k: opts.data[k] for k in p.override_settings.keys()}

    try:
        # if no checkpoint override or the override checkpoint can't be found, remove override entry and load opts checkpoint
        if sd_models.checkpoint_aliases.get(p.override_settings.get('sd_model_checkpoint')) is None:
            p.override_settings.pop('sd_model_checkpoint', None)
            sd_models.reload_model_weights()

        for k, v in p.override_settings.items():
            setattr(opts, k, v)

            if k == 'sd_model_checkpoint':
                sd_models.reload_model_weights()

            if k == 'sd_vae':
                sd_vae.reload_vae_weights()

        sd_models.apply_token_merging(p.sd_model, p.get_token_merging_ratio())

        ##modified for OV 
        ov_args = {
            "model_config" : model_state.model_config,
            "vae_ckpt" : model_state.vae_ckpt,
            "openvino_device" : model_state.device,
            "override_sampler" : model_state.override_sampler,
            "sampler_name" : model_state.sampler_name,
            "enable_caching" : model_state.enable_caching,
            "is_xl_ckpt" : model_state.is_xl_ckpt,
            "refiner_ckpt" : model_state.refiner_ckpt,
            "refiner_steps" : model_state.refiner_steps,
        }
        print(ov_args)
        res = ov_process_images_inner_wrapper(p, **ov_args)

    finally:
        sd_models.apply_token_merging(p.sd_model, 0)

        # restore opts to original state
        if p.override_settings_restore_afterwards:
            for k, v in stored_opts.items():
                setattr(opts, k, v)

                if k == 'sd_vae':
                    sd_vae.reload_vae_weights()

    return res

class Script(scripts.Script):


    md = sys.modules["modules.processing"]
    md.process_images = modified_process_images
    
    core = Core()
        
    def title(self):
        return "Accelerate with OpenVINO"

    def show(self, is_img2img):
        return scripts.AlwaysVisible
            

    def ui(self, is_img2img):
        core = Core()
        def get_config_list():
            config_dir_list = os.listdir(os.path.join(os.getcwd(), "configs"))
            config_list = []
            config_list.append("None")
            for file in config_dir_list:
                if file.endswith('.yaml'):
                    config_list.append(file)
            return config_list
        def get_vae_list():
            vae_dir_list = os.listdir(os.path.join(os.getcwd(), 'models', 'VAE'))
            vae_list = []
            vae_list.append("None")
            vae_list.append("Disable-VAE-Acceleration")
            for file in vae_dir_list:
                if file.endswith('.safetensors') or file.endswith('.ckpt') or file.endswith('.pt'):
                    vae_list.append(file)
            return vae_list
        def get_refiner_list():
            refiner_dir_list = os.listdir(os.path.join(os.getcwd(), 'models', 'Stable-diffusion'))
            refiner_list = []
            refiner_list.append("None")
            for file in refiner_dir_list:
                if file.endswith('.safetensors') or file.endswith('.ckpt') or file.endswith('.pt'):
                    refiner_list.append(file)
            return refiner_list
        
        #load config file

        with gr.Accordion('OpenVINO Accelerate Extension', open=True):
            with gr.Group():
                with gr.Row():
                    with gr.Row():
                        model_config = gr.Dropdown(label="Select a local config for the model from the configs directory of the webui root", choices=get_config_list(), value="None", visible=True)
                        create_refresh_button(model_config, get_config_list, lambda: {"choices": get_config_list()},"refresh_model_config")
                    with gr.Row():
                        vae_ckpt = gr.Dropdown(label="Custom VAE", choices=get_vae_list(), value="None", visible=True)
                        create_refresh_button(vae_ckpt, get_vae_list, lambda: {"choices": get_vae_list()},"refresh_vae_directory")
            openvino_device = gr.Dropdown(label="Select a device", choices=list(core.available_devices), value="CPU")
            is_xl_ckpt= gr.Checkbox(label="Loaded checkpoint is a SDXL checkpoint", value=False)
            with gr.Row():
                    refiner_ckpt = gr.Dropdown(label="Model", choices=get_refiner_list(), value="None")
                    refiner_steps = gr.Slider(minimum=0, maximum=100, step=4, label='Refiner steps:', value=20)
            override_sampler = gr.Checkbox(label="Override the sampling selection from the main UI (Recommended as only below sampling methods have been validated for OpenVINO)", value=True)
            sampler_name = gr.Radio(label="Select a sampling method", choices=["Euler a", "Euler", "LMS", "Heun", "DPM++ 2M", "LMS Karras", "DPM++ 2M Karras", "DDIM", "PLMS"], value="Euler a")
            enable_caching = gr.Checkbox(label="Cache the compiled models on disk for faster model load in subsequent launches (Recommended)", value=True, elem_id=self.elem_id("enable_caching"))
            warmup_status = gr.Textbox(label="Device", interactive=False, visible=False)
            vae_status = gr.Textbox(label="VAE", interactive=False, visible=False)
            gr.Markdown(
            """
            ###
            ### Note:
            - First inference involves compilation of the model for best performance.
            Since compilation happens only on the first run, the first inference (or warm up inference) will be slower than subsequent inferences.
            - For accurate performance measurements, it is recommended to exclude this slower first inference, as it doesn't reflect normal running time.
            - Model is recompiled when resolution, batchsize, device, or samplers like DPM++ or Karras are changed.
            After recompiling, later inferences will reuse the newly compiled model and achieve faster running times.
            So it's normal for the first inference after a settings change to be slower, while subsequent inferences use the optimized compiled model and run faster.
            """)
        def device_change(choice):
            if (model_state.device == choice):
                return gr.update(value="Device selected is " + choice, visible=True)
            else:
                model_state.device = choice
                model_state.recompile = 1
                os.environ['OPENVINO_TORCH_BACKEND_DEVICE'] = choice
                return gr.update(value="Device changed to " + choice + ". Model will be re-compiled", visible=True)
        openvino_device.change(device_change, openvino_device, warmup_status)
        def vae_change(choice):
            if (model_state.custom_vae == choice):
                return gr.update(value="Custom_VAE selected is " + choice, visible=True)
            else:
                model_state.custom_vae = choice
                model_state.recompile = 1
                return gr.update(value="Custom VAE changed to " + choice + ". Model will be re-compiled", visible=True)
        vae_ckpt.change(vae_change, vae_ckpt, vae_status)


        def model_state_update():
            model_state.model_config = model_config.value
            model_state.vae_ckpt = vae_ckpt.value
            model_state.override_sampler = override_sampler.value
            model_state.sampler_name = sampler_name.value
            model_state.enable_caching = enable_caching.value
            model_state.is_xl_ckpt = is_xl_ckpt.value
            model_state.refiner_ckpt = refiner_ckpt.value
            model_state.refiner_steps = refiner_steps.value
        model_state_update()
        
        
        return [model_config, vae_ckpt, openvino_device, override_sampler, sampler_name, enable_caching, is_xl_ckpt, refiner_ckpt, refiner_steps]

        

