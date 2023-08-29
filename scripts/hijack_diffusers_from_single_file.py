from pathlib import Path

from diffusers.utils import (
    DIFFUSERS_CACHE,
    HF_HUB_OFFLINE,

)

## fix load from single file in diffusers to use local file, this would not be needed after future diffusers update
## https://github.com/huggingface/diffusers/issues/4561

def modified_from_single_file(cls, pretrained_model_link_or_path, **kwargs):
    # import here to avoid circular dependency
    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt

    original_config_file = kwargs.pop("original_config_file", None)
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
    vae = kwargs.pop("vae", None)
    controlnet = kwargs.pop("controlnet", None)
    tokenizer = kwargs.pop("tokenizer", None)

    torch_dtype = kwargs.pop("torch_dtype", None)

    use_safetensors = kwargs.pop("use_safetensors", None)

    pipeline_name = cls.__name__
    file_extension = pretrained_model_link_or_path.rsplit(".", 1)[-1]
    from_safetensors = file_extension == "safetensors"

    if from_safetensors and use_safetensors is False:
        raise ValueError("Make sure to install `safetensors` with `pip install safetensors`.")

    # TODO: For now we only support stable diffusion
    stable_unclip = None
    model_type = None

    if pipeline_name in [
        "StableDiffusionControlNetPipeline",
        "StableDiffusionControlNetImg2ImgPipeline",
        "StableDiffusionControlNetInpaintPipeline",
    ]:
        from diffusers.models.controlnet import ControlNetModel
        from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

        # Model type will be inferred from the checkpoint.
        if not isinstance(controlnet, (ControlNetModel, MultiControlNetModel)):
            raise ValueError("ControlNet needs to be passed if loading from ControlNet pipeline.")
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
    has_valid_url_prefix = False
    valid_url_prefixes = ["https://huggingface.co/", "huggingface.co/", "hf.co/", "https://hf.co/"]
    for prefix in valid_url_prefixes:
        if pretrained_model_link_or_path.startswith(prefix):
            pretrained_model_link_or_path = pretrained_model_link_or_path[len(prefix) :]
            has_valid_url_prefix = True

    # Code based on diffusers.pipelines.pipeline_utils.DiffusionPipeline.from_pretrained
    ckpt_path = Path(pretrained_model_link_or_path)
    if not ckpt_path.is_file():
        if not has_valid_url_prefix:
            raise ValueError(
                f"The provided path is either not a file or a valid huggingface URL was not provided. Valid URLs begin with {', '.join(valid_url_prefixes)}"
            )

        # get repo_id and (potentially nested) file path of ckpt in repo
        repo_id = "/".join(ckpt_path.parts[:2])
        file_path = "/".join(ckpt_path.parts[2:])

        if file_path.startswith("blob/"):
            file_path = file_path[len("blob/") :]

        if file_path.startswith("main/"):
            file_path = file_path[len("main/") :]

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
        original_config_file = original_config_file,
        pipeline_class=cls,
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
        vae=vae,
        tokenizer=tokenizer,
    )

    if torch_dtype is not None:
        pipe.to(torch_dtype=torch_dtype)

    return pipe
