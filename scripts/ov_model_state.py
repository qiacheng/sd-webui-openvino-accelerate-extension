class ModelState:
    def __init__(self):
        self.recompile = 1
        self.device = "CPU"
        self.height = 512
        self.width = 512
        self.batch_size = 1
        self.mode = 0
        self.partition_id = 0
        self.model_hash = ""
        self.model_config = ""
        self.vae_ckpt = ""
        self.override_sampler = True
        self.sampler_name = ""
        self.enable_caching = True
        self.is_xl_ckpt = False
        self.refiner_ckpt = ""
        self.refiner_steps = 0

model_state = ModelState()
