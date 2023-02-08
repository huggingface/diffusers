import os


def Root():
    models_path = "models"  # @param {type:"string"}
    configs_path = "configs"  # @param {type:"string"}
    output_path = "output"  # @param {type:"string"}
    mount_google_drive = False  # @param {type:"boolean"}
    models_path_gdrive = "/content/drive/MyDrive/AI/models"  # @param {type:"string"}
    output_path_gdrive = "/content/drive/MyDrive/AI/StableDiffusion"  # @param {type:"string"}

    # @markdown **Model Setup**
    model_config = "v1-inference.yaml"  # @param ["custom","v1-inference.yaml"]
    model_checkpoint = "v1-5-pruned-emaonly.ckpt"  # @param ["custom","v1-5-pruned.ckpt","v1-5-pruned-emaonly.ckpt","sd-v1-4-full-ema.ckpt","sd-v1-4.ckpt","sd-v1-3-full-ema.ckpt","sd-v1-3.ckpt","sd-v1-2-full-ema.ckpt","sd-v1-2.ckpt","sd-v1-1-full-ema.ckpt","sd-v1-1.ckpt", "robo-diffusion-v1.ckpt","wd-v1-3-float16.ckpt"]
    custom_config_path = ""  # @param {type:"string"}
    custom_checkpoint_path = ""  # @param {type:"string"}
    half_precision = True
    return locals()


def DeforumAnimArgs():
    animation_mode = "3D"  # @param ['None', '2D', '3D', 'Video Input', 'Interpolation'] {type:'string'}
    max_frames = 200  # @param {type:"number"}
    border = "wrap"  # @param ['wrap', 'replicate'] {type:'string'}

    # @markdown ####**Motion Parameters:**
    angle = "0:(0)"  # @param {type:"string"}
    zoom = "0:(1.04)"  # @param {type:"string"}
    translation_x = "0:(0)"  # @param {type:"string"}
    translation_y = "0:(0)"  # @param {type:"string"}
    translation_z = "0:(0)"  # @param {type:"string"}
    rotation_3d_x = "0:(0)"  # @param {type:"string"}
    rotation_3d_y = "0:(0)"  # @param {type:"string"}
    rotation_3d_z = "0:(0)"  # @param {type:"string"}
    flip_2d_perspective = False  # @param {type:"boolean"}
    perspective_flip_theta = "0:(0)"  # @param {type:"string"}
    perspective_flip_phi = "0:(t%15)"  # @param {type:"string"}
    perspective_flip_gamma = "0:(0)"  # @param {type:"string"}
    perspective_flip_fv = "0:(0)"  # @param {type:"string"}
    noise_schedule = "0:(0.02)"  # @param {type:"string"}
    strength_schedule = "0:(0.65)"  # @param {type:"string"}
    contrast_schedule = "0:(1.0)"  # @param {type:"string"}

    # @markdown ####**Coherence:**
    color_coherence = "Match Frame 0 LAB"  # @param ['None', 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB'] {type:'string'}
    diffusion_cadence = "3"  # @param ['1','2','3','4','5','6','7','8'] {type:'string'}

    # @markdown #### 3D Depth Warping
    use_depth_warping = True  # @param {type:"boolean"}
    midas_weight = 0.3  # @param {type:"number"}
    near_plane = 200
    far_plane = 10000
    fov = 40  # @param {type:"number"}
    padding_mode = "border"  # @param ['border', 'reflection', 'zeros'] {type:'string'}
    sampling_mode = "bicubic"  # @param ['bicubic', 'bilinear', 'nearest'] {type:'string'}
    save_depth_maps = False  # @param {type:"boolean"}

    # @markdown ####**Video Input:**
    video_init_path = "./input/video_in.mp4"  # @param {type:"string"}
    extract_nth_frame = 1  # @param {type:"number"}
    overwrite_extracted_frames = True  # @param {type:"boolean"}
    use_mask_video = False  # @param {type:"boolean"}
    video_mask_path = ""  # @param {type:"string"}

    # @markdown ####**Interpolation:**
    interpolate_key_frames = False  # @param {type:"boolean"}
    interpolate_x_frames = 4  # @param {type:"number"}

    # @markdown ####**Resume Animation:**
    resume_from_timestring = False  # @param {type:"boolean"}
    resume_timestring = "20220829210106"  # @param {type:"string"}
    return locals()


def DeforumArgs():
    # @markdown **Image Settings**
    W = 512  # @param
    H = 512  # @param
    W, H = map(lambda x: x - x % 64, (W, H))  # resize to integer multiple of 64

    # @markdown **Sampling Settings**
    seed = 2022  # @param
    sampler = "klms"  # @param ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral","plms", "ddim", "dpm_fast", "dpm_adaptive", "dpmpp_2s_a", "dpmpp_2m"]
    steps = 50  # @param
    scale = 7  # @param
    ddim_eta = 0.0  # @param
    dynamic_threshold = None
    static_threshold = None

    # @markdown **Save & Display Settings**
    save_samples = True  # @param {type:"boolean"}
    save_settings = True  # @param {type:"boolean"}
    display_samples = True  # @param {type:"boolean"}
    save_sample_per_step = False  # @param {type:"boolean"}
    show_sample_per_step = False  # @param {type:"boolean"}

    # @markdown **Prompt Settings**
    prompt_weighting = True  # @param {type:"boolean"}
    normalize_prompt_weights = True  # @param {type:"boolean"}
    log_weighted_subprompts = False  # @param {type:"boolean"}

    # @markdown **Batch Settings**
    n_batch = 1  # @param
    batch_name = "data"  # @param {type:"string"}
    filename_format = "{timestring}_{index}_{prompt}.png"  # @param ["{timestring}_{index}_{seed}.png","{timestring}_{index}_{prompt}.png"]
    seed_behavior = "iter"  # @param ["iter","fixed","random"]
    make_grid = False  # @param {type:"boolean"}
    grid_rows = 2  # @param
    outdir = "./outputs"

    # @markdown **Init Settings**
    use_init = False  # @param {type:"boolean"}
    strength = 0.0  # @param {type:"number"}
    strength_0_no_init = True  # Set the strength to 0 automatically when no init image is used
    init_image = ""  # @param {type:"string"}
    # Whiter areas of the mask are areas that change more
    use_mask = False  # @param {type:"boolean"}
    use_alpha_as_mask = False  # use the alpha channel of the init image as the mask
    mask_file = ""  # @param {type:"string"}
    invert_mask = False  # @param {type:"boolean"}
    # Adjust mask image, 1.0 is no adjustment. Should be positive numbers.
    mask_brightness_adjust = 1.0  # @param {type:"number"}
    mask_contrast_adjust = 1.0  # @param {type:"number"}

    # Overlay the masked image at the end of the generation so it does not get degraded by encoding and decoding
    overlay_mask = True  # {type:"boolean"}
    # Blur edges of final overlay mask, if used. Minimum = 0 (no blur)
    mask_overlay_blur = 5  # {type:"number"}

    # @markdown **Exposure/Contrast Conditional Settings**
    mean_scale = 0  # @param {type:"number"}
    var_scale = 0  # @param {type:"number"}
    exposure_scale = 0  # @param {type:"number"}
    exposure_target = 0.5  # @param {type:"number"}

    # @markdown **Color Match Conditional Settings**
    colormatch_scale = 0  # @param {type:"number"}
    colormatch_image = ""  # @param {type:"string"}
    colormatch_n_colors = 4  # @param {type:"number"}
    ignore_sat_weight = 0  # @param {type:"number"}

    # @markdown **CLIP\Aesthetics Conditional Settings**
    clip_name = "ViT-L/14"  # @param ['ViT-L/14', 'ViT-L/14@336px', 'ViT-B/16', 'ViT-B/32']
    clip_scale = 0  # @param {type:"number"}
    aesthetics_scale = 0  # @param {type:"number"}
    cutn = 1  # @param {type:"number"}
    cut_pow = 0.0001  # @param {type:"number"}

    # @markdown **Other Conditional Settings**
    init_mse_scale = 0  # @param {type:"number"}
    init_mse_image = ""  # @param {type:"string"}

    blue_scale = 1  # @param {type:"number"}

    # @markdown **Conditional Gradient Settings**
    gradient_wrt = "x0_pred"  # @param ["x", "x0_pred"]
    gradient_add_to = "both"  # @param ["cond", "uncond", "both"]
    decode_method = "linear"  # @param ["autoencoder","linear"]
    grad_threshold_type = "dynamic"  # @param ["dynamic", "static", "mean", "schedule"]
    clamp_grad_threshold = 0.2  # @param {type:"number"}
    clamp_start = 0.2  # @param
    clamp_stop = 0.01  # @param
    grad_inject_timing = list(range(1, 10))  # @param

    # @markdown **Speed vs VRAM Settings**
    cond_uncond_sync = True  # @param {type:"boolean"}

    n_samples = 1  # doesnt do anything
    precision = "autocast"
    C = 4
    f = 8

    prompt = ""
    timestring = ""
    init_latent = None
    init_sample = None
    init_sample_raw = None
    mask_sample = None
    init_c = None

    return locals()
