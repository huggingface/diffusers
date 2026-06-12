try:
    from diffusers.utils.triton_compat import patch_triton_autotuner_prune_configs

    patch_triton_autotuner_prune_configs()
except Exception:
    # Import-time compatibility patches should never prevent interpreter startup.
    pass
