import runpy
import sys
sys.argv = [
    "profile_diffusion_gemma_quantized.py",
    "--gen_length", "32",
    "--num_inference_steps", "4",
    "--cache_implementation", "dynamic",
]
runpy.run_path("/content/profile_diffusion_gemma_quantized.py", run_name="__main__")
