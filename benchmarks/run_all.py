import glob
import subprocess
import sys
from typing import List


sys.path.append(".")
from benchmark_text_to_image import ALL_T2I_CKPTS  # noqa: E402


PATTERN = "benchmark_*.py"


class SubprocessCallException(Exception):
    pass


# Taken from `test_examples_utils.py`
def run_command(command: List[str], return_stdout=False):
    """
    Runs `command` with `subprocess.check_output` and will potentially return the `stdout`. Will also properly capture
    if an error occurred while running `command`
    """
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT)
        if return_stdout:
            if hasattr(output, "decode"):
                output = output.decode("utf-8")
            return output
    except subprocess.CalledProcessError as e:
        raise SubprocessCallException(
            f"Command `{' '.join(command)}` failed with the following error:\n\n{e.output.decode()}"
        ) from e


def main():
    python_files = glob.glob(PATTERN)

    for file in python_files:
        print(f"****** Running file: {file} ******")

        # Run with canonical settings.
        if file != "benchmark_text_to_image.py" and file != "benchmark_ip_adapters.py":
            command = f"python {file}"
            run_command(command.split())

            command += " --run_compile"
            run_command(command.split())

    # Run variants.
    for file in python_files:
        # See: https://github.com/pytorch/pytorch/issues/129637
        if file == "benchmark_ip_adapters.py":
            continue

        if file == "benchmark_text_to_image.py":
            for ckpt in ALL_T2I_CKPTS:
                command = f"python {file} --ckpt {ckpt}"

                if "turbo" in ckpt:
                    command += " --num_inference_steps 1"

                run_command(command.split())

                command += " --run_compile"
                run_command(command.split())

        elif file == "benchmark_sd_img.py":
            for ckpt in ["stabilityai/stable-diffusion-xl-refiner-1.0", "stabilityai/sdxl-turbo"]:
                command = f"python {file} --ckpt {ckpt}"

                if ckpt == "stabilityai/sdxl-turbo":
                    command += " --num_inference_steps 2"

                run_command(command.split())
                command += " --run_compile"
                run_command(command.split())

        elif file in ["benchmark_sd_inpainting.py", "benchmark_ip_adapters.py"]:
            sdxl_ckpt = "stabilityai/stable-diffusion-xl-base-1.0"
            command = f"python {file} --ckpt {sdxl_ckpt}"
            run_command(command.split())

            command += " --run_compile"
            run_command(command.split())

        elif file in ["benchmark_controlnet.py", "benchmark_t2i_adapter.py"]:
            sdxl_ckpt = (
                "diffusers/controlnet-canny-sdxl-1.0"
                if "controlnet" in file
                else "TencentARC/t2i-adapter-canny-sdxl-1.0"
            )
            command = f"python {file} --ckpt {sdxl_ckpt}"
            run_command(command.split())

            command += " --run_compile"
            run_command(command.split())


if __name__ == "__main__":
    main()
