import glob
import subprocess
from typing import List


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

        if file != "benchmark_text_to_image.py":
            command = f"python {file}"
            run_command(command.split())

            command += " --run_compile"
            run_command(command.split())

        if file == "benchmark_text_to_image.py":
            for ckpt in [
                "runwayml/stable-diffusion-v1-5",
                "segmind/SSD-1B",
                "stabilityai/stable-diffusion-xl-base-1.0",
                "kandinsky-community/kandinsky-2-2-decoder",
                "warp-ai/wuerstchen",
            ]:
                command = f"python {file} --ckpt {ckpt}"
                run_command(command.split())

                command += " --run_compile"
                run_command(command.split())

        elif file in ["benchmark_sd_img.py", "benchmark_sd_inpainting.py"]:
            sdxl_ckpt = (
                "stabilityai/stable-diffusion-xl-refiner-1.0"
                if "inpainting" not in file
                else "stabilityai/stable-diffusion-xl-base-1.0"
            )
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
