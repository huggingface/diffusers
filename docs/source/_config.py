# docstyle-ignore
INSTALL_CONTENT = """
# Diffusers installation
! pip install diffusers transformers datasets accelerate
# To install from source instead of the last release, comment the command above and uncomment the following one.
# ! pip install git+https://github.com/huggingface/diffusers.git
"""

notebook_first_cells = [{"type": "code", "content": INSTALL_CONTENT}]
