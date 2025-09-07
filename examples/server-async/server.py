# DiffusersServerApp already handles the inference server and everything else internally, you 
# just need to do these basic configurations and run the script with "python server.py" 
# and you already get access to the inference APIs.
from DiffusersServer import DiffusersServerApp

app = DiffusersServerApp(
    model='stabilityai/stable-diffusion-3.5-medium',
    type_model='t2im',
    threads=3,
    enable_memory_monitor=True
)