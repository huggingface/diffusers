import uvicorn
import logging
import gc
import psutil
import os
import threading
import time

def setup_logging():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger('uvicorn')

logger = setup_logging()

def memory_cleanup(interval=30):
    while True:
        try:
            gc.collect()
            
            process = psutil.Process(os.getpid())
            mem = process.memory_info().rss / 1024 / 1024
            logger.info(f"Memoria en uso: {mem:.2f} MB")
            
            time.sleep(interval)
        except Exception as e:
            logger.error(f"Error en limpieza de memoria: {str(e)}")
            time.sleep(interval)

def run_uvicorn_server(
    app, 
    host='0.0.0.0', 
    port=8500, 
    workers=5, 
    cleanup_interval=30, 
    channel_timeout=900,
    headers=[               
        ("server", "DiffusersServer")
    ],
    enable_memory_monitor=True
):
    gc.enable()
    gc.set_threshold(700, 10, 5)
    
    if enable_memory_monitor:
        cleanup_thread = threading.Thread(
            target=memory_cleanup, 
            args=(cleanup_interval,), 
            daemon=True
        )
        cleanup_thread.start()
        logger.info("Memory monitor activated")
    
    logger.info(f"Starting Uvicorn server in {host}:{port}...")

    config = uvicorn.Config(
        app=app,
        host=host,
        workers=workers,
        port=port,
        timeout_keep_alive=channel_timeout,
        headers=headers
    )

    server = uvicorn.Server(config)

    return server.serve()