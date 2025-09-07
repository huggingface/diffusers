# from https://github.com/F4k3r22/DiffusersServer/blob/main/DiffusersServer/uvicorn_diffu.py

import uvicorn
import logging
import gc
import psutil
import os
import threading
import time
import string

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
    """
    Ejecuta un servidor de FastAPI utilizando Uvicorn con monitoreo de memoria opcional
    
    Args:
        app: Aplicación FastAPI
        host (str): Host donde se servirá la aplicación
        port (int): Puerto para el servidor
        workers (int): Número de hilos para Uvicorn
        cleanup_interval (int): Intervalo de limpieza para Uvicorn
        channel_timeout (int): Tiempo de espera máximo para canales
        server_header (bool): Activar el identificador / Header del servidor
        headers (str): Identificador del servidor / Header del servidor
        enable_memory_monitor (bool): Si se debe activar el monitoreo de memoria
        
    Returns:
        El resultado de serve() (aunque normalmente no retorna)
    """
    gc.enable()
    gc.set_threshold(700, 10, 5)
    
    if enable_memory_monitor:
        cleanup_thread = threading.Thread(
            target=memory_cleanup, 
            args=(cleanup_interval,), 
            daemon=True
        )
        cleanup_thread.start()
        logger.info("Monitor de memoria activado")
    
    logger.info(f"Iniciando servidor Uvicorn en {host}:{port}...")

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