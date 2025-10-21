class ThreadSafeTokenizerWrapper:
    def __init__(self, tokenizer, lock):
        self._tokenizer = tokenizer
        self._lock = lock

        self._thread_safe_methods = {
            '__call__', 'encode', 'decode', 'tokenize', 
            'encode_plus', 'batch_encode_plus', 'batch_decode'
        }
    
    def __getattr__(self, name):
        attr = getattr(self._tokenizer, name)
        
        if name in self._thread_safe_methods and callable(attr):
            def wrapped_method(*args, **kwargs):
                with self._lock:
                    return attr(*args, **kwargs)
            return wrapped_method
        
        return attr

    def __call__(self, *args, **kwargs):
        with self._lock:
            return self._tokenizer(*args, **kwargs)
    
    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            setattr(self._tokenizer, name, value)
    
    def __dir__(self):
        return dir(self._tokenizer)


class ThreadSafeVAEWrapper:
    def __init__(self, vae, lock):
        self._vae = vae
        self._lock = lock

    def __getattr__(self, name):
        attr = getattr(self._vae, name)
        if name in {"decode", "encode", "forward"} and callable(attr):
            def wrapped(*args, **kwargs):
                with self._lock:
                    return attr(*args, **kwargs)
            return wrapped
        return attr

    def __setattr__(self, name, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            setattr(self._vae, name, value)

class ThreadSafeImageProcessorWrapper:
    def __init__(self, proc, lock):
        self._proc = proc
        self._lock = lock

    def __getattr__(self, name):
        attr = getattr(self._proc, name)
        if name in {"postprocess", "preprocess"} and callable(attr):
            def wrapped(*args, **kwargs):
                with self._lock:
                    return attr(*args, **kwargs)
            return wrapped
        return attr

    def __setattr__(self, name, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            setattr(self._proc, name, value)