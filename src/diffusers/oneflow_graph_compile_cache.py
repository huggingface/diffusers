from collections import deque
from timeit import default_timer as timer
from .utils import logging

logger = logging.get_logger(__name__)


class OneFlowGraph(object):
    def __init__(self, graph_class, *args, **kwargs):
        self.graph_ = graph_class(*args, **kwargs)
        self.is_compiled_ = False

    @property
    def is_compiled(self):
        return self.is_compiled_

    def compile(self, *args, **kwargs):
        if self.is_compiled_:
            return

        global_class_name = self.graph_.__class__.__name__
        logger.info(
            f"[oneflow] compiling {global_class_name} beforehand to make sure the progress bar is more accurate",
        )
        compilation_start = timer()
        compilation_time = 0
        self.graph_._compile(*args, **kwargs)
        compilation_time = timer() - compilation_start
        logger.info(f"[oneflow] [elapsed(s)] [{global_class_name} compilation] {compilation_time:.3f}")

        self.is_compiled_ = True

    def __call__(self, *args, **kwargs):
        if not self.is_compiled_:
            self.compile(*args, **kwargs)

        return self.graph_(*args, **kwargs)


class LRUCache(object):
    def __init__(self, cache_size):
        self.cache_size = cache_size
        self.queue = deque()
        self.hash_map = dict()

    def is_queue_full(self):
        return len(self.queue) == self.cache_size

    def pop(self):
        pop_key = self.queue.pop()
        value = self.hash_map.pop(pop_key)
        del value
        return pop_key

    def set(self, key, value):
        if key in self.hash_map:
            return None

        pop_key = None
        while self.is_queue_full():
            pop_key = self.pop()

        self.queue.appendleft(key)
        self.hash_map[key] = value
        return pop_key if pop_key is not None else key

    def get(self, key):
        if key in self.hash_map:
            self.queue.remove(key)
            self.queue.appendleft(key)
            return self.hash_map[key]

        return None


class OneFlowGraphCompileCache(object):
    def __init__(self, cache_size=1):
        self.cache_size_ = cache_size
        self.cache_bucket_ = dict()

    def set_cache_size(self, cache_size):
        self.cache_size_ = cache_size

        for cache in self.cache_bucket_.values():
            cache.cache_size = cache_size

    def get_graph(self, graph_class, cache_key, *args, **kwargs):
        graph_class_name = graph_class.__name__
        if graph_class_name not in self.cache_bucket_:
            self.cache_bucket_[graph_class_name] = LRUCache(self.cache_size_)

        compile_cache = self.cache_bucket_[graph_class_name]

        graph = compile_cache.get(cache_key)
        if graph is None:
            graph = OneFlowGraph(graph_class, *args, **kwargs)
            ret = compile_cache.set(cache_key, graph)
            assert ret is not None

            if ret != cache_key:
                logger.info(
                    f"[oneflow] a {graph_class_name} with cache key {ret} "
                    "is deleted from cache according to the LRU policy",
                )
                if self.cache_size_ == 1:
                    logger.info("[oneflow] cache size can be changed by `set_cache_size`")

            logger.info(
                f"[oneflow] a {graph_class_name} with cache key {cache_key} is appending to "
                f"cache (cache_size={compile_cache.cache_size})",
            )

        return graph
