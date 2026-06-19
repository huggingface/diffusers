    # HACK: Use Gloo backend for all_gather to avoid H2D and D2H overhead
    comm_backends = str(dist.get_backend(group=group))
    # NOTE: e.g., dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    if "cpu" in comm_backends:
        gather_device = "cpu"
    elif hasattr(torch, "accelerator"):
        gather_device = torch.accelerator.current_accelerator()
    else:
        # `torch.accelerator` is only available in PyTorch >= 2.6
        gather_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gathered_sizes = [torch.empty((1,), device=gather_device, dtype=torch.int64) for _ in range(world_size)]
    dist.all_gather(
        gathered_sizes,
