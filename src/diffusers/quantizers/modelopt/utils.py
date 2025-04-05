from ...utils import logging


logger = logging.get_logger(__name__)


def _replace_with_modelopt_layers(model, quantization_config):
    # ModelOpt imports diffusers internally. These are placed here to avoid circular imports
    import modelopt.torch.opt as mto
    import modelopt.torch.quantization as mtq

    model = mto.apply_mode(model, mode=[("quantize", quantization_config)], registry=mtq.mode.QuantizeModeRegistry)
    return model
