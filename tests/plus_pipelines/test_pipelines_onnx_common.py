from diffusers.utils.testing_utils import require_onnxruntime


@require_onnxruntime
class OnnxPipelineTesterMixin:
    """
    This mixin is designed to be used with unittest.TestCase classes.
    It provides a set of common tests for each ONNXRuntime pipeline, e.g. saving and loading the pipeline,
    equivalence of dict and tuple outputs, etc.
    """

    pass
