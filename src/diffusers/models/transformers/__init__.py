import sys
import types

from ...utils import deprecate, is_torch_available


class _DeprecatedModuleAlias(types.ModuleType):
    """Backwards-compat alias for a transformer module that's been moved to a subpackage.

    Lives only in ``sys.modules`` — no stub file. Emits a one-time ``deprecate`` warning on first attribute access,
    then forwards every attribute lookup to the new target module. Used when a flat ``transformer_<name>.py`` is split
    into a ``<name>/`` subpackage and we want the old import path to keep working for a release cycle.
    """

    def __init__(self, old_dotted_path: str, target: types.ModuleType):
        super().__init__(target.__name__, target.__doc__)
        # Bypass __getattr__ when writing internals.
        self.__dict__["_target"] = target
        self.__dict__["_old_path"] = old_dotted_path
        self.__dict__["_warned"] = False

    def __getattr__(self, name):
        if not self.__dict__["_warned"]:
            self.__dict__["_warned"] = True
            old = self.__dict__["_old_path"]
            new = self.__dict__["_target"].__name__
            deprecate(
                old,
                "1.0.0",
                f"Importing from `{old}` is deprecated. Import from `{new}` instead.",
                standard_warn=True,
                stacklevel=3,
            )
        return getattr(self.__dict__["_target"], name)


def _register_legacy_module_alias(old_name: str, new_name: str) -> None:
    """Register ``old_name`` as a deprecated alias for the already-loaded ``new_name`` submodule.

    Both names are relative to ``diffusers.models.transformers``. The new submodule must already be in ``sys.modules``
    (loaded by a prior ``from .<new_name> import ...`` in this file).
    """
    old_dotted = f"{__name__}.{old_name}"
    target = sys.modules[f"{__name__}.{new_name}"]
    sys.modules[old_dotted] = _DeprecatedModuleAlias(old_dotted, target)


if is_torch_available():
    # Load flux first and install the legacy alias before any other transformer module imports,
    # since some of them still pull from `transformer_flux` during their own load.
    from .flux import FluxTransformer2DModel

    _register_legacy_module_alias("transformer_flux", "flux")

    from .ace_step_transformer import AceStepTransformer1DModel
    from .auraflow_transformer_2d import AuraFlowTransformer2DModel
    from .cogvideox_transformer_3d import CogVideoXTransformer3DModel
    from .consisid_transformer_3d import ConsisIDTransformer3DModel
    from .dit_transformer_2d import DiTTransformer2DModel
    from .dual_transformer_2d import DualTransformer2DModel
    from .hunyuan_transformer_2d import HunyuanDiT2DModel
    from .latte_transformer_3d import LatteTransformer3DModel
    from .lumina_nextdit2d import LuminaNextDiT2DModel
    from .pixart_transformer_2d import PixArtTransformer2DModel
    from .prior_transformer import PriorTransformer
    from .sana_transformer import SanaTransformer2DModel
    from .stable_audio_transformer import StableAudioDiTModel
    from .t5_film_transformer import T5FilmDecoder
    from .transformer_2d import Transformer2DModel
    from .transformer_2d_dreamlite import DreamLiteTransformer2DModel
    from .transformer_allegro import AllegroTransformer3DModel
    from .transformer_anyflow import AnyFlowTransformer3DModel
    from .transformer_anyflow_far import AnyFlowFARTransformer3DModel
    from .transformer_bria import BriaTransformer2DModel
    from .transformer_bria_fibo import BriaFiboTransformer2DModel
    from .transformer_chroma import ChromaTransformer2DModel
    from .transformer_chronoedit import ChronoEditTransformer3DModel
    from .transformer_cogview3plus import CogView3PlusTransformer2DModel
    from .transformer_cogview4 import CogView4Transformer2DModel
    from .transformer_cosmos import CosmosTransformer3DModel
    from .transformer_cosmos3 import Cosmos3OmniTransformer
    from .transformer_easyanimate import EasyAnimateTransformer3DModel
    from .transformer_ernie_image import ErnieImageTransformer2DModel
    from .transformer_flux2 import Flux2Transformer2DModel
    from .transformer_glm_image import GlmImageTransformer2DModel
    from .transformer_helios import HeliosTransformer3DModel
    from .transformer_hidream_image import HiDreamImageTransformer2DModel
    from .transformer_hunyuan_video import HunyuanVideoTransformer3DModel
    from .transformer_hunyuan_video15 import HunyuanVideo15Transformer3DModel
    from .transformer_hunyuan_video_framepack import HunyuanVideoFramepackTransformer3DModel
    from .transformer_hunyuanimage import HunyuanImageTransformer2DModel
    from .transformer_ideogram4 import Ideogram4Transformer2DModel
    from .transformer_joyimage import JoyImageEditTransformer3DModel
    from .transformer_kandinsky import Kandinsky5Transformer3DModel
    from .transformer_krea2 import Krea2Transformer2DModel
    from .transformer_longcat_audio_dit import LongCatAudioDiTTransformer
    from .transformer_longcat_image import LongCatImageTransformer2DModel
    from .transformer_ltx import LTXVideoTransformer3DModel
    from .transformer_ltx2 import LTX2VideoTransformer3DModel
    from .transformer_lumina2 import Lumina2Transformer2DModel
    from .transformer_mochi import MochiTransformer3DModel
    from .transformer_motif_video import MotifVideoTransformer3DModel
    from .transformer_nucleusmoe_image import NucleusMoEImageTransformer2DModel
    from .transformer_omnigen import OmniGenTransformer2DModel
    from .transformer_ovis_image import OvisImageTransformer2DModel
    from .transformer_prx import PRXTransformer2DModel
    from .transformer_qwenimage import QwenImageTransformer2DModel
    from .transformer_sana_video import SanaVideoTransformer3DModel
    from .transformer_sd3 import SD3Transformer2DModel
    from .transformer_skyreels_v2 import SkyReelsV2Transformer3DModel
    from .transformer_temporal import TransformerTemporalModel
    from .transformer_wan import WanTransformer3DModel
    from .transformer_wan_animate import WanAnimateTransformer3DModel
    from .transformer_wan_vace import WanVACETransformer3DModel
    from .transformer_z_image import ZImageTransformer2DModel
