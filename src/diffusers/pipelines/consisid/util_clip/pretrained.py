import hashlib
import os
import urllib
import warnings
from functools import partial
from typing import Dict, Union

from tqdm import tqdm

try:
    from huggingface_hub import hf_hub_download
    _has_hf_hub = True
except ImportError:
    hf_hub_download = None
    _has_hf_hub = False


def _pcfg(url='', hf_hub='', filename='', mean=None, std=None):
    return dict(
        url=url,
        hf_hub=hf_hub,
        mean=mean,
        std=std,
    )

_VITB32 = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"),
    laion400m_e31=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e31-d867053b.pt"),
    laion400m_e32=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e32-46683a32.pt"),
    laion2b_e16=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-laion2b_e16-af8dbd0c.pth"),
    laion2b_s34b_b79k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-laion2B-s34B-b79K/')
)

_VITB32_quickgelu = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"),
    laion400m_e31=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e31-d867053b.pt"),
    laion400m_e32=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e32-46683a32.pt"),
)

_VITB16 = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"),
    laion400m_e31=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_16-laion400m_e31-00efa78f.pt"),
    laion400m_e32=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_16-laion400m_e32-55e67d44.pt"),
    laion2b_s34b_b88k=_pcfg(hf_hub='laion/CLIP-ViT-B-16-laion2B-s34B-b88K/'),
)

_EVAB16 = dict(
    eva=_pcfg(hf_hub='QuanSun/EVA-CLIP/EVA02_B_psz14to16.pt'),
    eva02=_pcfg(hf_hub='QuanSun/EVA-CLIP/EVA02_B_psz14to16.pt'),
    eva_clip=_pcfg(hf_hub='QuanSun/EVA-CLIP/EVA02_CLIP_B_psz16_s8B.pt'),
    eva02_clip=_pcfg(hf_hub='QuanSun/EVA-CLIP/EVA02_CLIP_B_psz16_s8B.pt'),
)

_VITB16_PLUS_240 = dict(
    laion400m_e31=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_16_plus_240-laion400m_e31-8fb26589.pt"),
    laion400m_e32=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_16_plus_240-laion400m_e32-699c4b84.pt"),
)

_VITL14 = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"),
    laion400m_e31=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_l_14-laion400m_e31-69988bb6.pt"),
    laion400m_e32=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_l_14-laion400m_e32-3d133497.pt"),
    laion2b_s32b_b82k=_pcfg(
        hf_hub='laion/CLIP-ViT-L-14-laion2B-s32B-b82K/',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
)

_EVAL14 = dict(
    eva=_pcfg(hf_hub='QuanSun/EVA-CLIP/EVA02_L_psz14.pt'),
    eva02=_pcfg(hf_hub='QuanSun/EVA-CLIP/EVA02_L_psz14.pt'),
    eva_clip=_pcfg(hf_hub='QuanSun/EVA-CLIP/EVA02_CLIP_L_psz14_s4B.pt'),
    eva02_clip=_pcfg(hf_hub='QuanSun/EVA-CLIP/EVA02_CLIP_L_psz14_s4B.pt'),
)

_VITL14_336 = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt"),
)

_EVAL14_336 = dict(
    eva_clip=_pcfg(hf_hub='QuanSun/EVA-CLIP/EVA02_CLIP_L_336_psz14_s6B.pt'),
    eva02_clip=_pcfg(hf_hub='QuanSun/EVA-CLIP/EVA02_CLIP_L_336_psz14_s6B.pt'),
    eva_clip_224to336=_pcfg(hf_hub='QuanSun/EVA-CLIP/EVA02_CLIP_L_psz14_224to336.pt'),
    eva02_clip_224to336=_pcfg(hf_hub='QuanSun/EVA-CLIP/EVA02_CLIP_L_psz14_224to336.pt'),
)

_VITH14 = dict(
    laion2b_s32b_b79k=_pcfg(hf_hub='laion/CLIP-ViT-H-14-laion2B-s32B-b79K/'),
)

_VITg14 = dict(
    laion2b_s12b_b42k=_pcfg(hf_hub='laion/CLIP-ViT-g-14-laion2B-s12B-b42K/'),
    laion2b_s34b_b88k=_pcfg(hf_hub='laion/CLIP-ViT-g-14-laion2B-s34B-b88K/'),
)

_EVAg14 = dict(
    eva=_pcfg(hf_hub='QuanSun/EVA-CLIP/'),
    eva01=_pcfg(hf_hub='QuanSun/EVA-CLIP/EVA01_g_psz14.pt'),
    eva_clip=_pcfg(hf_hub='QuanSun/EVA-CLIP/EVA01_CLIP_g_14_psz14_s11B.pt'),
    eva01_clip=_pcfg(hf_hub='QuanSun/EVA-CLIP/EVA01_CLIP_g_14_psz14_s11B.pt'),
)

_EVAg14_PLUS = dict(
    eva=_pcfg(hf_hub='QuanSun/EVA-CLIP/'),
    eva01=_pcfg(hf_hub='QuanSun/EVA-CLIP/EVA01_g_psz14.pt'),
    eva_clip=_pcfg(hf_hub='QuanSun/EVA-CLIP/EVA01_CLIP_g_14_plus_psz14_s11B.pt'),
    eva01_clip=_pcfg(hf_hub='QuanSun/EVA-CLIP/EVA01_CLIP_g_14_plus_psz14_s11B.pt'),
)

_VITbigG14 = dict(
    laion2b_s39b_b160k=_pcfg(hf_hub='laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/'),
)

_EVAbigE14 = dict(
    eva=_pcfg(hf_hub='QuanSun/EVA-CLIP/EVA02_E_psz14.pt'),
    eva02=_pcfg(hf_hub='QuanSun/EVA-CLIP/EVA02_E_psz14.pt'),
    eva_clip=_pcfg(hf_hub='QuanSun/EVA-CLIP/EVA02_CLIP_E_psz14_s4B.pt'),
    eva02_clip=_pcfg(hf_hub='QuanSun/EVA-CLIP/EVA02_CLIP_E_psz14_s4B.pt'),
)

_EVAbigE14_PLUS = dict(
    eva=_pcfg(hf_hub='QuanSun/EVA-CLIP/EVA02_E_psz14.pt'),
    eva02=_pcfg(hf_hub='QuanSun/EVA-CLIP/EVA02_E_psz14.pt'),
    eva_clip=_pcfg(hf_hub='QuanSun/EVA-CLIP/EVA02_CLIP_E_psz14_plus_s9B.pt'),
    eva02_clip=_pcfg(hf_hub='QuanSun/EVA-CLIP/EVA02_CLIP_E_psz14_plus_s9B.pt'),
)


_PRETRAINED = {
    # "ViT-B-32": _VITB32,
    "OpenaiCLIP-B-32": _VITB32,
    "OpenCLIP-B-32": _VITB32,

    # "ViT-B-32-quickgelu": _VITB32_quickgelu,
    "OpenaiCLIP-B-32-quickgelu": _VITB32_quickgelu,
    "OpenCLIP-B-32-quickgelu": _VITB32_quickgelu,

    # "ViT-B-16": _VITB16,
    "OpenaiCLIP-B-16": _VITB16,
    "OpenCLIP-B-16": _VITB16,

    "EVA02-B-16": _EVAB16,
    "EVA02-CLIP-B-16": _EVAB16,

    # "ViT-B-16-plus-240": _VITB16_PLUS_240,
    "OpenCLIP-B-16-plus-240": _VITB16_PLUS_240,

    # "ViT-L-14": _VITL14,
    "OpenaiCLIP-L-14": _VITL14,
    "OpenCLIP-L-14": _VITL14,

    "EVA02-L-14": _EVAL14,
    "EVA02-CLIP-L-14": _EVAL14,

    # "ViT-L-14-336": _VITL14_336,
    "OpenaiCLIP-L-14-336": _VITL14_336,

    "EVA02-CLIP-L-14-336": _EVAL14_336,

    # "ViT-H-14": _VITH14,
    # "ViT-g-14": _VITg14,
    "OpenCLIP-H-14": _VITH14,
    "OpenCLIP-g-14": _VITg14,

    "EVA01-CLIP-g-14": _EVAg14,
    "EVA01-CLIP-g-14-plus": _EVAg14_PLUS,

    # "ViT-bigG-14": _VITbigG14,
    "OpenCLIP-bigG-14": _VITbigG14,

    "EVA02-CLIP-bigE-14": _EVAbigE14,
    "EVA02-CLIP-bigE-14-plus": _EVAbigE14_PLUS,
}


def _clean_tag(tag: str):
    # normalize pretrained tags
    return tag.lower().replace('-', '_')


def list_pretrained(as_str: bool = False):
    """ returns list of pretrained models
    Returns a tuple (model_name, pretrain_tag) by default or 'name:tag' if as_str == True
    """
    return [':'.join([k, t]) if as_str else (k, t) for k in _PRETRAINED.keys() for t in _PRETRAINED[k].keys()]


def list_pretrained_models_by_tag(tag: str):
    """ return all models having the specified pretrain tag """
    models = []
    tag = _clean_tag(tag)
    for k in _PRETRAINED.keys():
        if tag in _PRETRAINED[k]:
            models.append(k)
    return models


def list_pretrained_tags_by_model(model: str):
    """ return all pretrain tags for the specified model architecture """
    tags = []
    if model in _PRETRAINED:
        tags.extend(_PRETRAINED[model].keys())
    return tags


def is_pretrained_cfg(model: str, tag: str):
    if model not in _PRETRAINED:
        return False
    return _clean_tag(tag) in _PRETRAINED[model]


def get_pretrained_cfg(model: str, tag: str):
    if model not in _PRETRAINED:
        return {}
    model_pretrained = _PRETRAINED[model]
    return model_pretrained.get(_clean_tag(tag), {})


def get_pretrained_url(model: str, tag: str):
    cfg = get_pretrained_cfg(model, _clean_tag(tag))
    return cfg.get('url', '')


def download_pretrained_from_url(
        url: str,
        cache_dir: Union[str, None] = None,
):
    if not cache_dir:
        cache_dir = os.path.expanduser("~/.cache/clip")
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.basename(url)

    if 'openaipublic' in url:
        expected_sha256 = url.split("/")[-2]
    elif 'mlfoundations' in url:
        expected_sha256 = os.path.splitext(filename)[0].split("-")[-1]
    else:
        expected_sha256 = ''

    download_target = os.path.join(cache_dir, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if expected_sha256:
            if hashlib.sha256(open(download_target, "rb").read()).hexdigest().startswith(expected_sha256):
                return download_target
            else:
                warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")
        else:
            return download_target

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.headers.get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if expected_sha256 and not hashlib.sha256(open(download_target, "rb").read()).hexdigest().startswith(expected_sha256):
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def has_hf_hub(necessary=False):
    if not _has_hf_hub and necessary:
        # if no HF Hub module installed, and it is necessary to continue, raise error
        raise RuntimeError(
            'Hugging Face hub model specified but package not installed. Run `pip install huggingface_hub`.')
    return _has_hf_hub


def download_pretrained_from_hf(
        model_id: str,
        filename: str = 'open_clip_pytorch_model.bin',
        revision=None,
        cache_dir: Union[str, None] = None,
):
    has_hf_hub(True)
    cached_file = hf_hub_download(model_id, filename, revision=revision, cache_dir=cache_dir)
    return cached_file


def download_pretrained(
        cfg: Dict,
        force_hf_hub: bool = False,
        cache_dir: Union[str, None] = None,
):
    target = ''
    if not cfg:
        return target

    download_url = cfg.get('url', '')
    download_hf_hub = cfg.get('hf_hub', '')
    if download_hf_hub and force_hf_hub:
        # use HF hub even if url exists
        download_url = ''

    if download_url:
        target = download_pretrained_from_url(download_url, cache_dir=cache_dir)
    elif download_hf_hub:
        has_hf_hub(True)
        # we assume the hf_hub entries in pretrained config combine model_id + filename in
        # 'org/model_name/filename.pt' form. To specify just the model id w/o filename and
        # use 'open_clip_pytorch_model.bin' default, there must be a trailing slash 'org/model_name/'.
        model_id, filename = os.path.split(download_hf_hub)
        if filename:
            target = download_pretrained_from_hf(model_id, filename=filename, cache_dir=cache_dir)
        else:
            target = download_pretrained_from_hf(model_id, cache_dir=cache_dir)

    return target
