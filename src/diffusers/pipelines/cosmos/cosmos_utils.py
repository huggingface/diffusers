import os
import re

import numpy as np
import torch

from ...utils import get_logger, is_opencv_available, is_pytorch_retinaface_available


if is_opencv_available():
    import cv2

if is_pytorch_retinaface_available():
    from pytorch_retinaface.utils.nms.py_cpu_nms import py_cpu_nms


logger = get_logger(__name__)  # pylint: disable=invalid-name


def read_keyword_list_from_dir(folder_path: str) -> list[str]:
    """Read keyword list from all files in a folder."""
    output_list = []
    file_list = []
    # Get list of files in the folder
    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)):
            file_list.append(file)

    # Process each file
    for file in file_list:
        file_path = os.path.join(folder_path, file)
        try:
            with open(file_path, "r") as f:
                output_list.extend([line.strip() for line in f.readlines()])
        except Exception as e:
            logger.error(f"Error reading file {file}: {str(e)}")

    return output_list


def to_ascii(prompt: str) -> str:
    """Convert prompt to ASCII."""
    return re.sub(r"[^\x00-\x7F]+", " ", prompt)


def pixelate_face(face_img: np.ndarray, blocks: int = 5) -> np.ndarray:
    """
    Pixelate a face region by reducing resolution and then upscaling.

    Args:
        face_img: Face region to pixelate
        blocks: Number of blocks to divide the face into (in each dimension)

    Returns:
        Pixelated face region
    """
    h, w = face_img.shape[:2]
    # Shrink the image and scale back up to create pixelation effect
    temp = cv2.resize(face_img, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    return pixelated


# Adapted from https://github.com/biubug6/Pytorch_Retinaface/blob/master/detect.py
def filter_detected_boxes(boxes, scores, confidence_threshold, nms_threshold, top_k, keep_top_k):
    """Filter boxes based on confidence score and remove overlapping boxes using NMS."""
    # Keep detections with confidence above threshold
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # Sort by confidence and keep top K detections
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    scores = scores[order]

    # Run non-maximum-suppression (NMS) to remove overlapping boxes
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]
    dets = dets[:keep_top_k, :]
    boxes = dets[:, :-1]
    return boxes


# Adapted from https://github.com/biubug6/Pytorch_Retinaface/blob/master/utils/box_utils.py to handle batched inputs
def decode_batch(loc, priors, variances):
    """Decode batched locations from predictions using priors and variances.

    Args:
        loc (tensor): Batched location predictions for loc layers.
            Shape: [batch_size, num_priors, 4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors, 4]
        variances: (list[float]): Variances of prior boxes.

    Return:
        Decoded batched bounding box predictions
            Shape: [batch_size, num_priors, 4]
    """
    batch_size = loc.size(0)
    priors = priors.unsqueeze(0).expand(batch_size, -1, -1)

    boxes = torch.cat(
        (
            priors[:, :, :2] + loc[:, :, :2] * variances[0] * priors[:, :, 2:],
            priors[:, :, 2:] * torch.exp(loc[:, :, 2:] * variances[1]),
        ),
        dim=2,
    )

    boxes[:, :, :2] -= boxes[:, :, 2:] / 2
    boxes[:, :, 2:] += boxes[:, :, :2]
    return boxes


# Adapted from https://github.com/biubug6/Pytorch_Retinaface/blob/master/detect.py
def _check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    logger.debug("Missing keys:{}".format(len(missing_keys)))
    logger.debug("Unused checkpoint keys:{}".format(len(unused_pretrained_keys)))
    logger.debug("Used keys:{}".format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, "load NONE from pretrained checkpoint"
    return True


# Adapted from https://github.com/biubug6/Pytorch_Retinaface/blob/master/detect.py
def _remove_prefix(state_dict, prefix):
    """Old version of the model is stored with all names of parameters sharing common prefix 'module.'"""
    logger.debug("Removing prefix '{}'".format(prefix))

    def f(x):
        return x.split(prefix, 1)[-1] if x.startswith(prefix) else x

    return {f(key): value for key, value in state_dict.items()}


# Adapted from https://github.com/biubug6/Pytorch_Retinaface/blob/master/detect.py
def load_model(model, pretrained_path):
    logger.debug("Loading pretrained model from {}".format(pretrained_path))
    pretrained_dict = torch.load(pretrained_path, weights_only=True)
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = _remove_prefix(pretrained_dict["state_dict"], "module.")
    else:
        pretrained_dict = _remove_prefix(pretrained_dict, "module.")
    _check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


VIOLENCE = {
    "category": "violence",
    "blocked_reason": """This prompt is blocked due to the presence of violent content.""",
    "instruction": """O1: Violence. Should not
- Help people plan or engage in violence.
- Encourage people to engage in violence.
- Contain unpleasant vivid violent imagery or violent description that arouses discomfort in the reader.
Can
- Provide information on violence.
- Discuss the topics of violence, and can discuss historical events involving violence.
""",
}

SEXUAL = {
    "category": "sexual",
    "blocked_reason": """This prompt is blocked due to the presence of sexual content.""",
    "instruction": """O2: Sexual. Should not
- Engage in sexually explicit (i.e., erotic) conversations and should not encourage people to engage in sexual
  activities.
- Engage or encourage sexual activity that is considered illegal.
Can
- Discuss topics related to sexuality and sexual education.
- Provide general information regarding sexual activities.
- Provide guidance on topics related to sexual health.
""",
}

CRIMINAL_PLANNING = {
    "category": "criminal_planning",
    "blocked_reason": """This prompt is blocked due to the presence of criminal planning/confessions content.""",
    "instruction": """O3: Criminal Planning/Confessions. Should not
- Help people plan or engage in criminal activities like theft, kidnapping, or financial crimes, especially when the
  person indicates that they may act on the information provided to them.
- Encourage people to engage in criminal activities or condone criminal behavior.
- Have a confession of crime that was committed or will be committed in future.
Can
- Help people understand how crimes are committed, provided that the response is not so detailed that it could be used
  to provide a roadmap for future crimes.
- Help people understand how systems work that are used to prevent crimes, provided those descriptions do not provide
  detailed information on how to defeat those systems.
""",
}

WEAPONS = {
    "category": "weapons",
    "blocked_reason": """This prompt is blocked due to the presence of weapons content.""",
    "instruction": """O4: Guns and Illegal Weapons. Should not
- Assist in the planning or execution of firearm or illegal weapons crimes, especially when the person indicates that
  they may act on the information provided to them.
- Encourage people to engage in firearm or illegal weapons crimes or condone such behavior.
- Assist or encourage people to execute or own other illegal weapons for example biological agents, or chemical
  weapons, including instructions on how to create or use them.
Can
- Discuss firearms and the arguments for and against firearm ownership.
""",
}

SUBSTANCE_ABUSE = {
    "category": "substance_abuse",
    "blocked_reason": """This prompt is blocked due to the presence of substance abuse content.""",
    "instruction": """O5: Controlled/Regulated Substances. Should not
- Encourage or assist people in the trafficking, consumption, or use of regulated or controlled substances, including
  but not limited to illegal drugs, tobacco, alcohol, and cannabis.
- Assist or encourage people to create such substances when it is illegal to do so.
Can
- Discuss the use of regulated or controlled substances, and can provide general information on those substances (ex:
  what their chemical composition is and what their effects are).
- Provide information on how those substances are manufactured, provided that the information is not so detailed as to
  provide step-by-step instructions for their creation.
""",
}

SUICIDE = {
    "category": "suicide",
    "blocked_reason": "This prompt is blocked due to the presence of suicide and self harm content.",
    "instruction": """O6: Suicide and Self Harm. Should not
- Encourage people to harm themselves.
- Romanticize or condone self-harm or suicide.
- Provide information regarding the methods of suicide or self-harm.
- Provide any information or advice, other than information regarding appropriate health resources, to people who
  express or imply that they wish to harm themselves.
- Contain confessions or intent of self harm
Can
- Provide general information regarding self-harm and suicide (ex: statistics, the mental health challenges faced by
  those who take these actions).
Should
- Provide information regarding appropriate health resources to people who imply or express the desire to harm
  themselves.
""",
}

CSAM = {
    "category": "csam",
    "blocked_reason": "This prompt contains material related to Child Sexual Abuse",
    "instruction": """O7: Sexual (minor). Should not
- Assist, engage or encourage intention to engage in sexual activity with an underage/minor.
- Discuss or plan sexual acts, sharing of media of sexual nature, personal information or intention to meet in person
  for sexual activity with a minor.
- Should not include discussion where private information is shared with underage minors for purposes of acting on them
  and taking actions that are sexual in nature including meeting or sharing media.
- Should not discuss child abuse or engage, encourage, assist in sexual grooming/trafficking of children.
Can
- Provide information about sex education for use in schools.
""",
}

HATRED = {
    "category": "hatred",
    "blocked_reason": "This prompt is blocked due to the presence of hate speech content.",
    "instruction": """O8: Hate/Identity Hate. Should not
- Express hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like
  their
race, color, ethnicity, religion, national origin, sexual orientation, gender, gender identity, or disability or caste.
- Encourage discrimination against people based on those characteristics and should not use slurs or create insults
  based on those characteristics.
- Should not discuss prejudice towards a group classified as protected according to the law of Germany, UK, USA or any
  other country.
- Should not contain slurs.
- Should not contain text that is distasteful, offensive, or hateful that causes others to feel grief, anger, or fear.
Can
- Provide information on discrimination.
- Discuss the topics of discrimination, and can discuss historical events involving discrimination.
- Discuss neutral statements referring to group identity and contextualized hate speech (that talks about hate but does
  not incite hate).
""",
}

HARASSMENT = {
    "category": "harassment",
    "blocked_reason": "This prompt is blocked due to the presence of harassment content.",
    "instruction": """O10: Harassment. Should not
- Have content that may be used to torment or annoy individuals in real life, or make harassment more likely to occur.
- Engage in harassment, misconduct and unlawful discrimination
- Single out an individual or marginalization and/or retaliation based on protected characteristics like Race, Color,
  Gender, Sex,
Sexual orientation, Gender identity, and gender expression, National origin, Ethnicity, Disability (including being
regarded as disabled), Religion, Age (40+), Pregnancy (including pregnancy, childbirth or related medical conditions),
Genetic information, Military or veteran status, Citizenship status, Political activity, or affiliation,
- Taking or requesting statutorily protected leave Body characteristics.
- Promise to give a benefit, or a threat to retaliate or take an adverse action based on a response to harassment and
  misconduct.
- Bullying.
- Being derogatory and humiliating toward an individual but not necessarily protected characteristics under law.
- Making rude or insulting comments, demeaning, and objectifying terms toward an individual.
Can
- Discuss negative impacts of harassment and bullying based on protected characteristics like race, color, gender, sex,
  nationality, within a contextualized setting.
- Provide information on protected characteristics or refers, cites material that describes what protected
  characteristics are.
- Provide information or refer or cite material that informs what harassment is.
""",
}

THREAT = {
    "category": "threat",
    "blocked_reason": "This prompt is blocked due to the presence of threat content.",
    "instruction": """O11: Threat. Should not
- Discuss a communication of intent to inflict harm or loss on another person.
- Should not contain discussion of intention of intimidation which is a tactic used between conflicting parties to make
  the other timid or psychologically insecure for coercion or control.
- Should not contain intention of knowingly putting another person in fear of bodily injury.
- Should not assist or plan injury, physical or mental damage, act or instance of injury, or material or any kind of
  loss to a person.
- Should not contain threats that are forbidden by law as intent to obtain a monetary advantage or to compel a person
  to act against their will.
Can
- Provide information about threat is and the meaning of threat.
""",
}

PROFANITY = {
    "category": "profanity",
    "blocked_reason": "This prompt is blocked due to the presence of profanity content.",
    "instruction": """O12: Profanity. Should not
- Contain swear words, curse words, or other obscene or profane language
- This includes offensive words used without any intention to act on them or without any explicit targeted recipient
""",
}

UNSAFE_CATEGORIES = [
    VIOLENCE,
    SEXUAL,
    CRIMINAL_PLANNING,
    WEAPONS,
    SUBSTANCE_ABUSE,
    SUICIDE,
    CSAM,
    HATRED,
    HARASSMENT,
    THREAT,
    PROFANITY,
]

CLASS_IDX_TO_NAME = {
    0: "Safe",
    1: "Sexual_Content",
    2: "Violence",
    3: "Drugs",
    4: "Child_Abuse",
    5: "Hate_and_Harassment",
    6: "Self-Harm",
}

# RetinaFace model constants from https://github.com/biubug6/Pytorch_Retinaface/blob/master/detect.py
TOP_K = 5_000
KEEP_TOP_K = 750
NMS_THRESHOLD = 0.4
