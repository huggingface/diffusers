import pandas as pd
import datasets
import os

_VERSION = datasets.Version("0.0.2")

_DESCRIPTION = "High-quality commercial image-mask + caption dataset for inpainting"
_HOMEPAGE = "TODO"
_LICENSE = "TODO"
_CITATION = "TODO"

_FEATURES = datasets.Features(
    {
        "image": datasets.Image(),
        "text": datasets.Value("string"),
        "canny": datasets.Image(),
        'mask': datasets.Image(),
    },
)

_DEFAULT_CONFIG = datasets.BuilderConfig(name="default", version=_VERSION)

class CannyFormat(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [_DEFAULT_CONFIG]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        root = '/home/gkalstn000/dataset/canny_format'

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "metadata_path": os.path.join(root, 'train.jsonl'),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "metadata_path": os.path.join(root, 'test.jsonl'),
                }
            )


        ]

    def _generate_examples(self, metadata_path):
        metadata = pd.read_json(metadata_path, lines=True)

        for _, row in metadata.iterrows():
            text = row["text"]

            image_path = row["image"]
            image = open(image_path, "rb").read()

            conditioning_image_path = row["canny_total"]
            conditioning_image = open(conditioning_image_path, "rb").read()

            mask_image_path = row["mask_background"]
            mask = open(mask_image_path, "rb").read()

            yield row["image"], {
                "image": {
                    "path": image_path,
                    "bytes": image,
                },
                "text": text,

                "canny": {
                    "path": conditioning_image_path,
                    "bytes": conditioning_image,
                },

                "mask": {
                    "path": mask_image_path,
                    "bytes": mask,
                }
            }