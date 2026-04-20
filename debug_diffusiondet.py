from diffusers.models.diffusiondet.configuration_diffusiondet import DiffusionDetConfig
from diffusers.models.diffusiondet.modeling_diffusiondet import DiffusionDet


def main():
    config = DiffusionDetConfig()
    model = DiffusionDet(config)
    model([])


if __name__ == '__main__':
    main()