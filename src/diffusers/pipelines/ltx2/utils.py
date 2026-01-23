# Pre-trained sigma values for distilled model are taken from
# https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py
DISTILLED_SIGMA_VALUES = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875]

# Reduced schedule for super-resolution stage 2 (subset of distilled values)
STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875]
