#!/usr/bin/env python3
import json
import os

from regex import P
from diffusers import UNetUnconditionalModel
from scripts.convert_ncsnpp_original_checkpoint_to_diffusers import convert_ncsnpp_checkpoint
from huggingface_hub import hf_hub_download
import torch



def convert_checkpoint(model_id, subfolder=None, checkpoint = "diffusion_model.pt", config = "config.json"):
    if subfolder is not None:
        checkpoint = os.path.join(subfolder, checkpoint)
        config = os.path.join(subfolder, config)

    original_checkpoint = torch.load(hf_hub_download(model_id, checkpoint),map_location='cpu')
    config_path = hf_hub_download(model_id, config)

    with open(config_path) as f:
        config = json.load(f)

    checkpoint = convert_ncsnpp_checkpoint(original_checkpoint, config)


    def current_codebase_conversion(path):
        model = UNetUnconditionalModel.from_pretrained(model_id, subfolder=subfolder, sde=True)
        model.eval()
        model.config.sde=False
        model.save_config(path)
        model.config.sde=True
        
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        noise = torch.randn(1, model.config.in_channels, model.config.image_size, model.config.image_size)
        time_step = torch.tensor([10] * noise.shape[0])

        with torch.no_grad():
            output = model(noise, time_step)

        return model.state_dict()

    path = f"{model_id}_converted"
    currently_converted_checkpoint = current_codebase_conversion(path)


    def diff_between_checkpoints(ch_0, ch_1):
        all_layers_included = False

        if not set(ch_0.keys()) == set(ch_1.keys()):
            print(f"Contained in ch_0 and not in ch_1 (Total: {len((set(ch_0.keys()) - set(ch_1.keys())))})")
            for key in sorted(list((set(ch_0.keys()) - set(ch_1.keys())))):
                print(f"\t{key}")

            print(f"Contained in ch_1 and not in ch_0 (Total: {len((set(ch_1.keys()) - set(ch_0.keys())))})")
            for key in sorted(list((set(ch_1.keys()) - set(ch_0.keys())))):
                print(f"\t{key}")
        else:
            print("Keys are the same between the two checkpoints")
            all_layers_included = True

        keys = ch_0.keys()
        non_equal_keys = []

        if all_layers_included:
            for key in keys:
                try:
                    if not torch.allclose(ch_0[key].cpu(), ch_1[key].cpu()):
                        non_equal_keys.append(f'{key}. Diff: {torch.max(torch.abs(ch_0[key].cpu() - ch_1[key].cpu()))}')

                except RuntimeError as e:
                    print(e)
                    non_equal_keys.append(f'{key}. Diff in shape: {ch_0[key].size()} vs {ch_1[key].size()}')

            if len(non_equal_keys):
                non_equal_keys = '\n\t'.join(non_equal_keys)
                print(f"These keys do not satisfy equivalence requirement:\n\t{non_equal_keys}")
            else:
                print("All keys are equal across checkpoints.")


    diff_between_checkpoints(currently_converted_checkpoint, checkpoint)
    os.makedirs( f"{model_id}_converted",exist_ok =True)
    torch.save(checkpoint, f"{model_id}_converted/diffusion_model.pt")


model_ids = ["fusing/ffhq_ncsnpp","fusing/church_256-ncsnpp-ve", "fusing/celebahq_256-ncsnpp-ve", 
             "fusing/bedroom_256-ncsnpp-ve","fusing/ffhq_256-ncsnpp-ve","fusing/ncsnpp-ffhq-ve-dummy"
            ]
for model in model_ids: 
    print(f"converting {model}")
    try:
        convert_checkpoint(model)
    except Exception as e:
        print(e)

from tests.test_modeling_utils import PipelineTesterMixin, NCSNppModelTests

tester1 = NCSNppModelTests()
tester2 = PipelineTesterMixin()

os.environ["RUN_SLOW"] = '1'
cmd = "export RUN_SLOW=1; echo $RUN_SLOW" # or whatever command
os.system(cmd)
tester2.test_score_sde_ve_pipeline(f"{model_ids[0]}_converted")
tester1.test_output_pretrained_ve_mid(f"{model_ids[2]}_converted")
tester1.test_output_pretrained_ve_large(f"{model_ids[-1]}_converted")
