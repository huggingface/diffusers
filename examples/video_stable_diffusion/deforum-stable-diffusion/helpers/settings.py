import os
import json

def load_args(args_dict, anim_args_dict, settings_file, custom_settings_file, verbose=True):
    default_settings_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'settings'))  
    if settings_file.lower() == 'custom':
        settings_filename = custom_settings_file
    else:
        settings_filename = os.path.join(default_settings_dir,settings_file)
    print(f"Reading custom settings from {settings_filename}...")
    if not os.path.isfile(settings_filename):
        print('The settings file does not exist. The in-notebook settings will be used instead.')
    else:
        if not verbose:
            print(f"Any settings not included in {settings_filename} will use the in-notebook settings by default.")
        with open(settings_filename, "r") as f:
            jdata = json.loads(f.read())
            if jdata.get("prompts") is not None:
                animation_prompts = jdata["prompts"]
            for i, k in enumerate(args_dict):
                if k in jdata:
                    args_dict[k] = jdata[k]
                else:
                    if verbose:
                        print(f"key {k} doesn't exist in the custom settings data! using the default value of {args_dict[k]}")
            for i, k in enumerate(anim_args_dict):
                if k in jdata:
                    anim_args_dict[k] = jdata[k]
                else:
                    if verbose:
                        print(f"key {k} doesn't exist in the custom settings data! using the default value of {anim_args_dict[k]}")
            if verbose:
                print(args_dict)
                print(anim_args_dict)