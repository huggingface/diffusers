# Logger to help me (UmerHA) debug controlnet-xs

import csv
import inspect
import os
import shutil
from datetime import datetime
from types import SimpleNamespace

import torch


class UmerDebugLogger:
    _FILE = "udl.csv"

    INPUT_SAVE = 'input_save'
    BLOCK = 'block'
    SUBBLOCK = 'subblock'
    SUBBLOCKM1 = 'subblock-minus-1'
    allowed_conditions = [INPUT_SAVE, BLOCK, SUBBLOCK, SUBBLOCKM1]

    input_files = None

    def __init__(self, log_dir="logs", condition=None):
        self.log_dir, self.condition, self.tensor_counter = log_dir, condition, 0
        os.makedirs(log_dir, exist_ok=True)
        self.fields = ["timestamp", "cls", "fn", "shape", "msg", "condition", "tensor_file"]
        self.create_file()
        self.warned_of_no_condition = False
        print(
            "Info: `UmerDebugLogger` created. This is a logging class that will be deleted when the PR to integrate ControlNet-XS is done."
        )

    @property
    def full_file_path(self):
        return os.path.join(self.log_dir, self._FILE)

    def create_file(self):
        file = self.full_file_path
        if not os.path.isfile(file):
            with open(file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fields)
                writer.writeheader()

    def set_dir(self, log_dir, clear=False):
        self.log_dir = log_dir
        if clear:
            self.clear_logs()
        self.create_file()

    def clear_logs(self):
        shutil.rmtree(self.log_dir, ignore_errors=True)
        os.makedirs(self.log_dir, exist_ok=True)
        self.create_file()

    def set_condition(self, condition):
        if not isinstance(condition, list): condition = [condition]
        self.condition = condition

    def check_condition(self, condition):
        if not condition in self.allowed_conditions: raise ValueError(f'Unknown condition: {condition}')
        return condition in self.condition

    def log_if(self, msg, t, condition, *, print_=False):
        self.maybe_warn_of_no_condition()

        if not self.check_condition(condition):
            return

        # Use inspect to get the current frame and then go back one level to find caller
        frame = inspect.currentframe()
        caller_frame = frame.f_back
        caller_info = inspect.getframeinfo(caller_frame)

        # Extract class and function name from the caller
        cls_name = (
            caller_frame.f_locals.get("self", None).__class__.__name__ if "self" in caller_frame.f_locals else None
        )
        function_name = caller_info.function

        if not hasattr(t, "shape"):
            t = torch.tensor(t)
        t = t.cpu().detach()
        
        # Save tensor to a file
        tensor_filename = f"tensor_{self.tensor_counter}.pt"
        torch.save(t, os.path.join(self.log_dir, tensor_filename))
        self.tensor_counter += 1

        # Log information to CSV
        log_info = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cls": cls_name,
            "fn": function_name,
            "shape": str(list(t.shape)),
            "msg": msg,
            "condition": condition,
            "tensor_file": tensor_filename,
        }

        with open(self.full_file_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow(log_info)

        if print_:
            print(f"{msg}\t{t.flatten()[:10]}")

    def print_if(self, msg, conditions, end="\n"):
        self.maybe_warn_of_no_condition()
        if not isinstance(conditions, (tuple, list)):
            conditions = [conditions]
        if any(self.condition == c for c in conditions):
            print(msg, end=end)

    def stop_if(self, condition, funny_msg):
        if self.check_condition(condition):
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            raise SystemExit(f"{funny_msg} - {current_time}")

    def maybe_warn_of_no_condition(self):
        if self.condition is None and not self.warned_of_no_condition:
            print("Info: No condition set for UmerDebugLogger")
            self.warned_of_no_condition = True

    def get_log_objects(self):
        log_objects = []
        file = self.full_file_path
        with open(file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["tensor"] = torch.load(os.path.join(self.log_dir, row["tensor_file"]))
                row["head"] = row["tensor"].flatten()[:10]
                del row["tensor_file"]
                log_objects.append(SimpleNamespace(**row))
        return log_objects

    @classmethod
    def load_log_objects_from_dir(self, log_dir):
        file = os.path.join(log_dir, self._FILE)
        log_objects = []
        with open(file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["t"] = torch.load(os.path.join(log_dir, row["tensor_file"]))
                row["head"] = row["t"].flatten()[:10]
                del row["tensor_file"]
                log_objects.append(SimpleNamespace(**row))
        return log_objects

    def save_input(self, dir_, x, t, xcross, hint, text_embeds=None, time_ids=None, minimize_bs=True):
        assert (text_embeds is None and time_ids is None) or (text_embeds is not None and time_ids is not None)
        is_sdxl = text_embeds is not None
        inputs = dict(
            x=os.path.join(dir_, x),
            t=os.path.join(dir_, t),
            xcross=os.path.join(dir_, xcross),
            hint=os.path.join(dir_,hint)
        )
        if is_sdxl:
            inputs.update(dict(
                text_embeds=os.path.join(dir_, text_embeds),
                time_ids=os.path.join(dir_, time_ids),
            ))
        self.input_files = SimpleNamespace(**inputs)
        self.input_action = 'save'
        self.minimize_bs = minimize_bs

    def load_input(self, dir_, x, t, xcross, hint, text_embeds=None, time_ids=None):
        assert (text_embeds is None and time_ids is None) or (text_embeds is not None and time_ids is not None)
        is_sdxl = text_embeds is not None
        inputs = dict(
            x=os.path.join(dir_, x),
            t=os.path.join(dir_, t),
            xcross=os.path.join(dir_, xcross),
            hint=os.path.join(dir_,hint)
        )
        if is_sdxl:
            inputs.update(dict(
                text_embeds=os.path.join(dir_, text_embeds),
                time_ids=os.path.join(dir_, time_ids),
            ))
        self.input_files = SimpleNamespace(**inputs)
        self.input_action = 'load'

    def dont_process_input(self):
        self.input_action = 'none'
        self.input_files = {}

    def do_input_action(self, x, t, xcross, hint, text_embeds=None, time_ids=None):
        assert self.input_files is not None, "self.input_files not set! Use `save_input`, `load_input` or `dont_process_input`"
        assert self.input_action in ['save', 'load', 'none']
        assert (text_embeds is None and time_ids is None) or (text_embeds is not None and time_ids is not None)
        is_sdxl = text_embeds is not None

        if self.input_action == 'save':
            assert x.shape[0]==t.shape[0]==xcross.shape[0]==hint.shape[0]
            if is_sdxl:
                assert x.shape[0]==text_embeds.shape[0]==time_ids.shape[0]

            bs = x.shape[0]
            if self.minimize_bs and bs > 1:
                print(f'[udl] Input has batch size {bs} but reducing to 1 before saving')
                x = x[0:1]
                t = t[0:1]
                xcross = xcross[0:1]
                hint = hint[0:1]
                if is_sdxl:
                    text_embeds = text_embeds[0:1]
                    time_ids = time_ids[0:1]

            torch.save(x, self.input_files.x)
            torch.save(t, self.input_files.t)
            torch.save(xcross, self.input_files.xcross)
            torch.save(hint, self.input_files.hint)

            assert x.shape[0]==t.shape[0]==xcross.shape[0]==hint.shape[0]

            if is_sdxl:
                torch.save(text_embeds, self.input_files.text_embeds)
                torch.save(time_ids, self.input_files.time_ids)
                assert x.shape[0]==text_embeds.shape[0]==time_ids.shape[0]
            
            print(f'[udl] Input saved (batch size = {x.shape[0]})')
        elif self.input_action == 'load':
            x = torch.load(self.input_files.x, map_location=x.device)
            t = torch.load( self.input_files.t, map_location=t.device)
            xcross = torch.load(self.input_files.xcross, map_location=xcross.device)
            hint = torch.load(self.input_files.hint, map_location=hint.device)
            assert x.shape[0]==t.shape[0]==xcross.shape[0]==hint.shape[0]

            if is_sdxl:
                text_embeds = torch.load(self.input_files.text_embeds, map_location=text_embeds.device)
                time_ids = torch.load(self.input_files.time_ids, map_location=time_ids.device)
                assert x.shape[0]==text_embeds.shape[0]==time_ids.shape[0]

            print(f'[udl] Input loaded (batch size = {x.shape[0]})')
        else:
            print(f'[udl] Neither saving nor loading input (batch size = {x.shape[0]})')
        return x, t, xcross, hint, text_embeds, time_ids


udl = UmerDebugLogger()
