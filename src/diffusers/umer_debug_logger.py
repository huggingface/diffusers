# Logger to help me (UmerHA) debug controlnet-xs

import os
import csv
import torch
import inspect
import logging
import shutil
from types import SimpleNamespace

from datetime import datetime

class UmerDebugLogger:

    _FILE = 'udl.csv'

    def __init__(self, log_dir='logs', condition=None):
        self.log_dir, self.condition, self.tensor_counter = log_dir, condition, 0
        os.makedirs(log_dir, exist_ok=True)
        self.fields = ['timestamp', 'cls', 'fn', 'shape', 'msg', 'condition', 'tensor_file']
        self.create_file()
        self.warned_of_no_condition = False

    @property
    def full_file_path(self): return os.path.join(self.log_dir, self._FILE) 

    def create_file(self):
        file = self.full_file_path    
        if not os.path.isfile(file):
            with open(file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fields)
                writer.writeheader()


    def set_dir(self, log_dir, clear=False):
        self.log_dir = log_dir
        if clear: self.clear_logs()
        self.create_file()

    def clear_logs(self):
        shutil.rmtree(self.log_dir, ignore_errors=True)
        os.makedirs(self.log_dir, exist_ok=True)
        self.create_file()

    def set_condition(self, condition): self.condition = condition

    def log_if(self, msg, t, condition, *, print_=False):
        self.maybe_warn_of_no_condition()
       
        # Use inspect to get the current frame and then go back one level to find caller
        frame = inspect.currentframe()
        caller_frame = frame.f_back
        caller_info = inspect.getframeinfo(caller_frame)

        # Extract class and function name from the caller
        cls_name = caller_frame.f_locals.get('self', None).__class__.__name__ if 'self' in caller_frame.f_locals else None
        function_name = caller_info.function

        if not hasattr(t, 'shape'): t = torch.tensor(t)
        t = t.cpu().detach()

        if condition == self.condition:
            # Save tensor to a file
            tensor_filename = f"tensor_{self.tensor_counter}.pt"
            torch.save(t, os.path.join(self.log_dir, tensor_filename))
            self.tensor_counter += 1

            # Log information to CSV
            log_info = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'cls': cls_name,
                'fn': function_name,
                'shape': str(list(t.shape)),
                'msg': msg,
                'condition': condition,
                'tensor_file': tensor_filename
            }

            with open(self.full_file_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fields)
                writer.writerow(log_info)

            if print_: print(f'{msg}\t{t.flatten()[:10]}')
    
    def print_if(self, msg, conditions, end='\n'):
        self.maybe_warn_of_no_condition()
        if not isinstance(conditions, (tuple, list)): conditions = [conditions]
        if any(self.condition==c for c in conditions): print(msg, end=end)

    def stop_if(self, condition, funny_msg):
        if condition == self.condition:
            print(funny_msg)
            raise SystemExit(funny_msg)

    def maybe_warn_of_no_condition(self):
        if self.condition is None and not self.warned_of_no_condition :
            print("Warning: No condition set for UmerDebugLogger")
            self.warned_of_no_condition = True

    def get_log_objects(self):
        log_objects = []
        file = self.full_file_path  
        with open(file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['tensor'] = torch.load(os.path.join(self.log_dir, row['tensor_file']))
                row['head'] = row['tensor'].flatten()[:10]
                del row['tensor_file']
                log_objects.append(SimpleNamespace(**row))
        return log_objects

    @classmethod
    def load_log_objects_from_dir(self, log_dir):
        file = os.path.join(log_dir, self._FILE)   
        log_objects = []
        with open(file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['t'] = torch.load(os.path.join(log_dir, row['tensor_file']))
                row['head'] = row['t'].flatten()[:10]
                del row['tensor_file']
                log_objects.append(SimpleNamespace(**row))
        return log_objects


udl = UmerDebugLogger()
