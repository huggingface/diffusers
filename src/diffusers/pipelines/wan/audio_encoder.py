# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


def get_sample_indices(original_fps, total_frames, target_fps, num_sample, fixed_start=None):
    required_duration = num_sample / target_fps
    required_origin_frames = int(np.ceil(required_duration * original_fps))
    if required_duration > total_frames / original_fps:
        raise ValueError("required_duration must be less than video length")

    if fixed_start is not None and fixed_start >= 0:
        start_frame = fixed_start
    else:
        max_start = total_frames - required_origin_frames
        if max_start < 0:
            raise ValueError("video length is too short")
        start_frame = np.random.randint(0, max_start + 1)
    start_time = start_frame / original_fps

    end_time = start_time + required_duration
    time_points = np.linspace(start_time, end_time, num_sample, endpoint=False)

    frame_indices = np.round(np.array(time_points) * original_fps).astype(int)
    frame_indices = np.clip(frame_indices, 0, total_frames - 1)
    return frame_indices


def linear_interpolation(features, input_fps, output_fps, output_len=None):
    """
    features: shape=[1, T, 512] input_fps: fps for audio, f_a output_fps: fps for video, f_m output_len: video length
    """
    features = features.transpose(1, 2)  # [1, 512, T]
    seq_len = features.shape[2] / float(input_fps)  # T/f_a
    if output_len is None:
        output_len = int(seq_len * output_fps)  # f_m*T/f_a
    output_features = F.interpolate(
        features, size=output_len, align_corners=True, mode="linear"
    )  # [1, 512, output_len]
    return output_features.transpose(1, 2)  # [1, output_len, 512]


class WanAudioEncoder:
    def __init__(self, device="cpu", model_id="facebook/wav2vec2-base-960h"):
        # load pretrained model
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id)

        self.model = self.model.to(device)

        self.video_rate = 30

    def extract_audio_feat(self, audio_path, return_all_layers=False, dtype=torch.float32):
        audio_input, sample_rate = librosa.load(audio_path, sr=16000)

        input_values = self.processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values

        # INFERENCE

        # retrieve logits & take argmax
        res = self.model(input_values.to(self.model.device), output_hidden_states=True)
        if return_all_layers:
            feat = torch.cat(res.hidden_states)
        else:
            feat = res.hidden_states[-1]
        feat = linear_interpolation(feat, input_fps=50, output_fps=self.video_rate)

        z = feat.to(dtype)  # Encoding for the motion
        return z

    def get_audio_embed_bucket(self, audio_embed, stride=2, batch_frames=12, m=2):
        num_layers, audio_frame_num, audio_dim = audio_embed.shape

        if num_layers > 1:
            return_all_layers = True
        else:
            return_all_layers = False

        min_batch_num = int(audio_frame_num / (batch_frames * stride)) + 1

        bucket_num = min_batch_num * batch_frames
        batch_idx = [stride * i for i in range(bucket_num)]
        batch_audio_eb = []
        for bi in batch_idx:
            if bi < audio_frame_num:
                audio_sample_stride = 2
                chosen_idx = list(
                    range(bi - m * audio_sample_stride, bi + (m + 1) * audio_sample_stride, audio_sample_stride)
                )
                chosen_idx = [0 if c < 0 else c for c in chosen_idx]
                chosen_idx = [audio_frame_num - 1 if c >= audio_frame_num else c for c in chosen_idx]

                if return_all_layers:
                    frame_audio_embed = audio_embed[:, chosen_idx].flatten(start_dim=-2, end_dim=-1)
                else:
                    frame_audio_embed = audio_embed[0][chosen_idx].flatten()
            else:
                frame_audio_embed = (
                    torch.zeros([audio_dim * (2 * m + 1)], device=audio_embed.device)
                    if not return_all_layers
                    else torch.zeros([num_layers, audio_dim * (2 * m + 1)], device=audio_embed.device)
                )
            batch_audio_eb.append(frame_audio_embed)
        batch_audio_eb = torch.cat([c.unsqueeze(0) for c in batch_audio_eb], dim=0)

        return batch_audio_eb, min_batch_num

    def get_audio_embed_bucket_fps(self, audio_embed, fps=16, batch_frames=81, m=0):
        num_layers, audio_frame_num, audio_dim = audio_embed.shape

        if num_layers > 1:
            return_all_layers = True
        else:
            return_all_layers = False

        scale = self.video_rate / fps

        min_batch_num = int(audio_frame_num / (batch_frames * scale)) + 1

        bucket_num = min_batch_num * batch_frames
        padd_audio_num = math.ceil(min_batch_num * batch_frames / fps * self.video_rate) - audio_frame_num
        batch_idx = get_sample_indices(
            original_fps=self.video_rate,
            total_frames=audio_frame_num + padd_audio_num,
            target_fps=fps,
            num_sample=bucket_num,
            fixed_start=0,
        )
        batch_audio_eb = []
        audio_sample_stride = int(self.video_rate / fps)
        for bi in batch_idx:
            if bi < audio_frame_num:
                chosen_idx = list(
                    range(bi - m * audio_sample_stride, bi + (m + 1) * audio_sample_stride, audio_sample_stride)
                )
                chosen_idx = [0 if c < 0 else c for c in chosen_idx]
                chosen_idx = [audio_frame_num - 1 if c >= audio_frame_num else c for c in chosen_idx]

                if return_all_layers:
                    frame_audio_embed = audio_embed[:, chosen_idx].flatten(start_dim=-2, end_dim=-1)
                else:
                    frame_audio_embed = audio_embed[0][chosen_idx].flatten()
            else:
                frame_audio_embed = (
                    torch.zeros([audio_dim * (2 * m + 1)], device=audio_embed.device)
                    if not return_all_layers
                    else torch.zeros([num_layers, audio_dim * (2 * m + 1)], device=audio_embed.device)
                )
            batch_audio_eb.append(frame_audio_embed)
        batch_audio_eb = torch.cat([c.unsqueeze(0) for c in batch_audio_eb], dim=0)

        return batch_audio_eb, min_batch_num
