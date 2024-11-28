import sys
import numpy as np
import onnxruntime
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from datasets import load_dataset
from snac import SNAC
from viitor_voice.utils.speaker_codebook_utils import encode as speaker_encode_with_codebook


def compute_mse(original, reconstructed):
    # Ensure both inputs are the same length
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]

    mse = torch.mean((original - reconstructed) ** 2)
    return mse


def combine_sequences(first_elements, second_elements, third_elements):
    group_size = 7
    sequence = []

    second_index = 0
    third_index = 0

    for first in first_elements:
        group = [None] * group_size

        # Assign the first element
        group[0] = first

        # Assign the second and fifth elements if they exist
        if second_index < len(second_elements):
            group[1] = second_elements[second_index]
            second_index += 1
        if second_index < len(second_elements):
            group[4] = second_elements[second_index]
            second_index += 1

        # Assign the remaining elements from third_elements if they exist
        for j in [2, 3, 5, 6]:
            if third_index < len(third_elements):
                group[j] = third_elements[third_index]
                third_index += 1

        # Remove None values at the end of the group if the group is incomplete
        sequence.extend([x for x in group if x is not None])

    return sequence


class Speaker:
    def __init__(self, speaker_model_path: str):
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        providers = [
            ('TensorrtExecutionProvider', {
                'trt_max_workspace_size': 2147483648,
                'trt_fp16_enable': True,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': 'speaker.engine',
                'trt_profile_min_shapes': "input:1x10x80",
                'trt_profile_opt_shapes': "input:1x896x80",
                'trt_profile_max_shapes': "input:1x5000x80",
            }),
            'CUDAExecutionProvider']
        self.speaker_model = onnxruntime.InferenceSession(speaker_model_path, sess_options=option,
                                                          providers=providers)
        self.binding = self.speaker_model.io_binding()

    def __call__(self, input):
        self.binding.bind_input(
            name='input',
            device_type='cuda',
            device_id=0,
            element_type=np.float32,
            shape=tuple(input.shape),
            buffer_ptr=input.data_ptr(),
        )

        output = torch.empty((input.shape[0], 192), dtype=torch.float32, device='cuda:0')
        self.binding.bind_output(
            name='output',
            device_type='cuda',
            device_id=0,
            element_type=np.float32,
            shape=tuple(output.shape),
            buffer_ptr=output.data_ptr(),
        )

        self.speaker_model.run_with_iobinding(self.binding)
        return output


def get_speed_encode(text, snac_result):
    ratio = len(snac_result) / (len(text.split(' ')) + 1e-5)
    if ratio < 19:
        return 5
    elif ratio < 22:
        return 1
    elif ratio < 26:
        return 6
    elif ratio < 30:
        return 2
    elif ratio < 36:
        return 3
    elif ratio < 45:
        return 0
    else:
        return 4


class DataPrepare:
    def __init__(self, origin_sampling_rate, speaker_sampling_rate, snac_sampling_rate, speaker_onnx_path,
                 snac_model_path, speaker_codebook_path):
        if origin_sampling_rate != speaker_sampling_rate:
            self.speaker_sampler = torchaudio.transforms.Resample(orig_freq=origin_sampling_rate,
                                                                  new_freq=speaker_sampling_rate).to('cuda')
        else:
            self.speaker_sampler = None

        if origin_sampling_rate != snac_sampling_rate:
            self.snac_sampler = torchaudio.transforms.Resample(orig_freq=origin_sampling_rate,
                                                               new_freq=snac_sampling_rate).to('cuda')
        else:
            self.snac_sampler = None

        self.snac_model = SNAC.from_pretrained(snac_model_path).eval().cuda()
        self.speaker_model = Speaker(speaker_onnx_path)
        self.codebook = np.load(speaker_codebook_path)
        self.stripe = 4096

    @staticmethod
    def get_speed_encode(text, snac_result):
        ratio = len(snac_result) / (len(text.split(' ')) + 1e-5)
        if ratio < 19:
            return [5]
        elif ratio < 22:
            return [1]
        elif ratio < 26:
            return [6]
        elif ratio < 30:
            return [2]
        elif ratio < 36:
            return [3]
        elif ratio < 45:
            return [0]
        else:
            return [4]

    def padding(self, audio):
        target_length = ((audio.shape[2] + self.stripe - 1) // self.stripe) * self.stripe
        padding_size = target_length - audio.shape[2]
        padded_audio = F.pad(audio, (0, padding_size))
        return padded_audio

    def get_speaker_encode(self, audio):
        speaker_audio = self.speaker_sampler(audio) if self.speaker_sampler else audio
        feat_v2 = kaldi.fbank(speaker_audio,
                              num_mel_bins=80,
                              dither=0,
                              sample_frequency=16000)
        feat_v2 = feat_v2 - feat_v2.mean(dim=0, keepdim=True)
        feat_v2 = feat_v2.unsqueeze(0)

        speaker_embedding = self.speaker_model(feat_v2).cpu().numpy()
        speaker_encode = speaker_encode_with_codebook(speaker_embedding, self.codebook)
        return speaker_encode[0].tolist()

    def get_snac_encode(self, audio):
        snac_audio = self.snac_sampler(audio) if self.snac_sampler else audio
        snac_audio = snac_audio.unsqueeze(0)
        audio_hat, snac_encode = self.snac_model(snac_audio)
        norm = compute_mse(audio_hat[0], snac_audio)
        first_elements, second_elements, third_elements = \
            snac_encode[0].cpu().numpy().tolist(), snac_encode[1].cpu().numpy().tolist(), snac_encode[
                2].cpu().numpy().tolist()
        flattened_snac_encode = combine_sequences(first_elements[0], second_elements[0], third_elements[0])
        return flattened_snac_encode, norm.cpu().numpy().tolist()

    @torch.no_grad()
    def __call__(self, audio, text):
        snac_encode, norm, = self.get_snac_encode(audio)
        speaker_encode = self.get_speaker_encode(audio)
        speed_encode = self.get_speed_encode(text, snac_encode)
        res = {"text": text, 'spk_codes': speaker_encode, 'speed_id': speed_encode[0], 'target': snac_encode,
               'norm': norm}

        return res

