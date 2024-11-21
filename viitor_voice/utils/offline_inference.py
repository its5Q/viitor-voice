from vllm import LLM, SamplingParams
import json
import os
import re
from snac import SNAC
import torch
import numpy as np

pattern = re.compile(r"<\|speech-(\d+)\|>")


def split_sequence(sequence):
    group_size = 7
    first_elements = []
    second_elements = []
    third_elements = []

    # Iterate over the sequence in chunks of 7
    for i in range(0, len(sequence), group_size):
        group = sequence[i:i + group_size]

        # Add elements to the respective lists based on their position in the group
        if len(group) >= 1:
            first_elements.append(group[0])
        if len(group) >= 5:
            second_elements.extend([group[1], group[4]])
        if len(group) >= 7:
            third_elements.extend([group[2], group[3], group[5], group[6]])
        else:
            third_elements.extend(group[2:])

    return first_elements, second_elements, third_elements


class OfflineInference:
    def __init__(self, model_path, config_path):
        with open(config_path, 'r') as f:
            self.vllm_config = json.load(f)
        stop = self.vllm_config['stop']
        tensor_parallel_size = self.vllm_config.get('tensor_parallel_size', 1)
        max_tokens = self.vllm_config['max_tokens']
        prompt_map = self.vllm_config['prompt_map']
        self.sampling_params = SamplingParams(temperature=0.0, best_of=1, stop=stop,
                                              max_tokens=max_tokens, repetition_penalty=1.1, n=1,
                                              allowed_token_ids=list(range(151640, 156014)))
        self.model = LLM(model=model_path,
                         tokenizer=model_path,
                         trust_remote_code=True,
                         tensor_parallel_size=tensor_parallel_size,
                         dtype='float16',
                         max_model_len=max_tokens,
                         # gpu_memory_utilization=self.vllm_config.get('gpu_memory_utilization', 0.5)
                         )
        self.prompt_map = prompt_map
        self.snac_model = SNAC.from_pretrained('hubertsiuzdak/snac_24khz').eval().to('cuda')

    def batch_infer(self, text_list, speaker, speed=2):
        if isinstance(speaker, str):
            prompts = [self.prompt_map[speaker].format(speed, text) for text in text_list]
        else:
            prompts = [self.prompt_map[x].format(speed, text) for x, text in zip(speaker, text_list)]
        outputs = self.model.generate(prompts, self.sampling_params)
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            snac_tokens = pattern.findall(generated_text)
            snac_tokens = [int(x) for x in snac_tokens]
            results.append(snac_tokens)
        audios = self.batch_decode_audios(results)
        return audios

    def batch_decode_audios(self, snac_tokens_list):
        audios = []
        with torch.no_grad():
            for snac_tokens in snac_tokens_list:
                first_elements, second_elements, third_elements = split_sequence(snac_tokens)
                codes = [torch.from_numpy(np.array(x).astype(np.int32)[None,]).to('cuda') for x in
                         [first_elements, second_elements, third_elements]]
                audio_hat_all = self.snac_model.decode(codes)
                audios.append(audio_hat_all[0].cpu().to(torch.float32))
        return audios
