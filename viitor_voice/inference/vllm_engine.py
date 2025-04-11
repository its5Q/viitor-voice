import numpy as np
import torch
import torchaudio
import traceback
from snac import SNAC
from vllm import LLM, SamplingParams
from viitor_voice.inference.common import combine_sequences, load_audio, pattern, split_sequence
from pprint import pprint


class VllmEngine:
    def __init__(self, model_path):
        tensor_parallel_size = 1
        max_tokens = 3072
        self.sampling_params = SamplingParams(temperature=0.0, best_of=1, stop_token_ids=[156027],
                                              max_tokens=max_tokens, repetition_penalty=1.1, n=1,
                                              allowed_token_ids=list(range(151665, 156033)), logprobs=10,
                                              skip_special_tokens=True)
        self.model = LLM(model=model_path,
                         tokenizer=model_path,
                         trust_remote_code=True,
                         tensor_parallel_size=tensor_parallel_size,
                         dtype='bfloat16',
                         max_model_len=max_tokens,
                         # gpu_memory_utilization=self.vllm_config.get('gpu_memory_utilization', 0.5),
                         )
        self.snac_model = SNAC.from_pretrained('hubertsiuzdak/snac_24khz').eval().to('cuda')

    def batch_infer(self, text_list):
        prompts = [
            '<|START_TEXT|>' + x + '<|END_TEXT|>' + '<|START_AUDIO|>'
            for x in text_list]
        outputs = self.model.generate(prompts, self.sampling_params)
        results = []
        for output in outputs:
            pprint(output)
            generated_text = output.outputs[0].text
            pprint(generated_text)
            snac_tokens = pattern.findall(generated_text)
            snac_tokens = [int(x) for x in snac_tokens]
            results.append(snac_tokens)
        audios = self.batch_decode_audios(results)
        return audios

    def batch_decode_audios(self, snac_tokens_list):
        audios = []
        with torch.no_grad():
            for snac_tokens in snac_tokens_list:
                print(snac_tokens)
                try:
                    first_elements, second_elements, third_elements = split_sequence(snac_tokens)
                    codes = [torch.from_numpy(np.array(x).astype(np.int32)[None,]).to('cuda') for x in
                             [first_elements, second_elements, third_elements]]
                    print(codes)
                    audio_hat_all = self.snac_model.decode(codes)[0].cpu()
                    audios.append(audio_hat_all.to(torch.float32))
                except Exception:
                    traceback.print_exc()
                    audios.append('error')
                    print('error')
        return audios
