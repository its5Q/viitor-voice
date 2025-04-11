import numpy as np
import torch
import torchaudio
from snac import SNAC
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM
from viitor_voice.inference.common import combine_sequences, load_audio, pattern, split_sequence


class TransformersEngine:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation = "flash_attention_2").to(device)
        self.snac_model = SNAC.from_pretrained('hubertsiuzdak/snac_24khz').eval().to(device)

    def batch_infer(self, text_list):
        prompts = [
            '<|START_TEXT|>' + x + '<|END_TEXT|>' + '<|START_AUDIO|>'
            for x in text_list]
        prompt_ids_list = self.tokenizer(prompts, add_special_tokens=False).input_ids
        results = []
        for prompt_ids in prompt_ids_list:
            prompt_ids = torch.tensor([prompt_ids], dtype=torch.int64).to(self.device)
            output_ids = self.model.generate(prompt_ids, eos_token_id=156027, no_repeat_ngram_size=0, num_beams=1,
                                             do_sample=False, repetition_penalty=1.3,
                                             suppress_tokens=list(range(151665)))
            output_ids = output_ids[0, prompt_ids.shape[-1]:].cpu().numpy().tolist()
            generated_text = self.tokenizer.batch_decode([output_ids], skip_special_tokens=False)[0]
            snac_tokens = pattern.findall(generated_text)
            snac_tokens = [int(x) for x in snac_tokens]
            print('SNAC token count: ', len(snac_tokens))
            results.append(snac_tokens)
        audios = self.batch_decode_audios(results)
        return audios

    def batch_decode_audios(self, snac_tokens_list):
        audios = []
        with torch.no_grad():
            for snac_tokens in snac_tokens_list:
                try:
                    first_elements, second_elements, third_elements = split_sequence(snac_tokens)
                    codes = [torch.from_numpy(np.array(x).astype(np.int32)[None,]).to(self.device) for x in
                             [first_elements, second_elements, third_elements]]
                    audio_hat_all = self.snac_model.decode(codes)[0].cpu()
                    audios.append(audio_hat_all.to(torch.float32))
                except:
                    traceback.print_exc()
                    audios.append('error')
                    print('error')
        return audios
