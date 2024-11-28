from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import sys
from tokenizers import AddedToken

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
    ds = load_dataset('json', data_files=sys.argv[2], split='train')

    added_tokens = ['<|speech-{}|>'.format(i) for i in range(4096)] + \
                   ['<|speaker-{}|>'.format(i) for i in range(256)] + \
                   ['<|speed-{}|>'.format(i) for i in range(7)]

    added_tokens = [AddedToken(x) for x in added_tokens]
    added_special_tokens = ['<|START_TEXT|>', '<|END_TEXT|>', '<|START_AUDIO|>', '<|END_AUDIO|>', "<|START_SPEAKER|>",
                            "<|END_SPEAKER|>", '<|START_SPEED|>', '<|END_SPEED|>', '<|SEP_AUDIO|>']
    added_special_tokens = [AddedToken(x, special=True) for x in added_special_tokens]
    tokenizer.add_tokens(added_tokens)
    tokenizer.add_tokens(added_special_tokens)


    def format_train(sample):
        input_text = \
            ''.join(
                ["<|START_SPEAKER|>"] + ['<|speaker-{}|>'.format(i + idx * 32) for idx, i in
                                         enumerate(sample['spk_codes'])] + [
                    "<|END_SPEAKER|>"]) + \
            ''.join(["<|START_SPEED|>", '<|speed-{}|>'.format(sample['speed_id']), "<|END_SPEAKER|>"]) + \
            '<|START_TEXT|>' + sample['text'] + '<|END_TEXT|>' + '<|START_AUDIO|>'
        output_text = ''.join(
            ['<|speech-{}|>'.format(i) if j % 7 != 0 else '<|SEP_AUDIO|><|speech-{}|>'.format(i) for j, i in
             enumerate(sample['target'])]) + '<|END_AUDIO|>'
        input_ids = tokenizer(input_text, add_special_tokens=False).input_ids
        output_ids = tokenizer(output_text, add_special_tokens=False).input_ids
        labels = [-100] * len(input_ids) + output_ids
        input_ids = input_ids + output_ids
        return {"input_ids": input_ids, 'labels': labels}


    ds = ds.map(format_train, num_proc=24, remove_columns=list(ds.column_names))
    ds.to_parquet(sys.argv[3])
    if len(sys.argv) == 5:
        tokenizer.save_pretrained(sys.argv[4])
        model = AutoModelForCausalLM.from_pretrained(sys.argv[1])
        model.resize_position_embeddings(len(tokenizer))
        model.save_pretrained(sys.argv[4])
