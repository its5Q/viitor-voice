## step1: Download dataset and preprocess models
```commandline
git lfs install

mkdir tool_models
cd tool_models
git clone https://huggingface.co/hubertsiuzdak/snac_24khz
wget https://huggingface.co/model-scope/CosyVoice-300M/resolve/main/campplus.onnx
git clone https://huggingface.co/Qwen/Qwen2-0.5B
cd ..

mkdir datasets
cd datasets
git clone https://huggingface.co/datasets/mythicinfinity/libritts_r
cd ..
```

## step2: Generate speaker token, speed token and snac tokens
```
python preprocess.py
```

## step3: Prepare dataset, model, tokenizer
```commandline
python preprocess_raw_ids.py tool_models/Qwen2-0.5B datasets/libritts_r_tokenized.json datasets/libritts_r_tokenized.parquet tool_models/Qwen2-0.5B-snac
```

## step4: start training
```
deepspeed --include localhost:0,1 train.py     \
    --model_path tool_models/Qwen2-0.5B-snac \
    --model_type decoder     \
    --data_type parquet     \
    --data_file "datasets/libritts_r_tokenized.parquet"     \
    --streaming false     \
    --max_length 2048 \
    --merge_inputs false     \
    --output_dir checkpoints/demo     \
    --deepspeed ds1.json     \
    --per_device_train_batch_size=4     \
    --gradient_accumulation_steps=32     \
    --per_device_eval_batch_size=4     \
    --save_steps 5000 \
    --save_total_limit 3     \
    --learning_rate 1e-4     \
    --logging_steps 10     \
    --num_train_epochs 100 \
    --group_by_length false \
    --torch_compile false     \
    --do_train \
    --overwrite_output_dir \
    --bf16 \
    --remove_unused_columns false \
    --preprocessing_num_workers 24 \
    --save_safetensors false \
    --split_batches false \
    --lr_scheduler_type "cosine_with_restarts" \
    --warmup_steps 2000 \
    --lr_scheduler_kwargs '{"num_cycles": 5}'
```