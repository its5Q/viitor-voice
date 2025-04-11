## step1: Download dataset and base models
```bash
mkdir tool_models
cd tool_models
git clone https://huggingface.co/Qwen/Qwen2.5-0.5B
cd ..


mkdir datasets
cd datasets
mkdir emilia
wget "https://huggingface.co/datasets/amphion/Emilia-Dataset/resolve/main/ZH/ZH-B000000.tar" -O emilia/ZH-B000000.tar
cd ..
```

## step2: preprocess model
```bash
python viitor_voice/preprocess/preprocess_model.py tool_models/Qwen2.5-0.5B tool_models/Qwen2.5-0.5B-snac
```

## step3: preprocess dataset
```bash
python preprocess_dataset.py

python pack_datasets.py \
    --tokenizer_path ./tool_models/Qwen2.5-0.5B-snac \
    --data_type json \
    --data_file "big_ru_book3.json" \
    --max_length 3072 \
    --save_path big_ru_book3.parquet
```

## step4: start training

accelerate train.py         --model_path tool_models/Qwen2.5-0.5B-snac     --model_type decoder         --data_type parquet         --data_file "big_ru_book3.parquet"         --streaming false         --max_length 3072     --merge_inputs false         --output_dir checkpoints/poc2         --per_device_train_batch_size=4         --gradient_accumulation_steps=16       --save_steps 400     --save_total_limit 3         --learning_rate 1e-4         --logging_steps 1         --num_train_epochs 50     --group_by_length false     --torch_compile False         --do_train     --overwrite_output_dir     --bf16     --remove_unused_columns false     --preprocessing_num_workers 24     --save_safetensors false     --split_batches false     --lr_scheduler_type "cosine_with_restarts"     --warmup_steps 1000     --lr_scheduler_kwargs '{"num_cycles": 5}'  --use_liger_kernel true  --optim adamw_torch_fused

