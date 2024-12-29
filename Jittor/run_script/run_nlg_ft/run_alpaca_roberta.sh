# finetune

export TASK_NAME=alpaca
export DATASET_NAME=alpaca
export CUDA_VISIBLE_DEVICES=0

bs=4
lr=7e-5
dropout=0.1
psl=8
epoch=12

python3 run.py \
  --model_name_or_path "roberta-large" \
  --train_data_path "tasks/alpaca/alpaca_data_train.json" \
  --eval_data_path "tasks/alpaca/alpaca_data_eval.json" \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 500 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --output_dir checkpoints/$DATASET_NAME-roberta/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 11 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --fp16
