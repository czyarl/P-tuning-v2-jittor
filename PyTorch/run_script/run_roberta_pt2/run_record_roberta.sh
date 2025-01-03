export TASK_NAME=superglue
export DATASET_NAME=record
export CUDA_VISIBLE_DEVICES=0

bs=8
lr=7e-3
dropout=0.1
psl=8
epoch=100

python3 run.py \
  --model_name_or_path roberta-large \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 256 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --output_dir checkpoints/$DATASET_NAME-roberta-pt2/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 11 \
  --save_strategy no \
  --evaluation_strategy steps \
  --eval_steps 500 \
  --max_eval_samples 2000 \
  --prefix \
  --fp16