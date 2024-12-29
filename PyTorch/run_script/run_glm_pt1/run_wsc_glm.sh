export TASK_NAME=superglue
export DATASET_NAME=wsc
export CUDA_VISIBLE_DEVICES=0

# bs=8
bs=12 # with lr=1e-3 psl=128 => 65.38 (epoch 25, key-value style)
# bs=16 # with configuration above => 67.3 (epoch 25)
# bs=4
# lr=2e-2
# lr=1e-2
# lr=5e-3
# lr=2e-3
lr=1e-3 # 64.4
# lr=5e-4
# lr=1e-4
# lr=9e-4
# lr=1e-6
dropout=0.1
# psl=8
psl=128
epoch=100
# epoch=5

python3 run.py \
  --model_name_or_path "model/glm2b" \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --output_dir checkpoints/$DATASET_NAME-glm-pt1/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 44 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --prompt \
  --fp16