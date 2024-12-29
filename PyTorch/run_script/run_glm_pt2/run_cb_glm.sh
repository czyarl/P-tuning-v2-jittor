export TASK_NAME=superglue
export DATASET_NAME=cb
export CUDA_VISIBLE_DEVICES=0

# bs=32
# bs=16
# bs=8
bs=4
# lr=7e-3 # 0.5
# lr=5e-4 # 0.5
# lr=3e-4 # 0.5
# lr=1e-4 # 0.73
lr=9e-5 # 0.78
# lr=8e-5 # 0.60
# lr=7e-5 # 0.625
# lr=5e-5 # 0.66
# lr=1e-5 # 0.64285
# lr=7e-6 # 0.57
dropout=0.1
# psl=8
# psl=64
# psl=128
psl=64
# psl=256
epoch=400

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
  --output_dir checkpoints/$DATASET_NAME-glm-pt2/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 11 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --prefix \
  --fp16 
