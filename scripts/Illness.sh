export CUDA_VISIBLE_DEVICES=0
cd ..
seq_len=96
model_name=RAFT

for pred_len in 96 192 336 720
do
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./data/forecasting \
      --data_path national_illness.csv \
      --model_id Illness_$seq_len_$pred_len \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len $seq_len \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --freq d \
      --itr 1
done
