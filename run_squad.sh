export  CUDA_VISIBLE_DEVICES=0
la_type='random'
iter_cnt=4
ann_cnt=8
for dataset in 'squad'
do
    python3 train_reqa.py \
        --seeds 666 \
        --do_train True \
        --do_test True \
        --dev_metric p1 \
        --dataset ${dataset} \
        --max_question_len 24 \
        --max_answer_len 168 \
        --epoch 10 \
        --batch_size 32 \
        --model_type dual_encoder \
        --encoder_type biobert \
        --plm_path 'bert-base-uncased' \
        --pooler_type mean \
        --matching_func 'dot' \
        --whitening False \
        --temperature 1 \
        --learning_rate 5e-5 \
        --save_model_path output/9b/biobert_baseline/ \
        --rm_saved_model True \
        --save_results True \
        --la_type ${la_type} \
        --la_iter_count ${iter_cnt} \
        --log_dir logs/ \
        --result_dir results/ \
        --ann_cnt ${ann_cnt} \
        --sparse
done
# --adv_training \
# --adv_norm 0.001
