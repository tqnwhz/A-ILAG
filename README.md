# Biomedical Retrieval Question Answering
This repository provides the PyTorch implementation of our paper "Enhancing Biomedical ReQA with AdversarialHard In-batch Negative Samples".

## Environment
We mainly rely on these packages:
```bash
# Install huggingface transformers and pytorch
transformers==4.23.1
torch==1.12.1
```
Our experiment is conducted on a NVIDIA RTX 3090 24G with CUDA version 11.6.

## Dataset
We follow [RBAR](https://github.com/Ba1Jun/BioReQA) to process data. The dataset and scripts can be found in their repo.

## Models
We use the following version of BioBERT in PyTorch from Transformers which achieves the best performance in our experiments, while the other pre-trained language models can also be used in our framework.
* [`dmis-lab/biobert-base-cased-v1.1`](https://huggingface.co/dmis-lab/biobert-base-cased-v1.1): BioBERT-Base v1.1 (+ PubMed 1M)


## Example

We show an example running script on ReQA BioASQ 9b dataset. Other experiments can also be reproduced by using the hyper-parameters provided in the paper.

### Transforming BioASQ dataset into ReQA BioASQ dataset
As introduced in [RBAR](https://github.com/Ba1Jun/BioReQA), the first thing is to trasform BioASQ dataset into ReQA format.
```bash
# Transform the BioASQ 9b dataset into ReQA BioASQ 9b.
python3 reqa_bioasq_data_processing.py --dataset 9b
```

### ReQA BioASQ dataset
Run the script "run.sh" to get result of `A-ILAG`. By removing `adv_training` flag and setting `ann_cnt` to 0, you can get results of `w/o adv` and `w/o ILAG` respectively. 
```bash
export  CUDA_VISIBLE_DEVICES=0
la_type='greedy-random'
iter_cnt=4
ann_cnt=8
for dataset in '9b'
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
        --plm_path dmis-lab/biobert-base-cased-v1.1 \
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
        --adv_training \
        --adv_norm 0.005
done
```

For SQuAD dataset, similarly , you can run the following command. The `sparse` flag controls whether to store the difficulty matrix in sparse format to save RAM, its logic can be seen in the code. For sparse matrix, we suggest using random initialization since it's more efficient without column indexing. 
```
bash ./run_squad.sh
```
## License and Disclaimer
Please see the LICENSE file for details. Downloading data indicates your acceptance of our disclaimer.


## Contact
For help or issues, please create an issue.