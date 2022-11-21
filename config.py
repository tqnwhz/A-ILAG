import argparse
from pytorch_lightning import Trainer


def parse_arguments():
    ''' Main function '''
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--data_file', type=str)
    parser.add_argument('--spec_fold_idx', type=int, default=0)
    parser.add_argument('--max_question_len', type=int)
    parser.add_argument('--max_answer_len', type=int)
    parser.add_argument("--overwrite_cache", action="store_true")
    # training
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--do_train", type=str)
    parser.add_argument("--do_test", type=str)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument('--save_results', type=str)
    # evalaution
    parser.add_argument("--dev_metric", type=str)
    # model
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--encoder_type', type=str)
    parser.add_argument('--plm_path', type=str)
    parser.add_argument('--pooler_type', type=str)
    parser.add_argument("--matching_func", type=str)
    parser.add_argument('--load_model_path', type=str, default="")
    parser.add_argument('--save_model_path', type=str, default="")
    parser.add_argument("--whitening", type=str, default="True")
    parser.add_argument("--rm_saved_model", type=str, default="True")
    parser.add_argument('--temperature', type=float)

    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--result_dir', type=str, default='./results')

    parser.add_argument('--ann_cnt', type=int, default=0)
    parser.add_argument('--la_iter_count', type=int, default=4)
    parser.add_argument('--la_type', type=str, default='random')
    parser.add_argument('--strategy', type=str, default='la')

    parser.add_argument('--adv_training', action='store_true')
    parser.add_argument("--adv_norm", type=float, default=0.005)
    parser.add_argument('--sparse', action='store_true')

    args = parser.parse_args()
    return args