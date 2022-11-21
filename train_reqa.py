'''
This script handling the training process.
'''
from genericpath import exists
import warnings
from ann import construct_batch

from config import parse_arguments

warnings.filterwarnings("ignore")
import os
import argparse
import math
import time
import json
import copy
import random
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
from logger import Logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def train_epoch(model, train_data_loader, optimizer, scheduler, epoch_i, args):
    model.train()
    total_tr_loss = 0.0
    total_tr_difficulty = 0.0
    total_train_batch = 0
    total_acc = 0.0
    start = time.time()

    for step, batch in enumerate(
            tqdm(train_data_loader, desc='  -(Training)', leave=False)):

        # forward
        tr_loss, tr_difficulty, tr_acc = model(**batch,
                                               adv_training=args.adv_training)

        # backward
        tr_loss.backward()

        # record
        total_acc += tr_acc
        total_tr_loss += tr_loss.item()
        total_tr_difficulty += tr_difficulty.item()
        total_train_batch += 1

        # update parameters
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        model.zero_grad()

    logger.info('[Epoch{epoch: d}] - (Train) Loss ={train_loss: 8.5f}, Difficulty ={diff:8.4f} %, Acc ={acc: 3.2f} %, '\
                'elapse ={elapse: 3.2f} min'.format(epoch=epoch_i,
                                                    train_loss=total_tr_loss / total_train_batch,
                                                    diff=total_tr_difficulty/total_train_batch*100,
                                                    acc=100 * total_acc / total_train_batch,
                                                    elapse=(time.time()-start)/60))


def dev_epoch(model, dev_data_loader, epoch_i, args):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    start = time.time()

    total_dev_acc = []
    question_embeddings = []
    answer_embeddings = []
    all_src_ids = []

    with torch.no_grad():
        for batch in tqdm(dev_data_loader, desc='  -(Dev)', leave=False):
            # forward
            if args.dev_metric == "p1":
                question_embedding, answer_embedding, src_ids = model(**batch)
                question_embeddings += [question_embedding]
                answer_embeddings += [answer_embedding]
                all_src_ids += [src_ids]
            elif args.dev_metric == "acc":
                _, acc = model(**batch)
                total_dev_acc += [acc]

    if args.dev_metric == "p1":
        question_embeddings = torch.cat(question_embeddings, 0)
        answer_embeddings = torch.cat(answer_embeddings, 0)
        all_src_ids = torch.cat(all_src_ids, 0)

        predict_logits = model.matching(question_embeddings, answer_embeddings,
                                        all_src_ids, False)
        predict_result = torch.argmax(predict_logits, dim=1)
        device = predict_result.device
        labels = torch.arange(0, predict_logits.shape[0], device=device)
        dev_p1 = labels == predict_result
        dev_p1 = (dev_p1.int().sum() /
                  (predict_logits.shape[0] * 1.0)).item() * 100

        logger.info('[Epoch{epoch: d}] - (Dev  ) P@1 ={p1: 3.2f} %, '\
                    'elapse ={elapse: 3.2f} min'.format(epoch=epoch_i,
                                                        p1=dev_p1,
                                                        elapse=(time.time()-start)/60))
        return dev_p1
    elif args.dev_metric == "acc":
        dev_acc = np.mean(total_dev_acc) * 100
        logger.info('[Epoch{epoch: d}] - (Dev  ) Acc ={acc: 3.2f} %, '\
                    'elapse ={elapse: 3.2f} min'.format(epoch=epoch_i,
                                                        acc=dev_acc,
                                                        elapse=(time.time()-start)/60))
        return dev_acc


class TorchWhitening():

    def __init__(self):
        pass

    def fit(self, sentence_embeddings):
        self.mu = torch.mean(sentence_embeddings, dim=0, keepdims=True)
        cov = (sentence_embeddings - self.mu).t().mm(
            (sentence_embeddings - self.mu)) / (sentence_embeddings.shape[0] -
                                                1)
        u, s, v = torch.svd(cov)
        self.W = u.mm(torch.diag(1 / torch.sqrt(s)))
        self.device = self.mu.device

    def transform(self, vecs):
        return (vecs.to(self.device) - self.mu).mm(self.W)


def obtain_whitening_params(model, data_loaders, args):
    model.eval()
    sentence_embeddings = []

    with torch.no_grad():
        for data_loader in data_loaders:
            # rebuilt the dataloader without shuffling which removes randomness
            data_loader = torch.utils.data.DataLoader(
                data_loader.dataset,
                shuffle=False,
                batch_size=args.batch_size,
                collate_fn=data_loader.dataset.collate_fn)
            for batch in tqdm(data_loader,
                              desc='[sentence encoding for whitening]',
                              leave=False):
                # forward
                src_embedding, tgt_embedding, _ = model(**batch)
                sentence_embeddings.append(src_embedding.cpu())
                sentence_embeddings.append(tgt_embedding.cpu())

    sentence_embeddings = torch.cat(sentence_embeddings, 0)

    whitening = TorchWhitening()
    whitening.fit(sentence_embeddings)

    return whitening


def whiten_sentence_embeddings(question_embeddings, candidate_embeddings,
                               whitening, args):
    return whitening.transform(question_embeddings), whitening.transform(
        candidate_embeddings)


def obtain_test_embeddings(model, test_question_data_loader,
                           test_candidate_data_loader, args):
    model.eval()
    with torch.no_grad():
        question_embeddings = []
        test_ground_truth = []
        for batch in tqdm(test_question_data_loader,
                          desc='[encoding test questions]',
                          leave=False):
            # forward
            test_ground_truth += list(batch["ground_truth"])
            del batch["ground_truth"]
            question_embedding = model.sentence_encoding(**batch,
                                                         is_query=True)
            question_embeddings.append(question_embedding)
        question_embeddings = torch.cat(question_embeddings, 0)

        candidate_embeddings = []
        for batch in tqdm(test_candidate_data_loader,
                          desc='[encoding test candidates]',
                          leave=False):
            # forward
            candidate_embedding = model.sentence_encoding(**batch,
                                                          is_query=False)
            candidate_embeddings.append(candidate_embedding)
        candidate_embeddings = torch.cat(candidate_embeddings, 0)

    return question_embeddings, candidate_embeddings, test_ground_truth


def test(model, question_embeddings, candidate_embeddings, test_ground_truth,
         used_dimension, seed, fold_idx):
    if used_dimension != -1:
        question_embeddings = question_embeddings[:, :used_dimension]
        candidate_embeddings = candidate_embeddings[:, :used_dimension]

    predict_logits = model.matching(question_embeddings, candidate_embeddings,
                                    None, False).cpu().numpy()

    p_counts = {1: 0.0}
    r_counts = {5: 0.0, 10: 0.0}
    mrr_counts = 0

    for idx in range(len(test_ground_truth)):
        pred = np.argsort(-predict_logits[idx]).tolist()

        # precision at K
        for rank in p_counts.keys():
            numerator = 0.0
            denominator = rank
            for gt in test_ground_truth[idx]:
                if numerator == rank:
                    break
                if gt in pred[:rank]:
                    numerator += 1
            p_counts[rank] += numerator / denominator

        # recall at K
        denominator = len(test_ground_truth[idx])
        for rank in r_counts.keys():
            numerator = 0.0
            for gt in test_ground_truth[idx]:
                if numerator == rank:
                    break
                if gt in pred[:rank]:
                    numerator += 1
            r_counts[rank] += numerator / denominator

        # mrr
        for r, p in enumerate(pred):
            if p in test_ground_truth[idx]:
                mrr_counts += 1 / (r + 1)
                break

    mrr = np.round(mrr_counts / len(test_ground_truth), 4) * 100
    p_at_k = {
        k: np.round(v / len(test_ground_truth), 4) * 100
        for k, v in p_counts.items()
    }
    r_at_k = {
        k: np.round(v / len(test_ground_truth), 4) * 100
        for k, v in r_counts.items()
    }

    logger.info('[Seed: {seed:d}][KFold {fold_idx:d}] - (Test ) MRR ={mrr: 3.2f} %, P@1 ={p1: 3.2f} %,'\
            ' R@5 ={r5: 3.2f} %, R@10 ={r10: 3.2f} %'.format(seed=seed,
                                                           fold_idx=fold_idx,
                                                           mrr=mrr,
                                                           p1=p_at_k[1],
                                                           r5=r_at_k[5],
                                                           r10=r_at_k[10]))

    return mrr, p_at_k[1], r_at_k[5], r_at_k[10]


def run(model, train_data_loader, dev_data_loader, test_question_data_loader,
        test_candidate_data_loader, args, seed, results):
    # Prepare optimizer and schedule (linear warmup and decay)
    args.num_train_instances = train_data_loader.dataset.__len__()
    args.num_training_steps = math.ceil(
        args.num_train_instances / args.batch_size) * args.epoch
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params': [
            p for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ],
        'weight_decay':
        args.weight_decay
    }, {
        'params': [
            p for n, p in model.named_parameters()
            if any(nd in n for nd in no_decay)
        ],
        'weight_decay':
        0.0
    }]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.ceil(args.warmup_proportion *
                                   args.num_training_steps),
        num_training_steps=args.num_training_steps)

    best_metrics = 0
    best_epoch = 0
    if args.do_train == "True":
        for epoch_i in range(1, args.epoch + 1):
            logger.info('[Epoch {}]'.format(epoch_i))
            train_data_loader.dataset.shuffle()
            if args.ann_cnt > 0:
                construct_batch(model,
                                logger,
                                train_data_loader,
                                args,
                                epoch_i,
                                sparse=args.sparse)
            train_epoch(model, train_data_loader, optimizer, scheduler,
                        epoch_i, args)

            dev_score = dev_epoch(model, dev_data_loader, epoch_i, args)

            current_metrics = dev_score

            model_name = args.save_model_path + f'/{seed}_{args.fold_idx}.pt'
            if not os.path.exists(args.save_model_path):
                os.makedirs(args.save_model_path)
            if current_metrics >= best_metrics:
                best_epoch = epoch_i
                best_metrics = current_metrics
                save_dict = {
                    "model_state_dict": model.state_dict(),
                }
                torch.save(save_dict, model_name)
                logger.info('  - [Info] The checkpoint file has been updated.')
        # results[seed]['dev']['p1'].append(current_metrics)
        logger.info(
            f'Got best test performance on epoch{best_epoch}, P@1 = {best_metrics:3.2f} %'
        )

    if args.do_test == "True":
        logger.info(f'Conduct evaluation on test dataset')
        if args.do_train == "True":
            model.load_state_dict(torch.load(model_name)["model_state_dict"])
            model.to(args.device)
            logger.info('reload best checkpoint')

        # normal evaluation
        question_embeddings, candidate_embeddings, test_ground_truth = obtain_test_embeddings(
            model, test_question_data_loader, test_candidate_data_loader, args)
        # import pdb; pdb.set_trace()
        test_mrr, test_p1, test_r5, test_r10 = test(model, question_embeddings,
                                                    candidate_embeddings,
                                                    test_ground_truth, -1,
                                                    seed, args.fold_idx)
        results[seed]['normal']['mrr'].append(test_mrr)
        results[seed]['normal']['p1'].append(test_p1)
        results[seed]['normal']['r5'].append(test_r5)
        results[seed]['normal']['r10'].append(test_r10)

        if args.whitening == "True":
            # evaluation after target whitening
            whitening = obtain_whitening_params(model, [train_data_loader],
                                                args)
            if not os.path.exists(args.save_model_path):
                os.makedirs(args.save_model_path)
            save_dict = {
                "model_state_dict": model.state_dict(),
                "whitening_params": whitening
            }
            torch.save(save_dict, model_name)
            whitened_question_embeddings, whitened_candidate_embeddings = whiten_sentence_embeddings(
                question_embeddings, candidate_embeddings, whitening, args)

            test_mrr, test_p1, test_r5, test_r10 = test(
                model, whitened_question_embeddings,
                whitened_candidate_embeddings, test_ground_truth, -1, seed,
                args.fold_idx)
            results[seed]['whiten']['mrr'].append(test_mrr)
            results[seed]['whiten']['p1'].append(test_p1)
            results[seed]['whiten']['r5'].append(test_r5)
            results[seed]['whiten']['r10'].append(test_r10)

    if args.do_train == "True" and args.rm_saved_model == "True":
        import shutil
        shutil.rmtree(args.save_model_path)


def prepare_dataloaders(args):
    from utils_data import BioASQKFDataset as Dataset
    # initialize datasets
    train_dataset = Dataset(args, split="train", logger=logger)
    dev_dataset = Dataset(args, split="dev", logger=logger)
    test_question_dataset = Dataset(args,
                                    split="test",
                                    data_type="question",
                                    logger=logger)
    test_candidate_dataset = Dataset(args,
                                     split="test",
                                     data_type="candidate",
                                     logger=logger)

    logger.info(f"train data size: {train_dataset.__len__()}")
    logger.info(f"dev data size: {dev_dataset.__len__()}")
    logger.info(f"test question size: {test_question_dataset.__len__()}")
    logger.info(f"test candidate size: {test_candidate_dataset.__len__()}")

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=train_dataset.collate_fn)

    dev_data_loader = torch.utils.data.DataLoader(
        dev_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=dev_dataset.collate_fn)

    test_question_data_loader = torch.utils.data.DataLoader(
        test_question_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=test_question_dataset.collate_fn)

    test_candidate_data_loader = torch.utils.data.DataLoader(
        test_candidate_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=test_candidate_dataset.collate_fn)

    return train_data_loader, dev_data_loader, test_question_data_loader, test_candidate_data_loader


def prepare_model(args):
    if args.model_type == "dual_encoder":
        from models.dual_encoder import RankModel
    elif args.model_type == "dual_encoder_wot":
        from models.dual_encoder_wot import RankModel

    model = RankModel(args)
    if args.load_model_path != "":
        pretrained_model_dict = torch.load(
            args.load_model_path.format(
                fold_idx=args.fold_idx))["model_state_dict"]
        model.load_state_dict(pretrained_model_dict, strict=False)
        logger.info(f"load model successfully from {args.load_model_path}")
    model.to(args.device)
    return model


def main():
    ''' Main function '''
    global logger

    args = parse_arguments()
    if args.dataset == "squad":
        args.plm_path = 'bert-base-uncased'
    log_dir = args.log_dir = args.log_dir + '/' + args.dataset
    result_dir = args.result_dir = args.result_dir + '/' + args.dataset
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    log_file = f'{args.ann_cnt}'
    if args.strategy == 'la':
        log_file += f'_{args.la_type}{args.la_iter_count}'
    if args.adv_training:
        log_file += f'_adv{args.adv_norm}'
    result_file = os.path.join(result_dir, log_file + '.json')
    log_file += '.log'
    logger = Logger(os.path.join(log_dir, log_file)).logger

    logger.info(args)

    # set device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(result_file):
        with open(result_file) as f:
            results = json.load(f)
    else:
        # record results
        results = dict()

    for seed in args.seeds:
        if seed in results or str(seed) in results:
            logger.info(f"Seed {seed} exists, skip!")
            if str(seed) in results:
                results[seed] = results[str(seed)]
                results.pop(str(seed))
            continue
        results[seed] = {"normal": defaultdict(list)}
        if args.whitening == "True":
            results[seed]["whiten"] = defaultdict(list)
        # prepare train dev and test
        num_train_questions = {
                "6b": 2251,
                "7b": 2747,
                "8b": 3243,
                "9b": 3742,
                "squad": 87599
            }[args.dataset]
        if args.dataset == "squad":
            num_dev_questions = int(num_train_questions * 0.1)
            all_ids = [i for i in range(num_train_questions)]
            random.seed(seed)
            dev_ids = random.sample(all_ids, num_dev_questions)
            train_ids = list(set(all_ids) - set(dev_ids))
            kf = [(train_ids, dev_ids)]
        else:
            from sklearn.model_selection import KFold
            all_idx = list(range(num_train_questions))
            kf = KFold(n_splits=5, shuffle=True, random_state=seed).split(all_idx)
            kf = [(train_ids.tolist(), dev_ids.tolist())
                for train_ids, dev_ids in kf]
        # start training fold by fold
        for fold_idx, (train_ids, dev_ids) in enumerate(kf):
            fold_idx += 1
            if args.spec_fold_idx != 0 and args.spec_fold_idx != fold_idx:
                continue
            # set seed
            set_seed(seed)
            logger.info('\n')
            logger.info(f"Current random seed: {seed}")
            logger.info(f"The {fold_idx}th fold:")
            args.fold_idx = fold_idx
            args.train_ids = train_ids
            args.dev_ids = dev_ids
            # loading dataset
            train_data_loader, dev_data_loader, test_question_data_loader, test_candidate_data_loader = prepare_dataloaders(
                args)

            # preparing model
            model = prepare_model(args)

            # running
            run(model, train_data_loader, dev_data_loader,
                test_question_data_loader, test_candidate_data_loader, args,
                seed, results)
            # break
        # results[seed]["dev"]["p1"] = np.mean(results[seed]["dev"]["p1"])
        # logger.info(
        #     f'[Seed: {seed:d}][Dev] P@1 ={results[seed]["dev"]["p1"]: 3.2f} % '
        # )
        if args.do_test == "True":
            # logger.info(f"Current random seed: {seed}")
            results[seed]["normal"]["mrr"] = np.mean(
                results[seed]["normal"]["mrr"])
            results[seed]["normal"]["p1"] = np.mean(
                results[seed]["normal"]["p1"])
            results[seed]["normal"]["r5"] = np.mean(
                results[seed]["normal"]["r5"])
            results[seed]["normal"]["r10"] = np.mean(
                results[seed]["normal"]["r10"])
            if args.whitening == "True":
                results[seed]["whiten"]["mrr"] = np.mean(
                    results[seed]["whiten"]["mrr"])
                results[seed]["whiten"]["p1"] = np.mean(
                    results[seed]["whiten"]["p1"])
                results[seed]["whiten"]["r5"] = np.mean(
                    results[seed]["whiten"]["r5"])
                results[seed]["whiten"]["r10"] = np.mean(
                    results[seed]["whiten"]["r10"])

            logger.info(f'[Seed: {seed:d}][Normal] MRR ={results[seed]["normal"]["mrr"]: 3.2f} %, '\
                                               f'P@1 ={results[seed]["normal"]["p1"]: 3.2f} %, '\
                                               f'R@5 ={results[seed]["normal"]["r5"]: 3.2f} %, '\
                                               f'R@10 ={results[seed]["normal"]["r10"]: 3.2f} %.')
            if args.whitening == "True":
                logger.info(f'[Seed: {seed:d}][Whiten] MRR ={results[seed]["whiten"]["mrr"]: 3.2f} %, '\
                                                    f'P@1 ={results[seed]["whiten"]["p1"]: 3.2f} %, '\
                                                    f'R@5 ={results[seed]["whiten"]["r5"]: 3.2f} %, '\
                                                    f'R@10 ={results[seed]["whiten"]["r10"]: 3.2f} %.')
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=4)

    results["overall"] = dict()
    results["overall"]['seeds'] = args.seeds
    results["overall"]["normal"] = defaultdict(dict)
    results["overall"]["normal"]["mrr"]["mean"] = np.mean(
        [results[seed]["normal"]["mrr"] for seed in args.seeds])
    results["overall"]["normal"]["mrr"]["std"] = np.std(
        [results[seed]["normal"]["mrr"] for seed in args.seeds])
    results["overall"]["normal"]["p1"]["mean"] = np.mean(
        [results[seed]["normal"]["p1"] for seed in args.seeds])
    results["overall"]["normal"]["p1"]["std"] = np.std(
        [results[seed]["normal"]["p1"] for seed in args.seeds])
    results["overall"]["normal"]["r5"]["mean"] = np.mean(
        [results[seed]["normal"]["r5"] for seed in args.seeds])
    results["overall"]["normal"]["r5"]["std"] = np.std(
        [results[seed]["normal"]["r5"] for seed in args.seeds])
    results["overall"]["normal"]["r10"]["mean"] = np.mean(
        [results[seed]["normal"]["r10"] for seed in args.seeds])
    results["overall"]["normal"]["r10"]["std"] = np.std(
        [results[seed]["normal"]["r10"] for seed in args.seeds])

    normal_results = results["overall"]["normal"]
    logger.info(f'[Overall][Normal] MRR ={results["overall"]["normal"]["mrr"]["mean"]: 3.2f}  ± {normal_results["mrr"]["std"]: 3.2f} %, '\
                                               f'P@1 ={results["overall"]["normal"]["p1"]["mean"]: 3.2f}  ± {normal_results["p1"]["std"]: 3.2f} %, '\
                                               f'R@5 ={results["overall"]["normal"]["r5"]["mean"]: 3.2f}  ± {normal_results["r5"]["std"]: 3.2f} %, '\
                                               f'R@10 ={results["overall"]["normal"]["r10"]["mean"]: 3.2f}  ± {normal_results["r10"]["std"]: 3.2f} %.')
    if args.whitening == "True":
        results["overall"]["whiten"] = defaultdict(dict)
        results["overall"]["whiten"]["mrr"]["mean"] = np.mean(
            [results[seed]["whiten"]["mrr"] for seed in args.seeds])
        results["overall"]["whiten"]["mrr"]["std"] = np.std(
            [results[seed]["whiten"]["mrr"] for seed in args.seeds])
        results["overall"]["whiten"]["p1"]["mean"] = np.mean(
            [results[seed]["whiten"]["p1"] for seed in args.seeds])
        results["overall"]["whiten"]["p1"]["std"] = np.std(
            [results[seed]["whiten"]["p1"] for seed in args.seeds])
        results["overall"]["whiten"]["r5"]["mean"] = np.mean(
            [results[seed]["whiten"]["r5"] for seed in args.seeds])
        results["overall"]["whiten"]["r5"]["std"] = np.std(
            [results[seed]["whiten"]["r5"] for seed in args.seeds])
        results["overall"]["whiten"]["r10"]["mean"] = np.mean(
            [results[seed]["whiten"]["r10"] for seed in args.seeds])
        results["overall"]["whiten"]["r10"]["std"] = np.std(
            [results[seed]["whiten"]["r10"] for seed in args.seeds])

        whiten_results = results["overall"]["whiten"]
        logger.info(f'[Overall][Whiten] MRR ={results["overall"]["whiten"]["mrr"]["mean"]: 3.2f}  ± {whiten_results["mrr"]["std"]: 3.2f} %, '\
                                                    f'P@1 ={results["overall"]["whiten"]["p1"]["mean"]: 3.2f}  ± {whiten_results["p1"]["std"]: 3.2f} %, '\
                                                    f'R@5 ={results["overall"]["whiten"]["r5"]["mean"]: 3.2f}  ± {whiten_results["r5"]["std"]: 3.2f} %, '\
                                                    f'R@10 ={results["overall"]["whiten"]["r10"]["mean"]: 3.2f}  ± {whiten_results["r10"]["std"]: 3.2f} %.')
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)
    if args.save_results == "True":
        if not os.path.exists(args.save_model_path):
            os.makedirs(args.save_model_path)
        with open(args.save_model_path + "/results.json",
                  'w',
                  encoding='utf-8') as f:
            json.dump(results, f, indent=4)


if __name__ == '__main__':
    main()