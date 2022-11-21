import math
import random
import time
import torch
from tqdm import tqdm
from assign import linear_assignment, sparse_linear_assignment
import numpy as np
from scipy.sparse import csr_matrix, vstack, coo_matrix


@torch.no_grad()
def construct_batch(model, logger, train_loader, args, epoch=0, sparse=False):
    if sparse:
        return sparse_construct_batch(model, logger, train_loader, args, epoch)
    model.eval()

    dataset = train_loader.dataset
    N = len(dataset)
    ann_index = []
    device = args.device
    time_st = time.time()
    src_embeddings = []
    tgt_embeddings = []
    src_ids = []
    for batch in tqdm(train_loader, desc='  -(Prepare)', leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        # forward
        src_embedding, tgt_embedding, _ = model(**batch)

        src_embeddings.append(src_embedding)
        tgt_embeddings.append(tgt_embedding)
        src_ids.append(batch['src_ids'])
    src_ids = torch.cat(src_ids)
    tgt_embeddings = torch.cat(tgt_embeddings, 0).T
    src_embeddings = torch.cat(src_embeddings, 0)
    logits = src_embeddings.mm(tgt_embeddings)
    mask = torch.eye(N, dtype=torch.bool, device=device)  # [N,N]
    false_negative_mask = (src_ids.unsqueeze(1).repeat(
        1, N) == src_ids.unsqueeze(0).repeat(N, 1)).float() - torch.eye(N).to(
            src_ids.device)
    false_negative_mask = false_negative_mask.bool() ^ mask
    logits = logits.masked_fill(false_negative_mask, float('-inf'))
    if args.matching_func == 'cos':
        logits = logits / args.temperature
        logits = logits.softmax(dim=-1)
    elif args.matching_func == 'dot':
        logits = logits.softmax(dim=-1)
    logits.masked_fill_(mask, 0)
    logits = (logits + logits.T) / 2
    logits = logits.cpu()
    cluster_st = time.time()
    if args.strategy == 'la':
        la_type = args.la_type
        cluster, difficulty = linear_assignment(logits,
                                                args.ann_cnt,
                                                times=args.la_iter_count,
                                                type=la_type)
        cluster = list(filter(lambda x: len(x) > 0, cluster))

        sizes = [len(x) for x in cluster]
        logger.info(
            f'[Epoch{epoch: d}] - (Prepare) N_Clusters {len(cluster)}, Size {np.mean(sizes):.2f} ± {np.std(sizes):.2f}, Elapse: {(time.time()-cluster_st)/60:.2f} min'
        )

    random.shuffle(cluster)
    cluster = [idx for c in cluster for idx in c]
    ann_index = cluster
    logger.info(
        f'[Epoch{epoch: d}] - (Prepare) Difficulty: {difficulty* 100:.2f} %, Total Elapse: {(time.time()-time_st)/60:.2f} min'
    )

    dataset.reorder(ann_index)
    torch.cuda.empty_cache()
    # dataset.check()
    return difficulty


@torch.no_grad()
def sparse_construct_batch(model, logger, train_loader, args, epoch=0, k=1000):
    model.eval()

    dataset = train_loader.dataset
    N = len(dataset)
    ann_index = []
    device = args.device
    time_st = time.time()
    time_st = time.time()
    src_embeddings = []
    tgt_embeddings = []
    for batch in tqdm(train_loader, desc='  -(Prepare)', leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        # forward
        src_embedding, tgt_embedding, _ = model(**batch)

        src_embeddings.append(src_embedding)
        tgt_embeddings.append(tgt_embedding)
    tgt_embeddings = torch.cat(tgt_embeddings, 0).T
    batch_size = args.batch_size

    cluster_st = time.time()
    if args.strategy == 'la':
        shift = torch.zeros((batch_size, batch_size),
                            device=device,
                            dtype=torch.bool)
        mask = torch.cat(
            [torch.eye(batch_size),
             torch.zeros((batch_size, N - batch_size))],
            dim=1)  # [bsz,N]
        mask = mask.to(device).bool()
        logits = []
        for batch in tqdm(src_embeddings):
            mask = mask[:batch.size(0)]
            shift = shift[:batch.size(0)]
            batch_logits = batch.mm(tgt_embeddings)
            groundtruth_scores = torch.masked_select(batch_logits,
                                                     mask).unsqueeze(1)
            false_negative_mask = batch_logits.eq(groundtruth_scores) ^ mask
            batch_logits = batch_logits.masked_fill(false_negative_mask,
                                                    float('-inf'))
            if args.matching_func == 'dot':
                batch_logits = batch_logits.softmax(dim=-1)
            topk_val = batch_logits.topk(k=k, dim=-1).values[:, -1:]
            batch_logits.masked_fill_(batch_logits <= topk_val, 0)
            batch_logits.masked_fill_(mask, 0)
            logits.append(csr_matrix(batch_logits.cpu().numpy()))
            mask = torch.cat([shift, mask], dim=1)[:, :N]
        logits = vstack(logits, format='csr')
        logits = (logits + logits.transpose()) / 2

        cluster, difficulty = sparse_linear_assignment(
            logits, args.ann_cnt, times=args.la_iter_count, type=args.la_type)
        cluster = list(filter(lambda x: len(x) > 0, cluster))
        sizes = [len(x) for x in cluster]
        logger.info(
            f'[Epoch{epoch: d}] - (Prepare) La N_Clusters {len(cluster)}, Size {np.mean(sizes):.2f} ± {np.std(sizes):.2f}, Elapse: {(time.time()-cluster_st)/60:.2f} min'
        )

    random.shuffle(cluster)
    cluster = [idx for c in cluster for idx in c]
    ann_index = cluster
    difficulty = difficulty * 100
    logger.info(
        f'[Epoch{epoch: d}] - (Prepare) Difficulty: {difficulty:.2f} %, Total Elapse: {(time.time()-time_st)/60:.2f} min'
    )

    dataset.reorder(ann_index)
    torch.cuda.empty_cache()
