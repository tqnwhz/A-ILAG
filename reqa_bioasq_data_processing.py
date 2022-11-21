import json
import torch
import random
import argparse
import logging
import numpy as np
from collections import defaultdict, OrderedDict
from utils_data import infer_sentence_breaks
from nltk.tokenize import word_tokenize


logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def break_sentences(text):
    '''
    splitting a text into a set of sentences
    '''
    sent_locs = list(infer_sentence_breaks(text))
    return [text[st:ed].strip() for (st,ed) in sent_locs]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()

    # reading original bioasq training dataset
    with open(f'data/BioASQ/{args.dataset}/training{args.dataset}.json') as data_file:
        train_items = json.load(data_file)['questions']
    
    # extracting questions and their answers
    train_question2answers = OrderedDict()
    for item in train_items:
        question = item['body']
        if not question in train_question2answers:
            train_question2answers[question] = []
        for answer in item['ideal_answer']:
            if not answer in train_question2answers[question]:
                train_question2answers[question] += [answer]
    train_question2answers = [(k, v) for k, v in train_question2answers.items()]
    logger.info(f"training questions size: {len(train_question2answers)}")

    # extracting question-answer pairs
    train_questions = []
    train_answers = []
    train_question_ids = []
    for i, item in enumerate(train_question2answers):
        question = item[0]
        for answer in item[1]:
            train_questions += [question]
            train_answers += [answer]
            train_question_ids += [i]
    logger.info(f"training question-answer pairs size: {len(train_questions)}")

    # saving the training dataset of reqa bioasq
    train_data = {
        'questions': train_questions,
        'answers': train_answers,
        'question_ids': train_question_ids,
    }
    with open(f'data/BioASQ/{args.dataset}/train.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f)
    logger.info(f"the training dataset of reqa bioasq {args.dataset} saved")

    # reading original bioasq test dataset
    test_items = []
    for batch_id in range(1, 6):
        with open(f'data/BioASQ/{args.dataset}/{args.dataset.upper()}{batch_id}_golden.json') as data_file:
            test_items += json.load(data_file)['questions']
    
    # reading the abstracts relevant to each question that will used to construct the candidates pool
    abstract_dict = OrderedDict()
    for batch_id in range(1, 6):
        with open(f'data/BioASQ/{args.dataset}/{args.dataset.upper()}{batch_id}_abstracts.json') as data_file:
            sub_abstract_dict = json.load(data_file)
            abstract_dict.update(sub_abstract_dict)
    
    # performing sentence breaking on abstracts
    for k, v in abstract_dict.items():
        abstract_dict[k] = list(break_sentences(abstract_dict[k]))

    # extracting test questions and build candidates pool
    test_question2answers = OrderedDict()
    test_candidates = []
    for item in test_items:
        question = item['body']
        if not question in test_question2answers:
            test_question2answers[question] = []
        for answer in item['ideal_answer']:
            if not answer in test_question2answers[question]:
                test_question2answers[question] += [answer]
            if not answer in test_candidates:
                test_candidates += [answer]
        
        # build candidate from the non-answer sentence of abstract
        for snippet_item in item['snippets']:
            if snippet_item['document'] in abstract_dict:
                snippet = snippet_item['text'].strip()
                for sent in abstract_dict[snippet_item['document']]:
                    if sent in snippet or snippet in sent:
                        continue
                    if any([sent in answer for answer in item['ideal_answer']]) or any([answer in sent for answer in item['ideal_answer']]):
                        continue
                    if not sent in test_candidates:
                        test_candidates += [sent]

    # build ground truth of each question
    test_questions = []
    test_ground_truths = []
    for q in test_question2answers.keys():
        test_questions.append(q)

    for q in test_questions:
        q_answers = test_question2answers[q]
        answer_ids = []
        for a in q_answers:
            answer_ids.append(test_candidates.index(a))
        test_ground_truths.append(answer_ids)
    logger.info(f"test questions size: {len(test_questions)}")
    logger.info(f"test candidates size: {len(test_candidates)}")

    test_data = {
            'questions': test_questions,
            'candidates': test_candidates,
            'ground_truths': test_ground_truths
    }
    with open(f'data/BioASQ/{args.dataset}/test.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f)
    
    logger.info(f"the test dataset of reqa bioasq {args.dataset} saved")