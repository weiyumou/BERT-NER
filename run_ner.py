from __future__ import absolute_import, division, print_function

import argparse
import copy
import json
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from allennlp.data.dataset_readers.dataset_utils.ontonotes import Ontonotes
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import (CONFIG_NAME, WEIGHTS_NAME,
                                              BertConfig,
                                              BertForTokenClassification)
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from pytorch_pretrained_bert.tokenization import BertTokenizer
from seqeval.metrics import classification_report, f1_score
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, labels=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            labels: (Optional) list. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_ids = label_ids


class OntoNotesIter(object):

    def __init__(self, data_dir) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.data_reader = Ontonotes()
        self.data_iter = self.data_reader.dataset_iterator(self.data_dir)

    def __iter__(self):
        self.data_iter = self.data_reader.dataset_iterator(self.data_dir)
        return self

    def __next__(self):
        sentence = next(self.data_iter)
        return sentence.words, sentence.named_entities


class NerProcessor(object):
    """Processor for the CoNLL-2012 data set."""

    def __init__(self):
        self.labels = ['O', 'I-GPE', 'B-DATE', 'B-LAW', 'B-NORP', 'B-QUANTITY', 'I-PERSON', 'I-EVENT', 'I-LAW', 'I-FAC',
                       'B-LOC', 'I-DATE', 'I-ORDINAL', 'B-ORG', 'B-WORK_OF_ART', 'I-CARDINAL', 'B-PERCENT', 'B-ORDINAL',
                       'I-NORP', 'B-GPE', 'B-PRODUCT', 'I-MONEY', 'B-LANGUAGE', 'I-PRODUCT', 'B-MONEY', 'I-LANGUAGE',
                       'B-EVENT', 'B-FAC', 'B-CARDINAL', 'I-LOC', 'I-TIME', 'I-PERCENT', 'B-PERSON', 'I-QUANTITY',
                       'B-TIME', 'I-WORK_OF_ART', 'I-ORG']

    def get_train_examples(self, data_dir):
        lines_iter = OntoNotesIter(os.path.join(data_dir, "train"))
        return self._create_examples(lines_iter, "train")

    def get_dev_examples(self, data_dir):
        lines_iter = OntoNotesIter(os.path.join(data_dir, "development"))
        return self._create_examples(lines_iter, "dev")

    def get_test_examples(self, data_dir):
        lines_iter = OntoNotesIter(os.path.join(data_dir, "test"))
        return self._create_examples(lines_iter, "test")

    def get_labels(self):
        return self.labels

    def _create_examples(self, lines_iter, set_type):
        for i, (sentence, labels) in enumerate(lines_iter):
            guid = f"{set_type}-{i}"
            yield InputExample(guid=guid, text_a=sentence, labels=labels)


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    num_examples = 0
    for ex_index, example in enumerate(examples):
        text_list = example.text_a
        label_list = example.labels

        tokens = []
        labels = []
        for i, word in enumerate(text_list):
            first, *_ = tokenizer.tokenize(word)
            tokens.append(first)
            labels.append(label_list[i])

            # token = tokenizer.tokenize(word)
            # tokens.extend(token)
            #
            # curr_label = label_list[i]
            # labels.append(curr_label)
            # if len(token) > 1:
            #     if curr_label.startswith("B"):
            #         labels.extend(["I" + curr_label[1:]] * (len(token) - 1))
            #     else:
            #         labels.extend([curr_label] * (len(token) - 1))

        tokens = tokens[:max_seq_length - 1]
        labels = labels[:max_seq_length - 1]
        tokens = ["[CLS]"] + tokens
        labels = ["O"] + labels

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        label_ids = [label_map[label] for label in labels]
        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        label_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          label_ids=label_ids))
        num_examples += 1
    return features, num_examples


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {"ner": NerProcessor}

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                   'distributed_{}'.format(args.local_rank))

    model = BertForTokenClassification.from_pretrained(args.bert_model,
                                                       cache_dir=cache_dir,
                                                       num_labels=num_labels)

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    global_step = 0
    label_map = {i: label for i, label in enumerate(label_list)}
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        train_features, num_train_examples = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)

        dev_examples = processor.get_dev_examples(args.data_dir)
        dev_features, num_dev_examples = convert_examples_to_features(
            dev_examples, label_list, args.max_seq_length, tokenizer)

        num_train_optimization_steps = int(
            num_train_examples / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
            warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                                 t_total=num_train_optimization_steps)
        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=num_train_optimization_steps)

        logger.info("***** Running training *****")
        logger.info("  Num training examples = %d", num_train_examples)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        all_train_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_train_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_train_label_ids = torch.tensor([f.label_ids for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_train_input_ids, all_train_input_mask, all_train_label_ids)

        all_dev_input_ids = torch.tensor([f.input_ids for f in dev_features], dtype=torch.long)
        all_dev_input_mask = torch.tensor([f.input_mask for f in dev_features], dtype=torch.long)
        all_dev_label_ids = torch.tensor([f.label_ids for f in dev_features], dtype=torch.long)
        dev_data = TensorDataset(all_dev_input_ids, all_dev_input_mask, all_dev_label_ids)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        dev_sampler = SequentialSampler(dev_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.eval_batch_size)

        best_model_wts = copy.deepcopy(model.state_dict())
        best_metric = 0.0

        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, label_ids = batch
                loss = model(input_ids, attention_mask=input_mask, labels=label_ids)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                          args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            model.eval()
            y_true, y_pred = evaluate(device, model, dev_dataloader, label_map)
            metric = f1_score(y_true, y_pred)
            if metric > best_metric:
                best_metric = metric
                best_model_wts = copy.deepcopy(model.state_dict())
            logger.info("  Metric on dev set = %.4f", metric)

        # Save a trained model and the associated configuration
        model.load_state_dict(best_model_wts)
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())
        model_config = {"bert_model": args.bert_model, "do_lower": args.do_lower_case,
                        "max_seq_length": args.max_seq_length, "num_labels": num_labels,
                        "label_map": label_map}
        json.dump(model_config, open(os.path.join(args.output_dir, "model_config.json"), "w"))

    else:
        # Load a trained model and config that you have fine-tuned
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        config = BertConfig(output_config_file)
        model = BertForTokenClassification(config, num_labels=num_labels)
        model.load_state_dict(torch.load(output_model_file))
        model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        test_examples = processor.get_test_examples(args.data_dir)
        test_features, num_test_examples = convert_examples_to_features(test_examples, label_list, args.max_seq_length,
                                                                        tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num test examples = %d", num_test_examples)
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_test_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_test_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_test_label_ids = torch.tensor([f.label_ids for f in test_features], dtype=torch.long)

        test_data = TensorDataset(all_test_input_ids, all_test_input_mask, all_test_label_ids)
        # Run prediction for full data
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)
        model.eval()
        y_true, y_pred = evaluate(device, model, test_dataloader, label_map)
        report = classification_report(y_true, y_pred, digits=4)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            logger.info("\n%s", report)
            writer.write(report)


def evaluate(device, model, eval_dataloader, label_map):
    y_true = []
    y_pred = []
    for input_ids, input_mask, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask=input_mask)

        preds = torch.argmax(logits, dim=2).detach().cpu()
        input_mask = input_mask.cpu().byte()
        preds = preds[input_mask].numpy().tolist()
        label_ids = label_ids[input_mask].numpy().tolist()

        y_pred.append([label_map[label_id] for label_id in preds])
        y_true.append([label_map[label_id] for label_id in label_ids])

    return y_true, y_pred


if __name__ == "__main__":
    main()
