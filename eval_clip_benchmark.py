# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
from collections import OrderedDict
import json
import os

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms

import datasets
import models
from tokenizer import SimpleTokenizer
import utils
from clip_benchmark.datasets.builder import build_dataset, get_dataset_collate_fn, zeroshot_classification_templates
from clip_benchmark.metrics import zeroshot_classification, zeroshot_retrieval


def get_args_parser():
    parser = argparse.ArgumentParser(description='SLIP 0-shot evaluations', add_help=False)
    parser.add_argument('--output-dir', default='./', type=str, help='output dir')
    parser.add_argument('--dataset', type=str, default="cifar10", help="Dataset to use for the benchmark")
    parser.add_argument('--batch-size', default=256, type=int, help='batch_size')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--recall_k', default=[5], type=int, help="for retrieval, select the k for Recall@K metric. ", nargs="+",)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--dataset_root', default="root", type=str, help="dataset root")
    parser.add_argument('--annotation_file', default="", type=str, help="text annotation file for retrieval datasets")
    parser.add_argument('--task', type=str, default="zeroshot_classification", choices=["zeroshot_classification", "zeroshot_retrieval"])
    parser.add_argument('--output', default="result.json", type=str, help="output file where to dump the metrics")
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--amp', default=False, action="store_true", help="whether to use mixed precision")
    parser.add_argument('--model', default='', type=str, help='path to latest checkpoint')
    parser.add_argument('--pretrained', default='', type=str, help='path to latest checkpoint')
    return parser


def main(args):
    # optionally resume from a checkpoint (takes precedence over autoresume)
    if args.resume:
        ckpt_path = args.resume
    elif os.path.isfile(os.path.join(args.output_dir, 'checkpoint_best.pt')):
        ckpt_path = os.path.join(args.output_dir, 'checkpoint_best.pt')
    else:
        raise Exception('no checkpoint found')

    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v
    
    # create model
    old_args = ckpt['args']
    print(old_args)
    print("=> creating model: {}".format(old_args.model))
    model = getattr(models, old_args.model)(rand_embed=False,
        ssl_mlp_dim=old_args.ssl_mlp_dim, ssl_emb_dim=old_args.ssl_emb_dim)
    model.cuda()
    model.load_state_dict(state_dict, strict=False)
    print("=> loaded resume checkpoint '{}' (epoch {})".format(args.resume, ckpt['epoch']))

    cudnn.benchmark = True

    cwd = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(cwd, 'dataset_catalog.json')) as f:
        catalog = json.load(f)

    with open(os.path.join(cwd, 'templates.json')) as f:
        all_templates = json.load(f)

    with open(os.path.join(cwd, 'labels.json')) as f:
        all_labels = json.load(f)

    # Data loading code
    print("=> creating dataset")
    tokenizer = SimpleTokenizer()
    def tokenizer_(x):
        t = tokenizer(x)
        if len(t.shape) == 1:
            t = t.view(1, -1)
        return t
    val_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            lambda x: x.convert('RGB'),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = build_dataset(args, transform=val_transform, train=False)
    collate_fn = get_dataset_collate_fn(args.dataset)
    print("Dataset size", len(dataset))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, 
        shuffle=False, num_workers=args.num_workers, 
        collate_fn=collate_fn
    )
    zeroshot_templates = zeroshot_classification_templates.get(args.dataset)
    classnames = dataset.classes if hasattr(dataset, "classes") else None
    if args.task == "zeroshot_classification":
        zeroshot_templates = zeroshot_classification_templates.get(args.dataset)
        classnames = dataset.classes if hasattr(dataset, "classes") else None
        assert (zeroshot_templates is not None and classnames is not None), "Dataset does not support classification"
        metrics = zeroshot_classification.evaluate(
            model, 
            dataloader, 
            tokenizer_, 
            classnames, zeroshot_templates, 
            device=args.device, amp=args.amp
        )
    elif args.task == "zeroshot_retrieval":
        metrics = zeroshot_retrieval.evaluate(
            model, 
            dataloader, 
            tokenizer_, 
            recall_k_list=args.recall_k,
            device=args.device, amp=args.amp
        )
    else:
        raise ValueError("Unsupported task: {}".format(args.task))
    dump = {
        "dataset": args.dataset,
        "model": old_args.model,
        "task": args.task,
        "metrics": metrics
    }
    with open(args.output, "w") as f:
        json.dump(dump, f)
    return 0
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('SLIP 0-shot evaluations', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
