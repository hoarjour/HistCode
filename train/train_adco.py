#Some codes adopted from https://github.com/facebookresearch/moco
import os
from Adco.ops.argparser import argparser
from Adco.training.main_worker import main_worker


def main(args):
    if args.choose is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.choose
        print("Current we choose gpu:%s" % args.choose)
    args.distributed = False

    main_worker(args.gpu, args)


if __name__ == '__main__':
    parser = argparser()
    args = parser.parse_args()
    main(args)
