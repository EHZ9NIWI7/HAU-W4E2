import argparse
import os

import yaml


def update_parameters(parser, args):
    if os.path.exists('./configs/{}.yaml'.format(args.config)):
        with open('./configs/{}.yaml'.format(args.config), 'r') as f:
            try:
                yaml_arg = yaml.load(f, Loader=yaml.FullLoader)
            except:
                yaml_arg = yaml.load(f)
            default_arg = vars(args)
            for k in yaml_arg.keys():
                if k not in default_arg.keys():
                    raise ValueError('Do NOT exist this parameter {}'.format(k))
            parser.set_defaults(**yaml_arg)
    else:
        raise ValueError('Do NOT exist this file in \'configs\' folder: {}.yaml!'.format(args.config))
    
    return parser.parse_args()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="W4E2 Benchmark for Human Action Understanding")
    
    parser.add_argument('--config', '-c', type=str, default='', help='config')
    parser.add_argument('--dataset_dir', '-d', type=str, default='./data/Videos')
    parser.add_argument('--gt_dir', '-d', type=str, default='./data/GT')
    parser.add_argument('--save_dir', '-sd', type=str, default=f'./results')
    parser.add_argument('--subset', '-s', type=str, default='NTU')
    parser.add_argument('--model_path', '-m', type=str, default='')
    parser.add_argument('--gpus', '-g', type=str, default='0')
    parser.add_argument('--api_key', '-api', type=str, default='')
    
    parser.add_argument('--inference', '-i', action='store_true')
    parser.add_argument('--evaluate', '-e', action='store_true')
    
    args = parser.parse_args()
    if args.config:
        args = update_parameters(parser, args)  # cmd > yaml > default
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    
    from benchmark import W4E2
    
    """
    Subsets:
        NTU, Kinetics, UCF-101, ActivityNet
    """
    
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.inference:
        benchmark = W4E2(args)
        benchmark.inference()
    elif args.evaluate:
        benchmark = W4E2(args)
        benchmark.evaluate()
    else:
        raise ValueError('Please select an option: --inference, --evaluate')