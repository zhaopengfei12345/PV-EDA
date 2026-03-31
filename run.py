import os
import warnings
import yaml
import argparse
import train
from lib.utils import get_project_path

warnings.filterwarnings('ignore')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--config_filename', default='configs/PV.yaml', type=str)
    parser.add_argument('--lr', default=None, type=float)
    parser.add_argument('--bs', default=None, type=int)
    parser.add_argument('--d', default=None, type=int)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--lr_mode', default=None, type=str)
    parser.add_argument('--max_epoch', default=None, type=int)
    parser.add_argument('--ablation', default='all', type=str)

    args = parser.parse_args()
    args.config_filename = os.path.join(get_project_path(), args.config_filename)
    print(f'Starting experiment with configurations in {args.config_filename}...')

    configs = yaml.load(open(args.config_filename), Loader=yaml.FullLoader)

    if args.lr is not None:
        configs['lr_init'] = args.lr
    if args.bs is not None:
        configs['batch_size'] = args.bs
    if args.seed is not None:
        configs['seed'] = args.seed
    if args.d is not None:
        configs['d_model'] = args.d
    if args.lr_mode is not None:
        configs['lr_mode'] = args.lr_mode
    if args.max_epoch is not None:
        configs['epochs'] = args.max_epoch

    configs['ablation'] = args.ablation

    if 'lambda_vclub' not in configs:
        configs['lambda_vclub'] = 0.1
    if 'MMI' not in configs:
        configs['MMI'] = True
    if 'use_dwa' not in configs:
        configs['use_dwa'] = True
    if 'temp' not in configs:
        configs['temp'] = 4
    if 'test_batch_size' not in configs:
        configs['test_batch_size'] = configs.get('batch_size', 32)
    if 'debug' not in configs:
        configs['debug'] = False
    if 'best_path' not in configs:
        configs['best_path'] = None

    args = argparse.Namespace(**configs)
    args.graph_file = os.path.join(get_project_path(), args.graph_file)
    args.data_dir = os.path.join(get_project_path(), args.data_dir)

    if args.mode in ['train', 'test']:
        train.main(args)
    else:
        raise ValueError(f'Unsupported mode: {args.mode}')
