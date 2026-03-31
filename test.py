import os
import warnings
import torch
import numpy as np
from munch import DefaultMunch

warnings.filterwarnings('ignore')

from lib.utils import init_seed, load_graph
from lib.dataloader import get_dataloader
from models.our_model import PVEDA


def text2args(text):
    args_dict = {}
    temp = text.split(", ")
    for s in temp:
        key, value = s.split("=")
        if '\'' in value:
            args_dict[key] = value.replace('\'', '')
        elif '.' in value:
            args_dict[key] = float(value)
        elif 'False' in value:
            args_dict[key] = False
        elif 'True' in value:
            args_dict[key] = True
        else:
            args_dict[key] = int(value)
    args = DefaultMunch.fromDict(args_dict)
    return args


def test(model, dataloader, scaler):
    model.eval()
    invariant_repr = []
    variant_repr = []
    alpha_list = []
    beta_list = []
    y_pred = []
    y_true = []

    with torch.no_grad():
        for batch_idx, (data, data_m, target, time_label, c) in enumerate(dataloader):
            repr1, repr2 = model(data, data_m)
            pred_output, alpha, beta = model.predict(repr1, repr2, data)

            invariant_repr.append(repr1)
            variant_repr.append(repr2)
            alpha_list.append(alpha)
            beta_list.append(beta)
            y_true.append(target)
            y_pred.append(pred_output)

    invariant_repr = torch.cat(invariant_repr, dim=0).cpu()
    variant_repr = torch.cat(variant_repr, dim=0).cpu()
    alpha = torch.cat(alpha_list, dim=0).cpu()
    beta = torch.cat(beta_list, dim=0).cpu()
    y_true = scaler.inverse_transform(torch.cat(y_true, dim=0)).cpu()
    y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0)).cpu()

    return invariant_repr, variant_repr, alpha, beta, y_true, y_pred


def main(args):
    A, A_m = load_graph(args.graph_file, device=args.device)
    init_seed(args.seed)

    dataloader = get_dataloader(
        data_dir=args.data_dir,
        dataset=args.dataset,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        device=args.device
    )

    model = PVEDA(
        args=args,
        adj=A,
        adj_m=A_m,
        embed_size=args.d_model,
        output_T_dim=52,
        output_dim=args.d_output,
        device=args.device
    ).to(args.device)

    if args.best_path is None:
        if hasattr(args, 'log_dir') and args.log_dir is not None:
            best_path = os.path.join(args.log_dir, 'best_model.pth')
        else:
            raise ValueError('best_path is not provided.')
    else:
        best_path = args.best_path

    print('load model from {}.'.format(best_path))
    state_dict = torch.load(best_path, map_location=torch.device(args.device))
    model.load_state_dict(state_dict['model'])

    invariant, variant, alpha, beta, y_true, y_pred = test(model, dataloader['test'], dataloader['scaler'])

    result_dir = os.path.dirname(best_path)
    result_path = os.path.join(result_dir, 'result_test.npz')
    rep_path = os.path.join(result_dir, 'representation_test.npz')
    weight_path = os.path.join(result_dir, 'weight_test.npz')

    print('save result in {}.'.format(result_path))
    np.savez(result_path, y_true=y_true.numpy(), y_pred=y_pred.numpy())
    np.savez(rep_path, RI=invariant.numpy(), RV=variant.numpy())
    np.savez(weight_path, alpha=alpha.numpy(), beta=beta.numpy())


if __name__ == '__main__':
    d = r'D:\python project\PV-EDA'
    best_paths = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d, o))]
    file_list = []
    mae_list = []

    for best_path in best_paths:
        config_file_path = os.path.join(best_path, 'run.log')
        if not os.path.exists(config_file_path):
            continue
        with open(config_file_path, 'r', encoding='utf-8') as config_file:
            config_str = config_file.readlines()
        if len(config_str) < 2:
            continue
        config = config_str[1]
        config = config[55:-2]
        args = text2args(config)

        if getattr(args, 'batch_size', None) == 32 and getattr(args, 'd_model', None) == 32 and getattr(args, 'seed', None) == 31:
            temp = config_str[-2]
            try:
                mae = float(temp[34:39])
            except Exception:
                mae = 100.0
            file_list.append(best_path)
            mae_list.append(mae)

    if len(file_list) > 0:
        index = np.argsort(mae_list)
        print(file_list[index[0]])
