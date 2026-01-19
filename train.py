import os
import traceback
from datetime import datetime
import warnings
from lib.metrics import test_metrics
warnings.filterwarnings('ignore')
import time
import torch

from lib.utils import (
    init_seed,
    get_model_params,
    load_graph, get_log_dir,
)

from lib.dataloader import get_dataloader
from lib.logger import get_logger, PD_Stats
from lib.utils import dwa
import numpy as np
from models.our_model import PVEDA


class Trainer(object):
    def __init__(self, model, optimizer, dataloader, graph, lr_scheduler,args, load_state=None):
        super(Trainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.train_loader = dataloader['train']
        self.val_loader = dataloader['test']
        self.test_loader = dataloader['test']
        self.scaler = dataloader['scaler']
        self.graph = graph
        self.lr_scheduler=lr_scheduler
        self.args = args
        self.time_dir = '/root/autodl-tmp/PVEDA/time'
        os.makedirs(self.time_dir, exist_ok=True)
        self.time_log_file = os.path.join(self.time_dir, f'{self.args.dataset}_time_log.txt')
        with open(self.time_log_file, 'w', encoding='utf-8') as f:
            f.write(f'[Timing Log] dataset={self.args.dataset}, '
                    f'epochs={self.args.epochs}, batch_size={self.args.batch_size}\n')
            f.write('epoch, train_phase_s, val_infer_s, epoch_training_total_s\n')
        def _time_log(msg: str):
            with open(self.time_log_file, 'a', encoding='utf-8') as _f:
                _f.write(msg.rstrip() + '\n')
        self._time_log = _time_log
        self.total_training_time = 0.0
        self.total_infer_time = 0.0

        self.train_per_epoch = len(self.train_loader)
        if self.val_loader != None:
            self.val_per_epoch = len(self.val_loader)
        args.log_dir = get_log_dir(args)
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.log_dir, debug=args.debug)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.training_stats = PD_Stats(
            os.path.join(args.log_dir, 'stats.pkl'),
            ['epoch', 'train_loss', 'val_loss'],
        )
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        self.logger.info('Experiment configs are: {}'.format(args))

    def train_epoch(self, epoch, loss_weights):
        self.model.train()
        total_loss = 0
        total_sep_loss = np.zeros(3)
        start_train = time.time()
        for batch_idx, (data, data_m, target, time_label, c) in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            repr1, repr2 = self.model(data, data_m)  # nvc
            loss, sep_loss = self.model.calculate_loss(
                data, repr1, repr2, target, c, time_label, self.scaler, loss_weights, True
            )

            assert not torch.isnan(loss)

            loss.backward()
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    get_model_params([self.model]),
                    self.args.max_grad_norm
                )
            self.optimizer.step()
            total_loss += loss.item()
            total_sep_loss += sep_loss
        train_time_s = time.time() - start_train
        train_epoch_loss = total_loss / self.train_per_epoch
        total_sep_loss = total_sep_loss / self.train_per_epoch
        self.logger.info('*******Train Epoch {}: averaged Loss : {:.6f}'.format(epoch, train_epoch_loss))

        return train_epoch_loss, total_sep_loss, train_time_s


    def val_epoch(self, epoch, val_dataloader, loss_weights):
        self.model.eval()
        total_val_loss = 0
        total_sep_loss = np.zeros(3)
        start_val = time.time()
        with torch.no_grad():
            for batch_idx, (data, data_m, target, time_label, c) in enumerate(val_dataloader):
                repr1, repr2 = self.model(data, data_m)
                loss, sep_loss = self.model.calculate_loss(
                    data, repr1, repr2, target, c, time_label, self.scaler, loss_weights
                )
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
                total_sep_loss += sep_loss
        val_time_s = time.time() - start_val
        val_loss = total_val_loss / len(val_dataloader)
        total_sep_loss = total_sep_loss / len(val_dataloader)
        self.logger.info('*******Val Epoch {}: averaged Loss : {:.6f} sep loss : {}'.format(
            epoch, val_loss, total_sep_loss
        ))
        return val_loss, val_time_s


    def train(self):
        best_loss = float('inf')
        best_epoch = 0
        not_improved_count = 0
        start_time = time.time()

        loss_tm1 = loss_t = np.ones(3)

        for epoch in range(1, self.args.epochs + 1):

            if self.args.use_dwa:
                loss_tm2 = loss_tm1
                loss_tm1 = loss_t
                if (epoch == 1) or (epoch == 2):
                    loss_weights = dwa(loss_tm1, loss_tm1, self.args.temp)
                else:
                    loss_weights = dwa(loss_tm1, loss_tm2, self.args.temp)
            self.logger.info('loss weights: {}'.format(loss_weights))
            train_epoch_loss, loss_t, train_time_s = self.train_epoch(epoch, loss_weights)

            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break

            val_dataloader = self.val_loader if self.val_loader is not None else self.test_loader
            val_epoch_loss, val_time_s = self.val_epoch(epoch, val_dataloader, loss_weights)

            # === 累计与记录 ===
            epoch_training_total = train_time_s + val_time_s
            self.total_training_time += epoch_training_total
            self.total_infer_time += val_time_s

            self._time_log(f'{epoch}, {train_time_s:.6f}, {val_time_s:.6f}, {epoch_training_total:.6f}')

            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                best_epoch = epoch
                not_improved_count = 0
                save_dict = {
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                }
                if not self.args.debug:
                    self.logger.info('**************Current best model saved to {}'.format(self.best_path))
                    torch.save(save_dict, self.best_path)
            else:
                not_improved_count += 1

            self.lr_scheduler.step(val_epoch_loss)

            if self.args.early_stop and not_improved_count == self.args.early_stop_patience:
                self.logger.info("Validation performance didn\'t improve for {} epochs. Training stops.".format(
                    self.args.early_stop_patience))
                break


        training_time = time.time() - start_time
        self._time_log(f'TOTAL_training_time_s(train+val)={self.total_training_time:.6f}')
        self._time_log(f'TOTAL_inference_time_s(val only)={self.total_infer_time:.6f}')

        self.logger.info("== Training finished.\n"
                         "Total training time: {:.2f} min\t"
                         "best loss: {:.4f}\t"
                         "best epoch: {}\t".format(
            (training_time / 60),
            best_loss,
            best_epoch))

        state_dict = save_dict if self.args.debug else torch.load(
            self.best_path, map_location=torch.device(self.args.device))
        self.model.load_state_dict(state_dict['model'])
        self.logger.info("== Test results.")
        test_results = self.test(self.model, self.test_loader, self.scaler,
                                 self.graph, self.logger, self.args)
        results = {
            'best_val_loss': best_loss,
            'best_val_epoch': best_epoch,
            'test_results': test_results,
        }

        return results

    @staticmethod
    def test(model, dataloader, scaler, graph, logger, args):
        model.eval()
        y_pred = []
        y_true = []
        Cs=[]
        Hs=[]
        Alpha = []
        Beta = []
        with torch.no_grad():
            for batch_idx, (data, data_m, target, time_label, c) in enumerate(dataloader):
                repr1, repr2 = model(data, data_m)
                pred_output, alpha, beta = model.predict(repr1, repr2, data)
                y_true.append(target)
                y_pred.append(pred_output)

                Cs.append(repr1.cpu().detach())
                Hs.append(repr2.cpu().detach())

                Alpha.append(alpha.cpu().detach())
                Beta.append(beta.cpu().detach())

        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
        Cs=torch.cat(Cs,dim=0)
        Hs=torch.cat(Hs,dim=0)
        Alpha = torch.cat(Alpha, dim=0)
        Beta = torch.cat(Beta, dim=0)
        save_path = os.path.join(args.log_dir, 'result.npz')
        np.savez(save_path, y_true=y_true.cpu().numpy(), y_pred=y_pred.cpu().numpy())
        rep_path = os.path.join(args.log_dir, 'representation.npz')
        np.savez(rep_path, C=Cs.cpu().numpy(), H=Hs.cpu().numpy())
        coefficient_path = os.path.join(args.log_dir, 'weight.npz')
        np.savez(coefficient_path, alpha = Alpha.cpu().numpy(), beta = Beta.cpu().numpy())
        test_results = []
        mae, mape = test_metrics(y_pred[..., 0], y_true[..., 0])
        logger.info("TEST RESULTS, MAE: {:.2f}, MAPE: {:.4f}%".format(mae, mape * 100))
        test_results.append([mae, mape])
        return np.stack(test_results, axis=0)


def make_one_hot(labels, classes):
    labels = labels.unsqueeze(dim=-1)
    one_hot = torch.FloatTensor(labels.size()[0], classes).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target

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
    model = PVEDA(args=args, adj=A, adj_m = A_m, in_channels=args.d_input, embed_size=args.d_model,
                T_dim=args.input_length, output_T_dim=52, output_dim=args.d_output,device=args.device).to(args.device)

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=args.lr_init,
        eps=1.0e-8,
        weight_decay=0,
        amsgrad=False
    )
    lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.lr_patience, verbose=True, threshold=0.0001, threshold_mode='rel', min_lr=0.000005, eps=1e-08)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        graph=A,
        lr_scheduler=lr_scheduler,
        args=args
    )

    results = None
    try:
        if args.mode == 'train':
            results = trainer.train()
        elif args.mode == 'test':
            # test
            state_dict = torch.load(
                args.best_path,
                map_location=torch.device(args.device)
            )
            model.load_state_dict(state_dict['model'])
            print("Load saved model")
            results = trainer.test(model, dataloader['test'], dataloader['scaler'],
                        A, trainer.logger, trainer.args)
        else:
            raise ValueError
    except:
        trainer.logger.info(traceback.format_exc())












