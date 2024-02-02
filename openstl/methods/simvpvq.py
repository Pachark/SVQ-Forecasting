import time
import torch
import torch.nn as nn
from tqdm import tqdm
from timm.utils import AverageMeter
import numpy as np
import matplotlib.pyplot as plt
import gc

from openstl.models import SimVPVQ_Model
from openstl.utils import reduce_tensor
from .base_method import Base_method
from openstl.utils import gather_tensors_batch, get_dist_info, ProgressBar, print_log
from openstl.core import metric

class SimVPVQ(Base_method):

    def __init__(self, args, device, steps_per_epoch):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model(self.config)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)
        if self.args.loss == 'mse':
            self.criterion = nn.MSELoss()
        elif self.args.loss == 'mae':
            self.criterion = nn.L1Loss()
        elif self.args.loss == 'huber':
            self.criterion = nn.HuberLoss(delta=self.args.delta)
        else:
            raise Exception('Wrong loss function! Only mse and mae are available')
        self.perplexity_train = None
        self.perplexity_val = None

    def _build_model(self, args):
        return SimVPVQ_Model(**args).to(self.device)

    def _predict(self, batch_x, batch_y=None, **kwargs):
        """Forward the model"""
        if self.args.aft_seq_length == self.args.pre_seq_length:
            pred_y, auxilary_loss, perplexity, continuous, quantize, code_weight, codebook = self.model(batch_x, batch_y)

        elif self.args.aft_seq_length < self.args.pre_seq_length:
            pred_y, auxilary_loss, perplexity, continuous, quantize, code_weight, codebook = self.model(batch_x)
            pred_y = pred_y[:, :self.args.aft_seq_length]
        elif self.args.aft_seq_length > self.args.pre_seq_length:
            pred_y = []
            d = self.args.aft_seq_length // self.args.pre_seq_length
            m = self.args.aft_seq_length % self.args.pre_seq_length
            
            cur_seq = batch_x.clone()
            for _ in range(d):
                cur_seq, auxilary_loss, perplexity, continuous, quantize, code_weight, codebook = self.model(cur_seq)
                pred_y.append(cur_seq)

            if m != 0:
                cur_seq, auxilary_loss, perplexity, continuous, quantize, code_weight, codebook = self.model(cur_seq)
                pred_y.append(cur_seq[:, :m])
            
            pred_y = torch.cat(pred_y, dim=1)
        
        if batch_y is None:
            return pred_y
        else:
            return pred_y, auxilary_loss, perplexity, continuous, quantize, code_weight, codebook

    def train_one_epoch(self, runner, train_loader, epoch, num_updates, eta=None, **kwargs):
        """Train the model with train_loader."""
        data_time_m = AverageMeter()
        losses_m = AverageMeter()
        self.model.train()
        if self.by_epoch:
            self.scheduler.step(epoch)
        train_pbar = tqdm(train_loader) if self.rank == 0 else train_loader

        end = time.time()
        count = 0

        if self.args.freeze_projection:
            for param in self.model.vq.vq.proj_regression_weight.parameters():
                param.requires_grad = False
            
        for batch_x, batch_y in train_pbar:
            count += 1
            
            data_time_m.update(time.time() - end)
            self.model_optim.zero_grad()

            if not self.args.use_prefetcher:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            runner.call_hook('before_train_iter')

            with self.amp_autocast():
                pred_y, auxilary_loss, perplexity, continuous, quantize, code_weight, codebook = self._predict(batch_x, batch_y)
            loss = self.criterion(pred_y, batch_y)
            
            for i in auxilary_loss.keys():
                loss = loss + auxilary_loss[i]
            
            if not self.dist:
                losses_m.update(loss.item(), batch_x.size(0))
                
            if self.loss_scaler is not None:
                if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                    raise ValueError("Inf or nan loss value. Please use fp32 training!")
                self.loss_scaler(
                    loss, self.model_optim,
                    clip_grad=self.args.clip_grad, clip_mode=self.args.clip_mode,
                    parameters=self.model.parameters())
            else:
                loss.backward()
                self.clip_grads(self.model.parameters())
                self.model_optim.step()

            torch.cuda.synchronize()
            num_updates += 1

            if self.dist:
                losses_m.update(reduce_tensor(loss), batch_x.size(0))

            if not self.by_epoch:
                self.scheduler.step()
            runner.call_hook('after_train_iter')
            runner._iter += 1

            if self.rank == 0:
                log_buffer = 'train loss: {:.4f}'.format(loss.item())
                # log_buffer += ' | data time: {:.4f}'.format(data_time_m.avg)
                for i in auxilary_loss.keys():
                    log_buffer += ' | {}: {:.4f}'.format(i, auxilary_loss[i].item())
                for i in perplexity.keys():
                    log_buffer += ' | {}: {:.4f}'.format(i, perplexity[i].item())
                
                train_pbar.set_description(log_buffer)

            end = time.time()  # end for

        perplexity_mean = []
        perplexity_record = ''
        for p in perplexity:
            perplexity_record += f'Training {p}: {perplexity[p].item()}, '
            perplexity_mean.append(perplexity[p].item())
        perplexity_mean = np.mean(perplexity_mean)
        perplexity_record += f'Training average perp: {perplexity_mean}\n'
        print_log(perplexity_record)
        self.perplexity_train = perplexity
        
        if hasattr(self.model_optim, 'sync_lookahead'):
            self.model_optim.sync_lookahead()

        return num_updates, losses_m, eta

    
    def _nondist_forward_collect(self, data_loader, length=None, gather_data=False):
        """Forward and collect predictios.

        Args:
            data_loader: dataloader of evaluation.
            length (int): Expected length of output arrays.
            gather_data (bool): Whether to gather raw predictions and inputs.

        Returns:
            results_all (dict(np.ndarray)): The concatenated outputs.
        """
        # preparation
        results = []
        prog_bar = ProgressBar(len(data_loader))
        length = len(data_loader.dataset) if length is None else length

        # loop
        for idx, (batch_x, batch_y) in enumerate(data_loader):
            with torch.no_grad():
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_y, auxilary_loss, perplexity, continuous, quantize, code_weight, codebook = self._predict(batch_x, batch_y)
                
            if gather_data:  # return raw datas
                if self.args.dataname == 'human': # To avoid OOM, Human dataset doesn't save inputs 
                    results_name = ['preds', 'trues']
                    results_file = [pred_y.cpu().numpy(), batch_y.cpu().numpy()]
                    results.append(dict(zip(results_name, results_file)))
                else:
                    results_name = ['inputs', 'preds', 'trues']
                    results_file = [batch_x.cpu().numpy(), pred_y.cpu().numpy(), batch_y.cpu().numpy()]
                    results.append(dict(zip(results_name, results_file)))
                    
            else:  # return metrics
                loss = self.criterion(pred_y, batch_y)
                for i in auxilary_loss.keys():
                    loss = loss + auxilary_loss[i]
                eval_res, _ = metric(pred_y.cpu().numpy(), batch_y.cpu().numpy(),
                                 data_loader.dataset.mean, data_loader.dataset.std,
                                 metrics=self.metric_list, spatial_norm=self.spatial_norm, return_log=False)

                eval_res['loss'] = loss.cpu().numpy() #self.criterion(pred_y, batch_y).cpu().numpy()

                for i in auxilary_loss.keys():
                    eval_res[i] = auxilary_loss[i].cpu().numpy()
                for i in perplexity.keys():
                    eval_res[i] = perplexity[i].reshape(1).cpu().numpy()
                for k in eval_res.keys():
                    eval_res[k] = eval_res[k].reshape(1)
                results.append(eval_res)

            prog_bar.update()
            if self.args.empty_cache:
                torch.cuda.empty_cache()

        # post gather tensors
        results_all = {}
        for k in results[0].keys():
            results_all[k] = np.concatenate([batch[k] for batch in results], axis=0)
        
        if gather_data:  # return raw datas
            results_name = ['code_weight', 'quantize', 'continuous', 'codebook']
            results_file = [code_weight.detach().cpu().numpy(), quantize.detach().cpu().numpy(), continuous.detach().cpu().numpy(), codebook.detach().cpu().numpy()]
            for k in range(len(results_name)):
                results_all[results_name[k]] = results_file[k]
            
        # perplexity_names = [i for i in results_all.keys() if 'usage' in i]
        perplexity_names = [i for i in results_all.keys() if 'perp' in i]
        perplexity_mean = []
        perplexity_log = ''
        for p in perplexity_names:
            perplexity_log = perplexity_log + f'Validation {p}: {results_all[p].mean()}, '
            perplexity_mean.append(results_all[p].mean())
        perplexity_mean = np.mean(perplexity_mean)
        perplexity_log = perplexity_log + f'Validation average perp: {perplexity_mean}'
        
        print_log(perplexity_log)
        if not gather_data:
            self.perplexity_val = perplexity_mean
        return results_all


    def _dist_forward_collect(self, data_loader, length=None, gather_data=False):
        """Forward and collect predictios in a distributed manner.

        Args:
            data_loader: dataloader of evaluation.
            length (int): Expected length of output arrays.
            gather_data (bool): Whether to gather raw predictions and inputs.

        Returns:
            results_all (dict(np.ndarray)): The concatenated outputs.
        """
        # preparation
        results = []
        length = len(data_loader.dataset) if length is None else length
        if self.rank == 0:
            prog_bar = ProgressBar(len(data_loader))

        # loop
        for idx, (batch_x, batch_y) in enumerate(data_loader):
            if idx == 0:
                part_size = batch_x.shape[0]
            with torch.no_grad():
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_y, auxilary_loss, perplexity, continuous, quantize, code_weight, codebook = self._predict(batch_x, batch_y)

            if gather_data:  # return raw datas
                if self.args.dataname == 'human': # To avoid OOM, Human dataset doesn't save inputs 
                    results.append(dict(zip(['preds', 'trues'],
                                            [pred_y.cpu().numpy(), batch_y.cpu().numpy()])))
                else:
                    results.append(dict(zip(['inputs', 'preds', 'trues'],
                                            [batch_x.cpu().numpy(), pred_y.cpu().numpy(), batch_y.cpu().numpy()])))
            else:  # return metrics
                eval_res, _ = metric(pred_y.cpu().numpy(), batch_y.cpu().numpy(),
                                     data_loader.dataset.mean, data_loader.dataset.std,
                                     metrics=self.metric_list, spatial_norm=self.spatial_norm, return_log=False)
                eval_res['loss'] = self.criterion(pred_y, batch_y).cpu().numpy()
                for i in auxilary_loss.keys():
                    eval_res[i] = auxilary_loss[i].cpu().numpy()
                for i in perplexity.keys():
                    eval_res[i] = perplexity[i].reshape(1).cpu().numpy()
                for k in eval_res.keys():
                    eval_res[k] = eval_res[k].reshape(1)
                results.append(eval_res)

            if self.args.empty_cache:
                torch.cuda.empty_cache()
            if self.rank == 0:
                prog_bar.update()

        # post gather tensors
        results_all = {}
        for k in results[0].keys():
            results_cat = np.concatenate([batch[k] for batch in results], axis=0)
            results_all[k] = np.concatenate(gather_tensors_batch(results_cat, part_size=min(part_size*8, 16)), axis=0)[:length]
            # Empty cache to avoid OOM
            results_cat = None
            for batch in range(len(results)):
                results[batch][k] = None
            gc.collect()

        if gather_data:  # return raw datas
            results_name = ['code_weight', 'quantize', 'continuous', 'codebook']
            results_file = [code_weight.detach().cpu().numpy(), quantize.detach().cpu().numpy(), continuous.detach().cpu().numpy(), codebook.detach().cpu().numpy()]
            for k in range(len(results_name)):
                results_all[results_name[k]] = results_file[k]

        perplexity_names = [i for i in results_all.keys() if 'perp' in i]
        perplexity_mean = []
        perplexity_log = ''
        for p in perplexity_names:
            perplexity_log = perplexity_log + f'Validation {p}: {results_all[p].mean()}, '
            perplexity_mean.append(results_all[p].mean())
        perplexity_mean = np.mean(perplexity_mean)
        perplexity_log = perplexity_log + f'Validation average perp: {perplexity_mean}'
        
        print_log(perplexity_log)
        if not gather_data:
            self.perplexity_val = perplexity_mean
        return results_all