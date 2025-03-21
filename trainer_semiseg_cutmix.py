import torch
import time, random, cv2, sys 
from math import ceil
import numpy as np
from itertools import cycle
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision import transforms
from base import BaseTrainer, BaseTrainer_semiseg
from utils.helpers import colorize_mask
from utils.metrics import eval_metrics, AverageMeter
from tqdm import tqdm
from PIL import Image
from utils.helpers import DeNormalize
import torch.distributed as dist

class Trainer(BaseTrainer_semiseg):
    def __init__(self, model, resume, config, supervised_loader, unsupervised_loader, unsupervised_loader_2, iter_per_epoch,
                val_loader=None, train_logger=None, gpu=None, gt_loader=None, test=False):
        super(Trainer, self).__init__(model, resume, config, iter_per_epoch, train_logger, gpu=gpu, test=test)
        
        self.supervised_loader = supervised_loader
        self.unsupervised_loader = unsupervised_loader
        self.unsupervised_loader_2 = unsupervised_loader_2
        self.val_loader = val_loader
        self.iter_per_epoch = iter_per_epoch

        self.ignore_index = self.val_loader.dataset.ignore_index
        self.wrt_mode, self.wrt_step = 'train_', 0
        self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(self.val_loader.batch_size)))
        if config['trainer']['log_per_iter']:
            self.log_step = int(self.log_step / self.val_loader.batch_size) + 1

        self.num_classes = self.val_loader.dataset.num_classes
        # self.mode = self.model.module.mode

        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            DeNormalize(self.val_loader.MEAN, self.val_loader.STD),
            transforms.ToPILImage()])
        self.viz_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()])

        self.start_time = time.time()

    def _train_epoch(self, epoch):
        if self.gpu == 0:
            self.logger.info('\n')

        self.model.train()

        self.supervised_loader.train_sampler.set_epoch(epoch)
        self.unsupervised_loader.train_sampler.set_epoch(epoch)
        self.unsupervised_loader_2.train_sampler.set_epoch(epoch)

        dataloader = iter(zip(cycle(self.supervised_loader), cycle(self.unsupervised_loader), cycle(self.unsupervised_loader_2)))
        tbar = tqdm(range(self.iter_per_epoch), ncols=135)

        self._reset_metrics()

        for batch_idx in tbar:

            (input_l, target_l), (input_ul, target_ul, mask_params), (input_ul_2, target_ul_2) = next(dataloader)

            input_l, target_l = input_l.cuda(non_blocking=True), target_l.cuda(non_blocking=True)
            input_ul, target_ul, mask_params = input_ul.cuda(non_blocking=True), target_ul.cuda(non_blocking=True), mask_params.cuda(non_blocking=True)
            input_ul_2, target_ul_2 = input_ul_2.cuda(non_blocking=True), target_ul_2.cuda(non_blocking=True)
            self.model.zero_grad()
            self.optimizer_l.zero_grad()
            self.optimizer_r.zero_grad()

            # print(input_ul.shape)
            # print(mask_params.shape)

            kargs = {'gpu': self.gpu, 'ul1': None, 'br1': None, 'ul2': None, 'br2': None, 'flip': None}

            # _, pred_sup_l = self.model(input_l, step=1)
            # _, pred_unsup_l = self.model(input_ul, step=1)
            # _, pred_sup_r = self.model(input_l, step=2)
            # _, pred_unsup_r = self.model(input_ul, step=2)

            # outputs = {'sup_pred': pred_sup_l}
            
            # config network and criterion
            criterion = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
            criterion_csst = torch.nn.MSELoss(reduction='mean')

            # ### cps loss ###
            # pred_l = torch.cat([pred_sup_l, pred_unsup_l], dim=0)
            # pred_r = torch.cat([pred_sup_r, pred_unsup_r], dim=0)
            # _, max_l = torch.max(pred_l, dim=1)
            # _, max_r = torch.max(pred_r, dim=1)
            # max_l = max_l.long()
            # max_r = max_r.long()
            # cps_loss = criterion(pred_l, max_r) + criterion(pred_r, max_l)
            # dist.all_reduce(cps_loss, dist.ReduceOp.SUM)
            # cps_loss = cps_loss * 1.5

            # unsupervised loss on model/branch#1
            batch_mix_masks = mask_params
            unsup_imgs_mixed = input_ul * (1 - batch_mix_masks) + input_ul_2 * batch_mix_masks
            with torch.no_grad():
                # Estimate the pseudo-label with branch#1 & supervise branch#2
                _, logits_u0_tea_1 = self.model(input_ul, step=1)
                _, logits_u1_tea_1 = self.model(input_ul_2, step=1)
                logits_u0_tea_1 = logits_u0_tea_1.detach()
                logits_u1_tea_1 = logits_u1_tea_1.detach()
                # Estimate the pseudo-label with branch#2 & supervise branch#1
                _, logits_u0_tea_2 = self.model(input_ul, step=2)
                _, logits_u1_tea_2 = self.model(input_ul_2, step=2)
                logits_u0_tea_2 = logits_u0_tea_2.detach()
                logits_u1_tea_2 = logits_u1_tea_2.detach()

            # Mix teacher predictions using same mask
            # It makes no difference whether we do this with logits or probabilities as
            # the mask pixels are either 1 or 0
            logits_cons_tea_1 = logits_u0_tea_1 * (1 - batch_mix_masks) + logits_u1_tea_1 * batch_mix_masks
            _, ps_label_1 = torch.max(logits_cons_tea_1, dim=1)
            ps_label_1 = ps_label_1.long()
            logits_cons_tea_2 = logits_u0_tea_2 * (1 - batch_mix_masks) + logits_u1_tea_2 * batch_mix_masks
            _, ps_label_2 = torch.max(logits_cons_tea_2, dim=1)
            ps_label_2 = ps_label_2.long()

            # Get student#1 prediction for mixed image
            _, logits_cons_stu_1 = self.model(unsup_imgs_mixed.type(torch.FloatTensor).cuda(non_blocking=True), step=1)
            # Get student#2 prediction for mixed image
            _, logits_cons_stu_2 = self.model(unsup_imgs_mixed.type(torch.FloatTensor).cuda(non_blocking=True), step=2)

            cps_loss = criterion(logits_cons_stu_1, ps_label_2) + criterion(logits_cons_stu_2, ps_label_1)
            dist.all_reduce(cps_loss, dist.ReduceOp.SUM)
            # cps_loss = cps_loss * 1.5
            cps_loss = cps_loss * 1.5

            # Supervised Prediction
            _, pred_sup_l = self.model(input_l, step=1)
            _, pred_sup_r = self.model(input_l, step=2)

            outputs = {'sup_pred': pred_sup_l}

            ### standard cross entropy loss ###
            loss_sup = criterion(pred_sup_l, target_l)
            dist.all_reduce(loss_sup, dist.ReduceOp.SUM)
            # loss_sup = loss_sup / engine.world_size

            loss_sup_r = criterion(pred_sup_r, target_l)
            dist.all_reduce(loss_sup_r, dist.ReduceOp.SUM)
            # loss_sup_r = loss_sup_r / engine.world_size

            unlabeled_loss = False

            # current_idx = epoch * config.niters_per_epoch + idx
            # lr = lr_policy.get_lr(current_idx)

            loss = loss_sup + loss_sup_r + cps_loss
            loss.backward()
            self.optimizer_l.step()
            self.optimizer_r.step()
            

            cur_losses = {'loss_sup': loss_sup}
            cur_losses['loss_sup_r'] = loss_sup_r
            cur_losses['cps_loss'] = cps_loss
            
            if self.gpu == 0:
                if batch_idx % 100 == 0:
                    self.logger.info("epoch: {} train_loss: {}".format(epoch, loss))

            if batch_idx == 0:
                self.loss_sup = AverageMeter()
                self.loss_sup_r  = AverageMeter()
                self.cps_loss = AverageMeter()

            # self._update_losses has already implemented synchronized DDP
            self._update_losses(cur_losses)

            self._compute_metrics(outputs, target_l, target_ul, epoch-1)
            
            if self.gpu == 0:
                logs = self._log_values(cur_losses)
            
                if batch_idx % self.log_step == 0:
                    self.wrt_step = (epoch - 1) * len(self.unsupervised_loader) + batch_idx
                    self._write_scalars_tb(logs)

                # if batch_idx % int(len(self.unsupervised_loader)*0.9) == 0:
                #     self._write_img_tb(input_l, target_l, input_ul, target_ul, outputs, epoch)

                descrip = 'T ({}) | '.format(epoch)
                for key in cur_losses:
                    descrip += key + ' {:.2f} '.format(getattr(self, key).average)
                # descrip += 'm1 {:.2f} m2 {:.2f}|'.format(self.mIoU_l, self.mIoU_ul)
                descrip += 'm1 {:.2f}|'.format(self.mIoU_l)
                tbar.set_description(descrip)

            del input_l, target_l, input_ul, target_ul
            del loss, cur_losses, outputs
            
            self.lr_scheduler_l.step(epoch=epoch-1)
            self.lr_scheduler_r.step(epoch=epoch-1)
            
        return logs if self.gpu == 0 else None

    def _valid_epoch(self, epoch):
        if self.val_loader is None:
            if self.gpu == 0:
                self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}
            
        if self.gpu == 0:
            self.logger.info('\n###### EVALUATION ######')

        self.model.eval()
        self.wrt_mode = 'val'
        
        total_loss_val = AverageMeter()
        total_inter, total_union = 0, 0
        total_correct, total_label = 0, 0

        tbar = tqdm(self.val_loader, ncols=130)
        with torch.no_grad():
            if self.gpu == 0:
                val_visual = []

            for batch_idx, (data, target) in enumerate(tbar):
                target, data = target.cuda(non_blocking=True), data.cuda(non_blocking=True)

                H, W = target.size(1), target.size(2)
                up_sizes = (ceil(H / 8) * 8, ceil(W / 8) * 8)
                pad_h, pad_w = up_sizes[0] - data.size(2), up_sizes[1] - data.size(3)
                data = F.pad(data, pad=(0, pad_w, 0, pad_h), mode='reflect')

                output = self.model.branch1(data)

                output = output[:, :, :H, :W]
                # LOSS
                loss = F.cross_entropy(output, target, ignore_index=self.ignore_index)

                total_loss_val.update(loss.item())

                # eval_metrics has already implemented DDP synchronized
                correct, labeled, inter, union = eval_metrics(output, target, self.num_classes, self.ignore_index)
                
                total_inter, total_union = total_inter+inter, total_union+union
                total_correct, total_label = total_correct+correct, total_label+labeled

                if self.gpu == 0:
                    # LIST OF IMAGE TO VIZ (15 images)
                    if len(val_visual) < 15:
                        if isinstance(data, list): data = data[0]
                        target_np = target.data.cpu().numpy()
                        output_np = output.data.max(1)[1].cpu().numpy()
                        val_visual.append([data[0].data.cpu(), target_np[0], output_np[0]])

                # PRINT INFO
                pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
                IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
                mIoU = IoU.mean()
                seg_metrics = {"Pixel_Accuracy": np.round(pixAcc, 3), "Mean_IoU": np.round(mIoU, 3),
                                "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))}

                if self.gpu == 0:
                    tbar.set_description('EVAL ({}) | Loss: {:.3f}, PixelAcc: {:.2f}, Mean IoU: {:.2f} |'.format( epoch,
                                                    total_loss_val.average, pixAcc, mIoU))

            if self.gpu == 0:
                self._add_img_tb(val_visual, 'val')

                # METRICS TO TENSORBOARD
                self.wrt_step = (epoch) * len(self.val_loader)
                self.writer.add_scalar(f'{self.wrt_mode}/loss', total_loss_val.average, self.wrt_step)
                for k, v in list(seg_metrics.items())[:-1]: 
                    self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)

            log = {
                'val_loss': total_loss_val.average,
                **seg_metrics
            }
            
        return log

    def _reset_metrics(self):
        self.loss_sup = AverageMeter()
        self.loss_unsup  = AverageMeter()
        self.loss_weakly = AverageMeter()
        self.pair_wise = AverageMeter()
        self.total_inter_l, self.total_union_l = 0, 0
        self.total_correct_l, self.total_label_l = 0, 0
        self.total_inter_ul, self.total_union_ul = 0, 0
        self.total_correct_ul, self.total_label_ul = 0, 0
        self.mIoU_l, self.mIoU_ul = 0, 0
        self.pixel_acc_l, self.pixel_acc_ul = 0, 0
        self.class_iou_l, self.class_iou_ul = {}, {}

    def _update_losses(self, cur_losses):
        for key in cur_losses:
            loss = cur_losses[key]
            n = loss.numel()
            count = torch.tensor([n]).long().cuda()
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            mean = loss.sum() / n
            if self.gpu == 0:
                getattr(self, key).update(mean.item())

    def _compute_metrics(self, outputs, target_l, target_ul, epoch):
        seg_metrics_l = eval_metrics(outputs['sup_pred'], target_l, self.num_classes, self.ignore_index)
        
        if self.gpu == 0:
            self._update_seg_metrics(*seg_metrics_l, True)
            seg_metrics_l = self._get_seg_metrics(True)
            self.pixel_acc_l, self.mIoU_l, self.class_iou_l = seg_metrics_l.values()

        # if 'unsup_pred' in outputs:
        #     seg_metrics_ul = eval_metrics(outputs['unsup_pred'], target_ul, self.num_classes, self.ignore_index)
            
        #     if self.gpu == 0:
        #         self._update_seg_metrics(*seg_metrics_ul, False)
        #         seg_metrics_ul = self._get_seg_metrics(False)
        #         self.pixel_acc_ul, self.mIoU_ul, self.class_iou_ul = seg_metrics_ul.values()
            
    def _update_seg_metrics(self, correct, labeled, inter, union, supervised=True):
        if supervised:
            self.total_correct_l += correct
            self.total_label_l += labeled
            self.total_inter_l += inter
            self.total_union_l += union
        else:
            self.total_correct_ul += correct
            self.total_label_ul += labeled
            self.total_inter_ul += inter
            self.total_union_ul += union



    def _get_seg_metrics(self, supervised=True):
        if supervised:
            pixAcc = 1.0 * self.total_correct_l / (np.spacing(1) + self.total_label_l)
            IoU = 1.0 * self.total_inter_l / (np.spacing(1) + self.total_union_l)
        else:
            pixAcc = 1.0 * self.total_correct_ul / (np.spacing(1) + self.total_label_ul)
            IoU = 1.0 * self.total_inter_ul / (np.spacing(1) + self.total_union_ul)
        mIoU = IoU.mean()
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))
        }



    def _log_values(self, cur_losses):
        logs = {}
        if "loss_sup" in cur_losses.keys():
            logs['loss_sup'] = self.loss_sup.average
        if "loss_unsup" in cur_losses.keys():
            logs['loss_unsup'] = self.loss_unsup.average
        if "loss_weakly" in cur_losses.keys():
            logs['loss_weakly'] = self.loss_weakly.average
        if "pair_wise" in cur_losses.keys():
            logs['pair_wise'] = self.pair_wise.average

        logs['mIoU_labeled'] = self.mIoU_l
        logs['pixel_acc_labeled'] = self.pixel_acc_l
        # if self.mode == 'semi':
        logs['mIoU_unlabeled'] = self.mIoU_ul
        logs['pixel_acc_unlabeled'] = self.pixel_acc_ul
        return logs


    def _write_scalars_tb(self, logs):
        for k, v in logs.items():
            if 'class_iou' not in k: self.writer.add_scalar(f'train/{k}', v, self.wrt_step)
        for i, opt_group in enumerate(self.optimizer_l.param_groups):
            self.writer.add_scalar(f'train/Learning_rate_{i}', opt_group['lr'], self.wrt_step)
        # current_rampup = self.model.module.unsup_loss_w.current_rampup
        # self.writer.add_scalar('train/Unsupervised_rampup', current_rampup, self.wrt_step)

    def _add_img_tb(self, val_visual, wrt_mode):
        val_img = []
        palette = self.val_loader.dataset.palette
        for imgs in val_visual:
            imgs = [self.restore_transform(i) if (isinstance(i, torch.Tensor) and len(i.shape) == 3) 
                        else colorize_mask(i, palette) for i in imgs]
            imgs = [i.convert('RGB') for i in imgs]
            imgs = [self.viz_transform(i) for i in imgs]
            val_img.extend(imgs)
        val_img = torch.stack(val_img, 0)
        val_img = make_grid(val_img.cpu(), nrow=val_img.size(0)//len(val_visual), padding=5)
        self.writer.add_image(f'{wrt_mode}/inputs_targets_predictions', val_img, self.wrt_step)



    def _write_img_tb(self, input_l, target_l, input_ul, target_ul, outputs, epoch):
        outputs_l_np = outputs['sup_pred'].data.max(1)[1].cpu().numpy()
        targets_l_np = target_l.data.cpu().numpy()
        imgs = [[i.data.cpu(), j, k] for i, j, k in zip(input_l, outputs_l_np, targets_l_np)]
        self._add_img_tb(imgs, 'supervised')

        # if self.mode == 'semi':
        #     outputs_ul_np = outputs['unsup_pred'].data.max(1)[1].cpu().numpy()
        #     targets_ul_np = target_ul.data.cpu().numpy()
        #     imgs = [[i.data.cpu(), j, k] for i, j, k in zip(input_ul, outputs_ul_np, targets_ul_np)]
        #     self._add_img_tb(imgs, 'unsupervised')

