# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>, Valentine Oleka <valentinenwakibeya@gmail.com>

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from datetime import datetime as dt


def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)

    return x


def init_weights(m):
    if type(m) == torch.nn.Conv2d or type(m) == torch.nn.Conv3d or type(m) == torch.nn.ConvTranspose3d:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.BatchNorm3d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.constant_(m.bias, 0)


def save_checkpoints(cfg, file_path, epoch_idx, encoder, encoder_solver, decoder, decoder_solver, refiner,
                     refiner_solver, merger, merger_solver, best_iou, best_epoch):
    print('[INFO] %s Saving checkpoint to %s ...' % (dt.now(), file_path))
    checkpoint = {
        'epoch_idx': epoch_idx,
        'best_iou': best_iou,
        'best_epoch': best_epoch,
        'encoder_state_dict': encoder.state_dict(),
        'encoder_solver_state_dict': encoder_solver.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'decoder_solver_state_dict': decoder_solver.state_dict()
    }

    if cfg.NETWORK.USE_REFINER:
        checkpoint['refiner_state_dict'] = refiner.state_dict()
        checkpoint['refiner_solver_state_dict'] = refiner_solver.state_dict()
    if cfg.NETWORK.USE_MERGER:
        checkpoint['merger_state_dict'] = merger.state_dict()
        checkpoint['merger_solver_state_dict'] = merger_solver.state_dict()

    torch.save(checkpoint, file_path)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def save_image_voxels(voxels, path, iteration):
    voxels = voxels[:8].__ge__(0.5)
    fig = plt.figure(figsize=(32, 16))
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(voxels):
        x, y, z = sample.nonzero()
        ax = plt.subplot(gs[i], projection='3d')
        ax.scatter(x, y, z, zdir='z', c='red')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('auto')
    plt.savefig(path + '/{}.png'.format(str(iteration).zfill(3)), bbox_inches='tight')
    plt.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
