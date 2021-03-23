"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import argparse

import numpy as np
import torch
import torch.backends.cudnn as cudnn
# import metrics
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score)

from trainer import MUNIT_Trainer, UNIT_Trainer
from utils import (get_all_data_loaders, get_config, prepare_sub_folder)

try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass
import os
# import tensorboardX
import shutil
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str,
                    default='configs/demo_cxr_lungs.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str,
                    default='output', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']
config['vgg_model_path'] = opts.output_path

cuda0 = torch.device('cuda:0')
cuda1 = torch.device('cuda:0')
devices = (cuda0, cuda1)

train_loader, test_loader, train_count, test_count = get_all_data_loaders(
    config)
max_sample = max(train_count)
loss_weights = [max_sample/x for x in train_count]
loss_weights = torch.Tensor(loss_weights).cuda()


# Setup model and data loader
if opts.trainer == 'MUNIT':
    trainer = MUNIT_Trainer(config, devices, loss_weights)
elif opts.trainer == 'UNIT':
    trainer = UNIT_Trainer(config, devices)
else:
    sys.exit("Only support MUNIT|UNIT")
# trainer.cuda(cuda0)


train_img_samples = []
train_lbl_samples = []
test_img_samples = []
test_lbl_samples = []

for d in range(config['n_datasets']):

    train_samples, train_lbl = train_loader.dataset.load_samples(
        display_size, d)
    test_samples, test_lbl = test_loader.dataset.load_samples(display_size, d)

    train_img_samples.append(torch.stack(train_samples))
    train_lbl_samples.append(np.stack(train_lbl))
    test_img_samples.append(torch.stack(test_samples))
    test_lbl_samples.append(np.stack(test_lbl))

# train_display_images_list = [torch.stack(train_loader.dataset.load_samples(display_size, d)[0]) for d in range(config['n_datasets'])]
# test_display_images_list = [torch.stack(test_loader.dataset.load_samples(display_size, d)[0]) for d in range(config['n_datasets'])]

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
# copy config file to output folder
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))
# creating log folder
if not os.path.exists('logs/'):
    print("Creating directory: {}".format('logs/'))
    os.makedirs('logs/')
# Start training
iterations = trainer.resume(
    checkpoint_directory, hyperparameters=config) if opts.resume else 0


print('Training starts...')
while True:

    for it, data in enumerate(train_loader):

        img_a = data[0].cuda(cuda0).detach()
        img_b = data[1].cuda(cuda0).detach()

        ind_a = data[2][0].item()
        ind_b = data[3][0].item()

        label_a = data[4].cuda(cuda1).detach()
        label_b = data[5].cuda(cuda1).detach()

        # Translation forward and backward.
        trainer.dis_update(img_a, img_b, ind_a, ind_b, config)
        gen_losses = trainer.gen_update(img_a, img_b, ind_a, ind_b, config)
        torch.cuda.synchronize()

        # Supervised forward and backward.
        sup_losses = trainer.sup_update(
            img_a, img_b, label_a, label_b, ind_a, ind_b, config)

        recon_x_w_a, recon_s_w_a, recon_c_w_a, recon_x_w_b, recon_s_w_b, recon_c_w_b, recon_x_cyc_w_a, recon_x_cyc_w_b, loss_gen = gen_losses
        sup_a, sup_b, sup_a_recon, sup_b_recon, sup_loss = sup_losses

        print('Iteration: %d/%d, datasets [%s, %s]' % (
            iterations + 1, max_iter, ind_a, ind_b),)
        print('    Gen Losses: x_w_a %.3f, s_w_a %.3f, c_w_a %.3f, x_w_b %.3f, s_w_b %.3f, c_w_b %.3f, x_cyc_w_a %.3f, x_cyc_w_b %.3f, loss_gen %.3f' % (
            recon_x_w_a, recon_s_w_a, recon_c_w_a, recon_x_w_b, recon_s_w_b, recon_c_w_b, recon_x_cyc_w_a, recon_x_cyc_w_b, loss_gen),)

        print('    Sup Losses: a %.3f, b %.3f, a_recon %.3f, b_recon %.3f, loss_sup %.3f' % (
            sup_a, sup_b, sup_a_recon, sup_b_recon, sup_loss),)

        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:

            # recon_x_w_a, recon_s_w_a, recon_c_w_a, recon_x_w_b, recon_s_w_b, recon_c_w_b, recon_x_cyc_w_a, recon_x_cyc_w_b, loss_gen = gen_losses
            # sup_a, sup_b, sup_a_recon, sup_b_recon, sup_loss = sup_losses

            print('Iteration: %d/%d, datasets [%s, %s]' % (
                iterations + 1, max_iter, ind_a, ind_b), file=open(str(opts.output_path + "/output.txt"), "a"))
            print('    Gen Losses: x_w_a %.3f, s_w_a %.3f, c_w_a %.3f, x_w_b %.3f, s_w_b %.3f, c_w_b %.3f, x_cyc_w_a %.3f, x_cyc_w_b %.3f, loss_gen %.3f' % (
                recon_x_w_a, recon_s_w_a, recon_c_w_a, recon_x_w_b, recon_s_w_b, recon_c_w_b, recon_x_cyc_w_a, recon_x_cyc_w_b, loss_gen), file=open(str(opts.output_path + "/output.txt"), "a"))

            print('    Sup Losses: a %.3f, b %.3f, a_recon %.3f, b_recon %.3f, loss_sup %.3f' % (
                sup_a, sup_b, sup_a_recon, sup_b_recon, sup_loss), file=open(str(opts.output_path + "/output.txt"), "a"))
            sys.stdout.flush()

        if (iterations + 1) % config['snapshot_test_epoch'] == 0:

            metric_file = open('logs/' + opts.config.split('/')
                               [-1].replace('.yaml', '_metric.log'), 'a')
            y_pred = list()
            y_true = list()

            print("Testing...")

            for it, data in enumerate(test_loader):
                x1, x2, ind1, ind2, y1, y2 = data

                x1 = x1.cuda(cuda1)
                x2 = x1.cuda(cuda1)
                y1 = y1.cuda(cuda1)
                y2 = y2.cuda(cuda1)

                preds1 = trainer.sup_forward(x1, ind1[0].item(), config)
                preds2 = trainer.sup_forward(x2, ind2[0].item(), config)

                y_pred.extend(preds1.cpu().numpy())
                y_pred.extend(preds2.cpu().numpy())

                y_true.extend(y1.cpu().numpy())
                y_true.extend(y2.cpu().numpy())

            weighted_acc = balanced_accuracy_score(y_true, y_pred)

            precision = precision_score(y_true, y_pred, average='macro')

            cm = confusion_matrix(y_true, y_pred, normalize='true')

            f1 = f1_score(y_true, y_pred, average='macro')

            recall = recall_score(y_true, y_pred, average='macro')

            print("Test Balanced Accuracy iteration {}: {}".format(
                (iterations+1), (100 * weighted_acc)))

            print("Test Balanced Accuracy iteration {}: {}".format(
                (iterations+1), (100 * weighted_acc)), file=metric_file)

            print("Test Precision iteration {}: {}".format(
                (iterations+1), (100 * precision)))

            print("Test Precision iteration {}: {}".format(
                (iterations+1), (100 * precision)), file=metric_file)

            print("Test F1 Score iteration {}: {}".format(
                (iterations+1), (100 * f1)))

            print("Test F1 Score iteration {}: {}".format(
                (iterations+1), (100 * f1)), file=metric_file)

            print("Test Recall iteration {}: {}".format(
                (iterations+1), (100 * recall)))

            print("Test Recall iteration {}: {}".format(
                (iterations+1), (100 * recall)), file=metric_file)

            print("Test Confusion Matrix iteration {}: \n{}\n".format(
                (iterations+1), cm))

            print("Test Confusion Matrix iteration {}: \n{}\n".format(
                (iterations+1), cm), file=metric_file)

        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')

        trainer.update_learning_rate()
