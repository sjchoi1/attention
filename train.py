'''
This script handles the training process.
'''

import argparse
import math
import time
import dill as pickle
from tqdm import tqdm
import numpy as np
import random
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import transformer.Constants as Constants
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
from transformer.Dataset import CustomDataset

__author__ = "Yu-Hsiang Huang"

def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''

    pred = pred.contiguous().view(-1, 16)
    loss = cal_loss(pred, gold, smoothing=smoothing)
    pred = pred.max(1)[1]
    # gold = gold.max(1)[1]

    n_char = pred.shape[0]
    n_word = n_char / 7
    n_char_correct = pred.eq(gold).sum().item()
    n_word_correct = torch.all(torch.eq(pred.view(-1, 7), gold.view(-1, 7))).sum().item()
    return loss, n_char_correct, n_word_correct, n_char, n_word

def cal_loss(pred, gold, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    loss = F.cross_entropy(pred, gold, reduction='sum')

    # pred = pred.contiguous().view(-1, 112)
    # gold = gold.contiguous().view(-1, 112)
    # loss = F.binary_cross_entropy_with_logits(pred, gold, reduction='mean')
    # print(loss)
    return loss

def patch_trg(trg):
    # trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1, 16)
    gold = gold.max(1)[1]

    # gold = torch.mm(gold, normalize).view(-1).long()
    return trg, gold

def train_epoch(epoch, model, training_data, optimizer, opt, device, smoothing):
    ''' Epoch operation in training phase'''

    model.train()
    total_loss, global_word_total, global_word_correct = 0, 0, 0 
    temp_total_loss, temp_word_total, temp_word_correct = 0, 0, 0
    global_char_total, temp_char_total = 0, 0
    global_char_correct, temp_char_correct = 0, 0
    desc = '  - (Training)   '
    for i, batch in enumerate(training_data):
        # prepare data
        src_seq = batch['src'].to(device)
        trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch['trg']))

        # forward
        optimizer.zero_grad()
        pred = model(src_seq, trg_seq)

        # backward and update parameters
        loss, n_char_correct, n_word_correct, n_char, n_word = cal_performance(pred, gold) 
        loss.backward()
        optimizer.step_and_update_lr()

        # note keeping
        global_word_total += n_word
        global_char_total += n_char 
        temp_word_total += n_word 
        temp_char_total += n_char

        global_word_correct += n_word_correct
        global_char_correct += n_char_correct 
        temp_word_correct += n_word_correct
        temp_char_correct += n_char_correct 

        total_loss += loss.item()
        temp_total_loss += loss.item()

        log_interval = 5
        if i % log_interval == 0 and i > 0:
            cur_loss = temp_total_loss / log_interval
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                'final acc {:5.2f} | char acc {:5.2f} | loss {:5.2f} | '
                'lr {:8.5f} |'.format(
                    epoch, i, len(training_data), 
                    temp_word_correct / temp_word_total, 
                    temp_char_correct / temp_char_total, 
                    cur_loss, optimizer._optimizer.param_groups[0]['lr']))
            temp_total_loss = 0
            temp_word_total = 0
            temp_char_total = 0
            temp_word_correct = 0
            temp_char_correct = 0

    loss_per_word = total_loss/global_word_total
    accuracy = global_word_correct/global_word_total
    return loss_per_word, accuracy

def eval_epoch(epoch, model, validation_data, device, opt):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    total_loss, global_word_total, global_word_correct = 0, 0, 0 
    temp_total_loss, temp_word_total, temp_word_correct = 0, 0, 0
    global_char_total, temp_char_total = 0, 0
    global_char_correct, temp_char_correct = 0, 0

    desc = '  - (Validation) '
    with torch.no_grad():
        for i, batch in enumerate(validation_data):
            # prepare data
            src_seq = batch['src'].to(device)
            trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch['trg']))

            # forward
            pred = model(src_seq, trg_seq)
            loss, n_char_correct, n_word_correct, n_char, n_word = cal_performance(pred, gold) 

            # note keeping
            global_word_total += n_word
            global_char_total += n_char 
            temp_word_total += n_word 
            temp_char_total += n_char

            global_word_correct += n_word_correct
            global_char_correct += n_char_correct 
            temp_word_correct += n_word_correct
            temp_char_correct += n_char_correct 

            total_loss += loss.item()
            temp_total_loss += loss.item()

            log_interval = 100
            if i % log_interval == 0 and i > 0:
                cur_loss = temp_total_loss / log_interval
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                    'final acc {:5.2f} | char acc {:5.2f} | loss {:5.2f} | '
                    'lr {:8.5f} |'.format(
                        epoch, i, len(validation_data), 
                        temp_word_correct / temp_word_total, 
                        temp_char_correct / temp_char_total, 
                        cur_loss, optimizer._optimizer.param_groups[0]['lr']))
                temp_total_loss = 0
                temp_word_total = 0
                temp_char_total = 0
                temp_word_correct = 0
                temp_char_correct = 0

    loss_per_word = total_loss/global_word_total
    accuracy = global_word_correct/global_word_total
    return loss_per_word, accuracy


def train(model, training_data, validation_data, optimizer, device, opt):
    ''' Start training '''

    # Use tensorboard to plot curves, e.g. perplexity, accuracy, learning rate
    if opt.use_tb:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=os.path.join(opt.output_dir, 'tensorboard'))

    log_train_file = os.path.join(opt.output_dir, 'train.log')
    log_valid_file = os.path.join(opt.output_dir, 'valid.log')

    print('[Info] Training performance will be written to file: {} and {}'.format(
        log_train_file, log_valid_file))

    with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
        log_tf.write('epoch,loss,ppl,accuracy\n')
        log_vf.write('epoch,loss,ppl,accuracy\n')

    def print_performances(header, ppl, accu, start_time, lr):
        print('  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, lr: {lr:8.5f}, '\
              'elapse: {elapse:3.3f} min'.format(
                  header=f"({header})", ppl=ppl,
                  accu=100*accu, elapse=(time.time()-start_time)/60, lr=lr))

    #valid_accus = []
    valid_losses = []
    for epoch_i in range(opt.epoch):
        # print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(
            epoch_i, model, training_data, optimizer, opt, device, smoothing=False)
        train_ppl = math.exp(min(train_loss, 100))
        # Current learning rate
        lr = optimizer._optimizer.param_groups[0]['lr']
        print_performances('Training', train_ppl, train_accu, start, lr)

        start = time.time()
        valid_loss, valid_accu = eval_epoch(epoch_i, model, validation_data, device, opt)
        valid_ppl = math.exp(min(valid_loss, 100))
        print_performances('Validation', valid_ppl, valid_accu, start, lr)

        valid_losses += [valid_loss]

        checkpoint = {'epoch': epoch_i, 'settings': opt, 'model': model.state_dict()}

        if opt.save_mode == 'all':
            model_name = 'model_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
            torch.save(checkpoint, model_name)
        elif opt.save_mode == 'best':
            model_name = 'model.chkpt'
            if valid_loss <= min(valid_losses):
                torch.save(checkpoint, os.path.join(opt.output_dir, model_name))
                print('    - [Info] The checkpoint file has been updated.')

        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
            log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=train_loss,
                ppl=train_ppl, accu=100*train_accu))
            log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=valid_loss,
                ppl=valid_ppl, accu=100*valid_accu))

        if opt.use_tb:
            tb_writer.add_scalars('ppl', {'train': train_ppl, 'val': valid_ppl}, epoch_i)
            tb_writer.add_scalars('accuracy', {'train': train_accu*100, 'val': valid_accu*100}, epoch_i)
            tb_writer.add_scalar('learning_rate', lr, epoch_i)

def main():
    ''' 
    Usage:
    python train.py
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-workload', type= str, default='bc')
    parser.add_argument('-thread_cnt', type=str, default='4')
    parser.add_argument('-look_back', type=int, default=8)
    parser.add_argument('-look_front', type=int, default=8)

    parser.add_argument('-epoch', type=int, default=1000)
    parser.add_argument('-b', '--batch_size', type=int, default=2048)

    parser.add_argument('-d_model', type=int, default=112)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    # parser.add_argument('-d_k', type=int, default=22)
    # parser.add_argument('-d_v', type=int, default=22)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-warmup','--n_warmup_steps', type=int, default=8000)
    parser.add_argument('-lr_mul', type=float, default=2.0)
    parser.add_argument('-seed', type=int, default=None)

    parser.add_argument('-dropout', type=float, default=0.1)

    parser.add_argument('-output_dir', type=str, default=None)
    parser.add_argument('-use_tb', action='store_true')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    # https://pytorch.org/docs/stable/notes/randomness.html
    # For reproducibility
    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = False
        torch.set_deterministic(True)
        np.random.seed(opt.seed)
        random.seed(opt.seed)

    if not opt.output_dir:
        opt.output_dir = ('output/' + opt.workload + '_t' + opt.thread_cnt + 
                            '_lb' + str(opt.look_back) + '_lf' + str(opt.look_front))

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    if opt.batch_size < 2048 and opt.n_warmup_steps <= 4000:
        print('[Warning] The warmup steps may be not enough.\n'\
              '(sz_b, warmup) = (2048, 4000) is the official setting.\n'\
              'Using smaller batch w/o longer warmup may cause '\
              'the warmup stage ends with only little data trained.')

    device = torch.device('cuda' if opt.cuda else 'cpu')

    #========= Loading Dataset =========#
    training_data, validation_data = prepare_dataloaders(opt)

    print(opt)

    transformer = Transformer(
        n_src_vocab=48,
        n_trg_vocab=48,
        # d_model=opt.d_model,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout)

    # Use multi GPU
    # if torch.cuda.device_count() > 1:
    #     transformer = torch.nn.DataParallel(transformer)

    transformer.to(device)

    optimizer = ScheduledOptim(
        optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09),
        opt.lr_mul, opt.d_model, opt.n_warmup_steps)

    train(transformer, training_data, validation_data, optimizer, device, opt)

def prepare_dataloaders(opt):
    train_csv = 'data/' + opt.workload + '/csv/t' + opt.thread_cnt + '/train.csv'
    val_csv = 'data/' + opt.workload + '/csv/t' + opt.thread_cnt + '/val.csv'

    print('[Info] Preparing custom dataset')
    train_dataset = CustomDataset(train_csv, opt.look_back, opt.look_front, opt.thread_cnt)
    val_dataset = CustomDataset(val_csv, opt.look_back, opt.look_front, opt.thread_cnt)

    print('[Info] Preparing dataloader')
    train_iterator = DataLoader(train_dataset, batch_size=opt.batch_size, pin_memory=True, shuffle=True)
    val_iterator = DataLoader(val_dataset, batch_size=opt.batch_size, pin_memory=True)

    return train_iterator, val_iterator

if __name__ == '__main__':
    main()
