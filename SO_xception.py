import argparse, os, random, time, datetime
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from timm.utils import AverageMeter
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

from logger import create_logger
from config import get_config

from network.models import model_selection
from utils import create_task_flags_FFSC, load_checkpoint, xception_default_data_transforms, \
    save_checkpoint, get_weight_str
from SO_loss import pLoss, graph_SO_FFSC
from lambda_optimizer import AutoLambda_FFSC
from dataset import set_dataset_FFSC_SLH, set_dataset_singleGPU_DFDC, set_dataset_singleGPU_FSh, \
    set_dataset_singleGPU_Deeper, MyDataset_FFSC, set_dataset_singleGPU_CDF

def parse_option():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--interpret', action='store_true')
    parser.add_argument('--name', type=str, default='xcp_binary_base')
    parser.add_argument('--output', type=str, default='./output')

    parser.add_argument('--gpu', default=0, type=int, help='gpu ID')
    parser.add_argument('--task', default='all', type=str, help='primary tasks, use all for MTL setting')
    parser.add_argument('--weight', default='equal', type=str, help='weighting methods: equal, dwa, uncert, autol')
    parser.add_argument('--grad_method', default='none', type=str, help='graddrop, pcgrad, cagrad')
    parser.add_argument('--autol_init', default=0.1, type=float, help='initialisation for auto-lambda')
    parser.add_argument('--autol_lr', default=1e-4, type=float, help='learning rate for auto-lambda')

    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--epochs', type=int, default=36)

    parser.add_argument('--num_out', type=int, default=2)
    parser.add_argument('--BATCH_SIZE', type=int, default=32)
    parser.add_argument('--NUM_WORKERS', type=int, default=8)
    parser.add_argument('--mode_label', choices=['global', 'all_local'], type=str)
    parser.add_argument('--is_SLH', action='store_true')
    parser.add_argument('--is_nWay', action='store_true')
    parser.add_argument('--aug', action='store_true')
    parser.add_argument('--aug_probs', type=float, default=0.3)

    parser.add_argument('--optim', type=str, choices=['adam', 'adamw'])
    parser.add_argument('--scheduler', type=str, choices=['step', 'cosineA'])
    parser.add_argument('--print_freq', type=int, default=20)
    parser.add_argument('--T_max', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--lr', type=float, default=0.0001)

    parser.add_argument('--txt_path_train', type=str, default='')
    parser.add_argument('--txt_path_val', type=str, default='')
    parser.add_argument('--n_frames', type=int, default=32)
    parser.add_argument('--dataset', type=str, default='CDF', help='FSh, Deeper')
    parser.add_argument('--ffsc_path', default="test", type=str)
    parser.add_argument('--datapath', type=str, default='/data/CDF/faces/')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return args, config

def test(args):
    model = model_selection(modelname='xception', num_out_classes=args.num_out,
                            dropout=0.)  # self.last_layer 2048-->num_out

    model = model.cuda()
    model_without_ddp = model

    criterion = pLoss(graph_SO_FFSC())

    if args.resume:
        max_accuracy = load_checkpoint(args, model_without_ddp, logger)

    transforms = xception_default_data_transforms['val']

    logger.info(f'eval dataset {args.dataset}......')
    if args.dataset == 'DFDC':
        dataset_val, data_loader_val = set_dataset_singleGPU_DFDC(preprocess=transforms, n_frames=args.n_frames)
        auc = test_gen(data_loader_val, model, criterion)
        logger.info(f' * AUC {auc: .4f}')

    elif args.dataset == 'Deeper':
        _, data_loader_dp = set_dataset_singleGPU_Deeper(preprocess=transforms,
                                                         datapath=args.datapath,#'/data/DF-1.0/',#'/data0/mian2/DeeperForensics/',
                                                         n_frames=args.n_frames)
        auc = test_gen(data_loader_dp, model, criterion)
        logger.info(f' * AUC {auc: .4f}')

    elif args.dataset == 'CDF':
        _, data_loader_CDF = set_dataset_singleGPU_CDF(preprocess=transforms,
                                                       datapath=args.datapath,#'/data0/mian2/celeb-df/dataset/',
                                                       n_frames=args.n_frames)
        auc = test_gen(data_loader_CDF, model, criterion)
        logger.info(f' * AUC {auc: .4f}')

    elif args.dataset == 'fsh':
        _, data_loader_fsh = set_dataset_singleGPU_FSh(preprocess=transforms,
                                                       n_frames=args.n_frames)
        auc = test_gen(data_loader_fsh, model, criterion)
        logger.info(f' * AUC {auc: .4f}')

    elif args.dataset == 'ffsc':
        txt_ffsc_path = '/data/FFSC/' + args.ffsc_path + '.txt'  # "/data0/mian2/vision-language/FFSC-test.txt"
        if not os.path.isfile(txt_ffsc_path):
            txt_ffsc_path = args.ffsc_path
        ffsc_val = MyDataset_FFSC(txt_ffsc_path, transforms, args.aug_probs)
        data_loader_ffsc = torch.utils.data.DataLoader(ffsc_val, batch_size=64, shuffle=False,
                                                       pin_memory=True, num_workers=4)

        auc, acc = test_gen_imgs(data_loader_ffsc, model, criterion)
        logger.info(f' * Acc {acc: .4f} * AUC {auc: .4f} ')


@torch.no_grad()
def test_gen(data_loader, model, criterion):
    model.eval()

    # for auc
    video_predict = []
    video_label = []

    for idx, (samples, targets) in enumerate(tqdm(data_loader)):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        preds = model(samples.squeeze(0))
        probs = criterion.infer(preds)[:, 0] #torch.sigmoid(preds[:, 0])

        probs_auc = probs
        frames_predict = probs_auc.detach().cpu().numpy().tolist()
        video_predict.append(np.mean(frames_predict))
        video_label.append(targets.detach().cpu().numpy())

    auc = roc_auc_score(video_label, video_predict) * 100
    return auc

@torch.no_grad()
def test_gen_imgs(data_loader, model, criterion):
    model.eval()

    video_predict_binary = []
    # for auc
    video_predict = []
    video_label = []

    for idx, (samples, targets) in enumerate(tqdm(data_loader)):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        preds = model(samples)
        probs = criterion.infer(preds)[:, 0] #torch.sigmoid(preds[:, 0])

        video_predict_binary.extend(
            (np.array(probs.detach().cpu()) >= 0.5).astype(int).tolist()
        )

        probs_auc = probs
        frames_predict = probs_auc.detach().cpu().numpy().tolist()
        video_predict.extend(frames_predict)
        video_label.extend(targets.detach().cpu().numpy().tolist())

    auc = roc_auc_score(video_label, video_predict) * 100
    acc = accuracy_score(video_label, video_predict_binary) * 100
    return auc, acc



def train(args):
    if args.eval:
        drop_value = 0
    else:
        drop_value = 0.5

    # define model, optimiser and scheduler
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    train_tasks = create_task_flags_FFSC('all')
    pri_tasks = create_task_flags_FFSC(args.task)
    train_tasks_str = ''.join(task.title() + ' + ' for task in train_tasks.keys())[:-3]
    pri_tasks_str = ''.join(task.title() + ' + ' for task in pri_tasks.keys())[:-3]
    logger.info('Training Task: {} | Primary Task: {} in Multi-task / Auxiliary Learning Mode '
                .format(train_tasks_str, pri_tasks_str))
    logger.info('Applying Multi-task Methods: Weighting-based: {} + Gradient-based: {}'
                .format(args.weight.title(), args.grad_method.upper()))

    model = model_selection(modelname='xception', num_out_classes=args.num_out,
                            dropout=drop_value)  # self.last_layer 2048-->num_out

    model = model.cuda()
    model_without_ddp = model

    criterion = pLoss(graph_SO_FFSC())

    # load the checkpoint
    if args.resume:
        max_accuracy = load_checkpoint(args, model_without_ddp, logger)

    total_epoch = args.epochs
    params = model.parameters()
    autol = AutoLambda_FFSC(model, device, train_tasks, pri_tasks, args.autol_init)
    meta_weight_ls = np.zeros([total_epoch, len(train_tasks)], dtype=np.float32)
    meta_optimizer = optim.Adam([autol.meta_weights], lr=args.autol_lr)

    if args.optim == 'adam': # default optimizer for xception
        optimizer = optim.Adam(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-08)
    elif args.optim == 'adamw':
        optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    if args.scheduler == 'step': # default scheduler for xception
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    elif args.scheduler == 'cosineA':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max)
    else:
        raise NotImplementedError


    transforms = xception_default_data_transforms
    dataset_train, data_loader_train = set_dataset_FFSC_SLH(args.txt_path_train, preprocess=transforms['train'],
                                                            args=args, logger=logger, phase='train', aug=args.aug)
    dataset_val, data_loader_val = set_dataset_FFSC_SLH(args.txt_path_val, logger=logger, preprocess=transforms['val'],
                                                        args=args, phase='val')
    # val_data for lambda optimization
    _, data_loader_val_autoL = set_dataset_FFSC_SLH(args.txt_path_train, preprocess=transforms['train'],
                                                            args=args, logger=logger, phase='train', aug=args.aug)
    val_loader_autoL = data_loader_val_autoL

    # train and evaluate
    log_writer = SummaryWriter(log_dir=output_path)
    logger.info("Start training")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.epochs):
        avg_train_loss, avg_train_bi_loss, avg_train_acc_bi = train_one_epoch(args, model, data_loader_train,
                                                                              val_loader_autoL, train_tasks,
                                                                              criterion,
                                                                              optimizer, meta_optimizer, autol,
                                                                              scheduler, epoch, log_writer)
        avg_val_acc_bi, auc = validate(args, data_loader_val, model, criterion)

        # record
        logger.info(
            f'Epoch {epoch} - avg_train_loss: {avg_train_loss:.4f} - avg_train_acc(bi): {avg_train_acc_bi:.4f} ')
        logger.info(f'Epoch {epoch} - avg_val_acc(bi): {avg_val_acc_bi:.4f} - AUC:{auc:.4f}')
        max_accuracy = max(max_accuracy, avg_val_acc_bi)
        logger.info(f'Max accuracy: {max_accuracy:.4f}')

        with open(os.path.join(output_path, 'logtxt.txt'), mode='a') as f:
            f.write(f'Epoch {epoch} - avg_train_loss: {avg_train_loss:.4f} - avg_train_acc(bi): {avg_train_acc_bi:.4f}')
            f.write('\n')
            f.write(f'Epoch {epoch} - avg_val_acc(bi): {avg_val_acc_bi:.4f} - AUC:{auc:.4f}')
            f.write('\n')
            f.write('\n')

        scheduler.step()
        # save checkpoint
        save_checkpoint(args, output_path, epoch, model_without_ddp, max_accuracy, optimizer, logger)
        # save/record the best model
        if avg_val_acc_bi >= max_accuracy:
            with open(os.path.join(output_path, 'logtxt.txt'), mode='a') as f:
                f.write(f'Current best model for binary classification is in Epoch {epoch}!')
                f.write('\n')

        meta_weight_ls[epoch] = autol.meta_weights.detach().cpu()
        logger.info(get_weight_str(meta_weight_ls[epoch], train_tasks))
        try:
            f.write(get_weight_str(meta_weight_ls[epoch], train_tasks))
            f.write('\n')
        except:
            continue


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    logger.info('Best val Acc: {:.4f}'.format(max_accuracy))



def train_one_epoch(args, model, data_loader, val_loader, train_tasks, criterion,
                    optimizer, meta_optimizer, autol, scheduler, epoch, log_writer):
    model.train()

    batch_time = AverageMeter()
    num_steps = len(data_loader)
    loss_meter = AverageMeter()
    loss_w_meter = AverageMeter()
    loss_meter_binary = AverageMeter()
    binary_acc_meter = AverageMeter()

    logger.info(optimizer.state_dict()['param_groups'][0]['lr'])
    if optimizer.state_dict()['param_groups'][0]['lr'] == 0:
        scheduler.step()
        logger.info(optimizer.state_dict()['param_groups'][0]['lr'])

    start = time.time()
    end = time.time()

    val_dataset = iter(val_loader)

    for idx, (samples, label, mask, label_infer) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets_train = label.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        targets_infer = label_infer.cuda(non_blocking=True)

        val_samples, val_label, val_mask, val_label_infer = next(val_dataset)  # .next()
        val_samples = val_samples.cuda(non_blocking=True)
        val_label = val_label.cuda(non_blocking=True)
        val_mask = val_mask.cuda(non_blocking=True)

        meta_optimizer.zero_grad()

        autol.unrolled_backward(samples, targets_train, mask,
                                val_samples, val_label, val_mask,
                                scheduler.get_last_lr()[0],
                                scheduler.get_last_lr()[0], optimizer)

        meta_optimizer.step()


        optimizer.zero_grad()
        outputs = model(samples)
        train_loss, pMargin = compute_loss_FFSC_graph(outputs, targets_train, mask, criterion)
        total_loss = sum(train_loss)

        train_loss_tmp = [w * train_loss[i] for i, w in enumerate(autol.meta_weights)]
        loss = sum(train_loss_tmp)

        loss.backward()
        optimizer.step()

        # metric computation
        binary_acc = compute_acc_graph(pMargin, targets_infer)
        # metric update
        loss_meter.update(total_loss.item(), targets_infer.size(0))
        loss_w_meter.update(loss.item(), targets_infer.size(0))
        loss_meter_binary.update(train_loss[0], targets_infer.size(0))
        binary_acc_meter.update(binary_acc.item(), targets_infer.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{args.epochs}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'loss_w {loss_w_meter.val:.4f} ({loss_w_meter.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'loss_b {loss_meter_binary.val:.4f} ({loss_meter_binary.avg:.4f})\t'
                f'bi_acc {binary_acc_meter.val:.4f} ({binary_acc_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

            log_writer.add_scalar('Train/total_loss', loss_meter.val,
                                  int((idx / len(data_loader) + epoch) * len(data_loader)))  # *1000
            log_writer.add_scalar('Train/loss_weight', loss_w_meter.val,
                                  int((idx / len(data_loader) + epoch) * len(data_loader)))
            log_writer.add_scalar('Train/binary_loss', loss_meter_binary.val,
                                  int((idx / len(data_loader) + epoch) * len(data_loader)))
            log_writer.add_scalar('Train/binary_acc', binary_acc_meter.val,
                                  int((idx / len(data_loader) + epoch) * len(data_loader)))

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    return loss_meter.avg, loss_meter_binary.avg, binary_acc_meter.avg

@torch.no_grad()
def validate(args, data_loader, model, criterion):
    model.eval()

    batch_time = AverageMeter()
    num_steps = len(data_loader)
    binary_acc_meter = AverageMeter()

    start = time.time()
    end = time.time()
    all_probs = []
    all_labels = []
    for idx, (samples, _, _, label) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        if args.train_dataset in ['ffsc']:
            targets = label.cuda(non_blocking=True)
        else:
            targets = label[:,1:].cuda(non_blocking=True) # if ffpp, cdf, others for binary cls

        test_pred = model(samples)
        pMargin = criterion.infer(test_pred)
        # acc
        binary_acc = compute_acc_graph(pMargin, targets)
        binary_acc_meter.update(binary_acc, targets.size(0))
        # auc
        logits = pMargin[:, 0] #torch.sigmoid(test_pred[:, 0])
        all_probs.append(logits)
        all_labels.append(targets[:, 0])

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_freq == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Validation: [{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} \t'
                f'bi_acc {binary_acc_meter.val:.4f} ({binary_acc_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

    all_probs = torch.cat(all_probs).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()
    auc = roc_auc_score(all_labels, all_probs) * 100
    return binary_acc_meter.avg, auc


def compute_loss_FFSC_graph(pred, gt, mask, criterion, tasks_num=3):
    total_loss, pMargin = criterion(pred, gt.float(), mask, auto_mode=True)
    if tasks_num == 12:
        return total_loss, pMargin

    # for 3-level tasks
    loss_1 = total_loss[0]
    loss_2 = compute_one_level_loss(total_loss[1:6])
    loss_3 = compute_one_level_loss(total_loss[6:])
    loss = [loss_1, loss_2, loss_3]

    return loss, pMargin

def compute_one_level_loss(loss):
    total_loss = 0
    for loss_item in loss:
        total_loss += loss_item
    total_loss = total_loss / len(loss)
    return total_loss

def compute_acc_graph(pred, gt):
    targets_gsa = gt
    try:
        targets_gsa_binary = targets_gsa[:, 0]
    except:
        targets_gsa_binary = targets_gsa

    y_pred_binary = pred[:, 0].data
    y_pred_binary_ls = torch.zeros_like(y_pred_binary) - 1
    y_pred_binary_ls[y_pred_binary >= 0.5] = 1
    binary_acc = torch.sum(y_pred_binary_ls == targets_gsa_binary).to(torch.float32) / gt.size(0)
    return binary_acc

def setup_seed(seed):
    # fix the seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  #
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    args, config = parse_option()

    setup_seed(args.seed)

    output_path = os.path.join(args.output, args.name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logger = create_logger(output_dir=output_path, name=f"{args.name}")

    if not args.eval:
        train(args)
    else:
        test(args)