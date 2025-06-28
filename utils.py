import torch
import os
from torchvision import transforms

xception_default_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ]),
    'val': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'plot': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        # transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
}

def create_task_flags_FFSC(task):
    semantic_tasks_3lvls = {'binary': 1, 'global': 1, 'local': 1}
    tasks = {}
    if task != 'all' and task != 'global':
        tasks[task] = semantic_tasks_3lvls[task]
    elif task == 'global':
        tasks = {'binary': 1, 'global': 1}
    else:
        tasks = semantic_tasks_3lvls
    return tasks


def load_checkpoint(args, model, logger):
    logger.info(f"==============> Resuming form {args.resume}....................")
    if args.resume.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.resume, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(args.resume, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy

def save_checkpoint(args, output_path, epoch, model, max_accuracy, optimizer, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  # 'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  # 'scaler': loss_scaler.state_dict(),
                  'epoch': epoch,
                  'args': args}

    save_path = os.path.join(output_path, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_weight_str(weight, tasks):
    """
    Record task weighting.
    """
    weight_str = 'Task Weighting | '
    for i, task_id in enumerate(tasks):
        weight_str += '{} {:.04f} '.format(task_id.title(), weight[i])
    return weight_str