from dataset.dataset_kpt import build_dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import config
from tensorboardX import SummaryWriter
from models.ResNet import resnet50
from utils import misc
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def resolve_device(requested_device: str) -> torch.device:
    device = torch.device(requested_device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        print('CUDA requested but unavailable. Falling back to CPU.')
        return torch.device('cpu')
    return device


if __name__ == '__main__':
    args = config.args
    save_path = './experiments/'
    model_name = 'train'
    torch.backends.cudnn.benchmark = True
    print(model_name)
    if model_name != 'test':
        if not os.path.exists('./experiments/param/' + model_name + '/'):
            os.makedirs('./experiments/conf_matrix/' + model_name)
            os.makedirs('./experiments/weights/' + model_name)
            os.makedirs('./experiments/savept/' + model_name)
            os.makedirs('./experiments/param/' + model_name)
    device = resolve_device(args.device)
    amp_enabled = args.amp and device.type == 'cuda'
    # data info
    Train_Dataset = build_dataset(args.dataset_root_kpt, True)
    train_loader_kwargs = {
        'num_workers': args.num_workers,
        'pin_memory': args.pin_memory,
        'shuffle': True,
        'drop_last': True,
    }
    test_loader_kwargs = {
        'num_workers': args.num_workers,
        'pin_memory': args.pin_memory,
        'shuffle': False,
        'drop_last': True,
    }
    if args.num_workers > 0:
        train_loader_kwargs['persistent_workers'] = args.persistent_workers
        test_loader_kwargs['persistent_workers'] = args.persistent_workers
        train_loader_kwargs['prefetch_factor'] = args.prefetch_factor
        test_loader_kwargs['prefetch_factor'] = args.prefetch_factor
    train_loader = DataLoader(
        dataset=Train_Dataset,
        batch_size=args.batch_size,
        **train_loader_kwargs,
    )
    train_size = int(Train_Dataset.__len__())

    Test_Dataset = build_dataset(args.dataset_root_kpt, False)
    test_loader = DataLoader(
        dataset=Test_Dataset,
        batch_size=args.batch_size,
        **test_loader_kwargs,
    )
    test_size = int(Test_Dataset.__len__())
    print('model_name: ', model_name)
    print('train_size: ', train_size)
    print('test_size: ', test_size)
    num_class = 8
    model = resnet50(num_classes=num_class)
    model = model.to(device)
    loss_ce = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    train_logger, test_logger = misc.create_log(save_path + model_name)
    train_writer = SummaryWriter('logs/' + model_name + '/train')
    test_writer = SummaryWriter('logs/' + model_name + '/test')
    best_acc_train = 0.0
    best_acc_test = 0.0
    temp_test = 0
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    for i in range(args.epoch):
        print('epoch: ', i)
        losses = 0
        correct_train = 0
        conf_matrix = [[0 for _ in range(num_class)] for _ in range(num_class)]
        model.train()
        for kpt, label in tqdm(train_loader):
            kpt = kpt.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                output = model(kpt)
                loss = loss_ce(output, label)
            prediction = output.data.max(1)[1]
            correct_train += prediction.eq(label.data).sum()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses += loss.item()
        lr_scheduler.step()
        train_acc = 100 * float(correct_train) / train_size
        best_acc_train, _ = misc.log_save_kpt(model_name, train_writer, train_logger, i, losses,
                                                         train_size, best_acc_train, train_acc, conf_matrix, mode=True)
        loss_dict = {}
        losses = 0
        correct_test = 0
        model.eval()
        for kpt, label in tqdm(test_loader):
            with torch.no_grad():
                kpt = kpt.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    output = model(kpt)
                    loss = loss_ce(output, label)
                prediction = output.data.max(1)[1]
                conf_matrix = misc.get_conf_matrix(prediction, label, conf_matrix)
                correct_test += prediction.eq(label.data).sum()
                losses += loss.item()
        test_acc = 100 * float(correct_test) / test_size
        best_acc_test, is_save = misc.log_save_kpt(model_name, test_writer, test_logger, i, losses,
                                                   test_size, best_acc_test, test_acc, conf_matrix, mode=False)
        if is_save:
            torch.save(model.state_dict(), f'./experiments/param/{model_name}/best_test_mpjpe.pth')

    train_writer.close()
    test_writer.close()
    print('model_name: ', model_name)
