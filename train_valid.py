# --------------------------------------------------------
# Copyright (c) 2022 BiRSwinT.
# Licensed under The MIT License.
# --------------------------------------------------------

import torch
from torchvision import datasets, transforms
import torch.nn as nn
import time
from pathlib import Path
from BiRSwinT import BiRSwinT
from timm.data.transforms import RandomResizedCropAndInterpolation
import PIL
from optimizer import build_optimizer
from lr_scheduler import build_scheduler
import argparse
from config import get_config
from context import ctx


def parse_option():
    parser = argparse.ArgumentParser("BiRSwinT Test script", add_help=False)
    parser.add_argument(
        "--cfg",
        default="configs/swin_small_patch4_window7_224.yaml",
        type=str,
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs="+",
    )

    # easy config modification
    parser.add_argument("--batch-size", type=int, help="batch size for single GPU")
    parser.add_argument("--data-path", type=str, help="path to dataset")
    parser.add_argument("--zip", action="store_true", help="use zipped dataset instead of folder dataset")
    parser.add_argument(
        "--cache-mode",
        type=str,
        default="part",
        choices=["no", "full", "part"],
        help="no: no cache, "
        "full: cache all data, "
        "part: sharding the dataset into nonoverlapping pieces and only cache one piece",
    )
    parser.add_argument(
        "--resume",
        default="output/swin_small_patch4_window7_224/default/ckpt_epoch_23.pth",
        help="resume from checkpoint",
    )
    parser.add_argument("--accumulation-steps", type=int, help="gradient accumulation steps")
    parser.add_argument(
        "--use-checkpoint", action="store_true", help="whether to use gradient checkpointing to save memory"
    )
    parser.add_argument(
        "--amp-opt-level",
        type=str,
        default="O1",
        choices=["O0", "O1", "O2"],
        help="mixed precision opt level, if O0, no amp is used",
    )
    parser.add_argument(
        "--output",
        default="output",
        type=str,
        metavar="PATH",
        help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)",
    )
    parser.add_argument("--tag", help="tag of experiment")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--throughput", action="store_true", help="Test throughput only")
    parser.add_argument("--local_rank", default="0", type=int, help="local rank for DistributedDataParallel")
    args, _ = parser.parse_known_args()

    config = get_config(args)

    return args, config


def train_and_valid(
    config, model, train_data, valid_data, loss_function, optimizer, lr_sche, train_data_size, valid_data_size
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 若有 GPU 可用则使用 GPU
    model.to(device)
    step = 1

    TRAINED_MODEL_PARA = "output/ckpt_"
    i = 0
    trained_param_path = Path(TRAINED_MODEL_PARA + str(i) + ".pth")
    while trained_param_path.is_file():
        i += 1
        trained_param_path = Path(TRAINED_MODEL_PARA + str(i) + ".pth")

    record = []
    best_acc = 0.0
    best_epoch = 0

    epochs = config.TRAIN.NUM_EPOCHS
    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        model.train()  # training

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        for _, (inputs, labels) in enumerate(train_data):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_function(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            _, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            train_acc += acc.item() * inputs.size(0)

            lr_sche.step()

            step += 1

        with torch.no_grad():
            model.eval()  # validation
            correct = 0
            correct3 = 0
            for _, (inputs, labels) in enumerate(valid_data):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                loss = loss_function(outputs, labels)

                valid_loss += loss.item() * inputs.size(0)

                maxk = max((1, 5))
                label_resize = labels.view(-1, 1)
                _, predicted = outputs.topk(maxk, 1, True, True)

                maxk3 = max((1, 3))

                _, predicted2 = outputs.topk(maxk3, 1, True, True)

                correct += torch.eq(predicted, label_resize).cpu().sum().float().item()

                correct3 += torch.eq(predicted2, label_resize).cpu().sum().float().item()

                _, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                valid_acc += acc.item() * inputs.size(0)

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size

        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size

        avg_valid_acc3 = correct3 / valid_data_size
        avg_valid_acc5 = correct / valid_data_size
        record.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc, avg_valid_acc3, avg_valid_acc5])

        if avg_valid_acc > best_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1
            torch.save({"model": "resnet50", "state_dict": model.state_dict()}, trained_param_path)
            ctx.latest_round_result = trained_param_path

        epoch_end = time.time()

        print(
            "Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Accuracy3: {:.4f}%, Accuracy5: {:.4f}%,Time: {:.4f}s".format(
                epoch + 1,
                avg_train_loss,
                avg_train_acc * 100,
                avg_valid_loss,
                avg_valid_acc * 100,
                avg_valid_acc3 * 100,
                avg_valid_acc5 * 100,
                epoch_end - epoch_start,
            )
        )
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))

    return model, record


def main():
    _, cfg = parse_option()

    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),  # 随机裁剪到256*256
            RandomResizedCropAndInterpolation(
                size=(224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=PIL.Image.BICUBIC
            ),
            transforms.RandomRotation(degrees=15),  # 随机旋转
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.CenterCrop(size=224),  # 中心裁剪到224*224
            transforms.ToTensor(),  # 转化成张量
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 归一化
        ]
    )
    test_valid_transforms = transforms.Compose(
        [
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_directory = cfg.DATA.TRAIN_DATASET_DIR
    valid_directory = cfg.DATA.VALID_DATASET_DIR
    batch_size = cfg.DATA.BATCH_SIZE

    train_datasets = datasets.ImageFolder(train_directory, transform=train_transforms)
    print(train_datasets.class_to_idx)
    train_data_size = len(train_datasets)
    train_data = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)

    valid_datasets = datasets.ImageFolder(valid_directory, transform=test_valid_transforms)
    print(valid_datasets.class_to_idx)
    valid_data_size = len(valid_datasets)
    valid_data = torch.utils.data.DataLoader(valid_datasets, batch_size=batch_size, shuffle=True)

    print(train_data_size, valid_data_size)

    # ctx.current_round = 1
    # ctx.latest_round_result = "output/ckpt_0.pth"

    import os

    if not os.path.exists("output"):
        os.makedirs("output")

    for round in range(2):
    # if True:
        ctx.current_round = round
        model = BiRSwinT(config=cfg)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # if gpu is available

        loss_func = nn.CrossEntropyLoss()
        loss_func.to(device=device)

        optimizer = build_optimizer(config=cfg, model=model)
        lr_sche = build_scheduler(config=cfg, optimizer=optimizer, train_data_size=train_data_size)

        _, record = train_and_valid(
            config=cfg,
            model=model,
            optimizer=optimizer,
            loss_function=loss_func,
            lr_sche=lr_sche,
            train_data=train_data,
            valid_data=valid_data,
            train_data_size=train_data_size,
            valid_data_size=valid_data_size,
        )

        TRAINED_MODEL = "output/checkpoint"
        i = 0
        trained_path = Path(TRAINED_MODEL + str(i) + ".pth")
        while trained_path.is_file():
            i += 1
            trained_path = Path(TRAINED_MODEL + str(i) + ".pth")
        torch.save(record, trained_path)


if __name__ == "__main__":
    main()
