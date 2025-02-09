import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from archs.cnnocr import CNNOCR
from utils.adan_sf import adan_sf
from utils.augment import (
    CutOut,
    RandomSqueezeWithPadding,
    LensBlur,
    Jpeg,
    RandomUnsharpMask,
)
from utils.ctc_decode import CTCLabelConverter
from utils.dataloader import OCRDataset
from torch.amp import GradScaler, autocast

from utils.metrics import TextMetrics
import logging


class Trainer:
    def __init__(self, model, opt):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.iter = 0
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        self.optimizer = adan_sf(self.model.parameters(), **opt["optim"])
        self.scaler = GradScaler(enabled=(device.type == "cuda"))
        self.convert = CTCLabelConverter([str(charter) for charter in opt["character"]])
        self.val_metric = TextMetrics()
        self.best_val_loss = float("inf")
        seed = opt["manual_seed"]
        self.logger = logging.getLogger("basic_logger")
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(opt["logger_file"], encoding="utf-8")
        file_handler.setFormatter(formatter)

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        if seed > -1:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def train_step(self, images, labels):
        self.model.train()
        images = images.to(self.device)

        self.optimizer.zero_grad()

        with autocast("cuda"):
            outputs = self.model(images).log_softmax(2)
            text, length = self.convert.encode(labels)
            preds_size = torch.tensor(
                [outputs.size(1)] * len(images), device=self.device
            )
            loss = self.criterion(
                outputs.permute(1, 0, 2),
                text.to(self.device),
                preds_size,
                length.to(self.device),
            )

        self.scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.iter += 1
        return loss.item()

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        wer = 0
        cer = 0
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(self.device)

                with autocast("cuda"):
                    outputs = self.model(images).log_softmax(2)
                    text, length = self.convert.encode(labels)
                    preds_size = torch.tensor(
                        [outputs.size(1)] * len(images), device=self.device
                    )
                    loss = self.criterion(
                        outputs.permute(1, 0, 2),
                        text.to(self.device),
                        preds_size,
                        length.to(self.device),
                    )

                total_loss += loss.item() * images.size(0)
                preds_size = torch.IntTensor([outputs.size(1)])
                _, preds_index = outputs.max(2)
                preds_index = preds_index.view(-1)
                pred_str = self.convert.decode_greedy(
                    preds_index.cpu().data, preds_size.cpu().data
                )
                metrics = self.val_metric.evaluate(labels, pred_str)
                wer += metrics[0]
                cer += metrics[1]

        avg_loss = total_loss / len(val_loader.dataset)
        wer = wer / len(val_loader.dataset)
        cer = cer / len(val_loader.dataset)
        return avg_loss, wer, cer

    def save_checkpoint(self):
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"model_iter_{self.iter}.pth"
        )
        torch.save(self.model.state_dict(), checkpoint_path)
        self.logger.debug(f"Saved checkpoint at iteration {self.iter}")

    def fit(self, train_loader, val_loader, max_iters, val_interval=1000):
        train_iter = iter(train_loader)

        while self.iter < max_iters:
            # Get next batch
            try:
                images, labels, _ = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                images, labels, _ = next(train_iter)

            # Train step
            train_loss = self.train_step(images, labels)

            # Logging
            if self.iter % 100 == 0:
                self.logger.debug(
                    f"Iter {self.iter}/{max_iters} | Train Loss: {train_loss:.4f}"
                )

            # Validation
            if self.iter % val_interval == 0 and self.iter > 0:
                val_loss, wer, cer = self.validate(val_loader)
                self.logger.debug(
                    f"Validation @ {self.iter} iters | Loss: {val_loss:.4f} | WER: {wer:.2f} | CER: {cer:.2f}"
                )

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint()

        # Final validation and save
        val_loss, wer, cer = self.validate(val_loader)
        self.logger.debug(
            f"Final Validation | Loss: {val_loss:.4f} | WER: {wer:.4f} | CER: {cer:.4f}"
        )
        self.save_checkpoint()


def main(opt):
    train_load = opt["train_load"]
    val_load = opt["val_load"]

    train_data = OCRDataset(
        train_load["csv_file"],
        train_load["img_dir"],
        False,
        opt["rgb"],
        train_load["tile_size"],
        transform=transforms.Compose(
            [
                transforms.RandomApply([RandomSqueezeWithPadding()]),
                transforms.RandomApply(
                    [
                        transforms.RandomRotation(
                            45, interpolation=transforms.InterpolationMode.BILINEAR
                        )
                    ]
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.25, 0.25, 0.25, 0.25)]
                ),
                CutOut(),
                transforms.RandomApply([LensBlur(), RandomUnsharpMask(), Jpeg()]),
            ]
        )
        if train_load["transforms"]
        else nn.Identity(),
        transform_warmup=train_load["transforms_warmup"],
    )
    # Создаём модуль данных и подготавливаем датасеты
    train_loader = DataLoader(
        train_data,
        batch_size=train_load["batch_size"],
        shuffle=True,
        persistent_workers=True,
        num_workers=train_load["num_workers"],  # Многозадачность
        pin_memory=True,
    )
    val_loader = DataLoader(
        OCRDataset(
            val_load["csv_file"],
            val_load["img_dir"],
            True,
            opt["rgb"],
        ),
        shuffle=False,
    )

    model = CNNOCR(**opt["model"])
    pretrain = opt.get("pretrain")
    if pretrain:
        state = torch.load(pretrain, weights_only=True)
        model.load_state_dict(state)

    trainer = Trainer(model, opt)
    trainer.fit(train_loader, val_loader, 300000)


if __name__ == "__main__":
    import argparse
    import toml

    parser = argparse.ArgumentParser(
        description="Train the model with a given config file."
    )
    parser.add_argument(
        "-f", "--file", type=str, required=True, help="Path to the config file"
    )
    args = parser.parse_args()
    with open(args.file) as file:
        opt = toml.load(file)

    main(opt)
