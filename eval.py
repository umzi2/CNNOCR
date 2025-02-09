import argparse
import numpy as np
import torch
from utils.ctc_decode import CTCLabelConverter
from utils.parse_model import parse_model
from pepeline import read, ImgColor, ImgFormat
import os


def image2tensor(value: np.ndarray, out_type: torch.dtype = torch.float32):
    def _to_tensor(img: np.ndarray) -> torch.Tensor:
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0

        if len(img.shape) == 2:
            tensor = torch.from_numpy(img[None, ...])
        else:
            tensor = torch.from_numpy(img.transpose(2, 0, 1))

        if tensor.dtype != out_type:
            tensor = tensor.to(out_type)

        return tensor.unsqueeze(0)

    return _to_tensor(value)


# Парсер аргументов командной строки
parser = argparse.ArgumentParser(description="Проверка модели.")
parser.add_argument(
    "-f", "--folder", type=str, required=True, help="Папка с изображениями"
)
parser.add_argument("-m", "--model", type=str, required=True, help="Файл модели")
args = parser.parse_args()

# Загрузка состояния модели и создание модели
state_dict = torch.load(args.model, "cpu", weights_only=True)
model, rgb = parse_model(state_dict)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.eval().to(device)

# Получаем список файлов изображений из указанной папки
image_list = os.listdir(args.folder)
# Создаем конвертер для декодирования выходов модели в текст (цифры от 0 до 9)
convert = CTCLabelConverter([str(c) for c in "0123456789"])

# Обрабатываем каждое изображение в папке
for img_name in image_list:
    img_path = os.path.join(args.folder, img_name)
    img = image2tensor(
        read(img_path, ImgColor.RGB if rgb else ImgColor.GRAY, ImgFormat.F32)
    ).to(device)
    output = model(img)
    preds_size = torch.IntTensor([output.size(1)])
    _, preds_index = output.max(2)
    preds_index = preds_index.view(-1)
    pred_str = convert.decode_greedy(preds_index.cpu().data, preds_size.cpu().data)
    print(f"Имя изображения - {img_name} | Результат - {pred_str[0]}")
