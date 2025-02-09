import cv2
import pandas as pd
import pyvips
from torch.utils.data import Dataset
import torch
import numpy as np
import os


class Resize:
    """
    Класс для изменения размера изображения до заданного размера плитки (tile_size).
    Использует библиотеку pyvips для масштабирования изображения.
    """

    def __init__(self, tile_size=(128, 128)):
        """
        Инициализация с заданным размером плитки.

        :param tile_size: Желаемый размер плитки (по умолчанию (128,128)).
        """
        self.tile_size = tile_size

    def __call__(self, image):
        """
        Изменяет размер изображения.

        :param image: Изображение в формате pyvips.Image.
        :return: Изменённое изображение.
        """
        # Получаем исходные размеры изображения
        H, W = image.height, image.width
        # Вычисляем коэффициенты масштабирования для высоты и ширины
        scale_y = self.tile_size[0] / H
        scale_x = self.tile_size[1] / W

        # Масштабируем изображение с использованием pyvips
        resized_image = image.resize(scale_x, vscale=scale_y)
        return resized_image


def image2tensor(
    value: list[np.ndarray] | np.ndarray,
    out_type: torch.dtype = torch.float32,
    rgb: bool = False,
):
    """
    Преобразует изображение (или список изображений) из формата NumPy в тензор PyTorch.

    :param value: Одно изображение или список изображений в формате np.ndarray.
    :param out_type: Желаемый тип тензора (по умолчанию torch.float32).
    :param rgb: Если True, сохраняется цветовое изображение, иначе преобразуется в оттенки серого.
    :return: Тензор или список тензоров.
    """

    def _to_tensor(img: np.ndarray, rgb: bool = False) -> torch.Tensor:
        # Если не требуется сохранять цвет, преобразуем изображение в оттенки серого
        if not rgb:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Нормализуем изображение, если оно представлено в формате uint8
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0

        # Если изображение двумерное (оттенки серого), добавляем измерение канала
        if len(img.shape) == 2:
            tensor = torch.from_numpy(img[None, ...])
        else:
            # Для цветного изображения меняем порядок осей на (каналы, высота, ширина)
            tensor = torch.from_numpy(img.transpose(2, 0, 1))

        # Приводим тензор к нужному типу, если требуется
        if tensor.dtype != out_type:
            tensor = tensor.to(out_type)

        return tensor

    # Если value является списком, преобразуем каждый элемент отдельно
    if isinstance(value, list):
        return [_to_tensor(i, rgb) for i in value]
    else:
        return _to_tensor(value, rgb)


def img_read(path: str) -> pyvips.Image:
    """
    Считывает изображение с диска с помощью библиотеки pyvips.

    :param path: Путь к файлу изображения.
    :return: Изображение в формате pyvips.Image.
    """
    img = pyvips.Image.new_from_file(path, access="sequential", fail=True)
    assert isinstance(img, pyvips.Image)
    return img


def img2rgb(image: np.ndarray) -> np.ndarray:
    """
    Преобразует изображение в формате NumPy в RGB.

    Ожидаемые формы изображения:
        - (H, W)
        - (H, W, 1)
        - (H, W, 3)
        - (H, W, 4+)

    :param image: Исходное изображение.
    :return: Изображение в формате RGB с формой (H, W, 3).
    """
    # Если изображение имеет один канал, убираем лишнее измерение
    if image.ndim == 3 and image.shape[2] == 1:
        image = image.squeeze(-1)

    # Если изображение уже в формате RGB, возвращаем его без изменений
    if image.ndim == 3 and image.shape[2] == 3:
        return image

    # Если изображение имеет больше трёх каналов (например, RGBA), оставляем только первые три
    elif image.ndim == 3 and image.shape[2] > 3:
        return image[:, :, :3]

    # Если изображение в оттенках серого (двумерное), создаём три одинаковых канала
    elif image.ndim == 2:
        return np.stack((image,) * 3, axis=-1)

    else:
        raise ValueError(
            "Неподдерживаемая форма изображения: ожидается (H, W), (H, W, 1), (H, W, 3) или (H, W, 4+)"
        )


class OCRDataset(Dataset):
    """
    Датасет для задачи OCR.

    CSV-файл должен содержать как минимум два столбца:
        - 'img_name': относительный путь к изображению.
        - 'text': строка с меткой (например, "27").
    """

    def __init__(
        self,
        csv_file,
        image_dir,
        val,
        rgb=True,
        tile_size=(128, 128),
        transform=None,
        transform_warmup=0,
    ):
        """
        Инициализация датасета.

        :param csv_file: Путь к CSV-файлу с данными.
        :param image_dir: Директория, содержащая изображения.
        :param val: Флаг, указывающий, используется ли датасет для валидации.
        :param rgb: Если True, изображения сохраняются в RGB, иначе преобразуются в оттенки серого.
        :param transform: Преобразования для изображений (например, аугментация).
        :param transform_warmup: Количество итераций, в течение которых преобразования не применяются.
        """
        self.data = pd.read_csv(csv_file)
        self.iter = 0  # Счётчик для контроля этапа warmup
        self.image_dir = image_dir
        self.transform = transform
        self.val = val
        self.warmup = transform_warmup
        self.rgb = rgb
        # Если датасет используется для обучения, задаём изменение размера изображений
        if not val:
            self.norm_size = Resize(tile_size)

    def __len__(self):
        """
        Возвращает общее количество образцов в датасете.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Возвращает элемент датасета по индексу.

        :param idx: Индекс элемента.
        :return: Кортеж (image, label, img_name), где:
            - image: изображение в виде тензора,
            - label: строковая метка,
            - img_name: имя файла изображения.
        """
        # Извлекаем строку из CSV по индексу
        row = self.data.iloc[idx]
        # Формируем полный путь к изображению
        img_path = os.path.join(self.image_dir, row["img_name"])
        # Считываем изображение с диска
        image = img_read(str(img_path))

        if not self.val:
            # Изменяем размер изображения до стандартного (tile_size)
            image = self.norm_size(image)
            # Преобразуем изображение в тензор: сначала переводим в формат RGB, затем в PyTorch тензор
            image = image2tensor(img2rgb(image.numpy()), rgb=self.rgb)
            # Если warmup-период завершён, применяем дополнительные преобразования (например, аугментацию)
            if self.iter >= self.warmup:
                image = self.transform(image)
            self.iter += 1
        else:
            # Для валидационного датасета: только преобразуем изображение в тензор
            image = image2tensor(img2rgb(image.numpy()), rgb=self.rgb)

        # Получаем строковую метку и преобразуем её в нужный формат
        label_str = str(row["text"])
        label = []
        # Преобразуем строку метки, прекращая при встрече точки (если требуется)
        for ch in label_str:
            if ch == ".":
                break
            label.append(str(ch))
        label = "".join(label)

        return image, label, row["img_name"]
