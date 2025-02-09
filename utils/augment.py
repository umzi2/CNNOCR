import torch
from torch import nn
import torch.nn.functional as F
import random
import torchvision.transforms.v2
import torchvision.transforms.functional as TF
from pepeline import TypeNoise, noise_generate


class RandomUnsharpMask:
    """
    Применяет случайное усиление резкости с использованием размытого изображения.
    При данном подходе создаётся эффект «нечеткой резкости», который помогает улучшить детали изображения.
    """

    def __init__(self, blur_kernel=(3, 10), alpha_range=(0.5, 3), probability=0.5):
        """
        :param blur_kernel: Кортеж с минимальным и максимальным значениями размера ядра для размытия.
        :param alpha_range: Диапазон коэффициента усиления резкости.
        :param probability: Вероятность применения операции.
        """
        self.blur_kernel = blur_kernel
        self.alpha_range = alpha_range
        self.p = probability

    def __call__(self, img):
        """
        Применяет операцию усиления резкости к изображению.

        :param img: Входное изображение в виде тензора.
        :return: Изображение с усиленной резкостью.
        """
        # Если случайное число больше заданной вероятности, возвращаем исходное изображение без изменений
        if random.random() > self.p:
            return img

        # Выбираем случайный размер ядра для размытия (только нечётные значения)
        kernel_size = random.choice(
            range(self.blur_kernel[0], self.blur_kernel[1] + 1, 2)
        )
        # Выбираем случайное значение альфа из заданного диапазона
        alpha = random.uniform(*self.alpha_range)

        # Применяем гауссово размытие к изображению
        blurred = TF.gaussian_blur(img, kernel_size)
        # Усиливаем резкость: исходное изображение + alpha * (исходное - размытое)
        sharpened = torch.clamp(img + alpha * (img - blurred), 0, 1)
        return sharpened


class CutOut(nn.Module):
    """
    Применяет операцию CutOut, которая случайным образом заменяет части изображения шумом или фиксированным цветом.
    """

    def __init__(self, probability=0.5):
        """
        :param probability: Вероятность применения CutOut.
        """
        super().__init__()
        self.p = probability

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Применяет операцию CutOut к входному тензору.

        :param tensor: Изображение в виде тензора с формой (C, H, W).
        :return: Изображение с применённым эффектом CutOut.
        """
        c, h, w = tensor.shape

        # Если случайное число больше или равно вероятности, возвращаем исходный тензор
        if torch.rand(1) >= self.p:
            return tensor

        # Генерируем маску с использованием функции noise_generate (тип шума SIMPLEX)
        mask = noise_generate((c, h, w), TypeNoise.SIMPLEX, 1, 0.02, 0.1) > 0.3

        # С вероятностью 50% заменяем пиксели в области маски случайными значениями
        if torch.rand(1) >= 0.5:
            noise = torch.rand((c, h, w), device=tensor.device, dtype=tensor.dtype)
            tensor[mask] = noise[mask]
        else:
            # Иначе заменяем пиксели в области маски на случайный цвет для каждого канала
            random_color = torch.rand(c, 1, 1, device=tensor.device, dtype=tensor.dtype)
            tensor = torch.where(
                torch.tensor(mask, device=tensor.device, dtype=torch.bool),
                random_color,
                tensor,
            )
        return tensor


class RandomSqueezeWithPadding(nn.Module):
    """
    Применяет случайное сжатие изображения по вертикали и горизонтали с последующим дополнением (padding),
    чтобы вернуть изображение исходного размера.
    """

    def __init__(
        self,
        vertical_scale_range=(0.5, 1.0),
        horizontal_scale_range=(0.5, 1.0),
        pad_value=0,
    ):
        """
        :param vertical_scale_range: Диапазон масштабирования по вертикали.
        :param horizontal_scale_range: Диапазон масштабирования по горизонтали.
        :param pad_value: Значение для дополнения (padding) (не используется в данном коде, используется репликация).
        """
        super().__init__()
        self.vertical_scale_range = vertical_scale_range
        self.horizontal_scale_range = horizontal_scale_range
        self.pad_value = pad_value

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Применяет операцию случайного сжатия с дополнением.

        :param image: Изображение в виде тензора с формой (C, H, W).
        :return: Изменённое изображение с сохранением исходных размеров.
        """
        C, H, W = image.shape

        new_h = H
        new_w = W

        # Случайно выбираем коэффициент масштабирования для высоты
        if self.vertical_scale_range is not None:
            scale_h = random.uniform(*self.vertical_scale_range)
            new_h = int(H * scale_h)

        # Случайно выбираем коэффициент масштабирования для ширины
        if self.horizontal_scale_range is not None:
            scale_w = random.uniform(*self.horizontal_scale_range)
            new_w = int(W * scale_w)

        # Добавляем измерение для батча и изменяем размер изображения
        image_unsqueezed = image.unsqueeze(0)  # (1, C, H, W)
        resized = F.interpolate(
            image_unsqueezed, size=(new_h, new_w), mode="bilinear", align_corners=False
        ).squeeze(0)  # (C, new_h, new_w)

        # Вычисляем случайное распределение отступов (padding) для восстановления исходного размера
        pad_top = int((H - new_h) * torch.rand(1))
        pad_bottom = H - new_h - pad_top
        pad_left = int((W - new_w) * torch.rand(1))
        pad_right = W - new_w - pad_left

        # Применяем padding с использованием режима "replicate" (повторение краёв)
        padded = F.pad(
            resized,
            (pad_left, pad_right, pad_top, pad_bottom),
            "replicate",
        )

        return padded


class Jpeg:
    """
    Симулирует сжатие изображения в формате JPEG для создания артефактов сжатия.
    """

    def __init__(self):
        # Создаем объект преобразования JPEG с диапазоном качества от 40 до 100
        self.jpeg = torchvision.transforms.v2.JPEG([40, 100])

    def __call__(self, img):
        """
        Применяет JPEG-сжатие к изображению.

        :param img: Изображение в виде тензора с плавающей точкой (значения в диапазоне [0, 1]).
        :return: Изображение после JPEG-сжатия, приведённое обратно к диапазону [0, 1].
        """
        # Приводим изображение к диапазону [0, 255] и типу uint8
        img = (img * 255.0).to(torch.uint8)
        # Применяем JPEG-сжатие и возвращаем изображение с нормализацией обратно в диапазон [0, 1]
        return self.jpeg(img).to(torch.float32) / 255.0


class LensBlur(nn.Module):
    """
    Применяет эффект линзового размытия к изображению.
    Эффект создаётся с использованием дискового (кругового) ядра размытия.
    """

    def __init__(self, kernel_range=(0, 5)):
        """
        :param kernel_range: Диапазон значений для размера ядра размытия.
        """
        super().__init__()
        self.kernel_range = kernel_range

    @staticmethod
    def __generate_circle(x: int, y: int, radius: int, center: int) -> bool:
        """
        Определяет, находится ли точка (x, y) внутри круга с заданным центром и радиусом.

        :param x: Координата x.
        :param y: Координата y.
        :param radius: Радиус круга.
        :param center: Координата центра круга (предполагается квадратное ядро).
        :return: True, если точка внутри круга, иначе False.
        """
        return (x - center) ** 2 + (y - center) ** 2 <= radius**2

    def __lens_blur(self, image: torch.Tensor, dimension: float = 10) -> torch.Tensor:
        """
        Применяет эффект линзового размытия к изображению с заданным размером ядра.

        :param image: Входное изображение в виде тензора.
        :param dimension: Размер ядра размытия.
        :return: Размазанное изображение.
        """
        # Если размер ядра равен нулю, возвращаем исходное изображение
        if dimension == 0:
            return image

        # Получаем дисковое (круговое) ядро размытия
        kernel = self.__disk_kernel(dimension)
        kernel = kernel.to(image.dtype).to(image.device).unsqueeze(0).unsqueeze(0)
        # Расширяем ядро для каждого канала изображения
        kernel = kernel.expand(image.shape[0], 1, -1, -1)
        padding = kernel.shape[-1] // 2
        # Дополняем изображение по краям, чтобы сохранить размеры после свёртки
        image_padded = F.pad(
            image.unsqueeze(0), (padding, padding, padding, padding), mode="reflect"
        )
        # Применяем свёртку с группами, равными количеству каналов
        convolved = F.conv2d(image_padded, kernel, groups=image.shape[0]).squeeze(0)
        return convolved

    def __disk_kernel(self, kernel_size: float) -> torch.Tensor:
        """
        Генерирует дисковое (круговое) ядро размытия заданного размера.

        :param kernel_size: Размер ядра размытия (может быть дробным).
        :return: Нормализованное ядро в виде тензора.
        """
        # Определяем размер ядра как ближайшее нечетное число, зависящее от kernel_size
        kernel_dim = int(torch.ceil(torch.tensor(kernel_size)) * 2 + 1)
        # Определяем дробную часть размера ядра
        fraction = kernel_size % 1
        kernel = torch.zeros((kernel_dim, kernel_dim), dtype=torch.float32)
        circle_center_coord = kernel_dim // 2
        # Если дробная часть не равна нулю, уменьшаем радиус на 1
        circle_radius = (
            circle_center_coord - 1 if fraction != 0 else circle_center_coord
        )

        # Заполняем ядро, отмечая пиксели, находящиеся внутри круга
        for i in range(kernel_dim):
            for j in range(kernel_dim):
                kernel[i, j] = self.__generate_circle(
                    i, j, circle_radius, circle_center_coord
                )

        # Если есть дробная часть, смешиваем два ядра для получения плавного перехода
        if fraction != 0:
            kernel2 = torch.zeros((kernel_dim, kernel_dim), dtype=torch.float32)
            circle_radius = circle_center_coord
            for i in range(kernel_dim):
                for j in range(kernel_dim):
                    kernel2[i, j] = self.__generate_circle(
                        i, j, circle_radius, circle_center_coord
                    )
            kernel = torch.clamp(kernel + kernel2 * fraction, 0, 1)

        # Нормализуем ядро, чтобы сумма элементов равнялась 1
        kernel /= kernel.sum()
        return kernel

    def forward(self, x):
        """
        Применяет эффект линзового размытия к входному изображению.

        :param x: Изображение в виде тензора.
        :return: Изображение с эффектом линзового размытия.
        """
        # Случайным образом выбираем размер ядра из заданного диапазона
        kernel_size = random.uniform(*self.kernel_range)
        return self.__lens_blur(x, kernel_size)
