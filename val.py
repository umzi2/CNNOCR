import numpy as np
import torch
from torch import autocast, nn
from torch.utils.data import DataLoader
from typing import List, Dict, Any

from utils.ctc_decode import CTCLabelConverter
from utils.dataloader import OCRDataset
from utils.metrics import TextMetrics
from utils.parse_model import parse_model


class Validate:
    """
    Класс для проверки (валидации) модели распознавания текста.
    """

    def __init__(self, model: str, folder: str, csv_file: str) -> None:
        """
        Инициализация валидации модели.

        Аргументы:
            model (str): Путь к файлу модели.
            folder (str): Папка с изображениями.
            csv_file (str): Путь к CSV файлу с данными.
        """
        # Определение устройства: используем CUDA, если доступно, иначе CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Загрузка весов модели из файла (только веса)
        state_dict = torch.load(model, weights_only=True)

        # Разбор состояния модели и получение дополнительного параметра (например, RGB)
        self.model, rgb = parse_model(state_dict)
        self.model.eval().to(
            self.device
        )  # Перевод модели в режим оценки и перемещение на устройство

        # Определение функции потерь CTC с настройками
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)

        # Создание загрузчика данных для валидации
        self.loader = DataLoader(OCRDataset(csv_file, folder, True, rgb))

        # Инициализация метрики для оценки качества распознавания текста (CER, WER)
        self.val_metric = TextMetrics()

        # Инициализация конвертера меток для CTC (используются символы цифр от 0 до 9)
        self.convert = CTCLabelConverter([str(char) for char in "0123456789"])

    def __call__(self) -> List[Dict[str, Any]]:
        """
        Выполнение проверки модели на валидационном наборе.

        Возвращает:
            List[Dict[str, Any]]: Список словарей с результатами для каждого примера, содержащий:
                - "name": имя файла/образца,
                - "ctc_loss": значение CTC потерь,
                - "wer": ошибка WER,
                - "cer": ошибка CER,
                - "label": исходная метка,
                - "output": предсказанный текст.
        """
        total_loss: List[Dict[str, Any]] = []  # Список для накопления результатов

        # Отключение вычисления градиентов для повышения производительности
        with torch.no_grad():
            # Перебор батчей из загрузчика данных
            for images, labels, name in self.loader:
                # Перемещение изображений на выбранное устройство (GPU/CPU)
                images = images.to(self.device)

                # Использование автоматического режима с пониженной точностью для ускорения вычислений на CUDA
                with autocast("cuda"):
                    # Получение предсказаний модели и применение функции log_softmax по третьему измерению
                    outputs = self.model(images).log_softmax(2)

                    # Кодирование меток в формат, подходящий для вычисления CTC потерь
                    text, length = self.convert.encode(labels)

                    # Формирование тензора размеров предсказаний для каждого образца в батче
                    preds_size = torch.tensor(
                        [outputs.size(1)] * len(images), device=self.device
                    )

                    # Вычисление CTC потерь
                    loss = self.criterion(
                        outputs.permute(
                            1, 0, 2
                        ),  # Транспонирование размерностей для соответствия требованиям функции
                        text.to(self.device),
                        preds_size,
                        length.to(self.device),
                    )

                # Приведение потерь к числовому значению и масштабирование на размер батча
                loss_value = loss.item() * images.size(0)
                if loss_value == 0:
                    loss_value = 100  # Если потеря равна 0, устанавливаем её в 100

                # Подготовка тензора с размером предсказания для декодирования
                preds_size = torch.IntTensor([outputs.size(1)])
                # Получение индексов предсказанных символов (максимум по последнему измерению)
                _, preds_index = outputs.max(2)
                preds_index = preds_index.view(-1)

                # Декодирование индексов в строку с помощью жадного алгоритма
                pred_str = self.convert.decode_greedy(
                    preds_index.cpu().data, preds_size.cpu().data
                )

                # Оценка метрик распознавания: WER и CER (ошибки на уровне слов и символов)
                metrics = self.val_metric.evaluate(labels, pred_str)

                # Добавление результатов для текущего образца в общий список
                total_loss.append(
                    {
                        "name": name[0],  # Имя файла/образца
                        "ctc_loss": loss_value,  # Значение CTC потерь
                        "wer": metrics[0],  # Ошибка WER
                        "cer": metrics[1],  # Ошибка CER
                        "label": labels[0],  # Исходная метка
                        "output": pred_str[0],  # Предсказанный текст
                    }
                )
        return total_loss


if __name__ == "__main__":
    import argparse
    import csv

    # Определение аргументов командной строки
    parser = argparse.ArgumentParser(description="Проверка модели.")
    parser.add_argument(
        "-f", "--folder", type=str, required=True, help="Папка с изображениями"
    )
    parser.add_argument(
        "-c", "--csv", type=str, required=True, help="CSV файл с данными"
    )
    parser.add_argument("-m", "--model", type=str, required=True, help="Файл модели")
    args = parser.parse_args()

    # Создание экземпляра класса для валидации модели
    valid = Validate(args.model, args.folder, args.csv)

    # Выполнение проверки модели и получение списка результатов
    result_list = valid()

    # Подготовка данных для записи в CSV файл
    data: List[List[Any]] = [
        ["Имя", "CER", "WER", "CTC потеря", "Реальный", "Предсказанный"]
    ]
    ctc_loss: List[float] = []
    cer: List[float] = []
    wer: List[float] = []

    # Обработка каждого результата валидации
    for dict_val in result_list:
        ctc_loss.append(dict_val["ctc_loss"])
        cer.append(dict_val["cer"])
        wer.append(dict_val["wer"])
        data.append(
            [
                dict_val["name"],
                dict_val["cer"],
                dict_val["wer"],
                dict_val["ctc_loss"],
                dict_val["label"],
                dict_val["output"],
            ]
        )

    # Вывод средних значений метрик
    print(
        "Средняя CTC",
        np.mean(ctc_loss),
        "Средняя CER",
        np.mean(cer),
        "Средняя WER",
        np.mean(wer),
    )

    # Запись результатов в CSV файл с кодировкой UTF-8
    with open("output.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(data)
