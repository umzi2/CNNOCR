import torch
import numpy as np


def consecutive(data, mode="first", stepsize=1):
    """
    Функция группирует последовательные элементы, различающиеся ровно на stepsize.

    :param data: Массив данных (numpy-массив).
    :param mode: Режим выбора элемента из группы: "first" - первый элемент группы, "last" - последний.
    :param stepsize: Шаг между последовательными элементами (по умолчанию 1).
    :return: Список выбранных элементов (либо первые, либо последние из каждой группы).
    """
    # Разбиваем массив на группы, где разница между соседними элементами не равна stepsize
    group = np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)
    group = [item for item in group if len(item) > 0]

    if mode == "first":
        result = [item[0] for item in group]
    elif mode == "last":
        result = [item[-1] for item in group]
    return result


def word_segmentation(
    mat, separator_idx={"th": [1, 2], "en": [3, 4]}, separator_idx_list=[1, 2, 3, 4]
):
    """
    Функция сегментирует последовательность (например, вывод CTC) на слова с учетом разделителей.

    :param mat: Массив индексов (например, argmax из матрицы вероятностей).
    :param separator_idx: Словарь, содержащий индексы-разделители для разных языков.
    :param separator_idx_list: Список всех индексов-разделителей.
    :return: Список сегментов в формате [язык, [начало, конец]].
    """
    result = []
    sep_list = []
    start_idx = 0
    sep_lang = ""

    # Для каждого разделителя ищем группы последовательных позиций в матрице, где значение равно sep_idx
    for sep_idx in separator_idx_list:
        # Если разделитель чётный, выбираем первый элемент группы, иначе последний
        mode = "first" if sep_idx % 2 == 0 else "last"
        a = consecutive(np.argwhere(mat == sep_idx).flatten(), mode)
        new_sep = [[item, sep_idx] for item in a]
        sep_list += new_sep

    # Сортируем список разделителей по индексу в матрице
    sep_list = sorted(sep_list, key=lambda x: x[0])

    # Обрабатываем найденные разделители и формируем сегменты
    for sep in sep_list:
        for lang in separator_idx.keys():
            if sep[1] == separator_idx[lang][0]:  # начало сегмента для языка
                sep_lang = lang
                sep_start_idx = sep[0]
            elif sep[1] == separator_idx[lang][1]:  # конец сегмента для языка
                if (
                    sep_lang == lang
                ):  # проверяем, что начало и конец относятся к одному языку
                    new_sep_pair = [lang, [sep_start_idx + 1, sep[0] - 1]]
                    if sep_start_idx > start_idx:
                        # Добавляем сегмент без указания языка
                        result.append(["", [start_idx, sep_start_idx - 1]])
                    start_idx = sep[0] + 1
                    result.append(new_sep_pair)
                sep_lang = ""  # сбрасываем язык

    # Если осталась часть последовательности после последнего разделителя, добавляем её
    if start_idx <= len(mat) - 1:
        result.append(["", [start_idx, len(mat) - 1]])
    return result


# Код основан на https://github.com/githubharald/CTCDecoder/blob/master/src/BeamSearch.py


class BeamEntry:
    """
    Информация об одной конкретной "лучевой" записи (beam) на определённом временном шаге.
    """

    def __init__(self):
        self.prTotal = 0  # Совокупная вероятность (сумма вероятностей для пустого и не пустого символа)
        self.prNonBlank = 0  # Вероятность окончания на не пустом символе
        self.prBlank = 0  # Вероятность окончания на пустом символе
        self.prText = 1  # Оценка (score) языковой модели (Language Model, LM)
        self.lmApplied = (
            False  # Флаг, указывающий, что LM уже был применён для этого beam
        )
        self.labeling = ()  # Метка (последовательность символов) для beam
        self.simplified = (
            True  # Флаг упрощения метки (для объединения повторяющихся символов)
        )


class BeamState:
    """
    Состояние всех beam'ов на определённом временном шаге.
    """

    def __init__(self):
        self.entries = {}  # Словарь, где ключ – метка (labeling), а значение – объект BeamEntry

    def norm(self):
        """
        Нормализует оценку LM, деля её на длину метки.
        """
        for k, _ in self.entries.items():
            labelingLen = len(self.entries[k].labeling)
            self.entries[k].prText = self.entries[k].prText ** (
                1.0 / (labelingLen if labelingLen else 1.0)
            )

    def sort(self):
        """
        Возвращает список меток beam'ов, отсортированных по убыванию их суммарной вероятности (prTotal * prText).
        """
        beams = [v for (_, v) in self.entries.items()]
        sortedBeams = sorted(beams, reverse=True, key=lambda x: x.prTotal * x.prText)
        return [x.labeling for x in sortedBeams]

    def wordsearch(self, classes, ignore_idx, maxCandidate, dict_list):
        """
        Поиск лучшего кандидата, используя словарь.

        :param classes: Список символов.
        :param ignore_idx: Список индексов, которые нужно игнорировать.
        :param maxCandidate: Максимальное количество кандидатов для проверки.
        :param dict_list: Список слов из словаря.
        :return: Лучшая найденная текстовая строка.
        """
        beams = [v for (_, v) in self.entries.items()]
        sortedBeams = sorted(beams, reverse=True, key=lambda x: x.prTotal * x.prText)
        if len(sortedBeams) > maxCandidate:
            sortedBeams = sortedBeams[:maxCandidate]

        for beam_index, candidate in enumerate(sortedBeams):
            idx_list = candidate.labeling
            text = ""
            # Собираем текст из метки, игнорируя повторения и нежелательные индексы
            for char_index, label in enumerate(idx_list):
                if label not in ignore_idx and not (
                    char_index > 0 and idx_list[char_index - 1] == idx_list[char_index]
                ):
                    text += classes[label]

            if beam_index == 0:
                best_text = text
            # Если текст найден в словаре, выбираем его и прерываем поиск
            if text in dict_list:
                best_text = text
                break

        return best_text


def applyLM(parentBeam, childBeam, classes, lm):
    """
    Рассчитывает оценку языковой модели (LM) для дочернего beam, используя вероятность биграммы последних двух символов.

    :param parentBeam: Родительский beam.
    :param childBeam: Дочерний beam, для которого рассчитывается LM оценка.
    :param classes: Список символов.
    :param lm: Объект языковой модели.
    """
    if lm and not childBeam.lmApplied:
        # Первый символ: последний символ родительского beam (или пробел, если пусто)
        c1 = classes[
            parentBeam.labeling[-1] if parentBeam.labeling else classes.index(" ")
        ]
        # Второй символ: последний символ дочернего beam
        c2 = classes[childBeam.labeling[-1]]
        lmFactor = 0.01  # Коэффициент влияния языковой модели
        # Вероятность биграммы для пары символов
        bigramProb = lm.getCharBigram(c1, c2) ** lmFactor
        # Обновляем LM оценку для дочернего beam
        childBeam.prText = parentBeam.prText * bigramProb
        childBeam.lmApplied = True  # Применяем LM только один раз для данного beam


def simplify_label(labeling, blankIdx=0):
    """
    Упрощает последовательность меток, убирая повторяющиеся пустые символы.

    :param labeling: Последовательность меток.
    :param blankIdx: Индекс пустого символа (по умолчанию 0).
    :return: Упрощённая последовательность меток (в виде кортежа).
    """
    labeling = np.array(labeling)

    # Объединяем повторяющиеся пустые символы
    idx = np.where(~((np.roll(labeling, 1) == labeling) & (labeling == blankIdx)))[0]
    labeling = labeling[idx]

    # Убираем пустые символы, находящиеся между различными символами
    idx = np.where(
        ~((np.roll(labeling, 1) != np.roll(labeling, -1)) & (labeling == blankIdx))
    )[0]

    if len(labeling) > 0:
        last_idx = len(labeling) - 1
        if last_idx not in idx:
            idx = np.append(idx, [last_idx])
    labeling = labeling[idx]

    return tuple(labeling)


def fast_simplify_label(labeling, c, blankIdx=0):
    """
    Быстро упрощает последовательность меток при добавлении нового символа.

    :param labeling: Текущая последовательность меток (кортеж).
    :param c: Новый символ (его индекс).
    :param blankIdx: Индекс пустого символа.
    :return: Новая, упрощённая последовательность меток.
    """
    # Если добавляем пустой символ после не пустого
    if labeling and c == blankIdx and labeling[-1] != blankIdx:
        newLabeling = labeling + (c,)

    # Если добавляем не пустой символ после пустого
    elif labeling and c != blankIdx and labeling[-1] == blankIdx:
        # Если пустой символ между одинаковыми символами – ничего не делаем
        if labeling[-2] == c:
            newLabeling = labeling + (c,)
        # Если пустой символ между разными символами – удаляем пустой символ
        else:
            newLabeling = labeling[:-1] + (c,)

    # Если подряд идут пустые символы – оставляем исходную метку
    elif labeling and c == blankIdx and labeling[-1] == blankIdx:
        newLabeling = labeling

    # Если beam пустой и первый символ пустой – оставляем пустоту
    elif not labeling and c == blankIdx:
        newLabeling = labeling

    # Если beam пустой и первый символ не пустой – добавляем символ
    elif not labeling and c != blankIdx:
        newLabeling = labeling + (c,)

    elif labeling and c != blankIdx:
        newLabeling = labeling + (c,)

    # Остальные случаи, требующие упрощения
    else:
        newLabeling = labeling + (c,)
        newLabeling = simplify_label(newLabeling, blankIdx)

    return newLabeling


def addBeam(beamState, labeling):
    """
    Добавляет beam в состояние, если он ещё не существует.

    :param beamState: Текущее состояние beam'ов.
    :param labeling: Метка, которая должна быть добавлена.
    """
    if labeling not in beamState.entries:
        beamState.entries[labeling] = BeamEntry()


def ctcBeamSearch(mat, classes, ignore_idx, lm, beamWidth=25, dict_list=[]):
    """
    Реализует поиск по лучам (beam search) для декодирования последовательности из матрицы вероятностей.

    :param mat: Матрица вероятностей (размер: время x классы).
    :param classes: Список символов.
    :param ignore_idx: Список индексов, которые необходимо игнорировать.
    :param lm: Объект языковой модели (может быть None).
    :param beamWidth: Ширина луча (количество кандидатов).
    :param dict_list: Список слов из словаря для поиска корректного слова (может быть пустым).
    :return: Декодированная строка.
    """
    blankIdx = 0
    maxT, maxC = mat.shape

    # Инициализируем состояние beam'ов
    last = BeamState()
    labeling = ()
    last.entries[labeling] = BeamEntry()
    last.entries[labeling].prBlank = 1
    last.entries[labeling].prTotal = 1

    # Проходим по всем временным шагам
    for t in range(maxT):
        curr = BeamState()
        # Получаем лучшие beam'ы (метки) из предыдущего состояния
        bestLabelings = last.sort()[0:beamWidth]
        # Проходим по каждому beam
        for labeling in bestLabelings:
            # Вероятность для путей, заканчивающихся не пустым символом
            prNonBlank = 0
            if labeling:
                # Если beam не пустой, учитываем вероятность повторения последнего символа
                prNonBlank = last.entries[labeling].prNonBlank * mat[t, labeling[-1]]

            # Вероятность для путей, заканчивающихся пустым символом
            prBlank = last.entries[labeling].prTotal * mat[t, blankIdx]

            prev_labeling = labeling
            # Если метка не упрощена, упрощаем её
            if not last.entries[labeling].simplified:
                labeling = simplify_label(labeling, blankIdx)

            # Добавляем beam в текущее состояние
            addBeam(curr, labeling)
            # Заполняем данные для beam
            curr.entries[labeling].labeling = labeling
            curr.entries[labeling].prNonBlank += prNonBlank
            curr.entries[labeling].prBlank += prBlank
            curr.entries[labeling].prTotal += prBlank + prNonBlank
            curr.entries[labeling].prText = last.entries[prev_labeling].prText

            # Расширяем текущий beam, перебирая все вероятные символы
            # Используем все символы с вероятностью не ниже 0.5 / maxC
            char_highscore = np.where(mat[t, :] >= 0.5 / maxC)[0]
            for c in char_highscore:
                # Быстро упрощаем метку при добавлении нового символа
                newLabeling = fast_simplify_label(labeling, c, blankIdx)

                # Если новый символ такой же, как последний в текущей метке, учитываем только путь с пустым окончанием
                if labeling and labeling[-1] == c:
                    prNonBlank = mat[t, c] * last.entries[prev_labeling].prBlank
                else:
                    prNonBlank = mat[t, c] * last.entries[prev_labeling].prTotal

                addBeam(curr, newLabeling)
                curr.entries[newLabeling].labeling = newLabeling
                curr.entries[newLabeling].prNonBlank += prNonBlank
                curr.entries[newLabeling].prTotal += prNonBlank

                # Применение языковой модели (закомментировано, если требуется)
                # applyLM(curr.entries[labeling], curr.entries[newLabeling], classes, lm)

        # Обновляем состояние beam'ов
        last = curr

    # Нормализуем LM оценки с учетом длины метки
    last.norm()

    # Если список словаря пуст, выбираем наиболее вероятную метку
    if dict_list == []:
        best_labeling = last.sort()[0]
        res = ""
        for index, label in enumerate(best_labeling):
            # Убираем повторяющиеся символы и пустые символы
            if label not in ignore_idx and not (
                index > 0 and best_labeling[index - 1] == best_labeling[index]
            ):
                res += classes[label]
    else:
        res = last.wordsearch(classes, ignore_idx, 20, dict_list)
    return res


class CTCLabelConverter(object):
    """
    Конвертер для преобразования между текстовыми метками и индексами символов.
    """

    def __init__(self, character, separator_list={}, dict_pathlist={}):
        # character (str): набор возможных символов.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i + 1

        # Добавляем фиктивный токен "[blank]" для CTCLoss (индекс 0)
        self.character = ["[blank]"] + dict_character

        self.separator_list = separator_list
        separator_char = []
        for lang, sep in separator_list.items():
            separator_char += sep
        self.ignore_idx = [0] + [i + 1 for i, item in enumerate(separator_char)]

        # Загрузка латинского словаря, если separator_list пустой
        if len(separator_list) == 0:
            dict_list = []
            for lang, dict_path in dict_pathlist.items():
                try:
                    with open(dict_path, "r", encoding="utf-8-sig") as input_file:
                        word_count = input_file.read().splitlines()
                    dict_list += word_count
                except (FileNotFoundError, OSError) as e:
                    print(f"Предупреждение: Не удалось прочитать {dict_path}: {e}")
        else:
            dict_list = {}
            for lang, dict_path in dict_pathlist.items():
                with open(dict_path, "r", encoding="utf-8-sig") as input_file:
                    word_count = input_file.read().splitlines()
                dict_list[lang] = word_count

        self.dict_list = dict_list

    def encode(self, text, batch_max_length=25):
        """
        Преобразует текстовые метки в последовательность индексов для CTCLoss.

        Входные данные:
            text: список текстовых меток для каждого изображения (batch_size).
        Выходные данные:
            text: конкатенированная последовательность индексов для CTCLoss.
            length: список длин для каждого текста.
        """
        length = [len(s) for s in text]
        text = "".join(text)
        text = [self.dict[char] for char in text]

        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode_greedy(self, text_index, length):
        """
        Преобразует последовательность индексов в текстовую метку (жадный алгоритм).

        :param text_index: Тензор с индексами символов.
        :param length: Список длин последовательностей.
        :return: Список декодированных текстовых меток.
        """
        texts = []
        index = 0
        for seq_len in length:
            t = text_index[index : index + seq_len]
            # Создаем логический массив: True, если соседние значения не совпадают
            a = np.insert(~(t[1:] == t[:-1]), 0, True)
            # Логический массив: True, если значение не входит в ignore_idx
            b = ~np.isin(t, np.array(self.ignore_idx))
            # Объединяем два логических массива
            c = a & b
            # Собираем строку из символов по соответствующим индексам
            text = "".join(np.array(self.character)[t[c.nonzero()]].squeeze())
            texts.append(text)
            index += seq_len
        return texts

    def decode_beamsearch(self, mat, beamWidth=5):
        """
        Декодирование с использованием поиска по лучам (beam search).

        :param mat: Матрица вероятностей (batch x время x классы).
        :param beamWidth: Ширина луча.
        :return: Список декодированных текстовых меток.
        """
        texts = []
        for i in range(mat.shape[0]):
            t = ctcBeamSearch(
                mat[i], self.character, self.ignore_idx, None, beamWidth=beamWidth
            )
            texts.append(t)
        return texts

    def decode_wordbeamsearch(self, mat, beamWidth=5):
        """
        Декодирование с использованием поиска по лучам и сегментации слов.

        :param mat: Матрица вероятностей (batch x время x классы).
        :param beamWidth: Ширина луча.
        :return: Список декодированных строк.
        """
        texts = []
        # Получаем индексы максимальных значений по классу для каждого временного шага
        argmax = np.argmax(mat, axis=2)

        for i in range(mat.shape[0]):
            string = ""
            # Если разделителей нет – используем пробел как разделитель
            if len(self.separator_list) == 0:
                space_idx = self.dict[" "]
                data = np.argwhere(argmax[i] != space_idx).flatten()
                group = np.split(data, np.where(np.diff(data) != 1)[0] + 1)
                group = [list(item) for item in group if len(item) > 0]

                for j, list_idx in enumerate(group):
                    matrix = mat[i, list_idx, :]
                    t = ctcBeamSearch(
                        matrix,
                        self.character,
                        self.ignore_idx,
                        None,
                        beamWidth=beamWidth,
                        dict_list=self.dict_list,
                    )
                    if j == 0:
                        string += t
                    else:
                        string += " " + t
            # Если разделители присутствуют
            else:
                words = word_segmentation(argmax[i])
                for word in words:
                    matrix = mat[i, word[1][0] : word[1][1] + 1, :]
                    if word[0] == "":
                        dict_list = []
                    else:
                        dict_list = self.dict_list[word[0]]
                    t = ctcBeamSearch(
                        matrix,
                        self.character,
                        self.ignore_idx,
                        None,
                        beamWidth=beamWidth,
                        dict_list=dict_list,
                    )
                    string += t
            texts.append(string)
        return texts
