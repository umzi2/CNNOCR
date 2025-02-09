class TextMetrics:
    """
    Класс для вычисления метрик качества текста:
      - WER (Word Error Rate, ошибка слов)
      - CER (Character Error Rate, ошибка символов)
    """

    @staticmethod
    def _compute_edit_distance(seq1, seq2):
        """
        Вычисляет расстояние редактирования (расстояние Левенштейна) между двумя последовательностями.

        :param seq1: Первая последовательность (например, список слов или символов).
        :param seq2: Вторая последовательность (например, список слов или символов).
        :return: Расстояние редактирования между seq1 и seq2.
        """
        rows = len(seq1) + 1
        cols = len(seq2) + 1

        # Инициализация матрицы для динамического программирования
        dp = [[0] * cols for _ in range(rows)]

        # Заполняем первую строку и первый столбец: стоимость преобразования пустой последовательности
        for i in range(rows):
            dp[i][0] = i  # удаление всех символов/слов из seq1
        for j in range(cols):
            dp[0][j] = j  # вставка всех символов/слов в seq1

        # Вычисляем расстояние редактирования для каждой пары префиксов последовательностей
        for i in range(1, rows):
            for j in range(1, cols):
                # Если текущие элементы совпадают, стоимость замены равна 0, иначе 1
                if seq1[i - 1] == seq2[j - 1]:
                    cost = 0
                else:
                    cost = 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,  # удаление элемента из seq1
                    dp[i][j - 1] + 1,  # вставка элемента в seq1
                    dp[i - 1][j - 1] + cost,  # замена элемента
                )
        # Возвращаем значение в правом нижнем углу матрицы, которое и есть расстояние редактирования
        return dp[-1][-1]

    def evaluate(self, references, hypotheses):
        """
        Вычисляет метрики WER и CER для набора референсных текстов и гипотез.

        :param references: Список правильных (референсных) текстов.
        :param hypotheses: Список распознанных текстов (гипотез).
        :return: Кортеж (wer, cer) - ошибка слов и ошибка символов.
        :raises ValueError: Если длины списков references и hypotheses не совпадают.
        """
        # Проверяем, что количество референсных текстов совпадает с количеством гипотез
        if len(references) != len(hypotheses):
            raise ValueError(
                "Количество референсных и гипотезных предложений должно совпадать!"
            )

        total_word_errors = 0  # Общее количество ошибок на уровне слов
        total_word_count = 0  # Общее количество слов в референсах
        total_char_errors = 0  # Общее количество ошибок на уровне символов
        total_char_count = 0  # Общее количество символов в референсах

        # Проходим по каждой паре (референс, гипотеза)
        for ref, hyp in zip(references, hypotheses):
            # Разбиваем референс и гипотезу на слова
            ref_words = ref.split()
            hyp_words = hyp.split()
            # Вычисляем количество ошибок (расстояние редактирования) для слов
            word_errors = self._compute_edit_distance(ref_words, hyp_words)
            total_word_errors += word_errors
            total_word_count += len(ref_words)

            # Разбиваем референс и гипотезу на символы
            ref_chars = list(ref)
            hyp_chars = list(hyp)
            # Вычисляем количество ошибок для символов
            char_errors = self._compute_edit_distance(ref_chars, hyp_chars)
            total_char_errors += char_errors
            total_char_count += len(ref_chars)

        # Вычисляем WER и CER (защита от деления на ноль)
        wer = total_word_errors / total_word_count if total_word_count > 0 else 0
        cer = total_char_errors / total_char_count if total_char_count > 0 else 0

        return wer, cer
