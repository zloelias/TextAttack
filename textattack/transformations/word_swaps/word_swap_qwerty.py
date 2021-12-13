"""
Word Swap by swaps characters with QWERTY adjacent keys
----------------------------------------------------------
"""

import random

from .word_swap import WordSwap


class WordSwapQWERTY(WordSwap):
    def __init__(
        self, random_one=True, skip_first_char=False, skip_last_char=False, **kwargs
    ):
        """A transformation that swaps characters with adjacent keys on a
        QWERTY keyboard, replicating the kind of errors that come from typing
        too quickly.

        :param random_one: Whether to return a single (random) swap, or all possible swaps.
        :param skip_first_char: When True, do not modify the first character of each word.
        :param skip_last_char: When True, do not modify the last character of each word.
        >>> from textattack.transformations import WordSwapQWERTY
        >>> from textattack.augmentation import Augmenter

        >>> transformation = WordSwapQWERT()
        >>> augmenter = Augmenter(transformation=transformation)
        >>> s = 'I am fabulous.'
        >>> augmenter.augment(s)
        """
        super().__init__(**kwargs)
        self.random_one = random_one
        self.skip_first_char = skip_first_char
        self.skip_last_char = skip_last_char

        self._keyboard_adjacency = {
            "q": [
                "w",
                "a",
                "s",
            ],
            "w": ["q", "e", "a", "s", "d"],
            "e": ["w", "s", "d", "f", "r"],
            "r": ["e", "d", "f", "g", "t"],
            "t": ["r", "f", "g", "h", "y"],
            "y": ["t", "g", "h", "j", "u"],
            "u": ["y", "h", "j", "k", "i"],
            "i": ["u", "j", "k", "l", "o"],
            "o": ["i", "k", "l", "p"],
            "p": ["o", "l"],
            "a": ["q", "w", "s", "z", "x"],
            "s": ["q", "w", "e", "a", "d", "z", "x"],
            "d": ["w", "e", "r", "f", "c", "x", "s"],
            "f": ["e", "r", "t", "g", "v", "c", "d"],
            "g": ["r", "t", "y", "h", "b", "v", "d"],
            "h": ["t", "y", "u", "g", "j", "b", "n"],
            "j": ["y", "u", "i", "k", "m", "n", "h"],
            "k": ["u", "i", "o", "l", "m", "j"],
            "l": ["i", "o", "p", "k"],
            "z": ["a", "s", "x"],
            "x": ["s", "d", "z", "c"],
            "c": ["x", "d", "f", "v"],
            "v": ["c", "f", "g", "b"],
            "b": ["v", "g", "h", "n"],
            "n": ["b", "h", "j", "m"],
            "m": ["n", "j", "k"],
            "й": ["ц", "ы", "ф"],
            "ц": ["у", "в", "ы", "ф", "й"],
            "у": ["к", "а", "в", "ы", "ц"],
            "к": ["е", "п", "а", "в", "у"],
            "е": ["н", "р", "п", "а", "к"],
            "н": ["г", "о", "р", "п", "е"],
            "г": ["ш", "л", "о", "р", "н"],
            "ш": ["щ", "д", "л", "о", "г"],
            "щ": ["з", "ж", "д", "л", "ш"],
            "з": ["х", "э", "ж", "д", "щ"],
            "х": ["ъ", "э", "ж", "з"],
            "ъ": ["э", "х"],
            "ф": ["й", "ц", "ы", "ч", "я"],
            "ы": ["в", "с", "ч", "я", "ф", "й", "ц", "у"],
            "в": ["а", "м", "с", "ч", "ы", "ц", "у", "к"],
            "а": ["п", "и", "м", "с", "в", "у", "к", "е"],
            "п": ["р", "т", "и", "м", "а", "к", "е", "н"],
            "р": ["о", "ь", "т", "и", "п", "е", "н", "г"],
            "о": ["л", "б", "ь", "т", "р", "н", "г", "ш"],
            "л": ["д", "ю", "б", "ь", "о", "г", "ш", "щ"],
            "д": ["ж", "ю", "б", "л", "ш", "щ", "з"],
            "ж": ["э", "ю", "д", "щ", "з", "х"],
            "э": ["ъ", "х", "з", "ж"],
            "я": ["ф", "ы", "ч"],
            "ч": ["с", "в", "ы", "ф", "я"],
            "с": ["м", "а", "в", "ы", "ч"],
            "м": ["и", "п", "а", "в", "с"],
            "и": ["т", "р", "п", "а", "м"],
            "т": ["ь", "о", "р", "п", "и"],
            "ь": ["б", "л", "о", "р", "т"],
            "б": ["ю", "д", "л", "о", "ь"],
            "ю": ["ж", "д", "л", "б"],
        }

    def _get_adjacent(self, s):
        s_lower = s.lower()
        if s_lower in self._keyboard_adjacency:
            adjacent_keys = self._keyboard_adjacency.get(s_lower, [])
            if s.isupper():
                return [key.upper() for key in adjacent_keys]
            else:
                return adjacent_keys
        else:
            return [s]

    def _get_replacement_words(self, word):
        if len(word) <= 1:
            return []

        candidate_words = []

        start_idx = 1 if self.skip_first_char else 0
        end_idx = len(word) - (1 + self.skip_last_char)

        if start_idx >= end_idx:
            return []

        if self.random_one:
            i = random.randrange(start_idx, end_idx + 1)
            candidate_word = (
                word[:i] + random.choice(self._get_adjacent(word[i])) + word[i + 1 :]
            )
            candidate_words.append(candidate_word)
        else:
            for i in range(start_idx, end_idx + 1):
                for swap_key in self._get_adjacent(word[i]):
                    candidate_word = word[:i] + swap_key + word[i + 1 :]
                    candidate_words.append(candidate_word)

        return candidate_words

    @property
    def deterministic(self):
        return not self.random_one
