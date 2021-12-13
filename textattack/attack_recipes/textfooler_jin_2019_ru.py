"""

TextFooler (Is BERT Really Robust?)
===================================================
A Strong Baseline for Natural Language Attack on Text Classification and Entailment)

"""

from textattack import Attack
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.pre_transformation import (
    InputColumnModification,
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import WordSwapEmbedding

from .attack_recipe import AttackRecipe

from textattack.shared.utils import LazyLoader
from textattack.shared import GensimWordEmbedding, WordEmbedding


class TextFoolerJin2019RU(AttackRecipe):
    """Jin, D., Jin, Z., Zhou, J.T., & Szolovits, P. (2019).

    Is BERT Really Robust? Natural Language Attack on Text Classification and Entailment.

    https://arxiv.org/abs/1907.11932
    """

    @staticmethod
    def build(model_wrapper):
        #
        # Swap words with their 50 closest embedding nearest-neighbors.
        # Embedding: Gensim 'ruscorpora_1_300_10.bin.gz'
        #

        gensim = LazyLoader("gensim", globals(), "gensim")
        keyed_vectors = (
            gensim.models.keyedvectors.Word2VecKeyedVectors.load_word2vec_format('https://rusvectores.org/static/models/ruscorpora_upos_skipgram_300_10_2017.bin.gz', binary=True)
        )
        word_embedding = GensimWordEmbedding(keyed_vectors)

        transformation = WordSwapEmbedding(max_candidates=20, embedding=word_embedding)
        #
        # Don't modify the same word twice or the stopwords defined
        # in the TextFooler public implementation.
        #
        # fmt: off
        # fmt: on
        constraints = [RepeatModification(), StopwordModification(language='russian')]
        #
        # During entailment, we should only edit the hypothesis - keep the premise
        # the same.
        #
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )
        constraints.append(input_column_modification)
        # Minimum word embedding cosine similarity of 0.5.
        # (The paper claims 0.7, but analysis of the released code and some empirical
        # results show that it's 0.5.)
        #
        constraints.append(WordEmbeddingDistance(min_cos_sim=0.5, embedding=word_embedding))
        #
        # Only replace words with the same part of speech (or nouns with verbs)
        #
        constraints.append(PartOfSpeech(allow_verb_noun_swap=True))
        #
        # Universal Sentence Encoder with a minimum angular similarity of ε = 0.5.
        #
        # In the TextFooler code, they forget to divide the angle between the two
        # embeddings by pi. So if the original threshold was that 1 - sim >= 0.5, the
        # new threshold is 1 - (0.5) / pi = 0.840845057
        #
        use_constraint = UniversalSentenceEncoder(
            threshold=0.840845057,
            metric="angular",
            compare_against_original=False,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
        constraints.append(use_constraint)
        #
        # Goal is untargeted classification
        #
        goal_function = UntargetedClassification(model_wrapper)
        #
        # Greedily swap words with "Word Importance Ranking".
        #
        search_method = GreedyWordSwapWIR(wir_method="delete")

        return Attack(goal_function, constraints, transformation, search_method)
