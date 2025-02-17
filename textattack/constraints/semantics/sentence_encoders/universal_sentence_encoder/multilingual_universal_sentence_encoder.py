"""
multilingual universal sentence encoder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""
import tensorflow_text  # noqa: F401

from textattack.constraints.semantics.sentence_encoders import SentenceEncoder
from textattack.shared.utils import LazyLoader

hub = LazyLoader("tensorflow_hub", globals(), "tensorflow_hub")


class MultilingualUniversalSentenceEncoder(SentenceEncoder):
    """Constraint using similarity between sentence encodings of x and x_adv
    where the text embeddings are created using the Multilingual Universal
    Sentence Encoder."""

    def __init__(self, threshold=0.8, large=False, metric="angular", **kwargs):
        super().__init__(threshold=threshold, metric=metric, **kwargs)
        #tensorflow_text._load()
        if large:
            tfhub_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3"
        else:
            tfhub_url = (
                "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
            )

        # TODO add QA SET. Details at: https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3
        self._tfhub_url = tfhub_url
        self.model = None

    def encode(self, sentences):
        if not self.model:
            self.model = hub.load(self._tfhub_url)
        return self.model(sentences).numpy()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["model"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        if not self.model:
            self.model = hub.load(self._tfhub_url)
