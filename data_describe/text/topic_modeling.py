import warnings
from typing import List, Optional, Dict, Tuple, TYPE_CHECKING

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD, NMF
import matplotlib.pyplot as plt
import seaborn as sns

from data_describe.config._config import get_option
from data_describe.text.text_preprocessing import (
    create_doc_term_matrix,
    create_tfidf_matrix,
    filter_dictionary,
)
from data_describe.backends import _get_viz_backend
from data_describe.compat import _compat, _requires, _in_notebook
from data_describe._widget import BaseWidget

warnings.filterwarnings("ignore", category=UserWarning, module="gensim")

if TYPE_CHECKING:
    gensim = _compat["gensim"]


def topic_model(
    text_docs: List[str],
    model_type: str = "LDA",
    num_topics: Optional[int] = None,
    min_topics: int = 2,
    max_topics: int = 10,
    no_below: int = 10,
    no_above: float = 0.2,
    tfidf: bool = True,
    model_kwargs: Optional[Dict] = None,
):
    """Topic modeling.

    Unsupervised methods of identifying topics in documents.

    Args:
        text_docs: A list of text documents in string format. These documents should
            generally be pre-processed
        model_type: {'LDA', 'LSA', 'LSI', 'SVD', 'NMF'}
            Defines the type of model/algorithm which will be used.
        num_topics: Sets the number of topics for the model. If None, will be optimized
            using coherence values
        min_topics: Starting number of topics to optimize for if number of topics not
            provided. Default is 2
        max_topics: Maximum number of topics to optimize for if number of topics not
            provided. Default is 10
        no_below: Minimum number of documents a word must appear in to be used in
            training. Default is 10
        no_above: Maximum proportion of documents a word may appear in to be used in
            training. Default is 0.2
        tfidf: If True, model created using TF-IDF matrix. Otherwise, document-term
            matrix with wordcounts is used. Default is True
        model_kwargs: Keyword arguments for the model, should be in agreement with
            `model_type`

    Returns:
        Topic model widget.
    """
    topicwidget = TopicModelWidget(model_type, num_topics, model_kwargs)
    topicwidget.fit(
        text_docs,
        model_type,
        min_topics,
        max_topics,
        no_below,
        no_above,
        tfidf,
        model_kwargs,
    )
    return topicwidget


@_requires("tqdm")
@_requires("gensim")
@_requires("pyLDAvis")
class TopicModelWidget(BaseWidget):
    """Create topic model widget."""

    def __init__(
        self,
        model_type: str = "LDA",
        num_topics: Optional[int] = None,
        model_kwargs: Optional[Dict] = None,
    ):
        """Topic Modeling made for easier training and understanding of topics.

        The exact model type, number of topics, and keyword arguments can be input
        to initialize the object. The object can then be used to train the model
        using the 'fit' function, and visualizations of the model can be displayed,
        such as an interactive visual (for LDA models only), an elbow plot displaying
        coherence values (for LDA or LSA/LSI models only), a DataFrame displaying the
        top keywords per topic, and a DataFrame displaying the top documents per topic.

        Args:
            model_type: {'LDA', 'LSA', 'LSI', 'SVD', 'NMF'}
                Defines the type of model/algorithm which will be used.
            num_topics: Sets the number of topics for the model. If None, will be
                optimized using coherence values (LDA or LSA/LSI) or becomes
                3 (SVD/NMF). Default is None.
            model_kwargs: Keyword arguments for the model, should be in agreement
                with `model_type`.

        Raises:
            ValueError: Invalid `model_type`.
        """
        self._model_type = model_type.upper()
        if self._model_type not in ["LDA", "LSA", "LSI", "SVD", "NMF"]:
            raise ValueError(
                "Model type must be one of either: 'LDA', 'LSA', 'LSI', 'SVD' or 'NMF'"
            )
        self._num_topics = num_topics
        self._model_kwargs = model_kwargs

    def __str__(self):
        return "data-describe Topic Model Widget"

    @property
    def model(self):
        """Trained topic model."""
        return self._model

    @property
    def model_type(self):
        """Type of model which either already has been or will be trained."""
        return self._model_type

    @property
    def num_topics(self):
        """The number of topics in the model."""
        return self._num_topics

    @property
    def coherence_values(self):
        """A list of coherence values mapped from min_topics to max_topics."""
        return self._coherence_values

    @property
    def dictionary(self):
        """A Gensim dictionary mapping the words from the documents to their token_ids."""
        return self._dictionary

    @property
    def corpus(self):
        """Bag of Words (BoW) representation of documents (token_id, token_count)."""
        return self._corpus

    @property
    def matrix(self):
        """Either TF-IDF or document-term matrix with documents as rows and words as columns."""
        return self._matrix

    @property
    def min_topics(self):
        """If num_topics is None, this number is the first number of topics a model will be trained on."""
        return self._min_topics

    @property
    def max_topics(self):
        """If num_topics is None, this number is the last number of topics a model will be trained on."""
        return self._max_topics

    def show(self, num_topic_words: int = 10, topic_names: Optional[List[str]] = None):
        """Displays most relevant terms for each topic.

        Args:
            num_topic_words: The number of words to be displayed for each topic. Default is 10
            topic_names: A list of pre-defined names set for each of the topics. Default is None

        Returns:
            display_topics_df: Pandas DataFrame displaying topics as columns and their
                relevant terms as rows. LDA/LSI models will display an extra column to
                the right of each topic column, showing each term's corresponding
                coefficient value
        """
        return self.display_topic_keywords(
            num_topic_words=num_topic_words, topic_names=topic_names
        )

    def _compute_lsa_svd_model(self, text_docs: List[str], tfidf: bool = True):
        """Trains LSA TruncatedSVD scikit-learn model.

        Args:
            text_docs: A list of text documents in string format. These documents should
                generally be pre-processed.
            tfidf: If True, model created using TF-IDF matrix. Otherwise, document-term
                matrix with wordcounts is used. Default is True.

        Returns:
            lsa_model: Trained LSA topic model
        """
        if not self._num_topics:
            self._num_topics = 3

        if tfidf:
            self._matrix = create_tfidf_matrix(text_docs)
        else:
            self._matrix = create_doc_term_matrix(text_docs)

        if not self._model_kwargs:
            self._model_kwargs = {}
        self._model_kwargs.update({"n_components": self._num_topics})

        lsa_model = TruncatedSVD(**self._model_kwargs)
        lsa_model.fit(self._matrix)
        return lsa_model

    def _compute_lsi_model(
        self,
        text_docs: List[str],
        min_topics: int = 2,
        max_topics: int = 10,
        no_below: int = 10,
        no_above: float = 0.2,
    ):
        """Trains LSA Gensim model.

        Args:
            text_docs: A list of text documents in string format. These documents should
                generally be pre-processed
            min_topics: Starting number of topics to optimize for if number of topics
                not provided. Default is 2
            max_topics: Maximum number of topics to optimize for if number of topics not
                provided. Default is 10
            no_below: Minimum number of documents a word must appear in to be used in
                training. Default is 10
            no_above: Maximum proportion of documents a word may appear in to be used in
                training. Default is 0.2

        Returns:
            lsa_model: Trained LSA topic model
        """
        tokenized_text_docs = [text_doc.split() for text_doc in text_docs]
        self._min_topics = min_topics
        self._max_topics = max_topics
        self._dictionary, self._corpus = filter_dictionary(
            tokenized_text_docs, no_below, no_above
        )
        lsa_model_list = []

        if not self._model_kwargs:
            self._model_kwargs = {}
        self._model_kwargs.update(
            {
                "corpus": self._corpus,
                "num_topics": self._num_topics,
                "id2word": self._dictionary,
            }
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            if self._num_topics is None:
                self._coherence_values = []
                pbar = _compat["tqdm"].tqdm(  # type: ignore
                    range(self._min_topics, self._max_topics + 1),
                    desc="Fitting topic model",
                )
                for num in pbar:
                    self._model_kwargs.update({"num_topics": num})
                    lsa_model = _compat[  # type: ignore
                        "gensim"
                    ].models.lsimodel.LsiModel(**self._model_kwargs)
                    coherence_model = _compat[  # type: ignore
                        "gensim"
                    ].models.coherencemodel.CoherenceModel(
                        model=lsa_model,
                        texts=tokenized_text_docs,
                        dictionary=self._dictionary,
                        coherence="c_v",
                    )
                    score = coherence_model.get_coherence()
                    self._coherence_values.append(score)
                    lsa_model_list.append(lsa_model)
                max_coherence_index = self._coherence_values.index(
                    max(self._coherence_values)
                )
                self._num_topics = len(lsa_model_list[max_coherence_index].get_topics())
                return lsa_model_list[max_coherence_index]
            else:
                lsa_model = _compat["gensim"].models.lsimodel.LsiModel(  # type: ignore
                    corpus=self._corpus,
                    id2word=self._dictionary,
                    num_topics=self._num_topics,
                )
                return lsa_model

    def _compute_lda_model(
        self,
        text_docs: List[str],
        min_topics: int = 2,
        max_topics: int = 10,
        no_below: int = 10,
        no_above: float = 0.2,
    ):
        """Trains LDA Gensim model.

        Args:
            text_docs: A list of text documents in string format. These documents
                should generally be pre-processed
            min_topics: Starting number of topics to optimize for if number of topics
                not provided. Default is 2
            max_topics: Maximum number of topics to optimize for if number of topics
                not provided. Default is 10
            no_below: Minimum number of documents a word must appear in to be used in
            training. Default is 10
            no_above: Maximum proportion of documents a word may appear in to be used in
                training. Default is 0.2

        Returns:
            lda_model (Gensim LdaModel): Trained LDA topic model
        """
        tokenized_text_docs = [text_doc.split() for text_doc in text_docs]
        self._min_topics = min_topics
        self._max_topics = max_topics
        self._dictionary, self._corpus = filter_dictionary(
            tokenized_text_docs, no_below, no_above
        )
        lda_model_list = []

        if not self._model_kwargs:
            self._model_kwargs = {}
        self._model_kwargs = {
            **{
                "random_state": 1,
                "corpus": self._corpus,
                "num_topics": self._num_topics,
                "id2word": self._dictionary,
            },
            **self._model_kwargs,
        }

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            if self._num_topics is None:
                self._coherence_values = []
                pbar = _compat["tqdm"].tqdm(  # type: ignore
                    range(self._min_topics, self._max_topics + 1),
                    desc="Fitting topic model",
                )
                for num in pbar:
                    self._model_kwargs.update({"num_topics": num})
                    lda_model = _compat[  # type: ignore
                        "gensim"
                    ].models.ldamodel.LdaModel(**self._model_kwargs)
                    coherence_model = _compat[  # type: ignore
                        "gensim"
                    ].models.coherencemodel.CoherenceModel(
                        model=lda_model,
                        texts=tokenized_text_docs,
                        dictionary=self._dictionary,
                        coherence="c_v",
                    )
                    score = coherence_model.get_coherence()
                    self._coherence_values.append(score)
                    lda_model_list.append(lda_model)
                max_coherence_index = self._coherence_values.index(
                    max(self._coherence_values)
                )
                self._num_topics = len(lda_model_list[max_coherence_index].get_topics())
                return lda_model_list[max_coherence_index]
            else:
                lda_model = _compat["gensim"].models.ldamodel.LdaModel(  # type: ignore
                    **self._model_kwargs
                )
                return lda_model

    def _compute_nmf_model(self, text_docs: List[str], tfidf: bool = True):
        """Trains NMF scikit-learn model.

        Args:
            text_docs: A list of text documents in string format. These documents should
                generally be pre-processed
            tfidf: If True, model created using TF-IDF matrix. Otherwise, document-term
                matrix with wordcounts is used. Default is True.

        Returns:
            lsa_model (scikit-learn NMF model): Trained NMF topic model
        """
        if not self._num_topics:
            self._num_topics = 3

        if tfidf:
            self._matrix = create_tfidf_matrix(text_docs)
        else:
            self._matrix = create_doc_term_matrix(text_docs)

        if not self._model_kwargs:
            self._model_kwargs = {}
        self._model_kwargs.update({"n_components": self._num_topics})

        nmf_model = NMF(**self._model_kwargs)
        nmf_model.fit(self._matrix)
        return nmf_model

    def fit(
        self,
        text_docs: List[str],
        model_type: Optional[str] = None,
        min_topics: int = 2,
        max_topics: int = 10,
        no_below: int = 10,
        no_above: float = 0.2,
        tfidf: bool = True,
        model_kwargs: Optional[Dict] = None,
    ):
        """Trains topic model and assigns model to object as attribute.

        Args:
            text_docs: A list of text documents in string format. These documents should
                generally be pre-processed
            model_type: {'LDA', 'LSA', 'LSI', 'SVD', 'NMF'}
                Defines the type of model/algorithm which will be used.
            min_topics: Starting number of topics to optimize for if number of topics
                not provided. Default is 2
            max_topics: Maximum number of topics to optimize for if number of topics not
                provided. Default is 10
            no_below: Minimum number of documents a word must appear in to be used in
                training. Default is 10
            no_above: Maximum proportion of documents a word may appear in to be used in
                training. Default is 0.2
            tfidf: If True, model created using TF-IDF matrix. Otherwise, document-term
                matrix with wordcounts is used. Default is True.
            model_kwargs: Keyword arguments for the model, should be in agreement with
                `model_type`.

        Raises:
            ValueError: Invalid `model_type`.
        """
        if model_kwargs is not None:
            self._model_kwargs = model_kwargs
        if model_type is not None:
            self._model_type = model_type.upper()
            if self._model_type not in ["LDA", "LSA", "LSI", "SVD", "NMF"]:
                raise ValueError(
                    "Model type must be one of either: 'LDA', 'LSA', 'LSI', 'SVD' or 'NMF'"
                )

        if self._model_type == "LDA":
            self._model = self._compute_lda_model(
                text_docs, min_topics, max_topics, no_below, no_above
            )
        elif self._model_type == "LSA" or self._model_type == "LSI":
            self._model = self._compute_lsi_model(
                text_docs, min_topics, max_topics, no_below, no_above
            )
        elif self._model_type == "SVD":
            self._model = self._compute_lsa_svd_model(text_docs, tfidf)
        elif self._model_type == "NMF":
            self._model = self._compute_nmf_model(text_docs, tfidf)

    def elbow_plot(self, viz_backend: str = None):
        """Creates an elbow plot displaying coherence values vs number of topics.

        Args:
            viz_backend: The visualization backend.

        Raises:
            ValueError: No coherence values to plot.

        Returns:
            fig: Elbow plot showing coherence values vs number of topics
        """
        try:
            self._coherence_values
        except AttributeError:
            raise ValueError(
                "Coherence values not defined. At least 2 LDA or LSI models need to be"
                " trained with different numbers of topics."
            )
        else:
            return _get_viz_backend(viz_backend).viz_elbow_plot(
                self._min_topics, self._max_topics, self._coherence_values
            )

    def get_topic_nums(self):
        """Obtains topic distributions (LDA model) or scores (LSA/NMF model).

        Returns:
            doc_topics: Array of topic distributions (LDA model) or scores (LSA/NMF model)
        """
        if self._model_type == "NMF" or self._model_type == "SVD":
            return self._model.transform(self._matrix)
        elif self._model_type == "LDA":
            doc_topics = []
            for doc in list(
                self._model.get_document_topics(self._corpus, minimum_probability=0)
            ):
                current_doc = [topic[1] for topic in doc]
                doc_topics.append(current_doc)
            return np.array(doc_topics)
        elif self._model_type == "LSI" or self._model_type == "LSA":
            doc_topics = []
            for doc in self._model[self._corpus]:
                current_doc = [topic[1] for topic in doc]
                if current_doc:
                    doc_topics.append(current_doc)
                else:
                    doc_topics.append([0] * len(self._model.get_topics()))
            return np.array(doc_topics)

    def display_topic_keywords(
        self, num_topic_words: int = 10, topic_names: Optional[List[str]] = None
    ):
        """Creates Pandas DataFrame to display most relevant terms for each topic.

        Args:
            num_topic_words: The number of words to be displayed for each topic.
                Default is 10
            topic_names: A list of pre-defined names set for each of the topics.
                Default is None

        Returns:
            display_topics_df: Pandas DataFrame displaying topics as columns and their
                relevant terms as rows. LDA/LSI models will display an extra column to
                the right of each topic column, showing each term's corresponding
                coefficient value
        """
        display_topics_dict = {}
        if self._model_type == "NMF" or self._model_type == "SVD":
            for topic_num, topic in enumerate(self._model.components_):
                if not topic_names or not topic_names[topic_num]:
                    key = "Topic {}".format(topic_num + 1)
                else:
                    key = "Topic: {}".format(topic_names[topic_num])
                display_topics_dict[key] = [
                    self._matrix.columns[i]
                    for i in topic.argsort()[: -num_topic_words - 1 : -1]
                ]
        elif (
            self._model_type == "LSI"
            or self._model_type == "LSA"
            or self._model_type == "LDA"
        ):
            for topic_num, topic in self._model.print_topics(num_words=num_topic_words):
                topic_words = [
                    topic.split()[num].split("*")[1].replace('"', "")
                    for num in range(0, len(topic.split()), 2)
                ]
                topic_coefficients = [
                    topic.split()[num].split("*")[0]
                    for num in range(0, len(topic.split()), 2)
                ]
                if not topic_names or not topic_names[topic_num]:
                    key = "Topic {}".format(topic_num + 1)
                    coefficient_key = "Topic {} Coefficient Value".format(topic_num + 1)
                else:
                    key = "Topic: {}".format(topic_names[topic_num])
                    coefficient_key = "Topic: {} - Coefficient Value".format(
                        topic_names[topic_num]
                    )
                display_topics_dict[key], display_topics_dict[coefficient_key] = (
                    topic_words,
                    topic_coefficients,
                )

        term_numbers = ["Term {}".format(num + 1) for num in range(num_topic_words)]
        display_topics_df = pd.DataFrame(display_topics_dict, index=term_numbers)
        return display_topics_df

    def top_documents_per_topic(
        self,
        text_docs: List[str],
        topic_names: Optional[List[str]] = None,
        num_docs: int = 10,
        summarize_docs: bool = False,
        summary_words: Optional[int] = None,
    ):
        """Creates Pandas DataFrame to display most relevant documents for each topic.

        Args:
            text_docs: A list of text documents in string format. Important to note that
                 this list of documents should be ordered in accordance with the matrix
                or corpus on which the document was trained
            topic_names: A list of pre-defined names set for each of the topics.
                Default is None
            num_docs: The number of documents to display for each topic. Default is 10
            summarize_docs: If True, the documents will be summarized (if this is the
                case, 'text_docs' should be formatted into sentences). Default is False
            summary_words: The number of words the summary should be limited to. Should
                only be specified if summarize_docs set to True

        Returns:
            all_top_docs_df: Pandas DataFrame displaying topics as columns and their
                most relevant documents as rows
        """
        topics = self.get_topic_nums()

        if summary_words and not summarize_docs:
            warnings.warn("'summary_words' specified, but 'summarize' set to False.")

        all_top_docs = {}
        sentence_check = 0
        # iterate through each topic
        for topic_num in range(topics.shape[1]):
            # sort documents in order of relevancy to current topic
            topic = [topic[topic_num] for topic in topics]
            indexes = list(np.argsort(topic)[::-1])[:num_docs]
            top_docs = [text_docs[index] for index in indexes]

            # summarize docs if specified
            if summarize_docs:
                summarized_docs = []
                # use set number of words if specified
                # try to summarize, does not work if document is only one sentence so then just append document
                for index, doc in enumerate(top_docs):
                    if summary_words:
                        try:
                            summarized_docs.append(
                                _compat[  # type: ignore
                                    "gensim"
                                ].summarization.summarizer.summarize(
                                    doc, word_count=summary_words
                                )
                            )
                        except ValueError:
                            sentence_check += 1
                            warnings.warn(
                                "Document #{} in Topic {} cannot be summarized.".format(
                                    str(index + 1), topic_num
                                )
                            )
                            summarized_docs.append(doc)
                    else:
                        try:
                            summarized_docs.append(
                                _compat[  # type: ignore
                                    "gensim"
                                ].summarization.summarizer.summarize(doc)
                            )
                        except ValueError:
                            sentence_check += 1
                            warnings.warn(
                                "Document #{} in Topic {} cannot be summarized.".format(
                                    str(index + 1), topic_num
                                )
                            )
                            summarized_docs.append(doc)
                top_docs = summarized_docs

            if not topic_names:
                all_top_docs["Topic " + str(topic_num + 1)] = top_docs
            else:
                all_top_docs["Topic: " + topic_names[topic_num]] = top_docs

        if sentence_check > (num_docs * len(topics[0]) / 4):
            warnings.warn("Documents must be formatted as sentences for summarization.")

        doc_numbers = ["Document #" + str(num + 1) for num in range(num_docs)]
        all_top_docs_df = pd.DataFrame(all_top_docs, index=doc_numbers)
        return all_top_docs_df

    def visualize_topic_summary(self, viz_backend: str = "pyLDAvis"):
        """Displays interactive pyLDAvis visual to understand topic model and documents.

        Args:
            viz_backend (str): The visualization backend.

        Raises:
            TypeError: Only valid for LDA models.

        Returns:
            A visual to understand topic model and/or documents relating to model
        """
        if self._model_type != "LDA":
            raise TypeError("Model must be an LDA Model")
        else:
            return _get_viz_backend(viz_backend).viz_visualize_topic_summary(
                self._model, self._corpus, self._dictionary
            )


@_requires("pyLDAvis")
@_requires("gensim")
def _pyldavis_viz_visualize_topic_summary(
    model: "gensim.models.ldamodel.LdaModel",  # type: ignore
    corpus: List[List[Tuple[int, int]]],
    dictionary: "gensim.corpora.dictionary.Dictionary",  # type: ignore
):
    """Displays interactive pyLDAvis visual to understand topic model and documents.

    Args:
        model: LDA topic model
        corpus: Bag of Words (BoW) representation of documents (token_id, token_count)
        dictionary: Gensim Dictionary encapsulates the mapping between normalized words
            and their integer ids.

    Raises:
        EnvironmentError: Must be in a Notebook.

    Returns:
        A visual to understand topic model and/or documents relating to model
    """
    if not _in_notebook():
        raise EnvironmentError("Not in Jupyter Notebook")

    _compat["pyLDAvis"].enable_notebook()  # type: ignore
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            module="pyLDAvis",
            message="Sorting because non-concatenation axis is not aligned.",
        )
        vis = _compat["pyLDAvis.gensim"].prepare(model, corpus, dictionary)  # type: ignore
        return vis


def _seaborn_viz_elbow_plot(
    min_topics: int, max_topics: int, coherence_values: List[float]
):
    """Creates an elbow plot displaying coherence values vs number of topics.

    Args:
        min_topics: Starting number of topics that were optimized for
        max_topics: Maximum number of topics that were optimized for
        coherence_values: A list of coherence values mapped from min_topics to
            max_topics

    Returns:
        Elbow plot showing coherence values vs number of topics
    """
    ax = sns.lineplot(
        x=[num for num in range(min_topics, max_topics + 1)],
        y=coherence_values,
    )
    ax.set_title("Coherence Values Across Topic Numbers")
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Values")
    plt.figure(
        figsize=(
            get_option("display.matplotlib.fig_width"),
            get_option("display.matplotlib.fig_height"),
        )
    )
    return ax
