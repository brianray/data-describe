import pytest
import pandas as pd
import numpy as np

_COMPUTE_BACKENDS = ["pandas", "modin.pandas"]
_VIZ_BACKENDS = ["seaborn", "plotly"]


@pytest.fixture
def _hdbscan():
    return pytest.importorskip("hdbscan")


@pytest.fixture
def _pyscagnostics():
    return pytest.importorskip("pyscagnostics")


@pytest.fixture
def _gensim():
    return pytest.importorskip("gensim")


@pytest.fixture
def _nltk():
    return pytest.importorskip("nltk")


@pytest.fixture
def _presidio_analyzer():
    return pytest.importorskip("presidio_analyzer")


@pytest.fixture
def _statsmodels():
    return pytest.importorskip("statsmodels")


@pytest.fixture
def data():
    np.random.seed(22)
    return pd.DataFrame(
        {
            "a": np.random.normal(2, 1.2, size=250),
            "b": np.random.normal(3, 1.5, size=250),
            "c": np.random.normal(9, 0.2, size=250),
            "d": np.random.choice(["x", "y"], size=250),
            "e": np.random.choice(["v", "w"], p=[0.01, 0.99], size=250),
            "f": None,
        }
    )


@pytest.fixture
def numeric_data(data):
    return data.select_dtypes("number")


@pytest.fixture(params=_COMPUTE_BACKENDS)
def compute_backend_df(request):
    try:
        mod = pytest.importorskip(request.param)
    except ImportError:
        raise ValueError(f"Could not import {request.param}")

    np.random.seed(22)
    return mod.DataFrame(
        {
            "a": np.random.normal(2, 1.2, size=250),
            "b": np.random.normal(3, 1.5, size=250),
            "c": np.random.normal(9, 0.2, size=250),
            "d": np.random.choice(["x", "y"], size=250),
            "e": np.random.choice(["v", "w"], p=[0.01, 0.99], size=250),
            "z": np.zeros(250),
        }
    )


@pytest.fixture
def compute_numeric_backend_df(compute_backend_df):
    return compute_backend_df.select_dtypes("number")


@pytest.fixture
def text_data():
    return {
        "test_list_main": [
            "This is an article talking about Spider-man.",
            "Spiders are one of the weirdest things on earth.",
        ],
        "answer_key_tokenized": [
            ["This", "is", "an", "article", "talking", "about", "Spider-man", "."],
            [
                "Spiders",
                "are",
                "one",
                "of",
                "the",
                "weirdest",
                "things",
                "on",
                "earth",
                ".",
            ],
        ],
        "answer_key_lower": [
            ["this", "is", "an", "article", "talking", "about", "spider-man", "."],
            [
                "spiders",
                "are",
                "one",
                "of",
                "the",
                "weirdest",
                "things",
                "on",
                "earth",
                ".",
            ],
        ],
        "answer_key_remove_punct": [
            ["This", "is", "an", "article", "talking", "about", "Spider-man"],
            ["Spiders", "are", "one", "of", "the", "weirdest", "things", "on", "earth"],
        ],
        "answer_key_replace_all_punct_with_pipe": [
            ["This", "is", "an", "article", "talking", "about", "Spider|man"],
            ["Spiders", "are", "one", "of", "the", "weirdest", "things", "on", "earth"],
        ],
        "answer_key_remove_all_punct": [
            ["This", "is", "an", "article", "talking", "about", "Spiderman"],
            ["Spiders", "are", "one", "of", "the", "weirdest", "things", "on", "earth"],
        ],
        "answer_key_remove_stop_words": [
            ["article", "talking", "spider-man", "."],
            ["spiders", "one", "weirdest", "things", "earth", "."],
        ],
        "answer_key_remove_stop_words_more": [
            ["article", "talking", "."],
            ["one", "weirdest", "things", "earth", "."],
        ],
        "test_list_digits": [
            ["th3is", "is", "8", "a", "d1s4ster", "but", "no", "numb3rs"]
        ],
        "answer_key_remove_digits": [["", "is", "", "a", "", "but", "no", ""]],
        "test_list_single_char_and_spaces": [
            ["a", " ", "nice", "    ", "day", "to", "b", "outside", "thank you"]
        ],
        "answer_key_single_char_and_spaces": [
            ["nice", "day", "to", "outside", "thank", "you"]
        ],
        "test_list_stopwords": [
            [
                "please",
                "do",
                "not",
                "eat",
                "the",
                "rest",
                "of",
                "my",
                "pineapple",
                "pizza",
            ]
        ],
        "more_words": ["spiders", "spider-man"],
        "answer_key_remove_stopwords": [
            ["please", "eat", "rest", "pineapple", "pizza"]
        ],
        "answer_key_remove_stopwords_more": [["please", "rest", "pineapple"]],
        "test_list_lem_and_stem": [
            "Mars is the greatest planet to start terraforming; it would be amazing to see geese flying!"
        ],
        "answer_key_lem": [
            [
                "mar",
                "greatest",
                "planet",
                "start",
                "terraforming",
                "would",
                "amazing",
                "see",
                "goose",
                "flying",
            ]
        ],
        "answer_key_stem": [
            [
                "mar",
                "greatest",
                "planet",
                "start",
                "terraform",
                "would",
                "amaz",
                "see",
                "gees",
                "fly",
            ]
        ],
        "test_list_custom": ["this is an absolutely phenomenal day."],
        "answer_key_custom": [
            ["THIS", "IS", "AN", "ABSOLUTELY", "PHENOMENAL", "DAY", "."]
        ],
    }


@pytest.fixture
def document_data():
    topical_docs = [
        "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."
        "Unicorns are large, horned, bipedal mammals with short heads, tails, and ears. Unicorns are most commonly seen in countries of Central America, South America, and South Africa. They are among the oldest creatures on this planet and are believed to have lived on prehistoric Earth between 5 and 3 billion years ago."
        "Scientists estimate that the earliest signs of unicorn behavior show up in Central America at least 60 million years ago, with the earliest known example found in the Andean highlands of Peru. The presence of a unicorn behavior in a region where it is otherwise highly uncommon has allowed scientists to study how animals communicate, adapt and survive."
        "In the early 1990s, three scientists had been studying the behavior of the largest of many species of unicorns, the Ganesha unicorn. They found that the animals communicate only in a rudimentary form and were very silent in the presence of humans, dogs or other domestic animals.",
        '"The herd\'s incredible success is exciting because it demonstrates the tremendous potential for using animals for communication among people," says Professor Jo-Ann Faubert of the University of New South Wales, who has been working in the area to study the animals. The scientists believe there have been no other animals to communicate with, or even speak their language, at this stage of their evolution.'
        'What made this animal even more amazing was that it appears to have used a combination of communication techniques in order to successfully avoid its own hunters. Using a simple horn, the unicorn could produce a low-pitched whistle with which it could warn others of impending danger. "That could be the first time we have seen the use of a language in animals that use it to defend themselves," explains Professor Faubert. "It means if this particular unicorn can communicate with other animals"'
        '"I was amazed by the incredible range of characters," the researcher, Peter Henningsen a professor at the University of Aarhus, told The Local.'
        '"he fact that the animals are able to communicate seems extraordinary," he explained.The creatures were living peacefully with their herds of goats – which it seems they often slept in – and are said by the researchers to have been using special dialects, known as pidgin.',
        '"It\'s certainly an extraordinary case," said Henningsen.'
        "The discovery was made by Dr.Stefan R.Käppi, and colleagues from the Centre for the Ecology of Birds, Aarhus University in Denmark, with the help of an aerial drone equipped with a microphone to identify the unique language."
        '"It is quite possible that unicorns use the same language as all other mammals," he said.',
        "Today, scientists confirmed the worst possible outcome: the massive asteroid will collide with Earth in 2027 and cause widespread destruction on a scale of the most devastating natural disaster in our country's history.That's because one of the asteroid's two main fragments "
        "has an extremely hard core made of nickel, diamond and iron — which will be very difficult to remove without a robotic rover to do the work, NASA officials said at a press briefing Thursday. But some of the larger fragments contain other metals, such as nickel and cobalt, which"
        " can be easily broken apart by a rover and then blasted apart by a powerful burst of energy.The first asteroid to hit the Earth is called Chicxulub and will be about 4.4 miles (7 kilometers) in diameter, NASA officials said. The second asteroid is known as 2012 DA14 and will be about 2.7 miles (4.3 km) in diameter."
        "The two large chunks of the asteroid will probably crash into each other when they collide within 1.5 million years, NASA officials said.",
    ]
    return topical_docs


@pytest.fixture(params=_COMPUTE_BACKENDS)
def compute_time_data(request):
    mod = pytest.importorskip(request.param)
    dates = mod.to_datetime(
        [
            "2015-01-01",
            "2015-01-02",
            "2015-01-03",
            "2015-01-04",
            "2015-01-05",
            "2015-01-06",
            "2015-01-07",
            "2015-01-08",
            "2015-01-09",
            "2015-01-10",
            "2015-01-11",
            "2015-01-12",
            "2015-01-13",
            "2015-01-14",
            "2015-01-15",
        ],
        format="%Y-%m-%d",
    )
    return mod.DataFrame({"var": list(range(15))}, index=dates)


@pytest.fixture(params=_COMPUTE_BACKENDS)
def compute_backend_pii_df(request):
    mod = pytest.importorskip(request.param)
    return mod.DataFrame({"domain": "gmail.com", "name": "John Doe"}, index=[1])


@pytest.fixture(params=_COMPUTE_BACKENDS)
def compute_backend_pii_text(request):
    return "this is not a dataframe"


@pytest.fixture(params=_COMPUTE_BACKENDS)
def compute_backend_column_infotype(request):
    mod = pytest.importorskip(request.param)
    return mod.Series(["This string contains a domain, gmail.com"])


@pytest.fixture
def auto_arima_args():
    return {
        "start_p": 1,
        "start_q": 1,
        "max_p": 1,
        "max_q": 1,
        "m": 1,
        "start_P": 0,
        "seasonal": True,
        "d": 1,
        "D": 1,
        "trace": True,
        "error_action": "ignore",
        "suppress_warnings": True,
        "stepwise": True,
    }
