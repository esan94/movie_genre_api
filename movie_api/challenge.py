"""
Flask REST API for movie challenge.

Author: Esteban M. Sanchez Garcia
LinkedIn: linkedin.com/in/estebanmsg/
GitHub: github.com/esan94
Medium: medium.com/@emsg94
"""

from io import StringIO
from typing import List

from flask import Flask, request, Response
from nltk import download, pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
from pandas import DataFrame, read_csv
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder


download("averaged_perceptron_tagger")
download("punkt")
download("wordnet")


class ModelNode():
    """Node to save model used in the train process."""

    def __init__(self):
        """Initialize the instance ModelNode."""
        self.clf = None
        self.classes = None
        self.count_vect = None
        self.one_hot = None

    def set_parameters(self, clf_model, classes, count_vect, one_hot):
        """Set parameters from training process."""
        self.clf = clf_model
        self.classes = classes
        self.count_vect = count_vect
        self.one_hot = one_hot

    def get_parameters(self):
        """Get parameters from training process."""
        return self.clf, self.classes, self.count_vect, self.one_hot


MODEL_NODE = ModelNode()

POS_MAP_LEMMA = {
    "JJ": "a", "JJR": "a", "JJS": "a", "NN": "n", "NNS": "n", "RB": "r",
    "RBR": "r", "RBS": "r", "VB": "v", "VBD": "v", "VBG": "v", "VBN": "v",
    "VBP": "v", "VBZ": "v"}
PUNCTUATION = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
app = Flask(__name__)


@app.route('/genres/predict', methods=["POST"])
def post_csv_and_predict():
    """Class used predict one or more movies based on the synopsis.

    Post method used to predict the movie genres.

    """
    PredictNode(StringIO(request.data.decode("utf-8")),
                MODEL_NODE.get_parameters()).run()
    return Response(status=200, headers={
        "description": "The top 5 predicted movie genres",
        "Content-Type": "text/csv"})


@app.route('/genres/train', methods=["POST"])
def post_csv_and_train():
    """Take binary data, train a model that predicts 5 genres of a movie.

    Post method used to read data and train model.

    """
    train = TrainNode(StringIO(request.data.decode("utf-8")))
    train.run()

    MODEL_NODE.set_parameters(train.clf, train.multilabel_bin.classes_,
                              train.count_vect, train.one_hot)

    return Response(status=200, headers={
        "message": "The model has been successfully trained"})


class BaseNode():
    """Base node for the training and predictions steps."""

    columns: List[str]

    def __init__(self):
        """Initialize the instance BaseNode."""
        self.punctuation = dict.fromkeys(PUNCTUATION, " ")
        self.pos_map_lemma = POS_MAP_LEMMA
        self.lemmatizer = WordNetLemmatizer()
        self.multilabel_bin = MultiLabelBinarizer()
        self.string_data = ""
        self.columns = list()

    def _read(self):
        """Read csv from the API."""
        return read_csv(self.string_data, sep=",")[self.columns]

    @staticmethod
    def _lemmatization(text, lemmatizer, pos_lemma):
        """Make lematization in a text.

        Make a lemmatization taking into account part-of-speech
        on each word of a sencente. And delete extra white
        space.

        Parameters
        ----------
            text: str
                Synopsis of a film.

            lemmatizer: class
                WordNetLemmatizer instance.

            pos_lemma: dict
                Part of speech mapping dictionary.

        Return
        ------
            str:
                Synopsys lemmatized.

        """
        return " ".join([lemmatizer.lemmatize(token[0],
                         pos=pos_lemma.get(token[1], "n"))
                         for token in pos_tag(word_tokenize(text))])

    @staticmethod
    def is_alpha(text):
        """Only letters.

        Keep only alphabet.

        Parameters
        ----------
            text: str
                Synopsis of a film.

        Return
        ------
            numpy.array:
                Data with synopsis cleaned.

        """
        return " ".join(word if word.isalpha()
                        else " " for word in text.split())

    def preprocessing_text_data(self, data):
        """Preprocessing text data.

        Take into account all preprocessing techniques and apply
        everyone to a synopsis column.

        Parameters
        ----------
            data: numpy.array
                Data from uploaded csv.

        Return
        ------
            numpy.array:
                Data with synopsis cleaned.

        """
        # Lowercase text.
        data["synopsis"] = data["synopsis"].apply(lambda syn: syn.lower())
        # Remove punctuation.
        data["synopsis"] = data["synopsis"].apply(
            lambda syn: syn.translate(str.maketrans(self.punctuation)))
        # Keep letters only.
        data["synopsis"] = data["synopsis"].apply(
            lambda syn: self.is_alpha(syn))
        # Lemmatize text.
        data["synopsis"] = data["synopsis"].apply(
            lambda syn: self._lemmatization(syn, self.lemmatizer,
                                            self.pos_map_lemma))

        return data


class TrainNode(BaseNode):
    """Training node for API train post."""

    def __init__(self, string_data):
        """Initialize the instance TrainNode."""
        super().__init__()
        self.string_data = string_data
        self.columns = ["movie_id", "year", "synopsis", "genres"]
        self.clf = None
        self.count_vect = CountVectorizer(stop_words="english")
        self.one_hot = OneHotEncoder()

    def _train(self, data):
        """Train process.

        Parameters
        ----------
            data: numpy.array
                Data from uploaded csv.

        Return
        ------
            None

        """
        # Tokenize target.
        y_train = [genres.split() for genres in data["genres"]]
        y_train = self.multilabel_bin.fit_transform(y_train)

        x_synopsis = self.count_vect.fit_transform(data["synopsis"])
        x_year = self.one_hot.fit_transform(data["year"].values.reshape(-1, 1))
        x_train = hstack([x_synopsis, x_year])

        self.clf = OneVsRestClassifier(ComplementNB())

        self.clf.fit(x_train, y_train)

    def run(self):
        """Execute the training process."""
        # Read data.
        data = self._read()

        # Preprocessing summary.
        data = self.preprocessing_text_data(data)

        # Train model.
        self._train(data)


class PredictNode(BaseNode):
    """Training node for API train post."""

    y_pred_gnr: List[str]

    def __init__(self, string_data, parameters):
        """Initialize the instance PredictNode."""
        super().__init__()
        self.string_data = string_data
        self.columns = ["movie_id", "year", "synopsis"]
        self.count_vect = parameters[2]
        self.one_hot = parameters[3]
        self.clf = parameters[0]
        self.classes = parameters[1]
        self.y_pred_gnr = list()

    def _predict(self, data):
        """Prediction process.

        Use the top5 high probabilities classes to save into
        a submission.csv.

        Parameters
        ----------
            data: numpy.array
                Data from uploaded csv.

        Return
        ------
            None

        """
        x_synopsis = self.count_vect.transform(data["synopsis"])
        x_year = self.one_hot.transform(data["year"].values.reshape(-1, 1))
        y_pred = self.clf.predict_proba(hstack([x_synopsis, x_year]))

        for id_mov in range(len(y_pred)):
            top5 = sorted(range(len(y_pred[id_mov])),
                          key=y_pred[id_mov].__getitem__)[-5:][::-1]
            self.y_pred_gnr.append(" ".join([self.classes[t5] for t5 in top5]))

        df = DataFrame({"movie_id": data["movie_id"],
                        "predicted_genres": self.y_pred_gnr})
        df.to_csv("submission.csv", index=False)

    def run(self):
        """Execute the prediction process."""
        # Read data.
        data = self._read()

        # Preprocessing summary.
        data = self.preprocessing_text_data(data)

        # Predict results.
        self._predict(data)
