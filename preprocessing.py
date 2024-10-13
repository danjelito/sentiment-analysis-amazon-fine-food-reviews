from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer

stops = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
tokenizer = TreebankWordTokenizer()


def lower_doc(doc):
    """
    Convert the input document to lowercase.
    """
    return doc.lower()


def tokenize_doc(doc):
    """
    Tokenize the input document using NLTK word_tokenize.
    """
    return tokenizer.tokenize(doc)


def lemmatize_doc(doc):
    """
    Lemmatize each token in the input document using WordNetLemmatizer.
    """
    return [lemmatizer.lemmatize(token) for token in doc]


def remove_stopwords(doc):
    """
    Remove stopwords from the input document.
    """
    return [token for token in doc if token not in stops]


def remove_digit(doc):
    """
    Remove tokens that consist entirely of digits.
    """
    return [
        token
        for token in doc
        if not (
            token.isdigit()
            or token.replace(".", "").isnumeric()
            or token.replace(",", "").isnumeric()
        )
    ]


def strip_doc(doc):
    """
    Strip leading and trailing whitespace from each token in the input document.
    """
    return [token.strip() for token in doc]


def remove_empty_strings(doc):
    """
    Remove empty strings from the list of tokens.
    """
    return [token for token in doc if token != "''"]


def preprocess(doc):
    """
    Preprocess a text by applying all preprocessing steps.
    """
    doc = lower_doc(doc)
    doc = tokenize_doc(doc)
    doc = remove_stopwords(doc)
    doc = remove_digit(doc)
    doc = lemmatize_doc(doc)
    doc = strip_doc(doc)
    doc = remove_empty_strings(doc)
    doc = [token for token in doc if len(token) > 1]
    # return " ".join(doc)
    return doc
