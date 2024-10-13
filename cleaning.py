import re
import string
import emoji
from bs4 import BeautifulSoup


def remove_html_tags(text):
    """
    Remove all HTML tags from the text using BeautifulSoup.
    """
    soup = BeautifulSoup(text, "html.parser")

    # Get the plain text without any HTML tags
    return soup.get_text()


def to_lowercase(text):
    """
    Convert text to lowercase.
    """
    return text.lower()


def replace_emojis(text):
    """
    Replace emojis with their textual description.
    """
    return emoji.demojize(text)


def decontracted(text):
    """
    Expand common contractions in a text.
    """
    # specific
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"ain\'t", "are not", text)
    text = re.sub(r"shan\'t", "shall not", text)
    text = re.sub(r"ma\'am", "maam", text)
    text = re.sub(r"y\'all", "you all", text)

    # general
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    return text


def remove_mentions(text):
    """
    Remove all mentions (@username) from the tweet.
    """
    return re.sub(r"@\w+", "", text)


def remove_retweets(text):
    """
    Remove 'RT' (retweets) from the tweet.
    """
    return re.sub(r"\bRT\b", "", text)


def remove_urls(text):
    """
    Remove all URLs from the tweet.
    """
    return re.sub(r"http\S+", "", text)


def remove_nonstandard_characters(text):
    """
    Remove non-standard characters like â€™ from the tweet.
    """
    text = text.replace("Ã¢â‚¬â„¢", "'")  # Common encoding issue
    # Remove any other non-ASCII characters
    return re.sub(r"[^\x00-\x7F]+", "", text)


def remove_punctuations(text):
    """
    Replace punctuations with a space.
    """
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    return text.translate(translator)


def clean_hashtag(text):
    """Remove "#" symbol and split camel case only in hashtags."""
    cleaned_text = re.sub(
        r"#(\w+)", lambda m: re.sub(r"([a-z])([A-Z])", r"\1 \2", m.group(1)), text
    )
    # # capitalize the first letter of each word
    # cleaned_text = " ".join(word.capitalize() for word in cleaned_text.split())
    return cleaned_text


def remove_multiple_whitespaces(text, replacement_text=" "):
    """Remove multiple whitespaces from string."""
    pattern = re.compile(r"\s{2,}")
    return pattern.sub(replacement_text, text)


def replace_newline_with_space(text):
    """Replace newline characters (\n) with spaces."""
    return text.replace("\n", " ")


def clean(text):
    """
    Clean a text by applying all cleaning steps.
    """
    text = remove_html_tags(text)
    text = replace_emojis(text)
    text = to_lowercase(text)
    text = decontracted(text)
    text = remove_mentions(text)
    text = remove_retweets(text)
    text = remove_urls(text)
    text = remove_nonstandard_characters(text)
    text = clean_hashtag(text)
    text = remove_punctuations(text)
    text = remove_multiple_whitespaces(text)
    text = replace_newline_with_space(text)
    return " ".join(text.split())
