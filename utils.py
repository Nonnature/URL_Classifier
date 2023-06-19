import os
import random
import numpy as np
import pandas as pd
import wordninja

# One-Hot Encoder
from sklearn.preprocessing import OneHotEncoder

# Text processing methods with TF
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Default Settings
VOCAB_SIZE = 10000  # Vocab size of tokenizer
MAX_LEN = 20        # Max length for padding


def preprocess_data(
        X, y=None,
        tokenizer=None, vocab_size=VOCAB_SIZE, max_len=MAX_LEN,
        encoder=None, categories=None,
        return_processors=False,

):
    """ Process URLs and Categorical Labels
    Steps:
    1. Process URLs
      1.1. Tokenization
      1.2. Zero-padding
    2. Process Labels
      2.1. One-hot encoding

    Arguments:
    - X (np.ndarray): Input URLs (str), shape = (n_urls, 1)
    - y (np.ndarray): Target categories (str), shape = (n_urls, 1)
    - tokenizer (keras_preprocessing.text.Tokenizer): Tokenizer object for 
         tokenization of URLs
    - vocab_size (int): Size of vocabulary to use by tokenizer
    - max_len (int): Max length of tokens after (zero-)padding
    - encoder (sklearn.preprocessing.OneHotEncoder): Encoder object 
         for converting labels to one-hot encoding
    - return_processors (bool): If True, return tokenizer and encoder used in this 
        function
    """

    ##### Process URLs #####
    # Convert url string to word tokens
    # Processing included:
    # - Split joined words (e.g. aiapplication -> "ai", "application")
    # - Remove special characters (e.g. ai-application -> "ai", "application")
    X = [wordninja.split(str(x)) for x in X]

    # Convert list of tokens into one long string with tokens separated by spaces " ".
    # e.g. ["t1", "t2", "t3"] ==> "t1 t2 t3".
    # This conversion is performed so that we can use tensorflow.keras.preprocessing.text.Tokenizer
    # to convert space-separated strings into list of integer tokens easily.
    X = [" ".join(tokens) for tokens in X]

    # Convert word tokens to integer tokens
    # If tokenizer is not provided, construct a new tokenizer and fit data to it
    if tokenizer is None:
        tokenizer = Tokenizer(oov_token=True, num_words=VOCAB_SIZE)
        tokenizer.fit_on_texts(X)
    # Transform texts to index sequences
    X = tokenizer.texts_to_sequences(X)

    # Zero-padding
    X = pad_sequences(X, maxlen=max_len, padding='post')

    # Return if no label to process
    if y is None:
        if not return_processors:
            # Return processed data only
            return X
        else:
            # Return processed data and the tokenizer & encoder used
            return X, tokenizer, encoder

    ##### Process Labels #####

    # Use one-hot encoding to convert string classes to one-hot encoding vectors.

    # If encoder not provided, construct a new encoder
    if encoder is None:
        # Construct new encoder
        encoder = OneHotEncoder(handle_unknown='ignore')

        if categories is not None:
            # If categories to use is specified, use them to fit the encoder
            encoder.fit(np.asarray(categories).reshape(-1, 1))
        else:
            # else, fit encoder using provided labels
            encoder.fit(np.asarray(y).reshape(-1, 1))

    # Convert categories into one-hot vectors
    y = encoder.transform(y.reshape(-1, 1)).toarray()

    if not return_processors:
        # Return processed data only
        return X, y
    else:
        # Return processed data and the tokenizer & encoder used
        return X, y, tokenizer, encoder


def predict_url(url, model, tokenizer, encoder, class_names=None, return_logits=False):
    """
    Preprocess and classify URL string(s).
    Steps:
    1. Preprocess URL string(s) and convert URL(s) to model inputs
    2. Feed input(s) to model
    3. Convert model outputs to integer/string class(es)

    Arguments:
    - url (str or list of str): URL string(s) to use for prediction
    - model (tf.keras.Sequential): URL classification model
    - tokenizer (keras_preprocessing.text.Tokenizer): Tokenizer object for 
         tokenization of URLs
    - encoder (sklearn.preprocessing.OneHotEncoder): Encoder object 
         for converting labels to one-hot encoding
    - class_names (list of str): class names of prediction targets
    - return_logits (bool): If true, the model outputs are returned immediately

    """
    # If only one url (i.e. string) is provided, convert variable url
    # to a list of just one string (required by preprocess_data(...))
    if isinstance(url, str):
        url = [url]

    # Process URL
    x = preprocess_data(url, tokenizer=tokenizer, encoder=encoder)

    # Feed input to model
    outs = model.predict(x)

    # Return model outputs if it is required
    if return_logits:
        return outs

    # Get the predicted class
    y = np.argmax(outs, axis=1)

    # If class names are provided, convert the integer class to the
    # category name.
    if class_names is not None:
        y = np.asarray(class_names)[y]

    return y
