import re
import string

import nltk
from nltk.corpus import stopwords, wordnet


def remove_links(text):
    text = re.sub(r'http\S+', '', text) # remove http links
    text = re.sub(r'bit.ly/\S+', '', text) # remove bitly links
    text = re.sub(r'www\S+', '', text) # remove www links
    text = re.sub(r'.*\.com', '', text)
    return text


def removeEmoticons(text):
    """ Removes emoticons from text """
    text = re.sub(':\)|;\)|:-\)|\(-:|:-D|=D|:P|xD|X-p|\^\^|:-*|\^\.\^|\^\-\^|\^\_\^|\,-\)|\)-:|:\'\(|:\(|:-\(|:\S|T\.T|\.\_\.|:<|:-\S|:-<|\*\-\*|:O|=O|=\-O|O\.o|XO|O\_O|:-\@|=/|:/|X\-\(|>\.<|>=\(|D:', '', text)
    return text


def remove_symbols(text):
    """ Removes all symbols and keep alphanumerics """
    whitelist = []
    return [re.sub(r'([^a-zA-Z0-9\s]+?)',' ',word) for word in text if word not in whitelist]


def keep_alphanum(text):
    """ Keep Alphanumeric characters """
    return [word for word in text if word.isalnum()]


def remove_stopwords(text):
    """ Remove Stopwords """
    stop_list = stopwords.words('english')
    stop_list += string.punctuation
    stop_list += [] #any other stop words
    return [word for word in text if word not in stop_list]


def remove_apostrophes(text):
    """ Remove words which have 's with a space """
    return [re.sub(r"'s", "",word) for word in text]


def replaceFromFile(text, path):
    """ Creates a dictionary with slangs or contractions and their equivalents and replaces them """
    with open(path, encoding="ISO-8859-1") as file:
        slang_map = dict(map(str.strip, line.partition('\t')[::2]) for line in file if line.strip())
    return [slang_map[word] if word in slang_map.keys() else word for word in text]


def nltk_tag_to_wordnet_tag(nltk_tag):
    """ function to convert nltk tag to wordnet tag """
    output = None
    if nltk_tag.startswith('J'):
        output = wordnet.ADJ
    elif nltk_tag.startswith('V'):
        output = wordnet.VERB
    elif nltk_tag.startswith('N'):
        output = wordnet.NOUN
    elif nltk_tag.startswith('R'):
        output = wordnet.ADV
    return output


def replaceElongated(word):
    """ Replaces an elongated word with its basic form, unless the word exists in the lexicon """
    repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
    repl = r'\1\2\3'
    if wordnet.synsets(word):
        return word
    repl_word = repeat_regexp.sub(repl, word)
    if repl_word != word:
        return replaceElongated(repl_word)
    else:
        return repl_word


def replaceElongatedText(text):
    return [replaceElongated(word) for word in text]


def remove_multispaces(text):
    """ Replace multiple spaces with only 1 space """
    return [re.sub(r' +', " ",word) for word in text]


def normalize_word(text):
    """ Own mapping function """
    replacement_dict = {"l'oreal": 'loreal', "l'oreals":'loreal','b/c': 'because','amazon.com':'amazon', \
                        'cake-y':'cakey', 'build-able':'buildable', 'wal-mart':'walmart', \
                        'q-tip':'cotton swab','l"oreal': 'loreal', 'l"oreals':'loreal', \
                        "a/c":"air conditioning", "co-workers":"colleague", "co-worker":"colleague", \
                        "y'all":"you all"}

    text = [replacement_dict[word] if word in replacement_dict.keys() else word for word in text]

    return text


def lemmatize_words(feedback):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    # Pos tagging
    nltk_tagged = nltk.pos_tag(feedback)

    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)

    # lemmatizing
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is not None and word not in stopwords.words('english'):
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
        else:
            lemmatized_sentence.append(lemmatizer.lemmatize(word))

    return (lemmatized_sentence)


def clean_text(texts, contractions_path, slangs_path):
    new_texts = [nltk.word_tokenize(str(text)) for text in texts]
    new_texts = [normalize_word((text)) for text in new_texts]
    new_texts = [' '.join(text) for text in new_texts]
    new_texts = [remove_links(text) for text in new_texts]
    new_texts = [removeEmoticons(text) for text in new_texts]
    new_texts = [nltk.word_tokenize(str(text)) for text in new_texts]
    new_texts = [replaceFromFile(text, slangs_path) for text in new_texts]
    new_texts = [replaceFromFile(text, contractions_path) for text in new_texts]
    new_texts = [normalize_word(text) for text in new_texts]
    new_texts = [remove_apostrophes(text) for text in new_texts]
    new_texts = [replaceElongatedText(text) for text in new_texts]
    new_texts = [keep_alphanum(text) for text in new_texts]
    new_texts = [lemmatize_words(text) for text in new_texts]
    new_texts = [remove_stopwords(text) for text in new_texts]
    new_texts = [remove_multispaces(text) for text in new_texts]
    new_texts = [normalize_word(text) for text in new_texts]
    new_texts = [' '.join(text) for text in new_texts]

    return new_texts