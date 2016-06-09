#!/usr/bin/env python3

from nltk import word_tokenize, FreqDist
from nltk.corpus import stopwords, names
import codecs
import os
import drive
import re

CORPUS = 'clean_ocr'


def generate_stopwords():
    """generates stopwords for use in tokenizing"""
    nltk_stopwords = stopwords.words('french')
    # stopwords from http://www.ranks.nl/stopwords/french
    ranks_stopwords = []
    with codecs.open('french_stopwords.txt', 'r', 'utf8') as f:
        ranks_stopwords = [x.strip('\n') for x in f]
    # put custom stopword list here. Could also
    # read in as a csv file if that's easier.
    extra_stopwords = []
    punctuation = ['»', '«', ',', '-', '.', '!',
                   "\"", '\'' ':', ';', '?', '...']
    return set(nltk_stopwords + ranks_stopwords + punctuation + extra_stopwords)


def names_list():
    """puts together a list to query the corpus by"""
    nltk_names = names.words()
    # put custom name list here.
    extra_names = ['Alexandre', 'Steinheil']
    return [w.lower() for w in nltk_names] + [w.lower() for w in extra_names]


LIST_OF_NAMES = names_list()


def all_file_names(dirname=CORPUS):
    """Reads in the files"""
    for (root, _, files) in os.walk(dirname):
        for fn in files:
            yield os.path.join(root, fn)


def strip_off_file_path(filename):
    """Takes off extraneous information from the file
     path and returns the filename alone"""
    filename = re.sub(r'^.*/|_clean.txt', '', filename)
    return filename.lower()


def read_all_texts(filenames):
    """Given a list of filenames, read each of them"""
    for f in filenames:
        yield (strip_off_file_path(f), read_text(f))


def get_articles_metadata(list_of_articles, debug=False):
    """takes articles in form of [filename, [tokens]],
    goes out to google drive and gives it the necessary
    date and time information."""
    metadata = drive.get_article_metadata()
    new_list_of_articles = []
    for article in list_of_articles:
        for row in metadata:
            if row['filename'] == article[0]:
                new_list_of_articles.append({'file_name': article[0],
                                             'journal': row['newspaper name'],
                                             'date': row['date'],
                                             'tokens': tokenize_text(article[1])})
            elif debug:
                print("********ERROR: FILENAME AND DATE MISMATCH ********")
                print(row['filename'] + '   ≠   ' + article[0])
                print("*************")
            else:
                pass
    return new_list_of_articles


def read_text(filename):
    """given a filename read in the text."""
    with codecs.open(filename, 'r', 'utf8') as f:
        return f.read()


def tokenize_text(file):
    """Tokenizes the text as is. Strips out page breaks and
    derniere sections but leaves them as part of the larger text."""
    return [w.lower() for w in word_tokenize(file)]


def remove_stopwords(tokens):
    """takes a list of tokens and strips out the stopwords"""
    stopword_list = generate_stopwords()
    return [w for w in tokens if w.lower() not in stopword_list]


def calc_article_length(tokens):
    """given an article's tokens calculates its length"""
    return len(tokens)


def count_punctuation(text):
    """Gives a count of the given punctuation marks for each text"""
    fd = FreqDist(text)
    punctuation_marks = ['»', '«', ',', '-', '.', '!',
                         "\"", ':', ';', '?', '...', '\'']
    for mark in punctuation_marks:
        count = str(fd[mark])
        yield "%(mark)s, %(count)s" % locals()


def most_common(text):
    """takes a series of tokens and returns most common 50 words."""
    fd = FreqDist(remove_stopwords(text))
    return fd.most_common(50)


def find_names(text, list_of_names=LIST_OF_NAMES):
    """creates a frequency distribution of the
    most common names in the texts"""
    name_tokens = [w for w in text if w in list_of_names]
    fd = FreqDist(name_tokens)
    return fd.most_common(50)


def read_out(articles):
    """given a series of articles, print out stats for them
    articles are given as a list of tuple pairs (filename, list of tokens)"""
    output = open('results.txt', 'w')
    for article in articles:
        output.write("===================\n")
        output.write(article['file_name'] + '\n')
        output.write("Number of tokens: " +
                     str(calc_article_length(article['tokens'])) + '\n')
        output.write("Most common tokens: " +
                     str(most_common(article['tokens'])) + '\n')
        output.write("Punctuation Counts: " +
                     str([mark for mark
                          in count_punctuation(article['tokens'])]) + '\n')
        output.write("Names: " + str(find_names(article['tokens'])) + '\n')


def prepare_all_texts(corpus=CORPUS):
    """Takes all files from filenames to dict. runs everything in between"""
    # reads in all filenames from the corpus directory.
    file_names = list(all_file_names(corpus))
    # reads in data of texts
    texts = list(read_all_texts(file_names))
    # reads in article metadata for texts
    texts_with_metadata = get_articles_metadata(texts)
    return texts_with_metadata


def main():
    """Main function to be called when the script is called"""
    # print(texts_with_metadata[0])
    text_data = prepare_all_texts()
    # print(text_data[0]['file_name'])
    read_out(text_data)

if __name__ == '__main__':
    main()
