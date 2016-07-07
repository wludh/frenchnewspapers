#!/usr/bin/env python3

import nltk
from nltk import word_tokenize, FreqDist, PorterStemmer
from nltk.corpus import stopwords, names
from nltk.stem.snowball import SnowballStemmer
import codecs
import os
import re
import csv
import operator
import datetime
import dateutil.parser

# TODO: stemming. will need to follow the example
# TODO:
# below, where you have each text map onto an indexed stem.
# TODO: take those stems and then integrate that into everything below.

""" A text should have:
    filename
    date
    journal
    tokens
    stem index for each token"""


CORPUS = 'clean_ocr'
STOPWORD_LIST = []
LIST_OF_NAMES = []


class Corpus(object):
    """Takes in a list of text objects. and you can then
    run things on the set as a whole."""

    def __init__(self, corpus_dir=CORPUS):
        self.corpus_dir = corpus_dir
        self.stopwords = self.generate_stopwords()
        self.names_list = self.generate_names_list()
        self.texts = self.build_corpus()
        self.sort_articles_by_date()

    def build_corpus(self):
        """given a corpus directory, make indexed text objects from it"""
        texts = []
        for (root, _, files) in os.walk(self.corpus_dir):
            for fn in files:
                if fn[0] == '.':
                    pass
                else:
                    texts.append(IndexedText(os.path.join(root, fn)))
        return texts

    def generate_stopwords(self):
        global STOPWORD_LIST
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
        STOPWORD_LIST = set(nltk_stopwords + ranks_stopwords +
                            punctuation + extra_stopwords)
        return STOPWORD_LIST

    def generate_names_list(self):
        global LIST_OF_NAMES
        """puts together a list to query the corpus by"""
        nltk_names = names.words()
        # put custom name list here.
        extra_names = ['Alexandre', 'Steinheil']
        LIST_OF_NAMES = \
            [w.lower() for w in nltk_names] + [w.lower() for w in extra_names]
        return LIST_OF_NAMES

    def read_out(self):
        """given a series of articles, print out stats for them
        articles are given as a list of tuple pairs
        (filename, list of tokens)"""
        output = open('results.txt', 'w')
        for text in self.texts:
            output.write("===================\n")
            output.write(text.filename + '\n')
            output.write("Number of tokens: " +
                         str(text.length) + '\n')
            output.write("Most common tokens: " +
                         str(text.most_common()) + '\n')
            output.write("Punctuation Counts: " +
                         str(text.count_punctuation()) + '\n')
            output.write("Names: " + str(text.find_names()) + '\n')

    def sort_articles_by_date(self):
        """Takes the corpus and sorts them by date. Defaults to this method.
        Calling it again will resort things by date."""
        self.texts.sort(key=lambda x:
                        datetime.datetime.strptime(x.date, "%Y-%m-%d"))

    def group_articles_by_publication(self):
        """group articles by publication. call it to sort."""

        pub_index = {}
        new_array = []
        for text in self.texts:
            key = text.publication
            value = pub_index.get(key)
            if value is None:
                pub_index[key] = [text]
            else:
                pub_index[key] += [text]
        for key in pub_index.keys():
            pub_index[key].sort(key=lambda x:
                           datetime.datetime.strptime(x.date, "%Y-%m-%d"))
            new_array += pub_index[key]
        self.texts = new_array

    def single_token_by_date(self, token):
        """Takes the list of articles and the parameter to sort by.
        returns all the data needed for the CSV
        returns a hash with key values of {(year, month, day):
        (target token counts for
        this month, total tokens)}"""
        index = {}
        for text in self.texts:
            # for now, graph by date
            key = text.date
            column_values = index.get(key)
            if column_values is None:
                total_doc_tokens = text.length
                column_values = text.token_count(token)
                index[key] = (column_values, total_doc_tokens)
            else:
                index[key] = (index[key][0] + text.token_count(token), index[key][1] + text.length)
            index['year-month-day'] = token
        return index

    def dict_to_list(self, a_dict):
        """takes the result dict and prepares it for writing to csv"""
        rows = []

        for (date_key, values) in a_dict.items():
            try:
                tf_idf = values[0] / values[1]
                rows.append([date_key, tf_idf])
            except:
                # for the csv header, put it at the beginning of the list
                rows.insert(0, [date_key, values])
        # sorts things by date
        rows.sort(key=operator.itemgetter(0))
        # takes the last row and makes it first, since it gets shuffled to the back
        rows.insert(0, rows.pop())
        return rows

    def csv_dump(self, results_dict):
        """writes some information to a CSV for graphing in excel."""
        print(results_dict)
        results_list = self.dict_to_list(results_dict)

        with open('results.csv', 'w') as csv_file:
            csvwriter = csv.writer(csv_file, delimiter=',')
            for row in results_list:
                csvwriter.writerow(row)


class IndexedText(object):
    """Text object"""
    def __init__(self, filepath):
        self.filepath = filepath
        self.filename = re.sub(r'^.*/|_clean.txt|_Clean.txt', '', filepath).lower()
        self.text = self.read_text()
        self.tokens = [w.lower() for w in word_tokenize(self.text)]
        self.length = len(self.tokens)
        self.fd = FreqDist(self.tokens)
        self.date = self.parse_dates()
        self.year = dateutil.parser.parse(self.date).year
        self.month = dateutil.parser.parse(self.date).month
        self.day = dateutil.parser.parse(self.date).day
        self.publication = self.parse_publication()
        self.stemmer = SnowballStemmer('french')
        self.index = nltk.Index((self.stem(word), i)
                         for (i, word) in enumerate(self.tokens))
        self.tokens_without_stopwords = self.remove_stopwords()

    def read_text(self):
        """given a filename read in the text."""
        with codecs.open(self.filepath, 'r', 'utf8') as f:
            print(self.filename)
            return f.read()

    def parse_publication(self):
        """parse the filename for the publication."""
        date_pattern = r'[jJ]anuary[a-zA-Z0-9_]*|[fF]ebruary[a-zA-Z0-9_]*|[mM]arch[a-zA-Z0-9_]*|[aA]pril[a-zA-Z0-9_]*|[mM]ay[a-zA-Z0-9_]*|[jJ]une[a-zA-Z0-9_]*|[jJ]uly[a-zA-Z0-9_]*|[aA]ugust[a-zA-Z0-9_]*|[sS]eptember[a-zA-Z0-9_]*|[oO]ctober[a-zA-Z0-9_]*|[nN]ovember[a-zA-Z0-9_]*|[dD]ecember[a-zA-Z0-9_]*'
        # strip out date from filename and pop off the trailing underscore.
        return re.sub(date_pattern, '', self.filename)[:-1]

    def parse_dates(self):
        """parse the filename for the dates."""
        date_pattern = r'[jJ]anuary[a-zA-Z0-9_]*|[fF]ebruary[a-zA-Z0-9_]*|[mM]arch[a-zA-Z0-9_]*|[aA]pril[a-zA-Z0-9_]*|[mM]ay[a-zA-Z0-9_]*|[jJ]une[a-zA-Z0-9_]*|[jJ]uly[a-zA-Z0-9_]*|[aA]ugust[a-zA-Z0-9_]*|[sS]eptember[a-zA-Z0-9_]*|[oO]ctober[a-zA-Z0-9_]*|[nN]ovember[a-zA-Z0-9_]*|[dD]ecember[a-zA-Z0-9_]*'
        date = re.findall(date_pattern, self.filename)
        # replaces the underscores with dashes
        date = dateutil.parser.parse(re.sub(r'_', '-', date[0]))
        return ('%s-%s-%s' % (date.year, date.month, date.day))

    def stem(self, word):
        return self.stemmer.stem(word).lower()

    def concordance(self, word, width=40):
        """given a token, produce a word in context view of it."""
        key = self.stem(word)
        # words of context
        wc = int(width / 4)
        for i in self.index[key]:
            lcontext = ' '.join(self.tokens[i - wc:i])
            rcontext = ' '.join(self.tokens[i:i + wc])
            ldisplay = '{:>{width}}'.format(lcontext[-width:], width=width)
            rdisplay = '{:{width}}'.format(rcontext[:width], width=width)
            print(ldisplay, rdisplay)

    def remove_stopwords(self):
        global STOPWORD_LIST
        """takes a list of tokens and strips out the stopwords"""
        return [w for w in self.tokens if w.lower() not in STOPWORD_LIST]

    def find_names(self):
        """creates a frequency distribution of the
        most common names in the texts"""
        names_list = LIST_OF_NAMES
        name_tokens = [w for w in self.tokens if w in names_list]
        fd = FreqDist(name_tokens)
        return fd.most_common(50)

    def count_punctuation(self):
        """Gives a count of the given punctuation marks for each text"""
        fd = FreqDist(self.tokens)
        punctuation_marks = ['»', '«', ',', '-', '.', '!',
                             "\"", ':', ';', '?', '...', '\'']
        results = []
        for mark in punctuation_marks:
            count = str(fd[mark])
            results.append("%(mark)s, %(count)s" % locals())
        return(results)

    def most_common(self):
        """takes a series of tokens and returns most common 50 words."""
        fd = FreqDist(self.tokens_without_stopwords)
        return fd.most_common(50)

    def token_count(self, token):
        """takes a token and returns the counts for it in the text."""
        return self.fd[token]


def main():
    """Main function to be called when the script is called"""
    corpus = Corpus()
    corpus.read_out()
    # corpus.csv_dump(corpus.single_token_by_date('crime'))


if __name__ == '__main__':
    main()
