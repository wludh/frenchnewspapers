import os
import matplotlib.pyplot as plt
import codecs
import re
import argparse
import main as french_main
import shutil
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import namedtuple

# CORPUS = 'processed'
MetaData = namedtuple('MetaData',
                            ['publication', 'month', 'day', 'year'])
ParsedMetaData = namedtuple('ParsedMetaData',
                            ['journal_key', 'date_key'])


def parse_args(argv=None):
    """This parses the command line."""
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('-t', '--tag', dest='tag_filter', action='store',
                        default='ADJ',
                        help='The tag to filter on')
    parser.add_argument('-o', '--output', dest='processed_dir',
                        action='store',
                        default='processed',
                        help='output directory to print the processed materials too.')
    parser.add_argument('-r', '--results', dest='results_folder',
                        action='store',
                        default='clustering_results',
                        help='The destination for clustering results.')

    return parser.parse_args(argv)

ARGS = parse_args()


class ProcCorpus:
    """the object for the processed text"""
    def __init__(self):
        args = ARGS
        self.process_corpus(args)
        self.filenames = self.generate_filenames(args)
        self.texts = self.read_texts()

    def process_corpus(self, args):
        processed_dir = args.processed_dir
        tag_filter = args.tag_filter

        if os.path.exists(processed_dir):
            shutil.rmtree(processed_dir)

        os.makedirs(processed_dir)

        corpus = french_main.Corpus()
        print([text.filename for text in corpus.texts])
        corpus.group_articles_by_publication()
        print([text.filename for text in corpus.texts])
        for text in corpus.texts:
            to_write = []
            with open(processed_dir + '/' + text.filename + '.txt', 'w') as current_text:
                for x in text.tree_tagged_tokens:
                    # something special for verbs, we want
                    # to collapse all verb tenses together.
                    if tag_filter == 'VER':
                        if x.pos[0:3] == tag_filter:
                            to_write.append(x.lemma)
                    elif x.pos == tag_filter:
                        to_write.append(x.lemma)
                    else:
                        pass
                current_text.write(' '.join(to_write))
        #for text in corpus.texts:
        #    to_write = []
        #    with open(processed_dir + '/' + text.filename + '.txt', 'w') as current_text:
        #         for x in text.tree_tagged_tokens:
        #             if x.pos == tag_filter:
        #                 to_write.append(x.lemma)
        #         current_text.write(' '.join(to_write))

    def generate_filenames(self, args):
        filenames = []
        for (root, _, files) in os.walk(args.processed_dir):
            for fn in files:
                if fn[0] == '.' or fn[-4:] == '.png':
                    pass
                else:
                    print(fn)
                    filenames.append(os.path.join(root, fn))
        return filenames

    def read_texts(self):
        texts = {}
        for fn in self.filenames:
            with codecs.open(fn, 'r', 'utf8') as f:
                texts[fn] = f.read()
        return texts

    def produce_tfidfs(self):
        tfidf = TfidfVectorizer()
        tfs = tfidf.fit_transform(self.texts.values())
        return tfs

    def graph_clusters(self, args=ARGS):
        labels = self.parse_names()
        tfs = self.produce_tfidfs()
        # will try and slot them into the nclusters
        fitted = KMeans(n_clusters=5).fit(tfs)
        classes = fitted.predict(tfs)
        sklearn = PCA(n_components=5)
        sklearn_transf = sklearn.fit_transform(tfs.toarray())
        plt.scatter(sklearn_transf[:, 0],
                    sklearn_transf[:, 1], c=classes, s=35)
        for i in range(len(classes)):
            plt.text(sklearn_transf[i, 0], sklearn_transf[i, 1], s=labels[i])
        # plt.show()
        plt.savefig(ARGS.results_folder + '/' + ARGS.tag_filter + '.png')

    def parse_names(self):
        keys = []
        for key in self.texts.keys():
            clean_key = re.sub(r'processed\/|\.txt', '', key)
            split_key = re.split(r'_', clean_key)
            parsed_key = MetaData(' '.join(split_key[:-3]), split_key[-3], split_key[-2], split_key[-1])
            pub_mapping_dict = {'croix': 'c', 'figaro': 'f', 'humanite': 'h', 'intransigeant': 'i', 'journal': 'j',
                'matin': 'm', 'petit journal': 'pj',
                'petit parisien': 'pp', 'radical': 'r',
                'temps': 't'}
            # 1: June 1908
            # 2: Nov. 1908
            # 3: and Nov 1909
            if parsed_key.month == 'june' and parsed_key.year == '1908':
                result = ParsedMetaData(pub_mapping_dict[
                    parsed_key.publication], '1')
            elif parsed_key.year == '1908' and parsed_key.month == 'november':
                result = ParsedMetaData(pub_mapping_dict[
                    parsed_key.publication], '2')
            elif parsed_key.year == '1909':
                result = ParsedMetaData(pub_mapping_dict[
                    parsed_key.publication], '3')
            else:
                result = 'SOMETHING HAS GONE WRONG'
            # ['journal_key', 'date_key']
            keys.append(result.journal_key + result.date_key)
        return keys

    def parse_shapes(self):
        pass


def main():
    """Main function to be called when the script is called"""
    corpus = ProcCorpus()
    corpus.graph_clusters()

if __name__ == '__main__':
    main()
