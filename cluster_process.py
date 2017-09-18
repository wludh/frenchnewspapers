import os
import matplotlib.pyplot as plt
import codecs
import re
import argparse
import main as french_main
import shutil
import sys
import pandas as pd

from matplotlib.font_manager import FontProperties
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import namedtuple

# CORPUS = 'processed'
MetaData = namedtuple('MetaData',
                      ['publication', 'month', 'day', 'year'])
GenreMetaData = namedtuple('GenreMetaData',
                           ['title', 'genre'])
ParsedMetaData = namedtuple('ParsedMetaData',
                            ['journal_key', 'date_key'])
ParsedGenreMetaData = namedtuple('ParsedGenreMetaData',
                                 ['title', 'genre'])


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
                        help="""output directory to print
                         the processed materials too.""")
    parser.add_argument('-r', '--results', dest='results_folder',
                        action='store',
                        default='clustering_results',
                        help='The destination for clustering results.')

    parser.add_argument('-gc', '--genre_corpus', dest='genre_corpus',
                        action='store', default='genre_corpus',
                        help="""the folder containing a corpus of
                         other texts tagged for genre.""")
    parser.add_argument('-g', '--genre_compare', dest='genre_compare',
                        action='store', default="True",
                        help="""true/false - whether you are
                         comparing against a genre corpus.""")

    return parser.parse_args(argv)

ARGS = parse_args()
print(ARGS.genre_compare)


class ProcCorpus:
    """the object for the processed text"""
    def __init__(self):
        args = ARGS
        print(args.genre_compare)
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
 #       if args.genre_compare == "True" and os.path.exists('genre_corpus'):
        genre_corpus = french_main.GenreCorpus()
#        elif args.genre_compare == "True":
#            raise ValueError("""Error: specified genre clustering but
#                 genre_corpus/ directory does not exist.""")
#        else:
#            pass

#These lines commented out because of error being raised
        corpus.group_articles_by_publication()
        for text in corpus.texts:
            self.filter_tags(processed_dir, tag_filter, text)

        if args.genre_compare == "True":
            for text in genre_corpus.texts:
                self.filter_tags(processed_dir, tag_filter,
                                 text, genre_text=True)

    def filter_text(self, processed_dir, tag_filter, text, genre_text=False):
        """filters a single text based on part of speech"""
        to_write = []
        # your if statement here is a bit wonky
        for x in text.tree_tagged_tokens:
            if tag_filter == 'VER':
                if x.pos[0:3] == tag_filter:
                    to_write.append(x.lemma)
            elif tag_filter == 'PRO':
                if x.pos[0:3] == tag_filter:
                    to_write.append(x.lemma)
            # if working on punctuation only, sub
            elif tag_filter == '!?' or tag_filter == '?!':
                mappings = {'!': 'exclamation', '?': 'question'}
                if x.lemma in mappings:
                    to_write.append(mappings[x.lemma])
            elif (tag_filter == 'PUN' or tag_filter == 'SENT') and \
                 (x.pos == 'PUN' or x.pos == 'SENT'):
                mappings = {',': 'comma', '(': 'open_paren',
                            ':': 'colon', '\'': 'apostrophe', '-': 'dash',
                            ';': 'semi-colon', '/': 'forward_slash',
                            '!': 'exclamation', '?': 'question',
                            '.': 'period'}
                if x.lemma in mappings:
                    to_write.append(mappings[x.lemma])
                else:
                    to_write.append(x.lemma)
            elif x.pos == tag_filter:
                to_write.append(x.lemma)
            else:
                pass

        return to_write

    def filter_tags(self, processed_dir, tag_filter, text, genre_text=False):
        """filters parts of speech"""
        print('processing ' + text.filename)
        to_write = self.filter_text(processed_dir, tag_filter, text)
        chunk_size = 2000
        if genre_text:
            i = 1
            total_tokens = len(to_write)

            while (i * chunk_size) < total_tokens:
                start = (i - 1) * chunk_size
                end = i * chunk_size
                with open(processed_dir + '/' + text.filename + '_chunk_' +
                          str(i) + '.txt', 'w') as current_text:
                    current_text.write(' '.join(to_write[start:end]))
                i += 1

            with open(processed_dir + '/' + text.filename + '_chunk_' +
                      str(i) + '.txt', 'w') as current_text:
                current_text.write(' '.join(to_write[(i - 1) *
                                   chunk_size:total_tokens]))
        else:
            with open(processed_dir + '/' + text.filename + '.txt',
                      'w') as current_text:
                current_text.write(' '.join(to_write))

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

 #   def produce_tfidfs(self):
#        tfidf = TfidfVectorizer()
#        tfs = tfidf.fit_transform(self.texts.values())
#        return tfs
# Added the above lines to the graph_clusters function to make scope a little
#easier to deal with
    def graph_clusters(self, args=ARGS):
        ## Added lines:
        tfidf = TfidfVectorizer()
        tfs = tfidf.fit_transform(self.texts.values())
        numclusters = 5
        ## end of added
        labels = self.parse_names()

        
 #       tfs = self.produce_tfidfs()
#Deleted the above line after adding produce_tfidfs lines
 
        # will try and slot them into the nclusters
        try:
            fitted = KMeans(n_clusters=numclusters).fit(tfs)
        except ValueError:
            fitted = KMeans(n_clusters=2).fit(tfs)
        classes = fitted.predict(tfs)

##Print the data regarding words found in each cluster.

        centroids = fitted.cluster_centers_

        print (labels)


        labelsnoarrow = [x[7:] for x in labels if x[0] == ' ']
        labelsnoletter = [x[2:] for x in labels if x[0] != ' ']
        newlabels = labelsnoarrow + labelsnoletter
        thedict = {'Cluster':classes, 'Journal and Date':newlabels}

        usethis = pd.DataFrame(thedict)

        
        order_centroids = fitted.cluster_centers_.argsort()[:, ::-1]
        terms = tfidf.get_feature_names()
        for i in range(numclusters):
            print("Cluster %d words:" % i, end='')
    
            for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
                print(" %s" % terms[ind], end = "")
                print() #add whitespace
            current = usethis["Cluster"] == i
            showthis = usethis[current]
            print (showthis)
        print ()
        
        try:
            sklearn = PCA(n_components=5)
        except ValueError:
            sklearn = PCA(n_components=2)
        try:
            sklearn_transf = sklearn.fit_transform(tfs.toarray())
        except:
            sklearn_transf = PCA(n_components=2).fit_transform(tfs.toarray())
        plt.scatter(sklearn_transf[:, 0],
                    sklearn_transf[:, 1], c=classes, s=75)
        for i in range(len(classes)):
            plt.text(sklearn_transf[i, 0], sklearn_transf[i, 1], s=labels[i], size = 'xx-small')
        # plt.show()
        if not os.path.exists(ARGS.results_folder):
            os.makedirs(ARGS.results_folder)
        if ARGS.genre_compare == "True":
            plt.savefig(ARGS.results_folder + '/' + ARGS.tag_filter +
                        '_with_genre_corpus' + '.png')
        else:
            plt.savefig(ARGS.results_folder + '/' + ARGS.tag_filter + '.png')

    def parse_names(self):
        keys = []
        for key in self.texts.keys():

            print (key)

            clean_key = re.sub(r'processed\/|\.txt|_chunk_[0-9]+\.txt',
                               '', key)
            split_key = re.split(r'_', clean_key)
            if split_key[1] in ['sex', 'crime', 'corruption','morality']:
                split_key = re.split(r'_', clean_key)
                parsed_key = GenreMetaData(split_key[0], split_key[1])
                pub_mapping_dict = {'sex': 'x', 'crime': 'y',
                                    'corruption': 'z', 'morality': 'm', 'prostitution':'p'}
                result = pub_mapping_dict[parsed_key.genre] + ' ' + split_key[0]
                # ['journal_key', 'date_key']
                keys.append(result)
            else:
                split_key = re.split(r'_', clean_key)
                parsed_key = MetaData(' '.join(split_key[:-3]), split_key[-3],
                                      split_key[-2], split_key[-1])
                pub_mapping_dict = {'croix': 'c', 'figaro': 'f',
                                    'humanite': 'h', 'intransigeant': 'i',
                                    'journal': 'j',
                                    'matin': 'm', 'petit journal': 'pj',
                                    'petit parisien': 'pp', 'radical': 'r',
                                    'temps': 't'}
                datekey = "-" + parsed_key.month + " " + parsed_key.day + "," +  parsed_key.year
                result = ParsedMetaData(pub_mapping_dict[parsed_key.publication], datekey) 
                # 1: June 1908
                # 2: Nov. 1908
                # 3: and Nov 1909
#                if parsed_key.month == 'june' and parsed_key.year == '1908':
#                    result = ParsedMetaData(pub_mapping_dict[
#                        parsed_key.publication], '1')
#                elif parsed_key.year == '1908' and \
#                        parsed_key.month == 'november':
#                    result = ParsedMetaData(pub_mapping_dict[
#                        parsed_key.publication], '2')
#                elif parsed_key.year == '1909':
#                    result = ParsedMetaData(pub_mapping_dict[
#                        parsed_key.publication], '3')
#                else:
#                    print(keys)
#                    print(split_key)
#                    print(parsed_key)
#                    result = 'SOMETHING HAS GONE WRONG'
                # ['journal_key', 'date_key']
                keys.append("   <-- " + result.journal_key + result.date_key)
        return keys

    def parse_shapes(self):
        pass


def main():
    """Main function to be called when the script is called"""
    corpus = ProcCorpus()
    corpus.graph_clusters()

if __name__ == '__main__':
    main()
