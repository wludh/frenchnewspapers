#!/usr/bin/env python3

import nltk
from nltk import FreqDist
import treetaggerwrapper
import nltk.data
from nltk.corpus import stopwords, names
from nltk.stem.snowball import SnowballStemmer
import codecs
import os
import re
import csv
import operator
import datetime
import dateutil.parser
import matplotlib.pyplot as plt
plt.rcdefaults()
import numpy as np
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator
import gensim
from gensim import corpora



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



plt.rcdefaults()
fig, ax = plt.subplots()



class Corpus(object):
    """Takes in a list of text objects. and you can then
    run things on the set as a whole."""

    def __init__(self, corpus_dir=CORPUS):
        self.corpus_dir = corpus_dir
        self.stopwords = self.generate_stopwords()
        self.names_list = self.generate_names_list()
        self.texts = self.build_corpus()
        self.sort_articles_by_date()
        self.publications = [text.publication for text in self.texts]

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

    def preprocess_for_topic_modeling(self):
        """stem the corpus so that you can pre-process for topic modeling"""
        if not os.path.exists('processed'):
            os.makedirs('processed')

        for text in self.texts:
            with open('processed/' + text.filename + '.txt', 'w') as current_text:
                current_text.write(' '.join(text.stems))


    def tokens_by_publication_cfd(self):
        cfd = nltk.ConditionalFreqDist(
            (text.publication, word)
            for text in self.texts
            for word in text.tokens
                )
        return cfd

    def generate_stopwords(self):
        global STOPWORD_LIST
        """generates stopwords for use in tokenizing"""
        nltk_stopwords = stopwords.words('french')
        # stopwords from http://www.ranks.nl/stopwords/french
        ranks_stopwords = []
        #with codecs.open('french_stopwords.txt', 'r', 'utf8') as f:
#            ranks_stopwords = [x.strip('\n') for x in f]
#            ranks_stopwords = [x.strip('\r') for x in f]
        # put custom stopword list here. Could also
        # read in as a csv file if that's easier.

        text = codecs.open("french_stopwords.txt", "r", 'utf8').read()

        text = text.replace(' ', ',')
        text = text.replace('\r',',')
        text = text.replace('\n',' ')

        ranks_stopwords = text.split(",")
        ranks_stopwords = [i.replace(' ', '') for i in ranks_stopwords]

        

        ## Stopwords not being read......
#        print (ranks_stopwords)
 #       print (nltk_stopwords)
        extra_stopwords = []
        punctuation = ['»', '«', ',', '-', '.', '!',
                       "\"", '\'' ':', ';', '?', '...']
        STOPWORD_LIST = set(nltk_stopwords + ranks_stopwords +
                            punctuation + extra_stopwords)
        print (STOPWORD_LIST)
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
        # takes the last row and makes it first,
        # since it gets shuffled to the back
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

    def list_all_filenames(self):
        for text in self.texts:
            print(text.filename)

    
    def lda(self):
        allthetokens = []
        numberoftopics = int(input("Please enter the number of topics for the LDA."))
        numberofwords = int(input("Please enter the number of words for each topic."))
        nltk_stopwords = stopwords.words('french')
        # stopwords from http://www.ranks.nl/stopwords/french
        ranks_stopwords = []
        text = codecs.open("french_stopwords.txt", "r", 'utf8').read()

        text = text.replace(' ', ',')
        text = text.replace('\r',',')
        text = text.replace('\n',' ')

        ranks_stopwords = text.split(",")
        # put custom stopword list here. Could also
        # read in as a csv file if that's easier.
 #       print (ranks_stopwords)
        extra_stopwords = []
        punctuation = ['»', '«', ',', '-', '.', '!',
                       "\"", '\'' ':', ';', '?', '...']
        thestopwords = set(nltk_stopwords + ranks_stopwords +
                            punctuation + extra_stopwords)
        thestopwords = list(thestopwords)
        print (STOPWORD_LIST)
  #      print (thestopwords)
        for text in self.texts:
#            currenttext = text
#            currenttext = [w for w in text if w.lower() not in thestopwords]
## The stopwords arn't getting removed, but the punctuation is

            currenttext = text.tokens_without_stopwords
##hmm removing only some of the stopwords i think
            
            textwithoutpunc = [word for word in currenttext if word.isalpha()]
 #           print (textwithoutpunc)
            allthetokens.append(textwithoutpunc)
 #       print (allthetokens[0])
        for subarray in range(0, len(allthetokens)):
#            print (allthetokens[subarray])
            for word in range(0, len(allthetokens[subarray])):
                #print (allthetokens[subarray][word])
                #print (thestopwords)
                if allthetokens[subarray][word] in thestopwords:
                    print ("yes")
                    allthetokens[subarray].remove(allthetokens[subarray][word])
                    
        print ("Please wait.  This could take some time...")

        dictionary = corpora.Dictionary(allthetokens)
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in allthetokens]
        Lda = gensim.models.ldamodel.LdaModel
        ldamodel = Lda(doc_term_matrix, num_topics=numberoftopics, id2word = dictionary, passes=50)
        returnthis = ldamodel.print_topics(num_topics=numberoftopics, num_words=numberofwords)
        return returnthis

## is tdf-if necessary before running lda??


    def find_by_filename(self, name):
        """given a filename, return the text associated with it."""
        for text in self.texts:
            if text.filename == name:
                return text

    def find_percentages_on_first_page(self):
        """prints out the number of tokens on the first page of
        an article as well as the percentage of total tokens"""
        print("filename" + "," + "tokens on first page" + "," + "percentage on first page")
        for text in self.texts:
            try:
                num_tokens_on_first_page = text.find_page_breaks()
                total_tokens = (len(text.bigrams) + 1)
                percentage_on_first_page = (num_tokens_on_first_page / total_tokens) * 100
                print(text.date + "," + text.publication + "," + str(num_tokens_on_first_page) + "," + str(percentage_on_first_page))
            except:
                # if there are no page breaks, pass this article
                pass

    def find_conjunctions(self):
        """finds the number of coordinating conjunctions for each text and prints them out."""
        for text in self.texts:
            conjunctions = FreqDist([tag for token, tag in text.tagged_tokens])
            conjunction_count = conjunctions['KON']
            normalized_conjunction_count = conjunction_count/len(text.tokens)
            print(text.date + ',' + text.publication + "," + str(normalized_conjunction_count))

    def dispersion_plots(self, character_or_token, file_name, bin_count=500):
        """\
        given a character or token and a filename to output the things to, produce a scatterplot of the output
        """

        fig, axes = plt.subplots(len(self.texts), 1, squeeze=True)
        fig.set_figheight(9.4)
        for (text, ax) in zip(self.texts, axes):
            print(text.filename)
            matches = list(re.finditer(character_or_token, text.text))
            locations = [m.start() for m in matches]
            n, bins = np.histogram(locations, bin_count)

            # fig.suptitle(text.filename, fontsize=14, fontweight='bold')
            left = np.array(bins[:-1])
            right = np.array(bins[1:])
            bottom = np.zeros(len(left))
            top = bottom + n

            XY = np.array(
                [[left, left, right, right], [bottom, top, top, bottom]]
            ).T

            barpath = path.Path.make_compound_path_from_polys(XY)

            patch = patches.PathPatch(
                barpath, facecolor='blue', edgecolor='gray', alpha=0.8,
                )

            ax.set_xlim(left[0], right[-1])
            ax.set_ylim(bottom.min(), top.max())
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            # plt.axis('off')
            ax.add_patch(patch)

            # ax.set_xlabel('Position in Text, Measured by Character')
            # ax.set_ylabel('Number of Quotations')

        output = os.path.join(file_name + '.png')
        print('writing to {}'.format(output))
        plt.savefig(output, transparent=True)
        plt.show()

class IndexedText(object):
    """Text object"""
    def __init__(self, filepath):
        self.filepath = filepath
        self.filename = re.sub(r'^.*/|_clean.txt|_Clean.txt', '', filepath).lower()
        self.text = self.read_text()
        self.sentences = self.get_text_sentences()
        self.tokens = self.flatten_sentences()
        self.tagger = treetaggerwrapper.TreeTagger(TAGLANG='fr', TAGDIR='tagger')
        self.tree_tagged_tokens = self.get_tree_tagged_tokens()
        self.tagged_tokens = [(foo.word, foo.pos) for foo in self.tree_tagged_tokens]
        self.stems = [foo.lemma for foo in self.tree_tagged_tokens]
        self.bigrams = list(nltk.bigrams(self.tokens))
        self.trigrams = list(nltk.trigrams(self.tokens))
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
        self.tokens_without_punctuation = [word for word in self.tokens if word.isalpha()]

    def get_tree_tagged_tokens(self):
        """takes the tokens and tags them"""
        tagger = self.tagger
        return treetaggerwrapper.make_tags(tagger.tag_text(self.tokens))

    def find_page_breaks(self):
        """take bigrams and return indexs of page breaks for a text"""
        # page_breaks = []
        return self.bigrams.index(('page', 'break'))
        # for index, bigram in enumerate(self.bigrams):
            # this will grab all indices
            # if bigram == ('page', 'break'):
            #     page_breaks.append(index)
        # return page_breaks

    def get_text_sentences(self):
        """returns sentences from a text"""
        tokenizer = nltk.data.load('tokenizers/punkt/french.pickle')
        untokenized_sentences = [w.lower() for w in tokenizer.tokenize(self.text)]
        matches = []
        for sent in untokenized_sentences:
            matches.append(re.findall(
                r'\w+|[\'\"\/^/\,\-\:\.\;\?\!\(0-9]', sent
            ))
        return matches

    def flatten_sentences(self):
        """takes those sentences and returns tokens"""
        return [item for sublist in self.sentences for item in sublist]


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


    def puncbysection_indiv(self):

## This function creates charts of the frequency of all punctuation marks in all subdivisions of the text.
## The character count of all of the subdivisions can be modified in the definition for the variable 'parts'
        plt.rcdefaults()
        fig, ax = plt.subplots()

        punctuation_marks = ['»', '«', ',', '-', '.', '!',
                             "\"", ':', ';', '?', '...', '\'']

        y_pos = np.arange(len(punctuation_marks))
        thetext = self.text
        parts = [thetext[i:i+1000] for i in range(0, len(thetext), 1000)]


        for q in range(0, len(parts)):
            thnumber = str(q + 1)
            newthing = parts[q]
            newthing = re.findall(r"[\w]+|[^\s\w]", newthing)
            occur = []


            for i in range(0, len(punctuation_marks)):
                z=0
                for x in range(0, len(newthing)):
                    if punctuation_marks[i] == newthing[x]:
                        z = z+1
                occur.append(z)
        
            y_pos = np.arange(len(punctuation_marks))

            plt.barh(y_pos, occur, align='center', alpha=0.5)
            plt.yticks(y_pos, punctuation_marks)
            plt.xlabel('Occur')
            plt.title('Punctuation marks in section #' + thnumber + " of text")
            plt.show()


    def puncbysection_total(self):
        
## This function plots the appears of a given punctuation mark over the course of an entire text.


        punctuation_marks = ['»', '«', ',', '-', '.', '!',
                             "\"", ':', ';', '?', '...', '\'']
        mark = input("Please enter the punctuation mark you want to plot: » « , - . ! : ; ? ...  ' ")
        for i in range (0, len(punctuation_marks)):

            if mark == punctuation_marks[i]:


                text = self.text

                parts = [text[i:i+500] for i in range(0, len(text), 500)]

    ##The 500 character count can be changed depending on how long one wants the length of each
    ## subdivisions to be
                totaltally =[]
                
                for q in range(0, len(parts)):
                    newthing = parts[q]
                    newthing = re.findall(r"[\w]+|[^\s\w]", newthing)                
                    z=0
                    for x in range(0, len(newthing)):
                        if punctuation_marks[i] == newthing[x]:
                            z = z+1
                    totaltally.append(z)
                        
                    y_pos = np.arange(len(punctuation_marks[i]))
                
                noofsectionsforxaxis = []

                newvari = punctuation_marks[i]

                for i in range (0, len(totaltally)):
                    addedone = i + 1
                    noofsectionsforxaxis.append(addedone)

        ##plt.rcdefaults()
        ##fig, ax = plt.subplots()       
        ax = figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.plot(noofsectionsforxaxis, totaltally)
        sortedtotaltally = totaltally
        sortedtotaltally.sort(reverse=True)
        heightofyaxis = sortedtotaltally[0] + 1
        plt.axis([1, len(noofsectionsforxaxis), 0, heightofyaxis])
        ax.set_xlabel('Chronological textual subdivision #')
        ax.set_title('The Punctuation Mark ' + newvari + ' Over the Course of Text')
        ax.set_ylabel('The number of occurrences of ' + newvari)
        plt.legend([mark])
        plt.show()

    def multiple_punctuation(self):
        punctuation_marks = ['»', '«', ',', '-', '.', '!',
                             "\"", ':', ';', '?', '...', '\'']

        mark1 = input("Please enter the punctuation mark you want to plot: » « , - . ! : ; ? ... ")
        mark2 = input("Enter the second punctuation mark.")
        totaltally = []
        newtotaltally = []
        noofsectionsforxaxis = []
        
        for i in range (0, len(punctuation_marks)):

            if mark1 == punctuation_marks[i]:

                y_pos = np.arange(len(punctuation_marks[i]))        
                text = self.text

                parts = [text[i:i+500] for i in range(0, len(text), 500)]

                for q in range(0, len(parts)):
                    thnumber = str(q + 1)
                    newthing = parts[q]
                    newthing = re.findall(r"[\w]+|[^\s\w]", newthing)                
                    z=0
                    for x in range(0, len(newthing)):
                        if punctuation_marks[i] == newthing[x]:
                            z = z+1
                    totaltally.append(z)
                        
                    y_pos = np.arange(len(punctuation_marks[i]))
            

                newvari = punctuation_marks[i]

                for i in range (0, len(totaltally)):
                    addedone = i + 1
                    noofsectionsforxaxis.append(addedone)


            elif mark2 == punctuation_marks[i]:

                text = self.text

                parts = [text[i:i+500] for i in range(0, len(text), 500)]
                
                for q in range(0, len(parts)):
                    thnumber = str(q + 1)
                    newthing = parts[q]
                    newthing = re.findall(r"[\w]+|[^\s\w]", newthing)                
                    z=0
                    for x in range(0, len(newthing)):
                        if punctuation_marks[i] == newthing[x]:
                            z = z+1
                    newtotaltally.append(z)
        print (totaltally)
        print (newtotaltally)
                
                
        ax = figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.plot(noofsectionsforxaxis, totaltally)
        plt.plot(noofsectionsforxaxis, newtotaltally)
        totaltallysizing = totaltally
        totaltallysizing.sort(reverse=True)
        newtotaltallysizing = newtotaltally
        newtotaltally.sort(reverse=True)
        if totaltallysizing[0] > newtotaltallysizing[0]:
            yaxassign = totaltallysizing[0]
            yaxassign = yaxassign + 1
        else:
            yaxassign = newtotaltallysizing[0]
            yaxassign = yaxassign + 1
        plt.axis([1, len(noofsectionsforxaxis), 0, yaxassign])
        ax.set_xlabel('Chronological textual subdivisions')
        ax.set_title('The Punctuation Mark ' + mark1 + " & " + mark2 + ' Over the Course of Text')
        ax.set_ylabel('The number of occurrences of ' + mark1 + " and " + mark2)
        plt.legend([mark1, mark2])
        plt.show()

                


    def most_common(self):
        """takes a series of tokens and returns most common 50 words."""
        fd = FreqDist(self.tokens_without_stopwords)
        return fd.most_common(50)

    def token_count(self, token):
        """takes a token and returns the counts for it in the text."""
        return self.fd[token]

    def stemmed_token_count(self, token):
        stem = treetaggerwrapper.make_tags(self.tagger.tag_text(token))[0].lemma
        return FreqDist(self.stems)[stem]

    def count_conjunctions(self):
        tagged_tokens = self.tagged_tokens
        counter = 0
        for index, (token, tag) in enumerate(tagged_tokens):
            if tag == 'KON':
                print(token + ": " + ' '.join(self.tokens[index-3:index+3]))
                counter += 1
        return counter


class GenreText(IndexedText):
    def __init__(self, filepath):
        self.filepath = filepath
        self.filename = re.sub(r'^.*/|_clean\.txt|_Clean\.txt|\.txt', '', os.path.basename(filepath)).lower()
        self.text = self.read_text()
        self.genre = re.split(r'_', self.filename)[1]
        self.sentences = self.get_text_sentences()
        self.tokens = self.flatten_sentences()
        self.tagger = treetaggerwrapper.TreeTagger(TAGLANG='fr', TAGDIR='tagger')
        self.tree_tagged_tokens = self.get_tree_tagged_tokens()
        self.tagged_tokens = [(foo.word, foo.pos) for foo in self.tree_tagged_tokens]
        self.stems = [foo.lemma for foo in self.tree_tagged_tokens]
        self.bigrams = list(nltk.bigrams(self.tokens))
        self.trigrams = list(nltk.trigrams(self.tokens))
        self.length = len(self.tokens)
        self.fd = FreqDist(self.tokens)
        self.stemmer = SnowballStemmer('french')
        self.index = nltk.Index((self.stem(word), i)
                         for (i, word) in enumerate(self.tokens))
        self.tokens_without_stopwords = self.remove_stopwords()
        self.tokens_without_punctuation = [word for word in self.tokens if word.isalpha()]



class GenreCorpus(Corpus):
    """specialized object for genre texts"""

    def __init__(self, genre_corpus_dir='genre_corpus'):
        self.corpus_dir = genre_corpus_dir
        self.stopwords = self.generate_stopwords()
        self.names_list = self.generate_names_list()
        self.texts = self.build_corpus()

    def build_corpus(self):
        """given a corpus directory, make indexed text objects from it"""
        texts = []
        for (root, _, files) in os.walk(self.corpus_dir):
            for fn in files:
                if fn[0] == '.':
                    pass
                else:
                    texts.append(GenreText(os.path.join(root, fn)))
        return texts


def main():
    """Main function to be called when the script is called"""
    corpus = Corpus()
    corpus.read_out()
    # corpus.csv_dump(corpus.single_token_by_date('crime'))


if __name__ == '__main__':
    main()
