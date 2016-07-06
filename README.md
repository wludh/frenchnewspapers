Collects scripts for processing texts for the C19 French Newspapers Scandal project at W&L.

Primary script for text analysis is main.py.

To begin, clone the repository and change into it.

# Setup

```bash
$ git clone git@github.com:wludh/frenchnewspapers.git
$ cd frenchnewspapers
```

The scripts are written in python 3. First, install [homebrew](http://brew.sh/) if you haven't already: 
```
$ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

Then use homebrew to install python3:
```bash
$ brew install python3
```
Then install dependencies using pip.

```bash
$ pip3 install nltk
$ pip3 install python-dateutil
```

# Usage
Individual scripts can be created for a particular purpose by modifying the main() function in main.py. By default, it outputs a series of basic statistics about the corpus to the file results.txt when run as a program. 

A more flexible way to interact with the corpus is by importing the main script in the python interpreter. First, fire up your python interpreter and import your main.py package.

```bash
$ python 3
>> import main
>> corpus = main.Corpus()
```

The third line here loads the corpus from a given directory. By default, it reads in from a folder called "clean_ocr". If you don't have such a folder, you will have to create one and populate it with plain text files available from our WLU Box folder. 

Once read in, the script prepares your corpus as a list of individual texts (organize by date by default) that can be accessed like this:

```bash
>> corpus.texts
[<main.IndexedText object at 0x10d87b550>, <main.IndexedText object at 0x10da060b8>, <main.IndexedText object at 0x10dbb11d0>, <main.IndexedText object at 0x10dc60dd8>, <main.IndexedText object at 0x10deef1d0>, <main.IndexedText object at 0x10e451710>, <main.IndexedText object at 0x10e680080>, <main.IndexedText object at 0x10e9cfa58>, <main.IndexedText object at 0x10eb57cf8>, <main.IndexedText object at 0x10d867898>, <main.IndexedText object at 0x10d9b84e0>, <main.IndexedText object at 0x10db7c5c0>, <main.IndexedText object at 0x10dbef048>, <main.IndexedText object at 0x10dd6a6a0>, <main.IndexedText object at 0x10e3a12e8>, <main.IndexedText object at 0x10e60ef98>, <main.IndexedText object at 0x10e88a978>, <main.IndexedText object at 0x10ea31978>, <main.IndexedText object at 0x10ec87e48>, <main.IndexedText object at 0x10d6a40f0>, <main.IndexedText object at 0x10d6a4630>, <main.IndexedText object at 0x10d8f4e80>, <main.IndexedText object at 0x10dabbcf8>, <main.IndexedText object at 0x10dcde278>, <main.IndexedText object at 0x10e1f2eb8>, <main.IndexedText object at 0x10e4ae940>, <main.IndexedText object at 0x10e721048>, <main.IndexedText object at 0x10ea1ab00>, <main.IndexedText object at 0x10ebccb38>]
```

You can then access any individual text by selecting it from the list:

```bash
>> corpus.texts[0]
<main.IndexedText object at 0x10d87b550>
>> corpus.texts[0].filename
'figaro_june_1_1908'
>> corpus.texts[0].tokens
['assassinat', 'du', 'peintre', 'steinheil', 'et', 'de', 'sa', 'belle-mère', 'mme', 'veuve', 'japy', 'mme', 'steinheil', 'échappe', 'a', 'la', 'mort', 'un', 'crime', 'épouvantable', ',', 'un', 'triple', 'assassinat', ',', 'a', 'été', 'commis', 'à', 'paris', 'dans', 'la', 'nuit', 'de', 'samedi', 'à', 'dimanche', '.', 'dans', 'la', 'série', 'des', 'meurtres', "qu'il", 'faut', 'enregistrer', 'chaque', 'jour', ',', 'celui-là', 'prend', 'une', 'place', 'à', 'part', '.',...
```

I've baked in a variety of methods, some tied to the corpus itself:

* corpus.corpus_dir 
    * give name of the corpus directory
* corpus.stopwords
    * give the list of stopwords currently being used.
* corpus.names_list
    * give the list of proper names used for querying by proper names
* corpus.texts 
    * give the list of all the texts
* corpus.sort_articles_by_date()
    * sort the articles by date.
* corpus.read_out()
    * output to a file called 'results.txt' a variety of stats about the texts in the corpus.
* corpus.group_articles_by_publication()
    * orders the texts by publication and then each of these groupings by date.
* corpus.csv_dump(corpus.single_token_by_date('token'))
    * Actually two methods in one - csv_dump and single_token_by_date. The latter charts the usage of a single token across the corpus and the former writes it to a csv file for graphing in excel.

Others are tied to the individual texts:

* text.filename
    * give filename of text
* text.text
    * give unprocessed text (includes line breaks, etc.)
* text.tokens
    * give tokenized version of a text
* text.length
    * give the length of a text (number of tokens)
* text.fd
    * give a frequency distribution of the text (number of uses of each individual token)
* text.date
    * give date of a text. Can be further broken down with text.year, text.month, and text.day
* text.publication
    * give the name of the journal that published the text.
* text.stemmer
    * produces a French stemmer for the text (not fully implemented yet)
* text.index
    * produces a stemmed index for the text (not fully implemented yet)
* text.tokens_without_stopwords
    * produces a list of tokens in the text with stopwords excluded.