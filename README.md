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

The third line here loads the corpus from a given directory. By default, it reads in from a folder called "clean_ocr". If you don't have such a folder 
