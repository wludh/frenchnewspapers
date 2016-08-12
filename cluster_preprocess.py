import main
import shutil
import os

FOLDER_DIR = 'processed'
FILTER = 'ADJ'

if os.path.exists(FOLDER_DIR):
    shutil.rmtree(FOLDER_DIR)

os.makedirs(FOLDER_DIR)

corpus = main.Corpus()
print([text.filename for text in corpus.texts])
corpus.group_articles_by_publication()
print([text.filename for text in corpus.texts])
for text in corpus.texts:
    to_write = []
    with open(FOLDER_DIR + '/' + text.filename + '.txt', 'w') as current_text:
        for x in text.tree_tagged_tokens:
            if x.pos == FILTER:
                to_write.append(x.lemma)
        current_text.write(' '.join(to_write))

