import main
from nltk import FreqDist
corpus = main.Corpus()

publications = {}

for text in corpus.texts:
	this_publication = []
	for token, tag in text.tagged_tokens:
		if tag == 'ADV':
			this_publication.append(token)

	if text.month in publications:
		publications[text.publication] += this_publication

	else:
		publications[text.publication] = this_publication

print(publications)

for key in publications.keys():
	publications[key] = FreqDist(publications[key]).most_common(100)

print(publications)