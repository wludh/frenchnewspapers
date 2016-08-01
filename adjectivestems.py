
import main
from nltk import FreqDist
corpus = main.Corpus()
dates = {}
for text in corpus.texts:
	adjs = []
	for x in text.tree_tagged_tokens:
          if x.pos[0:3] == 'ADJ':
               adjs.append(x.lemma)
	if text.date in dates:
		dates[text.date]+= adjs
	else: 
		dates[text.date] = adjs

for key in dates.keys():
	dates[key] = FreqDist(dates[key]).most_common(100)
print(dates)

