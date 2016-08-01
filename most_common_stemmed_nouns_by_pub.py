import main
from nltk import FreqDist
corpus = main.Corpus()

dates = {}

for text in corpus.texts:
	this_date = []
	current_month=str(text.year) + '-' + str(text.month)
	for token, tag in text.tagged_tokens:
		if tag == 'NOM':
			this_date.append(token)

	if text.month in dates:
		dates[current_month] += this_date

	else:
		dates[current_month] = this_date

print(dates)

for key in dates.keys():
	dates[key] = FreqDist(dates[key]).most_common(100)

print(dates)