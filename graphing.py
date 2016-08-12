import main
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# from bokeh.plotting import figure, output_file, show

corpus = main.Corpus()

token_dict = {}
# common_lemmas = [FreqDist(text.stems).most_common(3000)
# for text in corpus.texts]
for text in corpus.texts:
    token_dict[text.filename] = text.text

tfidf = TfidfVectorizer()
tfs = tfidf.fit_transform(token_dict.values())


names = [text.filename for text in corpus.texts]

# fit and then predict will try and slot them into the nclusters 
fitted = KMeans(n_clusters=5).fit(tfs)
classes = fitted.predict(tfs)
sklearn = PCA(n_components=5)
sklearn_transf = sklearn.fit_transform(tfs.toarray())
plt.scatter(sklearn_transf[:,0], sklearn_transf[:,1] ,c=classes, s=500)
for i in range(len(classes)):
    plt.text(sklearn_transf[i,0], sklearn_transf[i,1] , s=names[i])

plt.show()
savefig('clustering.png')