#first mods

import pandas as pd
import numpy as np
import string
import re
from datetime import datetime

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
import plotly.express as px

import streamlit as st

df_full = pd.read_csv('../articles1.csv')
df = df_full.sample(n=1000)
#create a function to clean the text
lemma = WordNetLemmatizer()
stopwords_add = []

stoplist = stopwords.words('english') + list(string.punctuation) + stopwords_add

@st.cache()
def preprocess(x):
    x = x.strip()
    a = string.punctuation.replace('-','')
    x = re.sub(r'[{}]'.format(a),'', x.lower())
    x = [w for w in x.split() if w not in stoplist and not w.isdigit()]  # remove stopwords
    x = " ".join(lemma.lemmatize(word) for word in x if word not in stoplist)
    x = [w for w in x.split() if w not in stoplist]
    x = " ".join(word for word in x if word not in stoplist)
    return x                                     # join the list

#data_preprocess_state = st.text('Preprocessing Text Data...')
df['text_processed'] = df['content'].apply(preprocess)
#data_preprocess_state.text('Preprocessing Text Data Complete!')

documents = df['text_processed'].to_list()

vectorizer = TfidfVectorizer(
                        ngram_range=(1,2),
                        min_df = 5,
                        max_df = 0.95,
                        max_features = 8000)

#data_vectorize_state = st.text('Vectorizing Data...')
x = vectorizer.fit_transform(documents)
#data_vectorize_state.text('Vectorizing Complete!')

#ENTER OPTIMAL CLUSTER NUMBER
optimal_clusters = 20
random_st = 21

clusters_final = MiniBatchKMeans(n_clusters=optimal_clusters, init_size=1024, batch_size=2048,
                                 random_state=random_st).fit_predict(x)

df['clusters_final_ID'] = pd.Series(clusters_final, index=df.index)

def get_top_keywords2(data, clusters, labels, n_terms):
    df_new = pd.DataFrame(data.todense()).groupby(clusters).mean()
    keywords = []
    clusters = []

    for i, r in df_new.iterrows():
        clusters.append(i)
        # print('\nCluster {}'.format(i))
        keywords.append(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))

    return clusters, keywords


clusters, keywords = get_top_keywords2(x, clusters_final, vectorizer.get_feature_names(), 10)
df_keywords = pd.DataFrame({'Cluster': clusters, 'Top_10_Keywords': keywords})
df_keywords['Top_10_Keywords'] = df_keywords['Top_10_Keywords'].astype(str)
df_keywords['Top_10_Keywords'] = df_keywords['Top_10_Keywords'].str.split(',').apply(lambda x: ', '.join(x[::-1]))

#create tSNE components
tSNE=TSNE(n_components=2)
tSNE_result = tSNE.fit_transform(x.todense())
x_tSNE=tSNE_result[:,0]
y_tSNE=tSNE_result[:,1]
df['x_tSNE']=x_tSNE
df['y_tSNE']=y_tSNE

#create a plotly express scatter plot

fig = px.scatter(data_frame=df, x='x_tSNE', y='y_tSNE', color='clusters_final_ID',
                 hover_data=['title'])

#create streamlit app code
st.title('20 News Groups Clustering')

#st.subheader('All Data')
st.dataframe(df)

#st.subheader('Cluster Top 10 Keywords')
st.dataframe(df_keywords)

#st.subheader('T-SNE Plot')
st.plotly_chart(fig)