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

st.title('20 News Groups Clustering')

st.sidebar.header('Inputs for Clustering')

optimal_clusters = st.sidebar.number_input('Enter the number of clusters', min_value=2, max_value=50, step=1)

stopwords_add = st.sidebar.text_area('Enter Program Specific Stopwords')
stopwords_add = [x.strip() for x in stopwords_add.split(',')]

file_loc = '../articles1.csv'

#@st.cache
def load_data():
    full = pd.read_csv(file_loc)
    sample = full.sample(n=1000, random_state=10)
    return sample

# constants for preprocessing function
lemma = WordNetLemmatizer()
#stopwords_add = []
stoplist = stopwords.words('english') + list(string.punctuation) + stopwords_add

# create a function to clean the text
@st.cache
def preprocess(x):
    x = x.strip()
    a = string.punctuation.replace('-','')
    x = re.sub(r'[{}]'.format(a),'', x.lower())
    x = [w for w in x.split() if w not in stoplist and not w.isdigit()]  # remove stopwords
    x = " ".join(lemma.lemmatize(word) for word in x if word not in stoplist)
    x = [w for w in x.split() if w not in stoplist]
    x = " ".join(word for word in x if word not in stoplist)
    return x                                     # join the list

# load the dataset
df = load_data()
# run preprocessing
# data_preprocess_state = st.text('Preprocessing Text Data...')
df['text_processed'] = df['content'].apply(preprocess)
# data_preprocess_state.text('Preprocessing Text Data Complete!')

documents = df['text_processed'].to_list()
vectorizer = TfidfVectorizer(
                        ngram_range=(1,2),
                        min_df = 5,
                        max_df = 0.95,
                        max_features = 8000)

#@st.cache
def vectorize(doc):
    x = vectorizer.fit_transform(doc)
    return x

#data_vectorize_state = st.text('Vectorizing Data...')
x = vectorize(documents)
#data_vectorize_state.text('Vectorizing Complete!')

# Begin cluster process
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

# multiselect to select clusters to isolate for graph?

fig = px.scatter(data_frame=df, x='x_tSNE', y='y_tSNE', color='clusters_final_ID',
                 hover_data=['title'])

# create a filtered dataframe to display

display_cols = ['id', 'title', 'publication', 'author', 'date', 'clusters_final_ID']
df_display = df[display_cols]

#create streamlit app code

# if st.checkbox('Show Article Level Data'):
#     st.subheader('Article Level Data')
#     st.write(df_display)

#st.subheader('Cluster Top 10 Keywords')
st.dataframe(df_keywords)

#st.subheader('T-SNE Plot')
st.plotly_chart(fig)

st.dataframe(df_display)