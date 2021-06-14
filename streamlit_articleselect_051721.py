import pandas as pd
import numpy as np
import string
import re
from datetime import datetime

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
import plotly.express as px
import streamlit as st
import base64

@st.cache(allow_output_mutation=True, ttl=60*5, max_entries=20)
def load_data():
    full = pd.read_csv(file_loc)
    sample = full.sample(n=1000, random_state=10)
    return sample

# create a function to clean the text
@st.cache(ttl=60*5)
def preprocess(x):
    x = x.strip()
    a = string.punctuation.replace('-', '')
    x = re.sub(r'[{}]'.format(a),'', x.lower())
    x = [w for w in x.split() if w not in stoplist and not w.isdigit()]  # remove stopwords
    x = " ".join(lemma.lemmatize(word) for word in x if word not in stoplist)
    x = [w for w in x.split() if w not in stoplist]
    x = " ".join(word for word in x if word not in stoplist)
    return x                                     # join the list

#@st.cache
def vectorize(doc):
    x = vectorizer.fit_transform(doc)
    return x

def get_top_keywords2(data, clusters, labels, n_terms):
    df_new = pd.DataFrame(data.todense()).groupby(clusters).mean()
    keywords = []
    clusters = []

    for i, r in df_new.iterrows():
        clusters.append(i)
        # print('\nCluster {}'.format(i))
        keywords.append(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))

    return clusters, keywords

def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.

    object_to_download (str, pd.DataFrame):  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
    download_link_text (str): Text to display for download link.

    Examples:
    download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
    download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')

    """
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

def main():
    # load the dataset
    df = load_data()
    df['id'] = df['id'].astype(str)

    #isolate data to only article ID's specified in user input
    df2 = df[df['id'].isin(article_ids)]

    # run preprocessing
    # data_preprocess_state = st.text('Preprocessing Text Data...')
    df2['text_processed'] = df2['content'].apply(preprocess)
    # data_preprocess_state.text('Preprocessing Text Data Complete!')

    #vectorize the docs
    documents = df2['text_processed'].to_list()
    x = vectorize(documents)

    #added step for singular value decomposition (dimensionality reduction)
    lsa_obj = TruncatedSVD(n_components=optimal_clusters, n_iter=500, random_state=42)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(lsa_obj, normalizer)
    tfidf_lsa_data = lsa.fit_transform(x)

    # Begin cluster process
    random_st = 21
    clusters_final = MiniBatchKMeans(n_clusters=optimal_clusters, init='k-means++',  n_init=300,
                                     max_iter=300, tol=0.00001, random_state=random_st).fit_predict(tfidf_lsa_data)
    df2['clusters_final_ID'] = pd.Series(clusters_final, index=df2.index)
    clusters, keywords = get_top_keywords2(x, clusters_final, vectorizer.get_feature_names(), 10)
    df_keywords = pd.DataFrame({'Cluster': clusters, 'Top_10_Keywords': keywords})
    df_keywords['Top_10_Keywords'] = df_keywords['Top_10_Keywords'].astype(str)
    df_keywords['Top_10_Keywords'] = df_keywords['Top_10_Keywords'].str.split(',').apply(lambda x: ', '.join(x[::-1]))

    # create tSNE components
    tSNE = TSNE(n_components=2, perplexity=50, learning_rate=10)
    tSNE_result = tSNE.fit_transform(tfidf_lsa_data)
    x_tSNE = tSNE_result[:, 0]
    y_tSNE = tSNE_result[:, 1]
    df2['x_tSNE'] = x_tSNE
    df2['y_tSNE'] = y_tSNE

    # create a plotly express scatter plot
    fig = px.scatter(data_frame=df2, x='x_tSNE', y='y_tSNE', color='clusters_final_ID',
                     hover_data=['title'])

    # create a filtered dataframe to display
    display_cols = ['id', 'title', 'publication', 'author', 'date', 'clusters_final_ID']
    df_display = df2[display_cols]
    # create a cluster frequency data table
    clust_freq = pd.DataFrame(df_display['clusters_final_ID'].value_counts(sort=False))
    clust_freq = clust_freq.reset_index()
    clust_freq = clust_freq.rename(columns={'index': 'cluster_id', 'clusters_final_ID': 'freq_count'})
    # join frequency table with df_keywords
    df_keywords2 = df_keywords.merge(clust_freq, how='left', left_on='Cluster', right_on='cluster_id')
    df_keywords3 = df_keywords2.drop(columns='cluster_id')

    # create streamlit app code
    # st.subheader('Cluster Top 10 Keywords')
    st.dataframe(df_keywords3)
    # col2.dataframe(df_keywords)
    if st.button('Download Top Keywords as CSV'):
        tmp_download_link = download_link(df_keywords, 'Top_Keywords_Test.csv',
                                          'Click Here to Download Top Keywords!')
        st.markdown(tmp_download_link, unsafe_allow_html=True)
    # st.subheader('T-SNE Plot')
    st.plotly_chart(fig)
    # col1.plotly_chart(fig)
    st.dataframe(df_display)
    if st.button('Download Document Data as CSV'):
        tmp_download_link = download_link(df_display, 'Document_clusters_Test.csv', 'Click Here to Download Data!')
        st.markdown(tmp_download_link, unsafe_allow_html=True)

#streamlit opening page template code

st.title('20 News Groups Clustering')
st.sidebar.header('Inputs for Clustering')
optimal_clusters = st.sidebar.number_input('Enter the number of clusters', min_value=2, max_value=50, step=1)
stopwords_add = st.sidebar.text_area('Enter Program Specific Stopwords')
stopwords_add = [x.strip() for x in stopwords_add.split(',')]
article_ids = st.sidebar.text_area('Enter Article IDs to Cluster')
article_ids = [x.strip() for x in article_ids.split(',')]
st.sidebar.subheader('Instructions: ')
st.sidebar.markdown('1. Enter comma delimited Article IDs into the text box')
st.sidebar.markdown('2. Choose the number of groupings expected for your dataset')
st.sidebar.markdown('3. Optional: Enter any undesirable words that are displaying in the top keywords '
                    'into the stopwords list (comma delimited)')
st.sidebar.markdown('4. Modify the settings and iterate as needed!')

file_loc = '../articles1.csv'
# constants for nlp work
lemma = WordNetLemmatizer()
stoplist = stopwords.words('english') + list(string.punctuation) + stopwords_add
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=0.15,
    max_df=0.85,
    max_features=10000)

if __name__ == "__main__":
    try:
        main()
    except:
        st.header('Enter Article IDs to start clustering!')
