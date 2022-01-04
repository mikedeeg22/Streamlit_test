import pandas as pd
import numpy as np
import string
import re
from nltk.corpus import stopwords, wordnet
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
import math


@st.cache(allow_output_mutation=True, ttl=60 * 5, max_entries=20, suppress_st_warning=True)
def load_data(article_ids):
    full = pd.read_csv(file_loc)
    #       sample = full.sample(n=1000, random_state=10)
    full['id'] = full['id'].astype(str)
    # isolate data to only article ID's specified in user input
    filtered = full[full['id'].isin(article_ids)]
    missings = [x for x in article_ids if x not in (full['id'].tolist())]
    # my_bar = st.progress(0)
    # for percent_complete in range(100):
    #     my_bar.progress(percent_complete + 1)
    # my_bar.empty()
    return filtered, missings


# create a function to clean the text
@st.cache(ttl=60 * 5)
def preprocess(x):
    x = x.strip()
    a = string.punctuation.replace('-', '')
    x = re.sub(r'[{}]'.format(a), '', x.lower())
    x = [w for w in x.split() if w not in stoplist and not w.isdigit()]  # remove stopwords
    x = " ".join(lemma.lemmatize(word, wordnet.VERB) for word in x if word not in stoplist)
    x = [w for w in x.split() if w not in stoplist]
    x = " ".join(word for word in x if word not in stoplist)
    return x  # join the list


# @st.cache
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
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


def tsne_plot(data, lsa_data):
    # create tSNE components
    tsne = TSNE(n_components=2, perplexity=50, learning_rate=10)
    tsne_result = tsne.fit_transform(lsa_data)
    x_tsne = tsne_result[:, 0]
    y_tsne = tsne_result[:, 1]
    data['x_tSNE'] = x_tsne
    data['y_tSNE'] = y_tsne

    # create a plotly express scatter plot
    fig = px.scatter(data_frame=data, x='x_tSNE', y='y_tSNE', color='clusters_final_ID',
                     hover_data=['title'])
    return fig

def dfs_tabs(df_list, sheet_list, file_name):
    writer = pd.ExcelWriter(file_name,engine='xlsxwriter')
    for dataframe, sheet in zip(df_list, sheet_list):
        dataframe.to_excel(writer, sheet_name=sheet, startrow=0 , startcol=0)
    #writer.save()
    return writer

def main():

    # load the dataset
    df2, bad_ids = load_data(article_ids)

    # run preprocessing
    df2['text_processed'] = df2['content'].apply(preprocess)

    # vectorize the docs
    documents = df2['text_processed'].to_list()
    x = vectorize(documents)

    # added step for singular value decomposition (dimensionality reduction)
    lsa_obj = TruncatedSVD(n_components=n_comp, n_iter=500, random_state=42)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(lsa_obj, normalizer)
    tfidf_lsa_data = lsa.fit_transform(x)
    exp_var = lsa_obj.explained_variance_ratio_.sum()

    # Begin cluster process
    kmeans_obj = MiniBatchKMeans(n_clusters=optimal_clusters, init='k-means++', n_init=300,
                                 max_iter=300, tol=0.00001, random_state=random_st)
    clusters_final = kmeans_obj.fit_predict(tfidf_lsa_data)
    df2['clusters_final_ID'] = pd.Series(clusters_final, index=df2.index)
    df2 = df2.reset_index()
    clusters, keywords = get_top_keywords2(x, clusters_final, vectorizer.get_feature_names(), 10)
    df_keywords = pd.DataFrame({'Cluster': clusters, 'Top_10_Keywords': keywords})
    df_keywords['Top_10_Keywords'] = df_keywords['Top_10_Keywords'].astype(str)
    df_keywords['Top_10_Keywords'] = df_keywords['Top_10_Keywords'].str.split(',').apply(lambda x: ', '.join(x[::-1]))

    # calculate the distance to centroid
    s = []
    for i, r in df2.iterrows():
        p_s = (kmeans_obj.cluster_centers_[r['clusters_final_ID']] - tfidf_lsa_data[i])
        s.append(math.sqrt(sum(p_s * p_s)))

    df2['distance'] = s

    # create t-sne plot
    fig = tsne_plot(df2, tfidf_lsa_data)

    # create a filtered dataframe to display
    display_cols = ['id', 'title', 'publication', 'author', 'date', 'clusters_final_ID', 'distance']
    df_display = df2[display_cols]
    # create a cluster frequency data table
    clust_freq = pd.DataFrame(df_display['clusters_final_ID'].value_counts(sort=False))
    clust_freq = clust_freq.reset_index()
    clust_freq = clust_freq.rename(columns={'index': 'cluster_id', 'clusters_final_ID': 'freq_count'})
    # join frequency table with df_keywords
    df_keywords2 = df_keywords.merge(clust_freq, how='left', left_on='Cluster', right_on='cluster_id')
    df_keywords3 = df_keywords2.drop(columns='cluster_id')

    # create streamlit app code
    row1_1, row1_2 = st.columns((3, 2))
    with row1_1:
        st.header('T-SNE Plot')
        st.plotly_chart(fig)
        # st.markdown(
        #     'Dimensionality reduction has captured **{:.1%}** of the original data variance (target: > 80%)'.format(
        #         exp_var))
        st.markdown('IDs not processed: {}'.format(bad_ids))

    with row1_2:
        st.header('Top 10 Keywords Per Cluster')
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.dataframe(df_keywords3)
        st.download_button(label='Download Top Keywords as CSV', data=df_keywords3.to_csv(),
                           file_name='Top_Keywords_test.csv')
        #if st.button('Download Top Keywords as CSV'):
        #    tmp_download_link = download_link(df_keywords, 'Top_Keywords_Test.csv',
        #                                      'Click Here to Download Top Keywords!')
        #    st.markdown(tmp_download_link, unsafe_allow_html=True)

    # row2_1, row2_2 = st.beta_columns((2,3))
    # with row2_1:
    st.header('Article Data with Cluster ID')
    st.dataframe(df_display)
    st.download_button(label='Download Document Data as CSV', data=df_display.to_csv(),
              file_name='Document_clusters_test.csv')
    params = {'Parameter':['N_Clusters', 'Stopwords', 'Max_tfidf', 'Min_tfidf', 'Random_st'],
              'Value':[optimal_clusters, stopwords_add, max_df, min_df, random_st]}
    df_params = pd.DataFrame(params)
    df_params['Value'] = df_params['Value'].astype(str)
    st.dataframe(df_params)
    df_list = [df_display, df_keywords3, df_params]
    sheets = ['Doc_data', 'Cluster_Top_Keywords', 'Model_Parameters']
    st.download_button(label='Download Document Data as CSV', data=dfs_tabs(df_list, sheets, 'Results.xlsx'))
    #if st.button('Download Document Data as CSV'):
    #    tmp_download_link = download_link(df_display, 'Document_clusters_Test.csv', 'Click Here to Download Data!')
    #    st.markdown(tmp_download_link, unsafe_allow_html=True)


# streamlit opening page template code

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")
# sidebar inputs
st.title('20 News Groups Clustering')
st.sidebar.header('Inputs for Clustering')
optimal_clusters = st.sidebar.number_input('Enter the number of clusters', min_value=2, max_value=50, value=10, step=1)
article_ids = st.sidebar.text_area('Enter Article IDs to Cluster')
#article_ids = [x.strip() for x in article_ids.split(',')]
article_ids = re.split(r'[\n\t,]', article_ids)
article_ids = [x.strip() for x in article_ids]
st.sidebar.subheader('Optional Parameters')
stopwords_add = st.sidebar.text_area('Enter Program Specific Stopwords')
stopwords_add = [x.strip() for x in stopwords_add.split(',')]
# Dimensionality reduction N components
n_comp = st.sidebar.number_input("N Components for Dimensionality Reduction", min_value=2,
                                 max_value=300, value=50, step=1)
# Upper bound for tf-idf value
max_df = st.sidebar.slider("Maximum tf-idf value for a word to be considered.", min_value=0.05,
                           max_value=1.0, value=0.85, step=0.01)
# Lower bound for tf-idf value.
min_df = st.sidebar.slider("Minimum tf-idf value for a word to be considered.", min_value=0.00,
                           max_value=1.0, value=0.15, step=0.01)
# Random Seed
random_st = st.sidebar.number_input("Random State (For Initial Cluster Centroid Placement)", min_value=1,
                                    max_value=300, value=20, step=1)

st.sidebar.subheader('Instructions: ')
st.sidebar.markdown('1. Enter comma delimited Article IDs into the text box')
st.sidebar.markdown('2. Choose the number of groupings expected for your dataset')
st.sidebar.markdown('3. Optional: Enter any undesirable words that are displaying in the top keywords '
                    'into the stopwords list (comma delimited)')
st.sidebar.markdown('4. Optional: Adjust any other parameters as desired (n_componenets, max_df, min_df')
st.sidebar.markdown('5. Modify the settings and iterate as needed!')

st.sidebar.subheader('Info on Optional Parameters:')
st.sidebar.markdown('**N Components:** When reducing the dimensionality of highly dimensional datasets, this value '
                    'determines the size of the smaller dataset that contains most of the information of the original '
                    'dataset')
st.sidebar.markdown('**Max_df:** When building the vocabulary, ignore terms that have a document frequency higher than '
                    'the given threshold (by proportion of documents)')
st.sidebar.markdown('**Min_df:** When building the vocabulary, ignore terms that have a document frequency lower than '
                    'the given threshold (by proportion of documents)')
st.sidebar.markdown(
    '**Random State:** The clustering algorithm randomly places cluster centroids to start the process, and then '
    'moves these centroids to determine the location where the distance to cluster center is optimized. '
    'This number changes the starting random location of the cluster centroid, which may influence the '
    'results (depending on the dataset)')

file_loc = '../articles1.csv'
# constants for nlp work
lemma = WordNetLemmatizer()
stoplist = stopwords.words('english') + list(string.punctuation) + stopwords_add
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=min_df,
    max_df=max_df,
    max_features=10000)

#st.markdown('Article IDs!: {}'.format(article_ids))

main()

#if __name__ == "__main__":
#    try:
#            main()
#    except:
#        st.header('Enter Article IDs to start clustering!')
