import pandas as pd
import numpy as np

file_loc = '../articles1.csv'

#@st.cache
def load_data():
    full = pd.read_csv(file_loc)
    sample = full.sample(n=1000, random_state=10)
    return sample

# load the dataset
df = load_data()

#df['id'] = df['id'].astype(str)
df.info()
#take random 500 article id's
all_ids= df['id'].to_list()
test_ids = np.random.choice(all_ids, 500)
test_ids = test_ids.tolist()
print(test_ids)