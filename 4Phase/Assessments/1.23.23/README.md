# Phase 4 Code Challenge

This code challenge is designed to test your understanding of the Phase 4 material. It covers:

* Principal Component Analysis
* Clustering
* Time Series
* Natural Language Processing

_Read the instructions carefully_. You will be asked both to write code and to answer short answer questions.

## Code Tests

We have provided some code tests for you to run to check that your work meets the item specifications. Passing these tests does not necessarily mean that you have gotten the item correct - there are additional hidden tests. However, if any of the tests do not pass, this tells you that your code is incorrect and needs changes to meet the specification. To determine what the issue is, read the comments in the code test cells, the error message you receive, and the item instructions.

## Short Answer Questions 

For the short answer questions...

* _Use your own words_. It is OK to refer to outside resources when crafting your response, but _do not copy text from another source_.

* _Communicate clearly_. We are not grading your writing skills, but you can only receive full credit if your teacher is able to fully understand your response. 

* _Be concise_. You should be able to answer most short answer questions in a sentence or two. Writing unnecessarily long answers increases the risk of you being unclear or saying something incorrect.


```python
# Run this cell without changes to import the necessary libraries

from numbers import Number
import matplotlib, sklearn, scipy, pickle
import numpy as np
import pandas as pd
```

---

## Part 1: Principal Component Analysis [Suggested Time: 15 minutes]

---

In this part, you will use Principal Component Analysis on the wine dataset. 


```python
# Run this cell without changes

# Relevant imports
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load data
wine = load_wine()
X, y = load_wine(return_X_y=True)
X = pd.DataFrame(X, columns=wine.feature_names)
y = pd.Series(y)
y.name = 'class'

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Scaling
scaler_1 = StandardScaler()
X_train_scaled = pd.DataFrame(scaler_1.fit_transform(X_train), columns=X_train.columns)

# Inspect the first five rows of the scaled dataset
X_train_scaled.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>alcohol</th>
      <th>malic_acid</th>
      <th>ash</th>
      <th>alcalinity_of_ash</th>
      <th>magnesium</th>
      <th>total_phenols</th>
      <th>flavanoids</th>
      <th>nonflavanoid_phenols</th>
      <th>proanthocyanins</th>
      <th>color_intensity</th>
      <th>hue</th>
      <th>od280/od315_of_diluted_wines</th>
      <th>proline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.104538</td>
      <td>-0.530902</td>
      <td>-0.136257</td>
      <td>-0.374157</td>
      <td>-1.294014</td>
      <td>-1.017096</td>
      <td>-0.444344</td>
      <td>1.266120</td>
      <td>0.159532</td>
      <td>-1.074295</td>
      <td>0.516454</td>
      <td>-0.418240</td>
      <td>-0.851947</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.608849</td>
      <td>-0.792240</td>
      <td>-0.573221</td>
      <td>-0.217310</td>
      <td>4.793609</td>
      <td>0.421716</td>
      <td>0.331268</td>
      <td>-0.403193</td>
      <td>2.946675</td>
      <td>-0.990146</td>
      <td>0.856550</td>
      <td>0.076074</td>
      <td>0.739762</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.170548</td>
      <td>-0.471890</td>
      <td>1.611596</td>
      <td>-0.091832</td>
      <td>0.660038</td>
      <td>1.141122</td>
      <td>1.036369</td>
      <td>0.014135</td>
      <td>0.363469</td>
      <td>-0.190727</td>
      <td>1.239159</td>
      <td>1.133355</td>
      <td>0.663137</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.371448</td>
      <td>1.559801</td>
      <td>0.118638</td>
      <td>0.410080</td>
      <td>-1.218858</td>
      <td>0.997241</td>
      <td>1.096806</td>
      <td>-1.321315</td>
      <td>2.317869</td>
      <td>-0.905997</td>
      <td>-0.886446</td>
      <td>1.462898</td>
      <td>-1.200242</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.443619</td>
      <td>0.000204</td>
      <td>-0.573221</td>
      <td>-0.374157</td>
      <td>-0.316988</td>
      <td>-0.985122</td>
      <td>-1.290465</td>
      <td>2.184241</td>
      <td>-1.030103</td>
      <td>0.903214</td>
      <td>-0.971470</td>
      <td>-1.365674</td>
      <td>-0.103112</td>
    </tr>
  </tbody>
</table>
</div>



### 1.1) Create a PCA object `wine_pca` and fit it using `X_train_scaled`.

Use parameter defaults with `n_components=0.9` and `random_state=1` for your classifier. You must use the Scikit-learn PCA (docs [here](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)).


```python
# CodeGrade step1.1
# Your code here

wine_pca = PCA()

wine_pca
```


```python
# This test confirms that you have created a PCA object named wine_pca

assert type(wine_pca) == PCA

# This test confirms that you have set random_state to 1

assert wine_pca.get_params()['random_state'] == 1

# This test confirms that wine_pca has been fit

sklearn.utils.validation.check_is_fitted(wine_pca)
```

### 1.2) Create a numeric variable `wine_pca_ncomps` containing the number of components in `wine_pca`

_Hint: Look at the list of attributes of trained `PCA` objects in the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)_


```python
# CodeGrade step1.2
# Replace None with appropriate code

wine_pca_ncomps = None
```


```python
# This test confirms that you have created a numeric variable named wine_pca_ncomps

assert isinstance(wine_pca_ncomps, Number)
```

### 1.3) Short Answer: Is PCA more useful or less useful when you have high multicollinearity among your features? Explain why.


```python
# Your answer here


```

--- 

## Part 2: Clustering [Suggested Time: 20 minutes]

---

In this part, you will answer general questions about clustering.


```python
# Run this cell without changes

from sklearn.cluster import KMeans
```

### 2.1) Short Answer: Describe the steps of the k-means clustering algorithm.

Hint: Refer to the animation below, which visualizes the process.

<img src='https://raw.githubusercontent.com/learn-co-curriculum/dsc-cc-images/main/phase_4/centroid.gif'>


```python
# Your answer here


```

### 2.2) Write a function `get_labels()` that meets the requirements below to find `k` clusters in a dataset of features `X`, and return the cluster assignment labels for each row of `X`. 

Review the doc-string in the function below to understand the requirements of this function.

_Hint: Within the function, you'll need to:_
* instantiate a [scikit-learn KMeans object](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), using `random_state = 1` for reproducibility
* fit the object to the data
* return the cluster assignment labels for each row of `X` 


```python
# CodeGrade step2.2
# Replace None with appropriate code

    def get_labels(k, X):
        """ 
        Finds the labels from a k-means clustering model 

        Parameters: 
        -----------
        k: float object
            number of clusters to use in the k-means clustering model
        X: Pandas DataFrame or array-like object
            Data to cluster

        Returns: 
        --------
        labels: array-like object
            Labels attribute from the k-means model

        """

        # Instantiate a k-means clustering model with random_state=1 and n_clusters=k
        kmeans = None

        # Fit the model to the data
        None

        # Return the predicted labels for each row in the data produced by the model
        return None
```


```python
# This test confirms that you have created a function named get_labels

assert callable(get_labels) 

# This test confirms that get_labels can take the correct parameter types

get_labels(1, np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]]))
```




    array([0, 0, 0], dtype=int32)



The next cell uses your `get_labels` function to cluster the wine data, looping through all $k$ values from 2 to 9. It saves the silhouette scores for each $k$ value in a list `silhouette_scores`.


```python
# Run this cell without changes

from sklearn.metrics import silhouette_score

# Preprocessing is needed. Scale the data
scaler_2 = StandardScaler()
X_scaled = scaler_2.fit_transform(X)

# Create empty list for silhouette scores
silhouette_scores = []

# Range of k values to try
k_values = range(2, 10)

for k in k_values:
    labels = get_labels(k, X_scaled)
    score = silhouette_score(X_scaled, labels, metric='euclidean')
    silhouette_scores.append(score)
```

Next, we plot the silhouette scores obtained for each different value of $k$, against $k$, the number of clusters we asked the algorithm to find. 


```python
# Run this cell without changes

import matplotlib.pyplot as plt

plt.plot(k_values, silhouette_scores, marker='o')
plt.title('Silhouette scores vs number of clusters')
plt.xlabel('k (number of clusters)')
plt.ylabel('silhouette score');
```


    
![png](index_files/index_22_0.png)
    


### 2.3) Create numeric variable `wine_nclust` containing the value of $k$ you would choose based on the above plot of silhouette scores. 


```python
# CodeGrade step2.3
# Replace None with appropriate code

wine_nclust = None
```


```python
# This test confirms that you have created a numeric variable named wine_nclust

assert isinstance(wine_nclust, Number)
```

---

## Part 3: Natural Language Processing [Suggested Time: 20 minutes]

---

In this third section we will attempt to classify text messages as "SPAM" or "HAM" using TF-IDF Vectorization.


```python
# Run this cell without changes

# Import necessary libraries 
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize

# Generate a list of stopwords 
nltk.download('stopwords')
stops = stopwords.words('english') + list(string.punctuation)

# Read in data
df_messages = pd.read_csv('./spam.csv', usecols=[0,1])

# Convert string labels to 1 or 0 
le = LabelEncoder()
df_messages['target'] = le.fit_transform(df_messages['v1'])

# Examine our data
print(df_messages.head())

# Separate features and labels 
X = df_messages['v2']
y = df_messages['target']

# Create test and train datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state=1)
```

         v1                                                 v2  target
    0   ham  Go until jurong point, crazy.. Available only ...       0
    1   ham                      Ok lar... Joking wif u oni...       0
    2  spam  Free entry in 2 a wkly comp to win FA Cup fina...       1
    3   ham  U dun say so early hor... U c already then say...       0
    4   ham  Nah I don't think he goes to usf, he lives aro...       0


    [nltk_data] Downloading package stopwords to
    [nltk_data]     /Users/gadamico/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!


### 3.1) Create CSR matrices `tf_idf_train` and `tf_idf_test` by using a `TfidfVectorizer` with stop word list `stops` to vectorize `X_train` and `X_test`, respectively.

Besides using the stop word list, use paramater defaults for your TfidfVectorizer. Refer to the documentation about [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).


```python
# CodeGrade step3.1
# Replace None with appropriate code

vectorizer = None

tf_idf_train = None

tf_idf_test = None
```


```python
# These tests confirm that you have created CSR matrices tf_idf_train and tf_idf_test

assert type(tf_idf_train) == scipy.sparse.csr.csr_matrix
assert type(tf_idf_test) == scipy.sparse.csr.csr_matrix
```

### 3.2) Create an array `y_preds` containing predictions from an untuned `RandomForestClassifier` that uses `tf_idf_train` and `tf_idf_test`.

Use parameter defaults with `random_state=1` for your classifier. Refer to the documentation on [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).


```python
# CodeGrade step3.2
# Replace None with appropriate code

classifier = None
    
y_preds = None
```


```python
# This test confirms that you have created an array named y_preds

assert type(y_preds) == np.ndarray
```

### 3.3) Short Answer: What would it mean if the word "genuine" had the highest TF-IDF value of all words in one document from our test data?


```python
# Your answer here


```

---

## Part 4: Time Series [Suggested Time: 20 minutes]

---
In this part you will analyze the price of one stock over time. Each row of the dataset has four prices tracked for each day: 

* Open: The price when the market opens.
* High: The highest price over the course of the day.
* Low: The lowest price over the course of the day.
* Close: The price when the market closes.

<!---Create stock_df and save as .pkl
stocks_df = pd.read_csv("raw_data/all_stocks_5yr.csv")
stocks_df["clean_date"] = pd.to_datetime(stocks_df["date"], format="%Y-%m-%d")
stocks_df.drop(["date", "clean_date", "volume", "Name"], axis=1, inplace=True)
stocks_df.rename(columns={"string_date": "date"}, inplace=True)
pickle.dump(stocks_df, open("write_data/all_stocks_5yr.pkl", "wb"))
--->


```python
# Run this cell without changes

stocks_df = pd.read_csv('./stocks_5yr.csv')
stocks_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15.07</td>
      <td>15.12</td>
      <td>14.63</td>
      <td>14.75</td>
      <td>February 08, 2013</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14.89</td>
      <td>15.01</td>
      <td>14.26</td>
      <td>14.46</td>
      <td>February 11, 2013</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14.45</td>
      <td>14.51</td>
      <td>14.10</td>
      <td>14.27</td>
      <td>February 12, 2013</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14.30</td>
      <td>14.94</td>
      <td>14.25</td>
      <td>14.66</td>
      <td>February 13, 2013</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14.94</td>
      <td>14.96</td>
      <td>13.16</td>
      <td>13.99</td>
      <td>February 14, 2013</td>
    </tr>
  </tbody>
</table>
</div>



### 4.1) For `stocks_df`, create a DatetimeIndex from the `date` column.

The resulting DataFrame should not have a `date` column, only `open`, `high`, `low`, and `close` columns. 

Hint: First convert the `date` column to Datetime datatype, then set it as the index.


```python
# CodeGrade step4.1
# Replace None with appropriate code

stocks_df['date'] = None
```


```python
# This test confirms that stocks_df has a DatetimeIndex

assert type(stocks_df.index) == pd.core.indexes.datetimes.DatetimeIndex

# This test confirms that stocks_df only has `open`, `high`, `low`, and `close` columns.

assert list(stocks_df.columns) == ['open', 'high', 'low', 'close']
```

### 4.2) Create a DataFrame `stocks_monthly_df` that resamples `stocks_df` each month with the 'MS' DateOffset to calculate the mean of the four features over each month.

Refer to the [resample documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.resample.html).


```python
# CodeGrade step4.2
# Replace None with appropriate code

stocks_monthly_df = None
```


```python
# This test confirms that you have created a DataFrame named stocks_monthly_df

assert type(stocks_monthly_df) == pd.DataFrame

# This test confirms that stocks_monthy_df has the correct dimensions

assert stocks_monthly_df.shape == (61, 4)
```

### 4.3) Create a matplotlib figure `rolling_open_figure` containing a line graph that visualizes the rolling quarterly mean of open prices from `stocks_monthly_df`.

You will use this graph to determine whether the average monthly open stock price is stationary or not.

Hint: use a window size of 3 to represent one quarter of a year


```python
# CodeGrade step4.3
# Your code here

rolling_open_figure, ax = plt.subplots(figsize=(10, 6))


```


```python
# This test confirms that you have created a figure named rolling_open_figure

assert type(rolling_open_figure) == plt.Figure

# This test confirms that the figure contains exactly one axis

assert len(rolling_open_figure.axes) == 1
```

### 4.4) Short Answer: Based on your graph from Question 4.3, does the monthly open stock price look stationary? Explain your answer.


```python
# Your answer here


```


```python

```
