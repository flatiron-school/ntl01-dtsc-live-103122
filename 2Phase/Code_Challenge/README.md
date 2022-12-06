# Phase 2 Code Challenge

This code challenge is designed to test your understanding of the Phase 2 material. It covers:

- Normal Distribution
- Statistical Tests
- Bayesian Statistics
- Linear Regression

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

import itertools
import numpy as np
import pandas as pd 
from numbers import Number
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

import pickle
```

---
## Part 1: Normal Distribution [Suggested time: 20 minutes]
---
In this part, you will analyze check totals at a TexMex restaurant. We know that the population distribution of check totals for the TexMex restaurant is normally distributed with a mean of \\$20 and a standard deviation of \\$3. 

### 1.1) Create a numeric variable `z_score_26` containing the z-score for a \\$26 check. 


```python
# CodeGrade step1.1
# Replace None with appropriate code

z_score_26 = None
```


```python
# This test confirms that you have created a numeric variable named z_score_26

assert isinstance(z_score_26, Number)
```

### 1.2) Create a numeric variable `p_under_26` containing the approximate proportion of all checks that are less than \\$26.

Hint: Use the answer from the previous question along with the empirical rule, a Python function, or this [z-table](https://www.math.arizona.edu/~rsims/ma464/standardnormaltable.pdf).


```python
# CodeGrade step1.2
# Replace None with appropriate code

p_under_26 = None
```


```python
# This test confirms that you have created a numeric variable named p_under_26

assert isinstance(p_under_26, Number)

# These tests confirm that p_under_26 is a value between 0 and 1

assert p_under_26 >= 0
assert p_under_26 <= 1
```

### 1.3) Create numeric variables `conf_low` and `conf_high` containing the lower and upper bounds (respectively) of a 95% confidence interval for the mean of one waiter's check amounts using the information below. 

One week, a waiter gets 100 checks with a mean of \\$19 and a standard deviation of \\$3.


```python
# CodeGrade step1.3
# Replace None with appropriate code

n = 100
mean = 19
std = 3

conf_low = None
conf_high = None
```


```python
# These tests confirm that you have created numeric variables named conf_low and conf_high

assert isinstance(conf_low, Number)
assert isinstance(conf_high, Number)

# This test confirms that conf_low is below conf_high

assert conf_low < conf_high

# These statements print your answers for reference to help answer the next question

print('The lower bound of the 95% confidence interval is {}'.format(conf_low))
print('The upper bound of the 95% confidence interval is {}'.format(conf_high))
```

    The lower bound of the 95% confidence interval is 18.412
    The upper bound of the 95% confidence interval is 19.588


### 1.4) Short Answer: Interpret the 95% confidence interval you just calculated in Question 1.3.


```python
# Your answer here


```

---
## Part 2: Statistical Testing [Suggested time: 20 minutes]
---
The TexMex restaurant recently introduced queso to its menu.

We have a random sample containing 2000 check totals, all from different customers: 1000 check totals for orders without queso ("no queso") and 1000 check totals for orders with queso ("queso").

In the cell below, we load the sample data for you into the arrays `no_queso` and `queso` for the "no queso" and "queso" order check totals, respectively.


```python
# Run this cell without changes

# Load the sample data 
no_queso = pickle.load(open('./no_queso.pkl', 'rb'))
queso = pickle.load(open('./queso.pkl', 'rb'))
```

### 2.1) Short Answer: State null and alternative hypotheses to use for testing whether customers who order queso spend different amounts of money from customers who do not order queso.


```python
# Your answer here


```

### 2.2) Short Answer: What would it mean to make a Type I error for this specific hypothesis test?

Your answer should be _specific to this context,_  not a general statement of what Type I error is.


```python
# Your answer here


```

### 2.3) Create a numeric variable `p_value` containing the p-value associated with a statistical test of your hypotheses. 

You must identify and implement the correct statistical test for this scenario. You can assume the two samples have equal variances.

Hint: Use `scipy.stats` to calculate the answer - it has already been imported as `stats`. Relevant documentation can be found [here](https://docs.scipy.org/doc/scipy/reference/stats.html#statistical-tests).


```python
# CodeGrade step2.3
# Replace None with appropriate code

p_value = 
```


```python
# This test confirms that you have created a numeric variable named p_value

assert isinstance(p_value, Number)
```

### 2.4) Short Answer: Can you reject the null hypothesis using a significance level of $\alpha$ = 0.05? Explain why or why not.


```python
# Your answer here


```

---
## Part 3: Bayesian Statistics [Suggested time: 15 minutes]
---
A medical test is designed to diagnose a certain disease. The test has a false positive rate of 10%, meaning that 10% of people without the disease will get a positive test result. The test has a false negative rate of 2%, meaning that 2% of people with the disease will get a negative result. Only 1% of the population has this disease.

### 3.1) Create a numeric variable `p_pos_test` containing the probability of a person receiving a positive test result.

Assume that the person being tested is randomly selected from the broader population.


```python
# CodeGrade step3.1
# Replace None with appropriate code
    
false_pos_rate = 0.1
false_neg_rate = 0.02
population_rate = 0.01

p_pos_test = None
```


```python
# This test confirms that you have created a numeric variable named p_pos_test

assert isinstance(p_pos_test, Number)

# These tests confirm that p_pos_test is a value between 0 and 1

assert p_pos_test >= 0
assert p_pos_test <= 1
```

### 3.2) Create a numeric variable `p_disease_given_pos` containing the probability of a person actually having the disease if they receive a positive test result.

Assume that the person being tested is randomly selected from the broader population.

Hint: Use your answer to the previous question to help answer this one.


```python
# CodeGrade step3.2
# Replace None with appropriate code
    
false_pos_rate = 0.1
false_neg_rate = 0.02
population_rate = 0.01

p_disease_given_pos = None
```


```python
# This test confirms that you have created a numeric variable named p_disease_given_pos

assert isinstance(p_disease_given_pos, Number)

# These tests confirm that p_disease_given_pos is a value between 0 and 1

assert p_disease_given_pos >= 0
assert p_disease_given_pos <= 1
```

---

## Part 4: Linear Regression [Suggested Time: 20 min]
---

In this section, you'll run regression models with [automobile price](https://archive.ics.uci.edu/ml/datasets/Automobile) data.

We will use these columns of the dataset:

- `body-style`: categorical, hardtop, wagon, sedan, hatchback, or convertible
- `length`: continuous
- `width`: continuous
- `height`: continuous
- `engine-size`: continuous
- `price`: continuous

We will use `price` as the target and all other columns as features. The units of `price` are US dollars in 1985.


```python
# Run this cell without changes

# Load data into pandas
data = pd.read_csv("automobiles.csv")

# Data cleaning
data = data[(data["horsepower"] != "?") & (data["price"] != "?")]
data["horsepower"] = data["horsepower"].astype(int)
data["price"] = data["price"].astype(int)

# Select subset of columns
data = data[["body-style", "length", "width", "height", "engine-size", "horsepower", "city-mpg", "price"]]
data
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
      <th>body-style</th>
      <th>length</th>
      <th>width</th>
      <th>height</th>
      <th>engine-size</th>
      <th>horsepower</th>
      <th>city-mpg</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>convertible</td>
      <td>168.8</td>
      <td>64.1</td>
      <td>48.8</td>
      <td>130</td>
      <td>111</td>
      <td>21</td>
      <td>13495</td>
    </tr>
    <tr>
      <th>1</th>
      <td>convertible</td>
      <td>168.8</td>
      <td>64.1</td>
      <td>48.8</td>
      <td>130</td>
      <td>111</td>
      <td>21</td>
      <td>16500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>hatchback</td>
      <td>171.2</td>
      <td>65.5</td>
      <td>52.4</td>
      <td>152</td>
      <td>154</td>
      <td>19</td>
      <td>16500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sedan</td>
      <td>176.6</td>
      <td>66.2</td>
      <td>54.3</td>
      <td>109</td>
      <td>102</td>
      <td>24</td>
      <td>13950</td>
    </tr>
    <tr>
      <th>4</th>
      <td>sedan</td>
      <td>176.6</td>
      <td>66.4</td>
      <td>54.3</td>
      <td>136</td>
      <td>115</td>
      <td>18</td>
      <td>17450</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>200</th>
      <td>sedan</td>
      <td>188.8</td>
      <td>68.9</td>
      <td>55.5</td>
      <td>141</td>
      <td>114</td>
      <td>23</td>
      <td>16845</td>
    </tr>
    <tr>
      <th>201</th>
      <td>sedan</td>
      <td>188.8</td>
      <td>68.8</td>
      <td>55.5</td>
      <td>141</td>
      <td>160</td>
      <td>19</td>
      <td>19045</td>
    </tr>
    <tr>
      <th>202</th>
      <td>sedan</td>
      <td>188.8</td>
      <td>68.9</td>
      <td>55.5</td>
      <td>173</td>
      <td>134</td>
      <td>18</td>
      <td>21485</td>
    </tr>
    <tr>
      <th>203</th>
      <td>sedan</td>
      <td>188.8</td>
      <td>68.9</td>
      <td>55.5</td>
      <td>145</td>
      <td>106</td>
      <td>26</td>
      <td>22470</td>
    </tr>
    <tr>
      <th>204</th>
      <td>sedan</td>
      <td>188.8</td>
      <td>68.9</td>
      <td>55.5</td>
      <td>141</td>
      <td>114</td>
      <td>19</td>
      <td>22625</td>
    </tr>
  </tbody>
</table>
<p>199 rows × 8 columns</p>
</div>



### 4.1) Build a StatsModels `OLS` model `numeric_mod` that uses all numeric features to predict `price`

In other words, this model should use all features in `data` except for `body-style` (because `body-style` is categorical).


```python
# CodeGrade step4.1
# Replace None with appropriate code
    
y = None
X = None

numeric_mod = None
```


```python
# This test confirms that you have created a variable named numeric_mod containing a StatsModels OLS model

assert type(numeric_mod) == sm.OLS
```


```python
# This code prints your model summary for reference to help answer the next question

numeric_results = numeric_mod.fit()
print(numeric_results.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  price   R-squared:                       0.820
    Model:                            OLS   Adj. R-squared:                  0.815
    Method:                 Least Squares   F-statistic:                     146.2
    Date:                Thu, 17 Nov 2022   Prob (F-statistic):           8.43e-69
    Time:                        09:13:29   Log-Likelihood:                -1899.0
    No. Observations:                 199   AIC:                             3812.
    Df Residuals:                     192   BIC:                             3835.
    Df Model:                           6                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    const        -7.07e+04   1.36e+04     -5.197      0.000   -9.75e+04   -4.39e+04
    length        -46.3698     50.107     -0.925      0.356    -145.200      52.461
    width         908.5449    244.437      3.717      0.000     426.419    1390.671
    height        273.9268    137.282      1.995      0.047       3.153     544.701
    engine-size    92.5917     12.613      7.341      0.000      67.713     117.470
    horsepower     62.2932     16.544      3.765      0.000      29.662      94.924
    city-mpg      -29.8524     77.298     -0.386      0.700    -182.314     122.609
    ==============================================================================
    Omnibus:                       22.902   Durbin-Watson:                   0.820
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               42.916
    Skew:                           0.582   Prob(JB):                     4.80e-10
    Kurtosis:                       4.955   Cond. No.                     1.44e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.44e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.


### 4.2) Short Answer: Are all of these features statististically significant? If not, which features are not? How did you determine this from the model output?

Include the alpha level you are using in your answer.


```python
# Your answer here


```

### 4.3) Short Answer: Let's say we want to add `body-style` to our model. Run the cell below to view the values of `body-style`. Given the output, how many one-hot encoded features should be added?

Explain your answer. ***Hint:*** you might want to mention the dummy variable trap and/or the reference category.


```python
# Run this cell without changes

data["body-style"].value_counts().sort_index()
```




    convertible     6
    hardtop         8
    hatchback      67
    sedan          94
    wagon          24
    Name: body-style, dtype: int64




```python
# Your answer here


```

### 4.4) Prepare `body-style` for modeling using `pd.get_dummies`. Then create a StatsModels `OLS` model `all_mod` that predicts `price` using all (including one-hot encoded) other features.


```python
# CodeGrade step4.4
# Replace None with appropriate code
    
X_ohe = None

all_mod = None
```


```python
# This test confirms that you have created a variable named all_mod containing a StatsModels OLS model

assert type(all_mod) == sm.OLS

# This code prints your model summary for reference to help answer the next question

all_results = all_mod.fit()
print(all_results.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  price   R-squared:                       0.840
    Model:                            OLS   Adj. R-squared:                  0.832
    Method:                 Least Squares   F-statistic:                     98.92
    Date:                Thu, 17 Nov 2022   Prob (F-statistic):           2.34e-69
    Time:                        09:14:21   Log-Likelihood:                -1887.3
    No. Observations:                 199   AIC:                             3797.
    Df Residuals:                     188   BIC:                             3833.
    Df Model:                          10                                         
    Covariance Type:            nonrobust                                         
    ========================================================================================
                               coef    std err          t      P>|t|      [0.025      0.975]
    ----------------------------------------------------------------------------------------
    const                -7.519e+04   1.32e+04     -5.718      0.000   -1.01e+05   -4.93e+04
    length                 -48.0812     51.558     -0.933      0.352    -149.789      53.626
    width                 1027.9245    241.176      4.262      0.000     552.165    1503.684
    height                 328.4788    142.750      2.301      0.022      46.880     610.077
    engine-size             78.4995     12.463      6.299      0.000      53.915     103.084
    horsepower              67.1887     15.839      4.242      0.000      35.944      98.434
    city-mpg               -10.5979     75.198     -0.141      0.888    -158.939     137.743
    body-style_hardtop   -3087.3076   1793.006     -1.722      0.087   -6624.304     449.689
    body-style_hatchback -6009.0658   1447.710     -4.151      0.000   -8864.909   -3153.222
    body-style_sedan     -4866.9489   1484.847     -3.278      0.001   -7796.051   -1937.847
    body-style_wagon     -6347.1371   1686.430     -3.764      0.000   -9673.895   -3020.379
    ==============================================================================
    Omnibus:                       21.919   Durbin-Watson:                   0.811
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               45.145
    Skew:                           0.521   Prob(JB):                     1.57e-10
    Kurtosis:                       5.088   Cond. No.                     1.46e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.46e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.


### 4.5) Short Answer: Does this model do a better job of explaining automobile price than the previous model using only numeric features? Explain how you determined this based on the model output. 


```python
# Your answer here


```


```python

```
