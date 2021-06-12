
## Install and Reproduce Results

To run the code in this package, one need only install the necessary packages used for the project. To this end, requirements.txt acts as a guide, or as a literal manual. When using requirements.txt together with pip, it is strongly advised to create a virtual environment to encapsulate code for this project. Run the following:

```python
pip install virtualenv
```

Then create a new environment for this project, for example 'urlscamdetection'. Afterwards, enter this new environment using the command below,


Finally run,

This should set up your newly created virtual environment to run the code in this repository.

## Data Description

## Feature Engineering

This forms the most important part of any machine learning exercise, with this project being no different.

Provided already in the dataset is the Alexa rank of the websites behind each of the URLs. But what is this rank exactly?

### What is Alexa Rank?

Alexa rank is a measure of website popularity. It ranks millions of websites in order of popularity, with an Alexa Rank of 1 being the most popular. Alexa Rank reveals how a website is doing relative to all other sites, which makes it a great KPI for benchmarking and competitive analysis. Alexa rank is calculated using a proprietary methodology that combines a site’s estimated traffic and visitor engagement over the past three months. Traffic and engagement are estimated from the browsing behavior of people in our global panel, which is a sample of all Internet users.


## Exploratory Analysis

In this section we discuss analyses done on the overall data for sanity checking the data before feature engineering and on the features constructed from the raw data before modelling.

### Alexa Ranking

What we see in the Alexa ranking is that the ranking has a bimodal distribution of values, where there is a right extrema located at 1e7. This indicates a very low Alexa ranking, and correlates very well with the label dependent variable, meaning that Alexa ranking is already an amazing feature to use in qualifying URLs as being fraudulent.

When looking at the remaining distribution, we see it is right-skewed, meaning that the median of sites have quite a good Alexa ranking, tailing off with low Alexa rankings for poorer quality sites.

