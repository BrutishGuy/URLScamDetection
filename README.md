
## Install and Reproduce Results

To run the code in this package, one need only install the necessary packages used for the project. To this end, requirements.txt acts as a guide, or as a literal manual. When using requirements.txt together with pip, it is strongly advised to create a virtual environment to encapsulate code for this project. This can be done using the package virtualenv via run the following:

```python
pip install virtualenv
```

Afterwhich one can run,

```python 
python virtualenv create env -n urlscamdetection
```

The other option is to use Anaconda's native virtual environment support as follows:

```python
conda create -n urlscamdetection 
```

And activating it as follows:

```python
conda activate urlscamdetection
```

Finally, regardless of which method was used to instantiate the virtual environment, you can install packages via requirements.txt using:

```python
pip install -r requirements.txt
```

This should set up your newly created virtual environment with all the packages necessary to run the code in this repository.

## Data Description

The data is a set of URLs used for phishing attacks, along with legitimate URLs. Every URL is also accompanied by its Alexa ranking. This is an important task to tackle in general, since many users browsing the internet do not themselves have a good eye or intuition as to what is malicious or a phishing attempt. As a result, this means many people are easily susceptible to have their information, credit card data, and money stolen online. Conversely, legitimate links or URLs are flooded by these scam URLs. It is therefore in the interest of search engines, internet providers and anti-malware/anti-phishing software to help prevent this and keep users safe.

In this project, we have the following tasks to complete:

	1. Transform the URLs into a dataframe, and extract features you'd want to explore.
	2. Perform exploratory analysis of the dataset and summarize and explain the key trends in the data, explaining which features can be used to identify phishing attacks.
	3. Build a model to predict if a URL is a phishing URL.
	4. Report on the model's success and show what features are most important in that model.

One final piece of information which will be useful in the modelling section is to discuss what would be an important evaluation metric for this model? It is more than just building a model with high accuracy. For one, it is very important to identify as many malicious/phishing URLs as possible. This corresponds to a model with a very high recall: Out of all the positive cases (malicious/phishing URLs), the model is able to capture as many as possible. On the converse, we'd also like to maintain traffic uninterrupted to URLs which are legitimate, hence making precision also a valuable, but secondary, objective for this project: The precision of a model is the proportion of positive cases (malicious/phishing URLs) predicted by the model which were actually positive cases. 

## Feature Engineering

This forms the most important part of any machine learning exercise, with this project being no different.

Provided already in the dataset is the Alexa rank of the websites behind each of the URLs. But what is this rank exactly?

### What is Alexa Rank?

Alexa rank is a measure of website popularity. It ranks millions of websites in order of popularity, with an Alexa Rank of 1 being the most popular. Alexa Rank reveals how a website is doing relative to all other sites, which makes it a great KPI for benchmarking and competitive analysis. Alexa rank is calculated using a proprietary methodology that combines a site’s estimated traffic and visitor engagement over the past three months. Traffic and engagement are estimated from the browsing behavior of people in our global panel, which is a sample of all Internet users.

### Alexa Ranking

What we see in the Alexa ranking is that the ranking has a bimodal distribution of values, where there is a right extrema located at 1e7. This indicates a very low Alexa ranking, and correlates very well with the label dependent variable, meaning that Alexa ranking is already an amazing feature to use in qualifying URLs as being fraudulent.

When looking at the remaining distribution, we see it is right-skewed, meaning that the median of sites have quite a good Alexa ranking, tailing off with low Alexa rankings for poorer quality sites.



## Exploratory Analysis

In this section we discuss analyses done on the overall data for sanity checking the data before feature engineering and on the features constructed from the raw data before modeling. The code used to generate the plots and graphics shown in the PDF report can be found in the *src* folder as *01_exploratory_analysis.py*. This can be run from the home directory of the repository via

```python
python ./src/01_exploratory_analysis.py
```

It is however recommended you do this via an iPython or jupyter environment for true interactive viewing. Plots towards the PDF report can be found in *analysis*

Following from the previous section on feature engineering, we can see good potential in many of the features. For example,


## Running Models

There are various experiments available for running. In the *src* folder you can find a view scripts, described as:

	1. 
### Using Pre-Trained Model Pipelines for Reproducibility

The model code should already allow you to reproduce all the results quoted in the report, however, if you would like to view the results through the perspective of the trained models, you can access the models [the following Google Drive folder which is publicly viewable](https://drive.google.com/drive/folders/1Khe2OZ04HBmRinO2lSp4ubHkJHXem4vD?usp=sharing). Models are stored here so as to save space on GitHub, as it is bad practice to store large files here.

