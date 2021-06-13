# URL Phishing Detection Using Machine Learning
## _Using Phishing URLs Against Themselves_

In this task we cover the important topic of phishing and other malicious attempts to steal users' personal data, financial information and money or banking access. This extends even past individual users and affects all sectors of society, from government to private companies: Essentially, wherever human error can present itself, this problem is a prevalent and important issue to cover. 

The main question is: How can we predict whether a URL being accessed on the internet is likely to be malicious in some way? There are many ways to approach this. Often, a human expert trained in web and more generally in IT technologies will have no issues looking out for tell-tale signs of URL fraudulency. This is not so easy for the average internet user to do, however. So, how do we train a machine learning model to do this? It turns out that this is already quite a solveable problem by simply looking at the structure of the URL being presented to the user while navigating to the fraudulent/malicious site: Key giveaways, such as the lack of a secure HTTPS connection, mispellings in the subdomain or top-level domain, long and convoluted links, and sometimes even the lack of a domain entirely (just an IP address is visible in this case) are key points to consider when looking at URLs for suspicious structure and likely phishing behaviour lurking behind the URL.

Naturally, there are other ways to help prevent users falling victim to phishing and other malicious attacks beyond looking just at the URL itself and these are discussed in the last section **Conclusions and Recommendations**. One example that immediately comes to mind is looking at the HTML structure of the website sitting behind the URL, to see the type of content presented, the number of video and audio tags in the URL etc. in order to detect the characteristic structure of a spam site looking to defraud users.

Once the machine learning model is built, its applications are quite diverse. It can be 

- Deployed as part of anti-virus and anti-malware software packages to protect users who browse the internet. McAfee and Avast already have such capabilities
- Browser extensions which help protect the user from engaging in business with sites which are detected to be likely sources of phishing scams and other malicious purposes.
- Installed on virtual machines to help protect users who are deploying valuable software and company infrastructure through these machines. The first part of prevention, after all, is detection.

This project is designed to answer the question of modelling URL phishing likelihood by performing the following:

1. Transforming the URLs into a dataframe, and extract features you'd want to explore.
2. Performing exploratory analysis of the dataset and summarize and explain the key trends in the data, explaining which features can be used to identify phishing attacks.
3. Building a model to predict if a URL is a phishing URL.
4. Reporting on the model's success and show what features are most important in that model.

**Reporting on the data, feature engineering, methodology, results and discussions/conclusions can be found in ** _./src/reporting/_ in the files _Reporting.md_, and _Report.pdf_.
## Install and Reproduce Results

To run the code in this package, one need only install the necessary packages used for the project. To this end, requirements.txt acts as a guide, or as a literal manual. When using requirements.txt together with pip, it is strongly advised to create a virtual environment to encapsulate code for this project. This can be done using the package virtualenv via run the following:

```sh
pip install virtualenv
```

Afterwhich one can run,

```sh 
python virtualenv create env -n urlscamdetection
```

The other option is to use Anaconda's native virtual environment support as follows:

```sh
conda create -n urlscamdetection 
```

And activating it as follows:

```sh
conda activate urlscamdetection
```

Finally, regardless of which method was used to instantiate the virtual environment, you can install packages via requirements.txt using:

```sh
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

```sh
python ./src/01\_exploratory\_analysis.py
```

It is however recommended you do this via an iPython or jupyter environment for true interactive viewing. Plots towards the PDF report can be found in *analysis*

Following from the previous section on feature engineering, we can see good potential in many of the features. For example,


## Running Models

There are various experiments available for running. In the *src* folder you can find and view scripts, described as:

1. _01\_exploratory\_analytics.py_ which has all results for the exploratory phase of the project, producing and storing all graphics/plots in the _./analytics_ folder.
2. _01\_model\_selection\_and\_tuning.py_ which has all the model selection and hyperparameter tuning experiments run using Hyperopt-Sklearn
3. _02\_model\_reporting.py_ which has the final tests run on various feature sets using the most promising models, with XGBoost feature importances quoted in plots.
4. _feature\_helper\_functions.py_ has many well-documented feature definitions for ease-of-use with Scikit-Learns FunctionTransformer and Pipeline interfaces. TODO: Add these to a URLTransformations static method class.
5.	_03\_packaged\_model.py_ is the class-ified packaged model ready to ship for deployment on the cloud, such as on GCP via the Endpoints API on Vertex AI or on AWS SageMaker.

### Using Pre-Trained Model Pipelines for Reproducibility

The model code should already allow you to reproduce all the results quoted in the report, however, if you would like to view the results through the perspective of the trained models, you can access the models [the following Google Drive folder which is publicly viewable](https://drive.google.com/drive/folders/1Khe2OZ04HBmRinO2lSp4ubHkJHXem4vD?usp=sharing). Models are stored here so as to save space on GitHub, as it is bad practice to store large files here.

