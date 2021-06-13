# VMWare URL Phishing Detection Challenge
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

Finally, this document is structured as follows:

1. Data Description
2. Feature Engineering
3. Exploratory Analysis
4. Methodology
4. Results 
5. Conclusions and Recommendations

## Data Description


The data is a set of URLs used for phishing attacks, along with legitimate URLs. Every URL is also accompanied by its Alexa ranking. This is an important task to tackle in general, since many users browsing the internet do not themselves have a good eye or intuition as to what is malicious or a phishing attempt. As a result, this means many people are easily susceptible to have their information, credit card data, and money stolen online. Conversely, legitimate links or URLs are flooded by these scam URLs. It is therefore in the interest of search engines, internet providers and anti-malware/anti-phishing software to help prevent this and keep users safe.

In this project, we have the following fields provided to us:

| Field | Description |
| ------ | ------ |
| domain | The URL or link pointing to the potentially malicious/phishing site. |
| ranking | This is the Alexa ranking of the above URL and site. The higher the number, the worse the Alexa ranking is (this is because it is a ranking score).|
| label | This is the label telling us the ground truth of whether the URL is a malicious phishing site (1) or not (0). |

One final piece of information which will be useful in the modelling section is to discuss what would be an important evaluation metric for this model? It is more than just building a model with high accuracy. For one, it is very important to identify as many malicious/phishing URLs as possible. This corresponds to a model with a very high recall: Out of all the positive cases (malicious/phishing URLs), the model is able to capture as many as possible. On the converse, we'd also like to maintain traffic uninterrupted to URLs which are legitimate, hence making precision also a valuable, but secondary, objective for this project: The precision of a model is the proportion of positive cases (malicious/phishing URLs) predicted by the model which were actually positive cases. 

## Feature Engineering

In this section we discuss what additional features can be used to identify useful properties of a URL to be used for predicting whether it is malicious or not. Additionally, we describe which existing features (in this case just the Alexa ranking) can be transformed or preprocessed into a more useful form.

To this effect, we divide features into two different sets: Set 1) Derived features taken from a hashing trick on the characters of the URL with bigrams and trigrams, the subsequenting vectors transformed via TF-IDF and reduced via a truncated SVD (to better handle sparse representations). As we will see during the modeling and methodology and results sections, this is already a great feature engineering step which can be fine-tuned further with hyperparameter tuning of the hashing trick first step. Set 2) of features contains hand-crafted lexical features from the URL, such as the length of the URL and number of special characters in the URL. Later in the results, we shall see that while this helps squeeze out the last few drops of performance from the data for our model, it doesn't completely outclass the previous algorithmic feature engineering. This makes sense, since a hashing trick to vectorize the URLs on character level using bigrams and trigrams already captures many such dynamics such as presence of special characters and complexity of the URL.

The types of lexical features defined are described below:

| Feature Name | Data Type | Description |
| ------ | ------ | ------ |
| Alexa Scaled Ranking | Numeric | The Alexa Scaled Ranking is the provided Alexa ranking transformed via a log transform and then scaled via standard scaling to a range -1 to 1 to make it easier for modelling (see EDA section below). |
| URL Length | Numeric | Length of the URL |
| Num. Special URL Symbols | Numeric | Number of semicolons, underscores, question marks, equals, ampersands |
| Ratio Digit Letter | Numeric | Digit to letter ratio |
| Contains IP | Boolean | Contains IPv4 or IPv6 Address |
| Num. Digits | Numeric | Number of digits |
| Ratio Digits Non-Alpha | Numeric | Number of digits to non-alphanumeric characters |
| Num. Prefix Suffix | Numeric | Number of hyphens |
| Num. At Symbols | Numeric | Number of @s |
| Is In Alexa 100 | Boolean | Presence in top 100 Alexa domains |
| Num. Dots | Numeric | Number of dots |
| Num. Sub-Domains | Numeric | Number of subdomains |
| Num. Paths | Numeric | Number of ‘//’ |
| Num. Query Terms | Numeric | Presence of ‘%20’ in path |
| Num. Special Chars | Numeric | Number of special characters |
| Letter ratio | Numeric | Ratio of uppercase to lowercase characters |

These lexical features can be expanded on much further, as there is always a rich set of information one can store inside a URL. Additionally, some of these can be broken up depending on the overall feature importance, for example, after iterating a few times with different models and hyperparameters, one might find that Num. Special URL Symbols can be broken down into its consituent symbols as binary features or as count features, depending on the importance of each in classifying a URL. All in all, there is much room to explore here. Additionally, to filter out variables, one can perform a Recursive Feature Elimination step before/after modelling, but this is not explored in this project.

## Exploratory Analysis
We finally come to the exploratory analysis after looking at the data description and feature engineering done on top of the data. We specifically want to analyze the relationship between the hashing trick derived features to the dependent variable, label, and then do the same with the lexical URL features we defined above, to see if any have any added value to be used as a predictor against the dependent variable in the model. One additional note to mention is that the dependent variable is already **quite balanced**, hence not re-balancing techniques such as oversampling or undersampling need to be employed later on before modelling.

We should also examine collinearity among variables to ensure that we don't harm the convergence of the models we try, nor skew the coefficients of models such as our baseline (see below in Methodology) Logistic Regression model.

To this effect, for the hashing trick based features, as was described above, we take the final output from the Truncated SVD step mentioned and create a set of boxplots grouped on the dependent variable gain a quick glimpse into how each features relates to the others and to the dependent variable, label. This is seen in the figure below.

For the lexical features, we perform similar analyses via a scatter plot matrix for numerical features, and grouped bar plots of instance counts grouped on the dependent variable. This can be seen in the figures below. 

## Methodology

We now have a look at modelling the dataset using the defined features against the dependent variable to see how well we can predict if a URL is a malicious site or a phishing scame, or a regular legitimate site.

 **It should be noted that the data is already very balanced** with 49.9 percent of the data belonging to the positive class. Hence, we do not have to worry about performing oversampling or undersampling or providing custom weights to models which accept them, such as XGBoost or decision trees etc. In principle though, one could use packages such as ROSE or imbalance-learn to perform such corrections or re-weight samples by inverse class proportion.

As discussed in the data description section prior, one important consideration is the evaluation metric. With the previous paragraph in mind, we continue by noting that it is very important to identify as many malicious/phishing URLs as possible. This corresponds to a model with a very high recall: Out of all the positive cases (malicious/phishing URLs), the model is able to capture as many as possible. On the converse, we'd also like to maintain traffic uninterrupted to URLs which are legitimate, hence making precision also a valuable, but secondary, objective for this project: The precision of a model is the proportion of positive cases (malicious/phishing URLs) predicted by the model which were actually positive cases. Therefore, for us in this project we will want to focus on the recall, precision and perhaps F1 (in that order of preference) as a summary of the two as our primary metrics of interest. ROC-AUC and accuracy can also be good to look at given that our dataset is balanced. We quote these 5 metrics in each of the cases we examine.

To this effect of modelling we take the following steps:

1. We first need to set a baseline. For this we use a Logistic Regression model, using LogisticRegressionCV from Scikit-Learn with 5-folds in order to set some default parameters and obtain a benchmark. This is around 92 recall, 91 precision.
2. We then have a look at doing model selection and hyperparameter tuning on a set of models available from Scikit-Learn. To do this, we look at using Bayesian optimization provided by Hyperopt-Sklearn to perform joint model selection and hyperparameter tuning on classification models from Scikit-Learn. Once we've arrived at a set of promising models and parameters, we test them on various combinations of our feature sets, We also perform hyperparameter tuning on one of the feature extraction techniques, namely the hashing trick method, in adjusting the n-gram parameter, and the number of features to hash to in the final vector representation. 
3. Namely, we test promising models on hashing trick features, lexical featues and all features combined. We also make sure there is no multi-collinearity between the variables. One quick way to do this is to perform a hierarchical clustering on the Spearman's rank correlation between numerical features. For us, we see there is minor clustering apparent in both the hashing trick features and the lexical numerical features, but nothing so strongly correlated to cause massive concern for multi-collinearity. 
4. We then have a look at the results of which in the following section in terms of the metrics we defined and also examine the feature importances of the most promising model on the lexical features defined (since the hashing trick features have the disadvantage of not being as interpretable.) **Naturally, all tests are run on data which is split into training, validation and test sets. Validation sets are used for model comparisons while test sets are used for final results quoted below in the subsequent section Results**
5. We then package our model up to be ready for deployment and ready to ship on the cloud, such as on GCP via the Endpoints API on Vertex AI or on AWS SageMaker.

**NB** When evaluating feature importances, we use the permutation based feature importance method. Often, people resort to using feature importance methods that come free with tree based models. However, this often creates a bias since these features are based on how often features appear in the splits of these tree based models. Hence, removing one feature can completely reshuffle feature importances across the entire feature set, making this very instable to use for interpretability. Another approach would be use Shapley values via the Python package _shap_ to have a look at the effects of features on the dependent variable. We leave this as out of scope, as this has a similar spirit to the permutation based feature importances, being a game theory inspired approach which attributes contributions of one feature on the outcome/dependent variable.

## Results

Finally, we arrive at the results of the entire project. The results presented here are the model performance metrics discussed before, namely the recall, precision, F1, ROCAUC and accuracy (the last two on account of the balanced dataset provided), in that order of importance to our investigation into the feasibility of predicting a URL as malicious or not. We also show permutation feature importances for the best performing model, XGBoost on our lexical feature sets (this is done as these hand-crafted features are most interpretable).

Firstly, we consider the results of models on three features sets: The hashing trick features, the lexical featuers, and all features together. These are presented in the tables below. 

#### Hashing Feature Model Performances
| MODEL | ACCURACY | F1 | PRECISION | RECALL | ROCAUC |
| ------ | ------ | ------ | ------ | ------ | ------ |
|XGBoost |	93|	93|	92|	94|	98|
|Random Forest|	93|	92|	90|	94|	98|
|MLP|	93|	92|	93|	93|	98|
|Extra Trees|	93|	93|	93|	93|	98|
|SGD|	91|	91|	90|	93|	97|
|Decision Trees|	91|	91|	90|	92|	97|
|Log Regr. (Baseline) |	89|	89|	88|	90|	95|

Then we have a look at performance on lexical features:
s
#### Lexical/Hand-Crafted Feature Model Performances
| MODEL | ACCURACY | F1 | PRECISION | RECALL | ROCAUC |
| ------ | ------ | ------ | ------ | ------ | ------ |
|XGBoost |	91|	91|	92|	91|	98|
|Random Forest|	91|	91|	90|	91|	97|
|MLP|	91|	91|	91|	91|	97|
|Decision Trees|	90|	90|	90|	90|	96|
|Extra Trees|	88|	88|	85|	90|	96|
|SGD|	85|	84|	80|	88|	91|
|Log Regr. (Baseline) |	85|	84|	81|	87|	91|

We can see that even in the absence of hand-crafted informative features, already the hashing trick with bigrams and trigrams provides remarkable results in predicting URL malicious behaviours. This is intuitive to some extend, since the hashes of bigrams and trigrams already conceivably capture much of the same information as the lexical features. For example, counts of digits, counts of '%20' characters and so on are captured by this hashing trick method. With the hashing trick, we already arrive at precision and recall scores of 92 and 93 respectively and and ROCAUC of 97, the strongest model being XGBoost. 

#### All/Combined Feature Model Performances

| MODEL | ACCURACY | F1 | PRECISION | RECALL | ROCAUC |
| ------ | ------ | ------ | ------ | ------ | ------ |
|XGBoost |	95|	95|	95|	96|	99|
|Random Forest|	94|	94|	93|	95|	99|
|MLP|	95|	95|	95|	95|	99|
|Extra Trees|	94|	93|	92|	95|	99|
|SGD|	91|	91|	88|	94|	97|
|Decision Trees|	92|	92|	91|	93|	97|
|Log Regr. (Baseline) |	92|	92|	91|	93|	98|

The subsequent addition of the lexical or hand-crafted features lets us really squeeze out the final performance we are looking for, giving us 95 precision and 96 recall. The ROCAUC already being a remarkable 99. Overall, this is remarkable given the only information we have is from the content of the URL itself, and nothing of the website sitting behind the URL. The Recall of 96 can be interpreted very well as follows: Out of all the malicious/phishing URLs in the data, for every 100 our model correctly captures 96 of them. This is a very good result indeed, although some may say that there are still 4 malicious sites for every 100 out there still phishing users. The precision conversely says that for every 100 URls the model decides are malicious, 95 are indeed malicious. This would mean that 5 legitimate sites would potentially be unintentionally blacklisted by the model.

The feature importances of the XGBoost model are shown for the lexical features below. The first feature importance plot below is that of XGBoost on lexical features alone.

![XGBoost feature importance on lexical features](https://github.com/BrutishGuy/URLScamDetection/blob/master/analysis/feature_importances_all_features.PNG)

## Conclusion and Discussion

To wrap up this short analysis, it should be noted that by simply using information from the contents of a URL, once can already very reliably predict whether a URL is malicious or an attempt at phishing. This is quite remarkable given that a URL is only a doorway into the website where this activity takes place. Nevertheless, our models using hashing trick features (with bigrams and trigrams and truncated SVD feature reduction) and lexical hand-crafted features managed a recall score of 96, precision of 94, and an ROCAUC of 99 using a XGBoost model.

This is already great performance, but one might argue that we need to do better. A recall of 96 means that for every 100 malicious URLs in the data, our model is able to correctly catch 96 of them. That is still 4 sites or URLs which the public could potentially fall victim to. This is not ideal as we would want this to go down as close to zero as possible. How can we do better? Obviously, via more feature engineering. Data is everything, and some ways this analysis can be improved in future can be via the following ways:

1. Currently features are all based on the URL itself. Many packages in Python such as BeautifulSoup allow us to quickly and easily grab the contents of a webpage without triggering any nasty side-effects, allowing us to read data on the website in a programmatically standardized way. Using this method, we can already start building new features by analyzing the webpage itself. From here we can:
2. Develop features based on webpage properties, such as various tags on the website, video or audio tags, embeddings etc.
3. Develop features based on text found on the webpage. Specific language or use of specific terms are dead giveaways of scamming sites, and these can be detected by topic modelling and other NLP techniques. One could also apply newer deep learning approaches such as BERT or GPT-3 to extract features or to directly query against the textual content of the website.

These tips alone should help tremendously in capturing additional websites and URLs which fall under the fraudulent category for whatever reason. It would also allow one to build more reliable systems which are more robust and future-proof against ever-developing malware attacks and advanced phishing attacks.

Some technical improvements on this note can also be done:

1. The use of data drift detection methods such as those available in Spark and in Python packages such as Scikit-Multiflow allow one to get a hold for when behaviours by scammers are changing over time. For example, should URLs beging shifting more towards using URL shorteners such as tinyurl, or if webpage structures being becoming more advanced to hide phishing activites, we will be able to have an early or at least on-time warning system via the data distribution shift over time.
2. Using online learning algorithms and self-supervised or semi-supervised learning algorithms will allow models to learn over time in partnership with the above methods of data drift detection. Additionally, semi-supervised methods can periodically query human experts to label data points (URLs or websites) deemed to need a helping hand in labelling due to being novel or outliers in nature.

All in all, while this is a very simple toy dataset, this is an active area of improvement and research that can be explored.

