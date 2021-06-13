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
4. Modeling 
5. Conclusions and Recommendations

## Data Description

- Import a HTML file and watch it magically convert to Markdown
- Drag and drop images (requires your Dropbox account be linked)
- Import and save files from GitHub, Dropbox, Google Drive and One Drive
- Drag and drop markdown and HTML files into Dillinger
- Export documents as Markdown, HTML and PDF

Markdown is a lightweight markup language based on the formatting conventions
that people naturally use in email.
As [John Gruber] writes on the [Markdown site][df1]

> The overriding design goal for Markdown's
> formatting syntax is to make it as readable
> as possible. The idea is that a
> Markdown-formatted document should be
> publishable as-is, as plain text, without
> looking like it's been marked up with tags
> or formatting instructions.

This text you see here is *actually- written in Markdown! To get a feel
for Markdown's syntax, type some text into the left window and
watch the results in the right.

## Tech

Dillinger uses a number of open source projects to work properly:

- [AngularJS] - HTML enhanced for web apps!
- [Ace Editor] - awesome web-based text editor
- [markdown-it] - Markdown parser done right. Fast and easy to extend.
- [Twitter Bootstrap] - great UI boilerplate for modern web apps
- [node.js] - evented I/O for the backend
- [Express] - fast node.js network app framework [@tjholowaychuk]
- [Gulp] - the streaming build system
- [Breakdance](https://breakdance.github.io/breakdance/) - HTML
to Markdown converter
- [jQuery] - duh

And of course Dillinger itself is open source with a [public repository][dill]
 on GitHub.

## Installation

Dillinger requires [Node.js](https://nodejs.org/) v10+ to run.


| Plugin | README |
| ------ | ------ |
| Dropbox | [plugins/dropbox/README.md][PlDb] |




```sh
cd dillinger
docker build -t <youruser>/dillinger:${package.json.version} .
```

> Note: `--capt-add=SYS-ADMIN` is required for PDF rendering.
