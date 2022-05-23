# Sentiment Analysis
Part of a project developed throughout 2021 and '22. Text sentiment analysis packages allowing to use **nltk** 
and **machine learning**.

This attempt allowed me to get **more accurate sentiment scores** than one of the most popular Python sentiment analysis
package - NLTK Vader. Getting in some cases the ~82% accuracy to NLTK's ~72%.


The **Sourcing** package provides functions allowing to get data from a local DB, cloud DB or .csv files.
The local DB - MySQL and the cloud DB - Firebase both are implementations of an interface defined in the package,
so that they can be later used in the same manner during the sentiment analysis.

The **SentAnalysis** package provides both NLTK functions and machine learning Tensorflow functions allowing
to built a custom sentiment prediction model.
 - The use of the NLTK is straight forward. The data is brought from a DB using batch generators and evaluated.

 - The second approach consists of more steps. First, one loads a Google's Universal Sentence Encoder, a state of the art
linguistic model which allows to seek for semantic similarity between phrases. This encoder is then used to encode
a Kaggle's dataset of 1.6mln tweets with their rated sentiment scores. Then one can create a simple Deep Learning model
consisting of a few layers which is then being trained on the encoded Kaggle dataset. This model then can then be tested
on a smaller set of a few hundred tweets, which tends to give more accurate results than NLTK.


