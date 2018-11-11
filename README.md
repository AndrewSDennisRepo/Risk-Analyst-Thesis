# Risk-Analyst-Thesis
This repository covers the code used in my MS Data Science Thesis.

## Write Up
This script is designed for my thesis. The main point of thesis is to determine if sentiment has more of an impact of stock price movement than EPS and Sales beat/miss.

## Sentiment
For sentiment analysis originally was an LSTM utilizing crowdsourced truth data. Unfortunately not enough data was able to be annotated (due to cost) to have a strong precision and recall.

### Dictionary - Lexicon 
Loughran-McDonald Sentiment Word Lists was used instead to determine sentiment specifically Negative, Positive, and Uncertainty
[Dictionary / Lexicon by Loughran-McDonald](https://sraf.nd.edu/textual-analysis/)

## Results
Random Forest was able to classify with ~77% precision with all variables. 
![Results](https://github.com/syphercrypt/Risk-Analyst-Thesis/blob/master/charts/all_vars.JPG "All Variables")

(please see charts folder for more information) 
