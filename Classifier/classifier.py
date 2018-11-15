""" This script is meant to utilized the sentiment lexicon.
This will count each negative word in each sentence. 
No words have been given a weight just a simple count. 

Author: Andrew Dennis
"""
import numpy as np
import pandas as pd
import re

from sentiment import positive, negative, uncertainty

df = pd.read_csv(r'D:\MS Data Science Files\Thesis\model_data.csv')


def neg_count(phrase):
	"""Negative counter."""
	neg_count= 0
	for word in re.split(' ',phrase):
		if word.lower() in negative:
			neg_count = neg_count - 1
	return neg_count


def pos_count(phrase):
	"""Positive counter."""
	pos_count= 0
	for word in re.split(' ',phrase):
		if word.lower() in positive:
			pos_count = pos_count + 1
	return pos_count


def unc_count(phrase):
	"""Uncertainty counter."""
	unc_count= 0
	for word in re.split(' ',phrase):
		if word.lower() in uncertainty:
			unc_count = unc_count + 1
	return unc_count


df['NEG'] = df['tokenized'].apply(neg_count)
df['POS'] = df['tokenized'].apply(pos_count)
df['UNC'] = df['tokenized'].apply(unc_count)


# df.to_csv('D:\\MS Data Science Files\\Thesis\\classified_data.csv')
