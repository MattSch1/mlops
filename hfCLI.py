#!/us/bin/env python

import click
from transformers import pipeline
import urllib.request
from bs4 import BeautifulSoup
import wikipedia

# make a function that extracts the text from a url
def extract_text(url):
    page = urllib.request.urlopen(url)
    soup = BeautifulSoup(page, "html.parser")
    text = " ".join(map(lambda p: p.text, soup.find_all("p")))
    return text


# write a function that uses hugging face to return a summary of the text
def summarize(text):
    summarizer = pipeline("summarization", model="t5-small")
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary
