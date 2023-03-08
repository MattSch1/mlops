#!/us/bin/env python

from transformers import pipeline
import urllib.request
from bs4 import BeautifulSoup

def extract_text(url):
    page = urllib.request.urlopen(url)
    soup = BeautifulSoup(page, "html.parser")
    text = " ".join(map(lambda p: p.text, soup.find_all("p")))
    return text


def summarize(text):
    summarizer = pipeline("summarization", model="t5-small")
    summary = summarizer(text, max_length=180)
    return summary[0]["summary_text"]
