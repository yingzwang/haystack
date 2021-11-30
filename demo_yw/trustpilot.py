import csv
import time
import json
import requests
import lxml.html as html
import os
import numpy as np
import pandas as pd
import re


def get_next_page_url(tree: html.HtmlElement, base_url: str) -> str:
    next_page = tree.xpath("//a[contains(@class, 'next-page')]")
    if next_page:
        next_page_url = f"{base_url}{next_page[0].get('href')}"
    else:
        next_page_url = None
    return next_page_url


def get_one_review(elem: html.HtmlElement, elem_d: html.HtmlElement):
    curr_item = json.loads(elem.text_content())
    curr_dt = json.loads(elem_d.text_content())
    timestamp = curr_dt["publishedDate"]  # '2021-10-23T06:26:37+00:00'
    title = curr_item["reviewHeader"] # str
    body = curr_item["reviewBody"] # str
    rating = curr_item["stars"] # int
    return timestamp, title, body, rating


def isolate_punctuation(s):
    """
    put a space in front of punctuation. [Why?]
    ref: https://github.com/hakimkhalafi/trustpilot-scraper/blob/master/clean.ipynb
    """
    s = re.sub('([.,!?();:"])', r' \1 ', s) # isolate punctuation
    s = ' '.join(s.split()) # Remove space multiples
    return s


def scrape_trustpilot():
    """
    ref:
    https://github.com/hakimkhalafi/trustpilot-scraper/blob/master/scrape.ipynb
    https://www.basecamp.ai/blog/tutorial-behavioral-analysis
    """

    # Trustpilot review page
    base_url = "https://nl.trustpilot.com"
    company_name = 'www.ziggo.nl'
    site_url = os.path.join(base_url, "review", company_name)
    output_path = os.path.join("data/trustpilot", 'ziggo_nl.csv') # output file
    csv_sep = '\t' # Use tab delimiter to allow for special characters

    page = requests.get(site_url, verify=False) # skip HTTPS as it gives certificate errors
    tree = html.fromstring(page.content)
    num_reviews = tree.xpath('//span[@class="headline__review-count"]') # total number of ratings
    thousand_sep = "." # dot for nl; comma for en
    num_reviews = int(num_reviews[0].text.replace(thousand_sep,''))
    sleepTime = 1 # pause per page
    print(f"Site to scrape: {site_url}")
    print(f"Save output to: {output_path}")
    print(f"Found {num_reviews} reviews")

    ## Main scraping section
    with open(output_path, 'w', newline='', encoding='utf8') as csvfile:
        datawriter = csv.writer(csvfile, delimiter=csv_sep)
        next_page_url = site_url # first page
        rr_in_total = 0
        # Loop over all pages
        while (next_page_url is not None) and (rr_in_total < num_reviews):
            time.sleep(np.random.normal(sleepTime, 0.1))
            page = requests.get(next_page_url)
            tree = html.fromstring(page.content)
            script_bodies = tree.xpath("//script[starts-with(@data-initial-state, 'review-info')]")
            script_dates = tree.xpath("//script[starts-with(@data-initial-state, 'review-date')]")
            # Loop over all reviews in a page
            for rr, (elem, elem_d) in enumerate(zip(script_bodies, script_dates)):
                timestamp, title, body, rating = get_one_review(elem, elem_d)
                datawriter.writerow([timestamp, title, body, rating])
                rr_in_total += 1
            next_page_url = get_next_page_url(tree, base_url)
            print(f"Processed {rr_in_total}/{num_reviews} reviews")

        print(f"Finished processing {rr_in_total} ratings!")


def clean_trustpilot_reviews():
    csv_path = os.path.join("data/trustpilot", 'ziggo_nl.csv')  # output file
    csv_sep = '\t'  # Use tab delimiter to allow for special characters
    names = ["timestamp", "title", "body", "rating"]
    df = pd.read_csv(csv_path, sep=csv_sep, names=names)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # TODO: pre-processing for BERTopic,
    #  - drop duplicates;
    #  - remove short reviews;