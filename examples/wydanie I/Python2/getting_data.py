# -*- coding: utf-8 -*-
from __future__ import division
from collections import Counter
import math, random, csv, json

from bs4 import BeautifulSoup
import requests

######
#
# Książki wydawnictwa O'Reilly dotyczące analizy danych
#
######

def is_video(td):
    """Za materiał wideo uznaje się obiekt zawierajacy dokładnie jeden element one pricelabel,
    w którym znajduje się tekst rozpoczynający się od słowa Video."""
    pricelabels = td('span', 'pricelabel')
    return (len(pricelabels) == 1 and
            pricelabels[0].text.strip().startswith("Video"))

def book_info(td):
    """Przekazanie zawartości znacznika td do biblioteki Beautiful Soup
    spowoduje odczytanie informacji o książce i zwrócenie ich w formie słownika"""
    
    title = td.find("div", "thumbheader").a.text
    by_author = td.find('div', 'AuthorName').text
    authors = [x.strip() for x in re.sub("^By ", "", by_author).split(",")]
    isbn_link = td.find("div", "thumbheader").a.get("href")
    isbn = re.match("/product/(.*)\.do", isbn_link).groups()[0]
    date = td.find("span", "directorydate").text.strip()
    
    return {
        "title" : title,
        "authors" : authors,
        "isbn" : isbn,
        "date" : date
    }

from time import sleep

def scrape(num_pages=31):
    base_url = "http://shop.oreilly.com/category/browse-subjects/" + \
           "data.do?sortby=publicationDate&page="

    books = []

    for page_num in range(1, num_pages + 1):
        print "souping page", page_num
        url = base_url + str(page_num)
        soup = BeautifulSoup(requests.get(url).text, 'html5lib')
            
        for td in soup('td', 'thumbtext'):
            if not is_video(td):
                books.append(book_info(td))

        # Działaj zgodnie z zasadami opisanymi w pliku robots.txt!
        sleep(30)

    return books

def get_year(book):
    """Element book["date"] zawiera tekst typu November 2014, a więc musimy
    podzielić go na znaku spacji i korzystać dalej z jego drugiej połowy."""
    return int(book["date"].split()[1])

def plot_years(plt, books):
    # W odczytanych przeze mnie danych rok 2014 był ostatnim pełnym cyklem.
    year_counts = Counter(get_year(book) for book in books
                          if get_year(book) <= 2014)

    years = sorted(year_counts)
    book_counts = [year_counts[year] for year in x]
    plt.bar([x - 0.5 for x in years], book_counts)
    plt.xlabel("Rok")
    plt.ylabel("Liczba ksiazek")
    plt.title("Analiza danych staje sie coraz bardziej popularna!")
    plt.show()

##
# 
# Korzystanie z interfejsów programistycznych
#
##

endpoint = "https://api.github.com/users/joelgrus/repos"

repos = json.loads(requests.get(endpoint).text)

from dateutil.parser import parse

dates = [parse(repo["created_at"]) for repo in repos]
month_counts = Counter(date.month for date in dates)
weekday_counts = Counter(date.weekday() for date in dates)

####
#
# Twitter
#
####

from twython import Twython

# fill these in if you want to use the code
CONSUMER_KEY = ""
CONSUMER_SECRET = ""
ACCESS_TOKEN = ""
ACCESS_TOKEN_SECRET = ""

def call_twitter_search_api():

    twitter = Twython(CONSUMER_KEY, CONSUMER_SECRET)

    # Wyszukaj posty zawierające frazę „data science”.
    for status in twitter.search(q='"data science"')["statuses"]:
        user = status["user"]["screen_name"].encode('utf-8')
        text = status["text"].encode('utf-8')
        print user, ":", text
        print

from twython import TwythonStreamer

# Dopisywanie danych do zmiennej globalnej to generalnie kiepski pomysł,
# ale znacznie ułatwia on ten przykład.
tweets = [] 

class MyStreamer(TwythonStreamer):
    """Nasza własna podklasa TwythonStreamer określająca
    sposób interakcji ze strumieniem."""

    def on_success(self, data):
        """Co zrobimy, gdy Twitter prześle nam dane?
        W tym przypadku dane zostaną umieszczone w obiekcie Pythona reprezentującym post."""

        # Chcemy zbierać tylko posty napisane w języku angielskim.
        if data['lang'] == 'en':
            tweets.append(data)

        # Przerwij operację po zebraniu wystarczającej liczby postów.
        if len(tweets) >= 1000:
            self.disconnect()

    def on_error(self, status_code, data):
        print status_code, data
        self.disconnect()

def call_twitter_streaming_api():
    stream = MyStreamer(CONSUMER_KEY, CONSUMER_SECRET, 
                        ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

    # Rozpoczyna proces zbierania publicznych statusów zawierających słowo „data”.
    stream.statuses.filter(track='data')
    

if __name__ == "__main__":

    def process(date, symbol, price):
        print date, symbol, price

    print "tab delimited stock prices:"

    with open('tab_delimited_stock_prices.txt', 'rb') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            date = row[0]
            symbol = row[1]
            closing_price = float(row[2])
            process(date, symbol, closing_price)

    print

    print "colon delimited stock prices:"

    with open('colon_delimited_stock_prices.txt', 'rb') as f:
        reader = csv.DictReader(f, delimiter=':')
        for row in reader:
            date = row["date"]
            symbol = row["symbol"]
            closing_price = float(row["closing_price"])
            process(date, symbol, closing_price)

    print

    print "writing out comma_delimited_stock_prices.txt"

    today_prices = { 'AAPL' : 90.91, 'MSFT' : 41.68, 'FB' : 64.5 }

    with open('comma_delimited_stock_prices.txt','wb') as f:
        writer = csv.writer(f, delimiter=',')
        for stock, price in today_prices.items():
            writer.writerow([stock, price])

    print "BeautifulSoup"
    html = requests.get("http://www.example.com").text
    soup = BeautifulSoup(html)
    print soup
    print

    print "parsing json"

    serialized = """{ "title" : "Data Science Book",
                      "author" : "Joel Grus",
                      "publicationYear" : 2014,
                      "topics" : [ "data", "science", "data science"] }"""

    # Przetwarza kod  JSON w celu utworzenia słownika Pythona.
    deserialized = json.loads(serialized)
    if "data science" in deserialized["topics"]:
        print deserialized 

    print

    print "GitHub API"
    print "dates", dates
    print "month_counts", month_counts
    print "weekday_count", weekday_counts

    last_5_repositories = sorted(repos,
                                 key=lambda r: r["created_at"],
                                 reverse=True)[:5]

    print "last five languages", [repo["language"] 
                                  for repo in last_5_repositories]

