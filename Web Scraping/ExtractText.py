import requests
import pprint
from bs4 import BeautifulSoup

# -------------------------------------------------------------------- #
# -- Constants you may care about ------------------------------------ #
# -------------------------------------------------------------------- #

# Url to be scraped. Currently only for static HTML
URL = 'https://m.wangshugu.com/book-101619/21723542.html'
# Whether using id to find DOM element
USE_ID = True
# The id name of DOM element to be extracted
ID_NAME = "nr1"

# -------------------------------------------------------------------- #
# -- Implements ------------------------------------------------------ #
# -------------------------------------------------------------------- #

# Request the whole HTML page
page = requests.get(URL)

# Using BeautifulSoup to parse it with utf-8
soup = BeautifulSoup(page.content, 'html.parser', None, None, 'utf-8')

# Extract text from the DOM identified by ID_NAME(if USE_ID is True)
rawText = None
finalText = None
if USE_ID:
    results = soup.find(id="".join(ID_NAME))
    rawText = results.text
if rawText == None:
    print("Not found wanted text!")
    exit()

# Deal with rawText string
translation = {
    160: None}
finalText = rawText.translate(translation)

# Write to file
myFile = open('Sister_3', 'w')
myFile.write(finalText)