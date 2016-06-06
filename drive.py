import gspread
from oauth2client.service_account import ServiceAccountCredentials
import re

def read_sheet():
    """reads our spreadsheet"""

    #TODO: Could be optimized so it doesn't pull all three every time depending on what we want to use it for.
    scope = ['https://spreadsheets.google.com/feeds']

    credentials = ServiceAccountCredentials.from_json_keyfile_name('Scandal-58e97c6d4a1a.json', scope)

    gc = gspread.authorize(credentials)
    wks = gc.open_by_key('1vUos1XNsV4Ec4aVPIVU-QkBeLacFoOzRa1-U9_ORKK4')

    articles = []
    titles = []
    corrections = []

    (articles, titles, corrections) = wks.worksheets()
    return articles, titles, corrections


def get_article_metadata():
    """Gets the filename, date, and publication for each article. Returns a list of dictionaries."""
    article_sheet, _, _ = read_sheet()
    results = []
    for article in article_sheet.get_all_values()[1:]:
        row = {}
        row["date"] = article[6]
        row["filename"] = re.sub(r'_clean|.txt|_clean.txt', '', article[2].lower())
        row["newspaper name"] = article[7].lower()
        results.append(row)
    return results


def main():
    articles, titles, corrections = read_sheet()
    print(get_article_metadata(articles))


if __name__ == '__main__':
    main()
