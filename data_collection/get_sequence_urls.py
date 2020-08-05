from bs4 import BeautifulSoup
import time, os
from selenium import webdriver
import math
import pickle
chromedriver = "/Applications/chromedriver"
os.environ["webdriver.chrome.driver"] = chromedriver
from selenium.webdriver.support.ui import Select

def get_sequences(class_type):
    # loading driver with cookies
    driver = webdriver.Chrome(chromedriver)
    driver.get("https://www.tummee.com/login?")

    cookies = pickle.load(open("cookies.pickle", "rb"))
    for cookie in cookies:
        driver.add_cookie(cookie)

    driver.get("https://www.tummee.com/sequences/teachers/daily")

    # inputting desired class type into website
    selector = Select(driver.find_element_by_id("js-select-yoga-type"))
    selector.select_by_value(class_type)

    # scraping urls of sequences on first page
    soup = BeautifulSoup(driver.page_source, "html.parser")
    classes = soup.find(class_="daily-sequences").find_all("a")
    urls = [("https://www.tummee.com" + i["href"]) for i in classes]

    # going to second page
    next_page = "https://www.tummee.com" + soup.find(class_="All " + class_type).find("a")["href"]
    driver.get(next_page)

    # scraping urls of classes from second page onward
    while True:
        try:
            soup = BeautifulSoup(driver.page_source, "html.parser")
            classes = soup.find(class_="daily-sequences").find_all("a")
            new_urls = [("https://www.tummee.com" + i["href"]) for i in classes]
            urls = urls + new_urls
            next_150 = soup.find(class_="All " + class_type).find("a").find_next("a")
            next_page = "https://www.tummee.com" + next_150["href"]
            driver.get(next_page)
        except:
            break

    return urls

