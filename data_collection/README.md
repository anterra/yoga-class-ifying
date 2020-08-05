# About Webscraping Notebooks

Due to the password-required nature of the website I scraped to get yoga class and pose information, I had to use Selenium in a chromedriver to store cookies almost exclusively to do my webscraping.

All of the get_urls, and then get_yoga_poses notebooks for each class type were run concurrently to maximize efficiency in webscraping.

You'll notice the same class scraping function defined in each of the get_yoga_poses notebooks -- trust me, I would have liked to put these in a .py module as much as you'd love to see them there. Unforunately, when calling an external module, a new chrome browser window is instantiated. This would have slowed my scraping _significantly_. For this reason I copy pasted the module into each notebook, so that once open, with cookies stored, Selenium would simply load the next url in the same window. It was much quicker, necessarily so, even if bad form!

-Anterra
