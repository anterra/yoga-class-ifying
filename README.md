### Metis Project 3: Classification and Web Apps

# Yoga Class-ification

### Anterra Kennedy

## Goal

The goal of this project is to use machine learning classification methods to determine what genre of yoga a given yoga class most closely fall into, based only on the poses practiced in that class, to help yoga teachers more accurately understand their practice and market their classes.

## Context

This problem and its results will be interesting and important for several reasons. There is a great deal of subjectivity in classifying a yoga sequence, it is ultimately up to the yoga teacher to determine what genre their class most closely falls into; but this can be hard to do when the genres themselves are ambiguous and have a lot of overlap. It is nonetheless important for yoga teachers to advertise their the classes they are offering under the accurate classification, so that students coming to the class can know what to expect.

By trying to accurately predict a type of yoga based solely on what positions are practiced, I am attempting to use data science to answer the question of what actually defines the various classifications of yoga, so that both students and teachers alike can have more concrete definitions and better communication.

The end goal of this project is to create a web-based yoga sequencing application for yoga teachers, where they can create their own yoga classes from drop down menus of all 3600+ yoga poses and variations, sorted by their various features, and have their sequence analyzed to find out what genre it most closely falls into.

## Methodologies

1. ) Used Selenium to store cookies and scrape over 40,000 yoga classes and 3,600 yoga poses with their diagnostic information from Tummee.com, a website where yoga teachers can make and upload their own classes.
2. ) Paired the collected data to include diagnostic info about every pose in every class.
3. ) One-hot encoded categorical information into dummy variables to get binary stats about whether each pose fell into each category.
4. ) Summed up dummy variable counts of each category for all poses in each class, to get total counts of number of poses of each type in each class.
5. ) For each yoga class, I divided counts by length of the class to normalize the data. In this way, dataset was comprised of rows of classes, with each value as a ratio of the total class spent in each type of pose. (Note, these did not add up to 1 of course, since a single pose can have many diagnostics applied.)
6. ) Accomodated for class imbalance using RandomOverSampler.
7. ) Tested 6 different classification models.
8. ) Optimized, evaluated, and selected the best model -- Logistic Regression.
9. ) Discovered feature importance.
10. ) Built a flask app for yoga teachers to build their own yoga classes, and have them classified by percentage likelihood of falling into each genre!

## Findings and Conclusions

In scraping and analyzing yoga class data, I found that the vast majority (over 60%) of yoga classes had been labelled by their teachers as "Vinyasa" or "Hatha". Far fewer classes get labeled with the other class types, even if technically the class being offered actually is more closely aligned with one of those types.

This can be problematic for both yoga teachers and yoga practitioners. For practitioners, if the vast majority of classes have the same 2 labels, but of course vary wildly in content, you could go in expecting one thing, but recieve a class of an entirely different intensity level you did not desire, since they've been obscured to be extremely broad and non-descript terms. As a teacher, it is important to be candid about what you are actually offering.

In choosing to label a class either Vinyasa or Hatha, the key difference is basically whether or not "Surya Namaskar A", a sun-salutation flow with a certain push-up pose called "Chaturanga" is included or not, and whether importance is placed on pairing one breath with one movement. If so, the class is called Vinyasa, and if not, Hatha.

Hatha is basically a huge catch-all term for any "other" classes that don't seem like Vinyasa. It translates to something like, "movement", so you see its very vague.

After training and optimizing 6 different classification models, I found that a Logistic Regression performed the best. It had good ROC-AUC and high recall for all classes except Hatha.

And this is actually a good thing! Many classes that probably really should fall into other categories get naively mis-labeled by their teachers as "Hatha", and now I can help classify them correctly.

I essentially used Data Science to find aggregate decisions across thousands of individual teachers' yoga classes about what poses are included in each type of class, to better define those ambiguous definitions. As it turns out, the most influential factors on class type are how much of the class is spent in Strengthening, Standing, Inversion, and Balance poses.

## Deliverables

Sequentially:

- [Data Collection](https://github.com/anterra/yoga-class-ifying/tree/master/data_collection)
- [EDA and Feature Engineering](https://github.com/anterra/yoga-class-ifying/blob/master/classification_modeling/eda_feature_engineering.ipynb)
- [Pipeline Modules](https://github.com/anterra/yoga-class-ifying/blob/master/classification_modeling/pipeline_modules.py)
- [Classification Modeling](https://github.com/anterra/yoga-class-ifying/blob/master/classification_modeling/classification_modeling.ipynb)
- [Flask App](https://github.com/anterra/yoga-class-ifying/tree/master/flask_app)
- [Presentation Slide]()

## Project Team

- [Anterra Kennedy](https://www.linkedin.com/in/anterrakennedy/)

## Technologies Used

- Jupyter Notebook
- Python
- Scikit-learn
- Flask
- HTML/CSS
- Selenium
- Beautiful Soup
- Pandas
- Matplotlib
- Seaborn
- YellowBrickRoad

## Approaches and Skills

**Classification Algorithms**:

- KNN
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

**Other**

- Classification Scoring Metrics
- ROC-AUC Curves
- Precision-Recall Trade-off
- EDA
- Webscraping
