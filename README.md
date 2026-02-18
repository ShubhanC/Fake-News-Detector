# Fake-News-Detector
Learning basics of Deep Learning with TF-IDF and Fake News Detection. Original project at https://data-flair.training/blogs/advanced-python-project-detecting-fake-news/, but will do some fun twists.

## Goals
Overall: The goal is to have different types of models that will predict whether a post, article, or other text is considered fake news or not. I will try to split each article into a different topic (currently within political atmospheres, will hopefully spread to non-political news as well). Goal is to predict with >95% accuracy, and reduce the number of false negatives that exist (aka. classifying text as real when it is actually fake).


1. Create a Fake News classifier by implementing basic NLP preprocessing techniques (such as vectorization) and training on an SGD-augmented linear model. Should be applicable to any news article - in progress in [model.ipynb](https://github.com/ShubhanC/Fake-News-Detector/blob/main/model/model.ipynb)
2. Create a web scraping pipeline where if someone puts in an article link, then it will extract the article and print out the inference
3. Create a new model that focuses on short-form videos and/or small statements
4. Create models that focus on specific topics (like war, economy, disasters, etc.)

The goal is to have different types of models that will predict whether a post, article, or other text is considered fake news or not. I will try to split each article into a different topic (currently within political atmospheres, will hopefully spread to non-political news as well). Goal is to predict with >95% accuracy, and reduce the number of false negatives that exist (aka. classifying text as real when it is actually fake).

requirements: `pandas`, `numpy`, `sklearn`

Data Citation: in [data_sources.md](https://github.com/ShubhanC/Fake-News-Detector/blob/main/data_sources.md)

Also check out another project of mine, [curly-waffle-spark](https://github.com/ShubhanC/curly-waffle-spark/tree/main)
