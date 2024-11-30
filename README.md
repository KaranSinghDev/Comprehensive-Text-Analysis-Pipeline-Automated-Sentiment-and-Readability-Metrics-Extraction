# Sentiment Analysis and Text Insights Engine

## Overview
This project is a sentiment analysis and text insights engine that extracts, preprocesses, and analyzes text data from web articles. It computes several text metrics, including sentiment scores, readability, and linguistic features like the **Fog Index**, **complex word count**, and **personal pronoun frequency**. The goal is to provide actionable insights into the content of articles, which can be useful for applications like content summarization, trend analysis, and sentiment tracking.

## Key Features
- **Sentiment Analysis**: Computes positive, negative, and polarity scores using **VADER SentimentIntensityAnalyzer** and **TextBlob**.
- **Text Readability**: Calculates readability metrics like **Fog Index**, **average sentence length**, and **percentage of complex words**.
- **Linguistic Features**: Extracts features such as **personal pronouns**, **syllables per word**, and **average word length**.
- **Web Scraping**: Scrapes articles from the web using **BeautifulSoup** and **requests**.
- **Data Export**: Saves processed results into an **Excel file** for easy analysis and downstream applications.

## Project Requirements
- Python 3.x
- pandas
- requests
- BeautifulSoup4
- nltk
- textblob
- syllapy
- openpyxl
