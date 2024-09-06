import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from textblob import TextBlob
import syllapy
import re

# Download necessary NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# Set file paths
input_file = r'C:\Users\karan\OneDrive\Desktop\Data Intern project\Input.xlsx'
output_file = r'C:\Users\karan\OneDrive\Desktop\Data Intern project\Output Data Structure.xlsx'

print("Reading input file...")
# Read input file
df_input = pd.read_excel(input_file)
print("Input file read successfully.")

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Define functions for text analysis
def get_positive_score(text):
    return sia.polarity_scores(text)['pos']

def get_negative_score(text):
    return sia.polarity_scores(text)['neg']

def get_polarity_score(text):
    return TextBlob(text).sentiment.polarity

def get_subjectivity_score(text):
    return TextBlob(text).sentiment.subjectivity

def get_avg_sentence_length(text):
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text)
    return len(words) / len(sentences) if sentences else 0

def get_percentage_of_complex_words(text):
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    complex_words = [word for word in words if len(word) > 2 and word not in stop_words and syllapy.count(word) > 2]
    return len(complex_words) / len(words) * 100 if words else 0

def get_fog_index(text):
    return 0.4 * (get_avg_sentence_length(text) + get_percentage_of_complex_words(text))

def get_avg_number_of_words_per_sentence(text):
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text)
    return len(words) / len(sentences) if sentences else 0

def get_complex_word_count(text):
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    return sum(1 for word in words if len(word) > 2 and word not in stop_words and syllapy.count(word) > 2)

def get_word_count(text):
    return len(nltk.word_tokenize(text))

def get_syllable_per_word(text):
    words = nltk.word_tokenize(text)
    return sum(syllapy.count(word) for word in words) / len(words) if words else 0

def get_personal_pronouns(text):
    pronouns = re.findall(r'\b(I|we|my|ours|us|We|My|Ours|Us)\b', text)
    return len(pronouns)

def get_avg_word_length(text):
    words = nltk.word_tokenize(text)
    return sum(len(word) for word in words) / len(words) if words else 0

# Function to extract article text from URL
def extract_article_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    title = soup.find('title').get_text() if soup.find('title') else ''
    paragraphs = soup.find_all('p')
    article_text = ' '.join(p.get_text() for p in paragraphs)
    return title + " " + article_text

# Perform extraction and analysis
output_data = []

print("Starting data extraction and analysis...")
for index, row in df_input.iterrows():
    url_id = row['URL_ID']
    url = row['URL']
    
    print(f"Processing URL_ID {url_id}: {url}")
    article_text = extract_article_text(url)
    
    positive_score = get_positive_score(article_text)
    negative_score = get_negative_score(article_text)
    polarity_score = get_polarity_score(article_text)
    subjectivity_score = get_subjectivity_score(article_text)
    avg_sentence_length = get_avg_sentence_length(article_text)
    percentage_of_complex_words = get_percentage_of_complex_words(article_text)
    fog_index = get_fog_index(article_text)
    avg_number_of_words_per_sentence = get_avg_number_of_words_per_sentence(article_text)
    complex_word_count = get_complex_word_count(article_text)
    word_count = get_word_count(article_text)
    syllable_per_word = get_syllable_per_word(article_text)
    personal_pronouns = get_personal_pronouns(article_text)
    avg_word_length = get_avg_word_length(article_text)
    
    output_data.append([
        url_id, url, positive_score, negative_score, polarity_score, subjectivity_score,
        avg_sentence_length, percentage_of_complex_words, fog_index, avg_number_of_words_per_sentence,
        complex_word_count, word_count, syllable_per_word, personal_pronouns, avg_word_length
    ])

# Save output to Excel
output_columns = [
    'URL_ID', 'URL', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE', 'SUBJECTIVITY SCORE',
    'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS', 'FOG INDEX', 'AVG NUMBER OF WORDS PER SENTENCE',
    'COMPLEX WORD COUNT', 'WORD COUNT', 'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH'
]

df_output = pd.DataFrame(output_data, columns=output_columns)
df_output.to_excel(output_file, index=False)

print("Text analysis completed and output saved to", output_file)
