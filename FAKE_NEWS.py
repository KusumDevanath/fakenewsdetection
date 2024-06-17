import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import spacy
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel, LsiModel, TfidfModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Streamlit App
st.title('Fake News Detection Analysis')

# Load Data
data = pd.read_csv('fake_news_data.csv')  # Replace with the actual path to your data
st.write("Data Overview")
st.write(data.head())

# Display data info
st.write("Data Information")
buffer = io.StringIO()
data.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

# Visualize the distribution of the classes
st.write("Count of Article Classification")
fig, ax = plt.subplots()
data['fake_or_factual'].value_counts().plot(kind='bar', color='#eebfbf', ax=ax)
st.pyplot(fig)

# Text preprocessing
nlp = spacy.load('en_core_web_sm')
data['text_clean'] = data.apply(lambda x: re.sub(r"^[^-]*-\s", "", x['text']), axis=1)
data['text_clean'] = data['text_clean'].str.lower()
data['text_clean'] = data.apply(lambda x: re.sub(r"([^\w\s])", "", str(x['text_clean'])), axis=1)
en_stopwords = stopwords.words('english')
data['text_clean'] = data['text_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in en_stopwords]))
data['text_clean'] = data.apply(lambda x: word_tokenize(x['text_clean']), axis=1)
lemmatizer = WordNetLemmatizer()
data['text_clean'] = data['text_clean'].apply(lambda tokens: [lemmatizer.lemmatize(token) for token in tokens])

# Unigrams
tokens_clean = sum(data['text_clean'], [])
unigrams = pd.Series(nltk.ngrams(tokens_clean, 1)).value_counts()
st.write("Most Common Unigrams After Preprocessing")
fig, ax = plt.subplots()
sns.barplot(x=unigrams.values[:10], y=unigrams.index[:10], orient='h', palette=['#eebfbf'], ax=ax)
st.pyplot(fig)

# Sentiment Analysis
vader_sentiment = SentimentIntensityAnalyzer()
data['vader_sentiment_score'] = data['text'].apply(lambda x: vader_sentiment.polarity_scores(x)['compound'])
bins = [-1, -0.1, 0.1, 1]
names = ['negative', 'neutral', 'positive']
data['vader_sentiment_label'] = pd.cut(data['vader_sentiment_score'], bins, labels=names)
st.write("Sentiment by News Type")
fig, ax = plt.subplots()
sns.countplot(x='fake_or_factual', hue='vader_sentiment_label', palette=sns.color_palette("hls"), data=data, ax=ax)
st.pyplot(fig)

# Topic Modeling with LDA
fake_news_text = data[data['fake_or_factual'] == "Fake News"]['text_clean'].reset_index(drop=True)
dictionary_fake = corpora.Dictionary(fake_news_text)
doc_term_fake = [dictionary_fake.doc2bow(text) for text in fake_news_text]
coherence_values = []
model_list = []

min_topics = 2
max_topics = 11

for num_topics_i in range(min_topics, max_topics + 1):
    model = gensim.models.LdaModel(doc_term_fake, num_topics=num_topics_i, id2word=dictionary_fake)
    model_list.append(model)
    coherence_model = CoherenceModel(model=model, texts=fake_news_text, dictionary=dictionary_fake, coherence='c_v')
    coherence_values.append(coherence_model.get_coherence())

st.write("LDA Topic Modeling Coherence Scores")
fig, ax = plt.subplots()
ax.plot(range(min_topics, max_topics + 1), coherence_values)
ax.set_xlabel("Number of Topics")
ax.set_ylabel("Coherence Scores")
st.pyplot(fig)

num_topics_lda = 7
lda_model = gensim.models.LdaModel(corpus=doc_term_fake, id2word=dictionary_fake, num_topics=num_topics_lda)
st.write("LDA Model Topics")
st.write(lda_model.print_topics(num_topics=num_topics_lda, num_words=10))

# Logistic Regression Model
X = [','.join(map(str, text)) for text in data['text_clean']]
Y = data['fake_or_factual']
countvec = CountVectorizer()
countvec_fit = countvec.fit_transform(X)
bag_of_words = pd.DataFrame(countvec_fit.toarray(), columns=countvec.get_feature_names_out())
X_train, X_test, y_train, y_test = train_test_split(bag_of_words, Y, test_size=0.3)
lr = LogisticRegression(random_state=0).fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
accuracy = accuracy_score(y_pred_lr, y_test)
report = classification_report(y_test, y_pred_lr)

st.write("Logistic Regression Model Accuracy")
st.write(accuracy)
st.write("Classification Report")
st.text(report)
