# Earnings Call Sentiment vs. Stock Price Reaction

This project analyzes whether the **sentiment of earnings call transcripts** is related to **short-term stock price movements**. Using natural language processing (NLP), we examine whether the tone used by company executives during earnings calls correlates with stock returns following the announcement.

The analysis focuses on four major U.S. companies:

- Apple (AAPL)
- Microsoft (MSFT)
- NVIDIA (NVDA)
- JPMorgan Chase (JPM)

We analyze **16 earnings events across multiple quarters** and compare transcript sentiment with stock returns **1, 3, and 5 days after the earnings call**.

---

# Project Goal

The main research question:

> Does the sentiment expressed in quarterly earnings call transcripts correlate with short-term stock returns after earnings announcements?

The idea is that the tone executives use when discussing financial performance might signal company outlook and influence investor reactions.

---

# Data Sources

The project combines **text data and financial data**.

## Earnings Call Transcripts

- Scraped from **The Motley Fool earnings call transcripts**
- Includes transcripts for:
  - AAPL
  - MSFT
  - NVDA
  - JPM

Text preprocessing steps:

- Removed speaker labels
- Removed boilerplate phrases
- Removed bracketed actions
- Normalized whitespace
- Cleaned special characters

Final Output:
- ticker
- quarter
- fiscal_year
- lmd_net_score
- vader_compound
- return_1d
- return_3d
- return_5d
- beat_miss

---

## Stock Price Data

Stock prices were collected using the **yfinance Python API**.

This data was used to calculate **post-earnings stock returns**.

Output file: price_returns.csv

Return windows analyzed:

- 1-day return
- 3-day return
- 5-day return

---

# Methods

## Sentiment Analysis

Two sentiment analysis methods were applied to the transcripts.

### 1. Loughran-McDonald Financial Sentiment Lexicon (LMD)

A financial-domain dictionary used to identify **positive and negative words in financial documents**.

Filtered lexicon used in the project:

- 354 positive words
- 2,355 negative words

The LMD score measures the balance of positive vs negative financial language in the transcript.

---

### 2. VADER Sentiment Analyzer

VADER is a **rule-based sentiment analysis tool** commonly used for general text sentiment.

Each transcript receives a **compound sentiment score** between:

- **1** → very positive
- **0** → neutral
- **-1** → very negative

---

# Data Processing

Sentiment scores were merged with stock return data to create a final dataset containing:
ticker
quarter
fiscal_year
lmd_net_score
vader_compound
return_1d
return_3d
return_5d
beat_miss


A **correlation matrix** was then used to evaluate relationships between:

- Sentiment scores
- Post-earnings stock returns

---

# Visualizations

The project includes several visualizations:

- Scatter plot of sentiment vs 1-day returns
- Average sentiment comparison (beat vs miss quarters)
- Sentiment trend across quarters for each company
- Correlation heatmap between sentiment and returns
- Distribution of 1-day post-earnings returns by ticker

Libraries used:

- matplotlib
- seaborn
- plotnine

---

# Key Findings

Main insights from the analysis:

- Sentiment had **only a weak relationship with stock returns**
- The **Loughran-McDonald lexicon performed better than VADER** for financial text
- The strongest correlation observed was: LMD sentiment vs 1-day returns: r = 0.157

Company-specific observations:

- **AAPL and MSFT**
  - Mostly positive and stable post-earnings returns
- **NVDA**
  - Higher volatility and wider return distribution
- **JPM**
  - Returns clustered close to zero

Overall conclusion:

> Earnings call sentiment alone is not sufficient to predict short-term stock market reactions.

---

# Limitations

Some limitations of this study include:

- Small dataset (16 earnings events)
- Market expectations may influence returns more than sentiment
- Macroeconomic factors were not included in the model

Future work could include:

- More companies
- Larger datasets
- Additional financial indicators
- Earnings surprise variables

---

# Technologies Used

- Python
- pandas
- numpy
- matplotlib
- seaborn
- plotnine
- yfinance
- VADER sentiment analysis
- Loughran-McDonald financial lexicon

---

# Contributors

- Faris Banaja
- Mateo Rocha
- Zina Zhang
- Sadia Fathima
