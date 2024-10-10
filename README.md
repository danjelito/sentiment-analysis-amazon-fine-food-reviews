# Amazon Fine Food Reviews Sentiment Analysis Project

This project performs sentiment analysis on a dataset of text reviews using two sentiment analysis models: VADER (from the NLTK library) and RoBERTa (from Hugging Face's Transformers library). We use regression metrics to evaluate the accuracy of both models in predicting review scores.

## Models Used

1. **VADER (Valence Aware Dictionary and sEntiment Reasoner)**:
   - VADER is a lexicon-based model designed for sentiment analysis that is sensitive to both the polarity (positive/negative) and the intensity (strength) of sentiments.
   - It is particularly effective for analyzing social media data such as tweets.
   - VADER outputs four sentiment scores: `neg` (negative), `neu` (neutral), `pos` (positive), and `compound` (overall sentiment score).
   - [Official Documentation](https://www.nltk.org/_modules/nltk/sentiment/vader.html).

2. **RoBERTa (Robustly Optimized BERT Pretraining Approach)**:
   - RoBERTa is a transformer-based model, fine-tuned for sentiment classification on tweets in this case. It outputs sentiment scores for negative, neutral, and positive sentiment classes.
   - [Hugging Face Pretrained Model](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest).

## Dataset

The dataset used in this project contains Amazon reviews. It includes three columns:
- **Text**: The body of the review.
- **Summary**: A brief summary of the review.
- **Score**: The sentiment score provided by the reviewer (on a scale from 1 to 5).

We only focus on the `text` and `score` columns for our sentiment analysis tasks.

The dataset is obtained from Kaggle: [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews).

## Project Workflow

1. **Data Cleaning**:
   - The review text is preprocessed for both models by normalizing the text and removing special characters.

2. **Sentiment Analysis with VADER**:
   - We apply VADER to each review text, extracting the compound score as the main sentiment prediction.
   - These scores are normalized to match the range of the original review scores (1 to 5).

3. **Sentiment Analysis with RoBERTa**:
   - We tokenize the text using the pretrained tokenizer from Hugging Face.
   - The RoBERTa model is used to classify the sentiment into three categories: negative, neutral, and positive.
   - A custom function combines the scores into a single sentiment prediction, which is also normalized to the review score range (1 to 5).

4. **Evaluation**:
   - We compare the performance of the models using regression metrics:
     - Mean Squared Error (MSE)
     - Mean Absolute Error (MAE)
     - Root Mean Squared Error (RMSE)
     - R-squared (R²)
     - Explained Variance Score (EVS)
   - These metrics assess how closely the predicted scores align with the actual review scores.

5. **Error Analysis**:
   - We examine the reviews where the models perform the worst, displaying the largest discrepancies between the true score and the predicted score.

## Results

For both VADER and RoBERTa, we display:
- The true review scores vs. the predicted scores from each model.
- The top 5 reviews where the models' predictions deviated most from the true score.

## References

- [VADER Sentiment Analysis Documentation](https://www.nltk.org/_modules/nltk/sentiment/vader.html)
- [RoBERTa Model on Hugging Face](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- [YouTube Tutorial: Python Sentiment Analysis Project with NLTK and 🤗 Transformers](https://www.youtube.com/watch?v=QpzMWQvxXWk) by Rob Mulla