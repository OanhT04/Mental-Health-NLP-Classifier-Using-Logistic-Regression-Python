# High-Risk-Distress-Natural-Language-Processing-Classifier-Using-Logistic-Regression-Python: Supervised Machine Learning
 A Personal project building a multi-class NLP classification model in Python to identify emotional distress risk–related language patterns and evaluating different pre-processing approaches. The main focus is proper detection of high risk text language patterns associated to suicide ideation. 

The dataset categorized posts related to stress, anxiety, depression, suicidal ideation, and no symptoms in 20,000+ user-generated text collected from X, Reddit and Instagram on Kaggle. 
https://www.kaggle.com/datasets/priyangshumukherjee/mental-health-text-classification-dataset

<img width="573" height="249" alt="image" src="https://github.com/user-attachments/assets/3f6bcf0c-616c-49c6-a5a9-7cd127880b1f" />

Murarka, A., Radhakrishnan, B., \& Ravichandran, S. (2021). Detection and Classification of Mental Illnesses on Social Media using RoBERTa

# Pre-processing Raw Text Data:  Regex, pandas and other python libraries
- Standardizes mental health labels into consistent categories
- Expands contractions and normalizes slang and abbreviations
- Removes URLs, punctuation, special characters, and extra whitespace
- Fixes elongated words (e.g. sooooo → so)
- Splits concatenated words using word segmentation
- Filters scrambled, duplicated, or low-quality text
- Removes duplicate posts
- Exports a cleaned dataset ready for NLP models



# Jupyter Notebook
1. Split data

2. Lemmatize + Vectorization with TD-IDF
- analyzer word
- n gram range = (1, 2)
- remove stop words
- remove words that commonly appear in >80% of text entries

# Vectorization using TD-IDF:
TD-IDF Transform the processed tokens into numerical feature vectors.
- high frquency words that carry little semantic value are more likely assigned low weights where as informative terms that are class specific carry higher weights 
- helps model learn from  signal words "upset" or "anxious" instead of frequent words in text such as "them", "he", etc; drawing clearer decision boundaries between categories
- Con: does not focus on semantic; only frequency of words

3. Model: Logistic Regression: OVR Classifier

       # Base binary classifier used per class (one-vs-rest)
           base_lr = LogisticRegression(
               solver="liblinear",
               max_iter=5000,
               random_state=42,
               class_weight="balanced"  
           )
   
           clf_ovr = OneVsRestClassifier(base_lr)
           
           # Train
           clf_ovr.fit(X_train_vec, y_train)
           
           # Predict
           pred_ovr = clf_ovr.predict(X_test_vec)
           
           print(classification_report(y_test, pred_ovr))
# Examples
<img width="2192" height="1137" alt="image" src="https://github.com/user-attachments/assets/7b33fb86-ad9d-410a-92a3-4a0567afafe9" />
]   




Note for improvement:
Emotions are often: implied, indirect, expressed differently by different people; “Im so happy" is similar to "Yay!" and "I feel so blessed"; 
spaCy groups these together semantically, even with different wording whereas this current model relies on frequency of words.
- Represents meaning as numbers so text can be compared and analyzed
- Understands context; less about key words and more about meaning
  
  

# Challenges and Revisions
- The first main challenge in this project was pre-processing unstructured social media text. Through experimentation I learnt the importance of thoughtful preprocessing and class grouping decisions, as overly aggressive cleaning could remove important signal words while insufficient cleaning could introduce noise and reduce model performance. 

   
- Future Improvements:

  -  Implement more robust text normalization techniques, including handling repeated characters, elongated words, and common social media abbreviations.
- Use a modern NLP approach by using spaCy or BERT and evaluate its performance.
  -  Reduced noise introduced by class imbalance by reassessing feature importance across categories rather than relying solely on frequency-based weighting.
  -  Improve feature interpretability by validating top-weighted terms against domain relevance and semantic meaning.
  
    



# Future Step: Risk-Flag and Resource Prompt System
- Upon further improving model performance and robustness, future work could involve the development of a risk-flag and resource prompt system designed to respond to extreme emotional distress with a banner pop up. Rather than functioning as a diagnostic tool, this system would identify high-risk language patterns and triggers supportive interventions.
- When a post is classified as high risk only; the system could prompt the application to display a mental health support message alongside accessible resources, such as crisis hotline information/mental health resources. The goal of this approach would be to provide timely, non-intrusive support and encourage help-seeking behavior, while maintaining user privacy and ethical safeguards.

