# High-Risk-Distress-Natural-Language-Processing-Classifier-Using-Logistic-Regression-Python: Supervised Machine Learning
Building a multi-class NLP classification model in Python to identify emotional distress risk–related language patterns and evaluating different pre-processing approaches. One using TD-IDF and Logistic Regression, and one using SpaCy!!
 
The dataset categorized posts related to stress, anxiety, depression, suicidal ideation, and no symptoms in 20,000+ user-generated text collected from X, Reddit and Instagram on Kaggle. 
-
The main focus: detection of **high risk text** language patterns associated to suicide ideation or extreme emotional distress and provide banner pop ups of mental health resouurce information.

#### Goal and metrics: Recall over Precision
Because the goal of the classifier is to detect high-risk language and surface mental health support resources, recall is prioritized over precision. It is preferable to identify as many high-risk cases as possible, even if some false positives occur. These false positives are considered low impact, as the popup is low-friction, non-accusatory, and easy to dismiss. In contrast, missing a truly high-risk post could mean missing an opportunity to encourage someone to seek support when it may be most needed. At the same time, a reasonable level of precision will be maintained to avoid overly intrusive alerts.

## Exploring the Data Before Modeling
Overlap Between Depression and Suicide Risk Language

Language associated with depression often overlaps with language indicative of suicidal ideation. Expressions such as hopelessness, worthlessness, emotional exhaustion, or feeling like a burden are common in both depression and suicide risk signals. As a result, classifiers trained to detect suicide risk may assign high-risk scores to strongly depressive content, even without explicit suicidal intent.

This overlap can reduce precision, as some depressive posts may be flagged as high risk. However, in an intervention context (e.g., triggering a supportive popup), this tradeoff is acceptable. Individuals expressing depressive symptoms may still benefit from supportive resources. Therefore, the system prioritizes sensitivity to distress-related language, recognizing that depression and suicide risk exist on a continuum and are not always clearly separable in natural language.

As seen below the most common misclassification for suicidal ideation belongs to depression classification under spaCy. 

<img width="1868" height="1361" alt="image" src="https://github.com/user-attachments/assets/e929a34d-ef7f-436e-8fc9-d0d462ee0c45" />




### Dataset Source:
https://www.kaggle.com/datasets/priyangshumukherjee/mental-health-text-classification-dataset
Murarka, A., Radhakrishnan, B., \& Ravichandran, S. (2021). Detection and Classification of Mental Illnesses on Social Media 

# Clean.py - Pre-processing Text Data:  Regex, pandas and other python libraries
- Standardizes mental health labels into consistent categories
- Expands contractions and normalizes slang and abbreviations
- Removes URLs, punctuation, special characters, and extra whitespace
- Fixes elongated words (e.g. sooooo → so)
- Splits concatenated words using word segmentation
- Filters scrambled, duplicated, or low-quality text
- Removes duplicate posts


# Jupyter Notebook -pandas, sci-kit learn, 
1. Load data and split for training/testing
2. Lemmatize (spacy) + Vectorization with TD-IDF
- analyzer word
- n gram range = (1, 2)
- remove stop words
- remove words that commonly appear in >60% of text entries
  

# Vectorization using TD-IDF:
- high frquency words that carry little semantic value are more likely assigned low weights where as informative terms that are class specific carry higher weights 
- helps model learn from  signal words "upset" or "anxious" instead of frequent words in text such as "them", "he", etc; drawing clearer decision boundaries between categories
- Con: does not focus on semantic; only frequency of words


# Training and Testing: Logistic Regression: 


        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report
        
        
        clf = LogisticRegression(
            solver="lbfgs",
            multi_class="multinomial",
            class_weight="balanced",
            max_iter=5000
        )
        clf.fit(X_train_vec, y_train)
        pred = clf.predict(X_test_vec)
        print(classification_report(y_test, pred, target_names=all_target_names))






## Analysis



# Challenges
-  pre-processing unstructured social media text. Through experimentation I learnt the importance of thoughtful preprocessing and class grouping decisions, as overly aggressive cleaning could remove important signal words while insufficient cleaning could introduce noise and reduce model performance.

   
- Future Improvements:
- Use a modern NLP approach by using spaCy or/and transformers and evaluate performance.
  -  Reduced noise introduced by class imbalance by reassessing feature importance across categories rather than relying solely on frequency-based weighting.
  -  Improve feature interpretability by validating top-weighted terms against domain relevance and semantic meaning.
-   semantic analysis:
     Emotions are often: implied, indirect, expressed differently by different people; “Im so happy" is
     similar to "Yay!" and "I feel so blessed"; 
     spaCy can group these phrases together semantically, even with different wording or spelling. 
     - Represents meaning as numbers so text can be compared and analyzed
     - Understands context
       


# Future Step: Risk-Flag and Resource Prompt System
-  a future personal project could involve the development of a risk-flag and resource prompt system designed to respond to extreme emotional distress with a banner pop up. Rather than functioning as a diagnostic tool, this system would identify high-risk language patterns and encourage help-seeking behavior.
- When a post is classified as high risk only; the system could prompt the application to display a mental health support message alongside accessible resources, such as crisis hotline information/mental health resources. 

