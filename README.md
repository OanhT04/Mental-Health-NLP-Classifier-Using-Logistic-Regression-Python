# High-Risk-Distress-Natural-Language-Processing-Classifier-Using-Logistic-Regression-Python: Supervised Machine Learning
project: Building a multi-class NLP classification model in Python to identify emotional distress risk–related language patterns and evaluating different pre-processing approaches. 
 
The dataset categorized posts related to stress, anxiety, depression, suicidal ideation, and no symptoms in 20,000+ user-generated text collected from X, Reddit and Instagram on Kaggle. 

The main focus: detection of **high risk text** language patterns associated to suicide ideation or extreme emotional distress and provide banner pop ups of mental health resouurce information.

#### Goal and metrics 
Focus on Recall -> Because the goal of this classifier is to detect high-risk language and trigger a popup with mental health support resources, recall is prioritized over precision. It is more desirable to successfully identify as many high-risk cases as possible, even if that results in some false positives. The cost of a failing to detect genuinely high-risk language, is significantly higher than the cost of detecting false positives. Therefore, the model will be optimized for high recall on the high-risk class, accepting an increased false positive rate as a tradeoff.

##### While this may result in some users seeing the popup of mental health services unnecessarily, the impact of these false positives is not considered highly significant, as the intervention is low-friction, non-accusatory, and easy to dismiss.  In contrast, missing a truly high-risk case carries substantially greater potential harm.  At most, a false positive may cause brief and minor inconvenience. In contrast, missing a truly high-risk social media post could mean missing an important opportunity to surface supportive resources, encourage someone to reach out for help, or raise awareness of available services at a moment when they may be most needed.

##### While this may result in some users seeing the popup of mental health services unnecessarily, the impact of these false positives is not considered highly significant, as the intervention is low-friction, non-accusatory, and easy to dismiss; and still can benefit users through awareness of support services. In contrast, missing a truly high-risk case carries substantially greater potential harm. At the same time I hope to find a reasomable balance of precision 


#### Dataset Source:
https://www.kaggle.com/datasets/priyangshumukherjee/mental-health-text-classification-dataset
Murarka, A., Radhakrishnan, B., \& Ravichandran, S. (2021). Detection and Classification of Mental Illnesses on Social Media using RoBERTa

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
TD-IDF Transform the processed tokens into numerical feature vectors.
- high frquency words that carry little semantic value are more likely assigned low weights where as informative terms that are class specific carry higher weights 
- helps model learn from  signal words "upset" or "anxious" instead of frequent words in text such as "them", "he", etc; drawing clearer decision boundaries between categories
- Con: does not focus on semantic; only frequency of words


# Training and Testing: Logistic Regression: 
Goal: High Risk Emotional Distress Scores: precision score >85, recall > 85, f1 score  > 90


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




Precision > 
→ precision tells us how often the model is correct when classifying a post as high risk. High precision means the model makes few false positive errors.

Recall > 
→ Recall shows how often the model successfully detects high risk user text.



#Analysis ---The model correctly identifies most positive cases (high recall), while ensuring a reasonable balance of its predictions being correct ( precision). Since the primary goal is to detect the majority of high-risk posts, this balanced performance results in consistently high classification reliability rather than reliance on accuracy alone; suitable for low-intensity risk alert systems that provide non-intrusive, support banner notifications when detecting extreme emotional distress in user text language. 

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

