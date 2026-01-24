# High-Risk-Distress-Natural-Language-Processing-Classifier-Using-Logistic-Regression-Python: Supervised Machine Learning
 A Personal project building a multi-class NLP classification model in Python to identify emotional distress risk–related language patterns and evaluating different pre-processing approaches. 

The dataset categorized posts related to stress, anxiety, depression, suicidal ideation, and no symptoms in 1,000+ user-generated text collected from X, Reddit and Instagram on Kaggle. (Sci-kit-learn).



# Pre-processing Raw Text Data:  Regex, pandas and other python libraries
- Standardizes mental health labels into consistent categories
- Expands contractions and normalizes slang and abbreviations
- Removes URLs, punctuation, special characters, and extra whitespace
- Fixes elongated words (e.g. sooooo → so)
- Splits concatenated words using word segmentation
- Filters scrambled, duplicated, or low-quality text
- Removes duplicate posts
- Exports a cleaned dataset ready for NLP models

status
Depression                   248
HighRiskEmotionalDistress    248
Anxiety                      247
Normal                       243


# Jupyter Notebook
1. Split the model;

<img width="1504" height="465" alt="image" src="https://github.com/user-attachments/assets/fd0e9868-41b1-471a-956b-6726857b47b4" />

2. Lemmatize + Vectorize TD-IDF
- analyzer word
- n gram range = (1, 2)
- remove stop words

3. Model: Logistic Regression: OVR Classifier
   
 precision    recall  f1-score   support

               Depression       0.67      0.71      0.69        49
                  Anxiety       0.63      0.52      0.57        50
HighRiskEmotionalDistress       0.61      0.72      0.66        50
                   Normal       0.76      0.71      0.74        49

                 accuracy                           0.67       198
                macro avg       0.67      0.67      0.67       198
             weighted avg       0.67      0.67      0.66       198
<img width="928" height="289" alt="image" src="https://github.com/user-attachments/assets/209877a8-52a9-4f7e-b5fd-a3fc23af1adb" />



# Vectorization using TD-IDF:
TD-IDF Transform the processed tokens into numerical feature vectors.
- high frquency words that carry little semantic value are more likely assigned low weights where as informative terms that are class specific carry higher weights 
- helps model learn from  signal words "upset" or "anxious" instead of frequent words in text such as "them", "apple", etc; drawing clearer decision boundaries between categories
- Con: does not focus on semantic; only frequency of words



Note:
Emotions are often: implied, indirect, expressed differently by different people; “Im so happy" is similar to "Yay!" and "I feel so blessed"; 
spaCy groups these together semantically, even with different wording.
- Represents meaning as numbers so text can be compared and analyzed
- Understands context; less about key words and more about meaning
- Long posts often repeat ideas. spaCy’s Doc.vector averages meaning across the entire post, so repetition doesn’t overpower the result.



<img width="928" height="289" alt="image" src="https://github.com/user-attachments/assets/209877a8-52a9-4f7e-b5fd-a3fc23af1adb" />
  

# Challenges and Revisions
- The first main challenge in this project was pre-processing unstructured social media text during my first run with a large data set! Through experimentation I learnt the importance of thoughtful preprocessing and class grouping decisions, as overly aggressive cleaning could remove important signal words while insufficient cleaning could introduce noise and reduce model performance. 
<img width="1097" height="437" alt="image" src="https://github.com/user-attachments/assets/37a93690-f509-4683-92d4-e02c72b2e6b4" />

   
- Future Improvements:

  -  Implement more robust text normalization techniques, including handling repeated characters, elongated words, and common social media abbreviations.
- Use a modern NLP approach by using spaCy or BERT and evaluate its performance.
  -  Reduced noise introduced by class imbalance by reassessing feature importance across categories rather than relying solely on frequency-based weighting.
  -  Improve feature interpretability by validating top-weighted terms against domain relevance and semantic meaning.
    
  



# Future Step: Risk-Flag and Resource Prompt System
- Upon further improving model performance and robustness, future work could involve the development of a risk-flag and resource prompt system designed to respond to extreme emotional distress detected in social media posts, particularly indicators of extreme emoyional distress. Rather than functioning as a diagnostic tool, this system would act as an early-warning mechanism that identifies high-risk language patterns and triggers supportive interventions.
- When a post is classified as high risk, the system could prompt the application to display a mental health support message alongside accessible resources, such as crisis hotline information, text-based support services, or links to mental health organizations. The goal of this approach would be to provide timely, non-intrusive support and encourage help-seeking behavior, while maintaining user privacy and ethical safeguards.

