# High-Risk-Distress-NLP-Classifier-Using-Logistic-Regression-Python: Supervised Machine Learning
 A Personal project building a multi-class NLP classification model in Python to identify emotional distress risk–related language patterns and evaluating different pre-processing approaches. 

 The dataset categorized posts related to stress, anxiety, depression, suicidal ideation, and no symptoms in 18,000+ user-generated text collected from X, Reddit and Instagram on Kaggle. (Sci-kit-learn). Due to substantial class imbalance and linguistic overlap among several mental health categories, the label space was consolidated into fewer, semantically coherent groups. 
 
This reduction improved model stability, interpretability, and alignment with the capabilities of social media text data, while avoiding clinically unsupported fine-grained classification and diagnosis; 

Neutral <- Normal 

Mild-Moderate Emotional Distress <- stress, depression, anxiety have overlapping semantics and language structure

High Risk Emotional Distress <- suicidal (language patterns is more distinctive)

<img width="928" height="289" alt="image" src="https://github.com/user-attachments/assets/209877a8-52a9-4f7e-b5fd-a3fc23af1adb" />

Pre-processing Raw Text Data: 
- Regex, pandas and other python libraries
- Tokenization: split cleaned text into word/tokens
- Lemmatization: reduce words to base/root form
- Corpus generation 

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

  

# Challenges and Revisions
- The first main challenge in this project was pre-processing unstructured social media text during my first run with a large data set! Through experimentation I learnt the importance of thoughtful preprocessing and class grouping decisions, as overly aggressive cleaning could remove important signal words while insufficient cleaning could introduce noise and reduce model performance. 
- When I experimented with the original class labels from the dataset; depression, anxiety and stress tended to overlap; It proved more effective to consolidate the labels into broader, semantically coherent groups that reflect emotional severity rather than diagnostic categories. Grouping posts along a continuum from normal, to emotional distress, to high-risk (suicidal ideation) improved model stability, interpretability, and alignment with the linguistic characteristics of the data, while preserving the ability to identify critical high-risk content (which is the main focus)

   
- Future Improvements:

  -  Implement more robust text normalization techniques, including handling repeated characters, elongated words, and common social media abbreviations.
- Use a modern NLP approach by using spaCy or BERT and evaluate its performance.
  -  Reduced noise introduced by class imbalance by reassessing feature importance across categories rather than relying solely on frequency-based weighting.
  -  Improve feature interpretability by validating top-weighted terms against domain relevance and semantic meaning.
    
  



# Future Step: Risk-Flag and Resource Prompt System
- Upon further improving model performance and robustness, future work could involve the development of a risk-flag and resource prompt system designed to respond to extreme emotional distress detected in social media posts, particularly indicators of extreme emoyional distress. Rather than functioning as a diagnostic tool, this system would act as an early-warning mechanism that identifies high-risk language patterns and triggers supportive interventions.
- When a post is classified as high risk, the system could prompt the application to display a mental health support message alongside accessible resources, such as crisis hotline information, text-based support services, or links to mental health organizations. The goal of this approach would be to provide timely, non-intrusive support and encourage help-seeking behavior, while maintaining user privacy and ethical safeguards.

