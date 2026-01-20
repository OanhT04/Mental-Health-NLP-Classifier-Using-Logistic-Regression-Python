# Mental-Health-NLP-Classifier-Using-Logistic-Regression-Python: Supervised Machine Learning
 A Personal project building a multi-class NLP classification model in Python to identify mental health–related language patterns including stress, anxiety, depression, suicidal ideation, and no symptoms in 18,000+ user-generated text collected from X, Reddit and Instagram on Kaggle. (Sci-kit-learn). Due to substantial class imbalance and linguistic overlap among several mental health categories, the label space was consolidated into fewer, semantically coherent groups. 

This reduction improved model stability, interpretability, and alignment with the capabilities of social media text data, while avoiding clinically unsupported fine-grained classification!
 <img width="1125" height="334" alt="image" src="https://github.com/user-attachments/assets/cdc11a24-d77f-4d81-8102-2daa673c487f" />
<img width="928" height="289" alt="image" src="https://github.com/user-attachments/assets/209877a8-52a9-4f7e-b5fd-a3fc23af1adb" />

Pre-processing Raw Text Data: 
- Regex, pandas and other python libraries
- Tokenization: split cleaned text into word/tokens
- Lemmatization: reduce words to base/root form

# Vectorization using TD-IDF:
Transform the processed tokens into numerical feature vectors. I chose TD-IDF due to it's compatbility with text classification helping identify patterns in text; and the goal to find patterns associated to different emotional states.
- quantifies the importance of a term in a document with respect to frequency in a document and rarity across entire corpus
- high frquency words that carry little semantic value are more likely assigned low weights where as informative terms that are class specific carry higher weights 
- helps model learn from meaningful signal words "upset" or "anxious" instead of frequent words in text such as "them", "apple", etc; drawing clearer decision boundaries between categories
  
# Model : Logistic Regression
I chose Logistic Regression due to it's compatibility for classification problems; returing probabilities vs labels and pairing well with TD-IDF.

# Challenges and Revisions
- The first main challenge in this project was pre-processing unstructured social media text during my first run with a large data set! Through experimentation I learnt the importance of thoughtful preprocessing and class grouping decisions, as overly aggressive cleaning could remove important signal words while insufficient cleaning could introduce noise and reduce model performance. 
- Especially with imbalanced data and classifications with overlapping semantic meaning vs classes such as "suicidal" that tend to have greater prediction due to more distinct language; when I experimented with the original class labels from the dataset; depression, anxiety and stress tended to overlap; It proved more effective to consolidate the labels into broader, semantically coherent groups that reflect emotional severity rather than diagnostic categories. Grouping posts along a continuum from normal, to emotional distress, to high-risk (suicidal ideation) improved model stability, interpretability, and alignment with the linguistic characteristics of the data, while preserving the ability to identify critical high-risk content (which is the main focus)

   
- Future Improvements:

  -  Implement more robust text normalization techniques, including handling repeated characters, elongated words, and common social media abbreviations.
  -  Refine tokenization to better preserve meaningful expressions while removing non-informative artifacts (e.g., random character sequences).
  -  Reduced noise introduced by class imbalance by reassessing feature importance across categories rather than relying solely on frequency-based weighting.
  -  Introduced additional filtering to remove low-information tokens without overly aggressive cleaning that could eliminate important signal words.
  -  Improve feature interpretability by validating top-weighted terms against domain relevance and semantic meaning.
  -  Iteratively evaluate preprocessing revisions to balance model performance and explainability, particularly for sensitive emotional health–related content.
  



- compare results of different classifiers; analyze performance on multiple metrics

# Next Goal:
Perform analysis on revisions and further experimentation
Upon improving model; make some tweaks and create a risk-flag + resource prompt system when extreme emotional distress (suicide ideation) is detected: example of a mental health resources (hotlines) and support message pop up on app, when high emotional distress is detected in social media user post
