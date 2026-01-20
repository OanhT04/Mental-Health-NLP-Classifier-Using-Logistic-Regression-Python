# Mental-Health-NLP-Classifier-Using-Logistic-Regression-Python: Supervised Machine Learning
 My first learning ML project building a multi-class NLP classification model in Python to identify mental health–related language patterns including stress, anxiety, depression, suicidal ideation, and no symptoms in 2,000+ user-generated text collected from X, Reddit and Instagram on Kaggle. (Sci-kit-learn)

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

Challenges/Revisions
- The first main challenges in this project was pre-processing unstructured social media text during my first run. This challenge highlighted the importance of thoughtful preprocessing decisions, as overly aggressive cleaning could remove important signal words while insufficient cleaning could introduce noise and reduce model performance. For example, the top classifying words weighted random letters/non meaningful words for depression reflecting the need for better data cleaning and pre-processing
- <img width="800" height="500" alt="image" src="https://github.com/user-attachments/assets/5a3c0f41-24c2-4d6e-9ccc-fb8327b2c0b1" />

- uneven class distributions, with some mental health categories having significantly fewer samples than others which led me to apply class weighting; however this imbalance posed a challenge because bias was still present towards oversampled classes and trade offed performance
- focusing more time on cleaning and pre-processing; removing more words with no semantic value through adding custome stopword lists to filter, remove short posts or non coherent posts that lack context, handle abbreviations, etc. 
- experiment with different n-gram rages
- exploring different methods of balancing classes; resampling methods vs reliance on class weights: Below is the initial class weights;
  <img width="2531" height="625" alt="image" src="https://github.com/user-attachments/assets/88b390b7-eb78-4f29-849d-73bf404247ac" />
  
- Method 1: Downsampling
- //
- Method 2: SMOTE: addresses imbalanced datasets by synthetically generating new instances for the minority class. Unlike simply duplicating records
- //
- Model comparisons: Random Forest, Balanced Bagging Classifier
  

- compare results of different classifiers; analyze performance on multiple metrics

# Next Goal:
Perform analysis on revisions and further experimentation
Upon improving model; make some tweaks and create a risk-flag + resource prompt system: example of a resources and support message pop up on app, when high emotional distress is detected in social media user post
