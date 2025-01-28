# Spam-SMS-Detection

## Project Overview
The **SMS Spam Detection** project involves the classification of SMS messages as either "Spam" or "Ham" (not spam). Using natural language processing (NLP) techniques, this project aims to identify and filter spam messages from legitimate ones. This implementation is demonstrated in a Jupyter Notebook (`SMS_Spam_Detection.ipynb`).

## Key Features
- Preprocessing of SMS text data.
- Feature extraction using techniques like Bag-of-Words or TF-IDF.
- Classification using machine learning algorithms.
- Model evaluation metrics like accuracy, precision, recall, and F1-score.

## Requirements
Ensure you have the following software and Python libraries installed:

- **Python** (>= 3.7)
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- NLTK (Natural Language Toolkit)
- Matplotlib / Seaborn (optional, for visualization)

You can install the required libraries using pip:
```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn
```

## Dataset
The dataset used in this project should contain SMS messages labeled as "Spam" or "Ham." It typically has two columns:
- `label`: Indicates whether the message is "spam" or "ham."
- `message`: The SMS text content.

Ensure that your dataset is loaded correctly into the notebook. You can replace the dataset path with your local file path in the notebook.

## Workflow
1. **Data Loading**:
   - Import the dataset into a Pandas DataFrame.
   
2. **Data Preprocessing**:
   - Convert text to lowercase.
   - Remove punctuation, stopwords, and special characters.
   - Tokenize and lemmatize words.

3. **Feature Extraction**:
   - Convert text data into numerical features using Bag-of-Words or TF-IDF vectorization.

4. **Model Training**:
   - Use machine learning algorithms (e.g., Naive Bayes, Logistic Regression) to classify messages.

5. **Evaluation**:
   - Evaluate model performance using metrics such as accuracy, precision, recall, and F1-score.

6. **Prediction**:
   - Test the model with new SMS messages to predict whether they are spam or ham.

## Results
The performance of the model can be assessed using:
- **Confusion Matrix**
- **Classification Report**
- **Accuracy Score**

The results will vary based on the dataset and the preprocessing techniques applied.

## How to Use
1. Open the `SMS_Spam_Detection.ipynb` file in Jupyter Notebook.
2. Follow the code cells sequentially to:
   - Preprocess the data.
   - Train the model.
   - Evaluate and test the model.
3. Modify or extend the notebook to include other models or techniques as needed.

## Customization
- Experiment with different vectorization techniques (e.g., Word2Vec, GloVe).
- Try different machine learning or deep learning models.
- Use cross-validation to improve model robustness.

## Contributions
Contributions are welcome! If you have ideas for improving this project, feel free to fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. Feel free to use, modify, and distribute this project as needed.

## Acknowledgments
- Dataset source : https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset/code
- Tools and libraries: Scikit-learn, NLTK, Pandas, and others used.

---


