# Email Phishing Detection Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-In%20Progress-brightgreen)

## 📌 Overview
This project focuses on detecting phishing emails using machine learning techniques. The goal is to build a reliable model that can differentiate between legitimate and phishing emails to enhance cybersecurity.

## 💡 Features
- Preprocessing and cleaning email datasets.
- Feature extraction using techniques such as **TF-IDF** or **Bag of Words**.
- Implementing machine learning algorithms like:
  - Logistic Regression
  - Random Forest
  - Support Vector Machines
- Model evaluation using metrics like accuracy, precision, recall, and F1-score.

## 🚀 Installation
Follow these steps to set up the project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/dharxhan/email-phishing-detection-using-ML.git
   ```
2. Navigate to the project directory:
   ```bash
   cd email-phishing-detection-using-ML
   ```
3. Create a virtual environment:
   ```bash
   python -m venv env
   ```
4. Activate the virtual environment:
   - Windows:
     ```bash
     .\env\Scripts\activate
     ```
   - Mac/Linux:
     ```bash
     source env/bin/activate
     ```
5. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📂 Project Structure
The project is organized as follows:

```
email-phishing-detection-using-ML/
├── data/                   # Dataset files
├── notebooks/              # Jupyter Notebooks
├── src/                    # Source code for the project
├── models/                 # Saved models
├── requirements.txt        # Required Python packages
├── README.md               # Project documentation
└── LICENSE                 # License information
```

## 🔍 Usage
1. Add your dataset files in the `data/` folder.
2. Run the preprocessing script to clean and prepare the data:
   ```bash
   python src/preprocess.py
   ```
3. Train the model:
   ```bash
   python src/train.py
   ```
4. Evaluate the model's performance:
   ```bash
   python src/evaluate.py
   ```

## 📊 Results
The trained model achieves:
- **Accuracy**: 92%
- **Precision**: 91%
- **Recall**: 90%
- **F1-Score**: 90.5%

## 🛠️ Technologies Used
- **Python**
- **Scikit-Learn**
- **Pandas**
- **Numpy**

## 🔗 References
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/)
- [Dataset Source](#)

## 🤝 Contributing
Contributions are welcome! Please fork this repository and submit a pull request for any suggestions or improvements.

---

### ✨ Author
**Dharshan Devarajan**  
Feel free to reach out via [LinkedIn](#) or [Email](#) for any queries or collaboration opportunities!
