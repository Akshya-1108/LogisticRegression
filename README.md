# 📊 Logistic Regression

This repository contains a simple and intuitive implementation of **Logistic Regression** from scratch using Python and NumPy. It is designed for educational purposes to help understand the internal workings of this fundamental classification algorithm.

---

## 🚀 Features

- Logistic Regression without scikit-learn
- Supports binary classification
- Gradient Descent optimization
- Cost function minimization
- Accuracy evaluation

---

## 🧠 What is Logistic Regression?

Logistic Regression is a supervised machine learning algorithm used for binary classification. It estimates the probability that a given input belongs to a particular category using the sigmoid function:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

---

## 🛠️ Technologies Used

- Python
- NumPy
- Matplotlib (for visualizations)
- (Optional) scikit-learn for dataset generation or evaluation

---

## 📦 Installation & Usage

```bash
# Clone the repository
git clone https://github.com/Akshya-1108/LogisticRegression.git
cd LogisticRegression

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the main script
python main.py
```

---

## 📈 Sample Output

```
Accuracy: 91.00%
```

---

## 📊 Dataset

You can use:
- Synthetic datasets (`make_classification` from sklearn)
- Public datasets like Iris (binary classes), Breast Cancer dataset, etc.

---

## ✏️ To-Do

- [ ] Add L2 Regularization
- [ ] Visualize Cost Function
- [ ] Visualize Decision Boundaries
- [ ] Multi-class support (One-vs-Rest)

---

## 🙌 Acknowledgements

- Andrew Ng’s ML Course
- NumPy Documentation
- scikit-learn (for dataset generation and benchmarking)

---

## 🧑‍💻 Author

**Akshya-1108**  
GitHub: [@Akshya-1108](https://github.com/Akshya-1108)
