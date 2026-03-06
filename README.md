# Overview
This project focuses on SemEval-2026 Task 2, which aims to predict emotional valence and arousal from ecological essays. Rather than framing emotion as discrete categories (e.g., happy, sad, angry), the task models emotion as a continuous and ordered signal, requiring models that can capture both the semantic meaning of text and the relative intensity of emotions.
The objective is to develop a text-based prediction system capable of estimating valence and arousal values for each essay according to the official task framework. Since the test set is hidden, all model development, tuning, and evaluation are performed using the provided training and trial datasets.
## Approach Summary
Our approach relies on a transformer-based language encoder combined with ordinal-aware prediction layers designed to estimate valence and arousal.

## Key Components
## Text Encoder
We employ a pretrained RoBERTa model to generate contextualized embeddings for the input essays. RoBERTa was selected because of its strong performance in sentiment analysis and emotion-related NLP tasks, making it well suited for capturing nuanced emotional signals in text.

## Ordinal Emotion Modeling
Valence and arousal are treated as ordered variables rather than simple categories. This design enables the model to account for the degree of prediction error, penalizing large emotional deviations more than small ones. For example, predicting “slightly positive” instead of “very positive” is considered less severe than predicting “very negative.”

## Regression-to-Ordinal Framework
The model first predicts continuous emotion scores, which are later converted into discrete ordinal labels during evaluation. This approach stabilizes training while still preserving the underlying ordinal structure of the emotional scale.

## Loss Function Design
We explore several regression-oriented loss functions (such as MSE and ordinal-aware objectives) to better align the optimization process with the task’s evaluation goals and the ordinal nature of emotional predictions.
## Evaluation Strategy
Because the official test set is not publicly accessible, all experiments are conducted using the training and trial splits provided by the task organizers.
To ensure reproducibility, fixed random seeds are used throughout the experiments, which is particularly important given the relatively small size of the trial dataset.
Model performance is evaluated using the official task metrics, along with additional diagnostic analyses such as confusion matrices and prediction error-distance analysis to better understand how the model handles ordinal emotional predictions.

# Notes
This repository includes only our own implementation and experimental work.
External frameworks such as PyTorch and HuggingFace Transformers are used solely as supporting libraries for model development.
Consistent with the CS445 course project guidelines, the primary focus of this project is on methodological exploration and analysis, rather than optimizing specifically for leaderboard performance.
