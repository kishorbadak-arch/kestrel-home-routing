# KESTREL HOME – SUBMISSION NOTES

## 1. Project Overview

Project: Kestrel Home – Service Request Routing (Variant B)

The project is a locally developed machine-learning-based service request routing system. It analyses customer messages and predicts the appropriate support team, such as Repairs, Billing, Installs & Demo, Returns & Replacement, and Warranty Claims.

The system uses TF-IDF feature extraction and Logistic Regression to classify incoming customer requests.

It also includes a confidence-based routing mechanism. Requests with confidence of 80% or higher are automatically routed, while uncertain requests are flagged for clarification instead of making unreliable predictions.

## 2. Development and AI Tools Disclosure

- Code development: Developed and implemented by me.
- Development environment: Local machine.
- Execution: Code developed and executed locally.
- GitHub usage: Publicly available GitHub source code was consulted as a technical reference. The project implementation was written by me.
- AI coding assistants: None used.
- ChatGPT / Claude: Not used during development.
- External AI APIs: None integrated or used.
- Model implementation: Scikit-learn TF-IDF Vectorizer and Logistic Regression.
- API cost: ₹0.
- Runtime cost: No per-request API charges. Runs locally.

## 3. Model Performance and Expected Score

The following results are based on the project's forward-in-time validation using April–June 2026 data.

- Overall validation accuracy against actual final team outcomes: 84.3%.
- High-confidence requests: 79.6% of requests.
- Accuracy on high-confidence requests: 97.1%.
- Low-confidence requests: 20.4%.
- Accuracy on low-confidence requests if forced to predict: 34.4%.

The validation results indicate that confidence-based routing is more reliable than forcing predictions for every request.

These figures represent validation performance and are not a guarantee of the final external evaluation score.

## 4. Technical Implementation

Technologies used:

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Joblib
- HTML
- Python HTTP Server

Main implementation files:

- train.py – Model training, validation and prediction generation.
- prep.py – Data cleaning and feature preparation.
- app.py – Local routing application and HTTP endpoints.
- static/index.html – User interface.
- model/model.joblib – Trained model.
- predictions.csv – Final prediction output.
- tests/ – Automated test cases.

## 5. Testing and Validation

The project includes 13 automated test cases covering:

- Text cleaning and preprocessing.
- Removal of order and service request identifiers.
- Correct routing of customer requests.
- Confidence-based clarification.
- Empty request validation.
- Missing optional input handling.
- Model loading failure handling.
- HTTP endpoint responses.
- Invalid JSON handling.
- Health endpoint.
- Prediction file format and consistency.

Testing is performed locally using Python's unittest framework.

Command:

python -m unittest discover -s tests -v

## 6. Approaches Tried and Discarded

1. Bot-label training: Evaluated but not selected because historical bot labels do not reliably represent actual final team outcomes.
2. Word-level and character-level TF-IDF features: Used in the final model.
3. Logistic Regression with different regularization values: C=1 selected based on validation results.
4. Text preprocessing: Mojibake correction and removal of order/service request identifiers implemented.
5. API-based classification: Not implemented because the project is designed to run locally without external API dependencies.

## 7. Limitations

- Approximately 20% of requests contain insufficient information for reliable automatic routing.
- Low-confidence requests require additional customer clarification.
- The model's performance depends on the quality and consistency of historical resolution data.
- The validation accuracy is approximately 84.3%, not a guaranteed final evaluation score.

## 8. Privacy and Deployment

The application is designed to run locally.

No external AI API is required.

The project contains customer-related datasets. Therefore, the repository and data must remain private and must not be uploaded to a public GitHub repository, in accordance with Policy Section 10.

## 9. Remaining Submission Activities

- Record the required 3-minute screen recording demonstrating development, testing and implementation decisions.
- Complete the official submission form.
- Verify the final predictions.csv file.
- Test the README instructions on a clean Python virtual environment.
- Keep the repository private and verify that no restricted customer data is publicly accessible.

## 10. Final Declaration

I developed and implemented this project independently using Python and machine-learning libraries. Development and execution were performed locally. Public GitHub source code was used only as a technical reference.

No AI coding assistant or external AI API was used during development or execution.

The application operates locally and does not require external AI API services.