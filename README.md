# AI vs. Human Rating Label Detection Framework
This project is an end-to-end framework for identifying AI-generated vs. human-generated rating labels. It uses a subset of the Netflix Prize Dataset and synthetic data from Vertex AI Gemini, providing a comprehensive analysis with feature engineering, model training, and evaluation.

## Overview
This project directly addresses the growing need to distinguish between human and AI-generated content used in training machine learning models. Traditional methods can struggle to differentiate subtle variances, leading to compromised data integrity. By providing a clear and replicable method, this framework is designed to enhance the efficiency and interpretability of the detection process, aiding in data curation and model development.

## Key Elements
The effectiveness of the framework hinges on several key components:

*   **Dataset Creation:** The project relies on a combined dataset. A specific subset of the Netflix Prize Dataset representing genuine human rating behavior is paired with a synthetically generated dataset of ratings from Vertex AI Gemini.

*   **Data Preprocessing:** Critical for preparing the datasets, this step ensures the cleanliness and consistency required for accurate feature engineering and model training. This encompasses removing incomplete entries, data standardization, and format homogenization.

*   **Feature Engineering:** The core of the analysis lies in extracting a compact yet highly informative set of statistical features from user rating patterns. Based on empirical analysis of Variance Inflation Factors (VIF) and Feature Importance, this has been refined to include only:
    *   **Standard Deviation (SD) of Ratings:** Measures the dispersion of a user's ratings, indicating consistency or variability in their preferences.
    *   **Kurtosis of Rating Distribution:** Quantifies the "tailedness" of the rating distribution, revealing the presence of outliers or extreme values that might characterize AI behavior.
    *   **Skewness of Rating Distribution:** Assesses the asymmetry of the rating distribution, capturing potential biases or trends in rating tendencies.

*   **Model Training:** A selection of machine learning models – XGBoost, RandomForest, Support Vector Machines (SVM), K-Nearest Neighbors (KNN), and Logistic Regression – are trained using the engineered features. The goal is to create accurate classifiers that differentiate between AI-generated and human-generated rating labels, evaluated on both their individual performance and their collective performance across different cross-validation folds.

*   **Evaluation:** The models undergo rigorous evaluation using a variety of metrics to assess their performance. These metrics include:
    *   **Classification Reports:** Precision, Recall, F1-Score for each class (AI/Human) providing detailed insights into the model's ability to correctly classify both types of labels.
    *   **ROC AUC Score:** The Area Under the Receiver Operating Characteristic curve, quantifying the model's overall ability to discriminate between AI and human-generated ratings.
    *   **Confusion Matrices:** Visual representations that reveal the nature and frequency of classification errors, allowing for fine-tuning of the model.
    *   **Learning Curves:** Illustrate the model's performance as a function of training data size, indicating potential areas for improvement or overfitting.

*   **User Interface (Gradio):** A user-friendly Gradio UI simplifies the interaction with the trained models, enabling users to upload custom rating data in CSV format and receive interactive predictions. This allows for continuous monitoring of data quality and model performance.

## Workflow - In Detail

1.  **Dataset Preparation:**

    *   **Netflix Prize Dataset:** A subset of the Netflix Prize Dataset is utilized. To create a balanced representation of human raters, the initial dataset is filtered to include only users with at least 300 ratings. A random sample of 300 ratings per user is selected. This sampled ratings dataset is created via the use of `analyze_ratings.py`.
        *   **`analyze_ratings.py`**: this file reads the entire data and creates sample of 300 random ratings for each user.

    *   **AI-Generated Ratings:** Vertex AI Gemini is leveraged to generate synthetic movie ratings, crafting prompts that elicit ratings on a scale from 1 to 5. To introduce diversity into the dataset, the `temperature` parameter is varied during generation. The prompt data is generated via the creation of an NDJSON and uploaded to the Vertex AI Batch Prediction API. This can be done with the use of the `create_netflix_batch_input_jsonl.py` script. The JSONL data is taken and given to a `BatchPredictionJob`.
        *   **`create_netflix_batch_input_jsonl.py`**: this file takes the original dataset and creates NDJSON data to upload to the Vertex AI Gemini batch prediction API

    *   **Data Exportation**: the `BatchPredictionJob` then outputs the dataset into a JSONL format. That needs to be converted into a CSV that has the ratings. This CSV is created by use of the script `process_output_to_csv.py`.
        *   **`process_output_to_csv.py`**: takes the output from Vertex AI, retrieves the numeric rating (and skips any non-numeric rating), and saves the numeric rating with the user, movie, and temperature.

    *   **Data Organization:** Each dataset (human and AI-generated) is organized in a consistent CSV format, including the user ID, movie title, movie ID, and the rating itself.

2.  **Data Preprocessing (Spark):**

    *   **User Statistics Calculation:** The statistical computations and aggregations are performed using Apache Spark, allowing for efficient processing of large datasets. This is done with the `calculate_user_statistics.py` and `calculate_user_statistics.py` files.

        *   **`calculate_user_statistics.py`**: This script calculates mode rating for each user, and computes several basic statistical measures that describe the ratings distribution of a user. For each user the following are computed: mode, mean, median, standard deviation, max, min, IQR, kurtosis, skewness, and entropy.

        *   **`ai_synthetic_ratings_user_statistics.py`**: Same operation as `calculate_user_statistics.py`, however applied to the synthetic data generated by the Vertex AI.

    *   **Feature Selection**: After the data has been processed by the two `statistics` files, the `data_preparation_for_ml.py` loads the `ai_label` and encoded `user_id`.

        *   **`data_preparation_for_ml.py`**: Loads the statistics from the synthetic data and the original ratings data, and does preprocessing, feature engineering, and class balancing.

    *   **Feature Engineering:** The `data_preparation_for_ml.py` script uses this to create the labels.
        *   **Labeling:** An `ai_label` column is created to designate whether a rating is human-generated (0) or AI-generated (1).
        *   **User ID Encoding:** For later tracking, the `user_id` is encoded based on the `ai_label` for tracking purposes.

3.  **Data Combination and Balancing:** The data from the different source are combined and there are multiple classes, each with a label of `0` and `1`.

4.  **Model Training and Evaluation:** With the fully prepared dataset, the `main.py` script is initiated.

    *   **`main.py`**: This script performs model training, evaluation, and saving, with a comprehensive analysis.
        *   **Model Training:** The `main.py` file then trains a variety of machine learning models. The model evaluation uses the training metrics from the training and test sets.
        *   **Model Output**: The model then is saved, and can be called using the name of the model on the command line when launching the Gradio UI (`gradio_app.py`).

5.  **Gradio UI**: Finally, with the saved model, you can then call the model and upload your own rating data for it to generate predictions of AI or human via the code in `gradio_app.py`.
    *   **`gradio_app.py`**: This script creates and runs the Gradio application.

## Technologies Used

*   Python
*   Pandas
*   NumPy
*   Scikit-learn
*   XGBoost
*   Apache Spark
*   Vertex AI Gemini API
*   Gradio
*   Matplotlib
*   Scipy

## Installation and Usage

1.  **Clone the Repository:**

    ```bash
    git clone [repository URL]
    cd [repository directory]
    ```

2.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Vertex AI:**

    *   Set up a Vertex AI project and enable the Gemini API.
    *   Authenticate your environment to access the Gemini API.
    *   Update the relevant configuration parameters (e.g., project ID, location) in the code.

4.  **Run the Pipeline:**
    * First, run the code in `create_netflix_batch_input_jsonl.py`. Then, run the data in the Vertex AI UI. Then run the code in `process_output_to_csv.py` to move the output data into a csv.
    *   Run the Spark code in `user_statistics.py`, `ai_synthetic_ratings_user_statistics.py`, and `data_preparation_for_ml.py` to perform the necessary data processing and feature engineering. *Remember to configure the Spark scripts to ONLY output SD, Kurtosis, and Skewness*.
    *   Run the model training and evaluation script (`main.py`).

5.  **Launch the Gradio UI:**

    ```bash
    python gradio_app.py
    ```
    Access the interactive interface via the link provided. Specify the location of the trained model when launching the UI.

## Future Enhancements

Future directions include:

*   Further refinement of the feature set through experimentation with other statistical measures.
*   Integration of Explainable AI techniques (SHAP values) to enhance the interpretability of model predictions.
*   Implementation of active learning to improve model performance with minimal labeling effort.
*   Expansion of the framework to support rating data from alternative platforms.
