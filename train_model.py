import os
import datetime
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve, auc,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import StratifiedKFold, learning_curve
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def main():
    # Define a mapping for class labels
    label_mapping = {0: "Class 0: Human", 1: "Class 1: AI"}

    # -------------------------------
    # 1. Load data from the parquet file
    # -------------------------------
    data_file = r"C:\Works\Data\projects\Python\AI_labler_detection\combined_dataset_ml.parquet"  # Update path as needed
    data = pd.read_parquet(data_file)
    # Drop the 'user_id' column if it exists
    if 'user_id' in data.columns:
        data = data.drop(columns=['user_id'])

    # -------------------------------
    # 2. Balance the classes
    # -------------------------------
    # Downsample class 0 to match the number of samples in class 1.
    class0 = data[data['ai_label'] == 0]
    class1 = data[data['ai_label'] == 1]
    class0_downsampled = class0.sample(n=len(class1), random_state=42)
    balanced_data = pd.concat([class0_downsampled, class1], axis=0)
    balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Print sample counts after balancing
    total_samples = balanced_data.shape[0]
    class0_count = balanced_data[balanced_data['ai_label'] == 0].shape[0]
    class1_count = balanced_data[balanced_data['ai_label'] == 1].shape[0]
    print("Balanced dataset sample counts:")
    print(f"Total samples             : {total_samples}")
    print(f"ai_label 0 ({label_mapping[0]}): {class0_count}")
    print(f"ai_label 1 ({label_mapping[1]}): {class1_count}")
    print("-" * 50)

    # -------------------------------
    # 3. Print the mean of each feature for each class
    # -------------------------------
    feature_cols = [
        'sd_rating', 'kurtosis_rating', 'skewness_rating'
    ]
    print("\nFeature Means per Class:")
    class_feature_means = balanced_data.groupby('ai_label')[feature_cols].mean()
    for label, row in class_feature_means.iterrows():
        mapped_label = label_mapping.get(label, f"Class {label}")
        print(f"\n{mapped_label}:")
        for feature, value in row.items():
            print(f"  {feature:20s}: {value:.4f}")
    print("-" * 50)

    # -------------------------------
    # 4. Select features and target for training
    # -------------------------------
    X = balanced_data[feature_cols]
    y = balanced_data['ai_label']

    # -------------------------------
    # 5. XGBoost Evaluation via 6-Fold Cross Validation with Regularization
    # -------------------------------
    skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
    fold_reports = []
    auc_scores = []
    feature_importances_list = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []

    for fold, (train_idx, test_idx) in enumerate(tqdm(skf.split(X, y), total=6, desc="CV Folds (XGBoost)"), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]
        tqdm.write(f"Fold {fold}: Train: {len(y_train)} samples; Test: {len(y_val)} samples")
        # Initialize XGBoost with L1 and L2 regularization
        xgb_model = XGBClassifier(
            tree_method='hist',
            device='cuda',
            eval_metric='logloss',
            random_state=42,
            reg_alpha=0.1,   # L1 regularization
            reg_lambda=1.0   # L2 regularization
        )
        xgb_model.fit(X_train, y_train)
        feature_importances_list.append(xgb_model.feature_importances_)
        y_pred = xgb_model.predict(X_val)
        y_proba = xgb_model.predict_proba(X_val)[:, 1]
        all_y_true.append(y_val.values)
        all_y_pred.append(y_pred)
        all_y_proba.append(y_proba)
        report = classification_report(y_val, y_pred, output_dict=True)
        fold_reports.append(report)
        auc_fold = roc_auc_score(y_val, y_proba)
        auc_scores.append(auc_fold)

    avg_report = {}
    avg_report['accuracy'] = np.mean([r['accuracy'] for r in fold_reports])
    for key in fold_reports[0]:
        if key == 'accuracy':
            continue
        avg_metrics = {}
        for metric in ['precision', 'recall', 'f1-score']:
            avg_metrics[metric] = np.mean([fold_report[key][metric] for fold_report in fold_reports])
        avg_metrics['support'] = np.sum([fold_report[key]['support'] for fold_report in fold_reports])
        avg_report[key] = avg_metrics
    avg_auc = np.mean(auc_scores)
    print("\nAverage Classification Report (XGBoost over 6 folds):")
    print(f"Accuracy: {avg_report['accuracy']:.4f}\n")
    for key in avg_report:
        if key == 'accuracy':
            continue
        display_key = label_mapping[int(key)] if key in ['0', '1'] else key
        print(f"--- {display_key} ---")
        for metric in ['precision', 'recall', 'f1-score']:
            print(f"{metric:10s}: {avg_report[key][metric]:.4f}")
        print(f"support   : {avg_report[key]['support']:.0f}\n")
    print(f"Average AUC (XGBoost): {avg_auc:.4f}")

    # -------------------------------
    # 6. XGBoost Feature Importances & Histograms
    # -------------------------------
    feature_importances = np.mean(np.array(feature_importances_list), axis=0)
    importance_sorted_idx = np.argsort(feature_importances)[::-1]
    print("\nXGBoost Feature Importances (averaged over folds):")
    for idx in importance_sorted_idx:
        print(f"{feature_cols[idx]:20s}: {feature_importances[idx]:.4f}")

    # Histograms for selected features (with KDE) for each class
    features_to_plot = ['mode_rating', 'mean_rating', 'kurtosis_rating', 'iqr_rating', 'entropy_rating']
    classes = sorted(balanced_data['ai_label'].unique())
    nrows = len(features_to_plot)
    ncols = len(classes)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 4), dpi=300)
    if nrows == 1 and ncols == 1:
        axs = np.array([[axs]])
    elif nrows == 1:
        axs = np.array([axs])
    elif ncols == 1:
        axs = np.array([[ax] for ax in axs])
    for i, feature in enumerate(features_to_plot):
        for j, cls in enumerate(classes):
            ax = axs[i, j]
            data_feature = balanced_data[balanced_data['ai_label'] == cls][feature]
            counts, bins, _ = ax.hist(data_feature, bins=30, color='skyblue', edgecolor='black')
            bin_width = bins[1] - bins[0]
            kde = gaussian_kde(data_feature)
            x_range = np.linspace(data_feature.min(), data_feature.max(), 200)
            kde_values = kde(x_range) * len(data_feature) * bin_width
            ax.plot(x_range, kde_values, color='red', linewidth=2)
            ax.set_title(f"{feature} - {label_mapping.get(cls, f'Class {cls}')}")
            ax.set_xlabel(feature)
            ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Horizontal bar chart for XGBoost feature importances
    plt.figure(figsize=(10, 6), dpi=300)
    sorted_idx = importance_sorted_idx[::-1]
    sorted_importances = feature_importances[sorted_idx]
    sorted_features = [feature_cols[i] for i in sorted_idx]
    plt.barh(sorted_features, sorted_importances, color='skyblue')
    plt.xlabel('Feature Importance')
    plt.title('XGBoost Feature Importances')
    plt.tight_layout()
    plt.show()

    # ROC Curve for XGBoost (aggregated over folds)
    all_y_true_xgb = np.concatenate(all_y_true)
    all_y_proba_xgb = np.concatenate(all_y_proba)
    fpr, tpr, _ = roc_curve(all_y_true_xgb, all_y_proba_xgb)
    roc_auc_value = auc(fpr, tpr)
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'XGBoost ROC (AUC = {roc_auc_value:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('XGBoost Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    # Confusion Matrix for XGBoost (aggregated over folds)
    all_y_pred_xgb = np.concatenate(all_y_pred)
    cm = confusion_matrix(all_y_true_xgb, all_y_pred_xgb)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[label_mapping[0], label_mapping[1]])
    plt.figure(figsize=(6, 6), dpi=300)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('XGBoost Confusion Matrix')
    plt.tight_layout()
    plt.show()

    # -------------------------------
    # 7. Model Comparison: XGBoost, RandomForest, SVM, KNN, and LogisticRegression
    # -------------------------------
    def evaluate_model(model, X, y, skf):
        auc_scores = []
        all_y_true_model = []
        all_y_proba_model = []
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_val)[:, 1]
            auc_fold = roc_auc_score(y_val, y_proba)
            auc_scores.append(auc_fold)
            all_y_true_model.append(y_val.values)
            all_y_proba_model.append(y_proba)
        avg_auc = np.mean(auc_scores)
        return {"avg_auc": avg_auc,
                "all_y_true": np.concatenate(all_y_true_model),
                "all_y_proba": np.concatenate(all_y_proba_model)}

    skf2 = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
    results = {}
    models = {
        "XGBoost": XGBClassifier(tree_method='hist', device='cuda', eval_metric='logloss', random_state=42,
                                 reg_alpha=0.1, reg_lambda=1.0),
        "RandomForest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(),
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000)
    }
    print("\nModel Comparison (Average AUC):")
    for name, model in models.items():
        print(f"Evaluating {name}...")
        res = evaluate_model(model, X, y, skf2)
        results[name] = res
        print(f"Average AUC for {name}: {res['avg_auc']:.4f}")

    # Bar chart comparing average AUC of all models with distinct colors and AUC value inside each bar.
    plt.figure(figsize=(8, 6), dpi=300)
    model_names = list(results.keys())
    auc_values = [results[name]['avg_auc'] for name in model_names]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 5 distinct colors
    bars = plt.bar(model_names, auc_values, color=colors)
    plt.ylabel("Average AUC")
    plt.title("Model Comparison: Average AUC")
    plt.ylim([0.0, 1.0])
    for bar, auc_value in zip(bars, auc_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height/2, f'{auc_value:.2f}', ha='center', va='center', color='white', fontsize=12)
    plt.tight_layout()
    plt.show()

    # Plot ROC curves for each model on the same plot.
    plt.figure(figsize=(8, 6), dpi=300)
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(res["all_y_true"], res["all_y_proba"])
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc_val:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Model Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    ########################################
    # 8. Learning Curve for XGBoost
    ########################################
    train_sizes, train_scores, test_scores = learning_curve(
        XGBClassifier(tree_method='hist', device='cuda', eval_metric='logloss', random_state=42, reg_alpha=0.1, reg_lambda=1.0),
        X, y, cv=skf2, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), scoring='roc_auc'
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label="Training ROC AUC")
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label="CV ROC AUC")
    plt.title("XGBoost Learning Curve")
    plt.xlabel("Number of Training Examples")
    plt.ylabel("ROC AUC Score")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -------------------------------
    # 9. Train and Save Final XGBoost Model
    # -------------------------------
    final_xgb_model = XGBClassifier(
        tree_method='hist',
        device='cuda',
        eval_metric='logloss',
        random_state=42,
        reg_alpha=0.1,
        reg_lambda=1.0
    )
    final_xgb_model.fit(X, y)
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    model_filename = os.path.join(models_dir, f"xgboost_model_{current_date}.json")
    final_xgb_model.save_model(model_filename)
    print(f"Final XGBoost model saved to {model_filename}")

if __name__ == '__main__':
    main()
