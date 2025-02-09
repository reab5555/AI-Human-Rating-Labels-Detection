import os
import io
import tempfile
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import scipy.stats as stats
import gradio as gr

import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for servers
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import make_interp_spline
from PIL import Image

#########################
# 1) HELPER FUNCTIONS
#########################

def compute_features_spark_style(ratings):
    """
    Example aggregator:
    - Removes NaNs
    - Returns sd_rating, kurtosis_rating, skewness_rating
    """
    arr = np.array(ratings, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return {
            'sd_rating': np.nan,
            'kurtosis_rating': np.nan,
            'skewness_rating': np.nan
        }
    feats = {}
    feats['sd_rating'] = np.std(arr, ddof=1)
    feats['kurtosis_rating'] = stats.kurtosis(arr)
    feats['skewness_rating'] = stats.skew(arr)
    return feats


def create_smooth_hist_image(ratings, title="Rating Distribution"):
    arr = np.array(ratings, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        fig, ax = plt.subplots(figsize=(4,3), dpi=300)
        ax.set_title("No Ratings Available")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)

    fig, ax = plt.subplots(figsize=(4,3), dpi=300)

    # Integer bin range
    min_rating = int(np.floor(arr.min()))
    max_rating = int(np.ceil(arr.max()))
    bins = np.arange(min_rating, max_rating + 2, 1)  # e.g. [1,2,3,4,5,6]

    counts, bin_edges, _ = ax.hist(arr, bins=bins, color='skyblue',
                                   edgecolor='black', alpha=0.3)

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    if len(bin_centers) >= 3:
        x_spline = np.linspace(bin_centers[0], bin_centers[-1], 200)
        spline = make_interp_spline(bin_centers, counts, k=3)
        y_spline = spline(x_spline)
        ax.plot(x_spline, y_spline, color='red')
    else:
        ax.plot(bin_centers, counts, color='red')

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_xlim(min_rating - 0.5, max_rating + 0.5)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def read_file_bytes(file_obj):
    """Helper to handle reading Gradio file objects or direct file paths."""
    if file_obj is None:
        return None
    if isinstance(file_obj, dict):
        return file_obj.get("data", None)
    elif isinstance(file_obj, str):
        with open(file_obj, "rb") as f:
            return f.read()
    else:
        return file_obj.read()

def update_dropdowns(csv_file):
    """Update user & rating column dropdowns after CSV upload."""
    data = read_file_bytes(csv_file)
    if data is None:
        return gr.update(choices=[], value=None), gr.update(choices=[], value=None)
    try:
        df = pd.read_csv(io.BytesIO(data))
        columns = list(df.columns)
        user_default = columns[0] if len(columns) > 0 else None
        rating_default = columns[1] if len(columns) > 1 else None
        return gr.update(choices=columns, value=user_default), gr.update(choices=columns, value=rating_default)
    except Exception as e:
        print("Error in update_dropdowns:", e)
        return gr.update(choices=[], value=None), gr.update(choices=[], value=None)

def update_user_ids(csv_file, user_col):
    """Update 'Select Specific User' dropdown after user_col changes."""
    data = read_file_bytes(csv_file)
    if data is None or not user_col:
        return gr.update(choices=[], value=None)
    try:
        df = pd.read_csv(io.BytesIO(data))
        if user_col not in df.columns:
            return gr.update(choices=[], value=None)

        unique_users = df[user_col].astype(str).unique().tolist()
        default = unique_users[0] if unique_users else None
        return gr.update(choices=unique_users, value=default)
    except Exception as e:
        print("Error in update_user_ids:", e)
        return gr.update(choices=[], value=None)

def toggle_user_selection(combine_all):
    """Hide or show user dropdown if 'Combine all' is checked."""
    if combine_all:
        return gr.update(visible=False), gr.update(visible=False)
    else:
        return gr.update(visible=True), gr.update(visible=True)

#########################
# 2) MAIN PREDICT FUNCTION
#########################

def predict_csv(model_file, csv_file, user_col, rating_col, target_user, combine_all):
    """
    Loads XGBoost model, reads CSV, aggregates user ratings with compute_features_spark_style,
    and returns:
       1) A DataFrame of user/prediction info
       2) A DataFrame of aggregator stats (SD, Skewness, Kurtosis)
       3) A PIL Image of the rating distribution histogram + smooth line
    """
    if model_file is None or csv_file is None:
        return "Please upload both a model file and a CSV file.", None, None

    # --- Load Model ---
    model_bytes = read_file_bytes(model_file)
    if model_bytes is None:
        return "Error reading model file.", None, None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp.write(model_bytes)
        tmp.flush()
        model_path = tmp.name

    model = XGBClassifier()
    model.load_model(model_path)
    os.remove(model_path)

    # --- Load CSV ---
    csv_bytes = read_file_bytes(csv_file)
    if csv_bytes is None:
        return "Error reading CSV file.", None, None
    try:
        df = pd.read_csv(io.BytesIO(csv_bytes))
    except Exception as e:
        return f"Error reading CSV file: {e}", None, None

    # Check columns
    if user_col not in df.columns or rating_col not in df.columns:
        return "Selected columns not found in CSV.", None, None

    # Our model expects these columns
    feature_cols = ["sd_rating", "kurtosis_rating", "skewness_rating"]

    dist_image = None

    # The two DataFrames we will build:
    # 1) results_df -> user, Prediction, Probability AI
    # 2) stats_df   -> user, SD, Skewness, Kurtosis

    if combine_all:
        combined_ratings = df[rating_col].dropna().values
        feats = compute_features_spark_style(combined_ratings)
        dist_image = create_smooth_hist_image(combined_ratings, title="All Ratings Distribution")

        features_df = pd.DataFrame([feats])
        if features_df.isnull().values.any():
            return "No valid ratings found after filtering. Possibly all NaN.", None, None

        # Predict
        X_features = features_df[feature_cols]
        preds = model.predict(X_features)
        pred_proba = model.predict_proba(X_features)[:, 1]
        label_mapping = {0: "Human", 1: "AI"}
        label = label_mapping.get(preds[0], "Unknown")

        # Build main results table
        results_df = pd.DataFrame({
            user_col: ["Combined"],
            "Prediction": [label],
            "Probability AI": [pred_proba[0]]
        })

        # Build aggregator stats table
        stats_df = pd.DataFrame({
            user_col: ["Combined"],
            "SD": [feats["sd_rating"]],
            "Skewness": [feats["skewness_rating"]],
            "Kurtosis": [feats["kurtosis_rating"]]
        })

        return results_df, stats_df, dist_image

    else:
        if target_user:
            df = df[df[user_col].astype(str) == str(target_user)]
            if df.empty:
                return "No data found for the selected user.", None, None

        # We'll accumulate aggregator stats and final predictions per user
        feature_list = []
        user_ids = []
        aggregator_stats = []  # store dicts with {SD, Skewness, Kurtosis}

        first_plot_done = False

        for user, group in df.groupby(user_col):
            user_ratings = group[rating_col].dropna().values
            feats = compute_features_spark_style(user_ratings)
            if feats is None:
                continue

            feature_list.append(feats)
            user_ids.append(user)
            aggregator_stats.append({
                "SD": feats["sd_rating"],
                "Skewness": feats["skewness_rating"],
                "Kurtosis": feats["kurtosis_rating"]
            })

            # Make a plot for the first user
            if not first_plot_done and len(user_ratings) > 0:
                dist_image = create_smooth_hist_image(
                    user_ratings,
                    title=f"Ratings Distribution - User {user}"
                )
                first_plot_done = True

        if not feature_list:
            return "No valid ratings found after filtering. Possibly all NaN.", None, None

        feats_df = pd.DataFrame(feature_list)
        for c in feature_cols:
            if c not in feats_df.columns:
                return f"Missing {c} in feature dataframe.", None, None

        X_features = feats_df[feature_cols]
        preds = model.predict(X_features)
        pred_proba = model.predict_proba(X_features)[:, 1]
        label_mapping = {0: "Human", 1: "AI"}
        pred_labels = [label_mapping.get(p, "Unknown") for p in preds]

        # Build main results DataFrame
        results_df = pd.DataFrame({
            user_col: user_ids,
            "Prediction": pred_labels,
            "Probability AI": pred_proba
        })

        # Build aggregator stats DataFrame
        stats_df = pd.DataFrame(aggregator_stats)
        stats_df.insert(0, user_col, user_ids)  # put user_col as the first column

        stats_df.rename(columns={
            "SD": "SD",
            "Skewness": "Skewness",
            "Kurtosis": "Kurtosis"
        }, inplace=True)

        return results_df, stats_df, dist_image

#########################
# 3) BUILD GRADIO INTERFACE
#########################

with gr.Blocks() as demo:
    gr.Markdown("## AI/Human Rating Prediction")

    with gr.Row():
        model_input = gr.File(label="Upload Model JSON", file_types=[".json"])
        csv_input = gr.File(label="Upload CSV with Ratings", file_types=[".csv"])

    combine_all_checkbox = gr.Checkbox(label="Combine all ratings into a single user?", value=False)

    with gr.Row():
        user_col_dropdown = gr.Dropdown(label="Select User Column", choices=[], interactive=True)
        rating_col_dropdown = gr.Dropdown(label="Select Rating Column", choices=[], interactive=True)

    with gr.Row():
        target_user_dropdown = gr.Dropdown(label="Select Specific User", choices=[], interactive=True)

    # Update user/rating columns after CSV upload
    csv_input.upload(
        fn=update_dropdowns,
        inputs=csv_input,
        outputs=[user_col_dropdown, rating_col_dropdown]
    )
    # Update target user dropdown
    user_col_dropdown.change(
        fn=update_user_ids,
        inputs=[csv_input, user_col_dropdown],
        outputs=target_user_dropdown
    )
    csv_input.change(
        fn=update_user_ids,
        inputs=[csv_input, user_col_dropdown],
        outputs=target_user_dropdown
    )

    # Toggle user selection if "Combine All" is checked
    combine_all_checkbox.change(
        fn=toggle_user_selection,
        inputs=combine_all_checkbox,
        outputs=[target_user_dropdown, user_col_dropdown]
    )

    predict_button = gr.Button("Predict")

    # We want 2 tables + 1 image
    # Let's arrange them so that the tables stack vertically on the left,
    # and the image is on the right.
    with gr.Row():
        with gr.Column():
            output_table_main = gr.Dataframe(label="Prediction Results")
            output_table_stats = gr.Dataframe(label="Aggregator Stats (SD, Skewness, Kurtosis)")
        with gr.Column():
            output_image = gr.Image(type='pil', label="Rating Distribution Chart")

    # The predict function returns (results_df, stats_df, dist_image)
    predict_button.click(
        fn=predict_csv,
        inputs=[
            model_input, csv_input, user_col_dropdown, rating_col_dropdown,
            target_user_dropdown, combine_all_checkbox
        ],
        outputs=[output_table_main, output_table_stats, output_image]
    )

demo.launch()
