import time
import json
import random
import uuid  # Imported if needed elsewhere
from datetime import datetime

import pandas as pd
import vertexai
from vertexai.batch_prediction import BatchPredictionJob
from google.cloud import storage


def read_ratings_data():
    """
    Reads the ratings CSV from a local file.
    The CSV is expected to have at least the following columns:
      - "user_id": the user's identifier.
      - "movie": the title of the movie rated.
      - "movie_id": the identifier for the movie.
    Returns:
        pd.DataFrame: A DataFrame containing the ratings data.
    """
    input_path = r"C:\Works\Data\projects\Python\AI_labler_detection\datasets\datasets_netflix_ratings_300_per_user_ratings.csv_part-00000-ec43e2a1-c8e4-4641-8e32-1eef1f122cc7-c000.csv"
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} ratings records from local file.")
    unique_users = df['user_id'].nunique()
    print(f"Number of unique users: {unique_users}")
    return df


def clean_text(text):
    """Optional: Clean text to remove problematic characters."""
    if not isinstance(text, str):
        text = str(text)
    return text.strip()


def create_netflix_batch_input_jsonl(ratings_df, bucket_name="tests1234", total_users=4000):
    """
    Creates and uploads a JSONL (NDJSON) file to GCS containing instances for batch prediction.
    Validates each JSON line locally, skipping any malformed ones.
    Returns:
        str: The GCS URI of the uploaded NDJSON file.
    """
    instances = []
    possible_temperatures = [1.0, 0.8, 0.5, 0.25, 0.1, 0.01]

    unique_users = ratings_df['user_id'].unique().tolist()
    if len(unique_users) < total_users:
        raise ValueError("Not enough users in the dataset to sample the requested number.")

    sampled_users = random.sample(unique_users, total_users)
    total_instances = 0

    for user in sampled_users:
        user_temperature = random.choice(possible_temperatures)
        user_ratings = ratings_df[ratings_df["user_id"] == user]
        for _, row in user_ratings.iterrows():
            movie = clean_text(row["movie"])
            movie_id = row["movie_id"]
            instance_data = {
                "request": {
                    "contents": [{
                        "role": "user",
                        "parts": [
                            {
                                "text": (
                                    f"As a Netflix user, please rate the following movie on a scale of 1 to 5.\n"
                                    f"Movie: {movie}\n"
                                    "Provide only the numeric rating."
                                )
                            }
                        ]
                    }],
                    "generationConfig": {
                        "max_output_tokens": 64,
                        "temperature": user_temperature
                    }
                },
                "user_id": user,
                "movie": movie,
                "movie_id": movie_id,
                "temperature": user_temperature
            }
            try:
                instance_str = json.dumps(instance_data)
                instances.append(instance_str)
                total_instances += 1
            except Exception as e:
                print(f"Skipping instance for user {user} and movie {movie} due to error: {e}")
                continue

    # Validate each JSON line.
    valid_lines = []
    for i, line in enumerate(instances):
        try:
            json.loads(line)
            valid_lines.append(line)
        except Exception as e:
            print(f"Skipping invalid JSON line at index {i}: {e}")

    ndjson_content = "\n".join(valid_lines) + "\n"
    print(f"Created NDJSON content with {len(valid_lines)} valid instances (out of {total_instances}).")
    print("NDJSON preview (first 500 characters):", ndjson_content[:500])

    # Local validation: write to file and check each line.
    local_ndjson_path = "local_ndjson.jsonl"
    with open(local_ndjson_path, "w", encoding="utf-8") as f:
        f.write(ndjson_content)
    with open(local_ndjson_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                json.loads(line)
            except Exception as e:
                print(f"Error in line {i} of local NDJSON: {e}")

    # Upload NDJSON to GCS.
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    input_blob = f"vertex_ai_input/netflix_batch_input_{timestamp}.jsonl"
    blob = bucket.blob(input_blob)
    blob.upload_from_string(ndjson_content, content_type="text/plain")
    print("Uploaded NDJSON file to GCS.")

    return f"gs://{bucket_name}/{input_blob}"


def process_output_to_csv(bucket_name, output_prefix, output_csv_path):
    """
    Processes the output NDJSON blobs from GCS and writes predictions to a CSV.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=output_prefix))
    if not blobs:
        print("No output blobs found under the prefix:", output_prefix)
        return

    records = []
    invalid_count = 0
    for blob in blobs:
        print("Processing blob:", blob.name)
        content = blob.download_as_string().decode("utf-8")
        for line in content.splitlines():
            try:
                record = json.loads(line)
                movie = record.get("movie", "")
                user_id = record.get("user_id", "")
                movie_id = record.get("movie_id", "")
                temperature = record.get("temperature", None)
                rating = None
                if "response" in record and record["response"]:
                    response = record["response"]
                    if ("candidates" in response and isinstance(response["candidates"], list)
                        and response["candidates"]):
                        candidate = response["candidates"][0]
                        parts = candidate.get("content", {}).get("parts", [])
                        if parts:
                            rating_text = parts[0].get("text", "").strip()
                            try:
                                rating = float(rating_text)
                            except Exception:
                                invalid_count += 1
                                continue
                if rating is not None:
                    records.append({
                        "movie": movie,
                        "user_id": user_id,
                        "movie_id": movie_id,
                        "temperature": temperature,
                        "rating": rating
                    })
            except Exception as e:
                print(f"Error parsing line in blob {blob.name}: {e}")
    if records:
        df = pd.DataFrame(records)
        df.to_csv(output_csv_path, index=False)
        print(f"CSV output written to {output_csv_path}")
        print(f"Number of total samples in the output CSV file: {len(records)}")
    else:
        print("No records were parsed from the output blobs.")

    print(f"Number of samples with non-numeric ratings skipped: {invalid_count}")


def main():
    vertexai.init(location="us-central1")
    ratings_df = read_ratings_data()
    input_uri = create_netflix_batch_input_jsonl(ratings_df, bucket_name="tests1234", total_users=4000)
    print("Input file created at:", input_uri)
    output_uri_prefix = "gs://tests1234/datasets/ai_set/ratings_output/"
    print("Submitting batch prediction job...")
    batch_prediction_job = BatchPredictionJob.submit(
        source_model="gemini-2.0-flash-001",
        input_dataset=input_uri,
        output_uri_prefix=output_uri_prefix
    )
    print(f"Job resource name: {batch_prediction_job.resource_name}")
    while not batch_prediction_job.has_ended:
        time.sleep(5)
        batch_prediction_job.refresh()
        print(f"Job state: {batch_prediction_job.state.name}")
    if batch_prediction_job.has_succeeded:
        print("Batch prediction job succeeded!")
        print("Results saved to:", batch_prediction_job.output_location)
        output_prefix = batch_prediction_job.output_location.split("gs://tests1234/")[1]
        output_csv_path = "synthetic_ratings.csv"
        process_output_to_csv(bucket_name="tests1234", output_prefix=output_prefix, output_csv_path=output_csv_path)
    else:
        print("Batch prediction job failed!")
        print("Error:", batch_prediction_job.error)


if __name__ == "__main__":
    main()
