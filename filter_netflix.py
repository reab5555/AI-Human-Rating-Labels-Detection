from pyspark.sql import SparkSession
from pyspark.sql.functions import count, rand, row_number
from pyspark.sql.window import Window


def analyze_ratings():
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("Netflix Ratings Analysis") \
        .getOrCreate()

    # Read the CSV file
    input_path = "gs://tests1234/datasets/netflix_ratings_with_titles.csv"
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    # Get users with at least 300 ratings
    users_with_enough_ratings = df.groupBy("user_id") \
        .count() \
        .filter("count >= 300") \
        .select("user_id")

    # Join back with original data to get all ratings for these users
    qualified_ratings = df.join(users_with_enough_ratings, "user_id")

    # Create a window spec for row numbering
    window = Window.partitionBy("user_id").orderBy(rand())

    # Sample exactly 300 random ratings for each user
    sampled_ratings = qualified_ratings \
        .withColumn("row_num", row_number().over(window)) \
        .filter("row_num <= 300") \
        .drop("row_num")

    # Count unique users in the sampled dataset
    unique_users = sampled_ratings.select("user_id").distinct().count()

    # Save the sampled ratings to CSV
    output_path = "gs://tests1234/datasets/netflix_ratings_300_per_user/ratings.csv"
    sampled_ratings.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)

    print(f"\nSaved sampled ratings to: {output_path}")
    print(f"Number of unique users in the output dataset: {unique_users}")
    print(f"Each qualifying user has exactly 300 random ratings in the output file.\n")

    spark.stop()


if __name__ == "__main__":
    analyze_ratings()