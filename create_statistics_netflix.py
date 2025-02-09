from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window

def calculate_user_statistics():
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("Netflix User Statistics") \
        .getOrCreate()

    # Read the CSV file from GCS
    input_path = "gs://tests1234/datasets/netflix_ratings_300_per_user/ratings.csv"
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    # Remove rows with null values
    df_clean = df.dropna()

    # Calculate mode separately
    w = Window.partitionBy("user_id")
    mode_df = df_clean.groupBy("user_id", "rating") \
        .count() \
        .withColumn("max_count", F.max("count").over(w)) \
        .where(F.col("count") == F.col("max_count")) \
        .groupBy("user_id") \
        .agg(F.first("rating").alias("mode_rating"))

    # Calculate other statistics (note: variance is removed)
    user_stats = df_clean.groupBy("user_id").agg(
        F.mean("rating").alias("mean_rating"),
        F.expr("percentile_approx(rating, 0.5)").alias("median_rating"),
        F.stddev("rating").alias("sd_rating"),
        F.max("rating").alias("max_rating"),
        F.min("rating").alias("min_rating"),
        F.expr("percentile_approx(rating, 0.75) - percentile_approx(rating, 0.25)").alias("iqr_rating"),
        F.kurtosis("rating").alias("kurtosis_rating"),
        F.skewness("rating").alias("skewness_rating")
    )

    # Compute entropy for each user's rating distribution
    # Entropy: H = -sum(p * log(p)) where p = count(rating) / total_ratings
    user_rating_counts = df_clean.groupBy("user_id", "rating") \
        .agg(F.count("*").alias("rating_count"))
    w_user = Window.partitionBy("user_id")
    user_rating_counts = user_rating_counts.withColumn("total_count", F.sum("rating_count").over(w_user))
    user_rating_counts = user_rating_counts.withColumn("p", F.col("rating_count") / F.col("total_count"))
    user_rating_counts = user_rating_counts.withColumn("entropy_component", -F.col("p") * F.log(F.col("p")))
    user_entropy = user_rating_counts.groupBy("user_id") \
        .agg(F.sum("entropy_component").alias("entropy_rating"))

    # Join the mode and entropy calculations with the other statistics
    user_stats = user_stats.join(mode_df, "user_id").join(user_entropy, "user_id")

    # Remove any remaining null values from the statistics
    user_stats_clean = user_stats.dropna()

    # Save results to a single CSV file
    output_path = "gs://tests1234/datasets/netflix_user_statistics/stats.csv"
    user_stats_clean.coalesce(1) \
        .write \
        .option("header", "true") \
        .mode("overwrite") \
        .csv(output_path)

    # Print sample of results
    print("\nSample of calculated statistics:")
    user_stats_clean.show(5)

    print(f"\nStatistics saved to: {output_path}")
    print(f"Number of users processed: {user_stats_clean.count()}")

    spark.stop()

if __name__ == "__main__":
    calculate_user_statistics()
