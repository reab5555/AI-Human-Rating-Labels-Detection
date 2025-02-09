from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *


def create_spark_session():
    return (SparkSession.builder
            .appName("CSV to Parquet Converter")
            .config("spark.jars.packages", "com.google.cloud:google-cloud-storage:2.22.0")
            .config("spark.sql.parquet.compression.codec", "uncompressed")
            .getOrCreate())


def process_dataframes():
    spark = create_spark_session()

    # 1) Remove the manual schema usage
    df1 = (spark.read.format("csv")
           .option("header", "true")
           .option("inferSchema", "true")  # Let Spark figure it out
           .load("gs://tests1234/datasets/netflix_user_statistics/stats.csv"))

    df2 = (spark.read.format("csv")
           .option("header", "true")
           .option("inferSchema", "true")
           .load("gs://tests1234/datasets/ai_set/stats-ai.csv"))

    # 2) Proceed with sampling for df1
    df1 = df1.sample(False, fraction=4000 / df1.count(), seed=42)

    # 3) Append 0 or 1 to user_id, add ai_label
    df1 = (df1.withColumn("user_id", F.concat(F.col("user_id").cast("string"), F.lit("0")).cast("long"))
               .withColumn("ai_label", F.lit(0)))
    df2 = (df2.withColumn("user_id", F.concat(F.col("user_id").cast("string"), F.lit("1")).cast("long"))
               .withColumn("ai_label", F.lit(1)))

    # 4) Union the dataframes
    final_df = df1.union(df2)

    # 5) Now reorder columns if you *really* want mode_rating after median_rating, etc.
    columns = [
        'user_id',
        'mean_rating',
        'median_rating',
        'mode_rating',
        'sd_rating',
        'max_rating',
        'min_rating',
        'iqr_rating',
        'kurtosis_rating',
        'skewness_rating',
        'entropy_rating',
        'ai_label'
    ]
    final_df = final_df.select(columns)

    # 6) Write to Parquet
    final_df.coalesce(1).write.mode("overwrite").parquet("gs://tests1234/combined_dataset_ml.parquet")
    spark.stop()



if __name__ == "__main__":
    process_dataframes()