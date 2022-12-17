import six
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import unix_timestamp, round, stddev, avg, mean, expr, col
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("single_csv") \
        .getOrCreate()
    
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    #get version of pyspark for future refernece
    print('Apache Spark Version :'+spark.version)
    print('Apache Spark Version :'+spark.sparkContext.version)

    file_list =  ['s3a://nyc-tlc/trip data/yellow_tripdata_2021-01.parquet',
                  's3a://nyc-tlc/trip data/yellow_tripdata_2021-02.parquet',
                  's3a://nyc-tlc/trip data/yellow_tripdata_2021-03.parquet',
                  's3a://nyc-tlc/trip data/yellow_tripdata_2021-04.parquet',
                  's3a://nyc-tlc/trip data/yellow_tripdata_2021-05.parquet',
                  's3a://nyc-tlc/trip data/yellow_tripdata_2021-06.parquet'
                 ]

    df = spark.read.parquet(*file_list)

    df.printSchema()
    print("Number of rows:", df.count())
    print("Number of columns:", len(df.columns))

    # creating a list of column names
    cols = df.columns

    # iterate through the list of column names
    for col in cols:
        # get the number of missing values for each column
        missing_count = df.filter(df[col].isNull()).count()
        # print the column name and number of missing values
        print(f"Column {col}: {missing_count} missing values")

    # Calculate the difference between the "start_time" and "end_time" columns in seconds and then making them secondds

    df = df.withColumn("total_time", unix_timestamp(df["tpep_dropoff_datetime"]) - unix_timestamp(df["tpep_pickup_datetime"]))
    df = df.withColumn("total_time", round(df["total_time"]/60, 2))

    #finding the columns with the datatype double and converting them into int
    double_cols=["passenger_count", "trip_distance", "RatecodeID", "fare_amount", "extra", "mta_tax", "tip_amount", "tolls_amount","improvement_surcharge", "total_amount","congestion_surcharge", "total_time"]
    new_df = df
    for col in double_cols:
        new_df = new_df.withColumn(col,new_df[col].cast('integer'))
    new_df=new_df.drop("tpep_pickup_datetime", "tpep_dropoff_datetime", "VendorID", "payment_type")
    #check new datatypes
    new_df.printSchema()

    int_cols = ["passenger_count", "trip_distance", "fare_amount", "extra", "mta_tax", "tip_amount", "tolls_amount","improvement_surcharge", "total_amount","congestion_surcharge", "total_time"]
    new_df.describe(int_cols).show()

#///////////////////////////////////////////////////////////////////////////////////////////remving outlier portion /

    #making a df with only integrs 
    int_only_df = df.select(int_cols)

    cols = int_only_df.columns

    #corrrelation before outlier removal. Good to see how much the data was effected
    #this code is copied from online. Finding correlations from features and total_amount
    for i in int_only_df.columns:
        if not( isinstance(int_only_df.select(i).take(1)[0][0], six.string_types)):
            print( "Correlation to total_amount for ", i, int_only_df.stat.corr('total_amount',i))

    #removing outliers from data
    for col in cols:
        col_mean = int_only_df.select(mean(int_only_df[col])).first()[0]
        col_std = int_only_df.select(stddev(int_only_df[col])).first()[0]

        # Filter out the outliers using the mean and standard deviation
        int_only_df = int_only_df.filter((int_only_df[col] > col_mean - 3 * col_std) & (int_only_df[col] < col_mean + 3 * col_std))

    #this code is copied from online. Checkinig correlation again. HUGE DIFFERENCE
    for i in int_only_df.columns:
        if not( isinstance(int_only_df.select(i).take(1)[0][0], six.string_types)):
            print( "Correlation to total_amount for ", i, int_only_df.stat.corr('total_amount',i))

#///////////////////////////////////////////////////////////////////////////////////////////lin portion /
    featureassembler = VectorAssembler(inputCols=["trip_distance","tip_amount", "total_time"], outputCol= "Independant Features")
    output= featureassembler.transform(int_only_df)
    output.select("Independant Features").show()
    finalised_data = output.select("Independant Features", "total_amount")

    train_set, test_set = finalised_data.randomSplit([0.8, 0.2])

    reg = LinearRegression(featuresCol="Independant Features", labelCol= "total_amount")
    lr_model  = reg.fit(train_set)

    print("Coefficients: " + str(lr_model.coefficients))
    print("Intercept: " + str(lr_model.intercept))

    trainingSummary = lr_model.summary
    print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
    print("r2: %f" % trainingSummary.r2)

    lr_predictions = lr_model.transform(test_set)
    lr_predictions.show()
    
    lr_evaluator = RegressionEvaluator(predictionCol="prediction", \
                     labelCol="total_amount",metricName="r2")

    print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))
    
    #///////////////////////////////////////////////////////////////////////////////////////////log  portion /
    int_only_df = int_only_df.withColumn("tip_binary", expr("CASE WHEN tip_amount > '0' THEN '1' " + 
                   "WHEN tip_amount <= '0' THEN '0' WHEN tip_amount IS NULL THEN ''" +
                   "ELSE tip_amount END"))
    int_only_df = int_only_df.withColumn("tip_binary",int_only_df["tip_binary"].cast('integer'))

    featureassembler = VectorAssembler(inputCols=["passenger_count", "trip_distance", "fare_amount", "tolls_amount", "congestion_surcharge"], outputCol= "features")
    output= featureassembler.transform(int_only_df)
    log_moel_df= output.select("features", "tip_binary")
    train_set, test_set = log_moel_df.randomSplit([0.7, 0.3])

    log_reg = LogisticRegression(labelCol="tip_binary").fit(train_set)
    log_results = log_reg.evaluate(train_set).predictions
    results = log_reg.evaluate(test_set).predictions
    tp = results[(results.tip_binary == 1) & (results.prediction == 1)].count()
    tn = results[(results.tip_binary == 0) & (results.prediction == 0)].count()
    fp = results[(results.tip_binary == 0) & (results.prediction == 1)].count()
    fn = results[(results.tip_binary == 1) & (results.prediction == 0)].count()
    print(f"The TP is {tp}")
    print(f"The TN is {tn}")
    print(f"The FP is {fp}")
    print(f"The FN is {fn}")

    print(f"The accuracy of this model is {float((tp+tn)/results.count())}")
    print(f"The recall of this model is {float(tn)/(tp+tn)}")

    