# NYC Taxi Big Data Analysis With Python Pyspark

This project delves into the New York City Taxi and Limousine Commission (TLC) Trip Record Data, spanning over 13 years of operational TLCs in NYC. Leveraging Python and PySpark, the goal is to gain insights into the dynamics of the taxi industry, understand the impact of ride-hailing services like Uber and Lyft, and forecast future trends in taxi prices and demand. The dataset is sourced from Amazon S3 as part of the OpenData initiative. It includes records of taxi trips, encompassing trip prices, durations, distances, and other statistics. The dataset is segmented into various categories, including Yellow cab trip records, Green cab trip records, For-hire vehicle trip records, and High-volume For-hire trip records. Each segment is further divided by year and month, resulting in a large volume of data spread across multiple CSV and Parquet files.

Files below are these. 

1. ```cis_4130_project_7.pdf```
(contains all writings of the project, and the step by step work for how I approached this project)

2. ```first_analysis.ipynb```
(the code written to do basic analysis and visualizations for the given data in jupyter notebook)

3. ```python_final_4.py```
(the code written to launch in an AWS EMR cluster and yield results)

5. ```test_for_emr.ipynb```
(the same code as above but incremental as its a Jupyter Notebook file)

4. ```Folder named visualizations```
(contains visualizations from the project)

 
## Project Objectives

- Calculate averages for price and ride length to assess the overall market dynamics.
- Investigate the relationship between price, trip length, and location.
- Analyze tipping behavior and factors influencing tipping decisions.
- Identify irregularities in the data and perform data preprocessing to ensure data quality.
- Build regression models to predict future taxi prices and demand.

## Insights and Conclusions

Through exploratory data analysis and regression modeling, several insights were gained:

- Tolls appear to have a significant influence on tipping behavior, with customers more likely to tip when tolls are incurred during the trip.
- Despite the majority of rides being short distances (less than one mile), a significant portion of customers do not tip, potentially due to the perception that short rides do not warrant a tip.
- Data irregularities, such as negative values for miles driven, highlight challenges in data reporting by taxi drivers, necessitating robust data preprocessing techniques.

## Future Directions

Plans for this project include:

- Conducting in-depth analysis to identify specific conditions that influence tipping behavior, enabling drivers to optimize their earnings.
- Exploring additional factors, such as weather conditions and time of day, that may impact taxi demand and pricing.
- Implementing advanced machine learning algorithms to improve prediction accuracy and gain deeper insights into market dynamics.
- Collaborating with industry stakeholders to incorporate real-time data and enhance the predictive capabilities of the models.

By continuing to analyze and leverage NYC taxi big data, this project aims to contribute valuable insights to the transportation industry and inform decision-making processes for both taxi drivers and policymakers.

