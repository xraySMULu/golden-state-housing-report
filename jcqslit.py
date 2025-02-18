import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import configparser as config
import altair as alt


st.set_page_config(
    page_title="California Living",
    page_icon="🌉",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Sidebar Navigation
with st.sidebar:
   st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to Model", ["Home", "ROI Predictions", "Interest Predictions","Home Price Predictions", "California Investor Recommendation"])
        # Home Page
if page == "Home":
       st.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <h1 style="margin: 0;">Golden State Housing 🌉</h1>
        <img src="https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F544e2af6-1620-4f6e-a6a7-c7604d5aa54a_800x450.gif" width="450">
    </div>
    """,
    unsafe_allow_html=True
)
       st.markdown(
    r""" 
    
    ## Project Overview

    The primary goal of our project, Golden State Housing Insights, is to predict housing prices in the state of California. 
     Our team aims to achieve this by leveraging machine learning models to analyze various factors, including investor return on investment (ROI), affordability, distance between homes and cities, and specific home features.
     By integrating these elements, we strive to provide accurate and actionable insights into the California housing market, aiding investors, homebuyers, and policymakers in making informed decisions.
   ☀️ 
    ## Business Scenario 
    
      We approached our business problem as independent real estate consultants. Our client, who recently accepted a job offer in California, seeks to purchase a home in the area. They have tasked us with identifying the top 5 metro areas with the highest return on investment (ROI) based on a home feature analysis. Additionally, we will predict near-term interest rates using Time Series modeling and linear regression to support their decision-making.
    
    ## Data Pre-Processing and Machine Learning Steps

     Housing Price Prediction using Zillow Data Analysis
     Our team employed a comprehensive data pre-processing approach to ensure the accuracy and reliability of our housing price predictions. 
     We utilized powerful libraries such as NumPy and Pandas for efficient data manipulation and analysis. Matplotlib was used for visualizing data trends and patterns. We also applied data melting techniques to reshape our datasets,
     making them more suitable for analysis. Additionally, we incorporated time series analysis to account for temporal trends and seasonality in housing prices. This robust pre-processing framework enabled us to prepare our data effectively for machine learning modeling.
     Our team conducted an extensive exploratory data analysis (EDA) to uncover underlying patterns and relationships within the housing data. This initial step allowed us to gain valuable insights and informed our subsequent modeling approach.
     We employed both ARIMA (AutoRegressive Integrated Moving Average) and Auto-ARIMA models to predict housing prices. The ARIMA model helped us understand the temporal dependencies and trends in the data, while the Auto-ARIMA model automated the process of identifying the optimal parameters for our time series forecasting. 
     By comparing the results from these models, we were able to enhance the accuracy and robustness of our housing price predictions.

     ## Model Training & Testing

     Housing Price Prediction using Zillow Data Analysis
      We employed both ARIMA (AutoRegressive Integrated Moving Average) and Auto-ARIMA models to predict housing prices. The ARIMA model helped us understand the temporal dependencies and trends in the data, while the Auto-ARIMA model automated the process of identifying the optimal parameters for our time series forecasting. By comparing the results from these models, we were able to enhance the accuracy and robustness of our housing price predictions.
     Housing Feature Analysis
      We utilized a Linear Regression model for our feature analysis. The model was trained using the model.fit(X_train, y_train) method, allowing it to learn from the training data. After training, we made predictions on the test data with the model.predict(X_test) function. This approach enabled us to assess the model's performance and accuracy effectively. A scatter plot comparing actual and predicted prices based on these three features shows a general alignment, though some deviations indicate potential areas for further model refinement. These findings underscore the significant impact of economic and structural factors on housing prices and demonstrate that while linear regression offers a reasonable predictive capability, there is room for improvement to enhance accuracy.
      We trained models using Linear Regression and Random Forest, employing recursive feature elimination to iteratively remove less important features. Model performance was evaluated using RMSE, R², and MAE metrics, providing a comprehensive understanding of the factors influencing home prices.
     Interest Rate Prediction Analysis
      The visualization includes three key elements: the actual historical interest rates, represented as a line with circle markers; the Linear Regression predictions, depicted as a dashed line with 'x' markers, which illustrate the general trend but may miss some finer details; and the K-Nearest Neighbors (KNN) predictions, shown as a dotted line with square markers, which are potentially more responsive to recent changes in interest rates. According to the Linear Regression and KNN models, interest rates are likely to continue to rise into early 2025. This prediction suggests further challenges for the housing market, potentially leading to continued reduced buyer activity and potential downward pressure on prices.

      """
)

#Page on ROI 
elif page== "ROI Predictions":
         st.title("Return On Investment 💰")
         st.write("### Results of the predictive model created by Chris Gilbert")
         #Show ROI from 2018-2024
         st.image("/Users/jacintocepedaquroz/jcqslit/cg_roi.png", caption="ROI % Change 2018-2023", use_column_width=True)
         #Load CSV file
         file_path = "/Users/jacintocepedaquroz/jcqslit/caliroi.csv"
         roi_df = pd.read_csv(file_path)
         st.subheader("2025 ROI Prediction Results Table")
         st.dataframe(roi_df, use_container_width=True)
         st.bar_chart(roi_df, x="City", y= "ROI")
         st.write(roi_df)
         roi_csv = roi_df.to_csv(index=False).encode("utf-8")
         st.download_button("Download CSV", roi_csv, "roi_predictions.csv", "text/csv", key="download-csv")
          ## Housing Price Prediction using Zillow Data Analysis
         st.subheader("Housing Price Prediction using Zillow Data Analysis" )
         st.markdown(r"""To identify the five most optimal metro areas for investment in California, we automated the process of running time series models (ARIMA and SARIMA) for each of the 34 metro areas. This automation was necessary due to the impracticality of manually analyzing each area. We evaluated the models' accuracy by comparing their predictions for December 2024 with actual observed values. The ARIMA model's predictions were 9.52% off, while the SARIMA model's predictions were 6.76% off. Despite ARIMA performing better for the top 5 metro areas, SARIMA's lower error rate suggests it could be more accurate with more data. The ARIMA model predicted a return on investment (ROI) percentage range of 4% to 5%, whereas the SARIMA model predicted an ROI range of 0% to 2%. Given SARIMA's lower error rate, it is considered more reliable for future predictions.
         """)
         
#Page of interest Rate Predictions
elif page== "Interest Predictions":
         st.title("Interest Predictions 📈")
         st.write("Interest rate predictive model created by Dexter Johnson")
         #Load CSV File 
         file_path_2 = "/Users/jacintocepedaquroz/jcqslit/future_predictions.csv"
         fp_df = pd.read_csv(file_path_2)
         st.write("### Interest Rate Predictions 2025")
         st.dataframe(fp_df)
         fp_csv = fp_df.to_csv(index=False).encode("utf=8")
         st.download_button("Download CSV", fp_csv, "future_predictions.csv", "text/csv", key= "download-csv")
         st.write("### Dexter's Interest Rate Predicition Model Demo")
         st.video("dexdemo.mp4")
         st.subheader("Interest Prediction Analysis")
         st.markdown(r""" The models aim to predict future interest rates based on past trends. Assuming the historical patterns continue, the models could potentially forecast whether interest rates are likely to increase, decrease, or remain stable in the near future (e.g., the next few months). From the code's plot titled "Interest Rate Prediction until Feb 2025," we see an upward trend in interest rates over the past years leading to now. In this scenario, we would generally anticipate a slowdown in the housing market with potentially decreased sales and slower home price appreciation. This model predicts an increase in the interest rates into Feb 2025. This is primarily because interest rates directly affect mortgage rates. Higher mortgage rates make homes less affordable, leading to decreased demand and potentially lower home prices. Lower mortgage rates make homes more affordable, potentially increasing demand and driving up home prices.             
         """ )

#Page of Predicting California House Pricing Using Features
elif page == "Home Price Predictions": 
         st.title("Predicting California House Pricing using Features 🏡")
         st.write("Predictive housing model created by Roderick Burroughs")
         st.subheader("Roderick's Predicting Home Prices Using Home Features Demo")
         st.video("/Users/jacintocepedaquroz/jcqslit/rb.mp4")
         st.subheader("Feature Correlation Heatmap")
         st.image("/Users/jacintocepedaquroz/jcqslit/rb_heatmap.png")
         st.subheader("Linear Regression: Actual vs Predicted House Prices")
         st.image("/Users/jacintocepedaquroz/jcqslit/rb_linearreg.png")
         st.subheader("Housing Feature Analysis")
         st.markdown(r"""  The models aim to predict future interest rates based on past trends. Assuming the historical patterns continue, the models could potentially forecast whether interest rates are likely to increase, decrease, or remain stable in the near future (e.g., the next few months). From the code's plot titled "Interest Rate Prediction until Feb 2025," we see an upward trend in interest rates over the past years leading to now. In this scenario, we would generally anticipate a slowdown in the housing market with potentially decreased sales and slower home price appreciation. This model predicts an increase in the interest rates into Feb 2025. This is primarily because interest rates directly affect mortgage rates. Higher mortgage rates make homes less affordable, leading to decreased demand and potentially lower home prices. Lower mortgage rates make homes more affordable, potentially increasing demand and driving up home prices.             
         """)

#Conclusion Page
elif page == "California Investor Recommendation":
         st.title("California Investor Explanation & Recommendations 🧠")
         st.image("/Users/jacintocepedaquroz/jcqslit/gs1.jpg")
         st.markdown(r"""
         
         ## Client Recommendations
            
         Top 5 California Metro Areas with the highest ROI and best potential for a solid investment
         - Riverside, CA - ROI @ 1 year - 5%
         - Fresno, CA - ROI @ 1 year - 5%
         - Yuba City, CA - ROI @ 1 year - 5%
         - Santa Rosa, CA - ROI @ 1 year - 4%
         - Merced, CA - ROI @ 1 year - 4%
         
         ## Features to consider
         - House age
         - Exterior Quality
         - Garage Size
         - Living space size
         - Kitchen quality
         - Conclusion
         
         ## Model Updates
         The prediction percentages come from models trained on five years of data to predict two years ahead. We plan to use ten years of training data to predict just one year ahead. This suggests that our future predictions will likely be even more accurate than those in this validation test. By leveraging a larger dataset, we aim to enhance the reliability and precision of our investment recommendations.
                     
         """ )
         


