# 🚗 Car Resale Price Predictor

A machine learning web application that predicts the resale price of a used car based on its features. The application is built using Python, Scikit-learn, and Streamlit.

## 📌 Project Overview

Buying or selling a used car can be difficult because the price depends on several factors such as the car's age, brand, mileage, engine, fuel type, transmission, and kilometers driven.

This project uses machine learning to estimate the resale price of a used car based on these factors.

## 🎯 Objective

The main objective of this project is to build a regression model that can predict the selling price of a used car using historical Indian car sales data.

## ✨ Features

- 🚗 Used car resale price prediction
- 📊 Real Indian used-car dataset
- 🔤 Handles categorical features using One-Hot Encoding
- 🌳 Random Forest Regression model
- 📈 Model evaluation using MAE and R² Score
- 🖥️ Interactive Streamlit web application
- 💰 Displays predicted price in Indian Rupees and Lakhs

## 📊 Input Features

The model uses the following car details:

- Brand
- Model
- Vehicle Age
- Kilometers Driven
- Seller Type
- Fuel Type
- Transmission Type
- Mileage
- Engine
- Maximum Power
- Number of Seats

## 🤖 Machine Learning Workflow

```text
Real Indian Used-Car Dataset
            ↓
       Data Cleaning
            ↓
      Feature Selection
            ↓
   Categorical Encoding
            ↓
      Train/Test Split
            ↓
   Random Forest Regressor
            ↓
      Model Evaluation
            ↓
     Price Prediction
            ↓
      Streamlit Web App
