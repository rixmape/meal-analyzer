# 🥗 Meal Nutritional Analyzer

This application is a nutritional analyzer that estimates the caloric and macronutrient content of a meal from a single photograph.

![](screen.png)

It uses the Gemini AI to visually identify food items and estimate their weight, then fetches precise nutritional data for those items from the USDA FoodData Central API. The user is guided through a workflow to capture an image, review and edit the AI-generated food list and portion sizes, and finally view a detailed breakdown of the meal's nutritional value.

```mermaid
sequenceDiagram
    participant User
    participant Streamlit App
    participant Gemini API
    participant USDA API

    User->>Streamlit App: 1. Provide API Keys
    activate Streamlit App
    Streamlit App->>User: 2. Request Meal Image
    deactivate Streamlit App

    User->>Streamlit App: 3. Upload Image
    activate Streamlit App
    Streamlit App->>Gemini API: 4. Identify Ingredients from Image
    activate Gemini API
    Gemini API-->>Streamlit App: Return Ingredients
    deactivate Gemini API

    Streamlit App->>User: 5. Display Ingredients for Confirmation
    deactivate Streamlit App
    User->>Streamlit App: 6. Confirm/Edit Ingredients
    activate Streamlit App

    Streamlit App->>Gemini API: 7. Estimate Portions from Image
    activate Gemini API
    Gemini API-->>Streamlit App: Return Portion Estimates (grams)
    deactivate Gemini API

    Streamlit App->>User: 8. Display Portions for Confirmation
    deactivate Streamlit App
    User->>Streamlit App: 9. Confirm/Edit Portions
    activate Streamlit App

    Streamlit App->>USDA API: 10. Fetch Nutritional Data for Final Portions
    activate USDA API
    USDA API-->>Streamlit App: Return Aggregated Nutritional Data
    deactivate USDA API

    Streamlit App->>User: 11. Display Final Nutritional Analysis
    deactivate Streamlit App

    alt Start Over
        User->>Streamlit App: 12. Analyze Another Meal
    end
```
