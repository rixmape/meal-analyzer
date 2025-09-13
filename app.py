import json
import os
import time

import pandas as pd
import requests
import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="Meal Nutritional Analyzer", page_icon="🥗", layout="centered", initial_sidebar_state="collapsed")


def initialize_state():
    if "app_stage" not in st.session_state:
        st.session_state.app_stage = "config"
    if "gemini_api_key" not in st.session_state:
        st.session_state.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    if "usda_api_key" not in st.session_state:
        st.session_state.usda_api_key = os.getenv("USDA_API_KEY", "")
    if "results" not in st.session_state:
        st.session_state.results = None
    if "captured_image" not in st.session_state:
        st.session_state.captured_image = None
    if st.session_state.app_stage == "config" and st.session_state.gemini_api_key and st.session_state.usda_api_key:
        st.session_state.app_stage = "capture"


def get_ingredients_from_image(image_bytes, api_key):
    try:
        client = genai.Client(api_key=api_key)
        model_name = "gemma-3-27b-it"  # Never change this model.
        prompt = (
            "Analyze the image of this meal. Identify the major food ingredients visible. "
            "Exclude minor ingredients such as sauce, spices, and garnishes. "
            "Remove all qualifiers such as size, shape, or preparation style. "
            "Return your response as a valid JSON array of strings. "
            "For example: ['salmon', 'asparagus', 'lemon']. Only output the JSON array."
        )
        image_part = types.Part.from_bytes(mime_type="image/jpeg", data=image_bytes)
        text_part = types.Part.from_text(text=prompt)
        contents = [types.Content(role="user", parts=[image_part, text_part])]
        response = client.models.generate_content(model=model_name, contents=contents)
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned_text)
    except Exception as e:
        st.error(f"Gemini Error: Could not parse ingredients. Please try another photo. Details: {e}")
        return None


def get_nutritional_data(ingredients, api_key, status_container):
    all_data = {}
    target_nutrients = {
        "208": "Energy",
        "203": "Protein",
        "205": "Carbohydrates",
        "204": "Total Fat",
        "606": "Saturated Fat",
        "291": "Dietary Fiber",
        "269": "Sugars",
        "307": "Sodium",
    }
    for ingredient in ingredients:
        status_container.write(f"🔬 Fetching data for '{ingredient}'...")
        try:
            search_url = f"https://api.nal.usda.gov/fdc/v1/foods/search?query={requests.utils.quote(ingredient)}&api_key={api_key}&pageSize=1&dataType=Foundation,SR Legacy"
            search_res = requests.get(search_url)
            search_res.raise_for_status()
            search_data = search_res.json()
            if not search_data.get("foods"):
                status_container.warning(f"Could not find '{ingredient}' in USDA database.")
                continue
            fdc_id = search_data["foods"][0]["fdcId"]
            details_url = f"https://api.nal.usda.gov/fdc/v1/food/{fdc_id}?api_key={api_key}"
            details_res = requests.get(details_url)
            details_res.raise_for_status()
            food_data = details_res.json()
            nutrients_found = {}
            for n in food_data.get("foodNutrients", []):
                nutrient_num = str(n.get("nutrient", {}).get("number"))
                if nutrient_num in target_nutrients:
                    key = target_nutrients[nutrient_num]
                    value = n.get("amount", 0)
                    unit = n.get("nutrient", {}).get("unitName", "g").lower()
                    nutrients_found[key] = {"value": value, "unit": unit}
            all_data[ingredient.capitalize()] = nutrients_found
        except Exception as e:
            status_container.warning(f"Could not fetch data for '{ingredient}'. Error: {e}")
    return all_data


def display_config_page():
    st.header("🔑 API Key Configuration")
    st.markdown("Before you can analyze a meal, please provide your API keys.")
    st.text_input("Google Gemini API Key", key="gemini_api_key_input", type="password")
    st.text_input("USDA FoodData Central API Key", key="usda_api_key_input", type="password")
    if st.button("Save Keys and Start", type="primary", use_container_width=True):
        if st.session_state.gemini_api_key_input and st.session_state.usda_api_key_input:
            st.session_state.gemini_api_key = st.session_state.gemini_api_key_input
            st.session_state.usda_api_key = st.session_state.usda_api_key_input
            st.session_state.app_stage = "capture"
            st.rerun()
        else:
            st.error("Please provide both API keys to continue.")


def display_capture_page():
    st.header("📸 Capture Your Meal")
    st.markdown("Center your meal in the frame and click the capture button.")
    img_file_buffer = st.camera_input("Point your camera at the meal:", key="camera", label_visibility="collapsed")
    if img_file_buffer:
        st.session_state.captured_image = img_file_buffer.getvalue()
        with st.status("Analyzing your meal...", expanded=True) as status:
            status.write("🔎 Identifying ingredients with Gemini...")
            ingredients = get_ingredients_from_image(st.session_state.captured_image, st.session_state.gemini_api_key)
            if ingredients:
                status.write(f"✅ Ingredients identified: **{', '.join(ingredients)}**")
                nutritional_data = get_nutritional_data(ingredients, st.session_state.usda_api_key, status)
                st.session_state.results = nutritional_data
                status.update(label="Analysis Complete!", state="complete", expanded=False)
                time.sleep(1)
                st.session_state.app_stage = "results"
                st.rerun()
            else:
                status.update(label="Could not identify ingredients.", state="error", expanded=True)


def display_results_page():
    st.header("🥗 Nutritional Information")
    results = st.session_state.get("results")
    if not results:
        st.warning("No nutritional data could be generated from your photo. Please try another one.")
    else:
        all_rows = []
        for ingredient, nutrients in results.items():
            if not nutrients:
                continue
            for name, data in nutrients.items():
                all_rows.append({"Ingredient": ingredient, "Nutrient": name, "Value": data.get("value", 0), "Unit": data.get("unit", "")})
        if not all_rows:
            st.warning("Could not find detailed nutritional data for the identified ingredients.")
        else:
            df = pd.DataFrame(all_rows)
            st.subheader("Total Estimated Nutrition")
            st.caption("The combined nutritional values for all identified ingredients (per 100g of each).")
            totals = df.groupby("Nutrient").agg(Value=("Value", "sum"), Unit=("Unit", "first")).reset_index()
            metric_order = ["Energy", "Protein", "Carbohydrates", "Total Fat", "Saturated Fat", "Sugars", "Dietary Fiber", "Sodium"]
            num_metrics = len(metric_order)
            cols = st.columns(min(num_metrics, 4))
            col_idx = 0
            for nutrient_name in metric_order:
                nutrient_data = totals[totals["Nutrient"] == nutrient_name]
                if not nutrient_data.empty:
                    value = nutrient_data.iloc[0]["Value"]
                    unit = nutrient_data.iloc[0]["Unit"]
                    cols[col_idx].metric(label=nutrient_name, value=f"{value:.1f} {unit}")
                    col_idx = (col_idx + 1) % 4
            st.subheader("Per-Ingredient Breakdown")
            st.caption("Nutritional values estimated per 100g of each ingredient.")
            df["Amount"] = df.apply(lambda row: f"{row['Value']:.2f} {row['Unit']}", axis=1)
            pivot_df = df.pivot(index="Nutrient", columns="Ingredient", values="Amount")
            pivot_df.fillna("—", inplace=True)
            pivot_df = pivot_df.reindex(metric_order, fill_value="—").dropna(how="all")
            st.dataframe(pivot_df, use_container_width=True)
    if st.button("📸 Analyze Another Meal", type="primary", use_container_width=True):
        st.session_state.app_stage = "capture"
        st.session_state.captured_image = None
        st.session_state.results = None
        st.rerun()


if __name__ == "__main__":
    initialize_state()
    st.title("Meal Nutritional Analyzer")
    if st.session_state.app_stage == "config":
        display_config_page()
    elif st.session_state.app_stage == "capture":
        display_capture_page()
    elif st.session_state.app_stage == "results":
        display_results_page()
