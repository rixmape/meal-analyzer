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
        st.session_state.gemini_api_key = os.getenv("GEMINI_API_KEY", "") or "AIzaSyCyVaBliLslOBkXV9L4j3vKqZz8wfCBsDA"
    if "usda_api_key" not in st.session_state:
        st.session_state.usda_api_key = os.getenv("USDA_API_KEY", "") or "4Sw4Jcf9q3TmlA7bmznPHsNdDyuioXWqyGxCv5lH"
    if "results" not in st.session_state:
        st.session_state.results = None
    if "captured_image" not in st.session_state:
        st.session_state.captured_image = None
    if "identified_ingredients" not in st.session_state:
        st.session_state.identified_ingredients = []
    if "portion_estimates" not in st.session_state:
        st.session_state.portion_estimates = None
    if st.session_state.app_stage == "config" and st.session_state.gemini_api_key and st.session_state.usda_api_key:
        st.session_state.app_stage = "capture"


def get_ingredients_from_image(image_bytes, api_key):
    try:
        client = genai.Client(api_key=api_key)
        model_name = "gemma-3-27b-it"
        prompt = (
            "Analyze the image of this meal. Identify the major food ingredients visible. "
            "Exclude minor ingredients like spices or garnishes. "
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
        st.error(f"Gemini Error (Ingredient ID): Could not parse ingredients. Please try another photo. Details: {e}")
        return None


def get_portion_sizes_from_image(image_bytes, ingredients, api_key):
    try:
        client = genai.Client(api_key=api_key)
        model_name = "gemma-3-27b-it"  # Do not change this model.
        ingredients_list_str = ", ".join(ingredients)
        prompt = (
            f"Analyze the image of the meal. The following ingredients are present: {ingredients_list_str}. "
            "For each ingredient, estimate its weight in grams. Assume a standard dinner plate for scale if visible. "
            "Return your response as a single valid JSON object where keys are the ingredient names and values are their estimated integer weights in grams. "
            'For example: {"salmon": 180, "asparagus": 100, "lemon": 20}. '
            "Only output the JSON object."
        )
        image_part = types.Part.from_bytes(mime_type="image/jpeg", data=image_bytes)
        text_part = types.Part.from_text(text=prompt)
        contents = [types.Content(role="user", parts=[image_part, text_part])]
        response = client.models.generate_content(model=model_name, contents=contents)
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        return {k.lower(): v for k, v in json.loads(cleaned_text).items()}
    except Exception as e:
        st.error(f"Gemini Error (Portion Sizing): Could not estimate portion sizes. Details: {e}")
        return None


def get_nutritional_data(ingredient_portions, api_key, status_container):
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
    for ingredient, portion_size in ingredient_portions.items():
        status_container.write(f"🔬 Fetching data for ~{portion_size}g of '{ingredient}'...")
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
                    value_per_100g = n.get("amount", 0)
                    scaled_value = (value_per_100g / 100) * portion_size
                    unit = n.get("nutrient", {}).get("unitName", "g").lower()
                    nutrients_found[key] = {"value": scaled_value, "unit": unit}
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
    st.header("📸 Capture or Upload Your Meal")
    st.markdown("Use your camera or upload a file, then wait for the ingredient identification.")
    img_file_buffer = st.camera_input("Point your camera at the meal:", key="camera", label_visibility="collapsed")
    uploaded_file = st.file_uploader("Or upload an image from your device:", type=["jpg", "jpeg", "png"])
    image_buffer = img_file_buffer or uploaded_file
    if image_buffer:
        st.session_state.captured_image = image_buffer.getvalue()
        with st.spinner("🔎 Identifying ingredients with Gemini..."):
            ingredients = get_ingredients_from_image(st.session_state.captured_image, st.session_state.gemini_api_key)
        if ingredients:
            st.session_state.identified_ingredients = ingredients
            st.session_state.app_stage = "edit_ingredients"
            st.rerun()
        else:
            st.session_state.captured_image = None


def display_edit_page():
    st.header("✍️ Confirm or Edit Ingredients")
    st.markdown("Review the ingredients identified by the AI. You can add, remove, or change them below.")
    if st.session_state.captured_image:
        st.image(st.session_state.captured_image, caption="Your Meal", width=250)
    ingredients_str = st.text_area("Edit the ingredients list:", value=", ".join(st.session_state.get("identified_ingredients", [])), height=100)
    col1, col2 = st.columns(2)
    if col1.button("Analyze Nutrition", type="primary", use_container_width=True):
        final_ingredients = [item.strip().lower() for item in ingredients_str.split(",") if item.strip()]
        if not final_ingredients:
            st.warning("Please enter at least one ingredient.")
        else:
            with st.status("Analyzing your meal...", expanded=True) as status:
                status.write(f"✅ Ingredients confirmed: **{', '.join(final_ingredients)}**")
                status.write("⚖️ Estimating portion sizes with Gemini...")
                portion_estimates = get_portion_sizes_from_image(st.session_state.captured_image, final_ingredients, st.session_state.gemini_api_key)
                if portion_estimates:
                    st.session_state.portion_estimates = portion_estimates
                    status.write("✅ Portion sizes estimated!")
                    nutritional_data = get_nutritional_data(portion_estimates, st.session_state.usda_api_key, status)
                    st.session_state.results = nutritional_data
                    status.update(label="Analysis Complete!", state="complete", expanded=False)
                    time.sleep(1)
                    st.session_state.app_stage = "results"
                    st.rerun()
                else:
                    status.update(label="Portion estimation failed.", state="error", expanded=True)
    if col2.button("Go Back & Recapture", use_container_width=True):
        st.session_state.app_stage = "capture"
        st.session_state.captured_image = None
        st.session_state.identified_ingredients = []
        st.rerun()


def display_results_page():
    st.header("🥗 Nutritional Information")
    results = st.session_state.get("results")
    if not results:
        st.warning("No nutritional data could be generated. Please try again.")
    else:
        all_rows = []
        for ingredient, nutrients in results.items():
            if not nutrients:
                continue
            for name, data in nutrients.items():
                all_rows.append({"Ingredient": ingredient, "Nutrient": name, "Value": data.get("value", 0), "Unit": data.get("unit", "")})
        if not all_rows:
            st.warning("Could not find detailed nutritional data for the specified ingredients.")
        else:
            df = pd.DataFrame(all_rows)
            st.subheader("Estimated Portions")
            portions = st.session_state.get("portion_estimates", {})
            formatted_portions = {ing.capitalize(): w for ing, w in portions.items()}
            display_portions = {ing: w for ing, w in formatted_portions.items() if ing in df["Ingredient"].unique()}
            if display_portions:
                cols = st.columns(len(display_portions))
                for i, (ingredient, weight) in enumerate(display_portions.items()):
                    cols[i].metric(label=ingredient, value=f"{weight} g")
            st.subheader("Total Estimated Nutrition")
            st.caption("The combined nutritional values based on the estimated portion sizes.")
            totals = df.groupby("Nutrient").agg(Value=("Value", "sum"), Unit=("Unit", "first")).reset_index()
            metric_order = ["Energy", "Protein", "Carbohydrates", "Total Fat", "Saturated Fat", "Sugars", "Dietary Fiber", "Sodium"]
            cols = st.columns(4)
            col_idx = 0
            for nutrient_name in metric_order:
                nutrient_data = totals[totals["Nutrient"] == nutrient_name]
                if not nutrient_data.empty:
                    value = nutrient_data.iloc[0]["Value"]
                    unit = nutrient_data.iloc[0]["Unit"]
                    val_format = "{:,.0f}" if unit == "kcal" else "{:,.1f}"
                    cols[col_idx].metric(label=nutrient_name, value=f"{val_format.format(value)} {unit}")
                    col_idx = (col_idx + 1) % 4
            st.subheader("Per-Ingredient Breakdown")
            st.caption("Nutritional values for the estimated portion size of each ingredient.")
            df["Amount"] = df.apply(lambda row: f"{row['Value']:.2f} {row['Unit']}", axis=1)
            pivot_df = df.pivot(index="Nutrient", columns="Ingredient", values="Amount")
            pivot_df.fillna("—", inplace=True)
            pivot_df = pivot_df.reindex(metric_order, fill_value="—").dropna(how="all")
            st.dataframe(pivot_df, use_container_width=True)
    if st.button("📸 Analyze Another Meal", type="primary", use_container_width=True):
        st.session_state.app_stage = "capture"
        st.session_state.captured_image = None
        st.session_state.results = None
        st.session_state.identified_ingredients = []
        st.session_state.portion_estimates = None
        st.rerun()


if __name__ == "__main__":
    initialize_state()
    st.title("Meal Nutritional Analyzer")
    if st.session_state.app_stage == "config":
        display_config_page()
    elif st.session_state.app_stage == "capture":
        display_capture_page()
    elif st.session_state.app_stage == "edit_ingredients":
        display_edit_page()
    elif st.session_state.app_stage == "results":
        display_results_page()
