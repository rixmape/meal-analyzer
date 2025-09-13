import json
import os
import time

import pandas as pd
import requests
import streamlit as st
from google import genai
from google.genai import types

CONFIG = {
    "STAGES": {
        "CONFIG": "config",
        "CAPTURE": "capture",
        "EDIT_INGREDIENTS": "edit_ingredients",
        "EDIT_PORTIONS": "edit_portions",
        "RESULTS": "results",
    },
    "MODELS": {
        "INGREDIENT_ID": "gemma-3-27b-it",
        "PORTION_ESTIMATE": "gemma-3-27b-it",
    },
    "USDA_API": {
        "BASE_SEARCH_URL": "https://api.nal.usda.gov/fdc/v1/foods/search",
        "BASE_DETAILS_URL": "https://api.nal.usda.gov/fdc/v1/food/",
    },
    "TARGET_NUTRIENTS": {
        "208": "Energy",
        "203": "Protein",
        "205": "Carbohydrates",
        "204": "Total Fat",
        "606": "Saturated Fat",
        "291": "Dietary Fiber",
        "269": "Sugars",
        "307": "Sodium",
    },
    "PROMPTS": {
        "INGREDIENT": (
            "Analyze the image of this meal. Identify the major food ingredients visible. "
            "Exclude minor ingredients like spices or garnishes. "
            "Remove all qualifiers such as size, shape, or preparation style. "
            "Return your response as a valid JSON array of strings. "
            'For example: ["salmon", "asparagus", "lemon"]. '
            "Only output the JSON array."
        ),
        "PORTION": (
            "Analyze the image of the meal. The following ingredients are present: {ingredients}. "
            "For each ingredient, estimate its weight in grams. Assume a standard dinner plate for scale if visible. "
            "Return your response as a single valid JSON object where keys are the ingredient names and values are their estimated integer weights in grams. "
            'For example: {{"salmon": 180, "asparagus": 100, "lemon": 20}}. '
            "Only output the JSON object."
        ),
    },
}

DEFAULT_STATE = {
    "app_stage": CONFIG["STAGES"]["CONFIG"],
    "gemini_api_key": os.getenv("GEMINI_API_KEY", ""),
    "usda_api_key": os.getenv("USDA_API_KEY", ""),
    "results": None,
    "captured_image": None,
    "identified_ingredients": [],
    "portion_estimates": None,
}


def _call_gemini_vision_api(api_key, model_name, prompt, image_bytes):
    try:
        client = genai.Client(api_key=api_key)
        image_part = types.Part.from_bytes(mime_type="image/jpeg", data=image_bytes)
        text_part = types.Part.from_text(text=prompt)
        contents = [types.Content(role="user", parts=[image_part, text_part])]
        response = client.models.generate_content(model=model_name, contents=contents)
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned_text)
    except Exception as e:
        st.error(f"Gemini API Error: Could not process the request. Details: {e}")
        return None


def _reset_analysis_state():
    st.session_state.app_stage = CONFIG["STAGES"]["CAPTURE"]
    st.session_state.captured_image = None
    st.session_state.results = None
    st.session_state.identified_ingredients = []
    st.session_state.portion_estimates = None


def _display_captured_image():
    if st.session_state.captured_image:
        _, col_img, _ = st.columns([1, 2, 1])
        with col_img:
            st.image(st.session_state.captured_image, caption="Your Meal", use_container_width=True)


def get_ingredients_from_image(image_bytes, api_key):
    return _call_gemini_vision_api(
        api_key=api_key,
        model_name=CONFIG["MODELS"]["INGREDIENT_ID"],
        prompt=CONFIG["PROMPTS"]["INGREDIENT"],
        image_bytes=image_bytes,
    )


def get_portion_sizes_from_image(image_bytes, ingredients, api_key):
    ingredients_str = ", ".join(ingredients)
    prompt = CONFIG["PROMPTS"]["PORTION"].format(ingredients=ingredients_str)
    response = _call_gemini_vision_api(
        api_key=api_key,
        model_name=CONFIG["MODELS"]["PORTION_ESTIMATE"],
        prompt=prompt,
        image_bytes=image_bytes,
    )
    return {k.lower(): v for k, v in response.items()} if response else None


def get_nutritional_data(ingredient_portions, api_key, status_container):
    all_data = {}
    for ingredient, portion_size in ingredient_portions.items():
        status_container.write(f"🔬 Fetching data for {portion_size}g of '{ingredient}'...")
        try:
            search_url = f"{CONFIG['USDA_API']['BASE_SEARCH_URL']}?query={requests.utils.quote(ingredient)}&api_key={api_key}&pageSize=1&dataType=Foundation,SR Legacy"
            search_res = requests.get(search_url)
            search_res.raise_for_status()
            search_data = search_res.json()
            if not search_data.get("foods"):
                status_container.warning(f"Could not find '{ingredient}' in USDA database.")
                continue

            fdc_id = search_data["foods"][0]["fdcId"]
            details_url = f"{CONFIG['USDA_API']['BASE_DETAILS_URL']}{fdc_id}?api_key={api_key}"
            details_res = requests.get(details_url)
            details_res.raise_for_status()
            food_data = details_res.json()

            nutrients_found = {}
            for n in food_data.get("foodNutrients", []):
                nutrient_num = str(n.get("nutrient", {}).get("number"))
                if nutrient_num in CONFIG["TARGET_NUTRIENTS"]:
                    key = CONFIG["TARGET_NUTRIENTS"][nutrient_num]
                    value_per_100g = n.get("amount", 0)
                    scaled_value = (value_per_100g / 100) * portion_size
                    unit = n.get("nutrient", {}).get("unitName", "g").lower()
                    nutrients_found[key] = {"value": scaled_value, "unit": unit}
            all_data[ingredient.capitalize()] = nutrients_found
        except Exception as e:
            status_container.warning(f"Could not fetch data for '{ingredient}'. Error: {e}")
    return all_data


def _prepare_results_dataframes(results):
    all_rows = []
    for ingredient, nutrients in results.items():
        if not nutrients:
            continue
        for name, data in nutrients.items():
            all_rows.append({"Ingredient": ingredient, "Nutrient": name, "Value": data.get("value", 0), "Unit": data.get("unit", "")})

    if not all_rows:
        return None, None, None

    df = pd.DataFrame(all_rows)
    totals_df = df.groupby("Nutrient").agg(Value=("Value", "sum"), Unit=("Unit", "first")).reset_index()

    df["Amount"] = df.apply(lambda row: f"{row['Value']:.2f} {row['Unit']}", axis=1)
    pivot_df = df.pivot(index="Nutrient", columns="Ingredient", values="Amount").fillna("—")

    metric_order = ["Energy", "Protein", "Carbohydrates", "Total Fat", "Saturated Fat", "Sugars", "Dietary Fiber", "Sodium"]
    pivot_df = pivot_df.reindex(metric_order, fill_value="—").dropna(how="all")

    return totals_df, pivot_df, df["Ingredient"].unique()


def initialize_state():
    for key, value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state.app_stage == CONFIG["STAGES"]["CONFIG"] and st.session_state.gemini_api_key and st.session_state.usda_api_key:
        st.session_state.app_stage = CONFIG["STAGES"]["CAPTURE"]


def display_config_page():
    st.header("🔑 API Key Configuration")
    st.markdown("Before you can analyze a meal, please provide your API keys.")
    st.text_input("Google Gemini API Key", key="gemini_api_key_input", type="password", value=st.session_state.gemini_api_key)
    st.text_input("USDA FoodData Central API Key", key="usda_api_key_input", type="password", value=st.session_state.usda_api_key)
    if st.button("Save Keys and Start", type="primary", use_container_width=True):
        st.session_state.gemini_api_key = st.session_state.gemini_api_key_input
        st.session_state.usda_api_key = st.session_state.usda_api_key_input
        if st.session_state.gemini_api_key and st.session_state.usda_api_key:
            st.session_state.app_stage = CONFIG["STAGES"]["CAPTURE"]
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
            st.session_state.app_stage = CONFIG["STAGES"]["EDIT_INGREDIENTS"]
            st.rerun()
        else:
            st.session_state.captured_image = None


def display_edit_ingredients_page():
    st.header("✍️ Confirm or Edit Ingredients")
    st.markdown("Review the ingredients identified by the AI. You can add, remove, or change them below.")
    _display_captured_image()
    ingredients_str = st.text_area("Edit the ingredients list:", value=", ".join(st.session_state.get("identified_ingredients", [])), height=100)
    col1, col2 = st.columns(2)
    if col1.button("Estimate Portions", type="primary", use_container_width=True):
        final_ingredients = [item.strip().lower() for item in ingredients_str.split(",") if item.strip()]
        if not final_ingredients:
            st.warning("Please enter at least one ingredient.")
        else:
            with st.spinner("⚖️ Estimating portion sizes with Gemini..."):
                portion_estimates = get_portion_sizes_from_image(st.session_state.captured_image, final_ingredients, st.session_state.gemini_api_key)
            if portion_estimates:
                st.session_state.portion_estimates = portion_estimates
                st.session_state.app_stage = CONFIG["STAGES"]["EDIT_PORTIONS"]
                st.rerun()
    if col2.button("Go Back & Recapture", use_container_width=True):
        _reset_analysis_state()
        st.rerun()


def display_edit_portions_page():
    st.header("⚖️ Confirm or Edit Portion Sizes")
    st.markdown("The AI has estimated the weights of your ingredients. Fine-tune them below for a more accurate analysis.")
    _display_captured_image()
    with st.form(key="portions_form"):
        st.subheader("Edit Estimated Weights (in grams)")
        initial_estimates = st.session_state.get("portion_estimates", {})
        edited_portions = {}
        ingredients = list(initial_estimates.keys())
        cols = st.columns(3)
        for i, ingredient in enumerate(ingredients):
            with cols[i % 3]:
                weight = initial_estimates[ingredient]
                edited_portions[ingredient] = st.number_input(label=ingredient.capitalize(), key=f"weight_{ingredient}", min_value=0, value=int(weight), step=5, help=f"AI estimated this to be ~{weight}g. Edit as needed.")
        if st.form_submit_button("Calculate Nutrition", type="primary", use_container_width=True):
            st.session_state.portion_estimates = edited_portions
            with st.status("Analyzing your meal...", expanded=True) as status:
                nutritional_data = get_nutritional_data(edited_portions, st.session_state.usda_api_key, status)
                st.session_state.results = nutritional_data
                status.update(label="Analysis Complete!", state="complete", expanded=False)
                time.sleep(1)
            st.session_state.app_stage = CONFIG["STAGES"]["RESULTS"]
            st.rerun()
    if st.button("Go Back & Edit Ingredients", use_container_width=True):
        st.session_state.app_stage = CONFIG["STAGES"]["EDIT_INGREDIENTS"]
        st.session_state.portion_estimates = None
        st.rerun()


def display_results_page():
    st.header("🥗 Nutritional Information")
    results = st.session_state.get("results")
    if not results:
        st.warning("No nutritional data could be generated. Please try again.")
    else:
        totals_df, pivot_df, valid_ingredients = _prepare_results_dataframes(results)
        if totals_df is None:
            st.warning("Could not find detailed nutritional data for the specified ingredients.")
        else:
            st.subheader("Final Portions")
            st.caption("The gram weights used for the final calculation.")
            portions = st.session_state.get("portion_estimates", {})
            display_portions = {ing.capitalize(): w for ing, w in portions.items() if ing.capitalize() in valid_ingredients}
            if display_portions:
                cols = st.columns(len(display_portions))
                for i, (ingredient, weight) in enumerate(display_portions.items()):
                    cols[i].metric(label=ingredient, value=f"{weight} g")

            st.subheader("Total Estimated Nutrition")
            st.caption("The combined nutritional values based on the final portion sizes.")
            cols = st.columns(4)
            col_idx = 0
            for nutrient_name in pivot_df.index:
                nutrient_data = totals_df[totals_df["Nutrient"] == nutrient_name]
                if not nutrient_data.empty:
                    value = nutrient_data.iloc[0]["Value"]
                    unit = nutrient_data.iloc[0]["Unit"]
                    val_format = "{:,.0f}" if unit == "kcal" else "{:,.1f}"
                    cols[col_idx % 4].metric(label=nutrient_name, value=f"{val_format.format(value)} {unit}")
                    col_idx += 1

            st.subheader("Per-Ingredient Breakdown")
            st.caption("Nutritional values for the final portion size of each ingredient.")
            st.dataframe(pivot_df, use_container_width=True)

    if st.button("📸 Analyze Another Meal", type="primary", use_container_width=True):
        _reset_analysis_state()
        st.rerun()


if __name__ == "__main__":
    st.set_page_config(page_title="Meal Nutritional Analyzer", page_icon="🥗", layout="centered", initial_sidebar_state="collapsed")
    initialize_state()
    st.title("Meal Nutritional Analyzer")

    PAGE_ROUTER = {
        CONFIG["STAGES"]["CONFIG"]: display_config_page,
        CONFIG["STAGES"]["CAPTURE"]: display_capture_page,
        CONFIG["STAGES"]["EDIT_INGREDIENTS"]: display_edit_ingredients_page,
        CONFIG["STAGES"]["EDIT_PORTIONS"]: display_edit_portions_page,
        CONFIG["STAGES"]["RESULTS"]: display_results_page,
    }

    page_function = PAGE_ROUTER.get(st.session_state.app_stage)
    if page_function:
        page_function()
