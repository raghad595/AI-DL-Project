import streamlit as st
import importlib

# Title
st.set_page_config(page_title="Multi Feature App", layout="wide")
st.title("🔧 Select a Feature")

# Sidebar feature selection
feature = st.sidebar.selectbox(
    "Choose a feature:",
    (
        "Card Scanning",
        "Face Detection",
        "Object Detection",
        "Named Entity",
        "Chatbot",
        "Filters",
    )
)

# Mapping feature name to file/module
feature_map = {
    "Card Scanning": "card_scanner",
    "Face Detection": "face_detector",
    "Object Detection": "object_detector",
    "Named Entity": "text_detector",
    "Chatbot": "chatbot",
    "Filters": "filters",
}

# Import and run the selected module dynamically
selected_module = feature_map[feature]

try:
    module = importlib.import_module(selected_module)
    if hasattr(module, "main"):
        module.main()  # all feature files must have a `main()` function
    else:
        st.error(f"The module '{selected_module}' does not have a main() function.")
except ModuleNotFoundError:
    st.error(f"Module '{selected_module}' not found.")
except Exception as e:
    st.error(f"An error occurred while loading the module '{selected_module}': {e}")
