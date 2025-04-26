import streamlit as st
import pandas as pd
import time
from data_processing.preprocessing import DataPreprocessor
from recommender import LaptopRecommender

# Set page config
st.set_page_config(page_title="Laptop Recommender", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for styling
def load_css():
    st.markdown("""
        <style>
        /* Light blue styling for text input */
        .stTextInput input {
            background-color: #f0f8ff;
            border: 1px solid #d1e3ff;
            border-radius: 4px;
            color: black;
        }
        .stTextInput input:focus {
            border-color: #0077ff;
            box-shadow: 0 0 0 2px rgba(0, 119, 255, 0.2);
        }
        .stTextInput input::placeholder {
            color: #666666;
            opacity: 1;
        }

        /* Buttons */
        button, .stButton>button {
            background: linear-gradient(135deg, #00ffe7, #0077ff);
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.50em 0.80em;
            font-weight: bold;
            box-shadow: 0 0 8px #00ffe799;
            transition: all 0.3s ease-in-out;
        }
        button:hover, .stButton>button:hover {
            transform: scale(1.05);
            background: linear-gradient(135deg, #0077ff, #00ffe7);
            box-shadow: 0 0 12px #00ffe7;
        }


        /* Loading Spinner Placeholder */
        .loading-spinner {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 200px;
        }
        .spinner {
            width: 50px;
            height: 50px;
            border: 6px solid #00ffe7;
            border-top: 6px solid transparent;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>
    """, unsafe_allow_html=True)

def show_loading_animation():
    """Displays a loading spinner"""
    st.markdown("""
        <div class="loading-spinner">
            <div class="spinner"></div>
        </div>
        <p style='text-align: center;'>Processing your data...</p>
    """, unsafe_allow_html=True)

def load_default_data():
    """Loads and caches the default preprocessed data"""
    if 'default_data' not in st.session_state:
        with st.spinner("Loading default data..."):
            preprocessor = DataPreprocessor()
            try:
                df = (preprocessor.load_data("assets/default_data.csv")
                      .initial_clean()
                      .process_features()
                      .extract_processor_brand()
                      .process_ram()
                      .process_storage_details()
                      .process_warranty()
                      .process_display()
                      .process_gpu()
                      .get_processed_data())
                st.session_state.default_data = df
            except FileNotFoundError:
                st.error("Default data file not found. Please upload your own data.")
                return None
    return st.session_state.default_data

def process_uploaded_data(uploaded_file):
    """Processes user uploaded data with loading animation"""
    placeholder = st.empty()
    with placeholder:
        show_loading_animation()
        
    try:
        with open("temp_upload.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        preprocessor = DataPreprocessor()
        df = (preprocessor.load_data("temp_upload.csv")
                  .initial_clean()
                  .process_features()
                  .extract_processor_brand()
                  .process_ram()
                  .process_storage_details()
                  .process_warranty()
                  .process_display()
                  .process_gpu()
                  .get_processed_data())
        
        time.sleep(2)
        placeholder.empty()
        return df
        
    except Exception as e:
        placeholder.empty()
        st.error(f"Error processing file: {str(e)}")
        return None

def display_laptop_comparison(recommendations):
    """Display recommended laptops in a 3x2 grid using Streamlit components"""
    st.markdown(f"<h1 style='text-align: center;'>🌟 Top Recommendations</h1>", unsafe_allow_html=True)
    
    # Create mapping dictionaries to decode the encoded values
    drive_type_map = {0: 'Unknown', 1: 'HDD', 2: 'eMMC', 3: 'SSD'}
    os_map = {0: 'Unknown', 1: 'Chrome OS', 2: 'Windows', 3: 'macOS'}
    
    # Limit to 6 recommendations max
    recommendations = recommendations.head(6)
    
    # Create rows and columns
    for i in range(0, len(recommendations), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(recommendations):
                row = recommendations.iloc[i + j]
                with cols[j]:
                    # Create card container with a subtle border
                    with st.container():
                        st.markdown(
                            f"""
                            <div style='
                                padding: 1rem;
                                border-radius: 0.5rem;
                                border: 1px solid var(--border-color);
                                height: 100%;
                            '>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        # Name (with truncation)
                        st.markdown(f"**{row['Name'][:100]}{'...' if len(row['Name']) > 100 else ''}**")
                        
                        # Image - handle case where column doesn't exist
                        if 'Image URL' in recommendations.columns and pd.notna(row["Image URL"]):
                            try:
                                st.image(row["Image URL"], use_container_width=True)
                            except:
                                st.write("Image unavailable")
                        else:
                            st.write("No image available")
                        
                        # Price (keep the red color for price as it's important)
                        st.markdown(f"<span style='color:#d32f2f; font-weight:bold; font-size:18px;'>₹{row['Price']:,}</span>", 
                                   unsafe_allow_html=True)
                        
                        # Rating
                        if 'Rating' in recommendations.columns:
                            num_ratings = f"({row['Num Ratings']} reviews)" if 'Num Ratings' in recommendations.columns else ""
                            rating_display = f"<span style='color:#ffb400;'>★</span> {row['Rating']} {num_ratings}"
                            st.markdown(
                                f"<div style='display:flex; justify-content:space-between; margin:8px 0;'>"
                                f"<span style='font-weight:600;'>Rating:</span>"
                                f"<span style='font-weight:500;'>{rating_display}</span>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                        
                        # Processor
                        if 'Processor' in recommendations.columns:
                            processor = row['Processor'][:20] + ('...' if len(row['Processor']) > 20 else '')
                            st.markdown(
                                f"<div style='display:flex; justify-content:space-between; margin:8px 0;'>"
                                f"<span style='font-weight:600;'>Processor:</span>"
                                f"<span style='font-weight:500;'>{processor}</span>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                        
                        # RAM - show original values
                        if 'Ram_Size' in recommendations.columns:
                            ram_display = f"{row['Ram_Size']} GB"
                            st.markdown(
                                f"<div style='display:flex; justify-content:space-between; margin:8px 0;'>"
                                f"<span style='font-weight:600;'>RAM:</span>"
                                f"<span style='font-weight:500;'>{ram_display}</span>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                        
                        # Storage - show original values
                        if 'Primary_Drive_Size_GB' in recommendations.columns:
                            drive_type = drive_type_map.get(row['Primary_Drive_Type'], '') if 'Primary_Drive_Type' in recommendations.columns else ''
                            storage_display = f"{int(row['Primary_Drive_Size_GB'])} GB {drive_type}"
                            st.markdown(
                                f"<div style='display:flex; justify-content:space-between; margin:8px 0;'>"
                                f"<span style='font-weight:600;'>Storage:</span>"
                                f"<span style='font-weight:500;'>{storage_display}</span>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                        
                        # Operating System - show original name
                        if 'Operating System' in recommendations.columns:
                            os_name = os_map.get(row['Operating System'], row['Operating System'])
                            st.markdown(
                                f"<div style='display:flex; justify-content:space-between; margin:8px 0;'>"
                                f"<span style='font-weight:600;'>OS:</span>"
                                f"<span style='font-weight:500;'>{os_name}</span>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                        
                        # Warranty - show original value
                        if 'Warranty' in recommendations.columns:
                            st.markdown(
                                f"<div style='display:flex; justify-content:space-between; margin:8px 0;'>"
                                f"<span style='font-weight:600;'>Warranty:</span>"
                                f"<span style='font-weight:500;'>{row['Warranty']} Year</span>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                        
                        # View Product button - only if column exists
                        if 'Product Link' in recommendations.columns:
                            st.link_button("View Product", row['Product Link'])
                        
                        st.markdown("</div>", unsafe_allow_html=True)

def main():
    # Load custom CSS (if you want to keep other styling)
    load_css()

    # Center the title and subtitle
    st.markdown("<h3 style='text-align: center;'>Laptop Recommendation</h1>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; font-style: italic;'>Find the perfect laptop based on your preferences</h4>", unsafe_allow_html=True)

    # Initialize session state
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None

    # Create a container with columns to control width
    col1, container_col, col2 = st.columns([1, 6, 1])
    
    # Use the middle column as our container
    with container_col:
        # Use streamlit's native container with padding
        with st.container():
            # Create a card effect using expander but keep it always open
            with st.expander("🔍 Search Parameters", expanded=True):
                col1, col2 = st.columns([2, 3])
                with col1:
                    data_source = st.radio(
                        "Data Source:",
                        ["Use Default Dataset", "Upload Your Own Data"],
                        help="Data should be scraped using the scraper provided",
                        horizontal=True
                    )
                with col2:
                    max_price = st.slider(
                        "Maximum Price (₹)",
                        min_value=10000,
                        max_value=300000,
                        value=50000,
                        step=5000,
                        help="Recommendations will be within ±20% of this price"
                    )


                # Rest of your inputs inside this container_col
                if data_source == "Upload Your Own Data":
                    upload_col1, upload_col2 = st.columns(2)
                    with upload_col1:
                        uploaded_file = st.file_uploader(
                            "Upload CSV File",
                            type=["csv"],
                            label_visibility="visible"
                        )
                    with upload_col2:
                        st.write("")
                        if st.button("Get Scraper", use_container_width=True):
                            st.switch_page("pages/scrap_info.py")

                    if uploaded_file:
                        processed_data = process_uploaded_data(uploaded_file)
                        if processed_data is not None:
                            st.session_state.processed_data = processed_data
                            st.success("Data processed successfully!")
                    else:
                        st.session_state.processed_data = None
                else:
                    processed_data = load_default_data()
                    st.session_state.processed_data = processed_data

                # Keyword input + Get Recommendations button with more prominent layout
                search_cols = st.columns([1, 4, 1.5])

                with search_cols[0]:
                    # Add vertical spacing to align with the text input
                    st.markdown("""
                        <div style="display: flex; align-items: center; height: 38px; margin-top: 4px;">
                            <span style="color: #0077ff; font-weight: 600; display: flex; align-items: center;">
                                <span style="margin-left: 5px;">Keyword Search:</span>
                            </span>
                        </div>
                    """, unsafe_allow_html=True)

                with search_cols[1]:
                    keyword = st.text_input(
                        "Keyword Search (Optional)",
                        placeholder="e.g., nvidia, apple",
                        help="Enter keywords to find specific laptops",
                        label_visibility="collapsed",
                    )

                with search_cols[2]:
                    
                    recommend_clicked = st.button("Get Recommendations", use_container_width=True)

    # Rest of your code outside the container
    if st.session_state.processed_data is None:
        st.warning("Please load or upload data to get recommendations")
        st.stop()

    # Metrics - Centered using columns
    st.markdown("<h3 style='text-align: center;'>Data Analytics</h3>", unsafe_allow_html=True)
            
    # Create a container with centered columns for metrics
    metrics_col1, metrics_container, metrics_col2 = st.columns([1, 6, 1])
    
    with metrics_container:
        if len(st.session_state.processed_data) > 0:
            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("Total Laptops", len(st.session_state.processed_data))
            mcol2.metric("Average Price", f"₹{st.session_state.processed_data['Price'].mean():,.0f}")
            mcol3.metric("Average Rating", f"{st.session_state.processed_data['Rating'].mean():.1f} ★")
            
            # Center the caption
            st.markdown(f"<p style='text-align: center;'>Showing recommendations under ₹{max_price:,}</p>", unsafe_allow_html=True)

    # Recommendations
    if recommend_clicked or (keyword and keyword != st.session_state.get('last_keyword', '')):
        with st.spinner("Finding best matches..."):
            try:
                recommender = LaptopRecommender(st.session_state.processed_data)
                if keyword and str(keyword).strip():
                    st.session_state.recommendations = recommender.recommend_laptops(
                        keyword=keyword,
                        price_limit=max_price,
                        top_n=6
                    )
                    st.session_state.last_keyword = keyword
                else:
                    st.session_state.recommendations = recommender.recommend_top_rated(
                        price_limit=max_price,
                        top_n=6
                    )
                    st.session_state.last_keyword = ''
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")
                st.session_state.recommendations = None

    if st.session_state.recommendations is not None:
        display_laptop_comparison(st.session_state.recommendations)


if __name__ == "__main__":
    main()