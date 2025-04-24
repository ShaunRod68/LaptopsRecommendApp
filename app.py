import streamlit as st
import pandas as pd
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from data_processing.preprocessing import DataPreprocessor

# Set page config
st.set_page_config(page_title="Laptop Recommender", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    /* Main container styling */
    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
    }
    
    /* Card styling */
    .laptop-card {
        border: 1px solid #e1e4e8;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s;
        height: 100%;
    }
    .laptop-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Image styling */
    .laptop-image-container {
        height: 160px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 10px;
    }
    .laptop-image {
        max-height: 140px;
        width: auto;
        border-radius: 5px;
    }
    
    /* Price tag */
    .price-tag {
        font-size: 18px;
        font-weight: bold;
        color: #d32f2f;
        margin: 10px 0;
    }
    
    /* Feature rows */
    .feature-row {
        display: flex;
        justify-content: space-between;
        margin: 8px 0;
        font-size: 14px;
        line-height: 1.4;
    }
    .feature-name {
        font-weight: 600;
        color: #555;
    }
    .feature-value {
        font-weight: 500;
        color: #222;
    }
    
    /* Rating stars */
    .rating {
        display: flex;
        align-items: center;
        gap: 5px;
    }
    .star {
        color: #ffb400;
    }
    
    /* View button */
    .view-button {
        width: 100%;
        margin-top: 10px;
        background: #d32f2f;
        color: white;
        border: none;
        padding: 8px;
        border-radius: 5px;
        cursor: pointer;
        transition: background 0.2s;
    }
    .view-button:hover {
        background: #b71c1c;
    }
    
    /* Loading spinner */
    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 20px 0;
    }
    .spinner {
        border: 4px solid rgba(0, 0, 0, 0.1);
        width: 36px;
        height: 36px;
        border-radius: 50%;
        border-left-color: #09f;
        animation: spin 1s linear infinite;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Responsive columns */
    @media (max-width: 900px) {
        .stColumn {
            min-width: 50% !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

class LaptopRecommender:
    def __init__(self, data):
        self.df = data.copy()
        self._prepare_data()
        self._create_feature_matrix()
        
    def _prepare_data(self):
        """Prepare the data for recommendation by encoding categorical features"""
        # Custom mappings for ordinal encoding
        drive_type_order = {'Unknown': 0, 'HDD': 1, 'eMMC': 2, 'SSD': 3}
        ram_type_order = {
            'DDR3': 1, 'LPDDR3': 2, 'DDR4': 3, 'LPDDR4': 4, 'LPDDR4X': 5,
            'DDR5': 6, 'LPDDR5': 7, 'LPDDR5X': 8, 'Unified': 0, 'Unknown': 0
        }
        os_order = {'Unknown': 0, 'Chrome OS': 1, 'Windows': 2, 'macOS': 3}
        gpu_brand_order = {'Unknown': 0, 'Intel': 1, 'AMD': 2, 'NVIDIA': 3}
        processor_brand_order = {'Unknown': 0, 'Intel': 1, 'AMD': 2, 'Apple': 3}

        # Apply mappings
        self.df['Primary_Drive_Type'] = self.df['Primary_Drive_Type'].map(drive_type_order).fillna(0)
        self.df['Secondary_Drive_Type'] = self.df['Secondary_Drive_Type'].map(drive_type_order).fillna(0)
        self.df['Ram_Type'] = self.df['Ram_Type'].map(ram_type_order).fillna(0)
        self.df['Operating System'] = self.df['Operating System'].map(os_order).fillna(0)
        self.df['GPU_Brand'] = self.df['GPU_Brand'].map(gpu_brand_order).fillna(0)
        self.df['Processor_Brand'] = self.df['Processor_Brand'].map(processor_brand_order).fillna(0)

        # Combine Name and Processor into a single text feature
        self.df['Name_Processor'] = self.df['Name'] + " " + self.df['Processor']

    def _create_feature_matrix(self):
        """Create a weighted feature matrix for recommendation"""
        # Define weights
        self.feature_weights = {
            'Name': 5.0,
            'Processor': 10.0,
            'Processor_Brand': 1.5,
            'Ram_Size': 2.5,
            'Ram_Type': 1.0,
            'Primary_Drive_Size_GB': 1.0,
            'Primary_Drive_Type': 2.0,
            'Secondary_Drive_Size_GB': 0.5,
            'Secondary_Drive_Type': 0.5,
            'Total_Storage_GB': 2.0,
            'Operating System': 1.0,
            'GPU_Brand': 1.5,
            'GPU_VRAM': 2.0,
            'Price': 1.5
        }

        # Create a preprocessor pipeline
        numeric_pipeline = Pipeline([
            ('scaler', StandardScaler())
        ])

        self.preprocessor = make_column_transformer(
            (TfidfVectorizer(stop_words='english'), 'Name_Processor'),
            (numeric_pipeline, ['Processor_Brand']),
            (numeric_pipeline, ['Ram_Size']),
            (numeric_pipeline, ['Ram_Type']),
            (numeric_pipeline, ['Primary_Drive_Size_GB']),
            (numeric_pipeline, ['Primary_Drive_Type']),
            (numeric_pipeline, ['Secondary_Drive_Size_GB']),
            (numeric_pipeline, ['Secondary_Drive_Type']),
            (numeric_pipeline, ['Total_Storage_GB']),
            (numeric_pipeline, ['Operating System']),
            (numeric_pipeline, ['GPU_Brand']),
            (numeric_pipeline, ['GPU_VRAM']),
            (numeric_pipeline, ['Price']),
            remainder='drop'
        )

        # pipeline = Pipeline(steps=[('preprocessor', self.preprocessor)])
        self.pipeline = Pipeline(steps=[('preprocessor', self.preprocessor)])

        # Fit and transform - convert to dense array
        try:
            self.feature_matrix = self.pipeline.fit_transform(self.df)
            if hasattr(self.feature_matrix, 'toarray'):  # Check if sparse matrix
                self.feature_matrix = self.feature_matrix.toarray()
            else:
                self.feature_matrix = np.array(self.feature_matrix)
        except Exception as e:
            st.error(f"Error during feature transformation: {e}")
            # Fallback to simple features if transformation fails
            simple_features = self.df[['Ram_Size', 'Total_Storage_GB', 'GPU_VRAM', 'Price']]
            self.feature_matrix = simple_features.values

        # Get feature names
        try:
            self.text_features = self.pipeline.named_steps['preprocessor'].named_transformers_['tfidfvectorizer'].get_feature_names_out()
            numeric_features = [
                'Processor_Brand', 'Ram_Size', 'Ram_Type',
                'Primary_Drive_Size_GB', 'Primary_Drive_Type',
                'Secondary_Drive_Size_GB', 'Secondary_Drive_Type',
                'Total_Storage_GB', 'Operating System',
                'GPU_Brand', 'GPU_VRAM', 'Price'
            ]
            self.all_features = list(self.text_features) + numeric_features
        except:
            # Fallback feature names if pipeline failed
            self.all_features = ['Ram_Size', 'Total_Storage_GB', 'GPU_VRAM', 'Price']

        # Create DataFrame (handle case where dimensions might not match)
        try:
            self.feature_df = pd.DataFrame(self.feature_matrix, columns=self.all_features[:self.feature_matrix.shape[1]])
        except:
            self.feature_df = pd.DataFrame(self.feature_matrix)

        self.feature_df.fillna(0, inplace=True)

        # Apply weights (only to columns that exist)
        weight_array = np.ones(self.feature_df.shape[1])
        for i, col in enumerate(self.feature_df.columns):
            if col in self.feature_weights:
                weight_array[i] = self.feature_weights[col]
            elif any(processor_term in col for processor_term in ['i3', 'i5', 'i7', 'i9', 'ryzen', 'amd', 'intel']):
                weight_array[i] = self.feature_weights['Processor']
            elif col in ['Name', 'Processor']:  # For text features from Name_Processor
                weight_array[i] = self.feature_weights['Name']

        self.weighted_features = self.feature_df.values * weight_array
        self.weighted_feature_df = pd.DataFrame(self.weighted_features, columns=self.feature_df.columns)

        # Precompute overall scores
        self.overall_scores = self.weighted_feature_df.sum(axis=1)
        self.top_overall_indices = self.overall_scores.argsort()[::-1]

    def recommend_laptops(self, keyword=None, price_limit=None, top_n=5):
        """Recommend laptops based on keyword or overall ranking"""
        # 1. Compute sim_scores
        if keyword and str(keyword).strip():
            input_text = keyword.lower()
            tfidf = self.pipeline.named_steps['preprocessor']\
                        .named_transformers_['tfidfvectorizer']
            kv = tfidf.transform([input_text]).toarray()[0]

            input_features = np.zeros(len(self.all_features))
            for i, word in enumerate(self.text_features):
                if word in self.all_features:
                    idx = self.all_features.index(word)
                    w = (self.feature_weights['Processor']
                        if any(t in word for t in ['i3','i5','i7','i9','ryzen','amd','intel'])
                        else self.feature_weights['Name'])
                    input_features[idx] = kv[i] * w

            sim_scores = cosine_similarity([input_features], self.weighted_features).flatten()
        else:
            # no keyword: use precomputed overall_scores
            sim_scores = self.overall_scores

        # 2. Score & pick top 20
        scored = self.df.assign(_score=sim_scores).sort_values('_score', ascending=False)
        top20 = scored.head(20)

        # 3. Apply price window (20% range)
        if price_limit is not None:
            low = price_limit * 0.8  # 20% below
            high = price_limit * 1.2  # 20% above
            in_range = top20[(top20['Price'] >= low) & (top20['Price'] <= high)]
            
            # If no results in range, show closest matches
            if in_range.empty:
                # Find closest matches to price limit
                top20['price_diff'] = abs(top20['Price'] - price_limit)
                recommendations = top20.sort_values(['price_diff', '_score'], ascending=[True, False])
            else:
                recommendations = in_range

            recommendations = recommendations.drop(columns=['price_diff'], errors='ignore')
        else:
            recommendations = top20

        # 4. Return top_n
        return recommendations.head(top_n)



    def recommend_top_rated(self, price_limit=None, top_n=5):
        """Recommend top rated laptops within price limit (±20%)"""
        if 'Rating' not in self.df.columns:
            return self.recommend_laptops(top_n=top_n, price_limit=price_limit)
        
        working_df = self.df.copy()
        
        # Apply price filter if specified
        if price_limit is not None:
            price_low = price_limit * 0.8
            price_high = price_limit * 1.2
            working_df = working_df[(working_df['Price'] >= price_low) & 
                                (working_df['Price'] <= price_high)]
            
            if working_df.empty:
                # If no results in range, find closest matches
                working_df = self.df.copy()
                working_df['price_diff'] = abs(working_df['Price'] - price_limit)
                working_df = working_df.sort_values(['price_diff', 'Rating'], 
                                                ascending=[True, False])
                working_df = working_df.drop(columns=['price_diff'])
                st.warning(f"No laptops found within 20% of ₹{price_limit}. Showing closest matches.")
        
        # Sort by rating and return top N
        return working_df.sort_values('Rating', ascending=False).head(top_n)

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
                # Complete preprocessing pipeline
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
        # Save uploaded file temporarily
        with open("temp_upload.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process data
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
        
        time.sleep(2)  # Simulate processing time
        placeholder.empty()  # Clear the loading animation
        return df
        
    except Exception as e:
        placeholder.empty()
        st.error(f"Error processing file: {str(e)}")
        return None

def display_laptop_comparison(recommendations):
    """Display recommended laptops in a 3x2 grid using Streamlit components"""
    st.subheader("🌟 Top Recommendations")
    
    # Limit to 6 recommendations max
    recommendations = recommendations.head(6)
    
    # Create rows and columns
    for i in range(0, len(recommendations), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(recommendations):
                row = recommendations.iloc[i + j]
                with cols[j]:
                    # Create card container
                    with st.container():
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
                        
                        # Price
                        st.markdown(f"<span style='color:#d32f2f; font-weight:bold; font-size:18px;'>₹{row['Price']:,}</span>", 
                                   unsafe_allow_html=True)
                        
                        # Rating
                        rating_display = ""
                        if 'Rating' in recommendations.columns and 'Num Ratings' in recommendations.columns:
                            rating_display = f"<span style='color:#ffb400;'>★</span> {row['Rating']} ({row['Num Ratings']} reviews)"
                        elif 'Rating' in recommendations.columns:
                            rating_display = f"<span style='color:#ffb400;'>★</span> {row['Rating']}"
                        
                        st.markdown(
                            f"<div style='display:flex; justify-content:space-between; margin:8px 0;'>"
                            f"<span style='font-weight:600; color:#555;'>Rating:</span>"
                            f"<span style='font-weight:500; color:#222;'>{rating_display}</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                        
                        # Processor
                        processor = row['Processor'][:20] + ('...' if len(row['Processor']) > 20 else '') if 'Processor' in recommendations.columns else "N/A"
                        st.markdown(
                            f"<div style='display:flex; justify-content:space-between; margin:8px 0;'>"
                            f"<span style='font-weight:600; color:#555;'>Processor:</span>"
                            f"<span style='font-weight:500; color:#222;'>{processor}</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                        
                        # RAM
                        ram_display = ""
                        if 'Ram_Size' in recommendations.columns and 'Ram_Type' in recommendations.columns:
                            ram_display = f"{row['Ram_Size']} GB {row['Ram_Type']}"
                        elif 'Ram_Size' in recommendations.columns:
                            ram_display = f"{row['Ram_Size']} GB"
                        st.markdown(
                            f"<div style='display:flex; justify-content:space-between; margin:8px 0;'>"
                            f"<span style='font-weight:600; color:#555;'>RAM:</span>"
                            f"<span style='font-weight:500; color:#222;'>{ram_display}</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                        
                        # Storage
                        storage_display = ""
                        if 'Primary_Drive_Size_GB' in recommendations.columns and 'Primary_Drive_Type' in recommendations.columns:
                            storage_display = f"{int(row['Primary_Drive_Size_GB'])} GB {row['Primary_Drive_Type']}"
                        elif 'Total_Storage_GB' in recommendations.columns:
                            storage_display = f"{int(row['Total_Storage_GB'])} GB"
                        st.markdown(
                            f"<div style='display:flex; justify-content:space-between; margin:8px 0;'>"
                            f"<span style='font-weight:600; color:#555;'>Storage:</span>"
                            f"<span style='font-weight:500; color:#222;'>{storage_display}</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                        
                        # View Product button - only if column exists
                        if 'Product Link' in recommendations.columns:
                            st.link_button("View Product", row['Product Link'])
            
def main():
    st.title("💻 Laptop Recommendation System")
    st.markdown("Find the perfect laptop based on your preferences")
    
    # Initialize session state
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("🔍 Search Parameters")
        
        # Data source selection
        data_source = st.radio("Data Source:", 
                              ["Use Default Dataset", "Upload Your Own Data"],help="Data Should be scraped using the scraper provided")
        
        if data_source == "Upload Your Own Data":
            uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
            if st.button("Get Scraper"):
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
        
        # Price filter
        max_price = st.slider("Maximum Price (₹)", 
                            min_value=10000, 
                            max_value=300000, 
                            value=50000, 
                            step=5000,
                            help="Recommendations will be within ±20% of this price")
        
        # Keyword search
        keyword = st.text_input("Search Keyword (e.g., 'nvidia', 'i7','apple')")
        
        # Recommendation button
        recommend_clicked = st.button("🚀 Get Recommendations", 
                                    use_container_width=True,
                                    disabled=st.session_state.processed_data is None)
    
    # Main content area
    if st.session_state.processed_data is None:
        st.warning("Please load or upload data to get recommendations")
        return
    
    # Show data stats
    if len(st.session_state.processed_data) > 0:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Laptops", len(st.session_state.processed_data))
        col2.metric("Average Price", f"₹{st.session_state.processed_data['Price'].mean():,.0f}")
        col3.metric("Average Rating", f"{st.session_state.processed_data['Rating'].mean():.1f} ★")
        
        # Show price range info
        st.caption(f"Showing recommendations under ₹{max_price:,}")
    
    # Generate recommendations when button clicked
    if recommend_clicked or (keyword and keyword != st.session_state.get('last_keyword', '')):
        with st.spinner("Finding best matches..."):
            try:
                # Pass the full dataset to the recommender
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
    
    # Display recommendations if available
    if st.session_state.recommendations is not None:
        display_laptop_comparison(st.session_state.recommendations)

if __name__ == "__main__":
    main()
