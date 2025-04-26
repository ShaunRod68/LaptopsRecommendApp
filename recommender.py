import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler

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

        self.pipeline = Pipeline(steps=[('preprocessor', self.preprocessor)])

        # Fit and transform - convert to dense array
        try:
            self.feature_matrix = self.pipeline.fit_transform(self.df)
            if hasattr(self.feature_matrix, 'toarray'):
                self.feature_matrix = self.feature_matrix.toarray()
            else:
                self.feature_matrix = np.array(self.feature_matrix)
        except Exception as e:
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
            elif col in ['Name', 'Processor']:
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
            low = price_limit * 0.8
            high = price_limit * 1.2
            in_range = top20[(top20['Price'] >= low) & (top20['Price'] <= high)]
            
            if in_range.empty:
                top20['price_diff'] = abs(top20['Price'] - price_limit)
                recommendations = top20.sort_values(['price_diff', '_score'], ascending=[True, False])
            else:
                recommendations = in_range

            recommendations = recommendations.drop(columns=['price_diff'], errors='ignore')
        else:
            recommendations = top20

        return recommendations.head(top_n)

    def recommend_top_rated(self, price_limit=None, top_n=5):
        """Recommend top rated laptops within price limit (±20%)"""
        if 'Rating' not in self.df.columns:
            return self.recommend_laptops(top_n=top_n, price_limit=price_limit)
        
        working_df = self.df.copy()
        
        if price_limit is not None:
            price_low = price_limit * 0.8
            price_high = price_limit * 1.2
            working_df = working_df[(working_df['Price'] >= price_low) & 
                                (working_df['Price'] <= price_high)]
            
            if working_df.empty:
                working_df = self.df.copy()
                working_df['price_diff'] = abs(working_df['Price'] - price_limit)
                working_df = working_df.sort_values(['price_diff', 'Rating'], 
                                                ascending=[True, False])
                working_df = working_df.drop(columns=['price_diff'])
        
        return working_df.sort_values('Rating', ascending=False).head(top_n)