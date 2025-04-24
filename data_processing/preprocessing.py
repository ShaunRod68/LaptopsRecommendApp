import pandas as pd
import re

class DataPreprocessor:
    def __init__(self):
        self.df = None
    
    def load_data(self, filepath):
        """Load data from CSV file"""
        data = pd.read_csv(filepath)
        self.df = pd.DataFrame(data)
        return self
    
    def initial_clean(self):
        """Perform initial data cleaning"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        # Drop unnecessary columns
        self.df = self.df.drop(columns=['Unnamed: 0','Original Price','Discount','Availability',
                                      'Seller','Delivery','Exchange Offer'], errors='ignore')
        
        # Remove duplicates
        self.df = self.df.drop_duplicates(subset=['Name', 'Price','Rating','Num Ratings'], keep='first')
        
        # Keep lowest price for each product
        self.df = self.df.loc[self.df.groupby("Name")["Price"].idxmin()].reset_index(drop=True)
        return self
    
    def process_features(self):
        """Process and transform features"""
        if self.df is None:
            raise ValueError("Data not loaded and cleaned. Call load_data() and initial_clean() first.")
        
        # Clean price column
        self.df["Price"] = self.df["Price"].astype(str).str.replace("₹", "").str.replace(",", "").astype(int)
        
        # Clean numeric columns
        self.df["Num Ratings"] = pd.to_numeric(
            self.df["Num Ratings"].astype(str).str.replace(",", ""), 
            errors='coerce'
        )
        self.df['Num Reviews'] = pd.to_numeric(self.df['Num Reviews'], errors='coerce').fillna(0)
        
        # Standardize operating system
        self.df["Operating System"] = self.df["Operating System"].apply(
            lambda x: "Windows" if "win" in str(x).lower() else
                      "ChromeOS" if "chrome" in str(x).lower() else
                      "macOS" if "mac" in str(x).lower() else "other"
        )
        return self
    
    def process_storage(self):
        """Handle storage-related processing"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        # Salvage null storage from name
        null_storage = self.df[self.df['Storage'].isnull()]
        self.df = self.df.dropna(subset=['Storage'])
        
        storage_pattern = r'\d+\s*GB\s*/\s*(\d+\s*(?:GB|TB)\s*[A-Za-z]*)'
        null_storage['Storage'] = null_storage['Name'].str.extract(storage_pattern, flags=re.IGNORECASE)[0]
        null_storage = null_storage.dropna(subset=['Storage'])
        
        self.df = pd.concat([self.df, null_storage], ignore_index=True)
        return self
    
    def extract_processor_brand(self):
        """Extract processor brand from processor name"""
        def _extract_brand(processor_name):
            processor_name = str(processor_name).strip()
            if re.search(r'intel', processor_name, re.I):
                return 'Intel'
            elif re.search(r'AMD|Ryzen|Athlon', processor_name, re.I):
                return 'AMD'
            elif re.search(r'Apple|M\d', processor_name, re.I):
                return 'Apple'
            elif re.search(r'dragon|qual', processor_name, re.I):
                return 'Qualcomm'
            elif re.search(r'Media', processor_name, re.I):
                return 'MediaTek'
            return None
        
        self.df['Processor_Brand'] = self.df['Processor'].apply(_extract_brand)
        return self
    
    def process_ram(self):
        """Extract RAM features"""
        def _ram_extract(ram):
            entire_ram = str(ram).strip()
            features = {'Ram_Size': None, 'Ram_Type': None}
            
            size_match = re.search(r'(\d+)\s?GB', entire_ram)
            if size_match:
                features['Ram_Size'] = int(size_match.group(1))
            
            type_match = re.search(r'GB\s+([A-Za-z0-9]+)', entire_ram)
            if type_match:
                features['Ram_Type'] = type_match.group(1).strip()
            return features
        
        ram = self.df['RAM'].apply(_ram_extract)
        ram = pd.DataFrame(ram.tolist())
        self.df = self.df.drop(columns=['RAM'])
        self.df['Ram_Size'] = ram['Ram_Size']
        self.df['Ram_Type'] = ram['Ram_Type']
        return self
    
    def process_storage_details(self):
        """Extract detailed storage features"""
        def _extract_storage(storage_text):
            storage_text = str(storage_text).strip()
            features = {
                'Primary_Drive_Size_GB': 0,
                'Secondary_Drive_Size_GB': 0,
                'Primary_Drive_Type': None,
                'Secondary_Drive_Type': None,
                'Total_Storage_GB': 0
            }

            drives = re.split(r'\s*[|+]\s*|\s+and\s+', storage_text)

            for i, drive in enumerate(drives[:2]):
                if not drive:
                    continue

                size_match = re.search(r'(\d+)\s*(GB|TB)', drive, re.IGNORECASE)
                if not size_match:
                    continue

                size = float(size_match.group(1))
                unit = size_match.group(2).upper()
                size_gb = size * 1024 if unit == 'TB' else size

                drive_type = None
                if 'SSD' in drive.upper():
                    drive_type = 'SSD'
                elif 'HDD' in drive.upper():
                    drive_type = 'HDD'
                elif 'EMMC' in drive.upper():
                    drive_type = 'EMMC'

                if i == 0:
                    features['Primary_Drive_Size_GB'] = size_gb
                    features['Primary_Drive_Type'] = drive_type
                else:
                    features['Secondary_Drive_Size_GB'] = size_gb
                    features['Secondary_Drive_Type'] = drive_type

            features['Total_Storage_GB'] = (
                features['Primary_Drive_Size_GB'] +
                features['Secondary_Drive_Size_GB']
            )
            return features

        storage_features = self.df['Storage'].apply(_extract_storage)
        storage_df = pd.DataFrame(storage_features.tolist())
        
        # Insert storage features into main dataframe
        for col in storage_df.columns:
            self.df[col] = storage_df[col]
            
        return self
    
    def process_warranty(self):
        """Standardize warranty information"""
        def _extract_warranty(warranty):
            warranty = str(warranty).lower()
            match = re.search(r"(\d+)\s*year", warranty)
            return int(match.group(1)) if match else None
        
        self.df['Warranty'] = self.df['Warranty'].apply(_extract_warranty)
        self.df['Warranty'] = pd.to_numeric(self.df['Warranty'], errors='coerce').fillna(1).astype(int) # Default to 1 year if missing
        return self
    
    def get_processed_data(self):
        """Return the fully processed dataframe"""
        return self.df
    
    def save_processed_data(self, filepath):
        """Save processed data to CSV"""
        if self.df is not None:
            self.df.to_csv(filepath, index=False)
        return self
    def process_display(self):
        """Extract display size in cm"""
        def _extract_display_cm(display_text):
            match = re.search(r'(\d+\.?\d*)\s*cm', str(display_text))
            return float(match.group(1)) if match else None

        self.df['Display_cm'] = self.df['Display'].apply(_extract_display_cm)
        return self

    def process_gpu(self):
        """Extract GPU brand and VRAM from product name"""
        def _extract_gpu_info_precise(name):
            name_lower = str(name).lower()
            vram_match = re.search(r'(\d+)\s*gb\s+graphics', name_lower)
            if vram_match:
                vram = int(vram_match.group(1))
                # Look in a 40‑character window around the match
                window = name_lower[max(0, vram_match.start() - 40):vram_match.end() + 40]

                # Brand patterns
                gpu_keywords = {
                    "NVIDIA": [r"nvidia", r"geforce", r"gtx", r"rtx", r"mx"],
                    "AMD":    [r"radeon", r"vega"],
                    "Intel":  [r"intel uhd", r"iris", r"intel graphics"]
                }
                for brand, patterns in gpu_keywords.items():
                    for p in patterns:
                        if re.search(p, window):
                            return pd.Series([brand, vram])
                return pd.Series(["UNKNOWN", vram])
            return pd.Series([None, None])

        self.df[['GPU_Brand', 'GPU_VRAM']] = self.df['Name'].apply(_extract_gpu_info_precise)
        # Fill and coerce
        self.df['GPU_Brand'] = self.df['GPU_Brand'].fillna("Not Mentioned")
        self.df['GPU_VRAM'] = pd.to_numeric(self.df['GPU_VRAM'], errors='coerce').fillna(0).astype(int)
        return self
