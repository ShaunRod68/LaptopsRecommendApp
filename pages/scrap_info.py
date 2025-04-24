import streamlit as st
import pyperclip  # For copy-to-clipboard functionality

# Scraper code as a string
SCRAPER_CODE = """
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import os

# File paths
csv_filename = "flipkart_laptops_combined.csv"
failed_links_filename = "failed_links.txt"

# Replace with your ScraperAPI key (get it from https://www.scraperapi.com/)
SCRAPER_API_KEY = "YOUR_API_KEY_HERE"

# Price range URLs to scrape
PRICE_RANGE_URLS = [
    ("Under ₹50,000", "https://www.flipkart.com/search?q=laptops&p[]=facets.price_range.from%3DMin&p[]=facets.price_range.to%3D50000&page="),
    ("₹50,000 - ₹75,000", "https://www.flipkart.com/search?q=laptops&p%5B%5D=facets.price_range.from%253D50000&p%5B%5D=facets.price_range.to%253D75000&page="),
    ("Over ₹75,000", "https://www.flipkart.com/search?q=laptops&p[]=facets.price_range.from%3D75000&p[]=facets.price_range.to%3DMax&page=")
]

COLUMNS = [
    "Name", "Price", "Original Price", "Discount", "Processor", "RAM", 
    "Storage", "Display", "Operating System", "Warranty", "Rating", 
    "Num Ratings", "Num Reviews", "Availability", "Seller", 
    "Delivery", "Exchange Offer", "Image URL", "Product Link", "Price Range"
]

def fetch_url(url, max_retries=3):
    for _ in range(max_retries):
        try:
            payload = {'api_key': SCRAPER_API_KEY, 'url': url}
            response = requests.get("http://api.scraperapi.com", params=payload, timeout=30)
            if response.status_code == 200:
                return response.text
            elif response.status_code == 429:
                time.sleep(random.uniform(10, 20))  # Rate limiting
            else:
                print(f"Failed: {url} (Status: {response.status_code})")
                with open(failed_links_filename, "a") as f:
                    f.write(f"{url}\\n")
                return None
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(random.uniform(5, 10))
    return None

def scrape_all_ranges():
    all_laptops = []
    for range_name, base_url in PRICE_RANGE_URLS:
        print(f"Scraping {range_name}...")
        for page in range(1, 28):  # 27 pages max per range
            url = base_url + str(page)
            html = fetch_url(url)
            if not html:
                continue
            
            soup = BeautifulSoup(html, "html.parser")
            products = soup.find_all("div", {"class": "tUxRFH"})
            
            for product in products:
                # [Add your existing product extraction logic here]
                # Example: name = product.find("div", {"class": "KzDlHZ"}).text.strip()
                # Append to all_laptops with range_name as last column
            
            time.sleep(random.uniform(2, 5))  # Avoid rate limits
    
    pd.DataFrame(all_laptops, columns=COLUMNS).to_csv(csv_filename, index=False)
    print(f"Saved {len(all_laptops)} laptops to {csv_filename}")

if __name__ == "__main__":
    scrape_all_ranges()
"""

def main():
    st.set_page_config(page_title="Flipkart Scraper", page_icon="💻")
    st.title("💻 Flipkart Laptop Scraper")
    
    st.markdown("""
    ### How to Use This Scraper:
    1. **Get a ScraperAPI Key** (free tier available at [scraperapi.com](https://www.scraperapi.com))
    2. **Run the Python script** (modify `YOUR_API_KEY_HERE` with your key)
    3. **Wait** (scraping all 3 price ranges takes ~30 mins)
    4. **Check** `flipkart_laptops_combined.csv` for results

    ⚠️ **Note**: Flipkart may block scrapers without proxies.  
    ScraperAPI handles this automatically.
    """)
    
    # Download button for the script
    st.download_button(
        label="⬇️ Download Scraper Script (scraper.py)",
        data=SCRAPER_CODE,
        file_name="flipkart_laptop_scraper.py",
        mime="text/python"
    )
    
    # Copy-to-clipboard option
    if st.button("📋 Copy Code to Clipboard"):
        pyperclip.copy(SCRAPER_CODE)
        st.success("Code copied to clipboard!")

    # 👉 Return to Home page - FIXED VERSION
    if st.button("🏠 Return to Home"):
        st.switch_page("app.py")

if __name__ == "__main__":
    main()