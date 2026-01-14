from bing_image_downloader import downloader
from pathlib import Path
import logging
import shutil

logging.basicConfig(level=logging.INFO)

def scrape_vegeta():
    logging.info("Attempting to rescue Vegeta...")
    
    DATA_DIR = Path("data/raw")
    char = "Vegeta Dragon Ball" # slightly different query to maybe avoid bad image causing crash
    clean_name = "vegeta"
    
    output_dir = DATA_DIR
    
    try:
        downloader.download(
            char, 
            limit=100, 
            output_dir=str(output_dir), 
            adult_filter_off=True, 
            force_replace=False, 
            timeout=10, 
            verbose=False
        )
        
        # Rename logic
        downloaded_folder_name = char
        downloaded_path = output_dir / downloaded_folder_name
        target_path = output_dir / clean_name
        
        if downloaded_path.exists():
            if target_path.exists():
                for item in downloaded_path.iterdir():
                    shutil.move(str(item), str(target_path / item.name))
                downloaded_path.rmdir()
            else:
                downloaded_path.rename(target_path)
        
        logging.info(f"Successfully downloaded Vegeta rescue mission.")
        
    except Exception as e:
        logging.error(f"Failed to download Vegeta again: {e}")

if __name__ == "__main__":
    scrape_vegeta()
