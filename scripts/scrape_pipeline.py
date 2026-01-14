from bing_image_downloader import downloader
from pathlib import Path
import logging
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CHARACTERS = [
    "Monkey D. Luffy",
    "Roronoa Zoro",
    "Naruto Uzumaki",
    "Sasuke Uchiha",
    "Son Goku",
    "Vegeta",
    "Saitama One Punch Man",
    "Light Yagami",
    "Levi Ackerman",
    "Satoru Gojo"
]

DATA_DIR = Path("data/raw")
MAX_IMAGES = 100 

def scrape_data():
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True)

    for char in CHARACTERS:
        logging.info(f"Downloading images for {char}...")
        
        # bing-image-downloader creates its own subdirectory structure based on query
        # We will download to a temp folder then move/rename if needed, 
        # but actually it downloads to 'dataset/<query_string>'. 
        # We want 'data/raw/<clean_name>'
        
        clean_name = char.replace(" ", "_").lower()
        output_dir = DATA_DIR
        
        try:
            downloader.download(
                char + " anime", 
                limit=MAX_IMAGES, 
                output_dir=str(output_dir), 
                adult_filter_off=True, 
                force_replace=False, 
                timeout=10, 
                verbose=False
            )
            
            # Key step: bing downloader creates folder 'data/raw/Monkey D. Luffy anime'
            # We want to rename it to 'data/raw/monkey_d_luffy'
            
            downloaded_folder_name = f"{char} anime"
            downloaded_path = output_dir / downloaded_folder_name
            target_path = output_dir / clean_name
            
            if downloaded_path.exists():
                if target_path.exists():
                    # Merge if exists
                    for item in downloaded_path.iterdir():
                        shutil.move(str(item), str(target_path / item.name))
                    downloaded_path.rmdir()
                else:
                    downloaded_path.rename(target_path)
            
            logging.info(f"Successfully downloaded {char}")
            
        except Exception as e:
            logging.error(f"Failed to download {char}: {e}")

if __name__ == "__main__":
    scrape_data()
