"""
Script to reshape training data for image-LP relevance task.
Reads OriginalData.tsv and converts it to the format required for multi-modal LLM training.
"""
import argparse
import json
import os
import hashlib
import random
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from PIL import Image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Define multiple prompt templates for diversity
PROMPT_TEMPLATES = [
    "Please evaluate the relevance between an image and a URL landing page (LP). Answer with only Good, Fair, or Bad.\nURL: {url}\nLP: {doc}\nOutput:\n<image>",
    "Please evaluate the relevance between an image and a URL landing page (LP). Answer with only Good, Fair, or Bad.\nURL: {url}\nLP: {doc}\nOutput:\n<image>",
    "Please evaluate the relevance between an image and a URL landing page (LP). Answer with only Good, Fair, or Bad.\nURL: {url}\nLP: {doc}\nOutput:\n<image>",
    "Please evaluate the relevance between an image and a URL landing page (LP). Answer with only Good, Fair, or Bad.\nURL: {url}\nLP: {doc}\nOutput:\n<image>",
    "Please evaluate the relevance between an image and a URL landing page (LP). Answer with only Good, Fair, or Bad.\nURL: {url}\nLP: {doc}\nOutput:\n<image>",
    "<image>Please evaluate the relevance between an image and a URL landing page (LP). Answer with only Good, Fair, or Bad.\nURL: {url}\nLP: {doc}\nOutput:\n",
    "<image>Please evaluate the relevance between an image and a URL landing page (LP). Answer with only Good, Fair, or Bad.\nURL: {url}\nLP: {doc}\nOutput:\n",
    "<image>Please evaluate the relevance between an image and a URL landing page (LP). Answer with only Good, Fair, or Bad.\nURL: {url}\nLP: {doc}\nOutput:\n",
    "<image>Please evaluate the relevance between an image and a URL landing page (LP). Answer with only Good, Fair, or Bad.\nURL: {url}\nLP: {doc}\nOutput:\n",
    "<image>Please evaluate the relevance between an image and a URL landing page (LP). Answer with only Good, Fair, or Bad.\nURL: {url}\nLP: {doc}\nOutput:\n",
    "Assess the relevance level between the following image and landing page content:\nURL: {url}\nLP Content: {doc}\n<image>",
    "Determine how relevant this image is to the given landing page:\nURL: {url}\nLP Content: {doc}\nOutput:\n<image>",
    "Judge the relevance of the image to the following landing page:\nURL: {url}\nLP: {doc}\n<image>",
    "How relevant is this image to the landing page?\nURL: {url}\nLP Content: {doc}\n<image>",
    "Please analyze the relevance between the image and landing page:\nURL: {url}\nLP: {doc}\n<image>",
    "Check the relevance between the provided image and landing page:\nURL: {url}\nLP Content: {doc}\n<image>",
    "<image>\nEvaluate the relevance of this image to the landing page:\nURL: {url}\nLP: {doc}",
    "<image>\nHow well does this image match the landing page content?\nURL: {url}\nLP Content: {doc}",
    "<image>\nPlease rate the relevance between this image and the LP:\nURL: {url}\nLP Content: {doc}",
    "<image>\nJudge how relevant this image is to the given landing page:\nURL: {url}\nLP Content: {doc}",
    "Please evaluate the relevance between an image and a URL landing page (LP). Answer with only Good, Fair, or Bad.\nURL: {url}\nLP: {doc}\nOutput:\n<image>",
    "Determine how relevant this image is to the given landing page. Your answer should be one of: Good, Fair, Bad.\nURL: {url}\nPage Content: {doc}\nOutput:\n<image>",
    "Rate the relevance between the image and the landing page described below. Only output Good, Fair, or Bad.\nURL: {url}\nLP: {doc}\nOutput:\n<image>",
    "Judge the relevance of the image to the following landing page. Choose one: Good, Fair, or Bad.\nURL: {url}\nLP: {doc}\nOutput:\n<image>",
    "How relevant is this image to the landing page? Respond with only Good, Fair, or Bad.\nURL: {url}\nLP Content: {doc}\nOutput:\n<image>",
    "Analyze the relevance between the image and landing page. Your response should be Good, Fair, or Bad.\nURL: {url}\nLP: {doc}\nOutput:\n<image>",
    "<image>\nEvaluate the relevance of this image to the landing page. Respond with only Good, Fair, or Bad.\nURL: {url}\nLP: {doc}\nOutput:",
    "<image>\nPlease rate the relevance between this image and the LP. Your answer should be Good, Fair, or Bad.\nURL: {url}\nLP Content: {doc}",
    "<image>\nDetermine the relevance level between this image and the landing page. Output Good, Fair, or Bad.\nURL: {url}\nLP: {doc}",
    "<image>\nEvaluate whether this image is relevant to the landing page. Choose Good, Fair, or Bad.\nURL: {url}\nPage Content: {doc}",
]


def generate_image_id(img_url: str) -> str:
    """
    Generate a unique ID for an image URL using MD5 hash.
    
    Args:
        img_url: The image URL
        
    Returns:
        A unique hash ID for the image
    """
    return hashlib.md5(img_url.encode('utf-8')).hexdigest()


def download_image(img_url: str, save_dir: Path, image_id: str) -> Optional[str]:
    """
    Download an image from URL and save it with the given ID.
    
    Args:
        img_url: The URL of the image to download
        save_dir: Directory to save the image
        image_id: Unique ID for the image
        
    Returns:
        Relative path to the saved image, or None if download failed
    """
    # Determine file extension from URL or default to .jpg
    ext = '.jpg'
    if '.' in img_url.split('/')[-1]:
        url_ext = img_url.split('.')[-1].split('?')[0].lower()
        if url_ext in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
            ext = f'.{url_ext}'
    
    image_filename = f"{image_id}{ext}"
    image_path = save_dir / image_filename
    
    # Check if image already exists
    if image_path.exists():
        logger.debug(f"Image already exists: {image_filename}")
        return str(Path("ImageLPRelevance") / "image_data" / image_filename)
    
    # Download the image
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(img_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        with open(image_path, 'wb') as f:
            f.write(response.content)
        
        logger.debug(f"Downloaded image: {image_filename}")
        return str(Path("ImageLPRelevance") / "image_data" / image_filename)
    
    except Exception as e:
        logger.warning(f"Failed to download image from {img_url}: {str(e)}")
        return None


def read_tsv_data(file_path: str, max_samples: int = -1) -> List[Dict]:
    """
    Read the TSV file and extract required columns.

    Args:
        file_path: Path to the TSV file
        max_samples: Maximum total samples (if > 0, limit each label to 1/3 of this)

    Returns:
        List of dictionaries containing the data
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        # Read header
        header = f.readline().strip().split('\t')
        logger.info(f"TSV columns: {header}")

        # Find column indices
        try:
            url_idx = header.index('FinalUrl')
            img_idx = header.index('ImgUrl')
            label_idx = header.index('Label')
            doc_idx = header.index('doc')
        except ValueError as e:
            logger.error(f"Required column not found: {e}")
            raise

        # Read data rows
        valid_labels = {'good', 'fair', 'bad'}
        filtered_count = 0
        for line_num, line in enumerate(f, start=2):
            try:
                parts = line.strip().split('\t')
                if len(parts) >= max(url_idx, img_idx, label_idx, doc_idx) + 1:
                    label = parts[label_idx].strip().lower()
                    # Only keep rows with good/fair/bad labels
                    if label in valid_labels:
                        data.append({
                            'FinalUrl': parts[url_idx],
                            'ImgUrl': parts[img_idx],
                            'Label': label.capitalize(),  # Convert to Good/Fair/Bad
                            'doc': parts[doc_idx]
                        })
                    else:
                        filtered_count += 1
            except Exception as e:
                logger.warning(f"Error parsing line {line_num}: {e}")
                continue

    logger.info(f"Successfully read {len(data)} rows from TSV file")
    logger.info(f"Filtered out {filtered_count} rows with invalid labels")

    # Apply max_samples limit if specified
    if max_samples > 0:
        data = apply_max_samples_limit(data, max_samples)

    return data


def apply_max_samples_limit(data: List[Dict], max_samples: int) -> List[Dict]:
    """
    Apply max_samples limit by ensuring each label gets at most 1/3 of max_samples.

    Args:
        data: List of data dictionaries
        max_samples: Maximum total samples

    Returns:
        Limited list of data dictionaries
    """
    logger.info(f"\nApplying max_samples limit of {max_samples}...")

    # Separate by label
    label_data = {
        'Good': [d for d in data if d['Label'] == 'Good'],
        'Fair': [d for d in data if d['Label'] == 'Fair'],
        'Bad': [d for d in data if d['Label'] == 'Bad']
    }

    original_counts = {label: len(items) for label, items in label_data.items()}
    logger.info(f"Original counts - Good: {original_counts['Good']}, Fair: {original_counts['Fair']}, Bad: {original_counts['Bad']}")

    # Calculate max samples per label (1/3 of max_samples)
    max_per_label = max_samples // 3
    logger.info(f"Max samples per label (1/3 of {max_samples}): {max_per_label}")

    # Limit each label
    limited_data = []
    for label in ['Good', 'Fair', 'Bad']:
        items = label_data[label]
        if len(items) > max_per_label:
            random.shuffle(items)
            items = items[:max_per_label]
            logger.info(f"Limited {label} from {original_counts[label]} to {len(items)}")
        else:
            logger.info(f"Kept all {len(items)} {label} samples (within limit)")
        limited_data.extend(items)

    # Shuffle combined data
    random.shuffle(limited_data)

    final_counts = {
        'Good': len([d for d in limited_data if d['Label'] == 'Good']),
        'Fair': len([d for d in limited_data if d['Label'] == 'Fair']),
        'Bad': len([d for d in limited_data if d['Label'] == 'Bad'])
    }
    logger.info(f"Final counts - Good: {final_counts['Good']}, Fair: {final_counts['Fair']}, Bad: {final_counts['Bad']}")
    logger.info(f"Total samples after limiting: {len(limited_data)}")

    return limited_data


def process_single_row(row: Dict, image_dir: Path, templates: List[str], download_new_images: bool = True, download_bad_only: bool = False) -> Optional[Tuple[Dict, str, bool]]:
    """
    Process a single row of data.
    
    Args:
        row: Data row dictionary
        image_dir: Directory to save images
        templates: List of prompt templates
        download_new_images: Whether to download new images or only use existing ones
        download_bad_only: If True, only download images for Bad label
        
    Returns:
        Tuple of (training_example, label, is_new_download) or None if failed
    """
    try:
        # Generate unique image ID
        image_id = generate_image_id(row['ImgUrl'])
        
        # Check if image already exists
        image_path = None
        is_new_download = False
        
        for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
            potential_path = image_dir / f"{image_id}{ext}"
            if potential_path.exists():
                image_path = str(Path("ImageLPRelevance") / "image_data" / f"{image_id}{ext}")
                break
        
        # If image doesn't exist
        if image_path is None:
            if download_new_images:
                # Check if we should download this image based on label
                should_download = True
                if download_bad_only and row['Label'] != 'Bad':
                    # Skip downloading for non-Bad labels
                    should_download = False
                
                if should_download:
                    # Download new image
                    image_path = download_image(row['ImgUrl'], image_dir, image_id)
                    if image_path is None:
                        # Download failed, skip this row
                        return None
                    is_new_download = True
                else:
                    # Skip this row if image doesn't exist and should not download
                    return None
            else:
                # Skip this row if image doesn't exist and downloading is disabled
                return None
        
        # Randomly select a prompt template
        template = random.choice(templates)
        
        # Truncate doc if too long (keep first 1000 characters)
        doc_content = row['doc'][:1000] if len(row['doc']) > 1000 else row['doc']
        
        # Create prompt
        prompt = template.format(url=row['FinalUrl'], doc=doc_content)
        
        # Create message structure
        training_example = {
            "messages": [
                {
                    "content": prompt,
                    "role": "user"
                },
                {
                    "content": row['Label'],
                    "role": "assistant"
                }
            ],
            "images": [image_path]
        }
        
        return (training_example, row['Label'], is_new_download)
    
    except Exception as e:
        logger.warning(f"Failed to process row: {str(e)}")
        return None


def create_training_data(data: List[Dict], image_dir: Path, max_workers: int = 16, download_new_images: bool = True, download_bad_only: bool = False) -> List[Dict]:
    """
    Create training data in the required format using parallel processing.
    
    Args:
        data: List of data dictionaries
        image_dir: Directory to save images
        max_workers: Maximum number of parallel workers for downloading
        download_new_images: Whether to download new images or only use existing ones
        download_bad_only: If True, only download images for Bad label
        
    Returns:
        List of training examples in the required format
    """
    training_data = []
    downloaded_count = 0
    skipped_count = 0
    failed_count = 0
    no_image_count = 0  # Count for rows skipped due to missing images
    label_stats = {}
    
    # Use thread lock for thread-safe counter updates
    stats_lock = Lock()
    
    logger.info(f"Processing {len(data)} rows with {max_workers} parallel workers...")
    logger.info(f"Download new images: {download_new_images}")
    if download_bad_only:
        logger.info(f"Download only Bad label images: True")
    
    # Process data in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(process_single_row, row, image_dir, PROMPT_TEMPLATES, download_new_images, download_bad_only): idx 
            for idx, row in enumerate(data, start=1)
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            completed += 1
            
            try:
                result = future.result()
                
                if result is not None:
                    training_example, label, is_new_download = result
                    
                    with stats_lock:
                        training_data.append(training_example)
                        
                        if is_new_download:
                            downloaded_count += 1
                        else:
                            skipped_count += 1
                        
                        # Update label statistics
                        label_stats[label] = label_stats.get(label, 0) + 1
                else:
                    with stats_lock:
                        if download_new_images:
                            failed_count += 1
                        else:
                            no_image_count += 1
            
            except Exception as e:
                logger.error(f"Error processing row {idx}: {str(e)}")
                with stats_lock:
                    failed_count += 1
            
            # Log progress every 10000 examples
            if completed % 10000 == 0:
                logger.info(f"Processed {completed}/{len(data)} examples")
    
    logger.info(f"Processed all {len(data)} examples")
    
    # Log statistics
    logger.info(f"\n{'='*50}")
    logger.info(f"Processing complete!")
    logger.info(f"{'='*50}")
    logger.info(f"Total examples processed: {len(training_data)}")
    logger.info(f"Images downloaded: {downloaded_count}")
    logger.info(f"Images skipped (already exists): {skipped_count}")
    
    if download_new_images:
        logger.info(f"Failed examples (image download failed): {failed_count}")
    else:
        logger.info(f"Skipped examples (image not found, download disabled): {no_image_count}")
        logger.info(f"Failed examples (other errors): {failed_count}")
    
    logger.info(f"\nLabel distribution:")
    for label, count in sorted(label_stats.items()):
        logger.info(f"  {label}: {count} ({count/len(training_data)*100:.1f}%)")
    logger.info(f"{'='*50}\n")
    
    return training_data


def clean_unused_images(training_data_file: Path, image_dir: Path) -> None:
    """
    Remove images from image_dir that are not referenced in the training data.
    
    Args:
        training_data_file: Path to the training data JSON file
        image_dir: Directory containing the images
    """
    logger.info(f"\nCleaning unused images from {image_dir}...")
    
    # Check if training data file exists
    if not training_data_file.exists():
        logger.error(f"Training data file not found: {training_data_file}")
        return
    
    # Read training data to get used image paths
    try:
        with open(training_data_file, 'r', encoding='utf-8') as f:
            training_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read training data file: {e}")
        return
    
    # Collect all image filenames used in training data
    used_images = set()
    for example in training_data:
        if 'images' in example:
            for img_path in example['images']:
                # Extract just the filename from the path
                img_filename = Path(img_path).name
                used_images.add(img_filename)
    
    logger.info(f"Found {len(used_images)} unique images referenced in training data")
    
    # Get all image files in the directory
    all_images = []
    for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
        all_images.extend(image_dir.glob(f'*{ext}'))
    
    logger.info(f"Found {len(all_images)} total images in {image_dir}")
    
    # Delete unused images
    deleted_count = 0
    deleted_size = 0
    
    for img_path in all_images:
        if img_path.name not in used_images:
            try:
                file_size = img_path.stat().st_size
                img_path.unlink()
                deleted_count += 1
                deleted_size += file_size
                logger.debug(f"Deleted unused image: {img_path.name}")
            except Exception as e:
                logger.warning(f"Failed to delete {img_path.name}: {e}")
    
    if deleted_count > 0:
        size_mb = deleted_size / (1024 * 1024)
        logger.info(f"Deleted {deleted_count} unused images, freed {size_mb:.2f} MB of space")
    else:
        logger.info("No unused images found to delete")


def check_image_integrity(image_dir: Path, max_workers: int = 16) -> None:
    """
    Check all images in the directory for integrity and delete corrupted ones.
    Uses parallel processing for faster checking.
    
    Args:
        image_dir: Directory containing the images
        max_workers: Maximum number of parallel workers for checking
    """
    from PIL import Image
    
    logger.info(f"\nChecking image integrity in {image_dir}...")
    
    if not image_dir.exists():
        logger.error(f"Image directory not found: {image_dir}")
        return
    
    # Get all image files in the directory
    all_images = []
    for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
        all_images.extend(image_dir.glob(f'*{ext}'))
    
    logger.info(f"Found {len(all_images)} total images to check with {max_workers} parallel workers")
    
    corrupted_count = 0
    corrupted_size = 0
    stats_lock = Lock()
    
    def check_single_image(img_path: Path) -> Optional[Tuple[Path, int, str]]:
        """Check a single image and return info if corrupted."""
        try:
            # Quick file size check first
            if img_path.stat().st_size == 0:
                return (img_path, img_path.stat().st_size, "Empty file")
            
            # Try to open and verify the image
            with Image.open(img_path) as img:
                img.verify()
            
            # Reopen for a more thorough check (verify() closes the file)
            with Image.open(img_path) as img:
                img.load()
            
            return None  # Image is OK
            
        except Exception as e:
            # Image is corrupted
            file_size = img_path.stat().st_size if img_path.exists() else 0
            return (img_path, file_size, str(e))
    
    # Process images in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(check_single_image, img_path): img_path for img_path in all_images}
        
        completed = 0
        for future in as_completed(future_to_path):
            completed += 1
            
            try:
                result = future.result()
                
                if result is not None:
                    img_path, file_size, error_msg = result
                    logger.warning(f"Corrupted image detected: {img_path.name} - {error_msg}")
                    
                    try:
                        img_path.unlink()
                        with stats_lock:
                            corrupted_count += 1
                            corrupted_size += file_size
                        logger.info(f"Deleted corrupted image: {img_path.name}")
                    except Exception as del_error:
                        logger.error(f"Failed to delete corrupted image {img_path.name}: {del_error}")
            
            except Exception as e:
                logger.error(f"Error checking image: {e}")
            
            # Log progress every 10000 images
            if completed % 10000 == 0:
                logger.info(f"Checked {completed}/{len(all_images)} images")
    
    logger.info(f"Checked all {len(all_images)} images")
    
    if corrupted_count > 0:
        size_mb = corrupted_size / (1024 * 1024)
        logger.info(f"\nTotal corrupted images deleted: {corrupted_count}")
        logger.info(f"Space freed: {size_mb:.2f} MB")
    else:
        logger.info("\nAll images are valid. No corrupted images found.")


def main():
    """Main function to process the data."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Reshape training data for image-LP relevance task')
    parser.add_argument('--input-file', type=str, default=None,
                        help='Path to input TSV file (default: OriginalData.tsv in script directory)')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Path to output JSON file (default: training_data.json in script directory)')
    parser.add_argument('--image-dir', type=str, default=None,
                        help='Directory to store images (default: image_data in script directory)')
    parser.add_argument('--no-download', action='store_true',
                        help='Do not download new images, only use existing ones')
    parser.add_argument('--download-bad-only', action='store_true',
                        help='Only download images for Bad label (Good/Fair use existing images only)')
    parser.add_argument('--max-samples', type=int, default=-1,
                        help='Maximum number of total samples. Each label (Good/Fair/Bad) limited to 1/3 of this value (default: -1, no limit)')
    parser.add_argument('--clean-images', action='store_true',
                        help='Clean unused images from image directory (standalone mode)')
    parser.add_argument('--check-images', action='store_true',
                        help='Check image integrity and delete corrupted images (standalone mode)')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Define paths
    script_dir = Path(__file__).parent
    input_file = Path(args.input_file) if args.input_file else script_dir / "OriginalDataRound2.tsv"
    output_file = Path(args.output_file) if args.output_file else script_dir / "training_data_1.5MRound2.json"
    image_dir = Path(args.image_dir) if args.image_dir else script_dir / "image_data"
    
    # Handle standalone modes
    if args.clean_images:
        logger.info("Running in clean-images mode...")
        clean_unused_images(output_file, image_dir)
        return
    
    if args.check_images:
        logger.info("Running in check-images mode...")
        check_image_integrity(image_dir)
        return
    
    # Normal processing mode - Determine download settings
    download_new_images = not args.no_download
    download_bad_only = args.download_bad_only
    max_samples = args.max_samples
    
    # Create image directory if it doesn't exist
    image_dir.mkdir(exist_ok=True)
    logger.info(f"Image directory: {image_dir}")
    
    # Check if input file exists
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return
    
    # Read TSV data
    logger.info(f"Reading data from: {input_file}")
    data = read_tsv_data(str(input_file), max_samples=max_samples)

    if not data:
        logger.error("No data read from input file")
        return

    # Shuffle data (already shuffled in apply_max_samples_limit if max_samples > 0)
    if max_samples <= 0:
        random.shuffle(data)
        logger.info(f"Data shuffled with random seed 42")
    
    # Create training data
    logger.info("Starting data processing...")
    training_data = create_training_data(data, image_dir, download_new_images=download_new_images, download_bad_only=download_bad_only)

    if not training_data:
        logger.error("No training data created")
        return

    # Save to JSON file
    logger.info(f"Saving training data to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Training data saved successfully!")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Total examples: {len(training_data)}")


if __name__ == "__main__":
    main()

