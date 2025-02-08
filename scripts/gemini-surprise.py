import os
import pandas as pd
from PIL import Image
import google.generativeai as genai
from tqdm import tqdm
import time
from datetime import datetime
import json
import re


class RateLimiter:
    def __init__(self, rpm_limit=5, rpd_limit=1500):
        self.rpm_limit = rpm_limit
        self.rpd_limit = rpd_limit
        self.requests_today = 0
        self.request_times = []  # Track timestamp of each request
        self.state_file = "rate_limiter_state.json"
        self.load_state()

    def load_state(self):
        """Load previous state if exists"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                    self.requests_today = state.get("requests_today", 0)
                    last_date = datetime.fromisoformat(
                        state.get("last_date", datetime.now().isoformat())
                    )

                    # Reset counter if it's a new day
                    if last_date.date() != datetime.now().date():
                        self.requests_today = 0
            except Exception as e:
                print(f"Error loading state: {e}")
                self.requests_today = 0

    def save_state(self):
        """Save current state"""
        state = {
            "requests_today": self.requests_today,
            "last_date": datetime.now().isoformat(),
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f)

    def get_current_rpm(self):
        """Calculate current requests per minute"""
        current_time = time.time()
        minute_ago = current_time - 60
        return len([t for t in self.request_times if t > minute_ago])

    def wait_if_needed(self):
        """Check and wait if needed to comply with rate limits"""
        current_time = time.time()

        # Clean up old request times
        minute_ago = current_time - 60
        self.request_times = [t for t in self.request_times if t > minute_ago]

        # Print current RPM
        current_rpm = self.get_current_rpm()
        print(
            f"\rCurrent RPM: {current_rpm}/{self.rpm_limit} | Requests today: {self.requests_today}/{self.rpd_limit}",
            end="",
        )

        # Check daily limit
        if self.requests_today >= self.rpd_limit:
            raise Exception("Daily request limit reached")

        # Check and wait for RPM limit
        while len(self.request_times) >= self.rpm_limit:
            sleep_time = self.request_times[0] + 60 - current_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            current_time = time.time()
            self.request_times = [
                t for t in self.request_times if t > current_time - 60
            ]

        # Add current request
        self.request_times.append(current_time)
        self.requests_today += 1
        self.save_state()


def extract_rating_and_reasoning(text):
    """Extract rating and reasoning from the model's response"""
    match = re.search(r"<<\d+>>", text)
    rating = int(match.group(0)[2:-2]) if match else None
    reasoning = text.replace(match.group(0), "").strip() if match else text.strip()
    reasoning = re.sub(r"[\t\n]", " ", reasoning).strip()  # Remove tabs and new lines
    return rating, reasoning


def get_image_surprisals(
    image_folder,
    features_csv,
    api_key,
    output_file="image_surprisals.csv",
    start_index=0,
):
    """
    Process images listed in features CSV file and save surprisal ratings and reasoning
    """
    # Configure Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Initialize rate limiter
    rate_limiter = RateLimiter()

    # Load features CSV
    try:
        features_df = pd.read_csv(features_csv)
        if "image_id" not in features_df.columns:
            raise ValueError("'image_id' column not found in features CSV")
        image_files = features_df["image_id"].tolist()
        print(f"Loaded {len(image_files)} image IDs from features CSV")
    except Exception as e:
        raise Exception(f"Error loading features CSV: {str(e)}")

    # Initialize results list and processed images set
    results = []
    processed_images = set()

    # Load existing results if output file exists
    if os.path.exists(output_file):
        print(f"Found existing output file: {output_file}")
        try:
            existing_df = pd.read_csv(output_file)
            results = existing_df.to_dict("records")
            processed_images = set(existing_df["image_id"])
            print(f"Loaded {len(processed_images)} existing ratings")
        except Exception as e:
            print(f"Error loading existing file: {e}")
            print("Starting fresh...")

    # Add "jpg" extension to image files if not present
    processed_images = set(
        [
            img + ".jpg" if not img.endswith((".png", ".bmp", ".jpg", ".jpeg")) else img
            for img in processed_images
        ]
    )
    image_files = [
        (
            str(img) + ".jpg"
            if not str(img).endswith((".png", ".bmp", ".jpg", ".jpeg"))
            else img
        )
        for img in image_files
    ]

    # Filter out already processed images and apply start_index
    remaining_images = [
        img for img in image_files[start_index:] if img not in processed_images
    ]

    print(f"Found {len(image_files)} total images in CSV")
    print(f"Found {len(remaining_images)} images remaining to process")

    if not remaining_images:
        print("No new images to process!")
        return pd.DataFrame(results)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Process each remaining image
    for img_file in tqdm(remaining_images, desc="Processing images"):
        try:
            # Check rate limits
            rate_limiter.wait_if_needed()

            # Load image
            img_file = str(img_file)
            if not img_file.endswith((".png", ".bmp", ".jpg", ".jpeg")):
                img_file = img_file + ".jpg"
            img_path = os.path.join(image_folder, img_file)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")

            img = Image.open(img_path)

            # Generate surprisal rating with reasoning
            prompt = [
                "Step by step, explain why this image is surprising or not. Consider factors like rare events, or unexpected content. Be precise in your reasoning. Then, on a precise scale from 0 to 100, rate the surprisal of this image. Provide your reasoning and numeric rating as follows: 'Reasoning: [your explanation] Rating: <<number>>'.",
                img,
            ]
            response = model.generate_content(prompt)

            # Extract rating and reasoning
            rating, reasoning = extract_rating_and_reasoning(response.text.strip())
            if rating is None:
                raise ValueError("Could not extract valid rating from response")

            # Store result
            results.append(
                {
                    "image_id": img_file,
                    "surprisal_rating": rating,
                    "reasoning": reasoning,
                    "raw_response": response.text.strip(),
                }
            )

            # Save progress after each image
            pd.DataFrame(results).to_csv(output_file, index=False)

        except Exception as e:
            print(f"\nError processing {img_file}: {str(e)}")
            if "Daily request limit reached" in str(e):
                print("\nReached daily limit. Try again tomorrow.")
                break

            results.append(
                {
                    "image_id": img_file,
                    "complexity_rating": None,
                    "reasoning": "ERROR",
                    "raw_response": f"ERROR: {str(e)}",
                }
            )
            pd.DataFrame(results).to_csv(output_file, index=False)

    print("\nProcessing complete!")
    return pd.DataFrame(results)


# Usage example
if __name__ == "__main__":
    # Configuration

    IMAGE_FOLDER = "../datasets/SVG"  # Base folder containing images
    FEATURES_CSV = "../features/sample_features.csv"
    API_KEY = ""  # Replace with your Gemini API key
    OUTPUT_FILE = "SVG_surprise_scores.csv"
    START_INDEX = 0  # Change this if you want to start from a specific index

    # Process images and get surprisal ratings
    surprisals_df = get_image_surprisals(
        IMAGE_FOLDER, FEATURES_CSV, API_KEY, OUTPUT_FILE, START_INDEX
    )

    # Display summary
    print("\nProcessing Summary:")
    print(f"Total ratings in output: {len(surprisals_df)}")
    print("\nFirst few ratings:")
    print(surprisals_df.head())
