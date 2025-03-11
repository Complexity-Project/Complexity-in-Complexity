`gemini-surprise.py` generates surprise scores for images using the [Gemini 1.5 Flash](https://ai.google.dev/gemini-api/docs/models/gemini#gemini-1.5-flash) model. Provide the folder with images, the feature CSV file, an output folder, and your API key. The script only processes images listed in the feature CSV, so make sure the filenames match. If it stops due to an error, just rerun it and it will continue from where it left off, based on the existing output file.

The script saves its output in a CSV with the following columns:

- **image_id**: The filename or identifier of the image processed.
- **surprisal_rating**: Numerical value indicating how surprising the Gemini model finds the image.
- **reasoning**: A brief text explaining why the model assigned that surprisal rating.
- **raw_response**: The raw response text from the Gemini API call, included for reference.

You can use `algorithms.py` to calculate Multi-Scale Sobel Gradients (MSG) and Multi-Scale Unique Colors (MUC) scores for any image. The script takes an image file and prints the scores to the console. 
 