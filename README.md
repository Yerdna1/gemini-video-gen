# Gemini Image Chat with Video Generation

A Streamlit application that allows users to upload multiple images, chat with Google's Gemini 2.0 model, and generate videos from Gemini-generated images using Replicate's WAN-2.1 model. The app can generate both text and image responses from Gemini, and animate AI-generated images into videos.

## Features

- Upload multiple images at once through a simple interface
- Images are sent to Gemini with the first message only
- Subsequent messages contain only text for a more natural conversation flow
- View all your images together in the first chat message
- Toggle option to show/hide images in the chat for a cleaner interface
- Reset conversation option to send images again
- Receive both text and image responses from Gemini
- Generate videos from Gemini-generated images using Replicate's WAN-2.1 model
- Download generated videos or add them to the chat
- Persistent chat history during the session
- Clear chat and image management functionality
- Tabbed interface for image chat and video generation
- All generated files (images and videos) are saved to a 'temp' folder in the project root directory

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory with the following variables:
   ```
   GEMINI_API_KEY=your_gemini_api_key
   HELICONE_API_KEY=your_helicone_api_key
   REPLICATE_API_TOKEN=your_replicate_api_token
   ```
4. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Usage

1. Upload multiple images at once using the sidebar uploader
2. Your first message will include all uploaded images
3. Subsequent messages will only contain text (like a normal chat)
4. Use the "Show all images in chat" toggle to control image visibility
5. Use the "Reset conversation" button if you want to send images again
6. View Gemini's responses in the chat area
7. Switch to the "Video Generation" tab to create videos from Gemini-generated images
8. Select an image, enter a prompt, and generate a video
9. Download the generated video or add it to the chat
10. Use the management buttons to:
    - Remove specific images
    - Clear all images
    - Clear the chat history

## Video Generation

The app uses Replicate's WAN-2.1 model to generate videos from Gemini-generated images. This model can animate images based on text prompts, creating short videos that bring AI-generated images to life.

To generate a video:
1. Chat with Gemini and get it to generate images
2. Switch to the "Video Generation" tab
3. Select one of the Gemini-generated images by clicking "Select for Video"
4. Enter a prompt describing the motion or action you want to see
5. Click "Generate Video" to start the process (this may take a minute or two)
6. Once generated, you can download the video or add it to the chat

## Requirements

- Python 3.7+
- Streamlit
- Google Generative AI Python SDK
- Replicate Python SDK
- Python-dotenv
- Pillow (PIL)

## Note

Generated images and videos are saved to a 'temp' folder in the project's root directory. The path to this directory is displayed at the top of the app. This keeps your root directory organized while still allowing you to download any generated files you want to keep. 