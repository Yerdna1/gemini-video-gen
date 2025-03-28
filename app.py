import base64
import os
import io
import uuid
import shutil
from dotenv import load_dotenv
import google.generativeai as genai  # Correct import
import pathlib
import streamlit as st
from PIL import Image
import wan2  # Import the video generation module

# Load environment variables
load_dotenv()
gemini_api_key = os.environ.get("GEMINI_API_KEY")
helicone_api_key = os.environ.get("HELICONE_API_KEY")

# Set page config
st.set_page_config(
    page_title="Gemini Image Chat",
    page_icon="🖼️",
    layout="wide"
)

# Create temp directory in the root folder for storing generated files
if "temp_dir" not in st.session_state:
    # Create a 'temp' folder in the root directory if it doesn't exist
    temp_dir = os.path.join(os.getcwd(), "temp")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    st.session_state.temp_dir = temp_dir

# Initialize session state for chat history and images
if "messages" not in st.session_state:
    st.session_state.messages = []

if "images" not in st.session_state:
    st.session_state.images = []

if "images_sent" not in st.session_state:
    st.session_state.images_sent = False

if "generated_images" not in st.session_state:
    st.session_state.generated_images = []

if "video_generation_state" not in st.session_state:
    st.session_state.video_generation_state = {
        "selected_image_idx": None,
        "prompt": "",
        "generating": False,
        "video_path": None
    }

# We no longer need active_image_index since we're using all images
if "show_all_images_in_chat" not in st.session_state:
    st.session_state.show_all_images_in_chat = True

def save_binary_file(data, mime_type):
    """Save binary data to a file with a unique name based on mime type in the temp directory"""
    extension = mime_type.split("/")[1]
    file_name = f"generated_{uuid.uuid4()}.{extension}"
    file_path = os.path.join(st.session_state.temp_dir, file_name)
    
    with open(file_path, "wb") as f:
        f.write(data)
    
    return file_path

def initialize_gemini_client():
    """Initialize and return the Gemini client"""
    genai.configure(api_key=gemini_api_key)
    
    # Create a generation config for text and image generation
    generation_config = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
    }
    
    # Create the model
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config=generation_config,
    )
    
    return model

def generate_response(prompt, include_images=False):
    """Generate a response from Gemini model"""
    model = initialize_gemini_client()
    
    try:
        # Prepare content
        content_parts = [prompt]
        
        # Add all images if this is the first message and we have images
        if include_images and st.session_state.images and not st.session_state.images_sent:
            for img_data in st.session_state.images:
                # Prepare image for Gemini API
                image_part = {
                    "mime_type": "image/jpeg",
                    "data": img_data["data"]
                }
                content_parts.append(image_part)
            
            # Mark that we've sent images
            st.session_state.images_sent = True
        
        # Generate response
        response = model.generate_content(content_parts)
        
        # Extract text response
        response_text = response.text if hasattr(response, 'text') else ""
        
        # For now, we're not handling image responses as the current API might not support it
        response_image = None
        response_mime_type = None
        
        return response_text, response_image, response_mime_type
    
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return f"Error: {str(e)}", None, None

# App UI
st.title("Gemini Image Chat")
st.markdown("Upload images and chat with Gemini about them. Gemini can generate both text and images in response.")

# Sidebar for image upload and management
with st.sidebar:
    st.header("Upload & Manage Images")
    
    # Image uploader - with multiple file support
    uploaded_files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        # Process all uploaded images
        for uploaded_file in uploaded_files:
            # Check if this image already exists
            image_exists = False
            for existing_img in st.session_state.images:
                if existing_img["name"] == uploaded_file.name:
                    image_exists = True
                    break
                    
            if not image_exists:
                # Process the uploaded image
                image = Image.open(uploaded_file)
                
                # Convert to bytes for Gemini API
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format or "JPEG")
                img_bytes = img_byte_arr.getvalue()
                
                # Add to session state
                st.session_state.images.append({
                    "name": uploaded_file.name,
                    "data": img_bytes,
                    "format": image.format or "JPEG"
                })
                
                # Reset images_sent flag when new images are added
                st.session_state.images_sent = False
    
    # Display and manage uploaded images
    if st.session_state.images:
        st.subheader(f"Your Images ({len(st.session_state.images)})")
        
        # Toggle for showing all images in chat
        st.checkbox("Show all images in chat", value=st.session_state.show_all_images_in_chat, 
                   key="show_images_toggle", 
                   help="When enabled, all images will be displayed in the chat. When disabled, only a count will be shown.")
        
        # Create columns for the gallery
        cols = st.columns(2)
        
        for i, img_data in enumerate(st.session_state.images):
            col_idx = i % 2
            with cols[col_idx]:
                # Load image from bytes
                img = Image.open(io.BytesIO(img_data["data"]))
                
                # Display image with caption
                st.image(img, caption=f"{i+1}. {img_data['name']}")
                
                # Remove button
                if st.button("Remove", key=f"remove_{i}"):
                    st.session_state.images.pop(i)
                    # Reset images_sent flag when images are removed
                    st.session_state.images_sent = False
                    st.rerun()
                
    # Clear all button
    if st.session_state.images:
        if st.button("Clear All Images"):
            st.session_state.images = []
            st.session_state.images_sent = False
            st.rerun()
    
    # Clear chat button
    if st.session_state.messages:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.images_sent = False
            st.session_state.generated_images = []  # Also clear generated images
            st.rerun()
            
    # Information about image handling
    if st.session_state.images:
        if not st.session_state.images_sent:
            st.info(f"All {len(st.session_state.images)} images will be sent with your next message.")

# Main content area with tabs
tab1, tab2 = st.tabs(["Image Chat", "Video Generation"])

with tab1:
    # Display chat messages
    st.header("Chat")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # If message has images and we want to show them
            if "images" in message and message["images"] and st.session_state.show_all_images_in_chat:
                # Create a horizontal layout for images
                image_cols = st.columns(min(len(message["images"]), 4))
                
                for idx, img_idx in enumerate(message["images"]):
                    col_idx = idx % min(len(message["images"]), 4)
                    with image_cols[col_idx]:
                        img_data = st.session_state.images[img_idx]
                        img = Image.open(io.BytesIO(img_data["data"]))
                        st.image(img, caption=f"{img_data['name']}", width=150)
                
                # Add a note about the images
                st.caption(f"Message included {len(message['images'])} images")
            elif "images" in message and message["images"]:
                # Just show a note about the images
                st.caption(f"Message included {len(message['images'])} images")
            
            # Display video if the message has one
            if "video_path" in message and message["video_path"]:
                video_path = message["video_path"]
                video_file = open(video_path, 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes)
            
            # Display text content
            if isinstance(message["content"], str):
                # Text message
                st.markdown(message["content"])
            else:
                # Image message from model
                file_name = save_binary_file(message["content"], message["mime_type"])
                st.image(file_name, caption="Generated Image")
                
                # Store the generated image for video generation
                if message["role"] == "assistant" and not isinstance(message["content"], str):
                    # Add to generated images if not already there
                    image_exists = False
                    for img in st.session_state.generated_images:
                        if img["data"] == message["content"]:
                            image_exists = True
                            break
                    
                    if not image_exists:
                        st.session_state.generated_images.append({
                            "name": os.path.basename(file_name),
                            "data": message["content"],
                            "mime_type": message["mime_type"],
                            "file_path": file_name
                        })

    # Chat input
    if prompt := st.chat_input("Message Gemini..."):
        # Get indices of all images
        all_image_indices = list(range(len(st.session_state.images))) if st.session_state.images else []
        
        # Add user message to chat history with all images (for display purposes)
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt,
            "images": all_image_indices if not st.session_state.images_sent else []
        })
        
        # Display user message
        with st.chat_message("user"):
            # Display all images if we have any, haven't sent them yet, and the toggle is on
            if all_image_indices and not st.session_state.images_sent and st.session_state.show_all_images_in_chat:
                # Create a horizontal layout for images
                image_cols = st.columns(min(len(all_image_indices), 4))
                
                for idx, img_idx in enumerate(all_image_indices):
                    col_idx = idx % min(len(all_image_indices), 4)
                    with image_cols[col_idx]:
                        img_data = st.session_state.images[img_idx]
                        img = Image.open(io.BytesIO(img_data["data"]))
                        st.image(img, caption=f"{img_data['name']}", width=150)
                
                # Add a note about the images
                st.caption(f"Message includes all {len(all_image_indices)} images")
            elif all_image_indices and not st.session_state.images_sent:
                # Just show a note about the images
                st.caption(f"Message includes all {len(all_image_indices)} images")
            
            st.markdown(prompt)
        
        # Get response from Gemini
        with st.chat_message("assistant"):
            with st.spinner("Gemini is thinking..."):
                # Send images only if they haven't been sent yet
                response_text, response_image, response_mime_type = generate_response(
                    prompt, 
                    include_images=True  # The function will check st.session_state.images_sent
                )
                
                # Display response
                if response_text:
                    st.markdown(response_text)
                    # Add text response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                
                if response_image:
                    # Save and display image
                    file_name = save_binary_file(response_image, response_mime_type)
                    st.image(file_name, caption="Generated Image")
                    
                    # Add image response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response_image,
                        "mime_type": response_mime_type
                    })
                    
                    # Store the generated image for video generation
                    st.session_state.generated_images.append({
                        "name": os.path.basename(file_name),
                        "data": response_image,
                        "mime_type": response_mime_type,
                        "file_path": file_name
                    })

with tab2:
    st.header("Video Generation from Gemini Images")
    st.markdown("Generate videos from images that Gemini has created during your conversation.")
    
    if not st.session_state.generated_images:
        st.info("No images have been generated by Gemini yet. Chat with Gemini and ask it to generate images first.")
    else:
        st.subheader(f"Generated Images ({len(st.session_state.generated_images)})")
        
        # Create a grid for the generated images
        cols = st.columns(3)
        
        for i, img_data in enumerate(st.session_state.generated_images):
            col_idx = i % 3
            with cols[col_idx]:
                # Display image
                st.image(img_data["file_path"], caption=f"Image {i+1}: {img_data['name']}", use_container_width=True)
                
                # Select button
                if st.button("Select for Video", key=f"select_for_video_{i}"):
                    st.session_state.video_generation_state["selected_image_idx"] = i
                    st.rerun()
        
        # Video generation UI
        if st.session_state.video_generation_state["selected_image_idx"] is not None:
            st.divider()
            st.subheader("Generate Video")
            
            # Get the selected image
            img_idx = st.session_state.video_generation_state["selected_image_idx"]
            img_data = st.session_state.generated_images[img_idx]
            
            # Display the selected image
            st.image(img_data["file_path"], caption=f"Selected image: {img_data['name']}", width=300)
            
            # Prompt input
            prompt = st.text_input("Enter a prompt for the video:", 
                                value=st.session_state.video_generation_state["prompt"],
                                key="video_prompt",
                                help="Describe what action or movement you want to see in the video")
            
            # Store the prompt in session state
            st.session_state.video_generation_state["prompt"] = prompt
            
            # Generate button
            generate_col, cancel_col = st.columns([1, 1])
            
            with generate_col:
                if st.button("Generate Video", key="generate_video_btn"):
                    if prompt:
                        with st.spinner("Generating video... This may take a minute or two."):
                            st.session_state.video_generation_state["generating"] = True
                            
                            # Generate a unique filename for the video
                            video_filename = f"video_{uuid.uuid4()}.mp4"
                            video_path = os.path.join(st.session_state.temp_dir, video_filename)
                            
                            try:
                                # Generate the video using the wan2 module
                                video_path = wan2.generate_video_from_bytes(
                                    img_data["data"],
                                    prompt,
                                    output_path=video_path
                                )
                                
                                # Store the video path in session state
                                st.session_state.video_generation_state["video_path"] = video_path
                                st.session_state.video_generation_state["generating"] = False
                                
                                # Force a rerun to display the video
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error generating video: {str(e)}")
                                st.session_state.video_generation_state["generating"] = False
                    else:
                        st.warning("Please enter a prompt for the video.")
            
            with cancel_col:
                if st.button("Cancel", key="cancel_video_btn"):
                    st.session_state.video_generation_state["selected_image_idx"] = None
                    st.session_state.video_generation_state["prompt"] = ""
                    st.session_state.video_generation_state["generating"] = False
                    st.rerun()
            
            # Display the generated video if available
            if st.session_state.video_generation_state["video_path"]:
                st.divider()
                st.subheader("Generated Video")
                video_path = st.session_state.video_generation_state["video_path"]
                
                # Display the video
                video_file = open(video_path, 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes)
                
                # Download button
                st.download_button(
                    label="Download Video",
                    data=video_bytes,
                    file_name=os.path.basename(video_path),
                    mime="video/mp4"
                )
                
                # Add to chat button
                if st.button("Add to Chat", key="add_video_to_chat"):
                    # Create a message with the video
                    st.session_state.messages.append({
                        "role": "user",
                        "content": f"I generated a video from Gemini-generated image with prompt: '{prompt}'",
                        "video_path": video_path
                    })
                    
                    # Reset the video generation state
                    st.session_state.video_generation_state["selected_image_idx"] = None
                    st.session_state.video_generation_state["prompt"] = ""
                    st.session_state.video_generation_state["video_path"] = None
                    st.rerun()