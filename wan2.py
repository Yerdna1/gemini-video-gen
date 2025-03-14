import replicate
import os
from dotenv import load_dotenv
import base64

# Load environment variables
load_dotenv()

def generate_video_from_local_image(image_path, prompt, output_path="output.mp4"):
    """
    Generate a video from a local image using Replicate's WAN-2.1 model
    
    Args:
        image_path (str): Path to the local image file
        prompt (str): Text prompt describing the video to generate
        output_path (str): Path where the output video will be saved
        
    Returns:
        str: Path to the generated video file
    """
    # Read the image file and encode it as base64
    with open(image_path, "rb") as image_file:
        # Convert the image to base64 for API consumption
        image_data = base64.b64encode(image_file.read()).decode("utf-8")
        
    # Prepare the input for the model
    input = {
        "image": f"data:image/jpeg;base64,{image_data}",
        "prompt": prompt
    }
    
    # Run the model
    output = replicate.run(
        "wavespeedai/wan-2.1-i2v-480p",
        input=input
    )
    
    # Save the output video
    with open(output_path, "wb") as file:
        file.write(output.read())
    
    return output_path

def generate_video_from_bytes(image_bytes, prompt, output_path="output.mp4"):
    """
    Generate a video from image bytes using Replicate's WAN-2.1 model
    
    Args:
        image_bytes (bytes): Image data as bytes
        prompt (str): Text prompt describing the video to generate
        output_path (str): Path where the output video will be saved
        
    Returns:
        str: Path to the generated video file
    """
    # Convert the image bytes to base64
    image_data = base64.b64encode(image_bytes).decode("utf-8")
    
    # Prepare the input for the model
    input = {
        "image": f"data:image/jpeg;base64,{image_data}",
        "prompt": prompt
    }
    
    # Run the model
    output = replicate.run(
        "wavespeedai/wan-2.1-i2v-480p",
        input=input
    )
    
    # Save the output video
    with open(output_path, "wb") as file:
        file.write(output.read())
    
    return output_path

# Example usage when script is run directly
if __name__ == "__main__":
    # Example with a local image
    image_path = "image_croissant.jpeg"  # Replace with your local image path
    prompt = "A croissant with chocolate drizzle"
    
    output_file = generate_video_from_local_image(image_path, prompt)
    print(f"Video generated and saved to: {output_file}")