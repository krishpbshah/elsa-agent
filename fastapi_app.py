import vertexai
from vertexai.generative_models import GenerativeModel, Part
import base64
import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

# --- 1. CONFIGURATION ---

PROJECT_ID = "elsaugc"
LOCATION = "us-central1"
BUCKET_NAME = "elsa-ai-training-dataset-bucket"

# ---
# CREDENTIALS HANDLING
# Checks for environment variable first (Vercel), then falls back to local file.
import tempfile

SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")

if SERVICE_ACCOUNT_JSON:
    # We are likely on Vercel or an environment with the JSON string
    # Create a temporary file to store the credentials
    try:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as temp:
            temp.write(SERVICE_ACCOUNT_JSON)
            temp_path = temp.name
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_path
        print(f"Loaded credentials from environment variable to {temp_path}")
    except Exception as e:
        print(f"Error creating temp credential file: {e}")
else:
    # Fallback to local file (for local development if file exists)
    SERVICE_ACCOUNT_PATH = "elsaugc-6755d1abf892.json"
    if os.path.exists(SERVICE_ACCOUNT_PATH):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_PATH
        print(f"Loaded credentials from local file: {SERVICE_ACCOUNT_PATH}")
    else:
        print("WARNING: No credentials found! Set GOOGLE_APPLICATION_CREDENTIALS_JSON env var.")
# ---

# Your 14 Chibi style images
CHIBI_STYLE_GUIDE_URIS = [
    f"gs://{BUCKET_NAME}/chibi_dataset_images/elsa_01.png",
    f"gs://{BUCKET_NAME}/chibi_dataset_images/elsa_02.png",
    f"gs://{BUCKET_NAME}/chibi_dataset_images/elsa_03.png",
    f"gs://{BUCKET_NAME}/chibi_dataset_images/elsa_04.png",
    f"gs://{BUCKET_NAME}/chibi_dataset_images/elsa_10.png",
    f"gs://{BUCKET_NAME}/chibi_dataset_images/elsa_15.png",
    f"gs://{BUCKET_NAME}/chibi_dataset_images/elsa_09_dollar.png",
    f"gs://{BUCKET_NAME}/chibi_dataset_images/elsa_09_heart.png",
    f"gs://{BUCKET_NAME}/chibi_dataset_images/elsa_09_sad.png",
    f"gs://{BUCKET_NAME}/chibi_dataset_images/elsa_13_wink.png",
    f"gs://{BUCKET_NAME}/chibi_dataset_images/elsa_13_star.png",
    f"gs://{BUCKET_NAME}/chibi_dataset_images/elsa_13_sleep.png",
    f"gs://{BUCKET_NAME}/chibi_dataset_images/elsa_13_angry.png",
    f"gs://{BUCKET_NAME}/chibi_dataset_images/elsa_20_cropped.png"
]

# Your 9 Robot style images
ROBOT_STYLE_GUIDE_URIS = [
    f"gs://{BUCKET_NAME}/robot_images/elsa_05.png",
    f"gs://{BUCKET_NAME}/robot_images/elsa_06.png",
    f"gs://{BUCKET_NAME}/robot_images/elsa_07.png",
    f"gs://{BUCKET_NAME}/robot_images/elsa_08.png",
    f"gs://{BUCKET_NAME}/robot_images/elsa_11.png",
    f"gs://{BUCKET_NAME}/robot_images/elsa_12.png",
    f"gs://{BUCKET_NAME}/robot_images/elsa_17.png",
    f"gs://{BUCKET_NAME}/robot_images/elsa_18.png",
    f"gs://{BUCKET_NAME}/robot_images/elsa_19.png"
]

# --- 2. INITIALIZE SERVICES ---
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Load the models
image_model = GenerativeModel(model_name="gemini-2.5-flash-image")
text_model = GenerativeModel(model_name="gemini-2.5-pro")

# --- 3. DEFINE IMAGE GENERATION FUNCTIONS (RETURNING BASE64) ---
# (These functions are unchanged from your file)

def generate_chibi_image(prompt: str) -> dict:
    """Generates an image in the 'elsa_chibi_face' style and returns base64."""
    print(f"Tool Call: generate_chibi_image('{prompt}')...")
    
    prompt_parts = [
        f"Using the {len(CHIBI_STYLE_GUIDE_URIS)} uploaded images as a visual style reference for the 'chibi mascot' character, fulfill this request: '{prompt}'"
    ]
    prompt_parts.extend([Part.from_uri(uri, mime_type="image/png") for uri in CHIBI_STYLE_GUIDE_URIS])

    try:
        response = image_model.generate_content(prompt_parts)
        
        image_bytes = None
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                image_bytes = part.inline_data.data
                break

        if image_bytes is None:
            raise Exception("Model did not return an image. It may have only returned text.")
        
        # Convert to base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        return {
            "success": True,
            "type": "image",
            "data": image_base64,
            "message": "Chibi image generated successfully!"
        }

    except Exception as e:
        return {
            "success": False,
            "type": "error",
            "message": f"Error generating Chibi image: {e}"
        }


def generate_robot_image(prompt: str) -> dict:
    """Generates an image in the 'elsa_robot' (red robot) style and returns base64."""
    print(f"Tool Call: generate_robot_image('{prompt}')...")
    
    prompt_parts = [
        f"Using the {len(ROBOT_STYLE_GUIDE_URIS)} uploaded images as a visual style reference for the 'red robot' character, fulfill this request: '{prompt}'"
    ]
    prompt_parts.extend([Part.from_uri(uri, mime_type="image/png") for uri in ROBOT_STYLE_GUIDE_URIS])

    try:
        response = image_model.generate_content(prompt_parts)
        
        image_bytes = None
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                image_bytes = part.inline_data.data
                break

        if image_bytes is None:
            raise Exception("Model did not return an image. It may have only returned text.")
        
        # Convert to base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        return {
            "success": True,
            "type": "image",
            "data": image_base64,
            "message": "Robot image generated successfully!"
        }
        
    except Exception as e:
        return {
            "success": False,
            "type": "error",
            "message": f"Error generating Robot image: {e}"
        }

def generate_general_image(prompt: str) -> dict:
    """Generates a general image with NO style guide and returns base64."""
    print(f"Tool Call: generate_general_image('{prompt}')...")

    try:
        response = image_model.generate_content([prompt]) 
        
        image_bytes = None
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                image_bytes = part.inline_data.data
                break

        if image_bytes is None:
            raise Exception("Model did not return an image. It may have only returned text.")
        
        # Convert to base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        return {
            "success": True,
            "type": "image",
            "data": image_base64,
            "message": "General image generated successfully!"
        }

    except Exception as e:
        return {
            "success": False,
            "type": "error",
            "message": f"Error generating general image: {e}"
        }


def generate_diagram_code(topic: str) -> dict:
    """Generates Mermaid.js flowchart code for a specific topic."""
    print(f"Generating Diagram code for: '{topic}'...")
    
    system_instruction = (
        "You are a diagramming expert. Your only job is to generate valid "
        "Mermaid.js code for the user's request. Respond ONLY with the "
        "Mermaid.js code block, starting with ```mermaid and ending with ```. "
        "Do not add any other explanatory text."
    )
    
    diagram_model = GenerativeModel(
        model_name="gemini-2.5-pro",
        system_instruction=system_instruction
    )
    
    try:
        response = diagram_model.generate_content(topic)
        return {
            "success": True,
            "type": "text",
            "data": response.text,
            "message": "Diagram code generated successfully!"
        }
    except Exception as e:
        return {
            "success": False,
            "type": "error",
            "message": f"Error generating diagram: {e}"
        }

# --- 4. AGENT BRAIN & NEW PROMPT ENHANCER TOOL ---

def get_agent_decision(user_prompt: str) -> tuple:
    """
    This function acts as the "agent brain."
    It decides which tool to use.
    """
    print(f"\nAgent is thinking about: '{user_prompt}'...")
    
    system_instruction = (
        "You are a routing agent. Your job is to analyze the user's prompt and "
        "decide which tool is the correct one to use. "
        "You must respond with a JSON object like: {\"tool\": \"...\"}"
        "\n"
        "Here are the four tools:"
        "1. 'chibi': Use this if the prompt mentions 'chibi', '2D face', or the 'chibi mascot'."
        "2. 'robot': Use this if the prompt mentions 'red robot', '3D robot', or the 'robot mascot'."
        "3. 'diagram': Use this for any 'flowchart', 'diagram', 'schematic', or general text question."
        "4. 'general': Use this ONLY if the prompt asks for an image but does NOT mention 'chibi', 'robot', or 'diagram'."
        "\n"
        "Prioritize the mascots. If a complex prompt (like a 'banner' or 'scene') mentions 'chibi', "
        "the tool MUST be 'chibi'. 'general' is only a fallback."
    )
    
    routing_model = GenerativeModel(
        model_name="gemini-2.5-pro",
        system_instruction=system_instruction
    )
    
    try:
        response = routing_model.generate_content(user_prompt)
        
        json_text = response.text.strip().lstrip("```json").rstrip("```")
        decision = json.loads(json_text)
        
        tool = decision.get("tool", "general").lower()
        
        if tool not in ['chibi', 'robot', 'diagram', 'general']:
            print(f"Agent made an invalid decision: '{tool}'. Defaulting to 'general'.")
            tool = 'general'
        
        print(f"Agent decided to use tool: '{tool}'")
        return tool, user_prompt
        
    except Exception as e:
        print(f"Error in agent brain: {e}. Defaulting to 'general'.")
        return 'general', user_prompt

# --- NEW: PROMPT ENHANCER FUNCTION FOR X/TWITTER BANNERS ---
def enhance_prompt(topic: str) -> str:
    """
    Uses Gemini 2.5 Pro to turn article text or a simple idea into a detailed 
    X/Twitter banner image prompt. Optimized for banner dimensions and visual impact.
    """
    print(f"Enhancing prompt for topic: '{topic[:100]}...'")
    
    system_instruction = (
        "You are an expert at creating prompts for X (Twitter) banner images. "
        "The user will provide article text, a blog post, or a simple idea. "
        "Your job is to analyze the content and create a detailed, visually descriptive "
        "prompt for generating a professional X/Twitter banner image.\n\n"
        
        "X/Twitter banner requirements:\n"
        "- Aspect ratio: 3:1 (typically 1500x500px or 1200x400px)\n"
        "- Wide horizontal format suitable for header images\n"
        "- Visual elements should be readable and impactful when viewed as a banner\n"
        "- Text should be minimal or integrated visually (the image model may add text)\n"
        "- Professional, modern design aesthetic\n\n"
        
        "Your enhanced prompt should:\n"
        "1. Extract the core theme, key concepts, and main message from the content\n"
        "2. Describe a compelling visual scene that represents the article's essence\n"
        "3. Include specific details about: composition (wide horizontal layout), colors, "
        "lighting, style, mood, and visual elements\n"
        "4. If the article mentions specific characters (like 'Elsa's agents', 'robot mascot', "
        "'chibi mascot'), incorporate them naturally into the banner design\n"
        "5. Create a prompt that will result in a professional, shareable banner image\n\n"
        
        "Respond ONLY with the enhanced image generation prompt. Do not include explanations, "
        "markdown formatting, or any other text. Just the prompt itself."
    )
    
    prompt_enhancer_model = GenerativeModel(
        model_name="gemini-2.5-pro",
        system_instruction=system_instruction
    )
    
    try:
        # Truncate very long articles to avoid token limits (keep first 8000 chars which should be enough)
        truncated_topic = topic[:8000] if len(topic) > 8000 else topic
        if len(topic) > 8000:
            print(f"Truncating topic from {len(topic)} to {len(truncated_topic)} characters")
        
        # Add context about banner creation
        enhanced_request = (
            f"Create a detailed image generation prompt for an X/Twitter banner based on this content:\n\n"
            f"{truncated_topic}\n\n"
            f"Generate a prompt that will create a professional, visually striking banner image "
            f"that captures the essence of this content."
        )
        
        print(f"Calling Gemini API for prompt enhancement...")
        response = prompt_enhancer_model.generate_content(enhanced_request)
        
        if not response or not hasattr(response, 'text') or not response.text:
            print("Warning: Empty response from Gemini API")
            raise Exception("Empty response from prompt enhancer")
        
        enhanced_prompt = response.text.strip()
        
        if not enhanced_prompt:
            print("Warning: Enhanced prompt is empty after processing")
            raise Exception("Enhanced prompt is empty")
        
        # Clean up any markdown formatting that might have been added
        enhanced_prompt = enhanced_prompt.lstrip("```").rstrip("```")
        enhanced_prompt = enhanced_prompt.replace("prompt:", "").replace("Prompt:", "").strip()
        
        # Remove any leading/trailing quotes
        if enhanced_prompt.startswith('"') and enhanced_prompt.endswith('"'):
            enhanced_prompt = enhanced_prompt[1:-1]
        if enhanced_prompt.startswith("'") and enhanced_prompt.endswith("'"):
            enhanced_prompt = enhanced_prompt[1:-1]
        
        # Ensure banner dimensions are mentioned in the prompt
        if "banner" not in enhanced_prompt.lower() and "1500x500" not in enhanced_prompt.lower():
            enhanced_prompt = f"{enhanced_prompt}, X/Twitter banner format, 3:1 aspect ratio, 1500x500 pixels, wide horizontal composition"
        
        print(f"Enhanced prompt generated successfully, length: {len(enhanced_prompt)}")
        print(f"Enhanced prompt preview: {enhanced_prompt[:200]}...")
        return enhanced_prompt
    except Exception as e:
        print(f"Error enhancing prompt: {e}")
        import traceback
        traceback.print_exc()
        raise  # Re-raise so the calling function can handle it

# --- 5. FASTAPI APP ---

app = FastAPI(title="Elsa AI Image Generator")

class ChatRequest(BaseModel):
    prompt: str

@app.get("/", response_class=HTMLResponse)
async def get_chat_interface():
    """Serve the chat interface HTML."""
    html_path = os.path.join(os.path.dirname(__file__), "chat_interface.html")
    with open(html_path, "r") as f:
        return HTMLResponse(content=f.read())

# This is your existing endpoint for regular messages
@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Handle chat requests and generate images/text."""
    try:
        tool_to_use, task_prompt = get_agent_decision(request.prompt)
        
        if tool_to_use == 'chibi':
            result = generate_chibi_image(task_prompt)
        elif tool_to_use == 'robot':
            result = generate_robot_image(task_prompt)
        elif tool_to_use == 'diagram':
            result = generate_diagram_code(task_prompt)
        elif tool_to_use == 'general':
            result = generate_general_image(task_prompt)
        else:
            result = {"success": False, "type": "error", "message": "Unknown tool selected"}
        
        return JSONResponse(content=result)
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "type": "error", "message": f"Server error: {str(e)}"})

# --- NEW: API ENDPOINT FOR THE ENHANCE BUTTON ---
@app.post("/api/enhance_and_generate")
async def enhance_and_generate(request: ChatRequest):
    """
    Handles requests from the 'Enhance' button.
    1. Enhances the user's simple prompt.
    2. Decides which tool to use for the *enhanced* prompt.
    3. Generates the final image.
    """
    try:
        print(f"[ENHANCE] Received request with prompt length: {len(request.prompt)}")
        
        # Step 1: Enhance the user's simple idea
        try:
            enhanced_prompt = enhance_prompt(request.prompt)
            if not enhanced_prompt or len(enhanced_prompt.strip()) == 0:
                print(f"[ENHANCE] Enhanced prompt is empty, using original prompt")
                enhanced_prompt = request.prompt
            print(f"[ENHANCE] Enhanced prompt generated, length: {len(enhanced_prompt)}")
        except Exception as e:
            print(f"[ENHANCE] Error in enhance_prompt function: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to using original prompt if enhancement fails
            print(f"[ENHANCE] Falling back to original prompt")
            enhanced_prompt = request.prompt
        
        # Step 2: Decide which tool to use for the *new* enhanced prompt
        try:
            tool_to_use, task_prompt = get_agent_decision(enhanced_prompt)
            print(f"[ENHANCE] Tool selected: {tool_to_use}")
        except Exception as e:
            print(f"[ENHANCE] Error in get_agent_decision: {e}")
            import traceback
            traceback.print_exc()
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "type": "error",
                    "message": f"Error deciding tool: {str(e)}"
                }
            )
        
        # Step 3: Call the chosen tool with the enhanced prompt
        try:
            if tool_to_use == 'chibi':
                result = generate_chibi_image(task_prompt)
            elif tool_to_use == 'robot':
                result = generate_robot_image(task_prompt)
            elif tool_to_use == 'diagram':
                result = generate_diagram_code(task_prompt)
            elif tool_to_use == 'general':
                result = generate_general_image(task_prompt)
            else:
                result = {"success": False, "type": "error", "message": "Unknown tool selected"}
            
            print(f"[ENHANCE] Generation result success: {result.get('success', False)}")
        except Exception as e:
            print(f"[ENHANCE] Error in image generation: {e}")
            import traceback
            traceback.print_exc()
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "type": "error",
                    "message": f"Error generating image: {str(e)}"
                }
            )
        
        # Return the final result, but also include the enhanced prompt so the user sees it
        if result.get("success", False):
            result["enhanced_prompt"] = enhanced_prompt
        
        return JSONResponse(content=result)

    except Exception as e:
        print(f"[ENHANCE] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "type": "error",
                "message": f"Server error: {str(e)}"
            }
        )


if __name__ == "__main__":
    import uvicorn
    # Make sure to set the GOOGLE_APPLICATION_CREDENTIALS environment variable!
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        print("="*50)
        print("ERROR: GOOGLE_APPLICATION_CREDENTIALS is not set.")
        print(f"Please run this in your terminal before starting:")
        print(f"export GOOGLE_APPLICATION_CREDENTIALS=\"{os.path.join(os.path.dirname(__file__), 'elsaugc-6755d1abf892.json')}\"")
        print("="*50)
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000)