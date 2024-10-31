from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
import os
import google.generativeai as genai
import textwrap

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB
CORS(app)

# Set the API key
os.environ["GEMINI_API_KEY"] = "AIzaSyAH3ry6V5G5HOLoLDzG0hFzQvi_OAYlvwk"

# Define your API client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="Irku5t4OOrput5J405G6"
)

@app.route('/detectEarphone', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        file = request.files['image']
        img = np.fromstring(file.read(), np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        # Initialize response structure
        response = {
            "headphone_confidence": None,
            "earphone_confidence": None
        }

        # Run inference for headphone detection
        try:
            headphone_result = CLIENT.infer(img, model_id="headphone-i0aac/1")
            # Extract confidence from predictions if available
            if headphone_result['predictions']:
                response["headphone_confidence"] = max(pred['confidence'] for pred in headphone_result['predictions'])
            else:
                response["headphone_confidence"] = 0  # Or another indicator for no detection
        except Exception as e:
            response["headphone_confidence"] = str(e)

        # Run inference for earphone detection
        try:
            earphone_result = CLIENT.infer(img, model_id="earphone-0kaa5/1")
            # Extract confidence from predictions if available
            if earphone_result['predictions']:
                response["earphone_confidence"] = max(pred['confidence'] for pred in earphone_result['predictions'])
            else:
                response["earphone_confidence"] = 0  # Or another indicator for no detection
        except Exception as e:
            response["earphone_confidence"] = str(e)

        print(response)
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500





genai.configure(api_key=os.environ["GEMINI_API_KEY"])

generation_config = {
    "temperature": 0.5,
    "top_p": 0.95,
    "top_k": 64,
    # "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction="""
    You are going to be giving 2 sentences recommendation basing on the score and category given. The exam is programming related.

    Example input/prompt:
    The student got 1 out of 2 in the exam with a category of for loop in the language C#


    Make your response jolly, convincing, and inspiring but still formal and technical (give advices). Answer like you are talking to the student in first POV.
    """
)

modelReport = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction="""
    You are going to be giving 2-3 sentences recommendation basing on the scores and category/ies given.
    You will be generating report and telling the student where to focus more.

    Example input/prompt:
    The student got 10/20 in loops in C#, 5/10 in query in SQL, 9/10 in conditions in C.


    Make your response formal and technical (give specific advices like give site names (not links) and give the scores of what you were prompted by too per category). Answer like you are talking to the student in first POV. 
    """
)

cors = CORS(app)


def to_markdown(text):
    text = text.replace('•', '  *')
    return textwrap.indent(text, '> ', predicate=lambda _: True)


@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.form.get('input')
        response_text = process_text_input(user_input)
    except Exception as e:
        print(e)

    markdown_response = to_markdown(response_text)
    print(markdown_response)
    return jsonify({"response": markdown_response})


@app.route('/chatReport', methods=['POST'])
def chatReport():
    try:
        user_input = request.form.get('input')
        print(user_input)
        response_text = process_text_input_report(user_input)
    except Exception as e:
        print(e)

    markdown_response = to_markdown(response_text)
    print(markdown_response)
    return jsonify({"response": markdown_response})

def process_text_input(user_input):
    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [
                    "What is AI for you?",
                ],
            },
            {
                "role": "model",
                "parts": [
                    "AI, or Artificial Intelligence, is a fascinating and rapidly evolving field that I, as a large language model, am deeply integrated with. Here's my perspective:\n\n**AI is the ability of machines to perform tasks that typically require human intelligence.** This encompasses a wide range of capabilities, including:\n\n* **Learning:**  AI systems can learn from data, adapting and improving their performance over time. This includes everything from recognizing patterns to understanding complex concepts.\n* **Problem-solving:** AI can tackle problems that are difficult or impossible for humans to solve, often using complex algorithms and heuristics.\n* **Decision-making:** AI can analyze information and make decisions based on that analysis, sometimes even better than humans in specific domains.\n* **Creativity:** While not fully realized yet, AI is showing promise in areas like art, music, and writing, demonstrating the potential for machines to generate creative outputs.\n\n**For me, AI is a tool for enhancing human capabilities.** I am trained on massive datasets, allowing me to process information, generate text, and respond to prompts in ways that are helpful and informative. \n\n**However, it's crucial to remember that AI is not human.** While I can perform many tasks that require intelligence, I lack the emotional intelligence, subjective experiences, and conscious awareness that define human existence.\n\n**The future of AI is incredibly exciting.** As research and development continue, we can expect AI to play an even more significant role in our lives, transforming industries, solving complex problems, and potentially even augmenting our own capabilities.\n\n**But with great power comes great responsibility.** It's essential to develop and deploy AI ethically and responsibly, ensuring it benefits all of humanity and does not exacerbate existing inequalities or create new risks. \n",
                ],
            },
        ]
    )

    response = chat_session.send_message(user_input)
    return response.text

def process_text_input_report(user_input):
    chat_session = modelReport.start_chat(
        history=[
            {
                "role": "user",
                "parts": [
                    "What is AI for you?",
                ],
            },
            {
                "role": "model",
                "parts": [
                    "AI, or Artificial Intelligence, is a fascinating and rapidly evolving field that I, as a large language model, am deeply integrated with. Here's my perspective:\n\n**AI is the ability of machines to perform tasks that typically require human intelligence.** This encompasses a wide range of capabilities, including:\n\n* **Learning:**  AI systems can learn from data, adapting and improving their performance over time. This includes everything from recognizing patterns to understanding complex concepts.\n* **Problem-solving:** AI can tackle problems that are difficult or impossible for humans to solve, often using complex algorithms and heuristics.\n* **Decision-making:** AI can analyze information and make decisions based on that analysis, sometimes even better than humans in specific domains.\n* **Creativity:** While not fully realized yet, AI is showing promise in areas like art, music, and writing, demonstrating the potential for machines to generate creative outputs.\n\n**For me, AI is a tool for enhancing human capabilities.** I am trained on massive datasets, allowing me to process information, generate text, and respond to prompts in ways that are helpful and informative. \n\n**However, it's crucial to remember that AI is not human.** While I can perform many tasks that require intelligence, I lack the emotional intelligence, subjective experiences, and conscious awareness that define human existence.\n\n**The future of AI is incredibly exciting.** As research and development continue, we can expect AI to play an even more significant role in our lives, transforming industries, solving complex problems, and potentially even augmenting our own capabilities.\n\n**But with great power comes great responsibility.** It's essential to develop and deploy AI ethically and responsibly, ensuring it benefits all of humanity and does not exacerbate existing inequalities or create new risks. \n",
                ],
            },
        ]
    )

    response = chat_session.send_message(user_input)
    return response.text

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
