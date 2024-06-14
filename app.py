from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
import cv2
import base64
from moviepy.editor import VideoFileClip
import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI
from dotenv import load_dotenv
import time

load_dotenv()

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['VIDEO_FOLDER'] = os.path.join('static', 'video')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['VIDEO_FOLDER'], exist_ok=True)

api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
model = os.getenv("DEPLOYMENT_NAME")
speech_key = os.getenv("AZURE_SPEECH_KEY")
speech_region = os.getenv("AZURE_SPEECH_REGION")

client = AzureOpenAI(
    api_key=api_key,
    api_version="2024-02-01",
    azure_endpoint=azure_endpoint
)

def process_video(video_path, seconds_per_frame=2):
    base64Frames = []
    base_video_path, _ = os.path.splitext(video_path)

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps * seconds_per_frame)
    curr_frame = 0

    while curr_frame < total_frames - 1 and len(base64Frames) < 18:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip
    video.release()

    audio_path = f"{base_video_path}.wav"
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, codec="pcm_s16le")
    clip.audio.close()
    clip.close()

    return base64Frames, audio_path

def transcribeaudio(audio_path):
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    transcription = []

    def result_callback(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            transcription.append(evt.result.text)
        elif evt.result.reason == speechsdk.ResultReason.NoMatch:
            transcription.append("No speech could be recognized.")
        elif evt.result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = evt.result.cancellation_details
            transcription.append(f"Speech recognition canceled: {cancellation_details.reason}. Error: {cancellation_details.error_details}")

    done = False

    def stop_callback(evt):
        nonlocal done
        done = True

    speech_recognizer.recognized.connect(result_callback)
    speech_recognizer.session_stopped.connect(stop_callback)
    speech_recognizer.canceled.connect(stop_callback)

    speech_recognizer.start_continuous_recognition()
    while not done:
        time.sleep(0.5)

    return ' '.join(transcription)

def analyze_video(frames, transcription, model):
    system_prompt = """
        You are an expert in behavioral analysis with a focus on interpreting non-verbal and verbal cues to understand personality and behavior.
        Your task is to analyze a video and audio transcription of a person speaking. You will consider both the visual frames and the spoken content to provide a detailed assessment of the individual's behavior and personality.
        Give points abput the person's Confidence and Demeanor,Communication Style,Behavioral Insights,Emotional State and Overall Impressions.
    """
    
    messages = [
        {"role": "system", "content":system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": "These are the frames from the video."},
            *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, frames),
            {"type": "text", "text": f"The audio transcription is: {transcription}"}
        ]}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(video_path)

            frames, audio_path = process_video(video_path)

            transcription = transcribeaudio(audio_path)

            analysis = analyze_video(frames, transcription, model)

            return render_template('index.html', video_file=file.filename, transcription=transcription, analysis=analysis)

    return render_template('index.html', video_file=None, transcription=None, analysis=None)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
