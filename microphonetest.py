import whisper
import gradio as gr 
import librosa

model = whisper.load_model("base")

audio_input = gr.inputs.Audio(source="microphone", type="filepath")

def predict_audio(audio_file):
    # 오디오를 로드하고 스펙트로그램으로 변환합니다.
    audio, sr = librosa.load(audio_file)
    spectrogram = librosa.feature.melspectrogram(audio, sr=sr)
    
    # "whisper" 모델에 입력 스펙트로그램을 전달하여 예측을 수행합니다.
    prediction = model.predict(spectrogram)
    
    return prediction

# "gradio" 인터페이스를 생성합니다.
interface = gr.Interface(fn=predict_audio, inputs=audio_input, outputs="text")

# 인터페이스를 실행합니다.
interface.launch()