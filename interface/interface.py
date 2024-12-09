import openai
from openai import OpenAI
import sounddevice as sd
import soundfile as sf
import numpy as np
import queue
import time
import tempfile
import os

# 필요한 라이브러리를 설치해야 합니다.
# pip install openai sounddevice soundfile numpy gtts playsound

client = OpenAI(api_key="")

# 녹음 설정
samplerate = 16000  # Whisper는 16kHz를 권장합니다.
channels = 1  # 모노 채널
dtype = 'int16'  # 녹음 데이터 타입

def record_audio(duration=5):
    """
    사용자의 음성을 녹음합니다.

    Args:
        duration (int): 녹음 시간(초)

    Returns:
        str: 오디오 파일의 경로
    #todo:
    버튼 토글을 통한 녹음
    """
    print("질문을 말씀해주세요... ({}초 동안 녹음)".format(duration))
    q = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status)  # 오류 상태를 출력
        q.put(indata.copy())

    with sd.InputStream(samplerate=samplerate, channels=channels, dtype=dtype, callback=callback):
        frames = []
        start_time = time.time()
        while time.time() - start_time < duration:
            if not q.empty():
                frames.append(q.get())

    if frames:  # 프레임이 비어있지 않을 때만
        audio_data = np.concatenate(frames)
        temp_dir = tempfile.gettempdir()
        filename = "input.wav"
        sf.write(filename, audio_data, samplerate)
        print("녹음이 완료되었습니다. 파일 경로: {}".format(filename))
        return filename
    else:
        print("녹음된 데이터가 없습니다.")
        return None

def transcribe_audio(audio_file_path):
    """
    녹음된 오디오를 텍스트로 변환합니다.

    Args:
        audio_file (str): 오디오 파일의 경로

    Returns:
        str: 변환된 텍스트
    """
    audio_file = open(audio_file_path, "rb")
    print("음성을 텍스트로 변환 중입니다...")
    transcription = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file
    )
    
    text = transcription.text.strip()
    print("인식된 질문: {}".format(text))
    return text

def generate_response(conversation):
    """
    대화 내역을 GPT 모델에 전달하여 응답을 생성합니다.

    Args:
        conversation (list): 이전 대화 내역이 담긴 메시지 리스트

    Returns:
        str: GPT 모델의 응답
    """
    print("GPT 모델이 응답 중입니다...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # 또는 'gpt-4' 등 다른 모델로 변경 가능
        messages=conversation,
        max_tokens=512,
    )
    answer = response.choices[0].message.content.strip()
    print("GPT의 응답: {}".format(answer))
    return answer

def speak_text(text):
    """
    텍스트를 음성으로 변환하여 출력합니다.

    Args:
        text (str): 출력할 텍스트

    """
    print("응답을 음성으로 출력합니다...")
    
    speech_file_path = "speech.mp3"
    response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input=text
    )
    
    response.stream_to_file(speech_file_path)
    # os.remove(speech_file_path)  # 재생 후 임시 파일 삭제
    

# 사용 예시
def main():
    """
    메인 함수로, 대화 루프를 관리합니다.
    이전 대화 내용을 유지하여 대화의 맥락을 반영합니다.
    """
    print("음성 대화 애플리케이션을 시작합니다. '종료'라고 말씀하시면 대화가 종료됩니다.")
    conversation = [] # -> 매 턴 conversationreset 하고 작업 판단하게끔 설정
    while True:
        audio_file = record_audio(duration=5)
        input_text = transcribe_audio(audio_file)
        # os.remove(audio_file)  # 임시 파일 삭제

        if any(trigger in input_text.lower() for trigger in ['종료', 'bye', '끝']):
            print("대화를 종료합니다.")
            speak_text("대화를 종료합니다.")
            break

        # 사용자 메시지를 대화 내역에 추가
        conversation.append({"role": "user", "content": input_text})

        # GPT 모델로부터 응답 생성
        response_text = generate_response(conversation)

        # GPT의 응답을 대화 내역에 추가
        conversation.append({"role": "assistant", "content": response_text})

        # 응답을 음성으로 출력
        speak_text(response_text)

if __name__ == "__main__":
    main()