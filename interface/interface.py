import openai
from openai import OpenAI
import sounddevice as sd
import soundfile as sf
import numpy as np
import queue
import sys
import time
import tempfile
import os
from interface.prompts import SYSTEM_PROMPT, SPECIAL_TOKENS, CLASSIFICATION_PROMPT
# import keyboard

# 필요한 라이브러리를 설치해야 합니다.
# pip install openai sounddevice soundfile numpy gtts playsound

client = OpenAI()

# 녹음 설정
dtype = 'int16'  # 녹음 데이터 타입
sd.default.samplerate = 16000 # Whisper는 16kHz를 권장합니다.
sd.default.channels = 2

def record_audio():
    q = queue.Queue()
    filename = "input.mp3"
    channels = 1
    samplerate = 16000
    
    def callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())
        
    print("질문을 말씀해주세요...")
    if os.path.exists(filename):
        os.remove(filename)
    try:
        # Make sure the file is opened before recording anything:
        with sf.SoundFile(filename, mode='x',channels=channels,samplerate=samplerate) as file:
            with sd.InputStream(device=1,channels=channels, samplerate=samplerate,callback=callback):
                # print('#' * 20)
                # print('press "q" to stop the recording')
                # print('#' * 20)
                # while not keyboard.is_pressed('q'):
                #     file.write(q.get())
    except KeyboardInterrupt:
        print('\nRecording finished: ' + repr(filename))
    except Exception as e:
        print(type(e).__name__ + ': ' + str(e))

def transcribe_audio(audio_file_path=None):
    """
    녹음된 오디오를 텍스트로 변환합니다.

    Args:
        audio_file (str): 오디오 파일의 경로

    Returns:
        str: 변환된 텍스트
    """
    if audio_file_path is None:
        audio_file_path = "input.mp3"
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
    speech_file_path = "speech.mp3"
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="alloy",
        input=text
    ) as response:
        response.stream_to_file(speech_file_path)

    data, samplerate = sf.read(speech_file_path)  # 음성 파일 읽기
    sd.play(data, samplerate)  # 음성 파일 재생
    sd.wait()  # 재생이 끝날 때까지 대기
    os.remove(speech_file_path)  # 재생 후 임시 파일 삭제


def input_text_test():
    """
    텍스트 입력을 통해 테스트합니다.
    """
    conversation = [{"role":"system", "content": SYSTEM_PROMPT}]
    while True:
        input_text = input("질문을 입력하세요 ('종료'라고 입력하면 대화가 종료됩니다): ")

        if any(trigger in input_text.lower() for trigger in ['종료', 'bye', '끝']):
            print("대화를 종료합니다.")
            # speak_text("대화를 종료합니다.")
            break

        # 사용자 메시지를 대화 내역에 추가
        conversation.append({"role": "user", "content": input_text})

        # GPT 모델로부터 응답 생성
        response_text = generate_response(conversation)

        # GPT의 응답을 대화 내역에 추가
        conversation.append({"role": "assistant", "content": response_text})

        # 응답을 음성으로 출력
        speak_text(response_text)
        
        if any(trigger in response_text for trigger in [SPECIAL_TOKENS["DanceForUser"], SPECIAL_TOKENS["DanceWithUser"]]):
            print("대화를 종료합니다.")
            # speak_text("대화를 종료합니다.")
            break
    return conversation[-1]["content"]

# 사용 예시
def conversation():
    """
    대화 루프를 관리합니다.
    이전 대화 내용을 유지하여 대화의 맥락을 반영합니다.
    """
    print("음성 대화 애플리케이션을 시작합니다. '종료'라고 말씀하시면 대화가 종료됩니다.")
    conversation = [{"role":"system", "content": SYSTEM_PROMPT}] # -> 매 턴 conversationreset 하고 작업 판단하게끔 설정
    while True:
        record_audio()
        input_text = transcribe_audio()
        # os.remove(audio_file)  # 임시 파일 삭제

        if any(trigger in input_text.lower() for trigger in ['종료', 'bye', '끝']):
            print("대화를 종료합니다.")
            # speak_text("대화를 종료합니다.")
            break

        # 사용자 메시지를 대화 내역에 추가
        conversation.append({"role": "user", "content": input_text})

        # GPT 모델로부터 응답 생성
        response_text = generate_response(conversation)

        # GPT의 응답을 대화 내역에 추가
        conversation.append({"role": "assistant", "content": response_text})

        # 응답을 음성으로 출력
        speak_text(response_text)
        
        if any(trigger in response_text for trigger in [SPECIAL_TOKENS["DanceForUser"], SPECIAL_TOKENS["DanceWithUser"]]):
            print("대화를 종료합니다.")
            # speak_text("대화를 종료합니다.")
            break
    return conversation[-1]["content"]

def classification_songname(text):
    """
    GPT 모델을 사용하여 텍스트를 분류합니다.
    """
    conversation = [{"role":"user", "content": CLASSIFICATION_PROMPT.format(text=text)}]

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # 또는 'gpt-4' 등 다른 모델로 변경 가능
        messages=conversation,
        max_tokens=20,
    )
    answer = response.choices[0].message.content.strip()
    print(f"변환된 song_name: {answer}")
    return answer.strip()

def dance_for_user():
    print("춤을 춰드릴게요")
    pass

def dance_with_user():
    print("함께 춤을 춰요")
    pass

def run_function(trigger):
    if SPECIAL_TOKENS["DanceForUser"] in trigger:
        dance_for_user()
    elif SPECIAL_TOKENS["DanceWithUser"] in trigger:
        dance_with_user()
    else:
        print("잘못된 입력입니다.")



if __name__ == "__main__":
    # final_sentence = conversation()  # 음성 입력을 통한 대화
    final_sentence = input_text_test()  # 텍스트 입력을 통한 대화
    song_path = classification_songname(final_sentence)  # 노래 분류
    run_function(final_sentence, song_path)  # 대화 결과에 따른 기능 실행
