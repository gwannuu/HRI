from interface.interface import input_text_test,conversation, classification_songname
from interface.prompts import SPECIAL_TOKENS
from robot_interaction.sync_music_robot_v2 import run_robot

if __name__=="__main__":
    # final_sentence = input_text_test()  # 텍스트 입력을 통한 대화
    final_sentence = conversation()
    print(f"확인용: {final_sentence}")
    song_name = classification_songname(final_sentence)  # 노래 분류
    
    
    if SPECIAL_TOKENS["DanceForUser"] in final_sentence:
        run_robot(
            music_path = f"music/{song_name}/test_7.wav",
            dance_path = f"motions/{song_name}/6_joints_robot/test_7.pkl",
            withme=False,
        )
    elif SPECIAL_TOKENS["DanceWithUser"] in final_sentence:
        run_robot(
            music_path = f"music/{song_name}/test_7.wav",
            dance_path = f"motions/{song_name}/6_joints_robot/test_7.pkl",
            withme=True,
        )
    else:
        print("잘못된 입력입니다.")
    
    # run_robot(
    #     music_path = "music/Papa Nugs - Hyperdrive.mp3",
    #     dance_path = "motions/We_Will_Rock_You/6_joints_robot/test_7.pkl",
    #     withme=True,
    # )
        
