from interface.interface import input_text_test,classification_songname
from interface.prompts import SPECIAL_TOKENS
from robot_interaction.sync_music_robot_v2 import run_robot

if __name__=="__main__":
    final_sentence = input_text_test()  # 텍스트 입력을 통한 대화
    song_path = classification_songname(final_sentence)  # 노래 분류
    
    # if SPECIAL_TOKENS["DanceForUser"]:
    #     run_robot(
    #         music_path = "music/Papa Nugs - Hyperdrive.mp3",
    #         dance_path = "motions/APT/6_joints_robot/test_7.pkl",
    #         withme=False,
    #     )
    # elif SPECIAL_TOKENS["DanceWithUser"] in trigger:
    #     run_robot(
    #         music_path = "music/Papa Nugs - Hyperdrive.mp3",
    #         dance_path = "motions/APT/6_joints_robot/test_7.pkl",
    #         withme=True,
    #     )
    # else:
    #     print("잘못된 입력입니다.")
        
