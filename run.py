from robot_interaction.sync_music_robot_v2 import run_robot

if __name__=="__main__":
    run_robot(
        music_path = "music/Papa Nugs - Hyperdrive.mp3",
        dance_path = "motions/APT/6_joints_robot/test_7.pkl",
        withme=False,
    )