#  Install package
```
pip install -r requirements.txt
```

# Make sure placing stl files
You should locate stl files under low_cost_robot/assets/
I didn't add *.stl files under that directory because sti file has
relatively huge file size.
```
low_cost_robot/
├─ assets/
│  ├─ *.stl
    ...
```

# Run mujoco simulation in mac OS
- use `mjpython` binary file instead of `python3`
- `mjpython` is located under the package `mujoco`


# 파일 설명
### get_angle_from_aist
1. aist_plusplus의 키포인트 데이터셋이 하위폴더로 있어야함. 
    ```python
    folder_path = 'aist_plusplus_final/keypoints3d'
    file_name = 'gBR_sBM_cAll_d04_mBR1_ch01.pkl'
    ```
2. aist_angles 하위 폴더가 존재해야함

### run.ipynb
1. aist_angles 폴더의 치환 각도들을 로디암
2. 현재 맵핑 상황은 다음과 같으며 일부는 맵핑되어있지 않음
    ```python
    data.qpos[0] = toe_angle
    data.qpos[1] = lower_body_angle
    data.qpos[2] = upper_body_angle
    data.qpos[4] = shoulder_angle
    data.qpos[5] = np.pi/2
    ```