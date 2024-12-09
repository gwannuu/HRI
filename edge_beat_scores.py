import librosa
import numpy as np
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import argrelextrema
import scipy.signal as scisignal
import torch

def extract_music_beats(wav_file, duration, fps=60):
    """
    Extracts beat information from a .wav file and returns a one-hot encoded array
    that matches the specified duration and frames per second (fps).

    Args:
        wav_file (str): Path to the .wav file.
        duration (float): Total duration of the music in seconds.
        fps (int): Frames per second for the output beat representation.

    Returns:
        np.ndarray: One-hot encoded array of shape (num_frames,), where 1 indicates a beat.
    """
    # Load the audio file
    y, sr = librosa.load(wav_file, sr=None)
    
    # Detect beats
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    
    # Convert beat frames to time
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    # Create one-hot encoded array for beats
    num_frames = int(duration * fps)
    beat_onehot = np.zeros(num_frames, dtype=bool)
    beat_indices = np.round(beat_times * fps).astype(int)
    beat_indices = beat_indices[beat_indices < num_frames]  # Ensure indices are in range
    beat_onehot[beat_indices] = 1
    
    return beat_onehot

def load_motion(pkl_file):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    motion = data.get('full_pose')
    return motion # (seq_len, 24, 3)

# def eye(n, batch_shape):
#     iden = np.zeros(np.concatenate([batch_shape, [n, n]]))
#     iden[..., 0, 0] = 1.0
#     iden[..., 1, 1] = 1.0
#     iden[..., 2, 2] = 1.0
#     return iden

# def get_closest_rotmat(rotmats):
#     u, s, vh = np.linalg.svd(rotmats)
#     r_closest = np.matmul(u, vh)

#     # if the determinant of UV' is -1, we must flip the sign of the last column of u
#     det = np.linalg.det(r_closest)  # (..., )
#     iden = eye(3, det.shape)
#     iden[..., 2, 2] = np.sign(det)
#     r_closest = np.matmul(np.matmul(u, iden), vh)
#     return r_closest

# def recover_to_axis_angles(motion):
#     batch_size, seq_len, dim = motion.shape
#     assert dim == 225
#     transl = motion[:, :, 6:9]
#     rotmats = get_closest_rotmat(
#         np.reshape(motion[:, :, 9:], (batch_size, seq_len, 24, 3, 3))
#     )
#     axis_angles = R.from_matrix(
#         rotmats.reshape(-1, 3, 3)
#     ).as_rotvec().reshape(batch_size, seq_len, 24, 3)
#     return axis_angles, transl

# def recover_motion_to_keypoints(motion, smpl_model):
#     smpl_poses, smpl_trans = recover_to_axis_angles(motion)
#     smpl_poses = np.squeeze(smpl_poses, axis=0)  # (seq_len, 24, 3)
#     smpl_trans = np.squeeze(smpl_trans, axis=0)  # (seq_len, 3)
#     keypoints3d = smpl_model.forward(
#         global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
#         body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
#         transl=torch.from_numpy(smpl_trans).float(),
#     ).joints.detach().numpy()[:, :24, :]   # (seq_len, 24, 3)
#     return keypoints3d

def motion_peak_onehot(joints):
    velocity = np.zeros_like(joints, dtype=np.float32)
    velocity[1:] = joints[1:] - joints[:-1]
    velocity_norms = np.linalg.norm(velocity, axis=2)
    envelope = np.sum(velocity_norms, axis=1)  # (seq_len,)

    # Find local minima in velocity -- beats
    peak_idxs = scisignal.argrelextrema(envelope, np.less, axis=0, order=10) # 10 for 60FPS
    peak_onehot = np.zeros_like(envelope, dtype=bool)
    peak_onehot[peak_idxs] = 1
    return peak_onehot

def alignment_score(music_beats, motion_beats, sigma=3):
    if motion_beats.sum() == 0:
        return 0.0
    music_beat_idxs = np.where(music_beats)[0]
    motion_beat_idxs = np.where(motion_beats)[0]
    score_all = []
    for motion_beat_idx in motion_beat_idxs:
        dists = np.abs(music_beat_idxs - motion_beat_idx).astype(np.float32)
        ind = np.argmin(dists)
        score = np.exp(- dists[ind]**2 / 2 / sigma**2)
        score_all.append(score)
    return sum(score_all) / len(score_all)

# Paths to your files
wav_file = "./data/test_Twist_And_Shout.wav"
pkl_file = "./data/test_Twist_And_Shout.pkl"

duration = 30
fps = 60
# Step 1: Extract Music Beats
music_beats = extract_music_beats(wav_file, duration, fps)

# Step 2: Extract Motion Beats
motion_data = load_motion(pkl_file)  # Load motion data from .pkl
# motion_keypoints = recover_motion_to_keypoints(motion_data, smpl_model)  # SMPL required
# motion_beats = motion_peak_onehot(motion_keypoints)
motion_beats = motion_peak_onehot(motion_data)

# Step 3: Calculate Beat Score
beat_score = alignment_score(music_beats, motion_beats, sigma=3)
print(f"Beat Alignment Score: {beat_score}")
