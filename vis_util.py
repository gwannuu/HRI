import pickle
from enum import Enum

class Joint(Enum):
    NOSE = 0
    L_EYE = 1
    R_EYE = 2
    L_EAR = 3
    R_EAR = 4
    L_SHOLDER = 5
    R_SHOLDER = 6
    L_ELBOW = 7
    R_ELBOW = 8
    L_WRIST = 9
    R_WRIST = 10
    L_HIP = 11
    R_HIP = 12
    L_KNEE = 13
    R_KNEE = 14
    L_ANKLE = 15
    R_ANKLE = 16


# COCO 포맷의 키포인트 연결 정의
connections = [
    # (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (Joint.NOSE, Joint.L_EYE),
    (Joint.NOSE, Joint.R_EYE),
    (Joint.L_EYE, Joint.L_EAR),
    (Joint.R_EYE, Joint.R_EAR),

    # (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (Joint.L_SHOLDER, Joint.R_SHOLDER),
    (Joint.L_SHOLDER, Joint.L_ELBOW),
    (Joint.L_ELBOW, Joint.L_WRIST),
    (Joint.R_SHOLDER, Joint.R_ELBOW),
    (Joint.R_ELBOW, Joint.R_WRIST),
    
    # (11, 12), (5, 11), (6, 12),  # Torso
    (Joint.L_HIP, Joint.R_HIP),
    (Joint.L_SHOLDER, Joint.L_HIP),
    (Joint.R_SHOLDER, Joint.R_HIP),

    # (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    (Joint.L_HIP, Joint.L_KNEE),
    (Joint.L_KNEE, Joint.L_ANKLE),
    (Joint.R_HIP, Joint.R_KNEE),
    (Joint.R_KNEE, Joint.R_ANKLE),
]
connections = [(jf.value, jt.value) for (jf, jt) in connections]

connection_info = {
    "head": [
        (Joint.NOSE, Joint.L_EYE),
        (Joint.NOSE, Joint.R_EYE),
        (Joint.L_EYE, Joint.L_EAR),
        (Joint.R_EYE, Joint.R_EAR),
    ],
    "torso": [
        (Joint.L_SHOLDER, Joint.R_SHOLDER),
        (Joint.L_HIP, Joint.R_HIP),
        (Joint.L_SHOLDER, Joint.L_HIP),
        (Joint.R_SHOLDER, Joint.R_HIP),
    ],
    "left_arm": [
        (Joint.L_SHOLDER, Joint.L_ELBOW),
        (Joint.L_ELBOW, Joint.L_WRIST),
    ],
    "right_arm": [
        (Joint.R_SHOLDER, Joint.R_ELBOW),
        (Joint.R_ELBOW, Joint.R_WRIST),
    ],
    "left_leg": [
        (Joint.L_HIP, Joint.L_KNEE),
        (Joint.L_KNEE, Joint.L_ANKLE),
    ],
    "right_leg": [
        (Joint.R_HIP, Joint.R_KNEE),
        (Joint.R_KNEE, Joint.R_ANKLE),
    ]
}

def set_part_color(link):
    for idx, (k, vs) in enumerate(connection_info.items()):
        for v in vs:
            if v[0].value == link[0] and v[1].value == link[1]:
                return idx, k
            if v[1].value == link[0] and v[0].value == link[1]:
                return idx, k
    raise Exception()



class SMPL_JOINT(Enum):
    ROOT = 0
    LHIP = 1
    RHIP = 2
    BELLY = 3
    LKNEE = 4
    RKNEE = 5
    SPINE = 6
    LANKLE = 7
    RANKLE = 8
    CHEST = 9
    LTOES = 10
    RTOES = 11
    NECK = 12
    LINSHOULDER = 13
    RINSHOULDER = 14
    HEAD = 15
    LSHOULDER = 16
    RSHOULDER = 17
    LELBOW = 18
    RELBOW = 19
    LWRIST = 20
    RWRIST = 21
    LHAND = 22
    RHAND = 23

smpl_connection_info = {
    "head": [
        (SMPL_JOINT.HEAD.value, SMPL_JOINT.NECK.value),
        (SMPL_JOINT.NECK.value, SMPL_JOINT.CHEST.value),
    ],
    "torso": [
        (SMPL_JOINT.CHEST.value, SMPL_JOINT.RINSHOULDER.value),
        (SMPL_JOINT.RINSHOULDER.value, SMPL_JOINT.RSHOULDER.value),
        (SMPL_JOINT.CHEST.value, SMPL_JOINT.LINSHOULDER.value),
        (SMPL_JOINT.LINSHOULDER.value, SMPL_JOINT.LSHOULDER.value),
        (SMPL_JOINT.CHEST.value, SMPL_JOINT.SPINE.value),
        (SMPL_JOINT.SPINE.value, SMPL_JOINT.BELLY.value),
        (SMPL_JOINT.BELLY.value, SMPL_JOINT.ROOT),
    ],
    "left_arm": [
        (SMPL_JOINT.LSHOULDER.value, SMPL_JOINT.LELBOW.value),
        (SMPL_JOINT.LELBOW.value, SMPL_JOINT.LWRIST.value),
        (SMPL_JOINT.LWRIST.value, SMPL_JOINT.LHAND.value),
    ],
    "right_arm": [
        (SMPL_JOINT.RSHOULDER.value, SMPL_JOINT.RELBOW.value),
        (SMPL_JOINT.RELBOW.value, SMPL_JOINT.RWRIST.value),
        (SMPL_JOINT.RWRIST.value, SMPL_JOINT.RHAND.value),
    ],
    "left_leg": [
        (SMPL_JOINT.ROOT.value, SMPL_JOINT.LHIP.value),
        (SMPL_JOINT.LHIP.value, SMPL_JOINT.LKNEE.value),
        (SMPL_JOINT.LKNEE.value, SMPL_JOINT.LANKLE.value),
        (SMPL_JOINT.LANKLE.value, SMPL_JOINT.LTOES.value),
    ],
    "right_leg": [
        (SMPL_JOINT.ROOT.value, SMPL_JOINT.RHIP.value),
        (SMPL_JOINT.RHIP.value, SMPL_JOINT.RKNEE.value),
        (SMPL_JOINT.RKNEE.value, SMPL_JOINT.RANKLE.value),
        (SMPL_JOINT.RANKLE.value, SMPL_JOINT.RTOES.value),
    ],
}