
def convert_to_mirror_list(minds):
    mirror_lists = [
        minds["sideneg_obs_inds"] + minds["neg_obs_inds"],
        minds["right_obs_inds"],
        minds["left_obs_inds"],
        minds["neg_act_inds"] + minds["sideneg_act_inds"],
        minds["right_act_inds"],
        minds["left_act_inds"],
    ]
    return mirror_lists

MIRROR_DICTS={
    "Walked2d":{
            #### observation:
            #"com_obs_inds": [0, 8, 9],
            #"neg_obs_inds": [1, 10],
            "com_obs_inds": [0,1, 8, 9,10],
            "neg_obs_inds": [],
            "right_obs_inds": list(range(2, 5)) + list(range(11, 14)),
            "left_obs_inds": list(range(5, 8)) + list(range(14, 17)),
            "sideneg_obs_inds": [],
            #### action:
            "com_act_inds": [],
            "neg_act_inds": [],
            "sideneg_act_inds": [],
            "right_act_inds": list(range(0, 3)),
            "left_act_inds": list(range(3, 6)),
        },
    "HalfCheetah":{
            #### observation:
            "com_obs_inds": [0,1, 8, 9,10],
            "neg_obs_inds": [],
            "right_obs_inds": list(range(5, 8)) + list(range(14, 17)),
            "left_obs_inds": list(range(2, 5)) + list(range(11, 14)),
            "sideneg_obs_inds": [list(range(2, 8)) + list(range(11, 17))],#list(range(2, 8)) + list(range(11, 17)),#[],
            #### action:
            "com_act_inds": [],
            "neg_act_inds": [],
            "sideneg_act_inds": [list(range(0, 6))],#list(range(0, 6)),#[],
            "right_act_inds": list(range(3, 6)),
            "left_act_inds": list(range(0, 3)),
        },

    # BL:2,3,4  /17,18,19
    # FL:5,6,7 /20,21,22
    # BR:8,9,10 /23,24,25
    # FR:11,12,13 /26,27,28
    # "FullCheetah_gallop": {
    #     #### observation:
    #     "com_obs_inds": [0, 1,14,15,16],
    #     "neg_obs_inds": [],
    #     "right_obs_inds": list(range(5, 8)) + list(range(11, 14))+list(range(20, 23))+list(range(26, 29)),
    #     "left_obs_inds": list(range(2, 5)) + list(range(8, 11))+list(range(17, 20))+list(range(23, 26)),
    #     "sideneg_obs_inds": [],  # list(range(2, 8)) + list(range(11, 17)),#[],
    #     #### action:
    #     "com_act_inds": [],
    #     "neg_act_inds": [],
    #     "sideneg_act_inds": [],  # list(range(0, 6)),#[],
    #     "right_act_inds": list(range(3, 6))+list(range(9, 12)),
    #     "left_act_inds": list(range(0, 3))+list(range(6, 9)),
    # },
    "FullCheetah_lr": {#FullCheetah_gallop
        #### observation:
        "com_obs_inds": [0, 1, 14, 15, 16],
        "neg_obs_inds": [],
        "right_obs_inds":  list(range(8, 14)) + list(range(23, 29)),
        "left_obs_inds": list(range(2, 8)) + list(range(17, 23)),
        "sideneg_obs_inds": [],  # list(range(2, 8)) + list(range(11, 17)),#[],
        #### action:
        "com_act_inds": [],
        "neg_act_inds": [],
        "sideneg_act_inds": [],  # list(range(0, 6)),#[],
        "right_act_inds": list(range(6, 12)) ,
        "left_act_inds": list(range(0, 6)) ,
    },
    #BL:2,3,4  /17,18,19-> FR
    #FL:5,6,7 /20,21,22-> BR
    #BR:8,9,10 /23,24,25 -> FL
    #FR:11,12,13 /26,27,28->BL
    "FullCheetah_gallopFBLR": {
        #### observation:
        "com_obs_inds": [0, 1, 14, 15, 16],
        "neg_obs_inds": [],
        "right_obs_inds": list(range(2, 14)) + list(range(17, 29)),
        "left_obs_inds": list(range(11, 14)) +list(range(8, 11)) +list(range(5, 8))+list(range(2, 5)) +list(range(26, 29)) +list(range(23, 26)) +list(range(20, 23))+list(range(17, 20)) ,
        "sideneg_obs_inds": [],  # list(range(2, 8)) + list(range(11, 17)),#[],
        #### action:
        "com_act_inds": [],
        "neg_act_inds": [],
        "sideneg_act_inds": [],  # list(range(0, 6)),#[],
        "right_act_inds": list(range(0, 12)),
        "left_act_inds": list(range(9, 12)) + list(range(6, 9))+ list(range(3, 6))+ list(range(0, 3)),
    },

    "FullCheetah_fb": {#FullCheetah_trot
        #### observation:
        "com_obs_inds": [0, 1, 14, 15, 16],
        "neg_obs_inds": [],
        "right_obs_inds": list(range(5, 8))  + list(range(20, 23))+ list(range(11, 14)) + list(range(26, 29)),#fL+bR=right


        "left_obs_inds": list(range(2, 5))  + list(range(17, 20))+ list(range(8, 11)) + list(range(23, 26)),#fR+bL=left

        "sideneg_obs_inds": [],  # list(range(2, 8)) + list(range(11, 17)),#[],
        #### action:
        "com_act_inds": [],
        "neg_act_inds": [],
        "sideneg_act_inds": [],  # list(range(0, 6)),#[],
        "right_act_inds": list(range(3, 6)) + list(range(9, 12)),
        "left_act_inds": list(range(0, 3))+ list(range(6, 9)) ,
    },

}
"""
#x and/or y shoud be 
0 rootz     slider      C  
1 rooty     hinge       C
2 bthigh    hinge       L
3 bshin     hinge       L
4 bfoot     hinge       L
5 fthigh    hinge       R
6 fshin     hinge       R
7 ffoot     hinge       R
8 rootx     slider      C
9 rootz     slider      C
10 rooty     hinge      C
11 bthigh    hinge      L
12 bshin     hinge      L
13 bfoot     hinge      L
14 fthigh    hinge      R
15 fshin     hinge      R
16 ffoot     hinge      R
"""