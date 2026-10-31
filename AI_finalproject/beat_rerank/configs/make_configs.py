"""Write the job lists used in this project (configs/*.json)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MUSIC = "data/aist_music"  # AIST Dance DB music (mp3)
OFFSET = 8.0  # every AIST excerpt is 8-18 s of the song

AIST_GENRES = {"BR": "break", "PO": "pop", "LO": "lock", "MH": "middle hip-hop", "LH": "LA style hip-hop",
               "HO": "house", "WA": "waack", "KR": "krump", "JS": "street jazz", "JB": "ballet jazz"}
# AIST++ cross-modal test songs (one per genre) and a second song per genre
TEST_SONG = {"BR": 0, "PO": 1, "LO": 2, "MH": 3, "LH": 4, "HO": 5, "WA": 0, "KR": 2, "JS": 3, "JB": 5}
SECOND_SONG = {"BR": 3, "PO": 4, "LO": 5, "MH": 0, "LH": 1, "HO": 2, "WA": 3, "KR": 5, "JS": 0, "JB": 2}


def dump(name, jobs):
    with open(os.path.join(HERE, name), "w") as f:
        json.dump({"jobs": jobs}, f, indent=1, ensure_ascii=False)
    print(name, len(jobs))


# ---------------------------------------------------------------- reproduction of the original project
repro = []
# text -> music + motion: the two prompts of the GPT-o3 experiment, the two prompts of the
# genre-classifier experiment, the two example styles of the GPT-3 experiment and three more
# descriptions, nine in total for the CLAP similarity matrix
t2mm = [
    ("latinpop_lock", "The audio is a Latin pop song.", "The style of the dance is lock."),
    ("indie_house", "The music is a mix of indie pop and indie rock.", "The genre of the dance is house."),
    ("pop_genre", "The music is pop genre.", "The style of the dance is pop."),
    ("disco_genre", "The music is disco genre.", "The style of the dance is lock."),
    ("dreamy_indiepop", "The audio is a dreamy indie pop song with jangly guitars, airy vocals, and shimmering synth pads.",
     "The style of the dance is street jazz."),
    ("blues_rock", "The audio is a gritty blues-rock track with overdriven guitar riffs, raw vocals, and a shuffling shuffle beat.",
     "The style of the dance is break."),
    ("classical_piano", "The audio is a calm classical piano piece.", "The style of the dance is ballet jazz."),
    ("hiphop_beat", "The audio is an energetic hip-hop beat with heavy bass and rap vocals.",
     "The style of the dance is middle hip-hop."),
    ("heavy_metal", "The audio is a fast heavy metal song with distorted guitars and pounding drums.",
     "The style of the dance is krump."),
]
for i, (k, mu, mo) in enumerate(t2mm):
    repro.append({"id": f"t2mm_{i}_{k}", "task": "mumo", "music_desc": mu, "motion_desc": mo})
# text + music -> motion: the four prompts on the slides plus five more AIST genres (nine motions,
# as in the Beat Alignment Score experiment)
tm2m = [("HO", "The genre of the dance is house."), ("JB", "The style of the dance is ballet jazz"),
        ("WA", "The genre of the dance is waack"), ("LO", "This is a lock style dance."),
        ("BR", "The style of the dance is break."), ("PO", "The style of the dance is pop."),
        ("KR", "The style of the dance is krump."), ("MH", "The style of the dance is middle hip-hop."),
        ("JS", "The style of the dance is street jazz.")]
for g, mo in tm2m:
    song = f"m{g}{TEST_SONG[g]}"
    repro.append({"id": f"tm2m_{song}", "task": "mo", "music_desc": "", "motion_desc": mo,
                  "music_path": f"{MUSIC}/{song}.mp3", "offset": OFFSET, "genre": AIST_GENRES[g]})
# text + motion -> music: the two style descriptions quoted in the report, on the AIST++ motion
# that ships with the UniMuMo repository
for k, mu in [("dreamy_indiepop", t2mm[4][1]), ("blues_rock", t2mm[5][1])]:
    repro.append({"id": f"mt2mu_{k}", "task": "mu", "music_desc": mu, "motion_desc": "",
                  "motion_path": "UniMuMo/data/motion/train/joint_vecs/gWA_sBM_cAll_d25_mWA0_ch01.npy",
                  "offset": 0.0})
dump("reproduce.json", repro)

# ---------------------------------------------------------------- experiment A: text + music -> motion
exp_a = []
for g in AIST_GENRES:
    for song_idx in (TEST_SONG[g], SECOND_SONG[g]):
        song = f"m{g}{song_idx}"
        exp_a.append({"id": song, "task": "mo", "music_desc": "",
                      "motion_desc": f"The style of the dance is {AIST_GENRES[g]}.",
                      "music_path": f"{MUSIC}/{song}.mp3", "offset": OFFSET, "genre": AIST_GENRES[g]})
dump("exp_a_music2dance.json", exp_a)

# ---------------------------------------------------------------- experiment B: text -> music + motion
GTZAN = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
SPOKEN = {"hiphop": "hip-hop"}
DANCE_FOR = {"blues": "street jazz", "classical": "ballet jazz", "country": "house", "disco": "lock",
             "hiphop": "middle hip-hop", "jazz": "street jazz", "metal": "krump", "pop": "pop",
             "reggae": "LA style hip-hop", "rock": "break"}
exp_b = []
for g in GTZAN:
    name = SPOKEN.get(g, g)
    for t, mu in enumerate([f"The music is {name} genre.", f"The audio is a {name} track."]):
        exp_b.append({"id": f"{g}_{t}", "task": "mumo", "music_desc": mu,
                      "motion_desc": f"The style of the dance is {DANCE_FOR[g]}.",
                      "gtzan_genre": g, "genre": DANCE_FOR[g]})
dump("exp_b_text2musicdance.json", exp_b)
