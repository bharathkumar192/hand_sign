from ultralytics import YOLO
from flask import Flask, request, jsonify
import cv2
import numpy as np
import gdown
import os
from flask_ngrok import run_with_ngrok
from pyngrok import ngrok

app = Flask(__name__)
run_with_ngrok(app)  

MODEL_ID = "1eO5f_WpQsYbcCsB26F0hSLO-OrZ2EsGT"
MODEL_NAME = "best_3.pt"

if not os.path.exists(MODEL_NAME):
    print(f"Downloading {MODEL_NAME} from Google Drive...")
    url = f"https://drive.google.com/uc?id={MODEL_ID}"
    gdown.download(url, output=MODEL_NAME, quiet=False)
    print("Download complete.")
else:
    print(f"{MODEL_NAME} already exists in the current directory.")




id_to_tamil = {
    1: "அ(a)",
    2: "ஆ(ā)",
    3: "இ(i)", 
    4: "ஈ(ī)",
    5: "உ(u)",
    6: "ஊ(ū)",
    7: "எ(e)",
    8: "ஏ(ē)",
    9: "ஐ(ai)",
    10: "ஒ(o)",
    11: "ஓ(ō)",
    12: "ஔ(au)",
    13: "ஃ(ak)",
    14: "க்(k)",
    15: "ங்(ṅ)",
    16: "ச்(c)", 
    17: "ஞ்(ñ)",
    18: "ட்(ṭ)",
    19: "ண்(ṇ)",
    20: "த்(t)",
    21: "ந்(n)",
    22: "ப்(p)",
    23: "ம்(m)",
    24: "ய்(y)",
    25: "ர்(r)",
    26: "ல்(l)",
    27: "வ்(v)",
    28: "ழ்(lzh)",
    29: "ள்(ll)",
    30: "ற்(ṟ)",
    31: "ன்(ṉ)",
    32: "க(Ka)",
    33: "கா(Kā)",
    34: "கி(Ki)",
    35: "கீ(Kī)",
    36: "கு(Ku)",
    37: "கூ(Kū)", 
    38: "கெ(Ke)",
    39: "கே(Kē)",
    40: "கை(Kai)",
    41: "கொ(Ko)",
    42: "கோ(Kō)",
    43: "கௌ(Kau)",
    44: "ங(Nga)",
    45: "ஙா(Ngā)",
    46: "ஙி(Ngi)",
    47: "ஙீ(Ngī)",
    48: "ஙு(Ngu)",
    49: "ஙூ(Ngū)",
    50: "ஙெ(Nge)",
    51: "ஙே(Ngē)",
    52: "ஙை(Ngai)",
    53: "ஙொ(Ngo)",
    54: "ஙோ(Ngō)",
    55: "ஙௌ(Ngau)",
    56: "ச(Sa)",
    57: "சா(Sā)",
    58: "சி(Si)",
    59: "சீ(Sī)",
    60: "சு(Su)",
    61: "சூ(Sū)",
    62: "செ(Se)",
    63: "சே(Sē)",
    64: "சை(Sai)",
    65: "சொ(So)",
    66: "சோ(Sō)",
    67: "சௌ(Sau)",
    68: "ஞ(Ña)",
    69: "ஞா(Ñā)",
    70: "ஞி(Ñi)",
    71: "ஞீ(Ñī)",
    72: "ஞு(Ñu)",
    73: "ஞூ(Ñū)",
    74: "ஞெ(Ñe)",
    75: "ஞே(Ñē)",
    76: "ஞை(Ñai)",
    77: "ஞொ(Ño)",
    78: "ஞோ(Ñō)",
    79: "ஞௌ(Ñau)",
    80: "ட(Ṭa)",
    81: "டா(Ṭā)",
    82: "டி(Ṭi)",
    83: "டீ(Ṭī)",
    84: "டு(Ṭu)",
    85: "டூ(Ṭū)",
    86: "டெ(Ṭe)",
    87: "டே(Ṭē)",
    88: "டை(Ṭai)",
    89: "டொ(Ṭo)",
    90: "டோ(Ṭō)",
    91: "டௌ(Ṭau)",
    92: "ண(Ṇa)",
    93: "ணா(Ṇā)",
    94: "ணி(Ṇi)",
    95: "ணீ(Ṇī)",
    96: "ணு(Ṇu)",
    97: "ணூ(Ṇū)",
    98: "ணெ(Ṇe)",
    99: "ணே(Ṇē)",
    100: "ணை(Ṇai)",
    101: "ணொ(Ṇo)",
    102: "ணோ(Ṇō)",
    103: "ணௌ(Ṇau)",
    104: "த(Ta)",
    105: "தா(Tā)",
    106: "தி(Ti)",
    107: "தீ(Tī)",
    108: "து(Tu)",
    109: "தூ(Tū)",
    110: "தெ(Te)",
    111: "தே(Tē)",
    112: "தை(Tai)",
    113: "தொ(To)",
    114: "தோ(Tō)", 
    115: "தௌ(Tau)",
    116: "ந(Na)",
    117: "நா(Nā)",
    118: "நி(Ni)",
    119: "நீ(Nī)",
    120: "நு(Nu)",
    121: "நூ(Nū)",
    122: "நெ(Ne)",
    123: "நே(Nē)",
    124: "நை(Nai)",
    125: "நொ(No)",
    126: "நோ(Nō)",
    127: "நௌ(Nau)",
    128: "ப(Pa)",
    129: "பா(Pā)",
    130: "பி(Pi)",
    131: "பீ(Pī)",
    132: "பு(Pu)",
    133: "பூ(Pū)",
    134: "பெ(Pe)",
    135: "பே(Pē)",
    136: "பை(Pai)", 
    137: "பொ(Po)",
    138: "போ(Pō)",
    139: "பௌ(Pau)",
    140: "ம(Ma)",
    141: "மா(Mā)",
    142: "மி(Mi)",
    143: "மீ(Mī)",
    144: "மு(Mu)",
    145: "மூ(Mū)",
    146: "மெ(Me)",
    147: "மே(Mē)",
    148: "மை(Mai)",
    149: "மொ(Mo)",
    150: "மோ(Mō)",
    151: "மௌ(Mau)",
    152: "ய(Ya)",
    153: "யா(Yā)",
    154: "யி(Yi)",
    155: "யீ(Yī)",
    156: "யு(Yu)",
    157: "யூ(Yū)",
    158: "யெ(Ye)",
    159: "யே(Yē)",
    160: "யை(Yai)",
    161: "யொ(Yo)",
    162: "யோ(Yō)",
    163: "யௌ(Yau)",
    164: "ர(Ra)",
    165: "ரா(Rā)",
    166: "ரி(Ri)",
    167: "ரீ(Rī)",
    168: "ரு(Ru)",
    169: "ரூ(Rū)",
    170: "ரெ(Re)",
    171: "ரே(Rē)",
    172: "ரை(Rai)",
    173: "ரொ(Ro)",
    174: "ரோ(Rō)",
    175: "ரௌ(Rau)",
    176: "ல(La)",
    177: "லா(Lā)",
    178: "லி(Li)",
    179: "லீ(Lī)",
    180: "லு(Lu)",
    181: "லூ(Lū)",
    182: "லெ(Le)",
    183: "லே(Lē)",
    184: "லை(Lai)",
    185: "லொ(Lo)",
    186: "லோ(Lō)",
    187: "லௌ(Lau)",
    188: "வ(Va)",
    189: "வா(Vā)",
    190: "வி(Vi)",
    191: "வீ(Vī)",
    192: "வு(Vu)",
    193: "வூ(Vū)",
    194: "வெ(Ve)",
    195: "வே(Vē)",
    196: "வை(Vai)",
    197: "வொ(Vo)",
    198: "வோ(Vō)",
    199: "வௌ(Vau)",
    200: "ழ(Lzha)",
    201: "ழா(Lzhā)",
    202: "ழி(Lzhi)", 
    203: "ழீ(Lzhī)",
    204: "ழு(Lzhu)",
    205: "ழூ(Lzhū)",
    206: "ழெ(Lzhe)",
    207: "ழே(Lzhē)",
    208: "ழை(Lzhai)",
    209: "ழொ(Lzho)",
    210: "ழோ(Lzhō)",
    211: "ழௌ(Lzhau)",
    212: "ள(Lla)",
    213: "ளா(Llā)",
    214: "ளி(Lli)",
    215: "ளீ(Llī)",
    216: "ளு(Llu)",
    217: "ளூ(Llū)",
    218: "ளெ(Lle)",
    219: "ளே(Llē)",
    220: "ளை(Llai)",
    221: "ளொ(Llo)",
    222: "ளோ(Llō)",
    223: "ளௌ(Llau)",
    224: "ற(Ṟa)",
    225: "றா(Ṟā)",
    226: "றி(Ṟi)",
    227: "றீ(Ṟī)",
    228: "று(Ṟu)",
    229: "றூ(Ṟū)",
    230: "றெ(Ṟe)",
    231: "றே(Ṟē)",
    232: "றை(Ṟai)",
    233: "றொ(Ṟo)",
    234: "றோ(Ṟō)",
    235: "றௌ(Ṟau)",
    236: "ன(Ṉa)",
    237: "னா(Ṉā)",
    238: "னி(Ṉi)",
    239: "னீ(Ṉī)",
    240: "னு(Ṉu)",
    241: "னூ(Ṉū)",
    242: "னெ(Ṉe)",
    243: "னே(Ṉē)",
    244: "னை(Ṉai)",
    245: "னொ(Ṉo)",
    246: "னோ(Ṉō)",
    247: "னௌ(Ṉau)"
}

# Load the model
model = YOLO("best_3.pt")

@app.route('/')
def home():
    return "Flask App for Tamil Hand Sign"

@app.route('/predictions', methods=['POST'])
def predict():
    # Get the image data from the request
    image_data = request.files['image'].read()
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

    # Make predictions
    results = model.predict(source=image)
    
    # Extract predictions and map to Tamil characters
    predictions = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls)
            tamil_character = id_to_tamil.get(cls, "Unknown")  # Get Tamil character or return "Unknown"
            predictions.append(tamil_character)
            print(predictions)
    print(predictions)
    # Return the predictions as a JSON response
    return jsonify({'predictions': predictions}), 200, {'Content-Type': 'application/json'}



if __name__ == '__main__':
    ngrok.set_auth_token("2gtde6QPcIZPmHSSq0ZhCfW3jli_3ANYfarF8RhuJQbFB7ym1")
    public_url = ngrok.connect(5000, hostname="monarch-modest-cod.ngrok-free.app")
    print(f"ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:5000/\"")
    app.config["BASE_URL"] = public_url
    app.run()
