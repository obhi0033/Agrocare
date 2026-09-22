import streamlit as st
from pathlib import Path
import cv2, numpy as np, joblib, pandas as pd, base64
import streamlit.components.v1 as components
from image_quality import decode_uploaded_image, check_image_quality
from feature_extraction import extract_features, features_for_model
from weather_context import weather_risk, weather_risk_breakdown
from database import init_db, save_scan, get_history, get_summary, clear_history

st.set_page_config(page_title="AgroCare AI", page_icon="🌿", layout="wide")
init_db()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Bengali:wght@400;600;700;800&family=Orbitron:wght@500;700;800&display=swap');
:root{--bg:#0A0E17;--panel:#121824;--green:#39FF14;--cyan:#00FFFF;--text:#E9FFF4;--muted:#91A8A3}
html,body,[class*="css"]{font-family:'Noto Sans Bengali',sans-serif}
.stApp{background:#0A0E17;color:var(--text);background-image:radial-gradient(circle at 15% 20%,rgba(57,255,20,.08),transparent 24%),radial-gradient(circle at 85% 18%,rgba(0,255,255,.07),transparent 25%),linear-gradient(rgba(0,255,255,.025) 1px,transparent 1px),linear-gradient(90deg,rgba(57,255,20,.02) 1px,transparent 1px);background-size:auto,auto,42px 42px,42px 42px;background-attachment:fixed}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#121824 0%,#0C111B 100%);border-right:1px solid rgba(0,255,255,.16)}
[data-testid="stSidebar"] *{color:#DFFCF1!important}.block-container{padding-top:1.25rem;max-width:1450px}
h1,h2,h3,h4{color:#F1FFF7!important}.hero{position:relative;overflow:hidden;padding:32px 36px;border-radius:28px;background:linear-gradient(125deg,rgba(18,24,36,.96),rgba(8,40,35,.92));border:1px solid rgba(57,255,20,.28);box-shadow:0 0 45px rgba(57,255,20,.08),inset 0 0 30px rgba(0,255,255,.035);margin-bottom:24px}.hero:after{content:'';position:absolute;inset:-50%;background:conic-gradient(from 90deg,transparent 0 82%,rgba(57,255,20,.16),rgba(0,255,255,.13),transparent 92%);animation:spinGlow 10s linear infinite;pointer-events:none}.hero>*{position:relative;z-index:2}.hero h1{font-family:Orbitron,sans-serif;font-size:43px;margin:0 0 8px;text-shadow:0 0 18px rgba(57,255,20,.25)}.hero p{font-size:16px;color:#C8DDD6;margin:0}.badge{display:inline-block;background:rgba(57,255,20,.09);color:#8CFF78;border:1px solid rgba(57,255,20,.4);padding:7px 13px;border-radius:999px;font-weight:800;font-size:11px;margin-bottom:12px;box-shadow:0 0 18px rgba(57,255,20,.08)}
@keyframes spinGlow{to{transform:rotate(360deg)}}
.cyber-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:22px;margin:14px 0 22px}.glow-wrap{position:relative;border-radius:24px;padding:1.5px;overflow:hidden;transition:.35s ease}.glow-wrap:before{content:'';position:absolute;width:180%;height:180%;left:-40%;top:-40%;background:conic-gradient(transparent 0 25%,#39FF14 34%,#00FFFF 44%,transparent 54% 100%);animation:spinGlow 4.5s linear infinite}.glow-card{position:relative;z-index:1;min-height:170px;border-radius:23px;padding:25px;background:rgba(18,24,36,.82);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);box-shadow:inset 0 0 28px rgba(0,255,255,.025)}.glow-wrap:hover{transform:translateY(-9px) scale(1.015);filter:drop-shadow(0 0 18px rgba(57,255,20,.25))}.num{font-family:Orbitron,sans-serif;font-size:34px;font-weight:800;color:#39FF14;text-shadow:0 0 16px rgba(57,255,20,.6)}.glow-card b{font-size:18px;color:#F1FFF7}.glow-card p{color:#91A8A3;margin-bottom:0}
.card{background:rgba(18,24,36,.78);border:1px solid rgba(0,255,255,.17);border-radius:22px;padding:22px;box-shadow:0 10px 30px rgba(0,0,0,.22),inset 0 0 20px rgba(0,255,255,.025);margin-bottom:15px;backdrop-filter:blur(10px)}.metric{font-family:Orbitron,sans-serif;font-size:30px;font-weight:800;color:#39FF14;text-shadow:0 0 14px rgba(57,255,20,.24)}.muted{color:#91A8A3}.good{background:rgba(57,255,20,.07);border:1px solid rgba(57,255,20,.3);border-left:4px solid #39FF14;padding:14px;border-radius:12px;color:#DFFFF0}.warn{background:rgba(255,190,60,.08);border:1px solid rgba(255,190,60,.28);border-left:4px solid #FFC247;padding:14px;border-radius:12px;color:#FFF1C8}
div[data-testid="stFileUploader"]{background:rgba(18,24,36,.76);border:1px dashed rgba(57,255,20,.65);padding:15px;border-radius:20px;box-shadow:0 0 25px rgba(57,255,20,.05)}
.stButton>button,.stDownloadButton>button{border-radius:12px;font-weight:800;background:#111B24;color:#39FF14;border:1px solid rgba(57,255,20,.45);box-shadow:0 0 12px rgba(57,255,20,.06)}.stButton>button:hover,.stDownloadButton>button:hover{border-color:#00FFFF;color:#00FFFF;box-shadow:0 0 20px rgba(0,255,255,.16)}
.lang-pill{padding:10px 14px;border-radius:14px;background:linear-gradient(90deg,rgba(57,255,20,.08),rgba(0,255,255,.07));border:1px solid rgba(57,255,20,.22);color:#7BFF67;font-weight:800;margin-bottom:10px}.bn-title{font-weight:800}.scan-frame{position:relative;border-radius:24px;overflow:hidden;border:1px solid rgba(57,255,20,.35);box-shadow:0 0 35px rgba(57,255,20,.09);background:#060A0E}.scan-frame img{display:block;width:100%;height:auto}.laser{position:absolute;left:0;right:0;height:4px;top:0;background:linear-gradient(90deg,transparent,#39FF14 20%,#E8FFE3 50%,#00FFFF 80%,transparent);box-shadow:0 0 10px #39FF14,0 0 24px #00FFFF,0 0 44px rgba(57,255,20,.7);animation:laserScan 2.7s ease-in-out infinite;z-index:5}.laser:after{content:'';position:absolute;left:0;right:0;top:-28px;height:58px;background:linear-gradient(to bottom,transparent,rgba(57,255,20,.10),transparent)}@keyframes laserScan{0%{top:2%;opacity:.65}50%{top:96%;opacity:1}100%{top:2%;opacity:.65}}.scan-label{text-align:center;padding:9px;color:#8CFF78;font-size:12px;letter-spacing:.08em;background:rgba(10,14,23,.88)}
[data-testid="stMetric"]{background:rgba(18,24,36,.58);border:1px solid rgba(0,255,255,.12);padding:12px;border-radius:16px}[data-testid="stMetricValue"]{color:#39FF14}.stProgress>div>div>div>div{background:linear-gradient(90deg,#39FF14,#00FFFF)!important}
[data-testid="stDataFrame"]{border:1px solid rgba(0,255,255,.14);border-radius:14px;overflow:hidden}.stAlert{background:rgba(18,24,36,.72)!important;color:#E9FFF4!important;border:1px solid rgba(0,255,255,.13)!important}
hr{border-color:rgba(0,255,255,.12)!important}.stCaption,p,label{color:#B7C9C3}.stSelectbox>div>div,.stTextInput>div>div>input,.stNumberInput input{background:#121824!important;color:#E9FFF4!important;border-color:rgba(0,255,255,.16)!important}
@media(max-width:900px){.cyber-grid{grid-template-columns:1fr}.hero h1{font-size:34px}}

/* Screenshot-matched premium dashboard */
.block-container{max-width:1320px!important;padding-left:2.1rem!important;padding-right:2.1rem!important}
.stApp:before{content:"";position:fixed;inset:0;pointer-events:none;z-index:0;background:radial-gradient(circle at 22% 15%,rgba(57,255,20,.08),transparent 19%),radial-gradient(circle at 82% 22%,rgba(0,255,255,.07),transparent 24%),repeating-linear-gradient(0deg,transparent 0 34px,rgba(57,255,20,.018) 35px),repeating-linear-gradient(90deg,transparent 0 34px,rgba(0,255,255,.015) 35px)}
[data-testid="stSidebar"]{box-shadow:14px 0 45px rgba(0,0,0,.38)}
.hero{min-height:185px;padding:28px 34px!important;border-radius:0 0 26px 26px!important;background:linear-gradient(100deg,rgba(7,55,37,.97),rgba(9,106,60,.80),rgba(7,44,35,.94))!important;border:1px solid rgba(57,255,20,.20)!important;box-shadow:0 16px 38px rgba(0,0,0,.38),0 0 28px rgba(57,255,20,.10)!important}
.hero:before{content:"";position:absolute;inset:0;background-image:var(--hero-roots);background-size:cover;background-position:right center;opacity:.82;mix-blend-mode:screen;pointer-events:none}
.hero h1{font-family:'Noto Sans Bengali',sans-serif!important;font-size:42px!important;letter-spacing:-1px}.hero p{font-size:16px!important;color:#e2f8ee!important;max-width:760px}.badge{background:#d8ffe4!important;color:#0c5d39!important;border:none!important;box-shadow:none!important}
.cyber-grid{gap:16px!important;align-items:stretch}.glow-wrap{padding:1px!important;background:linear-gradient(145deg,rgba(255,255,255,.16),rgba(57,255,20,.18),rgba(0,255,255,.12));box-shadow:0 16px 34px rgba(0,0,0,.34);border-radius:22px!important}.glow-wrap:before{opacity:.35}.glow-card{min-height:420px!important;padding:18px!important;border-radius:21px!important;background:linear-gradient(160deg,rgba(26,31,39,.96),rgba(17,23,31,.92))!important;border:1px solid rgba(255,255,255,.07)}.num{font-size:26px!important}.home-card-title{font-size:17px;font-weight:800;color:#f3fff8;margin-top:2px}.home-card-desc{height:52px;color:#b8cbc4!important;font-size:14px;line-height:1.45}.home-visual{position:relative;height:275px;margin-top:12px;border-radius:14px;overflow:hidden;border:1px solid rgba(57,255,20,.16);background:#071018;box-shadow:inset 0 0 28px rgba(0,0,0,.5)}.home-visual img{width:100%;height:100%;object-fit:cover;display:block;transition:transform .5s ease,filter .5s ease}.glow-wrap:hover .home-visual img{transform:scale(1.035);filter:saturate(1.15) brightness(1.08)}.home-visual:after{content:"";position:absolute;inset:0;background:linear-gradient(180deg,transparent 60%,rgba(5,12,16,.42));pointer-events:none}
.home-cta{margin-top:8px;padding:14px 16px;border-radius:12px;background:rgba(0,255,255,.07);border:1px solid #00FFFF;box-shadow:0 0 9px rgba(0,255,255,.7),inset 0 0 18px rgba(0,255,255,.06);color:#dfffff;font-weight:700}
@media(max-width:900px){.glow-card{min-height:auto!important}.home-visual{height:230px}}

</style>""", unsafe_allow_html=True)

# Particles.js cyber background. The CSS grid above remains as a fallback if a browser blocks the CDN.
components.html("""
<div id="particles-js"></div>
<style>html,body{margin:0;background:transparent;overflow:hidden}#particles-js{position:fixed;inset:0;width:100vw;height:100vh;pointer-events:none}</style>
<script src="https://cdn.jsdelivr.net/npm/particles.js@2.0.0/particles.min.js"></script>
<script>
particlesJS('particles-js',{particles:{number:{value:42,density:{enable:true,value_area:1000}},color:{value:['#39FF14','#00FFFF']},shape:{type:'circle'},opacity:{value:.18,random:true},size:{value:2,random:true},line_linked:{enable:true,distance:145,color:'#39FF14',opacity:.10,width:1},move:{enable:true,speed:.65,direction:'none',random:true,straight:false,out_mode:'out',bounce:false}},interactivity:{detect_on:'canvas',events:{onhover:{enable:false},onclick:{enable:false},resize:true}},retina_detect:true});
</script>""", height=1)


TXT={
"English": {"nav":"Navigation","home":"🏠 Home","scan":"🔬 Scan & Diagnose","weather":"🌦️ Weather Risk","model":"🧠 Model Center","history":"🕘 Scan History","language":"Language / ভাষা","dataset":"Dataset folder","sharp":"Image sharpness threshold","tag":"AI-POWERED CROP HEALTH ASSISTANT","subtitle":"Smart Crop Disease Diagnostics & Management • Scan a leaf. Understand the risk. Take informed action.","command":"Your crop health command center","upload":"Upload Leaf","upload_d":"Use a clear photo of the affected leaf.","analysis":"AI Analysis","analysis_d":"Quality, crop/disease class and confidence.","action":"Action Guidance","action_d":"Severity, weather context and practical next steps.","goto":"Go to Scan & Diagnose to test a leaf image.","scan_title":"🔬 Scan & Diagnose","scan_cap":"Upload a clear leaf photo. Low-confidence or unsupported matches will be marked Unknown instead of forcing a diagnosis.","drop":"Drop a leaf image here","preview":"Leaf preview","quality":"Image Quality","score":"Sharpness score","badimg":"Could not read this image.","blurry":"Image is too blurry. Please upload a clearer photo.","nomodel":"No trained model found in this project folder.","copy":"Copy the models folder containing best_model.joblib into this project.","reliable":"Reliable prediction threshold","unknown":"⚠️ Unknown / Unsupported","unknown_d":"This leaf does not confidently match the trained classes.","best":"Best match confidence","done":"✓ AI diagnosis completed","crop":"Crop","disease":"Disease","confidence":"Confidence","severity":"Severity / Status","recommend":"💡 Recommended Action","matches":"Top AI matches","class":"Class","prob":"Probability","weather_title":"🌦️ Environmental Disease Pressure","temp":"Temperature °C","hum":"Humidity %","rain":"Rainfall mm","assessment":"Current environmental assessment","no_risk":"No major rule-based risk trigger.","weather_note":"Weather is contextual information and is not used as an image-classification feature in this prototype.","factors_title":"Factors checked","met":"✅ Triggered","notmet":"⬜ Not triggered","factor_humidity":"Humidity ≥ 80%","factor_temp":"Temperature 18–30°C","factor_rain":"Rainfall > 0mm","updated":"Updated","model_title":"🧠 Model Center","model_info":"Training tools remain in the Week 2 MultiCrop version. Week 3 keeps technical model work separate from the farmer-facing interface.","use_model":"The trained best_model.joblib in the models folder is used for diagnosis.","history_title":"🕘 Scan History","total":"Total Scans","detected":"Detected","unknown_m":"Unknown","avg":"Avg. Confidence","recent":"Recent diagnosis records","export":"⬇️ Export History CSV","clear":"🗑️ Clear Scan History","empty":"No scans saved yet","empty_d":"Go to Scan & Diagnose and analyze a leaf. The result will be saved automatically here.","footer":"AgroCare AI is a classroom prototype. Predictions should be confirmed with qualified local agricultural guidance before treatment decisions."},
"বাংলা": {"nav":"মেনু","home":"🏠 হোম","scan":"🔬 রোগ শনাক্তকরণ","weather":"🌦️ আবহাওয়া ঝুঁকি","model":"🧠 মডেল সেন্টার","history":"🕘 স্ক্যান ইতিহাস","language":"ভাষা / Language","dataset":"ডেটাসেট ফোল্ডার","sharp":"ছবির স্পষ্টতার সীমা","tag":"এআই-চালিত ফসল স্বাস্থ্য সহায়ক","subtitle":"স্মার্ট ফসল রোগ শনাক্তকরণ ও ব্যবস্থাপনা • পাতার ছবি দিন • ঝুঁকি বুঝুন • সঠিক পদক্ষেপ নিন।","command":"আপনার ফসল স্বাস্থ্য কমান্ড সেন্টার","upload":"পাতার ছবি দিন","upload_d":"আক্রান্ত পাতার একটি পরিষ্কার ছবি ব্যবহার করুন।","analysis":"এআই বিশ্লেষণ","analysis_d":"ছবির মান, ফসল/রোগের ধরন এবং আত্মবিশ্বাস বিশ্লেষণ।","action":"করণীয় পরামর্শ","action_d":"তীব্রতা, আবহাওয়ার ঝুঁকি এবং পরবর্তী করণীয়।","goto":"পাতার ছবি পরীক্ষা করতে রোগ শনাক্তকরণ মেনুতে যান।","scan_title":"🔬 রোগ শনাক্তকরণ","scan_cap":"পাতার পরিষ্কার ছবি আপলোড করুন। কম আত্মবিশ্বাসের বা অজানা ফলাফলকে জোর করে রোগ হিসেবে দেখানো হবে না।","drop":"এখানে পাতার ছবি দিন","preview":"পাতার ছবি","quality":"ছবির মান","score":"স্পষ্টতার স্কোর","badimg":"ছবিটি পড়া যায়নি।","blurry":"ছবিটি বেশি ঝাপসা। আরও পরিষ্কার ছবি দিন।","nomodel":"এই প্রজেক্ট ফোল্ডারে trained model পাওয়া যায়নি।","copy":"best_model.joblib সহ models ফোল্ডারটি এই প্রজেক্টে কপি করুন।","reliable":"নির্ভরযোগ্য ফলাফলের সীমা","unknown":"⚠️ অজানা / সমর্থিত নয়","unknown_d":"এই পাতাটি প্রশিক্ষিত শ্রেণিগুলোর সাথে যথেষ্ট আত্মবিশ্বাসে মিলছে না।","best":"সর্বোচ্চ মিলের আত্মবিশ্বাস","done":"✓ এআই বিশ্লেষণ সম্পন্ন","crop":"ফসল","disease":"রোগ","confidence":"আত্মবিশ্বাস","severity":"তীব্রতা / অবস্থা","recommend":"💡 করণীয় পরামর্শ","matches":"এআই-এর সম্ভাব্য মিল","class":"শ্রেণি","prob":"সম্ভাবনা","weather_title":"🌦️ পরিবেশগত রোগের ঝুঁকি","temp":"তাপমাত্রা °C","hum":"আর্দ্রতা %","rain":"বৃষ্টিপাত mm","assessment":"বর্তমান পরিবেশগত মূল্যায়ন","no_risk":"উল্লেখযোগ্য নিয়মভিত্তিক ঝুঁকি পাওয়া যায়নি।","weather_note":"এই প্রোটোটাইপে আবহাওয়ার তথ্য সহায়ক প্রেক্ষাপট হিসেবে ব্যবহৃত হয়; ছবি শ্রেণিবিন্যাসের feature হিসেবে নয়।","factors_title":"যে বিষয়গুলো যাচাই করা হয়েছে","met":"✅ ট্রিগার হয়েছে","notmet":"⬜ ট্রিগার হয়নি","factor_humidity":"আর্দ্রতা ≥ ৮০%","factor_temp":"তাপমাত্রা ১৮–৩০°C","factor_rain":"বৃষ্টিপাত > ০mm","updated":"আপডেট হয়েছে","model_title":"🧠 মডেল সেন্টার","model_info":"Training tools Week 2 MultiCrop সংস্করণে রয়েছে। Week 3-এ technical model অংশকে ব্যবহারকারীর interface থেকে আলাদা রাখা হয়েছে।","use_model":"models ফোল্ডারের trained best_model.joblib রোগ শনাক্তকরণে ব্যবহৃত হচ্ছে।","history_title":"🕘 স্ক্যান ইতিহাস","total":"মোট স্ক্যান","detected":"শনাক্ত হয়েছে","unknown_m":"অজানা","avg":"গড় আত্মবিশ্বাস","recent":"সাম্প্রতিক রোগ শনাক্তকরণ রেকর্ড","export":"⬇️ ইতিহাস CSV ডাউনলোড","clear":"🗑️ স্ক্যান ইতিহাস মুছুন","empty":"এখনও কোনো স্ক্যান সংরক্ষিত হয়নি","empty_d":"রোগ শনাক্তকরণে গিয়ে একটি পাতার ছবি বিশ্লেষণ করুন। ফলাফল এখানে স্বয়ংক্রিয়ভাবে সংরক্ষিত হবে।","footer":"AgroCare AI একটি শিক্ষামূলক প্রোটোটাইপ। চিকিৎসা বা ব্যবস্থাপনার সিদ্ধান্তের আগে যোগ্য স্থানীয় কৃষি বিশেষজ্ঞের পরামর্শ নিন।"}}

DISEASE_INFO={"Early_blight":("Moderate","মাঝারি","Remove badly affected leaves, keep foliage dry, improve airflow and follow locally approved crop-protection guidance.","আক্রান্ত পাতা সরান, পাতা ভেজা কম রাখুন, বাতাস চলাচল বাড়ান এবং স্থানীয়ভাবে অনুমোদিত কৃষি পরামর্শ অনুসরণ করুন।"),"Late_blight":("High","উচ্চ","Isolate affected plants where practical, avoid overhead watering and seek local agricultural guidance promptly.","সম্ভব হলে আক্রান্ত গাছ আলাদা করুন, উপর থেকে পানি দেওয়া এড়িয়ে চলুন এবং দ্রুত স্থানীয় কৃষি পরামর্শ নিন।"),"Leaf_Mold":("Moderate","মাঝারি","Reduce humidity around plants, improve ventilation and remove heavily affected foliage.","গাছের আশেপাশে আর্দ্রতা কমান, বাতাস চলাচল বাড়ান এবং বেশি আক্রান্ত পাতা সরান।"),"healthy":("Healthy","সুস্থ","No disease action suggested. Continue routine monitoring and good crop hygiene.","রোগের ব্যবস্থা প্রয়োজন মনে হচ্ছে না। নিয়মিত পর্যবেক্ষণ ও পরিচ্ছন্ন চাষাবাদ চালিয়ে যান।")}

CROP_BN={
    "Tomato":"টমেটো","Potato":"আলু","Pepper  bell":"ক্যাপসিকাম","Pepper bell":"ক্যাপসিকাম","Pepper, bell":"ক্যাপসিকাম",
    "Corn (maize)":"ভুট্টা","Grape":"আঙুর","Apple":"আপেল","Blueberry":"ব্লুবেরি",
    "Cherry (including sour)":"চেরি","Peach":"পীচ","Orange":"কমলা","Raspberry":"রাস্পবেরি",
    "Soybean":"সয়াবিন","Squash":"স্কোয়াশ","Strawberry":"স্ট্রবেরি"
}
DISEASE_BN={
    "Early blight":"আর্লি ব্লাইট","Late blight":"লেট ব্লাইট","Leaf Mold":"লিফ মোল্ড","healthy":"সুস্থ",
    "Bacterial spot":"ব্যাকটেরিয়াল স্পট","Target Spot":"টার্গেট স্পট","Septoria leaf spot":"সেপ্টোরিয়া লিফ স্পট",
    "Tomato Yellow Leaf Curl Virus":"টমেটো ইয়েলো লিফ কার্ল ভাইরাস","Tomato mosaic virus":"টমেটো মোজাইক ভাইরাস",
    "Spider mites Two-spotted spider mite":"স্পাইডার মাইট","Northern Leaf Blight":"নর্দার্ন লিফ ব্লাইট",
    "Common rust":"কমন রাস্ট","Cercospora leaf spot Gray leaf spot":"গ্রে লিফ স্পট",
    "Black rot":"ব্ল্যাক রট","Esca (Black Measles)":"এস্কা (ব্ল্যাক মিজলস)",
    "Leaf blight (Isariopsis Leaf Spot)":"লিফ ব্লাইট","Haunglongbing (Citrus greening)":"সিট্রাস গ্রিনিং"
}
STATUS_BN={"Detected":"শনাক্ত","Unknown / Unsupported":"অজানা / সমর্থিত নয়","Unknown":"অজানা","Unsupported":"সমর্থিত নয়","Review":"পর্যালোচনা","High":"উচ্চ","Moderate":"মাঝারি","Healthy":"সুস্থ"}

def _norm_key(s):
    # Collapse commas/multiple spaces/case so "Pepper, bell", "Pepper  bell",
    # "pepper bell" etc. all match the same translation entry.
    return " ".join(str(s).replace(",", " ").split()).lower()

CROP_BN_NORM = {_norm_key(k): v for k, v in CROP_BN.items()}
DISEASE_BN_NORM = {_norm_key(k): v for k, v in DISEASE_BN.items()}

_BN_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
def bn_num(s, lang):
    # Convert English digits to Bengali digits in Bangla mode; leave English mode untouched.
    return str(s).translate(_BN_DIGITS) if lang=="বাংলা" else str(s)

def notify_change(state_key, value, label):
    """Show a small toast the moment a control's value actually changes, so the
    user gets explicit feedback instead of the result silently updating."""
    prev = st.session_state.get(state_key)
    if prev is not None and prev != value:
        msg = f"{label}: {prev} → {value}"
        if hasattr(st, "toast"):
            st.toast(msg, icon="🔔")
        else:
            st.info(msg)
    st.session_state[state_key] = value

def display_crop_disease(crop,disease,lang):
    if lang!="বাংলা": return crop,disease
    return (CROP_BN_NORM.get(_norm_key(crop), crop),
            DISEASE_BN_NORM.get(_norm_key(disease), disease))

def display_class_label(label,lang):
    crop,disease=pretty_class(str(label))
    crop,disease=display_crop_disease(crop,disease,lang)
    return f"{crop} — {disease}"


if "saved_upload_key" not in st.session_state: st.session_state.saved_upload_key=None
if "lang" not in st.session_state: st.session_state.lang="English"

def pretty_class(c):
    p=c.replace("___","|").split("|")
    return (p[0].replace("_"," "),p[1].replace("_"," ")) if len(p)==2 else ("Detected crop",c.replace("_"," "))
def info_for(label):
    for k,v in DISEASE_INFO.items():
        if k.lower() in label.lower(): return v
    return ("Review","পর্যালোচনা","Monitor the plant and confirm the result with local agricultural guidance before treatment.","গাছটি পর্যবেক্ষণ করুন এবং চিকিৎসার আগে স্থানীয় কৃষি পরামর্শ দিয়ে ফলাফল নিশ্চিত করুন।")

def bn_weather(level,reasons):
    level_map={"Higher environmental disease pressure":"উচ্চ পরিবেশগত রোগের ঝুঁকি","Moderate environmental disease pressure":"মাঝারি পরিবেশগত রোগের ঝুঁকি","Lower environmental disease pressure":"কম পরিবেশগত রোগের ঝুঁকি"}
    reason_map={"High humidity may support fungal disease development":"উচ্চ আর্দ্রতা ছত্রাকজনিত রোগ বৃদ্ধিতে সহায়তা করতে পারে","Temperature is within a common disease-favorable range":"তাপমাত্রা রোগ বিস্তারের অনুকূল সীমায় রয়েছে","Recent rainfall can increase leaf wetness":"সাম্প্রতিক বৃষ্টিপাত পাতার ভেজাভাব বাড়াতে পারে"}
    return level_map.get(level,level),[reason_map.get(x,x) for x in reasons]

HERO_ROOTS = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA5MDAgMjYwIj48ZGVmcz48ZmlsdGVyIGlkPSJnIj48ZmVHYXVzc2lhbkJsdXIgc3RkRGV2aWF0aW9uPSIzIiByZXN1bHQ9ImIiLz48ZmVNZXJnZT48ZmVNZXJnZU5vZGUgaW49ImIiLz48ZmVNZXJnZU5vZGUgaW49IlNvdXJjZUdyYXBoaWMiLz48L2ZlTWVyZ2U+PC9maWx0ZXI+PGxpbmVhckdyYWRpZW50IGlkPSJsZyIgeDE9IjAiIHgyPSIxIj48c3RvcCBzdG9wLWNvbG9yPSIjMzlGRjE0Ii8+PHN0b3Agb2Zmc2V0PSIxIiBzdG9wLWNvbG9yPSIjMDBGRkZGIi8+PC9saW5lYXJHcmFkaWVudD48L2RlZnM+PGcgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ1cmwoI2xnKSIgc3Ryb2tlLXdpZHRoPSIyIiBvcGFjaXR5PSIuNzIiIGZpbHRlcj0idXJsKCNnKSI+PHBhdGggZD0iTTcyMCAwYzAgNDItOCA2My0yNCA4NS0xOCAyNC0zNCAzNi00OSA1NS0xMyAxNi0xOSAzOC0yMCA3MiIvPjxwYXRoIGQ9Ik03MjAgNDhjMzggMTggNTUgMzYgNjcgNjggOSAyNSAzMSAzNyA2NSA0MyIvPjxwYXRoIGQ9Ik03MDIgODZjLTM3IDEwLTU3IDMwLTcyIDYwLTExIDIyLTMzIDM4LTY4IDQ4Ii8+PHBhdGggZD0iTTc0NiA3MGMtOCAyOSAxIDUyIDI3IDY5IDE5IDEzIDI4IDMyIDI4IDU4Ii8+PHBhdGggZD0iTTY3MSAxMThjLTI5LTItNTEgOC02NSAzMC0xMSAxOC0zMSAyNy01OSAyOCIvPjxwYXRoIGQ9Ik03OTQgMTE4YzMxLTcgNTcgMSA3OCAyMyIvPjwvZz48ZyBmaWxsPSIjMzlGRjE0IiBvcGFjaXR5PSIuNzUiPjxjaXJjbGUgY3g9IjcyMCIgY3k9IjQ4IiByPSIzIi8+PGNpcmNsZSBjeD0iNjk2IiBjeT0iODUiIHI9IjIiLz48Y2lyY2xlIGN4PSI3NDYiIGN5PSI3MCIgcj0iMiIvPjxjaXJjbGUgY3g9IjY3MSIgY3k9IjExOCIgcj0iMiIvPjxjaXJjbGUgY3g9Ijc5NCIgY3k9IjExOCIgcj0iMiIvPjxjaXJjbGUgY3g9IjYzMCIgY3k9IjE0NiIgcj0iMiIvPjwvZz48L3N2Zz4="
CARD_SCAN = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA2MDAgMzkwIj48ZGVmcz48cmFkaWFsR3JhZGllbnQgaWQ9ImIiPjxzdG9wIHN0b3AtY29sb3I9IiMxNzNjMzAiLz48c3RvcCBvZmZzZXQ9IjEiIHN0b3AtY29sb3I9IiMwNzEwMTgiLz48L3JhZGlhbEdyYWRpZW50PjxmaWx0ZXIgaWQ9ImciPjxmZUdhdXNzaWFuQmx1ciBzdGREZXZpYXRpb249IjUiIHJlc3VsdD0ieCIvPjxmZU1lcmdlPjxmZU1lcmdlTm9kZSBpbj0ieCIvPjxmZU1lcmdlTm9kZSBpbj0iU291cmNlR3JhcGhpYyIvPjwvZmVNZXJnZT48L2ZpbHRlcj48L2RlZnM+PHJlY3Qgd2lkdGg9IjYwMCIgaGVpZ2h0PSIzOTAiIHJ4PSIyNCIgZmlsbD0idXJsKCNiKSIvPjxnIG9wYWNpdHk9Ii43IiBmaWxsPSIjMWQ1YjM3Ij48ZWxsaXBzZSBjeD0iMTE1IiBjeT0iMTAwIiByeD0iODUiIHJ5PSIzNSIgdHJhbnNmb3JtPSJyb3RhdGUoLTMwIDExNSAxMDApIi8+PGVsbGlwc2UgY3g9IjQ3MCIgY3k9Ijk1IiByeD0iOTUiIHJ5PSIzOCIgdHJhbnNmb3JtPSJyb3RhdGUoMjUgNDcwIDk1KSIvPjxlbGxpcHNlIGN4PSIxMDAiIGN5PSIzMTAiIHJ4PSIxMDAiIHJ5PSI0MiIgdHJhbnNmb3JtPSJyb3RhdGUoMTggMTAwIDMxMCkiLz48L2c+PGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMjEwIDM1KSByb3RhdGUoLTggOTUgMTYwKSI+PHJlY3QgeD0iMCIgeT0iMCIgd2lkdGg9IjE5MCIgaGVpZ2h0PSIzMjAiIHJ4PSIzMCIgZmlsbD0iIzIwMmIzNCIgc3Ryb2tlPSIjMDBGRkZGIiBzdHJva2Utd2lkdGg9IjQiLz48cmVjdCB4PSIxMiIgeT0iMjIiIHdpZHRoPSIxNjYiIGhlaWdodD0iMjU1IiByeD0iMTgiIGZpbGw9IiMwYTBlMTciLz48cGF0aCBkPSJNNDggMTk4YzEyLTkxIDk4LTEzNSAxMjAtMTEzIDIyIDIzLTkgMTExLTkxIDEzMi0xOSA1LTM1LTEtMjktMTl6IiBmaWxsPSIjMzlGRjE0IiBvcGFjaXR5PSIuNzIiLz48cGF0aCBkPSJNNjkgMTk5YzI5LTQwIDU1LTY3IDg5LTk4IiBzdHJva2U9IiNlOGZmZTMiIHN0cm9rZS13aWR0aD0iNCIgZmlsbD0ibm9uZSIvPjxnIGZpbGw9IiNmZjc0NWUiPjxjaXJjbGUgY3g9IjkzIiBjeT0iMTUwIiByPSIxMiIvPjxjaXJjbGUgY3g9IjEyMyIgY3k9IjEzMiIgcj0iOSIvPjxjaXJjbGUgY3g9IjgwIiBjeT0iMTc4IiByPSI4Ii8+PC9nPjxjaXJjbGUgY3g9Ijk1IiBjeT0iMjk4IiByPSIxMCIgZmlsbD0iI2I5ZDlkMiIvPjwvZz48ZyBmaWx0ZXI9InVybCgjZykiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzM5RkYxNCIgc3Ryb2tlLXdpZHRoPSIzIj48cGF0aCBkPSJNMTcwIDcwaDU1TTE3MCA3MHY0NU00MzAgNzBoLTU1TTQzMCA3MHY0NU0xNzAgMzIwaDU1TTE3MCAzMjB2LTQ1TTQzMCAzMjBoLTU1TTQzMCAzMjB2LTQ1Ii8+PC9nPjx0ZXh0IHg9IjI4IiB5PSIzNjAiIGZpbGw9IiMwMEZGRkYiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNiI+QUkgTEVBRiBWSVNJT04g4oCiIExJVkUgU0NBTjwvdGV4dD48L3N2Zz4="
CARD_AI = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA2MDAgMzkwIj48ZGVmcz48cmFkaWFsR3JhZGllbnQgaWQ9ImIiPjxzdG9wIHN0b3AtY29sb3I9IiMwYjNjMzEiLz48c3RvcCBvZmZzZXQ9IjEiIHN0b3AtY29sb3I9IiMwNjEwMTgiLz48L3JhZGlhbEdyYWRpZW50PjxmaWx0ZXIgaWQ9ImciPjxmZUdhdXNzaWFuQmx1ciBzdGREZXZpYXRpb249IjYiIHJlc3VsdD0ieCIvPjxmZU1lcmdlPjxmZU1lcmdlTm9kZSBpbj0ieCIvPjxmZU1lcmdlTm9kZSBpbj0iU291cmNlR3JhcGhpYyIvPjwvZmVNZXJnZT48L2ZpbHRlcj48L2RlZnM+PHJlY3Qgd2lkdGg9IjYwMCIgaGVpZ2h0PSIzOTAiIHJ4PSIyNCIgZmlsbD0idXJsKCNiKSIvPjxnIHN0cm9rZT0iIzAwRkZGRiIgc3Ryb2tlLXdpZHRoPSIyIiBvcGFjaXR5PSIuNSI+PHBhdGggZD0iTTgwIDcwbDEwMCA1NU01MjAgNzBsLTEwMCA1NU03MCAzMTBsMTIwLTU1TTUzMCAzMTBsLTEyMC01NSIvPjxjaXJjbGUgY3g9IjgwIiBjeT0iNzAiIHI9IjYiIGZpbGw9IiMzOUZGMTQiLz48Y2lyY2xlIGN4PSI1MjAiIGN5PSI3MCIgcj0iNiIgZmlsbD0iIzM5RkYxNCIvPjxjaXJjbGUgY3g9IjcwIiBjeT0iMzEwIiByPSI2IiBmaWxsPSIjMzlGRjE0Ii8+PGNpcmNsZSBjeD0iNTMwIiBjeT0iMzEwIiByPSI2IiBmaWxsPSIjMzlGRjE0Ii8+PC9nPjxnIGZpbHRlcj0idXJsKCNnKSI+PGNpcmNsZSBjeD0iMzAwIiBjeT0iMTkwIiByPSIxMDUiIGZpbGw9IiMwNzE1MWEiIHN0cm9rZT0iIzM5RkYxNCIgc3Ryb2tlLXdpZHRoPSI0Ii8+PHBhdGggZD0iTTI0NyAxNDhjLTI3LTM0IDE2LTY3IDQ3LTQzIDI1LTMxIDczLTMgNTYgMzQgMzkgNiA0MCA1NSAxMCA2NyAxOCAzNy0yOCA2OC01OCA0Mi0zMSAyOC03Ny03LTU1LTQyLTM5LTE0LTM0LTU4IDAtNTh6IiBmaWxsPSJub25lIiBzdHJva2U9IiMwMEZGRkYiIHN0cm9rZS13aWR0aD0iNyIvPjxwYXRoIGQ9Ik0zMDAgMTEwdjE1NU0yNTIgMTU0aDk2TTI2MCAyMTNoODAiIHN0cm9rZT0iIzM5RkYxNCIgc3Ryb2tlLXdpZHRoPSIzIiBvcGFjaXR5PSIuNyIvPjwvZz48ZyBmaWxsPSIjMGQyYTI0IiBzdHJva2U9IiMzOUZGMTQiPjxyZWN0IHg9IjM1IiB5PSIxMTAiIHdpZHRoPSIxMTAiIGhlaWdodD0iNzAiIHJ4PSIxMiIvPjxyZWN0IHg9IjQ1NSIgeT0iMTEwIiB3aWR0aD0iMTEwIiBoZWlnaHQ9IjcwIiByeD0iMTIiLz48cmVjdCB4PSI1NSIgeT0iMjM1IiB3aWR0aD0iMTE1IiBoZWlnaHQ9IjcwIiByeD0iMTIiLz48cmVjdCB4PSI0MzAiIHk9IjIzNSIgd2lkdGg9IjExNSIgaGVpZ2h0PSI3MCIgcng9IjEyIi8+PC9nPjxnIGZpbGw9IiNjYWZmYzIiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxMyI+PHRleHQgeD0iNTIiIHk9IjEzNyI+RElTRUFTRTwvdGV4dD48dGV4dCB4PSI1MiIgeT0iMTU4Ij5DTEFTU0lGSUVSPC90ZXh0Pjx0ZXh0IHg9IjQ3NCIgeT0iMTM3Ij5DT05GSURFTkNFPC90ZXh0Pjx0ZXh0IHg9IjQ3NCIgeT0iMTU4Ij5FTkdJTkU8L3RleHQ+PHRleHQgeD0iNzMiIHk9IjI2MyI+TEVBRjwvdGV4dD48dGV4dCB4PSI3MyIgeT0iMjg0Ij5GRUFUVVJFUzwvdGV4dD48dGV4dCB4PSI0NDgiIHk9IjI2MyI+UklTSzwvdGV4dD48dGV4dCB4PSI0NDgiIHk9IjI4NCI+QU5BTFlTSVM8L3RleHQ+PC9nPjwvc3ZnPg=="
CARD_ACTION = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA2MDAgMzkwIj48ZGVmcz48bGluZWFyR3JhZGllbnQgaWQ9ImIiIHgxPSIwIiB5MT0iMSIgeDI9IjEiPjxzdG9wIHN0b3AtY29sb3I9IiMwNzEyMWEiLz48c3RvcCBvZmZzZXQ9IjEiIHN0b3AtY29sb3I9IiMwZDNhMzUiLz48L2xpbmVhckdyYWRpZW50PjxmaWx0ZXIgaWQ9ImciPjxmZUdhdXNzaWFuQmx1ciBzdGREZXZpYXRpb249IjUiIHJlc3VsdD0ieCIvPjxmZU1lcmdlPjxmZU1lcmdlTm9kZSBpbj0ieCIvPjxmZU1lcmdlTm9kZSBpbj0iU291cmNlR3JhcGhpYyIvPjwvZmVNZXJnZT48L2ZpbHRlcj48L2RlZnM+PHJlY3Qgd2lkdGg9IjYwMCIgaGVpZ2h0PSIzOTAiIHJ4PSIyNCIgZmlsbD0idXJsKCNiKSIvPjxnIHN0cm9rZT0iIzAwRkZGRiIgc3Ryb2tlLXdpZHRoPSIzIiBvcGFjaXR5PSIuNiI+PHBhdGggZD0iTTMwMCAxODVMMTIwIDk1TTMwMCAxODVsMTgwLTkwTTMwMCAxODVMMTMwIDMwME0zMDAgMTg1bDE4MCAxMTUiLz48L2c+PGcgZmlsdGVyPSJ1cmwoI2cpIj48cGF0aCBkPSJNMzAwIDExMmMtNTAgMjItODAgNjctNzAgMTEzIDExIDUwIDY4IDYzIDk4IDI5IDMzLTM3IDIyLTEwNy0yOC0xNDJ6IiBmaWxsPSIjMzlGRjE0IiBvcGFjaXR5PSIuNyIgc3Ryb2tlPSIjY2FmZmMyIiBzdHJva2Utd2lkdGg9IjMiLz48cGF0aCBkPSJNMjgyIDI0NWM4LTQ2IDE2LTc4IDMxLTExMiIgc3Ryb2tlPSIjZThmZmUzIiBzdHJva2Utd2lkdGg9IjUiLz48L2c+PGcgZmlsbD0iIzEwMmQyYiIgc3Ryb2tlPSIjMDBGRkZGIiBzdHJva2Utd2lkdGg9IjIiPjxyZWN0IHg9IjUwIiB5PSI0NSIgd2lkdGg9IjE0NSIgaGVpZ2h0PSI5MCIgcng9IjE4Ii8+PHJlY3QgeD0iNDA1IiB5PSI0NSIgd2lkdGg9IjE0NSIgaGVpZ2h0PSI5MCIgcng9IjE4Ii8+PHJlY3QgeD0iNTAiIHk9IjI1NSIgd2lkdGg9IjE0NSIgaGVpZ2h0PSI5MCIgcng9IjE4Ii8+PHJlY3QgeD0iNDA1IiB5PSIyNTUiIHdpZHRoPSIxNDUiIGhlaWdodD0iOTAiIHJ4PSIxOCIvPjwvZz48ZyBmaWxsPSIjMzlGRjE0IiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtd2VpZ2h0PSJib2xkIiBmb250LXNpemU9IjE3Ij48dGV4dCB4PSI4MCIgeT0iNzgiPldFQVRIRVI8L3RleHQ+PHRleHQgeD0iODIiIHk9IjEwNSI+UklTSzwvdGV4dD48dGV4dCB4PSI0NDAiIHk9Ijc4Ij5BQ1RJT048L3RleHQ+PHRleHQgeD0iNDMyIiB5PSIxMDUiPkdVSURFPC90ZXh0Pjx0ZXh0IHg9IjgzIiB5PSIyOTAiPlNDQU48L3RleHQ+PHRleHQgeD0iNzUiIHk9IjMxNyI+SElTVE9SWTwvdGV4dD48dGV4dCB4PSI0MzUiIHk9IjI5MCI+Q1JPUDwvdGV4dD48dGV4dCB4PSI0MjciIHk9IjMxNyI+Q0FSRTwvdGV4dD48L2c+PC9zdmc+"

with st.sidebar:
    st.title("🌱 AgroCare")
    lang=st.selectbox("Language / ভাষা",["English","বাংলা"],index=0 if st.session_state.lang=="English" else 1)
    st.session_state.lang=lang; t=TXT[lang]
    st.markdown(f'<div class="lang-pill">{"🇬🇧 English Mode" if lang=="English" else "🇧🇩 বাংলা মোড"}</div>',unsafe_allow_html=True)
    page=st.radio(t["nav"],[t["home"],t["scan"],t["weather"],t["model"],t["history"]])
    st.divider(); threshold=st.slider(t["sharp"],20,300,100); notify_change("prev_threshold", threshold, t["sharp"]); st.caption("AgroCare • Final Integrated")

st.markdown(f'<div class="hero" style="--hero-roots:url({HERO_ROOTS})"><div class="badge">{t["tag"]}</div><h1>🌿 AgroCare AI</h1><p>{t["subtitle"]}</p></div>',unsafe_allow_html=True)

if page==t["home"]:
    st.subheader(t["command"])
    st.markdown(f"""<div class="cyber-grid">
      <div class="glow-wrap"><div class="glow-card"><div class="num">01</div><div class="home-card-title">{t['upload']}</div><p class="home-card-desc">{t['upload_d']}</p><div class="home-visual"><img src="{CARD_SCAN}" alt="AI leaf scan"></div></div></div>
      <div class="glow-wrap"><div class="glow-card"><div class="num">02</div><div class="home-card-title">{t['analysis']}</div><p class="home-card-desc">{t['analysis_d']}</p><div class="home-visual"><img src="{CARD_AI}" alt="AI crop analysis"></div></div></div>
      <div class="glow-wrap"><div class="glow-card"><div class="num">03</div><div class="home-card-title">{t['action']}</div><p class="home-card-desc">{t['action_d']}</p><div class="home-visual"><img src="{CARD_ACTION}" alt="Agro action guidance"></div></div></div>
    </div><div class="home-cta">◉ {t['goto']}</div>""",unsafe_allow_html=True)
elif page==t["scan"]:
    st.subheader(t["scan_title"]); st.caption(t["scan_cap"]); up=st.file_uploader(t["drop"],type=["jpg","jpeg","png","bmp","webp"])
    if up:
        img=decode_uploaded_image(up)
        if img is None: st.error(t["badimg"])
        else:
            left,right=st.columns([1,1.25],gap="large")
            with left:
                ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                if ok:
                    b64 = base64.b64encode(enc.tobytes()).decode("ascii")
                    st.markdown(f'<div class="scan-frame"><img src="data:image/jpeg;base64,{b64}"><div class="laser"></div><div class="scan-label">◉ AI VISION SCAN • {t["preview"]}</div></div>', unsafe_allow_html=True); q=check_image_quality(img,threshold); score_disp=bn_num("%.1f" % q["score"], lang); st.markdown(f'<div class="card"><b>{t["quality"]}</b><div class="metric">{score_disp}</div><span class="muted">{t["score"]}</span></div>',unsafe_allow_html=True)
            with right:
                model_path=Path("models/best_model.joblib")
                if not q["is_sharp"]: st.warning(t["blurry"])
                elif not model_path.exists(): st.warning(t["nomodel"]); st.write(t["copy"])
                else:
                    saved=joblib.load(model_path); model=saved["model"] if isinstance(saved,dict) and "model" in saved else saved; feat=np.array(features_for_model(img, model),dtype=float).reshape(1,-1); pred=model.predict(feat)[0]; conf=None; probs=None
                    if hasattr(model,"predict_proba"): probs=model.predict_proba(feat)[0]; conf=float(np.max(probs))
                    cutoff=st.slider(t["reliable"],0.30,0.95,0.55,0.05); notify_change("prev_cutoff", cutoff, t["reliable"])
                    if conf is not None and conf<cutoff:
                        st.markdown(f'<div class="warn"><b>{t["unknown"]}</b><br>{t["unknown_d"]}</div>',unsafe_allow_html=True); st.metric(t["best"],bn_num("%.1f%%" % (conf*100), lang)); upload_key=f"{up.name}:{up.size}:{conf:.6f}:unknown"
                        if st.session_state.saved_upload_key!=upload_key: save_scan(up.name,"Unknown","Unsupported",conf*100,"Review","Unknown / Unsupported",q["score"]); st.session_state.saved_upload_key=upload_key
                    else:
                        crop,disease=pretty_class(str(pred)); display_crop,display_disease=display_crop_disease(crop,disease,lang); sev_en,sev_bn,en,bn=info_for(str(pred)); severity=sev_bn if lang=="বাংলা" else sev_en
                        st.markdown(f'<div class="good"><b>{t["done"]}</b></div>',unsafe_allow_html=True); x,y,z=st.columns(3); x.metric(t["crop"],display_crop); y.metric(t["disease"],display_disease); z.metric(t["confidence"],bn_num("%.1f%%" % (conf*100), lang) if conf is not None else "—"); st.progress(conf if conf is not None else 0); st.markdown(f'<div class="card"><b>{t["severity"]}</b><div class="metric">{severity}</div></div>',unsafe_allow_html=True); st.markdown("#### "+t["recommend"]); st.success(bn if lang=="বাংলা" else en); upload_key=f"{up.name}:{up.size}:{pred}:{conf if conf is not None else 0:.6f}"
                        if st.session_state.saved_upload_key!=upload_key: save_scan(up.name,crop,disease,(conf*100) if conf is not None else None,sev_en,"Detected",q["score"]); st.session_state.saved_upload_key=upload_key
                    if probs is not None:
                        classes=model.classes_; top=np.argsort(probs)[::-1][:5]; st.markdown("#### "+t["matches"]); st.dataframe(pd.DataFrame({t["class"]:[display_class_label(classes[i],lang) for i in top],t["prob"]:[bn_num("%.2f%%" % (probs[i]*100), lang) for i in top]}),use_container_width=True,hide_index=True)
elif page==t["weather"]:
    st.subheader(t["weather_title"]); c1,c2,c3=st.columns(3); temp=c1.number_input(t["temp"],value=28.0); hum=c2.number_input(t["hum"],0,100,75); rain=c3.number_input(t["rain"],0.0,500.0,0.0)
    notify_change("prev_temp", temp, t["temp"]); notify_change("prev_hum", hum, t["hum"]); notify_change("prev_rain", rain, t["rain"])
    level,reasons=weather_risk(temp,hum,rain); breakdown=weather_risk_breakdown(temp,hum,rain)
    if lang=="বাংলা": level,reasons=bn_weather(level,reasons)
    st.markdown(f'<div class="card"><b>{t["assessment"]}</b><div class="metric">{level}</div><p>{", ".join(reasons) if reasons else t["no_risk"]}</p></div>',unsafe_allow_html=True); st.caption(t["weather_note"])
    factor_labels={"humidity":t["factor_humidity"],"temp":t["factor_temp"],"rainfall":t["factor_rain"]}
    st.markdown(f"#### {t['factors_title']}")
    for f in breakdown:
        st.markdown(f'<div class="card" style="padding:12px 16px;margin-bottom:8px"><b>{factor_labels[f["key"]]}</b> — {t["met"] if f["met"] else t["notmet"]}<br><span class="muted">{f["detail"]}</span></div>', unsafe_allow_html=True)
elif page==t["model"]:
    st.subheader(t["model_title"]); st.info(t["model_info"]); st.success(t["use_model"])
else:
    st.subheader(t["history_title"]); summary=get_summary(); a,b,c,d=st.columns(4); a.metric(t["total"],bn_num(summary["total"],lang)); b.metric(t["detected"],bn_num(summary["detected"],lang)); c.metric(t["unknown_m"],bn_num(summary["unknown"],lang)); d.metric(t["avg"],bn_num("%.1f%%" % summary["avg_confidence"], lang)); rows=get_history(200)
    if rows:
        df=pd.DataFrame(rows);
        if lang=="বাংলা":
            df["crop"]=df["crop"].map(lambda x: CROP_BN_NORM.get(_norm_key(x), "অজানা" if str(x)=="Unknown" else x))
            df["disease"]=df["disease"].map(lambda x: DISEASE_BN_NORM.get(_norm_key(x), "সমর্থিত নয়" if str(x)=="Unsupported" else x))
            df["severity"]=df["severity"].map(lambda x: STATUS_BN.get(str(x),x))
            df["status"]=df["status"].map(lambda x: STATUS_BN.get(str(x),x))
            df["confidence"]=df["confidence"].map(lambda x: bn_num("%.1f" % x, lang) if pd.notna(x) else x)
            df["sharpness"]=df["sharpness"].map(lambda x: bn_num("%.1f" % x, lang) if pd.notna(x) else x)
        names={"scan_time":"স্ক্যানের সময়" if lang=="বাংলা" else "Scan Date/Time","image_name":"ছবির নাম" if lang=="বাংলা" else "Image Name","crop":"ফসল" if lang=="বাংলা" else "Crop","disease":"রোগ" if lang=="বাংলা" else "Disease","confidence":"আত্মবিশ্বাস %" if lang=="বাংলা" else "Confidence %","severity":"তীব্রতা" if lang=="বাংলা" else "Severity","status":"অবস্থা" if lang=="বাংলা" else "Status","sharpness":"স্পষ্টতা" if lang=="বাংলা" else "Sharpness"}; df=df.rename(columns=names); st.markdown("### "+t["recent"]); st.dataframe(df.drop(columns=["id"]),use_container_width=True,hide_index=True); csv=df.drop(columns=["id"]).to_csv(index=False).encode("utf-8-sig"); c1,c2=st.columns(2)
        with c1: st.download_button(t["export"],csv,"agrocare_scan_history.csv","text/csv")
        with c2:
            if st.button(t["clear"]): clear_history(); st.session_state.saved_upload_key=None; st.rerun()
    else: st.markdown(f'<div class="card"><b>{t["empty"]}</b><p class="muted">{t["empty_d"]}</p></div>',unsafe_allow_html=True)

st.divider(); st.caption(t["footer"])
