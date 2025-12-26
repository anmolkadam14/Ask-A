import os, re, json, time, math, base64, sys, sqlite3
from io import BytesIO
from datetime import datetime
from typing import List, Optional
from fastapi.staticfiles import StaticFiles
from fastapi import Request
import os
from fastapi import FastAPI, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


os.makedirs("static", exist_ok=True)


app.mount("/static", StaticFiles(directory="static"), name="static")



@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/image")
def create_image(payload: dict = Body(...), request: Request = None):
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        return {"files": [], "caption": "⚠️ No prompt provided"}

    files, cap = generate_image(prompt)   # use your existing generate_image()
    base = str(request.base_url).rstrip("/")

    urls = []
    for f in files:
        if not f.startswith("http"):
            fixed = f.replace("\\", "/")
            url = f"{base}/{fixed}"
        else:
            url = f
        urls.append(url)

    save_chat(user_input=f"/image {prompt}", bot_reply=cap or "", image_path=";".join(files))
    return {"files": urls, "caption": cap or ""}


os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")



from google import genai
from google.genai import types


from PIL import Image


import speech_recognition as sr
import pyttsx3


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn


from dotenv import load_dotenv
load_dotenv()



EXIT_WORDS = {"bye", "exit", "quit", "goodbye", "by", "byyy"}
MEM_FILE = "memory.json"
DB_FILE = "chat_history.db"
ASR_LANG = "en-IN"
LISTEN_TIMEOUT = 6
PHRASE_TIME_LIMIT = 12
TTS_RATE = 180
TEXT_MODEL = "gemini-2.0-flash-001"
IMAGE_MODEL = "gemini-2.0-flash-preview-image-generation"
INR_SYMBOL = "₹"


conn = sqlite3.connect(DB_FILE, check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS chat_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    user_input TEXT,
    bot_reply TEXT,
    image_path TEXT
)
""")
conn.commit()

def save_chat(user_input: str, bot_reply: Optional[str], image_path: Optional[str] = None):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO chat_log (timestamp, user_input, bot_reply, image_path) VALUES (?, ?, ?, ?)",
        (ts, user_input, bot_reply, image_path),
    )
    conn.commit()

def load_memory():
    if not os.path.exists(MEM_FILE):
        return {"username": None, "bot_name": "ANU", "mic_index": None, "finance_mode": False}
    try:
        with open(MEM_FILE, "r", encoding="utf-8") as f:
            mem = json.load(f)
            mem.setdefault("bot_name", "ANU")
            mem.setdefault("mic_index", None)
            mem.setdefault("finance_mode", False)
            return mem
    except Exception:
        return {"username": None, "bot_name": "ANU", "mic_index": None, "finance_mode": False}

def save_memory(mem: dict):
    try:
        with open(MEM_FILE, "w", encoding="utf-8") as f:
            json.dump(mem, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

mem = load_memory()
username = mem.get("username") or "friend"
bot_name = "CHAT WITH VIJAY"
MIC_INDEX = mem.get("mic_index")

# ========= TTS =========
def init_tts():
    try:
        eng = pyttsx3.init("sapi5")
    except Exception:
        try:
            eng = pyttsx3.init()
        except Exception:
            eng = None
    if eng:
        eng.setProperty("rate", TTS_RATE)
    return eng

engine = init_tts()

def speak(text: str):
    if not text or engine is None:
        return
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception:
        pass

# ========= Mic =========
def list_mics():
    try:
        return sr.Microphone.list_microphone_names()
    except Exception:
        return []

def set_mic_index(idx: Optional[int]):
    global MIC_INDEX
    MIC_INDEX = idx
    mem["mic_index"] = idx
    save_memory(mem)

def listen() -> str:
    r = sr.Recognizer()
    r.dynamic_energy_threshold = True
    r.energy_threshold = 150
    try:
        with sr.Microphone(device_index=MIC_INDEX) as source:
            print("Listening...")
            r.adjust_for_ambient_noise(source, duration=0.5)
            audio = r.listen(source, timeout=LISTEN_TIMEOUT, phrase_time_limit=PHRASE_TIME_LIMIT)
    except sr.WaitTimeoutError:
        print(f"{bot_name}: No speech detected.")
        return ""
    except OSError:
        print(f"{bot_name}: No default input device. Use /mic to select one.")
        return ""
    except Exception:
        return ""
    try:
        text = r.recognize_google(audio, language=ASR_LANG)
        print("You said:", text)
        return text
    except sr.UnknownValueError:
        speak("Sorry, I could not understand. Please try again.")
        return ""
    except sr.RequestError:
        speak("Speech service error. Please check your internet.")
        return ""

# ========= Finance utils =========
def format_money(v: float) -> str:
    try:
        if v >= 1e7:
            return f"{INR_SYMBOL}{v/1e7:.2f} Cr"
        if v >= 1e5:
            return f"{INR_SYMBOL}{v/1e5:.2f} L"
        return f"{INR_SYMBOL}{v:,.0f}"
    except Exception:
        return f"{INR_SYMBOL}{v}"

def finance_disclaimer() -> str:
    return ("Educational only—NOT financial advice. Markets/taxes change. "
            "Consider consulting a SEBI-registered advisor for personalized guidance.")

def fv_sip(monthly: float, rate_pct: float, years: float) -> float:
    r = rate_pct / 100.0
    n = 12
    t = years
    i = r / n
    return monthly * (((1 + i) ** (n * t) - 1) / i) * (1 + i)

def req_sip_for_goal(target: float, rate_pct: float, years: float) -> float:
    r = rate_pct / 100.0
    n = 12
    t = years
    i = r / n
    denom = (((1 + i) ** (n * t) - 1) / i) * (1 + i)
    return target / denom if denom > 0 else float('inf')

def emi(principal: float, rate_pct: float, years: float) -> float:
    m = years * 12
    i = (rate_pct / 100.0) / 12.0
    if i == 0:
        return principal / m
    return principal * i * (1 + i) ** m / ((1 + i) ** m - 1)

def cagr(initial: float, final: float, years: float) -> float:
    if initial <= 0 or years <= 0:
        return float('nan')
    return (final / initial) ** (1 / years) - 1

def parse_floats(parts, count):
    try:
        vals = [float(x.replace(",", "")) for x in parts[:count]]
        if len(vals) < count:
            return None
        return vals
    except Exception:
        return None

def finance_help_text() -> str:
    return (
        "Finance commands:\n"
        "/finon  — enable finance mode\n"
        "/finoff — disable finance mode\n"
        "/finhelp — show this help\n"
        "/sip <monthly> <rate%> <years>\n"
        "/goal <target> <years> <rate%>\n"
        "/emi <principal> <rate%> <years>\n"
        "/cagr <initial> <final> <years>\n"
        f"Note: {finance_disclaimer()}"
    )

# ========= Prompts =========
def system_instruction_content_general(_bot, _user):
    return types.Content(
        role="system",
        parts=[types.Part.from_text(
            text=(
                f"You are an AI assistant named '{_bot}'. Refer to yourself as {_bot}. "
                f"Always address the user as '{_user}'. Be concise, friendly, and helpful."
            )
        )]
    )

def system_instruction_content_finance(_bot, _user):
    return types.Content(
        role="system",
        parts=[types.Part.from_text(
            text=(
                f"You are a finance education assistant named '{_bot}'. "
                f"Always address the user as '{_user}'. "
                "Give clear, structured, India-relevant explanations with examples when useful. "
                "Do NOT provide personalized investment advice, stock tips, or tax filings. "
                "Always include this line at the end of finance responses: "
                f"'{finance_disclaimer()}'"
            )
        )]
    )

# ========= GenAI Client =========
def get_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_GENAI_API_KEY")or "AIzaSyAC06KN6DooRl9DyQyFUO75PspnHTeUZao"
    if not api_key:
        print("ERROR: Set GOOGLE_GENAI_API_KEY")
        sys.exit(1)
    return genai.Client(api_key=api_key)

client = get_client()

def ask_model(prompt: str, finance_mode: bool, _bot: str, _user: str) -> str:
    sys_inst = system_instruction_content_finance(_bot, _user) if finance_mode else system_instruction_content_general(_bot, _user)
    r = client.models.generate_content(
        model=TEXT_MODEL,
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
        config=types.GenerateContentConfig(
            system_instruction=sys_inst,
            temperature=0.7 if finance_mode else 0.8,
            top_p=0.95,
            max_output_tokens=2048
        ),
    )
    return (getattr(r, "text", "") or "").strip()

def _decode_inline_data(data):
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(data, str):
        if data.startswith("data:"):
            try:
                data = data.split(",", 1)[1]
            except Exception:
                pass
        try:
            return base64.b64decode(data)
        except Exception:
            return b""
    return b""

def generate_image(prompt: str):
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    r = client.models.generate_content(
        model=IMAGE_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])
    )
    files, caption = [], None
    ts, idx = int(time.time()), 1
    for cand in (getattr(r, "candidates", None) or []):
        for part in (getattr(cand.content, "parts", []) or []):
            if getattr(part, "text", None) and not caption:
                caption = part.text.strip()
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                raw = _decode_inline_data(inline.data)
                if not raw: continue
                mime = (getattr(inline, "mime_type", None) or "image/png").lower()
                ext = ".png" if "png" in mime else ".jpg"
                fname = f"static/gemini_image_{ts}_{idx}{ext}"   # <- save into static/
                try:
                    img = Image.open(BytesIO(raw))
                    img.save(fname)
                    files.append(fname)
                    idx += 1
                except Exception:
                    pass
    return files, caption


# ========= FastAPI =========
app = FastAPI(title="Ask-A– Chat + Finance API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ChatIn(BaseModel):
    prompt: str
    finance_mode: bool = False
    username: Optional[str] = None
    bot_name: Optional[str] = None

class ChatOut(BaseModel):
    reply: str

class ImageIn(BaseModel):
    prompt: str

class SIPIn(BaseModel):
    monthly: float
    rate_pct: float
    years: float

class GoalIn(BaseModel):
    target: float
    years: float
    rate_pct: float

class EMIIn(BaseModel):
    principal: float
    rate_pct: float
    years: float

class CAGRIn(BaseModel):
    initial: float
    final: float
    years: float

@app.get("/")
def index():
    return FileResponse("index.html")

@app.get("/health")
def health():
    return {"status": "ok", "bot": bot_name, "user": username}

@app.post("/chat", response_model=ChatOut)
def chat(req: ChatIn):
    _user = (req.username or username).strip() or "friend"
    _bot = (req.bot_name or bot_name).strip() or "CHAT WITH VIJAY"
    reply = ask_model(req.prompt, req.finance_mode, _bot, _user)
    if req.finance_mode and reply and finance_disclaimer() not in reply:
        reply = f"{reply}\n\n{finance_disclaimer()}"
    save_chat(req.prompt, reply)
    return {"reply": reply or "(no text in response)"}

@app.post("/image")
def image(req: ImageIn):
    files, cap = generate_image(req.prompt)
    if files:
        for f in files:
            save_chat(user_input=f"/image {req.prompt}", bot_reply=cap or "", image_path=f)
    else:
        save_chat(user_input=f"/image {req.prompt}", bot_reply="(image generation failed)")
    return {"files": files, "caption": cap}

@app.post("/finance/sip")
def api_sip(req: SIPIn):
    future = fv_sip(req.monthly, req.rate_pct, req.years)
    msg = (f"Projected corpus for SIP {format_money(req.monthly)}/mo @ {req.rate_pct:.2f}% for {req.years:.1f} yrs ≈ {format_money(future)}.\n"
           f"Assumes monthly compounding. {finance_disclaimer()}")
    save_chat(user_input=f"/sip {req.monthly} {req.rate_pct} {req.years}", bot_reply=msg)
    return {"future_value": future, "message": msg}

@app.post("/finance/goal")
def api_goal(req: GoalIn):
    monthly = req_sip_for_goal(req.target, req.rate_pct, req.years)
    msg = (f"To reach {format_money(req.target)} in {req.years:.1f} yrs @ {req.rate_pct:.2f}% expected return, "
           f"required SIP ≈ {format_money(monthly)}/mo.\n"
           f"Assumes monthly compounding. {finance_disclaimer()}")
    save_chat(user_input=f"/goal {req.target} {req.years} {req.rate_pct}", bot_reply=msg)
    return {"required_sip": monthly, "message": msg}

@app.post("/finance/emi")
def api_emi(req: EMIIn):
    m = emi(req.principal, req.rate_pct, req.years)
    total_pay = m * req.years * 12
    interest = total_pay - req.principal
    msg = (f"EMI ≈ {format_money(m)} for {req.years:.1f} yrs @ {req.rate_pct:.2f}% on {format_money(req.principal)}.\n"
           f"Total Payable ≈ {format_money(total_pay)} (Interest ≈ {format_money(interest)}). {finance_disclaimer()}")
    save_chat(user_input=f"/emi {req.principal} {req.rate_pct} {req.years}", bot_reply=msg)
    return {"emi": m, "total_pay": total_pay, "interest": interest, "message": msg}

@app.post("/finance/cagr")
def api_cagr(req: CAGRIn):
    rate = cagr(req.initial, req.final, req.years) * 100.0
    if math.isnan(rate):
        msg = "Invalid inputs for CAGR."
    else:
        msg = (f"CAGR from {format_money(req.initial)} to {format_money(req.final)} over {req.years:.1f} yrs ≈ {rate:.2f}% p.a. "
               f"{finance_disclaimer()}")
    save_chat(user_input=f"/cagr {req.initial} {req.final} {req.years}", bot_reply=msg)
    return {"cagr_pct": rate, "message": msg}

# ========= CLI mode =========
def run_cli():
    global username, bot_name

    # Ask once
    nm = input("Your name (blank to keep current): ").strip()
    if nm:
        username = nm
        mem["username"] = username
        save_memory(mem)

    bn = input("Bot name (default ANU, blank to keep current): ").strip()
    if bn:
        bot_name = bn
        mem["bot_name"] = bot_name
        save_memory(mem)

    print(f"{bot_name}: Hi {username}! Press Enter to talk by mic, or type. (type 'bye' to exit)")
    print(f"{bot_name}: /botname <new>  |  /image <prompt>  |  /mic (choose device)")
    print(f"{bot_name}: /finon  /finoff  /finhelp  /sip  /goal  /emi  /cagr")

    finance_mode = bool(mem.get("finance_mode"))

    while True:
        prompt = input(f"{username}: ").strip()
        if not prompt:
            print(f"{bot_name}: Listening…")
            spoken = listen().strip()
            if not spoken:
                print(f"{bot_name}: I didn't catch that.")
                continue
            prompt = spoken
            print(f"{username} (mic): {prompt}")

        if prompt.lower() in EXIT_WORDS:
            msg = f"Bye {username}! Have a nice day."
            print(f"{bot_name}: {msg}")
            speak(msg)
            break

        # Commands
        if prompt.lower().startswith("/botname "):
            new_name = prompt.split(" ", 1)[1].strip()
            if new_name:
                bot_name = new_name
                mem["bot_name"] = bot_name
                save_memory(mem)
                say = f"Done! I’ll go by {bot_name} now."
                print(f"{bot_name}: {say}")
                speak(say)
                save_chat(user_input=f"/botname {new_name}", bot_reply=say)
            else:
                msg = "Please provide a name after /botname"
                print(f"{bot_name}: {msg}")
                speak(msg)
            continue

        if prompt.lower() == "/mic":
            names = list_mics()
            if not names:
                print(f"{bot_name}: No microphones detected.")
                continue
            print(f"{bot_name}: Available microphones:")
            for i, n in enumerate(names):
                print(f"  [{i}] {n}")
            choice = input(f"{username}: Choose mic index (Enter for default): ").strip()
            idx = None if choice == "" else (int(choice) if choice.isdigit() else None)
            set_mic_index(idx)
            chosen = "default device" if idx is None else f"[{idx}] {names[idx]}" if names and idx is not None and 0 <= idx < len(names) else "device set"
            msg = f"Mic set to {chosen}."
            print(f"{bot_name}: {msg}")
            speak(msg)
            continue

        if prompt.lower().startswith("/image "):
            img_prompt = prompt.split(" ", 1)[1].strip()
            if not img_prompt:
                msg = "Please provide a prompt after /image"
                print(f"{bot_name}: {msg}")
                speak(msg)
                continue
            files, cap = generate_image(img_prompt)
            if files:
                for f in files:
                    print(f"{bot_name}: Image saved as {f}")
                    save_chat(user_input=prompt, bot_reply=cap or "", image_path=f)
            else:
                print(f"{bot_name}: I couldn't generate an image.")
                save_chat(user_input=prompt, bot_reply="(image generation failed)")
            if cap:
                print(f"{bot_name}: Caption: {cap}")
                speak(cap)
            continue

        if prompt.lower() == "/finon":
            finance_mode = True
            mem["finance_mode"] = True
            save_memory(mem)
            msg = "Finance mode is ON. Use /finhelp to see calculators."
            print(f"{bot_name}: {msg}")
            speak(msg)
            save_chat(user_input=prompt, bot_reply=msg)
            continue

        if prompt.lower() == "/finoff":
            finance_mode = False
            mem["finance_mode"] = False
            save_memory(mem)
            msg = "Finance mode is OFF. Back to general chat."
            print(f"{bot_name}: {msg}")
            speak(msg)
            save_chat(user_input=prompt, bot_reply=msg)
            continue

        if prompt.lower() == "/finhelp":
            msg = finance_help_text()
            print(f"{bot_name}: {msg}")
            speak("Finance help opened.")
            save_chat(user_input=prompt, bot_reply=msg)
            continue

        if prompt.lower().startswith("/sip "):
            parts = prompt.split()[1:]
            vals = parse_floats(parts, 3)
            if not vals:
                msg = "Usage: /sip <monthly> <rate%> <years>"
                print(f"{bot_name}: {msg}")
                speak(msg)
                continue
            monthly, rate, years = vals
            future = fv_sip(monthly, rate, years)
            msg = (f"Projected corpus for SIP {format_money(monthly)}/mo @ {rate:.2f}% for {years:.1f} yrs ≈ {format_money(future)}.\n"
                   f"Assumes monthly compounding. {finance_disclaimer()}")
            print(f"{bot_name}: {msg}")
            speak(f"Projected corpus is {format_money(future)}.")
            save_chat(user_input=prompt, bot_reply=msg)
            continue

        if prompt.lower().startswith("/goal "):
            parts = prompt.split()[1:]
            vals = parse_floats(parts, 3)
            if not vals:
                msg = "Usage: /goal <target> <years> <rate%>"
                print(f"{bot_name}: {msg}")
                speak(msg)
                continue
            target, years, rate = vals
            monthly = req_sip_for_goal(target, rate, years)
            msg = (f"To reach {format_money(target)} in {years:.1f} yrs @ {rate:.2f}% expected return, "
                   f"required SIP ≈ {format_money(monthly)}/mo.\n"
                   f"Assumes monthly compounding. {finance_disclaimer()}")
            print(f"{bot_name}: {msg}")
            speak(f"Required monthly SIP is {format_money(monthly)}.")
            save_chat(user_input=prompt, bot_reply=msg)
            continue

        if prompt.lower().startswith("/emi "):
            parts = prompt.split()[1:]
            vals = parse_floats(parts, 3)
            if not vals:
                msg = "Usage: /emi <principal> <rate%> <years>"
                print(f"{bot_name}: {msg}")
                speak(msg)
                continue
            principal, rate, years = vals
            m = emi(principal, rate, years)
            total_pay = m * years * 12
            interest = total_pay - principal
            msg = (f"EMI ≈ {format_money(m)} for {years:.1f} yrs @ {rate:.2f}% on {format_money(principal)}.\n"
                   f"Total Payable ≈ {format_money(total_pay)} (Interest ≈ {format_money(interest)}). {finance_disclaimer()}")
            print(f"{bot_name}: {msg}")
            speak(f"Your EMI is {format_money(m)} per month.")
            save_chat(user_input=prompt, bot_reply=msg)
            continue

        if prompt.lower().startswith("/cagr "):
            parts = prompt.split()[1:]
            vals = parse_floats(parts, 3)
            if not vals:
                msg = "Usage: /cagr <initial> <final> <years>"
                print(f"{bot_name}: {msg}")
                speak(msg)
                continue
            initial, final_amt, years = vals
            rate = cagr(initial, final_amt, years) * 100.0
            if math.isnan(rate):
                msg = "Invalid inputs for CAGR."
            else:
                msg = (f"CAGR from {format_money(initial)} to {format_money(final_amt)} over {years:.1f} yrs ≈ {rate:.2f}% p.a. "
                       f"{finance_disclaimer()}")
            print(f"{bot_name}: {msg}")
            speak(f"CAGR is approximately {rate:.2f} percent per annum." if not math.isnan(rate) else "Invalid CAGR inputs.")
            save_chat(user_input=prompt, bot_reply=msg)
            continue

    
        if re.search(r"\bwhat('?| i)s your name\b|\byour name\b|\bbot name\b|\bwho are you\b", prompt.lower()):
            print(f"{bot_name}: My name is {bot_name}. Do you want to rename me? (yes/no)")
            ans = input(f"{username}: ").strip().lower()
            if ans in {"yes", "y"}:
                print(f"{bot_name}: What should my new name be?")
                new_name = input(f"{username}: ").strip() or bot_name
                bot_name = new_name
                mem["bot_name"] = bot_name
                save_memory(mem)
                msg = f"Noted. I’m {bot_name} now."
                print(f"{bot_name}: {msg}")
                speak(msg)
                save_chat(user_input="rename flow", bot_reply=msg)
            else:
                msg = f"Cool. I’ll stay {bot_name}."
                print(f"{bot_name}: {msg}")
                speak(msg)
                save_chat(user_input="rename flow", bot_reply=msg)
            continue

        # Normal chat
        reply = ask_model(prompt, finance_mode, bot_name, username)
        if finance_mode and reply and finance_disclaimer() not in reply:
            reply = f"{reply}\n\n{finance_disclaimer()}"
        print(f"{bot_name}:", reply or "(no text in response)")
        speak(reply or "")
        save_chat(user_input=prompt, bot_reply=reply or "")


if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "server"
    if mode == "cli":
        run_cli()
    else:
        uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
