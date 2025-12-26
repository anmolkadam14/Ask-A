from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import base64
import os, json, re, time, sys, math
import sqlite3
from datetime import datetime
import speech_recognition as sr
import pyttsx3

EXIT_WORDS = {"bye", "exit", "quit", "goodbye", "by", "byyy"}
MEM_FILE = "memory.json"
ASR_LANG = "en-IN"
LISTEN_TIMEOUT = 6
PHRASE_TIME_LIMIT = 12
TTS_RATE = 180
TEXT_MODEL = "gemini-2.0-flash-001"
IMAGE_MODEL = "gemini-2.0-flash-preview-image-generation"
INR_SYMBOL = "₹"

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

conn = sqlite3.connect("chat_history.db")
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

def save_chat(user_input: str, bot_reply: str | None, image_path: str | None = None):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO chat_log (timestamp, user_input, bot_reply, image_path) VALUES (?, ?, ?, ?)",
        (ts, user_input, bot_reply, image_path),
    )
    conn.commit()

def load_memory():
    if not os.path.exists(MEM_FILE):
        return {"username": None, "bot_name": None, "mic_index": None, "finance_mode": False}
    try:
        with open(MEM_FILE, "r", encoding="utf-8") as f:
            mem = json.load(f)
            mem.setdefault("mic_index", None)
            mem.setdefault("finance_mode", False)
            return mem
    except Exception:
        return {"username": None, "bot_name": None, "mic_index": None, "finance_mode": False}

def save_memory(mem: dict):
    try:
        with open(MEM_FILE, "w", encoding="utf-8") as f:
            json.dump(mem, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def is_asking_bot_name(t: str) -> bool:
    t = t.lower().strip()
    pats = [
        r"\bwhat('?| i)s your name\b",
        r"\byour name\b",
        r"\bbot name\b",
        r"\bwho are you\b",
        r"\bnaam kya hai\b",
        r"\btumhara naam\b",
        r"\baapka naam\b",
        r"\bतुम्हारा नाम\b",
        r"\bतुझे क्या कहते हैं\b",
        r"\bनाव काय आहे\b",
        r"\bतुम्हें क्या कहते हैं\b",
    ]
    return any(re.search(p, t) for p in pats)

mem = load_memory()
username = mem.get("username") or input("Your name: ").strip() or "friend"
if mem.get("username") != username:
    mem["username"] = username
    save_memory(mem)

bot_name = mem.get("bot_name") or (input("Bot name (default: Ask-A): ").strip() or "ANU")
if mem.get("bot_name") != bot_name:
    mem["bot_name"] = bot_name
    save_memory(mem)

def list_mics():
    try:
        return sr.Microphone.list_microphone_names()
    except Exception:
        return []

def pick_mic_interactive() -> int | None:
    names = list_mics()
    if not names:
        print(f"{bot_name}: No microphones detected.")
        return None
    print(f"{bot_name}: Available microphones:")
    for i, n in enumerate(names):
        print(f"  [{i}] {n}")
    choice = input(f"{username}: Choose mic index (Enter for default): ").strip()
    if choice == "":
        return None
    try:
        idx = int(choice)
        if 0 <= idx < len(names):
            return idx
    except Exception:
        pass
    print(f"{bot_name}: Invalid choice. Using default device.")
    return None

if mem.get("mic_index") is None:
    names = list_mics()
    auto = None
    for i, n in enumerate(names):
        ln = (n or "").lower()
        if "microphone" in ln or "mic" in ln or "realtek" in ln or "array" in ln:
            auto = i
            break
    mem["mic_index"] = auto if auto is not None else pick_mic_interactive()
    save_memory(mem)

MIC_INDEX = mem.get("mic_index")

def set_mic_index(idx: int | None):
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
        f"/sip <monthly> <rate%> <years> — future value (e.g., /sip 5000 12 10)\n"
        f"/goal <target> <years> <rate%> — required monthly SIP (e.g., /goal 1000000 10 12)\n"
        f"/emi <principal> <rate%> <years> — loan EMI (e.g., /emi 800000 9 5)\n"
        f"/cagr <initial> <final> <years> — CAGR (e.g., /cagr 100000 250000 5)\n"
        f"Note: {finance_disclaimer()}"
    )

def system_instruction_content_general(bot_name, username):
    return types.Content(
        role="system",
        parts=[types.Part.from_text(
            text=(
                f"You are an AI assistant named '{bot_name}'. Refer to yourself as {bot_name}. "
                f"Always address the user as '{username}'. Be concise, friendly, and helpful."
            )
        )]
    )

def system_instruction_content_finance(bot_name, username):
    return types.Content(
        role="system",
        parts=[types.Part.from_text(
            text=(
                f"You are a finance education assistant named '{bot_name}'. "
                f"Always address the user as '{username}'. "
                "Give clear, structured, India-relevant explanations with examples when useful. "
                "Do NOT provide personalized investment advice, stock tips, or tax filings. "
                "Always include this line at the end of finance responses: "
                f"'{finance_disclaimer()}'"
            )
        )]
    )

api_key ="AIzaSyAC06KN6DooRl9DyQyFUO75PspnHTeUZao"or os.getenv("GOOGLE_GENAI_API_KEY")
client = genai.Client(api_key=api_key)

def ask_model(prompt: str, finance_mode: bool) -> str:
    sys_inst = system_instruction_content_finance(bot_name, username) if finance_mode else system_instruction_content_general(bot_name, username)
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
    files = []
    caption = None
    ts = int(time.time())
    idx = 1
    for cand in (getattr(r, "candidates", None) or []):
        parts = getattr(cand.content, "parts", []) or []
        for part in parts:
            if getattr(part, "text", None) and not caption:
                caption = part.text.strip()
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                raw = _decode_inline_data(inline.data)
                if not raw:
                    continue
                mime = (getattr(inline, "mime_type", None) or "image/png").lower()
                ext = ".png" if "png" in mime else ".jpg"
                fname = f"gemini_image_{ts}_{idx}{ext}"
                try:
                    img = Image.open(BytesIO(raw))
                    img.save(fname)
                    files.append(fname)
                    idx += 1
                except Exception:
                    pass
    return files, caption

finance_mode = bool(mem.get("finance_mode"))
print(f"{bot_name}: Hi {username}! Press Enter to talk by mic, or type. (type 'bye' to exit)")
print(f"{bot_name}: /botname <new>  |  /image <prompt>  |  /mic (choose device)")
print(f"{bot_name}: /finon  /finoff  /finhelp  /sip  /goal  /emi  /cagr")

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
        idx = pick_mic_interactive()
        set_mic_index(idx)
        names = list_mics()
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

    if is_asking_bot_name(prompt):
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

    reply = ask_model(prompt, finance_mode)
    if finance_mode and reply and finance_disclaimer() not in reply:
        reply = f"{reply}\n\n{finance_disclaimer()}"
    print(f"{bot_name}:", reply or "(no text in response)")
    speak(reply or "")
    save_chat(user_input=prompt, bot_reply=reply or "")
