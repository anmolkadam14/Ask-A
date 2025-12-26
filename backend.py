from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import math

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Models ----------
class Message(BaseModel):
    role: str
    text: str

class Goal(BaseModel):
    name: str
    target: float = 0.0          # total amount needed (₹)
    monthly: float = 0.0         # planned monthly contribution (₹)
    bucket: str = "equity"       # "equity" or "debt"

class Payload(BaseModel):
    messages: List[Message] = Field(default_factory=list)
    profile: Dict = Field(default_factory=dict)
    goals: List[Goal] = Field(default_factory=list)

# ---------- Helpers ----------
def inr_commas(n: float) -> str:
    """Indian numbering format: 12,34,567"""
    try:
        x = int(round(n))
    except Exception:
        x = 0
    s = str(abs(x))
    if len(s) <= 3:
        out = s
    else:
        out = s[-3:]
        s = s[:-3]
        while s:
            out = s[-2:] + "," + out
            s = s[:-2]
    return ("-₹" if x < 0 else "₹") + out

def clamp(v, a, b):
    return max(a, min(v, b))

def alloc(risk: float):
    """Simple thumb-rule allocation from risk 0–10."""
    eq = clamp(0.30 + 0.05 * risk, 0.30, 0.80)
    debt = clamp(0.60 - 0.04 * risk, 0.15, 0.70)
    cash = max(0.05, 1 - eq - debt)
    s = eq + debt + cash
    return {"eq": round(eq / s, 2), "debt": round(debt / s, 2), "cash": round(cash / s, 2)}

def fv_annuity_due(months: int, monthly: float, annual_rate: float) -> float:
    r = annual_rate / 12.0
    if r == 0:
        return monthly * months
    return monthly * (((1 + r) ** months - 1) / r) * (1 + r)

def months_needed_for_target(target: float, monthly: float, annual_rate: float) -> int:
    """Solve for n in annuity-due FV >= target."""
    if monthly <= 0:
        return math.inf
    r = annual_rate / 12.0
    if r == 0:
        return math.ceil(target / monthly)
    
