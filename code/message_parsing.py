"""
Hybrid Message Parser and Context Event Integrator.

This module implements a two-step hybrid pipeline:
1. High-precision NLP/Regex matching across 4 financial categories:
   - Cash flow amendment (AMEND_SALARY_DATE, AMEND_SALARY_AMOUNT)
   - Cash flow termination (STOP_RECURRING)
   - Phantom inflow exclusion (EXCLUDE_EVENT)
   - Confirmed windfall (CONFIRM_INFLOW)
   Returns confidence: 1.0 on match.
2. LLM fallback (ChatOpenAI) as an escape hatch when no pattern matches:
   Returns confidence: 0.85 on successful parse, else 0.0 (uncertain).

The resulting actions are directly applied to ctx["events"] to prepare
the financial cash flow stream before feeding into the 90-day balance simulator.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union




# ----------------------------------------------------------------------
# Step 1: Regex Pattern Rules Dictionary
# ----------------------------------------------------------------------
REGEX_RULES: Dict[str, Dict[str, List[str]]] = {
    "Cash flow amendment": {
        "AMEND_SALARY_DATE": [
            r"(?:confirmed salary is now expected on|replaces the payroll date shown|revised date for anything you normally pay)\s*([0-9]{4}-[0-9]{2}-[0-9]{2})?",
            r"(?:gaji yang sudah dikonfirmasi kini diperkirakan masuk pada|menggantikan tanggal penggajian yang tertera|gunakan tanggal terbaru ini)\s*([0-9]{4}-[0-9]{2}-[0-9]{2})?",
        ],
        "AMEND_SALARY_AMOUNT": [
            r"(?:monthly salary has increased to|next salary is reduced to|temporary monthly pay is|renewed lease increases monthly rent|regular salary for the next payroll is|regular salary of [A-Z]{3} [0-9\.,]+ resumes on|salary of [A-Z]{3} [0-9\.,]+ is confirmed for)\b",
            r"(?:gaji bulanan anda naik menjadi|gaji bulanan sementara anda adalah|gaji rutin anda untuk penggajian berikutnya adalah|gaji berikutnya berkurang menjadi|gaji sebesar [A-Z]{3} [0-9\.,]+ dikonfirmasi)\b",
        ],
    },
    "Cash flow termination": {
        "STOP_RECURRING": [
            r"(?:contract has ended|employment has ended|household employment record has ended|no regular salary payments scheduled|no off-season income or renewal has been confirmed)\b",
            r"(?:kontrak musiman saat ini telah berakhir|hubungan kerja anda telah berakhir|pendapatan kerja rumah tangga telah berakhir|tidak ada pembayaran gaji rutin yang dijadwalkan)\b",
        ]
    },
    "Phantom inflow exclusion": {
        "EXCLUDE_EVENT": [
            r"(?:displayed market value has increased|no units have been sold and no cash proceeds|prize claim has been verified and is still in payment processing|payment has not been credited to your account yet|pay the release charge today|refund has been initiated but has not reached your account yet|matching debit and credit came from a transfer between your two accounts|commission shown for open deals is still pending approval|bonus(?: kuartalan)? (?:is still subject to the final performance review|anda masih menunggu hasil akhir)|payout is still pending|weekly earnings shown in the .*? app can change|foreign-currency refund is still processing|extra card charge is still being investigated|previous debit attempt failed)\b",
            r"(?:nilai pasar portofolio|belum ada unit yang dijual|klaim hadiah anda telah diverifikasi|masih dalam proses pembayaran|belum (?:masuk|dikreditkan) ke rekening|pengembalian dana anda telah dimulai tetapi belum masuk|debit dan kredit yang cocok berasal dari transfer|komisi dari transaksi yang masih berjalan belum disetujui|pembayaran berikutnya dari .*? masih tertunda)\b",
        ]
    },
    "Confirmed windfall": {
        "CONFIRM_INFLOW": [
            r"(?:prize proceeds have reached your account|proceeds from your investment sale have settled in the cash account|client approved an invoice payment of|first salary will be|first salary from the new employer is|first salary of [A-Z]{3} [0-9\.,]+ is scheduled|reimbursement for your earlier work expense)\b",
            r"(?:hasil penjualan investasi anda sudah masuk ke rekening tunai|klien menyetujui pembayaran faktur sebesar|gaji pertama anda sebesar|gaji pertama dari perusahaan baru adalah|faktur sebesar .*? penyelesaian diperkirakan)\b",
        ]
    },
}


# ----------------------------------------------------------------------
# Extraction Helpers
# ----------------------------------------------------------------------
def _extract_date(text: str) -> Optional[str]:
    """Extracts ISO date (YYYY-MM-DD) from text if present."""
    match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if match:
        return match.group(1)
    return None


def _extract_amount(text: str) -> Optional[float]:
    """Extracts numeric currency amount from text (handles both European and standard notation)."""
    match = re.search(r"\b(?:IDR|EUR|USD|ZAR|GBP|INR)\s*([\d\.,]+)", text, re.IGNORECASE)
    if not match:
        return None
    raw_num = match.group(1).strip()

    # Detect European/Indonesian period-as-thousands format (e.g. 42.750.000 or 1.422,85)
    euro_match = re.search(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?", raw_num)
    if euro_match:
        normalized = euro_match.group().replace(".", "").replace(",", ".")
        try:
            return float(normalized)
        except ValueError:
            pass

    clean_num = raw_num.replace(",", "")
    try:
        return float(clean_num)
    except ValueError:
        return None


# ----------------------------------------------------------------------
# Step 1: Regex Matching
# ----------------------------------------------------------------------
def parse_message_regex(text: str) -> Optional[Tuple[str, str]]:
    """
    Step 1: NLP/Regex matching against REGEX_RULES.
    Returns (category, action_type) if matched, else None.
    """
    for category, actions in REGEX_RULES.items():
        for action_type, patterns in actions.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return category, action_type
    return None


# ----------------------------------------------------------------------
# Step 2: LLM Fallback
# ----------------------------------------------------------------------
def parse_message_llm(
    message_text: str,
    llm: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """
    Step 2: LLM fallback for unparsed or uncertain messages.
    Returns a structured dictionary if resolved, else None.
    """
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError:
        # If LangChain is not installed in the current environment
        return None

    if llm is None:
        api_key = os.environ.get("OPENROUTER_API_KEY", os.environ.get("OPENAI_API_KEY", "<API_KEY>"))
        model_name = os.environ.get("OPENROUTER_MODEL_NAME", os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini"))
        try:
            llm = ChatOpenAI(
                model=model_name,
                api_key=api_key,
                temperature=0.0,
            )
        except Exception:
            return None

    system_prompt = (
        "You are an expert financial message parser. Classify the user/service message into one of these action types:\n"
        "- AMEND_SALARY_DATE: date of recurring salary or cash flow has shifted\n"
        "- AMEND_SALARY_AMOUNT: amount of salary, income, or recurring expense (e.g. rent) has changed\n"
        "- STOP_RECURRING: contract ended, employment ended, or recurring stream terminated\n"
        "- EXCLUDE_EVENT: phantom inflow, uncredited prize, unearned bonus, or duplicate transfer to exclude\n"
        "- CONFIRM_INFLOW: confirmed prize, completed investment sale, approved invoice, or new salary credit\n\n"
        "Return ONLY a valid JSON object formatted exactly as:\n"
        "{\n"
        '    "action_type": "AMEND_SALARY_DATE" | "AMEND_SALARY_AMOUNT" | "STOP_RECURRING" | "EXCLUDE_EVENT" | "CONFIRM_INFLOW",\n'
        '    "new_date": "YYYY-MM-DD" or null,\n'
        '    "new_amount": float or null\n'
        "}\n"
        "If you cannot classify the message or it is purely informational, return null."
    )

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Message to parse: {message_text}"),
        ])
        content = str(response.content).strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

        if content.lower() == "null":
            return None

        parsed = json.loads(content)
        if isinstance(parsed, dict) and parsed.get("action_type"):
            return parsed
    except Exception:
        return None

    return None


# ----------------------------------------------------------------------
# Pipeline Execution
# ----------------------------------------------------------------------
def run_hybrid_message_pipeline(
    message: Dict[str, Any],
    llm: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Runs the hybrid two-step message parsing pipeline on a single message.
    
    Output Format:
    {
        "message_id": "message_05",
        "action_type": "AMEND_SALARY_DATE",
        "related_event_id": "event_123",    # or None
        "new_date": "2024-09-23",
        "new_amount": None,
        "confidence": 1.0                   # 1.0 (regex), 0.85 (LLM fallback), 0.0 (uncertain)
    }
    """
    message_id = message.get("message_id")
    message_text = message.get("message_text", "")
    related_event_id = message.get("related_event_id") or None

    # Step 1: NLP / Regex pattern matching
    regex_match = parse_message_regex(message_text)

    if regex_match is not None:
        _, action_type = regex_match
        new_date = _extract_date(message_text) if action_type in ("AMEND_SALARY_DATE", "CONFIRM_INFLOW") else None
        new_amount = _extract_amount(message_text) if action_type in ("AMEND_SALARY_AMOUNT", "CONFIRM_INFLOW") else None

        return {
            "message_id": message_id,
            "action_type": action_type,
            "related_event_id": related_event_id,
            "new_date": new_date,
            "new_amount": new_amount,
            "confidence": 1.0,
        }

    # Step 2: LLM Fallback (only on unparsed / uncertain)
    llm_result = parse_message_llm(message_text, llm=llm)

    if llm_result is not None and llm_result.get("action_type"):
        return {
            "message_id": message_id,
            "action_type": llm_result.get("action_type"),
            "related_event_id": related_event_id,
            "new_date": llm_result.get("new_date"),
            "new_amount": float(llm_result["new_amount"]) if llm_result.get("new_amount") is not None else None,
            "confidence": 0.85,
        }

    # Return uncertain with confidence 0.0
    return {
        "message_id": message_id,
        "action_type": None,
        "related_event_id": related_event_id,
        "new_date": None,
        "new_amount": None,
        "confidence": 0.0,
    }


# ----------------------------------------------------------------------
# Direct Integration with ctx["events"] for 90-Day Simulator
# ----------------------------------------------------------------------
def apply_message_action_to_events(
    events: List[Dict[str, Any]],
    action: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Applies a parsed message action directly to an events list.
    Modifies matching event dictionary fields in-place and returns the list.
    """
    action_type = action.get("action_type")
    if not action_type or action.get("confidence", 0.0) < 0.8:
        return events

    target_event_id = action.get("related_event_id")
    new_date = action.get("new_date")
    new_amount = action.get("new_amount")
    message_id = action.get("message_id")

    # Helper match predicate for when related_event_id is not explicitly provided
    def _is_salary_or_income(ev: Dict[str, Any]) -> bool:
        category = str(ev.get("category", "")).lower()
        ev_type = str(ev.get("event_type", "")).lower()
        desc = str(ev.get("description", "")).lower()
        return category == "salary" or ev_type == "income" or "salary" in desc or "payroll" in desc

    applied = False

    for ev in events:
        ev_id = ev.get("event_id")
        
        # Match by explicit event_id if present, else fallback to category heuristics
        is_target = (target_event_id and ev_id == target_event_id) or (not target_event_id and _is_salary_or_income(ev))

        if is_target:
            if action_type == "AMEND_SALARY_DATE" and new_date:
                ev["event_date"] = new_date
                ev["settlement_date"] = new_date
                ev["amended_date"] = new_date
                ev["applied_message_id"] = message_id
                applied = True

            elif action_type == "AMEND_SALARY_AMOUNT" and new_amount is not None:
                ev["amount"] = new_amount
                ev["amended_amount"] = new_amount
                if new_date:
                    ev["event_date"] = new_date
                    ev["settlement_date"] = new_date
                ev["applied_message_id"] = message_id
                applied = True

            elif action_type == "STOP_RECURRING":
                ev["status"] = "cancelled"
                ev["is_active"] = False
                ev["excluded"] = True
                ev["applied_message_id"] = message_id
                applied = True

            elif action_type == "EXCLUDE_EVENT":
                ev["status"] = "excluded"
                ev["excluded"] = True
                ev["applied_message_id"] = message_id
                applied = True

            elif action_type == "CONFIRM_INFLOW":
                # Bug 3 fix: when no explicit related_event_id is given (heuristic match),
                # skip events that are already settled or cancelled to prevent overwriting
                # historical cash flow records.
                # Proven by test: event_2850 (status='settled', date=2024-12-15) was being
                # mutated to settlement_date=2025-02-15, corrupting the historical record.
                if not target_event_id:
                    ev_status = str(ev.get("status", "")).lower()
                    if ev_status not in ("pending", "scheduled"):
                        continue
                ev["status"] = "confirmed"
                ev["excluded"] = False
                if new_amount is not None:
                    ev["amount"] = new_amount
                if new_date:
                    ev["settlement_date"] = new_date
                    ev["event_date"] = new_date
                ev["applied_message_id"] = message_id
                applied = True

    # If CONFIRM_INFLOW was not linked to an existing event and not matched, create a confirmed event
    if action_type == "CONFIRM_INFLOW" and not applied and new_amount is not None:
        synthetic_event = {
            "event_id": f"confirmed_inflow_{message_id}",
            "user_id": events[0].get("user_id", "") if events else "",
            "event_type": "income",
            "category": "windfall",
            "direction": "credit",
            "amount": new_amount,
            "currency": events[0].get("currency", "EUR") if events else "EUR",
            "event_date": new_date or "",
            "settlement_date": new_date or "",
            "status": "confirmed",
            "excluded": False,
            "applied_message_id": message_id,
        }
        events.append(synthetic_event)

    return events


def integrate_messages_into_context_events(
    ctx: Dict[str, Any],
    llm: Optional[Any] = None,
    min_confidence: float = 0.8,
) -> Dict[str, Any]:
    """
    Parses all request and user messages in ctx and applies the parsed
    actions directly to ctx["events"] and ctx["events_by_id"] before
    the events are fed into the 90-day balance simulator.
    
    Returns the mutated RequestContext dictionary with:
    - ctx["events"] modified according to message instructions
    - ctx["parsed_message_actions"] containing the list of parsed action dictionaries
    """
    events = ctx.get("events", [])
    events_by_id = ctx.get("events_by_id", {})
    messages_dict = ctx.get("messages", {})

    # Collect both request-level and user-level messages
    all_messages = messages_dict.get("request_messages", []) + messages_dict.get("user_messages", [])

    parsed_actions: List[Dict[str, Any]] = []

    for msg in all_messages:
        action = run_hybrid_message_pipeline(msg, llm=llm)
        parsed_actions.append(action)

        if action.get("confidence", 0.0) >= min_confidence:
            apply_message_action_to_events(events, action)

    # Keep events_by_id in sync
    for ev in events:
        ev_id = ev.get("event_id")
        if ev_id:
            events_by_id[ev_id] = ev

    ctx["events"] = events
    ctx["events_by_id"] = events_by_id
    ctx["parsed_message_actions"] = parsed_actions

    return ctx
