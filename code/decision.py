"""
Hybrid Decision Engine for the 90-Day Financial Simulation.

================================================================================
ARCHITECTURE OVERVIEW: THE 5 MINI-PIPELINES
================================================================================
1. Mini-Pipeline 1: Deterministic Decision Tree Gatekeeper (Outer Safety Gate)
   - Checks if the user accepts 'full_payment' and if amount_safe covers requested_amount.
   - If safe today, short-circuits to 'affordable_now' with 'full_payment'.

2. Mini-Pipeline 2: Partial Payment Resolver
   - If the request allows partial payment, the user accepts 'partial_payment',
     0 < amount_safe < requested_amount, and earliest_date <= deadline:
     Recommends 'affordable_with_plan' with 'partial_payment' (zero financing fee).

3. Mini-Pipeline 3: Installment Plan Simulation & Dynamic Multi-Criteria Scorer
   - If the user accepts 'installments', simulates candidate plans from request_payment_options.csv.
   - Ranks safe plans via DynamicScorer using context-aware weights and Min-Max scaling.

4. Mini-Pipeline 4: Deferral & Fallback Branch (Wait vs. Not Affordable)
   - If full payment is accepted and earliest_date <= deadline: recommends 'affordable_later' ('wait').
   - Otherwise flags as 'not_affordable'.

5. Mini-Pipeline 5: LLM Decision Explanation Generator
   - Generates a strictly bounded 10-20 token (1-2 sentence) explanation via prompt constraints
     and hard max_tokens limits, with an intelligent deterministic fallback template.
================================================================================
"""

from datetime import datetime, timedelta
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

# Automatically load environment variables from .env if present
def _load_env_file() -> None:
    for env_path in [Path(__file__).resolve().parent / ".env", Path(__file__).resolve().parent.parent / ".env"]:
        if env_path.exists():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k, v = k.strip(), v.strip().strip("'\"")
                            if k and k not in os.environ:
                                os.environ[k] = v
            except Exception:
                pass

_load_env_file()

# Attempt package and relative imports for Stage3BusinessLogic simulation
try:
    from code.simulation import Stage3BusinessLogic
except ImportError:
    try:
        from simulation import Stage3BusinessLogic
    except ImportError:
        Stage3BusinessLogic = None


# ==============================================================================
# MINI-PIPELINE 3: DYNAMIC MULTI-CRITERIA SCORING SYSTEM
# ==============================================================================
class DynamicScorer:
    """
    Mini-Pipeline 3: Dynamic Multi-Criteria Scorer.
    Evaluates and ranks candidate installment options using adaptive weights.
    """

    @staticmethod
    def _get_dynamic_weights(request_type: str) -> Dict[str, float]:
        req_type = (request_type or "").lower().strip()
        if req_type in ("emergency_expense", "debt_repayment", "medical", "urgent"):
            return {"cost": 0.10, "speed": 0.55, "simplicity": 0.10, "margin": 0.25}
        if req_type in ("housing", "family_transfer", "education"):
            return {"cost": 0.30, "speed": 0.20, "simplicity": 0.10, "margin": 0.40}
        return {"cost": 0.45, "speed": 0.05, "simplicity": 0.15, "margin": 0.35}

    @staticmethod
    def score_options(
        safe_options: List[Dict[str, Any]],
        request_type: str,
        sim_results: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not safe_options or not sim_results:
            return None
        if len(safe_options) == 1:
            return safe_options[0]

        weights = DynamicScorer._get_dynamic_weights(request_type)

        def _safe_float(val: Any, default: float = 0.0) -> float:
            try:
                f = float(val)
                return default if math.isnan(f) else f
            except Exception:
                return default

        fees = [_safe_float(opt.get("financing_fee"), 0.0) for opt in safe_options]
        payments = [int(_safe_float(opt.get("number_of_payments"), 1.0)) for opt in safe_options]
        cushions = [_safe_float(res.get("min_cushion_with_plan"), 0.0) for res in sim_results]

        def _date_to_num(d_str: Any) -> float:
            try:
                return float(datetime.strptime(str(d_str)[:10], "%Y-%m-%d").toordinal())
            except Exception:
                return 0.0

        date_nums = [_date_to_num(opt.get("first_payment_date")) for opt in safe_options]
        valid_dates = [d for d in date_nums if d > 0.0]
        min_d = min(valid_dates) if valid_dates else 0.0
        max_d = max(valid_dates) if valid_dates else 0.0

        min_fee, max_fee = min(fees), max(fees)
        min_pmts, max_pmts = min(payments), max(payments)
        min_margin, max_margin = min(cushions), max(cushions)

        def _normalize(val: float, min_v: float, max_v: float, reverse: bool = False) -> float:
            if max_v == min_v:
                return 1.0
            ratio = (val - min_v) / (max_v - min_v)
            return (1.0 - ratio) if reverse else ratio

        best_score = -1.0
        best_option = safe_options[0]

        for opt, res, fee, pmts, margin, d_num in zip(safe_options, sim_results, fees, payments, cushions, date_nums):
            cost_score = _normalize(fee, min_fee, max_fee, reverse=True)
            speed_score = _normalize(d_num, min_d, max_d, reverse=True) if min_d > 0.0 else 1.0
            simplicity_score = _normalize(pmts, min_pmts, max_pmts, reverse=True)
            margin_score = _normalize(margin, min_margin, max_margin, reverse=False)

            total_score = (
                (cost_score * weights["cost"])
                + (speed_score * weights["speed"])
                + (simplicity_score * weights["simplicity"])
                + (margin_score * weights["margin"])
            ) * 100.0

            if total_score > best_score:
                best_score = total_score
                best_option = opt

        return best_option


# ==============================================================================
# MINI-PIPELINE 5: LLM DECISION EXPLANATION GENERATOR
# ==============================================================================
class ExplanationGenerator:
    """
    Mini-Pipeline 5: LLM Decision Explanation Generator.
    Enforces a strict 10-20 token / 1-2 sentence constraint with deterministic fallback.
    """

    @staticmethod
    def _create_default_llm() -> Optional[Any]:
        try:
            from langchain_openai import ChatOpenAI
            api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                return None
            model_name = os.getenv("OPENROUTER_MODEL_NAME") or os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
            base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

            kwargs = {
                "model": model_name,
                "api_key": api_key,
                "base_url": base_url,
                "temperature": 0.2,
                "max_tokens": 25,
            }
            return ChatOpenAI(**kwargs)
        except Exception:
            return None

    @staticmethod
    def _generate_fallback(
        decision: Dict[str, Any],
        profile: Dict[str, Any],
        request: Dict[str, Any],
    ) -> str:
        currency = profile.get("home_currency", "USD")
        min_bal = float(profile.get("minimum_balance_to_keep", 0.0))
        amt = float(request.get("requested_amount", 0.0))
        status = decision.get("affordability_status")
        method = decision.get("recommended_payment_method")
        plan = decision.get("payment_plan", "")

        if status == "affordable_now":
            date_str = plan.split(":")[0] if ":" in plan else "today"
            return f"Pay {currency} {amt:,.2f} in full on {date_str}. Your minimum balance of {currency} {min_bal:,.0f} remains protected."

        if method == "partial_payment":
            parts = plan.split("|")
            p1_date, p1_amt = parts[0].split(":")
            p2_date, p2_amt = parts[1].split(":")
            return f"Pay {currency} {float(p1_amt):,.2f} on {p1_date} and {currency} {float(p2_amt):,.2f} on {p2_date}. Your minimum balance stays safe."

        if method == "installments":
            parts = plan.split("|")
            first_date, first_amt = parts[0].split(":") if ":" in parts[0] else ("soon", amt)
            return f"Use {len(parts)} installments of {currency} {float(first_amt):,.2f} starting {first_date}. This fits your cash flow safely."

        if status == "affordable_later" or method == "wait":
            earliest_date = decision.get("earliest_date_for_full_payment", "a later date")
            return f"Wait to pay {currency} {amt:,.2f} in full on {earliest_date}. Paying earlier would breach your required balance."

        return f"Do not proceed with the {currency} {amt:,.2f} request. Your balance cannot safely support this expense within 90 days."

    @classmethod
    def generate(
        cls,
        decision: Dict[str, Any],
        profile: Dict[str, Any],
        request: Dict[str, Any],
        llm: Optional[Any] = None,
    ) -> str:
        fallback_text = cls._generate_fallback(decision, profile, request)
        active_llm = llm or cls._create_default_llm()
        if not active_llm:
            return fallback_text

        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            currency = profile.get("home_currency", "USD")
            min_bal = float(profile.get("minimum_balance_to_keep", 0.0))
            amt = float(request.get("requested_amount", 0.0))

            system_prompt = (
                "You are an expert financial advisor. Summarize the payment decision in exactly 1 to 2 short sentences "
                "(strictly between 10 and 20 words/tokens total). Never use markdown, bullet points, or filler phrases. "
                "Explicitly mention the action, amount, currency, and safety protection."
            )
            raw_req_type = str(request.get("request_type", "general"))
            clean_req_type = re.sub(r"[^a-zA-Z0-9_\- ]", "", raw_req_type)[:30].strip() or "general"

            user_prompt = (
                f"Request Type: {clean_req_type}\n"
                f"Requested Amount: {currency} {amt:,.2f}\n"
                f"Protected Minimum Balance: {currency} {min_bal:,.2f}\n"
                f"Decision Status: {decision.get('affordability_status')}\n"
                f"Recommended Method: {decision.get('recommended_payment_method')}\n"
                f"Payment Plan: {decision.get('payment_plan')}\n"
                f"Reference Example: {fallback_text}\n"
                "Provide the concise 10-20 token explanation now:"
            )

            invoker = active_llm
            if hasattr(active_llm, "bind"):
                try:
                    invoker = active_llm.bind(max_tokens=25)
                except Exception:
                    pass

            response = invoker.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            text = str(response.content).strip().replace('"', '').replace('`', '').strip()
            if text and len(text.split()) >= 5:
                return text
        except Exception:
            pass

        return fallback_text


# ==============================================================================
# MAIN ORCHESTRATOR: HYBRID DECISION ENGINE
# ==============================================================================
class HybridDecisionEngine:
    """
    Main Orchestrator combining all 5 Mini-Pipelines:
    1. Immediate Full Payment Gate
    2. Partial Payment Resolver
    3. Installment Plan Simulation & Dynamic Scoring
    4. Deferral & Fallback Branch (Wait vs. Reject)
    5. Token-Bounded LLM Explanation Generator
    """

    @staticmethod
    def _build_installment_plan_string(option: Dict[str, Any]) -> str:
        first_date = str(option.get("first_payment_date", ""))[:10]
        try:
            num_payments = int(float(option.get("number_of_payments", 1)))
            amount = float(option.get("payment_amount", 0.0))
            raw_freq = option.get("payment_frequency_days")
            if raw_freq is None or (isinstance(raw_freq, float) and math.isnan(raw_freq)) or str(raw_freq).strip() == "":
                freq_days = 30
            else:
                freq_days = int(float(raw_freq))
        except Exception:
            return "none"

        if not first_date or num_payments <= 0:
            return "none"

        parts = []
        try:
            curr_dt = datetime.strptime(first_date, "%Y-%m-%d")
            for _ in range(num_payments):
                parts.append(f"{curr_dt.strftime('%Y-%m-%d')}:{amount:.2f}")
                curr_dt += timedelta(days=freq_days)
            return "|".join(parts)
        except Exception:
            return "none"

    @classmethod
    def make_decision(
        cls,
        request: Dict[str, Any],
        profile: Dict[str, Any],
        sim_result: Dict[str, Any],
        payment_options: List[Dict[str, Any]],
        stage3_metrics: Dict[str, Any],
        llm: Optional[Any] = None,
    ) -> Dict[str, Any]:
        req_id = request.get("request_id", "")
        req_amount = float(request.get("requested_amount", 0.0))
        req_type = str(request.get("request_type", ""))
        req_date = str(request.get("request_date", sim_result.get("start_date", "1970-01-01")))[:10]
        deadline = request.get("desired_completion_date")
        allows_partial = bool(request.get("allows_partial_payment", False))

        amount_safe = float(stage3_metrics.get("amount_safe_to_pay", 0.0))
        earliest_date = stage3_metrics.get("earliest_date_for_full_payment")

        # Parse user's accepted payment methods
        raw_accepted = profile.get("payment_methods_user_will_consider")
        if raw_accepted:
            accepted_methods = [m.strip().lower() for m in str(raw_accepted).split("|") if m.strip()]
        else:
            accepted_methods = ["full_payment", "installments", "partial_payment"]

        output: Dict[str, Any] = {
            "request_id": req_id,
            "amount_safe_to_pay": amount_safe,
            "affordability_status": "not_affordable",
            "recommended_payment_method": "not_recommended",
            "payment_plan": "none",
            "earliest_date_for_full_payment": earliest_date,
            "spending_changes_needed": "none",
            "decision_explanation": "",
        }

        # ----------------------------------------------------------------------
        # 1. IMMEDIATE FULL PAYMENT (if user accepts full_payment and balance covers it)
        # ----------------------------------------------------------------------
        if "full_payment" in accepted_methods and amount_safe >= req_amount:
            output["affordability_status"] = "affordable_now"
            output["recommended_payment_method"] = "full_payment"

            fp_option = next(
                (opt for opt in payment_options if opt.get("payment_method") == "full_payment"),
                None,
            )
            if fp_option and fp_option.get("first_payment_date"):
                date_str = str(fp_option.get("first_payment_date", ""))[:10]
                output["payment_plan"] = f"{date_str}:{req_amount:.2f}"
            else:
                output["payment_plan"] = f"{req_date}:{req_amount:.2f}"

            output["decision_explanation"] = ExplanationGenerator.generate(output, profile, request, llm=llm)
            return output

        # ----------------------------------------------------------------------
        # 2. PARTIAL PAYMENT (zero financing fee, prioritized over installments if accepted)
        # ----------------------------------------------------------------------
        if (
            "partial_payment" in accepted_methods
            and allows_partial
            and 0.0 < amount_safe < req_amount
            and earliest_date is not None
        ):
            earliest_dt_str = str(earliest_date)[:10]
            deadline_str = str(deadline)[:10] if deadline else ""
            if not deadline_str or earliest_dt_str <= deadline_str:
                remaining_amt = req_amount - amount_safe
                output["affordability_status"] = "affordable_with_plan"
                output["recommended_payment_method"] = "partial_payment"
                output["payment_plan"] = f"{req_date}:{amount_safe:.2f}|{earliest_dt_str}:{remaining_amt:.2f}"
                output["decision_explanation"] = ExplanationGenerator.generate(output, profile, request, llm=llm)
                return output

        # ----------------------------------------------------------------------
        # 3. INSTALLMENT PLANS (if user accepts installments)
        # ----------------------------------------------------------------------
        safe_installments = []
        sim_results_for_safe = []

        if "installments" in accepted_methods and Stage3BusinessLogic is not None:
            installment_opts = [
                opt for opt in payment_options if opt.get("payment_method") == "installments"
            ]
            for opt in installment_opts:
                plan_str = cls._build_installment_plan_string(opt)
                if plan_str == "none":
                    continue

                payments = []
                for part in plan_str.split("|"):
                    if ":" in part:
                        d_val, a_val = part.split(":")
                        payments.append({"date": d_val, "amount": float(a_val)})

                test_res = Stage3BusinessLogic.test_payment_plan(
                    sim_result, payments, desired_completion_date=deadline
                )
                if test_res.get("is_safe", False):
                    safe_installments.append(opt)
                    sim_results_for_safe.append(test_res)

        if safe_installments:
            best_opt = DynamicScorer.score_options(
                safe_installments, req_type, sim_results_for_safe
            )
            if best_opt:
                output["affordability_status"] = "affordable_with_plan"
                output["recommended_payment_method"] = "installments"
                output["payment_plan"] = cls._build_installment_plan_string(best_opt)
                output["decision_explanation"] = ExplanationGenerator.generate(output, profile, request, llm=llm)
                return output

        # ----------------------------------------------------------------------
        # 4. DEFERRAL / WAIT (only if user considers full_payment)
        # ----------------------------------------------------------------------
        if "full_payment" in accepted_methods and earliest_date:
            earliest_dt_str = str(earliest_date)[:10]
            deadline_str = str(deadline)[:10] if deadline else ""
            if not deadline_str or earliest_dt_str <= deadline_str:
                output["affordability_status"] = "affordable_later"
                output["recommended_payment_method"] = "wait"
                output["payment_plan"] = f"{earliest_dt_str}:{req_amount:.2f}"
                output["decision_explanation"] = ExplanationGenerator.generate(output, profile, request, llm=llm)
                return output

        # ----------------------------------------------------------------------
        # 5. FINAL FALLBACK: NOT AFFORDABLE
        # ----------------------------------------------------------------------
        output["decision_explanation"] = ExplanationGenerator.generate(output, profile, request, llm=llm)
        return output
