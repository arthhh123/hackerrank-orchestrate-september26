"""
Stage 1 of 90-Day Balance Simulation: Real Cash Flow Ledger.

Builds a clean, conservative ledger of real cash flow:
- Outflows: Treated generously (settled, pending, and scheduled debit commitments).
  Funds are reserved on the earlier date before settlement to avoid shortfall.
- Inflows: Treated strictly (only confirmed or settled credits).
  Pending credits, uncredited bonuses, and unrealized valuations are excluded.
- Result: Clean list of 'this money moves on this date, for this amount' events.
"""

from datetime import datetime, timedelta
import math
from typing import Any, Dict, List, Optional


class RealCashFlowEvent:
    """Represents a single verified cash flow event in the ledger."""

    __slots__ = (
        "event_id",
        "date",
        "amount",
        "direction",
        "category",
        "description",
        "status",
        "currency",
        "flexibility",
    )

    def __init__(
        self,
        event_id: str,
        date: str,
        amount: float,
        direction: str,  # 'outflow' or 'inflow'
        category: str = "",
        description: str = "",
        status: str = "",
        currency: str = "",
        flexibility: str = "fixed",
    ):
        self.event_id = event_id
        self.date = date
        self.amount = amount
        self.direction = direction
        self.category = category
        self.description = description
        self.status = status
        self.currency = currency
        self.flexibility = flexibility

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "date": self.date,
            "amount": self.amount,
            "direction": self.direction,
            "category": self.category,
            "description": self.description,
            "status": self.status,
            "currency": self.currency,
            "flexibility": self.flexibility,
        }

    def __repr__(self) -> str:
        return (
            f"RealCashFlowEvent(date='{self.date}', amount={self.amount:.2f}, "
            f"direction='{self.direction}', event_id='{self.event_id}')"
        )


class Stage1CashFlowLedger:
    """Stage 1 pipeline to construct a verified cash-flow ledger."""

    @staticmethod
    def build_ledger(
        events: List[Dict[str, Any]],
        home_currency: Optional[str] = None,
        exchange_rates: Optional[Dict[tuple, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Filters and normalizes events into a clean list of
        'this money moves on this date, for this amount' events.
        """
        ledger_events: List[RealCashFlowEvent] = []

        for ev in events:
            # 1. Skip cancelled, failed, or message-excluded records
            status = str(ev.get("status", "")).lower().strip()
            if status in ("cancelled", "failed", "excluded") or ev.get("excluded") is True:
                continue

            direction = str(ev.get("direction", "")).lower().strip()
            event_type = str(ev.get("event_type", "")).lower().strip()

            # 2. Skip non-cash items (e.g. unrealized stock/portfolio valuations)
            if (
                direction == "non_cash"
                or event_type == "investment_valuation"
                or status == "unrealized"
            ):
                continue

            # Amount validation
            raw_amount = ev.get("amount")
            if raw_amount is None or raw_amount == "":
                continue
            try:
                amount = float(raw_amount)
                if math.isnan(amount) or amount < 0:
                    continue
            except (ValueError, TypeError):
                continue

            # Dates
            event_date = str(ev.get("event_date", "")).strip()[:10]
            settlement_date = str(ev.get("settlement_date", "")).strip()[:10]
            if not event_date and not settlement_date:
                continue

            # Currency normalization if exchange rates are provided
            currency = str(ev.get("currency", "")).strip()
            effective_currency = home_currency or currency
            if home_currency and currency and currency != home_currency and exchange_rates:
                rate_date = settlement_date or event_date
                rate = exchange_rates.get((rate_date, currency, home_currency), 1.0)
                amount = amount * rate

            # 3. OUTFLOWS: Treated generously
            # Count rent, bills, loan payments even if pending or scheduled.
            # Reserve before settled: use earlier date to protect balance in advance.
            if direction == "debit":
                if status in ("settled", "pending", "scheduled"):
                    move_date = event_date
                    if event_date and settlement_date:
                        move_date = min(event_date, settlement_date)
                    elif settlement_date:
                        move_date = settlement_date

                    ledger_events.append(
                        RealCashFlowEvent(
                            event_id=str(ev.get("event_id", "")),
                            date=move_date,
                            amount=round(amount, 4),
                            direction="outflow",
                            category=str(ev.get("category", "")),
                            description=str(ev.get("description", "")),
                            status=status,
                            currency=effective_currency,
                            flexibility=str(ev.get("flexibility", "fixed")),
                        )
                    )

            # 4. INFLOWS: Treated strictly
            # Only count money that is confirmed or settled.
            # Pending bonuses, lottery in review, pending refunds strictly excluded.
            elif direction == "credit":
                is_confirmed_or_settled = False
                if status in ("settled", "confirmed"):
                    is_confirmed_or_settled = True
                elif status == "scheduled":
                    desc = str(ev.get("description", "")).lower()
                    cat = str(ev.get("category", "")).lower()
                    if "salary" in desc or "salary" in cat or "confirmed" in desc:
                        is_confirmed_or_settled = True

                if is_confirmed_or_settled:
                    # Settled/confirmed cash moves on settlement date (or event date)
                    move_date = settlement_date or event_date
                    ledger_events.append(
                        RealCashFlowEvent(
                            event_id=str(ev.get("event_id", "")),
                            date=move_date,
                            amount=round(amount, 4),
                            direction="inflow",
                            category=str(ev.get("category", "")),
                            description=str(ev.get("description", "")),
                            status=status,
                            currency=effective_currency,
                            flexibility=str(ev.get("flexibility", "fixed")),
                        )
                    )

        # Chronological sort: on same date, inflows precede outflows
        ledger_events.sort(key=lambda x: (x.date, 0 if x.direction == "inflow" else 1))
        return [entry.to_dict() for entry in ledger_events]


# ----------------------------------------------------------------------
# Stage 2: 90-Day Balance & Cushion Simulation Engine (Day 0 to Day 90)
# ----------------------------------------------------------------------
class DailySimulationStep:
    """Represents the financial state for a single day t in [0, 90]."""

    __slots__ = (
        "day",
        "date",
        "inflows",
        "outflows",
        "net_change",
        "projected_balance",
        "cushion",
    )

    def __init__(
        self,
        day: int,
        date: str,
        inflows: float,
        outflows: float,
        net_change: float,
        projected_balance: float,
        cushion: float,
    ):
        self.day = day
        self.date = date
        self.inflows = inflows
        self.outflows = outflows
        self.net_change = net_change
        self.projected_balance = projected_balance
        self.cushion = cushion

    def to_dict(self) -> Dict[str, Any]:
        return {
            "day": self.day,
            "date": self.date,
            "inflows": round(self.inflows, 2),
            "outflows": round(self.outflows, 2),
            "net_change": round(self.net_change, 2),
            "projected_balance": round(self.projected_balance, 2),
            "cushion": round(self.cushion, 2),
        }

    def __repr__(self) -> str:
        return (
            f"Day {self.day:02d} ({self.date}): "
            f"In={self.inflows:.2f}, Out={self.outflows:.2f}, "
            f"Δ={self.net_change:+.2f} -> Balance={self.projected_balance:.2f} "
            f"(Cushion={self.cushion:+.2f})"
        )


class Stage2SimulationEngine:
    """
    Stage 2 of 90-Day Simulation Pipeline: Balance & Cushion Calculation (Day 0 to Day 90).

    For each day t in [0, 90]:
    - Δ(t) = Inflows(t) - Outflows(t)
    - B(t) = B(t-1) + Δ(t) (projected balance on every single day)
    - Cushion(t) = B(t) - M (slack above minimum balance floor line M)
    """

    @staticmethod
    def simulate_90_days(
        start_date_str: str,
        starting_balance: float,
        minimum_balance: float,
        ledger: List[Dict[str, Any]],
        days: int = 90,
    ) -> Dict[str, Any]:
        """
        Calculates daily balance and cushion for day 0 to day 90.

        Args:
            start_date_str: Request evaluation date (Day 0) in YYYY-MM-DD.
            starting_balance: User's current available balance.
            minimum_balance: Floor line M (minimum_balance_to_keep).
            ledger: Clean cash flow events list from Stage 1.
            days: Simulation horizon (default 90).

        Returns:
            Dictionary containing:
            - trajectory: Day-by-day steps from Day 0 through Day 90
            - min_cushion: Lowest cushion value observed in the 90-day window
            - min_balance: Lowest projected balance reached
            - min_cushion_date: The date when cushion is at its minimum
            - is_safe: True if cushion >= 0 on all 91 days (never crosses floor M)
        """
        start_dt = datetime.strptime(start_date_str[:10], "%Y-%m-%d")

        # Map daily cash flows from Stage 1 clean ledger
        daily_inflows: Dict[str, float] = {}
        daily_outflows: Dict[str, float] = {}

        for ev in ledger:
            ev_date = str(ev.get("date", ""))[:10]
            direction = str(ev.get("direction", "")).lower()
            amt = float(ev.get("amount", 0.0))

            if direction == "inflow":
                daily_inflows[ev_date] = daily_inflows.get(ev_date, 0.0) + amt
            elif direction == "outflow":
                daily_outflows[ev_date] = daily_outflows.get(ev_date, 0.0) + amt

        trajectory: List[DailySimulationStep] = []
        current_balance = float(starting_balance)
        min_cushion = float("inf")
        min_balance = float("inf")
        min_cushion_date = start_date_str[:10]

        # Simulate Day 0 through Day 90 (91 days total)
        for day in range(days + 1):
            curr_dt = start_dt + timedelta(days=day)
            curr_date_str = curr_dt.strftime("%Y-%m-%d")

            inflows = daily_inflows.get(curr_date_str, 0.0)
            outflows = daily_outflows.get(curr_date_str, 0.0)
            net_change = inflows - outflows

            # B(t) = B(t-1) + Δ(t)
            current_balance += net_change

            # Cushion(t) = B(t) - M
            cushion = current_balance - float(minimum_balance)

            if cushion < min_cushion:
                min_cushion = cushion
                min_cushion_date = curr_date_str
            if current_balance < min_balance:
                min_balance = current_balance

            trajectory.append(
                DailySimulationStep(
                    day=day,
                    date=curr_date_str,
                    inflows=inflows,
                    outflows=outflows,
                    net_change=net_change,
                    projected_balance=current_balance,
                    cushion=cushion,
                )
            )

        return {
            "start_date": start_date_str[:10],
            "starting_balance": float(starting_balance),
            "minimum_balance": float(minimum_balance),
            "days_simulated": len(trajectory),
            "trajectory": [step.to_dict() for step in trajectory],
            "min_cushion": round(min_cushion, 2),
            "min_balance": round(min_balance, 2),
            "min_cushion_date": min_cushion_date,
            "is_safe": min_cushion >= 0.0,
        }

    @staticmethod
    def render_ascii_chart(
        sim_result: Dict[str, Any],
        width: int = 45,
        height: int = 10,
    ) -> str:
        """Renders ASCII chart displaying B(t) vs floor line M."""
        trajectory = sim_result["trajectory"]
        M = sim_result["minimum_balance"]
        balances = [s["projected_balance"] for s in trajectory]

        all_vals = balances + [M]
        min_v, max_v = min(all_vals), max(all_vals)
        v_range = max_v - min_v if max_v != min_v else 1.0

        step_stride = len(trajectory) / width
        sampled = [balances[min(int(c * step_stride), len(trajectory) - 1)] for c in range(width)]

        m_row = max(0, min(height - 1, int(round((M - min_v) / v_range * (height - 1)))))

        lines = [
            f"Projected Balance B(t) vs Floor (M = {M:,.0f})",
            "-" * (width + 15),
        ]
        for row in range(height - 1, -1, -1):
            row_val = min_v + (row / (height - 1)) * v_range
            chars = []
            for b in sampled:
                b_row = int(round((b - min_v) / v_range * (height - 1)))
                if b_row == row:
                    chars.append("*")
                elif row == m_row:
                    chars.append("-")
                elif b_row > row >= m_row:
                    chars.append("|")
                else:
                    chars.append(" ")
            tag = " [M floor]" if row == m_row else ""
            lines.append(f"{row_val:>10,.0f} |" + "".join(chars) + tag)

        lines.append("-" * (width + 15))
        lines.append(f"{'Day 0':>12}" + " " * (width - 12) + "Day 90")
        lines.append(
            f"Min Cushion: {sim_result['min_cushion']:+,.2f} on {sim_result['min_cushion_date']} | "
            f"Safe: {sim_result['is_safe']}"
        )
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Stage 3: Answering Business Questions from Simulation Trajectory
# ----------------------------------------------------------------------
class Stage3BusinessLogic:
    """
    Stage 3 of Simulation Pipeline: Answering the Business Questions.

    1. compute_amount_safe_to_pay:
       Largest one-time payment safe today = min over all t in [0, 90] of (B(t) - M).
       Capped at requested_amount, floored at 0.0.

    2. find_earliest_date_for_full_payment:
       Soonest day d such that paying requested_amount on day d keeps all days from d
       onward above M (min_{t >= d} Cushion(t) >= requested_amount).
       Returns Day 0 if safe today, or None if never safe within 90 days.

    3. test_payment_plan:
       Reusable fast checker for proposed payment plans (installments/partial):
       (a) Checks if balance ever dips below M across the 90-day simulation.
       (b) Checks if the last payment happens on or before desired_completion_date.
    """

    @staticmethod
    def compute_amount_safe_to_pay(
        sim_result: Dict[str, Any],
        requested_amount: float,
    ) -> float:
        """
        Computes the largest one-time payment the user could make today without
        ever breaching the minimum balance on any of the next 90 days.
        """
        min_cushion = float(sim_result.get("min_cushion", 0.0))
        if min_cushion <= 0.0:
            return 0.0
        return round(min(min_cushion, float(requested_amount)), 2)

    @staticmethod
    def find_earliest_date_for_full_payment(
        sim_result: Dict[str, Any],
        requested_amount: float,
    ) -> Optional[str]:
        """
        Finds the earliest date d where paying requested_amount on day d ensures
        every day from d onward stays above M.
        Returns 'today' (Day 0) if safe immediately; None if never safe in 90 days.
        """
        trajectory = sim_result.get("trajectory", [])
        if not trajectory:
            return None

        req_amt = float(requested_amount)
        n = len(trajectory)

        # Precompute suffix minimums of cushion:
        # suffix_min[d] = min_{t = d..n-1} Cushion(t)
        suffix_min = [0.0] * n
        current_min = float("inf")
        for i in range(n - 1, -1, -1):
            cushion_val = float(trajectory[i]["cushion"])
            if cushion_val < current_min:
                current_min = cushion_val
            suffix_min[i] = current_min

        # Earliest day d where all subsequent days maintain required cushion
        for d in range(n):
            if suffix_min[d] >= req_amt:
                return trajectory[d]["date"]

        return None

    @staticmethod
    def test_payment_plan(
        sim_result: Dict[str, Any],
        payments: List[Dict[str, Any]],
        desired_completion_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Tests any proposed payment schedule against the baseline trajectory.
        Subtracts each proposed payment starting from its date onward and checks:
        (a) does the balance ever dip below M anywhere in the 90 days?
        (b) does the last payment happen by the user's desired deadline?

        payments format: [{'date': 'YYYY-MM-DD', 'amount': float}, ...]
        """
        trajectory = sim_result.get("trajectory", [])
        M = float(sim_result.get("minimum_balance", 0.0))

        if not payments:
            return {
                "is_safe": True,
                "balance_safe": True,
                "completed_by_deadline": True,
                "min_cushion_with_plan": sim_result.get("min_cushion", 0.0),
                "min_balance_with_plan": sim_result.get("min_balance", 0.0),
                "last_payment_date": None,
                "total_paid": 0.0,
            }

        sorted_payments = sorted(payments, key=lambda x: str(x.get("date", ""))[:10])
        last_payment_date = str(sorted_payments[-1].get("date", ""))[:10]
        total_paid = sum(float(p.get("amount", 0.0)) for p in sorted_payments)

        # (b) Check deadline condition
        completed_by_deadline = True
        if desired_completion_date:
            deadline_str = str(desired_completion_date)[:10]
            if last_payment_date > deadline_str:
                completed_by_deadline = False

        # Map total payment additions per day
        payments_by_date: Dict[str, float] = {}
        for p in sorted_payments:
            p_date = str(p.get("date", ""))[:10]
            p_amt = float(p.get("amount", 0.0))
            payments_by_date[p_date] = payments_by_date.get(p_date, 0.0) + p_amt

        # (a) Check balance cushion with payments deducted from their date onward
        min_cushion_with_plan = float("inf")
        min_balance_with_plan = float("inf")
        cumulative_paid = 0.0

        for step in trajectory:
            s_date = step["date"]
            if s_date in payments_by_date:
                cumulative_paid += payments_by_date[s_date]

            adjusted_balance = float(step["projected_balance"]) - cumulative_paid
            adjusted_cushion = adjusted_balance - M

            if adjusted_cushion < min_cushion_with_plan:
                min_cushion_with_plan = adjusted_cushion
            if adjusted_balance < min_balance_with_plan:
                min_balance_with_plan = adjusted_balance

        balance_safe = min_cushion_with_plan >= 0.0
        is_safe = balance_safe and completed_by_deadline

        return {
            "is_safe": is_safe,
            "balance_safe": balance_safe,
            "completed_by_deadline": completed_by_deadline,
            "min_cushion_with_plan": round(min_cushion_with_plan, 2),
            "min_balance_with_plan": round(min_balance_with_plan, 2),
            "last_payment_date": last_payment_date,
            "total_paid": round(total_paid, 2),
        }


