"""
DataStore and RequestContext Assembler.

Loads all provided CSV datasets once into memory at startup,
indexes by join keys (user_id, request_id, related_event_id, linked_event_id),
supports OCR path resolution and memoized missing-amount filling,
and assembles an isolated, clean RequestContext dictionary per request on demand.
"""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import pandas as pd


# ----------------------------------------------------------------------
# Sanitization & Type Conversion Helpers
# ----------------------------------------------------------------------
def _clean_str(val: Any) -> str:
    """Converts NaN, None, or float representations to clean stripped strings."""
    if pd.isna(val) or val is None:
        return ""
    return str(val).strip()


def _parse_pipe_set(val: Any) -> Set[str]:
    """Parses a pipe-delimited string (e.g. 'rent|groceries') into a set of strings."""
    clean = _clean_str(val)
    if not clean:
        return set()
    return {item.strip() for item in clean.split("|") if item.strip()}


def _to_bool(val: Any) -> bool:
    """Safely converts CSV string/bool values to a strict Python boolean."""
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "1", "yes")


# ----------------------------------------------------------------------
# Core DataStore Class
# ----------------------------------------------------------------------
class DataStore:
    """
    In-memory data store that loads each CSV once at startup
    and provides indexed lookups, OCR image resolution, and isolated RequestContext assembly.
    """

    def __init__(
        self,
        dataset_dir: Optional[Union[str, Path]] = None,
        requests_filename: str = "requests.csv",
    ):
        if dataset_dir is not None:
            self.dataset_dir = Path(dataset_dir)
        else:
            default_path = Path(__file__).resolve().parent.parent / "dataset"
            self.dataset_dir = default_path if default_path.exists() else Path("dataset")

        self.requests_filename = requests_filename
        self.images_dir = self.dataset_dir / "media" / "images"

        # In-memory cache for resolved OCR values: {event_id: extracted_amount}
        self.ocr_cache: Dict[str, float] = {}

        # Load raw dataframes once
        self._load_datasets()

        # Build join key indexes
        self._build_indexes()

    def _load_datasets(self) -> None:
        """Loads each CSV file once into memory."""
        requests_path = self.dataset_dir / self.requests_filename
        profiles_path = self.dataset_dir / "financial_profiles.csv"
        events_path = self.dataset_dir / "financial_events.csv"
        options_path = self.dataset_dir / "request_payment_options.csv"
        messages_path = self.dataset_dir / "messages.csv"
        images_path = self.dataset_dir / "images.csv"
        rates_path = self.dataset_dir / "exchange_rates.csv"

        self.requests_df = pd.read_csv(requests_path) if requests_path.exists() else pd.DataFrame()
        self.profiles_df = pd.read_csv(profiles_path) if profiles_path.exists() else pd.DataFrame()
        self.events_df = pd.read_csv(events_path) if events_path.exists() else pd.DataFrame()
        self.options_df = pd.read_csv(options_path) if options_path.exists() else pd.DataFrame()
        self.messages_df = pd.read_csv(messages_path) if messages_path.exists() else pd.DataFrame()
        self.images_df = pd.read_csv(images_path) if images_path.exists() else pd.DataFrame()
        self.rates_df = pd.read_csv(rates_path) if rates_path.exists() else pd.DataFrame()

    def _build_indexes(self) -> None:
        """Builds lookup hash indexes by join keys with sanitized fields."""
        # 1. Index requests by request_id and user_id
        self.requests_by_id: Dict[str, Dict[str, Any]] = {}
        self.requests_by_user: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in self.requests_df.to_dict(orient="records"):
            req_id = _clean_str(row.get("request_id"))
            user_id = _clean_str(row.get("user_id"))
            if req_id:
                # Normalize boolean flag
                row["allows_partial_payment"] = _to_bool(row.get("allows_partial_payment"))
                self.requests_by_id[req_id] = row
            if user_id:
                self.requests_by_user[user_id].append(row)

        # 2. Index financial profiles by user_id with pre-parsed category sets
        self.profiles_by_user: Dict[str, Dict[str, Any]] = {}
        for row in self.profiles_df.to_dict(orient="records"):
            user_id = _clean_str(row.get("user_id"))
            if user_id:
                # Pre-parse category sets to eliminate string splitting errors
                row["priorities_set"] = _parse_pipe_set(row.get("financial_priorities"))
                row["protected_set"] = _parse_pipe_set(row.get("expense_categories_to_protect"))
                row["reducible_set"] = _parse_pipe_set(
                    row.get("expense_categories_user_is_willing_to_reduce")
                )
                row["stoppable_set"] = _parse_pipe_set(
                    row.get("expense_categories_user_is_willing_to_stop")
                )
                row["payment_methods_set"] = _parse_pipe_set(
                    row.get("payment_methods_user_will_consider")
                )
                self.profiles_by_user[user_id] = row

        # 3. Index images with resolved absolute disk file paths
        self.images_by_user: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.images_by_request: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.images_by_event: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in self.images_df.to_dict(orient="records"):
            img_id = _clean_str(row.get("image_id"))
            user_id = _clean_str(row.get("user_id"))
            req_id = _clean_str(row.get("request_id"))
            rel_event_id = _clean_str(row.get("related_event_id"))

            # Resolve image path on disk
            row["image_path"] = str(self.images_dir / f"{img_id}.png")

            if user_id:
                self.images_by_user[user_id].append(row)
            if req_id:
                self.images_by_request[req_id].append(row)
            if rel_event_id:
                self.images_by_event[rel_event_id].append(row)

        # 4. Index messages cleanly partitioned between request-level and user-level
        self.messages_by_user: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.messages_by_request: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.messages_by_event: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in self.messages_df.to_dict(orient="records"):
            user_id = _clean_str(row.get("user_id"))
            req_id = _clean_str(row.get("request_id"))
            rel_event_id = _clean_str(row.get("related_event_id"))

            # Partition: only index under messages_by_user if NOT request-specific
            if user_id and not req_id:
                self.messages_by_user[user_id].append(row)
            if req_id:
                self.messages_by_request[req_id].append(row)
            if rel_event_id:
                self.messages_by_event[rel_event_id].append(row)

        # 5. Index financial events by user_id, event_id, and linked_event_id
        self.events_by_user: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.events_by_id: Dict[str, Dict[str, Any]] = {}
        self.events_by_linked_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in self.events_df.to_dict(orient="records"):
            user_id = _clean_str(row.get("user_id"))
            event_id = _clean_str(row.get("event_id"))
            linked_id = _clean_str(row.get("linked_event_id"))

            if user_id:
                self.events_by_user[user_id].append(row)
            if event_id:
                self.events_by_id[event_id] = row
            if linked_id:
                self.events_by_linked_id[linked_id].append(row)

        # 6. Index payment options by request_id
        self.options_by_request: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in self.options_df.to_dict(orient="records"):
            req_id = _clean_str(row.get("request_id"))
            if req_id:
                self.options_by_request[req_id].append(row)

        # 7. Index exchange rates by (rate_date, from_currency, to_currency)
        self.exchange_rates: Dict[tuple, float] = {}
        for row in self.rates_df.to_dict(orient="records"):
            date_str = _clean_str(row.get("rate_date"))[:10]
            from_curr = _clean_str(row.get("from_currency"))
            to_curr = _clean_str(row.get("to_currency"))
            rate = float(row.get("rate", 1.0))
            self.exchange_rates[(date_str, from_curr, to_curr)] = rate

    def get_exchange_rate(
        self, date_str: str, from_currency: str, to_currency: str
    ) -> Optional[float]:
        """Returns exchange rate between two currencies for a specific date (YYYY-MM-DD)."""
        if from_currency == to_currency:
            return 1.0
        clean_date = _clean_str(date_str)[:10]
        return self.exchange_rates.get((clean_date, from_currency, to_currency))

    def get_all_request_ids(self) -> List[str]:
        """Returns all request IDs loaded in the data store."""
        return list(self.requests_by_id.keys())

    def resolve_blank_amount(self, event_id: str, amount: float) -> None:
        """Updates the internal cache and event record with an OCR-extracted amount."""
        self.ocr_cache[event_id] = float(amount)
        if event_id in self.events_by_id:
            self.events_by_id[event_id]["amount"] = float(amount)

    def assemble_request_context(self, request_id: str) -> Dict[str, Any]:
        """
        Assembles an isolated, clean RequestContext dictionary per request on demand.

        Returns:
            Dict containing isolated copies of:
            - request_id: Target request ID
            - user_id: User identifier
            - request: Specific request details
            - profile: Financial profile of the user (with parsed category sets)
            - events: Enriched financial events for this user
            - events_by_id: O(1) map of event_id -> event dict
            - payment_options: Payment options for the request
            - messages: Partitioned request-level and user-level messages
            - images: Request-level and user-level images (with image_path)
            - blank_amount_events: Financial events requiring OCR extraction
            - exchange_rates: Exchange rate mapping table
        """
        req = self.requests_by_id.get(request_id)
        if req is None:
            raise KeyError(f"Request ID '{request_id}' not found.")

        user_id = _clean_str(req.get("user_id"))
        profile = self.profiles_by_user.get(user_id, {})
        user_events = self.events_by_user.get(user_id, [])
        payment_options = self.options_by_request.get(request_id, [])

        # Partitioned messages
        request_messages = [dict(m) for m in self.messages_by_request.get(request_id, [])]
        user_messages = [dict(m) for m in self.messages_by_user.get(user_id, [])]

        # Partitioned images
        request_images = [dict(img) for img in self.images_by_request.get(request_id, [])]
        user_images = [dict(img) for img in self.images_by_user.get(user_id, [])]

        # Assemble enriched events using fast dictionary copies
        enriched_events: List[Dict[str, Any]] = []
        events_by_id: Dict[str, Dict[str, Any]] = {}
        blank_amount_events: List[Dict[str, Any]] = []

        for event in user_events:
            ev_copy = dict(event)
            event_id = _clean_str(ev_copy.get("event_id"))

            # Attach related messages and images
            ev_copy["related_messages"] = [
                dict(m) for m in self.messages_by_event.get(event_id, [])
            ]
            related_imgs = [
                dict(img) for img in self.images_by_event.get(event_id, [])
            ]
            ev_copy["related_images"] = related_imgs

            # Check OCR cache first
            if event_id in self.ocr_cache:
                ev_copy["amount"] = self.ocr_cache[event_id]

            # Detect blank amount events
            amount_val = ev_copy.get("amount")
            if pd.isna(amount_val) or amount_val == "" or amount_val is None:
                # Pre-package primary image metadata for OCR convenience
                img_info = related_imgs[0] if related_imgs else {}
                ev_copy["ocr_target"] = {
                    "event_id": event_id,
                    "image_id": img_info.get("image_id", ""),
                    "image_path": img_info.get("image_path", ""),
                    "currency": ev_copy.get("currency", ""),
                }
                blank_amount_events.append(ev_copy)

            enriched_events.append(ev_copy)
            if event_id:
                events_by_id[event_id] = ev_copy

        # Assemble final context dictionary
        context: Dict[str, Any] = {
            "request_id": request_id,
            "user_id": user_id,
            "request": dict(req),
            "profile": dict(profile),
            "events": enriched_events,
            "events_by_id": events_by_id,
            "payment_options": [dict(opt) for opt in payment_options],
            "messages": {
                "request_messages": request_messages,
                "user_messages": user_messages,
            },
            "images": {
                "request_images": request_images,
                "user_images": user_images,
            },
            "blank_amount_events": blank_amount_events,
            "exchange_rates": dict(self.exchange_rates),
        }

        return context
