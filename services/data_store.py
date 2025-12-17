from __future__ import annotations

from datetime import datetime
from threading import Lock
from typing import Dict, Iterable, List, Optional


class ReliabilityDataStore:
    """Thread-safe in-memory store for reliability records."""

    def __init__(self) -> None:
        self._records: List[Dict[str, object]] = []
        self._lock = Lock()

    def _normalize_record(self, record: Dict[str, object], source: str) -> Dict[str, object]:
        failures = int(record.get("failures", 0) or 0)
        mtbf = float(record.get("mtbf", 0) or 0) or 1.0
        runtime = float(record.get("runtime", 0) or 0) or mtbf
        module = str(record.get("module") or "Imported module")
        timestamp = record.get("timestamp")
        if not timestamp:
            timestamp = datetime.utcnow().isoformat()

        return {
            "module": module,
            "failures": failures,
            "mtbf": round(mtbf, 3),
            "runtime": round(runtime, 3),
            "source": source,
            "timestamp": timestamp,
        }

    def add_record(self, record: Dict[str, object], source: str) -> Dict[str, object]:
        normalized = self._normalize_record(record, source)
        with self._lock:
            self._records.append(normalized)
        return normalized

    def add_records(self, records: Iterable[Dict[str, object]], source: str) -> int:
        count = 0
        with self._lock:
            for record in records:
                normalized = self._normalize_record(record, source)
                self._records.append(normalized)
                count += 1
        return count

    def get_records(self) -> List[Dict[str, object]]:
        with self._lock:
            return list(self._records)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()


store = ReliabilityDataStore()
