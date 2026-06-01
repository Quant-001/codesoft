import os
from typing import Annotated

from fastapi import Header, HTTPException


ADMIN_KEY = os.getenv("FRAUD_ADMIN_KEY")
ANALYST_KEY = os.getenv("FRAUD_ANALYST_KEY")


def require_role(x_api_key: Annotated[str | None, Header()] = None) -> str:
    if not ADMIN_KEY and not ANALYST_KEY:
        return "Fraud Analyst"

    if ADMIN_KEY and x_api_key == ADMIN_KEY:
        return "Admin"
    if ANALYST_KEY and x_api_key == ANALYST_KEY:
        return "Fraud Analyst"

    raise HTTPException(status_code=401, detail="Valid analyst API key required")
