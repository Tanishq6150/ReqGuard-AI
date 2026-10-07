"""ReqGuard AI FastAPI application."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import get_settings
from app.core.exceptions import ReqGuardError
from app.core.logging import logger

settings = get_settings()

app = FastAPI(
    title=settings.application.name,
    version=settings.application.version,
    description="NLP-based software requirement analysis and intelligent test case generation.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix=settings.api.prefix)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
    details = exc.errors()
    message = "Requirement cannot be empty."
    for item in details:
        loc = item.get("loc") or []
        if "requirement" in loc:
            raw = str(item.get("msg") or "")
            if "maximum length" in raw.lower() or "exceeds" in raw.lower():
                message = raw.replace("Value error, ", "")
            else:
                message = "Requirement cannot be empty."
            break
        message = item.get("msg") or message
    return JSONResponse(status_code=400, content={"error": message})


@app.exception_handler(ReqGuardError)
async def handle_domain_error(_request: Request, exc: ReqGuardError) -> JSONResponse:
    logger.warning("Handled application error: %s", exc.error)
    return JSONResponse(status_code=exc.status_code, content={"error": exc.error})


@app.exception_handler(Exception)
async def handle_unexpected_error(_request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled server error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": "An unexpected server error occurred."},
    )
