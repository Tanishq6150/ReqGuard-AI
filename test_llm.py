import os
import sys
import traceback
from pathlib import Path

print("=" * 70)
print("REQGUARD PRODUCTION PATH DIAGNOSTIC")
print("=" * 70)

# ---------------------------------------------------------
# 1. BASIC ENVIRONMENT
# ---------------------------------------------------------

print("\n[1] PYTHON ENVIRONMENT")
print("-" * 70)

print("Python executable:")
print(sys.executable)

print("\nPython version:")
print(sys.version)

print("\nCurrent working directory:")
print(os.getcwd())

print("\nSSL_CERT_FILE:")
print(os.getenv("SSL_CERT_FILE") or "<not set>")


# ---------------------------------------------------------
# 2. CONFIG MODULE
# ---------------------------------------------------------

print("\n[2] REQGUARD CONFIG")
print("-" * 70)

try:
    import app.core.config as config

    print("Config module:")
    print(config.__file__)

    print("\n_ENV_FILE:")
    print(config._ENV_FILE)

    print("\n_ENV_FILE exists:")
    print(config._ENV_FILE.exists())

except Exception as e:
    print("CONFIG IMPORT FAILED")
    print(type(e).__name__)
    print(repr(e))
    traceback.print_exc()
    raise SystemExit(1)


# ---------------------------------------------------------
# 3. RAW ENVIRONMENT
# ---------------------------------------------------------

print("\n[3] ENVIRONMENT VARIABLES")
print("-" * 70)

for key in [
    "LLM_PROVIDER",
    "LLM_MODEL",
    "LLM_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "API_BASE_URL",
    "API_HOST",
    "API_PORT",
]:
    value = os.getenv(key)

    if "KEY" in key:
        print(f"{key}: {'SET' if value else 'NOT SET'}")
        if value:
            print(f"    length = {len(value)}")
    else:
        print(f"{key}: {value!r}")


# ---------------------------------------------------------
# 4. REQGUARD SETTINGS
# ---------------------------------------------------------

print("\n[4] REQGUARD SETTINGS")
print("-" * 70)

try:
    from app.core.config import get_settings

    settings = get_settings()

    print("Settings object:")
    print(type(settings))

    print("\nProvider:")
    print(repr(settings.llm.provider))

    print("\nModel:")
    print(repr(settings.llm.model))

    print("\nAPI key configured:")
    print(bool(settings.llm_api_key))

    print("\nAPI key length:")
    print(len(settings.llm_api_key or ""))

except Exception as e:
    print("SETTINGS FAILED")
    print(type(e).__name__)
    print(repr(e))
    traceback.print_exc()
    raise SystemExit(1)


# ---------------------------------------------------------
# 5. PROVIDER SELECTION
# ---------------------------------------------------------

print("\n[5] PROVIDER SELECTION")
print("-" * 70)

try:
    from app.services.llm_service import get_provider

    provider = get_provider()

    print("Provider object:")
    print(provider)

    print("\nProvider class:")
    print(type(provider).__name__)

    print("\nProvider module:")
    print(type(provider).__module__)

except Exception as e:
    print("PROVIDER SELECTION FAILED")
    print(type(e).__name__)
    print(repr(e))
    traceback.print_exc()
    raise SystemExit(1)


# ---------------------------------------------------------
# 6. PROVIDER INTERNAL STATE
# ---------------------------------------------------------

print("\n[6] PROVIDER INTERNAL STATE")
print("-" * 70)

try:

    print("Provider attributes:")

    for name, value in vars(provider).items():

        if "key" in name.lower() or "token" in name.lower():
            if value:
                print(f"{name}: SET (length={len(str(value))})")
            else:
                print(f"{name}: NOT SET")

        else:
            print(f"{name}: {value!r}")

except Exception as e:
    print("Could not inspect provider state:")
    print(type(e).__name__)
    print(repr(e))


# ---------------------------------------------------------
# 7. ACTUAL PRODUCTION GENERATE_JSON
# ---------------------------------------------------------

print("\n[7] PRODUCTION generate_json()")
print("-" * 70)

try:

    from app.services.llm_service import generate_json

    print("Calling generate_json()...")

    result = generate_json(
        "Return JSON with exactly one key called status and "
        "the value TEST."
    )

    print("\nSUCCESS")

    print("Result type:")
    print(type(result))

    print("\nResult:")
    print(result)

except Exception as e:

    print("\nGENERATE_JSON FAILED")

    print("Exception type:")
    print(type(e).__name__)

    print("\nException:")
    print(repr(e))

    print("\nFULL TRACEBACK:")
    traceback.print_exc()


# ---------------------------------------------------------
# 8. IMPORT FRONTEND
# ---------------------------------------------------------

print("\n[8] FRONTEND IMPORT CHECK")
print("-" * 70)

try:

    import frontend.app as frontend_app

    print("Frontend imported successfully.")

    print("Frontend file:")
    print(frontend_app.__file__)

except Exception as e:

    print("FRONTEND IMPORT FAILED")

    print(type(e).__name__)
    print(repr(e))

    traceback.print_exc()


# ---------------------------------------------------------
# FINISHED
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)