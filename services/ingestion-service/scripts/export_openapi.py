import json
import os
import sys
from pathlib import Path

# Add service directory and repo root to Python path
current_dir = Path(__file__).resolve().parent
service_root = current_dir.parent
repo_root = service_root.parent.parent

if str(service_root) not in sys.path:
    sys.path.insert(0, str(service_root))
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.main import app


def export_openapi() -> None:
    """Exports OpenAPI JSON spec to docs/openapi/ingestion-service.json."""
    output_dir = repo_root / "docs" / "openapi"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "ingestion-service.json"

    openapi_schema = app.openapi()
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(openapi_schema, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] Exported OpenAPI specification to {output_file}")


if __name__ == "__main__":
    export_openapi()
