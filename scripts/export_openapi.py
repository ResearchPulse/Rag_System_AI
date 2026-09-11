import json
import sys
from pathlib import Path

# Add project root to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.main import app


def export_openapi() -> None:
    """Exports unified OpenAPI JSON specification to docs/openapi/openapi.json."""
    output_dir = project_root / "docs" / "openapi"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "openapi.json"

    openapi_schema = app.openapi()
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(openapi_schema, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] Exported unified OpenAPI specification to {output_file}")


if __name__ == "__main__":
    export_openapi()
