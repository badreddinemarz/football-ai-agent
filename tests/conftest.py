import os
import sys
from pathlib import Path

# Make `import src...` work when running pytest from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Never send the tests' fake runs to LangSmith. Set before src.llm loads .env: load_dotenv keeps existing values.
os.environ["LANGSMITH_TRACING"] = "false"
