from pathlib import Path
from pydantic_settings import BaseSettings

class Paths():

    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    INDEX_DIR: Path = BASE_DIR / "indexes"