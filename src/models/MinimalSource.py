from pydantic import BaseModel
from typing import Optional


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int
    content: Optional[str] = None
