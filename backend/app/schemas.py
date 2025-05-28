from pydantic import BaseModel

class StoryOut(BaseModel):
    scene: str
    confidence: float
    story: str
