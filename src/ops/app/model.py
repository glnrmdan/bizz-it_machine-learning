from pydantic import BaseModel

class ImageParser(BaseModel):
    image: str