import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

USERS_FILE = Path(__file__).parent.parent / "users.json"


def _load_users() -> list[dict]:
    with open(USERS_FILE) as f:
        return json.load(f)


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/login")
def login(body: LoginRequest):
    users = _load_users()
    for user in users:
        if user["username"] == body.username and user["password"] == body.password:
            return {"success": True, "username": body.username}
    raise HTTPException(status_code=401, detail="Invalid username or password")
