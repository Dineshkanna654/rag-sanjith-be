"""Seed script: creates default org, roles, and migrates users from users.json.

Run via: python -m app.db.seed
"""

import asyncio
import json
from pathlib import Path

from passlib.hash import bcrypt
from sqlalchemy import select

from app.db.engine import async_session
from app.db.models import Organization, Role, User, UserRole

USERS_JSON = Path(__file__).parent.parent / "users.json"


async def seed() -> None:
    async with async_session() as session:
        # --- Default organization ---
        result = await session.execute(select(Organization).where(Organization.slug == "default"))
        org = result.scalar_one_or_none()
        if org is None:
            org = Organization(name="Default Organization", slug="default")
            session.add(org)
            await session.flush()
            print("Created organization: Default Organization")
        else:
            print("Organization 'default' already exists, skipping")

        # --- Roles ---
        role_map: dict[str, Role] = {}
        for role_name, description in [
            ("admin", "Full system access"),
            ("user", "Standard user access"),
            ("viewer", "Read-only access"),
        ]:
            result = await session.execute(select(Role).where(Role.name == role_name))
            role = result.scalar_one_or_none()
            if role is None:
                role = Role(name=role_name, description=description)
                session.add(role)
                await session.flush()
                print(f"Created role: {role_name}")
            else:
                print(f"Role '{role_name}' already exists, skipping")
            role_map[role_name] = role

        # --- Migrate users from JSON ---
        with open(USERS_JSON) as f:
            json_users = json.load(f)

        for entry in json_users:
            username = entry["username"]
            password = entry["password"]

            result = await session.execute(select(User).where(User.username == username))
            if result.scalar_one_or_none() is not None:
                print(f"User '{username}' already exists, skipping")
                continue

            user = User(
                username=username,
                password_hash=bcrypt.hash(password),
                org_id=org.id,
                is_active=True,
            )
            session.add(user)
            await session.flush()

            # Assign role: admin gets "admin", others get "user"
            assigned_role = role_map["admin"] if username == "admin" else role_map["user"]
            session.add(UserRole(user_id=user.id, role_id=assigned_role.id))
            print(f"Created user: {username} (role: {assigned_role.name})")

        await session.commit()
        print("Seed complete.")


if __name__ == "__main__":
    asyncio.run(seed())
