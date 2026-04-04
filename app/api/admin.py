import uuid

from fastapi import APIRouter, Depends, HTTPException
from passlib.hash import bcrypt
from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser, require_role
from app.db.engine import get_db
from app.db.models import Organization, Role, User, UserRole

router = APIRouter(prefix="/admin", tags=["admin"])

_admin_dep = require_role("admin")


# ─── Schemas ──────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    username: str
    password: str
    email: str | None = None
    org_id: str | None = None
    role_names: list[str] = []


class UserUpdate(BaseModel):
    email: str | None = None
    is_active: bool | None = None
    org_id: str | None = None


class UserOut(BaseModel):
    id: str
    username: str
    email: str | None
    org_id: str | None
    is_active: bool
    roles: list[str]

    model_config = {"from_attributes": True}


class RoleCreate(BaseModel):
    name: str
    description: str | None = None


class RoleOut(BaseModel):
    id: str
    name: str
    description: str | None

    model_config = {"from_attributes": True}


class RoleAssign(BaseModel):
    role_name: str


class OrgCreate(BaseModel):
    name: str
    slug: str


class OrgOut(BaseModel):
    id: str
    name: str
    slug: str

    model_config = {"from_attributes": True}


# ─── Helpers ──────────────────────────────────────────────────────────

def _user_out(user: User) -> dict:
    return {
        "id": str(user.id),
        "username": user.username,
        "email": user.email,
        "org_id": str(user.org_id) if user.org_id else None,
        "is_active": user.is_active,
        "roles": [r.name for r in user.roles],
    }


def _role_out(role: Role) -> dict:
    return {"id": str(role.id), "name": role.name, "description": role.description}


def _org_out(org: Organization) -> dict:
    return {"id": str(org.id), "name": org.name, "slug": org.slug}


# ─── User CRUD ────────────────────────────────────────────────────────

@router.post("/users", response_model=UserOut, status_code=201)
async def create_user(
    body: UserCreate,
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(_admin_dep),
):
    # Check username uniqueness
    existing = await db.execute(select(User).where(User.username == body.username))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Username already exists")

    user = User(
        username=body.username,
        password_hash=bcrypt.hash(body.password),
        email=body.email,
        org_id=uuid.UUID(body.org_id) if body.org_id else current_user.org_id,
    )
    db.add(user)
    await db.flush()

    # Assign requested roles
    if body.role_names:
        result = await db.execute(select(Role).where(Role.name.in_(body.role_names)))
        roles = result.scalars().all()
        for role in roles:
            db.add(UserRole(user_id=user.id, role_id=role.id))

    await db.commit()
    # Re-fetch to get roles populated
    result = await db.execute(select(User).where(User.id == user.id))
    user = result.scalar_one()
    return _user_out(user)


@router.get("/users", response_model=list[UserOut])
async def list_users(
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(_admin_dep),
):
    result = await db.execute(select(User).order_by(User.created_at))
    users = result.scalars().all()
    return [_user_out(u) for u in users]


@router.get("/users/{user_id}", response_model=UserOut)
async def get_user(
    user_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(_admin_dep),
):
    result = await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return _user_out(user)


@router.patch("/users/{user_id}", response_model=UserOut)
async def update_user(
    user_id: str,
    body: UserUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(_admin_dep),
):
    result = await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    if body.email is not None:
        user.email = body.email
    if body.is_active is not None:
        user.is_active = body.is_active
    if body.org_id is not None:
        user.org_id = uuid.UUID(body.org_id)

    await db.commit()
    await db.refresh(user)
    return _user_out(user)


@router.post("/users/{user_id}/roles", response_model=UserOut)
async def assign_role(
    user_id: str,
    body: RoleAssign,
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(_admin_dep),
):
    result = await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    result = await db.execute(select(Role).where(Role.name == body.role_name))
    role = result.scalar_one_or_none()
    if role is None:
        raise HTTPException(status_code=404, detail="Role not found")

    # Check for existing assignment
    existing = await db.execute(
        select(UserRole).where(UserRole.user_id == user.id, UserRole.role_id == role.id)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Role already assigned")

    db.add(UserRole(user_id=user.id, role_id=role.id))
    await db.commit()

    # Expire cached state so selectin reload picks up new roles
    db.expire_all()
    result = await db.execute(select(User).where(User.id == user.id))
    user = result.scalar_one()
    return _user_out(user)


@router.delete("/users/{user_id}/roles/{role_name}", response_model=UserOut)
async def remove_role(
    user_id: str,
    role_name: str,
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(_admin_dep),
):
    result = await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    result = await db.execute(select(Role).where(Role.name == role_name))
    role = result.scalar_one_or_none()
    if role is None:
        raise HTTPException(status_code=404, detail="Role not found")

    await db.execute(
        delete(UserRole).where(UserRole.user_id == user.id, UserRole.role_id == role.id)
    )
    await db.commit()

    # Expire cached state so selectin reload picks up removed roles
    db.expire_all()
    result = await db.execute(select(User).where(User.id == user.id))
    user = result.scalar_one()
    return _user_out(user)


# ─── Role CRUD ────────────────────────────────────────────────────────

@router.post("/roles", response_model=RoleOut, status_code=201)
async def create_role(
    body: RoleCreate,
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(_admin_dep),
):
    existing = await db.execute(select(Role).where(Role.name == body.name))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Role already exists")

    role = Role(name=body.name, description=body.description)
    db.add(role)
    await db.commit()
    await db.refresh(role)
    return _role_out(role)


@router.get("/roles", response_model=list[RoleOut])
async def list_roles(
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(_admin_dep),
):
    result = await db.execute(select(Role).order_by(Role.name))
    return [_role_out(r) for r in result.scalars().all()]


# ─── Organization CRUD ────────────────────────────────────────────────

@router.post("/orgs", response_model=OrgOut, status_code=201)
async def create_org(
    body: OrgCreate,
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(_admin_dep),
):
    existing = await db.execute(select(Organization).where(Organization.slug == body.slug))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Organization slug already exists")

    org = Organization(name=body.name, slug=body.slug)
    db.add(org)
    await db.commit()
    await db.refresh(org)
    return _org_out(org)


@router.get("/orgs", response_model=list[OrgOut])
async def list_orgs(
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(_admin_dep),
):
    result = await db.execute(select(Organization).order_by(Organization.name))
    return [_org_out(o) for o in result.scalars().all()]
