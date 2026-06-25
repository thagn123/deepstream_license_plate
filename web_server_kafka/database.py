import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

_raw_db_url = os.environ.get("DATABASE_URL", "sqlite:///./data/events.db")
if _raw_db_url.startswith("sqlite:///") and "+aiosqlite" not in _raw_db_url:
    DATABASE_URL = _raw_db_url.replace("sqlite:///", "sqlite+aiosqlite:///", 1)
else:
    DATABASE_URL = _raw_db_url


engine = create_async_engine(DATABASE_URL, echo=False)

AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


class Base(DeclarativeBase):
    pass


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


async def init_db():
    from models import Base as ModelBase  # noqa: F401
    async with engine.begin() as conn:
        await conn.run_sync(ModelBase.metadata.create_all)
