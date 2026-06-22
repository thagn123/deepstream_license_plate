from datetime import datetime
from sqlalchemy import (
    Boolean, DateTime, Float, Index, Integer, String, UniqueConstraint,
    ForeignKey, func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database import Base


class Device(Base):
    __tablename__ = "devices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    device_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    project_name: Mapped[str] = mapped_column(String, index=True, nullable=False)
    hostname: Mapped[str | None] = mapped_column(String, nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String, nullable=True)
    last_fps: Mapped[float] = mapped_column(Float, default=0.0)
    last_gpu_temp: Mapped[float] = mapped_column(Float, default=0.0)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    __table_args__ = (
        UniqueConstraint("device_id", "project_name", name="uq_device_project"),
        Index("idx_devices_last_seen", "last_seen_at"),
    )


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    project_name: Mapped[str] = mapped_column(String, nullable=False)
    device_id: Mapped[str] = mapped_column(String, nullable=False)
    camera_id: Mapped[str | None] = mapped_column(String, nullable=True)
    camera_name: Mapped[str | None] = mapped_column(String, nullable=True)
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    message: Mapped[str | None] = mapped_column(String, nullable=True)
    plate_text: Mapped[str | None] = mapped_column(String, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    object_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    track_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bbox: Mapped[str | None] = mapped_column(String, nullable=True)
    fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_name: Mapped[str | None] = mapped_column(String, nullable=True)
    model_version: Mapped[str | None] = mapped_column(String, nullable=True)
    raw_metadata: Mapped[str | None] = mapped_column(String, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True, default=func.now())
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    images: Mapped[list["EventImage"]] = relationship(
        "EventImage", back_populates="event", foreign_keys="EventImage.event_id",
        primaryjoin="Event.event_id == EventImage.event_id",
        lazy="select",
    )

    __table_args__ = (
        Index("idx_events_timestamp", "timestamp"),
        Index("idx_events_search", "project_name", "device_id", "timestamp"),
        Index("idx_events_plate", "plate_text"),
    )


class EventImage(Base):
    __tablename__ = "event_images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[str] = mapped_column(
        String, ForeignKey("events.event_id"), nullable=False
    )
    image_type: Mapped[str] = mapped_column(String, nullable=False)
    image_url: Mapped[str | None] = mapped_column(String, nullable=True)
    thumb_url: Mapped[str | None] = mapped_column(String, nullable=True)
    file_path: Mapped[str | None] = mapped_column(String, nullable=True)
    is_available: Mapped[bool] = mapped_column(Boolean, default=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    event: Mapped["Event"] = relationship(
        "Event", back_populates="images",
        foreign_keys=[event_id],
        primaryjoin="EventImage.event_id == Event.event_id",
    )

    __table_args__ = (
        Index("idx_event_images_event_id", "event_id"),
    )
