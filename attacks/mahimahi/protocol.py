from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Final

CONTROL_MAGIC: Final[int] = 0x5341474541445631
CONTROL_VERSION: Final[int] = 2
DIRECTION_FLAG_SHARED_BIN_LOSS: Final[int] = 1 << 0

HEADER_STRUCT: Final[struct.Struct] = struct.Struct("<QIIQQQ")
DIRECTION_CONFIG_STRUCT: Final[struct.Struct] = struct.Struct("<dddIIIIdd")
DIRECTION_TELEMETRY_STRUCT: Final[struct.Struct] = struct.Struct("<dddIIQQQQQQQdddd")
CONTROL_SETTINGS_STRUCT: Final[struct.Struct] = struct.Struct("<dd")

HEADER_OFFSET: Final[int] = 0
UPLINK_CONFIG_OFFSET: Final[int] = HEADER_OFFSET + HEADER_STRUCT.size
DOWNLINK_CONFIG_OFFSET: Final[int] = UPLINK_CONFIG_OFFSET + DIRECTION_CONFIG_STRUCT.size
UPLINK_TELEMETRY_OFFSET: Final[int] = DOWNLINK_CONFIG_OFFSET + DIRECTION_CONFIG_STRUCT.size
DOWNLINK_TELEMETRY_OFFSET: Final[int] = UPLINK_TELEMETRY_OFFSET + DIRECTION_TELEMETRY_STRUCT.size
LABEL_OFFSET: Final[int] = DOWNLINK_TELEMETRY_OFFSET + DIRECTION_TELEMETRY_STRUCT.size
LABEL_SIZE: Final[int] = 64
RESERVED_SIZE: Final[int] = 256
CONTROL_BLOCK_SIZE: Final[int] = LABEL_OFFSET + LABEL_SIZE + RESERVED_SIZE
CONTROL_SETTINGS_OFFSET: Final[int] = LABEL_OFFSET + LABEL_SIZE
SEQUENCE_OFFSET: Final[int] = 16
UPDATE_COUNTER_OFFSET: Final[int] = 24


@dataclass(frozen=True)
class ControlSettings:
    shared_bin_loss_bin_ms: float = 0.0
    attack_interval_ms: float = 0.0

    def clamp(self) -> "ControlSettings":
        return ControlSettings(
            shared_bin_loss_bin_ms=max(float(self.shared_bin_loss_bin_ms), 0.0),
            attack_interval_ms=max(float(self.attack_interval_ms), 0.0),
        )


@dataclass(frozen=True)
class DirectionConfig:
    bandwidth_mbps: float
    loss_rate: float
    delay_ms: float
    queue_packets: int
    queue_bytes: int
    episode_step: int = 0
    flags: int = 0
    effective_after_abs_ms: float = 0.0
    reserved1: float = 0.0

    def clamp(self) -> "DirectionConfig":
        return DirectionConfig(
            bandwidth_mbps=max(float(self.bandwidth_mbps), 0.0),
            loss_rate=min(max(float(self.loss_rate), 0.0), 1.0),
            delay_ms=max(float(self.delay_ms), 0.0),
            queue_packets=max(int(self.queue_packets), 0),
            queue_bytes=max(int(self.queue_bytes), 0),
            episode_step=max(int(self.episode_step), 0),
            flags=int(self.flags),
            effective_after_abs_ms=max(float(self.effective_after_abs_ms), 0.0),
            reserved1=float(self.reserved1),
        )


@dataclass(frozen=True)
class DirectionTelemetry:
    applied_bandwidth_mbps: float = 0.0
    applied_loss_rate: float = 0.0
    applied_delay_ms: float = 0.0
    applied_queue_packets: int = 0
    applied_queue_bytes: int = 0
    enqueued_packets: int = 0
    dequeued_packets: int = 0
    dropped_packets: int = 0
    dropped_bytes: int = 0
    queue_occupancy_packets: int = 0
    queue_occupancy_bytes: int = 0
    last_apply_ms: int = 0
    departure_rate_mbps: float = 0.0
    queue_delay_ms: float = 0.0
    applied_step: float = 0.0
    applied_effective_after_abs_ms: float = 0.0


@dataclass(frozen=True)
class ControlBlockSnapshot:
    magic: int
    version: int
    byte_size: int
    sequence: int
    update_counter: int
    created_ms: int
    uplink: DirectionConfig
    downlink: DirectionConfig
    uplink_telemetry: DirectionTelemetry
    downlink_telemetry: DirectionTelemetry
    label: str
    settings: ControlSettings


def pack_control_settings(settings: ControlSettings) -> bytes:
    safe = settings.clamp()
    return CONTROL_SETTINGS_STRUCT.pack(
        float(safe.shared_bin_loss_bin_ms),
        float(safe.attack_interval_ms),
    )


def unpack_control_settings(buf: bytes) -> ControlSettings:
    values = CONTROL_SETTINGS_STRUCT.unpack(buf)
    return ControlSettings(
        shared_bin_loss_bin_ms=float(values[0]),
        attack_interval_ms=float(values[1]),
    )


def pack_direction_config(config: DirectionConfig) -> bytes:
    safe = config.clamp()
    return DIRECTION_CONFIG_STRUCT.pack(
        float(safe.bandwidth_mbps),
        float(safe.loss_rate),
        float(safe.delay_ms),
        int(safe.queue_packets),
        int(safe.queue_bytes),
        int(safe.episode_step),
        int(safe.flags),
        float(safe.effective_after_abs_ms),
        float(safe.reserved1),
    )


def unpack_direction_config(buf: bytes) -> DirectionConfig:
    values = DIRECTION_CONFIG_STRUCT.unpack(buf)
    return DirectionConfig(
        bandwidth_mbps=float(values[0]),
        loss_rate=float(values[1]),
        delay_ms=float(values[2]),
        queue_packets=int(values[3]),
        queue_bytes=int(values[4]),
        episode_step=int(values[5]),
        flags=int(values[6]),
        effective_after_abs_ms=float(values[7]),
        reserved1=float(values[8]),
    )


def unpack_direction_telemetry(buf: bytes) -> DirectionTelemetry:
    values = DIRECTION_TELEMETRY_STRUCT.unpack(buf)
    return DirectionTelemetry(
        applied_bandwidth_mbps=float(values[0]),
        applied_loss_rate=float(values[1]),
        applied_delay_ms=float(values[2]),
        applied_queue_packets=int(values[3]),
        applied_queue_bytes=int(values[4]),
        enqueued_packets=int(values[5]),
        dequeued_packets=int(values[6]),
        dropped_packets=int(values[7]),
        dropped_bytes=int(values[8]),
        queue_occupancy_packets=int(values[9]),
        queue_occupancy_bytes=int(values[10]),
        last_apply_ms=int(values[11]),
        departure_rate_mbps=float(values[12]),
        queue_delay_ms=float(values[13]),
        applied_step=float(values[14]),
        applied_effective_after_abs_ms=float(values[15]),
    )


def build_control_block(
    *,
    label: str,
    uplink: DirectionConfig,
    downlink: DirectionConfig,
    created_ms: int,
    settings: ControlSettings | None = None,
) -> bytes:
    block = bytearray(CONTROL_BLOCK_SIZE)
    HEADER_STRUCT.pack_into(
        block,
        HEADER_OFFSET,
        CONTROL_MAGIC,
        CONTROL_VERSION,
        CONTROL_BLOCK_SIZE,
        0,
        0,
        int(created_ms),
    )
    block[UPLINK_CONFIG_OFFSET : UPLINK_CONFIG_OFFSET + DIRECTION_CONFIG_STRUCT.size] = pack_direction_config(uplink)
    block[DOWNLINK_CONFIG_OFFSET : DOWNLINK_CONFIG_OFFSET + DIRECTION_CONFIG_STRUCT.size] = pack_direction_config(downlink)
    label_bytes = label.encode("utf-8", errors="ignore")[: LABEL_SIZE - 1]
    block[LABEL_OFFSET : LABEL_OFFSET + len(label_bytes)] = label_bytes
    block[
        CONTROL_SETTINGS_OFFSET : CONTROL_SETTINGS_OFFSET + CONTROL_SETTINGS_STRUCT.size
    ] = pack_control_settings(settings or ControlSettings())
    return bytes(block)


def unpack_control_block(buf: bytes) -> ControlBlockSnapshot:
    header = HEADER_STRUCT.unpack_from(buf, HEADER_OFFSET)
    label_bytes = bytes(buf[LABEL_OFFSET : LABEL_OFFSET + LABEL_SIZE]).split(b"\0", 1)[0]
    return ControlBlockSnapshot(
        magic=int(header[0]),
        version=int(header[1]),
        byte_size=int(header[2]),
        sequence=int(header[3]),
        update_counter=int(header[4]),
        created_ms=int(header[5]),
        uplink=unpack_direction_config(
            bytes(buf[UPLINK_CONFIG_OFFSET : UPLINK_CONFIG_OFFSET + DIRECTION_CONFIG_STRUCT.size])
        ),
        downlink=unpack_direction_config(
            bytes(buf[DOWNLINK_CONFIG_OFFSET : DOWNLINK_CONFIG_OFFSET + DIRECTION_CONFIG_STRUCT.size])
        ),
        uplink_telemetry=unpack_direction_telemetry(
            bytes(buf[UPLINK_TELEMETRY_OFFSET : UPLINK_TELEMETRY_OFFSET + DIRECTION_TELEMETRY_STRUCT.size])
        ),
        downlink_telemetry=unpack_direction_telemetry(
            bytes(buf[DOWNLINK_TELEMETRY_OFFSET : DOWNLINK_TELEMETRY_OFFSET + DIRECTION_TELEMETRY_STRUCT.size])
        ),
        label=label_bytes.decode("utf-8", errors="ignore"),
        settings=unpack_control_settings(
            bytes(buf[CONTROL_SETTINGS_OFFSET : CONTROL_SETTINGS_OFFSET + CONTROL_SETTINGS_STRUCT.size])
        ),
    )
