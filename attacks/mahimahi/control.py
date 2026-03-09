from __future__ import annotations

from dataclasses import replace
import mmap
import os
import struct
import time
from typing import Mapping

from .protocol import (
    CONTROL_BLOCK_SIZE,
    CONTROL_MAGIC,
    CONTROL_VERSION,
    DOWNLINK_CONFIG_OFFSET,
    HEADER_STRUCT,
    SEQUENCE_OFFSET,
    UPLINK_CONFIG_OFFSET,
    UPDATE_COUNTER_OFFSET,
    ControlBlockSnapshot,
    DirectionConfig,
    build_control_block,
    pack_direction_config,
    unpack_control_block,
)


class MahimahiControlClient:
    def __init__(
        self,
        path: str,
        *,
        label: str = "sage-adv",
        initial_uplink: DirectionConfig | None = None,
        initial_downlink: DirectionConfig | None = None,
    ) -> None:
        self.path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        os.ftruncate(self._fd, CONTROL_BLOCK_SIZE)
        self._mmap = mmap.mmap(self._fd, CONTROL_BLOCK_SIZE)

        uplink = initial_uplink or DirectionConfig(12.0, 0.0, 0.0, 128, 0)
        downlink = initial_downlink or DirectionConfig(12.0, 0.0, 0.0, 128, 0)
        if not self._is_initialized():
            block = build_control_block(
                label=label,
                uplink=uplink,
                downlink=downlink,
                created_ms=int(time.time() * 1000.0),
            )
            self._mmap.seek(0)
            self._mmap.write(block)
            self._mmap.flush()

    def close(self) -> None:
        self._mmap.close()
        os.close(self._fd)

    def __enter__(self) -> "MahimahiControlClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _is_initialized(self) -> bool:
        magic, version, byte_size, _, _, _ = HEADER_STRUCT.unpack_from(self._mmap, 0)
        return (
            int(magic) == CONTROL_MAGIC
            and int(version) == CONTROL_VERSION
            and int(byte_size) == CONTROL_BLOCK_SIZE
        )

    def _stable_read(self) -> bytes:
        for _ in range(16):
            seq_before = struct.unpack_from("<Q", self._mmap, SEQUENCE_OFFSET)[0]
            if seq_before & 1:
                time.sleep(0.001)
                continue
            buf = self._mmap[:CONTROL_BLOCK_SIZE]
            seq_after = struct.unpack_from("<Q", buf, SEQUENCE_OFFSET)[0]
            if seq_before == seq_after and not (seq_after & 1):
                return bytes(buf)
        return bytes(self._mmap[:CONTROL_BLOCK_SIZE])

    def snapshot(self) -> ControlBlockSnapshot:
        return unpack_control_block(self._stable_read())

    def update(
        self,
        *,
        uplink: DirectionConfig | None = None,
        downlink: DirectionConfig | None = None,
    ) -> ControlBlockSnapshot:
        current = self.snapshot()
        next_uplink = (uplink or current.uplink).clamp()
        next_downlink = (downlink or current.downlink).clamp()

        sequence = int(current.sequence)
        if sequence & 1:
            sequence += 1
        odd_sequence = sequence + 1
        even_sequence = sequence + 2

        struct.pack_into("<Q", self._mmap, SEQUENCE_OFFSET, odd_sequence)
        self._mmap[UPLINK_CONFIG_OFFSET : UPLINK_CONFIG_OFFSET + len(pack_direction_config(next_uplink))] = pack_direction_config(next_uplink)
        self._mmap[DOWNLINK_CONFIG_OFFSET : DOWNLINK_CONFIG_OFFSET + len(pack_direction_config(next_downlink))] = pack_direction_config(next_downlink)
        struct.pack_into("<Q", self._mmap, UPDATE_COUNTER_OFFSET, int(current.update_counter) + 1)
        struct.pack_into("<Q", self._mmap, SEQUENCE_OFFSET, even_sequence)
        self._mmap.flush()
        return self.snapshot()

    def update_from_mapping(self, params: Mapping[str, float | int]) -> ControlBlockSnapshot:
        current = self.snapshot()
        uplink = replace(
            current.uplink,
            bandwidth_mbps=float(params.get("uplink_bw_mbps", current.uplink.bandwidth_mbps)),
            loss_rate=float(params.get("uplink_loss", current.uplink.loss_rate)),
            delay_ms=float(params.get("uplink_delay_ms", current.uplink.delay_ms)),
            queue_packets=int(params.get("uplink_queue_packets", current.uplink.queue_packets)),
            queue_bytes=int(params.get("uplink_queue_bytes", current.uplink.queue_bytes)),
        )
        downlink = replace(
            current.downlink,
            bandwidth_mbps=float(params.get("downlink_bw_mbps", current.downlink.bandwidth_mbps)),
            loss_rate=float(params.get("downlink_loss", current.downlink.loss_rate)),
            delay_ms=float(params.get("downlink_delay_ms", current.downlink.delay_ms)),
            queue_packets=int(params.get("downlink_queue_packets", current.downlink.queue_packets)),
            queue_bytes=int(params.get("downlink_queue_bytes", current.downlink.queue_bytes)),
        )
        return self.update(uplink=uplink, downlink=downlink)
