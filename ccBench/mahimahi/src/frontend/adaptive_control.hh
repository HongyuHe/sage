/* -*-mode:c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

#ifndef ADAPTIVE_CONTROL_HH
#define ADAPTIVE_CONTROL_HH

#include <cstdint>
#include <string>

namespace adaptive {

constexpr uint64_t CONTROL_MAGIC = 0x5341474541445631ULL; /* SAGEADV1 */
constexpr uint32_t CONTROL_VERSION = 2;

enum class Direction : uint32_t
{
    Uplink = 0,
    Downlink = 1,
};

#pragma pack(push, 1)
struct DirectionConfig
{
    double bandwidth_mbps;
    double loss_rate;
    double delay_ms;
    uint32_t queue_packets;
    uint32_t queue_bytes;
    uint32_t episode_step;
    uint32_t flags;
    double effective_after_abs_ms;
    double reserved1;
};

struct DirectionTelemetry
{
    double applied_bandwidth_mbps;
    double applied_loss_rate;
    double applied_delay_ms;
    uint32_t applied_queue_packets;
    uint32_t applied_queue_bytes;
    uint64_t enqueued_packets;
    uint64_t dequeued_packets;
    uint64_t dropped_packets;
    uint64_t dropped_bytes;
    uint64_t queue_occupancy_packets;
    uint64_t queue_occupancy_bytes;
    uint64_t last_apply_ms;
    double departure_rate_mbps;
    double queue_delay_ms;
    double applied_step;
    double applied_effective_after_abs_ms;
};

struct ControlBlock
{
    uint64_t magic;
    uint32_t version;
    uint32_t byte_size;
    uint64_t sequence;
    uint64_t update_counter;
    uint64_t created_ms;
    DirectionConfig uplink;
    DirectionConfig downlink;
    DirectionTelemetry uplink_telemetry;
    DirectionTelemetry downlink_telemetry;
    char label[64];
    uint8_t reserved[256];
};
#pragma pack(pop)

class ControlBlockView
{
private:
    int fd_;
    std::string path_;
    ControlBlock * block_;

    DirectionConfig defaults_[ 2 ];

    void initialize_if_needed( const std::string & label );

public:
    ControlBlockView( const std::string & path,
                      const std::string & label,
                      const DirectionConfig & uplink_defaults,
                      const DirectionConfig & downlink_defaults );
    ~ControlBlockView();
    ControlBlockView( const ControlBlockView & other ) = delete;
    ControlBlockView & operator=( const ControlBlockView & other ) = delete;

    DirectionConfig read( const Direction direction ) const;
    void update_telemetry( const Direction direction, const DirectionTelemetry & telemetry ) const;
};

} /* namespace adaptive */

#endif /* ADAPTIVE_CONTROL_HH */
