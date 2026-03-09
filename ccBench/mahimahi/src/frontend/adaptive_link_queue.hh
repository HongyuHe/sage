/* -*-mode:c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

#ifndef ADAPTIVE_LINK_QUEUE_HH
#define ADAPTIVE_LINK_QUEUE_HH

#include <cstdint>
#include <deque>
#include <fstream>
#include <memory>
#include <optional>
#include <queue>
#include <random>
#include <string>

#include "adaptive_control.hh"
#include "file_descriptor.hh"
#include "queued_packet.hh"

class AdaptiveLinkQueue
{
private:
    struct DelayedPacket
    {
        uint64_t release_time_ms;
        std::string contents;
    };

    const static unsigned int PACKET_SIZE = 1504;

    adaptive::Direction direction_;
    std::string link_name_;
    adaptive::DirectionConfig defaults_;
    adaptive::DirectionConfig active_config_;
    adaptive::ControlBlockView control_;

    std::deque<QueuedPacket> packet_queue_;
    std::optional<QueuedPacket> packet_in_service_;
    unsigned int packet_in_service_bytes_left_;
    uint64_t queue_bytes_;

    std::deque<DelayedPacket> delayed_output_;

    double last_service_time_ms_;
    double service_credit_bytes_;

    std::default_random_engine prng_;
    std::unique_ptr<std::ofstream> log_;

    uint64_t enqueued_packets_;
    uint64_t dequeued_packets_;
    uint64_t dropped_packets_;
    uint64_t dropped_bytes_;
    uint64_t departed_bytes_;
    uint64_t first_departure_ms_;
    uint64_t last_apply_ms_;

    void record_arrival( const uint64_t now, const size_t pkt_size );
    void record_drop( const uint64_t now, const size_t packets, const size_t bytes );
    void record_departure( const uint64_t now, const QueuedPacket & packet );

    void refresh_config( const uint64_t now );
    void trim_queue_to_limits( const uint64_t now );
    bool queue_accepts( const size_t pkt_size ) const;
    uint64_t queue_occupancy_packets( void ) const;
    uint64_t queue_occupancy_bytes( void ) const;
    void rationalize( const uint64_t now );
    void update_telemetry( const uint64_t now );

public:
    AdaptiveLinkQueue( const std::string & link_name,
                       const std::string & control_file,
                       adaptive::Direction direction,
                       const adaptive::DirectionConfig & defaults,
                       const std::string & logfile,
                       const std::string & init_timestamp_file,
                       const std::string & command_line );

    void read_packet( const std::string & contents );
    void write_packets( FileDescriptor & fd );
    unsigned int wait_time( void );
    bool pending_output( void ) const;

    static bool finished( void ) { return false; }
};

#endif /* ADAPTIVE_LINK_QUEUE_HH */
