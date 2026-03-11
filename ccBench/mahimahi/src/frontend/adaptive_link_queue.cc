/* -*-mode:c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <time.h>

#include "adaptive_link_queue.hh"
#include "timestamp.hh"

using namespace std;

namespace {

constexpr double BITS_PER_BYTE = 8.0;
constexpr double BYTES_PER_MEGABIT = 1000000.0 / BITS_PER_BYTE;

double bytes_per_ms_from_mbps( const double bandwidth_mbps )
{
    return bandwidth_mbps * BYTES_PER_MEGABIT / 1000.0;
}

double clamp_nonnegative( const double value )
{
    if ( not isfinite( value ) or value < 0.0 ) {
        return 0.0;
    }
    return value;
}

uint64_t clamp_wait_time( const double value_ms )
{
    if ( not isfinite( value_ms ) or value_ms <= 0.0 ) {
        return 0;
    }

    const double bounded = min( value_ms,
                                double( numeric_limits<uint16_t>::max() ) );
    return uint64_t( ceil( bounded ) );
}

uint64_t monotonic_abs_timestamp_ms( void )
{
    timespec ts {};
    clock_gettime( CLOCK_MONOTONIC, &ts );
    uint64_t millis = uint64_t( ts.tv_nsec ) / 1000000;
    millis += uint64_t( ts.tv_sec ) * 1000;
    return millis;
}

} /* namespace */

AdaptiveLinkQueue::AdaptiveLinkQueue( const string & link_name,
                                      const string & control_file,
                                      adaptive::Direction direction,
                                      const adaptive::DirectionConfig & defaults,
                                      const string & logfile,
                                      const string & init_timestamp_file,
                                      const string & command_line )
    : direction_( direction ),
      link_name_( link_name ),
      defaults_( defaults ),
      active_config_( defaults ),
      control_( control_file, link_name, defaults, defaults ),
      packet_queue_(),
      packet_in_service_( nullopt ),
      packet_in_service_bytes_left_( 0 ),
      queue_bytes_( 0 ),
      delayed_output_(),
      last_service_time_ms_( timestamp() ),
      service_credit_bytes_( 0.0 ),
      prng_( random_device()() ),
      log_(),
      enqueued_packets_( 0 ),
      dequeued_packets_( 0 ),
      dropped_packets_( 0 ),
      dropped_bytes_( 0 ),
      departed_bytes_( 0 ),
      first_departure_ms_( 0 ),
      last_apply_ms_( timestamp() )
{
    const string effective_init_timestamp_file =
        not init_timestamp_file.empty() ? init_timestamp_file
                                        : (not logfile.empty() ? logfile + "_init_timestamp" : string());

    if ( not logfile.empty() ) {
        log_.reset( new ofstream( logfile ) );
        if ( not log_->good() ) {
            throw runtime_error( logfile + ": error opening for writing" );
        }

        *log_ << "# mahimahi mm-adv-net (" << link_name_ << ") > " << logfile << endl;
        *log_ << "# command line: " << command_line << endl;
        *log_ << "# init timestamp: " << initial_timestamp() << endl;
        *log_ << "# base timestamp: " << timestamp() << endl;
    }

    if ( not effective_init_timestamp_file.empty() ) {
        std::unique_ptr<std::ofstream> init_log;
        init_log.reset( new ofstream( effective_init_timestamp_file ) );
        if ( not init_log->good() ) {
            throw runtime_error( effective_init_timestamp_file + ": error opening for writing" );
        }
        *init_log << initial_timestamp() + timestamp() << endl;
    }
}

void AdaptiveLinkQueue::record_arrival( const uint64_t now, const size_t pkt_size )
{
    if ( log_ ) {
        *log_ << now << " + " << pkt_size << endl;
    }
}

void AdaptiveLinkQueue::record_drop( const uint64_t now,
                                     const size_t packets,
                                     const size_t bytes )
{
    if ( log_ ) {
        *log_ << now << " d " << packets << " " << bytes << endl;
    }
}

void AdaptiveLinkQueue::record_departure( const uint64_t now,
                                          const QueuedPacket & packet )
{
    if ( log_ ) {
        *log_ << now << " - " << packet.contents.size()
              << " " << (now - packet.arrival_time)
              << " " << packet.queue_num << endl;
    }
}

void AdaptiveLinkQueue::refresh_config( const uint64_t now )
{
    adaptive::DirectionConfig next = control_.read( direction_ );
    const uint64_t now_abs_ms = monotonic_abs_timestamp_ms();

    if ( not isfinite( next.bandwidth_mbps ) or next.bandwidth_mbps < 0.0 ) {
        next.bandwidth_mbps = defaults_.bandwidth_mbps;
    }
    if ( not isfinite( next.loss_rate ) ) {
        next.loss_rate = defaults_.loss_rate;
    }
    next.loss_rate = clamp( next.loss_rate, 0.0, 1.0 );
    if ( not isfinite( next.delay_ms ) or next.delay_ms < 0.0 ) {
        next.delay_ms = defaults_.delay_ms;
    }
    if ( next.queue_packets == 0 ) {
        next.queue_packets = defaults_.queue_packets;
    }
    if ( next.queue_bytes == 0 ) {
        next.queue_bytes = defaults_.queue_bytes;
    }
    if ( not isfinite( next.effective_after_abs_ms ) or next.effective_after_abs_ms < 0.0 ) {
        next.effective_after_abs_ms = active_config_.effective_after_abs_ms;
    }
    if ( next.episode_step < active_config_.episode_step ) {
        return;
    }
    if ( next.episode_step > active_config_.episode_step
         and uint64_t( ceil( next.effective_after_abs_ms ) ) > now_abs_ms ) {
        return;
    }

    const bool changed = fabs( next.bandwidth_mbps - active_config_.bandwidth_mbps ) > 1e-9
                         or fabs( next.loss_rate - active_config_.loss_rate ) > 1e-9
                         or fabs( next.delay_ms - active_config_.delay_ms ) > 1e-9
                         or next.queue_packets != active_config_.queue_packets
                         or next.queue_bytes != active_config_.queue_bytes
                         or next.episode_step != active_config_.episode_step;
    active_config_ = next;
    if ( changed ) {
        last_apply_ms_ = now;
        trim_queue_to_limits( now );
    }
}

bool AdaptiveLinkQueue::queue_accepts( const size_t pkt_size ) const
{
    if ( active_config_.queue_packets > 0
         and queue_occupancy_packets() + 1 > active_config_.queue_packets ) {
        return false;
    }

    if ( active_config_.queue_bytes > 0
         and queue_occupancy_bytes() + pkt_size > active_config_.queue_bytes ) {
        return false;
    }

    return true;
}

uint64_t AdaptiveLinkQueue::queue_occupancy_packets( void ) const
{
    return packet_queue_.size() + (packet_in_service_.has_value() ? 1U : 0U);
}

uint64_t AdaptiveLinkQueue::queue_occupancy_bytes( void ) const
{
    return queue_bytes_ + packet_in_service_bytes_left_;
}

void AdaptiveLinkQueue::trim_queue_to_limits( const uint64_t now )
{
    while ( not packet_queue_.empty() and not queue_accepts( 0 ) ) {
        const QueuedPacket & packet = packet_queue_.back();
        dropped_packets_++;
        dropped_bytes_ += packet.contents.size();
        queue_bytes_ -= packet.contents.size();
        record_drop( now, 1, packet.contents.size() );
        packet_queue_.pop_back();
    }
}

void AdaptiveLinkQueue::rationalize( const uint64_t now )
{
    refresh_config( now );

    if ( now < last_service_time_ms_ ) {
        last_service_time_ms_ = now;
    }

    const double service_rate_bytes_per_ms = bytes_per_ms_from_mbps( active_config_.bandwidth_mbps );
    if ( service_rate_bytes_per_ms <= 0.0 ) {
        update_telemetry( now );
        return;
    }

    if ( not packet_in_service_.has_value() and packet_queue_.empty() ) {
        service_credit_bytes_ = 0.0;
        last_service_time_ms_ = now;
        update_telemetry( now );
        return;
    }

    double service_cursor_ms = last_service_time_ms_;
    double available_bytes = service_credit_bytes_
                             + max( 0.0, double( now ) - last_service_time_ms_ ) * service_rate_bytes_per_ms;

    while ( available_bytes > 1e-9 ) {
        if ( not packet_in_service_.has_value() ) {
            if ( packet_queue_.empty() ) {
                break;
            }

            packet_in_service_ = move( packet_queue_.front() );
            packet_queue_.pop_front();
            packet_in_service_bytes_left_ = packet_in_service_->contents.size();
            queue_bytes_ -= packet_in_service_->contents.size();
        }

        const double bytes_needed = double( packet_in_service_bytes_left_ );
        if ( available_bytes + 1e-9 < bytes_needed ) {
            packet_in_service_bytes_left_ = unsigned( ceil( bytes_needed - available_bytes ) );
            available_bytes = 0.0;
            break;
        }

        const double time_needed_ms = bytes_needed / service_rate_bytes_per_ms;
        service_cursor_ms += time_needed_ms;
        available_bytes -= bytes_needed;

        if ( first_departure_ms_ == 0 ) {
            first_departure_ms_ = uint64_t( ceil( service_cursor_ms ) );
        }
        departed_bytes_ += packet_in_service_->contents.size();
        dequeued_packets_++;
        record_departure( uint64_t( ceil( service_cursor_ms ) ), *packet_in_service_ );

        delayed_output_.push_back(
            DelayedPacket {
                uint64_t( ceil( service_cursor_ms + active_config_.delay_ms ) ),
                move( packet_in_service_->contents ),
            }
        );
        packet_in_service_.reset();
        packet_in_service_bytes_left_ = 0;
    }

    if ( not packet_in_service_.has_value() and packet_queue_.empty() ) {
        service_credit_bytes_ = 0.0;
    } else {
        service_credit_bytes_ = available_bytes;
    }
    last_service_time_ms_ = now;
    update_telemetry( now );
}

void AdaptiveLinkQueue::update_telemetry( const uint64_t now )
{
    adaptive::DirectionTelemetry telemetry {};
    telemetry.applied_bandwidth_mbps = clamp_nonnegative( active_config_.bandwidth_mbps );
    telemetry.applied_loss_rate = clamp( active_config_.loss_rate, 0.0, 1.0 );
    telemetry.applied_delay_ms = clamp_nonnegative( active_config_.delay_ms );
    telemetry.applied_queue_packets = active_config_.queue_packets;
    telemetry.applied_queue_bytes = active_config_.queue_bytes;
    telemetry.enqueued_packets = enqueued_packets_;
    telemetry.dequeued_packets = dequeued_packets_;
    telemetry.dropped_packets = dropped_packets_;
    telemetry.dropped_bytes = dropped_bytes_;
    telemetry.queue_occupancy_packets = queue_occupancy_packets();
    telemetry.queue_occupancy_bytes = queue_occupancy_bytes();
    telemetry.last_apply_ms = last_apply_ms_;

    if ( first_departure_ms_ > 0 and now > first_departure_ms_ ) {
        telemetry.departure_rate_mbps = double( departed_bytes_ ) * 8.0 / double( now - first_departure_ms_ ) / 1000.0;
    }

    const double bytes_per_ms = bytes_per_ms_from_mbps( active_config_.bandwidth_mbps );
    if ( bytes_per_ms > 0.0 ) {
        telemetry.queue_delay_ms = double( queue_occupancy_bytes() ) / bytes_per_ms;
    }
    telemetry.applied_step = double( active_config_.episode_step );
    telemetry.applied_effective_after_abs_ms = active_config_.effective_after_abs_ms;

    control_.update_telemetry( direction_, telemetry );
}

void AdaptiveLinkQueue::read_packet( const string & contents )
{
    const uint64_t now = timestamp();
    rationalize( now );
    record_arrival( now, contents.size() );

    const double loss_rate = clamp( active_config_.loss_rate, 0.0, 1.0 );
    if ( loss_rate > 0.0 ) {
        bernoulli_distribution dist( loss_rate );
        if ( dist( prng_ ) ) {
            dropped_packets_++;
            dropped_bytes_ += contents.size();
            record_drop( now, 1, contents.size() );
            update_telemetry( now );
            return;
        }
    }

    if ( not queue_accepts( contents.size() ) ) {
        dropped_packets_++;
        dropped_bytes_ += contents.size();
        record_drop( now, 1, contents.size() );
        update_telemetry( now );
        return;
    }

    packet_queue_.emplace_back( contents, now );
    queue_bytes_ += contents.size();
    enqueued_packets_++;
    update_telemetry( now );
}

void AdaptiveLinkQueue::write_packets( FileDescriptor & fd )
{
    const uint64_t now = timestamp();
    rationalize( now );

    while ( not delayed_output_.empty() and delayed_output_.front().release_time_ms <= now ) {
        fd.write( delayed_output_.front().contents );
        delayed_output_.pop_front();
    }
}

unsigned int AdaptiveLinkQueue::wait_time( void )
{
    const uint64_t now = timestamp();
    rationalize( now );

    if ( not delayed_output_.empty() and delayed_output_.front().release_time_ms <= now ) {
        return 0;
    }

    uint64_t next_wait = numeric_limits<uint16_t>::max();

    if ( not delayed_output_.empty() ) {
        next_wait = min( next_wait,
                         uint64_t( delayed_output_.front().release_time_ms - now ) );
    }

    const double bytes_per_ms = bytes_per_ms_from_mbps( active_config_.bandwidth_mbps );
    if ( bytes_per_ms > 0.0 and (packet_in_service_.has_value() or not packet_queue_.empty()) ) {
        const size_t bytes_needed = packet_in_service_.has_value()
                                        ? packet_in_service_bytes_left_
                                        : packet_queue_.front().contents.size();
        const double wait_ms = (double( bytes_needed ) - service_credit_bytes_) / bytes_per_ms;
        next_wait = min( next_wait, clamp_wait_time( wait_ms ) );
    }

    return next_wait;
}

bool AdaptiveLinkQueue::pending_output( void ) const
{
    return not delayed_output_.empty() and delayed_output_.front().release_time_ms <= timestamp();
}
