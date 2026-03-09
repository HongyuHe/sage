/* -*-mode:c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "adaptive_control.hh"
#include "exception.hh"
#include "timestamp.hh"

using namespace std;

namespace adaptive {

namespace {

DirectionConfig sanitize_defaults( const DirectionConfig & cfg )
{
    DirectionConfig out = cfg;
    if ( not isfinite( out.bandwidth_mbps ) or out.bandwidth_mbps < 0.0 ) {
        out.bandwidth_mbps = 0.0;
    }
    if ( not isfinite( out.loss_rate ) ) {
        out.loss_rate = 0.0;
    }
    out.loss_rate = clamp( out.loss_rate, 0.0, 1.0 );
    if ( not isfinite( out.delay_ms ) or out.delay_ms < 0.0 ) {
        out.delay_ms = 0.0;
    }
    return out;
}

DirectionTelemetry * telemetry_ptr( ControlBlock * block, const Direction direction )
{
    return direction == Direction::Uplink ? &block->uplink_telemetry : &block->downlink_telemetry;
}

DirectionConfig * config_ptr( ControlBlock * block, const Direction direction )
{
    return direction == Direction::Uplink ? &block->uplink : &block->downlink;
}

} /* namespace */

ControlBlockView::ControlBlockView( const string & path,
                                    const string & label,
                                    const DirectionConfig & uplink_defaults,
                                    const DirectionConfig & downlink_defaults )
    : fd_( -1 ),
      path_( path ),
      block_( nullptr ),
      defaults_ { sanitize_defaults( uplink_defaults ), sanitize_defaults( downlink_defaults ) }
{
    fd_ = SystemCall( "open adaptive control block",
                      open( path_.c_str(), O_RDWR | O_CREAT, 0600 ) );
    SystemCall( "ftruncate adaptive control block",
                ftruncate( fd_, sizeof( ControlBlock ) ) );

    void * const mapped = mmap( nullptr, sizeof( ControlBlock ),
                                PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0 );
    if ( mapped == MAP_FAILED ) {
        throw unix_error( "mmap adaptive control block" );
    }

    block_ = static_cast<ControlBlock *>( mapped );
    initialize_if_needed( label );
}

ControlBlockView::~ControlBlockView()
{
    if ( block_ != nullptr ) {
        munmap( static_cast<void *>( block_ ), sizeof( ControlBlock ) );
    }
    if ( fd_ >= 0 ) {
        close( fd_ );
    }
}

void ControlBlockView::initialize_if_needed( const string & label )
{
    if ( block_->magic == CONTROL_MAGIC
         and block_->version == CONTROL_VERSION
         and block_->byte_size == sizeof( ControlBlock ) ) {
        return;
    }

    memset( block_, 0, sizeof( ControlBlock ) );
    block_->magic = CONTROL_MAGIC;
    block_->version = CONTROL_VERSION;
    block_->byte_size = sizeof( ControlBlock );
    block_->sequence = 0;
    block_->update_counter = 0;
    block_->created_ms = initial_timestamp() + timestamp();
    block_->uplink = defaults_[ 0 ];
    block_->downlink = defaults_[ 1 ];
    strncpy( block_->label, label.c_str(), sizeof( block_->label ) - 1 );
}

DirectionConfig ControlBlockView::read( const Direction direction ) const
{
    if ( block_->magic != CONTROL_MAGIC
         or block_->version != CONTROL_VERSION
         or block_->byte_size != sizeof( ControlBlock ) ) {
        return defaults_[ direction == Direction::Uplink ? 0 : 1 ];
    }

    DirectionConfig result {};
    for ( unsigned int attempt = 0; attempt < 8; attempt++ ) {
        const uint64_t seq_before = block_->sequence;
        if ( seq_before & 1U ) {
            continue;
        }

        result = *config_ptr( block_, direction );
        const uint64_t seq_after = block_->sequence;
        if ( seq_before == seq_after and ((seq_after & 1U) == 0) ) {
            return result;
        }
    }

    return defaults_[ direction == Direction::Uplink ? 0 : 1 ];
}

void ControlBlockView::update_telemetry( const Direction direction,
                                         const DirectionTelemetry & telemetry ) const
{
    if ( block_->magic != CONTROL_MAGIC
         or block_->version != CONTROL_VERSION
         or block_->byte_size != sizeof( ControlBlock ) ) {
        return;
    }

    *telemetry_ptr( block_, direction ) = telemetry;
}

} /* namespace adaptive */
