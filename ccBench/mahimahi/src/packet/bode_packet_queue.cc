#include "bode_packet_queue.hh"

#include <math.h>
#include "timestamp.hh"

#define DISABLE_BODE_CODE 999


BoDePacketQueue::BoDePacketQueue( const string & args )
  : DroppingPacketQueue(args),
    target_ ( get_arg( args, "target") ),
	min_thr_ ( get_arg( args, "min_thr") ),
	first_above_time_ ( 0 ),
	drop_next_( 0 ),
	count_ ( 0 ),
	lastcount_ ( 0 ),
	dropping_ ( 0 )
{
  if ( target_ == 0 /*|| min_thr_ == 0*/ ) {
    throw runtime_error( "BoDe queue must have target and min_thr (packets) arguments." );
  }
}

void BoDePacketQueue::Init(const std::string & args)
{
	DroppingPacketQueue::Init(args);
	target_=get_arg( args, "target" );
	min_thr_=get_arg( args, "min_thr" );
	cerr<<"target_:"<<target_<<"min_thr_:"<<min_thr_<<"\n";
}


//NOTE: BoDe makes drop decisions at dequeue
//However, this function cannot return NULL. Therefore we ignore
//the drop decision if the current packet is the only one in the queue.
//We know that if this function is called, there is at least one packet in the queue.
dodequeue_result BoDePacketQueue::dodequeue ( uint64_t now )
{
  uint64_t sojourn_time;

  dodequeue_result r;
  r.p = std::move( DroppingPacketQueue::dequeue () );
  r.ok_to_drop = false;

  if ( empty() ) {
    first_above_time_ = 0;
    return r;
  }

  sojourn_time = now - r.p.arrival_time;
  if ( sojourn_time <= target_ || (min_thr_!=DISABLE_BODE_CODE && size_packets()<=min_thr_) || size_bytes() <= PACKET_SIZE ) {
	  r.ok_to_drop = false;
  }
  else {
      r.ok_to_drop = true;
  }
  return r;
}

QueuedPacket BoDePacketQueue::dequeue( void )
{   
  const uint64_t now = timestamp();
  dodequeue_result r = dodequeue( now );

  while ( r.ok_to_drop ) {
      r = std::move(dodequeue(now));
      if ( ! r.ok_to_drop ) {
    	  dropping_ = false;
      }
  }
  return r.p;
}

void BoDePacketQueue::enqueue( QueuedPacket && p )
{

  if ( good_with( size_bytes() + p.contents.size(),
		  size_packets() + 1 ) ) {
    accept( std::move( p ) );
  }

  assert( good() );
}
