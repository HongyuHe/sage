/* -*-mode:c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

#ifndef BODE_PACKET_QUEUE_HH
#define BODE_PACKET_QUEUE_HH

#include <random>
#include <string>
#include "dropping_packet_queue.hh"
#include "codel_packet_queue.hh"

using namespace std;

class BoDePacketQueue : public DroppingPacketQueue
{
private:
    const static unsigned int PACKET_SIZE = 1504;
    //Configuration parameters
    uint32_t target_, min_thr_;

    //State variables
    uint64_t first_above_time_, drop_next_;
    uint32_t count_, lastcount_;
    bool dropping_;

    virtual const std::string & type( void ) const override
    {
        static const std::string type_ { "bode" };
        return type_;
    }

    dodequeue_result dodequeue ( uint64_t now );

public:

    BoDePacketQueue( const std::string & args );
    void Init(const std::string & args) override;

    void enqueue( QueuedPacket && p ) override;
    QueuedPacket dequeue( void ) override;

};

#endif /* BODE_PACKET_QUEUE_HH */
