/* -*-mode:c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

#ifndef MULTI_PACKET_QUEUE_HH
#define MULTI_PACKET_QUEUE_HH

#include <random>
#include <memory>
#include <iostream>
#include <cstdint>
#include <string>
#include <fstream>
#include "dropping_packet_queue.hh"
#include "codel_packet_queue.hh"
#include "drop_tail_packet_queue.hh"
#include "bode_packet_queue.hh"
#include "header.hh"

using namespace std;
enum queue_type {
	hq_fifo=0,
	hq_bode=1,
};
struct dequeue_result {
    QueuedPacket p;
    dequeue_result ( ): p ( "", 0 )
    {}
};


class MultiPacketQueue : public DroppingPacketQueue
{
private:
    const static unsigned int PACKET_SIZE = 1504;

    virtual const std::string & type( void ) const override
    {
        static const std::string type_ { "multi" };
        return type_;
    }
public:
    unique_ptr<std::ofstream> log__;
    //Type of higher priority queues!
    int h_queue_type;
    DropTailPacketQueue fifo1;
    DropTailPacketQueue fifo2;
    DropTailPacketQueue fifo3;
    BoDePacketQueue bode2;
    BoDePacketQueue bode3;

    MultiPacketQueue( const std::string & args );

    void enqueue( QueuedPacket && p ) override;
    QueuedPacket dequeue( void ) override;

    bool empty( void ) const override;

    void ParseHD(ip_tcp_header_t *ip_tcp_header);
    port_t GetSrcPort(ip_tcp_header_t *ip_tcp_header);

    ip_header_t* ParseIP(const std::string contents);
    tcp_header_t* ParseTCP(const std::string contents);

};

#endif /* MULTI_PACKET_QUEUE_HH */
