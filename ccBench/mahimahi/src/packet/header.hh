/* -*-mode:c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

#ifndef HEADER_HH
#define HEADER_HH

using namespace std;

#include <arpa/inet.h>
//Parsing TCP/IP Header
typedef uint32_t addr_t;
typedef uint16_t port_t;

typedef struct {
  uint8_t  ver_ihl;  // 4 bits version and 4 bits internet header length
  uint8_t  tos;
  uint16_t total_length;
  uint16_t id;
  uint16_t flags_fo; // 3 bits flags and 13 bits fragment-offset
  uint8_t  ttl;
  uint8_t  protocol;
  uint16_t checksum;
  addr_t   src_addr;
  addr_t   dst_addr;
}__attribute__((packed)) ip_header_t ;

enum IPHR_INDEX{
	ver_ihl=0,
	tos=1,
	total_length=2,
	id=4,
	flags_fo=6,
	ttl=8,
	protocol=9,
	checksum=10,
	src_addr=12,
	dst_addr=16,
};
enum TCPHR_INDEX{
	src_port=20,
	dst_port=22,
	seq=24,
	ack=28,
};

typedef struct{
  uint16_t src_port;
  uint16_t dst_port;
  uint32_t seq;
  uint32_t ack;
  uint8_t  data_offset;  // 4 bits
  uint8_t  flags;
  uint16_t window_size;
  uint16_t checksum;
  uint16_t urgent_p;
}__attribute__((packed)) tcp_header_t ;

typedef struct {
//	uint8_t pad[4];
	  //IP
	  uint8_t  ver_ihl;  // 4 bits version and 4 bits internet header length
	  uint8_t  tos;
	  uint16_t total_length;
	  uint16_t id;
	  uint16_t flags_fo; // 3 bits flags and 13 bits fragment-offset
	  uint8_t  ttl;
	  uint8_t  protocol;
	  uint16_t checksum;
	  addr_t   src_addr;
	  addr_t   dst_addr;
	  //TCP (~UDP we use only src_port/dst_port which are at similar position for both TCP and UDP)
	  uint16_t src_port;
	  uint16_t dst_port;
	  uint32_t seq;
	  uint32_t ack;
	  uint8_t  data_offset;  // 4 bits
	  uint8_t  flags;
	  uint16_t window_size;
	  uint16_t tcp_checksum;
	  uint16_t urgent_p;
}__attribute__((packed)) ip_tcp_header_t ;

#endif /* HEADER_HH */
