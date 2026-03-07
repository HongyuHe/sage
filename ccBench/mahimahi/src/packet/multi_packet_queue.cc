#include "multi_packet_queue.hh"

#include <math.h>
#include "timestamp.hh"

#include <stdio.h>
#include <sstream>

#define DISABLE_BODE_CODE 999

MultiPacketQueue::MultiPacketQueue( const string & args )
  : DroppingPacketQueue(args),
	log__(),
	h_queue_type(get_arg(args,"type")),
	fifo1(DropTailPacketQueue(args)),
	fifo2(DropTailPacketQueue(args)),
	fifo3(DropTailPacketQueue(args)),
	bode2(BoDePacketQueue(args)),
	bode3(BoDePacketQueue(args))
{
	log__.reset(new ofstream("header_test"));

	std::ostringstream oss;
	oss << "packets=" << get_arg(args,"packets");
	const std::string fifo1_args = oss.str();
	fifo1.Init(fifo1_args);
	cerr<<"FIFO1\n";

	oss.clear();
	oss.str("");
	oss << "packets=" << get_arg(args,"packets2");
	const std::string fifo2_args = oss.str();
	fifo2.Init(fifo2_args);
	cerr<<"FIFO2\n";

	oss.clear();
	oss.str("");
	oss << "packets=" << get_arg(args,"packets3");
	const std::string fifo3_args = oss.str();
	fifo3.Init(fifo3_args);
	cerr<<"FIFO3\n";

	oss.clear();
	oss.str("");
	oss << "packets=" << get_arg(args,"packets2")<<",target="<<get_arg(args,"target")<<",min_thr="<<get_arg(args,"min_thr");
	const std::string bode2_args = oss.str();
	bode2.Init(bode2_args);
	cerr<<"BODE2\n";

	oss.clear();
	oss.str("");
	oss << "packets=" << get_arg(args,"packets3")<<",target="<<get_arg(args,"target3")<<",min_thr="<<get_arg(args,"min_thr3");
	const std::string bode3_args = oss.str();
	bode3.Init(bode3_args);
	cerr<<"BODE3\n";

//	bode2=new BoDePacketQueue(args);
//	fifo1=new DropTailPacketQueue(args);
}

bool MultiPacketQueue::empty( void ) const
{
    return (bode3.empty() && bode2.empty() && fifo1.empty() && fifo2.empty() && fifo3.empty());
}

QueuedPacket MultiPacketQueue::dequeue( void )
{   
	QueuedPacket r("",0);
	AbstractPacketQueue *hq3=&bode3;
	AbstractPacketQueue *hq2=&bode2;
	if (h_queue_type==hq_fifo)
	{
		hq3=&fifo3;
		hq2=&fifo2;
	}
	if(!hq3->empty())
	{
		r=std::move(hq3->dequeue());
	}
	else if(!hq2->empty())
	{
		r=std::move(hq2->dequeue());
	}
	else
	{
		r=std::move(fifo1.dequeue());
	}
	return r;
}

void MultiPacketQueue::enqueue( QueuedPacket && p )
{
  string contents=p.contents;
  ip_tcp_header_t *hd;
  bool done=false;
  //Highest priority port==5050
  AbstractPacketQueue *hq3=&bode3;
  AbstractPacketQueue *hq2=&bode2;
  if (h_queue_type==hq_fifo)
  {
	  hq3=&fifo3;
	  hq2=&fifo2;
  }
  for(size_t i=0;i<contents.size();++i)
  {
	  hd =  (ip_tcp_header_t*)(contents.c_str()+i);
	  if (hd->ver_ihl==0x45)
	  {
		  done=true;
		  break;
	  }
  }
  if(!done)
  {
	  hd =  (ip_tcp_header_t*)(contents.c_str());
  }

  	if(GetSrcPort(hd)==5100)
	{
		p.queue_num=3;
		hq3->enqueue(std::move(p));
	}
	else if(GetSrcPort(hd)==5050)
	{
		p.queue_num=2;
		hq2->enqueue(std::move(p));
	}
	else
	{
		fifo1.enqueue(std::move(p));
	}
}
port_t MultiPacketQueue::GetSrcPort(ip_tcp_header_t *ip_tcp_header)
{
	return ntohs(ip_tcp_header->src_port);
}

void MultiPacketQueue::ParseHD(ip_tcp_header_t *ip_tcp_header){
//    	return;
	ip_tcp_header->src_port		= ntohs(ip_tcp_header->src_port);
	ip_tcp_header->dst_port     = ntohs(ip_tcp_header->dst_port);

/*
	ip_tcp_header->total_length = ntohs(ip_tcp_header->total_length);
	ip_tcp_header->id           = ntohs(ip_tcp_header->id);
	ip_tcp_header->flags_fo     = ntohs(ip_tcp_header->flags_fo);
	ip_tcp_header->checksum     = ntohs(ip_tcp_header->checksum);
	ip_tcp_header->src_addr     = ntohl(ip_tcp_header->src_addr);
	ip_tcp_header->dst_addr     = ntohl(ip_tcp_header->dst_addr);

	ip_tcp_header->src_port		= ntohs(ip_tcp_header->src_port);
	ip_tcp_header->dst_port     = ntohs(ip_tcp_header->dst_port);
	ip_tcp_header->seq		    = ntohl(ip_tcp_header->seq);
	ip_tcp_header->ack     	 	= ntohl(ip_tcp_header->ack);
//    	ip_tcp_header->data_offset  = ntohl(ip_tcp_header->data_offset);
//    	ip_tcp_header->flags   	 	= ntohl(ip_tcp_header->flags);
	ip_tcp_header->window_size  = ntohs(ip_tcp_header->window_size);
	ip_tcp_header->tcp_checksum	 	= ntohs(ip_tcp_header->tcp_checksum);
	ip_tcp_header->urgent_p     = ntohs(ip_tcp_header->urgent_p);
*/

//    	return ip_tcp_header;
}

ip_header_t* MultiPacketQueue::ParseIP(const std::string contents)
{
	ip_header_t* ip_header =  (ip_header_t*)(contents.c_str());
	ip_header->total_length = ntohs(ip_header->total_length);
	ip_header->id           = ntohs(ip_header->id);
	ip_header->flags_fo     = ntohs(ip_header->flags_fo);
	ip_header->checksum     = ntohs(ip_header->checksum);
	ip_header->src_addr     = ntohl(ip_header->src_addr);
	ip_header->dst_addr     = ntohl(ip_header->dst_addr);
	return ip_header;
}

tcp_header_t* MultiPacketQueue::ParseTCP(const std::string contents)
{
//    	ip_header_t* ip_header=ParseIP(contents);

	unsigned int ip_size = 20;//4 * (ip_header->ver_ihl & 0x0F);
	tcp_header_t* tcp_header =  (tcp_header_t*)(contents.c_str()+ip_size);
	tcp_header->src_port	 = ntohs(tcp_header->src_port);
	tcp_header->dst_port     = ntohs(tcp_header->dst_port);
	tcp_header->seq		     = ntohs(tcp_header->seq);
	tcp_header->ack     	 = ntohs(tcp_header->ack);
	tcp_header->data_offset  = ntohl(tcp_header->data_offset);
	tcp_header->flags   	 = ntohl(tcp_header->flags);
	tcp_header->window_size  = ntohl(tcp_header->window_size);
	tcp_header->checksum	 = ntohl(tcp_header->checksum);
	tcp_header->urgent_p     = ntohl(tcp_header->urgent_p);
	return tcp_header;
}
