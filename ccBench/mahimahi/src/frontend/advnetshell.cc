/* -*-mode:c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "adaptive_control.hh"
#include "adaptive_link_queue.hh"
#include "ezio.hh"
#include "packetshell.cc"
#include "util.hh"

using namespace std;

namespace {

struct ParsedArgs
{
    string control_file;
    string uplink_logfile;
    string downlink_logfile;
    string uplink_init_timestamp_file;
    string downlink_init_timestamp_file;
    adaptive::DirectionConfig uplink;
    adaptive::DirectionConfig downlink;

    ParsedArgs()
        : control_file(),
          uplink_logfile(),
          downlink_logfile(),
          uplink_init_timestamp_file(),
          downlink_init_timestamp_file(),
          uplink(),
          downlink()
    {}
};

double parse_double( const char * const raw, const string & label )
{
    if ( raw == nullptr ) {
        throw runtime_error( "missing value for " + label );
    }
    return myatof( raw );
}

unsigned int parse_uint( const char * const raw, const string & label )
{
    if ( raw == nullptr ) {
        throw runtime_error( "missing value for " + label );
    }
    return myatoi( raw );
}

void usage_error( const string & program_name )
{
    cerr << "Usage: " << program_name << " --control-file=PATH [OPTION]... [COMMAND]" << endl;
    cerr << "Options:" << endl;
    cerr << "  --uplink-bw=MBPS --downlink-bw=MBPS" << endl;
    cerr << "  --uplink-loss=RATE --downlink-loss=RATE" << endl;
    cerr << "  --uplink-delay=MS --downlink-delay=MS" << endl;
    cerr << "  --uplink-queue=PACKETS --downlink-queue=PACKETS" << endl;
    cerr << "  --uplink-queue-bytes=BYTES --downlink-queue-bytes=BYTES" << endl;
    cerr << "  --uplink-log=FILE --downlink-log=FILE" << endl;
    cerr << "  --uplink-init-timestamp=FILE --downlink-init-timestamp=FILE" << endl;
    throw runtime_error( "invalid arguments" );
}

} /* namespace */

int main( int argc, char *argv[] )
{
    try {
        const bool passthrough_until_signal = getenv( "MAHIMAHI_PASSTHROUGH_UNTIL_SIGNAL" );

        char ** const user_environment = environ;
        environ = nullptr;

        check_requirements( argc, argv );

        ParsedArgs args;
        args.uplink.bandwidth_mbps = 12.0;
        args.downlink.bandwidth_mbps = 12.0;
        args.uplink.loss_rate = 0.0;
        args.downlink.loss_rate = 0.0;
        args.uplink.delay_ms = 0.0;
        args.downlink.delay_ms = 0.0;
        args.uplink.queue_packets = 128;
        args.downlink.queue_packets = 128;
        args.uplink.queue_bytes = 0;
        args.downlink.queue_bytes = 0;

        const option command_line_options[] = {
            { "control-file", required_argument, nullptr, 'c' },
            { "uplink-bw", required_argument, nullptr, 'u' },
            { "downlink-bw", required_argument, nullptr, 'd' },
            { "uplink-loss", required_argument, nullptr, 'U' },
            { "downlink-loss", required_argument, nullptr, 'D' },
            { "uplink-delay", required_argument, nullptr, 'x' },
            { "downlink-delay", required_argument, nullptr, 'y' },
            { "uplink-queue", required_argument, nullptr, 'q' },
            { "downlink-queue", required_argument, nullptr, 'w' },
            { "uplink-queue-bytes", required_argument, nullptr, 'Q' },
            { "downlink-queue-bytes", required_argument, nullptr, 'W' },
            { "uplink-log", required_argument, nullptr, 'l' },
            { "downlink-log", required_argument, nullptr, 'm' },
            { "uplink-init-timestamp", required_argument, nullptr, 'L' },
            { "downlink-init-timestamp", required_argument, nullptr, 'M' },
            { 0, 0, nullptr, 0 },
        };

        while ( true ) {
            const int opt = getopt_long( argc, argv, "", command_line_options, nullptr );
            if ( opt == -1 ) {
                break;
            }

            switch ( opt ) {
            case 'c':
                args.control_file = optarg;
                break;
            case 'u':
                args.uplink.bandwidth_mbps = parse_double( optarg, "uplink-bw" );
                break;
            case 'd':
                args.downlink.bandwidth_mbps = parse_double( optarg, "downlink-bw" );
                break;
            case 'U':
                args.uplink.loss_rate = parse_double( optarg, "uplink-loss" );
                break;
            case 'D':
                args.downlink.loss_rate = parse_double( optarg, "downlink-loss" );
                break;
            case 'x':
                args.uplink.delay_ms = parse_double( optarg, "uplink-delay" );
                break;
            case 'y':
                args.downlink.delay_ms = parse_double( optarg, "downlink-delay" );
                break;
            case 'q':
                args.uplink.queue_packets = parse_uint( optarg, "uplink-queue" );
                break;
            case 'w':
                args.downlink.queue_packets = parse_uint( optarg, "downlink-queue" );
                break;
            case 'Q':
                args.uplink.queue_bytes = parse_uint( optarg, "uplink-queue-bytes" );
                break;
            case 'W':
                args.downlink.queue_bytes = parse_uint( optarg, "downlink-queue-bytes" );
                break;
            case 'l':
                args.uplink_logfile = optarg;
                break;
            case 'm':
                args.downlink_logfile = optarg;
                break;
            case 'L':
                args.uplink_init_timestamp_file = optarg;
                break;
            case 'M':
                args.downlink_init_timestamp_file = optarg;
                break;
            default:
                usage_error( argv[ 0 ] );
            }
        }

        if ( args.control_file.empty() ) {
            usage_error( argv[ 0 ] );
        }

        vector<string> command;
        if ( optind == argc ) {
            command.push_back( shell_path() );
        } else {
            for ( int i = optind; i < argc; i++ ) {
                command.push_back( argv[ i ] );
            }
        }

        string command_line { string( argv[ 0 ] ) };
        for ( int i = 1; i < argc; i++ ) {
            command_line += " ";
            command_line += argv[ i ];
        }

        PacketShell<AdaptiveLinkQueue> link_shell_app( "adv", user_environment, passthrough_until_signal );

        link_shell_app.start_uplink(
            "[adv-net up] ",
            command,
            "Uplink",
            args.control_file,
            adaptive::Direction::Uplink,
            args.uplink,
            args.uplink_logfile,
            args.uplink_init_timestamp_file,
            command_line
        );

        link_shell_app.start_downlink(
            "Downlink",
            args.control_file,
            adaptive::Direction::Downlink,
            args.downlink,
            args.downlink_logfile,
            args.downlink_init_timestamp_file,
            command_line
        );

        return link_shell_app.wait_for_exit();
    } catch ( const exception & e ) {
        print_exception( e );
        return EXIT_FAILURE;
    }
}
