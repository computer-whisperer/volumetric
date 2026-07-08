use std::net::SocketAddr;
use std::time::Duration;

use clap::Parser;

/// Volumetric processing daemon: runs project builds and model meshing for
/// remote clients (the UI's "remote build" mode, the CLI's --remote flag).
#[derive(Parser)]
#[command(version)]
struct Args {
    /// Address to listen on. Use 0.0.0.0:<port> to accept LAN clients —
    /// there is no authentication yet, so only do that on a trusted network.
    #[arg(long, default_value_t = SocketAddr::from(([127, 0, 0, 1], volumetric_protocol::DEFAULT_PORT)))]
    bind: SocketAddr,

    /// Jobs executing concurrently; further submissions queue.
    #[arg(long, default_value_t = 4)]
    max_jobs: usize,

    /// Seconds to retain finished job results for pickup.
    #[arg(long, default_value_t = 600)]
    result_ttl_secs: u64,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let handle = volumetric_daemon::start(volumetric_daemon::DaemonConfig {
        bind: args.bind,
        max_concurrent_jobs: args.max_jobs,
        result_ttl: Duration::from_secs(args.result_ttl_secs),
        ..Default::default()
    })?;
    eprintln!(
        "volumetric_daemon {} listening on http://{} ({} concurrent jobs)",
        env!("CARGO_PKG_VERSION"),
        handle.addr(),
        args.max_jobs
    );
    handle.wait();
    Ok(())
}
