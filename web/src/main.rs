mod blocking_stream;
mod config;
mod data;
mod paths;
mod queue;
mod routes;
mod slice;
mod smart_bufreader;

use ::config::{Config, Environment, File, FileFormat};
use actix_cors::Cors;
use actix_web::{
    App, HttpServer,
    middleware::{Condition, Logger, NormalizePath, TrailingSlash},
    web::Data,
};
use actix_web_helmet::{Helmet, XContentTypeOptions};
use actix_web_prom::PrometheusMetricsBuilder;
use data::GameData;
use prometheus::Registry;
use shadow_rs::shadow;
use std::{io, sync::Arc};
use thiserror::Error;

use crate::{
    paths::{PathIndex, report::Collector},
    queue::MessageQueue,
    routes::api::{SLICE, STREAM_KIND},
};

shadow!(build);

#[derive(Error, Debug)]
pub enum ServerError {
    #[error("Join error")]
    JoinError(#[from] tokio::task::JoinError),
    #[error("Actix error")]
    ActixError(#[from] io::Error),
    #[error("Dotenvy error")]
    DotenvyError(#[from] dotenvy::Error),
    #[error("Prometheus error")]
    PrometheusError(#[from] prometheus::Error),
    #[error("Other error")]
    OtherError(#[from] anyhow::Error),
}

#[tokio::main]
async fn main() -> Result<(), ServerError> {
    #[cfg(debug_assertions)]
    {
        _ = dotenvy::from_filename(".env");
        _ = dotenvy::from_filename(".secrets.env");
        unsafe { std::env::set_var("RUST_BACKTRACE", "1") };
        // console_subscriber::init();
    }

    let mut config: config::Config = Config::builder()
        .add_source(File::new("config", FileFormat::Yaml).required(false))
        .add_source(Environment::default())
        .build()
        .and_then(Config::try_deserialize)
        .unwrap();

    env_logger::init_from_env(
        env_logger::Env::new()
            .default_filter_or(config.log_filter.clone().unwrap_or("info".to_string())),
    );

    let prometheus_registry = Registry::new();

    config.cache = config
        .cache
        .prometheus_registry(prometheus_registry.clone());
    let config = config;

    let game_data = Arc::new(
        GameData::new(
            config.cache.clone(),
            config.assets.clone(),
            config.file_readahead,
        )
        .await?,
    );

    let server_prometheus = PrometheusMetricsBuilder::new("public")
        .registry(prometheus_registry.clone())
        .build()
        .map_err(|e| {
            *e.downcast::<prometheus::Error>()
                .expect("Unknown error from prometheus builder")
        })?;
    let server_config = config.clone();
    let path_index = Arc::new(PathIndex::new(config.path_list.clone()));
    let collector = Data::new(Collector::new(path_index.clone(), config.report.clone()));
    let server_game_data = MessageQueue::new(game_data.clone(), path_index, config.api_workers)?;

    routes::github::prewarm();

    log::info!("Binding to {}", config.server_addr);
    let server = HttpServer::new(move || {
        App::new()
            .wrap(
                Helmet::new()
                    .add(XContentTypeOptions::nosniff())
                    .into_middleware()
                    .expect("valid helmet config"),
            )
            .wrap(
                Cors::default()
                    // localhost + private/loopback LAN IPs (any port) so a LAN trunk serve reaches the API.
                    .allowed_origin_fn(|origin, _req_head| origin.to_str().is_ok_and(is_dev_origin))
                    .allowed_methods(vec!["GET", "POST"])
                    .allowed_headers(vec!["Content-Type"])
                    .expose_headers(vec![SLICE, STREAM_KIND]),
            )
            .wrap(NormalizePath::new(TrailingSlash::Always))
            .wrap(Condition::new(
                server_config.metrics_server_addr.is_some(),
                server_prometheus.clone(),
            ))
            .wrap(
                server_config
                    .log_access_format
                    .as_deref()
                    .map_or_else(Logger::default, Logger::new),
            )
            .app_data(Data::new(server_config.clone()))
            .app_data(Data::new(server_game_data.clone()))
            .app_data(collector.clone())
            .service(routes::api::service())
            .service(routes::assets::service())
    })
    .bind(config.server_addr.clone())?
    .run();

    log::info!("Http server running at http://{}", config.server_addr);

    let private_prometheus = PrometheusMetricsBuilder::new("private")
        .registry(prometheus_registry)
        .endpoint("/metrics")
        .build()
        .map_err(|e| {
            *e.downcast::<prometheus::Error>()
                .expect("Unknown error from prometheus builder")
        })?;
    let prometheus_server = if let Some(metrics_addr) = &config.metrics_server_addr {
        let ret = HttpServer::new(move || {
            App::new().wrap(private_prometheus.clone()).wrap(
                config
                    .log_access_format
                    .as_deref()
                    .map_or_else(Logger::default, Logger::new),
            )
        })
        .workers(1)
        .bind(metrics_addr)?
        .run();
        log::info!("Metrics http server running at http://{}", metrics_addr);
        Some(ret)
    } else {
        log::info!("Metrics server disabled");
        None
    };

    let server_task = tokio::task::spawn(server);
    let prometheus_server_task = prometheus_server.map(|s| tokio::task::spawn(s));

    let server_ret = server_task.await;

    let prometheus_server_ret = match prometheus_server_task {
        Some(task) => task.await,
        None => Ok(Ok(())),
    };

    server_ret??;
    prometheus_server_ret??;

    game_data.close().await?;

    log::info!("Goodbye!");

    Ok(())
}

/// A CORS origin whose host is localhost or a private/loopback IPv4 address.
fn is_dev_origin(origin: &str) -> bool {
    let Some(rest) = origin
        .strip_prefix("http://")
        .or_else(|| origin.strip_prefix("https://"))
    else {
        return false;
    };
    let host = rest.split(['/', ':']).next().unwrap_or_default();
    if host == "localhost" {
        return true;
    }
    matches!(
        host.parse::<std::net::Ipv4Addr>(),
        Ok(ip) if ip.is_private() || ip.is_loopback()
    )
}

#[cfg(test)]
mod tests {
    use super::is_dev_origin;

    #[test]
    fn allows_lan_and_localhost_only() {
        assert!(is_dev_origin("https://192.168.1.217:8080"));
        assert!(is_dev_origin("http://localhost:8080"));
        assert!(is_dev_origin("http://127.0.0.1:3000"));
        assert!(is_dev_origin("http://10.0.0.5:3000"));
        assert!(is_dev_origin("https://172.20.1.1"));
        assert!(!is_dev_origin("https://xiviewer.app")); // same-origin, no CORS needed
        assert!(!is_dev_origin("https://8.8.8.8"));
        assert!(!is_dev_origin("https://172.32.0.1")); // outside 172.16-31 private range
        assert!(!is_dev_origin("https://evil.com:8080"));
        assert!(!is_dev_origin("ftp://192.168.1.1"));
    }
}
