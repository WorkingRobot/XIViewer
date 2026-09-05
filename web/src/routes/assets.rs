use std::{
    env::{current_dir, current_exe},
    path::PathBuf,
    sync::LazyLock,
};

use actix_files::{Files, NamedFile};
use actix_web::{
    HttpResponse,
    dev::{HttpServiceFactory, ServiceRequest, ServiceResponse, fn_service},
};

static SERVICE_DIRECTORY: LazyLock<PathBuf> = LazyLock::new(|| {
    current_exe()
        .map(|p| p.parent().map(|p| p.to_path_buf()).unwrap_or(p))
        .unwrap_or_else(|_| current_dir().unwrap())
        .join("static")
});

/// Every first segment the client routes itself, which is what tells a deep link with a dot in it
/// from a missing file. A zone is named by its own `.lvb` and an asset by its own extension, so a
/// tab left out of this list answers its own reloads with a 404.
///
/// These are the routes `App::build` registers, less the sheet listing, which the client takes as
/// its default. **A new tab has to be added here too.**
const CLIENT_ROUTES: &[&str] = &[
    "assets",
    "auth",
    "character",
    "icons",
    "music",
    "quests",
    "sheet",
    "zones",
];

fn routed_by_client(path: &str) -> bool {
    let segment = path.trim_start_matches('/').split('/').next();
    segment.is_some_and(|segment| CLIENT_ROUTES.contains(&segment))
}

pub fn service() -> impl HttpServiceFactory {
    Files::new("/", SERVICE_DIRECTORY.clone())
        .index_file("index.html")
        .default_handler(fn_service(|req: ServiceRequest| async {
            let path = req.match_info().unprocessed();
            if path.contains('.') && !routed_by_client(path) {
                return Ok(req.into_response(HttpResponse::NotFound().finish()));
            }
            let (req, _) = req.into_parts();
            let file = NamedFile::open_async(SERVICE_DIRECTORY.join("index.html")).await?;
            let res = file.into_response(&req);
            Ok(ServiceResponse::new(req, res))
        }))
}

#[cfg(test)]
mod tests {
    use super::routed_by_client;

    #[test]
    fn client_deep_links_are_pages_not_missing_files() {
        for path in [
            "assets/exd/root.exl",
            "/assets/music/ffxiv/BGM_Null.scd",
            "sheet/Item.foo",
            "music/1",
            "auth/github/callback",
            // A zone is named by the `.lvb` it stands in, so every reload of one carries a dot.
            "zones/bg/ffxiv/wil_w1/dun/w1d6/level/w1d6.lvb",
            "/zones/bg/ex1/01_roc_r2/dun/r2d1/level/r2d1.lvb",
            "quests/1234",
            "character",
            "icons/60042",
        ] {
            assert!(routed_by_client(path), "{path}");
        }
    }

    #[test]
    fn everything_else_still_reports_a_missing_file() {
        for path in ["nope.js", "/viewer_bg.wasm", "", "assetsfoo/x.js"] {
            assert!(!routed_by_client(path), "{path}");
        }
    }
}
