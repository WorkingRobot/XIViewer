use std::{
    collections::HashMap,
    io::Write,
    sync::{Arc, LazyLock, Mutex},
    time::{Duration, Instant},
};

use actix_web::post;
use actix_web::{
    HttpRequest, HttpResponse, Result,
    body::{EitherBody, MessageBody},
    dev::{HttpServiceFactory, ServiceResponse},
    error::{ErrorBadRequest, ErrorInternalServerError, ErrorNotFound, ErrorTooManyRequests},
    get,
    http::header::{
        self, ByteRangeSpec, ContentDisposition, ContentRange, ContentRangeSpec, ETag, EntityTag,
        Header, IfNoneMatch, Range,
    },
    middleware::{ErrorHandlerResponse, ErrorHandlers},
    web::{self, Bytes},
};
use actix_web_lab::header::{CacheControl, CacheDirective};
use flate2::{Compression, write::GzEncoder};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use xiv_core::file::{slug::Slug, version::GameVersion};

use crate::{
    config::Config,
    data::{Region, RepositoryInfo, Target},
    paths::report::{Collector, Submission},
    queue::MessageQueue,
    slice::Slice,
};

/// What format a file was stored as.
pub const STREAM_KIND: header::HeaderName = header::HeaderName::from_static("x-stream-kind");

/// Header naming which part of a file was served, for a request that asked for one. Absent from a
/// server predating it, which is how a client knows to take the file whole.
pub const SLICE: header::HeaderName = header::HeaderName::from_static("x-slice");

pub fn service() -> impl HttpServiceFactory {
    web::scope("/api")
        // Literal-prefixed routes first. They cannot collide with the region routes by segment
        // count, but keeping the rule "literals before variables" makes that obvious.
        .service(get_github_oauth_config)
        .service(post_github_oauth_token)
        .configure(super::github::configure)
        .service(get_repositories)
        .service(get_regions)
        .service(get_list_id)
        .service(post_report)
        .service(get_global_paths)
        .service(get_songs)
        .service(get_bnpc)
        .service(get_versions_repo)
        .service(get_latest_repo)
        .service(get_file_repo)
        .service(get_hash_repo)
        .service(get_paths_repo)
        .service(get_exists_repo)
        .service(get_versions_region)
        .service(get_latest_region)
        .service(get_file_region)
        .service(get_hash_region)
        .service(get_paths_region)
        .service(get_exists_region)
        .wrap(
            ErrorHandlers::new()
                .default_handler_client(|r| log_error(true, r))
                .default_handler_server(|r| log_error(false, r)),
        )
}

#[derive(Debug, Clone, Serialize)]
struct RegionsInfo {
    regions: Vec<Region>,
}

#[derive(Debug, Serialize)]
struct RepositoriesInfo {
    repositories: Vec<RepositoryInfo>,
}

#[derive(Debug, Serialize)]
struct ListInfo {
    list: String,
}

/// Every content response is for a pinned version — `latest` is a redirect, never a resource — so
/// there is one cache policy rather than a branch per handler.
fn pinned() -> Vec<CacheDirective> {
    vec![
        CacheDirective::Public,
        CacheDirective::Immutable,
        CacheDirective::MaxAge(60 * 60 * 24 * 365),
    ]
}

/// Reject a target that carries no sqpack data. Boot ships the launcher, so answering with an
/// empty result would be a confident lie.
async fn require_sqpack(data: &MessageQueue, target: Target) -> Result<()> {
    if data.has_sqpack(target).await {
        Ok(())
    } else {
        Err(ErrorNotFound(format!(
            "{target} has no sqpack data; only /versions/ is available for it"
        )))
    }
}

/// Reject a version no repository behind the target ever published, so a typo fails loudly rather
/// than silently backfilling to something older.
async fn check_version(data: &MessageQueue, target: Target, version: &GameVersion) -> Result<()> {
    if data.version_valid(target, version).await {
        Ok(())
    } else {
        Err(ErrorNotFound(format!("{target} has no version {version}")))
    }
}

/// 307 to the resolved version, so `latest` stays usable by hand without ever being a cacheable
/// resource itself. Temporary, because the target legitimately moves.
async fn redirect_latest(
    data: &MessageQueue,
    request: &HttpRequest,
    target: Target,
    prefix: &str,
    rest: &str,
) -> Result<HttpResponse> {
    let latest = data
        .versions_for(target)
        .await
        .ok_or_else(|| ErrorBadRequest("No version info available"))?
        .latest;
    // `rest` is captured without its trailing slash; every route it can land on has one.
    let query = request.query_string();
    let separator = if query.is_empty() { "" } else { "?" };
    let location = format!("/api/{prefix}/{latest}/{rest}/{separator}{query}");
    Ok(HttpResponse::TemporaryRedirect()
        .insert_header((actix_web::http::header::LOCATION, location))
        .insert_header(CacheControl(vec![
            CacheDirective::Public,
            CacheDirective::MaxAge(60),
        ]))
        .finish())
}

/// The one span of a file a `Range` header asks for. A set of them would have to be answered as a
/// multipart body, which nothing asks for.
fn partial(request: &HttpRequest, bytes: &Bytes) -> Option<(Bytes, ContentRange)> {
    let length = bytes.len() as u64;
    let Ok(Range::Bytes(asked)) = Range::parse(request) else {
        return None;
    };
    let [spec] = asked.as_slice() else {
        return None;
    };
    let (from, to) = ByteRangeSpec::to_satisfiable_range(spec, length)?;
    Some((
        bytes.slice(from as usize..=to as usize),
        ContentRange(ContentRangeSpec::Bytes {
            range: Some((from, to)),
            instance_length: Some(length),
        }),
    ))
}

async fn serve_file(
    data: &MessageQueue,
    request: &HttpRequest,
    target: Target,
    version: GameVersion,
    path: String,
) -> Result<HttpResponse> {
    if path.is_empty() {
        return Err(ErrorBadRequest("File path cannot be empty"));
    }
    require_sqpack(data, target).await?;
    check_version(data, target, &version).await?;

    let file_name = path
        .rsplit_once('/')
        .map_or(path.as_str(), |(_, name)| name);
    let directives = pinned();

    let data = data.get_file(target, Some(version), path.clone()).await;
    match data {
        Ok(data) => {
            let cut = web::Query::<Slice>::from_query(request.query_string())
                .ok()
                .and_then(|asked| asked.cut(&data.bytes));
            let bytes = cut.as_ref().map_or(&data.bytes, |(bytes, _)| bytes);
            let asked = partial(request, bytes);
            let mut response = match asked {
                Some(_) => HttpResponse::PartialContent(),
                None => HttpResponse::Ok(),
            };
            response
                .insert_header(ContentDisposition::attachment(file_name))
                .insert_header(CacheControl(directives))
                .insert_header((header::ACCEPT_RANGES, "bytes"))
                .insert_header((STREAM_KIND, data.kind.name()));
            if let Some((_, name)) = &cut {
                response.insert_header((SLICE, name.clone()));
            }
            Ok(match asked {
                Some((bytes, range)) => response.insert_header(range).body(bytes),
                None => response.body(bytes.clone()),
            })
        }
        Err(err) if matches!(err, ironworks::Error::NotFound(_)) => Err(ErrorNotFound(err)),
        Err(err) => Err(ErrorInternalServerError(err)),
    }
}

#[get("/paths/")]
async fn get_list_id(data: web::Data<MessageQueue>, request: HttpRequest) -> Result<HttpResponse> {
    let current = data.get_list_id().await.map_err(ErrorInternalServerError)?;
    let tag = EntityTag::new_strong(format!("{current:016x}"));

    let known = match IfNoneMatch::parse(&request) {
        Ok(IfNoneMatch::Any) => true,
        Ok(IfNoneMatch::Items(tags)) => tags.iter().any(|seen| seen.weak_eq(&tag)),
        Err(_) => false,
    };
    let mut response = if known {
        HttpResponse::NotModified()
    } else {
        HttpResponse::Ok()
    };
    response.insert_header(ETag(tag));
    if known {
        return Ok(response.finish());
    }
    Ok(response.json(ListInfo {
        list: format!("{current:016x}"),
    }))
}

/// Who submitted, as far as the edge will say. The peer address is Cloudflare's, so on its own it
/// would limit every client as one; a header can be forged, which makes this a nuisance limiter
/// rather than a control.
fn client_key(request: &HttpRequest) -> String {
    let header = |name| {
        request
            .headers()
            .get(name)
            .and_then(|value| value.to_str().ok())
    };
    header("cf-connecting-ip")
        .or_else(|| header("x-forwarded-for").and_then(|value| value.split(',').next()))
        .map(|ip| ip.trim().to_owned())
        .or_else(|| request.peer_addr().map(|addr| addr.ip().to_string()))
        .unwrap_or_default()
}

#[post("/report/")]
async fn post_report(
    collector: web::Data<Collector>,
    request: HttpRequest,
    body: web::Json<Submission>,
) -> Result<HttpResponse> {
    let body = body.into_inner();
    if body.paths.is_empty() {
        return Err(ErrorBadRequest("No paths submitted"));
    }
    if body.paths.len() > collector.batch_limit() {
        return Err(ErrorBadRequest(format!(
            "At most {} paths per submission",
            collector.batch_limit()
        )));
    }
    if !collector.allow(&client_key(&request)) {
        return Err(ErrorTooManyRequests("Too many submissions"));
    }

    let outcome = collector
        .submit(body)
        .await
        .map_err(ErrorInternalServerError)?;
    Ok(HttpResponse::Accepted().json(outcome))
}

#[get("/paths/{list_id}/")]
async fn get_global_paths(
    data: web::Data<MessageQueue>,
    request: HttpRequest,
    list_id: web::Path<String>,
) -> Result<HttpResponse> {
    let wanted = parse_list_id(&list_id)?;
    let (current, frame) = data
        .get_global_paths()
        .await
        .map_err(ErrorInternalServerError)?;
    if wanted != current {
        return Err(stale_list(wanted, current));
    }
    serve_frame(&request, frame, 60 * 60 * 24 * 365)
}

async fn serve_presence(
    data: &MessageQueue,
    request: &HttpRequest,
    target: Target,
    version: GameVersion,
    list_id: &str,
) -> Result<HttpResponse> {
    let wanted = parse_list_id(list_id)?;
    require_sqpack(data, target).await?;
    check_version(data, target, &version).await?;
    let frame = data
        .get_presence(target, Some(version), wanted)
        .await
        .map_err(ErrorInternalServerError)?;
    match frame {
        Some(frame) => serve_frame(request, frame, 60 * 60 * 24 * 365),
        None => {
            let current = data.get_list_id().await.map_err(ErrorInternalServerError)?;
            Err(stale_list(wanted, current))
        }
    }
}

fn parse_list_id(list_id: &str) -> Result<u64> {
    u64::from_str_radix(list_id, 16)
        .map_err(|_| ErrorBadRequest("Path list id must be 16 hex digits"))
}

fn stale_list(wanted: u64, current: u64) -> actix_web::Error {
    ErrorNotFound(format!(
        "Path list {wanted:016x} is no longer served; the current one is {current:016x}"
    ))
}

fn serve_frame(request: &HttpRequest, frame: Bytes, max_age: u32) -> Result<HttpResponse> {
    let mut directives = vec![CacheDirective::Public, CacheDirective::MaxAge(max_age)];
    if max_age > 60 * 60 {
        directives.insert(1, CacheDirective::Immutable);
    }

    let mut response = HttpResponse::Ok();
    response
        .content_type("application/octet-stream")
        .insert_header(CacheControl(directives))
        .insert_header((header::VARY, "Accept-Encoding"));

    if accepts(request, "zstd") {
        return Ok(response
            .insert_header((header::CONTENT_ENCODING, "zstd"))
            .body(frame));
    }
    let body = pathlist::decompress(&frame).map_err(ErrorInternalServerError)?;
    if accepts(request, "gzip") {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&body).map_err(ErrorInternalServerError)?;
        let gzipped = encoder.finish().map_err(ErrorInternalServerError)?;
        return Ok(response
            .insert_header((header::CONTENT_ENCODING, "gzip"))
            .body(gzipped));
    }
    Ok(response.body(body))
}

pub fn accepts(request: &HttpRequest, encoding: &str) -> bool {
    request
        .headers()
        .get(header::ACCEPT_ENCODING)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| {
            value.split(',').any(|part| {
                let mut fields = part.split(';');
                let name = fields.next().unwrap_or_default().trim();
                name.eq_ignore_ascii_case(encoding) && !fields.any(|f| f.trim() == "q=0")
            })
        })
}

/// Unnamed files have no path, only the hash the index records them under.
async fn serve_hash(
    data: &MessageQueue,
    target: Target,
    version: GameVersion,
    repository: u8,
    category: u8,
    hash: String,
) -> Result<HttpResponse> {
    require_sqpack(data, target).await?;
    check_version(data, target, &version).await?;

    // 16 hex digits is the `.index` form, split into directory and file halves; 8 is the
    // `.index2` whole-path form.
    let hash = match hash.len() {
        16 => u64::from_str_radix(&hash, 16)
            .map(ironworks::sqpack::IndexHash::Split)
            .map_err(|_| ErrorBadRequest("Malformed index hash")),
        8 => u32::from_str_radix(&hash, 16)
            .map(ironworks::sqpack::IndexHash::Whole)
            .map_err(|_| ErrorBadRequest("Malformed index2 hash")),
        _ => Err(ErrorBadRequest(
            "Hash must be 16 hex digits for .index or 8 for .index2",
        )),
    }?;

    match data
        .get_file_by_hash(target, Some(version), repository, category, hash)
        .await
    {
        Ok(data) => Ok(HttpResponse::Ok()
            .insert_header(CacheControl(pinned()))
            .insert_header((STREAM_KIND, data.kind.name()))
            .body(data.bytes.clone())),
        // A whole-path hash can name more than one file, which the caller has to resolve by asking
        // for a path instead.
        Err(err) if matches!(err, ironworks::Error::NotFound(_)) => Err(ErrorNotFound(err)),
        Err(err) if matches!(err, ironworks::Error::Invalid(..)) => Err(ErrorBadRequest(err)),
        Err(err) => Err(ErrorInternalServerError(err)),
    }
}

#[derive(Debug, Deserialize)]
struct ExistsQuery {
    /// Comma-separated list of file paths
    files: String,
}

#[derive(Debug, Serialize)]
struct ExistsResponse {
    exists: Vec<bool>,
}

async fn serve_exists(
    data: &MessageQueue,
    target: Target,
    version: GameVersion,
    files_param: &str,
) -> Result<HttpResponse> {
    let files: Vec<String> = files_param
        .split(',')
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .collect();
    if files.is_empty() {
        return Err(ErrorBadRequest("No files specified"));
    }

    require_sqpack(data, target).await?;
    check_version(data, target, &version).await?;
    let directives = pinned();

    match data.exists(target, Some(version), files).await {
        Ok(exists) => Ok(HttpResponse::Ok()
            .insert_header(CacheControl(directives))
            .json(ExistsResponse { exists })),
        Err(err) if matches!(err, ironworks::Error::NotFound(_)) => Err(ErrorNotFound(err)),
        Err(err) => Err(ErrorInternalServerError(err)),
    }
}

#[get("/regions/")]
async fn get_regions(data: web::Data<MessageQueue>) -> Result<HttpResponse> {
    let regions = data.regions().await.map_err(ErrorInternalServerError)?;
    Ok(HttpResponse::Ok().json(RegionsInfo { regions }))
}

#[get("/{region}/versions/")]
async fn get_versions_region(
    data: web::Data<MessageQueue>,
    path_info: web::Path<Region>,
) -> Result<HttpResponse> {
    serve_versions(&data, Target::Region(path_info.into_inner())).await
}

#[get("/repo/{slug}/versions/")]
async fn get_versions_repo(
    data: web::Data<MessageQueue>,
    path_info: web::Path<Slug>,
) -> Result<HttpResponse> {
    serve_versions(&data, Target::Repo(path_info.into_inner())).await
}

async fn serve_versions(data: &MessageQueue, target: Target) -> Result<HttpResponse> {
    match data.versions_for(target).await {
        Some(info) => Ok(HttpResponse::Ok().json(info)),
        None => Err(ErrorBadRequest("No version info available")),
    }
}

// Region-keyed routes: the game as a whole. Every sibling under a version is a literal segment,
// so no endpoint can be shadowed by a file path and registration order is not load-bearing.

#[get("/{region}/latest/{rest:.*}/")]
async fn get_latest_region(
    data: web::Data<MessageQueue>,
    request: HttpRequest,
    path_info: web::Path<(Region, String)>,
) -> Result<HttpResponse> {
    let (region, rest) = path_info.into_inner();
    redirect_latest(
        &data,
        &request,
        Target::Region(region),
        &region.to_string(),
        &rest,
    )
    .await
}

#[get("/{region}/{version}/file/{path:.*}/")]
async fn get_file_region(
    data: web::Data<MessageQueue>,
    request: HttpRequest,
    path_info: web::Path<(Region, GameVersion, String)>,
) -> Result<HttpResponse> {
    let (region, version, path) = path_info.into_inner();
    serve_file(&data, &request, Target::Region(region), version, path).await
}

#[get("/{region}/{version}/hash/{repository}/{category}/{hash}/")]
async fn get_hash_region(
    data: web::Data<MessageQueue>,
    path_info: web::Path<(Region, GameVersion, u8, u8, String)>,
) -> Result<HttpResponse> {
    let (region, version, repository, category, hash) = path_info.into_inner();
    serve_hash(
        &data,
        Target::Region(region),
        version,
        repository,
        category,
        hash,
    )
    .await
}

#[get("/{region}/{version}/paths/{list_id}/")]
async fn get_paths_region(
    data: web::Data<MessageQueue>,
    request: HttpRequest,
    path_info: web::Path<(Region, GameVersion, String)>,
) -> Result<HttpResponse> {
    let (region, version, list_id) = path_info.into_inner();
    serve_presence(&data, &request, Target::Region(region), version, &list_id).await
}

#[get("/{region}/{version}/exists/")]
async fn get_exists_region(
    data: web::Data<MessageQueue>,
    path_info: web::Path<(Region, GameVersion)>,
    query: web::Query<ExistsQuery>,
) -> Result<HttpResponse> {
    let (region, version) = path_info.into_inner();
    serve_exists(&data, Target::Region(region), version, &query.files).await
}

// Per-repository escape hatch. Structurally distinct from the region routes by segment count, so
// the two can never collide.

#[get("/repo/{slug}/latest/{rest:.*}/")]
async fn get_latest_repo(
    data: web::Data<MessageQueue>,
    request: HttpRequest,
    path_info: web::Path<(Slug, String)>,
) -> Result<HttpResponse> {
    let (slug, rest) = path_info.into_inner();
    redirect_latest(
        &data,
        &request,
        Target::Repo(slug),
        &format!("repo/{slug}"),
        &rest,
    )
    .await
}

#[get("/repo/{slug}/{version}/file/{path:.*}/")]
async fn get_file_repo(
    data: web::Data<MessageQueue>,
    request: HttpRequest,
    path_info: web::Path<(Slug, GameVersion, String)>,
) -> Result<HttpResponse> {
    let (slug, version, path) = path_info.into_inner();
    serve_file(&data, &request, Target::Repo(slug), version, path).await
}

#[get("/repo/{slug}/{version}/hash/{repository}/{category}/{hash}/")]
async fn get_hash_repo(
    data: web::Data<MessageQueue>,
    path_info: web::Path<(Slug, GameVersion, u8, u8, String)>,
) -> Result<HttpResponse> {
    let (slug, version, repository, category, hash) = path_info.into_inner();
    serve_hash(
        &data,
        Target::Repo(slug),
        version,
        repository,
        category,
        hash,
    )
    .await
}

#[get("/repo/{slug}/{version}/paths/{list_id}/")]
async fn get_paths_repo(
    data: web::Data<MessageQueue>,
    request: HttpRequest,
    path_info: web::Path<(Slug, GameVersion, String)>,
) -> Result<HttpResponse> {
    let (slug, version, list_id) = path_info.into_inner();
    serve_presence(&data, &request, Target::Repo(slug), version, &list_id).await
}

#[get("/repo/{slug}/{version}/exists/")]
async fn get_exists_repo(
    data: web::Data<MessageQueue>,
    path_info: web::Path<(Slug, GameVersion)>,
    query: web::Query<ExistsQuery>,
) -> Result<HttpResponse> {
    let (slug, version) = path_info.into_inner();
    serve_exists(&data, Target::Repo(slug), version, &query.files).await
}

#[get("/repositories/")]
async fn get_repositories(data: web::Data<MessageQueue>) -> Result<HttpResponse> {
    let repositories = data
        .repositories()
        .await
        .map_err(ErrorInternalServerError)?;
    Ok(HttpResponse::Ok().json(RepositoriesInfo { repositories }))
}

#[derive(Debug, Serialize)]
struct GithubOAuthConfig {
    client_id: String,
}

#[get("/github/oauth/config/")]
async fn get_github_oauth_config(config: web::Data<Config>) -> Result<HttpResponse> {
    Ok(HttpResponse::Ok().json(GithubOAuthConfig {
        client_id: config.github_client_id.clone(),
    }))
}

#[derive(Debug, Deserialize)]
struct GithubOAuthRequest {
    code: String,
    code_verifier: Option<String>,
    redirect_uri: Option<String>,
}

#[post("/github/oauth/token/")]
async fn post_github_oauth_token(
    config: web::Data<Config>,
    body: web::Json<GithubOAuthRequest>,
) -> Result<HttpResponse> {
    if config.github_client_id.is_empty() || config.github_client_secret.is_empty() {
        return Err(ErrorInternalServerError("GitHub OAuth is not configured"));
    }

    let mut params = Map::new();
    params.insert(
        "client_id".into(),
        Value::String(config.github_client_id.clone()),
    );
    params.insert(
        "client_secret".into(),
        Value::String(config.github_client_secret.clone()),
    );
    params.insert("code".into(), Value::String(body.code.clone()));
    if let Some(verifier) = &body.code_verifier {
        params.insert("code_verifier".into(), Value::String(verifier.clone()));
    }
    if let Some(redirect_uri) = &body.redirect_uri {
        params.insert("redirect_uri".into(), Value::String(redirect_uri.clone()));
    }

    let response = reqwest::Client::new()
        .post("https://github.com/login/oauth/access_token")
        .header("Accept", "application/json")
        .json(&params)
        .send()
        .await
        .map_err(ErrorInternalServerError)?;

    let value: Value = response.json().await.map_err(ErrorInternalServerError)?;
    Ok(HttpResponse::Ok().json(value))
}

/// BGM song metadata proxied from OrchestrionPlugin Google Sheet, keyed by BGM row id and language.
const SONGS_SHEET: &str = "https://docs.google.com/spreadsheets/d/1s-xJjxqp6pwS7oewNy1aOQnr3gaJbewvIBbyYchZ6No/gviz/tq?tqx=out:csv&sheet=";
const SONG_SHEETS: [&str; 5] = ["en", "ja", "fr", "de", "zh"];
const SONGS_TTL: Duration = Duration::from_secs(6 * 60 * 60);
type SongsCache = Mutex<HashMap<&'static str, (Instant, Arc<String>)>>;
static SONGS_CACHE: LazyLock<SongsCache> = LazyLock::new(|| Mutex::new(HashMap::new()));

#[get("/songs/{lang}/")]
async fn get_songs(lang: web::Path<String>) -> Result<HttpResponse> {
    let sheet = SONG_SHEETS
        .into_iter()
        .find(|&s| s == lang.as_str())
        .unwrap_or("en");

    let cached = SONGS_CACHE
        .lock()
        .unwrap()
        .get(sheet)
        .filter(|(fetched, _)| fetched.elapsed() < SONGS_TTL)
        .map(|(_, json)| json.clone());

    let json = match cached {
        Some(json) => json,
        None => {
            let json = Arc::new(build_songs(sheet).await.map_err(ErrorInternalServerError)?);
            SONGS_CACHE
                .lock()
                .unwrap()
                .insert(sheet, (Instant::now(), json.clone()));
            json
        }
    };

    Ok(HttpResponse::Ok()
        .insert_header(CacheControl(vec![
            CacheDirective::Public,
            CacheDirective::MaxAge(60 * 60 * 6),
        ]))
        .content_type("application/json")
        .body(json.as_ref().clone()))
}

async fn build_songs(sheet: &str) -> anyhow::Result<String> {
    let client = reqwest::Client::new();
    let meta_csv = client
        .get(format!("{SONGS_SHEET}metadata"))
        .send()
        .await?
        .error_for_status()?
        .text()
        .await?;
    let lang_csv = client
        .get(format!("{SONGS_SHEET}{sheet}"))
        .send()
        .await?
        .error_for_status()?
        .text()
        .await?;

    // metadata sheet: id, duration (seconds)
    let mut durations = std::collections::HashMap::new();
    for record in csv::Reader::from_reader(meta_csv.as_bytes()).records() {
        let record = record?;
        if let (Some(Ok(id)), Some(Ok(duration))) = (
            record.get(0).map(str::parse::<u32>),
            record.get(1).map(str::parse::<f64>),
        ) {
            durations.insert(id, duration.round() as u64);
        }
    }

    // language sheet: id, title, alt title, special mode title, locations, comments
    let mut songs = Map::new();
    for record in csv::Reader::from_reader(lang_csv.as_bytes()).records() {
        let record = record?;
        let Some(Ok(id)) = record.get(0).map(str::parse::<u32>) else {
            continue;
        };
        let title = record.get(1).unwrap_or("").trim();
        if title.is_empty() || title == "None" {
            continue;
        }
        let mut song = Map::new();
        song.insert("t".into(), Value::from(title));
        for (key, column) in [("a", 2), ("s", 3), ("l", 4), ("i", 5)] {
            let value = record.get(column).unwrap_or("").trim();
            if !value.is_empty() {
                song.insert(key.into(), Value::from(value));
            }
        }
        if let Some(&duration) = durations.get(&id).filter(|&&d| d > 0) {
            song.insert("d".into(), Value::from(duration));
        }
        songs.insert(id.to_string(), Value::Object(song));
    }

    Ok(serde_json::to_string(&Value::Object(songs))?)
}

/// BNpc base -> crowdsourced name ids, distilled from FFXIVGachaSpreadsheet's pairing dumps.
const BNPC_DATA: &str =
    "https://raw.githubusercontent.com/Infiziert90/FFXIVGachaSpreadsheet/master/website/static/data/";
const BNPC_TTL: Duration = Duration::from_secs(12 * 60 * 60);
type BnpcCache = Mutex<Option<(Instant, Arc<String>)>>;
static BNPC_CACHE: LazyLock<BnpcCache> = LazyLock::new(|| Mutex::new(None));

#[derive(Deserialize)]
struct BnpcSimple {
    #[serde(rename = "Base")]
    base: u32,
    #[serde(rename = "Names")]
    names: Vec<u32>,
}

#[derive(Deserialize)]
struct BnpcSighting {
    #[serde(rename = "Base")]
    base: u32,
    #[serde(rename = "Name")]
    name: u32,
    #[serde(rename = "Records")]
    records: u64,
}

#[derive(Deserialize)]
struct BnpcPairings {
    #[serde(rename = "BnpcPairings")]
    pairings: HashMap<String, BnpcSighting>,
}

#[get("/bnpc/")]
async fn get_bnpc() -> Result<HttpResponse> {
    let cached = BNPC_CACHE
        .lock()
        .unwrap()
        .as_ref()
        .filter(|(fetched, _)| fetched.elapsed() < BNPC_TTL)
        .map(|(_, json)| json.clone());

    let json = match cached {
        Some(json) => json,
        None => {
            let json = Arc::new(build_bnpc().await.map_err(ErrorInternalServerError)?);
            *BNPC_CACHE.lock().unwrap() = Some((Instant::now(), json.clone()));
            json
        }
    };

    Ok(HttpResponse::Ok()
        .insert_header(CacheControl(vec![
            CacheDirective::Public,
            CacheDirective::MaxAge(60 * 60 * 12),
        ]))
        .content_type("application/json")
        .body(json.as_ref().clone()))
}

/// Keeps the pairing map hot, so the first client is not the one that waits for the upstream dumps.
pub fn prewarm_bnpc() {
    tokio::spawn(async {
        loop {
            match build_bnpc().await {
                Ok(json) => *BNPC_CACHE.lock().unwrap() = Some((Instant::now(), Arc::new(json))),
                Err(error) => log::warn!("Could not prewarm BNpc names: {error}"),
            }
            tokio::time::sleep(BNPC_TTL).await;
        }
    });
}

async fn build_bnpc() -> anyhow::Result<String> {
    let client = reqwest::Client::new();
    let simple: Vec<BnpcSimple> = client
        .get(format!("{BNPC_DATA}BnpcPairsSimple.json"))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let seen: BnpcPairings = client
        .get(format!("{BNPC_DATA}BnpcPairsV2.json"))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    distil(simple, seen)
}

fn distil(simple: Vec<BnpcSimple>, seen: BnpcPairings) -> anyhow::Result<String> {
    let mut sightings: HashMap<(u32, u32), u64> = HashMap::new();
    for held in seen.pairings.into_values() {
        *sightings.entry((held.base, held.name)).or_default() += held.records;
    }

    // A base wearing several names is only ambiguous on paper: the one sighted most is the one it
    // goes by, so the caller can take the head of the list and ignore the tail.
    let mut bases = Map::new();
    for BnpcSimple { base, mut names } in simple {
        names.sort_by_key(|&name| {
            std::cmp::Reverse(sightings.get(&(base, name)).copied().unwrap_or(0))
        });
        bases.insert(base.to_string(), Value::from(names));
    }

    Ok(serde_json::to_string(&Value::Object(bases))?)
}

fn log_error<B: MessageBody + 'static>(
    is_client: bool,
    res: ServiceResponse<B>,
) -> actix_web::Result<ErrorHandlerResponse<B>> {
    Ok(ErrorHandlerResponse::Future(Box::pin(log_error2(
        is_client, res,
    ))))
}

async fn log_error2<B: MessageBody + 'static>(
    is_client: bool,
    res: ServiceResponse<B>,
) -> actix_web::Result<ServiceResponse<EitherBody<B>>> {
    let (req, res) = res.into_parts();
    let (res, body) = res.into_parts();

    let body = {
        let data = actix_web::body::to_bytes_limited(body, 1 << 12).await;
        let line = match &data {
            Ok(Ok(data)) => String::from_utf8_lossy(data).into_owned(),
            Ok(Err(_)) => "Error reading body".to_string(),
            Err(_) => "Body too large".to_string(),
        };
        if is_client {
            log::error!("Client Error: {}", line);
        } else {
            log::error!("Server Error: {}", line);
        }

        match data {
            Ok(Ok(bytes)) => bytes,
            Ok(Err(_)) => Bytes::from_static(b"Body conversion failure"),
            Err(_) => Bytes::from_static(b"Body too large"),
        }
    };

    let res = ServiceResponse::new(req, res.map_body(|_head, _body| body))
        .map_into_boxed_body()
        .map_into_right_body();

    Ok(res)
}

/// Every case here is refused before the list is consulted, so no test fetches one. A 429 rather
/// than a 404 is also what says the route is reachable past `/paths/{list_id}/`.
#[cfg(test)]
mod tests {
    use actix_web::{App, http::StatusCode, test};
    use serde_json::json;

    use super::*;
    use crate::{config::Report, paths::PathIndex};


    #[actix_web::test]
    async fn a_base_takes_the_name_it_was_sighted_under_most() {
        let simple = vec![BnpcSimple {
            base: 7,
            names: vec![20, 9, 44],
        }];
        let seen = BnpcPairings {
            pairings: HashMap::from([
                (
                    "a".to_owned(),
                    BnpcSighting {
                        base: 7,
                        name: 20,
                        records: 3,
                    },
                ),
                (
                    "b".to_owned(),
                    BnpcSighting {
                        base: 7,
                        name: 9,
                        records: 100,
                    },
                ),
            ]),
        };

        // 44 was never sighted, so it sorts behind both.
        assert_eq!(distil(simple, seen).unwrap(), r#"{"7":[9,20,44]}"#);
    }

    fn collector() -> web::Data<Collector> {
        web::Data::new(Collector::new(
            std::sync::Arc::new(PathIndex::new(crate::config::PathList::default())),
            Report {
                enabled: false,
                forward_url: String::new(),
                max_paths: 2,
                per_hour: 0,
            },
        ))
    }

    #[actix_web::test]
    async fn a_submission_is_refused_before_it_can_cost_anything() {
        let app = test::init_service(App::new().app_data(collector()).service(service())).await;

        for (body, want) in [
            (json!({ "paths": [] }), StatusCode::BAD_REQUEST),
            (
                json!({ "paths": ["ui/a.uld", "ui/b.uld", "ui/c.uld"] }),
                StatusCode::BAD_REQUEST,
            ),
            (
                json!({ "entries": ["ui/uld/a.uld"] }),
                StatusCode::BAD_REQUEST,
            ),
            (
                json!({ "paths": ["ui/uld/a.uld"] }),
                StatusCode::TOO_MANY_REQUESTS,
            ),
        ] {
            let request = test::TestRequest::post()
                .uri("/api/report/")
                .set_json(&body)
                .to_request();
            assert_eq!(
                test::call_service(&app, request).await.status(),
                want,
                "{body}"
            );
        }
    }
}
