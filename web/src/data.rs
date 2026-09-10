use std::{
    collections::{HashMap, HashSet},
    fmt,
    io::{Read, Seek},
    path::Path,
    str::FromStr,
    sync::Arc,
    time::Duration,
};

use anyhow::bail;
use bytes::Bytes;
use futures_util::StreamExt;
use ironworks::{
    Ironworks,
    sqpack::{self, FileKind, SqPack, VInstall, Vfs},
};
use mini_moka::sync::{Cache, CacheBuilder};
use serde::{Deserialize, Serialize};
use tokio::runtime::Handle;
use xiv_cache::{
    builder::ServerBuilder,
    file::CacheFile,
    server::{Server, SlugData},
    stream::CacheFileStream,
};
use xiv_core::file::{slug::Slug, version::GameVersion};

use crate::{blocking_stream::BlockingReader, config::AssetCache, smart_bufreader::SmartBufReader};

#[derive(Debug, Clone, Serialize)]
pub struct VersionInfo {
    pub latest: GameVersion,
    pub versions: Vec<GameVersion>,
    /// Human-readable name per version, where the patch index records one.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub names: std::collections::BTreeMap<GameVersion, String>,
}

impl From<SlugData> for VersionInfo {
    fn from(value: SlugData) -> Self {
        Self {
            latest: value.latest_version,
            versions: value.versions,
            names: std::collections::BTreeMap::new(),
        }
    }
}

/// Attach the patch index's human names to a version list. A region without metadata, or a
/// version the index does not name, is left unlabelled rather than guessed at.
pub async fn label_versions(info: &mut VersionInfo, region: Region) {
    static CLIENT: std::sync::OnceLock<reqwest::Client> = std::sync::OnceLock::new();
    let client = CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .user_agent("FFXIV PATCH CLIENT")
            .build()
            .unwrap_or_default()
    });
    let indexed = match region {
        Region::Global => xiv_core::index::Region::Global,
        Region::Korea => xiv_core::index::Region::Korea,
        Region::China => xiv_core::index::Region::China,
        Region::Taiwan => xiv_core::index::Region::Taiwan,
    };
    let metadata = match xiv_core::index::fetch_metadata(
        client,
        xiv_core::index::DEFAULT_INDEX,
        indexed,
    )
    .await
    {
        Ok(metadata) => metadata,
        Err(error) => {
            log::debug!("no metadata for {region}: {error}");
            return;
        }
    };
    for version in std::iter::once(&info.latest).chain(info.versions.iter()) {
        if let Some(label) = metadata.label(version) {
            info.names.insert(version.clone(), label);
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RepositoryInfo {
    pub slug: Slug,
    pub name: String,
    /// Both derived from `name`, which is the raw publisher path. Modelled rather than left for
    /// the client to re-parse, since the API is keyed by region.
    pub region: Option<Region>,
    pub repo: Option<Repo>,
    pub latest: GameVersion,
}

impl RepositoryInfo {
    fn from_slug_data(slug: Slug, value: SlugData) -> Self {
        let parsed = parse_repo(&value.repository);
        Self {
            slug,
            name: value.repository,
            region: parsed.map(|(region, _)| region),
            repo: parsed.map(|(_, repo)| repo),
            latest: value.latest_version,
        }
    }
}

type CacheIronworks = Ironworks<SqPack<VInstall<CacheVfs>>>;

#[derive(Debug)]
pub struct StoredFile {
    pub kind: FileKind,
    pub bytes: Bytes,
}

impl StoredFile {
    fn read<R: Read + Seek>(mut file: sqpack::File<R>) -> Result<Self, ironworks::Error> {
        let kind = file.kind();
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        Ok(Self {
            kind,
            bytes: bytes.into(),
        })
    }
}

#[derive(Debug)]
pub struct GameData {
    cache: Server,
    readahead_size: usize,
    ironworks_cache: Cache<(Target, GameVersion), Arc<CacheIronworks>>,
    file_cache: Cache<(Target, GameVersion, String), Arc<StoredFile>>,
}

impl GameData {
    pub async fn new(
        cache_config: ServerBuilder,
        asset_config: AssetCache,
        readahead_size: usize,
    ) -> anyhow::Result<Self> {
        let server = cache_config.build().await?;

        Ok(Self {
            cache: server,
            readahead_size,
            ironworks_cache: CacheBuilder::new(asset_config.version_capacity)
                .time_to_live(Duration::from_secs(60 * asset_config.version_ttl_minutes))
                .build(),
            file_cache: CacheBuilder::new(asset_config.file_capacity)
                .time_to_live(Duration::from_secs(60 * asset_config.file_ttl_minutes))
                .build(),
        })
    }

    /// The version chain a target offers, for resolving `latest`.
    pub async fn versions_for(&self, target: Target) -> Option<VersionInfo> {
        target_versions(&self.cache, target).await.ok().flatten()
    }

    /// Which regions the slug list covers.
    pub async fn regions(&self) -> anyhow::Result<Vec<Region>> {
        regions(&self.cache).await
    }

    /// Whether a target carries sqpack data; boot does not.
    pub async fn has_sqpack(&self, target: Target) -> bool {
        has_sqpack(&self.cache, target).await
    }

    /// Whether any repository backing the target published exactly this version.
    pub async fn version_valid(&self, target: Target, version: &GameVersion) -> bool {
        match target {
            Target::Region(region) => version_valid(&self.cache, region, version)
                .await
                .unwrap_or(false),
            Target::Repo(slug) => self
                .cache
                .get_slug(slug)
                .await
                .is_ok_and(|data| data.versions.contains(version)),
        }
    }

    pub async fn repositories(&self) -> anyhow::Result<Vec<RepositoryInfo>> {
        let slugs = self.cache.get_slug_list().await?;
        let mut repositories = Vec::with_capacity(slugs.len());
        for slug in slugs {
            if let Ok(slug_data) = self.cache.get_slug(slug).await {
                repositories.push(RepositoryInfo::from_slug_data(slug, slug_data));
            }
        }
        Ok(repositories)
    }

    pub async fn get_version(
        &self,
        target: Target,
        version: GameVersion,
    ) -> Result<Arc<CacheIronworks>, ironworks::Error> {
        let key = (target, version);
        if let Some(ret) = self.ironworks_cache.get(&key) {
            return Ok(ret);
        }
        let (target, version) = key;

        log::info!("Fetching ironworks for {target}, version: {version}");
        let vfs = CacheVfs::new(
            self.cache.clone(),
            self.readahead_size,
            target,
            version.clone(),
        )
        .await
        .map_err(|e| ironworks::Error::Resource(Box::new(std::io::Error::other(e))))?;
        let resource = VInstall::at_sqpack(vfs);
        let resource = ironworks::sqpack::SqPack::new(resource);
        let ironworks = Arc::new(Ironworks::new().with_resource(resource));
        self.ironworks_cache
            .insert((target, version), ironworks.clone());
        Ok(ironworks)
    }

    pub async fn get(
        &self,
        target: Target,
        version: GameVersion,
        file: String,
    ) -> Result<Arc<StoredFile>, ironworks::Error> {
        let key = (target, version, file);
        if let Some(ret) = self.file_cache.get(&key) {
            return Ok(ret);
        }
        let (target, version, file) = key;

        let ironworks = self.get_version(target, version.clone()).await?;

        log::info!("Fetching file: {file} for {target}, version: {version}");
        let stream = ironworks.find_first(&file, |package| package.file(&file))?;
        let file_data = StoredFile::read(stream)?;
        log::info!(
            "File fetched: {file} for {target}, version: {version}, size: {}",
            file_data.bytes.len()
        );

        let data = Arc::new(file_data);
        self.file_cache
            .insert((target, version, file), data.clone());
        Ok(data)
    }

    /// Read a file the game records only as a hash. Unnamed files have no path to look up, so the
    /// repository and category from the index the hash came out of stand in for one.
    pub async fn get_by_hash(
        &self,
        target: Target,
        version: GameVersion,
        repository: u8,
        category: u8,
        hash: ironworks::sqpack::IndexHash,
    ) -> Result<Arc<StoredFile>, ironworks::Error> {
        let ironworks = self.get_version(target, version.clone()).await?;

        log::info!(
            "Fetching unnamed file: {hash:?} in {repository}/{category} for {target} {version}"
        );
        let mut last = None;
        for package in ironworks.resources() {
            match package.file_by_hash(repository, category, hash) {
                Ok(file) => {
                    let data = StoredFile::read(file)?;
                    log::info!("Unnamed file fetched: {hash:?}, size: {}", data.bytes.len());
                    return Ok(Arc::new(data));
                }
                Err(error) => last = Some(error),
            }
        }
        Err(
            last.unwrap_or(ironworks::Error::NotFound(ironworks::ErrorValue::Other(
                format!("{hash:?}"),
            ))),
        )
    }

    pub async fn warm_indexes(&self, target: Target, version: GameVersion) -> anyhow::Result<()> {
        let mut indexes = Vec::new();
        for (slug, version) in contributions(&self.cache, target, version).await? {
            let clut = self.cache.get_clut(slug, version.clone()).await?;
            indexes.extend(
                clut.files()
                    .filter(|path| path.ends_with(".index") || path.ends_with(".index2"))
                    .map(|path| (slug, version.clone(), path.to_owned())),
            );
        }

        // Each read buffers a whole index file, so this is bounded rather than unleashed.
        const CONCURRENCY: usize = 12;

        let started = std::time::Instant::now();
        let count = indexes.len();
        let failures: Vec<_> =
            futures_util::stream::iter(indexes.into_iter().map(|(slug, version, path)| {
                let server = self.cache.clone();
                async move {
                    let file = CacheFile::new(server, slug, version, path.clone())
                        .await
                        .map_err(|e| (path.clone(), e))?;
                    let mut buffer = vec![0u8; file.len() as usize];
                    file.pread(0, &mut buffer).await.map_err(|e| (path, e))
                }
            }))
            .buffer_unordered(CONCURRENCY)
            .filter_map(|result| std::future::ready(result.err()))
            .collect()
            .await;
        log::info!(
            "Warmed {} of {count} sqpack indexes in {:?}",
            count - failures.len(),
            started.elapsed()
        );
        for (path, error) in failures.iter().take(5) {
            log::warn!("Could not warm {path}: {error}");
        }
        Ok(())
    }

    pub async fn exists(
        &self,
        target: Target,
        version: GameVersion,
        files: Vec<String>,
    ) -> Result<Vec<bool>, ironworks::Error> {
        let ironworks = self.get_version(target, version).await?;
        Ok(files
            .iter()
            .map(|file| ironworks.exists(file).unwrap_or(false))
            .collect())
    }

    pub async fn close(&self) -> anyhow::Result<()> {
        self.cache.close().await
    }
}

/// A publishing region. This, not a slug, is how the API addresses the game: a slug identifies
/// one repository within a region, which is an implementation detail of how patches are shipped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Region {
    Global,
    Korea,
    China,
    Taiwan,
}

impl Region {
    pub const ALL: [Self; 4] =
        [Region::Global, Region::Korea, Region::China, Region::Taiwan];

    fn from_publisher(publisher: &str) -> Option<Self> {
        match publisher {
            "ffxivneo" => Some(Region::Global),
            "actoz" => Some(Region::Korea),
            "shanda" => Some(Region::China),
            "ffxivtc" => Some(Region::Taiwan),
            _ => None,
        }
    }
}

impl fmt::Display for Region {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Region::Global => "global",
            Region::Korea => "korea",
            Region::China => "china",
            Region::Taiwan => "taiwan",
        })
    }
}

impl FromStr for Region {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "global" => Ok(Region::Global),
            "korea" => Ok(Region::Korea),
            "china" => Ok(Region::China),
            "taiwan" => Ok(Region::Taiwan),
            other => bail!("unknown region {other}"),
        }
    }
}

impl<'a> Deserialize<'a> for Region {
    fn deserialize<D: serde::Deserializer<'a>>(deserializer: D) -> Result<Self, D::Error> {
        String::deserialize(deserializer)?
            .parse()
            .map_err(|_| serde::de::Error::custom("unknown region"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(into = "String")]
pub enum Repo {
    Boot,
    Game,
    Ex(u8),
}

impl Repo {
    fn from_node(node: &str) -> Option<Self> {
        match node {
            "boot" => Some(Repo::Boot),
            "game" => Some(Repo::Game),
            _ => node
                .strip_prefix("ex")
                .and_then(|n| n.parse::<u8>().ok())
                .filter(|n| (1..=5).contains(n))
                .map(Repo::Ex),
        }
    }

    /// Whether this repository carries sqpack data. Boot ships the launcher rather than game
    /// files, so every sqpack-backed endpoint has to refuse it.
    pub fn is_game_side(self) -> bool {
        matches!(self, Repo::Game | Repo::Ex(_))
    }
}

impl fmt::Display for Repo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Repo::Boot => f.write_str("boot"),
            Repo::Game => f.write_str("game"),
            Repo::Ex(n) => write!(f, "ex{n}"),
        }
    }
}

impl From<Repo> for String {
    fn from(repo: Repo) -> Self {
        repo.to_string()
    }
}

/// The base game plus its expansions, in sqpack load order.
const GAME_SIDE: [Repo; 6] = [
    Repo::Game,
    Repo::Ex(1),
    Repo::Ex(2),
    Repo::Ex(3),
    Repo::Ex(4),
    Repo::Ex(5),
];

fn parse_repo(repository: &str) -> Option<(Region, Repo)> {
    let publisher = repository.split('/').next()?;
    let node = repository.rsplit('/').next()?;
    Some((Region::from_publisher(publisher)?, Repo::from_node(node)?))
}

/// Expand a base game slug into itself plus its region's expansions, each pinned to the newest
/// version at or before `version` (an expansion that didn't exist yet is dropped). A non-game
/// slug (boot, unknown publisher) contributes only itself, so per-repo browsing is unchanged.
/// What a request addresses. A region is the game as a whole; a slug is one repository, the
/// escape hatch for per-repository work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Target {
    Region(Region),
    Repo(Slug),
}

impl Target {
    /// Filename-safe rendering, for the stored presence cache. `Display` uses a slash.
    pub fn file_key(self) -> String {
        match self {
            Target::Region(region) => region.to_string(),
            Target::Repo(slug) => format!("repo-{slug}"),
        }
    }
}

impl fmt::Display for Target {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Target::Region(region) => write!(f, "{region}"),
            Target::Repo(slug) => write!(f, "repo/{slug}"),
        }
    }
}

/// The (slug, version) pairs backing a target, which is what a CLUT set is assembled from.
pub async fn contributions(
    server: &Server,
    target: Target,
    version: GameVersion,
) -> anyhow::Result<Vec<(Slug, GameVersion)>> {
    match target {
        Target::Region(region) => region_contributions(server, region, version).await,
        Target::Repo(slug) => game_contributions(server, slug, version).await,
    }
}

/// Whether a target carries sqpack data at all. Boot ships the launcher, so every sqpack-backed
/// endpoint has to refuse it rather than answer emptily.
pub async fn has_sqpack(server: &Server, target: Target) -> bool {
    match target {
        Target::Region(_) => true,
        Target::Repo(slug) => server
            .get_slug(slug)
            .await
            .ok()
            .and_then(|data| parse_repo(&data.repository))
            .is_some_and(|(_, repo)| repo.is_game_side()),
    }
}

/// The version chain a target offers.
pub async fn target_versions(
    server: &Server,
    target: Target,
) -> anyhow::Result<Option<VersionInfo>> {
    match target {
        Target::Region(region) => game_versions(server, region).await,
        Target::Repo(slug) => Ok(server.get_slug(slug).await.map(VersionInfo::from).ok()),
    }
}

/// The game-side repositories of a region, each with the versions it has published, keyed by repo.
async fn region_repos(
    server: &Server,
    region: Region,
) -> anyhow::Result<HashMap<Repo, (Slug, Vec<GameVersion>)>> {
    let mut by_repo = HashMap::new();
    for slug in server.get_slug_list().await? {
        let Ok(data) = server.get_slug(slug).await else {
            continue;
        };
        if let Some((r, repo)) = parse_repo(&data.repository)
            && r == region
            && repo.is_game_side()
        {
            by_repo.insert(repo, (slug, data.versions));
        }
    }
    Ok(by_repo)
}

/// Which regions the slug list actually covers.
pub async fn regions(server: &Server) -> anyhow::Result<Vec<Region>> {
    let mut found = Vec::new();
    for slug in server.get_slug_list().await? {
        let Ok(data) = server.get_slug(slug).await else {
            continue;
        };
        if let Some((region, repo)) = parse_repo(&data.repository)
            && repo.is_game_side()
            && !found.contains(&region)
        {
            found.push(region);
        }
    }
    found.sort_by_key(|region| Region::ALL.iter().position(|r| r == region));
    Ok(found)
}

/// The base game's version chain, which is the canonical list to offer for a region.
pub async fn game_versions(server: &Server, region: Region) -> anyhow::Result<Option<VersionInfo>> {
    let by_repo = region_repos(server, region).await?;
    let Some((slug, _)) = by_repo.get(&Repo::Game) else {
        return Ok(None);
    };
    Ok(server.get_slug(*slug).await.map(VersionInfo::from).ok())
}

/// Whether any repository in the region published exactly this version.
///
/// Versions are date-ordered, so they are comparable across repositories: a version taken from an
/// expansion's chain still orders correctly against the base game's, which is what lets
/// [`region_contributions`] backfill around it without caring where it came from.
pub async fn version_valid(
    server: &Server,
    region: Region,
    version: &GameVersion,
) -> anyhow::Result<bool> {
    Ok(region_repos(server, region)
        .await?
        .values()
        .any(|(_, versions)| versions.contains(version)))
}

/// Every repository backing a region at `version`, each pinned to its newest version at or before
/// it. Reconstructs the install as it would have existed at that point in time; an expansion that
/// did not exist yet is dropped.
pub async fn region_contributions(
    server: &Server,
    region: Region,
    version: GameVersion,
) -> anyhow::Result<Vec<(Slug, GameVersion)>> {
    Ok(backfill(&region_repos(server, region).await?, &version))
}

/// The pure half of [`region_contributions`], so the ordering and drop rules can be tested without
/// a live server. In sqpack load order, since that is the order the resulting install layers in.
fn backfill(
    by_repo: &HashMap<Repo, (Slug, Vec<GameVersion>)>,
    version: &GameVersion,
) -> Vec<(Slug, GameVersion)> {
    let mut contributions = Vec::new();
    for repo in GAME_SIDE {
        let Some((slug, versions)) = by_repo.get(&repo) else {
            continue;
        };
        if let Some(pinned) = versions.iter().filter(|v| *v <= version).max() {
            contributions.push((*slug, pinned.clone()));
        }
    }
    contributions
}

async fn game_contributions(
    server: &Server,
    slug: Slug,
    version: GameVersion,
) -> anyhow::Result<Vec<(Slug, GameVersion)>> {
    let region = server
        .get_slug(slug)
        .await
        .ok()
        .and_then(|data| parse_repo(&data.repository))
        .filter(|(_, repo)| repo.is_game_side())
        .map(|(region, _)| region);
    let Some(region) = region else {
        return Ok(vec![(slug, version)]);
    };

    let Ok(slugs) = server.get_slug_list().await else {
        return Ok(vec![(slug, version)]);
    };
    let mut by_repo: HashMap<Repo, (Slug, Vec<GameVersion>)> = HashMap::new();
    for slug in slugs {
        let Ok(data) = server.get_slug(slug).await else {
            continue;
        };
        if let Some((r, repo)) = parse_repo(&data.repository)
            && r == region
            && repo.is_game_side()
        {
            by_repo.insert(repo, (slug, data.versions));
        }
    }

    let mut contributions = Vec::new();
    for repo in GAME_SIDE {
        let Some((slug, versions)) = by_repo.get(&repo) else {
            continue;
        };
        if let Some(pinned) = versions.iter().filter(|v| **v <= version).max() {
            contributions.push((*slug, pinned.clone()));
        }
    }
    if contributions.is_empty() {
        contributions.push((slug, version));
    }
    Ok(contributions)
}

/// A read-only sqpack Vfs spanning every repository of one game install: the base game and all
/// expansions merged into one file set, with each path routed back to the slug that owns it.
pub struct CacheVfs {
    server: Server,
    readahead_size: usize,
    files: HashMap<String, (Slug, GameVersion)>,
    folders: HashSet<String>,
}

impl CacheVfs {
    pub async fn new(
        server: Server,
        readahead_size: usize,
        target: Target,
        version: GameVersion,
    ) -> anyhow::Result<Self> {
        let mut files = HashMap::new();
        let mut folders = HashSet::new();
        let mut resident = 0;
        for (slug, version) in contributions(&server, target, version).await? {
            let clut = server.get_clut(slug, version.clone()).await?;
            resident += clut.resident_size();
            for path in clut.files() {
                files.insert(path.to_owned(), (slug, version.clone()));
            }
            folders.extend(clut.folders.iter().cloned());
        }
        log::info!(
            "Install for {target}: {} files, CLUTs resident in {resident} bytes",
            files.len()
        );

        Ok(Self {
            server,
            readahead_size,
            files,
            folders,
        })
    }
}

impl Vfs for CacheVfs {
    type File = SmartBufReader<BlockingReader<CacheFileStream>>;

    fn exists(&self, path: impl AsRef<Path>) -> bool {
        let path = Path::new("sqpack").join(path);
        let path_str = path.to_str().unwrap_or_default();
        self.files.contains_key(path_str)
            || self.folders.contains(path_str)
            || self.files.keys().chain(self.folders.iter()).any(|k| {
                Path::new(k)
                    .parent()
                    .map(|parent| parent == path)
                    .unwrap_or(false)
                    || Path::new(k).ancestors().any(|a| a == path)
            })
    }

    fn open(&self, path: impl AsRef<Path>) -> std::io::Result<Self::File> {
        let path = Path::new("sqpack").join(path);
        let path = path.to_str().unwrap_or_default();

        let Some((slug, version)) = self.files.get(path) else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "file not found",
            ));
        };

        let file = tokio::task::block_in_place(|| {
            Handle::current().block_on(async move {
                CacheFile::new(
                    self.server.clone(),
                    *slug,
                    version.clone(),
                    path.to_string(),
                )
                .await
            })
        })?;

        Ok(SmartBufReader::unchecked_new(
            BlockingReader::new(file.into_reader()),
            self.readahead_size,
        ))
    }
}

#[cfg(test)]
mod tests {
    fn version(s: &str) -> GameVersion {
        GameVersion::new(s).unwrap()
    }

    #[test]
    fn regions_round_trip_through_strings() {
        for region in Region::ALL {
            assert_eq!(region.to_string().parse::<Region>().unwrap(), region);
        }
        assert_eq!("GLOBAL".parse::<Region>().unwrap(), Region::Global);
        assert!("nintendo".parse::<Region>().is_err());
        assert_eq!(Region::Global.to_string(), "global");
    }

    /// The presence cache is stored under this key, so a region and a repository colliding would
    /// serve one target's map for the other.
    #[test]
    fn file_keys_never_collide_across_target_kinds() {
        let regions = Region::ALL.map(|r| Target::Region(r).file_key());
        for slug in ["4e9a232b", "6b936f08", "2b5cbc63"] {
            let repo = Target::Repo(Slug::from_str(slug).unwrap()).file_key();
            assert!(!regions.contains(&repo), "{repo}");
        }
    }

    #[test]
    fn repos_name_themselves() {
        assert_eq!(Repo::Boot.to_string(), "boot");
        assert_eq!(Repo::Game.to_string(), "game");
        assert_eq!(Repo::Ex(3).to_string(), "ex3");
    }

    /// The whole point of backfilling: a version taken from any repository's chain reconstructs a
    /// coherent install, with every other repository snapped to its newest release at or before it.
    #[test]
    fn backfill_snaps_siblings_and_drops_future_expansions() {
        let mut by_repo = HashMap::new();
        by_repo.insert(
            Repo::Game,
            (
                Slug::from_str("4e9a232b").unwrap(),
                vec![
                    version("2026.05.01.0000.0000"),
                    version("2026.06.18.0000.0000"),
                ],
            ),
        );
        by_repo.insert(
            Repo::Ex(1),
            (
                Slug::from_str("6b936f08").unwrap(),
                vec![version("2026.05.15.0000.0000")],
            ),
        );
        // Did not exist yet at the version we ask for.
        by_repo.insert(
            Repo::Ex(5),
            (
                Slug::from_str("00000005").unwrap(),
                vec![version("2026.07.01.0000.0000")],
            ),
        );

        let out = backfill(&by_repo, &version("2026.06.18.0000.0000"));
        let versions: Vec<String> = out.iter().map(|(_, v)| v.to_string()).collect();
        assert_eq!(
            versions,
            vec!["2026.06.18.0000.0000", "2026.05.15.0000.0000"],
            "game takes the exact version, ex1 backfills, ex5 is dropped"
        );
        assert_eq!(out[0].0, Slug::from_str("4e9a232b").unwrap(), "game leads");
    }

    /// A version from an expansion's chain is just as valid a reference point; everything still
    /// resolves around it, because versions are date-ordered across repositories.
    #[test]
    fn backfill_accepts_a_version_from_another_repos_chain() {
        let mut by_repo = HashMap::new();
        by_repo.insert(
            Repo::Game,
            (
                Slug::from_str("4e9a232b").unwrap(),
                vec![
                    version("2026.05.01.0000.0000"),
                    version("2026.06.18.0000.0000"),
                ],
            ),
        );
        by_repo.insert(
            Repo::Ex(1),
            (
                Slug::from_str("6b936f08").unwrap(),
                vec![version("2026.05.15.0000.0000")],
            ),
        );

        let out = backfill(&by_repo, &version("2026.05.15.0000.0000"));
        let versions: Vec<String> = out.iter().map(|(_, v)| v.to_string()).collect();
        assert_eq!(
            versions,
            vec!["2026.05.01.0000.0000", "2026.05.15.0000.0000"],
            "the game falls back to its older release, ex1 matches exactly"
        );
    }

    use super::*;

    #[test]
    fn parses_repository_names() {
        assert_eq!(
            parse_repo("ffxivneo/win32/release/game"),
            Some((Region::Global, Repo::Game))
        );
        assert_eq!(
            parse_repo("ffxivneo/win32/release/ex5"),
            Some((Region::Global, Repo::Ex(5)))
        );
        assert_eq!(
            parse_repo("actoz/win32/release_ko/ex1"),
            Some((Region::Korea, Repo::Ex(1)))
        );
        assert_eq!(
            parse_repo("shanda/win32/release_chs/game"),
            Some((Region::China, Repo::Game))
        );
        assert_eq!(
            parse_repo("ffxivneo/win32/release/boot"),
            Some((Region::Global, Repo::Boot))
        );
        assert_eq!(parse_repo("nintendo/win32/release/game"), None);
        assert_eq!(parse_repo("ffxivneo/win32/release/ex6"), None);
    }

    #[test]
    fn only_game_and_expansions_are_game_side() {
        assert!(Repo::Game.is_game_side());
        assert!(Repo::Ex(3).is_game_side());
        assert!(!Repo::Boot.is_game_side());
    }
}
