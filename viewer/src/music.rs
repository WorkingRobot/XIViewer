use std::collections::{HashMap, HashSet};
use std::io::Cursor;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, anyhow};
use egui::{
    Align, Button, CentralPanel, Color32, Layout, Rect, RichText, ScrollArea, Sense, Slider,
    TextEdit, UiBuilder, Vec2, Widget, containers::panel::Panel, pos2, vec2,
};
use ironworks::excel::Language;
use ironworks::file::File;
use ironworks::file::scd::{Codec, SoundContainer};
use serde::Deserialize;

use crate::audio::{self, Decoded, Player};
use crate::backend::Backend;
use crate::data::{FileProvider, FileProviderExt};
use crate::excel::base::CachedProvider;
use crate::excel::provider::{ExcelHeader, ExcelProvider, ExcelSheet};
use crate::goto::{ListNav, Palette, SUGGESTIONS};
use crate::settings::{LANGUAGE, api_base};
use crate::utils::{
    CollapsibleSidePanel, FuzzyMatcher, PromiseKind, Side, TrackedPromise, center, empty_view,
    export, fetch_url_str, file_name,
};

const FILTER_ID: &str = "music_filter";
const LIST_WIDTH: f32 = 340.0;
const LIST_MIN_WIDTH: f32 = 160.0;

#[derive(Deserialize, Default)]
struct SongInfo {
    #[serde(rename = "t", default)]
    title: String,
    #[serde(rename = "a", default)]
    alt: String,
    #[serde(rename = "s", default)]
    special: String,
    #[serde(rename = "l", default)]
    locations: String,
    #[serde(rename = "i", default)]
    info: String,
    #[serde(rename = "d", default)]
    duration: u32,
}

struct BgmTrack {
    row_id: u32,
    path: String,
}

enum Index {
    Idle,
    Loading(TrackedPromise<Result<Vec<BgmTrack>>>),
    Loaded(Vec<BgmTrack>),
    Failed(String),
}

enum Avail {
    Idle,
    Loading(TrackedPromise<Result<HashSet<String>>>),
    Ready(HashSet<String>),
    Failed,
}

enum Songs {
    Idle,
    Loading(TrackedPromise<Result<HashMap<u32, SongInfo>>>),
    Done,
}

#[derive(Clone, Copy)]
struct StreamInfo {
    codec: Codec,
    file_size: usize,
    stream_size: usize,
}

enum Stage {
    Downloading(TrackedPromise<Result<(SoundContainer, usize)>>),
    Decoding(StreamInfo, TrackedPromise<Result<(Decoded, Arc<[u8]>)>>),
}

struct Loading {
    row_id: u32,
    name: String,
    path: String,
    stage: Stage,
}

impl Loading {
    fn phase(&self) -> &'static str {
        match self.stage {
            Stage::Downloading(_) => "Downloading",
            Stage::Decoding(..) => "Decoding",
        }
    }
}

struct NowPlaying {
    name: String,
    path: String,
    row_id: u32,
    channels: u16,
    sample_rate: u32,
    loop_range_secs: Option<(f64, f64)>,
    info: StreamInfo,
    stream: Arc<[u8]>,
}

struct TrackRow {
    row_id: u32,
    path: String,
    name: String,
    available: bool,
}

pub struct MusicPlayer {
    player: Option<Player>,
    index: Index,
    avail: Avail,
    songs: HashMap<u32, SongInfo>,
    songs_load: Songs,
    songs_lang: Option<Language>,
    loading: Option<Loading>,
    now_playing: Option<NowPlaying>,
    pending: Option<u32>,
    volume: f32,
    search: String,
    show_unavailable: bool,
    show_visualizer: bool,
    rows: Vec<TrackRow>,
    rows_stale: bool,
    matcher: FuzzyMatcher,
    palette: Option<Palette>,
    nav: ListNav,
    scrub: Option<f64>,
    viz: Vec<f32>,
    export_promise: Option<TrackedPromise<()>>,
}

impl Default for MusicPlayer {
    fn default() -> Self {
        Self {
            player: None,
            index: Index::Idle,
            avail: Avail::Idle,
            songs: HashMap::new(),
            songs_load: Songs::Idle,
            songs_lang: None,
            loading: None,
            now_playing: None,
            pending: None,
            volume: 1.0,
            search: String::new(),
            show_unavailable: false,
            show_visualizer: true,
            rows: Vec::new(),
            rows_stale: true,
            matcher: FuzzyMatcher::new(),
            palette: None,
            nav: ListNav::default(),
            scrub: None,
            viz: Vec::new(),
            export_promise: None,
        }
    }
}

pub enum Action {
    /// A track was picked from the list or the palette; reflect it in the URL.
    Select(u32),
    /// The now-playing panel's path was followed to the Assets tab.
    Navigate(String),
}

enum Cmd {
    Toggle,
    Scrub(f64),
    Seek(f64),
    Volume(f32),
    ToggleVisualizer,
    OpenInAssets,
}

impl MusicPlayer {
    /// The track playing, or the one about to be, so that entering the bare route restores the
    /// same URL either way.
    pub fn now_playing_row(&self) -> Option<u32> {
        self.now_playing
            .as_ref()
            .map(|track| track.row_id)
            .or_else(|| self.loading.as_ref().map(|track| track.row_id))
            .or(self.pending)
    }

    pub fn name_of(&self, row_id: u32) -> Option<&str> {
        self.now_playing
            .as_ref()
            .filter(|track| track.row_id == row_id)
            .map(|track| track.name.as_str())
    }

    pub fn request(&mut self, row_id: u32) {
        self.pending = Some(row_id);
    }

    /// Drop everything that came from the install, so a reconnect reads it all again.
    ///
    /// The track playing is stopped here rather than left for `begin_load`, which never runs if the
    /// new install has no such row, and re-armed as pending so it plays again from the new one.
    pub fn reset(&mut self) {
        if let Some(player) = &mut self.player {
            player.stop();
        }
        self.index = Index::Idle;
        self.avail = Avail::Idle;
        self.rows_stale = true;
        // Both have to be cleared before the re-arm: `poll` drops a pending row that is already
        // loading or playing, which after a reset is exactly the row that has to be fetched again.
        self.pending = self
            .loading
            .take()
            .map(|track| track.row_id)
            .or(self.now_playing.take().map(|track| track.row_id))
            .or(self.pending);
    }

    pub fn open_palette(&mut self) {
        self.palette = Some(Palette::new("Find Track…", "Filter", self.search.clone()));
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, backend: &Backend) -> Option<Action> {
        self.poll(backend, &api_base(ui.ctx()), LANGUAGE.get(ui.ctx()));
        if let Some(player) = &mut self.player {
            player.take_media_action();
        }
        if self.rows_stale {
            self.rebuild_rows();
        }

        let playing = self.player.as_ref().is_some_and(Player::is_playing);
        if playing && self.show_visualizer && self.now_playing.is_some() {
            ui.ctx().request_repaint();
        } else if playing || self.loading.is_some() {
            ui.ctx().request_repaint_after(Duration::from_millis(100));
        }

        let picked = self.draw_palette(ui.ctx());
        let listed = matches!(self.index, Index::Loaded(_))
            && !CollapsibleSidePanel::is_collapsed(ui.ctx(), "music_list");
        self.nav
            .claim(ui.ctx(), listed, Some(egui::Id::new(FILTER_ID)));

        let clicked = self.side_panel(ui);
        let navigate = self.now_playing_panel(ui);
        picked
            .or(clicked)
            .map(Action::Select)
            .or_else(|| navigate.map(Action::Navigate))
    }

    fn draw_palette(&mut self, ctx: &egui::Context) -> Option<u32> {
        let palette = self.palette.take()?;
        match palette.draw(ctx, |query| {
            self.search = query.to_owned();
            self.matches()
                .into_iter()
                .take(SUGGESTIONS)
                .map(|row| (row.row_id, row.name.clone()))
                .collect()
        }) {
            Ok(picked) => picked,
            Err(palette) => {
                self.palette = Some(palette);
                None
            }
        }
    }

    fn poll(&mut self, backend: &Backend, api_url: &str, lang: Language) {
        if matches!(self.index, Index::Idle) {
            let excel = backend.excel().clone();
            self.index = Index::Loading(TrackedPromise::spawn_local(async move {
                load_index(excel).await
            }));
        }
        if matches!(&self.index, Index::Loading(p) if p.try_get().is_some()) {
            let Index::Loading(promise) = std::mem::replace(&mut self.index, Index::Idle) else {
                unreachable!()
            };
            self.index = match promise.block_and_take() {
                Ok(tracks) => Index::Loaded(tracks),
                Err(error) => Index::Failed(error.to_string()),
            };
            self.rows_stale = true;
        }

        if self.songs_lang != Some(lang) && !matches!(self.songs_load, Songs::Loading(_)) {
            self.songs_lang = Some(lang);
            let url = format!("{api_url}/songs/{}/", song_sheet(lang));
            self.songs_load = Songs::Loading(TrackedPromise::spawn_local(async move {
                Ok(serde_json::from_str(&fetch_url_str(url).await?)?)
            }));
        }
        if matches!(&self.songs_load, Songs::Loading(p) if p.try_get().is_some()) {
            let Songs::Loading(promise) = std::mem::replace(&mut self.songs_load, Songs::Idle)
            else {
                unreachable!()
            };
            match promise.block_and_take() {
                Ok(songs) => {
                    self.songs = songs;
                    self.rows_stale = true;
                }
                Err(error) => log::warn!("BGM song list unavailable, using file names: {error}"),
            }
            self.songs_load = Songs::Done;
        }

        if matches!(self.avail, Avail::Idle)
            && let Index::Loaded(tracks) = &self.index
        {
            let files = backend.files().clone();
            let paths: Vec<String> = tracks.iter().map(|track| track.path.clone()).collect();
            self.avail = Avail::Loading(TrackedPromise::spawn_local(async move {
                check_availability(files, paths).await
            }));
        }
        if matches!(&self.avail, Avail::Loading(p) if p.try_get().is_some()) {
            let Avail::Loading(promise) = std::mem::replace(&mut self.avail, Avail::Idle) else {
                unreachable!()
            };
            self.avail = match promise.block_and_take() {
                Ok(available) => Avail::Ready(available),
                Err(_) => Avail::Failed,
            };
            self.rows_stale = true;
        }

        self.poll_loading();

        if let Some(row_id) = self.pending {
            let active = self.now_playing.as_ref().map(|n| n.row_id) == Some(row_id)
                || self.loading.as_ref().map(|l| l.row_id) == Some(row_id);
            if active {
                self.pending = None;
            } else if let Index::Loaded(tracks) = &self.index {
                let path = tracks
                    .iter()
                    .find(|t| t.row_id == row_id)
                    .map(|t| t.path.clone());
                self.pending = None;
                if let Some(path) = path {
                    self.begin_load(backend, row_id, path);
                }
            }
        }
    }

    fn poll_loading(&mut self) {
        let ready = match &self.loading {
            Some(l) => match &l.stage {
                Stage::Downloading(p) => p.try_get().is_some(),
                Stage::Decoding(_, p) => p.try_get().is_some(),
            },
            None => return,
        };
        if !ready {
            return;
        }
        let Loading {
            row_id,
            name,
            path,
            stage,
        } = self.loading.take().unwrap();
        match stage {
            Stage::Downloading(promise) => match promise.block_and_take() {
                Ok((container, file_size)) => {
                    let Some(info) = stream_info(&container, file_size) else {
                        log::error!("no audio streams in {path}");
                        return;
                    };
                    let decode = TrackedPromise::spawn_local(async move {
                        let entry = container
                            .entries()
                            .first()
                            .ok_or_else(|| anyhow!("no audio streams"))?;
                        let stream: Arc<[u8]> = Arc::from(entry.data().as_slice());
                        Ok((audio::decode(entry)?, stream))
                    });
                    self.loading = Some(Loading {
                        row_id,
                        name,
                        path,
                        stage: Stage::Decoding(info, decode),
                    });
                }
                Err(error) => log::error!("BGM download failed: {error}"),
            },
            Stage::Decoding(info, promise) => match promise.block_and_take() {
                Ok((decoded, stream)) => self.start(row_id, name, path, info, decoded, stream),
                Err(error) => log::error!("BGM decode failed: {error}"),
            },
        }
    }

    fn title(&self, row_id: u32, path: &str) -> String {
        self.songs
            .get(&row_id)
            .filter(|song| !song.title.is_empty())
            .map_or_else(|| file_stem(path), |song| song.title.clone())
    }

    fn begin_load(&mut self, backend: &Backend, row_id: u32, path: String) {
        if !self.ensure_player() {
            return;
        }
        if let Some(player) = &mut self.player {
            player.unlock();
            player.stop();
        }
        self.now_playing = None;

        let name = self.title(row_id, &path);
        let files = backend.files().clone();
        let fetch_path = path.clone();
        let promise = TrackedPromise::spawn_local(async move {
            let bytes = files.file::<Vec<u8>>(&fetch_path).await?;
            let file_size = bytes.len();
            let container = SoundContainer::read(Cursor::new(bytes))?;
            Ok((container, file_size))
        });
        self.loading = Some(Loading {
            row_id,
            name,
            path,
            stage: Stage::Downloading(promise),
        });
    }

    fn start(
        &mut self,
        row_id: u32,
        name: String,
        path: String,
        info: StreamInfo,
        decoded: Decoded,
        stream: Arc<[u8]>,
    ) {
        if !self.ensure_player() {
            return;
        }
        let rate = f64::from(decoded.sample_rate);
        // OS media-session integration is only for a track that actually loops; a fanfare-length
        // BGM entry with no real loop region gets no more OS surface than an Assets tab preview.
        let announce = decoded
            .loop_start
            .zip(decoded.loop_end)
            .is_some_and(|(start, end)| end > start);
        let now_playing = NowPlaying {
            name: name.clone(),
            path,
            row_id,
            channels: decoded.channels,
            sample_rate: decoded.sample_rate,
            loop_range_secs: decoded
                .loop_start
                .zip(decoded.loop_end)
                .map(|(start, end)| (f64::from(start) / rate, f64::from(end) / rate)),
            info,
            stream,
        };
        let player = self.player.as_mut().unwrap();
        player.set_volume(self.volume);
        if let Err(error) = player.play(decoded, announce) {
            log::error!("BGM playback failed: {error}");
            return;
        }
        player.set_metadata(&name);
        self.now_playing = Some(now_playing);
    }

    fn ensure_player(&mut self) -> bool {
        if self.player.is_none() {
            match Player::new() {
                Ok(player) => self.player = Some(player),
                Err(error) => {
                    log::error!("audio init failed: {error}");
                    return false;
                }
            }
        }
        true
    }

    fn rebuild_rows(&mut self) {
        let Index::Loaded(tracks) = &self.index else {
            return;
        };
        let rows = tracks
            .iter()
            .map(|track| {
                let name = self
                    .songs
                    .get(&track.row_id)
                    .filter(|song| !song.title.is_empty())
                    .map_or_else(|| file_stem(&track.path), |song| song.title.clone());
                let available = match &self.avail {
                    Avail::Ready(set) => set.contains(&track.path),
                    _ => true,
                };
                TrackRow {
                    row_id: track.row_id,
                    path: track.path.clone(),
                    name,
                    available,
                }
            })
            .collect();
        self.rows = rows;
        self.rows_stale = false;
    }

    fn side_panel(&mut self, ui: &mut egui::Ui) -> Option<u32> {
        let mut clicked = None;
        let mut nav = std::mem::take(&mut self.nav);
        CollapsibleSidePanel::new("music_list", Side::Left)
            .min_width(LIST_MIN_WIDTH)
            .max_width(LIST_WIDTH)
            .show(ui, |ui, is_open| {
                if !is_open {
                    return;
                }
                Panel::top("music_list_header").show(ui, |ui| {
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.with_layout(Layout::right_to_left(Align::Min), |ui| {
                            CollapsibleSidePanel::draw_arrow(ui, "music_list", Side::Left);
                            ui.vertical_centered_justified(|ui| ui.heading("Tracks"));
                        });
                    });
                    ui.add_space(4.0);
                    ui.with_layout(Layout::right_to_left(Align::Min), |ui| {
                        if ui
                            .add_enabled(!self.search.is_empty(), Button::new("↩"))
                            .on_hover_text("Clear")
                            .clicked()
                        {
                            self.search.clear();
                        }
                        let unavailable = self.rows.iter().filter(|row| !row.available).count();
                        if unavailable > 0 {
                            ui.toggle_value(&mut self.show_unavailable, "🚫")
                                .on_hover_text(format!("Show {unavailable} unavailable"));
                        }
                        ui.add_sized(
                            Vec2::new(ui.available_width(), 0.0),
                            TextEdit::singleline(&mut self.search)
                                .id(egui::Id::new(FILTER_ID))
                                .hint_text("Filter"),
                        );
                    });
                    ui.add_space(4.0);
                });

                CentralPanel::default().show(ui, |ui| match &self.index {
                    Index::Idle | Index::Loading(_) => {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label("Loading BGM list…");
                        });
                    }
                    Index::Failed(error) => {
                        ui.colored_label(Color32::RED, format!("Failed to load BGM list: {error}"));
                    }
                    Index::Loaded(_) => clicked = self.draw_rows(ui, &mut nav),
                });
            });
        self.nav = nav;
        clicked
    }

    /// The tracks the filter leaves, best match first.
    fn matches(&self) -> Vec<&TrackRow> {
        let query = (!self.search.is_empty()).then_some(self.search.as_str());
        self.matcher.match_list_indirect(
            query,
            self.rows
                .iter()
                .filter(|row| self.show_unavailable || row.available),
            |row| row.name.as_str(),
        )
    }

    fn draw_rows(&self, ui: &mut egui::Ui, nav: &mut ListNav) -> Option<u32> {
        let selected = self
            .now_playing
            .as_ref()
            .map(|n| n.row_id)
            .or_else(|| self.loading.as_ref().map(|l| l.row_id));
        let filtered = self.matches();

        let mut clicked = nav
            .apply(filtered.len())
            .filter(|at| filtered[*at].available)
            .map(|at| filtered[at].row_id);
        let row_height = ui.text_style_height(&egui::TextStyle::Button);
        let mut area = ScrollArea::vertical().auto_shrink(false);
        if let Some(offset) = nav.scroll(ui, row_height, filtered.len()) {
            area = area.vertical_scroll_offset(offset);
        }
        let output = area.show_rows(ui, row_height, filtered.len(), |ui, range| {
            ui.with_layout(Layout::top_down_justified(Align::Min), |ui| {
                for (at, row) in filtered[range.clone()].iter().enumerate() {
                    ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Truncate);
                    let response = ui
                        .add_enabled_ui(row.available, |ui| {
                            Button::selectable(selected == Some(row.row_id), row.name.as_str())
                                .ui(ui)
                        })
                        .inner
                        .on_hover_ui(|ui| self.row_hover(ui, row));
                    nav.mark(ui, range.start + at, response.rect);
                    if response.clicked() {
                        clicked = Some(row.row_id);
                    }
                }
            });
        });
        nav.seen(&output);
        clicked
    }

    fn row_hover(&self, ui: &mut egui::Ui, row: &TrackRow) {
        ui.strong(&row.name);
        if let Some(song) = self.songs.get(&row.row_id) {
            if !song.alt.is_empty() {
                ui.label(format!("Also known as: {}", song.alt));
            }
            if !song.special.is_empty() {
                ui.label(format!("Special mode: {}", song.special));
            }
            if !song.locations.is_empty() {
                ui.label(format!("Locations: {}", song.locations));
            }
            if !song.info.is_empty() {
                ui.label(format!("Notes: {}", song.info));
            }
            if song.duration > 0 {
                ui.label(format!(
                    "Duration: {}",
                    format_time(f64::from(song.duration))
                ));
            }
        }
        ui.separator();
        ui.label(RichText::new(&row.path).weak());
        ui.label(RichText::new(format!("BGM #{}", row.row_id)).weak());
        if !row.available {
            ui.colored_label(
                Color32::from_rgb(0xE0, 0x8C, 0x3C),
                "Not available on this data source",
            );
        }
    }

    fn now_playing_panel(&mut self, ui: &mut egui::Ui) -> Option<String> {
        let mut navigate = None;
        CentralPanel::default().show(ui, |ui| {
            if CollapsibleSidePanel::is_collapsed(ui.ctx(), "music_list") {
                Panel::top("music_reexpand").show(ui, |ui| {
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        CollapsibleSidePanel::draw_arrow(ui, "music_list", Side::Left)
                    });
                    ui.add_space(4.0);
                });
            }
            if self.now_playing.is_some() {
                navigate = self.draw_player(ui);
            } else if let Some(loading) = &self.loading {
                let phase = loading.phase();
                let name = loading.name.clone();
                center(ui, |ui| {
                    ui.spinner();
                    ui.add_space(8.0);
                    ui.label(RichText::new(format!("{phase}…")).heading());
                    ui.label(RichText::new(name).weak());
                });
            } else {
                empty_view(ui, "♪", "Select a track to play");
            }
        });
        navigate
    }

    fn draw_player(&mut self, ui: &mut egui::Ui) -> Option<String> {
        self.export_promise
            .take_if(|promise| promise.try_get().is_some());
        let now = self.now_playing.as_ref().unwrap();
        let (name, path, loop_range, channels, sample_rate, info, row_id, stream) = (
            now.name.clone(),
            now.path.clone(),
            now.loop_range_secs,
            now.channels,
            now.sample_rate,
            now.info,
            now.row_id,
            now.stream.clone(),
        );
        let locations = self
            .songs
            .get(&row_id)
            .filter(|song| !song.locations.is_empty())
            .map(|song| song.locations.clone());
        let playing = self.player.as_ref().is_some_and(Player::is_playing);
        let (position, duration) = self
            .player
            .as_ref()
            .map_or((0.0, 0.0), |player| (player.position(), player.duration()));
        let bar_position = self.scrub.unwrap_or(position);
        let mut volume = self.volume;
        let show_viz = self.show_visualizer;
        let exporting = self.export_promise.is_some();

        let mut spectrum = [0u8; 4096];
        if show_viz && let Some(player) = &self.player {
            player.spectrum(&mut spectrum);
        }
        let bars = if show_viz {
            self.viz_bars(&spectrum, sample_rate, playing)
        } else {
            Vec::new()
        };

        let outer = ui.available_rect_before_wrap();
        let col_w = outer.width().min(600.0);
        let col_rect = Rect::from_min_size(
            pos2(outer.center().x - col_w / 2.0, outer.top()),
            vec2(col_w, outer.height()),
        );
        let mut cmd = None;
        let mut export_start = None;
        ui.scope_builder(
            UiBuilder::new()
                .max_rect(col_rect)
                .layout(Layout::top_down(Align::Center)),
            |ui| {
                let sp = ui.spacing().item_spacing.x;

                ui.add_space(16.0);
                if show_viz {
                    draw_visualizer(ui, &bars, 176.0);
                    ui.add_space(18.0);
                }

                ui.label(RichText::new(&name).size(26.0).strong());
                if let Some(locations) = &locations {
                    ui.label(RichText::new(locations).weak());
                }
                ui.add_space(18.0);

                ui.horizontal(|ui| {
                    ui.label(RichText::new(format_time(bar_position)).weak());
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        ui.label(RichText::new(format_time(duration)).weak());
                        ui.spacing_mut().slider_width = (ui.available_width() - sp).max(80.0);
                        let mut seek = bar_position;
                        let response = ui.add_enabled(
                            duration > 0.0,
                            Slider::new(&mut seek, 0.0..=duration.max(0.001)).show_value(false),
                        );
                        if response.dragged() {
                            cmd = Some(Cmd::Scrub(seek));
                        } else if response.drag_stopped() || response.changed() {
                            cmd = Some(Cmd::Seek(seek));
                        }
                    });
                });
                ui.add_space(14.0);

                ui.horizontal(|ui| {
                    if ui
                        .add_sized(
                            vec2(40.0, 34.0),
                            Button::new(RichText::new(if playing { "⏸" } else { "▶" }).size(19.0)),
                        )
                        .clicked()
                    {
                        cmd = Some(Cmd::Toggle);
                    }
                    if ui
                        .add_sized(vec2(40.0, 34.0), Button::selectable(show_viz, "📊"))
                        .on_hover_text("Visualizer")
                        .clicked()
                    {
                        cmd = Some(Cmd::ToggleVisualizer);
                    }
                    // `Original` hands back the entry's own bytes untouched, multichannel and all;
                    // `Wav` re-decodes them, so it carries the same stereo downmix the player is
                    // producing.
                    let codec = info.codec;
                    let extension = codec_extension(codec);
                    export_start = export::menu(
                        ui,
                        "📤",
                        Some("Export"),
                        exporting,
                        vec![
                            export::Choice::bytes(
                                "Original file",
                                format!("{}.{extension}", safe_file_name(&name)),
                                {
                                    let stream = stream.clone();
                                    move || Ok(stream.to_vec())
                                },
                            )
                            .title("Export Original Audio")
                            .filter(extension.to_ascii_uppercase(), &[extension]),
                            export::Choice::bytes(
                                "WAV",
                                format!("{}.wav", safe_file_name(&name)),
                                move || {
                                    audio::decode_data(codec, &stream)
                                        .and_then(|decoded| audio::encode_wav(&decoded))
                                },
                            )
                            .title("Export WAV Audio")
                            .filter("WAV", &["wav"]),
                        ],
                        vec2(40.0, 34.0),
                    );

                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        ui.spacing_mut().slider_width = 150.0;
                        if ui
                            .add(Slider::new(&mut volume, 0.0..=1.0).show_value(false))
                            .changed()
                        {
                            cmd = Some(Cmd::Volume(volume));
                        }
                        ui.label("🔊");
                    });
                });
                ui.add_space(18.0);

                if draw_info(
                    ui,
                    &info,
                    channels,
                    sample_rate,
                    duration,
                    loop_range,
                    &path,
                ) {
                    cmd = Some(Cmd::OpenInAssets);
                }
            },
        );

        if export_start.is_some() {
            self.export_promise = export_start;
        }

        match cmd {
            Some(Cmd::Toggle) => {
                if let Some(player) = &self.player {
                    if playing {
                        player.pause();
                    } else {
                        player.resume();
                    }
                }
                None
            }
            Some(Cmd::Scrub(seconds)) => {
                self.scrub = Some(seconds);
                None
            }
            Some(Cmd::Seek(seconds)) => {
                if let Some(player) = &mut self.player {
                    player.seek(seconds);
                }
                self.scrub = None;
                None
            }
            Some(Cmd::Volume(value)) => {
                self.volume = value;
                if let Some(player) = &mut self.player {
                    player.set_volume(value);
                }
                None
            }
            Some(Cmd::ToggleVisualizer) => {
                self.show_visualizer = !self.show_visualizer;
                None
            }
            Some(Cmd::OpenInAssets) => Some(path),
            None => None,
        }
    }

    fn viz_bars(&mut self, spectrum: &[u8], sample_rate: u32, playing: bool) -> Vec<f32> {
        if self.viz.len() != VIZ_BARS {
            self.viz = vec![0.0; VIZ_BARS];
        }
        if !playing {
            return self.viz.clone();
        }
        let bins = spectrum.len().max(1);
        let nyquist = (sample_rate as f32 / 2.0).max(1.0);
        let f_min = 40.0;
        let f_max = nyquist.min(16_000.0);
        let ratio = f_max / f_min;
        let bin_of = |freq: f32| ((freq / nyquist) * bins as f32).round() as usize;
        for (i, bar) in self.viz.iter_mut().enumerate() {
            let lo = bin_of(f_min * ratio.powf(i as f32 / VIZ_BARS as f32)).min(bins - 1);
            let hi =
                bin_of(f_min * ratio.powf((i + 1) as f32 / VIZ_BARS as f32)).clamp(lo + 1, bins);
            let peak = spectrum[lo..hi].iter().copied().max().unwrap_or(0) as f32 / 255.0;
            let target = peak.powf(1.4);
            let rate = if target > *bar { 0.55 } else { 0.16 };
            *bar += (target - *bar) * rate;
        }
        self.viz.clone()
    }
}

const VIZ_BARS: usize = 64;

fn draw_visualizer(ui: &mut egui::Ui, bars: &[f32], height: f32) {
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), height), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 10.0, ui.visuals().extreme_bg_color);

    let inner = rect.shrink(12.0);
    let n = bars.len().max(1);
    let gap = 3.0;
    let bar_w = ((inner.width() - gap * (n as f32 - 1.0)) / n as f32).max(1.0);
    let accent = ui.visuals().selection.bg_fill;
    for (i, &m) in bars.iter().enumerate() {
        let m = m.clamp(0.0, 1.0);
        let bar_h = (m * inner.height()).max(2.0);
        let x = inner.left() + i as f32 * (bar_w + gap);
        let bar = Rect::from_min_max(
            pos2(x, inner.bottom() - bar_h),
            pos2(x + bar_w, inner.bottom()),
        );
        painter.rect_filled(bar, 2.0, accent.gamma_multiply(0.45 + 0.55 * m));
    }
}

fn draw_info(
    ui: &mut egui::Ui,
    info: &StreamInfo,
    channels: u16,
    sample_rate: u32,
    duration: f64,
    loop_range: Option<(f64, f64)>,
    path: &str,
) -> bool {
    let looping = loop_range.is_some();
    let bitrate = if duration > 0.0 {
        (info.stream_size as f64 * 8.0 / duration / 1000.0).round() as u64
    } else {
        0
    };
    let freq = if sample_rate.is_multiple_of(1000) {
        format!("{} kHz", sample_rate / 1000)
    } else {
        format!("{:.1} kHz", f64::from(sample_rate) / 1000.0)
    };
    let chan = match channels {
        1 => "Mono".to_string(),
        2 => "Stereo".to_string(),
        6 => "5.1".to_string(),
        n => format!("{n} ch"),
    };
    let sep = "   ·   ";
    let line1 = [codec_name(info.codec).to_string(), freq, chan].join(sep);
    let mut parts = vec![format!("{bitrate} kbps"), format_size(info.file_size)];
    if looping {
        parts.push("Looping".to_string());
    }
    let line2 = parts.join(sep);

    ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
    ui.label(RichText::new(line1).weak());
    let stats = ui.label(RichText::new(line2).weak());
    if let Some((start, end)) = loop_range {
        stats.on_hover_text(format!(
            "Loops {} → {}",
            format_time(start),
            format_time(end)
        ));
    }
    ui.add_space(4.0);
    crate::assets::viewers::link(ui, file_name(path), path)
}

fn codec_name(codec: Codec) -> &'static str {
    match codec {
        Codec::OggVorbis => "Ogg Vorbis",
        Codec::Hca => "HCA",
        Codec::Mp3 => "MP3",
        Codec::MsAdpcm => "MS ADPCM",
        Codec::Atrac9 => "ATRAC9",
        Codec::Pcm => "PCM",
        Codec::Empty => "Empty",
        Codec::Unknown(_) => "Unknown",
    }
}

/// Ogg and Hca are the only codecs `audio::decode` ever hands back a `NowPlaying` for, so
/// anything else here is unreachable; the fallback stays honest rather than naming a container
/// the raw entry bytes do not actually have.
fn codec_extension(codec: Codec) -> &'static str {
    match codec {
        Codec::OggVorbis => "ogg",
        Codec::Hca => "hca",
        Codec::MsAdpcm => "wav",
        _ => "bin",
    }
}

fn safe_file_name(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|character| {
            if "<>:\"/\\|?*".contains(character) {
                '_'
            } else {
                character
            }
        })
        .collect();
    let trimmed = cleaned.trim().trim_end_matches(['.', ' ']);
    if trimmed.is_empty() {
        "music".to_string()
    } else {
        trimmed.to_string()
    }
}

fn format_size(bytes: usize) -> String {
    let bytes = bytes as f64;
    if bytes >= 1_048_576.0 {
        format!("{:.1} MB", bytes / 1_048_576.0)
    } else if bytes >= 1024.0 {
        format!("{:.0} KB", bytes / 1024.0)
    } else {
        format!("{bytes:.0} B")
    }
}

fn stream_info(container: &SoundContainer, file_size: usize) -> Option<StreamInfo> {
    let entry = container.entries().first()?;
    Some(StreamInfo {
        codec: entry.format(),
        file_size,
        stream_size: entry.data().len(),
    })
}

fn format_time(seconds: f64) -> String {
    if !seconds.is_finite() || seconds < 0.0 {
        return "0:00".to_string();
    }
    let total = seconds as u64;
    format!("{}:{:02}", total / 60, total % 60)
}

fn file_stem(path: &str) -> String {
    path.rsplit('/')
        .next()
        .unwrap_or(path)
        .trim_end_matches(".scd")
        .to_string()
}

fn song_sheet(lang: Language) -> &'static str {
    match lang {
        Language::Japanese => "ja",
        Language::French => "fr",
        Language::German => "de",
        Language::ChineseSimplified | Language::ChineseTraditional | Language::TaiwanChinese => {
            "zh"
        }
        _ => "en",
    }
}

async fn load_index(excel: CachedProvider) -> Result<Vec<BgmTrack>> {
    let sheet = excel.get_sheet("BGM", Language::None).await?;
    let offset = u32::from(
        sheet
            .columns()
            .first()
            .ok_or_else(|| anyhow!("BGM sheet has no columns"))?
            .offset(),
    );

    let mut tracks = Vec::new();
    for row_id in sheet.get_row_ids() {
        let Ok(row) = sheet.get_row(row_id) else {
            continue;
        };
        let Ok(cell) = row.read_string(offset) else {
            continue;
        };
        let path = cell.format().to_string();
        if path.ends_with(".scd") {
            tracks.push(BgmTrack { row_id, path });
        }
    }
    Ok(tracks)
}

async fn check_availability(
    files: Rc<dyn FileProvider>,
    paths: Vec<String>,
) -> Result<HashSet<String>> {
    let mut available = HashSet::with_capacity(paths.len());
    for chunk in paths.chunks(100) {
        let exists = files.exists_many(chunk).await?;
        for (path, ok) in chunk.iter().zip(exists) {
            if ok {
                available.insert(path.clone());
            }
        }
    }
    Ok(available)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `poll` drops a pending row that is already playing, so re-arming the track before clearing
    /// it would leave the previous install's audio playing and never fetch it again.
    #[test]
    fn reconnecting_rearms_the_playing_track() {
        let mut music = MusicPlayer {
            index: Index::Loaded(Vec::new()),
            avail: Avail::Ready(HashSet::new()),
            now_playing: Some(NowPlaying {
                name: "Prelude".to_string(),
                path: "music/ffxiv/bgm_system_title.scd".to_string(),
                row_id: 7,
                channels: 2,
                sample_rate: 44100,
                loop_range_secs: None,
                info: StreamInfo {
                    codec: Codec::OggVorbis,
                    file_size: 0,
                    stream_size: 0,
                },
                stream: Arc::from(&[][..]),
            }),
            ..Default::default()
        };

        music.reset();

        assert_eq!(
            music.pending,
            Some(7),
            "the track has to be read again from the new install"
        );
        assert!(
            music.now_playing.is_none(),
            "or poll sees the row as already playing and drops it"
        );
        assert!(
            matches!(music.index, Index::Idle),
            "the BGM sheet is per version"
        );
        assert!(matches!(music.avail, Avail::Idle), "as is which files ship");
    }
}
