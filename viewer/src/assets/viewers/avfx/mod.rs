//! `.avfx` effects: what the game spawns, over what span of frames, and the curves driving it.
//!
//! An effect is a tree of tagged blocks. Its own settings are named, and everything below them goes
//! by the four-character tag it is written under, so the tree here is the file as it stands rather
//! than a reading of it. What is read is the shape: schedulers start timelines, a timeline runs
//! items over a span of frames, an emitter spawns particles and further emitters, and anything
//! animated is a curve.
//!
//! Nothing in the file states the rate its frames are counted at, so the rate is a setting: it sets
//! how fast the preview runs, and a curve reads out in frames and in seconds both.
//!
//! The preview itself is [`sim`]: an emitter bursting particles, each carrying its curves forward
//! from the frame it was spawned on. It is not the game's renderer and does not try to be.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Cursor;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use egui::{
    Color32, RichText, ScrollArea, Sense, TextureHandle, TextureOptions, Vec2,
    collapsing_header::paint_default_icon, vec2,
};
use glam::{Mat4, Vec3, Vec4};
use ironworks::file::{
    File,
    avfx::{Avfx, Block, Clip, Item, Model, Payload},
};

use super::{Preview, facts, headers, heading, link, section};
use crate::backend::Backend;
use crate::data::DecodedTexture;
use crate::utils::TrackedPromise;
use crate::{settings::AVFX_FRAME_RATE, utils::file_name};

mod curve;
pub(crate) mod gpu;
pub(crate) mod program;
pub(crate) mod sim;

use curve::Curve;

/// Vertical field of view.
const FOV: f32 = 40.0_f32.to_radians();

/// How much of the effect's reach the opening view stands back by.
const MARGIN: f32 = 1.6;

/// Longest edge an effect's textures are decoded to, and the bytes one effect may hold of them.
const TEXTURE_SIZE: u16 = 256;
const TEXTURE_BUDGET: usize = 64 << 20;

/// Models one definition may draw at once. The model package reads one transform at a time, so each
/// is a draw call of its own and an effect spawning thousands would take a browser tab with it.
const MODELS: usize = 512;

enum Texture {
    Fetching(TrackedPromise<Result<DecodedTexture>>),
    Ready(TextureHandle),
    Absent,
}

/// A shader package, from the moment it is asked for to the moment it can be translated.
pub(crate) enum Package {
    Fetching(TrackedPromise<Result<Vec<u8>>>),
    Ready(Vec<u8>),
    Failed,
}

/// Where the camera is looking from.
#[derive(Clone, Copy)]
struct Camera {
    yaw: f32,
    pitch: f32,
    distance: f32,
    target: Vec3,
}

impl Camera {
    fn eye(&self) -> Vec3 {
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        self.target + self.distance * Vec3::new(cos_pitch * sin_yaw, sin_pitch, cos_pitch * cos_yaw)
    }
}

/// Space each level of the tree is set in by.
const INDENT: f32 = 12.0;

/// Room the expander takes, kept on rows without one so their labels still line up.
const TRIANGLE: f32 = 12.0;

/// Side of the square a color is drawn in.
const CHIP: f32 = 10.0;

/// One row of the tree.
struct Row {
    depth: u8,

    label: String,

    /// What the row says beside its label, drawn weakly.
    detail: String,

    /// The file this row names, drawn as a link.
    asset: Option<String>,

    /// The curve this row draws, indexed into [`Rendered::curves`].
    curve: Option<usize>,

    /// Everything the row carries, for the rows the file gives a shape of their own. A block row
    /// says all it has to say in its label and detail, and leaves this empty.
    fields: Vec<(&'static str, String)>,

    /// Whether the row starts expanded.
    open: bool,
}

pub struct Rendered {
    identity: Vec<(&'static str, String)>,
    settings: Vec<(&'static str, String)>,
    rows: Vec<Row>,
    curves: Vec<Curve>,
    /// Where the open rows and the selected one are kept, since drawing takes the file by
    /// reference.
    state: egui::Id,

    effect: sim::Effect,
    gpu: Arc<Mutex<gpu::Particles>>,
    live: RefCell<sim::State>,
    /// Where playback has reached, which runs between frames where the rate is not the display's.
    at: Cell<f32>,
    playing: Cell<bool>,
    camera: Cell<Camera>,
    home: Camera,
    textures: RefCell<Textures>,
    shaders: RefCell<Shaders>,
}

/// The textures an effect samples, asked for once each and handed to egui as they arrive.
#[derive(Default)]
pub(crate) struct Textures {
    held: HashMap<String, Texture>,
    /// What the ones already handed over take of the budget.
    resident: usize,
}

impl Textures {
    /// Asks for whatever `wanted` names that is not in hand yet. Runs every frame; one already
    /// resolved costs a lookup.
    pub(crate) fn poll(&mut self, ctx: &egui::Context, backend: &Backend, wanted: &[String]) {
        let Self { held, resident } = self;
        for path in wanted {
            if held.contains_key(path) {
                continue;
            }
            if *resident >= TEXTURE_BUDGET {
                log::warn!("assets/avfx: {path}: past this effect's texture budget");
                held.insert(path.clone(), Texture::Absent);
                continue;
            }
            let files = backend.files().clone();
            let wanted = path.clone();
            held.insert(
                path.clone(),
                Texture::Fetching(TrackedPromise::spawn_local(async move {
                    files.read_texture(&wanted, Some(TEXTURE_SIZE)).await
                })),
            );
        }
        for (path, texture) in held.iter_mut() {
            let Texture::Fetching(promise) = texture else {
                continue;
            };
            let Some(result) = promise.try_get() else {
                continue;
            };
            *texture = match result {
                Ok(decoded) => {
                    let handled = crate::utils::tex_loader::fit(ctx, &decoded.image);
                    let size = [handled.width() as usize, handled.height() as usize];
                    *resident += size[0] * size[1] * 4;
                    let image = egui::ColorImage::from_rgba_premultiplied(
                        size,
                        handled.as_flat_samples().as_slice(),
                    );
                    Texture::Ready(ctx.load_texture(
                        format!("avfx:{path}"),
                        image,
                        TextureOptions {
                            magnification: egui::TextureFilter::Linear,
                            minification: egui::TextureFilter::Linear,
                            wrap_mode: egui::TextureWrapMode::ClampToEdge,
                            mipmap_mode: Some(egui::TextureFilter::Linear),
                        },
                    ))
                }
                Err(why) => {
                    log::error!("assets/avfx: {path}: {why}");
                    Texture::Absent
                }
            };
        }
    }

    /// The handle bound to each of `wanted`, or `None` where one has not arrived: held back rather
    /// than drawn white, since an additive quad with no sampler reads as flat white.
    pub(crate) fn bound(&self, wanted: &[String]) -> Vec<Option<egui::TextureId>> {
        wanted
            .iter()
            .map(|path| match self.held.get(path) {
                Some(Texture::Ready(handle)) => Some(handle.id()),
                _ => None,
            })
            .collect()
    }
}

/// The two apricot packages every effect is shaded by, asked for once however many are drawn.
#[derive(Default)]
pub(crate) struct Shaders {
    held: HashMap<&'static str, Package>,
    resolved: Arc<gpu::Packages>,
}

impl Shaders {
    pub(crate) fn poll(&mut self, backend: &Backend) {
        for path in [program::SHAPE, program::MODEL] {
            self.held.entry(path).or_insert_with(|| {
                let files = backend.files().clone();
                Package::Fetching(TrackedPromise::spawn_local(async move {
                    files.read(path).await
                }))
            });
        }
        let mut arrived = false;
        for (path, package) in self.held.iter_mut() {
            let Package::Fetching(promise) = package else {
                continue;
            };
            let Some(result) = promise.try_get() else {
                continue;
            };
            arrived = true;
            *package = match result {
                Ok(bytes) => Package::Ready(bytes.clone()),
                Err(why) => {
                    log::error!("assets/avfx: {path}: {why}");
                    Package::Failed
                }
            };
        }
        if !arrived {
            return;
        }
        let held = |path| match self.held.get(path) {
            Some(Package::Ready(bytes)) => Some(bytes.clone()),
            _ => None,
        };
        self.resolved = Arc::new(gpu::Packages {
            shape: held(program::SHAPE),
            model: held(program::MODEL),
        });
    }

    pub(crate) fn resolved(&self) -> Arc<gpu::Packages> {
        self.resolved.clone()
    }
}

/// The first block carrying `name`.
fn find<'a>(blocks: &'a [Block], name: &str) -> Option<&'a Block> {
    blocks.iter().find(|block| block.name() == name)
}

fn integer(blocks: &[Block], name: &str) -> Option<i32> {
    find(blocks, name)?.i32()
}

/// An index into one of the effect's lists, which is written `-1` where there is none.
fn reference(value: Option<i32>) -> String {
    match value {
        Some(-1) | None => "none".to_owned(),
        Some(index) => index.to_string(),
    }
}

/// What a row says about an entry that can be switched off.
fn disabled(blocks: &[Block], name: &str) -> &'static str {
    match find(blocks, name).and_then(Block::bool) {
        Some(false) => "  off",
        _ => "",
    }
}

/// A four-byte payload, read as whichever of the two the bits can be. An integer small enough for
/// this format to be writing lands in the exponent range a float leaves for zero, so a normal float
/// is a float and anything else is an integer.
fn scalar(bytes: [u8; 4]) -> String {
    let value = f32::from_le_bytes(bytes);
    match value.is_normal() {
        true => value.to_string(),
        false => i32::from_le_bytes(bytes).to_string(),
    }
}

/// A block's payload as it reads.
fn payload(block: &Block, bytes: &[u8]) -> String {
    let name = block.name();
    match (name.as_str(), bytes) {
        ("SdNm" | "Name", _) => block.text().unwrap_or_default(),
        (_, [byte]) => byte.to_string(),
        (_, [a, b, c, d]) => scalar([*a, *b, *c, *d]),
        (_, []) => String::new(),
        _ => format!("{} bytes", bytes.len()),
    }
}

fn axes(values: [f32; 3]) -> String {
    format!("{:.3}, {:.3}, {:.3}", values[0], values[1], values[2])
}

fn numbers<const N: usize>(values: [impl ToString; N]) -> String {
    values
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

#[derive(Default)]
struct Build {
    rows: Vec<Row>,
    curves: Vec<Curve>,
}

impl Build {
    fn row(
        &mut self,
        depth: usize,
        label: impl Into<String>,
        detail: impl Into<String>,
    ) -> &mut Row {
        self.rows.push(Row {
            depth: depth as u8,
            label: label.into(),
            detail: detail.into(),
            asset: None,
            curve: None,
            fields: Vec::new(),
            open: false,
        });
        self.rows.last_mut().unwrap()
    }

    /// One of the effect's lists, which is left out where the file holds none of it.
    fn list(&mut self, label: &str, count: usize) -> bool {
        if count > 0 {
            self.row(0, label, count.to_string()).open = true;
        }
        count > 0
    }

    /// One block and everything under it. A curve collapses into the row it sits on: the tags
    /// beside its keys say what happens either side of them, and the plot says both.
    fn block(&mut self, block: &Block, depth: usize) {
        if let Some(curve) = curve::read(block) {
            let (detail, index) = (curve.summary(), self.curves.len());
            self.curves.push(curve);
            self.row(depth, block.name().as_str(), detail).curve = Some(index);
            return;
        }

        match block.payload() {
            Payload::Blocks(blocks) => {
                self.row(depth, block.name().as_str(), String::new());
                for child in blocks {
                    self.block(child, depth + 1);
                }
            }
            Payload::Keys(keys) => {
                self.row(depth, block.name().as_str(), format!("{} keys", keys.len()));
            }
            Payload::Bytes(bytes) => {
                let value = payload(block, bytes);
                let row = self.row(depth, block.name().as_str(), value.clone());
                if block.name() == "SdNm" && !value.is_empty() {
                    row.detail = String::new();
                    row.asset = Some(value);
                }
            }
        }
    }

    /// One entry of a scheduler, timeline or emitter list.
    fn item(&mut self, label: String, detail: String, item: &Item, depth: usize) {
        self.row(depth, label, detail);
        for block in item.blocks() {
            self.block(block, depth + 1);
        }
    }

    /// The effect's own settings, which start collapsed: the panel already reads out the ones that
    /// are named.
    fn settings(&mut self, file: &Avfx) {
        if file.properties().is_empty() {
            return;
        }
        self.row(0, "Settings", file.properties().len().to_string());
        for block in file.properties() {
            self.block(block, 1);
        }
    }

    fn schedulers(&mut self, file: &Avfx) {
        if !self.list("Schedulers", file.schedulers().len()) {
            return;
        }
        for (index, scheduler) in file.schedulers().iter().enumerate() {
            let detail = format!(
                "{} items, {} triggers",
                scheduler.items().len(),
                scheduler.triggers().len()
            );
            self.row(1, format!("Scheduler {index}"), detail);
            for block in scheduler.properties() {
                self.block(block, 2);
            }
            for (label, items) in [
                ("Item", scheduler.items()),
                ("Trigger", scheduler.triggers()),
            ] {
                for (index, item) in items.iter().enumerate() {
                    let blocks = item.blocks();
                    let detail = format!(
                        "timeline {}  start {}{}",
                        reference(integer(blocks, "TlNo")),
                        reference(integer(blocks, "StTm")),
                        disabled(blocks, "bEna")
                    );
                    self.item(format!("{label} {index}"), detail, item, 2);
                }
            }
        }
    }

    fn timelines(&mut self, file: &Avfx) {
        if !self.list("Timelines", file.timelines().len()) {
            return;
        }
        for (index, timeline) in file.timelines().iter().enumerate() {
            let properties = timeline.properties();
            let detail = format!(
                "loop {}..{}  {} items",
                reference(integer(properties, "LpSt")),
                reference(integer(properties, "LpEd")),
                timeline.items().len()
            );
            self.row(1, format!("Timeline {index}"), detail).fields = vec![
                (
                    "Loop",
                    format!(
                        "{}..{}",
                        reference(integer(properties, "LpSt")),
                        reference(integer(properties, "LpEd"))
                    ),
                ),
                ("Binder", reference(integer(properties, "BnNo"))),
                ("Items", timeline.items().len().to_string()),
                ("Clips", timeline.clips().len().to_string()),
            ];
            for block in properties {
                self.block(block, 2);
            }
            for (index, item) in timeline.items().iter().enumerate() {
                let blocks = item.blocks();
                let detail = format!(
                    "{}..{}  emitter {}  binder {}  effector {}{}",
                    reference(integer(blocks, "StTm")),
                    reference(integer(blocks, "EdTm")),
                    reference(integer(blocks, "EmNo")),
                    reference(integer(blocks, "BdNo")),
                    reference(integer(blocks, "EfNo")),
                    disabled(blocks, "bEna")
                );
                self.item(format!("Item {index}"), detail, item, 2);
            }
            for (index, clip) in timeline.clips().iter().enumerate() {
                self.clip(index, clip);
            }
        }
    }

    fn clip(&mut self, index: usize, clip: &Clip) {
        let kind = format!("{:?}", clip.kind());
        self.row(2, format!("Clip {index}"), &kind).fields = vec![
            ("Kind", kind.clone()),
            ("Integers", numbers(clip.integers())),
            (
                "Floats",
                numbers(clip.floats().map(|value| format!("{value:.3}"))),
            ),
        ];
    }

    fn emitters(&mut self, file: &Avfx) {
        if !self.list("Emitters", file.emitters().len()) {
            return;
        }
        for (index, emitter) in file.emitters().iter().enumerate() {
            let properties = emitter.properties();
            let detail = format!(
                "kind {}  life {}  {} particles, {} emitters",
                reference(integer(properties, "EVT")),
                reference(integer(properties, "Life")),
                emitter.particles().len(),
                emitter.emitters().len()
            );
            self.row(1, format!("Emitter {index}"), detail).fields = vec![
                (
                    "Kind",
                    format!("EVT {}", reference(integer(properties, "EVT"))),
                ),
                ("Life", reference(integer(properties, "Life"))),
                (
                    "Loop",
                    format!(
                        "{}..{}",
                        reference(integer(properties, "LpSt")),
                        reference(integer(properties, "LpEd"))
                    ),
                ),
                ("Sound", reference(integer(properties, "SdNo"))),
                ("Particles", emitter.particles().len().to_string()),
                ("Emitters", emitter.emitters().len().to_string()),
            ];
            for block in properties {
                self.block(block, 2);
            }
            for (label, items) in [
                ("Particle", emitter.particles()),
                ("Emitter", emitter.emitters()),
            ] {
                for (index, item) in items.iter().enumerate() {
                    let blocks = item.blocks();
                    let detail = format!(
                        "{} {}{}",
                        label.to_lowercase(),
                        reference(integer(blocks, "TgtB")),
                        disabled(blocks, "bEnb")
                    );
                    self.item(format!("{label} {index}"), detail, item, 2);
                }
            }
        }
    }

    /// The three lists the file writes as one block each, which are read only by their tags.
    fn blocks(&mut self, label: &str, kind: &str, blocks: &[Block]) {
        if !self.list(&format!("{label}s"), blocks.len()) {
            return;
        }
        for (index, block) in blocks.iter().enumerate() {
            let inner = block.blocks();
            // An effector carries no life of its own, where a particle and a binder both do.
            let life = integer(inner, "Life");
            let mut fields = vec![(
                "Kind",
                format!("{kind} {}", reference(integer(inner, kind))),
            )];
            fields.extend(life.map(|life| ("Life", life.to_string())));
            fields.push((
                "Loop",
                format!(
                    "{}..{}",
                    reference(integer(inner, "LpSt")),
                    reference(integer(inner, "LpEd"))
                ),
            ));

            let detail = match life {
                Some(life) => format!("kind {}  life {life}", reference(integer(inner, kind))),
                None => format!("kind {}", reference(integer(inner, kind))),
            };
            self.row(1, format!("{label} {index}"), detail).fields = fields;
            for child in inner {
                self.block(child, 2);
            }
        }
    }

    fn textures(&mut self, file: &Avfx) {
        if !self.list("Textures", file.textures().len()) {
            return;
        }
        for (index, path) in file.textures().iter().enumerate() {
            self.row(1, format!("Tex {index}"), String::new()).asset = Some(path.clone());
        }
    }

    fn models(&mut self, file: &Avfx) {
        if !self.list("Models", file.models().len()) {
            return;
        }
        for (index, model) in file.models().iter().enumerate() {
            self.row(1, format!("Model {index}"), summary(model)).fields = vec![
                ("Vertices", model.vertices().len().to_string()),
                ("Triangles", model.triangles().len().to_string()),
                ("Emit points", model.emit_vertices().len().to_string()),
            ];
        }
    }
}

/// A model, which the file holds whole rather than naming a `.mdl` beside it.
fn summary(model: &Model) -> String {
    format!(
        "{} vertices, {} triangles, {} emit points",
        model.vertices().len(),
        model.triangles().len(),
        model.emit_vertices().len()
    )
}

fn identity(file: &Avfx) -> Vec<(&'static str, String)> {
    vec![
        ("Version", format!("{:#010x}", file.version())),
        ("Schedulers", file.schedulers().len().to_string()),
        ("Timelines", file.timelines().len().to_string()),
        ("Emitters", file.emitters().len().to_string()),
        ("Particles", file.particles().len().to_string()),
        ("Effectors", file.effectors().len().to_string()),
        ("Binders", file.binders().len().to_string()),
        ("Textures", file.textures().len().to_string()),
        ("Models", file.models().len().to_string()),
    ]
}

/// The effect's own settings, leaving out the ones sitting at a value that does nothing.
fn settings(file: &Avfx) -> Vec<(&'static str, String)> {
    let mut rows: Vec<(&'static str, String)> = Vec::new();
    rows.extend(file.draw_layer().map(|v| ("Draw layer", format!("{v:?}"))));
    rows.extend(file.draw_order().map(|v| ("Draw order", format!("{v:?}"))));
    rows.extend(
        file.directional_light_source()
            .map(|v| ("Directional light", format!("{v:?}"))),
    );
    for (label, source) in ["Point light 1", "Point light 2"]
        .into_iter()
        .zip(file.point_light_sources())
    {
        rows.extend(source.map(|v| (label, format!("{v:?}"))));
    }

    if file.clip_box_enabled() == Some(true) {
        rows.extend(file.clip_box().map(|v| ("Clip box", axes(v))));
        rows.extend(file.clip_box_size().map(|v| ("Clip box size", axes(v))));
    }
    if file.clip_own_setting() == Some(true) {
        rows.extend(
            file.near_clip()
                .map(|(from, to)| ("Near clip", format!("{from:.3} to {to:.3}"))),
        );
        rows.extend(
            file.far_clip()
                .map(|(from, to)| ("Far clip", format!("{from:.3} to {to:.3}"))),
        );
    }
    if file.global_fog_enabled() == Some(true) {
        rows.extend(
            file.global_fog_influence()
                .map(|v| ("Global fog", format!("{v:.3}"))),
        );
    }

    rows.extend(
        file.revised_position()
            .filter(|v| *v != [0.0; 3])
            .map(|v| ("Position", axes(v))),
    );
    rows.extend(
        file.revised_rotation()
            .filter(|v| *v != [0.0; 3])
            .map(|v| ("Rotation", axes(v))),
    );
    rows.extend(
        file.revised_scale()
            .filter(|v| *v != [1.0; 3])
            .map(|v| ("Scale", axes(v))),
    );
    rows.extend(
        file.revised_colour()
            .filter(|v| *v != [1.0; 3])
            .map(|v| ("Color", axes(v))),
    );

    for (label, fade) in [
        ("Fade X", file.fade_x()),
        ("Fade Y", file.fade_y()),
        ("Fade Z", file.fade_z()),
    ] {
        rows.extend(
            fade.filter(|fade| fade.enabled())
                .map(|fade| (label, format!("{:.3} to {:.3}", fade.inner(), fade.outer()))),
        );
    }

    for (label, value) in [
        ("Soft particle fade", file.soft_particle_fade_range()),
        ("Sort key offset", file.sort_key_offset()),
        ("Bias Z scale", file.bias_z_max_scale()),
        ("Bias Z distance", file.bias_z_max_distance()),
    ] {
        rows.extend(
            value
                .filter(|v| *v != 0.0)
                .map(|v| (label, format!("{v:.3}"))),
        );
    }

    for (label, value) in [
        ("Delay fast particle", file.is_delay_fast_particle()),
        ("Fit ground", file.is_fit_ground()),
        ("Transform skip", file.is_transform_skip()),
        ("All stop on hide", file.is_all_stop_on_hide()),
        ("Can be clipped out", file.can_be_clipped_out()),
        ("Camera space", file.is_camera_space()),
        ("Full env light", file.is_full_env_light()),
    ] {
        rows.extend(value.map(|v| {
            (
                label,
                match v {
                    true => "yes",
                    false => "no",
                }
                .to_owned(),
            )
        }));
    }
    rows
}

pub fn decode(path: &str, bytes: &[u8]) -> Result<Preview> {
    let file = Avfx::read(Cursor::new(bytes.to_vec()))?;

    let mut build = Build::default();
    build.settings(&file);
    build.schedulers(&file);
    build.timelines(&file);
    build.emitters(&file);
    build.blocks("Particle", "PrVT", file.particles());
    build.blocks("Effector", "EfVT", file.effectors());
    build.blocks("Binder", "BnVr", file.binders());
    build.textures(&file);
    build.models(&file);

    let mut effect = sim::Effect::read(&file);
    let (target, reach) = effect.fit();
    // The models are the effect's own geometry and nothing reads them again once they are on the
    // card: a particle already carries the index it draws.
    let models = std::mem::take(&mut effect.models);
    let home = Camera {
        yaw: 0.6,
        pitch: 0.25,
        distance: reach * MARGIN / (FOV * 0.5).tan(),
        target,
    };

    log::info!(
        "assets/avfx: {path} {} timelines, {} emitters, {} particles, {} curves, {} frames",
        file.timelines().len(),
        file.emitters().len(),
        file.particles().len(),
        build.curves.len(),
        effect.length
    );

    Ok(Preview::Avfx(Box::new(Rendered {
        identity: identity(&file),
        settings: settings(&file),
        rows: build.rows,
        curves: build.curves,
        state: egui::Id::new(path).with("avfx_tree"),
        gpu: gpu::Particles::new(models),
        effect,
        live: RefCell::default(),
        at: Cell::new(0.0),
        playing: Cell::new(true),
        camera: Cell::new(home),
        home,
        textures: RefCell::default(),
        shaders: RefCell::default(),
    })))
}

/// The preview over the tree, with the split between them draggable.
pub fn ui(ui: &mut egui::Ui, file: &Rendered, backend: &Backend) -> Option<String> {
    file.poll(ui, backend);
    let follow = egui::containers::panel::Panel::bottom(file.state.with("split"))
        .default_size((ui.available_height() * 0.4).max(120.0))
        .show(ui, |ui| tree_ui(ui, file))
        .inner;
    file.preview_ui(ui);
    follow
}

fn tree_ui(ui: &mut egui::Ui, file: &Rendered) -> Option<String> {
    let mut follow = None;
    let mut open = file.open(ui);
    let mut shown = Vec::new();
    let mut collapsed_at = None;
    for (index, row) in file.rows.iter().enumerate() {
        match collapsed_at {
            Some(depth) if row.depth > depth => continue,
            _ => collapsed_at = None,
        }
        let parent = file.parent(index);
        let expanded = parent && (row.open != open.contains(&index));
        if parent && !expanded {
            collapsed_at = Some(row.depth);
        }
        shown.push((index, expanded));
    }

    section(ui, "Effect");
    let picked = file.selected(ui);
    let mut selected = picked;
    let mut toggled = None;
    // A curve's row carries a sparkline, which is taller than either the text or the expander, and
    // the height handed to the scroll area has to cover the tallest of the three or the rows it
    // places drift out of their own space.
    let height = ui
        .text_style_height(&egui::TextStyle::Monospace)
        .max(TRIANGLE)
        .max(curve::SPARK.y)
        + 2.0 * ui.spacing().button_padding.y
        + ui.spacing().item_spacing.y;
    ScrollArea::vertical()
        .auto_shrink(false)
        .show_rows(ui, height, shown.len(), |ui, range| {
            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
            for &(index, expanded) in &shown[range] {
                let row = &file.rows[index];
                ui.horizontal(|ui| {
                    ui.add_space(f32::from(row.depth) * INDENT);
                    match file.parent(index) {
                        false => ui.add_space(TRIANGLE),
                        true => {
                            let (_, response) =
                                ui.allocate_exact_size(Vec2::splat(TRIANGLE), Sense::click());
                            let openness = match expanded {
                                true => 1.0,
                                false => 0.0,
                            };
                            paint_default_icon(ui, openness, &response);
                            if response.clicked() {
                                toggled = Some(index);
                            }
                        }
                    }

                    if ui
                        .selectable_label(
                            picked == Some(index),
                            RichText::new(&row.label).monospace(),
                        )
                        .clicked()
                    {
                        selected = Some(index);
                    }
                    if let Some(curve) = row.curve {
                        curve::spark(ui, &file.curves[curve]);
                    }
                    if let Some(asset) = &row.asset
                        && link(ui, file_name(asset), asset)
                    {
                        follow = Some(asset.clone());
                    }
                    if !row.detail.is_empty() {
                        ui.label(RichText::new(&row.detail).monospace().weak());
                    }
                });
            }
        });

    if selected != picked {
        ui.data_mut(|data| data.insert_temp(file.state.with("selected"), selected));
    }
    if let Some(index) = toggled {
        if !open.insert(index) {
            open.remove(&index);
        }
        ui.data_mut(|data| data.insert_temp(file.state, open));
    }
    follow
}

impl Rendered {
    fn open(&self, ui: &egui::Ui) -> HashSet<usize> {
        ui.data(|data| data.get_temp(self.state).unwrap_or_default())
    }

    fn selected(&self, ui: &egui::Ui) -> Option<usize> {
        ui.data(|data| data.get_temp(self.state.with("selected")).flatten())
    }

    /// Whether anything sits under a row. Rows are held parents-first, so the row after one is its
    /// first child where it has any.
    fn parent(&self, index: usize) -> bool {
        self.rows
            .get(index + 1)
            .is_some_and(|next| next.depth > self.rows[index].depth)
    }

    /// The curves a row draws: its own where it is one, and the ones written directly under it
    /// otherwise, which is how a position or a color arrives: one curve an axis.
    fn drawn(&self, index: usize) -> Vec<usize> {
        let row = &self.rows[index];
        if let Some(curve) = row.curve {
            return vec![curve];
        }
        self.rows[index + 1..]
            .iter()
            .take_while(|under| under.depth > row.depth)
            .filter(|under| under.depth == row.depth + 1)
            .filter_map(|under| under.curve)
            .collect()
    }

    /// Asks for the textures the effect samples and the packages it is shaded by.
    fn poll(&self, ui: &egui::Ui, backend: &Backend) {
        self.shaders.borrow_mut().poll(backend);
        self.textures
            .borrow_mut()
            .poll(ui.ctx(), backend, &self.effect.textures);
    }

    /// The playback bar, and the effect running under it.
    fn preview_ui(&self, ui: &mut egui::Ui) {
        let rate = AVFX_FRAME_RATE.get(ui.ctx());
        let length = self.effect.length;
        ui.horizontal(|ui| {
            let playing = self.playing.get();
            if ui
                .selectable_label(
                    playing,
                    match playing {
                        true => "Pause",
                        false => "Play",
                    },
                )
                .clicked()
            {
                self.playing.set(!playing);
            }
            if ui.button("Restart").clicked() {
                self.at.set(0.0);
            }
            if ui.button("Reset view").clicked() {
                self.camera.set(self.home);
            }

            let mut at = self.at.get();
            ui.spacing_mut().slider_width = (ui.available_width() - 140.0).max(80.0);
            // On the response rather than on `changed`: the slider rewrites the value it is handed
            // every frame, so playing through it reads as the user having moved it.
            let response =
                ui.add(egui::Slider::new(&mut at, 0.0..=length as f32).show_value(false));
            if response.dragged() || response.clicked() {
                self.at.set(at);
                self.playing.set(false);
            }
            ui.label(
                RichText::new(format!("{at:.0}/{length}  {}", curve::seconds(at, rate)))
                    .monospace()
                    .weak(),
            );
        });

        if let Some(why) = self.gpu.lock().unwrap().failure() {
            ui.centered_and_justified(|ui| {
                ui.colored_label(Color32::RED, format!("Could not build the shader: {why}"));
            });
            return;
        }

        if self.playing.get() {
            let step = ui.input(|input| input.stable_dt).clamp(0.0, 0.1) * rate;
            self.at.set((self.at.get() + step) % length as f32);
            ui.ctx().request_repaint();
        }
        self.effect
            .seek(&mut self.live.borrow_mut(), self.at.get() as i32);
        self.viewport(ui);
    }

    /// The effect itself: an orbit camera over a paint callback.
    fn viewport(&self, ui: &mut egui::Ui) {
        let (rect, response) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        if rect.width() < 1.0 || rect.height() < 1.0 {
            return;
        }

        let mut camera = self.camera.get();
        let pan = |camera: &mut Camera, delta: egui::Vec2| {
            let (sin_yaw, cos_yaw) = camera.yaw.sin_cos();
            let right = Vec3::new(cos_yaw, 0.0, -sin_yaw);
            let scale = camera.distance * 0.002;
            camera.target += (right * -delta.x + Vec3::Y * delta.y) * scale;
        };
        let zoom = |camera: &mut Camera, scale: f32| {
            camera.distance = (camera.distance * scale)
                .clamp(self.home.distance * 0.02, self.home.distance * 20.0);
        };

        // A second finger takes the gesture over: egui carries on reporting a primary drag through
        // one, so leaving the orbit armed would spin the effect while it is being pinched.
        let touch = ui.input(|input| input.multi_touch());
        match touch.filter(|_| response.dragged()) {
            Some(touch) => {
                zoom(&mut camera, 1.0 / touch.zoom_delta);
                pan(&mut camera, touch.translation_delta);
            }
            None => {
                if response.dragged_by(egui::PointerButton::Primary) {
                    let delta = response.drag_delta();
                    camera.yaw -= delta.x * 0.01;
                    camera.pitch = (camera.pitch + delta.y * 0.01).clamp(-1.5, 1.5);
                }
                if response.dragged_by(egui::PointerButton::Secondary) {
                    pan(&mut camera, response.drag_delta());
                }
            }
        }
        if response.hovered() {
            let scroll = ui.input(|input| input.smooth_scroll_delta.y);
            if scroll != 0.0 {
                zoom(&mut camera, 1.0 - scroll * 0.002);
            }
        }
        self.camera.set(camera);

        let eye = camera.eye();
        let view = Mat4::look_at_rh(eye, camera.target, Vec3::Y);
        // Nothing here depth tests, so the planes do nothing but clip. A slab fitted to the effect's
        // own reach cuts the corners off the very bounds it was taken from, and loses whatever a
        // particle does past the frames the fit ran over.
        let projection = Mat4::perspective_rh_gl(
            FOV,
            rect.width() / rect.height(),
            camera.distance * 0.005,
            camera.distance * 200.0,
        );

        // A sprite is set into the screen's plane, which is what the camera's own axes are for. The
        // shape package reads a stream already placed in the world, so the billboard happens here.
        let axes = glam::Mat3::from_mat4(view).transpose();
        let frame = gpu::Frame {
            scene: program::Scene {
                view,
                projection,
                size: (rect.width(), rect.height()),
                light: (eye - camera.target).normalize_or(Vec3::Y),
                fade_range: self.effect.fade_range,
                ..program::Scene::default()
            },
            batches: self.batches(view, eye, axes.x_axis, axes.y_axis),
            packages: self.shaders.borrow().resolved(),
            tested: false,
            // Nothing behind a particle here, so the soft-particle variant reads an unbound
            // sampler same as before: the preview has no scene depth to copy.
            depth: None,
        };

        // The context is taken from the painter rather than captured: `glow::Context` is neither
        // `Send` nor `Sync` on wasm, and a callback has to be both.
        let particles = self.gpu.clone();
        ui.painter().add(egui::PaintCallback {
            rect,
            callback: Arc::new(egui_glow::CallbackFn::new(move |_info, painter| {
                particles
                    .lock()
                    .unwrap()
                    .draw(painter.gl(), painter, &frame);
            })),
        });
    }

    /// The live particles, gathered into one draw apiece per particle definition and blend, furthest
    /// group first. Blending reads what is already there, so the order is part of the picture.
    fn batches(&self, view: Mat4, eye: Vec3, right: Vec3, up: Vec3) -> Vec<gpu::Batch> {
        let bound = self.textures.borrow().bound(&self.effect.textures);
        batches(
            &self.effect,
            self.effect.drawn(&self.live.borrow()),
            &bound,
            view,
            eye,
            right,
            up,
        )
    }

    pub fn details_ui(&self, ui: &mut egui::Ui, follow: &mut Option<String>) {
        let mut rate = AVFX_FRAME_RATE.get(ui.ctx());
        ui.horizontal(|ui| {
            ui.label(RichText::new("Frame rate").weak());
            if ui
                .add(
                    egui::DragValue::new(&mut rate)
                        .speed(1.0)
                        .range(1.0..=240.0)
                        .suffix(" fps"),
                )
                .changed()
            {
                AVFX_FRAME_RATE.set(ui.ctx(), rate);
            }
        });
        ui.separator();

        ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
            if let Some(index) = self.selected(ui)
                && let Some(row) = self.rows.get(index)
            {
                ui.label(RichText::new(&row.label).strong());
                ui.add_space(4.0);
                if !row.detail.is_empty() {
                    ui.label(RichText::new(&row.detail).monospace());
                    ui.add_space(4.0);
                }
                if let Some(path) = &row.asset
                    && link(ui, file_name(path), path)
                {
                    *follow = Some(path.clone());
                }
                if !row.fields.is_empty() {
                    facts(ui, "avfx_selected", &row.fields);
                }

                let drawn = self.drawn(index);
                for (position, curve) in drawn.iter().enumerate() {
                    let curve = &self.curves[*curve];
                    ui.add_space(8.0);
                    ui.separator();
                    if drawn.len() > 1 || curve.name != row.label {
                        heading(ui, &curve.name);
                    }
                    curve_ui(ui, curve, position, rate);
                }

                ui.add_space(8.0);
                ui.separator();
            }

            facts(ui, "avfx_identity", &self.identity);
            if !self.settings.is_empty() {
                ui.add_space(8.0);
                ui.separator();
                heading(ui, "Settings");
                facts(ui, "avfx_settings", &self.settings);
            }
        });
    }
}

/// The two edges one sprite's quad spans, each as long as the scale along it. The shape package
/// reads a stream the viewer has already placed in the world, so which way a particle is turned is
/// settled here rather than in the shader.
fn facing(drawn: &sim::Drawn, eye: Vec3, right: Vec3, up: Vec3) -> (Vec3, Vec3) {
    let scale = Vec3::from(drawn.scale);
    // A quad in the screen's plane can carry only the turn about its own normal, and the file writes
    // that one as the last of its Euler angles.
    let spun = |across: Vec3, down: Vec3| {
        let (sin, cos) = drawn.roll.sin_cos();
        (
            (across * cos + down * sin) * scale.x,
            (down * cos - across * sin) * scale.y,
        )
    };
    match drawn.facing {
        // Billed about the particle's own axis rather than about the camera's up: two captures of
        // one emote hold the two lightsticks at opposite diagonals and each glow runs along its own
        // stick, where a bill taken about the camera would stand both of them the same way up. The
        // turn already carries every angle the file states, so there is no roll to lay over it.
        sim::Facing::Camera => {
            let axis = (glam::Quat::from_array(drawn.turn) * Vec3::Y).normalize_or(Vec3::Y);
            let away = (eye - Vec3::from(drawn.center)).normalize_or(-Vec3::Z);
            let across = axis.cross(away).normalize_or(right);
            (across * scale.x, axis * scale.y)
        }
        // Standing upright is the whole of what this one asks for, so it takes no turn: a roll would
        // lean the quad off the axis it is billed about. Which of the two bills a sprite takes went
        // unmeasured, so both meet the eye here as they always did.
        sim::Facing::Upright(_) => {
            let across = Vec3::Y
                .cross(eye - Vec3::from(drawn.center))
                .normalize_or(right);
            (across * scale.x, Vec3::Y * scale.y)
        }
        // A quad the camera has no say in carries the whole of its own turn rather than the roll, and
        // spans the two axes it is scaled along rather than the first two.
        sim::Facing::Still(axis) => {
            let (across, down) = match axis {
                sim::Axis::X => (Vec3::Z * scale.z, Vec3::Y * scale.y),
                sim::Axis::Y => (Vec3::X * scale.x, Vec3::Z * scale.z),
                sim::Axis::Z => (Vec3::X * scale.x, Vec3::Y * scale.y),
            };
            let turn = glam::Quat::from_array(drawn.turn);
            (turn * across, turn * down)
        }
        sim::Facing::Screen => spun(right, up),
    }
}

/// The turn a model billed about the world's up axis is drawn under. The game hands its own model
/// package a world matrix and the package never reads the view, so this is the whole of the bill:
/// the particle's own turn does not survive it.
fn billed(toward: sim::Toward, center: Vec3, eye: Vec3, back: Vec3) -> glam::Quat {
    let aim = match toward {
        sim::Toward::Eye => eye - center,
        sim::Toward::Screen => back,
    };
    let aim = Vec3::new(aim.x, 0.0, aim.z).normalize_or(Vec3::Z);
    glam::Quat::from_mat3(&glam::Mat3::from_cols(Vec3::Y.cross(aim), Vec3::Y, aim))
}

/// One batch per particle definition, shape and blend, furthest group first since blending reads
/// what is already there. Takes already-placed draws: a zone merges every instance of the same file
/// into one set before calling this, so a placement costs a transform rather than a draw of its own.
pub(crate) fn batches(
    effect: &sim::Effect,
    drawn: impl IntoIterator<Item = sim::Drawn>,
    textures: &[Option<egui::TextureId>],
    view: Mat4,
    eye: Vec3,
    right: Vec3,
    up: Vec3,
) -> Vec<gpu::Batch> {
    let back = right.cross(up);
    let mut groups: BTreeMap<_, Vec<(f32, sim::Drawn)>> = BTreeMap::new();
    for drawn in drawn {
        let center = Vec3::from(drawn.center);
        let depth = (view * Vec4::from((center, 1.0))).z;
        groups
            .entry((drawn.def, drawn.shape, drawn.blend))
            .or_default()
            .push((depth, drawn));
    }

    let mut batches: Vec<(f32, gpu::Batch)> = groups
        .into_iter()
        .filter_map(|((def, shape, blend), mut held)| {
            held.sort_by(|(a, _), (b, _)| a.total_cmp(b));
            let mean = held.iter().map(|(depth, _)| depth).sum::<f32>() / held.len() as f32;
            let mut vertices = Vec::new();
            let mut instances = Vec::new();
            for (_, drawn) in &held {
                match shape {
                    sim::Shape::Sprite => {
                        let (across, down) = facing(drawn, eye, right, up);
                        gpu::quad(
                            Vec3::from(drawn.center),
                            across,
                            down,
                            drawn.color,
                            &drawn.uv,
                            &mut vertices,
                        );
                    }
                    sim::Shape::Model(_) if instances.len() < MODELS => {
                        instances.push(program::Instance {
                            transform: Mat4::from_scale_rotation_translation(
                                Vec3::from(drawn.scale),
                                match drawn.facing {
                                    sim::Facing::Upright(toward) => {
                                        billed(toward, Vec3::from(drawn.center), eye, back)
                                    }
                                    _ => glam::Quat::from_array(drawn.turn),
                                },
                                Vec3::from(drawn.center),
                            ),
                            color: Vec4::from(drawn.color),
                            rim: drawn.rim,
                            uv: drawn.uv,
                            ..program::Instance::default()
                        });
                    }
                    sim::Shape::Model(_) => {}
                }
            }
            Some((
                mean,
                gpu::Batch {
                    shape,
                    textures: textures.to_vec(),
                    blend,
                    def,
                    shading: effect.shading(def)?,
                    vertices,
                    instances,
                },
            ))
        })
        .collect();
    batches.sort_by(|(a, _), (b, _)| a.total_cmp(b));
    batches.into_iter().map(|(_, batch)| batch).collect()
}

/// One curve: the plot, what it does either side of its keys, and every key it holds.
fn curve_ui(ui: &mut egui::Ui, curve: &Curve, position: usize, rate: f32) {
    let range = curve::plot(ui, curve, rate);
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(format!("before {:?}, after {:?}", curve.pre, curve.post))
                .monospace()
                .weak(),
        );
        if let Some(random) = curve.random {
            ui.label(RichText::new(format!("{random:?}")).monospace().weak());
        }
    });
    if let Some((low, high)) = range {
        ui.label(
            RichText::new(format!("{low:.3} to {high:.3}"))
                .monospace()
                .weak(),
        );
    }
    ui.add_space(4.0);

    let columns = match curve.color {
        true => 5,
        false => 4,
    };
    egui::Grid::new(("avfx_keys", position))
        .num_columns(columns)
        .striped(true)
        .show(ui, |ui| {
            match curve.color {
                true => headers(ui, &["Frame", "Time", "Kind", "", "Color"]),
                false => headers(ui, &["Frame", "Time", "Kind", "Value"]),
            }
            for key in &curve.keys {
                ui.label(RichText::new(key.time().to_string()).monospace());
                ui.label(
                    RichText::new(curve::seconds(f32::from(key.time()), rate))
                        .monospace()
                        .weak(),
                );
                ui.label(
                    RichText::new(format!("{:?}", key.kind()))
                        .monospace()
                        .weak(),
                );
                match curve.color {
                    true => {
                        let (at, _) = ui.allocate_exact_size(Vec2::splat(CHIP), Sense::hover());
                        ui.painter()
                            .rect_filled(at, 2.0, curve.swatch(f32::from(key.time())));
                        let [r, g, b] = key.data().map(|channel| (channel * 255.0).round() as u8);
                        ui.label(RichText::new(format!("{r}, {g}, {b}")).monospace());
                    }
                    false => {
                        ui.label(RichText::new(format!("{}", key.value())).monospace());
                    }
                }
                ui.allocate_space(vec2(ui.available_width(), 0.0));
                ui.end_row();
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One block: the tag back to front, then its length, then its payload padded out to four.
    fn block(tag: &str, payload: &[u8]) -> Vec<u8> {
        let mut bytes = tag.bytes().rev().collect::<Vec<_>>();
        bytes.resize(4, 0);
        bytes.extend(u32::try_from(payload.len()).unwrap().to_le_bytes());
        bytes.extend(payload);
        bytes.resize(8 + payload.len().next_multiple_of(4), 0);
        bytes
    }

    fn nest(tag: &str, children: &[Vec<u8>]) -> Vec<u8> {
        block(tag, &children.concat())
    }

    fn integer(value: i32) -> Vec<u8> {
        value.to_le_bytes().into()
    }

    /// One curve, as the file writes it: the behaviours either side of its keys, then the keys.
    fn curve(tag: &str, pre: i32, post: i32, keys: &[(i16, i16, [f32; 3])]) -> Vec<u8> {
        let mut written = Vec::new();
        for &(time, kind, data) in keys {
            written.extend(time.to_le_bytes());
            written.extend(kind.to_le_bytes());
            written.extend(data.iter().flat_map(|value| value.to_le_bytes()));
        }
        nest(
            tag,
            &[
                block("BvPr", &integer(pre)),
                block("BvPo", &integer(post)),
                block("Keys", &written),
            ],
        )
    }

    fn read(children: &[Vec<u8>]) -> Rendered {
        let mut blocks = vec![block("Ver", &integer(0x2011_0913))];
        blocks.extend_from_slice(children);
        match decode("test.avfx", &nest("AVFX", &blocks)).unwrap() {
            Preview::Avfx(effect) => *effect,
            _ => panic!("read as something other than an effect"),
        }
    }

    /// The one curve a file holds, which every curve test builds under a particle.
    fn only(keys: &[(i16, i16, [f32; 3])], tag: &str, pre: i32, post: i32) -> Rendered {
        read(&[nest("Ptcl", &[curve(tag, pre, post, keys)])])
    }

    fn scalars(values: &[(i16, i16, f32)]) -> Vec<(i16, i16, [f32; 3])> {
        values
            .iter()
            .map(|&(time, kind, value)| (time, kind, [1.0, 1.0, value]))
            .collect()
    }

    #[test]
    fn a_payload_reads_as_whichever_of_the_two_its_bits_can_be() {
        assert_eq!(scalar(0i32.to_le_bytes()), "0");
        assert_eq!(scalar(30i32.to_le_bytes()), "30");
        assert_eq!(scalar((-1i32).to_le_bytes()), "-1");
        assert_eq!(scalar(1.0f32.to_le_bytes()), "1");
        assert_eq!(scalar(0.5f32.to_le_bytes()), "0.5");
        assert_eq!(scalar((-2.25f32).to_le_bytes()), "-2.25");
    }

    #[test]
    fn a_curve_collapses_into_the_row_it_sits_on() {
        let effect = only(&scalars(&[(0, 1, 0.0), (10, 1, 1.0)]), "X", 0, 0);
        let labels = effect
            .rows
            .iter()
            .map(|row| row.label.as_str())
            .collect::<Vec<_>>();
        assert_eq!(labels, ["Particles", "Particle 0", "X"]);
        assert_eq!(effect.curves.len(), 1);
        assert_eq!(effect.rows[2].curve, Some(0));
        assert_eq!(effect.rows[2].detail, "2 keys  0..10");
    }

    #[test]
    fn linear_runs_between_its_keys() {
        let effect = only(&scalars(&[(0, 1, 0.0), (10, 1, 1.0)]), "X", 0, 0);
        let curve = &effect.curves[0];
        assert_eq!(curve.sample(0.0)[2], 0.0);
        assert_eq!(curve.sample(5.0)[2], 0.5);
        assert_eq!(curve.sample(10.0)[2], 1.0);
    }

    #[test]
    fn a_step_holds_until_the_next_key() {
        let effect = only(&scalars(&[(0, 2, 3.0), (10, 1, 7.0)]), "X", 0, 0);
        let curve = &effect.curves[0];
        assert_eq!(curve.sample(0.0)[2], 3.0);
        assert_eq!(curve.sample(9.9)[2], 3.0);
        assert_eq!(curve.sample(10.0)[2], 7.0);
    }

    /// The tangent scales beside a spline key go unread, so all that is pinned down is that the
    /// curve meets its keys and bulges between them.
    #[test]
    fn a_spline_meets_its_keys() {
        let effect = only(
            &scalars(&[(0, 0, 0.0), (10, 0, 1.0), (20, 0, 0.0)]),
            "X",
            0,
            0,
        );
        let curve = &effect.curves[0];
        assert_eq!(curve.sample(0.0)[2], 0.0);
        assert_eq!(curve.sample(10.0)[2], 1.0);
        assert_eq!(curve.sample(20.0)[2], 0.0);
        assert!(curve.sample(5.0)[2] > 0.5);
    }

    #[test]
    fn behaviours_carry_a_curve_outside_its_keys() {
        let keys = scalars(&[(0, 1, 1.0), (10, 1, 3.0)]);

        let hold = only(&keys, "X", 0, 0);
        assert_eq!(hold.curves[0].sample(-10.0)[2], 1.0);
        assert_eq!(hold.curves[0].sample(20.0)[2], 3.0);

        let repeat = only(&keys, "X", 1, 1);
        assert_eq!(repeat.curves[0].sample(15.0)[2], 2.0);
        assert_eq!(repeat.curves[0].sample(-5.0)[2], 2.0);

        let add = only(&keys, "X", 2, 2);
        assert_eq!(add.curves[0].sample(15.0)[2], 4.0);
        assert_eq!(add.curves[0].sample(25.0)[2], 6.0);
    }

    #[test]
    fn a_single_key_holds_everywhere() {
        let effect = only(&scalars(&[(30, 1, 2.5)]), "X", 1, 1);
        assert_eq!(effect.curves[0].sample(-100.0)[2], 2.5);
        assert_eq!(effect.curves[0].sample(100.0)[2], 2.5);
    }

    #[test]
    fn keys_out_of_order_are_read_in_time_order() {
        let effect = only(&scalars(&[(10, 1, 2.0), (0, 1, 1.0)]), "X", 0, 0);
        assert_eq!(effect.curves[0].sample(5.0)[2], 1.5);
    }

    /// An `RGB` curve writes a channel in each of its three floats rather than a value and two
    /// tangent scales, so all three interpolate.
    #[test]
    fn a_color_curve_interpolates_every_channel() {
        let effect = only(
            &[(0, 1, [1.0, 0.0, 0.0]), (10, 1, [0.0, 1.0, 0.0])],
            "RGB",
            0,
            0,
        );
        let curve = &effect.curves[0];
        assert!(curve.color);
        assert_eq!(curve.sample(5.0), [0.5, 0.5, 0.0]);
        assert_eq!(curve.swatch(0.0), egui::Color32::from_rgb(255, 0, 0));
    }

    #[test]
    fn an_absent_reference_reads_as_none() {
        let effect = read(&[nest(
            "TmLn",
            &[
                block("LpSt", &integer(0)),
                block("LpEd", &integer(-1)),
                block("TICn", &integer(1)),
                block(
                    "Item",
                    &[
                        block("bEna", &integer(1)),
                        block("StTm", &integer(0)),
                        block("EdTm", &integer(60)),
                        block("BdNo", &integer(-1)),
                        block("EfNo", &integer(2)),
                        block("EmNo", &integer(-1)),
                    ]
                    .concat(),
                ),
            ],
        )]);
        let item = effect
            .rows
            .iter()
            .find(|row| row.label == "Item 0")
            .expect("the timeline's item");
        assert_eq!(item.detail, "0..60  emitter none  binder none  effector 2");
        let timeline = &effect.rows[1];
        assert_eq!(timeline.detail, "loop 0..none  1 items");
    }

    #[test]
    fn a_sound_is_a_link_and_a_texture_is_its_own_row() {
        let effect = read(&[
            nest("Emit", &[block("SdNm", b"sound/vfx/se_vfx_test.scd\0")]),
            block("Tex", b"vfx/common/texture/uv_r.atex\0"),
        ]);
        let sound = effect
            .rows
            .iter()
            .find(|row| row.label == "SdNm")
            .expect("the emitter's sound");
        assert_eq!(sound.asset.as_deref(), Some("sound/vfx/se_vfx_test.scd"));
        let texture = effect
            .rows
            .iter()
            .find(|row| row.label == "Tex 0")
            .expect("the effect's texture");
        assert_eq!(
            texture.asset.as_deref(),
            Some("vfx/common/texture/uv_r.atex")
        );
    }

    #[test]
    fn a_container_draws_the_curves_written_under_it() {
        let effect = read(&[nest(
            "Ptcl",
            &[nest(
                "Col",
                &[
                    curve("RGB", 0, 0, &[(0, 1, [1.0, 1.0, 1.0])]),
                    curve("A", 0, 0, &scalars(&[(0, 1, 1.0)])),
                ],
            )],
        )]);
        let column = effect
            .rows
            .iter()
            .position(|row| row.label == "Col")
            .expect("the colour container");
        assert_eq!(effect.drawn(column), vec![0, 1]);
        assert_eq!(effect.drawn(column + 1), vec![0]);
    }

    fn life(frames: f32) -> Vec<u8> {
        nest("Life", &[block("Val", &frames.to_le_bytes())])
    }

    /// One emitter, running one particle over `span`, bursting one a frame.
    fn playing(particle: &[Vec<u8>], span: (i32, i32)) -> Rendered {
        read(&[
            nest(
                "TmLn",
                &[
                    block("TICn", &integer(1)),
                    block(
                        "Item",
                        &[
                            block("bEna", &integer(1)),
                            block("StTm", &integer(span.0)),
                            block("EdTm", &integer(span.1)),
                            block("EmNo", &integer(0)),
                        ]
                        .concat(),
                    ),
                ],
            ),
            nest(
                "Emit",
                &[
                    block("PrCn", &integer(1)),
                    block(
                        "ItPr",
                        &[
                            block("bEnb", &integer(1)),
                            block("TgtB", &integer(0)),
                            block("CrCn", &integer(1)),
                        ]
                        .concat(),
                    ),
                    life(-1.0),
                    curve("CrC", 0, 0, &scalars(&[(0, 1, 1.0)])),
                    curve("CrI", 0, 0, &scalars(&[(0, 1, 1.0)])),
                ],
            ),
            nest("Ptcl", particle),
        ])
    }

    fn at(effect: &sim::Effect, frame: i32) -> Vec<sim::Drawn> {
        let mut state = sim::State::default();
        effect.seek(&mut state, frame);
        effect.drawn(&state)
    }

    #[test]
    fn an_emitter_runs_over_the_span_its_timeline_gives_it() {
        let effect = &playing(&[life(2.0)], (3, 6)).effect;
        assert!(at(effect, 2).is_empty());
        assert_eq!(at(effect, 4).len(), 2);
        // The last one spawns on frame six and lives two frames past it.
        assert!(!at(effect, 8).is_empty());
        assert!(at(effect, 9).is_empty());
    }

    /// A particle's position is the sum of every step it has taken, so seeking back has to replay.
    #[test]
    fn seeking_back_lands_where_playing_forward_did() {
        let effect = &playing(&[life(4.0)], (0, 20)).effect;
        let forward: Vec<_> = at(effect, 7)
            .into_iter()
            .map(|drawn| drawn.center)
            .collect();

        let mut state = sim::State::default();
        effect.seek(&mut state, 15);
        effect.seek(&mut state, 7);
        let back: Vec<_> = effect
            .drawn(&state)
            .into_iter()
            .map(|drawn| drawn.center)
            .collect();
        assert_eq!(forward, back);
    }

    #[test]
    fn a_particle_carries_its_own_curves_from_the_frame_it_spawned_on() {
        let effect = &playing(
            &[
                life(10.0),
                nest(
                    "Pos",
                    &[curve("Y", 0, 0, &scalars(&[(0, 1, 0.0), (10, 1, 10.0)]))],
                ),
            ],
            (0, 0),
        )
        .effect;
        assert_eq!(at(effect, 0)[0].center[1], 0.0);
        assert_eq!(at(effect, 5)[0].center[1], 5.0);
    }

    #[test]
    fn a_particle_expires_at_the_life_the_file_gives_it() {
        let effect = &playing(&[life(3.0)], (0, 0)).effect;
        assert_eq!(at(effect, 3).len(), 1);
        assert!(at(effect, 4).is_empty());
    }

    /// A life the file writes as `-1` is one the particle never reaches.
    #[test]
    fn a_particle_with_no_life_runs_to_the_end() {
        let effect = &playing(&[life(-1.0)], (0, 0)).effect;
        assert_eq!(at(effect, 200).len(), 1);
    }

    fn scale(value: f32) -> Vec<u8> {
        let axes = ["X", "Y", "Z"].map(|axis| curve(axis, 0, 0, &scalars(&[(0, 1, value)])));
        nest("Scl", &axes)
    }

    /// A sprite is drawn one scale across about its own center, so that is the reach the camera opens
    /// on: taking the scale for a half extent stands it off at several times the distance.
    #[test]
    fn a_sprite_is_framed_on_the_quad_it_draws() {
        let effect = &playing(&[life(-1.0), scale(4.0)], (0, 0)).effect;
        assert_eq!(effect.fit(), (Vec3::ZERO, 2.0));

        // Left lying across x and z, the height it is framed on is the one it is deep by.
        let flat = nest(
            "Scl",
            &["X", "Y", "Z"]
                .iter()
                .zip([1.0, 1.0, 8.0])
                .map(|(axis, value)| curve(axis, 0, 0, &scalars(&[(0, 1, value)])))
                .collect::<Vec<_>>(),
        );
        let effect = &playing(&[life(-1.0), block("RBDT", &integer(1)), flat], (0, 0)).effect;
        assert_eq!(effect.fit(), (Vec3::ZERO, 4.0));
    }

    #[test]
    fn a_model_kind_draws_the_model_it_names() {
        let mut vertex = vec![0; 36];
        vertex[0] = 0x3c;
        let model = nest(
            "Modl",
            &[
                block("VDrw", &vertex),
                block("VIdx", &[0u16, 0, 0].map(u16::to_le_bytes).concat()),
            ],
        );
        let effect = read(&[
            model,
            nest(
                "TmLn",
                &[block(
                    "Item",
                    &[block("bEna", &integer(1)), block("EmNo", &integer(0))].concat(),
                )],
            ),
            nest(
                "Emit",
                &[
                    block(
                        "ItPr",
                        &[block("bEnb", &integer(1)), block("TgtB", &integer(0))].concat(),
                    ),
                    life(-1.0),
                ],
            ),
            nest(
                "Ptcl",
                &[
                    block("PrVT", &integer(5)),
                    block("RMT", &integer(2)),
                    life(-1.0),
                    nest("Data", &[block("MdNo", &[0])]),
                ],
            ),
        ]);
        let drawn = at(&effect.effect, 0);
        assert_eq!(drawn[0].shape, sim::Shape::Model(0));
        assert_eq!(drawn[0].blend, sim::Blend::Add);
    }

    /// A model index past what the file holds is not one, and falls back to the sprite.
    #[test]
    fn a_model_index_the_file_does_not_hold_draws_as_a_sprite() {
        let effect = &playing(
            &[
                life(-1.0),
                block("PrVT", &integer(5)),
                nest("Data", &[block("MdNo", &[3])]),
            ],
            (0, 0),
        )
        .effect;
        assert_eq!(at(effect, 0)[0].shape, sim::Shape::Sprite);
    }

    /// An axis the file ties to another is written no curve of its own.
    #[test]
    fn a_tied_axis_takes_the_curve_of_the_one_it_follows() {
        let width = curve("X", 0, 0, &scalars(&[(0, 1, 3.0)]));
        let tied = nest("Scl", &[block("ACT", &integer(1)), width.clone()]);
        let effect = &playing(&[life(-1.0), tied], (0, 0)).effect;
        assert_eq!(at(effect, 0)[0].scale, [3.0; 3]);

        let apart = nest("Scl", &[width]);
        let effect = &playing(&[life(-1.0), apart], (0, 0)).effect;
        assert_eq!(at(effect, 0)[0].scale, [3.0, 1.0, 1.0]);
    }

    /// The sprite packages put no transform on a texture coordinate, so what a uv set does has to
    /// arrive with the coordinate.
    #[test]
    fn a_uv_set_scales_and_scrolls_about_the_texture_middle() {
        let axes = |tag: &str, values: [f32; 2]| {
            nest(
                tag,
                &["X", "Y"]
                    .iter()
                    .zip(values)
                    .map(|(axis, value)| curve(axis, 0, 0, &scalars(&[(0, 1, value)])))
                    .collect::<Vec<_>>(),
            )
        };
        let set = nest("UvSt", &[axes("Scl", [2.0, 2.0]), axes("Scr", [0.25, 0.0])]);
        let effect = &playing(&[life(-1.0), set], (0, 0)).effect;
        let uv = at(effect, 0)[0].uv;
        assert_eq!(uv[0], [2.0, 0.0, 0.0, 0.75]);
        assert_eq!(uv[1], [0.0, 2.0, 0.0, 0.5]);
        assert_eq!(uv[2], [1.0, 0.0, 0.0, 0.5]);
    }

    /// Naming no rotation base leaves a sprite in its own plane rather than against the screen, and
    /// a decal in the one it is cast onto.
    #[test]
    fn a_sprite_with_no_rotation_base_stays_in_its_own_plane() {
        let unbased = |kind: i32| {
            let effect = &playing(
                &[
                    life(-1.0),
                    block("PrVT", &integer(kind)),
                    block("RBDT", &integer(10)),
                ],
                (0, 0),
            )
            .effect;
            at(effect, 0)[0].facing
        };
        assert_eq!(unbased(1), sim::Facing::Still(sim::Axis::Z));
        assert_eq!(unbased(2), sim::Facing::Still(sim::Axis::Z));
        assert_eq!(unbased(5), sim::Facing::Still(sim::Axis::Z));
        assert_eq!(unbased(11), sim::Facing::Still(sim::Axis::Y));
        // The screen billboard is a base of its own, and still reads as one.
        let effect = &playing(
            &[
                life(-1.0),
                block("PrVT", &integer(8)),
                block("RBDT", &integer(5)),
            ],
            (0, 0),
        )
        .effect;
        assert_eq!(at(effect, 0)[0].facing, sim::Facing::Screen);
    }

    /// `RBDT` names the two upright bills apart, and the game turns them to meet different things.
    #[test]
    fn the_two_upright_bills_read_apart() {
        let based = |base| {
            let effect = &playing(
                &[
                    life(-1.0),
                    block("PrVT", &integer(5)),
                    block("RBDT", &integer(base)),
                ],
                (0, 0),
            )
            .effect;
            at(effect, 0)[0].facing
        };
        assert_eq!(based(4), sim::Facing::Upright(sim::Toward::Screen));
        assert_eq!(based(8), sim::Facing::Upright(sim::Toward::Eye));
    }

    /// The two bills a model takes about the world's up axis, against the world matrices the game
    /// hands its own model package: `b0025_aet1_o` over Ishgard at `RBDT 8`, which meets the eye,
    /// and `b2923_aet1_o` at Tuliyollal's aetheryte at `RBDT 4`, which meets the screen. The two
    /// are 9 degrees apart at the Ishgard camera, so neither reading can pass for the other.
    #[test]
    fn a_billed_model_turns_the_way_the_game_turns_it() {
        let eye = Vec3::new(-251.920_83, 8.874_07, 166.831_26);
        let back = -Vec3::new(0.674_39, 0.404_75, -0.617_55);
        let at = Vec3::new(-64.0, 8.53, 44.0);
        let basis = glam::Mat3::from_quat(super::billed(sim::Toward::Eye, at, eye, back));
        assert!(basis.y_axis.abs_diff_eq(Vec3::Y, 1e-6));
        assert!(basis.z_axis.abs_diff_eq(Vec3::new(-0.837_051, 0.0, 0.547_124), 1e-5));
        assert!(basis.x_axis.abs_diff_eq(Vec3::new(0.547_124, 0.0, 0.837_051), 1e-5));

        let eye = Vec3::new(-6.535_23, 18.582_67, 36.727_37);
        let back = -Vec3::new(-0.568_08, -0.186_75, -0.801_5);
        let at = Vec3::new(-24.067_76, 10.866_75, 7.599_18);
        let basis = glam::Mat3::from_quat(super::billed(sim::Toward::Screen, at, eye, back));
        assert!(basis.y_axis.abs_diff_eq(Vec3::Y, 1e-6));
        assert!(basis.z_axis.abs_diff_eq(Vec3::new(0.578_258, 0.0, 0.815_854), 1e-4));
        assert!(basis.x_axis.abs_diff_eq(Vec3::new(0.815_854, 0.0, -0.578_258), 1e-4));
    }

    /// A quad billed against the camera can carry the turn about its own normal, and that is the one
    /// the file writes as `Rot/Z`.
    #[test]
    fn a_sprite_carries_the_turn_the_billboard_can_hold() {
        let turn = nest("Rot", &[curve("Z", 0, 0, &scalars(&[(0, 1, 1.5)]))]);
        let effect = &playing(&[life(-1.0), turn], (0, 0)).effect;
        assert_eq!(at(effect, 0)[0].roll, 1.5);
    }

    #[test]
    fn a_single_axis_turn_does_not_care_about_order() {
        let turn = nest("Rot", &[curve("Z", 0, 0, &scalars(&[(0, 1, 0.77)]))]);
        let effect = &playing(&[life(-1.0), turn], (0, 0)).effect;
        let turn = glam::Quat::from_array(at(effect, 0)[0].turn);
        assert!(turn.abs_diff_eq(glam::Quat::from_rotation_z(0.77), 1e-6));
    }

    /// About more than one axis, a particle's turn composes the way `apricot_powder.shpk` builds it
    /// from `TEXCOORD.xyz`: Y outermost, then X, then Z.
    #[test]
    fn a_multi_axis_turn_composes_the_way_the_shader_does() {
        let turn = nest(
            "Rot",
            &[
                curve("X", 0, 0, &scalars(&[(0, 1, 0.4)])),
                curve("Y", 0, 0, &scalars(&[(0, 1, 0.9)])),
                curve("Z", 0, 0, &scalars(&[(0, 1, 1.3)])),
            ],
        );
        let effect = &playing(&[life(-1.0), turn], (0, 0)).effect;
        let turn = glam::Quat::from_array(at(effect, 0)[0].turn);
        let expect = glam::Quat::from_rotation_y(0.9)
            * glam::Quat::from_rotation_x(0.4)
            * glam::Quat::from_rotation_z(1.3);
        assert!(turn.abs_diff_eq(expect, 1e-6));
    }

    #[test]
    fn a_particle_samples_the_texture_its_first_layer_names() {
        let effect = &playing(&[life(-1.0), nest("TC1", &[block("TLst", &[1])])], (0, 0)).effect;
        assert_eq!(at(effect, 0)[0].texture, Some(1));
    }

    /// A timeline item with no end (`EdTm -1`) never bounds the effect, so `length` is only a
    /// fallback for scrubbing it in isolation. A particle stating no life of its own settles once
    /// its own emitter stops spawning, rather than being capped to that fallback.
    #[test]
    fn a_particle_with_no_life_settles_once_its_emitter_stops_spawning() {
        let effect = &read(&[
            nest(
                "TmLn",
                &[
                    block("TICn", &integer(1)),
                    block(
                        "Item",
                        &[
                            block("bEna", &integer(1)),
                            block("StTm", &integer(0)),
                            block("EdTm", &integer(-1)),
                            block("EmNo", &integer(0)),
                        ]
                        .concat(),
                    ),
                ],
            ),
            nest(
                "Emit",
                &[
                    block("PrCn", &integer(1)),
                    block(
                        "ItPr",
                        &[
                            block("bEnb", &integer(1)),
                            block("TgtB", &integer(0)),
                            block("CrCn", &integer(1)),
                        ]
                        .concat(),
                    ),
                    life(3.0),
                    curve("CrC", 0, 0, &scalars(&[(0, 1, 1.0)])),
                    curve("CrI", 0, 0, &scalars(&[(0, 1, 1.0)])),
                ],
            ),
            nest("Ptcl", &[life(-1.0)]),
        ])
        .effect;
        assert!(!effect.bounded);
        let settled = at(effect, 10).len();
        assert!((1..10).contains(&settled));
        assert_eq!(at(effect, 1000).len(), settled);
    }

    /// Where the emitter itself never stops either, only the pass an entry names bounds how many
    /// pile up. This one is made on the emitter's interval, so one lands every frame it comes round
    /// and nothing the file states clears them.
    #[test]
    fn a_particle_with_no_life_piles_up_where_its_emitter_never_stops_either() {
        let effect = &playing(&[life(-1.0)], (0, -1)).effect;
        assert!(!effect.bounded);
        assert_eq!(at(effect, 1000).len(), 1001);
    }

    #[test]
    fn length_is_the_latest_timeline_item_ends_not_a_particles_own_life_past_it() {
        let effect = &playing(&[life(50.0)], (0, 10)).effect;

        // The particle outlives the item that spawned it by 40 frames; the item's own end is
        // where the effect is done, not a floor its longest-lived particle can push past.
        assert_eq!(effect.length, 10);
    }
}
