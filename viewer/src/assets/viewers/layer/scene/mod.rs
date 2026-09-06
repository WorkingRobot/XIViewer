//! A zone's layers placed in space, flown through.
//!
//! Every `BgPart` is drawn at its own transform, and a `SharedGroup` is another file's tree read
//! under the transform that placed it. Composition is the ordinary one: a node's matrix is
//! translation times rotation times scale, the stored Euler triple turns about X first, then Y,
//! then Z, and a child is its parent's matrix times its own.
//!
//! Nothing is fetched up front. Files, models, materials and textures are asked for a few at a time
//! and nearest first, and the view draws whatever has arrived, so a zone fills in around the camera
//! rather than appearing at once. What is asked for at all is bounded by a load distance the user
//! sets: past it an instance is neither drawn nor fetched.
//!
//! The shading is the game's own. Every surface goes through the package its material names, into
//! the same deferred frame the model viewer draws into, and the lights the zone places are drawn as
//! the volumes its `.lcb` clips them against.

mod ambient;
mod gpu;
mod preset;
mod report;
mod sound;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::io::Cursor;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use anyhow::Result;
use egui::{Color32, RichText, ScrollArea, Sense, TextureHandle, TextureOptions};
use glam::{Mat3, Mat4, Quat, Vec3, Vec4};
use half::f16;
use ironworks::file::avfx::Avfx;
use ironworks::file::layer::{
    Colour, Glow, InstanceData, Lane, LayerGroup, LightKind, Rgba, SceneAnimation, SceneGlow,
    SceneSpin, SceneTimeline, ShadowMode, Transform,
};
use ironworks::file::mdl::ModelContainer;
use ironworks::file::shpk::ShaderPackage;
use ironworks::file::spm::ShaderParameters;
use ironworks::file::tmb;
use ironworks::file::{
    File, ggd, gzd, layer, lcb, lgb::LayerGroupFile, mtrl, sgb::SharedGroupFile, svb, tera,
};

use super::super::avfx;
use super::super::mdl;
use super::super::{facts, link, section};
use super::Source;
use crate::assets::deps::Deps;
use crate::backend::Backend;
use crate::data::DecodedTexture;
use crate::utils::{TrackedPromise, export};

use mdl::material::Material;
use mdl::program;

/// Vertical field of view.
const FOV: f32 = 55.0_f32.to_radians();

/// How wide a details grid is let grow before its value column wraps. A grid sizes to whatever its
/// widest row ever measured and never shrinks again, so left unbounded one long row drags the whole
/// panel past the cap the details side panel holds itself to. Held well under that cap rather than
/// tied to it: the panel's own chrome costs more than the difference once every row is measured.
const DETAILS_ROW_WIDTH: f32 = 250.0;

/// How deep a shared group may hold another. Files reach four; the cap guards against a cycle
/// rather than limiting anything real.
const DEPTH: u8 = 8;

/// Longest edge a scene's textures are decoded to. Smaller than the model viewer's: a zone binds
/// hundreds of materials rather than one model's handful, over the same connection.
const TEXTURE_SIZE: u16 = 256;

/// Decoded texture bytes one scene may hold. Past it the rest of its surfaces draw untextured.
const TEXTURE_BUDGET: usize = 128 << 20;

/// Longest edge a grass color map is decoded to. Over the cap above, since a map holds its tiles
/// side by side and a blade reads one of them.
const GRASS_SIZE: u16 = 1024;

/// Tiles a grass color map is laid out in, which a placement's own profile picks between.
const TILES: u8 = 8;

/// Blades one frame draws, each a quad of its own. The grids nearest the eye fill it, and the rest
/// stand undrawn until it comes closer.
const BLADES: usize = 200_000;

/// Lights the frame draws at once. Every one is a pass of its own over the volume it reaches, so a
/// zone's whole set would cost more than it shows; the nearest are kept.
const LAMPS: usize = 256;

/// The scene key deciding whether a background shader reads the normal map at all. A package
/// defaults it to off, and the variant that answer selects samples no normal map, so the frame it
/// writes is the geometry's own.
const GET_NORMAL_MAP: u32 = 0xcbdf_d5ec;
const GET_NORMAL_MAP_ON: u32 = 0xd999_4ef1;
/// The third value, which walks the normal map's blue channel as a parallax height under the
/// material's own `g_HeightScale`. Only `bg.shpk` ships a node for it - `bgprop.shpk` declares the
/// same key but has none, so asking it for this value finds no node at all.
const GET_NORMAL_MAP_PARALLAX: u32 = 0xd9fd_8a1c;

/// The scene key deciding whether a shader clips against its own alpha threshold. A package defaults
/// it to off, and the variant that answer selects carries no clip at all, so a material's cutout
/// leaves the geometry it was authored over standing.
const APPLY_ALPHA_CLIP: u32 = 0xdcfc_844e;
const APPLY_ALPHA_CLIP_ON: u32 = 0x59c4_e6db;

/// `ApplyDetailMap`, and the value that lays the tiled arrays over a surface. Left at the package's
/// own default: the game picks this per material and we have no `g_SamplerDetailColorMap`/
/// `g_SamplerDetailNormalMap` to feed it, so forcing it on tints every surface toward the grey
/// stand-in instead of leaving it off like a material that never asked for it.
const APPLY_DETAIL_MAP: u32 = 0x6313_fd87;
const APPLY_DETAIL_MAP_ON: u32 = 0x7a3d_9efd;

/// `ApplyWavingAnim`, and the value that lets the wind reach a surface. Only the models whose own
/// header allows it are drawn through the variant it selects.
const APPLY_WAVING_ANIM: u32 = 0x105c_6a52;
const APPLY_WAVING_ANIM_ON: u32 = 0xf801_b859;

/// `GetRLR`, water's own local-reflection toggle. A capture of a real frame carries it on; a
/// package defaults it off, and the variant that answer selects has no `g_SamplerReflectionMap` at
/// all rather than one nothing here fills.
const GET_RLR: u32 = 0x1143_3f2d;
const GET_RLR_ON: u32 = 0x4ba7_7904;

/// The keys the engine sets rather than the material, other than `GetNormalMap`: a package that
/// declares none of them resolves exactly as it did, since a key the package never declares is
/// never looked up.
const KEYS: [(u32, u32); 2] = [(APPLY_ALPHA_CLIP, APPLY_ALPHA_CLIP_ON), (GET_RLR, GET_RLR_ON)];

/// The engine keys this package's materials draw with. `GetNormalMap` is separate from `KEYS`
/// because only `bg.shpk` has a node for the parallax value; everything else stays on the plain
/// normal map.
fn engine_keys(package: &str, waving: bool) -> Vec<(u32, u32)> {
    let mut keys = KEYS.to_vec();
    keys.push((
        GET_NORMAL_MAP,
        match package.ends_with("/bg.shpk") {
            true => GET_NORMAL_MAP_PARALLAX,
            false => GET_NORMAL_MAP_ON,
        },
    ));
    if waving {
        keys.push((APPLY_WAVING_ANIM, APPLY_WAVING_ANIM_ON));
    }
    keys
}

/// How large a box a light is drawn as where the zone states none for it.
const REACH: f32 = 6.0;

/// How fast a shared group's timeline runs. An animation pack states one span both in seconds and
/// in these, and the two agree on thirty a second.
const TICKS: f32 = 30.0;

/// Requests of each kind in flight at once.
const FILES: usize = 12;
const PACKAGES: usize = 4;
const MODELS: usize = 24;
const MATERIALS: usize = 16;
const TEXTURES: usize = 24;

/// The share of a frame given to parsing files and decoding models, and the least that share is
/// worth whatever the frame costs. Both happen on the thread that draws, and a zone holds files
/// from a few hundred bytes to a few megabytes, so what bounds them is time rather than a count.
const SHARE: f32 = 0.3;
const LEAST: Duration = Duration::from_millis(6);

/// How far the eye moves before the instance buffers are written again.
const STEP: f32 = 8.0;

/// How large an instance has to look to be worth its highest detail level, and its middle one, as a
/// fraction of the distance to it.
const DETAIL: [f32; 2] = [0.04, 0.012];

/// What the load distance may be set to, and where it starts.
const NEAREST: f32 = 400.0;
const FURTHEST: f32 = 16000.0;
const LOADED: f32 = 4000.0;

/// How far the eye travels a second, before the user's multiplier.
const SPEED: f32 = 100.0;

/// How much of the fitted reach the opening view stands back by.
const MARGIN: f32 = 1.4;

/// The share of instances a fit is taken over. A zone holds placements a million units from
/// anything else, and a fit that covered them would leave the zone itself a speck.
const BULK: f32 = 0.9;

/// How far a terrain plate reaches, for culling. Nothing in the terrain file states it, and an
/// overestimate only loads a plate sooner than it needs to.
const PLATE: f32 = 128.0;

/// How many grass grids are asked for at once. A zone sorts hundreds of them and each is small, so
/// what this bounds is how much of the fetch budget grass takes from the models and materials.
const GRIDS: usize = 4;

/// One layer of one of the scene's files, as the picker offers it.
struct Layer {
    name: String,
    /// The file it came from, where a level merged several.
    origin: Option<String>,
    /// What the file says about whether it draws, which is what the picker starts at.
    visible: bool,
    festival: u16,
    shown: bool,
    placements: usize,
}

/// A placement one or more timelines move rather than leaving where the file put it: each motion
/// along the way with whatever fixed transform stands in front of it, and the tail below the last.
///
/// A chain rather than a single motion, since a group a timeline turns can hold another the same
/// timeline system turns again: composing them is what keeps a part turning with its parent instead
/// of against it.
struct Driven {
    chain: Vec<(usize, Mat4)>,
    tail: Mat4,
}

/// What moves a shared group's node, whichever way it is stated. Every span is in the ticks a
/// timeline is keyed in.
enum Motion {
    /// A timeline's nine curves over its span, which state where the node stands outright.
    Keyed {
        curves: Vec<(tmb::Channel, tmb::Curve)>,
        duration: f32,
        looping: bool,
    },
    /// Transforms a timeline states outright with no curve to play them over, each holding from the
    /// time its command runs. Where the node stands before the first is where the file put it.
    Placed {
        placement: Transform,
        steps: Vec<(f32, Mat4)>,
        duration: f32,
        looping: bool,
    },
    /// A swing the scene repeats with no timeline to play it, on top of where the file placed it.
    Repeat {
        placement: Transform,
        translation: Lane,
        rotation: Lane,
        scale: Lane,
    },
    /// A turn about one axis the scene never stops, on top of where the file placed it. The period
    /// is signed, and runs the turn the other way where it is negative.
    Spin {
        placement: Transform,
        axis: Vec3,
        period: f32,
    },
}

/// The lane a scene cycles one of its own instances through, where it names one and gives it a
/// colour of its own.
fn tinted(glows: &[SceneGlow], instance: u32, lane: fn(&SceneGlow) -> Glow) -> Option<Glow> {
    glows
        .iter()
        .find(|held| held.instances().contains(&instance))
        .map(lane)
        .filter(|held| held.active() && held.tints())
}

/// The colour a cycled lane stands at, and the strength it is taken at, from the two ends it names
/// and the ticks it swings between them in.
///
/// Swung out and back rather than started over: nothing in the file says which of the two it is.
fn cycled(lane: Glow, time: f32) -> (Vec3, f32) {
    let rgb = |held: Colour| {
        Vec3::new(
            f32::from(held.red()),
            f32::from(held.green()),
            f32::from(held.blue()),
        ) / 255.0
    };
    let (from, to) = (lane.from(), lane.to());
    let along = phase(lane.period(), 0, 1, time);
    (
        rgb(from).lerp(rgb(to), along),
        from.intensity() + (to.intensity() - from.intensity()) * along,
    )
}

/// How far into its span a timeline stands. One that does not loop plays through once and holds
/// where it ended, which is what the file's own flag asks for.
fn along(time: f32, span: f32, looping: bool) -> f32 {
    match looping {
        true => time.rem_euclid(span),
        false => time.min(span),
    }
}

/// How far along its swing a repeating lane stands, from nought at rest to one at full reach. A
/// lane that wraps at nought starts over, which is what a whole turn wants; the rest swing back.
fn phase(period: u32, delay: u32, wrap: u32, time: f32) -> f32 {
    let along = (time - delay as f32).max(0.0) / period.max(1) as f32;
    match wrap {
        0 => along.fract(),
        _ => 1.0 - (along.rem_euclid(2.0) - 1.0).abs(),
    }
}

impl Motion {
    /// Where the node stands at a time.
    fn at(&self, time: f32) -> Mat4 {
        match self {
            Self::Keyed {
                curves,
                duration,
                looping,
            } => {
                let span = duration.max(1.0);
                let along = along(time, span, *looping);
                let mut turn = Vec3::ZERO;
                let mut shift = Vec3::ZERO;
                let mut size = Vec3::ONE;
                for (channel, curve) in curves {
                    let Some(held) = curve.at(along) else {
                        continue;
                    };
                    let lane = |into: &mut Vec3, at: usize| into[at] = held;
                    match channel {
                        tmb::Channel::TranslationX => lane(&mut shift, 0),
                        tmb::Channel::TranslationY => lane(&mut shift, 1),
                        tmb::Channel::TranslationZ => lane(&mut shift, 2),
                        tmb::Channel::RotationX => lane(&mut turn, 0),
                        tmb::Channel::RotationY => lane(&mut turn, 1),
                        tmb::Channel::RotationZ => lane(&mut turn, 2),
                        tmb::Channel::ScaleX => lane(&mut size, 0),
                        tmb::Channel::ScaleY => lane(&mut size, 1),
                        tmb::Channel::ScaleZ => lane(&mut size, 2),
                    }
                }
                Mat4::from_scale_rotation_translation(
                    size,
                    Quat::from_euler(
                        glam::EulerRot::ZYX,
                        turn.z.to_radians(),
                        turn.y.to_radians(),
                        turn.x.to_radians(),
                    ),
                    shift,
                )
            }
            Self::Placed {
                placement,
                steps,
                duration,
                looping,
            } => {
                let along = along(time, duration.max(1.0), *looping);
                steps
                    .iter()
                    .rev()
                    .find(|(at, _)| *at <= along)
                    .map_or_else(|| matrix(*placement), |(_, held)| *held)
            }
            Self::Repeat {
                placement,
                translation,
                rotation: spin,
                scale,
            } => {
                let reach = |lane: Lane| {
                    let held = lane.amount();
                    Vec3::new(held[0], held[1], held[2])
                        * phase(lane.period(), lane.delay(), lane.wrap(), time)
                };
                let shift = match translation.active() {
                    true => reach(*translation),
                    false => Vec3::ZERO,
                };
                let turn = match spin.active() {
                    true => reach(*spin),
                    false => Vec3::ZERO,
                };
                // A scale rests at one rather than at nought, so its lane reaches towards what it
                // states instead of adding to it.
                let size = match scale.active() {
                    true => Vec3::ONE.lerp(
                        Vec3::from_slice(&scale.amount()[..3]),
                        phase(scale.period(), scale.delay(), scale.wrap(), time),
                    ),
                    false => Vec3::ONE,
                };
                Mat4::from_scale_rotation_translation(
                    Vec3::from_array(placement.scale()) * size,
                    Quat::from_mat3(&rotation(placement.rotation()))
                        * Quat::from_euler(glam::EulerRot::ZYX, turn.z, turn.y, turn.x),
                    Vec3::from_array(placement.translation()) + shift,
                )
            }
            Self::Spin {
                placement,
                axis,
                period,
            } => {
                // Kept as a fraction of a whole turn rather than wrapped in radians, so a negative
                // period runs backwards instead of mirroring.
                let along = (time / period).fract();
                Mat4::from_scale_rotation_translation(
                    Vec3::from_array(placement.scale()),
                    Quat::from_mat3(&rotation(placement.rotation()))
                        * Quat::from_axis_angle(*axis, along * std::f32::consts::TAU),
                    Vec3::from_array(placement.translation()),
                )
            }
        }
    }
}

/// One `BgPart`, in world space.
#[derive(Clone)]
struct Placement {
    model: usize,
    transform: Mat4,
    /// Set where a timeline moves this, in which case the transform above is only where it starts.
    driven: Option<Rc<Driven>>,
    center: Vec3,
    /// The instance's own bounding sphere, which the file states in world units.
    radius: f32,
    /// Past this an instance stops drawing whatever the load distance is. Zero means never.
    fade: f32,
    layer: usize,
    /// How the zone's own `.svb` reaches this part, the way an `.lcb` reaches a light.
    key: (u32, [u8; 4]),
    /// Set where the scene cycles the colour its material's emissive is taken at.
    glow: Option<Glow>,
    /// Whether the sun's own pass draws this at all, which the instance states for itself.
    casts: bool,
    /// Where a `.ggd` placement starts in the wind cycle, over `0.0..=1.0`. Nothing but grass states
    /// this, so a layer group placement carries none and the instance falls back to a guess.
    wind_phase: Option<f32>,
}

enum State {
    /// Wanted, but nothing has been asked for yet.
    Wanted,
    Fetching(TrackedPromise<Result<(Vec<u8>, u8)>>),
    Decoding(Vec<u8>, u8),
    Ready,
    Failed,
}

struct Model {
    path: String,
    state: State,
    /// Which detail levels hold geometry.
    drawn: [bool; 3],
    /// Per detail level, the scene material each of its meshes uses.
    meshes: Vec<Vec<usize>>,
    /// Whether the wind may reach it, which its own header states.
    waving: bool,
    /// Whether the sun's pass draws it, which its own header states as well.
    casts: bool,
    /// Placements drawing this model.
    instances: usize,
    /// How far the nearest of them was at the last rebuild, which is the order models are asked for
    /// in.
    nearest: f32,
    /// The finest detail level any of them would draw, and the level last asked for. A file is read
    /// again only where the eye has come close enough to want more of it than was taken.
    finest: u8,
    asked: u8,
}

enum Slot {
    Wanted,
    Fetching(TrackedPromise<Result<Vec<u8>>>),
    Ready(Box<Material>),
    Failed,
}

/// A shader package, which many materials name the same one of. A fetch answers whether the
/// bytecode came with it.
enum Package {
    Wanted,
    Fetching(TrackedPromise<Result<(Vec<u8>, bool)>>),
    Ready(Vec<u8>),
    Failed,
}

/// The blobs one read of a package answered with, each under the shader it belongs to.
type Blobbed = TrackedPromise<Result<Vec<(u32, Vec<u8>)>>>;

/// A package whose bytecode was left behind: where each of its shaders' blobs sits in the file, and
/// which of them the surfaces read so far have asked for.
struct Blobs {
    spans: Vec<std::ops::Range<u32>>,
    arrived: BTreeSet<u32>,
    wanted: BTreeSet<u32>,
    fetching: Option<Blobbed>,
}

impl Blobs {
    fn read(package: &ShaderPackage) -> Self {
        let base = package.blobs_offset() as u32;
        Self {
            spans: package
                .shaders()
                .iter()
                .map(|shader| {
                    let at = base + shader.blob_offset();
                    at..at + shader.blob_size()
                })
                .collect(),
            arrived: BTreeSet::new(),
            wanted: BTreeSet::new(),
            fetching: None,
        }
    }
}

/// The draws a zone makes of a surface, which is what a package is read for. A shader outside these
/// is never translated, so its bytecode is never asked for.
const DRAWS: [(program::Pass, u32); 8] = [
    (program::Pass::Buffer, program::SUB_VIEW_MAIN),
    (program::Pass::Blended, program::SUB_VIEW_MAIN),
    (program::Pass::Depth, program::SUB_VIEW_MAIN),
    (program::Pass::Depth, program::SUB_VIEW_SHADOW_0),
    (program::Pass::Water, program::SUB_VIEW_MAIN),
    (program::Pass::BlendedLighting, program::SUB_VIEW_MAIN),
    (program::Pass::Shaft, program::SUB_VIEW_MAIN),
    (program::Pass::Layer, program::SUB_VIEW_MAIN),
];

/// Whether a package is one of the surfaces that blend themselves into the frame.
fn wet_name(held: &str) -> bool {
    [
        "water.shpk",
        "river.shpk",
        "crystal.shpk",
        "lightshaft.shpk",
        "verticalfog.shpk",
    ]
    .iter()
    .any(|one| held.ends_with(one))
}

/// One material's shaders, and how much of the G-buffer they were translated for.
struct Translated {
    attachments: usize,
    buffer: Vec<Arc<program::Program>>,
    depth: Option<Arc<program::Program>>,
    shadow: Option<Arc<program::Program>>,
    resolve: Option<Arc<program::Program>>,
    sheer: Option<(Arc<program::Program>, Arc<program::Program>)>,
}

/// One light the zone places. The box it is clipped against is stated in its own space, so the
/// placement carries where it stands and the box how far it carries: a zone that cuts none states
/// this fallback instead, which is what the reach is worked out from either way.
struct Light {
    placement: Mat4,
    min: Vec3,
    max: Vec3,
    /// Which of its package's falloff variants shades it.
    falloff: usize,
    /// The range its record states, which its falloff is divided by.
    range: f32,
    center: Vec3,
    color: Vec3,
    kind: program::LampKind,
    /// Which way it throws, in world space.
    direction: Vec3,
    /// The cosines its cone is full strength within and cut at.
    inner: f32,
    cone: f32,
    /// How the zone's own `.lcb` reaches this light: the instance at the top of the tree, then an
    /// index per shared group under it.
    key: (u32, [u8; 4]),
    /// Set where the scene cycles its colour, in which case the colour above is only where the
    /// file left it.
    glow: Option<Glow>,
}

/// A placed `.avfx` glow: where it stands, which file names it, and the placement's own settings
/// on top of what the file itself draws.
struct Vfx {
    placement: Mat4,
    /// Set where a timeline moves this, in which case the placement above is only where it starts.
    driven: Option<Rc<Driven>>,
    path: String,
    layer: usize,
    key: (u32, [u8; 4]),
    tint: Vec4,
    /// Distances over which the effect ramps in near the camera and back out far from it. Only 10
    /// of 7,654 corpus placements open a real near range (almost always starting at 0, the camera's
    /// own near clip), against 3,554 with a real far one, so the far pair is where a cull actually
    /// happens; `no_far_clip` turns it off regardless of what the pair states.
    fade_near: [f32; 2],
    fade_far: [f32; 2],
    no_far_clip: bool,
}

/// Linear ramp from invisible at `near[0]` to full opacity at `near[1]`, or always visible where
/// the pair does not open a real range.
fn near_fade(distance: f32, near: [f32; 2]) -> f32 {
    let [start, end] = near;
    match end > start {
        true => ((distance - start) / (end - start)).clamp(0.0, 1.0),
        false => 1.0,
    }
}

/// Linear ramp from full opacity at `far[0]` down to invisible at `far[1]`, or always visible where
/// the pair does not open a real range.
fn far_fade(distance: f32, far: [f32; 2]) -> f32 {
    let [start, end] = far;
    match end > start {
        true => (1.0 - (distance - start) / (end - start)).clamp(0.0, 1.0),
        false => 1.0,
    }
}

/// The placement's colour override, which the corpus leaves at opaque white on 72.1% of
/// placements: that is already the identity multiply, so no sentinel case is needed.
fn vfx_tint(colour: Rgba) -> Vec4 {
    Vec4::new(
        colour.red() as f32 / 255.0,
        colour.green() as f32 / 255.0,
        colour.blue() as f32 / 255.0,
        colour.alpha() as f32 / 255.0,
    )
}

enum EffectState {
    Wanted,
    Fetching(TrackedPromise<Result<Vec<u8>>>),
    /// The clock reading the fetch resolved at, so the effect's own timeline starts at zero
    /// wherever real time happened to be when it arrived rather than at the clock's own zero.
    Ready(avfx::sim::Effect, avfx::sim::State, i32),
    Failed,
}

/// A `.avfx` some placement names, read and stepped once regardless of how many instances place it:
/// the particles it steps are the same wherever a copy of the file stands.
struct Effect {
    path: String,
    state: EffectState,
}

/// A file the scene names beside itself and reads once: the boxes its lights are clipped against,
/// how much of the sky reaches each of its parts, and the game's own textures its shaders read.
enum Aside {
    Wanted(String),
    Fetching(String, TrackedPromise<Result<Vec<u8>>>),
    Done,
}

enum Texture {
    Fetching(TrackedPromise<Result<DecodedTexture>>),
    Ready(TextureHandle),
    Absent,
}

/// A texture a material reads through a sampler declared over slices. Read whole and handed to the
/// graph rather than to egui, which holds nothing but planes.
enum Stack {
    Fetching(TrackedPromise<Result<Vec<u8>>>),
    Ready,
    Absent,
}

/// A file the scene still has to read placements out of.
enum Held {
    Fetching(TrackedPromise<Result<Vec<u8>>>),
    Parsing(Vec<u8>),
    Ready(Rc<Source>),
    Failed,
}

/// The ground, which no layer places: it is tiled from the plates a `.tera` beside the zone's
/// layer groups lists.
enum Terrain {
    Wanted(String),
    Fetching(String, TrackedPromise<Result<Vec<u8>>>),
    Done,
}

/// The grass, which no layer places either: a zone file beside the layer groups names the models
/// and sorts the grids, and a grid file per cell holds the placements themselves.
enum Grass {
    Wanted(String),
    Fetching(String, TrackedPromise<Result<Vec<u8>>>),
    Placing(Box<Placing>),
    Done,
}

struct Placing {
    directory: String,
    /// The scene's model for each grass slot, in the order the zone names them.
    models: Vec<usize>,
    /// The color map each auto layer's blades are cut out of, where the zone names one.
    maps: Vec<String>,
    grids: Vec<Patch>,
    layer: usize,
}

/// One grid's blades at one auto layer, as the scene stood them up.
struct Turf {
    origin: Vec3,
    radius: f32,
    layer: usize,
    blades: usize,
}

/// One grid of grass, which is only asked for once the eye reaches the sphere the zone sorts it by.
struct Patch {
    center: Vec3,
    radius: f32,
    file: String,
    fetch: Option<TrackedPromise<Result<Vec<u8>>>>,
    taken: bool,
}

/// A file named but not yet walked.
struct Expand {
    path: String,
    transform: Mat4,
    /// How an `.lcb` entry reaches into this subtree.
    key: (u32, [u8; 4]),
    /// The largest the transform above scales by, which the bounding spheres underneath it grow by.
    scale: f32,
    /// The layer everything found belongs to. A level names whole layer groups rather than placing
    /// anything itself, so what it names brings layers of its own.
    layer: Option<usize>,
    depth: u8,
    /// The motions this subtree hangs under, each with the fixed transform in front of it.
    chain: Vec<(usize, Mat4)>,
    /// What has accumulated since the last of them, which the walk goes on adding to.
    since: Mat4,
}

#[derive(Clone, Copy)]
struct Camera {
    position: Vec3,
    yaw: f32,
    pitch: f32,
}

impl Camera {
    fn forward(&self) -> Vec3 {
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        Vec3::new(cos_pitch * sin_yaw, sin_pitch, cos_pitch * cos_yaw)
    }

    fn right(&self) -> Vec3 {
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        Vec3::new(-cos_yaw, 0.0, sin_yaw)
    }
}

/// A camera driven for one frame by a host outside this view, in place of the free orbit camera.
/// `near` and `far` here are only the projection's clip planes; the world streaming distance stays
/// keyed on the load distance regardless of what a cutscene states.
pub struct Drive {
    pub position: Vec3,
    pub forward: Vec3,
    pub up: Vec3,
    /// Vertical, and stated for a 16:9 frame: [`refit_16_9_fov`] carries it to the viewport's own
    /// aspect.
    pub fov_degrees: f32,
    pub near: f32,
    pub far: f32,
}

/// The asset a [`Prop`] draws itself from.
pub enum Asset {
    /// A model, placed as one instance.
    Model(String),
    /// A shared group, expanded the way the scene expands one of its own.
    Group(String),
}

/// A character a host outside this view stands in the scene. It is assembled rather than placed:
/// what the scene draws of it is the model's own geometry, filling the same G-buffer as everything
/// else in the frame.
pub struct Standing {
    pub model: Rc<mdl::Rendered>,
    /// Where it stands, read the same way a layer group's own instances are.
    pub at: Transform,
    /// How much of it is drawn, which its own dither clip tests each pixel against.
    pub opacity: f32,
}

/// One effect a host outside this view is running: which file, where it stands, how far into its
/// own timeline it now is, and the color to draw it through. The id is the firing's own, so a
/// host that hands the list over again every frame keeps each firing's particles.
pub struct Fired {
    pub id: u64,
    pub path: String,
    pub at: Mat4,
    pub frame: i32,
    pub tint: Vec4,
}

/// Something placed in the scene by a host outside this view, alongside what the level itself
/// holds.
pub struct Prop {
    pub asset: Asset,
    /// Where it stands, read the same way a layer group's own instances are.
    pub transform: Transform,
    /// The instance it stands for, which is the key the level's `.svb` and `.lcb` are read by.
    pub id: u32,
}

/// A 16:9-authored vertical field of view, carried to another aspect ratio by keeping its
/// horizontal field rather than its vertical one.
fn refit_16_9_fov(vertical_degrees: f32, aspect: f32) -> f32 {
    const AUTHORED: f32 = 16.0 / 9.0;
    let half_horizontal = (vertical_degrees.to_radians() * 0.5).tan() * AUTHORED;
    2.0 * (half_horizontal / aspect).atan().to_degrees()
}

pub struct Scene {
    camera: Camera,
    home: Camera,
    /// A camera a host outside this view wants for the next frame, in place of the free orbit
    /// camera. Taken (and so cleared) as soon as that frame draws, so a host that stops calling
    /// [`Scene::drive`] hands control back to the free camera on its very next frame.
    drive: Option<Drive>,
    /// Markers a host outside this view wants drawn over the frame, in scene space with a label.
    /// Cleared the same way [`Self::drive`] is.
    markers: Vec<(Vec3, String)>,
    /// The characters a host outside this view wants standing in the scene. Held rather than
    /// cleared each frame: a model keeps asking for what it still needs, and the host hands its
    /// cast over once.
    cast: Vec<Standing>,
    /// The props a host outside this view has taken out of the frame, by the ids it placed them
    /// under. A shared group's own placements carry the id it was placed under as well, so hiding
    /// one takes the whole subtree with it.
    unplaced: BTreeSet<u32>,
    /// Whether the last frame drawn was driven, for the side panel to grey its own camera controls
    /// against: [`Self::drive`] itself is forgotten the instant a frame reads it.
    driving: bool,
    /// The level this view was opened for, which is what a preset's own is checked against.
    path: String,
    /// The last TitleEdit preset read, which is where a capture was taken from.
    preset: Option<preset::Preset>,
    /// A preset being picked or written, since a file dialog answers a frame or more later. Held
    /// rather than forgotten: dropping a promise cancels the future behind it.
    picking: Option<TrackedPromise<Option<Vec<u8>>>>,
    /// A preset pasted in whole, for a window nothing can open a file dialog over.
    pasted: String,
    saving: Option<TrackedPromise<()>>,
    /// Where the eye stood when the instance buffers were last written.
    written: Vec3,
    dirty: bool,
    load: f32,
    speed: f32,
    fov: f32,
    layers: Vec<Layer>,
    placements: Vec<Placement>,
    /// The placement the pointer last landed on, which the overlay outlines and the panel reads.
    selected: Option<usize>,
    models: Vec<Model>,
    model_at: HashMap<String, usize>,
    materials: Vec<(String, Slot)>,
    material_at: HashMap<String, usize>,
    packages: HashMap<String, Package>,
    /// The bytecode still owed on the packages a store served in part.
    blobs: HashMap<String, Blobs>,
    /// Materials whose own shaders have been asked for, so a package is only read again when a new
    /// one names it.
    picked: HashSet<usize>,
    /// Parameter files folded into the table the card holds, so a later one uploads it again.
    typed: usize,
    translated: HashMap<(usize, bool), Translated>,
    tables: HashMap<usize, Arc<(Vec<u16>, usize, usize)>>,
    lighting: Option<Arc<mdl::gpu::Lighting>>,
    /// The chain that works the frame's brightness out and reads it back through a curve, once its
    /// six shaders have arrived. Absent where the environment states no tone mapping of its own.
    exposure: Option<Arc<mdl::gpu::Exposure>>,
    /// The pass that fills whatever the frame did not cover, and the size and resource id of the
    /// volume it reads: a sky is addressed by its own texel centers, so the pass needs its shape.
    skybox: Option<Arc<program::Program>>,
    sunlight: Option<Arc<program::Program>>,
    moonlight: Option<Arc<program::Program>>,
    /// The pass that fades a distant pixel toward the weather's own fog and then toward that sky.
    haze: Option<Arc<program::Program>>,
    /// The two cloud draws, the band first, and the texture each reads: the weather names one per
    /// mesh by id, so moving the hour or the weather fetches the next.
    clouds: [Option<Arc<program::Program>>; 2],
    /// The sheet drawn again from where the sun stands, and the blur the map it fills is left in.
    cloud_shadow: Option<(Arc<program::Program>, Arc<program::Program>)>,
    cloud_files: [Aside; 2],
    cloud_wanted: [Option<u16>; 2],
    /// The night star field's tier 0, its own two `.shcd` translated into one program, and its three
    /// textures: fixed paths rather than ones the weather names, so unlike the cloud files these are
    /// only ever asked for once.
    starlight: Option<Arc<program::Program>>,
    star_files: [Aside; 3],
    star_wanted: bool,
    sky_volume: Option<(u32, (f32, f32), f32)>,
    /// The sky the volume was fetched for, so moving the picker fetches the next one.
    sky_wanted: Option<u16>,
    sky_file: Aside,
    /// The chain that spreads the bright end of the frame into a halo.
    glare: Option<Arc<mdl::gpu::Glare>>,
    /// The pair that smooths its edges, and the chain that works out how much sky reaches a pixel.
    smoothing: Option<Arc<mdl::gpu::Smoothing>>,
    occlusion: Option<Arc<mdl::gpu::Occlusion>>,
    /// The one that darkens its corners, and what the passes past the composite are run with.
    vignette: Option<Arc<program::Program>>,
    reflection: Option<Arc<mdl::deferred::Reflection>>,
    /// The chain that fills what water reflects itself through, which is not that one.
    water_mirror: Option<Arc<mdl::deferred::WaterMirror>>,
    look: program::Look,
    ambient: ambient::Ambient,
    lights: Vec<Light>,
    effects: Vec<Vfx>,
    /// The effects a host outside this view is running, replaced every frame the way
    /// [`Scene::stand`]'s cast is, and the particles each firing has run out.
    fired: Vec<Fired>,
    firing: HashMap<u64, avfx::sim::State>,
    effect_files: Vec<Effect>,
    effect_at: HashMap<String, usize>,
    sound: sound::SoundStage,
    /// The two apricot packages every effect is drawn with, fetched once for the whole scene.
    effect_shape: Option<avfx::Package>,
    effect_model: Option<avfx::Package>,
    effect_packages: Arc<avfx::gpu::Packages>,
    /// The box each light is clipped against, by the key its `.lcb` entry uses.
    clips: HashMap<(u32, [u8; 4]), (Vec3, Vec3)>,
    clip: Aside,
    /// How much of the sky reaches each part, by the key its `.svb` entry uses.
    visibility: HashMap<(u32, [u8; 4]), f32>,
    sky: Aside,
    /// The engine's own textures, by resource id. The ramp every placed light reads its falloff off
    /// is wanted from the start, since the lighting passes read it whatever a zone holds; the rest
    /// are only worth their fetch once a material's own shaders turn out to declare one.
    engine: BTreeMap<u32, Aside>,
    textures: BTreeMap<String, Texture>,
    /// The same, for the ones read through a sampler with slices. Keyed by an `Arc` so a surface
    /// built every frame names one without copying the path.
    stacked: BTreeMap<Arc<str>, Stack>,
    resident: usize,
    files: HashMap<String, Held>,
    waiting: Vec<Expand>,
    terrain: Terrain,
    grass: Grass,
    /// The two readings the grass is drawn with, once its package has arrived.
    sward: Option<Arc<gpu::Grass>>,
    turf: Vec<Turf>,
    /// The quads those stand as, and how many of them the last frame drew.
    blades: usize,
    standing: usize,
    /// Placements the view was last framed over, so a scene that arrived empty frames itself once
    /// its first file lands rather than leaving the camera at the origin.
    fitted: usize,
    renderer: Arc<Mutex<gpu::Renderer>>,
    /// Where each model stands at each detail level, as the last rebuild left them. The ones that
    /// cast lead, so the sun's pass takes a prefix of the same records rather than a list of its own.
    placed: Vec<[Vec<program::Instance>; 3]>,
    /// How long that prefix is.
    casts: Vec<[usize; 3]>,
    /// What the zone's shared groups animate, and how far along their timelines it stands, in the
    /// ticks a timeline is keyed in.
    motions: Vec<Motion>,
    /// Whether anything the zone places has a colour the clock moves, which a scene holding no
    /// motion at all can still have.
    cycling: bool,
    clock: f32,
    /// Placements the last rebuild would have drawn had their model arrived.
    absent: usize,
}

pub fn rotation(angles: [f32; 3]) -> Mat3 {
    Mat3::from_rotation_z(angles[2])
        * Mat3::from_rotation_y(angles[1])
        * Mat3::from_rotation_x(angles[0])
}

/// How an `.lcb` entry reaches one instance: the key of whatever stands at the top of the tree, then
/// an index per shared group under it.
fn reach(key: (u32, [u8; 4]), depth: u8, id: u32) -> (u32, [u8; 4]) {
    if depth == 0 {
        return (id, [0; 4]);
    }
    let mut held = key.1;
    if let Some(slot) = held.get_mut(usize::from(depth) - 1) {
        *slot = id as u8;
    }
    (key.0, held)
}

/// A `.lcb`/`.svb` key that finds nothing may still be inside a shared group the file only states
/// one answer for as a whole: sky visibility and a light's clip box apply to everything a group
/// places unless a deeper entry overrides part of it. A miss steps back through the membership path
/// one shared group at a time, ending at the root's own key, before giving up.
fn reached<V>(map: &HashMap<(u32, [u8; 4]), V>, mut key: (u32, [u8; 4])) -> Option<&V> {
    loop {
        if let Some(value) = map.get(&key) {
            return Some(value);
        }
        let at = key.1.iter().rposition(|byte| *byte != 0)?;
        key.1[at] = 0;
    }
}

/// The variant of its own package a light is shaded by, which is that same power: the corpus states
/// one, two or three and each package holds a shader for each.
fn falloff(attenuation: f32) -> usize {
    (attenuation as usize).clamp(1, program::ATTENUATION.len()) - 1
}

/// The quad one auto-layer placement stands as, measured from its grid's own origin. The blade
/// itself is stated in no file: what the grid holds is where each stands, how far it is turned, and
/// how wide and tall it is.
fn blade(placement: &ggd::Placement, into: &mut Vec<gpu::Corner>) {
    let turn = Quat::from_array(placement.rotation());
    let across = turn * Vec3::X * placement.scale_xz() * 0.5;
    let up = turn * Vec3::Y * placement.scale_y();
    let foot = Vec3::from_array(placement.position());
    let tile = 1.0 / f32::from(TILES);
    let column = f32::from(placement.profile() % TILES) * tile;
    let half = |values: [f32; 4]| values.map(f16::from_f32);
    for (side, height, u, v) in [
        (-1.0, 1.0, 0.0, 0.0),
        (1.0, 1.0, tile, 0.0),
        (-1.0, 0.0, 0.0, 1.0),
        (1.0, 0.0, tile, 1.0),
    ] {
        let at = foot + across * side + up * height;
        into.push(gpu::Corner {
            position: half([at.x, at.y, at.z, 0.0]),
            // .z is the bend weight the waving shader multiplies by `1 - v.y`; the game's own blade
            // subdivides that weight across three rows, but a flat quad has only the one, so it is
            // left at full rather than zero, which would leave the blade standing still regardless
            // of everything else.
            uv: half([u, v, 1.0, 0.0]),
            // .y is the phase the waving shader starts this blade's sway at, in radians: grass.shpk
            // adds it straight into a sine with no 2pi of its own, unlike bg.shpk's waving shader. A
            // whole grid shares one clock, so this is the only thing keeping neighbouring blades out
            // of step.
            color1: half([column, placement.wind_phase() * std::f32::consts::TAU, 0.0, 0.0]),
            // Nought weight, so the albedo is the color map's own texel: the map the tint would be
            // read off is the engine's and no file names it.
            color: [0; 4],
        });
    }
}

/// A file the scene names beside itself, wanted where it names one.
fn aside(path: Option<&String>) -> Aside {
    match path {
        Some(path) if !path.is_empty() => Aside::Wanted(path.clone()),
        _ => Aside::Done,
    }
}

pub fn matrix(transform: Transform) -> Mat4 {
    Mat4::from_scale_rotation_translation(
        Vec3::from_array(transform.scale()),
        Quat::from_mat3(&rotation(transform.rotation())),
        Vec3::from_array(transform.translation()),
    )
}

/// The detail level to draw an instance at, given how much of the view it covers. A model missing
/// the level it asked for falls back to the nearest one it has.
fn level(drawn: [bool; 3], apparent: f32) -> Option<usize> {
    let wanted = usize::from(detail(apparent));
    (wanted..3)
        .chain((0..wanted).rev())
        .find(|level| drawn[*level])
}

/// The detail level something this size on screen is drawn at, before what the file holds is known.
fn detail(apparent: f32) -> u8 {
    match apparent {
        size if size > DETAIL[0] => 0,
        size if size > DETAIL[1] => 1,
        _ => 2,
    }
}

/// A point the bulk of the placements sit around, and how far out that bulk reaches, from medians
/// rather than extremes.
fn bulk(points: &[Vec3]) -> (Vec3, f32) {
    if points.is_empty() {
        return (Vec3::ZERO, 100.0);
    }
    let order = |a: &f32, b: &f32| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);
    let median = |axis: fn(&Vec3) -> f32| {
        let mut values: Vec<f32> = points.iter().map(axis).collect();
        values.sort_by(order);
        values[values.len() / 2]
    };
    let center = Vec3::new(median(|at| at.x), median(|at| at.y), median(|at| at.z));
    let mut spans: Vec<f32> = points.iter().map(|at| (*at - center).length()).collect();
    spans.sort_by(order);
    let reach = spans[((spans.len() - 1) as f32 * BULK) as usize].max(10.0);
    (center, reach)
}

/// The nearest of the spheres the ray runs through. Everything the eye stands inside is met at
/// nought, so the tighter of those wins rather than whichever the file placed first.
fn nearest(
    from: Vec3,
    along: Vec3,
    spheres: impl Iterator<Item = (usize, Vec3, f32)>,
) -> Option<usize> {
    let mut found: Option<((f32, f32), usize)> = None;
    for (at, center, radius) in spheres {
        let Some(hit) = pierced(from, along, center, radius) else {
            continue;
        };
        let key = (hit, (center - from).length());
        if found.is_none_or(|(held, _)| key < held) {
            found = Some((key, at));
        }
    }
    found.map(|(_, at)| at)
}

/// How far along the ray a sphere is first met, or nought where the ray starts inside it.
fn pierced(from: Vec3, along: Vec3, center: Vec3, radius: f32) -> Option<f32> {
    let toward = center - from;
    let ahead = toward.dot(along);
    let off = toward.length_squared() - ahead * ahead;
    if off > radius * radius {
        return None;
    }
    let reach = (radius * radius - off).sqrt();
    match ahead - reach {
        held if held >= 0.0 => Some(held),
        _ if ahead + reach >= 0.0 => Some(0.0),
        _ => None,
    }
}

fn looking_at(center: Vec3, reach: f32) -> Camera {
    let back = reach * MARGIN;
    let position = center + Vec3::new(0.0, back * 0.45, -back);
    let to = center - position;
    Camera {
        position,
        yaw: to.x.atan2(to.z),
        pitch: to.y.atan2((to.x * to.x + to.z * to.z).sqrt()),
    }
}

impl Scene {
    pub(super) fn new(path: &str, source: &Source) -> Self {
        // Both files a zone is entered through sit in its `level` directory, and its ground sits in
        // `bgplate` beside that. A shared group states a zone root of its own but has no ground.
        let root = path.split_once("/level/").map(|(root, _)| root);
        let home = looking_at(Vec3::ZERO, 100.0);
        let mut scene = Self {
            camera: home,
            home,
            drive: None,
            markers: Vec::new(),
            driving: false,
            path: path.to_owned(),
            preset: preset::taken(path),
            picking: None,
            pasted: String::new(),
            saving: None,
            written: Vec3::splat(f32::INFINITY),
            dirty: true,
            load: LOADED,
            speed: 1.0,
            fov: FOV.to_degrees(),
            layers: Vec::new(),
            placements: Vec::new(),
            selected: None,
            models: Vec::new(),
            model_at: HashMap::new(),
            materials: Vec::new(),
            material_at: HashMap::new(),
            packages: HashMap::new(),
            blobs: HashMap::new(),
            picked: HashSet::new(),
            typed: 0,
            translated: HashMap::new(),
            tables: HashMap::new(),
            lighting: None,
            exposure: None,
            skybox: None,
            sunlight: None,
            moonlight: None,
            haze: None,
            clouds: [None, None],
            cloud_shadow: None,
            cloud_files: [Aside::Done, Aside::Done],
            cloud_wanted: [None, None],
            starlight: None,
            star_files: [Aside::Done, Aside::Done, Aside::Done],
            star_wanted: false,
            sky_volume: None,
            sky_wanted: None,
            sky_file: Aside::Done,
            glare: None,
            smoothing: None,
            occlusion: None,
            vignette: None,
            reflection: None,
            water_mirror: None,
            look: program::Look::default(),
            ambient: ambient::Ambient::new(source.scene()),
            lights: Vec::new(),
            effects: Vec::new(),
            fired: Vec::new(),
            firing: HashMap::new(),
            effect_files: Vec::new(),
            effect_at: HashMap::new(),
            effect_shape: None,
            effect_model: None,
            effect_packages: Arc::new(avfx::gpu::Packages::default()),
            sound: sound::SoundStage::default(),
            clips: HashMap::new(),
            clip: aside(source.scene().map(layer::Scene::light_culling_path)),
            visibility: HashMap::new(),
            sky: aside(source.scene().map(layer::Scene::sky_visibility_path)),
            engine: BTreeMap::from([(
                mdl::deferred::RAMP.0,
                Aside::Wanted(mdl::deferred::RAMP.1.to_owned()),
            )]),
            textures: BTreeMap::new(),
            stacked: BTreeMap::new(),
            resident: 0,
            files: HashMap::new(),
            waiting: Vec::new(),
            terrain: match root {
                Some(root) => Terrain::Wanted(format!("{root}/bgplate/terrain.tera")),
                None => Terrain::Done,
            },
            grass: match root {
                Some(root) => Grass::Wanted(format!("{root}/grass/grass_zone_data.gzd")),
                None => Grass::Done,
            },
            sward: None,
            turf: Vec::new(),
            blades: 0,
            standing: 0,
            fitted: 0,
            renderer: gpu::Renderer::new(),
            cast: Vec::new(),
            unplaced: BTreeSet::new(),
            placed: Vec::new(),
            casts: Vec::new(),
            motions: Vec::new(),
            cycling: false,
            clock: 0.0,
            absent: 0,
        };
        match source.scene() {
            // A level holds no instances of its own; the layer groups it names are where the zone
            // actually is.
            Some(named) if source.groups().is_empty() => {
                for path in named.layer_group_paths() {
                    scene.waiting.push(Expand {
                        path: path.clone(),
                        transform: Mat4::IDENTITY,
                        key: (0, [0; 4]),
                        scale: 1.0,
                        layer: None,
                        depth: 0,
                        chain: Vec::new(),
                        since: Mat4::IDENTITY,
                    });
                }
            }
            _ => scene.walk(
                source.groups(),
                source.scene().map_or(&[][..], SceneTimeline::of),
                source.scene().map_or(&[][..], SceneAnimation::of),
                source.scene().map_or(&[][..], SceneSpin::of),
                source.scene().map_or(&[][..], SceneGlow::of),
                Mat4::IDENTITY,
                (0, [0; 4]),
                1.0,
                None,
                0,
                None,
                &[],
                Mat4::IDENTITY,
            ),
        }
        scene.fit();
        // A preset held for this path was left by an import that had to open it first, and would
        // otherwise sit unapplied: nothing else ever stands the view where it says.
        if let Some(held) = scene.preset.take() {
            scene.stand_where(&held);
            scene.preset = Some(held);
        }
        scene
    }

    /// Drives the camera for the next frame in place of the free orbit camera, and suppresses the
    /// mouse and keyboard input that would otherwise fly it. Takes effect once and is forgotten, so
    /// a host that stops calling this hands control back to the free camera on its next frame.
    pub fn drive(&mut self, drive: Drive) {
        self.drive = Some(drive);
    }

    /// Labels drawn over the next frame at the given scene-space points, alongside whatever a
    /// driven camera shows. Forgotten the same way [`Self::drive`] is.
    pub fn mark(&mut self, markers: Vec<(Vec3, String)>) {
        self.markers = markers;
    }

    /// The characters standing in the scene, replacing whoever stood there before. Unlike
    /// [`Self::place`] a host may call this every frame: a cast that changes is one whose models
    /// have finished arriving, not a scene to build again.
    pub fn stand(&mut self, cast: Vec<Standing>) {
        self.cast = cast;
    }

    /// Which of the props a host placed are out of the frame, replacing whatever was out before.
    /// Called every frame the way [`Self::stand`] is.
    pub fn hide(&mut self, unplaced: BTreeSet<u32>) {
        if self.unplaced != unplaced {
            self.unplaced = unplaced;
            self.dirty = true;
        }
    }

    /// The effects a host outside this view is running, replacing whatever it ran before. Called
    /// every frame the way [`Self::stand`] is: what a firing has run out is kept by its id, so the
    /// list itself is only where each one stands and how far along it is.
    pub fn fire(&mut self, fired: Vec<Fired>) {
        self.fired = fired;
    }

    /// Adds props to the scene under a layer of their own. Unlike [`Self::drive`] and
    /// [`Self::mark`] this appends, so a host builds a scene's props once rather than every frame.
    pub fn place(&mut self, layer: &str, props: Vec<Prop>) {
        if props.is_empty() {
            return;
        }
        self.layers.push(Layer {
            name: layer.to_owned(),
            origin: None,
            visible: true,
            festival: 0,
            shown: true,
            placements: 0,
        });
        let at = self.layers.len() - 1;
        for prop in props {
            let here = matrix(prop.transform);
            let key = reach((0, [0; 4]), 0, prop.id);
            match prop.asset {
                Asset::Model(path) => {
                    let model = self.model(&path);
                    self.models[model].instances += 1;
                    self.layers[at].placements += 1;
                    self.placements.push(Placement {
                        model,
                        transform: here,
                        driven: None,
                        center: here.transform_point3(Vec3::ZERO),
                        // A prop states neither a bounding sphere nor a fade distance, and the
                        // record the game builds for one leaves both at nought as well, so it
                        // never fades.
                        radius: 0.0,
                        fade: 0.0,
                        layer: at,
                        key,
                        glow: None,
                        casts: true,
                        wind_phase: None,
                    });
                }
                Asset::Group(path) => self.waiting.push(Expand {
                    path,
                    transform: here,
                    key,
                    scale: Vec3::from_array(prop.transform.scale())
                        .abs()
                        .max_element()
                        .max(0.001),
                    layer: Some(at),
                    depth: 1,
                    chain: Vec::new(),
                    since: Mat4::IDENTITY,
                }),
            }
        }
        self.dirty = true;
    }

    /// Reads placements out of a file's layers, queueing every shared group it names.
    #[allow(clippy::too_many_arguments)]
    /// The motion a scene gives one of its own instances, where it gives it one: a timeline that
    /// plays curves over it, or a swing or a turn it repeats forever without one.
    fn motion(
        &mut self,
        timelines: &[SceneTimeline],
        animations: &[SceneAnimation],
        spins: &[SceneSpin],
        instance: u32,
        placement: Transform,
    ) -> Option<usize> {
        for timeline in timelines {
            if !timeline.auto_play() {
                continue;
            }
            let Some((actor, _)) = timeline
                .animated()
                .iter()
                .find(|(_, held)| *held as u32 == instance)
            else {
                continue;
            };
            let held = timeline.timeline();
            // The scene names an actor by the key the actor itself carries, not by its item id.
            let Some(tracks) = held.items().iter().find_map(|item| match item {
                tmb::Item::Actor(held) if i32::from(held.time()) == *actor => Some(held.tracks()),
                _ => None,
            }) else {
                continue;
            };
            // Every track of the actor and every command of each, since an actor states its motion
            // across all of them: eight of the game's aetherytes hang four tracks off one actor.
            // The first to name a channel keeps it, so a lone track reads exactly as it did.
            let mut curves: Vec<(tmb::Channel, tmb::Curve)> = Vec::new();
            let mut steps: Vec<(f32, Mat4)> = Vec::new();
            for track in tracks {
                let Some(commands) = held.items().iter().find_map(|item| match item {
                    tmb::Item::Track(held) if held.id() == *track => Some(held.commands()),
                    _ => None,
                }) else {
                    continue;
                };
                for command in commands {
                    let Some(tmb::Item::Command(found)) = held.items().iter().find(
                        |item| matches!(item, tmb::Item::Command(held) if held.id() == *command),
                    ) else {
                        continue;
                    };
                    match found.kind() {
                        tmb::CommandKind::C013(driven) => {
                            let Some(set) = held.items().iter().find_map(|item| match item {
                                tmb::Item::Curves(held)
                                    if i32::from(held.id()) == driven.curve_id() =>
                                {
                                    Some(held.curves())
                                }
                                _ => None,
                            }) else {
                                continue;
                            };
                            for curve in set {
                                let Some(channel) = curve.channel() else {
                                    continue;
                                };
                                if curves.iter().all(|(held, _)| *held != channel) {
                                    curves.push((channel, curve.clone()));
                                }
                            }
                        }
                        // The command's own scale is the identity in every file the game ships, so
                        // taking it would only throw away a scale the scene did state.
                        tmb::CommandKind::C018(driven) => steps.push((
                            f32::from(found.time()),
                            Mat4::from_scale_rotation_translation(
                                Vec3::from_array(placement.scale()),
                                Quat::from_mat3(&rotation(driven.rotation())),
                                Vec3::from_array(driven.translation()),
                            ),
                        )),
                        _ => (),
                    }
                }
            }
            let duration = held
                .items()
                .iter()
                .find_map(|item| match item {
                    tmb::Item::Header(held) => Some(f32::from(held.duration())),
                    _ => None,
                })
                .unwrap_or(1.0);
            if !curves.is_empty() {
                self.motions.push(Motion::Keyed {
                    curves,
                    duration,
                    looping: timeline.looping(),
                });
                return Some(self.motions.len() - 1);
            }
            if steps.is_empty() {
                continue;
            }
            steps.sort_by(|left, right| left.0.total_cmp(&right.0));
            self.motions.push(Motion::Placed {
                placement,
                steps,
                duration,
                looping: timeline.looping(),
            });
            return Some(self.motions.len() - 1);
        }
        // Only where no timeline already plays over it: a few scenes state both, and the curves are
        // the more particular of the two.
        for animation in animations {
            if !animation.instances().contains(&instance) {
                continue;
            }
            self.motions.push(Motion::Repeat {
                placement,
                translation: *animation.translation(),
                rotation: *animation.rotation(),
                scale: *animation.scale(),
            });
            return Some(self.motions.len() - 1);
        }
        for spin in spins {
            if spin.instance() != instance || spin.period() == 0.0 {
                continue;
            }
            self.motions.push(Motion::Spin {
                placement,
                axis: match spin.axis() {
                    0 => Vec3::X,
                    2 => Vec3::Z,
                    _ => Vec3::Y,
                },
                period: spin.period(),
            });
            return Some(self.motions.len() - 1);
        }
        None
    }

    #[allow(clippy::too_many_arguments)]
    fn walk(
        &mut self,
        groups: &[LayerGroup],
        timelines: &[SceneTimeline],
        animations: &[SceneAnimation],
        spins: &[SceneSpin],
        glows: &[SceneGlow],
        transform: Mat4,
        key: (u32, [u8; 4]),
        scale: f32,
        under: Option<usize>,
        depth: u8,
        origin: Option<&str>,
        chain: &[(usize, Mat4)],
        since: Mat4,
    ) {
        for group in groups {
            for layer in group.layers() {
                let at = match under {
                    Some(at) => at,
                    None => {
                        self.layers.push(Layer {
                            name: layer.name().clone(),
                            origin: origin.map(str::to_owned),
                            visible: layer.visible(),
                            festival: layer.festival_id(),
                            shown: layer.visible() && layer.festival_id() == 0,
                            placements: 0,
                        });
                        self.layers.len() - 1
                    }
                };
                for instance in layer.instances() {
                    let placed = instance.transform();
                    // What a timeline moves stands where the curves put it rather than where the
                    // file did, and everything under it follows.
                    let moved = self.motion(timelines, animations, spins, instance.id(), placed);
                    let local = match moved {
                        Some(at) => self.motions[at].at(0.0),
                        None => matrix(placed),
                    };
                    let here = transform * local;
                    // A moved node joins the chain with whatever fixed transform led up to it, and
                    // what follows accumulates from there. Everything else just lengthens the tail.
                    let (chain, since) = match moved {
                        Some(at) => {
                            let mut held = chain.to_vec();
                            held.push((at, since));
                            (held, Mat4::IDENTITY)
                        }
                        None => (chain.to_vec(), since * local),
                    };
                    match instance.data() {
                        InstanceData::BgPart(part)
                            if part.visible() && !part.asset_path().is_empty() =>
                        {
                            let model = self.model(part.asset_path());
                            self.models[model].instances += 1;
                            self.layers[at].placements += 1;
                            let glow = tinted(glows, instance.id(), SceneGlow::surface);
                            self.cycling |= glow.is_some();
                            self.placements.push(Placement {
                                model,
                                transform: here,
                                driven: (!chain.is_empty()).then(|| {
                                    Rc::new(Driven {
                                        chain: chain.clone(),
                                        tail: since,
                                    })
                                }),
                                center: here.transform_point3(Vec3::ZERO),
                                radius: part.bounding_sphere_size() * scale,
                                fade: part.fade_out_distance(),
                                layer: at,
                                key: reach(key, depth, instance.id()),
                                glow,
                                casts: part.world_light_shadow_mode() != ShadowMode::ForceOff,
                                wind_phase: None,
                            });
                        }
                        InstanceData::SharedGroup(shared)
                            if depth < DEPTH && !shared.asset_path().is_empty() =>
                        {
                            self.waiting.push(Expand {
                                path: shared.asset_path().clone(),
                                transform: here,
                                key: reach(key, depth, instance.id()),
                                scale: scale
                                    * Vec3::from_array(placed.scale())
                                        .abs()
                                        .max_element()
                                        .max(0.001),
                                layer: Some(at),
                                depth: depth + 1,
                                chain,
                                since,
                            });
                        }
                        InstanceData::EnvSpace(space) => {
                            self.ambient.spaces.push(ambient::Space {
                                placement: here,
                                // The composite reads the kind back with the bit pattern, not the
                                // value, so it goes in as one.
                                shape: f32::from_bits(space.shape() as u32),
                                range: space.effective_range(),
                                bound: space.bound_instance_id(),
                            });
                        }
                        InstanceData::EnvLocation(env) => {
                            self.ambient
                                .locate(instance.id(), env.ambient_light_asset_path());
                        }
                        InstanceData::Light(light) => {
                            let held = light.colour();
                            let color = Vec3::new(
                                f32::from(held.red()),
                                f32::from(held.green()),
                                f32::from(held.blue()),
                            ) / 255.0;
                            // Without the scale a parent carries: a light's own space is where the
                            // box it is clipped against is stated, so a shared group placed at
                            // eight tenths over would light a volume of a different size than the
                            // one the zone cut for it.
                            let (_, turn, at) = here.to_scale_rotation_translation();
                            let here = Mat4::from_rotation_translation(turn, at);
                            let glow = tinted(glows, instance.id(), SceneGlow::light);
                            self.cycling |= glow.is_some();
                            let kind = match light.kind() {
                                LightKind::Spot => program::LampKind::Spot,
                                LightKind::Line => program::LampKind::Line,
                                LightKind::Flat => program::LampKind::Plane,
                                _ => program::LampKind::Point,
                            };
                            let color = color * held.intensity();
                            let range = light.range().max(0.001);
                            // Halved, since each angle is stated across the whole cone. Only a spot
                            // carries them, and the light's own package reads a different lane
                            // where its kind does not have a cone at all.
                            let half = |angle: f32| match kind {
                                program::LampKind::Spot => (angle * 0.5).to_radians().cos(),
                                _ => 0.0,
                            };
                            self.lights.push(Light {
                                placement: here,
                                center: at,
                                min: Vec3::splat(-REACH),
                                max: Vec3::splat(REACH),
                                range,
                                falloff: falloff(light.attenuation()),
                                color,
                                kind,
                                direction: here.transform_vector3(Vec3::Z).normalize_or_zero(),
                                inner: half(light.spot_angle()),
                                cone: half(
                                    light.spot_angle() + light.attenuation_cone_coefficient(),
                                ),
                                key: reach(key, depth, instance.id()),
                                glow,
                            });
                        }
                        // A placement with auto_play unset only ever runs off a script trigger this
                        // viewer has no notion of, so it would never be seen this way in game either.
                        InstanceData::Vfx(vfx)
                            if !vfx.asset_path().is_empty() && vfx.auto_play() =>
                        {
                            self.effects.push(Vfx {
                                placement: here,
                                driven: (!chain.is_empty()).then(|| {
                                    Rc::new(Driven {
                                        chain: chain.clone(),
                                        tail: since,
                                    })
                                }),
                                path: vfx.asset_path().clone(),
                                layer: at,
                                key: reach(key, depth, instance.id()),
                                tint: vfx_tint(vfx.colour()),
                                fade_near: vfx.fade_near(),
                                fade_far: vfx.fade_far(),
                                no_far_clip: vfx.no_far_clip(),
                            });
                        }
                        InstanceData::Sound(placed_sound) => {
                            self.sound.collect(
                                placed_sound,
                                here.transform_point3(Vec3::ZERO),
                                reach(key, depth, instance.id()),
                            );
                        }
                        _ => {}
                    }
                }
            }
        }
        self.dirty = true;
    }

    fn model(&mut self, path: &str) -> usize {
        if let Some(at) = self.model_at.get(path) {
            return *at;
        }
        self.models.push(Model {
            path: path.to_owned(),
            state: State::Wanted,
            drawn: [false; 3],
            meshes: Vec::new(),
            waving: false,
            casts: true,
            instances: 0,
            nearest: f32::INFINITY,
            finest: 2,
            asked: 2,
        });
        self.model_at.insert(path.to_owned(), self.models.len() - 1);
        self.models.len() - 1
    }

    fn material(&mut self, path: &str) -> usize {
        if let Some(at) = self.material_at.get(path) {
            return *at;
        }
        self.materials.push((path.to_owned(), Slot::Wanted));
        self.material_at
            .insert(path.to_owned(), self.materials.len() - 1);
        self.materials.len() - 1
    }

    /// Puts the camera where the placements read so far are.
    fn fit(&mut self) {
        let points: Vec<Vec3> = self
            .placements
            .iter()
            .map(|placement| placement.center)
            .collect();
        let (center, reach) = bulk(&points);
        self.home = looking_at(center, reach);
        self.camera = self.home;
        self.fitted = points.len();
        self.dirty = true;
    }

    /// The ground, as one placement per plate. Meddle places a plate at the position the terrain
    /// file states with no rotation of its own, which is what this does.
    fn place_terrain(&mut self, path: &str, bytes: Vec<u8>) {
        let terrain = match tera::Terrain::read(Cursor::new(bytes)) {
            Ok(terrain) => terrain,
            Err(why) => {
                log::error!("assets/layer: {path}: {why}");
                return;
            }
        };
        let directory = path.trim_end_matches("terrain.tera");
        self.layers.push(Layer {
            name: "terrain".to_owned(),
            origin: Some(path.to_owned()),
            visible: true,
            festival: 0,
            shown: true,
            placements: terrain.plates().len(),
        });
        let at = self.layers.len() - 1;
        for (index, plate) in terrain.plates().iter().enumerate() {
            let (x, z) = terrain.plate_position(*plate);
            let model = self.model(&format!("{directory}{}", tera::Terrain::plate_file(index)));
            self.models[model].instances += 1;
            let center = Vec3::new(x, 0.0, z);
            self.placements.push(Placement {
                model,
                transform: Mat4::from_translation(center),
                driven: None,
                center,
                radius: PLATE,
                fade: 0.0,
                layer: at,
                key: (0, [0; 4]),
                glow: None,
                casts: true,
                wind_phase: None,
            });
        }
        self.dirty = true;
    }

    fn load_terrain(&mut self, backend: &Backend) {
        let mut arrived = None;
        let next = match &self.terrain {
            Terrain::Wanted(path) => {
                let files = backend.files().clone();
                let wanted = path.clone();
                Some(Terrain::Fetching(
                    path.clone(),
                    TrackedPromise::spawn_local(async move { files.read(&wanted).await }),
                ))
            }
            Terrain::Fetching(path, promise) => match promise.try_get() {
                Some(Ok(bytes)) => {
                    arrived = Some((path.clone(), bytes.clone()));
                    Some(Terrain::Done)
                }
                // Plenty of zones are interiors with no ground of their own.
                Some(Err(_)) => Some(Terrain::Done),
                None => None,
            },
            Terrain::Done => None,
        };
        if let Some(next) = next {
            self.terrain = next;
        }
        if let Some((path, bytes)) = arrived {
            self.place_terrain(&path, bytes);
        }
    }

    /// The zone's grass file, which names the models and sorts the grids but places nothing itself.
    fn open_grass(&mut self, path: &str, bytes: Vec<u8>) {
        let zone = match gzd::GrassZone::read(Cursor::new(bytes)) {
            Ok(zone) => zone,
            Err(why) => {
                log::error!("assets/layer: {path}: {why}");
                return;
            }
        };
        let directory = path.trim_end_matches("grass_zone_data.gzd").to_owned();
        // The zone names its models by full path, and shares them across zones: an s1f2 grid places
        // s1f1's plants.
        let models = zone
            .model_paths()
            .iter()
            .map(|path| self.model(path))
            .collect();
        let grids: Vec<Patch> = [gzd::Detail::High, gzd::Detail::Medium, gzd::Detail::Low]
            .into_iter()
            .flat_map(|detail| zone.grids(detail))
            .map(|grid| Patch {
                center: Vec3::from_array(grid.center()),
                radius: grid.radius(),
                file: grid.file(),
                fetch: None,
                taken: false,
            })
            .collect();
        self.layers.push(Layer {
            name: "grass".to_owned(),
            origin: Some(path.to_owned()),
            visible: true,
            festival: 0,
            shown: true,
            placements: 0,
        });
        let maps = zone
            .color_map()
            .iter()
            .map(|name| match name.is_empty() {
                true => String::new(),
                false => format!("{directory}{name}.tex"),
            })
            .collect();
        self.grass = Grass::Placing(Box::new(Placing {
            directory,
            models,
            maps,
            grids,
            layer: self.layers.len() - 1,
        }));
    }

    /// Every placement of one grid: the leading count slots are the procedural layers, whose
    /// placements stand as blades of the zone's own grass, and the rest name a model the zone lists.
    fn place_grass(&mut self, grid: usize, bytes: Vec<u8>) {
        let Grass::Placing(placing) = &self.grass else {
            return;
        };
        let (models, layer) = (placing.models.clone(), placing.layer);
        let radius = placing.grids[grid].radius;
        let file = match ggd::GrassGrid::read(Cursor::new(bytes)) {
            Ok(file) => file,
            Err(why) => {
                log::error!("assets/layer: grass grid {grid}: {why}");
                return;
            }
        };
        let origin = Vec3::from_array(file.world_origin());
        let mut sown: [Vec<gpu::Corner>; ggd::Chunk::AUTO_LAYERS] = Default::default();
        for chunk in file.chunks() {
            let mut at = 0;
            for (slot, count) in chunk.counts().iter().enumerate() {
                let placements = &chunk.placements()[at..at + usize::from(*count)];
                at += usize::from(*count);
                let Some(model) = slot.checked_sub(ggd::Chunk::AUTO_LAYERS) else {
                    for placement in placements {
                        blade(placement, &mut sown[slot]);
                    }
                    continue;
                };
                let Some(model) = models.get(model).copied() else {
                    continue;
                };
                for placement in placements {
                    let scale = Vec3::new(
                        placement.scale_xz(),
                        placement.scale_y(),
                        placement.scale_xz(),
                    );
                    let center = origin + Vec3::from_array(placement.position());
                    self.models[model].instances += 1;
                    self.layers[layer].placements += 1;
                    self.placements.push(Placement {
                        model,
                        driven: None,
                        transform: Mat4::from_scale_rotation_translation(
                            scale,
                            Quat::from_array(placement.rotation()),
                            center,
                        ),
                        center,
                        radius: scale.max_element(),
                        fade: 0.0,
                        layer,
                        key: (0, [0; 4]),
                        glow: None,
                        casts: true,
                        wind_phase: Some(placement.wind_phase()),
                    });
                }
            }
        }
        self.sow(origin, radius, sown);
        self.dirty = true;
    }

    /// One grid's blades handed to the card, a buffer per auto layer.
    fn sow(&mut self, origin: Vec3, radius: f32, sown: [Vec<gpu::Corner>; ggd::Chunk::AUTO_LAYERS]) {
        let Grass::Placing(placing) = &self.grass else {
            return;
        };
        let cut: [bool; ggd::Chunk::AUTO_LAYERS] =
            std::array::from_fn(|at| placing.maps.get(at).is_some_and(|path| !path.is_empty()));
        for (layer, corners) in sown.into_iter().enumerate() {
            let blades = corners.len() / 4;
            // A layer the zone names no map for is cut out of nothing, so it stands nothing up.
            if blades == 0 || !cut[layer] {
                continue;
            }
            let indices = (0..blades as u32)
                .flat_map(|at| [0, 1, 2, 2, 1, 3].map(|corner| at * 4 + corner))
                .collect();
            self.renderer.lock().unwrap().queue_turf(gpu::Sown {
                turf: self.turf.len(),
                corners,
                indices,
            });
            self.turf.push(Turf {
                origin,
                radius,
                layer,
                blades,
            });
            self.blades += blades;
        }
    }

    fn load_grass(&mut self, backend: &Backend) {
        let mut arrived = None;
        let next = match &self.grass {
            Grass::Wanted(path) => {
                let files = backend.files().clone();
                let wanted = path.clone();
                Some(Grass::Fetching(
                    path.clone(),
                    TrackedPromise::spawn_local(async move { files.read(&wanted).await }),
                ))
            }
            Grass::Fetching(path, promise) => match promise.try_get() {
                Some(Ok(bytes)) => {
                    arrived = Some((path.clone(), bytes.clone()));
                    None
                }
                // Interiors and instanced zones place no grass of their own.
                Some(Err(_)) => Some(Grass::Done),
                None => None,
            },
            Grass::Placing(_) | Grass::Done => None,
        };
        if let Some(next) = next {
            self.grass = next;
        }
        if let Some((path, bytes)) = arrived {
            self.open_grass(&path, bytes);
        }
        self.load_grids(backend);
    }

    /// Asks for the grids the eye has reached, a few at a time, and places each as it lands.
    fn load_grids(&mut self, backend: &Backend) {
        let (eye, load) = (self.camera.position, self.load);
        let mut arrived = Vec::new();
        let Grass::Placing(placing) = &mut self.grass else {
            return;
        };
        let mut flight = placing
            .grids
            .iter()
            .filter(|grid| grid.fetch.is_some())
            .count();
        for (at, grid) in placing.grids.iter_mut().enumerate() {
            let landed = match &grid.fetch {
                Some(promise) => match promise.try_get() {
                    Some(Ok(bytes)) => {
                        arrived.push((at, bytes.clone()));
                        true
                    }
                    Some(Err(_)) => true,
                    None => false,
                },
                None => false,
            };
            if landed {
                grid.fetch = None;
                flight -= 1;
            }
        }
        // Nearest first, rather than in the order the zone lists them. A zone holds the same ground
        // three times over at three levels of detail, and the models sit in the coarsest of them:
        // taken in order, six hundred grids of nothing but procedural layers are read before the
        // first grid that places anything at all.
        while flight < GRIDS {
            let wanted = placing
                .grids
                .iter()
                .enumerate()
                .filter(|(_, grid)| {
                    !grid.taken && eye.distance(grid.center) < load + grid.radius
                })
                .min_by(|(_, one), (_, two)| {
                    eye.distance(one.center).total_cmp(&eye.distance(two.center))
                })
                .map(|(at, _)| at);
            let Some(at) = wanted else { break };
            let grid = &mut placing.grids[at];
            let files = backend.files().clone();
            let path = format!("{}{}", placing.directory, grid.file);
            grid.fetch = Some(TrackedPromise::spawn_local(
                async move { files.read(&path).await },
            ));
            grid.taken = true;
            flight += 1;
        }
        for (grid, bytes) in arrived {
            self.place_grass(grid, bytes);
        }
    }

    /// Where every model stands for where the eye now is. The transforms go to the card each frame
    /// rather than here, since a record carries the object into view space and the camera turns.
    /// Where a placement stands now, which is where the file put it unless a timeline drives it.
    fn posed(&self, placement: &Placement) -> Mat4 {
        self.moved(placement.transform, &placement.driven)
    }

    /// Where a driven transform stands now, or the transform itself where nothing drives it. Shared
    /// with `Vfx`, which carries the same chain but is not a `Placement`.
    fn moved(&self, transform: Mat4, driven: &Option<Rc<Driven>>) -> Mat4 {
        match driven {
            Some(held) => {
                held.chain.iter().fold(Mat4::IDENTITY, |into, (at, fixed)| {
                    into * *fixed * self.motions[*at].at(self.clock)
                }) * held.tail
            }
            None => transform,
        }
    }

    /// The nearest placement the ray runs through, out of the ones the frame is drawing.
    fn under(&self, from: Vec3, along: Vec3) -> Option<usize> {
        let eye = self.camera.position;
        nearest(
            from,
            along,
            self.placements.iter().enumerate().filter_map(|(at, placement)| {
                if !self.layers[placement.layer].shown || self.unplaced.contains(&placement.key.0) {
                    return None;
                }
                let span = (placement.center - eye).length() - placement.radius;
                if span > self.load || (placement.fade > 0.0 && span > placement.fade) {
                    return None;
                }
                let center = self.posed(placement).transform_point3(Vec3::ZERO);
                Some((at, center, placement.radius.max(0.01)))
            }),
        )
    }

    fn rebuild(&mut self) {
        let eye = self.camera.position;
        let mut placed: Vec<[Vec<program::Instance>; 3]> = (0..self.models.len())
            .map(|_| std::array::from_fn(|_| Vec::new()))
            .collect();
        let mut blocked = placed.clone();
        for model in &mut self.models {
            model.nearest = f32::INFINITY;
            model.finest = 2;
        }
        self.absent = 0;

        for at in 0..self.placements.len() {
            let placement = self.placements[at].clone();
            if !self.layers[placement.layer].shown || self.unplaced.contains(&placement.key.0) {
                continue;
            }
            let span = (placement.center - eye).length() - placement.radius;
            if span > self.load || (placement.fade > 0.0 && span > placement.fade) {
                continue;
            }
            let apparent = placement.radius / span.max(0.01);
            let model = &mut self.models[placement.model];
            model.nearest = model.nearest.min(span);
            model.finest = model.finest.min(detail(apparent));
            let Some(level) = level(model.drawn, apparent) else {
                if !matches!(model.state, State::Ready | State::Failed) {
                    self.absent += 1;
                }
                continue;
            };
            let into = match placement.casts && model.casts {
                true => &mut placed,
                false => &mut blocked,
            };
            into[placement.model][level].push(program::Instance {
                transform: self.posed(&placement),
                sky_visibility: reached(&self.visibility, placement.key).copied().unwrap_or(1.0),
                emissive: placement.glow.map(|lane| {
                    let (color, power) = cycled(lane, self.clock);
                    color.extend(power)
                }),
                wind_phase: placement.wind_phase,
            });
        }
        self.casts = placed
            .iter_mut()
            .zip(&mut blocked)
            .map(|(held, rest)| {
                std::array::from_fn(|level| {
                    let casts = held[level].len();
                    held[level].append(&mut rest[level]);
                    casts
                })
            })
            .collect();
        self.placed = placed;
        self.written = eye;
        self.dirty = false;
    }

    /// The lights the frame draws, nearest first. Each is clipped against the box its zone states
    /// for it, in the light's own space.
    ///
    /// Nearest by how close a light's own volume comes, not by where its middle stands: a hall's far
    /// lamps cover more of the frame than a near one with a foot of reach, and an interior states
    /// hundreds, so a cap taken on the middles alone leaves whole galleries lit by nothing.
    fn lamps(&self) -> Vec<program::Lamp> {
        let eye = self.camera.position;
        let mut near: Vec<(f32, &Light)> = self
            .lights
            .iter()
            .map(|light| {
                let reach = light.min.abs().max(light.max.abs()).max_element();
                (((light.center - eye).length() - reach).max(0.0), light)
            })
            .filter(|(span, _)| *span <= self.load)
            .collect();
        near.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        near.into_iter()
            .take(LAMPS)
            .map(|(_, light)| {
                // A light the `.lcb` states no box for keeps one of this viewer's own. Either
                // way, how far the light carries is read off that box rather than off its
                // brightness: a captured frame keeps one reach for one box under a colour that
                // swings widely, and a reach solved from brightness alone can drift far past a
                // small box, leaving everything inside it lit at full strength up to a hard wall.
                let (min, max) = reached(&self.clips, light.key)
                    .copied()
                    .unwrap_or((light.min, light.max));
                let reach = min.abs().max(max.abs()).max_element().max(0.001);
                program::Lamp {
                    placement: light.placement,
                    min,
                    max,
                    reach,
                    falloff: light.falloff,
                    range: light.range,
                    color: match light.glow {
                        Some(lane) => {
                            let (color, power) = cycled(lane, self.clock);
                            color * power
                        }
                        None => light.color,
                    },
                    kind: light.kind,
                    direction: light.direction,
                    inner: light.inner,
                    cone: light.cone,
                }
            })
            .collect()
    }

    /// Every placed effect that has arrived and is textured, merged into one batch set a file no
    /// matter how many placements share it: the particles a shared simulation stepped, each copy
    /// carried into the world by its own placement's rotation, offset and scale.
    fn effect_draws(&self, view: Mat4, eye: Vec3) -> Vec<gpu::EffectDraw> {
        let axes = Mat3::from_mat4(view).transpose();
        let (right, up) = (axes.x_axis, axes.y_axis);
        self.effect_files
            .iter()
            .filter_map(|effect| {
                let EffectState::Ready(parsed, live, _) = &effect.state else {
                    return None;
                };
                let bound = self.bound_textures(parsed)?;
                let base = parsed.drawn(live);
                let mut drawn = Vec::new();
                for vfx in self.effects.iter().filter(|vfx| vfx.path == effect.path) {
                    let (scale, rotation, translation) = self
                        .moved(vfx.placement, &vfx.driven)
                        .to_scale_rotation_translation();
                    let scale = scale.abs().max_element().max(0.001);
                    let distance = eye.distance(translation);
                    let far = match vfx.no_far_clip {
                        true => 1.0,
                        false => far_fade(distance, vfx.fade_far),
                    };
                    let fade = near_fade(distance, vfx.fade_near) * far;
                    let tint = vfx.tint * Vec4::new(1.0, 1.0, 1.0, fade);
                    drawn.extend(
                        base.iter()
                            .map(|held| held.placed(rotation, translation, scale, tint)),
                    );
                }
                let batches = avfx::batches(parsed, drawn, &bound, view, eye, right, up);
                (!batches.is_empty()).then_some(gpu::EffectDraw {
                    path: effect.path.clone(),
                    batches,
                    fade_range: parsed.fade_range,
                })
            })
            .chain(self.fired.iter().filter_map(|held| {
                let Some(EffectState::Ready(parsed, ..)) = self
                    .effect_at
                    .get(&held.path)
                    .map(|at| &self.effect_files[*at].state)
                else {
                    return None;
                };
                let bound = self.bound_textures(parsed)?;
                let (scale, rotation, translation) = held.at.to_scale_rotation_translation();
                let scale = scale.abs().max_element().max(0.001);
                let drawn: Vec<_> = parsed
                    .drawn(self.firing.get(&held.id)?)
                    .into_iter()
                    .map(|item| item.placed(rotation, translation, scale, held.tint))
                    .collect();
                let batches = avfx::batches(parsed, drawn, &bound, view, eye, right, up);
                (!batches.is_empty()).then_some(gpu::EffectDraw {
                    path: held.path.clone(),
                    batches,
                    fade_range: parsed.fade_range,
                })
            }))
            .collect()
    }

    /// The handle bound to each texture an effect samples, or `None` where one has not arrived:
    /// held back rather than drawn white, since an additive quad with no sampler reads as flat
    /// white.
    fn bound_textures(&self, parsed: &avfx::sim::Effect) -> Option<Vec<Option<egui::TextureId>>> {
        parsed
            .textures
            .iter()
            .map(|path| match self.textures.get(path.as_str()) {
                Some(Texture::Ready(handle)) => Some(Some(handle.id())),
                _ => None,
            })
            .collect()
    }

    /// The files read once beside the scene, as they arrive: the boxes its lights are clipped
    /// against, how much of the sky reaches each of its parts, and the game's own textures its
    /// shaders read.
    fn load_asides(&mut self, backend: &Backend) {
        // The volume the sky pass reads, which is named by the id rather than by the zone. Asked for
        // only once the pass itself has translated, since the pass is what says which resource the
        // volume is bound under.
        if self.skybox.is_some() && self.sky_wanted != self.ambient.sky() {
            self.sky_wanted = self.ambient.sky();
            self.sky_volume = None;
            self.sky_file = match self.ambient.sky() {
                Some(id) => Aside::Wanted(program::sky_texture(id)),
                None => Aside::Done,
            };
        }
        // The two cloud textures, each named by the weather rather than by the zone, and asked for
        // only once the draw that reads it has translated.
        if self.clouds[0].is_some() {
            let held = self.ambient.clouds();
            let wanted = [
                held.as_ref().and_then(|held| held.band),
                held.as_ref().and_then(|held| held.sheet),
            ];
            for (at, id) in wanted.into_iter().enumerate() {
                if self.cloud_wanted[at] == id {
                    continue;
                }
                self.cloud_wanted[at] = id;
                self.cloud_files[at] = match (at, id) {
                    (0, Some(id)) => Aside::Wanted(program::cloudside_texture(id)),
                    (_, Some(id)) => Aside::Wanted(program::cloud_texture(id)),
                    _ => Aside::Done,
                };
            }
        }
        // The star field's three textures, fixed paths asked for once the zone's own weather ever
        // states a starfield set: unlike the sky and cloud files, nothing here ever moves them on.
        if !self.star_wanted && self.ambient.starfield().is_some() {
            self.star_wanted = true;
            self.star_files = [
                Aside::Wanted(program::STAR_COLOR.to_owned()),
                Aside::Wanted(program::STAR_BAND.to_owned()),
                Aside::Wanted(program::STAR_TWINKLE.to_owned()),
            ];
        }
        for held in [&mut self.clip, &mut self.sky, &mut self.sky_file]
            .into_iter()
            .chain(&mut self.cloud_files)
            .chain(&mut self.star_files)
            .chain(self.engine.values_mut())
        {
            *held = match std::mem::replace(held, Aside::Done) {
                Aside::Wanted(path) => {
                    let files = backend.files().clone();
                    let wanted = path.clone();
                    Aside::Fetching(
                        path,
                        TrackedPromise::spawn_local(async move { files.read(&wanted).await }),
                    )
                }
                held => held,
            };
        }
        let taken = |held: &mut Aside| match held {
            Aside::Fetching(path, promise) => match promise.try_get() {
                Some(Ok(bytes)) => {
                    let arrived = (path.clone(), bytes.clone());
                    *held = Aside::Done;
                    Some(arrived)
                }
                Some(Err(_)) => {
                    *held = Aside::Done;
                    None
                }
                None => None,
            },
            _ => None,
        };
        let clip = taken(&mut self.clip);
        let sky = taken(&mut self.sky);
        let volume = taken(&mut self.sky_file);
        let overcast: Vec<(usize, String, Vec<u8>)> = self
            .cloud_files
            .iter_mut()
            .enumerate()
            .filter_map(|(at, held)| taken(held).map(|(path, bytes)| (at, path, bytes)))
            .collect();
        let supplied: Vec<(u32, String, Vec<u8>)> = self
            .engine
            .iter_mut()
            .filter_map(|(id, held)| taken(held).map(|(path, bytes)| (*id, path, bytes)))
            .collect();
        let starlit: Vec<(usize, String, Vec<u8>)> = self
            .star_files
            .iter_mut()
            .enumerate()
            .filter_map(|(at, held)| taken(held).map(|(path, bytes)| (at, path, bytes)))
            .collect();

        if let Some((path, bytes)) = clip {
            match lcb::ClipBoxes::read(Cursor::new(bytes)) {
                Ok(held) => {
                    for group in held.groups() {
                        for entry in group.entries() {
                            self.clips.insert(
                                (entry.instance(), entry.members()),
                                (Vec3::from_array(entry.min()), Vec3::from_array(entry.max())),
                            );
                        }
                    }
                    self.dirty = true;
                }
                Err(why) => log::error!("assets/layer: {path}: {why}"),
            }
        }
        if let Some((path, bytes)) = sky {
            match svb::SkyVisibility::read(Cursor::new(bytes)) {
                Ok(held) => {
                    for group in held.groups() {
                        for entry in group.entries() {
                            self.visibility
                                .insert((entry.instance(), entry.members()), entry.visibility());
                        }
                    }
                    self.dirty = true;
                }
                Err(why) => log::error!("assets/layer: {path}: {why}"),
            }
        }
        if let Some((path, bytes)) = volume
            && let Some(held) = self
                .skybox
                .as_ref()
                .and_then(|held| held.textures.first())
                .map(|texture| texture.id)
        {
            // Read between its texels rather than at them: a sky is a handful of texels across a
            // whole sky, and the hour falls between two of its slices.
            match mdl::layered(&bytes, &path, glow::LINEAR) {
                Ok(decoded) => {
                    self.sky_volume = Some((
                        held,
                        (decoded.size.0 as f32, decoded.size.1 as f32),
                        decoded.layers as f32,
                    ));
                    self.renderer.lock().unwrap().queue_supplied(held, decoded);
                }
                Err(why) => log::error!("assets/layer: {path}: {why}"),
            }
        }
        for (at, path, bytes) in overcast {
            // Read between its texels: a cloud sheet is tiled over tens of thousands of units, so
            // one texel of it covers a good deal of sky.
            match mdl::layered(&bytes, &path, glow::LINEAR) {
                Ok(held) => self.renderer.lock().unwrap().queue_overcast(at, path, held),
                Err(why) => log::error!("assets/layer: {path}: {why}"),
            }
        }
        for (at, path, bytes) in starlit {
            // Wrapped rather than clamped: every one of them is sampled well past a single tile.
            match mdl::layered(&bytes, &path, glow::LINEAR) {
                Ok(held) => self.renderer.lock().unwrap().queue_starlit(at, held),
                Err(why) => log::error!("assets/layer: {path}: {why}"),
            }
        }
        for (id, path, bytes) in supplied {
            let Some((_, _, filter)) = mdl::deferred::ENGINE
                .into_iter()
                .find(|(held, _, _)| *held == id)
            else {
                continue;
            };
            match mdl::layered(&bytes, &path, filter) {
                Ok(held) => self.renderer.lock().unwrap().queue_supplied(id, held),
                Err(why) => log::error!("assets/layer: {path}: {why}"),
            }
        }
    }

    /// Reads each `.avfx` a placement names, once no matter how many placements share it, and steps
    /// every one that has arrived to where the clock now stands.
    fn load_effects(&mut self, backend: &Backend) {
        for path in self
            .effects
            .iter()
            .map(|vfx| vfx.path.clone())
            .chain(self.fired.iter().map(|held| held.path.clone()))
            .collect::<Vec<_>>()
        {
            if self.effect_at.contains_key(&path) {
                continue;
            }
            self.effect_at.insert(path.clone(), self.effect_files.len());
            self.effect_files.push(Effect {
                path,
                state: EffectState::Wanted,
            });
        }

        for effect in &mut self.effect_files {
            if !matches!(effect.state, EffectState::Wanted) {
                continue;
            }
            let files = backend.files().clone();
            let path = effect.path.clone();
            effect.state = EffectState::Fetching(TrackedPromise::spawn_local(async move {
                files.read(&path).await
            }));
        }

        for effect in &mut self.effect_files {
            let EffectState::Fetching(promise) = &effect.state else {
                continue;
            };
            let Some(result) = promise.try_get() else {
                continue;
            };
            effect.state = match result
                .as_ref()
                .map_err(ToString::to_string)
                .and_then(|bytes| {
                    Avfx::read(Cursor::new(bytes.clone())).map_err(|why| why.to_string())
                }) {
                Ok(file) => {
                    let mut parsed = avfx::sim::Effect::read(&file);
                    let models = std::mem::take(&mut parsed.models);
                    self.renderer
                        .lock()
                        .unwrap()
                        .queue_effect(effect.path.clone(), models);
                    self.dirty = true;
                    EffectState::Ready(parsed, avfx::sim::State::default(), self.clock as i32)
                }
                Err(why) => {
                    log::error!("assets/layer: {}: {why}", effect.path);
                    EffectState::Failed
                }
            };
        }

        let frame = self.clock as i32;
        for effect in &mut self.effect_files {
            let EffectState::Ready(parsed, live, born) = &mut effect.state else {
                continue;
            };
            // Relative to when the effect arrived, so its own timeline starts at zero there rather
            // than at the clock's zero. An unbounded one is left running past `length` rather than
            // wrapped back to it.
            let elapsed = frame - *born;
            let target = match parsed.bounded {
                true => elapsed.rem_euclid(parsed.length.max(1)),
                false => elapsed,
            };
            parsed.seek(live, target);
        }

        // A firing runs once rather than over and over: the host says when it started, so it plays
        // to the end its own file states and settles there. Held to that end, and to the longest
        // run the simulation reaches at all, since stepping is what moves a particle and a scrub
        // deep into a cutscene would otherwise replay every frame of it in one paint.
        self.firing
            .retain(|id, _| self.fired.iter().any(|held| held.id == *id));
        for held in &self.fired {
            let Some(EffectState::Ready(parsed, ..)) = self
                .effect_at
                .get(&held.path)
                .map(|at| &self.effect_files[*at].state)
            else {
                continue;
            };
            let end = match parsed.bounded {
                true => parsed.length,
                false => avfx::sim::LONGEST,
            };
            parsed.seek(self.firing.entry(held.id).or_default(), held.frame.clamp(0, end));
        }
    }

    /// The two apricot packages an effect is drawn with, fetched once the zone places any at all.
    fn load_effect_packages(&mut self, backend: &Backend) {
        if self.effects.is_empty() && self.fired.is_empty() {
            return;
        }
        for (held, path) in [
            (&mut self.effect_shape, avfx::program::SHAPE),
            (&mut self.effect_model, avfx::program::MODEL),
        ] {
            if held.is_none() {
                let files = backend.files().clone();
                *held = Some(avfx::Package::Fetching(TrackedPromise::spawn_local(
                    async move { files.read(path).await },
                )));
            }
        }

        let mut arrived = false;
        for held in [&mut self.effect_shape, &mut self.effect_model] {
            let Some(avfx::Package::Fetching(promise)) = held else {
                continue;
            };
            let Some(result) = promise.try_get() else {
                continue;
            };
            arrived = true;
            *held = Some(match result {
                Ok(bytes) => avfx::Package::Ready(bytes.clone()),
                Err(why) => {
                    log::error!("assets/layer: apricot: {why}");
                    avfx::Package::Failed
                }
            });
        }
        if arrived {
            let held = |package: &Option<avfx::Package>| match package {
                Some(avfx::Package::Ready(bytes)) => Some(bytes.clone()),
                _ => None,
            };
            self.effect_packages = Arc::new(avfx::gpu::Packages {
                shape: held(&self.effect_shape),
                model: held(&self.effect_model),
            });
        }
    }

    /// Asks for whatever the scene still needs and takes in whatever arrived. Runs every frame.
    fn poll(&mut self, ui: &egui::Ui, backend: &Backend) {
        let step = ui.input(|input| input.stable_dt).min(0.5);
        let until = Instant::now() + Duration::from_secs_f32(step * SHARE).max(LEAST);
        self.load_terrain(backend);
        self.load_grass(backend);
        self.load_asides(backend);
        self.load_effects(backend);
        self.load_effect_packages(backend);
        self.sound.poll(backend, self.camera.position);
        self.ambient.poll(backend);
        self.expand(backend, until);
        if self.fitted == 0 && !self.placements.is_empty() {
            self.fit();
        }
        self.load_models(backend, until);
        self.load_materials(backend);
        self.load_packages(backend);
        self.load_textures(ui, backend);
        self.translate();

        // Parsing, decoding and uploading are all spread over frames, and a promise only asks for
        // repaints while it is still in flight. Without this the last of a load stalls half drawn
        // until something else happens to redraw the browser.
        if !self.waiting.is_empty()
            || self.ambient.pending()
            || self.renderer.lock().unwrap().pending() > 0
            || self
                .files
                .values()
                .any(|held| matches!(held, Held::Parsing(_)))
            || self
                .models
                .iter()
                .any(|model| matches!(model.state, State::Decoding(..)))
            || self
                .effect_files
                .iter()
                .any(|effect| matches!(effect.state, EffectState::Fetching(_)))
        {
            ui.ctx().request_repaint();
        }
    }

    /// Drives the files the scene is still reading placements out of.
    fn expand(&mut self, backend: &Backend, until: Instant) {
        let mut fetching = self
            .files
            .values()
            .filter(|held| matches!(held, Held::Fetching(_)))
            .count();
        for expand in &self.waiting {
            if fetching >= FILES {
                break;
            }
            if self.files.contains_key(&expand.path) {
                continue;
            }
            let files = backend.files().clone();
            let wanted = expand.path.clone();
            self.files.insert(
                expand.path.clone(),
                Held::Fetching(TrackedPromise::spawn_local(async move {
                    files.read(&wanted).await
                })),
            );
            fetching += 1;
        }

        for (path, held) in &mut self.files {
            let Held::Fetching(promise) = held else {
                continue;
            };
            let Some(result) = promise.try_get() else {
                continue;
            };
            *held = match result {
                Ok(bytes) => Held::Parsing(bytes.clone()),
                Err(why) => {
                    log::error!("assets/layer: {path}: {why}");
                    Held::Failed
                }
            };
        }

        let parsing: Vec<String> = self
            .files
            .iter()
            .filter(|(_, held)| matches!(held, Held::Parsing(_)))
            .map(|(path, _)| path.clone())
            .collect();
        for path in parsing {
            let Some(Held::Parsing(bytes)) = self.files.remove(&path) else {
                continue;
            };
            let read = Cursor::new(bytes);
            let parsed = match path.ends_with(".sgb") {
                true => SharedGroupFile::read(read).map(Source::Shared),
                false => LayerGroupFile::read(read).map(Source::Group),
            };
            let held = match parsed {
                Ok(source) => Held::Ready(Rc::new(source)),
                Err(why) => {
                    log::error!("assets/layer: {path}: {why}");
                    Held::Failed
                }
            };
            self.files.insert(path, held);
            if Instant::now() >= until {
                break;
            }
        }

        let mut waiting = std::mem::take(&mut self.waiting);
        let mut ready = Vec::new();
        waiting.retain(|expand| match self.files.get(&expand.path) {
            Some(Held::Ready(source)) => {
                ready.push((
                    source.clone(),
                    expand.transform,
                    expand.key,
                    expand.scale,
                    expand.layer,
                    expand.depth,
                    expand.chain.clone(),
                    expand.since,
                ));
                false
            }
            // A file that would not arrive takes its subtree with it rather than being asked for
            // again every frame.
            Some(Held::Failed) => false,
            _ => true,
        });
        self.waiting = waiting;
        for (source, transform, key, scale, layer, depth, chain, since) in ready {
            self.walk(
                source.groups(),
                source.scene().map_or(&[][..], SceneTimeline::of),
                source.scene().map_or(&[][..], SceneAnimation::of),
                source.scene().map_or(&[][..], SceneSpin::of),
                source.scene().map_or(&[][..], SceneGlow::of),
                transform,
                key,
                scale,
                layer,
                depth,
                None,
                &chain,
                since,
            );
        }
    }

    fn load_models(&mut self, backend: &Backend, until: Instant) {
        let fetching = self
            .models
            .iter()
            .filter(|model| matches!(model.state, State::Fetching(_)))
            .count();
        if fetching < MODELS {
            let mut wanted: Vec<usize> = (0..self.models.len())
                .filter(|at| {
                    let model = &self.models[*at];
                    model.nearest <= self.load
                        && match model.state {
                            State::Wanted => true,
                            State::Ready => model.finest < model.asked,
                            _ => false,
                        }
                })
                .collect();
            wanted.sort_by(|a, b| {
                self.models[*a]
                    .nearest
                    .partial_cmp(&self.models[*b].nearest)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for at in wanted.into_iter().take(MODELS - fetching) {
                let files = backend.files().clone();
                let path = self.models[at].path.clone();
                let lod = self.models[at].finest;
                self.models[at].asked = lod;
                self.models[at].state = State::Fetching(TrackedPromise::spawn_local(async move {
                    files.read_model(&path, lod).await
                }));
            }
        }

        for model in &mut self.models {
            let State::Fetching(promise) = &model.state else {
                continue;
            };
            let Some(result) = promise.try_get() else {
                continue;
            };
            model.state = match result {
                Ok((bytes, level)) => State::Decoding(bytes.clone(), *level),
                Err(why) => {
                    log::error!("assets/layer: {}: {why}", model.path);
                    State::Failed
                }
            };
        }

        let decoding: Vec<usize> = (0..self.models.len())
            .filter(|at| matches!(self.models[*at].state, State::Decoding(..)))
            .collect();
        for at in decoding {
            let State::Decoding(bytes, level) =
                std::mem::replace(&mut self.models[at].state, State::Failed)
            else {
                continue;
            };
            match self.decode(at, bytes, level) {
                Ok(()) => {
                    self.models[at].state = State::Ready;
                    self.dirty = true;
                }
                Err(why) => log::error!("assets/layer: {}: {why}", self.models[at].path),
            }
            if Instant::now() >= until {
                break;
            }
        }
    }

    /// Reads one detail level of a model and hands its geometry to the card.
    fn decode(&mut self, at: usize, bytes: Vec<u8>, level: u8) -> Result<()> {
        let path = self.models[at].path.clone();
        let container = ModelContainer::read(Cursor::new(bytes))?;
        let model = container.model(mdl::detail(level));
        let mut built = Vec::new();
        let mut used = Vec::new();
        for mesh in model.meshes() {
            if !mdl::draws(&mesh) {
                continue;
            }
            let (Ok(attributes), Ok(indices)) = (mesh.attributes(), mesh.indices()) else {
                continue;
            };
            let Ok(geometry) = mdl::build(&attributes, indices) else {
                continue;
            };
            let name = mesh.material().unwrap_or_default();
            let resolved = mdl::material::path(&path, &name, 0, None).unwrap_or(name);
            used.push(self.material(&resolved));
            built.push(geometry);
        }

        let level = usize::from(level);
        // A model may carry no standard mesh at the level it was read at, which plenty of terrain
        // plates do not. That is what the model holds rather than a failure to read it, and `drawn`
        // already says so.
        let mut drawn = [false; 3];
        drawn[level] = !built.is_empty();
        let mut levels: Vec<Vec<_>> = (0..3).map(|_| Vec::new()).collect();
        levels[level] = built;
        let mut meshes: Vec<Vec<usize>> = (0..3).map(|_| Vec::new()).collect();
        meshes[level] = used;
        self.models[at].drawn = drawn;
        self.models[at].meshes = meshes;
        self.models[at].waving = model.waving();
        self.models[at].casts = model.shadowing();
        self.renderer
            .lock()
            .unwrap()
            .queue_model(gpu::Pending { model: at, levels });
        Ok(())
    }

    fn load_materials(&mut self, backend: &Backend) {
        let fetching = self
            .materials
            .iter()
            .filter(|(_, slot)| matches!(slot, Slot::Fetching(_)))
            .count();
        // Only what a model that has arrived actually names, so a slot claimed by a model still
        // waiting on its own bytes costs nothing.
        let mut wanted: Vec<usize> = self
            .models
            .iter()
            .filter(|model| matches!(model.state, State::Ready) && model.nearest <= self.load)
            .flat_map(|model| model.meshes.iter().flatten().copied())
            .filter(|at| matches!(self.materials[*at].1, Slot::Wanted))
            .collect();
        wanted.sort_unstable();
        wanted.dedup();
        for at in wanted.into_iter().take(MATERIALS.saturating_sub(fetching)) {
            let files = backend.files().clone();
            let path = self.materials[at].0.clone();
            self.materials[at].1 = Slot::Fetching(TrackedPromise::spawn_local(async move {
                files.read(&path).await
            }));
        }

        for (path, slot) in &mut self.materials {
            let Slot::Fetching(promise) = slot else {
                continue;
            };
            let Some(result) = promise.try_get() else {
                continue;
            };
            *slot = match result {
                Ok(bytes) => match Material::parse(bytes) {
                    Ok(material) => Slot::Ready(Box::new(material)),
                    Err(why) => {
                        log::error!("assets/layer: {path}: {why}");
                        Slot::Failed
                    }
                },
                Err(why) => {
                    log::error!("assets/layer: {path}: {why}");
                    Slot::Failed
                }
            };
        }
    }

    /// The packages the ready materials name, plus the ones the frame is lit and resolved with.
    fn load_packages(&mut self, backend: &Backend) {
        let mut wanted: Vec<String> = [
            program::VIEW_POSITION,
            program::DIRECTIONAL,
            program::POINT,
            program::COMPOSITE,
        ]
        .map(str::to_owned)
        .to_vec();
        wanted.extend(program::PARAMETERS.map(|(_, path)| path.to_owned()));
        // Only where the environment states an exposure to run them under. A zone with no tone
        // mapping set of its own is left as the composite resolved it, so the six files it would
        // take are never asked for.
        if self.ambient.exposure(0.0).is_some() {
            wanted.extend(program::MEASURE.map(str::to_owned));
        }
        wanted.extend(program::GLARE.map(str::to_owned));
        wanted.extend(program::REFLECTION.map(str::to_owned));
        wanted.extend(program::WATER_MIRROR.map(str::to_owned));
        wanted.extend([
            program::FXAA_LUMA.to_owned(),
            program::FXAA.to_owned(),
            program::SKY.to_owned(),
            program::SUN.to_owned(),
            program::MOON.to_owned(),
            program::SHADOW.to_owned(),
            program::VIGNETTE.to_owned(),
            program::DOWN_SCALE.to_owned(),
            program::GATHER.to_owned(),
            self.look.occluder(),
        ]);
        // Only where the weather states a fog of its own, the same way the exposure chain is only
        // asked for where there is something to run it under.
        if self.ambient.fog().is_some() {
            wanted.push(program::FOG.to_owned());
        }
        if self.ambient.clouds().is_some() {
            wanted.push(program::CLOUD.to_owned());
            wanted.push(program::CLOUD_SHADOW.to_owned());
            wanted.push(program::CLOUD_SHADOW_VERTEX.to_owned());
        }
        if self.ambient.starfield().is_some() {
            wanted.push(program::STAR_VERTEX.to_owned());
            wanted.push(program::STAR_PIXEL.to_owned());
        }
        if matches!(self.grass, Grass::Placing(_)) {
            wanted.push(program::GRASS.to_owned());
        }
        // A spot's package is twice the size of a point's and nothing can be lit with it until the
        // four above are in hand, so it is only worth a fetch of its own once they are and the zone
        // turns out to place one.
        for (kind, path) in [
            (program::LampKind::Line, program::LINE),
            (program::LampKind::Plane, program::PLANE),
        ] {
            if self.lighting.is_some() && self.lights.iter().any(|light| light.kind == kind) {
                wanted.push(path.to_owned());
            }
        }
        if self.lighting.is_some()
            && self
                .lights
                .iter()
                .any(|light| matches!(light.kind, program::LampKind::Spot))
        {
            wanted.push(program::SPOT.to_owned());
        }
        // A package the frame itself is drawn with selects off no material, so it is read whole
        // however many surfaces also name it: nothing here would know which of its blobs to ask for.
        let mut named: HashSet<String> = self
            .materials
            .iter()
            .filter_map(|(_, slot)| match slot {
                Slot::Ready(material) => Some(material.package()),
                _ => None,
            })
            .collect();
        for path in &wanted {
            named.remove(path);
        }
        wanted.extend(named.iter().cloned());
        for path in wanted {
            self.packages.entry(path).or_insert(Package::Wanted);
        }

        let mut fetching = self
            .packages
            .values()
            .filter(|held| matches!(held, Package::Fetching(_)))
            .count();
        for (path, held) in &mut self.packages {
            if fetching >= PACKAGES {
                break;
            }
            if !matches!(held, Package::Wanted) {
                continue;
            }
            let files = backend.files().clone();
            let wanted = path.clone();
            let holed = named.contains(path);
            *held = Package::Fetching(TrackedPromise::spawn_local(async move {
                match program::unnamed(&wanted) {
                    Some(hash) => Ok((
                        files.read_by_hash(program::SHADER.0, program::SHADER.1, hash, true).await?,
                        false,
                    )),
                    None if holed => files.read_package(&wanted).await,
                    None => Ok((files.read(&wanted).await?, false)),
                }
            }));
            fetching += 1;
        }

        let mut arrived = false;
        for (path, held) in &mut self.packages {
            let Package::Fetching(promise) = held else {
                continue;
            };
            let Some(result) = promise.try_get() else {
                continue;
            };
            *held = match result {
                Ok((bytes, holed)) => {
                    if *holed
                        && let Ok(package) = ShaderPackage::parse(bytes)
                    {
                        self.blobs.insert(path.clone(), Blobs::read(&package));
                    }
                    Package::Ready(bytes.clone())
                }
                Err(why) => {
                    log::error!("assets/layer: {path}: {why}");
                    Package::Failed
                }
            };
            arrived = true;
        }
        if arrived {
            self.load_types();
        }
        self.want_blobs();
        self.load_blobs(backend);
    }

    /// Which shaders the surfaces read so far will be drawn with, asked of each package once per
    /// wave of materials rather than once per material: reading a package's tables is what costs.
    fn want_blobs(&mut self) {
        let mut fresh: HashMap<String, Vec<usize>> = HashMap::new();
        for (at, (_, slot)) in self.materials.iter().enumerate() {
            let Slot::Ready(material) = slot else {
                continue;
            };
            if self.picked.contains(&at) || !self.blobs.contains_key(&material.package()) {
                continue;
            }
            fresh.entry(material.package()).or_default().push(at);
        }
        for (path, held) in fresh {
            let Some(Package::Ready(bytes)) = self.packages.get(&path) else {
                continue;
            };
            let package = match ShaderPackage::parse(bytes) {
                Ok(package) => package,
                Err(why) => {
                    log::error!("assets/layer: {path}: {why}");
                    continue;
                }
            };
            let Some(blobs) = self.blobs.get_mut(&path) else {
                continue;
            };
            for at in held {
                let Some((_, Slot::Ready(material))) = self.materials.get(at) else {
                    continue;
                };
                // Both readings of the wind, since a model that carries it may be read after the
                // material it shares with one that does not.
                for waving in [false, true] {
                    let keys = engine_keys(&path, waving);
                    for (pass, subview) in DRAWS {
                        let Some((vertex, pixel)) =
                            program::picks(&package, material, &keys, pass, subview)
                        else {
                            continue;
                        };
                        for shader in [vertex, pixel] {
                            if !blobs.arrived.contains(&shader)
                                && (shader as usize) < blobs.spans.len()
                            {
                                blobs.wanted.insert(shader);
                            }
                        }
                    }
                }
                self.picked.insert(at);
            }
        }
    }

    /// Asks for the bytecode a package still owes, and splices what arrives back where the file
    /// itself would have carried it.
    fn load_blobs(&mut self, backend: &Backend) {
        for (path, blobs) in &mut self.blobs {
            if let Some(promise) = blobs.fetching.take() {
                match promise.try_take() {
                    Err(promise) => blobs.fetching = Some(promise),
                    Ok(Err(why)) => {
                        log::error!("assets/layer: {path}: {why}");
                        blobs.wanted.clear();
                        self.packages.insert(path.clone(), Package::Failed);
                    }
                    Ok(Ok(filled)) => {
                        let Some(Package::Ready(bytes)) = self.packages.get_mut(path) else {
                            continue;
                        };
                        for (at, blob) in filled {
                            let span = blobs.spans[at as usize].clone();
                            if let Some(held) =
                                bytes.get_mut(span.start as usize..span.end as usize)
                            {
                                held.copy_from_slice(&blob);
                                blobs.arrived.insert(at);
                            }
                            blobs.wanted.remove(&at);
                        }
                        self.dirty = true;
                    }
                }
            }
            if blobs.fetching.is_some() || blobs.wanted.is_empty() {
                continue;
            }
            let files = backend.files().clone();
            let held = path.clone();
            let spans: Vec<(u32, std::ops::Range<u32>)> = blobs
                .wanted
                .iter()
                .map(|at| (*at, blobs.spans[*at as usize].clone()))
                .collect();
            blobs.fetching = Some(TrackedPromise::spawn_local(async move {
                futures_util::future::try_join_all(spans.into_iter().map(|(at, span)| {
                    let files = files.clone();
                    let held = held.clone();
                    async move { Ok((at, files.read_span(&held, span).await?)) }
                }))
                .await
            }));
        }
    }

    /// The table the shading passes index, from the parameter files that have arrived. A zone's
    /// surfaces name the background family's profiles, and the frame stands in with a table of
    /// nought until the file holding them lands.
    fn load_types(&mut self) {
        let files: Vec<(usize, ShaderParameters)> = program::PARAMETERS
            .iter()
            .filter_map(|(base, path)| {
                let Some(Package::Ready(bytes)) = self.packages.get(*path) else {
                    return None;
                };
                match ShaderParameters::read(Cursor::new(bytes.clone())) {
                    Ok(file) => Some((*base, file)),
                    Err(why) => {
                        log::error!("assets/layer: {path}: {why}");
                        None
                    }
                }
            })
            .collect();
        if files.len() == self.typed {
            return;
        }
        self.typed = files.len();
        let held: Vec<(usize, &ShaderParameters)> =
            files.iter().map(|(base, file)| (*base, file)).collect();
        self.renderer
            .lock()
            .unwrap()
            .queue_types(program::shader_types(&held));
    }

    /// One of the packages the frame itself is drawn with, translated where it has arrived.
    fn screen(
        &self,
        path: &str,
        pass: program::Pass,
        attachments: usize,
        keys: &[(u32, u32)],
    ) -> Option<Arc<program::Program>> {
        let Some(Package::Ready(bytes)) = self.packages.get(path) else {
            return None;
        };
        program::Program::screen(bytes, pass, attachments, keys)
            .inspect_err(|why| log::warn!("assets/layer: {path}: {why}"))
            .ok()
            .map(Arc::new)
    }

    /// A lighting package translated once at each falloff power a light can name, since a light
    /// picks its own and every one of them is one shader of its own.
    ///
    /// Clipped, which is what keeps a light inside the volume its zone cut for it: the pass reads
    /// the pixel it shades in the light's own space and drops it outside the box, and without that
    /// a lamp in a room reaches its whole distance through the walls.
    fn lamp(&self, path: &str, attachments: usize) -> Option<mdl::deferred::Falloffs> {
        let [linear, quadratic, cubic] = program::ATTENUATION.map(|value| {
            self.screen(
                path,
                program::Pass::Lamp,
                attachments,
                &[
                    (program::APPLY_ATTENUATION, value),
                    (program::LIGHT_CLIP, program::LIGHT_CLIP_ENABLE),
                ],
            )
        });
        Some([linear?, quadratic?, cubic?])
    }

    /// One member of the post chain, translated where its file has arrived.
    fn effect(&self, path: &str, vertex: &str) -> Option<Arc<program::Program>> {
        let Some(Package::Ready(bytes)) = self.packages.get(path) else {
            return None;
        };
        program::Program::posteffect(path, bytes, vertex)
            .inspect_err(|why| log::warn!("assets/layer: {path}: {why}"))
            .ok()
            .map(Arc::new)
    }

    /// The chain that reflects the frame off itself, translated once its nine shaders have arrived.
    /// Every member is drawn with the vertex shader the game pairs it with.
    fn mirror(&self) -> Option<Arc<mdl::deferred::Reflection>> {
        let ready = |path: &str| match self.packages.get(path) {
            Some(Package::Ready(bytes)) => Some(bytes),
            _ => None,
        };
        let held = |path: &str, vertex: &str| {
            program::Program::sampling(path, ready(path)?, ready(vertex)?)
                .inspect_err(|why| log::warn!("assets/layer: {path}: {why}"))
                .ok()
                .map(Arc::new)
        };
        let read = |path: &str| held(path, program::REFLECTION_VERTEX);
        Some(Arc::new(mdl::deferred::Reflection {
            normal: read(program::REFLECTION_NORMAL)?,
            mask: read(program::REFLECTION_MASK)?,
            march: read(program::REFLECTION_MARCH)?,
            blur: [
                read(program::REFLECTION_BLUR_X)?,
                read(program::REFLECTION_BLUR_Y)?,
            ],
            distort: read(program::REFLECTION_DISTORT)?,
            copy: held(program::REFLECTION_COPY, program::REFLECTION_MERGE_VERTEX)?,
        }))
    }

    /// The chain that fills what water reflects itself through, translated once its ten shaders
    /// have arrived. Each member is drawn with the vertex shader the game pairs it with, and the two
    /// that run over the water itself with the one shared vertex shader.
    fn watering(&self) -> Option<Arc<mdl::deferred::WaterMirror>> {
        let ready = |path: &str| match self.packages.get(path) {
            Some(Package::Ready(bytes)) => Some(bytes),
            _ => None,
        };
        let held = |path: &str, vertex: &str| {
            let mut held = program::Program::sampling(path, ready(path)?, ready(vertex)?)
                .inspect_err(|why| log::warn!("assets/layer: {path}: {why}"))
                .ok()?;
            held.pass = program::Pass::WaterMirror;
            Some(Arc::new(held))
        };
        let over = |path: &str| held(path, program::WATER_MIRROR_VERTEX);
        Some(Arc::new(mdl::deferred::WaterMirror {
            mask: over(program::WATER_MIRROR_MASK)?,
            march: over(program::WATER_MIRROR_MARCH)?,
            blur: [
                held(program::WATER_MIRROR_BLUR, program::WATER_MIRROR_BLUR_X)?,
                held(program::WATER_MIRROR_BLUR, program::WATER_MIRROR_BLUR_Y)?,
            ],
            wide: held(program::WATER_MIRROR_WIDE, program::WATER_MIRROR_WIDE_X)?,
            merge: held(
                program::WATER_MIRROR_MERGE,
                program::WATER_MIRROR_MERGE_VERTEX,
            )?,
        }))
    }

    /// The pair that smooths the frame's edges, and the three that work out how much sky reaches
    /// each pixel, each translated once all of its own shaders have arrived.
    fn edges(&self) -> Option<Arc<mdl::gpu::Smoothing>> {
        Some(Arc::new(mdl::gpu::Smoothing {
            luma: self.effect(program::FXAA_LUMA, program::POST_VERTEX)?,
            fxaa: self.effect(program::FXAA, program::POST_VERTEX)?,
        }))
    }

    /// The chain that spreads the bright end of the frame, translated once its shaders have arrived.
    /// The two smoothing passes read nine and seven coordinates rather than one, so each is drawn
    /// with the vertex shader the game pairs it with rather than the one every other pass here takes.
    fn halo(&self) -> Option<Arc<mdl::gpu::Glare>> {
        let sampled = |path: &str, vertex: &str| {
            let Some(Package::Ready(held)) = self.packages.get(vertex) else {
                return None;
            };
            let Some(Package::Ready(bytes)) = self.packages.get(path) else {
                return None;
            };
            program::Program::sampling(path, bytes, held)
                .inspect_err(|why| log::warn!("assets/layer: {path}: {why}"))
                .ok()
                .map(Arc::new)
        };
        Some(Arc::new(mdl::gpu::Glare {
            bright: self.effect(program::BRIGHT_PASS, program::POST_VERTEX)?,
            gauss: sampled(program::GAUSS_BLUR, program::SAMPLING_9)?,
            blur: sampled(program::BLOOM_BLUR, program::SAMPLING_7)?,
            merge: self.effect(program::GLARE_MERGE, program::POST_VERTEX)?,
            composite: self.effect(program::GLARE_COMPOSITE, program::POST_VERTEX)?,
        }))
    }

    /// The chain that works out how much of the sky reaches each pixel, translated once its three
    /// shaders have arrived. The taps ship as a file per quality, so a change there is a chain of
    /// its own rather than a constant.
    fn occluders(&self) -> Option<Arc<mdl::gpu::Occlusion>> {
        Some(Arc::new(mdl::gpu::Occlusion {
            scale: self.effect(program::DOWN_SCALE, program::POST_VERTEX)?,
            gather: self.effect(program::GATHER, program::GATHER_VERTEX)?,
            occlude: self.effect(&self.look.occluder(), program::POST_VERTEX)?,
        }))
    }

    /// The exposure chain, translated once all six of its shaders have arrived. The three that
    /// halve the frame read four texels of a square rather than one, so they are drawn with the
    /// vertex shader that names those four.
    fn measure(&self) -> Option<Arc<mdl::gpu::Exposure>> {
        let held = |path: &str, vertex| self.effect(path, vertex);
        Some(Arc::new(mdl::gpu::Exposure {
            initial: held(program::MEASURE_INITIAL, program::SAMPLING_VERTEX)?,
            iterative: held(program::MEASURE_ITERATIVE, program::SAMPLING_VERTEX)?,
            last: held(program::MEASURE_FINAL, program::SAMPLING_VERTEX)?,
            adapt: held(program::ADAPT_LUM, program::POST_VERTEX)?,
            curve: held(program::TONE_MAP_LUT, program::POST_VERTEX)?,
            tone: held(program::TONE_MAPPING, program::POST_VERTEX)?,
        }))
    }

    /// Every ready material's shaders, translated once its package has arrived. A context that
    /// turned out to write fewer of the G-buffer's targets at once has them translated again.
    fn translate(&mut self) {
        let attachments = self.renderer.lock().unwrap().attachments();
        if self.lighting.is_none()
            && let (Some(position), Some(directional), Some(point), Some(composite)) = (
                self.screen(program::VIEW_POSITION, program::Pass::Lighting, attachments, &[]),
                self.screen(program::DIRECTIONAL, program::Pass::Lighting, attachments, &[]),
                self.lamp(program::POINT, attachments),
                self.screen(program::COMPOSITE, program::Pass::Composite, attachments, &[]),
            )
        {
            self.lighting = Some(Arc::new(mdl::gpu::Lighting {
                position,
                directional,
                point,
                spot: None,
                shadow: None,
                line: None,
                plane: None,
                subsurface: None,
                // The fifth target's alpha is a background surface's emissive flag rather than the
                // scale a strand is marched along, so the fur pass has nothing here to read.
                fur: None,
                composite,
            }));
        }
        // The frame lights without a spot's own package and takes it up on whichever frame it
        // arrives on: waiting for it would leave a zone unlit until it did. A package that arrived
        // and would not translate is marked failed rather than translated again every frame.
        if let Some(lighting) = self.lighting.clone()
            && lighting.spot.is_none()
            && matches!(self.packages.get(program::SPOT), Some(Package::Ready(_)))
        {
            let spot = self.lamp(program::SPOT, attachments);
            if spot.is_none() {
                self.packages
                    .insert(program::SPOT.to_owned(), Package::Failed);
            }
            self.lighting = Some(Arc::new(mdl::gpu::Lighting {
                spot,
                ..(*lighting).clone()
            }));
        }
        for (path, take) in [
            (program::LINE, 0usize),
            (program::PLANE, 1usize),
        ] {
            let Some(lighting) = self.lighting.clone() else {
                continue;
            };
            let held = match take {
                0 => lighting.line.is_none(),
                _ => lighting.plane.is_none(),
            };
            if !held || !matches!(self.packages.get(path), Some(Package::Ready(_))) {
                continue;
            }
            let built = self.lamp(path, attachments);
            if built.is_none() {
                self.packages.insert(path.to_owned(), Package::Failed);
            }
            self.lighting = Some(Arc::new(match take {
                0 => mdl::gpu::Lighting {
                    line: built,
                    ..(*lighting).clone()
                },
                _ => mdl::gpu::Lighting {
                    plane: built,
                    ..(*lighting).clone()
                },
            }));
        }
        // The same, for the pass that works out how much of the sun reaches a pixel: a zone lights
        // unshadowed until its package is in hand rather than waiting on one.
        if let Some(lighting) = self.lighting.clone()
            && lighting.shadow.is_none()
            && matches!(self.packages.get(program::SHADOW), Some(Package::Ready(_)))
        {
            // The strongest softening rather than a fixed square: a square of any width blurs an
            // edge by as much where it meets the thing casting it as it does far away from it. Both
            // keys are asked for here alone, so no other package moves with them.
            let shadow = self.screen(
                program::SHADOW,
                program::Pass::Lighting,
                attachments,
                &[
                    (program::SHADOW_SOFT, program::SHADOW_SOFT_PCSS),
                    (program::TRANSFORM_PROJ, program::TRANSFORM_PROJ_PLANE_FAR),
                ],
            );
            if shadow.is_none() {
                self.packages
                    .insert(program::SHADOW.to_owned(), Package::Failed);
            }
            // The dither it turns each pixel's disc by, which the engine binds and no material ever
            // names as a path.
            for texture in shadow.iter().flat_map(|held| &held.textures) {
                if let Some((id, path, _)) = mdl::deferred::ENGINE
                    .iter()
                    .find(|(held, _, _)| *held == texture.id)
                {
                    self.engine
                        .entry(*id)
                        .or_insert_with(|| Aside::Wanted(path.to_string()));
                }
            }
            self.lighting = Some(Arc::new(mdl::gpu::Lighting {
                shadow,
                ..(*lighting).clone()
            }));
        }
        if self.exposure.is_none() {
            self.exposure = self.measure();
        }
        if self.sunlight.is_none() {
            self.sunlight = self.effect(program::SUN, program::POST_VERTEX);
        }
        if self.moonlight.is_none() {
            self.moonlight = self.effect(program::MOON, program::MOON_VERTEX);
        }
        if self.skybox.is_none() {
            self.skybox = self.effect(program::SKY, program::SKY_VERTEX);
        }
        if self.haze.is_none() {
            self.haze = self.effect(program::FOG, program::POST_VERTEX);
        }
        if self.clouds[0].is_none()
            && let Some(Package::Ready(bytes)) = self.packages.get(program::CLOUD)
        {
            for (at, pass) in [program::Pass::CloudBand, program::Pass::CloudSheet]
                .into_iter()
                .enumerate()
            {
                self.clouds[at] = program::Program::cloud(bytes, pass, attachments)
                    .inspect_err(|why| log::warn!("assets/layer: {}: {why}", program::CLOUD))
                    .ok()
                    .map(Arc::new);
            }
        }
        // One target, whatever the frame's own packing takes: the map holds a weight rather than a
        // channel of the G-buffer.
        if self.cloud_shadow.is_none()
            && let Some(Package::Ready(bytes)) = self.packages.get(program::CLOUD)
            && let (Some(Package::Ready(blur)), Some(Package::Ready(vertex))) = (
                self.packages.get(program::CLOUD_SHADOW),
                self.packages.get(program::CLOUD_SHADOW_VERTEX),
            )
        {
            let sheet = program::Program::cloud(bytes, program::Pass::CloudShadow, 1)
                .inspect_err(|why| log::warn!("assets/layer: {}: {why}", program::CLOUD))
                .ok();
            let held = program::Program::sampling(program::CLOUD_SHADOW, blur, vertex)
                .inspect_err(|why| log::warn!("assets/layer: {}: {why}", program::CLOUD_SHADOW))
                .ok()
                .map(|mut held| {
                    held.pass = program::Pass::CloudShadow;
                    held
                });
            self.cloud_shadow = sheet
                .zip(held)
                .map(|(sheet, blur)| (Arc::new(sheet), Arc::new(blur)));
        }
        if self.starlight.is_none()
            && let (Some(Package::Ready(vertex)), Some(Package::Ready(fragment))) = (
                self.packages.get(program::STAR_VERTEX),
                self.packages.get(program::STAR_PIXEL),
            )
        {
            self.starlight = program::Program::stars(vertex, fragment)
                .inspect_err(|why| log::warn!("assets/layer: {}: {why}", program::STAR_VERTEX))
                .ok()
                .map(Arc::new);
        }
        if self.sward.is_none()
            && let Some(Package::Ready(bytes)) = self.packages.get(program::GRASS)
        {
            let read = |normal, page| {
                program::Program::grass(bytes, normal, page, attachments)
                    .inspect_err(|why| log::warn!("assets/layer: {}: {why}", program::GRASS))
                    .ok()
                    .map(Arc::new)
            };
            self.sward = read(false, 0).map(|first| {
                let pages = first.outputs.len().div_ceil(attachments.max(1)).max(1);
                let mut buffer = vec![first];
                buffer.extend((1..pages).filter_map(|page| read(false, page)));
                Arc::new(gpu::Grass {
                    buffer,
                    normal: (0..pages).filter_map(|page| read(true, page)).collect(),
                })
            });
        }
        if self.glare.is_none() {
            self.glare = self.halo();
        }
        if self.reflection.is_none() {
            self.reflection = self.mirror();
        }
        if self.water_mirror.is_none() {
            self.water_mirror = self.watering();
        }
        if self.smoothing.is_none() {
            self.smoothing = self.edges();
        }
        if self.occlusion.is_none() {
            self.occlusion = self.occluders();
        }
        if self.vignette.is_none() {
            // Against the sky's own vertex shader, which is the one here handing a fragment where it
            // stands rather than what to read.
            self.vignette = self.effect(program::VIGNETTE, program::SKY_VERTEX);
        }

        // Which reading of a material to take is the model's to say, and most of the materials on a
        // model the wind reaches are also on one that stands still. So a material shared that way
        // is translated twice, and one that is not is translated once.
        let mut readings: Vec<(usize, bool)> = self
            .models
            .iter()
            .flat_map(|model| {
                model
                    .meshes
                    .iter()
                    .flatten()
                    .map(move |at| (*at, model.waving))
            })
            .collect();
        readings.sort_unstable();
        readings.dedup();
        // One reading of each package, shared by every surface that names it.
        let mut read: HashMap<String, ShaderPackage> = HashMap::new();
        for (at, waving) in readings {
            let Some((_, Slot::Ready(material))) = self.materials.get(at) else {
                continue;
            };
            if self
                .translated
                .get(&(at, waving))
                .is_some_and(|held| held.attachments == attachments)
            {
                continue;
            }
            let name = material.package();
            let Some(Package::Ready(bytes)) = self.packages.get(&name) else {
                continue;
            };
            // A package still owed bytecode holds nought where a shader would be, which translates
            // to a program that draws nothing rather than to an error worth reporting.
            if self
                .blobs
                .get(&name)
                .is_some_and(|held| !held.wanted.is_empty())
            {
                continue;
            }
            if !read.contains_key(&name) {
                match ShaderPackage::parse(bytes) {
                    Ok(held) => {
                        read.insert(name.clone(), held);
                    }
                    Err(why) => {
                        log::error!("assets/layer: {name}: {why}");
                        continue;
                    }
                }
            }
            let package = &read[&name];
            let keys = engine_keys(&name, waving);
            let page = |pass, page| {
                program::Program::build(
                    package,
                    bytes,
                    material,
                    &keys,
                    pass,
                    program::SUB_VIEW_MAIN,
                    page,
                    attachments,
                )
            };
            // A package with no opaque pass is a surface that blends itself into the frame - water,
            // and the glass a zone places. It fills the same G-buffer through a pass of its own and
            // answers into the lit frame afterward, so which one it took has to be remembered. One
            // with neither fills none of it: a light shaft and a slab of fog are drawn over the
            // frame the lighting left and nowhere else.
            let (pass, first, opaque) = match page(program::Pass::Buffer, 0) {
                Ok(held) => (program::Pass::Buffer, Some(held), String::new()),
                Err(why) => (
                    program::Pass::Blended,
                    page(program::Pass::Blended, 0).ok(),
                    why,
                ),
            };
            let blended = pass != program::Pass::Buffer;
            let mut buffer = Vec::new();
            if let Some(first) = first {
                let pages = first.outputs.len().div_ceil(attachments.max(1)).max(1);
                buffer.push(Arc::new(first));
                buffer.extend((1..pages).filter_map(|held| page(pass, held).ok().map(Arc::new)));
            }
            // Only where the same vertex shader settled the depth. A blending surface fills the
            // buffer through a pass whose vertices are lifted by its own waves, and the depth pass
            // leaves them where the file put them: every later test against it fails.
            let depth = match blended {
                true => Err("a blending surface writes its own depth".into()),
                false => program::Program::build(
                    package,
                    bytes,
                    material,
                    &keys,
                    program::Pass::Depth,
                    program::SUB_VIEW_MAIN,
                    0,
                    attachments,
                ),
            };
            // Only where the material states a clip the semi-transparent pass's own reaches under.
            // Below that the two passes cover the same fragments, and the resolve drops every one
            // the opaque half already drew.
            let sheer = (!blended && material.clip() > program::SHEER_CLIP)
                .then(|| {
                    page(program::Pass::Blended, 0).and_then(|held| {
                        Ok((Arc::new(held), Arc::new(page(program::Pass::CompositeBlended, 0)?)))
                    })
                })
                .and_then(|result| {
                    result
                        .inspect_err(|why| {
                            log::warn!("assets/layer: {name}: no semi-transparent pass: {why}")
                        })
                        .ok()
                });
            // The same depth pass as the light sees it. A package that answers no shadow subview
            // casts none, which is what the flag on a placed instance says anyway. One that answers
            // it and then fails to translate is a fault, and is reported rather than dropped.
            let shadow = program::picks(
                package,
                material,
                &keys,
                program::Pass::Depth,
                program::SUB_VIEW_SHADOW_0,
            )
            .and_then(|_| {
                program::Program::build(
                    package,
                    bytes,
                    material,
                    &keys,
                    program::Pass::Depth,
                    program::SUB_VIEW_SHADOW_0,
                    0,
                    attachments,
                )
                .inspect_err(|why| log::warn!("assets/layer: {name}: no shadow pass: {why}"))
                .ok()
            });
            // What it answers into the lit frame with, which only a blending surface has.
            // Water reads the lit frame back and shades itself from it, where anything else that
            // blends is lit where it stands and an overlay carries its own colour.
            let resolve = blended
                .then(|| {
                    [
                        program::Pass::Water,
                        program::Pass::BlendedLighting,
                        program::Pass::Shaft,
                        program::Pass::Layer,
                    ]
                    .into_iter()
                    .find_map(|pass| page(pass, 0).ok())
                    .map(Arc::new)
                })
                .flatten();
            if buffer.is_empty() && resolve.is_none() {
                log::warn!("assets/layer: {name}: {opaque}");
                continue;
            }
            // The engine binds these rather than the material, so nothing names them as a path;
            // what a surface's own shaders declare is what says the file is worth reading at all.
            for texture in buffer.iter().chain(&resolve).flat_map(|held| &held.textures) {
                if let Some((id, path, _)) = mdl::deferred::ENGINE
                    .iter()
                    .find(|(held, _, _)| *held == texture.id)
                {
                    self.engine
                        .entry(*id)
                        .or_insert_with(|| Aside::Wanted(path.to_string()));
                }
            }
            self.translated.insert(
                (at, waving),
                Translated {
                    attachments,
                    buffer,
                    depth: depth.ok().map(Arc::new),
                    shadow: shadow.map(Arc::new),
                    resolve,
                    sheer,
                },
            );
            if let Some((values, columns, rows)) =
                material.held().color_table().and_then(program::table)
            {
                self.tables
                    .entry(at)
                    .or_insert_with(|| Arc::new((values.to_vec(), columns, rows)));
            }
            self.dirty = true;
        }
    }

    /// The textures a material names for a sampler its package declares over slices rather than a
    /// plane: an environment cube, an array, a volume.
    fn sliced(&self) -> BTreeSet<String> {
        self.translated
            .iter()
            .filter_map(|((at, _), held)| match self.materials.get(*at) {
                Some((_, Slot::Ready(material))) => Some((held, material)),
                _ => None,
            })
            .flat_map(|(held, material)| {
                material.bound().filter(|(id, _)| {
                    held.buffer
                        .iter()
                        .chain(&held.depth)
                        .chain(&held.shadow)
                        .chain(&held.resolve)
                        .flat_map(|pass| &pass.textures)
                        .any(|texture| texture.id == *id && texture.kind != program::Kind::Plane)
                })
            })
            .map(|(_, path)| path.to_owned())
            .collect()
    }

    /// Every texture the ready materials name, since the game's own shaders read all of them. Held
    /// for the whole scene rather than per model, since a zone's models share theirs heavily.
    /// The color maps the zone's grass is cut out of, where it names any.
    fn maps(&self) -> Vec<String> {
        let Grass::Placing(placing) = &self.grass else {
            return Vec::new();
        };
        placing
            .maps
            .iter()
            .filter(|path| !path.is_empty())
            .cloned()
            .collect()
    }

    fn load_textures(&mut self, ui: &egui::Ui, backend: &Backend) {
        let sliced = self.sliced();
        for path in sliced
            .iter()
            .filter(|path| !self.stacked.contains_key(path.as_str()))
            .cloned()
            .collect::<Vec<_>>()
        {
            let files = backend.files().clone();
            let held = path.clone();
            self.stacked.insert(
                path.into(),
                Stack::Fetching(TrackedPromise::spawn_local(async move {
                    files.read(&held).await
                })),
            );
        }
        for (path, stack) in &mut self.stacked {
            let Stack::Fetching(promise) = stack else {
                continue;
            };
            let Some(result) = promise.try_get() else {
                continue;
            };
            *stack = match result
                .as_ref()
                .map_err(ToString::to_string)
                .and_then(|bytes| {
                    mdl::layered(bytes, path, glow::LINEAR).map_err(|why| why.to_string())
                }) {
                Ok(held) => {
                    self.renderer.lock().unwrap().queue_stack(path.clone(), held);
                    self.dirty = true;
                    Stack::Ready
                }
                Err(why) => {
                    log::error!("assets/layer: {path}: {why}");
                    Stack::Absent
                }
            };
        }

        let mut fetching = self
            .textures
            .values()
            .filter(|texture| matches!(texture, Texture::Fetching(_)))
            .count();
        let maps = self.maps();
        // Either reading: a material the wind reaches and no still model carries is translated
        // only as a waving one.
        let drawn: HashSet<usize> = self.translated.keys().map(|(at, _)| *at).collect();
        let live: Vec<&Material> = self
            .materials
            .iter()
            .enumerate()
            .filter(|(at, _)| drawn.contains(at))
            .filter_map(|(_, (_, slot))| match slot {
                Slot::Ready(material) => Some(material.as_ref()),
                _ => None,
            })
            .collect();
        // Past its last texel a texture addresses the way its own sampler states, which for a
        // cutout is mirrored far more often than the repeat this otherwise assumes.
        let wrap_paths: BTreeMap<&str, mtrl::AddressMode> = live
            .iter()
            .flat_map(|material| {
                material
                    .bound()
                    .map(move |(id, path)| (path, material.wrap(id)))
            })
            .collect();
        let wanted: Vec<String> = live
            .iter()
            // By the sampler each is bound to, not by the four roles this viewer's own shading
            // knows: the game's shaders read every one, and water names its wave maps through
            // samplers no other package declares.
            .flat_map(|material| material.bound().map(|(_, path)| path.to_owned()))
            .chain(maps.iter().cloned())
            .chain(self.effect_files.iter().flat_map(|effect| match &effect.state {
                EffectState::Ready(parsed, ..) => parsed.textures.clone(),
                _ => Vec::new(),
            }))
            .filter(|path| !self.textures.contains_key(path) && !sliced.contains(path))
            .collect();
        for path in wanted {
            if fetching >= TEXTURES {
                break;
            }
            if self.textures.contains_key(&path) {
                continue;
            }
            if self.resident >= TEXTURE_BUDGET {
                self.textures.insert(path, Texture::Absent);
                continue;
            }
            let files = backend.files().clone();
            let held = path.clone();
            let size = match maps.contains(&path) {
                true => GRASS_SIZE,
                false => TEXTURE_SIZE,
            };
            self.textures.insert(
                path,
                Texture::Fetching(TrackedPromise::spawn_local(async move {
                    files.read_texture(&held, Some(size)).await
                })),
            );
            fetching += 1;
        }

        let mut taken = 0;
        for (path, texture) in &mut self.textures {
            let Texture::Fetching(promise) = texture else {
                continue;
            };
            let Some(result) = promise.try_get() else {
                continue;
            };
            *texture = match result {
                Ok(decoded) => {
                    let held = crate::utils::tex_loader::fit(ui.ctx(), &decoded.image);
                    let size = [held.width() as usize, held.height() as usize];
                    taken += size[0] * size[1] * 4;
                    // Premultiplied is the one path that copies the bytes through untouched, and a
                    // diffuse map's alpha is opacity rather than something the other channels
                    // should be scaled by.
                    Texture::Ready(ui.ctx().load_texture(
                        format!("scene:{path}"),
                        egui::ColorImage::from_rgba_premultiplied(
                            size,
                            held.as_flat_samples().as_slice(),
                        ),
                        TextureOptions {
                            magnification: egui::TextureFilter::Linear,
                            minification: egui::TextureFilter::Linear,
                            wrap_mode: match wrap_paths.get(path.as_str()) {
                                Some(mtrl::AddressMode::MirroredRepeat) => {
                                    egui::TextureWrapMode::MirroredRepeat
                                }
                                Some(
                                    mtrl::AddressMode::ClampToEdge
                                    | mtrl::AddressMode::ClampToBorder,
                                ) => egui::TextureWrapMode::ClampToEdge,
                                Some(mtrl::AddressMode::Repeat) | None => {
                                    egui::TextureWrapMode::Repeat
                                }
                            },
                            mipmap_mode: Some(egui::TextureFilter::Linear),
                        },
                    ))
                }
                Err(why) => {
                    log::error!("assets/layer: {path}: {why}");
                    Texture::Absent
                }
            };
        }
        self.resident += taken;
    }

    /// Which of the passes past the lighting ran. A weather that names no clouds draws none, and so
    /// does a draw that quietly went wrong; only the graph knows which of the two a frame without
    /// any is.
    fn passes(&self) -> String {
        let held = self.renderer.lock().unwrap().drawn();
        // A lamp kind whose own package is not in hand is drawn through the point one, which lights
        // a cone or a line as a sphere: naming them here is what tells the two apart on screen.
        let lit = |take: fn(&mdl::gpu::Lighting) -> bool| {
            self.lighting.as_deref().is_some_and(take)
        };
        let ran: Vec<&str> = [
            (lit(|held| held.spot.is_some()), "spot"),
            (lit(|held| held.line.is_some()), "line"),
            (lit(|held| held.plane.is_some()), "plane"),
            (held.occlusion, "occlusion"),
            (held.shadow, "shadow"),
            (held.sky, "sky"),
            (held.sun, "sun"),
            (held.moon, "moon"),
            (held.stars, "stars"),
            (held.clouds[0], "band"),
            (held.clouds[1], "sheet"),
            (held.cloud_shadow, "cloud shadow"),
            (held.fog, "fog"),
            (held.reflection, "reflection"),
            (held.water, "water mirror"),
            (held.vignette, "vignette"),
            (
                !(self.effects.is_empty() && self.fired.is_empty())
                    && self
                        .effect_files
                        .iter()
                        .any(|effect| matches!(effect.state, EffectState::Ready(..))),
                "effects",
            ),
        ]
        .into_iter()
        .filter_map(|(ran, name)| ran.then_some(name))
        .collect();
        match ran.is_empty() {
            true => "none".to_owned(),
            false => ran.join(", "),
        }
    }

    /// The viewport, and the navigation over it.
    fn viewport(&mut self, ui: &mut egui::Ui) {
        let (rect, _) = ui.allocate_exact_size(ui.available_size(), Sense::hover());
        // Interacted with under an id of its own: the details panel beside it is content sized, so
        // the rect this takes moves as the counts in there change, and an id taken from the rect
        // changes with it, which loses a press and the release that answers it.
        let response = ui.interact(rect, ui.id().with("scene"), Sense::click_and_drag());
        if rect.width() < 1.0 || rect.height() < 1.0 {
            return;
        }

        // A driven camera answers to whatever is driving it, not the mouse and keyboard. Kept on
        // the struct too, since the side panel's own controls are drawn from outside this method.
        self.driving = self.drive.is_some();
        let driving = self.driving;
        if !driving && response.dragged_by(egui::PointerButton::Primary) {
            let delta = response.drag_delta();
            self.camera.yaw -= delta.x * 0.005;
            self.camera.pitch = (self.camera.pitch - delta.y * 0.005).clamp(-1.5, 1.5);
        }
        let mut moved = Vec3::ZERO;
        if !driving && response.dragged_by(egui::PointerButton::Secondary) {
            let delta = response.drag_delta();
            moved += (self.camera.right() * delta.x + Vec3::Y * delta.y) * self.load * 0.0005;
        }
        if !driving && response.hovered() {
            let scroll = ui.input(|input| input.smooth_scroll_delta.y);
            if scroll != 0.0 {
                moved += self.camera.forward() * scroll * self.load * 0.002;
            }
        }
        // Keys only where nothing else has taken them, so typing in the browser's own fields does
        // not fly the camera along with it.
        let flying = !driving
            && (response.hovered() || response.dragged())
            && ui.memory(|memory| memory.focused().is_none());
        if flying {
            let (ahead, side, up, step) = ui.input(|input| {
                let held = |key| f32::from(input.key_down(key));
                (
                    held(egui::Key::W) - held(egui::Key::S),
                    held(egui::Key::D) - held(egui::Key::A),
                    held(egui::Key::E) - held(egui::Key::Q),
                    input.stable_dt.min(0.1)
                        * match input.modifiers.shift {
                            true => 4.0,
                            false => 1.0,
                        },
                )
            });
            if ahead != 0.0 || side != 0.0 || up != 0.0 {
                moved +=
                    (self.camera.forward() * ahead + self.camera.right() * side + Vec3::Y * up)
                        * step
                        * SPEED
                        * self.speed;
                ui.ctx().request_repaint();
            }
        }
        // Taken rather than borrowed: a host that stops driving next frame gets the free camera
        // back, and this frame still needs to know it was driven for the clip planes below.
        let driven = self.drive.take();
        if let Some(drive) = &driven {
            let forward = drive.forward.normalize_or_zero();
            self.camera = Camera {
                position: drive.position,
                yaw: forward.x.atan2(forward.z),
                pitch: forward.y.clamp(-1.0, 1.0).asin(),
            };
            self.fov = drive.fov_degrees;
        }
        if moved != Vec3::ZERO {
            self.camera.position += moved;
        }
        // Scaled by how fast the camera is set to move, so raising the speed does not turn a
        // rebuild every few seconds into one every frame.
        if (self.camera.position - self.written).length() > STEP * self.speed {
            self.dirty = true;
        }
        // A timeline states where its node stands rather than how far it has moved, so what a frame
        // draws follows the clock rather than the frame before it. The placements themselves are
        // already worked out; only where the moving ones stand is done again.
        let animated = !self.motions.is_empty() || self.cycling;
        // An effect steps off the same clock but never touches a placement, so it asks for the
        // clock and a repaint without asking for the whole scene's placements to be redone.
        if animated || !self.effects.is_empty() {
            self.clock += ui.input(|input| input.stable_dt).min(0.25) * TICKS;
            ui.ctx().request_repaint();
        }
        if animated {
            self.dirty = true;
        }
        if self.dirty {
            self.rebuild();
        }

        let eye = self.camera.position;
        // A drive's own forward/up are used directly rather than rebuilt from the yaw/pitch
        // `self.camera` stores them as: that round trip degenerates for a shot looking straight up
        // or down, which an orbit camera's clamped pitch never reaches but a cutscene's camera can.
        // The free camera never rolls itself, so a raw `Vec3::Y` hint is exactly what it wants;
        // `look_at_rh` levels it against forward on its own.
        let (forward, up) = match &driven {
            Some(drive) => (drive.forward.normalize_or_zero(), drive.up.normalize_or_zero()),
            None => (self.camera.forward(), Vec3::Y),
        };
        let view = Mat4::look_at_rh(eye, eye + forward, up);
        // A driven camera's own clip planes, where it states them: the world streaming distance
        // stays keyed on the load distance regardless.
        let (near, far) = match &driven {
            Some(drive) => (drive.near, drive.far),
            None => {
                let far = self.load * 1.5;
                // Capped as well as scaled: at the largest load distance a proportional near plane
                // would sit further out than the walls of an interior.
                ((far * 0.0002).min(0.2), far)
            }
        };
        // A driven shot's own field of view is stated for a 16:9 frame (see the `C004` doc); refit
        // it to the viewport's actual aspect so its horizontal field is what the shot states rather
        // than whatever a narrower or wider panel would crop it to.
        let vertical_fov = match &driven {
            Some(_) => refit_16_9_fov(self.fov, rect.width() / rect.height()),
            None => self.fov,
        };
        // The game's own shaders were compiled for a clip depth running from nought to one, and the
        // backend moves what they compute into the range GL clips against.
        let projection = Mat4::perspective_rh(
            vertical_fov.to_radians(),
            rect.width() / rect.height(),
            near,
            far,
        );

        let mut batches = Vec::new();
        for (at, model) in self.models.iter().enumerate() {
            for level in 0..3 {
                let instances = match self.placed.get(at) {
                    Some(held) if !held[level].is_empty() => held[level].clone(),
                    _ => continue,
                };
                let Some(meshes) = model.meshes.get(level) else {
                    continue;
                };
                batches.push(gpu::Batch {
                    model: at,
                    level,
                    casts: self.casts[at][level],
                    instances,
                    surfaces: meshes
                        .iter()
                        .map(|slot| self.surface(*slot, model.waving))
                        .collect(),
                });
            }
        }

        let attachments = self.renderer.lock().unwrap().attachments();
        let cast: Vec<mdl::Cast> = self
            .cast
            .iter()
            .map(|held| mdl::Cast {
                opacity: held.opacity,
                ..held.model.cast(matrix(held.at), attachments)
            })
            .collect();

        let (light, color) = self.ambient.light();
        let blades = self.sown();
        self.standing = blades.iter().map(|held| self.turf[held.turf].blades).sum();
        let effects = self.effect_draws(view, eye);
        let effects_drawn = effects.iter().map(|held| held.batches.len()).sum();
        let frame = gpu::Frame {
            casts: cast,
            scene: program::Scene {
                view,
                projection,
                model: Mat4::IDENTITY,
                light,
                reach: self.ambient.reach,
                diffuse: color,
                specular: color,
                ambient: self.ambient.scene(),
                // How far the adaptation moves is stated per second, so it needs to know how long a
                // frame took. A frame after an idle spell is capped by the pass itself.
                // The adaptation the last frame settled on, and the scale every pass writing into
                // the lit frame takes against it so the tone pass can divide it back out.
                exposure: {
                    let adapted = self.renderer.lock().unwrap().exposed();
                    program::Exposure {
                        adapted,
                        // Nothing divides the frame back out until the chain that does has arrived,
                        // so until then it is written as the composite resolved it.
                        encode: match self.exposure.is_some() {
                            true => program::encode(adapted),
                            false => 1.0,
                        },
                        ..self
                            .ambient
                            .exposure(ui.input(|input| input.stable_dt))
                            .unwrap_or_default()
                    }
                },
                fog: self.ambient.fog().unwrap_or_default(),
                cloud: self
                    .ambient
                    .clouds()
                    .map_or_else(program::Cloud::default, |held| held.scene),
                shaft: self.ambient.shafts().unwrap_or_default(),
                star: self.ambient.starfield().unwrap_or_default(),
                bloom: self.ambient.bloom().unwrap_or_default(),
                look: self.look,
                clock: self.clock / TICKS,
                wind: self.ambient.wind().unwrap_or(program::Wind {
                    reach: 0.0,
                    layers: [program::WindLayer::default(); 2],
                    ..Default::default()
                }),
                sky: program::Sky {
                    time: self.ambient.time,
                    tilt: self.ambient.tilt,
                    size: self
                        .sky_volume
                        .map_or_else(|| program::Sky::default().size, |(_, size, _)| size),
                    depth: self
                        .sky_volume
                        .map_or_else(|| program::Sky::default().depth, |(_, _, depth)| depth),
                    moon: self.ambient.moon,
                    moonlight: self.ambient.moonlight(),
                    moon_fade: self.ambient.moon_fade(),
                    day: self.ambient.day,
                },
                ..Default::default()
            },
            lighting: self.lighting.clone(),
            exposure: self.exposure.clone(),
            skybox: self.skybox.clone(),
            sunlight: self.sunlight.clone(),
            moonlight: self.moonlight.clone(),
            starlight: self.starlight.clone(),
            haze: self.haze.clone(),
            clouds: self.clouds.clone(),
            cloud_shadow: self.cloud_shadow.clone(),
            glare: self.glare.clone(),
            smoothing: self.smoothing.clone(),
            occlusion: self.look.occlude.then(|| self.occlusion.clone()).flatten(),
            vignette: self.look.vignette.then(|| self.vignette.clone()).flatten(),
            reflection: self.look.reflect.then(|| self.reflection.clone()).flatten(),
            water_mirror: self.look.reflect.then(|| self.water_mirror.clone()).flatten(),
            lamps: self.lamps(),
            batches,
            grass: self.sward.clone(),
            blades,
            effects,
            effect_packages: self.effect_packages.clone(),
        };

        // A click picks whatever the pointer runs through, and a click on nothing lets go of what
        // was held. Dragging turns the camera, so only a click that did not drag counts.
        if response.clicked()
            && let Some(pointer) = response.interact_pointer_pos()
        {
            let ndc = egui::vec2(
                2.0 * (pointer.x - rect.left()) / rect.width() - 1.0,
                1.0 - 2.0 * (pointer.y - rect.top()) / rect.height(),
            );
            let inverse = (projection * view).inverse();
            let near = inverse.project_point3(Vec3::new(ndc.x, ndc.y, 0.0));
            let far = inverse.project_point3(Vec3::new(ndc.x, ndc.y, 1.0));
            self.selected = self.under(near, (far - near).normalize_or_zero());
        }

        // The context is taken from the painter rather than captured: `glow::Context` is neither
        // `Send` nor `Sync` on wasm, and a callback has to be both.
        let renderer = self.renderer.clone();
        ui.painter().add(egui::PaintCallback {
            rect,
            callback: Arc::new(egui_glow::CallbackFn::new(move |info, painter| {
                renderer
                    .lock()
                    .unwrap()
                    .draw(painter.gl(), painter, &frame, &info);
            })),
        });
        self.outline(ui, rect, projection * view);
        // Taken the same way the drive is: a host that stops asking for markers gets none next
        // frame, rather than a stale label left over another view.
        for (point, label) in std::mem::take(&mut self.markers) {
            self.mark_point(ui, rect, projection * view, point, &label);
        }
        self.state(
            rect,
            ui.ctx().pixels_per_point(),
            ui.input(|input| input.stable_dt),
            effects_drawn,
            vertical_fov,
        );
    }

    /// Publishes what this frame was drawn from, for a harness measuring it against a capture.
    /// `vertical_fov` is what the projection matrix actually used, which for a driven shot is not
    /// [`Self::fov`]: that field states the shot's own 16:9 value, refit here to the viewport's
    /// real aspect.
    fn state(&self, rect: egui::Rect, scale: f32, step: f32, effects_drawn: usize, vertical_fov: f32) {
        let (exposure, measured) = match self.exposure.is_some() {
            true => {
                let held = self.renderer.lock().unwrap();
                (held.exposed(), held.measured())
            }
            false => (f32::NAN, f32::NAN),
        };
        report::publish(&report::Frame {
            commit: crate::build::COMMIT_HASH,
            clean: crate::build::GIT_CLEAN,
            built: crate::build::BUILD_TIME,
            level: &self.path,
            preset: self.preset.as_ref().map(|held| held.name.as_str()),
            eye: self.camera.position.to_array(),
            forward: self.camera.forward().to_array(),
            fov: vertical_fov,
            viewport: [
                rect.left() * scale,
                rect.top() * scale,
                rect.width() * scale,
                rect.height() * scale,
            ],
            time: self.ambient.time,
            weather: self.ambient.weather_id().unwrap_or_default(),
            exposure,
            measured,
            step,
            placed: self.placements.len(),
            drawn: self.placed.iter().flatten().map(Vec::len).sum(),
            effects: self.effects.len() + self.fired.len(),
            effects_drawn,
            casting: self.casts.iter().flatten().sum(),
            models: format!(
                "{} of {}",
                self.models
                    .iter()
                    .filter(|model| matches!(model.state, State::Ready))
                    .count(),
                self.models.len()
            ),
            materials: format!(
                "{} of {}",
                self.translated.keys().filter(|(_, waving)| !waving).count(),
                self.materials.len()
            ),
            passes: self.passes(),
        });
    }

    /// Everything the zone states about the placement the pointer picked.
    fn chosen_ui(&self, ui: &mut egui::Ui, follow: &mut Option<String>) {
        let Some(placement) = self.selected.and_then(|at| self.placements.get(at)) else {
            return;
        };
        ui.add_space(8.0);
        ui.separator();
        section(ui, "Selected");
        ui.add_space(4.0);

        let path = &self.models[placement.model].path;
        if link(ui, crate::utils::file_name(path), path) {
            *follow = Some(path.clone());
        }
        ui.add_space(4.0);

        let held = self.posed(placement);
        let basis = Mat3::from_cols(
            held.x_axis.truncate(),
            held.y_axis.truncate(),
            held.z_axis.truncate(),
        );
        let scale = Vec3::new(
            basis.x_axis.length(),
            basis.y_axis.length(),
            basis.z_axis.length(),
        );
        // A placement is free to state a scale of nought, and taking a rotation out of a basis that
        // flat gives nothing back.
        let angles = (scale.min_element() > 1e-6).then(|| {
            let upright = Mat3::from_cols(
                basis.x_axis / scale.x,
                basis.y_axis / scale.y,
                basis.z_axis / scale.z,
            );
            let (z, y, x) = Quat::from_mat3(&upright).to_euler(glam::EulerRot::ZYX);
            Vec3::new(x.to_degrees(), y.to_degrees(), z.to_degrees())
        });
        let place = |held: Vec3| format!("{:.3}, {:.3}, {:.3}", held.x, held.y, held.z);
        let (group, id) = placement.key;
        ui.scope(|ui| {
            ui.set_max_width(ui.available_width().min(DETAILS_ROW_WIDTH));
            facts(ui, "scene_selected", &[
                ("Layer", self.layers[placement.layer].name.clone()),
                ("Position", place(held.w_axis.truncate())),
                ("Rotation", match angles {
                    Some(held) => place(held),
                    None => "flat".to_owned(),
                }),
                ("Scale", place(scale)),
                ("Size", format!("{:.3}", placement.radius)),
                ("Fade", match placement.fade {
                    held if held > 0.0 => format!("{held:.1}"),
                    _ => "never".to_owned(),
                }),
                ("Motion", match &placement.driven {
                    Some(held) => format!("{} driven", held.chain.len()),
                    None => "still".to_owned(),
                }),
                (
                    "Sky",
                    format!(
                        "{:.3}",
                        reached(&self.visibility, placement.key).copied().unwrap_or(1.0)
                    ),
                ),
                (
                    "Key",
                    format!("{group:08x} {:02x}{:02x}{:02x}{:02x}", id[0], id[1], id[2], id[3]),
                ),
            ]);
        });
    }

    /// A box around what the pointer picked, drawn over the frame rather than into it, and the name
    /// of what it holds.
    fn outline(&self, ui: &egui::Ui, rect: egui::Rect, clip: Mat4) {
        let Some(placement) = self.selected.and_then(|at| self.placements.get(at)) else {
            return;
        };
        let held = self.posed(placement);
        let reach = placement.radius.max(0.01);
        let center = held.transform_point3(Vec3::ZERO);
        let corner = |at: usize| {
            let step = |bit: usize| match at >> bit & 1 {
                0 => -reach,
                _ => reach,
            };
            let point = center + Vec3::new(step(0), step(1), step(2));
            let clipped = clip * point.extend(1.0);
            (clipped.w > 0.0).then(|| {
                egui::pos2(
                    rect.left() + (clipped.x / clipped.w * 0.5 + 0.5) * rect.width(),
                    rect.top() + (0.5 - clipped.y / clipped.w * 0.5) * rect.height(),
                )
            })
        };
        let points: Vec<Option<egui::Pos2>> = (0..8).map(corner).collect();
        let painter = ui.painter_at(rect);
        let stroke = egui::Stroke::new(1.5, Color32::from_rgb(255, 190, 60));
        let mut seen = egui::Rect::NOTHING;
        for from in 0..8 {
            for bit in 0..3 {
                let to = from ^ 1usize << bit;
                if to < from {
                    continue;
                }
                let (Some(a), Some(b)) = (points[from], points[to]) else {
                    continue;
                };
                painter.line_segment([a, b], stroke);
                seen = seen.union(egui::Rect::from_two_pos(a, b));
            }
        }
        if seen.is_negative() {
            return;
        }
        painter.text(
            egui::pos2(seen.center().x, seen.top() - 4.0),
            egui::Align2::CENTER_BOTTOM,
            self.models[placement.model]
                .path
                .rsplit('/')
                .next()
                .unwrap_or_default(),
            egui::FontId::monospace(11.0),
            stroke.color,
        );
    }

    /// A labelled point, projected the same way [`Self::outline`] projects a box's corners. Behind
    /// the eye, it draws nothing rather than a label pinned to the wrong edge of the screen.
    fn mark_point(&self, ui: &egui::Ui, rect: egui::Rect, clip: Mat4, point: Vec3, label: &str) {
        let clipped = clip * point.extend(1.0);
        if clipped.w <= 0.0 {
            return;
        }
        let at = egui::pos2(
            rect.left() + (clipped.x / clipped.w * 0.5 + 0.5) * rect.width(),
            rect.top() + (0.5 - clipped.y / clipped.w * 0.5) * rect.height(),
        );
        if !rect.contains(at) {
            return;
        }
        let painter = ui.painter_at(rect);
        let color = Color32::from_rgb(120, 200, 255);
        painter.circle_stroke(at, 4.0, egui::Stroke::new(1.5, color));
        painter.text(
            at + egui::vec2(6.0, -6.0),
            egui::Align2::LEFT_BOTTOM,
            label,
            egui::FontId::monospace(11.0),
            color,
        );
    }

    /// The grids nearest the eye, up to the blades one frame draws, and the color map each is cut
    /// out of. Whatever stands within the distance the rest of the zone is drawn over is a
    /// candidate; the cap takes them in the order the eye reaches them.
    fn sown(&self) -> Vec<gpu::Blades> {
        let Grass::Placing(placing) = &self.grass else {
            return Vec::new();
        };
        let eye = self.camera.position;
        let mut near: Vec<(f32, usize, &Turf)> = self
            .turf
            .iter()
            .enumerate()
            .map(|(at, turf)| (eye.distance(turf.origin) - turf.radius, at, turf))
            .filter(|(span, _, _)| *span < self.load)
            .collect();
        near.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut standing = 0;
        let mut drawn = Vec::new();
        for (_, at, turf) in near {
            // Nothing until the map itself is in hand. A blade is cut out of its alpha, and the
            // flat stand-in an unfilled sampler answers with is opaque: the whole quad would stand
            // there as a grey sheet. A grid waiting on one costs nothing against the cap.
            let held = placing
                .maps
                .get(turf.layer)
                .and_then(|path| self.textures.get(path));
            let Some(Texture::Ready(color_map)) = held else {
                continue;
            };
            // Stopped rather than skipped past: what the frame draws is then a run of the nearest
            // grids, which a step of the camera moves the end of rather than shuffling.
            if standing + turf.blades > BLADES {
                break;
            }
            standing += turf.blades;
            drawn.push(gpu::Blades {
                turf: at,
                origin: turf.origin,
                color_map: color_map.id(),
            });
        }
        drawn
    }

    fn surface(&self, slot: usize, waving: bool) -> gpu::Surface {
        let held = self.materials.get(slot).and_then(|(_, held)| match held {
            Slot::Ready(material) => Some(material),
            _ => None,
        });
        // The graph's own store first: a sliced texture reaches egui as a plane on the frame before
        // its package is translated, and answering with that one would pin the sampler to it.
        let bind = |path: &str, aniso: f32| match self.stacked.get_key_value(path) {
            Some((held, Stack::Ready)) => Some(mdl::gpu::Bound::Stacked(held.clone())),
            _ => match self.textures.get(path) {
                Some(Texture::Ready(handle)) => Some(mdl::gpu::Bound::Plane(handle.id(), aniso)),
                _ => None,
            },
        };
        // Bare geometry until the material and its package arrive, rather than a hole where they
        // will be.
        let shaded = held
            .zip(self.translated.get(&(slot, waving)))
            .map(|(material, held)| mdl::gpu::Shaded {
                buffer: held.buffer.clone(),
                depth: held.depth.clone(),
                shadow: held.shadow.clone(),
                resolve: held.resolve.clone(),
                sheer: held.sheer.clone(),
                table: self.tables.get(&slot).cloned(),
                textures: material
                    .bound()
                    .map(|(id, path)| (id, bind(path, material.anisotropic(id))))
                    .collect(),
            });
        gpu::Surface {
            material: slot,
            waving,
            shaded,
            cull: held.is_some_and(|material| material.cull()),
            hidden: held.is_some_and(|material| !material.drawn()),
        }
    }

    /// Stands the view where a preset says a capture was taken from: the camera and what it looks
    /// at, the lens, and the weather and hour the frame was under. The level it names is left to the
    /// link beside it, since opening one builds a scene of its own and would undo all of this.
    fn stand_where(&mut self, held: &preset::Preset) {
        let (yaw, pitch) = held.angles();
        self.camera.position = held.camera;
        self.camera.yaw = yaw;
        self.camera.pitch = pitch;
        if let Some(fov) = held.fov {
            self.fov = fov;
        }
        if let Some(time) = held.time {
            self.ambient.time = time;
        }
        if let Some(id) = held.weather
            && !self.ambient.stand_in_weather(id)
        {
            log::warn!("assets/layer: this zone states no weather {id}");
        }
        // Counts as fitted, so `poll`'s first-placements auto-frame does not undo this once the
        // zone's own content streams in.
        self.fitted = self.fitted.max(1);
        self.dirty = true;
    }

    pub fn details_ui(
        &mut self,
        ui: &mut egui::Ui,
        follow: &mut Option<String>,
        deps: &mut Deps,
        backend: &Backend,
    ) {
        self.saving.take_if(|promise| promise.try_get().is_some());
        let mut refit = false;
        let mut changed = false;
        // A preset dropped on the window, or picked with the button below, stands this view where a
        // capture was taken from, which is what makes the two comparable at all.
        let mut arrived: Vec<Vec<u8>> = ui.ctx().input(|input| {
            input
                .raw
                .dropped_files
                .iter()
                .filter_map(|held| held.bytes.as_ref().map(|held| held.to_vec()))
                .collect()
        });
        if let Some(promise) = &self.picking
            && let Some(held) = promise.try_get()
        {
            arrived.extend(held.clone());
            self.picking = None;
        }
        for bytes in &arrived {
            match preset::Preset::read(bytes) {
                Ok(held) => {
                    // A preset for somewhere else opens that level instead, and is applied on the
                    // other side: opening one builds a scene of its own.
                    match held.level == self.path {
                        true => self.stand_where(&held),
                        false => {
                            *follow = Some(held.level.clone());
                            preset::hold(held);
                            return;
                        }
                    }
                    self.preset = Some(held);
                    changed = true;
                }
                Err(why) => log::warn!("assets/layer: this is no TitleEdit preset: {why}"),
            }
        }
        ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
            section(ui, "View");
            // Pasted rather than picked, since a file dialog is the one way in that nothing outside
            // the window can drive: a headless run positions the camera through here.
            let pasted = ui.add(
                egui::TextEdit::singleline(&mut self.pasted)
                    .hint_text("paste a TitleEdit preset")
                    .desired_width(f32::INFINITY),
            );
            let mut load =
                pasted.lost_focus() && ui.input(|held| held.key_pressed(egui::Key::Enter));
            // Wrapped rather than run on: four buttons in one row is wider than the panel's own
            // minimum, and a row that can't shrink pins the whole panel at its own width.
            ui.horizontal_wrapped(|ui| {
                if ui.button("Import preset").clicked() {
                    self.picking = Some(TrackedPromise::spawn_local(async {
                        let held = rfd::AsyncFileDialog::new()
                            .set_title("Import a TitleEdit preset")
                            .add_filter("TitleEdit preset", &["json"])
                            .pick_file()
                            .await?;
                        Some(held.read().await)
                    }));
                }
                load |= ui.button("Load pasted").clicked();
                if load {
                    match preset::Preset::read(self.pasted.as_bytes()) {
                        Ok(held) => {
                            match held.level == self.path {
                                true => self.stand_where(&held),
                                false => {
                                    *follow = Some(held.level.clone());
                                    preset::hold(held);
                                    return;
                                }
                            }
                            self.preset = Some(held);
                            changed = true;
                        }
                        Err(why) => log::warn!("assets/layer: this is no TitleEdit preset: {why}"),
                    }
                }
                let held = preset::Preset::of(
                    &self.path,
                    self.camera.position,
                    self.camera.forward(),
                    self.fov,
                    self.ambient.weather_id(),
                    self.ambient.time,
                );
                let file_name = format!("TE_{}.json", held.name);
                let choices = match held.write() {
                    Ok(text) => vec![
                        export::Choice::bytes("Export preset", file_name, move || {
                            Ok(text.into_bytes())
                        })
                        .title("Export a TitleEdit preset")
                        .filter("JSON", &["json"]),
                    ],
                    Err(why) => {
                        log::error!("assets/layer: {why}");
                        Vec::new()
                    }
                };
                let promise =
                    export::menu(ui, "Export preset", None, self.saving.is_some(), choices, egui::Vec2::ZERO);
                if promise.is_some() {
                    self.saving = promise;
                }
                // The same shape the plugin hands over its own clipboard, so a paste elsewhere
                // reads it back.
                if ui.button("Copy preset").clicked() {
                    match held.share() {
                        Ok(text) => ui.ctx().copy_text(text),
                        Err(why) => log::error!("assets/layer: {why}"),
                    }
                }
            });
            if let Some(held) = &self.preset {
                ui.label(RichText::new(format!("Preset  {}", held.name)).weak());
                ui.add_space(4.0);
            }
            ui.horizontal(|ui| {
                if ui.button("Fit").clicked() {
                    refit = true;
                }
                ui.label(
                    RichText::new(format!(
                        "{:.0}, {:.0}, {:.0}",
                        self.camera.position.x, self.camera.position.y, self.camera.position.z
                    ))
                    .monospace()
                    .weak(),
                );
            });
            ui.add_space(4.0);
            ui.label(RichText::new("Load distance").weak());
            changed |= ui
                .add(egui::Slider::new(&mut self.load, NEAREST..=FURTHEST).logarithmic(true))
                .changed();
            ui.label(RichText::new("Speed").weak());
            ui.add(egui::Slider::new(&mut self.speed, 0.1..=20.0).logarithmic(true));
            ui.label(RichText::new("Field of view").weak());
            match self.driving {
                // Derived from the shot's own focal length while a cutscene drives the camera,
                // so it is a label rather than a slider here.
                true => {
                    ui.label(RichText::new(format!("{:.1}\u{b0}", self.fov)).monospace());
                }
                false => {
                    changed |= ui
                        .add(egui::Slider::new(&mut self.fov, 20.0..=120.0).suffix("\u{b0}"))
                        .changed();
                }
            }
            let quality = self.look.quality;
            ui.checkbox(&mut self.look.occlude, "Occlusion").on_hover_text(
                "Shade the creases with the game's own HDAO, which every light past the sun and \
                 the composite weight what they work out by",
            );
            ui.add_enabled_ui(self.look.occlude, |ui| {
                egui::ComboBox::from_id_salt("layer-occluder")
                    .selected_text(program::OCCLUDERS[self.look.quality])
                    .show_ui(ui, |ui| {
                        for (at, what) in program::OCCLUDERS.iter().enumerate() {
                            ui.selectable_value(&mut self.look.quality, at, *what);
                        }
                    });
            });
            if self.look.quality != quality {
                self.occlusion = None;
            }
            ui.checkbox(&mut self.look.vignette, "Vignette").on_hover_text(
                "Darken the frame's corners with the game's own pass. The ellipse it spreads over \
                 follows the frame's own shape, but the two below are choices: no file states \
                 either",
            );
            ui.add_enabled_ui(self.look.vignette, |ui| {
                ui.label(RichText::new("Onset").weak());
                ui.add(egui::Slider::new(&mut self.look.onset, 0.0..=1.0))
                    .on_hover_text(
                        "How far out the darkening starts, as a squared distance with a corner at \
                         one",
                    );
                ui.label(RichText::new("Darkening").weak());
                ui.add(egui::Slider::new(&mut self.look.darkening, 0.0..=2.0))
                    .on_hover_text("How steeply it deepens past that");
            });
            ui.add_space(8.0);
            ui.separator();
            let mut sound_on = self.sound.enabled();
            if ui.checkbox(&mut sound_on, "Play in-zone sound").changed() {
                match sound_on {
                    true => self.sound.enable(),
                    false => self.sound.disable(),
                }
            }
            ui.add_enabled_ui(sound_on, |ui| {
                ui.label(RichText::new("Sound volume").weak());
                let mut volume = self.sound.volume();
                if ui.add(egui::Slider::new(&mut volume, 0.0..=1.0)).changed() {
                    self.sound.set_volume(volume);
                }
            });
            ui.label(
                RichText::new(format!(
                    "{} placed, {} playing",
                    self.sound.placed(),
                    self.sound.playing()
                ))
                .weak(),
            );
            if let Some(error) = self.sound.error() {
                ui.colored_label(Color32::RED, error);
            }

            ui.add_space(8.0);
            ui.separator();
            let drawn: usize = self.placed.iter().flatten().map(Vec::len).sum();
            let ready = self
                .models
                .iter()
                .filter(|model| matches!(model.state, State::Ready))
                .count();
            ui.scope(|ui| {
                ui.set_max_width(ui.available_width().min(DETAILS_ROW_WIDTH));
                facts(
                    ui,
                    "scene_counts",
                    &[
                        ("Placed", self.placements.len().to_string()),
                        ("Drawn", drawn.to_string()),
                        ("Waiting on a model", self.absent.to_string()),
                        ("Models", format!("{ready} of {}", self.models.len())),
                        ("Groups to read", self.waiting.len().to_string()),
                        (
                            "Materials",
                            format!(
                                "{} of {}",
                                self.translated.keys().filter(|(_, waving)| !waving).count(),
                                self.materials.len()
                            ),
                        ),
                        (
                            "Lights",
                            format!("{} of {}", self.lamps().len(), self.lights.len()),
                        ),
                        (
                            "Effects",
                            format!(
                                "{} of {}",
                                self.effect_files
                                    .iter()
                                    .filter(|effect| matches!(effect.state, EffectState::Ready(..)))
                                    .count(),
                                self.effect_files.len()
                            ),
                        ),
                        ("Wind", {
                            let count = self.models.iter().filter(|model| model.waving).count();
                            let plural = match count {
                                1 => "",
                                _ => "s",
                            };
                            match self.ambient.wind() {
                                Some(held) => format!(
                                    "clock {:.1}s, reach {:.2} at {:.0} deg, {count} model{plural}",
                                    self.clock / TICKS,
                                    held.reach,
                                    held.heading.x.atan2(held.heading.z).to_degrees(),
                                ),
                                None => format!("no wind set stated, {count} model{plural}"),
                            }
                        }),
                        (
                            "Exposure",
                            match self.exposure.is_some() {
                                true => {
                                    let held = self.renderer.lock().unwrap();
                                    format!(
                                        "{:.3} from a frame measuring {:.3}, written at {:.3}",
                                        held.exposed(),
                                        held.measured(),
                                        program::encode(held.exposed())
                                    )
                                }
                                false => "not run".to_owned(),
                            },
                        ),
                        // Which of the passes past the lighting ran. A weather that names no clouds
                        // draws none, and so does a draw that quietly went wrong; only the graph knows
                        // which of the two a frame without any is.
                        // How much of the sky reaches each part, which the zone's own `.svb` states by
                        // the same key an `.lcb` reaches a light by. A part it does not name stands in
                        // full sky, so a file that matches nothing looks exactly like no file at all.
                        // A zone with no grass of its own and a grass file that would not read look the
                        // same from the outside, and so does a grid nothing has asked for yet.
                        ("Grass", match &self.grass {
                            Grass::Wanted(_) => "waiting on the zone's own file".to_owned(),
                            Grass::Fetching(_, _) => "reading the zone's own file".to_owned(),
                            Grass::Done => "none".to_owned(),
                            Grass::Placing(held) => {
                                let read = held.grids.iter().filter(|grid| grid.taken).count();
                                format!(
                                    "{read} of {} grids, {} models, {} placed, {} of {} blades drawn",
                                    held.grids.len(),
                                    held.models.len(),
                                    self.layers.get(held.layer).map_or(0, |held| held.placements),
                                    self.standing,
                                    self.blades,
                                )
                            }
                        }),
                        (
                            "Sky visibility",
                            format!(
                                "{} of {} placed",
                                self.placements
                                    .iter()
                                    .filter(|held| reached(&self.visibility, held.key).is_some())
                                    .count(),
                                self.placements.len()
                            ),
                        ),
                        ("Blended materials", {
                            let mut tally: BTreeMap<String, (usize, usize, usize)> = BTreeMap::new();
                            for (at, (_, slot)) in self.materials.iter().enumerate() {
                                let Slot::Ready(material) = slot else { continue };
                                let name = material.package();
                                if !wet_name(&name) {
                                    continue;
                                }
                                let held = tally.entry(name).or_default();
                                held.0 += 1;
                                match self.translated.get(&(at, false)) {
                                    Some(one) if one.resolve.is_some() => held.1 += 1,
                                    Some(_) => held.2 += 1,
                                    None => {}
                                }
                            }
                            match tally.is_empty() {
                                true => "none named".to_owned(),
                                false => tally
                                    .iter()
                                    .map(|(name, (all, wet, dry))| {
                                        format!("{name} {all}: {wet} blended, {dry} opaque")
                                    })
                                    .collect::<Vec<_>>()
                                    .join("\n"),
                            }
                        }),
                        (
                            "Blended surfaces",
                            format!(
                                "{} of {} translated",
                                self.translated
                                    .iter()
                                    .filter(|((_, waving), held)| !waving && held.resolve.is_some())
                                    .count(),
                                self.translated.keys().filter(|(_, waving)| !waving).count()
                            ),
                        ),
                        (
                            "Shadow pass",
                            match (
                                self.packages.get(program::SHADOW),
                                self.lighting.as_ref().map(|held| held.shadow.is_some()),
                            ) {
                                (Some(Package::Ready(_)), Some(true)) => {
                                    let reaches: Vec<String> = (0..program::SPLITS)
                                        .map(|at| {
                                            format!("{:.0}", program::shadow_reach(self.ambient.reach, at))
                                        })
                                        .collect();
                                    format!(
                                        "translated, {} splits reaching {} (the game draws 5)",
                                        program::SPLITS,
                                        reaches.join(", ")
                                    )
                                }
                                (Some(Package::Ready(_)), _) => "arrived, not translated".to_owned(),
                                (Some(Package::Failed), _) => "failed".to_owned(),
                                (Some(Package::Fetching(_)), _) => "fetching".to_owned(),
                                (Some(Package::Wanted), _) => "wanted".to_owned(),
                                (None, _) => "never asked for".to_owned(),
                            },
                        ),
                        ("Passes", self.passes()),
                        (
                            "Textures",
                            format!(
                                "{}, {}, {} with slices",
                                self.textures.len(),
                                crate::assets::Bytes(self.resident),
                                self.stacked
                                    .values()
                                    .filter(|held| matches!(held, Stack::Ready))
                                    .count(),
                            ),
                        ),
                    ],
                );
            });

            self.chosen_ui(ui, follow);

            ui.add_space(8.0);
            ui.separator();
            changed |= self.ambient.ui(ui, follow, deps, backend);

            ui.add_space(8.0);
            ui.separator();
            section(ui, "Layers");
            ui.horizontal(|ui| {
                if ui.button("All").clicked() {
                    for layer in &mut self.layers {
                        layer.shown = true;
                    }
                    changed = true;
                }
                if ui.button("None").clicked() {
                    for layer in &mut self.layers {
                        layer.shown = false;
                    }
                    changed = true;
                }
            });
            ui.add_space(4.0);
            // Truncated rather than run on: a zone's layer names are unbounded, and one long name
            // in an unwrapped checkbox pins the whole panel at its own width forever.
            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Truncate);
            for layer in &mut self.layers {
                let mut label = format!("{} ({})", layer.name, layer.placements);
                if layer.festival != 0 {
                    label.push_str(&format!("  festival {}", layer.festival));
                }
                let mut hover = label.clone();
                hover.push('\n');
                hover.push_str(match layer.visible {
                    true => "drawn by default",
                    false => "hidden by default",
                });
                if let Some(origin) = &layer.origin {
                    hover.push('\n');
                    hover.push_str(origin);
                }
                changed |= ui
                    .checkbox(&mut layer.shown, RichText::new(label).monospace())
                    .on_hover_text(hover)
                    .changed();
            }
        });
        if refit {
            self.fit();
        }
        if changed {
            self.dirty = true;
        }
    }
}

pub fn ui(ui: &mut egui::Ui, scene: &mut Scene, backend: &Backend) {
    if let Some(why) = scene.renderer.lock().unwrap().failure() {
        ui.centered_and_justified(|ui| {
            ui.colored_label(Color32::RED, format!("Could not build the shader: {why}"));
        });
        return;
    }
    scene.poll(ui, backend);
    scene.viewport(ui);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Euler order the files are read under. A pure yaw reduces to `Mat3::from_rotation_y`,
    /// which is what ring tests over the corpus settled; this pins the rest of it.
    #[test]
    fn a_rotation_turns_about_x_first() {
        let quarter = std::f32::consts::FRAC_PI_2;
        assert!((rotation([0.0, quarter, 0.0]) * Vec3::Z - Vec3::X).length() < 1e-5);
        assert!((rotation([quarter, 0.0, quarter]) * Vec3::Z - Vec3::X).length() < 1e-5);
        // The timeline deltas take the same order through glam's own sequence instead.
        assert!(
            Quat::from_euler(glam::EulerRot::ZYX, 1.1, 0.7, 0.3)
                .abs_diff_eq(Quat::from_mat3(&rotation([0.3, 0.7, 1.1])), 1e-5)
        );
    }

    #[test]
    fn refitting_to_the_authored_aspect_changes_nothing() {
        assert!((refit_16_9_fov(30.0, 16.0 / 9.0) - 30.0).abs() < 1e-4);
    }

    /// A narrower viewport than 16:9 must widen the vertical field to keep the same horizontal
    /// one, not crop it.
    #[test]
    fn refitting_to_a_narrower_aspect_widens_the_vertical_field() {
        assert!(refit_16_9_fov(30.0, 1.0) > 30.0);
    }

    /// A whole turn starts over where it ends, so it never runs backwards; anything else swings
    /// back to where it began.
    #[test]
    fn a_repeating_lane_wraps_or_swings_back() {
        assert_eq!(phase(360, 0, 0, 0.0), 0.0);
        assert_eq!(phase(360, 0, 0, 270.0), 0.75);
        assert_eq!(phase(360, 0, 0, 450.0), 0.25);
        assert_eq!(phase(180, 0, 1, 180.0), 1.0);
        assert_eq!(phase(180, 0, 1, 270.0), 0.5);
        assert_eq!(phase(180, 0, 1, 360.0), 0.0);
        assert_eq!(phase(60, 30, 1, 20.0), 0.0);
    }

    #[test]
    fn a_missed_key_steps_back_to_its_own_group() {
        let mut map = HashMap::new();
        map.insert((9, [0, 0, 0, 0]), 0.5_f32);
        map.insert((9, [1, 2, 0, 0]), 0.1_f32);
        assert_eq!(reached(&map, (9, [1, 2, 3, 0])), Some(&0.1));
        assert_eq!(reached(&map, (9, [1, 0, 0, 0])), Some(&0.5));
        assert_eq!(reached(&map, (9, [4, 0, 0, 0])), Some(&0.5));
        assert_eq!(reached(&map, (7, [1, 0, 0, 0])), None);
    }

    #[test]
    fn a_fit_leaves_out_what_sits_nowhere_near_the_rest() {
        let mut points: Vec<Vec3> = (0..100).map(|at| Vec3::new(at as f32, 0.0, 0.0)).collect();
        points.push(Vec3::splat(1_400_000.0));
        let (center, reach) = bulk(&points);
        assert!((center - Vec3::new(50.0, 0.0, 0.0)).length() < 5.0);
        assert!(reach < 100.0);
    }

    #[test]
    fn a_model_falls_back_to_the_level_it_has() {
        assert_eq!(level([true, false, false], 0.0001), Some(0));
        assert_eq!(level([true, true, true], 0.5), Some(0));
        assert_eq!(level([true, true, true], 0.0001), Some(2));
        assert_eq!(level([false, false, false], 0.5), None);
    }

    #[test]
    fn a_ray_meets_the_near_side_of_a_sphere() {
        let along = Vec3::NEG_Z;
        let ahead = Vec3::new(0.0, 0.0, -10.0);
        assert_eq!(pierced(Vec3::ZERO, along, ahead, 1.0), Some(9.0));
        assert_eq!(pierced(Vec3::ZERO, along, ahead, 2.0), Some(8.0));
        assert_eq!(pierced(Vec3::ZERO, along, Vec3::new(0.0, 0.0, 10.0), 1.0), None);
        assert_eq!(pierced(Vec3::ZERO, along, Vec3::new(5.0, 0.0, -10.0), 1.0), None);
        assert_eq!(pierced(Vec3::ZERO, along, Vec3::new(0.0, 0.0, -1.0), 5.0), Some(0.0));
    }

    #[test]
    fn the_tighter_bound_wins_where_the_eye_stands_inside_both() {
        let wide = (0, Vec3::new(0.0, 0.0, -40.0), 100.0);
        let close = (1, Vec3::new(0.0, 0.0, -2.0), 5.0);
        assert_eq!(nearest(Vec3::ZERO, Vec3::NEG_Z, [wide, close].into_iter()), Some(1));
        assert_eq!(nearest(Vec3::ZERO, Vec3::NEG_Z, [close, wide].into_iter()), Some(1));
    }

    #[test]
    fn the_nearer_of_two_spheres_ahead_wins_either_way_round() {
        let far = (0, Vec3::new(0.0, 0.0, -30.0), 2.0);
        let near = (1, Vec3::new(0.0, 0.0, -10.0), 2.0);
        assert_eq!(nearest(Vec3::ZERO, Vec3::NEG_Z, [far, near].into_iter()), Some(1));
        assert_eq!(nearest(Vec3::ZERO, Vec3::NEG_Z, [near, far].into_iter()), Some(1));
        assert_eq!(nearest(Vec3::ZERO, Vec3::NEG_Z, [].into_iter()), None);
    }
}
