//! What an effect does, read out of the tags it is written under and stepped a frame at a time.
//!
//! A scheduler starts a timeline, a timeline runs an emitter over a span of frames, and an emitter
//! bursts particles at an interval. A particle carries its own velocity forward and reads its
//! position, rotation, scale and color off curves indexed by how long it has been alive.
//!
//! What the tags mean comes from VFXEditor, which is the only place they are named. Only the ones
//! the corpus actually writes are read here; the rest of a particle's `Data` goes unread, so a kind
//! that draws a ribbon along its own path or warps what is behind it falls back to the sprite the
//! rest use. Nothing random is read: the `R`-suffixed curves beside the ones below go unread, so an
//! effect plays the same way every time and scrubbing back to a frame lands where it did before.

use std::hash::{Hash, Hasher};

use glam::{Quat, Vec3, Vec4};
use ironworks::file::avfx::{Avfx, Block, DirectionalLightSource, Item, Model as Geometry};

use super::curve::{self, Curve};
use super::find;
use super::program::{self, UV_REGISTERS, UV_SETS};

/// Live particles and running emitters one effect may hold. Both counts come off the file unchecked
/// and this ships to a browser, where an effect asking for millions takes the tab with it.
const PARTICLES: usize = 8192;
const EMITTERS: usize = 512;

/// How deep an emitter may spawn another.
const DEPTH: u8 = 4;

/// Frames a loop runs for where nothing in the file bounds it, and the longest one it may reach.
const LOOP: i32 = 300;
pub const LONGEST: i32 = 3600;

/// Frames a fit is taken over.
const FITTED: i32 = 300;

/// The Euler order `apricot_powder.shpk` builds a particle's basis under: about Z first, then X,
/// then Y.
fn rotation(angles: Vec3) -> Quat {
    Quat::from_rotation_y(angles.y)
        * Quat::from_rotation_x(angles.x)
        * Quat::from_rotation_z(angles.z)
}

fn integer(blocks: &[Block], name: &str) -> Option<i32> {
    find(blocks, name)?.i32()
}

fn nested<'a>(blocks: &'a [Block], name: &str) -> &'a [Block] {
    find(blocks, name).map_or(&[][..], Block::blocks)
}

/// A tag naming one of the effect's lists. These are written as a list of one-byte indices where
/// the tag allows several, and as a plain integer where it does not.
fn index(blocks: &[Block], name: &str) -> Option<usize> {
    let block = find(blocks, name)?;
    let value = match block.bytes() {
        [only] => i32::from(*only),
        bytes if bytes.len() == 4 => block.i32()?,
        [first, ..] => i32::from(*first),
        [] => return None,
    };
    usize::try_from(value).ok()
}

/// Whether something the file can switch off is switched off.
fn off(blocks: &[Block], name: &str) -> bool {
    find(blocks, name).and_then(Block::bool) == Some(false)
}

/// How long something lives, in frames. A life it never reaches is written as `-1`.
fn life(blocks: &[Block]) -> Option<f32> {
    let value = find(blocks, "Life")?.find("Val")?.f32()?;
    (value >= 0.0).then_some(value)
}

/// One animated value, or the constant the file leaves where it writes no curve.
struct Track {
    curve: Option<Curve>,
    idle: f32,
}

impl Track {
    fn read(blocks: &[Block], name: &str, idle: f32) -> Self {
        Self {
            curve: find(blocks, name).and_then(curve::read),
            idle,
        }
    }

    fn at(&self, frame: f32) -> f32 {
        self.curve
            .as_ref()
            .map_or(self.idle, |curve| curve.sample(frame)[2])
    }
}

fn triple(blocks: &[Block], names: [&str; 3], idle: f32) -> [Track; 3] {
    names.map(|name| Track::read(blocks, name, idle))
}

fn read(tracks: &[Track; 3], frame: f32) -> Vec3 {
    Vec3::from(tracks.each_ref().map(|track| track.at(frame)))
}

/// Which curve each axis reads, `ACT`. An axis tied to another is written no curve of its own, so
/// leaving it at the idle value is what makes a sprite whose file animates only its width come out
/// a fixed height.
fn tied<const N: usize>(blocks: &[Block]) -> [usize; N] {
    let mut out = std::array::from_fn(|axis| axis);
    let (from, onto): (usize, &[usize]) = match (N, integer(blocks, "ACT").unwrap_or_default()) {
        (2, 1) => (0, &[1]),
        (2, 2) => (1, &[0]),
        (_, 1) => (0, &[1, 2]),
        (_, 2) => (0, &[1]),
        (_, 3) => (0, &[2]),
        (_, 4) => (1, &[0, 2]),
        (_, 5) => (1, &[0]),
        (_, 6) => (1, &[2]),
        (_, 7) => (2, &[0, 1]),
        (_, 8) => (2, &[0]),
        (_, 9) => (2, &[1]),
        _ => return out,
    };
    for &axis in onto.iter().filter(|&&axis| axis < N) {
        out[axis] = from;
    }
    out
}

/// A value the file writes one curve an axis for, under a container of its own.
struct Axes {
    tracks: [Track; 3],
    tied: [usize; 3],
}

impl Axes {
    fn read(blocks: &[Block], name: &str, idle: f32) -> Self {
        let inner = nested(blocks, name);
        Self {
            tracks: triple(inner, ["X", "Y", "Z"], idle),
            tied: tied(inner),
        }
    }

    fn at(&self, frame: f32) -> Vec3 {
        Vec3::from(self.tied.map(|axis| self.tracks[axis].at(frame)))
    }
}

/// The same over two axes, which is how a uv set writes its scale and its scroll.
struct Pair {
    tracks: [Track; 2],
    tied: [usize; 2],
}

impl Pair {
    fn read(blocks: &[Block], name: &str, idle: f32) -> Self {
        let inner = nested(blocks, name);
        Self {
            tracks: ["X", "Y"].map(|axis| Track::read(inner, axis, idle)),
            tied: tied(inner),
        }
    }

    fn at(&self, frame: f32) -> [f32; 2] {
        self.tied.map(|axis| self.tracks[axis].at(frame))
    }
}

/// A value in `[0, 1)` fixed by whatever asks for it. The `Smpl` layer draws all of its variety
/// from ranges the file states, and a slot handed the middle of every one of them is a rigid ladder
/// of one sprite; hashing what the slot is keeps that variety without a generator, so an effect
/// still plays the same way every time and scrubbing back to a frame lands where it did before.
fn noise(key: [u64; 3], lane: u64) -> f32 {
    let mut hasher = std::hash::DefaultHasher::new();
    (key, lane).hash(&mut hasher);
    (hasher.finish() >> 40) as f32 / 16_777_216.0
}

fn float(blocks: &[Block], name: &str, idle: f32) -> f32 {
    find(blocks, name).and_then(Block::f32).unwrap_or(idle)
}

fn triplet(blocks: &[Block], names: [&str; 3]) -> Vec3 {
    Vec3::from(names.map(|name| float(blocks, name, 0.0)))
}

/// A direction the file states as a cone about `+Y`, between two polar angles.
fn cone(low: f32, high: f32, along: f32, around: f32) -> Vec3 {
    let height = low.cos() + (high.cos() - low.cos()) * along;
    let radius = (1.0 - height * height).max(0.0).sqrt();
    let (sin, cos) = (std::f32::consts::TAU * around).sin_cos();
    Vec3::new(sin * radius, height, cos * radius)
}

/// One slot of the `Smpl` layer as it stands at some age.
struct Slot {
    offset: Vec3,
    angles: Vec3,
    size: [f32; 2],
    color: Vec4,
    /// Which cell of the atlas it takes, and whether its `u` runs backwards.
    cell: [i32; 2],
    mirrored: bool,
}

/// The sub-sprite layer a particle carries under `Smpl`: a small particle system of its own, whose
/// slots each take one cell of the particle's texture atlas and run a life of their own. A particle
/// this is on draws its slots and no quad of its own.
struct Simple {
    slots: usize,
    group: i32,
    per: i32,
    interval: f32,
    interval_random: i32,
    life: f32,
    life_random: i32,
    remade: bool,
    /// `SIPT`, and the box it spreads a slot over where it is nought.
    injection: i32,
    spread: Vec3,
    /// `SIDT`, and the cone `IRD0`..`IRD1` the two kinds that read one take.
    heading: i32,
    cone: [f32; 2],
    speed: [f32; 2],
    /// `CGX`..`CGZ`, an acceleration the game applies over a slot's whole age at once.
    accel: Vec3,
    begin: [f32; 2],
    end: [f32; 2],
    curve: f32,
    random_size: [[f32; 2]; 2],
    linked_size: bool,
    angles: Vec3,
    angles_random: Vec3,
    rate: Vec3,
    rate_random: Vec3,
    cells: [i32; 2],
    cell_interval: i32,
    cell_random: i32,
    cell_loops: i32,
    mirrors: bool,
    colors: [[u8; 4]; 4],
    frames: [i16; 4],
}

impl Simple {
    /// Read where `bSCt` turns the layer on, which is the bit the game gates its whole path on.
    fn read(blocks: &[Block]) -> Option<Self> {
        let inner = find(blocks, "Smpl")?.blocks();
        let slots = integer(inner, "CCnt")?.clamp(0, PARTICLES as i32) as usize;
        if slots == 0 || find(blocks, "bSCt").and_then(Block::bool) != Some(true) {
            return None;
        }
        let held = |name: &str, at: usize| find(inner, name).map_or(&[][..], Block::bytes).get(at..);
        let colors = std::array::from_fn(|key| {
            held("Cols", key * 4)
                .and_then(|bytes| bytes.first_chunk::<4>().copied())
                .unwrap_or([255; 4])
        });
        let frames = std::array::from_fn(|key| {
            held("Frms", key * 2)
                .and_then(|bytes| bytes.first_chunk::<2>().copied())
                .map_or(0, i16::from_le_bytes)
        });
        let on = |name: &str| find(inner, name).and_then(Block::bool) == Some(true);
        Some(Self {
            slots,
            group: integer(inner, "BlkN").unwrap_or(1).max(1),
            per: integer(inner, "CrIC").unwrap_or(1).max(1),
            interval: integer(inner, "CrI").unwrap_or(1) as f32,
            interval_random: integer(inner, "CrIR").unwrap_or_default().max(0),
            life: integer(inner, "CrIL").unwrap_or(-1) as f32,
            life_random: integer(inner, "CrLR").unwrap_or_default().max(0),
            remade: on("bCrN"),
            injection: integer(inner, "SIPT").unwrap_or_default(),
            spread: triplet(inner, ["CrAX", "CrAY", "CrAZ"]),
            heading: integer(inner, "SIDT").unwrap_or_default(),
            cone: [float(inner, "IRD0", 0.0), float(inner, "IRD1", 0.0)],
            speed: [float(inner, "VMin", 0.0), float(inner, "VMax", 0.0)],
            accel: triplet(inner, ["CGX", "CGY", "CGZ"]),
            begin: [float(inner, "SBX", 1.0), float(inner, "SBY", 1.0)],
            end: [float(inner, "SEX", 1.0), float(inner, "SEY", 1.0)],
            curve: float(inner, "SC", 1.0),
            random_size: [
                [float(inner, "SRX0", 1.0), float(inner, "SRX1", 1.0)],
                [float(inner, "SRY0", 1.0), float(inner, "SRY1", 1.0)],
            ],
            linked_size: on("bSRL"),
            angles: triplet(inner, ["RIX", "RIY", "RIZ"]),
            angles_random: triplet(inner, ["RBX", "RBY", "RBZ"]),
            rate: triplet(inner, ["RAX", "RAY", "RAZ"]),
            rate_random: triplet(inner, ["RVX", "RVY", "RVZ"]),
            cells: [
                integer(inner, "UvCU").unwrap_or(1).max(1),
                integer(inner, "UvCV").unwrap_or(1).max(1),
            ],
            cell_interval: integer(inner, "UvIv").unwrap_or_default(),
            cell_random: integer(inner, "UvNR").unwrap_or_default().max(0),
            cell_loops: integer(inner, "UvLC").unwrap_or_default(),
            mirrors: on("bRUV"),
            colors,
            frames,
        })
    }

    /// The color the four `Cols` keys hold at `age`, against the frames `Frms` puts them at.
    fn tint(&self, age: f32) -> Vec4 {
        let key = |at: usize| Vec4::from(self.colors[at].map(|lane| f32::from(lane) / 255.0));
        if age <= f32::from(self.frames[0]) {
            return key(0);
        }
        let Some(at) = (1..4).find(|&at| f32::from(self.frames[at]) > age) else {
            return key(3);
        };
        let (low, high) = (f32::from(self.frames[at - 1]), f32::from(self.frames[at]));
        key(at - 1) + (key(at) - key(at - 1)) * ((age - low) / (high - low))
    }

    /// Slot `slot` at `age` frames into the particle carrying it, or nothing where it has not been
    /// made yet or has run out a life of its own.
    fn at(&self, slot: usize, age: f32, key: [u64; 3]) -> Option<Slot> {
        let random = |lane: u64| noise(key, lane);
        let between = |lane, [low, high]: [f32; 2]| low + (high - low) * random(lane);
        let whole = |lane, span: i32| ((span + 1) as f32 * random(lane)).floor();

        let group = (slot as i32 / self.group) / self.per;
        let age = age - (group as f32 * self.interval + whole(0, self.interval_random));
        if age < 0.0 {
            return None;
        }
        let life = self.life + whole(1, self.life_random);
        let age = match (life > 0.0, self.remade) {
            (true, true) => age % life,
            (true, false) if age >= life => return None,
            _ => age,
        };

        let held = self.cells[0] * self.cells[1];
        let mut cell = whole(2, self.cell_random) as i32 % held;
        if self.cell_interval > 0 {
            let walked = cell + (age / self.cell_interval as f32) as i32;
            if self.cell_loops > 0 && walked >= held * self.cell_loops {
                return None;
            }
            cell = match self.cell_loops < 0 && walked >= -(held * self.cell_loops) {
                true => held - 1,
                false => walked.rem_euclid(held),
            };
        }

        let along = match self.life > 0.0 {
            true => (age / self.life).powf(self.curve),
            false => 0.0,
        };
        let across = between(3, self.random_size[0]);
        let down = match self.linked_size {
            true => across,
            false => between(4, self.random_size[1]),
        };
        let heading = match self.heading {
            0 => cone(0.0, std::f32::consts::PI, random(5), random(6)),
            1 => cone(self.cone[0], self.cone[1], random(5), random(6)),
            2 => Vec3::X,
            3 => Vec3::Y,
            4 => Vec3::Z,
            _ => Vec3::ZERO,
        };
        let spread = match self.injection {
            0 => self.spread * Vec3::from([7, 8, 9].map(|lane| 2.0 * random(lane) - 1.0)),
            _ => Vec3::ZERO,
        };
        let swing = |base: Vec3, span: Vec3, lane: u64| {
            base + span * Vec3::from([0, 1, 2].map(|axis| 2.0 * random(lane + axis) - 1.0))
        };

        Some(Slot {
            offset: spread + heading * between(10, self.speed) * age + 0.5 * self.accel * age * age,
            angles: swing(self.angles, self.angles_random, 11)
                + swing(self.rate, self.rate_random, 14) * age,
            size: [
                (self.begin[0] + (self.end[0] - self.begin[0]) * along) * across,
                (self.begin[1] + (self.end[1] - self.begin[1]) * along) * down,
            ],
            color: self.tint(age),
            cell: [cell % self.cells[0], cell / self.cells[0]],
            mirrored: self.mirrors && random(17) >= 0.5,
        })
    }
}

/// The uv rows a particle hands over, narrowed onto one cell of a `UvCU` by `UvCV` atlas. The game
/// writes the cell straight into the vertex it builds, so this stands ahead of whatever the
/// particle's own uv sets do to a coordinate.
fn celled(
    mut rows: [[f32; 4]; UV_SETS * UV_REGISTERS],
    slot: &Slot,
    cells: [i32; 2],
) -> [[f32; 4]; UV_SETS * UV_REGISTERS] {
    let span = [1.0 / cells[0] as f32, 1.0 / cells[1] as f32];
    let middle = [
        (slot.cell[0] as f32 + 0.5) * span[0] - 0.5,
        (slot.cell[1] as f32 + 0.5) * span[1] - 0.5,
    ];
    for row in &mut rows {
        row[3] += row[0] * middle[0] + row[1] * middle[1];
        row[0] *= match slot.mirrored {
            true => -span[0],
            false => span[0],
        };
        row[1] *= span[1];
    }
    rows
}

/// One of the up to four uv sets a particle carries, `UvSt`. The sprite packages read a coordinate
/// the viewer has already transformed and the model packages read the transform itself, so both take
/// the same two rows: `uv' = dot(vec3(uv, 1), row.xyw)`, over a coordinate an effect's own models
/// write centered on the texture's middle.
struct UvSet {
    scale: Pair,
    scroll: Pair,
    turn: Track,
}

impl UvSet {
    fn read(block: &Block) -> Self {
        let blocks = block.blocks();
        Self {
            scale: Pair::read(blocks, "Scl", 1.0),
            scroll: Pair::read(blocks, "Scr", 0.0),
            turn: Track::read(blocks, "Rot", 0.0),
        }
    }

    fn at(&self, frame: f32) -> [[f32; 4]; UV_REGISTERS] {
        let [width, height] = self.scale.at(frame);
        let [across, down] = self.scroll.at(frame);
        let (sin, cos) = self.turn.at(frame).sin_cos();
        [
            [cos * width, -sin * height, 0.0, 0.5 + across],
            [sin * width, cos * height, 0.0, 0.5 + down],
        ]
    }
}

/// Every set a particle carries, as the registers a draw hands over.
fn transform(sets: &[UvSet], frame: f32) -> [[f32; 4]; UV_SETS * UV_REGISTERS] {
    let mut out = program::UV_IDENTITY;
    for (set, held) in sets.iter().take(UV_SETS).enumerate() {
        let rows = held.at(frame);
        out[set * UV_REGISTERS..][..UV_REGISTERS].copy_from_slice(&rows);
    }
    out
}

/// A color: three channels in one curve, with an alpha, a brightness and a per-channel scale
/// written beside them.
struct Tint {
    rgb: Option<Curve>,
    alpha: Track,
    brightness: Track,
    scale: [Track; 4],
}

impl Tint {
    fn read(blocks: &[Block], name: &str) -> Self {
        let inner = nested(blocks, name);
        Self {
            rgb: find(inner, "RGB").and_then(curve::read),
            alpha: Track::read(inner, "A", 1.0),
            brightness: Track::read(inner, "Bri", 1.0),
            scale: ["SclR", "SclG", "SclB", "SclA"].map(|name| Track::read(inner, name, 1.0)),
        }
    }

    fn at(&self, frame: f32) -> Vec4 {
        let rgb = self
            .rgb
            .as_ref()
            .map_or([1.0; 3], |curve| curve.sample(frame));
        let brightness = self.brightness.at(frame);
        Vec4::new(
            rgb[0] * brightness * self.scale[0].at(frame),
            rgb[1] * brightness * self.scale[1].at(frame),
            rgb[2] * brightness * self.scale[2].at(frame),
            self.alpha.at(frame) * self.scale[3].at(frame),
        )
    }
}

/// Where something sits, so a spawned thing can be placed under whatever spawned it.
#[derive(Clone, Copy)]
struct Place {
    origin: Vec3,
    turn: Quat,
    scale: Vec3,
}

impl Place {
    const NONE: Self = Self {
        origin: Vec3::ZERO,
        turn: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    fn under(&self, inner: Place) -> Place {
        Place {
            origin: self.origin + self.turn * (inner.origin * self.scale),
            turn: self.turn * inner.turn,
            scale: self.scale * inner.scale,
        }
    }
}

/// How a particle's color reaches what is already drawn. `RMT`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Blend {
    Opaque,
    Alpha,
    Multiply,
    Screen,
    Subtract,
    Add,
}

impl From<i32> for Blend {
    fn from(value: i32) -> Self {
        match value {
            1 | 9 => Self::Multiply,
            2 | 10 => Self::Add,
            3 | 11 => Self::Subtract,
            4 | 12 => Self::Screen,
            8 => Self::Opaque,
            _ => Self::Alpha,
        }
    }
}

/// A world axis, as `RBDT` names one.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Axis {
    X,
    Y,
    Z,
}

/// What a bill turns to meet: the eye itself, or the plane the frame is drawn on. `RBDT` names the
/// second by `Billboard` and the first by `CameraBillboard`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Toward {
    Eye,
    Screen,
}

/// Which way a sprite is turned to be drawn, `RBDT`. The two that read a velocity are drawn against
/// the screen.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Facing {
    /// Set into the screen's own plane.
    Screen,
    /// Turned to look at the eye.
    Camera,
    /// Billed about the world's up axis, so it turns with the camera but never leans.
    Upright(Toward),
    /// Left lying in the world, across the two axes the one it names stands out of.
    Still(Axis),
}

impl Facing {
    /// What `RBDT` reads as for a particle of `kind`. A decal is cast onto what lies under it, so
    /// the axis it names settles nothing: it is scaled across x and z, where every other kind is
    /// scaled across the two axes the one it names leaves. Naming no base at all is not a default:
    /// the powder package turns a corner by the particle's own angles and never reads the view, so
    /// what names none is left in the plane its own rotation puts it in.
    fn read(kind: i32, base: i32) -> Self {
        match (kind, base) {
            (10..=12, 0..=2 | 10) => Self::Still(Axis::Y),
            (_, 0) => Self::Still(Axis::X),
            (_, 1) => Self::Still(Axis::Y),
            (_, 2 | 10) => Self::Still(Axis::Z),
            (_, 8) => Self::Upright(Toward::Eye),
            (_, 4 | 9) => Self::Upright(Toward::Screen),
            (_, 6) => Self::Camera,
            _ => Self::Screen,
        }
    }
}

/// What a particle draws as.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Shape {
    /// A quad turned to face the camera.
    Sprite,
    /// One of the effect's own models, indexed into [`Effect::models`].
    Model(usize),
}

/// One of the texture roles a particle names, as the package's own sampler is called.
const ROLES: [&str; 8] = [
    "g_SamplerColor1",
    "g_SamplerColor2",
    "g_SamplerColor3",
    "g_SamplerColor4",
    "g_SamplerNormal",
    "g_SamplerDistortion",
    "g_SamplerPalette",
    "g_SamplerReflection_",
];

/// The tags each of those roles is written under.
const SETS: [&str; 8] = ["TC1", "TC2", "TC3", "TC4", "TN", "TD", "TP", "TR"];

/// How a texture set combines with what came before it. `TCCT` and `TCAT`, whose orders VFXEditor
/// names and which the package's own key values follow one for one.
const CALCULATE_COLOR: [&str; 6] = ["Mul", "Add", "Sub", "Max", "Min", "None"];
const CALCULATE_ALPHA: [&str; 4] = ["Mul", "Max", "Min", "None"];

/// How a texture is filtered and wrapped. `TFT` runs from off through three degrees of anisotropy,
/// none of which GL ES has, so anything past off is filtered.
const WRAPS: [u32; 3] = [glow::REPEAT, glow::CLAMP_TO_EDGE, glow::MIRRORED_REPEAT];

/// What a particle asks its shader package for: the keys its own texture sets resolve to, and which
/// of the effect's textures fills each role the package names.
pub struct Shading {
    pub keys: Vec<(u32, u32)>,
    /// The light keys, kept apart because they come off the effect's own settings rather than the
    /// particle's, and a package that carries no such node should still draw the particle textured.
    pub lights: Vec<(u32, u32)>,
    /// The package's own sampler id, the effect's texture behind it, and how it is sampled.
    pub textures: Vec<(u32, usize, u32, [u32; 2])>,
    /// `CalculateColor` and `CalculateAlpha`, the two ratios the package lerps the first color
    /// set's texel towards white by. An alpha-only texture is written with the color ratio at
    /// nought, which is what leaves such a particle its own color rather than the texture's.
    pub calculate: [f32; 2],
    /// Whether this is drawn from a stream the viewer places in the world rather than from one of
    /// the effect's own models.
    pub sprite: bool,
}

/// What a model particle's rim ramp is written as, `FrC` against `ColB` and `ColE`. A file that
/// states none reads as two white ends, which the shader lerps to nothing.
struct Fresnel {
    power: Track,
    begin: Tint,
    end: Tint,
}

impl Fresnel {
    fn read(blocks: &[Block]) -> Self {
        Self {
            power: Track::read(blocks, "FrC", 1.0),
            begin: Tint::read(blocks, "ColB"),
            end: Tint::read(blocks, "ColE"),
        }
    }

    fn at(&self, frame: f32) -> program::Rim {
        program::Rim {
            power: self.power.at(frame),
            begin: self.begin.at(frame).to_array(),
            end: self.end.at(frame).to_array(),
        }
    }
}

struct Particle {
    life: Option<f32>,
    gravity: Track,
    drag: Track,
    position: Axes,
    rotation: Axes,
    scale: Axes,
    spin: [Track; 3],
    color: Tint,
    fresnel: Fresnel,
    uv: Vec<UvSet>,
    texture: Option<usize>,
    shape: Shape,
    facing: Facing,
    blend: Blend,
    simple: Option<Simple>,
    shading: std::sync::Arc<Shading>,
}

/// The keys and textures a particle's own texture sets and depth handling resolve to. Everything a
/// drawing package would read off an `.mtrl` an effect states here, and apricot declares no material
/// keys at all, so all of it lands in the scene group.
fn shading(block: &Block, lights: Option<Vec<(u32, u32)>>, sprite: bool) -> Shading {
    let blocks = block.blocks();
    let mut keys = Vec::new();
    let mut textures = Vec::new();
    let mut key = |name: &str, value: String| {
        keys.push((program::id(name), program::id(&value)));
    };

    let sets = integer(blocks, "UvSN").unwrap_or_default().clamp(0, 4);
    key("UvSetCount_Table", format!("UvSetCount_{sets}"));
    // Each blend family is handed a color prepared differently: a multiply lerps it towards white
    // by the particle's own opacity and a screen scales it by that opacity, where the two families
    // whose own source factor already carries the opacity take the color as it stands.
    key(
        "ComputeFinalColorType_Table",
        match Blend::from(integer(blocks, "RMT").unwrap_or_default()) {
            Blend::Multiply => "ComputeFinalColorType_LerpWhite",
            Blend::Screen => "ComputeFinalColorType_ModulateAlpha",
            _ => "ComputeFinalColorType_NoneControl",
        }
        .to_owned(),
    );
    key(
        "DepthOffsetType_Table",
        match integer(blocks, "DOTy") == Some(1) {
            true => "DepthOffsetType_FixedIntervalNDC",
            false => "DepthOffsetType_Legacy",
        }
        .to_owned(),
    );

    for (at, (tag, role)) in SETS.iter().zip(ROLES).enumerate() {
        let inner = nested(blocks, tag);
        let held = index(inner, "TxNo").or_else(|| index(inner, "TLst"));
        let name = match at {
            0..=3 => format!("TextureColor{}", at + 1),
            4 => "TextureNormal".to_owned(),
            5 => "TextureDistortion".to_owned(),
            6 => "TexturePalette".to_owned(),
            _ => "TextureReflection".to_owned(),
        };
        let on = find(inner, "bEna").and_then(Block::bool) == Some(true) && held.is_some();
        key(
            &format!("{name}_Table"),
            format!("{name}_{}", if on { "Enable" } else { "Disable" }),
        );
        if !on {
            continue;
        }
        let uv = integer(inner, "UvSN").unwrap_or_default().clamp(0, 3);
        // The palette is a lookup rather than a surface, so it has no uv set of its own.
        if at != 6 {
            key(&format!("{name}_UvNo_Table"), format!("{name}_Uv_{uv}"));
        }
        if at <= 3 {
            key(
                &format!("{name}_ColorToAlpha_Table"),
                format!(
                    "{name}_ColorToAlpha_{}",
                    match find(inner, "bC2A").and_then(Block::bool) == Some(true) {
                        true => "On",
                        false => "Off",
                    }
                ),
            );
        }
        // The first color set is what the others are combined into, so it has no arithmetic.
        let combine = |table: &[&str], tag: &str| {
            let held = integer(inner, tag).unwrap_or_default();
            table
                .get(usize::try_from(held).unwrap_or(0))
                .copied()
                .unwrap_or(table[0])
                .to_owned()
        };
        if (1..=3).contains(&at) || at == 7 {
            key(
                &format!("{name}_CalculateColor_Table"),
                format!(
                    "{name}_CalculateColor_{}",
                    combine(&CALCULATE_COLOR, "TCCT")
                ),
            );
        }
        if (1..=3).contains(&at) {
            key(
                &format!("{name}_CalculateAlpha_Table"),
                format!(
                    "{name}_CalculateAlpha_{}",
                    combine(&CALCULATE_ALPHA, "TCAT")
                ),
            );
        }
        if at == 5 {
            for set in 0..UV_SETS {
                let on = find(inner, &format!("bT{}", set + 1)).and_then(Block::bool) == Some(true);
                key(
                    &format!("TextureDistortion_UvSet{set}_Table"),
                    format!(
                        "TextureDistortion_UvSet_{}",
                        if on { "Enable" } else { "Disable" }
                    ),
                );
            }
        }
        let wrap = |tag: &str| {
            let held = integer(inner, tag).unwrap_or_default();
            WRAPS
                .get(usize::try_from(held).unwrap_or(0))
                .copied()
                .unwrap_or(glow::REPEAT)
        };
        let filter = match integer(inner, "TFT").unwrap_or(1) > 0 {
            true => glow::LINEAR,
            false => glow::NEAREST,
        };
        textures.push((
            program::id(role),
            held.unwrap_or_default(),
            filter,
            [wrap("TBUT"), wrap("TBVT")],
        ));
    }
    let first = nested(blocks, "TC1");
    Shading {
        keys,
        lights: lights.unwrap_or_default(),
        textures,
        // The first set combines with the particle's own color rather than with another set, so
        // these two are ratios where the sets below read them as the arithmetic above.
        calculate: [
            integer(first, "TCCT").unwrap_or(1) as f32,
            integer(first, "TCAT").unwrap_or(1) as f32,
        ],
        sprite,
    }
}

impl Particle {
    fn read(block: &Block, models: usize, lights: &[(u32, u32)]) -> Self {
        let blocks = block.blocks();
        let data = nested(blocks, "Data");
        let model = |name| match index(data, name) {
            Some(model) if model < models => Shape::Model(model),
            _ => Shape::Sprite,
        };
        let kind = integer(blocks, "PrVT").unwrap_or_default();
        // The kinds that draw geometry name it under a tag of their own.
        let shape = match kind {
            5 | 14 => model("MdNo"),
            13 => model("MNO"),
            _ => Shape::Sprite,
        };
        let sprite = shape == Shape::Sprite;
        Self {
            life: life(blocks),
            shading: std::sync::Arc::new(shading(
                block,
                (!sprite).then(|| lights.to_vec()),
                sprite,
            )),
            shape,
            gravity: Track::read(blocks, "Gra", 0.0),
            drag: Track::read(blocks, "ARs", 0.0),
            position: Axes::read(blocks, "Pos", 0.0),
            rotation: Axes::read(blocks, "Rot", 0.0),
            scale: Axes::read(blocks, "Scl", 1.0),
            spin: triple(blocks, ["VRX", "VRY", "VRZ"], 0.0),
            color: Tint::read(blocks, "Col"),
            fresnel: Fresnel::read(data),
            uv: blocks
                .iter()
                .filter(|block| block.name() == "UvSt")
                .take(UV_SETS)
                .map(UvSet::read)
                .collect(),
            texture: index(nested(blocks, "TC1"), "TLst"),
            facing: Facing::read(kind, integer(blocks, "RBDT").unwrap_or_default()),
            blend: integer(blocks, "RMT").unwrap_or_default().into(),
            simple: Simple::read(blocks),
        }
    }
}

/// The light keys an effect's own settings resolve to. Where a file names no light the package
/// defaults answer, and those draw an effect unlit.
fn lights(file: &Avfx) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    let mut key = |name: &str, value: &str| out.push((program::id(name), program::id(value)));
    if !matches!(
        file.directional_light_source(),
        None | Some(DirectionalLightSource::None)
    ) {
        key("DirectionalLight_Table", "DirectionalLight_Enable");
    }
    let held = file
        .point_light_sources()
        .iter()
        .filter(|source| {
            !matches!(
                source,
                None | Some(ironworks::file::avfx::PointLightSource::None)
            )
        })
        .count();
    if held > 0 {
        key(
            "PointLightCount_Table",
            match held {
                1 => "PointLightCount_1_0",
                _ => "PointLightCount_1_1",
            },
        );
    }
    out
}

/// When an emitter makes what one of its entries names, `CrTm`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Pass {
    /// Every `CrI` frames, for as long as the emitter runs.
    Interval,
    /// Once, as the emitter starts.
    Start,
    /// Once, as the emitter is destroyed. Nothing reaches this: an emitter here runs out the span
    /// its timeline gives it, with no moment answering to the one the game tears one down at.
    End,
}

/// One entry of an emitter's particle or emitter list.
struct Spawn {
    target: usize,
    /// `CrCn`, how many the start pass makes. The interval pass counts by the emitter's own `CrC`
    /// and never reads this.
    count: i32,
    delay: f32,
    pass: Pass,
}

impl Spawn {
    fn read(item: &Item, of: usize) -> Option<Self> {
        let blocks = item.blocks();
        let target = usize::try_from(integer(blocks, "TgtB")?).ok()?;
        (target < of && !off(blocks, "bEnb")).then(|| Self {
            target,
            count: integer(blocks, "CrCn").unwrap_or(1).clamp(0, 64),
            delay: integer(blocks, "GenD").unwrap_or_default() as f32,
            pass: match integer(blocks, "CrTm").unwrap_or_default() {
                0 => Pass::Interval,
                1 => Pass::Start,
                _ => Pass::End,
            },
        })
    }

    /// How many of this one to make on a burst `local` frames into the emitter's run that bursts
    /// `burst`, where the burst before it came at `previous`. A start entry the file delays is made
    /// on the first burst its delay has run out by rather than dropped.
    fn made(&self, burst: i32, previous: f32, local: f32) -> i32 {
        match self.pass {
            _ if local < self.delay => 0,
            Pass::Interval => burst,
            Pass::Start if previous < self.delay => self.count,
            _ => 0,
        }
    }
}

struct Emitter {
    life: Option<f32>,
    count: Track,
    interval: Track,
    position: Axes,
    rotation: Axes,
    scale: Axes,
    color: Tint,
    /// `Data/IjS`, how fast a particle leaves, along the direction `Data/AnX`..`AnZ` turns `+Y` to.
    speed: Track,
    heading: [Track; 3],
    particles: Vec<Spawn>,
    emitters: Vec<Spawn>,
}

impl Emitter {
    fn read(emitter: &ironworks::file::avfx::Emitter, particles: usize, emitters: usize) -> Self {
        let blocks = emitter.properties();
        let data = nested(blocks, "Data");
        Self {
            life: life(blocks),
            count: Track::read(blocks, "CrC", 1.0),
            interval: Track::read(blocks, "CrI", 1.0),
            position: Axes::read(blocks, "Pos", 0.0),
            rotation: Axes::read(blocks, "Rot", 0.0),
            scale: Axes::read(blocks, "Scl", 1.0),
            color: Tint::read(blocks, "Col"),
            speed: Track::read(data, "IjS", 0.0),
            heading: triple(data, ["AnX", "AnY", "AnZ"], 0.0),
            particles: emitter
                .particles()
                .iter()
                .filter_map(|item| Spawn::read(item, particles))
                .collect(),
            emitters: emitter
                .emitters()
                .iter()
                .filter_map(|item| Spawn::read(item, emitters))
                .collect(),
        }
    }
}

/// One emitter a timeline runs, and the frames it runs over.
struct Run {
    emitter: usize,
    start: i32,
    until: i32,
}

/// The emitters one timeline runs, added to `runs` at `at`.
fn timeline(file: &Avfx, index: usize, at: i32, runs: &mut Vec<Run>) {
    let Some(timeline) = file.timelines().get(index) else {
        return;
    };
    for item in timeline.items() {
        let blocks = item.blocks();
        if off(blocks, "bEna") {
            continue;
        }
        let Some(emitter) = integer(blocks, "EmNo")
            .and_then(|value| usize::try_from(value).ok())
            .filter(|&emitter| emitter < file.emitters().len())
        else {
            continue;
        };
        let end = integer(blocks, "EdTm").unwrap_or(-1);
        runs.push(Run {
            emitter,
            start: at + integer(blocks, "StTm").unwrap_or_default(),
            until: match end < 0 {
                true => i32::MAX,
                false => at + end,
            },
        });
    }
}

fn runs(file: &Avfx) -> Vec<Run> {
    let mut runs = Vec::new();
    for scheduler in file.schedulers() {
        for item in scheduler.items() {
            let blocks = item.blocks();
            if off(blocks, "bEna") {
                continue;
            }
            let Some(index) = integer(blocks, "TlNo").and_then(|value| usize::try_from(value).ok())
            else {
                continue;
            };
            timeline(
                file,
                index,
                integer(blocks, "StTm").unwrap_or_default(),
                &mut runs,
            );
        }
    }
    // An effect whose schedulers start nothing still holds the timelines and emitters it would
    // have run, and is worth showing rather than leaving blank.
    if runs.is_empty() {
        for index in 0..file.timelines().len() {
            timeline(file, index, 0, &mut runs);
        }
    }
    if runs.is_empty() {
        runs.extend((0..file.emitters().len()).map(|emitter| Run {
            emitter,
            start: 0,
            until: i32::MAX,
        }));
    }
    runs
}

/// A model vertex as the game's own shaders read it: four uv sets, and a normal and tangent the
/// shader takes the bias off itself, which is why they go up as the bytes the file holds rather than
/// as the signed values ironworks reads them into.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 4],
    pub normal: [u8; 4],
    pub tangent: [u8; 4],
    pub color: [u8; 4],
    pub uv: [f32; 8],
}

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u16>,
}

fn mesh(model: &Geometry) -> Mesh {
    let biased = |held: [i8; 4]| held.map(|lane| (lane as u8).wrapping_add(128));
    Mesh {
        vertices: model
            .vertices()
            .iter()
            .map(|vertex| Vertex {
                position: vertex.position(),
                normal: biased(vertex.normal()),
                tangent: biased(vertex.tangent()),
                color: vertex.colour(),
                uv: std::array::from_fn(|lane| vertex.uv()[lane / 2][lane % 2]),
            })
            .collect(),
        indices: model
            .triangles()
            .iter()
            .flat_map(|triangle| triangle.indices())
            .collect(),
    }
}

/// One emitter running: a timeline started it, or a parent emitter did.
struct Running {
    def: usize,
    born: i32,
    until: i32,
    place: Place,
    tint: Vec4,
    /// Frames since the last burst.
    since: f32,
    depth: u8,
}

struct Live {
    def: usize,
    born: i32,
    life: f32,
    /// How far it has carried itself under its own velocity, in the frame it was spawned into.
    at: Vec3,
    velocity: Vec3,
    /// Where the emitter stood when it spawned, which its own curves run under.
    place: Place,
    tint: Vec4,
}

pub struct State {
    pub frame: i32,
    running: Vec<Running>,
    particles: Vec<Live>,
}

impl Default for State {
    fn default() -> Self {
        Self {
            // Nothing has run yet, so the first step lands on frame zero.
            frame: -1,
            running: Vec::new(),
            particles: Vec::new(),
        }
    }
}

/// One thing to draw, in the effect's own space.
#[derive(Clone, Copy)]
pub struct Drawn {
    pub center: [f32; 3],
    pub scale: [f32; 3],
    pub turn: [f32; 4],
    /// How far the sprite is spun in the plane it is billed onto, which is the one part of its turn
    /// a quad facing the camera can carry.
    pub roll: f32,
    pub color: [f32; 4],
    /// The rim ramp, which only the model packages read.
    pub rim: program::Rim,
    /// What each uv set does to a texture coordinate, two registers a set.
    pub uv: [[f32; 4]; UV_SETS * UV_REGISTERS],
    pub texture: Option<usize>,
    pub shape: Shape,
    pub facing: Facing,
    pub blend: Blend,
    /// Which of the effect's particles this is one of, which is what its shading is read off.
    pub def: usize,
}

impl Drawn {
    /// Carried into a placement external to the effect itself: a zone stands its own copy wherever
    /// an instance says, so what the emitters ran out in their own space is turned by the
    /// placement's rotation and scale before it is offset into the world, and tinted by whatever
    /// colour and distance fade the placement itself states.
    pub(crate) fn placed(mut self, rotation: Quat, offset: Vec3, scale: f32, tint: Vec4) -> Self {
        self.center = (offset + rotation * (Vec3::from(self.center) * scale)).to_array();
        self.turn = (rotation * Quat::from_array(self.turn)).to_array();
        self.scale = (Vec3::from(self.scale) * scale).to_array();
        self.color = (Vec4::from(self.color) * tint).to_array();
        self
    }
}

pub struct Effect {
    emitters: Vec<Emitter>,
    particles: Vec<Particle>,
    runs: Vec<Run>,
    /// The `.atex` files the particles sample, in the order they index them.
    pub textures: Vec<String>,
    pub models: Vec<Mesh>,
    /// Frames the effect runs for before it starts over. Only meaningful where `bounded` is true;
    /// where it is not, this is `LOOP`, a fallback for scrubbing the file on its own rather than a
    /// span the effect actually starts over at.
    pub length: i32,
    /// Whether every run the file states truly ends and every particle it can spawn has a life of
    /// its own, so nothing outlives the point `length` names. An effect a placement runs forever
    /// has no such point: it settles once its emitters stop spawning and holds there, and wrapping
    /// its frame back to zero anyway restarts it from empty on a cycle nothing in the file states.
    pub bounded: bool,
    /// `SPFR`: how far behind a particle the soft-particle fade reaches. Zero where the file states
    /// none, which the shader divides by, but only the apricot_model technique that samples depth
    /// reads it at all.
    pub fade_range: f32,
}

impl Effect {
    pub fn read(file: &Avfx) -> Self {
        let lights = lights(file);
        let particles: Vec<Particle> = file
            .particles()
            .iter()
            .map(|particle| Particle::read(particle, file.models().len(), &lights))
            .collect();
        let emitters: Vec<Emitter> = file
            .emitters()
            .iter()
            .map(|emitter| Emitter::read(emitter, particles.len(), file.emitters().len()))
            .collect();
        let runs = runs(file);

        // A timeline item's own end is where the effect it placed is done, not a lower bound a
        // particle's own life can run past: an `EdTm` an artist tunes to the effect's length would
        // otherwise need every particle's life hand-matched to it as well.
        let bounded = runs.iter().all(|run| run.until != i32::MAX)
            && particles.iter().all(|particle| particle.life.is_some());
        let length = match bounded {
            true => runs.iter().map(|run| run.until).max().unwrap_or_default(),
            false => LOOP,
        }
        .clamp(1, LONGEST);

        Self {
            emitters,
            particles,
            runs,
            textures: file.textures().to_vec(),
            models: file.models().iter().map(mesh).collect(),
            length,
            bounded,
            fade_range: file.soft_particle_fade_range().unwrap_or(0.0),
        }
    }

    /// Steps to `frame`, replaying from the start where the state sits past it: a particle's
    /// position is the sum of every step it has taken, so there is no stepping backwards.
    pub fn seek(&self, state: &mut State, frame: i32) {
        if frame < state.frame {
            *state = State::default();
        }
        while state.frame < frame {
            self.step(state);
        }
    }

    fn step(&self, state: &mut State) {
        let frame = state.frame + 1;
        state.frame = frame;

        state.particles.retain_mut(|live| {
            let age = (frame - live.born) as f32;
            if age > live.life {
                return false;
            }
            let def = &self.particles[live.def];
            live.velocity *= (1.0 - def.drag.at(age)).clamp(0.0, 1.0);
            live.velocity.y -= def.gravity.at(age);
            live.at += live.velocity;
            true
        });

        for run in &self.runs {
            if run.start == frame && state.running.len() < EMITTERS {
                state.running.push(Running {
                    def: run.emitter,
                    born: frame,
                    until: run.until,
                    place: Place::NONE,
                    tint: Vec4::ONE,
                    since: f32::INFINITY,
                    depth: 0,
                });
            }
        }
        state.running.retain(|running| frame <= running.until);

        let mut spawned = Vec::new();
        let room = EMITTERS.saturating_sub(state.running.len());
        for running in &mut state.running {
            let def = &self.emitters[running.def];
            let local = (frame - running.born) as f32;
            if def.life.is_some_and(|life| local > life) {
                continue;
            }
            running.since += 1.0;
            if running.since < def.interval.at(local).max(1.0) {
                continue;
            }
            // Where the last burst came, which is before the run began until there has been one.
            let previous = local - running.since;
            running.since = 0.0;

            let burst = def.count.at(local).round().clamp(0.0, 64.0) as i32;
            let place = running.place.under(Place {
                origin: def.position.at(local),
                turn: rotation(def.rotation.at(local)),
                scale: def.scale.at(local),
            });
            let tint = running.tint * def.color.at(local);
            let velocity = rotation(read(&def.heading, local)) * Vec3::Y * def.speed.at(local);

            for spawn in &def.particles {
                let life = self.particles[spawn.target].life.unwrap_or(f32::INFINITY);
                for _ in 0..spawn.made(burst, previous, local) {
                    if state.particles.len() >= PARTICLES {
                        break;
                    }
                    state.particles.push(Live {
                        def: spawn.target,
                        born: frame,
                        life,
                        at: Vec3::ZERO,
                        velocity,
                        place,
                        tint,
                    });
                }
            }

            if running.depth < DEPTH {
                for spawn in &def.emitters {
                    if spawn.made(burst, previous, local) == 0 {
                        continue;
                    }
                    if spawned.len() >= room {
                        break;
                    }
                    spawned.push(Running {
                        def: spawn.target,
                        born: frame,
                        until: self.emitters[spawn.target]
                            .life
                            .map_or(i32::MAX, |life| frame + life as i32),
                        place,
                        tint,
                        since: f32::INFINITY,
                        depth: running.depth + 1,
                    });
                }
            }
        }
        state.running.extend(spawned);
    }

    pub fn drawn(&self, state: &State) -> Vec<Drawn> {
        let mut out = Vec::with_capacity(state.particles.len());
        for live in &state.particles {
            let def = &self.particles[live.def];
            let age = (state.frame - live.born) as f32;
            let angles = def.rotation.at(age) + read(&def.spin, age) * age;
            let origin = live.at + def.position.at(age);
            let scale = def.scale.at(age);
            let held = Drawn {
                center: [0.0; 3],
                scale: [0.0; 3],
                turn: [0.0; 4],
                roll: angles.z,
                color: (live.tint * def.color.at(age)).to_array(),
                rim: def.fresnel.at(age),
                uv: transform(&def.uv, age),
                texture: def.texture,
                shape: def.shape,
                facing: def.facing,
                blend: def.blend,
                def: live.def,
            };
            let Some(simple) = &def.simple else {
                let place = live.place.under(Place {
                    origin,
                    turn: rotation(angles),
                    scale,
                });
                out.push(Drawn {
                    center: place.origin.to_array(),
                    scale: place.scale.to_array(),
                    turn: place.turn.to_array(),
                    ..held
                });
                continue;
            };
            for at in 0..simple.slots {
                if out.len() >= PARTICLES {
                    break;
                }
                let key = [live.def as u64, live.born as u64, at as u64];
                let Some(slot) = simple.at(at, age, key) else {
                    continue;
                };
                // The corners the game builds a slot's quad from sit either side of its stated
                // size, where a plain sprite's sit either side of half its scale. Nothing of the
                // particle's own scale reaches them: what the game multiplies them by is the scale
                // of the node the effect stands under, which a placement carries already.
                let place = live.place.under(Place {
                    origin: origin + slot.offset,
                    turn: rotation(angles + slot.angles),
                    scale: Vec3::new(2.0 * slot.size[0], 2.0 * slot.size[1], 1.0),
                });
                out.push(Drawn {
                    center: place.origin.to_array(),
                    scale: place.scale.to_array(),
                    turn: place.turn.to_array(),
                    roll: angles.z + slot.angles.z,
                    color: (live.tint * def.color.at(age) * slot.color).to_array(),
                    uv: celled(held.uv, &slot, simple.cells),
                    ..held
                });
            }
        }
        out
    }

    /// What the shader package a particle is drawn with is asked for.
    pub fn shading(&self, def: usize) -> Option<std::sync::Arc<Shading>> {
        self.particles.get(def).map(|held| held.shading.clone())
    }

    /// A sphere the whole run fits inside, for the camera to open on. A scale is not an extent: a
    /// sprite is drawn one scale wide about its own center and only across the two axes it is billed
    /// onto, and a model is drawn its own geometry wide, so taking the scale for either stands the
    /// camera off by several times too far.
    pub fn fit(&self) -> (Vec3, f32) {
        let models: Vec<f32> = self
            .models
            .iter()
            .map(|mesh| {
                mesh.vertices
                    .iter()
                    .map(|vertex| Vec3::from_slice(&vertex.position).length())
                    .fold(0.0f32, f32::max)
            })
            .collect();

        let mut state = State::default();
        let (mut low, mut high) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for _ in 0..self.length.min(FITTED) {
            self.step(&mut state);
            for live in &state.particles {
                let def = &self.particles[live.def];
                let age = (state.frame - live.born) as f32;
                let at = live.place.origin
                    + live.place.turn * ((live.at + def.position.at(age)) * live.place.scale);
                let scale = (def.scale.at(age) * live.place.scale).abs();
                let reach = match def.shape {
                    Shape::Sprite => {
                        0.5 * match def.facing {
                            Facing::Still(Axis::X) => scale.y.max(scale.z),
                            Facing::Still(Axis::Y) => scale.x.max(scale.z),
                            _ => scale.x.max(scale.y),
                        }
                    }
                    Shape::Model(index) => {
                        scale.max_element() * models.get(index).copied().unwrap_or(0.5)
                    }
                }
                .max(0.05);
                low = low.min(at - reach);
                high = high.max(at + reach);
            }
        }
        match low.cmple(high).all() {
            true => (
                (low + high) * 0.5,
                ((high - low) * 0.5).max_element().max(0.1),
            ),
            false => (Vec3::ZERO, 1.0),
        }
    }
}

#[cfg(test)]
mod test {
    use ironworks::file::File;
    use ironworks::file::avfx::Avfx;

    use glam::{Vec3, Vec4};

    use super::{Blend, Effect, Fresnel, Live, Place, State, nested};

    /// One block as the format writes it: the tag back to front and null-padded, its length, then
    /// a payload rounded up to the next four bytes.
    fn block(tag: &str, payload: &[u8]) -> Vec<u8> {
        let mut head = [0u8; 4];
        for (at, byte) in tag.bytes().rev().enumerate() {
            head[at] = byte;
        }
        let mut out = head.to_vec();
        out.extend(u32::try_from(payload.len()).expect("a short payload").to_le_bytes());
        out.extend(payload);
        out.resize(out.len().next_multiple_of(4), 0);
        out
    }

    /// A curve holding one key, whose three floats a scalar reads the last of and a colour all of.
    fn curve(tag: &str, data: [f32; 3]) -> Vec<u8> {
        let mut key = vec![0u8; 4];
        for value in data {
            key.extend(value.to_le_bytes());
        }
        block(tag, &block("Keys", &key))
    }

    fn scalar(tag: &str, value: i32) -> Vec<u8> {
        block(tag, &value.to_le_bytes())
    }

    #[test]
    fn a_model_particle_reads_the_rim_ramp_its_data_block_states() {
        let colour = |tag: &str, rgb: [f32; 3], alpha: f32| {
            block(tag, &[curve("RGB", rgb), curve("A", [0.0, 0.0, alpha])].concat())
        };
        let data = block(
            "Data",
            &[
                curve("FrC", [0.0, 0.0, 3.0]),
                colour("ColB", [1.0, 1.0, 1.0], 0.0),
                colour("ColE", [0.5, 0.25, 0.125], 1.0),
            ]
            .concat(),
        );
        let particle = block("Ptcl", &[scalar("PrVT", 5), data].concat());
        let bytes = block("AVFX", &[scalar("Ver", 0x0001_0000), particle].concat());

        let file = Avfx::read(std::io::Cursor::new(bytes)).expect("a whole file");
        let effect = Effect::read(&file);
        let state = State {
            frame: 0,
            running: Vec::new(),
            particles: vec![Live {
                def: 0,
                born: 0,
                life: 1.0,
                at: Vec3::ZERO,
                velocity: Vec3::ZERO,
                place: Place::NONE,
                tint: Vec4::ONE,
            }],
        };
        let rim = effect.drawn(&state)[0].rim;
        assert_eq!(rim.power, 3.0);
        assert_eq!(rim.begin, [1.0, 1.0, 1.0, 0.0]);
        assert_eq!(rim.end, [0.5, 0.25, 0.125, 1.0]);
    }

    /// The rows a capture's own pipeline state settles. Nought is `SRC_ALPHA -> INV_SRC_ALPHA` and
    /// two is `SRC_ALPHA -> ONE`; ten sets the same pipeline as two, which is what says the eighth
    /// bit leaves the family alone rather than naming one of its own. The one glow the game draws
    /// against `ZERO -> SRC_COLOR` is the second of a pair whose first is that additive two, which
    /// leaves the multiply to the one beside it.
    #[test]
    fn the_blend_modes_are_the_ones_the_captures_set() {
        assert_eq!(Blend::from(0), Blend::Alpha);
        assert_eq!(Blend::from(2), Blend::Add);
        assert_eq!(Blend::from(10), Blend::Add);
        assert_eq!(Blend::from(1), Blend::Multiply);
        assert_eq!(Blend::from(9), Blend::Multiply);
        for mode in 1..=4 {
            assert_eq!(Blend::from(mode), Blend::from(mode + 8), "mode {mode}");
        }
    }

    /// `TCCT` and `TCAT` under the first color set are the two ratios the package lerps its texel
    /// towards white by, so a set that states no color has to reach the shader as nought.
    #[test]
    fn the_first_color_set_states_the_two_ratios() {
        let particle = |color: i32, alpha: i32| {
            block(
                "Ptcl",
                &[
                    scalar("PrVT", 1),
                    block(
                        "TC1",
                        &[scalar("TCCT", color), scalar("TCAT", alpha)].concat(),
                    ),
                ]
                .concat(),
            )
        };
        let bytes = block(
            "AVFX",
            &[scalar("Ver", 0x0001_0000), particle(0, 1), particle(1, 0)].concat(),
        );
        let file = Avfx::read(std::io::Cursor::new(bytes)).expect("a whole file");
        let effect = Effect::read(&file);

        assert_eq!(effect.particles[0].shading.calculate, [0.0, 1.0]);
        assert_eq!(effect.particles[1].shading.calculate, [1.0, 0.0]);
    }

    /// A multiply is drawn against a color the package has already lerped towards white by the
    /// particle's opacity, and a screen against one it has scaled by that opacity, so the blend a
    /// particle names has to reach the key naming which.
    #[test]
    fn the_blend_family_names_how_the_color_is_computed() {
        let final_color = |mode: i32| {
            let particle = block("Ptcl", &[scalar("PrVT", 1), scalar("RMT", mode)].concat());
            let bytes = block("AVFX", &[scalar("Ver", 0x0001_0000), particle].concat());
            let file = Avfx::read(std::io::Cursor::new(bytes)).expect("a whole file");
            let table = super::program::id("ComputeFinalColorType_Table");
            Effect::read(&file).particles[0]
                .shading
                .keys
                .iter()
                .find(|(held, _)| *held == table)
                .expect("the key the package declares")
                .1
        };
        assert_eq!(final_color(9), super::program::id("ComputeFinalColorType_LerpWhite"));
        assert_eq!(final_color(4), super::program::id("ComputeFinalColorType_ModulateAlpha"));
        assert_eq!(final_color(2), super::program::id("ComputeFinalColorType_NoneControl"));
    }

    /// A curve ramping to `end` over `span` frames and starting over, as a scroll is written.
    fn ramp(tag: &str, span: i16, end: f32) -> Vec<u8> {
        let mut keys = Vec::new();
        for (time, value) in [(0i16, 0.0f32), (span, end)] {
            keys.extend(time.to_le_bytes());
            keys.extend(1i16.to_le_bytes());
            for float in [1.0, 1.0, value] {
                keys.extend(f32::to_le_bytes(float));
            }
        }
        block(tag, &[block("Keys", &keys), scalar("BvPo", 1)].concat())
    }

    fn uv_set(width: f32, height: f32, scroll: &[u8]) -> Vec<u8> {
        let scale = [curve("X", [1.0, 1.0, width]), curve("Y", [1.0, 1.0, height])].concat();
        block(
            "UvSt",
            &[block("Scl", &scale), block("Scr", scroll)].concat(),
        )
    }

    /// The rows the game hands `apricot_model` for the Elpis waterfall, taken out of a capture of
    /// it: `n5f104taki1_h1.avfx` particle 1, at the age the two lanes nothing randomizes solve to
    /// together. The other two sets carry a per-particle random the viewer does not draw.
    #[test]
    fn a_uv_set_lands_where_the_game_puts_it() {
        let particle = block(
            "Ptcl",
            &[
                scalar("PrVT", 5),
                uv_set(2.0, 1.0, &ramp("Y", 100, -1.0)),
                uv_set(0.5, 0.5, &[ramp("X", 300, 1.0), ramp("Y", 150, 1.0)].concat()),
                uv_set(2.0, 1.0, &ramp("Y", 120, -1.0)),
                uv_set(1.0, 1.0, &ramp("Y", 80, -1.0)),
            ]
            .concat(),
        );
        let bytes = block("AVFX", &[scalar("Ver", 0x0001_0000), particle].concat());
        let file = Avfx::read(std::io::Cursor::new(bytes)).expect("a whole file");
        let effect = Effect::read(&file);
        let rows = super::transform(&effect.particles[0].uv, 164.9012);

        assert_eq!(rows[4], [2.0, 0.0, 0.0, 0.5]);
        assert_eq!(rows[6], [1.0, 0.0, 0.0, 0.5]);
        assert_eq!([rows[5][0], rows[5][1]], [0.0, 1.0]);
        assert_eq!([rows[7][0], rows[7][1]], [0.0, 1.0]);
        assert!((rows[5][3] - 0.125_823).abs() < 1e-5, "{}", rows[5][3]);
        assert!((rows[7][3] - 0.438_735).abs() < 1e-5, "{}", rows[7][3]);
    }

    /// Whatever angle a set turns through, the game leaves both offsets at a half, so the turn is
    /// about the middle of the texture and not the corner.
    #[test]
    fn a_turned_uv_set_stays_on_the_middle_of_its_texture() {
        let turned = |angle: f32| {
            let particle = block(
                "Ptcl",
                &[
                    scalar("PrVT", 5),
                    block(
                        "UvSt",
                        &[
                            block(
                                "Scl",
                                &[curve("X", [1.0, 1.0, 1.0]), curve("Y", [1.0, 1.0, 1.0])].concat(),
                            ),
                            curve("Rot", [1.0, 1.0, angle]),
                        ]
                        .concat(),
                    ),
                ]
                .concat(),
            );
            let bytes = block("AVFX", &[scalar("Ver", 0x0001_0000), particle].concat());
            let file = Avfx::read(std::io::Cursor::new(bytes)).expect("a whole file");
            super::transform(&Effect::read(&file).particles[0].uv, 0.0)
        };
        for angle in [0.0, 0.566_55, 1.570_796_3, 3.141_592_7] {
            let rows = turned(angle);
            assert_eq!([rows[0][3], rows[1][3]], [0.5, 0.5], "at {angle}");
            assert!((rows[0][0] - angle.cos()).abs() < 1e-6, "at {angle}");
            assert!((rows[1][0] - angle.sin()).abs() < 1e-6, "at {angle}");
        }
    }

    /// `CrTm` names which of an emitter's creation passes makes an entry: nought on the emitter's
    /// own interval, counting by its `CrC`, and one as the emitter starts, counting by the entry's
    /// own `CrCn`. A particle stating no life outlives every interval, so only that once-only pass
    /// keeps one from piling up.
    #[test]
    fn an_emitter_makes_its_start_entries_once_and_its_interval_entries_over_and_over() {
        let span = |tag: &str, value: f32| block(tag, &block("Val", &value.to_le_bytes()));
        let entry = |target: i32, when: i32, count: i32, delay: i32| {
            [
                scalar("bEnb", 1),
                scalar("TgtB", target),
                scalar("CrTm", when),
                scalar("CrCn", count),
                scalar("GenD", delay),
            ]
            .concat()
        };
        let emitter = block(
            "Emit",
            &[
                span("Life", -1.0),
                curve("CrC", [1.0, 1.0, 1.0]),
                curve("CrI", [1.0, 1.0, 15.0]),
                block(
                    "ItPr",
                    &[
                        entry(0, 0, 0, 0),
                        entry(1, 1, 1, 0),
                        entry(2, 1, 2, 0),
                        entry(3, 1, 1, 1),
                    ]
                    .concat(),
                ),
            ]
            .concat(),
        );
        // A second emitter counting nothing at all, whose start entry the count never reaches.
        let idle = block(
            "Emit",
            &[
                span("Life", -1.0),
                curve("CrC", [1.0, 1.0, 0.0]),
                block("ItPr", &entry(4, 1, 1, 0)),
            ]
            .concat(),
        );
        let particle = |life: f32| block("Ptcl", &[scalar("PrVT", 1), span("Life", life)].concat());
        let bytes = block(
            "AVFX",
            &[
                scalar("Ver", 0x0001_0000),
                particle(32.0),
                particle(-1.0),
                particle(-1.0),
                particle(-1.0),
                particle(-1.0),
                emitter,
                idle,
            ]
            .concat(),
        );

        let file = Avfx::read(std::io::Cursor::new(bytes)).expect("a whole file");
        let effect = Effect::read(&file);
        let mut state = State::default();
        let alive = |state: &State| {
            let mut out = [0; 5];
            for live in &state.particles {
                out[live.def] += 1;
            }
            out
        };

        // One burst in, which the delayed entry misses by a frame.
        effect.seek(&mut state, 10);
        assert_eq!(alive(&state), [1, 1, 2, 0, 1]);

        // A 32-frame life over a 15-frame interval leaves the two bursts before this one still
        // running, and the delayed entry stays at the burst that first cleared its delay.
        effect.seek(&mut state, 330);
        assert_eq!(alive(&state), [3, 1, 2, 1, 1]);
    }

    /// A file stating no ramp has to leave the lerp an identity rather than a black, invisible one.
    #[test]
    fn a_particle_with_no_ramp_reads_as_two_white_ends() {
        let rim = Fresnel::read(nested(&[], "Data")).at(0.0);
        assert_eq!(rim.power, 1.0);
        assert_eq!(rim.begin, [1.0; 4]);
        assert_eq!(rim.end, [1.0; 4]);
    }

    /// A particle carrying a `Smpl` layer, whose slots the tests below read.
    fn simple(inner: &[Vec<u8>], flag: i32) -> Vec<u8> {
        block(
            "Ptcl",
            &[
                scalar("PrVT", 1),
                scalar("bSCt", flag),
                block("Smpl", &inner.concat()),
            ]
            .concat(),
        )
    }

    fn shown(bytes: Vec<u8>, frame: i32) -> Vec<super::Drawn> {
        let file = Avfx::read(std::io::Cursor::new(bytes)).expect("a whole file");
        let effect = Effect::read(&file);
        let state = State {
            frame,
            running: Vec::new(),
            particles: vec![Live {
                def: 0,
                born: 0,
                life: f32::INFINITY,
                at: Vec3::ZERO,
                velocity: Vec3::ZERO,
                place: Place::NONE,
                tint: Vec4::ONE,
            }],
        };
        effect.drawn(&state)
    }

    fn effect(inner: &[Vec<u8>], flag: i32) -> Vec<u8> {
        block(
            "AVFX",
            &[scalar("Ver", 0x0001_0000), simple(inner, flag)].concat(),
        )
    }

    /// `w_fire208n1y`'s flame makes one slot every `CrI` frames and gives each `CrIL` frames of its
    /// own, so its sixteen slots are one frame apart and the last dies fifteen frames after the
    /// first. Four slots on a four-frame life run the same shape out in seven.
    #[test]
    fn a_simple_layer_makes_a_slot_an_interval_and_retires_it_a_life_later() {
        let held = [
            scalar("CCnt", 4),
            scalar("CrI", 1),
            scalar("CrIC", 1),
            scalar("BlkN", 1),
            scalar("CrIL", 4),
        ];
        let counts: Vec<usize> = (0..9).map(|frame| shown(effect(&held, 1), frame).len()).collect();
        assert_eq!(counts, [1, 2, 3, 4, 3, 2, 1, 0, 0]);
    }

    /// The whole of the atlas drawn at once is what made the brazier read as a grid of little
    /// flames. A slot takes one cell of it, walking one cell every `UvIv` frames of its own age and
    /// starting over at the end, and the rows it hands the shader put that cell's corners where the
    /// texture holds them.
    #[test]
    fn a_simple_slot_takes_one_cell_of_its_atlas_and_walks_it() {
        let held = [
            scalar("CCnt", 1),
            scalar("CrIL", -1),
            scalar("UvCU", 4),
            scalar("UvCV", 4),
            scalar("UvIv", 1),
        ];
        // The corner a quad's own uv reaches: the shader is handed a coordinate about the middle.
        let corners = |frame: i32| {
            let uv = shown(effect(&held, 1), frame)[0].uv;
            let at = |u: f32, v: f32| {
                [
                    uv[0][0] * u + uv[0][1] * v + uv[0][3],
                    uv[1][0] * u + uv[1][1] * v + uv[1][3],
                ]
            };
            [at(-0.5, -0.5), at(0.5, 0.5)]
        };
        assert_eq!(corners(0), [[0.0, 0.0], [0.25, 0.25]]);
        assert_eq!(corners(1), [[0.25, 0.0], [0.5, 0.25]]);
        assert_eq!(corners(4), [[0.0, 0.25], [0.25, 0.5]]);
        assert_eq!(corners(5), [[0.25, 0.25], [0.5, 0.5]]);
        assert_eq!(corners(15), [[0.75, 0.75], [1.0, 1.0]]);
        assert_eq!(corners(16), [[0.0, 0.0], [0.25, 0.25]]);
    }

    /// A slot is drawn between the size it begins at and the one it ends at, over its own life and
    /// either side of its own middle, where a plain sprite is drawn either side of half its scale.
    #[test]
    fn a_simple_slot_runs_from_the_size_it_begins_at_to_the_one_it_ends_at() {
        let held = [
            scalar("CCnt", 1),
            scalar("CrIL", 4),
            scalar("SBX", 0.5f32.to_bits() as i32),
            scalar("SEX", 0.25f32.to_bits() as i32),
            scalar("SBY", 1.0f32.to_bits() as i32),
            scalar("SEY", 1.0f32.to_bits() as i32),
            scalar("SC", 1.0f32.to_bits() as i32),
        ];
        let width = |frame: i32| shown(effect(&held, 1), frame)[0].scale[0];
        assert_eq!(width(0), 1.0);
        assert_eq!(width(2), 0.75);
        assert_eq!(width(3), 0.625);
        assert_eq!(shown(effect(&held, 1), 3)[0].scale[1], 2.0);
    }

    /// `Cols` is four colors against the four frames `Frms` states, lerped over a slot's own age and
    /// held at either end.
    #[test]
    fn a_simple_slot_reads_the_color_keys_beside_its_frames() {
        let mut keys = Vec::new();
        for color in [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [255, 255, 255, 0]] {
            keys.extend(color);
        }
        let mut frames = Vec::new();
        for frame in [2i16, 4, 6, 8] {
            frames.extend(frame.to_le_bytes());
        }
        let held = [
            scalar("CCnt", 1),
            scalar("CrIL", -1),
            block("Cols", &keys),
            block("Frms", &frames),
        ];
        let color = |frame: i32| shown(effect(&held, 1), frame)[0].color;
        assert_eq!(color(0), [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(color(2), [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(color(3), [0.5, 0.5, 0.0, 1.0]);
        assert_eq!(color(6), [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(color(9), [1.0, 1.0, 1.0, 0.0]);
    }

    /// The game runs the whole layer behind `bSCt`, so a particle that holds the block without
    /// turning it on is drawn as the one quad it was before.
    #[test]
    fn a_particle_that_states_no_bsct_keeps_the_quad_of_its_own() {
        let held = [scalar("CCnt", 16), scalar("CrIL", 16), scalar("UvCU", 4), scalar("UvCV", 4)];
        assert_eq!(shown(effect(&held, 1), 0).len(), 1);
        let plain = shown(effect(&held, 0), 0);
        assert_eq!(plain.len(), 1);
        assert_eq!(plain[0].uv, super::program::UV_IDENTITY);
    }

    fn real(tag: &str, value: f32) -> Vec<u8> {
        block(tag, &value.to_le_bytes())
    }

    /// A slot carries a heading off `SIDT` at a speed between `VMin` and `VMax`, an acceleration
    /// `CGX`..`CGZ` the game applies over its whole age at once, and an angle that starts at `RI`
    /// and turns by `RA` a frame.
    #[test]
    fn a_simple_slot_rises_turns_and_falls_under_what_its_file_states() {
        let held = [
            scalar("CCnt", 1),
            scalar("CrIL", -1),
            scalar("SIDT", 3),
            real("VMin", 0.25),
            real("VMax", 0.25),
            real("CGZ", 0.5),
            real("RIZ", 0.5),
            real("RAZ", 0.25),
        ];
        let at = |frame: i32| {
            let held = &shown(effect(&held, 1), frame)[0];
            (held.center, held.roll)
        };
        assert_eq!(at(0), ([0.0, 0.0, 0.0], 0.5));
        assert_eq!(at(4), ([0.0, 1.0, 4.0], 1.5));

        // The same angle again as the quaternion the shape packages are handed, which is what a
        // sprite the camera has no say in is turned by.
        let turned = glam::Quat::from_array(shown(effect(&held, 1), 4)[0].turn) * Vec3::X;
        assert!(turned.abs_diff_eq(Vec3::new(0.070_737_2, 0.997_495_0, 0.0), 1e-5), "{turned:?}");

        // `SIPT` nought spreads a slot over the box `CrAX`..`CrAZ` states rather than standing it
        // on the particle's own middle.
        let spread = [scalar("CCnt", 1), scalar("CrIL", -1), real("CrAX", 2.0)];
        let placed = shown(effect(&spread, 1), 0)[0].center;
        assert!(placed[0] != 0.0 && placed[0].abs() <= 2.0, "{placed:?}");
        assert_eq!([placed[1], placed[2]], [0.0, 0.0]);
    }

    /// The size a slot is drawn at is the one it states scaled by a random between `SRX0` and
    /// `SRX1`, walked from begin to end along `SC`.
    #[test]
    fn a_simple_slot_takes_the_size_random_and_the_curve_beside_it() {
        let flat = [
            scalar("CCnt", 1),
            scalar("CrIL", -1),
            real("SRX0", 2.0),
            real("SRX1", 2.0),
            real("SRY0", 3.0),
            real("SRY1", 3.0),
        ];
        assert_eq!(shown(effect(&flat, 1), 0)[0].scale, [4.0, 6.0, 1.0]);

        let curved = [
            scalar("CCnt", 1),
            scalar("CrIL", 4),
            real("SBX", 1.0),
            real("SEX", 0.0),
            real("SC", 2.0),
        ];
        assert_eq!(shown(effect(&curved, 1), 2)[0].scale[0], 1.5);
    }

    /// `bCrN` makes a slot start over where it would have died, and `UvLC` retires it once it has
    /// walked its atlas that many times.
    #[test]
    fn a_simple_slot_starts_over_or_stops_where_its_file_says() {
        let remade = [
            scalar("CCnt", 1),
            scalar("CrIL", 4),
            block("bCrN", &[1]),
            real("SBX", 1.0),
            real("SEX", 0.0),
        ];
        let counts: Vec<usize> = (0..9).map(|frame| shown(effect(&remade, 1), frame).len()).collect();
        assert_eq!(counts, [1; 9]);
        // A slot that starts over is back where it began rather than carrying its age past its life.
        let width = |frame: i32| shown(effect(&remade, 1), frame)[0].scale[0];
        assert_eq!(width(1), 1.5);
        assert_eq!(width(5), 1.5);
        assert_eq!(width(3), 0.5);

        let looped = [
            scalar("CCnt", 1),
            scalar("CrIL", -1),
            scalar("UvCU", 2),
            scalar("UvCV", 2),
            scalar("UvIv", 1),
            scalar("UvLC", 1),
        ];
        let counts: Vec<usize> = (0..6).map(|frame| shown(effect(&looped, 1), frame).len()).collect();
        assert_eq!(counts, [1, 1, 1, 1, 0, 0]);
    }

    /// `UvNR` is the range a slot draws its first cell from, which is what stands the glow sprites
    /// of one effect on different frames of the same atlas.
    #[test]
    fn a_simple_slot_starts_on_a_cell_uvnr_picks() {
        let held = |range: i32| {
            [
                scalar("CCnt", 16),
                scalar("CrI", 0),
                scalar("CrIL", -1),
                scalar("UvCU", 4),
                scalar("UvCV", 4),
                scalar("UvNR", range),
            ]
        };
        let starts = |range: i32| {
            let mut seen: Vec<[f32; 4]> = shown(effect(&held(range), 1), 0)
                .into_iter()
                .map(|one| one.uv[0])
                .collect();
            seen.dedup();
            seen.len()
        };
        assert_eq!(starts(0), 1);
        assert!(starts(15) > 1, "every slot started on the same cell");
    }
}
