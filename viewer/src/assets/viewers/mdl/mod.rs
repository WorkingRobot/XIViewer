//! `.mdl` models, drawn.
//!
//! Geometry comes off the file when it is decoded; the materials it names are fetched afterwards and
//! land on meshes already on screen, so a model shows as untextured geometry first and dresses
//! itself as its textures arrive.
//!
//! The shading approximates the game's rather than reproducing it: a color table row is picked the
//! way the game picks one and drives a diffuse color, a specular color and a specular exponent, the
//! mask map scales all three, and everything is lit by three lights that follow the camera instead
//! of by the scene's. Skinning, dyes and decals are all absent, so a character stands in bind pose.
//!
//! Shape keys are applied by rewriting the indices they name, which is what the file states rather
//! than a blend, so a shape is either on or off.

pub(super) mod deferred;
mod deform;
pub mod dye;
mod effects;
mod emote;
mod export;
pub(super) mod gpu;
mod grid;
pub(crate) mod material;
mod noise;
pub(super) mod program;
mod skin;
mod wield;

pub use deform::{Deform, Deformers};
pub use skin::motion_names;
pub use dye::Templates as DyeTemplates;
pub use program::Customize;

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ops::Range;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use egui::{Color32, Label, RichText, ScrollArea, Sense, TextureHandle, TextureOptions};
use glam::{Mat3, Mat4, Vec3, Vec4};
use ironworks::file::{
    File,
    imc::ImageChange,
    mdl::{Lod, MeshKind, ModelContainer, VertexAttributeKind, VertexFormat, VertexValues},
    mtrl,
    shpk::ShaderPackage,
    spm::ShaderParameters,
};
use std::io::Cursor;

use super::{Preview, facts, link, placed, section};
use crate::assets::Bytes;
use crate::backend::Backend;
use crate::data::DecodedTexture;
use crate::utils::TrackedPromise;
use crate::utils::export::Choice;

use material::{Material, Role};

/// What a model's textures may be decoded to, as the longest edge of the mipmap taken. The last
/// takes the file's own mip nought, which is what the game itself draws.
const DETAIL: [(Option<u16>, &str); 4] = [
    (Some(512), "512"),
    (Some(1024), "1024"),
    (Some(2048), "2048"),
    (None, "Authored"),
];

/// Decoded texture bytes one model may hold. Past it the rest of its surfaces draw untextured.
const TEXTURE_BUDGET: usize = 256 << 20;

/// The attributes an imc entry's mask reaches, which the format gives ten bits.
const IMC_ATTRIBUTES: u32 = 0x3ff;

/// Vertical field of view.
const FOV: f32 = 40.0_f32.to_radians();

/// How much of the model's radius the initial framing leaves as margin.
const MARGIN: f32 = 1.25;

/// The scene key deciding whether a shader skins, and the value asking it to. Nothing in a file
/// says it; a mesh carrying bone indices is what the engine would set it from.
const TRANSFORM_VIEW: u32 = 0xa5a1_910d;
const TRANSFORM_VIEW_SKIN: u32 = 0x9c14_c8e9;

/// The scene key deciding whether a background shader reads the normal map at all. A package
/// defaults it to off, and the variant that answer selects samples no normal map, so the frame it
/// writes is the geometry's own.
const GET_NORMAL_MAP: u32 = 0xcbdf_d5ec;
const GET_NORMAL_MAP_ON: u32 = 0xd999_4ef1;
/// The third value, which walks the normal map's blue channel as a parallax height under the
/// material's own `g_HeightScale`. Only `bg.shpk` ships a node for it.
const GET_NORMAL_MAP_PARALLAX: u32 = 0xd9fd_8a1c;

/// The scene key deciding whether a character shader clips against its own alpha threshold. A
/// package defaults it to off, and the variant that answer selects carries no clip at all, so a
/// material's cutout leaves the geometry it was authored over standing.
const APPLY_ALPHA_CLIP: u32 = 0xdcfc_844e;
const APPLY_ALPHA_CLIP_ON: u32 = 0x59c4_e6db;

/// `ApplyDitherClip`, and the value that puts the clip a partly faded character is drawn through
/// in. The variant it selects discards each pixel whose `-dither.tex` texel stands above
/// `m_MulColor.w`, so at full opacity it discards nothing and the two readings draw the same frame.
const APPLY_DITHER_CLIP: u32 = 0x8b03_6665;
const APPLY_DITHER_CLIP_ON: u32 = 0x61b0_cf19;

/// `ApplyWavingAnim`, and the value that lets the wind reach a surface. Set only where the model's
/// own header allows it, which is what keeps a wall from swaying with the leaves.
const APPLY_WAVING_ANIM: u32 = 0x105c_6a52;
const APPLY_WAVING_ANIM_ON: u32 = 0xf801_b859;

/// `GetRLR`, water's own local-reflection toggle. A capture of a real frame carries it on; a
/// package defaults it off, and the variant that answer selects has no `g_SamplerReflectionMap` at
/// all rather than one nothing here fills.
const GET_RLR: u32 = 0x1143_3f2d;
const GET_RLR_ON: u32 = 0x4ba7_7904;

/// Packages with no opaque pass that still belong on the dedicated semi-transparent buffer rather
/// than water's shared-G-buffer fallback: glass and veils are transparent everywhere, not a fringe
/// around an opaque fill, and a captured frame keeps their own composite out of the frame's alpha
/// (glare share) entirely, which only the semi-transparent buffer's resolve does.
const CHARACTER_TRANSPARENT_PACKAGES: [&str; 11] = [
    "/character.shpk",
    "/characterlegacy.shpk",
    "/characterglass.shpk",
    "/charactertattoo.shpk",
    "/charactertransparency.shpk",
    "/characterocclusion.shpk",
    "/characterinc.shpk",
    "/characterscroll.shpk",
    "/hair.shpk",
    "/iris.shpk",
    "/skin.shpk",
];

/// Where the key light stands, in the model's own space. Anchored rather than carried with the
/// camera: a rig that turns with the eye shades every angle alike, so orbiting reveals no form.
const KEY: Vec3 = Vec3::new(-0.45, 0.78, 0.44);

/// The marker a drawn weapon's own effect is stood in for by, apart from the emote vfx marker's
/// own tint so the two read as different things.
const WEAPON_VFX_COLOR: [f32; 4] = [0.35, 0.95, 0.65, 1.0];

/// How far the placed light reaches, in radii of the model. A lamp is drawn as the box it covers
/// and cut off at the sphere of its own reach, and both of those show as a hard edge where they
/// cross what is drawn, so the box stands well outside it. The near and far planes have to hold the
/// whole box: a face of it that the planes cut leaves a straight edge that moves with the camera.
const LAMP_SPAN: f32 = 4.0;

/// What the placed light is worth beside the sun, stated as the color its pass squares. At one it is
/// a second key rather than a fill, and a pale surface under two keys is lit to the top of what the
/// frame holds, with no shading left.
const LAMP_FILL: f32 = 0.55;

/// A vertex as the shader reads it. `#[repr(C)]` with no padding, so a mesh uploads as its own
/// slice.
///
/// Every semantic a drawing package asks for is here whether or not a given shader reads it, so one
/// upload serves both this viewer's own shading and the game's, which bind different subsets of it.
/// Tangents are kept as the file states them, since the game's own shaders do their own unbiasing.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    tangent: [f32; 4],
    bitangent: [f32; 4],
    uv: [f32; 4],
    uv1: [f32; 4],
    color: [u8; 4],
    color1: [u8; 4],
    /// Sixteen bits each, since a skinned shader reads the low byte as the first four influences
    /// and the high byte as the next four.
    weights: [u16; 4],
    bones: [u16; 4],
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

/// One part of a mesh, drawn with the rest of it but hideable on its own.
struct Part {
    range: Range<usize>,
    /// A cell rather than a plain bool: the imc variant this defaults from is fetched after the
    /// level is built, and applying it only has to reach the part, not rebuild the level around it.
    shown: Cell<bool>,
    /// What the model's attribute table calls this part, which is the only name it carries. Empty
    /// where the part claims no attribute.
    attributes: String,
    /// The bits behind that name, as the file's own attribute mask states them. Nought where the
    /// part claims no attribute, which is what leaves it shown whatever variant is picked.
    mask: u32,
}

/// One mesh of the model, as far as the browser cares about it.
struct Mesh {
    /// Which of the level's pieces this came out of, since each carries its own `.imc` and so its
    /// own attribute mask.
    piece: usize,
    material: usize,
    vertices: usize,
    triangles: usize,
    /// The runs of indices the file splits the mesh into, and whether each draws. A mesh the file
    /// does not split holds the one run covering all of them.
    parts: Vec<Part>,
    /// The mesh's indices as the file lists them, kept only where the model has a shape key that
    /// could rewrite them, since applying one is a rewrite of these rather than of what is on the
    /// card.
    base: Vec<u16>,
}

/// Which of the level's meshes a shape touches, and for each the indices it replaces.
type Rewrites = Vec<(usize, Vec<(u16, u16)>)>;

/// One shape key, and where it rewrites the geometry.
struct Shape {
    name: String,
    rewrites: Rewrites,
}

/// Shape keys the file names as variants of one thing, which the browser offers as alternatives
/// rather than as switches that stack. A name carrying no variant stands in a group of its own.
struct Group {
    /// The file's own abbreviation, left as it writes it. Empty for a shape standing alone.
    category: String,
    /// Positions in [`Level::shapes`], each with the variant its name ends in.
    variants: Vec<(usize, String)>,
}

/// A texture, from the moment it is asked for to the moment it can be bound.
enum Texture {
    Fetching(TrackedPromise<Result<DecodedTexture>>),
    Ready(TextureHandle),
    /// It would not load, or the model had already spent its budget.
    Absent,
}

/// A material, from the moment it is asked for to the moment it can be drawn with.
enum Slot {
    Fetching(TrackedPromise<Result<Vec<u8>>>),
    Ready(Box<Material>),
    Failed(String),
}

/// A shader package, from the moment a material names it to the moment it can be translated.
enum Package {
    Fetching(TrackedPromise<Result<Vec<u8>>>),
    Ready(Vec<u8>),
    Failed(String),
}

/// One of the game's own texture arrays. Kept once decoded, since a level built later has a context
/// of its own to hand it to and asking for it again would be a fetch the user watches land.
enum Array {
    Fetching(TrackedPromise<Result<Vec<u8>>>),
    Ready(deferred::Layered),
    Failed,
}

/// One of the game's own shader parameter files. Kept once parsed, since the table every file writes
/// into is built again from all of them each time one more arrives.
enum Parameters {
    Fetching(TrackedPromise<Result<Vec<u8>>>),
    Ready(ShaderParameters),
    Failed,
}

/// The model's own `.imc`, which states which attribute-gated parts a variant draws. `Absent` covers
/// both a model this could name no such file for and one whose file would not read, and either way
/// means what today means: every part shows, whatever it claims an attribute for.
enum Imc {
    Fetching(TrackedPromise<Result<Vec<u8>>>),
    Ready(ImageChange),
    Absent,
}

/// One material's translated shaders, and what they were translated for: how many attachments the
/// context allows decides how many of the G-buffer's targets fit in one reading.
struct Translated {
    attachments: usize,
    held: Result<Passes, String>,
}

/// What a material draws with. A semitransparent package declares no G pass at all: it is drawn
/// over the frame the composite resolved rather than into the buffer that frame came from.
#[derive(Default)]
struct Passes {
    /// The buffer pass, one reading per page of its targets.
    buffer: Vec<Arc<program::Program>>,
    depth: Option<Arc<program::Program>>,
    /// What the material resolves itself into the frame with, drawn as its own geometry after the
    /// lighting. A package with no buffer pass has only the semitransparent one.
    resolve: Option<Arc<program::Program>>,
    /// The pair that draws what the buffer pass clipped away: the surface into a buffer of its own,
    /// and what resolves that over the frame the opaque half left.
    sheer: Option<(Arc<program::Program>, Arc<program::Program>)>,
}

/// The color table in the game's own layout: its halfs, the texels a row takes, and the rows.
type Table = Arc<(Vec<u16>, usize, usize)>;
/// A material's last-applied stains, and the table they were dyed into.
type Dyed = (Table, [Option<u8>; 2]);

/// A piece to carry rigidly at another bone's placement rather than pose on the shared rig: a
/// weapon, hanging off whichever bone its attach point names this frame.
struct Attachment {
    /// The file this piece was worn as, which is what a mesh's own `piece` index names it by.
    path: String,
    bone: String,
    local: Mat4,
}

/// One detail level's geometry, and everything the browser says about it.
struct Level {
    identity: Vec<(&'static str, String)>,
    meshes: Vec<Mesh>,
    /// Shape keys reaching this detail level, in the order the file declares them.
    shapes: Vec<Shape>,
    /// The same shapes, gathered by the category their names share.
    groups: Vec<Group>,
    /// Material paths, in the order meshes index them.
    materials: Vec<String>,
    /// Meshes the file lists but whose vertices would not read, with why.
    unreadable: Vec<(usize, String)>,
    /// Framing the model starts at, so the view can be put back.
    home: Camera,
    /// Half the bounding box's diagonal, which the depth range is cut to.
    radius: f32,
    /// Whether any mesh carries bone indices, which is what decides whether the game would draw
    /// this model through its skinning variant.
    skinned: bool,
    /// Whether the wind may reach it, which the model's own header states.
    waving: bool,
    /// The bones each mesh's blend indices name, in the order they index them.
    bones: Vec<Vec<String>>,
    /// How many attributes each piece declares. An imc variant's mask means something over only
    /// this many of its bits; the rest are padding the format reserves rather than states.
    attributes: Vec<usize>,
    gpu: Arc<Mutex<gpu::Model>>,
}

/// One file the level was built from. A character is worn out of several, and each carries its own
/// `.imc`, so the variant a part's visibility is read from is the piece's rather than the level's.
struct Piece {
    path: String,
    /// The file, read once. A change of clothes builds its level out of the pieces it already
    /// holds, and re-reading a file nothing changed about is most of what that used to cost.
    container: ModelContainer,
    /// The file's own `.imc`, once asked for.
    imc: RefCell<Option<Imc>>,
    /// Which of that imc's variants a part's default visibility is drawn from. Nought is the file's
    /// own default entry.
    variant: Cell<u16>,
    /// Which one it was asked for, which is not always where it settles: a file whose variants are
    /// alternatives rather than toggles is drawn at the first of them.
    asked: u16,
    /// The imc's own `material_id` for `variant`, where a caller has already resolved one. `None`
    /// wherever nothing states one, and the folder `variant` names is what the piece draws with.
    material: Option<u16>,
    deform: Option<Arc<Deform>>,
    skin: Option<u16>,
    rigid: bool,
}

/// One file to build a model out of, and the imc variant it is worn at. Nought is the file's own
/// default entry, which is what anything inspected on its own is shown at.
pub struct Source {
    pub path: String,
    pub bytes: Vec<u8>,
    pub variant: u16,
    /// The imc's own `material_id` for `variant`, where a caller that has the imc in hand has
    /// resolved one: several imc variants commonly share one material, and nought is the entry
    /// stating that the slot draws no material at all. `None` wherever nothing states one.
    pub material: Option<u16>,
    /// What to move the file's vertices by, where it was modelled for a body other than the one
    /// wearing it.
    pub deform: Option<Arc<Deform>>,
    /// The body whose skin to draw it with, where it is a body's own model.
    pub skin: Option<u16>,
    /// Whether this piece hangs rigidly off a bone rather than posing on the shared rig: a weapon,
    /// carried at the placement its own attach point states.
    pub rigid: bool,
}

impl Piece {
    fn new(source: &Source) -> Result<Self> {
        Ok(Self {
            path: source.path.clone(),
            container: ModelContainer::read(Cursor::new(source.bytes.clone()))?,
            imc: RefCell::new(None),
            variant: Cell::new(source.variant),
            asked: source.variant,
            material: source.material,
            deform: source.deform.clone(),
            skin: source.skin,
            rigid: source.rigid,
        })
    }

    /// Whether this is already the file being asked for, worn the same way. The deform is compared
    /// by identity: one is built per body a piece is borrowed from and handed to every piece
    /// borrowing from it, so two that are not the same allocation were built for different bodies.
    fn wears(&self, source: &Source) -> bool {
        self.path == source.path
            && self.asked == source.variant
            && self.material == source.material
            && self.skin == source.skin
            && self.rigid == source.rigid
            && match (&self.deform, &source.deform) {
                (None, None) => true,
                (Some(held), Some(wanted)) => Arc::ptr_eq(held, wanted),
                _ => false,
            }
    }

    /// The picked variant's attribute mask, once the imc has arrived. `None` before it has, or where
    /// it named nothing to read: a part with an attribute then draws exactly as one without.
    fn mask(&self) -> Option<u32> {
        let held = self.imc.borrow();
        let Some(Imc::Ready(image_change)) = held.as_ref() else {
            return None;
        };
        image_change
            .entry(imc_part(&self.path), self.variant.get())
            .map(|entry| u32::from(entry.attribute_mask()))
    }

    /// How many variants past the default the imc carries, once it has arrived.
    fn variants(&self) -> Option<u16> {
        match self.imc.borrow().as_ref() {
            Some(Imc::Ready(image_change)) => Some(image_change.variant_count()),
            _ => None,
        }
    }
}

/// Everything a material owns that outlives the level it was built for.
type Kept = (Option<Slot>, Option<Translated>, Option<Table>);

/// A model, decoded and ready to draw. Everything a detail level owns is rebuilt when one is
/// picked; the camera and the fetched materials and textures are not, so switching neither moves
/// the view nor asks for anything twice.
pub struct Rendered {
    /// The files the level was merged from, in the order its meshes name them.
    pieces: Vec<Piece>,
    lod: Cell<u8>,
    /// Which detail levels the file draws anything at.
    drawn: [bool; 3],
    level: RefCell<Level>,
    /// Shape keys the user has switched on, by name: a detail level built later carries its own
    /// shapes, and the names are what survives the switch.
    shapes: RefCell<BTreeSet<String>>,
    slots: RefCell<Vec<Option<Slot>>>,
    textures: RefCell<BTreeMap<String, Texture>>,
    /// Shader packages the materials name, by path, since several materials share one.
    packages: RefCell<BTreeMap<String, Package>>,
    /// The textures the shaders read that no material names, by resource id.
    arrays: RefCell<BTreeMap<u32, Array>>,
    /// The ones a material does name that egui cannot hold, since their sampler is declared over
    /// slices. Keyed by an `Arc` so a surface built every frame names one without copying the path.
    stacks: RefCell<BTreeMap<Arc<str>, Array>>,
    /// The parameter files the shader type table is filled from, by the record their first profile
    /// lands at.
    parameters: RefCell<BTreeMap<usize, Parameters>>,
    /// The translated shaders, by material.
    translated: RefCell<BTreeMap<usize, Translated>>,
    /// The skeleton the model is skinned to, and the motion it is posed by.
    animation: skin::Animation,
    /// The passes that light the G-buffer, which belong to the frame rather than to a material. Kept
    /// against the attachment count they were built at: a piece change stands up a fresh G-buffer of
    /// its own, and one drawn against a page split this was not built for reads channels nothing
    /// wrote.
    lighting: RefCell<Option<(usize, Arc<gpu::Lighting>)>>,
    /// The pass that grades the frame they resolve, and whether the table it reads has landed.
    post: RefCell<Option<Arc<program::Program>>>,
    graded: Cell<bool>,
    /// The pair that smooths the graded frame's edges, and the chain that occludes it. The second
    /// is kept against the quality it was built at, since that decides which file it came from.
    smoothing: RefCell<Option<Arc<gpu::Smoothing>>>,
    occlusion: RefCell<Option<(usize, Arc<gpu::Occlusion>)>>,
    /// The chain that spreads the bright end of it into a halo.
    glare: RefCell<Option<Arc<gpu::Glare>>>,
    /// The pass that darkens its corners.
    vignette: RefCell<Option<Arc<program::Program>>>,
    reflection: RefCell<Option<Arc<gpu::Reflection>>>,
    /// What those passes are run with, and whether the settings row is open.
    look: Cell<program::Look>,
    settings: Cell<bool>,
    /// The color table in the game's own layout, by material.
    tables: RefCell<BTreeMap<usize, Table>>,
    /// The staining templates a wearer's dye picks read from, handed over once by the character tab.
    dye_templates: RefCell<Option<Rc<dye::Templates>>>,
    /// The stains worn in each piece's slot, by piece index, and the color table each material last
    /// dyed itself into for them: recomputed only where the stains a piece carries change, so a frame
    /// with nothing newly picked costs a lookup rather than a rebuild.
    stains: RefCell<Vec<[Option<u8>; 2]>>,
    dyed: RefCell<BTreeMap<usize, Dyed>>,
    /// A piece hung rigidly off a bone rather than skinned to the shared rig, by the path it was
    /// worn as: a weapon, carried at the placement its attach point states this frame.
    attachments: RefCell<Vec<Attachment>>,
    /// The bones a weapon's own effect would play from, for the weapons carrying one and drawn.
    /// The effect a drawn weapon plays and the bone it hangs from, one pair a weapon, with the
    /// clock it started running on.
    glowing: RefCell<Vec<(String, String)>>,
    /// The rig each carried weapon moves on, whether they are drawn, and the frame clock kept from
    /// the poll so the pose can read it without a context of its own.
    wield: RefCell<wield::Wield>,
    wielded: std::cell::Cell<bool>,
    wall: std::cell::Cell<f64>,
    glowing_at: std::cell::Cell<Option<f64>>,
    /// The props, sound and vfx an emote's own timeline states, read against whatever the body is
    /// playing.
    emote: RefCell<emote::Cue>,
    /// The effects the emote is firing, and where each stood the last frame it was drawn: the
    /// placement is the posed rig's to say, so it is taken during the draw and stepped on the next
    /// poll.
    effects: RefCell<effects::Effects>,
    fired: RefCell<Vec<effects::Fired>>,
    camera: Cell<Camera>,
    /// Which of the two viewers this is, which is what decides how much of the model it takes apart.
    chrome: Cell<Chrome>,
    /// The colours the character was made with, and the attributes it does not wear. A face
    /// declares one part per facial feature and no `.imc` to choose between them, so left to the
    /// variant alone it draws all seven at once over each other.
    customize: Cell<program::Customize>,
    /// The face paint the decal binding was last fetched for.
    painted: Cell<Option<u16>>,
    hidden: RefCell<BTreeSet<String>>,
    /// How tall the character was built, as a scale on everything it is drawn from.
    stature: Cell<f32>,
    /// Whether the rig it is posed on is drawn over it, and the boxes that is drawn as.
    skeleton: Cell<bool>,
    overlay: Arc<Mutex<placed::Placements>>,
    /// Whether a floor is ruled at the origin under it.
    grid: Cell<bool>,
    /// Decoded texture bytes handed to egui so far.
    resident: Cell<usize>,
    debug: Cell<gpu::Debug>,
    /// Whether to draw with the game's own shaders rather than with this viewer's approximation.
    shaded: Cell<bool>,
    /// Why the game's own shaders last failed to build, if they did: turns `shaded` back off so the
    /// model still draws with the plain pass instead of going blank, and names what failed.
    shade_failure: RefCell<Option<String>>,
    /// Seconds the viewer has been open, which is what water and foliage move against.
    clock: Cell<f32>,
    /// Which G-buffer channel the game's own shaders put on screen, starting at the frame their
    /// lighting resolves rather than at a channel of the buffer it is resolved from.
    target: Cell<usize>,
    /// An export in flight.
    export: RefCell<Option<TrackedPromise<()>>>,
}

pub fn decode(path: &str, bytes: &[u8]) -> Result<Preview> {
    let model = compose(&[Source {
        path: path.to_owned(),
        bytes: bytes.to_vec(),
        variant: 0,
        material: None,
        deform: None,
        skin: None,
        rigid: false,
    }])?;
    model.chrome.set(Chrome::Asset);
    model.shaded.set(false);
    Ok(Preview::Model(Box::new(model)))
}

/// Builds one model out of several files, which is how a character is worn. The first is what the
/// rest hang off: its path is what names the skeleton they are all posed on. A character is drawn
/// the way the game draws it, standing in its idle rather than in the pose its files hold.
pub fn compose(parts: &[Source]) -> Result<Rendered> {
    parts.first().context("a model of no files")?;
    let pieces = parts.iter().map(Piece::new).collect::<Result<Vec<_>>>()?;
    let drawn = drawn_levels(&pieces);
    let level = level_of(&pieces, 0, 0)?;
    let camera = level.home;
    Ok(Rendered {
        pieces,
        lod: Cell::new(0),
        drawn,
        slots: RefCell::new((0..level.materials.len()).map(|_| None).collect()),
        shapes: Default::default(),
        level: RefCell::new(level),
        textures: Default::default(),
        packages: Default::default(),
        arrays: Default::default(),
        stacks: Default::default(),
        parameters: Default::default(),
        translated: Default::default(),
        animation: skin::Animation::new(parts.iter().map(|part| part.path.as_str())),
        lighting: Default::default(),
        post: Default::default(),
        graded: Cell::new(false),
        smoothing: Default::default(),
        occlusion: Default::default(),
        glare: Default::default(),
        vignette: Default::default(),
        reflection: Default::default(),
        look: Cell::new(program::Look::default()),
        settings: Cell::new(false),
        tables: Default::default(),
        dye_templates: Default::default(),
        stains: Default::default(),
        dyed: Default::default(),
        attachments: Default::default(),
        glowing: Default::default(),
        wield: Default::default(),
        wielded: Default::default(),
        wall: Default::default(),
        glowing_at: Default::default(),
        emote: Default::default(),
        effects: Default::default(),
        fired: Default::default(),
        camera: Cell::new(camera),
        chrome: Cell::new(Chrome::Character),
        customize: Cell::new(program::Customize::default()),
        painted: Cell::new(None),
        hidden: Default::default(),
        stature: Cell::new(1.0),
        skeleton: Cell::new(false),
        overlay: placed::Placements::new(Vec::new()),
        grid: Cell::new(true),
        resident: Cell::new(0),
        debug: Cell::new(gpu::Debug::None),
        shaded: Cell::new(true),
        shade_failure: Default::default(),
        clock: Cell::new(0.0),
        target: Cell::new(gpu::LIT),
        export: Default::default(),
    })
}

fn level_of(pieces: &[Piece], lod: u8, attachments: usize) -> Result<Level> {
    let sources: Vec<_> = pieces
        .iter()
        .map(|piece| {
            (
                Worn {
                    path: piece.path.as_str(),
                    variant: piece.variant.get(),
                    material: piece.material,
                    deform: piece.deform.as_deref(),
                    skin: piece.skin,
                    rigid: piece.rigid,
                },
                &piece.container,
            )
        })
        .collect();
    read_level(&sources, lod, attachments)
}

/// Which detail levels the pieces draw anything at.
fn drawn_levels(pieces: &[Piece]) -> [bool; 3] {
    std::array::from_fn(|lod| {
        pieces.iter().any(|piece| {
            piece
                .container
                .model(detail(lod as u8))
                .meshes()
                .iter()
                .any(draws)
        })
    })
}

pub(super) fn detail(lod: u8) -> Lod {
    match lod {
        0 => Lod::High,
        1 => Lod::Medium,
        _ => Lod::Low,
    }
}

/// Whether this graph draws a mesh. Water fills the same G-buffer as anything else, through a
/// blended pass of its own, and the two overlays carry their own colour over the frame the lighting
/// left; the kinds left out are the engine's own passes, which nothing here runs.
pub(super) fn draws(mesh: &ironworks::file::mdl::Mesh) -> bool {
    mesh.kinds().iter().any(|kind| {
        matches!(
            kind,
            MeshKind::Standard
                | MeshKind::Water
                | MeshKind::LightShaft
                | MeshKind::VerticalFog
        )
    })
}

/// What a mesh a drawing pass leaves out is for.
fn kind_name(kind: MeshKind) -> &'static str {
    match kind {
        MeshKind::Water => "water",
        MeshKind::Shadow => "shadow",
        MeshKind::Terrain => "terrain shadow",
        MeshKind::VerticalFog => "vertical fog",
        MeshKind::LightShaft => "light shaft",
        MeshKind::Glass => "glass",
        MeshKind::MaterialChange => "material change",
        MeshKind::CrestChange => "crest change",
        MeshKind::Standard => "standard",
    }
}

/// The `.imc` this model's part draws with, derived from the model's own path rather than named
/// anywhere in the file: strip the `model/<name>.mdl` tail and the directory left names it.
///
/// A human ships none. Its body, face, hair and ears wear no variant, and asking for one is a
/// request for a file the game does not have.
pub(crate) fn imc_path(path: &str) -> Option<String> {
    if path.starts_with("chara/human/") {
        return None;
    }
    let base = &path[..path.rfind("/model/")?];
    let part = base.rsplit('/').next()?;
    // The off-hand half of a paired weapon ships none of its own and reads the main hand's.
    if let Some(tail) = base.strip_prefix("chara/weapon/w")
        && let Ok(set) = tail.get(..4)?.parse::<u32>()
    {
        let shared = material::shared_set(set);
        return Some(format!("chara/weapon/w{shared:04}{}/{part}.imc", &tail[4..]));
    }
    Some(format!("{base}/{part}.imc"))
}

/// Which of the imc's five parts this model's own slot reads: head or ears 0, body or neck 1,
/// hands or wrists 2, legs or right ring 3, feet or left ring 4, matching `imc.rs`'s own doc. A
/// monster or weapon has one part and no such suffix, so it falls back to 0, which is already
/// right for it.
fn imc_part(path: &str) -> u8 {
    let stem = path.rsplit('/').next().unwrap_or(path);
    let slot = stem
        .strip_suffix(".mdl")
        .unwrap_or(stem)
        .rsplit('_')
        .next()
        .unwrap_or("");
    match slot {
        "met" | "ear" => 0,
        "top" | "nek" => 1,
        "glv" | "wrs" => 2,
        "dwn" | "rir" => 3,
        "sho" | "ril" => 4,
        _ => 0,
    }
}

/// Whether an imc's variants pick between mutually exclusive geometry, which is what lets a first
/// look default past entry 0's own all-on catalog: variant 1's mask is set, at least one other
/// variant's is too, and no two share a bit. Masks are restricted to the model's own declared
/// attributes, since the format reserves ten bits regardless of how many exist; a lone alternative
/// is a toggle rather than a choice and does not count, which is what keeps this off ordinary
/// equipment.
fn exclusive_variants(image_change: &ImageChange, part: u8, declared: usize) -> bool {
    let cover: u32 = match declared {
        0 => return false,
        1..32 => (1u32 << declared) - 1,
        _ => u32::MAX,
    };
    let first = image_change
        .entry(part, 1)
        .map_or(0, |entry| u32::from(entry.attribute_mask()) & cover);
    if first == 0 {
        return false;
    }
    let mut seen = first;
    let mut count = 1;
    for variant in 2..=image_change.variant_count() {
        let Some(mask) = image_change
            .entry(part, variant)
            .map(|entry| u32::from(entry.attribute_mask()) & cover)
        else {
            continue;
        };
        if mask == 0 {
            continue;
        }
        if seen & mask != 0 {
            return false;
        }
        seen |= mask;
        count += 1;
    }
    count >= 2
}

/// What a piece contributes to a level beyond the file it was decoded from.
struct Worn<'a> {
    path: &'a str,
    variant: u16,
    material: Option<u16>,
    deform: Option<&'a Deform>,
    skin: Option<u16>,
    rigid: bool,
}

fn read_level(sources: &[(Worn<'_>, &ModelContainer)], lod: u8, attachments: usize) -> Result<Level> {
    let mut names: Vec<String> = Vec::new();
    let mut meshes = Vec::new();
    let mut unreadable = Vec::new();
    let mut pending = gpu::Pending::default();
    let mut low = Vec3::splat(f32::INFINITY);
    let mut high = Vec3::splat(f32::NEG_INFINITY);
    let mut bones: Vec<Vec<String>> = Vec::new();
    let mut shapes: Vec<Shape> = Vec::new();
    let mut declares: Vec<usize> = Vec::new();
    let mut skipped: Vec<MeshKind> = Vec::new();
    let mut unbound = 0usize;
    let mut skinned = false;
    let mut waving = false;

    for (piece, (worn, container)) in sources.iter().enumerate() {
        let model = container.model(detail(lod));
        waving |= model.waving();

        let attributes = model.attribute_names().unwrap_or_default();
        let bone_names = model.bone_names().unwrap_or_default();
        let declared = model.shapes();
        let mut rewrites: Vec<Rewrites> = declared.iter().map(|_| Vec::new()).collect();
        declares.push(attributes.len());

        for (index, mesh) in model.meshes().into_iter().enumerate() {
            if !draws(&mesh) {
                for kind in mesh.kinds() {
                    if !skipped.contains(kind) {
                        skipped.push(*kind);
                    }
                }
                continue;
            }
            let name = mesh.material().unwrap_or_default();
            // An imc entry naming material nought states that the slot draws no material at all:
            // every material its colourway files resolves to no path, and the game leaves a mesh
            // whose material never bound out of the frame.
            if worn.material == Some(0) && material::colourwayed(worn.path, &name) {
                unbound += 1;
                continue;
            }
            let built = match (mesh.attributes(), mesh.indices()) {
                (Ok(attributes), Ok(indices)) => {
                    skinned |= attributes.iter().any(|attribute| {
                        attribute.kind as u8 == VertexAttributeKind::BlendIndices as u8
                    });
                    build(&attributes, indices)
                }
                (Err(why), _) | (_, Err(why)) => Err(why.to_string()),
            };
            let (mut vertices, indices) = match built {
                Ok(built) => built,
                Err(why) => {
                    unreadable.push((index, why));
                    continue;
                }
            };

            // A rigid piece carries no bone table of its own to resolve against the shared rig:
            // one placeholder per bone it skins to is what gives it joints to be carried on, which
            // `Rendered::carried` fills in every frame. A weapon that skins to more than one, a
            // grimoire's pages or a bow's limbs, loses every vertex past the first without them.
            let slots: Vec<String> = mesh
                .bone_table()
                .iter()
                .map(|bone| {
                    bone_names
                        .get(usize::from(*bone))
                        .cloned()
                        .unwrap_or_default()
                })
                .collect();
            // A rigid piece resolves its own bone names against a rig of its own where it has one,
            // and against nothing at all where it does not, so an empty table still needs the one
            // slot every vertex of it indexes.
            let table = match worn.rigid && slots.is_empty() {
                true => vec![String::new()],
                false => slots,
            };
            if let Some(deform) = worn.deform {
                deform.apply(&mut vertices, &table);
            }

            for vertex in &vertices {
                let position = Vec3::from_array(vertex.position);
                low = low.min(position);
                high = high.max(position);
            }

            let resolved = material::path(
                worn.path,
                &name,
                worn.material.unwrap_or(worn.variant),
                worn.skin,
            )
            .unwrap_or(name);
            let material = names
                .iter()
                .position(|held| *held == resolved)
                .unwrap_or_else(|| {
                    names.push(resolved);
                    names.len() - 1
                });
            let submeshes = mesh.submeshes();
            let parts = match submeshes.is_empty() {
                true => vec![Part {
                    range: 0..indices.len(),
                    shown: Cell::new(true),
                    attributes: String::new(),
                    mask: 0,
                }],
                false => submeshes
                    .iter()
                    .map(|part| Part {
                        range: part.start..part.start + part.count,
                        shown: Cell::new(true),
                        attributes: named(&attributes, part.attributes),
                        mask: part.attributes,
                    })
                    .collect(),
            };
            for (shape, touched) in declared.iter().zip(&mut rewrites) {
                let values = shape.rewrites(&mesh);
                if !values.is_empty() {
                    touched.push((meshes.len(), values));
                }
            }
            bones.push(table);
            meshes.push(Mesh {
                piece,
                material,
                vertices: vertices.len(),
                triangles: indices.len() / 3,
                parts,
                base: match declared.is_empty() {
                    true => Vec::new(),
                    false => indices.clone(),
                },
            });
            pending.meshes.push((vertices, indices));
        }

        shapes.extend(
            declared
                .iter()
                .zip(rewrites)
                .filter(|(_, touched)| !touched.is_empty())
                .map(|(shape, touched)| Shape {
                    name: shape.name().unwrap_or_default(),
                    rewrites: touched,
                }),
        );
    }

    // A model whose every mesh carries a kind nothing here draws still has materials, a tree and a
    // browser worth opening, so the level comes back empty and names what it left out rather than
    // the read failing. A mesh that would not read at all is a different matter.
    if meshes.is_empty()
        && let Some((_, why)) = unreadable.first()
    {
        anyhow::bail!("no mesh of this model could be read: {why}");
    }
    if meshes.is_empty() {
        low = Vec3::NEG_ONE;
        high = Vec3::ONE;
    }

    let center = (low + high) * 0.5;
    let radius = ((high - low).length() * 0.5).max(0.01);
    let home = Camera {
        yaw: 0.0,
        pitch: 0.15,
        distance: radius / (FOV * 0.5).tan() * MARGIN,
        target: center,
    };

    let vertices: usize = meshes.iter().map(|mesh| mesh.vertices).sum();
    let triangles: usize = meshes.iter().map(|mesh| mesh.triangles).sum();
    let mut identity = vec![
        ("Meshes", meshes.len().to_string()),
        ("Vertices", vertices.to_string()),
        ("Triangles", triangles.to_string()),
        ("Materials", names.len().to_string()),
        (
            "Bounds",
            format!(
                "{:.2} x {:.2} x {:.2}",
                high.x - low.x,
                high.y - low.y,
                high.z - low.z
            ),
        ),
        (
            "Buffers",
            Bytes(vertices * size_of::<Vertex>() + triangles * 6).to_string(),
        ),
    ];
    let mut left_out: Vec<&str> = skipped.iter().map(|kind| kind_name(*kind)).collect();
    if unbound != 0 {
        left_out.push("no material");
    }
    if !left_out.is_empty() {
        identity.push(("Not drawn", left_out.join(", ")));
    }

    log::info!(
        "assets/mdl: {} {} meshes, {vertices} vertices, {} materials, {} unreadable",
        sources
            .iter()
            .map(|(worn, _)| crate::utils::file_name(worn.path))
            .collect::<Vec<_>>()
            .join(" + "),
        meshes.len(),
        names.len(),
        unreadable.len()
    );

    let gpu = gpu::Model::new(pending);
    if attachments != 0 {
        gpu.lock().unwrap().seed_attachments(attachments);
    }
    Ok(Level {
        identity,
        groups: group(&shapes),
        meshes,
        shapes,
        materials: names,
        unreadable,
        home,
        radius,
        skinned,
        waving,
        bones,
        attributes: declares,
        gpu,
    })
}

/// Interleaves the attributes a mesh declares into the one buffer the shader reads. A mesh missing
/// a normal, tangent, UV or color gets a default rather than being dropped.
pub(super) fn build(
    attributes: &[ironworks::file::mdl::VertexAttribute],
    indices: Vec<u16>,
) -> Result<(Vec<Vertex>, Vec<u16>), String> {
    let held = |kind: u8, usage: u8| {
        attributes
            .iter()
            .find(|attribute| attribute.kind as u8 == kind && attribute.usage_index == usage)
    };
    let positions = held(VertexAttributeKind::Position as u8, 0);
    let normals = held(VertexAttributeKind::Normal as u8, 0);
    let tangents = held(VertexAttributeKind::Tangent1 as u8, 0);
    let bitangents = held(VertexAttributeKind::Tangent2 as u8, 0);
    let uvs = held(VertexAttributeKind::Uv as u8, 0);
    let uvs1 = held(VertexAttributeKind::Uv as u8, 1);
    let colors = held(VertexAttributeKind::Color as u8, 0);
    let colors1 = held(VertexAttributeKind::Color as u8, 1);
    let weights = held(VertexAttributeKind::BlendWeights as u8, 0);
    let bones = held(VertexAttributeKind::BlendIndices as u8, 0);

    let Some(positions) = positions.map(|held| &held.values) else {
        return Err("mesh declares no vertex positions".into());
    };
    let count = match positions {
        VertexValues::Vector3(values) => values.len(),
        VertexValues::Vector4(values) => values.len(),
        _ => return Err("vertex positions are not a vector".into()),
    };
    if let Some(index) = indices.iter().find(|index| usize::from(**index) >= count) {
        return Err(format!(
            "index {index} names none of the mesh's {count} vertices"
        ));
    }

    let (normals, uvs, uvs1) = (values(normals), values(uvs), values(uvs1));
    let (colors, colors1) = (values(colors), values(colors1));
    let (weights, bones) = (values(weights), values(bones));
    // A byte tangent arrives scaled to nought and one, which is the convention the game's own
    // shaders unbias from, so a half or float one is put back into it rather than the other way.
    let frame = |held: Option<&ironworks::file::mdl::VertexAttribute>, at| {
        let held = held?;
        let value = xyzw(&held.values, at)?;
        Some(match held.format {
            VertexFormat::ByteFloat4 => value,
            _ => value.map(|channel| channel * 0.5 + 0.5),
        })
    };

    let vertices = (0..count)
        .map(|at| Vertex {
            position: xyz(positions, at).unwrap_or_default(),
            normal: normals
                .and_then(|held| xyz(held, at))
                .unwrap_or([0.0, 1.0, 0.0]),
            // A mesh with no frame gets a flat one, which unbiases to the surface normal rather
            // than to a basis nothing measured.
            tangent: frame(tangents, at).unwrap_or([0.5, 0.5, 1.0, 1.0]),
            bitangent: frame(bitangents, at).unwrap_or([0.5, 0.5, 1.0, 1.0]),
            uv: uvs.and_then(|held| uv(held, at)).unwrap_or_default(),
            uv1: uvs1.and_then(|held| uv(held, at)).unwrap_or_default(),
            color: colors.and_then(|held| bytes(held, at)).unwrap_or([255; 4]),
            // Nought, not white: the only thing that reads this is the sway, and a mesh with no
            // stream of its own is one the wind does not reach.
            color1: colors1.and_then(|held| bytes(held, at)).unwrap_or([0; 4]),
            weights: influences(weights, at, [255, 0, 0, 0]),
            bones: influences(bones, at, [0; 4]),
        })
        .collect();
    Ok((vertices, indices))
}

fn values(held: Option<&ironworks::file::mdl::VertexAttribute>) -> Option<&VertexValues> {
    Some(&held?.values)
}

fn xyz(values: &VertexValues, at: usize) -> Option<[f32; 3]> {
    match values {
        VertexValues::Vector3(held) => held.get(at).copied(),
        VertexValues::Vector4(held) => held.get(at).map(|value| [value[0], value[1], value[2]]),
        _ => None,
    }
}

fn xyzw(values: &VertexValues, at: usize) -> Option<[f32; 4]> {
    match values {
        VertexValues::Vector4(held) => held.get(at).copied(),
        _ => None,
    }
}

/// A half4 UV element carries two sets packed as `xy` and `zw`, so the whole element goes across
/// rather than only the first two components.
fn uv(values: &VertexValues, at: usize) -> Option<[f32; 4]> {
    match values {
        VertexValues::Vector2(held) => held.get(at).map(|value| [value[0], value[1], 0.0, 0.0]),
        VertexValues::Vector3(held) => held
            .get(at)
            .map(|value| [value[0], value[1], value[2], 0.0]),
        VertexValues::Vector4(held) => held.get(at).copied(),
        _ => None,
    }
}

/// One vertex's bone influences, in the sixteen bits a skinned shader reads each as: the low byte
/// of a pair is one of the first four influences and the high byte one of the second four.
fn influences(values: Option<&VertexValues>, at: usize, missing: [u16; 4]) -> [u16; 4] {
    match values {
        Some(VertexValues::Bytes8(held)) => match held.get(at) {
            Some(held) => {
                std::array::from_fn(|lane| u16::from_le_bytes([held[lane * 2], held[lane * 2 + 1]]))
            }
            None => missing,
        },
        held => held
            .and_then(|held| bytes(held, at))
            .map_or(missing, |held| held.map(u16::from)),
    }
}

/// Four bytes of an attribute the shader reads as bytes. An eight-byte element carries two sets
/// interleaved, the low half first, so its own four are every other one.
fn bytes(values: &VertexValues, at: usize) -> Option<[u8; 4]> {
    match values {
        VertexValues::Vector4(held) => held
            .get(at)
            .map(|value| value.map(|channel| (channel.clamp(0.0, 1.0) * 255.0) as u8)),
        VertexValues::Bytes8(held) => held
            .get(at)
            .map(|value| [value[0], value[2], value[4], value[6]]),
        VertexValues::Uint(held) => held.get(at).map(|value| value.to_le_bytes()),
        _ => None,
    }
}

/// Shapes gathered by category, in the order the file declares them. A name is read as
/// `shp_<category>_<variant>`; most carry no variant, and each of those stands alone.
fn group(shapes: &[Shape]) -> Vec<Group> {
    let mut groups: Vec<Group> = Vec::new();
    for (at, shape) in shapes.iter().enumerate() {
        let (category, variant) = match shape
            .name
            .strip_prefix("shp_")
            .and_then(|rest| rest.rsplit_once('_'))
        {
            Some((category, variant)) => (category.to_owned(), variant.to_owned()),
            None => (String::new(), shape.name.clone()),
        };
        match groups
            .iter_mut()
            .find(|group| !group.category.is_empty() && group.category == category)
        {
            Some(group) => group.variants.push((at, variant)),
            None => groups.push(Group {
                category,
                variants: vec![(at, variant)],
            }),
        }
    }
    groups
}

/// What the model's attribute table calls the bits a part sets. The mask is 32 bits wide however
/// many names the table holds.
fn named(attributes: &[String], mask: u32) -> String {
    attributes
        .iter()
        .take(32)
        .enumerate()
        .filter(|(bit, _)| mask & (1 << bit) != 0)
        .map(|(_, name)| name.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

/// The table the shading passes index, from the parameter files that have arrived. Nothing until one
/// has, since a table of nought is what the frame already stands in with.
fn types(parameters: &BTreeMap<usize, Parameters>) -> Option<Vec<u32>> {
    let held = parameters
        .iter()
        .filter_map(|(base, held)| match held {
            Parameters::Ready(file) => Some((*base, file)),
            _ => None,
        })
        .collect::<Vec<_>>();
    (!held.is_empty()).then(|| program::shader_types(&held))
}

/// One of the game's own textures as the card takes it. Mip nought alone: nothing tells a translated
/// shader how many levels a texture has, and the graph answers that with one.
pub(super) fn layered(bytes: &[u8], path: &str, filter: u32) -> Result<deferred::Layered> {
    use ironworks::file::tex::TextureKind;

    let texture = ironworks::file::tex::Texture::read(Cursor::new(bytes.to_vec()))?;
    let image = crate::utils::tex_loader::decode_stack(&texture, 0, path)?;
    let (width, height) = texture.mip_size(0);
    let layers = texture.layers(0);
    Ok(deferred::Layered {
        size: (width.into(), height.into()),
        layers: layers.into(),
        pixels: image.into_rgba8().into_raw(),
        filter,
        kind: match texture.kind() {
            TextureKind::D3 => program::Kind::Volume,
            TextureKind::Cube => program::Kind::Cube,
            _ if layers > 1 => program::Kind::Array,
            _ => program::Kind::Plane,
        },
    })
}

/// The parts still showing, as the fewest runs that cover them. A file lists a mesh's parts in
/// index order, so two neighbours that both draw are one call rather than two.
fn shown(parts: &[Part]) -> Vec<Range<i32>> {
    let mut runs: Vec<Range<i32>> = Vec::new();
    for part in parts.iter().filter(|part| part.shown.get()) {
        let run = part.range.start as i32..part.range.end as i32;
        match runs.last_mut() {
            Some(last) if last.end == run.start => last.end = run.end,
            _ => runs.push(run),
        }
    }
    runs
}

/// What the viewer is for: taking a file apart, standing a character up the way the game does, or
/// standing one in a frame this view does not draw.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Chrome {
    Asset,
    Character,
    Placed,
}

/// A model standing in a frame someone else draws: the card holding its geometry, what each of its
/// meshes draws with, and where in that frame it stands.
///
/// Everything the model owns that belongs to a frame - the camera, the G-buffer, the lighting and
/// every pass past it - is the host's here. What crosses is geometry, one joint palette per mesh,
/// and the two things a character puts into the scene constants beside every other surface.
pub struct Cast {
    pub gpu: Arc<Mutex<gpu::Model>>,
    pub surfaces: Vec<gpu::Surface>,
    /// One palette per mesh, in the model's own space, since a mesh's blend indices run over its
    /// own bone table.
    pub joints: Vec<Vec<Mat4>>,
    /// Where it stands, at the height it was built.
    pub model: Mat4,
    pub customize: program::Customize,
    /// How much of it is drawn, which its own dither clip tests each pixel against.
    pub opacity: f32,
}

/// What the debug row offers, in the order it offers it.
const VIEWS: [(gpu::Debug, &str); 9] = [
    (gpu::Debug::Normals, "Normals"),
    (gpu::Debug::Geometry, "Geometric"),
    (gpu::Debug::Tangents, "Tangents"),
    (gpu::Debug::Bitangents, "Bitangents"),
    (gpu::Debug::Handedness, "Handedness"),
    (gpu::Debug::Uv, "UVs"),
    (gpu::Debug::Color, "Vertex color"),
    (gpu::Debug::Alpha, "Vertex alpha"),
    (gpu::Debug::Meshes, "Meshes"),
];

pub fn ui(ui: &mut egui::Ui, model: &Rendered, backend: &Backend) {
    if let Some(why) = model.level.borrow().gpu.lock().unwrap().take_shader_failure() {
        log::error!("assets/mdl: game shaders: {why}");
        model.fail_shading(why);
    }
    ui.horizontal_wrapped(|ui| {
        // A character is shown as the game draws it, so the row that takes it apart is not offered:
        // the shaders are already on and there is nothing to switch them to.
        let inspecting = model.chrome.get() == Chrome::Asset;
        let shaded = model.shaded.get();
        if inspecting
            && ui
                .selectable_label(shaded, "Game shaders")
                .on_hover_text("Draw with the package the material names, into its own G-buffer")
                .clicked()
        {
            let now = !shaded;
            model.shaded.set(now);
            // A deliberate retry, so it gets its own fresh answer instead of the last one.
            if now {
                model.shade_failure.borrow_mut().take();
            }
        }
        match shaded {
            true if inspecting => {
                for (at, name) in model.channels() {
                    if ui
                        .selectable_label(model.target.get() == at, name)
                        .clicked()
                    {
                        model.target.set(at);
                    }
                }
            }
            false if inspecting => {
                let debug = model.debug.get();
                for (mode, label) in VIEWS {
                    if ui.selectable_label(debug == mode, label).clicked() {
                        model.debug.set(match debug == mode {
                            true => gpu::Debug::None,
                            false => mode,
                        });
                    }
                }
            }
            _ => {}
        }
        let level = model.level.borrow();
        if level.skinned {
            let skeleton = model.skeleton.get();
            if ui
                .selectable_label(skeleton, "Skeleton")
                .on_hover_text("Draw the rig it is posed on over it")
                .clicked()
            {
                model.skeleton.set(!skeleton);
            }
        }
        let grid = model.grid.get();
        if ui
            .selectable_label(grid, "Grid")
            .on_hover_text("Rule a floor at the origin, at the model's own scale")
            .clicked()
        {
            model.grid.set(!grid);
        }
        if shaded {
            let settings = model.settings.get();
            if ui
                .selectable_label(settings, "Graphics")
                .on_hover_text("What the passes past the composite are run with")
                .clicked()
            {
                model.settings.set(!settings);
            }
        }
        if ui.button("Reset view").clicked() {
            model.camera.set(level.home);
        }
        let (arrived, wanted) = model.arrived();
        if arrived < wanted {
            ui.add(egui::Spinner::new().size(14.0));
            ui.label(RichText::new(format!("{arrived}/{wanted}")).weak())
                .on_hover_text("Materials, shader packages and textures still on their way");
        }
        let ready = arrived >= wanted;
        let busy = model.export.borrow().is_some();
        let promise = crate::utils::export::menu(
            ui,
            "Export",
            None,
            busy,
            model.export_choices(backend, ready, wanted - arrived),
            egui::Vec2::ZERO,
        );
        if promise.is_some() {
            *model.export.borrow_mut() = promise;
        }
        if !level.unreadable.is_empty() {
            ui.label(
                RichText::new(format!("⚠ {} unreadable meshes", level.unreadable.len()))
                    .color(Color32::LIGHT_RED),
            )
            .on_hover_text(
                level
                    .unreadable
                    .iter()
                    .map(|(index, why)| format!("mesh {index}: {why}"))
                    .collect::<Vec<_>>()
                    .join("\n"),
            );
        }
        // Bound first, or the `Ref` from `borrow()` would still be held when the click handler
        // below reaches for `borrow_mut()`. The Character chrome hides the toggle that would
        // otherwise be the only way to clear this.
        let failure = model.shade_failure.borrow().clone();
        if let Some(why) = failure {
            ui.label(
                RichText::new("⚠ game shaders would not build, showing the plain pass")
                    .color(Color32::LIGHT_RED),
            )
            .on_hover_text(why.as_str());
            if ui.button("Retry").clicked() {
                model.shade_failure.borrow_mut().take();
                model.shaded.set(true);
            }
        }
    });

    if let Some(why) = model.level.borrow().gpu.lock().unwrap().failure() {
        ui.centered_and_justified(|ui| {
            ui.colored_label(Color32::RED, format!("Could not build the shader: {why}"));
        });
        return;
    }

    if model.shaded.get() && model.settings.get() {
        settings(ui, model);
    }

    if model.level.borrow().skinned {
        model.animation.ui(ui);
    }

    model.poll(ui.ctx(), backend);
    model.viewport(ui);
}

/// Which of the passes past the composite run, and what each is run with. Every constant here comes
/// out of a buffer no file describes, so each is stated beside its control rather than left to a
/// slider; the two lanes of the vignette are the exception and are picks.
fn settings(ui: &mut egui::Ui, model: &Rendered) {
    let mut look = model.look.get();
    ui.horizontal_wrapped(|ui| {
        ui.label("Textures").on_hover_text(
            "Which mipmap of a model's own textures is decoded. The file arrives whole whatever \
             this says, so only memory and decoding time follow it",
        );
        egui::ComboBox::from_id_salt("mdl-detail")
            .selected_text(label(look.detail))
            .show_ui(ui, |ui| {
                for (detail, what) in DETAIL {
                    ui.selectable_value(&mut look.detail, detail, what);
                }
            });
        ui.checkbox(&mut look.antialias, "Antialias")
            .on_hover_text("Smooth the frame's edges with the game's own FXAA");
        ui.add_enabled_ui(look.antialias, |ui| {
            ui.label(format!("Subpixel {}", program::FXAA_SUBPIX))
                .on_hover_text(
                    "How much of FXAA's own subpixel aliasing removal reaches the frame, off the \
                     game's own upload",
                );
            ui.label(format!("Edge {}", program::FXAA_EDGE))
                .on_hover_text(
                    "How much local contrast counts as an edge, likewise, over a floor the pass \
                     carries itself",
                );
        });
    });
    ui.horizontal_wrapped(|ui| {
        ui.checkbox(&mut look.bloom, "Bloom").on_hover_text(
            "Spread the bright end of the frame with the game's own glare chain. Nothing in it is \
             the viewer's to choose: the taps come out of the passes' own weights and the two \
             beside this off the frames the game drew",
        );
        ui.add_enabled_ui(look.bloom, |ui| {
            ui.label(format!("Threshold {:.5}", program::GLARE_THRESHOLD))
                .on_hover_text("What a pixel's glare has to average before any of it spreads");
            ui.label(format!("Veil {}", program::GLARE_VEIL))
                .on_hover_text("How dim a pixel has to be for the merge to pull it toward a grey");
        });
    });
    ui.horizontal_wrapped(|ui| {
        ui.checkbox(&mut look.vignette, "Vignette").on_hover_text(
            "Darken the frame's corners with the game's own pass. The ellipse it spreads over \
             follows the frame's own shape, but the two below are choices: no file states either",
        );
        ui.add_enabled_ui(look.vignette, |ui| {
            ui.add(egui::Slider::new(&mut look.onset, 0.0..=1.0).text("Onset"))
                .on_hover_text(
                    "How far out the darkening starts, as a squared distance with a corner at one",
                );
            ui.add(egui::Slider::new(&mut look.darkening, 0.0..=2.0).text("Darkening"))
                .on_hover_text("How steeply it deepens past that");
        });
    });
    ui.horizontal_wrapped(|ui| {
        ui.checkbox(&mut look.reflect, "Reflection").on_hover_text(
            "Reflect the frame off itself with the game's own chain, which is what a metal \
             surface answers with where nothing captured an environment for it",
        );
        ui.add_enabled_ui(look.reflect, |ui| {
            ui.add(
                Label::new(
                    RichText::new(format!(
                        "{}-{} units  x{}  rough<{}  {} levels",
                        program::REFLECTION_FADE[0],
                        program::REFLECTION_FADE[1],
                        program::REFLECTION_POWER,
                        program::REFLECTION_ROUGHNESS,
                        program::REFLECTION_LEVELS,
                    ))
                    .weak(),
                )
                .wrap(),
            )
            .on_hover_text(
                "What the chain runs with, read whole off a frame the game drew: how far a \
                 reflection reaches and where it starts fading, what a pixel's reflectance is \
                 scaled by, how rough a surface may be and still be marched, and how many levels \
                 the blur takes the answer down",
            );
        });
    });
    ui.horizontal_wrapped(|ui| {
        ui.checkbox(&mut look.occlude, "Occlusion")
            .on_hover_text("Shade the creases with the game's own HDAO");
        ui.add_enabled_ui(look.occlude, |ui| {
            egui::ComboBox::from_id_salt("mdl-occluder")
                .selected_text(program::OCCLUDERS[look.quality])
                .show_ui(ui, |ui| {
                    for (at, what) in program::OCCLUDERS.iter().enumerate() {
                        ui.selectable_value(&mut look.quality, at, *what);
                    }
                });
            ui.add(
                Label::new(
                    RichText::new(format!(
                        "{} texels  accept {}  reject {}  {}-{} units  bias {}  x{}  ^{}",
                        program::OCCLUSION_SPREAD,
                        program::OCCLUSION_ACCEPT,
                        program::OCCLUSION_REJECT,
                        program::OCCLUSION_NEAR,
                        program::OCCLUSION_REACH,
                        program::OCCLUSION_BIAS,
                        program::OCCLUSION_INTENSITY,
                        program::OCCLUSION_POWER,
                    ))
                    .weak(),
                )
                .wrap(),
            )
            .on_hover_text(
                "What the pass runs with, read whole off the game's own upload: how far the taps \
                 spread, how steeply a valley has to fall to count against the depth it stands at, \
                 the fall past which two samples are no longer one surface, the distances under \
                 and past which a pixel is left alone, how far a sample is pushed along its own \
                 normal, what the taps add up to is scaled by, and the exponent it is raised to",
            );
        });
    });
    let held = model.look.get();
    if look != held {
        // Which mipmap is taken is settled when a texture is fetched, so a change means fetching
        // and decoding every one of them again. Dropping the handles is what frees what they held.
        if look.detail != held.detail {
            model.textures.borrow_mut().clear();
            model.resident.set(0);
        }
        model.look.set(look);
    }
}

/// What the ladder calls one of its rungs, or the number itself where it names none.
fn label(detail: Option<u16>) -> String {
    DETAIL
        .iter()
        .find(|(held, _)| *held == detail)
        .map_or_else(|| format!("{detail:?}"), |(_, what)| (*what).to_owned())
}

impl Rendered {
    /// Asks for whatever the model still needs, and hands what arrived to egui. Runs every frame;
    /// a slot that is already resolved costs a lookup.
    /// The textures a material names for a sampler its package declares over slices rather than a
    /// plane: an environment cube, an array, a volume.
    fn sliced(&self, slots: &[Option<Slot>]) -> BTreeSet<String> {
        let translated = self.translated.borrow();
        slots
            .iter()
            .enumerate()
            .filter_map(|(at, slot)| match slot {
                Some(Slot::Ready(material)) => {
                    Some((translated.get(&at)?.held.as_ref().ok()?, material))
                }
                _ => None,
            })
            .flat_map(|(passes, material)| {
                material.bound().filter(|(id, _)| {
                    passes
                        .buffer
                        .iter()
                        .chain(&passes.depth)
                        .chain(&passes.resolve)
                        .chain(passes.sheer.iter().flat_map(|(g, over)| [g, over]))
                        .flat_map(|pass| &pass.textures)
                        .any(|texture| texture.id == *id && texture.kind != program::Kind::Plane)
                })
            })
            .map(|(_, path)| path.to_owned())
            .collect()
    }

    /// What a host outside this view draws this model with, at the transform it stands at and for
    /// as much of a G-buffer as that host's own frame writes.
    ///
    /// A motion is set the way the character tab sets one, through [`Self::act`] and [`Self::play`]:
    /// nothing about this seam names a pose, so what it stands in is whatever was last asked for.
    pub fn cast(&self, at: Mat4, attachments: usize) -> Cast {
        let level = self.level.borrow();
        self.translate(level.skinned, level.waving, attachments);
        let (pose, _) = self.posed(&level);
        Cast {
            gpu: level.gpu.clone(),
            surfaces: self.surfaces(&level),
            joints: pose.joints,
            model: at * Mat4::from_scale(Vec3::splat(self.stature.get())),
            customize: self.made_up(),
            opacity: 1.0,
        }
    }

    /// Asks for whatever this model still needs. Called by the view drawing it, which is not always
    /// this one.
    pub fn poll(&self, ctx: &egui::Context, backend: &Backend) {
        self.export
            .borrow_mut()
            .take_if(|promise| promise.try_get().is_some());

        let level = self.level.borrow();
        if level.skinned {
            self.animation.poll(ctx, backend);
            self.emote
                .borrow_mut()
                .poll(backend, self.animation.body_playing());
            self.effects
                .borrow_mut()
                .poll(ctx, backend, &self.fired.borrow());
            self.wall.set(ctx.input(|input| input.time));
            let worn: Vec<(u16, u16)> = self
                .attachments
                .borrow()
                .iter()
                .filter_map(|held| wield::worn(&held.path))
                .collect();
            self.wield
                .borrow_mut()
                .poll(backend, &worn, self.wielded.get(), self.wall.get());
        }
        let mut slots = self.slots.borrow_mut();
        for (index, slot) in slots.iter_mut().enumerate() {
            let path = &level.materials[index];
            match slot {
                None => {
                    let files = backend.files().clone();
                    let wanted = path.clone();
                    *slot = Some(Slot::Fetching(TrackedPromise::spawn_local(async move {
                        files.read(&wanted).await
                    })));
                }
                Some(Slot::Fetching(promise)) => {
                    let Some(result) = promise.try_get() else {
                        continue;
                    };
                    *slot = Some(match result {
                        Ok(bytes) => match Material::parse(bytes) {
                            Ok(material) => Slot::Ready(Box::new(material)),
                            Err(why) => Slot::Failed(why.to_string()),
                        },
                        Err(why) => {
                            log::error!("assets/mdl: {path}: {why}");
                            Slot::Failed(why.to_string())
                        }
                    });
                    if let Some(Slot::Ready(material)) = slot
                        && let Some(table) = material.table()
                    {
                        level.gpu.lock().unwrap().queue_table(index, table.to_vec());
                    }
                }
                Some(_) => {}
            }
        }

        let mut landed = false;
        for (at, piece) in self.pieces.iter().enumerate() {
            let mut imc = piece.imc.borrow_mut();
            match &mut *imc {
                None => {
                    *imc = Some(match imc_path(&piece.path) {
                        Some(path) => {
                            let files = backend.files().clone();
                            Imc::Fetching(TrackedPromise::spawn_local(async move {
                                files.read(&path).await
                            }))
                        }
                        None => Imc::Absent,
                    });
                }
                Some(Imc::Fetching(promise)) => {
                    if let Some(result) = promise.try_get() {
                        let read = result
                            .as_ref()
                            .map_err(ToString::to_string)
                            .and_then(|bytes| {
                                ImageChange::read(Cursor::new(bytes.clone()))
                                    .map_err(|why| why.to_string())
                            });
                        *imc = Some(match read {
                            Ok(image_change) => {
                                if exclusive_variants(
                                    &image_change,
                                    imc_part(&piece.path),
                                    level.attributes[at],
                                ) {
                                    piece.variant.set(1);
                                }
                                Imc::Ready(image_change)
                            }
                            Err(why) => {
                                log::warn!("assets/mdl: {}: {why}", piece.path);
                                Imc::Absent
                            }
                        });
                        landed = true;
                    }
                }
                Some(_) => {}
            }
        }
        if landed {
            self.apply_variant();
        }

        let mut packages = self.packages.borrow_mut();
        if self.shaded.get() {
            // The packages that light the frame belong to no material, so they are asked for
            // alongside the ones the materials name.
            let wanted = slots
                .iter()
                .flatten()
                .filter_map(|slot| match slot {
                    Slot::Ready(material) => Some(material.package()),
                    _ => None,
                })
                // The passes that light a frame and grade it belong to whoever draws the frame,
                // which for a placed model is not this one.
                .chain(
                    match self.chrome.get() {
                        Chrome::Placed => [].as_slice(),
                        _ => [
                            program::VIEW_POSITION,
                            program::DIRECTIONAL,
                            program::POINT,
                            program::COMPOSITE,
                            program::TONE_ADJUST,
                        ]
                        .as_slice(),
                    }
                    .iter()
                    .copied()
                    .map(str::to_owned),
                )
                // Asked for only where the viewer is drawing with them, so a frame nobody wants
                // smoothed costs no fetch at all.
                .chain(
                    self.look
                        .get()
                        .antialias
                        .then_some([program::FXAA_LUMA, program::FXAA])
                        .into_iter()
                        .flatten()
                        .map(str::to_owned),
                )
                .chain(
                    self.look
                        .get()
                        .bloom
                        .then_some(program::GLARE)
                        .into_iter()
                        .flatten()
                        .map(str::to_owned),
                )
                .chain(
                    self.look
                        .get()
                        .vignette
                        .then_some(program::VIGNETTE)
                        .map(str::to_owned),
                )
                .chain(
                    self.look
                        .get()
                        .reflect
                        .then_some(program::REFLECTION)
                        .into_iter()
                        .flatten()
                        .map(str::to_owned),
                )
                // Of the eight readings the quality ladder offers, only the one it is set to.
                .chain(
                    self.look
                        .get()
                        .occlude
                        .then(|| {
                            [
                                program::DOWN_SCALE.to_owned(),
                                program::GATHER.to_owned(),
                                self.look.get().occluder(),
                            ]
                        })
                        .into_iter()
                        .flatten(),
                );
            for path in wanted {
                if packages.contains_key(&path) {
                    continue;
                }
                let files = backend.files().clone();
                let wanted = path.clone();
                packages.insert(
                    path,
                    Package::Fetching(TrackedPromise::spawn_local(async move {
                        match program::unnamed(&wanted) {
                            Some(hash) => {
                                files
                                    .read_by_hash(program::SHADER.0, program::SHADER.1, hash, true)
                                    .await
                            }
                            None => files.read(&wanted).await,
                        }
                    })),
                );
            }
        }
        // Drained whether or not shading is still on: a package still in flight when a build
        // failure elsewhere turned shading off on its own would otherwise never be consumed, and
        // `arrived` would hold short of `wanted` forever.
        for (path, package) in packages.iter_mut() {
            let Package::Fetching(promise) = package else {
                continue;
            };
            let Some(result) = promise.try_get() else {
                continue;
            };
            *package = match result {
                Ok(bytes) => Package::Ready(bytes.clone()),
                Err(why) => {
                    log::error!("assets/mdl: {path}: {why}");
                    Package::Failed(why.to_string())
                }
            };
        }

        let mut arrays = self.arrays.borrow_mut();
        // Picking another face paint drops what was fetched for the last one, since the two are
        // the one binding and the fetch below is what fills it.
        let paint = self.customize.get().paint;
        if self.painted.replace(paint) != paint {
            arrays.remove(&deferred::FACE_PAINT);
        }
        let held = deferred::ENGINE
            .into_iter()
            .chain([deferred::GRADING])
            .map(|(id, path, filter)| (id, path.to_owned(), filter))
            .chain(paint.map(|set| {
                (
                    deferred::FACE_PAINT,
                    format!("{}{set}.tex", deferred::PAINTS),
                    glow::LINEAR,
                )
            }));
        for (id, path, filter) in held {
            let held = match arrays.entry(id) {
                std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::btree_map::Entry::Vacant(_) if !self.shaded.get() => continue,
                std::collections::btree_map::Entry::Vacant(entry) => {
                    let files = backend.files().clone();
                    let path = path.clone();
                    entry.insert(Array::Fetching(TrackedPromise::spawn_local(async move {
                        files.read(&path).await
                    })))
                }
            };
            let Array::Fetching(promise) = held else {
                continue;
            };
            let Some(result) = promise.try_get() else {
                continue;
            };
            *held = match result
                .as_ref()
                .map_err(ToString::to_string)
                .and_then(|bytes| layered(bytes, &path, filter).map_err(|why| why.to_string()))
            {
                Ok(decoded) => {
                    level.gpu.lock().unwrap().queue_array(id, decoded.clone());
                    self.graded
                        .set(self.graded.get() || id == deferred::GRADING.0);
                    Array::Ready(decoded)
                }
                Err(why) => {
                    log::error!("assets/mdl: {path}: {why}");
                    Array::Failed
                }
            };
        }
        // The one texture the game states no path for. Held in the same map as the fetched set so
        // that dropping that map drops this too, and drawn once: the field is deterministic.
        if let std::collections::btree_map::Entry::Vacant(entry) =
            arrays.entry(deferred::PERLIN_2D)
            && self.shaded.get()
        {
            let held = deferred::Layered {
                size: (noise::SIZE as i32, noise::SIZE as i32),
                layers: 1,
                pixels: noise::perlin(),
                filter: glow::LINEAR,
                kind: program::Kind::Plane,
            };
            level.gpu.lock().unwrap().queue_array(deferred::PERLIN_2D, held.clone());
            entry.insert(Array::Ready(held));
        }

        let mut parameters = self.parameters.borrow_mut();
        let mut arrived = false;
        for (base, path) in program::PARAMETERS {
            let held = match parameters.entry(base) {
                std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::btree_map::Entry::Vacant(_) if !self.shaded.get() => continue,
                std::collections::btree_map::Entry::Vacant(entry) => {
                    let files = backend.files().clone();
                    entry.insert(Parameters::Fetching(TrackedPromise::spawn_local(
                        async move { files.read(path).await },
                    )))
                }
            };
            let Parameters::Fetching(promise) = held else {
                continue;
            };
            let Some(result) = promise.try_get() else {
                continue;
            };
            *held = match result
                .as_ref()
                .map_err(ToString::to_string)
                .and_then(|bytes| {
                    ShaderParameters::read(Cursor::new(bytes.clone()))
                        .map_err(|why| why.to_string())
                }) {
                Ok(file) => Parameters::Ready(file),
                Err(why) => {
                    log::error!("assets/mdl: {path}: {why}");
                    Parameters::Failed
                }
            };
            arrived = true;
        }
        if arrived && let Some(values) = types(&parameters) {
            level.gpu.lock().unwrap().queue_types(values);
        }

        // The fur pass belongs to no material either, and nothing can be softened with it until
        // the frame is lit at all, so it is only worth a fetch of its own once the four above
        // are in hand and the model turns out to state a fur length.
        if self.shaded.get()
            && self.lighting.borrow().is_some()
            && !packages.contains_key(program::FUR)
            && let Some(values) = types(&parameters)
            && slots.iter().flatten().any(|slot| match slot {
                Slot::Ready(material) => program::furred(material, &values),
                _ => false,
            })
        {
            let files = backend.files().clone();
            packages.insert(
                program::FUR.to_owned(),
                Package::Fetching(TrackedPromise::spawn_local(async move {
                    files.read(program::FUR).await
                })),
            );
        }
        // Skin softens the light that fell on it, and every character has some, so this is
        // asked for as soon as the frame can be lit at all rather than off a material's own
        // record: the pass decides per pixel from the type table which ones scatter.
        if self.shaded.get()
            && self.lighting.borrow().is_some()
            && !packages.contains_key(program::SCATTER)
        {
            let files = backend.files().clone();
            packages.insert(
                program::SCATTER.to_owned(),
                Package::Fetching(TrackedPromise::spawn_local(async move {
                    files.read(program::SCATTER).await
                })),
            );
        }

        let sliced = self.sliced(&slots);
        let mut stacks = self.stacks.borrow_mut();
        for path in &sliced {
            let held = stacks.entry(path.as_str().into()).or_insert_with(|| {
                let files = backend.files().clone();
                let wanted = path.clone();
                Array::Fetching(TrackedPromise::spawn_local(async move {
                    files.read(&wanted).await
                }))
            });
            let Array::Fetching(promise) = held else {
                continue;
            };
            let Some(result) = promise.try_get() else {
                continue;
            };
            *held = match result
                .as_ref()
                .map_err(ToString::to_string)
                .and_then(|bytes| layered(bytes, path, glow::LINEAR).map_err(|why| why.to_string()))
            {
                Ok(decoded) => {
                    level
                        .gpu
                        .lock()
                        .unwrap()
                        .queue_stack(path.as_str().into(), decoded.clone());
                    Array::Ready(decoded)
                }
                Err(why) => {
                    log::error!("assets/mdl: {path}: {why}");
                    Array::Failed
                }
            };
        }
        drop(stacks);

        // A color table row index, not a color, so filtering it linearly would blend two rows'
        // indices into a third one that names neither. Point sampled and never mipmapped, the
        // same reasoning `deferred::ENGINE` gives for the subsurface kernel. Always a plane, so
        // this is the one place it is read: an array kind would go through `stacks` above instead.
        let index_paths: BTreeSet<&str> = slots
            .iter()
            .flatten()
            .filter_map(|slot| match slot {
                Slot::Ready(material) => material.texture(Role::Index).map(String::as_str),
                _ => None,
            })
            .collect();
        // Hair's cutout sampler states mirrored addressing, not the repeat this viewer otherwise
        // assumes; wrong past the last texel it is what turns a filtered strand edge into a bleed
        // of the texture's opposite side rather than a mirrored continuation of its own.
        let wrap_paths: BTreeMap<&str, mtrl::AddressMode> = slots
            .iter()
            .flatten()
            .filter_map(|slot| match slot {
                Slot::Ready(material) => Some(material),
                _ => None,
            })
            .flat_map(|material| {
                material
                    .bound()
                    .map(move |(id, path)| (path, material.wrap(id)))
            })
            .collect();
        let mut textures = self.textures.borrow_mut();
        let detail = self.look.get().detail;
        for slot in slots.iter().flatten() {
            let Slot::Ready(material) = slot else {
                continue;
            };
            let held: Vec<&String> = material.textures().collect();
            let bound: Vec<String> = match self.shaded.get() {
                true => material.bound().map(|(_, path)| path.to_owned()).collect(),
                false => Vec::new(),
            };
            for path in held.into_iter().chain(bound.iter()) {
                if textures.contains_key(path) || sliced.contains(path) {
                    continue;
                }
                if self.resident.get() >= TEXTURE_BUDGET {
                    log::warn!("assets/mdl: {path}: past this model's texture budget");
                    textures.insert(path.clone(), Texture::Absent);
                    continue;
                }
                let files = backend.files().clone();
                let wanted = path.clone();
                textures.insert(
                    path.clone(),
                    Texture::Fetching(TrackedPromise::spawn_local(async move {
                        files.read_texture(&wanted, detail).await
                    })),
                );
            }
        }
        for (path, texture) in textures.iter_mut() {
            let Texture::Fetching(promise) = texture else {
                continue;
            };
            let Some(result) = promise.try_get() else {
                continue;
            };
            *texture = match result {
                Ok(decoded) => {
                    let held = crate::utils::tex_loader::fit(ctx, &decoded.image);
                    let size = [held.width() as usize, held.height() as usize];
                    self.resident
                        .set(self.resident.get() + size[0] * size[1] * 4);
                    // Taken as premultiplied, which is the one path that copies the bytes through
                    // untouched. These are looked up channel by channel rather than composited, and
                    // a normal or mask map carrying anything but opacity in its alpha has its other
                    // three channels scaled away by the unmultiplied path.
                    let image = egui::ColorImage::from_rgba_premultiplied(
                        size,
                        held.as_flat_samples().as_slice(),
                    );
                    // Model UVs tile, and a texture bound to a surface is minified far more often
                    // than it is magnified, so this is the one place the browser wants mipmaps and
                    // repeat rather than the crisp clamped sampling a texture preview wants. A
                    // material's own sampler can ask for mirrored addressing instead; egui has a
                    // variant for that too.
                    //
                    // Except a color table row index: a mip is `glGenerateMipmap`'s box filter
                    // over indices, which is not itself an index, so this one goes unmipmapped
                    // rather than blended within or across levels.
                    let options = match index_paths.contains(path.as_str()) {
                        true => TextureOptions {
                            magnification: egui::TextureFilter::Nearest,
                            minification: egui::TextureFilter::Nearest,
                            wrap_mode: egui::TextureWrapMode::Repeat,
                            mipmap_mode: None,
                        },
                        false => TextureOptions {
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
                    };
                    Texture::Ready(ctx.load_texture(format!("mdl:{path}"), image, options))
                }
                Err(why) => {
                    log::error!("assets/mdl: {path}: {why}");
                    Texture::Absent
                }
            };
        }
    }

    /// The model itself: an orbit camera over a paint callback.
    /// The pose the model stands in, in its own space, and the rig it was posed on.
    ///
    /// The joints move the geometry after the file stated its bounds, so where the model stands has
    /// to be worked out before anything is framed or clipped against it.
    fn posed(&self, level: &Level) -> (skin::Pose, Option<skin::RigInfo>) {
        let worn: Vec<&str> = level
            .meshes
            .iter()
            .map(|mesh| self.pieces[mesh.piece].path.as_str())
            .collect();
        let mut pose = self.animation.pose(&level.bones, &worn, self.skeleton.get());
        let rig = self.animation.rig();
        // A carried piece has no bone table of its own to resolve against the shared rig, so
        // `pose` names it by an unresolvable placeholder and leaves it at `Mat4::IDENTITY`; this
        // overwrites that with the bone it actually hangs from, carried the way a rider is.
        if !self.attachments.borrow().is_empty()
            && let Some((names, ..)) = &rig
        {
            let attachments = self.attachments.borrow();
            for (at, attachment) in attachments.iter().enumerate() {
                let Some(bone) = names.iter().position(|name| *name == attachment.bone) else {
                    continue;
                };
                let Some(&world) = pose.world.get(bone) else {
                    continue;
                };
                let carried = world * attachment.local;
                // A motion that summons the same model twice wears it twice, so the nth carried
                // piece of a path is the nth piece built from it rather than every one of them.
                let which = attachments[..at]
                    .iter()
                    .filter(|held| held.path == attachment.path)
                    .count();
                let Some(worn) = self
                    .pieces
                    .iter()
                    .enumerate()
                    .filter(|(_, piece)| piece.path == attachment.path)
                    .nth(which)
                    .map(|(piece, _)| piece)
                else {
                    continue;
                };
                for (index, mesh) in level.meshes.iter().enumerate() {
                    if mesh.piece != worn {
                        continue;
                    }
                    // A prop that ships a pack of its own is skinned to a rig of its own, walked
                    // by that pack and carried whole to the point it hangs from: that is what puts
                    // one of the two things it holds in each hand.
                    // A prop moves out of the emote's own timeline; a weapon moves out of the pack
                    // its set ships, off the stance rather than off any motion the body plays.
                    let moved = self
                        .animation
                        .body_playing()
                        .and_then(|(_, _, time)| {
                            self.emote
                                .borrow()
                                .joints(&attachment.path, &level.bones[index], time)
                        })
                        .or_else(|| {
                            let (set, base) = wield::worn(&attachment.path)?;
                            self.wield.borrow().joints(
                                set,
                                base,
                                &level.bones[index],
                                self.wall.get(),
                            )
                        });
                    pose.joints[index] = match moved {
                        Some(joints) => joints.iter().map(|joint| carried * *joint).collect(),
                        None => vec![carried; level.bones[index].len()],
                    };
                }
            }
        }
        (pose, rig)
    }

    /// What each mesh of the level is drawn with, once its material and that material's own shaders
    /// have arrived.
    fn surfaces(&self, level: &Level) -> Vec<gpu::Surface> {
        let translated = self.translated.borrow();
        let tables = self.tables.borrow();
        let slots = self.slots.borrow();
        let textures = self.textures.borrow();
        let stacks = self.stacks.borrow();
        let bind = |path: &str| match textures.get(path) {
            Some(Texture::Ready(handle)) => Some(handle.id()),
            _ => None,
        };
        // The graph's own store first: a sliced texture reaches egui as a plane on the frame before
        // its package is translated, and answering with that one would pin the sampler to it.
        let sampled = |path: &str, aniso: f32| match stacks.get_key_value(path) {
            Some((held, Array::Ready(_))) => Some(gpu::Bound::Stacked(held.clone())),
            _ => bind(path).map(|handle| gpu::Bound::Plane(handle, aniso)),
        };
        // One that has not answered yet, as against one that answered with nothing. The flat
        // stand-in a draw reaches for meanwhile is opaque, so a cutout authored into a normal map's
        // alpha clips nothing and the quad it was cut out of stands as a solid card.
        let pending = |path: &str| {
            !matches!(
                textures.get(path),
                Some(Texture::Ready(_) | Texture::Absent)
            ) && !matches!(stacks.get(path), Some(Array::Ready(_) | Array::Failed))
        };
        level
            .meshes
            .iter()
            .map(|mesh| {
                let runs = shown(&mesh.parts);
                let Some(Some(Slot::Ready(material))) = slots.get(mesh.material) else {
                    return gpu::Surface {
                        material: mesh.material,
                        runs,
                        ..Default::default()
                    };
                };
                if !material.drawn() {
                    return gpu::Surface {
                        material: mesh.material,
                        ..Default::default()
                    };
                }
                let shaded = self.shaded.get().then(|| {
                    let passes = translated.get(&mesh.material)?.held.as_ref().ok()?;
                    if material.bound().any(|(_, path)| pending(path)) {
                        return None;
                    }
                    Some(gpu::Shaded {
                        buffer: passes.buffer.clone(),
                        depth: passes.depth.clone(),
                        // The model viewer lights one object against nothing, so it casts no shadow.
                        shadow: None,
                        resolve: passes.resolve.clone(),
                        sheer: passes.sheer.clone(),
                        table: tables
                            .get(&mesh.material)
                            .map(|base| self.dyed_table(mesh, material, base)),
                        textures: material
                            .bound()
                            .map(|(id, path)| (id, sampled(path, material.anisotropic(id))))
                            .collect(),
                    })
                });
                gpu::Surface {
                    material: mesh.material,
                    shaded: shaded.flatten(),
                    runs,
                    family: material.family(),
                    normal: material.texture(Role::Normal).and_then(|path| bind(path)),
                    index: material.texture(Role::Index).and_then(|path| bind(path)),
                    mask: material.texture(Role::Mask).and_then(|path| bind(path)),
                    diffuse: material.texture(Role::Diffuse).and_then(|path| bind(path)),
                    alpha_threshold: material.alpha_threshold(),
                    diffuse_color: material.diffuse(),
                    emissive_color: material.emissive(),
                    normal_scale: material.normal_scale(),
                    cull: material.cull(),
                }
            })
            .collect()
    }

    fn viewport(&self, ui: &mut egui::Ui) {
        let (rect, response) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        if rect.width() < 1.0 || rect.height() < 1.0 {
            return;
        }
        // The game's own shaders move a surface against a clock, so a frame under them is never the
        // same twice and the viewer has to keep asking for another.
        if self.shaded.get() {
            let step = ui.input(|input| input.stable_dt).min(0.25);
            self.clock.set(self.clock.get() + step);
            ui.ctx().request_repaint();
        }

        let level = self.level.borrow();
        let mut camera = self.camera.get();
        let pan = |camera: &mut Camera, delta: egui::Vec2| {
            let (sin_yaw, cos_yaw) = camera.yaw.sin_cos();
            let right = Vec3::new(cos_yaw, 0.0, -sin_yaw);
            let scale = camera.distance * 0.002;
            camera.target += (right * -delta.x + Vec3::Y * delta.y) * scale;
        };
        let zoom = |camera: &mut Camera, scale: f32| {
            camera.distance = (camera.distance * scale)
                .clamp(level.home.distance * 0.02, level.home.distance * 20.0);
        };

        // A second finger takes the gesture over: egui carries on reporting a primary drag through
        // one, so leaving the orbit armed would spin the model while it is being pinched.
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

        let (mut pose, rig) = self.posed(&level);
        // A marker standing in for the effect a weapon carries while it is drawn: this view runs
        // no particles for one, only where and when it would draw.
        let mut markers = std::mem::take(&mut pose.skeleton);
        if let Some((names, ..)) = &rig {
            for (_, bone) in self.glowing.borrow().iter() {
                let Some(&world) = names
                    .iter()
                    .position(|name| name == bone)
                    .and_then(|bone| pose.world.get(bone))
                else {
                    continue;
                };
                let (_, rotation, translation) = world.to_scale_rotation_translation();
                markers.push(placed::Batch {
                    shape: placed::Shape::Box,
                    instances: vec![placed::Instance {
                        center: translation.to_array(),
                        scale: [0.06; 3],
                        turn: rotation.to_array(),
                        color: WEAPON_VFX_COLOR,
                    }],
                });
            }
        }
        // Where each vfx the emote's own timeline is running stands this frame, which the next
        // poll steps and the callback below draws.
        let mut firing: Vec<effects::Fired> = match (self.animation.body_playing(), &rig) {
            (Some((_, _, time)), Some((names, ..))) => self
                .emote
                .borrow()
                .firing(time)
                .flat_map(|vfx| {
                    let place = |bone: &str| {
                        let at = names.iter().position(|name| *name == bone)?;
                        Some(*pose.world.get(at)? * vfx.local)
                    };
                    // A file states its own bind points and the client hangs one instance off
                    // each. Only the ids a capture has pinned are answered, and a bone this rig
                    // cannot name answers nothing, so either way what is left is where the command
                    // bound it rather than nothing at all.
                    let bound: Vec<Mat4> = self
                        .effects
                        .borrow()
                        .bound(vfx.path)
                        .iter()
                        .filter_map(|bone| place(bone))
                        .collect();
                    let placed = match bound.is_empty() {
                        true => place(vfx.bone).into_iter().collect(),
                        false => bound,
                    };
                    placed
                        .into_iter()
                        .enumerate()
                        .map(|(at, world)| effects::Fired {
                            id: vfx.id | (at as u64) << 16,
                            path: vfx.path.to_owned(),
                            at: world,
                            since: vfx.since,
                            tint: Vec4::from(vfx.tint),
                        })
                        .collect::<Vec<_>>()
                })
                .collect(),
            _ => Vec::new(),
        };
        // A drawn weapon's own effect, which runs for as long as it is drawn rather than off a
        // timeline of its own: the clock starts over whenever the set of them changes, so putting a
        // weapon away and taking it out again plays it from the beginning.
        if let Some((names, ..)) = &rig {
            let now = ui.input(|input| input.time);
            let start = match self.glowing_at.get() {
                Some(held) => held,
                None => {
                    self.glowing_at.set(Some(now));
                    now
                }
            };
            for (at, (path, bone)) in self.glowing.borrow().iter().enumerate() {
                let Some(&world) = names
                    .iter()
                    .position(|name| name == bone)
                    .and_then(|bone| pose.world.get(bone))
                else {
                    continue;
                };
                firing.push(effects::Fired {
                    // Past anything an emote's own timeline can number, so the two never share a
                    // running state.
                    id: 1 << 48 | at as u64,
                    path: path.clone(),
                    at: world,
                    since: (now - start) as f32,
                    tint: Vec4::ONE,
                });
            }
        }
        *self.fired.borrow_mut() = firing;
        // Carried rather than written into the camera, so a motion that walks runs in place and the
        // user's own orbit, pan and zoom still mean what they did.
        let focus = level.home.target + pose.drift;
        let reach = level.radius + pose.stretch;

        let target = camera.target + pose.drift;
        let eye = camera.eye() + pose.drift;
        let view = Mat4::look_at_rh(eye, target, Vec3::Y);
        // Cut to the model's own bounding sphere. A fixed ratio leaves a large piece with almost no
        // depth precision where it is actually drawn.
        let span = (eye - focus).length();
        let near = (span - reach).max(reach * 0.005);
        // Past the light box's own far corner rather than past the model, since the volume a lamp
        // is drawn as is clipped by these planes whether or not anything depth tests against them.
        let far = span + reach.max(level.radius * (1.0 + LAMP_SPAN * 2.0));
        let projection = Mat4::perspective_rh_gl(FOV, rect.width() / rect.height(), near, far);

        // Fill and rim follow the camera; a fill weighted toward the eye is the whole of what keeps
        // a surface turned away from the key from reading as a silhouette. Both are built from the
        // camera's axes rather than from a fragment's view vector, which would give every pixel a
        // rig of its own and sweep it across the surface as the camera moves.
        let axes = Mat3::from_mat4(view).transpose();
        let (right, up, back) = (axes.x_axis, axes.y_axis, axes.z_axis);
        let fill = back - right * 0.5 - up * 0.2;
        let rim = -back * 0.55 + up * 0.6 - right * 0.55;
        let mut lights = [0.0; 9];
        for (slot, light) in lights.chunks_exact_mut(3).zip([KEY, fill, rim]) {
            slot.copy_from_slice(&light.normalize().to_array());
        }

        let attachments = level.gpu.lock().unwrap().attachments();
        let lighting = match self.shaded.get() {
            true => {
                self.translate(level.skinned, level.waving, attachments);
                self.lighting(attachments)
            }
            false => None,
        };
        let surfaces = self.surfaces(&level);

        // The game's own shaders were compiled for a clip depth running from nought to one, and the
        // backend moves what they compute into the range GL clips against. A projection built for GL
        // would go through that move a second time and lose the near half of the frame.
        let held = Mat4::perspective_rh(FOV, rect.width() / rect.height(), near, far);

        // A cell of about half the model's radius, snapped to a one, a two or a five. Only the model
        // says what scale to rule at, and a bare decade is a tenfold jump: it leaves a piece of
        // landscape standing in one cell or a character ruled into mush.
        let cell = level.radius * 0.5;
        let decade = 10f32.powf(cell.log10().floor());
        let step = decade
            * match cell / decade {
                held if held < 1.5 => 1.0,
                held if held < 3.5 => 2.0,
                held if held < 7.5 => 5.0,
                _ => 10.0,
            };

        // The GL projection whichever path drew the frame: the game's own shaders are compiled for
        // a clip depth of nought to one and the backend moves what they write into GL's range, so
        // the depth both of them leave behind is this one's. The quad reaches past the far plane,
        // which is what leaves the fade rather than its own edge as where the grid stops.
        let grid = self.grid.get().then(|| grid::Ground {
            view_projection: (projection * view).to_cols_array(),
            // The camera carries the pose's drift and the lines do not, so a model that walks walks
            // over them.
            center: [eye.x, eye.z],
            extent: far * 1.5,
            range: [near, far],
            step,
        });

        let frame = gpu::Frame {
            view: view.to_cols_array(),
            projection: projection.to_cols_array(),
            target: self.target.get(),
            scene: program::Scene {
                view,
                projection: held,
                model: Mat4::from_scale(Vec3::splat(self.stature.get())),
                light: KEY,
                lamp: program::Lamp {
                    placement: Mat4::from_translation(
                        target + Vec3::new(0.0, level.radius, level.radius),
                    ),
                    min: Vec3::splat(-level.radius * LAMP_SPAN),
                    max: Vec3::splat(level.radius * LAMP_SPAN),
                    reach: level.radius * LAMP_SPAN,
                    color: Vec3::splat(LAMP_FILL),
                    ..Default::default()
                },
                look: self.look.get(),
                customize: self.made_up(),
                clock: self.clock.get(),
                ..Default::default()
            },
            lighting,
            post: match self.shaded.get() {
                true => self.post(),
                false => None,
            },
            smoothing: match self.shaded.get() {
                true => self.smoothing(),
                false => None,
            },
            glare: match self.shaded.get() {
                true => self.glare(),
                false => None,
            },
            occlusion: match self.shaded.get() {
                true => self.occlusion(),
                false => None,
            },
            reflection: match self.shaded.get() {
                true => self.mirror(),
                false => None,
            },
            vignette: match self.shaded.get() {
                true => self.corners(),
                false => None,
            },
            eye: eye.to_array(),
            lights,
            surfaces,
            joints: pose.joints,
            debug: self.debug.get(),
            grid,
            // The emote's own particles, drawn inside the frame rather than over the widget: only
            // there is the depth the character settled still attached to be tested against. On the
            // game's own clip depth, since these are game shaders and the soft-particle variant
            // rebuilds a world position out of that same depth buffer: handed a GL projection it
            // reads every depth half a range out, puts the scene surface on top of the particle and
            // discards it, so an effect vanishes wherever anything at all stands behind it.
            effects: std::sync::Mutex::new(self.effects.borrow().frames(
                view,
                held,
                (rect.width(), rect.height()),
                eye,
            )),
        };

        // Drawn with no depth test, which is what makes it an overlay rather than a rig buried in
        // the mesh it poses.
        let overlay = self.skeleton.get().then(|| {
            self.overlay.lock().unwrap().replace(markers);
            (self.overlay.clone(), (projection * view).to_cols_array())
        });

        // The context is taken from the painter rather than captured: `glow::Context` is neither
        // `Send` nor `Sync` on wasm, and a callback has to be both.
        let model = level.gpu.clone();
        ui.painter().add(egui::PaintCallback {
            rect,
            callback: Arc::new(egui_glow::CallbackFn::new(move |info, painter| {
                model
                    .lock()
                    .unwrap()
                    .draw(painter.gl(), painter, &frame, &info);
                if let Some((bones, view_projection)) = &overlay {
                    bones
                        .lock()
                        .unwrap()
                        .draw(painter.gl(), painter, view_projection, false);
                }
            })),
        });

    }

    /// How much of what the model needs has landed, against how much it asked for. A material names
    /// the package and textures it wants only once it has arrived itself, so the total grows as the
    /// fetches resolve rather than being known up front.
    fn arrived(&self) -> (usize, usize) {
        let slots = self.slots.borrow();
        let packages = self.packages.borrow();
        let textures = self.textures.borrow();
        let ready = slots
            .iter()
            .flatten()
            .filter(|slot| !matches!(slot, Slot::Fetching(_)))
            .count()
            + packages
                .values()
                .filter(|held| !matches!(held, Package::Fetching(_)))
                .count()
            + textures
                .values()
                .filter(|held| !matches!(held, Texture::Fetching(_)))
                .count();
        // A slot the model has not asked for yet still owes an answer, so every material the level
        // names counts against the total whether or not it is in flight.
        (ready, slots.len() + packages.len() + textures.len())
    }

    /// The raw file (or files, zipped, for a character worn out of several) plus the glTF bake.
    /// Gathering the scene runs synchronously against `self` so the async half that follows holds
    /// no reference to it and can outlive the frame that started it.
    fn export_choices<'a>(
        &'a self,
        backend: &Backend,
        ready: bool,
        waiting: usize,
    ) -> Vec<Choice<'a>> {
        let files = backend.files().clone();
        let raw_name = match self.pieces.as_slice() {
            [only] => crate::utils::file_name(&only.path).to_owned(),
            _ => "models.zip".to_owned(),
        };
        let raw = Choice::new("Raw file", raw_name, move || {
            let paths: Vec<String> = self.pieces.iter().map(|piece| piece.path.clone()).collect();
            Box::pin(async move {
                if let [only] = paths.as_slice() {
                    return files.read(only).await;
                }
                let mut entries = Vec::with_capacity(paths.len());
                for path in &paths {
                    let data = files.read(path).await?;
                    entries.push((crate::utils::file_name(path).to_owned(), data));
                }
                crate::utils::export::zip(&entries)
            })
        });

        let stem = crate::utils::file_name(&self.pieces[0].path)
            .strip_suffix(".mdl")
            .unwrap_or(&self.pieces[0].path)
            .to_owned();
        let files = backend.files().clone();
        let gltf = Choice::new(format!("{stem}.glb"), format!("{stem}.glb"), move || {
            let scene = export::gather(self);
            Box::pin(async move {
                let scene = scene?;
                export::finish(scene, files.as_ref()).await
            })
        })
        .hover(
            "Write the geometry, materials and posed skeleton as a self-contained .glb. Textures \
             are baked from this viewer's own preview shading, not the game shaders' G-buffer, so \
             a shaded render and the export will not match exactly",
        )
        .unless(
            ready,
            format!(
                "waiting on {waiting} materials, shader packages and textures to finish loading"
            ),
        );

        vec![raw, gltf]
    }

    /// What the channel row offers: the translated shaders' own names for their targets, and the
    /// frame the composite resolves once the passes that make it have arrived.
    fn channels(&self) -> Vec<(usize, String)> {
        let mut held: Vec<(usize, String)> = self
            .translated
            .borrow()
            .values()
            .filter_map(|held| held.held.as_ref().ok())
            .find_map(|passes| passes.buffer.first())
            .map(|buffer| buffer.names.iter().cloned().enumerate().collect())
            .unwrap_or_default();
        if !held.is_empty() && self.lighting.borrow().is_some() {
            held.push((gpu::LIT, "Lit".to_owned()));
            if self.look.get().reflect {
                held.push((deferred::REFLECTED, "Reflection".to_owned()));
            }
        }
        held
    }

    /// The pass that grades the resolved frame, translated once its shader has arrived. Withheld
    /// until the table it reads has landed too: a pass drawn against the flat stand-in would grade
    /// every pixel toward the one grey it answers with.
    fn post(&self) -> Option<Arc<program::Program>> {
        if !self.graded.get() {
            return None;
        }
        if let Some(held) = self.post.borrow().as_ref() {
            return Some(held.clone());
        }
        let mut packages = self.packages.borrow_mut();
        let built = match packages.get(program::TONE_ADJUST) {
            Some(Package::Ready(bytes)) => {
                program::Program::posteffect(program::TONE_ADJUST, bytes, program::POST_VERTEX)
            }
            _ => return None,
        };
        // Kept as a failure rather than tried again: the file will not translate differently on the
        // next frame, and the pass is skipped from here on.
        let built = match built {
            Ok(held) => Arc::new(held),
            Err(why) => {
                log::error!("assets/mdl: {}: {why}", program::TONE_ADJUST);
                packages.insert(program::TONE_ADJUST.to_owned(), Package::Failed(why));
                return None;
            }
        };
        drop(packages);
        *self.post.borrow_mut() = Some(built.clone());
        Some(built)
    }

    /// The pair that smooths the graded frame's edges, translated once both shaders have arrived.
    fn smoothing(&self) -> Option<Arc<gpu::Smoothing>> {
        if !self.look.get().antialias {
            return None;
        }
        if let Some(held) = self.smoothing.borrow().as_ref() {
            return Some(held.clone());
        }
        let packages = self.packages.borrow();
        let held = |path: &str| {
            let Some(Package::Ready(bytes)) = packages.get(path) else {
                return None;
            };
            program::Program::posteffect(path, bytes, program::POST_VERTEX)
                .inspect_err(|why| log::warn!("assets/mdl: {path}: {why}"))
                .ok()
                .map(Arc::new)
        };
        let built = gpu::Smoothing {
            luma: held(program::FXAA_LUMA)?,
            fxaa: held(program::FXAA)?,
        };
        drop(packages);
        let built = Arc::new(built);
        *self.smoothing.borrow_mut() = Some(built.clone());
        Some(built)
    }

    /// The pass that darkens the frame's corners, translated once its shader has arrived. Against
    /// the sky's own vertex shader, which is the one here handing a fragment where it stands rather
    /// than what to read.
    fn corners(&self) -> Option<Arc<program::Program>> {
        if !self.look.get().vignette {
            return None;
        }
        if let Some(held) = self.vignette.borrow().as_ref() {
            return Some(held.clone());
        }
        let packages = self.packages.borrow();
        let Some(Package::Ready(bytes)) = packages.get(program::VIGNETTE) else {
            return None;
        };
        let built = program::Program::posteffect(program::VIGNETTE, bytes, program::SKY_VERTEX)
            .inspect_err(|why| log::warn!("assets/mdl: {}: {why}", program::VIGNETTE))
            .ok()
            .map(Arc::new)?;
        drop(packages);
        *self.vignette.borrow_mut() = Some(built.clone());
        Some(built)
    }

    /// The chain that spreads the bright end of the frame, translated once its four shaders have
    /// arrived. The blur reads seven coordinates rather than one, so it is drawn with the vertex
    /// shader the game pairs it with rather than the one every other pass here takes.
    fn glare(&self) -> Option<Arc<gpu::Glare>> {
        if !self.look.get().bloom {
            return None;
        }
        if let Some(held) = self.glare.borrow().as_ref() {
            return Some(held.clone());
        }
        let packages = self.packages.borrow();
        let ready = |path: &str| match packages.get(path) {
            Some(Package::Ready(bytes)) => Some(bytes),
            _ => None,
        };
        let held = |path: &str| {
            program::Program::posteffect(path, ready(path)?, program::POST_VERTEX)
                .inspect_err(|why| log::warn!("assets/mdl: {path}: {why}"))
                .ok()
                .map(Arc::new)
        };
        let sampled = |path: &str, vertex: &str| {
            program::Program::sampling(path, ready(path)?, ready(vertex)?)
                .inspect_err(|why| log::warn!("assets/mdl: {path}: {why}"))
                .ok()
                .map(Arc::new)
        };
        let built = gpu::Glare {
            bright: held(program::BRIGHT_PASS)?,
            gauss: sampled(program::GAUSS_BLUR, program::SAMPLING_9)?,
            blur: sampled(program::BLOOM_BLUR, program::SAMPLING_7)?,
            merge: held(program::GLARE_MERGE)?,
            composite: held(program::GLARE_COMPOSITE)?,
        };
        drop(packages);
        let built = Arc::new(built);
        *self.glare.borrow_mut() = Some(built.clone());
        Some(built)
    }

    /// The chain that reflects the frame off itself, translated once its eight shaders have
    /// arrived. Every member is drawn with the vertex shader the game pairs it with, which hands a
    /// fragment both where it stands and what to read.
    fn mirror(&self) -> Option<Arc<gpu::Reflection>> {
        if !self.look.get().reflect {
            return None;
        }
        if let Some(held) = self.reflection.borrow().as_ref() {
            return Some(held.clone());
        }
        let packages = self.packages.borrow();
        let ready = |path: &str| match packages.get(path) {
            Some(Package::Ready(bytes)) => Some(bytes),
            _ => None,
        };
        let held = |path: &str, vertex: &str| {
            program::Program::sampling(path, ready(path)?, ready(vertex)?)
                .inspect_err(|why| log::warn!("assets/mdl: {path}: {why}"))
                .ok()
                .map(Arc::new)
        };
        let read = |path: &str| held(path, program::REFLECTION_VERTEX);
        let built = gpu::Reflection {
            normal: read(program::REFLECTION_NORMAL)?,
            mask: read(program::REFLECTION_MASK)?,
            march: read(program::REFLECTION_MARCH)?,
            blur: [
                read(program::REFLECTION_BLUR_X)?,
                read(program::REFLECTION_BLUR_Y)?,
            ],
            distort: read(program::REFLECTION_DISTORT)?,
            copy: held(program::REFLECTION_COPY, program::REFLECTION_MERGE_VERTEX)?,
        };
        drop(packages);
        let built = Arc::new(built);
        *self.reflection.borrow_mut() = Some(built.clone());
        Some(built)
    }

    /// The chain that occludes the frame, translated once its three shaders have arrived. Rebuilt
    /// where the quality changed, since that is a file of its own.
    fn occlusion(&self) -> Option<Arc<gpu::Occlusion>> {
        let look = self.look.get();
        if !look.occlude {
            return None;
        }
        if let Some((quality, held)) = self.occlusion.borrow().as_ref()
            && *quality == look.quality
        {
            return Some(held.clone());
        }
        let packages = self.packages.borrow();
        let held = |path: &str, vertex| {
            let Some(Package::Ready(bytes)) = packages.get(path) else {
                return None;
            };
            program::Program::posteffect(path, bytes, vertex)
                .inspect_err(|why| log::warn!("assets/mdl: {path}: {why}"))
                .ok()
                .map(Arc::new)
        };
        let built = gpu::Occlusion {
            scale: held(program::DOWN_SCALE, program::POST_VERTEX)?,
            gather: held(program::GATHER, program::GATHER_VERTEX)?,
            occlude: held(&look.occluder(), program::POST_VERTEX)?,
        };
        drop(packages);
        let built = Arc::new(built);
        *self.occlusion.borrow_mut() = Some((look.quality, built.clone()));
        Some(built)
    }

    /// Turns shading back off the moment it fails to build, so the model still draws with the plain
    /// pass instead of going blank. Keeps the first reason; flipping "Game shaders" back on is what
    /// clears it for a fresh attempt.
    fn fail_shading(&self, why: String) {
        self.shade_failure.borrow_mut().get_or_insert(why);
        self.shaded.set(false);
    }

    /// The passes that light the G-buffer, translated once their packages have arrived. They are the
    /// same whatever is being drawn, so they are built once and kept.
    fn lighting(&self, attachments: usize) -> Option<Arc<gpu::Lighting>> {
        let matches = |held: &Option<(usize, Arc<gpu::Lighting>)>| {
            held.as_ref().is_some_and(|(held, _)| *held == attachments)
        };
        if matches(&self.lighting.borrow()) {
            // Read again after these: each replaces `self.lighting` with a fur or subsurface pass
            // added, and returning the clone taken before them would hand the caller a frame short
            // of whichever just landed.
            self.soften(attachments);
            self.scatter(attachments);
            return self
                .lighting
                .borrow()
                .as_ref()
                .map(|(_, built)| built.clone());
        }
        let packages = self.packages.borrow();
        let held = |path: &str, pass| {
            let Some(Package::Ready(bytes)) = packages.get(path) else {
                return None;
            };
            program::Program::screen(bytes, pass, attachments, &[])
                .inspect_err(|why| {
                    log::warn!("assets/mdl: {path} attachments={attachments}: {why}");
                    self.fail_shading(format!("{path} attachments={attachments}: {why}"));
                })
                .ok()
                .map(Arc::new)
        };
        let built = gpu::Lighting {
            // The model viewer stands one object against nothing, so nothing casts onto it.
            shadow: None,
            subsurface: None,
            position: held(program::VIEW_POSITION, program::Pass::Lighting)?,
            directional: held(program::DIRECTIONAL, program::Pass::Lighting)?,
            // One studio light of this viewer's own, so the one falloff it stands at is enough.
            point: std::array::from_fn({
                let held = held(program::POINT, program::Pass::Lamp)?;
                move |_| held.clone()
            }),
            // A model stands under one studio light of this viewer's own, which is a point.
            spot: None,
            line: None,
            plane: None,
            fur: None,
            composite: held(program::COMPOSITE, program::Pass::Composite)?,
        };
        drop(packages);
        let built = Arc::new(built);
        *self.lighting.borrow_mut() = Some((attachments, built.clone()));
        Some(built)
    }

    /// The same for the skin blur, which every character wants and which the pass itself gates per
    /// pixel off the type table.
    fn scatter(&self, attachments: usize) {
        let lit = self.lighting.borrow().clone();
        let Some((_, lighting)) =
            lit.filter(|(held, lighting)| *held == attachments && lighting.subsurface.is_none())
        else {
            return;
        };
        let mut packages = self.packages.borrow_mut();
        let Some(Package::Ready(bytes)) = packages.get(program::SCATTER) else {
            return;
        };
        let held = match program::Program::screen(bytes, program::Pass::Lighting, attachments, &[])
        {
            Ok(held) => Arc::new(held),
            Err(why) => {
                log::warn!("assets/mdl: {}: {why}", program::SCATTER);
                packages.insert(program::SCATTER.to_owned(), Package::Failed(why));
                return;
            }
        };
        drop(packages);
        *self.lighting.borrow_mut() = Some((
            attachments,
            Arc::new(gpu::Lighting {
                subsurface: Some(held),
                ..(*lighting).clone()
            }),
        ));
    }

    /// Takes the fur pass up on whichever frame its package arrives on, the frame having lit without
    /// it until then. One that arrived and would not translate is marked failed rather than
    /// translated again every frame, which costs a whole one.
    fn soften(&self, attachments: usize) {
        let lit = self.lighting.borrow().clone();
        let Some((_, lighting)) =
            lit.filter(|(held, lighting)| *held == attachments && lighting.fur.is_none())
        else {
            return;
        };
        let mut packages = self.packages.borrow_mut();
        let Some(Package::Ready(bytes)) = packages.get(program::FUR) else {
            return;
        };
        let fur = match program::Program::screen(bytes, program::Pass::Fur, attachments, &[]) {
            Ok(held) => Arc::new(held),
            Err(why) => {
                log::warn!("assets/mdl: {}: {why}", program::FUR);
                packages.insert(program::FUR.to_owned(), Package::Failed(why));
                return;
            }
        };
        drop(packages);
        *self.lighting.borrow_mut() = Some((
            attachments,
            Arc::new(gpu::Lighting {
                fur: Some(fur),
                ..(*lighting).clone()
            }),
        ));
    }

    /// Translates every ready material's passes, again where the context's own limit changed how
    /// many of the G-buffer's targets one reading can write.
    fn translate(&self, skinned: bool, waving: bool, attachments: usize) {
        let slots = self.slots.borrow();
        let packages = self.packages.borrow();
        let mut translated = self.translated.borrow_mut();
        let mut tables = self.tables.borrow_mut();
        // The keys the engine sets rather than the material: a mesh carrying bone indices is one the
        // game would draw through the skinning variant. `GetNormalMap` is added per material below,
        // since only `bg.shpk` has a node for the parallax value.
        let mut base = vec![
            (APPLY_ALPHA_CLIP, APPLY_ALPHA_CLIP_ON),
            (APPLY_DITHER_CLIP, APPLY_DITHER_CLIP_ON),
            (GET_RLR, GET_RLR_ON),
        ];
        if skinned {
            base.push((TRANSFORM_VIEW, TRANSFORM_VIEW_SKIN));
        }
        if waving {
            base.push((APPLY_WAVING_ANIM, APPLY_WAVING_ANIM_ON));
        }
        let mut read: HashMap<String, ShaderPackage> = HashMap::new();
        for (index, slot) in slots.iter().enumerate() {
            let Some(Slot::Ready(material)) = slot else {
                continue;
            };
            // Not a failed one: a fresh attempt is what a deliberate re-enable after a build failure
            // is for, rather than the same answer coming back forever with no notice of it.
            if translated
                .get(&index)
                .is_some_and(|held| held.attachments == attachments && held.held.is_ok())
            {
                continue;
            }
            let name = material.package();
            let Some(Package::Ready(bytes)) = packages.get(&name) else {
                continue;
            };
            if !read.contains_key(&name) {
                match ShaderPackage::parse(bytes) {
                    Ok(held) => {
                        read.insert(name.clone(), held);
                    }
                    Err(why) => {
                        log::error!("assets/mdl: {name}: {why}");
                        self.fail_shading(format!("{name}: {why}"));
                        continue;
                    }
                }
            }
            let package = &read[&name];
            let mut keys = base.clone();
            keys.push((
                GET_NORMAL_MAP,
                match name.ends_with("/bg.shpk") {
                    true => GET_NORMAL_MAP_PARALLAX,
                    false => GET_NORMAL_MAP_ON,
                },
            ));
            let build = |pass, at| {
                program::Program::build(
                    package,
                    bytes,
                    material,
                    &keys,
                    pass,
                    program::SUB_VIEW_MAIN,
                    at,
                    attachments,
                )
            };
            let mut passes = Passes::default();
            // A package with no opaque pass is a surface that blends itself into the frame: water
            // and river fill the same G-buffer through a pass of their own. A character-family one
            // is transparent everywhere rather than a fringe around an opaque fill, so it takes the
            // dedicated semi-transparent buffer a fringe does instead: claiming the shared opaque
            // G-buffer and depth would be wrong, and its composite's real coverage would land in
            // the frame's alpha, which downstream is a glare share rather than an opacity.
            let filling = build(program::Pass::Buffer, 0)
                .map(|held| (program::Pass::Buffer, held))
                .or_else(|_| {
                    build(program::Pass::Blended, 0).map(|held| (program::Pass::Blended, held))
                });
            let all_transparent = matches!(&filling, Ok((program::Pass::Blended, _)))
                && CHARACTER_TRANSPARENT_PACKAGES
                    .iter()
                    .any(|suffix| name.ends_with(suffix));
            if let Ok((pass, first)) = filling {
                if all_transparent {
                    passes.sheer = build(program::Pass::CompositeBlended, 0)
                        .map(|resolve| (Arc::new(first), Arc::new(resolve)))
                        .inspect_err(|why| {
                            log::warn!(
                                "assets/mdl: {}: no semi-transparent resolve: {why}",
                                material.package()
                            )
                        })
                        .ok();
                } else {
                    let pages = first.outputs.len().div_ceil(attachments.max(1)).max(1);
                    passes.buffer.push(Arc::new(first));
                    passes
                        .buffer
                        .extend((1..pages).filter_map(|at| build(pass, at).ok().map(Arc::new)));
                    // Only where the same vertex shader settled the depth. A blending surface fills
                    // the buffer through a pass whose vertices are lifted by its own waves, and the
                    // depth pass leaves them where the file put them: every later test against it
                    // fails.
                    if pass == program::Pass::Buffer {
                        passes.depth = build(program::Pass::Depth, 0).ok().map(Arc::new);
                        // Only where the material states a clip the semi-transparent pass's own
                        // reaches under. Below that the two passes cover the same fragments, and the
                        // resolve drops every one the opaque half already drew.
                        if material.clip() > program::SHEER_CLIP {
                            passes.sheer = build(program::Pass::Blended, 0)
                                .and_then(|held| {
                                    Ok((Arc::new(held), Arc::new(build(program::Pass::CompositeBlended, 0)?)))
                                })
                                .inspect_err(|why| {
                                    log::warn!(
                                        "assets/mdl: {}: no semi-transparent pass: {why}",
                                        material.package()
                                    )
                                })
                                .ok();
                        }
                    }
                }
            }
            // A package carrying a composite of its own resolves itself with it. The screen-wide
            // pass is `bg`'s, and `bg` reserves values past one in the second target as the sign
            // that a pixel keeps its specular color in the fifth; a character writes a luminance
            // there that reaches one of its own accord, and is then read as that.
            //
            // Not where the semi-transparent buffer above already took the same composite: that one
            // draws through `sheer()`'s own resolve, against its own dedicated depth, rather than
            // here against the shared frame.
            if !all_transparent {
                passes.resolve = build(program::Pass::Composite, 0)
                    .or_else(|_| build(program::Pass::CompositeBlended, 0))
                    .or_else(|_| build(program::Pass::Water, 0))
                    .ok()
                    .map(Arc::new);
            }
            let held = match passes.buffer.is_empty()
                && passes.resolve.is_none()
                && passes.sheer.is_none()
            {
                true => Err("this material's keys reach no pass that draws it".into()),
                false => Ok(passes),
            };
            if let Err(why) = &held {
                let why = format!(
                    "material {index} ({}) attachments={attachments}: {why}",
                    material.package()
                );
                log::warn!("assets/mdl: {why}");
                self.fail_shading(why);
            }
            translated.insert(index, Translated { attachments, held });
            if let Some((values, columns, rows)) =
                material.held().color_table().and_then(program::table)
            {
                tables
                    .entry(index)
                    .or_insert_with(|| Arc::new((values, columns, rows)));
            }
        }
    }

    /// Defaults every part's visibility from the picked variant's attribute mask. Cheap enough to
    /// call on every arrival and every pick: it only sets cells, never rebuilds the level.
    ///
    /// A part gated past the ten bits an imc entry carries is one the file cannot speak about, so it
    /// draws whatever the variant says: a model may declare far more attributes than that. So is one
    /// whose entry enables nothing, which is what a racial outfit states for every slot but the body.
    /// The parts that gates are the seams between slots, and holding them back leaves a character's
    /// breeches ending mid-thigh where its boots start at the knee.
    fn apply_variant(&self) {
        let masks: Vec<Option<u32>> = self.pieces.iter().map(Piece::mask).collect();
        let level = self.level.borrow();
        let hidden = self.hidden.borrow();
        for (mask, part) in level.meshes.iter().flat_map(|mesh| {
            let mask = masks[mesh.piece];
            mesh.parts.iter().map(move |part| (mask, part))
        }) {
            // A part the tab has switched off is off whatever the variant says, which is how a face
            // draws one of the features it declares rather than every one of them at once.
            if part
                .attributes
                .split(", ")
                .any(|name| hidden.contains(name))
            {
                part.shown.set(false);
                continue;
            }
            let gated = part.mask & IMC_ATTRIBUTES;
            part.shown
                .set(mask.is_none_or(|mask| gated == 0 || mask == 0 || gated & mask != 0));
        }
    }

    /// Rewrites every touched mesh's indices from the file's own, so switching a shape off restores
    /// what it replaced and two shapes over the same mesh both land.
    fn apply(&self) {
        let level = self.level.borrow();
        let enabled = self.shapes.borrow();
        let mut rewritten: BTreeMap<usize, Vec<u16>> = BTreeMap::new();
        for shape in level
            .shapes
            .iter()
            .filter(|shape| enabled.contains(&shape.name))
        {
            for (mesh, values) in &shape.rewrites {
                let indices = rewritten
                    .entry(*mesh)
                    .or_insert_with(|| level.meshes[*mesh].base.clone());
                for (offset, vertex) in values {
                    if let Some(held) = indices.get_mut(usize::from(*offset)) {
                        *held = *vertex;
                    }
                }
            }
        }
        // A mesh a shape has just stopped touching still holds that shape's indices, so every mesh
        // any shape reaches is uploaded rather than only the ones still rewritten.
        let mut gpu = level.gpu.lock().unwrap();
        for mesh in level
            .shapes
            .iter()
            .flat_map(|shape| &shape.rewrites)
            .map(|(mesh, _)| *mesh)
            .collect::<BTreeSet<_>>()
        {
            let indices = rewritten
                .remove(&mesh)
                .unwrap_or_else(|| level.meshes[mesh].base.clone());
            gpu.queue_indices(mesh, indices);
        }
    }

    /// Draws another detail level of the same files.
    fn switch(&self, lod: u8) {
        let paths: Vec<&str> = self
            .pieces
            .iter()
            .map(|piece| piece.path.as_str())
            .collect();
        let attachments = self
            .level
            .borrow()
            .gpu
            .lock()
            .unwrap()
            .attachments_learned()
            .unwrap_or(0);
        match level_of(&self.pieces, lod, attachments) {
            Ok(level) => {
                self.lod.set(lod);
                self.rebuild(level);
            }
            Err(why) => log::error!(
                "assets/mdl: {}: detail level {lod}: {why}",
                paths.join(" + ")
            ),
        }
    }

    /// How the character was made: the colours its shaders tint with, the attributes its face draws
    /// and the shape keys that deform it. Taken together so a pick costs one pass over the parts.
    /// What the creator left, less a face paint whose own texture has not arrived: the flat
    /// stand-in an unbound sampler answers with would lay the paint's colour over the whole face.
    fn made_up(&self) -> program::Customize {
        let mut held = self.customize.get();
        if !matches!(
            self.arrays.borrow().get(&deferred::FACE_PAINT),
            Some(Array::Ready(_))
        ) {
            held.decal[3] = 0.0;
        }
        held
    }

    pub fn made(
        &self,
        customize: program::Customize,
        hidden: BTreeSet<String>,
        shapes: BTreeSet<String>,
        stature: f32,
        bust: Vec3,
    ) {
        self.customize.set(customize);
        self.stature.set(stature);
        self.animation.shaped(bust);
        *self.hidden.borrow_mut() = hidden;
        let changed = *self.shapes.borrow() != shapes;
        *self.shapes.borrow_mut() = shapes;
        self.apply_variant();
        if changed {
            self.apply();
        }
    }

    /// What the wearer picked to stain each worn piece with, by piece index in the same order the
    /// character tab built its parts. Cheap enough to hand over on every frame: a piece the stains
    /// have not changed for costs the cache lookup in [`Self::dyed_table`] and nothing more.
    pub fn dye(&self, templates: Option<Rc<dye::Templates>>, stains: Vec<[Option<u8>; 2]>) {
        *self.dye_templates.borrow_mut() = templates;
        *self.stains.borrow_mut() = stains;
    }

    /// The color table one mesh's material draws with: the base table a stain replaces nothing in,
    /// or, where the piece it came from carries one, that table with the wearer's picks applied.
    fn dyed_table(&self, mesh: &Mesh, material: &material::Material, base: &Table) -> Table {
        let stains = self
            .stains
            .borrow()
            .get(mesh.piece)
            .copied()
            .unwrap_or_default();
        if stains == [None, None] {
            return base.clone();
        }
        if let Some((table, held)) = self.dyed.borrow().get(&mesh.material)
            && *held == stains
        {
            return table.clone();
        }
        let built = self
            .dye_templates
            .borrow()
            .as_deref()
            .zip(material.held().color_table())
            .and_then(|(templates, colors)| dye::table(base, colors, templates, stains))
            .unwrap_or_else(|| base.clone());
        self.dyed
            .borrow_mut()
            .insert(mesh.material, (built.clone(), stains));
        built
    }

    /// How far a raised visor has turned, one angle per bone it hinges on.
    pub fn hinged(&self, visor: [f32; 3]) {
        self.animation.hinged(visor);
    }

    /// Which of the mount's own seats the rider takes.
    pub fn seated(&self, seat: usize) {
        self.animation.seated(seat);
    }

    /// Which pieces hang rigidly off a bone this frame rather than posing on the shared rig, each
    /// by the path it was worn as, the bone it hangs from, and its own placement relative to that
    /// bone. Replaces whatever was carried last frame outright: a weapon put away carries nothing.
    pub fn carried(&self, pieces: Vec<(String, String, Mat4)>, drawn: bool) {
        self.wielded.set(drawn);
        *self.attachments.borrow_mut() = pieces
            .into_iter()
            .map(|(path, bone, local)| Attachment { path, bone, local })
            .collect();
    }

    /// The models an emote's own timeline wants held right now, each by the path it is worn as, its
    /// material variant and the weapon set it is filed under, where a `C043`/`C198` prop command's
    /// window covers the current time. Empty once they have played through: the caller is what
    /// carries a prop away again, the way a weapon put away carries nothing.
    pub fn wanted_props(&self) -> Vec<(String, u16, u16)> {
        let Some((_, _, time)) = self.animation.body_playing() else {
            return Vec::new();
        };
        self.emote.borrow().active_props(time)
    }

    /// Poses the character out of the first of `packs` the install holds, which is what picking
    /// an emote is.
    pub fn play(&self, packs: &[String], then: Option<&str>) {
        self.animation.play(packs, then);
    }

    /// Poses the character out of the first of `packs` the install holds over whatever it is
    /// already doing, which is what picking an emote while mounted is.
    pub fn play_over(&self, packs: &[String]) {
        self.animation.play_over(packs);
    }

    /// What to price a change of clip against, out of the game's own blend tables.
    pub fn blending(&self, blend: impl Fn(&str, &str) -> f32 + 'static) {
        self.animation.blending(blend);
    }

    /// Where each drawn weapon's own effect would play, by the bone it hangs from. The character
    /// scene runs no particles, so this marks the place the way an emote's own vfx is marked.
    pub fn glowing(&self, effects: Vec<(String, String)>) {
        if *self.glowing.borrow() != effects {
            self.glowing_at.set(None);
        }
        *self.glowing.borrow_mut() = effects;
    }

    /// Stands the character in the first of `poses` its own pack actually holds, cross-fading out
    /// of whatever it was standing in, which is what a change of weapon or of stance is.
    pub fn stand(&self, poses: &[(String, &str)], fade: f32) {
        self.animation.stand(poses, fade);
    }

    /// Lays one motion over the pose the character is standing in for as long as it runs, which is
    /// what drawing or sheathing a weapon is.
    pub fn act(&self, packs: &[String], motion: &str, fade: f32) {
        self.animation.act(packs, motion, fade);
    }

    /// Puts the motion the character is playing `seconds` in rather than wherever wall time has
    /// run it to, which is what a transport seeking by a cutscene's own frame numbering asks for.
    /// Read after [`Self::stand`], since taking a clip up is what puts its clock back to nought.
    pub fn plays_at(&self, seconds: f32) {
        self.animation.plays_at(seconds);
    }

    /// The motion the character is standing in, by the name its own pack gives it.
    pub fn standing(&self) -> Option<String> {
        self.animation.standing()
    }

    /// The motion laid over the pose the character is standing in, where one is still running.
    pub fn acting(&self) -> Option<String> {
        self.animation.acting()
    }

    /// What each eye-size bone is scaled by, which no clip ever states.
    pub fn eyed(&self, eyes: [f32; 2]) {
        self.animation.eyed(eyes);
    }

    /// Puts an expression on the character's face, which is what picking an emote that only makes
    /// one is.
    pub fn express(&self, name: &str) {
        self.animation.express(name);
    }

    /// The bodies to read animation from, nearest first, which the caller reads off the same tree
    /// that says where a body borrows its clothes from.
    /// Marks this model as standing in a frame this view does not draw, which is what keeps it
    /// from asking for the passes that light one.
    pub fn placed(&self) {
        self.chrome.set(Chrome::Placed);
    }

    pub fn built_on(&self, lineage: Vec<String>) {
        self.animation.built_on(lineage);
    }

    /// Puts a different set of files on the same character, which is what a change of clothes is.
    /// The camera, the rig and the motion it is playing all stay where they are; the rig is rebuilt
    /// only where the body under the clothes changed.
    pub fn redress(&mut self, parts: &[Source]) -> Result<()> {
        parts.first().context("a model of no files")?;
        // Whatever is still being worn is kept as it stands, imc and all, so a change of one slot
        // reads one file rather than every file the character is drawn from.
        let mut held: BTreeMap<String, Piece> = std::mem::take(&mut self.pieces)
            .into_iter()
            .map(|piece| (piece.path.clone(), piece))
            .collect();
        let pieces = parts
            .iter()
            .map(|source| match held.remove(&source.path) {
                Some(piece) if piece.wears(source) => Ok(piece),
                _ => Piece::new(source),
            })
            .collect::<Result<Vec<_>>>()?;
        let drawn = drawn_levels(&pieces);
        let lod = match drawn[usize::from(self.lod.get())] {
            true => self.lod.get(),
            false => 0,
        };
        let attachments = self
            .level
            .borrow()
            .gpu
            .lock()
            .unwrap()
            .attachments_learned()
            .unwrap_or(0);
        let level = level_of(&pieces, lod, attachments)?;

        let rode = self.animation.rides().map(str::to_owned);
        if !self
            .animation
            .poses(parts.iter().map(|part| part.path.as_str()))
        {
            self.animation = skin::Animation::new(parts.iter().map(|part| part.path.as_str()));
        }
        self.animation
            .rewear(parts.iter().map(|part| part.path.as_str()));
        self.pieces = pieces;
        self.drawn = drawn;
        self.lod.set(lod);
        self.rebuild(level);
        // Getting on or off a mount is a whole second body coming and going rather than a change of
        // clothes, so the view is framed on what is there now.
        if rode.as_deref() != self.animation.rides() {
            self.camera.set(self.level.borrow().home);
        }
        Ok(())
    }

    /// Takes up a level built from the pieces the model now holds. Everything already fetched is
    /// kept and matched to the new geometry by material path, so nothing is asked for twice and
    /// nothing pops. Index is not a key that survives this: merging or changing a piece renumbers
    /// the materials, and an entry carried across by index would draw one material's geometry
    /// through another's shader.
    fn rebuild(&self, level: Level) {
        let mut slots = self.slots.borrow_mut();
        let mut translated = self.translated.borrow_mut();
        let mut tables = self.tables.borrow_mut();
        // Keyed by material index alone, and a rebuild renumbers materials same as everything else
        // matched by path above; kept, it would hand a stain a stranger's table.
        self.dyed.borrow_mut().clear();
        let was = std::mem::take(&mut self.level.borrow_mut().materials);
        let mut held: BTreeMap<String, Kept> = was
            .into_iter()
            .enumerate()
            .map(|(index, path)| {
                let slot = slots.get_mut(index).and_then(Option::take);
                (
                    path,
                    (slot, translated.remove(&index), tables.remove(&index)),
                )
            })
            .collect();
        translated.clear();
        tables.clear();
        slots.clear();
        for (index, path) in level.materials.iter().enumerate() {
            let (slot, program, table) = held.remove(path).unwrap_or_default();
            translated.extend(program.map(|program| (index, program)));
            tables.extend(table.map(|table| (index, table)));
            slots.push(slot);
        }
        // The new level's context has no color tables of its own, and a material kept from the old
        // one never transitions again to hand one over.
        for (index, slot) in slots.iter().enumerate() {
            if let Some(Slot::Ready(material)) = slot
                && let Some(table) = material.table()
            {
                level.gpu.lock().unwrap().queue_table(index, table.to_vec());
            }
        }
        // Nor the shader type table, nor the engine's own texture arrays, and the files all of
        // them are built from have already arrived.
        if let Some(values) = types(&self.parameters.borrow()) {
            level.gpu.lock().unwrap().queue_types(values);
        }
        for (id, bytes) in self.arrays.borrow().iter() {
            if let Array::Ready(bytes) = bytes {
                level.gpu.lock().unwrap().queue_array(*id, bytes.clone());
            }
        }

        drop((slots, translated, tables));
        *self.level.borrow_mut() = level;
        self.apply();
        self.apply_variant();
    }

    pub fn details_ui(&self, ui: &mut egui::Ui, follow: &mut Option<String>) {
        let mut picked = None;
        let mut toggled = None;
        let mut picked_shape = None;
        let mut picked_variant = None;
        ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
            let level = self.level.borrow();
            facts(ui, "mdl_identity", &level.identity);
            // A file drawing at one detail level has nothing to pick between.
            if self.drawn.iter().filter(|drawn| **drawn).count() > 1 {
                ui.add_space(8.0);
                section(ui, "Detail");
                let lod = self.lod.get();
                ui.horizontal(|ui| {
                    for (level, label) in [(0, "High"), (1, "Medium"), (2, "Low")] {
                        let picker = ui.add_enabled(
                            self.drawn[usize::from(level)],
                            egui::Button::selectable(lod == level, label),
                        );
                        if picker.clicked() && lod != level {
                            picked = Some(level);
                        }
                    }
                });
            }
            if !level.shapes.is_empty() {
                ui.add_space(8.0);
                section(ui, "Shapes");
                let enabled = self.shapes.borrow();
                let on = |at: usize| enabled.contains(&level.shapes[at].name);
                let hover = |at: usize| {
                    let shape = &level.shapes[at];
                    format!("{}\n{} meshes rewritten", shape.name, shape.rewrites.len())
                };
                // Clicking the variant already showing is what turns its category off, so a
                // category needs no entry of its own for having nothing applied.
                let chip = |ui: &mut egui::Ui, at: usize, label: &str| {
                    ui.selectable_label(on(at), label)
                        .on_hover_text(hover(at))
                        .clicked()
                };
                for (index, group) in level.groups.iter().enumerate() {
                    if group.category.is_empty() {
                        continue;
                    }
                    ui.label(RichText::new(&group.category).weak());
                    ui.horizontal_wrapped(|ui| {
                        for (at, variant) in &group.variants {
                            if chip(ui, *at, variant) {
                                picked_shape = Some((index, (!on(*at)).then_some(*at)));
                            }
                        }
                    });
                }
                // Whatever the file names without a variant, which is most of what a model
                // deforms. Each stands on its own, so they share one row rather than taking a
                // heading each.
                if level.groups.iter().any(|group| group.category.is_empty()) {
                    ui.horizontal_wrapped(|ui| {
                        for (index, group) in level.groups.iter().enumerate() {
                            if !group.category.is_empty() {
                                continue;
                            }
                            let (at, name) = &group.variants[0];
                            if chip(ui, *at, name) {
                                picked_shape = Some((index, (!on(*at)).then_some(*at)));
                            }
                        }
                    });
                }
            }

            for (at, piece) in self.pieces.iter().enumerate() {
                let Some(count) = piece.variants().filter(|count| *count > 0) else {
                    continue;
                };
                ui.add_space(8.0);
                section(ui, "Variant");
                if self.pieces.len() > 1 {
                    ui.label(RichText::new(crate::utils::file_name(&piece.path)).weak());
                }
                let current = piece.variant.get();
                ui.horizontal_wrapped(|ui| {
                    for variant in 0..=count {
                        if ui
                            .selectable_label(current == variant, variant.to_string())
                            .clicked()
                            && current != variant
                        {
                            picked_variant = Some((at, variant));
                        }
                    }
                });
            }

            if level.skinned {
                ui.add_space(8.0);
                self.animation.details_ui(ui, follow);
            }

            ui.add_space(8.0);
            section(ui, "Meshes");
            for (index, mesh) in level.meshes.iter().enumerate() {
                ui.horizontal_wrapped(|ui| {
                    let drawn = mesh.parts.iter().any(|part| part.shown.get());
                    if ui
                        .selectable_label(drawn, RichText::new(format!("Mesh {index}")).weak())
                        .on_hover_text(format!(
                            "{}\n{} triangles",
                            crate::utils::file_name(&level.materials[mesh.material]),
                            mesh.triangles
                        ))
                        .clicked()
                    {
                        toggled = Some((index, None));
                    }
                    for (part, held) in mesh.parts.iter().enumerate() {
                        let label = match held.attributes.is_empty() {
                            true => part.to_string(),
                            false => held.attributes.clone(),
                        };
                        if ui.selectable_label(held.shown.get(), label).clicked() {
                            toggled = Some((index, Some(part)));
                        }
                    }
                });
            }
            ui.add_space(8.0);
            section(ui, "Materials");
            let slots = self.slots.borrow();
            for (index, path) in level.materials.iter().enumerate() {
                if link(ui, crate::utils::file_name(path), path) {
                    *follow = Some(path.clone());
                }
                match slots.get(index).and_then(Option::as_ref) {
                    Some(Slot::Ready(material)) => {
                        ui.label(RichText::new(material.summary()).weak());
                    }
                    Some(Slot::Failed(why)) => {
                        ui.label(RichText::new(why).color(Color32::LIGHT_RED));
                    }
                    _ => {
                        ui.label(RichText::new("loading").weak());
                    }
                }
                ui.add_space(4.0);
            }
        });
        if let Some((mesh, part)) = toggled {
            let level = self.level.borrow();
            let parts = &level.meshes[mesh].parts;
            match part {
                Some(part) => parts[part].shown.set(!parts[part].shown.get()),
                None => {
                    let hide = parts.iter().any(|part| part.shown.get());
                    for part in parts {
                        part.shown.set(!hide);
                    }
                }
            }
        }
        if let Some((piece, variant)) = picked_variant {
            self.pieces[piece].variant.set(variant);
            self.apply_variant();
        }
        if let Some((group, variant)) = picked_shape {
            {
                let level = self.level.borrow();
                let mut enabled = self.shapes.borrow_mut();
                for (at, _) in &level.groups[group].variants {
                    enabled.remove(&level.shapes[*at].name);
                }
                if let Some(at) = variant {
                    enabled.insert(level.shapes[at].name.clone());
                }
            }
            self.apply();
        }
        if let Some(lod) = picked {
            self.switch(lod);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Source, compose};

    /// `w5341b0001`'s `.imc` names material nought for the one variant it carries, which is the
    /// game stating that the weapon draws no material at all: worn at that variant it contributes
    /// no mesh, and at the base colourway its own meshes are there.
    #[test]
    #[ignore = "reads the real local FFXIV install"]
    fn a_piece_whose_imc_names_no_material_contributes_no_mesh() {
        let path = "chara/weapon/w5341/obj/body/b0001/model/w5341b0001.mdl";
        let install = ironworks::Ironworks::new().with_resource(ironworks::sqpack::SqPack::new(
            ironworks::sqpack::Install::at_sqpack("/home/asriel/.xlcore/ffxiv/game/sqpack"),
        ));
        let bytes: Vec<u8> = install.file(path).expect("the model");
        let worn = |material| -> usize {
            compose(&[Source {
                path: path.to_owned(),
                bytes: bytes.clone(),
                variant: 1,
                material,
                deform: None,
                skin: None,
                rigid: true,
            }])
            .expect("a readable model")
            .level
            .borrow()
            .meshes
            .len()
        };
        assert_eq!(worn(Some(0)), 0, "the imc names no material");
        assert!(worn(Some(1)) > 0, "the base colourway draws");
        assert!(worn(None) > 0, "nothing states a colourway at all");
    }
}
