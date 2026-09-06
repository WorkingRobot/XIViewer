//! A model drawn with the shaders the game would draw it with.
//!
//! The material names a package and a set of keys; the package's node table turns those into the
//! vertex and pixel shader of one pass, and both are translated to GLSL ES 3.00. What the shaders
//! then read is the material's own textures and color table, the package's own parameter buffer, and
//! a camera reconstructed field by field off each buffer's reflection.
//!
//! The G-buffer is five targets and a context is only promised four draw buffers, so the pixel
//! shader is emitted with one page of its outputs at a time: a shader declaring a location the
//! context has no draw buffer for would not link, which makes the split a translation-time choice.

use std::collections::{BTreeSet, HashMap};

use glam::{Mat4, Vec2, Vec3, Vec4};
use ironworks::file::shpk::{self, ShaderPackage, Stage};
use ironworks::file::{mtrl, shcd, spm};

use super::material::{Family, Material};

/// What the semi-transparent buffer pass clips at, held in the shader rather than read off a
/// material. A surface whose own clip stands under this has no band for that pass to draw.
pub const SHEER_CLIP: f32 = 16.0 / 255.0;

const PASS_G_OPAQUE: u32 = 0x03ac_862e;
const PASS_G_SEMITRANSPARENCY: u32 = 0x6006_067f;
const PASS_Z_OPAQUE: u32 = 0xe412_a2d4;
const PASS_LIGHTING_OPAQUE: u32 = 0xfbde_0a8f;
const PASS_COMPOSITE_OPAQUE: u32 = 0x955c_0b73;
const PASS_COMPOSITE_SEMITRANSPARENCY: u32 = 0xc885_bbd3;
/// What a surface that lights itself answers into the frame with. Water resolves through this rather
/// than through the composite pass a glass package takes.
const PASS_LIGHTING_SEMITRANSPARENCY: u32 = 0x1f19_7698;
const PASS_WATER: u32 = 0x8ef4_0d56;

/// The pass with no name of its own, which two packages use for the one thing each of them does:
/// furblur marches along a strand at it, and every cloud node holds it and nothing else.
const PASS_7: u32 = 0x5bc1_ad3f;

/// What the two overlays a zone places answer under, each holding this pass and no other. The
/// second is named for a depth pass and writes colour all the same: which of the two things it does
/// is the technique's own, and the one it defaults to is `Color`.
const PASS_SEMITRANSPARENCY: u32 = 0x2d0c_1a37;
const PASS_WATER_Z: u32 = 0x24cd_f1ea;

/// The render pass a node is selected under. Holding everything else fixed, a drawing package
/// answers `SUB_VIEW_SHADOW_0` with its depth pass alone, which is what a shadow map is.
pub const SUB_VIEW_MAIN: u32 = 0xf43b_2f35;
/// The same view, as the older generation of packages spells it. Nothing declares both.
const MAIN: u32 = 0xa8f9_ffcc;
pub const SUB_VIEW_SHADOW_0: u32 = 0x99b2_2d1c;
pub const SUB_VIEW_CUBE_0: u32 = 0x6624_4231;
pub const SUB_VIEW_ROOF: u32 = 0xae5e_6a42;
pub const SUB_VIEW_MAIN_SELECT: u32 = 0x0c01_20ca;

/// The packages the frame is lit and resolved with, in the order they run, and the pass each is run
/// under. What each reads is what the one before it wrote.
pub const VIEW_POSITION: &str = "shader/sm5/shpk/createviewposition.shpk";
pub const DIRECTIONAL: &str = "shader/sm5/shpk/directionallighting.shpk";
pub const POINT: &str = "shader/sm5/shpk/pointlighting.shpk";
pub const SPOT: &str = "shader/sm5/shpk/spotlighting.shpk";
pub const LINE: &str = "shader/sm5/shpk/linelighting.shpk";
pub const PLANE: &str = "shader/sm5/shpk/planelighting.shpk";
pub const SHADOW: &str = "shader/sm5/shpk/directionalshadow.shpk";
pub const COMPOSITE: &str = "shader/sm5/shpk/bg_composite.shpk";
/// Softens the surface a strand grows out of, between the G-buffer and the light it is read under.
pub const FUR: &str = "shader/sm5/shpk/furblur.shpk";
pub const SCATTER: &str = "shader/sm5/shpk/subsurfaceblur.shpk";

/// The members of the game's post chain the viewer runs. The first reads a table a file holds. The
/// other two smooth the frame's edges, in the order they run: one writes each pixel's brightness
/// into the alpha the next reads its edges off.
pub const TONE_ADJUST: &str = "shader/sm5/posteffect/ToneAdjust.shcd";
pub const FXAA_LUMA: &str = "shader/sm5/posteffect/FXAALuma.shcd";
pub const FXAA: &str = "shader/sm5/posteffect/FXAA.shcd";

/// How bright the frame turned out and what that does to it. The first three halve the frame down
/// to one texel, accumulating the reciprocal of each tap's luminance, so what lands is a harmonic
/// mean; the fourth carries that toward the last frame's rather than jumping to it; the fifth builds
/// the curve as a 1024-wide strip; the last reads the frame through it.
pub const MEASURE_INITIAL: &str = "shader/sm5/posteffect/MeasureLumInitial.shcd";
pub const MEASURE_ITERATIVE: &str = "shader/sm5/posteffect/MeasureLumIterative.shcd";
pub const MEASURE_FINAL: &str = "shader/sm5/posteffect/MeasureLumFinal.shcd";
pub const ADAPT_LUM: &str = "shader/sm5/posteffect/AdaptLum.shcd";
pub const TONE_MAP_LUT: &str = "shader/sm5/posteffect/ToneMapLut.shcd";
pub const TONE_MAPPING: &str = "shader/sm5/posteffect/ToneMapping.shcd";

/// The six of them, for asking after at once.
pub const MEASURE: [&str; 6] = [
    MEASURE_INITIAL,
    MEASURE_ITERATIVE,
    MEASURE_FINAL,
    ADAPT_LUM,
    TONE_MAP_LUT,
    TONE_MAPPING,
];

/// Texels of the curve the tone pass reads, which the buffer states as the half-texel bounds
/// `(0.5/1024, 1 - 0.5/1024)`.
pub const CURVE: i32 = 1024;

/// What spreads the bright end of the frame into a halo, in the order it runs. The first keeps the
/// share of a pixel the composite left in its alpha, wherever that is bright enough to count; the
/// second smooths each level the halo is carried down through; the third spreads it along one axis
/// at a time; the fourth shapes what came back; and the last adds that to the frame.
///
/// The game binds `posteffect/ed8a98cf` for the first, which is this file one generation on with a
/// `sqrt` around the color it reads. That undoes the square the source pass ahead of it takes, and
/// the viewer runs no source pass: what it reads is already square-rooted, so the file without the
/// `sqrt` is what puts the same numbers into the blur.
pub const BRIGHT_PASS: &str = "shader/sm5/posteffect/BrightPassFilter.shcd";
pub const GAUSS_BLUR: &str = "shader/sm5/posteffect/GaussBlur5x5_Linear.shcd";
pub const BLOOM_BLUR: &str = "shader/sm5/posteffect/BloomBlur_Linear.shcd";
pub const GLARE_MERGE: &str = "shader/sm5/posteffect/GlareMerge.shcd";
pub const GLARE_COMPOSITE: &str = "shader/sm5/posteffect/90631568";

/// The vertex shaders the game pairs the two smoothing passes with, which hand them nine and seven
/// coordinates rather than the one every other pass here reads.
pub const SAMPLING_9: &str = "shader/sm5/posteffect/VSSampling9.shcd";
pub const SAMPLING_7: &str = "shader/sm5/posteffect/VSSampling7.shcd";

/// The seven of them, for asking after at once.
pub const GLARE: [&str; 7] = [
    BRIGHT_PASS,
    GAUSS_BLUR,
    BLOOM_BLUR,
    GLARE_MERGE,
    GLARE_COMPOSITE,
    SAMPLING_9,
    SAMPLING_7,
];

/// What the chain runs at, against the frame. The frame is kept at half, the bright pass and the
/// first smoothing stand at a quarter, and the blur that spreads the halo runs at an eighth, which
/// is what settles how far one reaches: the blur takes six texels of whatever it reads.
pub const GLARE_SOURCE: i32 = 2;
pub const GLARE_SCALE: i32 = 4;
pub const GLARE_SPREAD: i32 = 8;

/// Where the blur's six outer taps stand, in texels of what it reads. Its weights are a Gaussian of
/// two texels read at nought through six and paired off, and a pair taken as one bilinear tap sits
/// at the pair's own center of mass rather than on either texel.
const TAPS: [f32; 3] = [1.40737, 3.29421, 5.20181];

/// The same for the smoothing pass, whose weights are a Gaussian of one texel read at nought through
/// two, paired off on each axis and taken as a square of nine bilinear taps.
const GAUSS_TAP: f32 = 1.182_425;

/// What a pixel's glare has to average before any of it spreads, and how dim one has to be for the
/// merge to pull it toward a grey of one and a half times its own mean. Both come off the frames the
/// game drew. The halo goes back over the frame at its own strength, so the blur's threefold gain
/// along each axis stands and nothing takes it back.
pub const GLARE_THRESHOLD: f32 = 2.5 / 255.0;
pub const GLARE_VEIL: f32 = 10.0;

/// What takes the frame's four corners down toward black, which the game draws last of all and over
/// the graded frame. It measures a pixel against the middle of the frame rather than reading the
/// frame at all, so it is the one member of the chain wanting a centered coordinate.
pub const VIGNETTE: &str = "shader/sm5/posteffect/Vignetting.shcd";

/// The sky, drawn over whatever the frame did not cover.
pub const SKY: &str = "shader/sm5/posteffect/Sky.shcd";

/// The volume one is read out of. A sky is a strip a few texels wide by a few dozen tall, stacked
/// once per hour of the day, and the id picks which of them a place stands under.
pub fn sky_texture(id: u16) -> String {
    format!("bgcommon/nature/sky/texture/sky_{id:03}.tex")
}

/// The sun's own glow, drawn over the sky: a screen-wide pass that measures every pixel against
/// where the sun stands and answers a core, six rays and a wide halo.
pub const SUN: &str = "shader/sm5/posteffect/Sun.shcd";

/// Every lane of `cSunParam` past the two the frame decides, read out of the one draw a real frame
/// makes of this pass. The falloffs are in half-frame-heights rather than pixels, so they carry
/// across resolutions: the pass measures its radius after scaling x by the aspect.
///
/// The frame that holds them had the sun far off screen, so the rays and the core are live buffer
/// contents that nothing has been seen to rasterize.
const SUN_RAYS: [f32; 4] = [3.0, 0.965_246_44, 4.983_494_3, 0.499_174_03];
const SUN_FALLOFF: [f32; 4] = [-66.537_23, -92.146_774, -127.613_18, -575.917_3];
const SUN_CORE: [f32; 4] = [
    std::f32::consts::SQRT_2,
    1.0,
    std::f32::consts::FRAC_1_SQRT_2,
    1.0,
];
const SUN_HALO: [f32; 4] = [1.347_855_2, 1.350_965_7, 1.350_469_2, 0.1];

/// The moon, drawn over a disc of its own rather than the frame: the shader reads its coordinate as
/// a place on that disc and throws away what falls outside it.
pub const MOON: &str = "shader/sm5/posteffect/Moon.shcd";

/// That disc's own vertex shader, which stands the screen triangle over the moon and hands each
/// fragment where it falls on it. The far plane, so a depth test keeps it behind everything drawn.
pub const MOON_VERTEX: &str = "\
#version 300 es

layout(location = 0) in vec4 a_position;

uniform vec4 u_disc;

out vec2 TEXCOORD;

void main() {
\tTEXCOORD = a_position.xy;
\tgl_Position = vec4(u_disc.xy + a_position.xy * u_disc.zw, 1.0, 1.0);
}
";

/// `cMoonParam[0].xy`: rolls the quad's own uv into `sMoon`'s. It is the same turn `sun` takes
/// below, before the zone's own tilt splits it into up and flat.
pub fn moon_roll(time: f32) -> Vec2 {
    let (sin, cos) = turned(time).sin_cos();
    Vec2::new(cos, sin)
}

/// How far a day stands from new toward full, `1..=32`: nought at `1`, one at `17`, and back down
/// to nought by `32`. What `cMoonParam[1].w` states outright.
pub fn moon_phase(day: f32) -> f32 {
    1.0 - (day - 17.0).abs() / 16.0
}

/// `cMoonParam[0].zw`: the terminator's own axis, swept through a half turn as the phase runs from
/// new to full.
pub fn moon_terminator(phase: f32) -> Vec2 {
    let theta = (180.0 * phase).to_radians();
    Vec2::new(theta.sin(), -theta.cos())
}

/// `cMoonParam[1].xy`: the slope and offset the pixel shader folds its own `saturate(cos * slope +
/// offset)` shading through, tied by `offset = 1 - 1/slope`. A full or new moon squares the
/// terminator's own plane to the viewer, wanting the edge razored to the visible limb rather than
/// softened, which is what sends the slope toward infinity there.
fn moon_softness(phase: f32) -> (f32, f32) {
    let slope = ((1.0 + phase) / (1.0 - phase).max(1e-6)).min(999_999.0);
    (slope, 1.0 - 1.0 / slope)
}

/// The eight phases Eorzea's calendar names a day under, four days wide apiece.
const MOON_PHASE_NAME: [&str; 8] = [
    "New Moon",
    "Waxing Crescent",
    "Waxing Half Moon",
    "Waxing Gibbous",
    "Full Moon",
    "Waning Gibbous",
    "Waning Half Moon",
    "Waning Crescent",
];

/// What a day, `1..=32`, is called under that calendar.
pub fn moon_phase_name(day: f32) -> &'static str {
    let at = ((day - 1.0) / 4.0).floor().clamp(0.0, (MOON_PHASE_NAME.len() - 1) as f32);
    MOON_PHASE_NAME[at as usize]
}

/// What the weather's stated moon color is taken down by before it tints the disc, and its stated
/// alpha before it weighs the blend. Read off the one frame that states them; a second, independently
/// captured frame carries different figures for both, so these are approximations and not a rule.
const MOON_TINT: f32 = 5.0 / 6.0;
const MOON_WEIGHT: f32 = 0.19;

/// The clouds, which the engine draws over two meshes it builds itself: a band around the horizon
/// and a sheet overhead. One package holds both, under a technique apiece.
pub const CLOUD: &str = "shader/sm5/shpk/cloud.shpk";
pub const CLOUD_BAND: u32 = 0xa2f7_6b97;
pub const CLOUD_SHEET: u32 = 0xd9d5_8038;

/// The subview the sheet answers with the shadow it casts rather than with its own colour, which is
/// what fills the map the sun's lighting reads a cloud out of. Spelled the way this package's own
/// generation spells a subview, so it is none of the ids a drawing package answers to.
const CLOUD_SHADOW_VIEW: u32 = 0x344c_e408;

/// The blur that map is left in, over the four taps the vertex shader the game pairs it with builds.
pub const CLOUD_SHADOW: &str = "shader/sm5/posteffect/CloudShadow.shcd";
pub const CLOUD_SHADOW_VERTEX: &str = "shader/sm5/posteffect/VSSampling4.shcd";

/// How wide it is drawn.
pub const CLOUD_SHADOW_MAP: i32 = 256;

/// The textures each draws, which the environment's cloud set names by id. A sheet of nought is a
/// weather that draws none: no such file exists.
pub fn cloud_texture(id: u16) -> String {
    format!("bgcommon/nature/cloud/texture/cloud_{id:03}.tex")
}

pub fn cloudside_texture(id: u16) -> String {
    format!("bgcommon/nature/cloud/texture/cloudside_{id:03}.tex")
}

/// The night star field's tier 0: a dome the viewer builds itself (see `dome` in `deferred.rs`)
/// rather than a model or a mesh a file holds, and there is no `star.shpk` either - both stages are
/// standalone `.shcd`. The Dawntrail graphics update renamed these from `StarVS0`/`StarPS0`, so only
/// the `_gu` spelling resolves against the live data.
pub const STAR_VERTEX: &str = "shader/sm5/shcd/starvs0_gu.shcd";
pub const STAR_PIXEL: &str = "shader/sm5/shcd/starps0_gu.shcd";

/// The three textures tier 0 reads: per-point colour, the Milky Way band (read squared, this
/// engine's usual sqrt-encoded-colour convention), and a scrolling twinkle mask.
pub const STAR_COLOR: &str = "bgcommon/nature/star/texture/star0.tex";
pub const STAR_BAND: &str = "bgcommon/nature/star/texture/star1.tex";
pub const STAR_TWINKLE: &str = "bgcommon/nature/star/texture/star2.tex";

/// `cWorldMatrix`'s own rotation at hour 3, read off two independently captured night frames of the
/// same zone and byte-identical between them; the translation matches the eye exactly in both.
///
/// It does turn with the time of day: a third capture (Ultima Thule, hour 7) carries a genuinely
/// different rotation, and the transform between the two is not a rotation about a shared axis -
/// its own axis, worked out from the two matrices, is nowhere near vertical. Two hours are not
/// enough to fit whatever the real rule is (a plain turn about a fixed axis is ruled out, not
/// confirmed), so rather than invent one this stays the single measured snapshot: the dome renders
/// correctly at hour 3 and holds a plausible but not time-correct orientation at every other hour.
#[allow(clippy::approx_constant)]
const STAR_ROTATION: [[f32; 3]; 3] = [
    [-0.707_107, 0.0, 0.707_107],
    [-0.704_416, 0.087_156, -0.704_416],
    [-0.061_628, -0.996_195, -0.061_628],
];

/// `cParam[0].xyz`, constant across three independently captured frames of two different zones.
/// `.w` is a scrolling twinkle phase, animated: three captures of the same zone at the same
/// declared hour (byte-identical `cWorldMatrix`) read three different values, so it is driven by
/// [`STAR_TWINKLE_RATE`] rather than the day-night hour, which the same three captures rule out
/// as the driver.
const STAR_PARAM_0: [f32; 3] = [0.000_554, 0.000_985, 1.418_846];

/// Tiles a second the night sky's twinkle mask scrolls by, before wrapping mod one. Read off
/// `ffxiv_dx11.exe` rather than fitted: its own star update accumulates `rate * frame time` into
/// the phase every frame, and `rate` is a flat `1.0` with no other write to that field anywhere
/// in the binary.
pub const STAR_TWINKLE_RATE: f32 = 1.0;

/// `cParam[2].x`: the horizon fade's own scale, `saturate(x * dot(cWorldMatrix.row1, position) + 1)`.
/// Not envb: the exe carries this on the star object itself, next to [`STAR_TWINKLE_RATE`], as a
/// literal `10.0` default and a boolean beside it that zeroes it instead. Nothing in this project
/// places what sets that boolean, so this stays the engine's own default - right for every zone
/// sampled but Ultima Thule, where the flag is set and the term evaluates to `saturate(0 + 1) = 1`
/// everywhere: no fade at all, which is plausible for a zone with no real horizon.
const STAR_HORIZON: f32 = 10.0;

/// `cParam[3]`, read by neither shader tier 0 runs and not even constant across zones (Ultima Thule
/// carries a different `.xy` at the same slot), so nothing here is worth shipping as data.
const STAR_PARAM_3: [f32; 4] = [0.0; 4];

/// `cParam[4].xy`: the scale `sSky` is sampled at off the screen coordinate tier 0's own vertex
/// shader hands down. Measured as the identity in both captures.
const STAR_SKY_SCALE: [f32; 2] = [1.0, 1.0];

/// What the environment's starfield set states at the weather and time the frame stands at: the
/// scales tiers 0-2 take their point mask, Milky Way band and horizon fade by, and the flat alpha
/// tier 0 blends at.
#[derive(Clone, Copy, PartialEq, Default)]
pub struct Star {
    /// `a_intensity`. Unread by tier 0; both instanced tiers multiply it into their own horizon
    /// fade.
    pub horizon: f32,
    /// `b_intensity`, which scales the twinkling point mask.
    pub point: f32,
    /// `c_intensity`, which scales the Milky Way band.
    pub band: f32,
    /// `unknown`, tier 0's flat output alpha.
    pub alpha: f32,
}

impl Star {
    /// The dome's own transform: the measured rotation, recentred on the eye every frame the way
    /// the moon, the sun and both cloud meshes are.
    pub fn placement(eye: Vec3) -> Mat4 {
        let rows = STAR_ROTATION;
        Mat4::from_cols(
            Vec4::new(rows[0][0], rows[1][0], rows[2][0], 0.0),
            Vec4::new(rows[0][1], rows[1][1], rows[2][1], 0.0),
            Vec4::new(rows[0][2], rows[1][2], rows[2][2], 0.0),
            eye.extend(1.0),
        )
    }
}

/// The zone's own grass, which the engine draws off geometry it bakes per grid rather than off any
/// model, and binds with no material naming it.
pub const GRASS: &str = "shader/sm5/shpk/grass.shpk";

/// The technique that fills the channels the default one leaves at nought. Its own vertex shader
/// takes a clip position straight through, since the engine runs it over the pixels grass already
/// covered; paired here with the default vertex shader, which stands those same pixels up.
const GRASS_NORMAL: u32 = 0x5cf2_9b55;

/// `ApplyWavingAnimation`, and the value that sways a blade rather than standing it still. The
/// package's own default is `_Nothing`; `_Shigemi` is a third geometry class this viewer never
/// builds.
const APPLY_WAVING_ANIMATION: u32 = 0x764a_ece7;
const APPLY_WAVING_ANIMATION_AUTO_PLACEMENT: u32 = 0x7dda_17b6;

/// Which way a blade faces, at half of itself plus a half, which is how the channel holding it is
/// read. No file states one.
const GRASS_UP: [f32; 4] = [0.5, 1.0, 0.5, 0.0];

/// The fog, which drags a distant pixel toward the color the weather states and a further one toward
/// the sky itself, and hazes a near one by how much air stands between it and the camera.
///
/// The install ships this under no name: the shader category records 319 files against 318 known
/// names, and the one left over is this. `Fog.shcd` is an older shader that runs in no frame the
/// game draws. Named the way the asset browser shows an unnamed file, and read the same way.
pub const FOG: &str = "shader/sm5/posteffect/e8bf3721";
pub const FOG_DIRECTORY: &str = "shader/sm5/posteffect";

/// Texels of the table it reads the curve out of, which is what its own scale and bias address.
pub const FOG_TABLE: i32 = 256;

/// What the pass reads: the frame's depth, the sky on a plane of its own, and that table.
pub const FOG_DEPTH: &str = "sDepth";
pub const SKY_SAMPLER: &str = "sSky";
pub const FOG_LUT: &str = "sLut";

/// Which repository and category the shader files sit in.
pub const SHADER: (u8, u8) = (0, 5);

/// The index hash a shader path names where its last segment is the hash itself rather than a file
/// name. Several shaders the install ships have no name in the path list, so each is asked for the
/// way the asset browser asks for any file it can only see as a hash.
pub fn unnamed(path: &str) -> Option<u64> {
    let (directory, name) = path.rsplit_once('/')?;
    if name.len() != 8 || !name.bytes().all(|held| held.is_ascii_hexdigit()) {
        return None;
    }
    let (Some(ironworks::sqpack::IndexHash::Split(held)), _) =
        ironworks::sqpack::IndexHash::of(&format!("{directory}/x"))
    else {
        return None;
    };
    Some(held & !0xffff_ffff | u64::from(u32::from_str_radix(name, 16).ok()?))
}

/// The chain the game reflects a frame off itself with, which its own files call a reflection and
/// its shader keys call RLR. The frame's depth becomes a pyramid of its own, a march walks that
/// pyramid for whatever a pixel's surface points at, a blur pair takes the answer down five levels,
/// and a resolve picks a level per pixel and adds it back.
///
/// Four of the files carry names the path list knows. The other five ship under none, so they are
/// named here by the hash their directory records them under and read the same way the asset browser
/// reads any unnamed file; the crc32 of `ReflectionNormalPS.shcd`, `ReflectionBlurXPS.shcd` and
/// `ReflectionBlurYPS.shcd` is three of those hashes, so those three have their names back.
pub const REFLECTION_DIRECTORY: &str = "shader/sm5/shcd";
pub const REFLECTION_VERTEX: &str = "shader/sm5/shcd/ReflectionVS.shcd";
/// The same without the half-texel the rest of the chain reads at, which is what the copy back over
/// the frame is drawn with.
pub const REFLECTION_MERGE_VERTEX: &str = "shader/sm5/shcd/ReflectionMergeVS.shcd";
pub const REFLECTION_NORMAL: &str = "shader/sm5/shcd/38611e75";
pub const REFLECTION_MASK: &str = "shader/sm5/shcd/ReflectionMaskPS.shcd";
pub const REFLECTION_MARCH: &str = "shader/sm5/shcd/621a822b";
pub const REFLECTION_BLUR_X: &str = "shader/sm5/shcd/65887e77";
pub const REFLECTION_BLUR_Y: &str = "shader/sm5/shcd/a9227ee9";
pub const REFLECTION_DISTORT: &str = "shader/sm5/shcd/ReflectionDistortionPS.shcd";
pub const REFLECTION_COPY: &str = "shader/sm5/shcd/ReflectionCopyPS.shcd";

/// Every file the chain takes, for one fetch list.
pub const REFLECTION: [&str; 9] = [
    REFLECTION_VERTEX,
    REFLECTION_MERGE_VERTEX,
    REFLECTION_NORMAL,
    REFLECTION_MASK,
    REFLECTION_MARCH,
    REFLECTION_BLUR_X,
    REFLECTION_BLUR_Y,
    REFLECTION_DISTORT,
    REFLECTION_COPY,
];

/// What the chain reads, by the names its files give them. Two of these mean different things to
/// different members, so what each pass is handed is the pass's own to say.
pub const REFLECTION_DEPTH: &str = "g_SamplerHierarchicalZ";
pub const REFLECTION_FRAME: &str = "g_SamplerReflection";
pub const REFLECTION_BLURRED: &str = "g_SamplerReflectionBlur";
pub const REFLECTION_PLANE: &str = "g_SamplerGBuffer0";
pub const REFLECTION_MASKED: &str = "g_SamplerGBuffer1";

/// How far down the chain runs against the frame, which every buffer of it is sized by.
pub const REFLECTION_SCALE: i32 = 2;

/// How many levels the blur pair takes the marched reflection down, past the one the march wrote.
pub const REFLECTION_LEVELS: i32 = 5;

/// How many levels the pyramid the march walks holds. The pass that builds it writes four and the
/// march caps its own mip at the last of them.
pub const REFLECTION_DEPTHS: i32 = 4;

/// What the environment states the reflection reaches, read whole off a frame the game drew: the
/// view depth it starts fading at, the one past which a pixel is dropped outright, and one over the
/// distance between them.
pub const REFLECTION_FADE: [f32; 4] = [16.0, 32.0, 0.0625, 0.0];

/// What a pixel's own reflectance is scaled by before anything is marched for it, and how rough a
/// surface may be and still be marched at all. Both are the same on every frame measured.
pub const REFLECTION_POWER: f32 = 2.5;
pub const REFLECTION_ROUGHNESS: f32 = 0.8;

/// The chain water reflects itself off, which is not the one above: `river.shpk` and `water.shpk`
/// reach no cube and read a screen-wide `g_SamplerReflectionMap` this fills. The mask stamps a
/// stencil over the water the frame covers, the march walks the same pyramid the frame-wide chain
/// does for what each of those pixels reflects, a blur pair spreads it, a second one spreads it
/// further and the merge picks between the two per pixel.
///
/// The march ships under no name; the crc32 of `WaterRaytracingHighPS.shcd` is the hash its
/// directory records it under, which is how it is asked for.
pub const WATER_MIRROR_VERTEX: &str = "shader/sm5/shcd/WaterReflectionVS.shcd";
pub const WATER_MIRROR_MASK: &str = "shader/sm5/shcd/WaterReflectionMaskPS.shcd";
pub const WATER_MIRROR_MARCH: &str = "shader/sm5/shcd/9402c299";
pub const WATER_MIRROR_BLUR_X: &str = "shader/sm5/shcd/WaterReflectionFirstBlurXVS.shcd";
pub const WATER_MIRROR_BLUR_Y: &str = "shader/sm5/shcd/WaterReflectionFirstBlurYVS.shcd";
pub const WATER_MIRROR_BLUR: &str = "shader/sm5/shcd/WaterReflectionFirstBlurPS.shcd";
pub const WATER_MIRROR_WIDE_X: &str = "shader/sm5/shcd/WaterReflectionSecondBlurXVS.shcd";
pub const WATER_MIRROR_WIDE: &str = "shader/sm5/shcd/WaterReflectionSecondBlurPS.shcd";
pub const WATER_MIRROR_MERGE_VERTEX: &str = "shader/sm5/shcd/WaterReflectionBlurMergeVS.shcd";
pub const WATER_MIRROR_MERGE: &str = "shader/sm5/shcd/WaterReflectionBlurMergePS.shcd";

/// Every file that chain takes, for one fetch list.
pub const WATER_MIRROR: [&str; 10] = [
    WATER_MIRROR_VERTEX,
    WATER_MIRROR_MASK,
    WATER_MIRROR_MARCH,
    WATER_MIRROR_BLUR_X,
    WATER_MIRROR_BLUR_Y,
    WATER_MIRROR_BLUR,
    WATER_MIRROR_WIDE_X,
    WATER_MIRROR_WIDE,
    WATER_MIRROR_MERGE_VERTEX,
    WATER_MIRROR_MERGE,
];

/// The reference the mask stamps and every member after it draws against.
pub const WATER_MIRROR_STENCIL: i32 = 1;

/// The lane past the fog's own start distance that the game uploads and no member of the chain
/// reads.
pub const WATER_MIRROR_UNREAD: f32 = 3500.0;

/// The weights each blur reads its sixteen taps through, eight of them since the kernel is
/// symmetric. A Gaussian of variance twenty-five taken between texels and normalized over the whole
/// kernel, which is the game's own upload to six figures.
fn blur_weights() -> [f32; 8] {
    let held: [f32; 8] = std::array::from_fn(|at| (-(at as f32 + 0.5).powi(2) / 50.0).exp());
    let total = held.iter().sum::<f32>() * 2.0;
    held.map(|weight| weight / total)
}

/// The buffers the chain reads itself out of, and the one every stage of the engine takes the camera
/// from.
const REFLECTION_PARAM: &str = "g_ReflectionParameter";
const SCREEN_PARAM: &str = "g_ScreenParameter";
const CAMERA: &str = "g_CameraParameter";

/// The two passes that stand between the G-buffer and the occlusion read off it: one linearizes the
/// depth and brings the normal into view space, the other packs a square of four of those into the
/// channels of one texel, which is the shape the occlusion pass addresses.
pub const DOWN_SCALE: &str = "shader/sm5/posteffect/DownScaleDepthNormalZ.shcd";
pub const GATHER: &str = "shader/sm5/posteffect/GatherDepthNormalZ.shcd";

/// What each of those runs at, against the frame. The pass that fills the first is named for the
/// scaling and gathers a square of the depth buffer per texel, so this is the factor that makes that
/// gather the two-by-two it is written as.
pub const OCCLUSION_SCALE: i32 = 2;

/// What the occlusion pass reads and over how many taps, at each quality the game ships. Its file
/// is `SSAO` and the place in this list: the four depth-only readings first, then the four that read
/// the normal too, each set running the same taps as the other.
pub const OCCLUDERS: [&str; 8] = [
    "2 taps, depth",
    "6 taps, depth",
    "12 taps, depth",
    "20 taps, depth",
    "2 taps, depth and normal",
    "6 taps, depth and normal",
    "12 taps, depth and normal",
    "20 taps, depth and normal",
];

/// The buffers the exposure chain reads, and the frame, the measure and the table the passes read
/// them through. `cToneMapParam` is shared with the grading pass, which reads the same two lanes as
/// something else entirely: `.z` as how much of its table reaches the frame, `.w` as an exponent.
/// The most of a frame's own measurement the adaptation carries, however long the frame took.
const SETTLE: f32 = 1.0 / 3.0;

const TONE_MAP_PARAM: &str = "cToneMapParam";
const ADAPT_LUM_PARAM: &str = "cAdaptLumParam";
const SKY_PARAM: &str = "cSkyParam";
const SUN_PARAM: &str = "cSunParam";
const MOON_PARAM: &str = "cMoonParam";
const FOG_PARAM: &str = "cFogParam";
const HEIGHT_FOG_PARAM: &str = "cExpHeightFogParam";
const DIRECTIONAL_SHADOW_PARAM: &str = "g_DirectionalShadowParameter";
const SHADOW_BIAS_PARAM: &str = "g_ShadowBiasParameter";

/// The buffers the glare chain reads. The first two carry the threshold and the merge's weights; the
/// third is what the pass that adds the halo back writes into the frame's alpha. The last two are
/// the sampling vertex shader's own, one holding the scale and bias it builds a coordinate with and
/// the other where each of its taps stands.
const BRIGHT_PASS_PARAM: &str = "cBrightPassParam";
const MERGE_WEIGHT: &str = "cMergeWeight";
const SOFT_FOCUS_PARAM: &str = "cSoftFocusParam";
const SAMPLING_PARAM: &str = "cParam";
const SAMPLING_OFFSET: &str = "cSamplingOffset";

/// How far the blur takes the light down where the sheet is at its thickest. The first pair weighs
/// what the lighting reads under the middle of the map and the second what it reads at the edge, and
/// within each the first lane is the diffuse's share and the second the specular's.
const CLOUD_SHADOW_PARAM: &str = "cCloudShadowParam";
const CLOUD_SHADOW_WEIGHTS: [f32; 4] = [0.5, 0.5, 1.0, 0.5];

/// Where the lighting reads that map, which the package names its buffer and its one member alike.
const CLOUD_SHADOW_MATRIX: &str = "g_CloudShadowMatrix";
const VIGNETTING_PARAM: &str = "cVignettingParam";

/// How the shadow resolve softens an edge. The package names its first two levels a single
/// comparison and a nine-tap square; the strongest it leaves unnamed, and that one is a different
/// thing rather than a wider square. It gathers the depths under a disc, sizes a penumbra from how
/// far behind the pixel the blockers it found stand, and filters at that size.
pub const SHADOW_SOFT: u32 = 0xa89d_89f0;
pub const SHADOW_SOFT_PCSS: u32 = 0x2b16_de56;

/// How wide the sun stands: the penumbra a unit of distance between a blocker and what it falls on
/// widens by. Read off the five cascades of a captured frame, exact in each.
const SUN_SOFTNESS: f32 = 0.007;

/// What geometry a pass covers the frame with. The resolve's far-plane variant stands its quad at
/// the second lane of `m_ShadowDistance`, which is where a split stops.
pub const TRANSFORM_PROJ: u32 = 0x0950_0613;
pub const TRANSFORM_PROJ_PLANE_FAR: u32 = 0xd6e2_1545;

/// What power a lamp's distance falls off by, which every lighting package declares and each placed
/// light names for itself: its record's `attenuation` is one, two or three and picks the variant of
/// the same index below.
pub const APPLY_ATTENUATION: u32 = 0x53af_00ed;
pub const ATTENUATION: [u32; 3] = [0x2795_eaa4, 0xe79a_9e9b, 0x4495_a6b1];

/// Whether a lamp's pass drops the pixels standing outside the box its zone clipped it to, which it
/// reads out of `m_ClipMin` and `m_ClipMax` in the same units those are stated in.
/// `ApplyConeAttenuation`, and the value that softens a spot's edge. A spot's package defaults it
/// off, and off is a bare `discard` at the outer cosine: the cone meets the floor as a conic section
/// with no falloff across it at all. The variant it selects works out
/// `(dot(dir, toPixel) - cos(outer)) / (cos(inner) - cos(outer))`, which is the penumbra the two
/// cosines were always written for.
pub const APPLY_CONE_ATTENUATION: u32 = 0x52d2_1d34;
pub const APPLY_CONE_ATTENUATION_ENABLE: u32 = 0xe106_8eed;

pub const LIGHT_CLIP: u32 = 0x7db0_9695;
pub const LIGHT_CLIP_ENABLE: u32 = 0x6f0e_2969;

/// How wide the sun's own map is drawn, which is what a texel of it measures.
pub const SHADOW_MAP: i32 = 2048;
const CLIP_TO_WORLD: &str = "cC2W";

/// What the cloud draws read themselves against. Both are named the way any package names the two
/// buffers its own stages take, so nothing but a cloud pass fills them this way.
const VS_PARAM: &str = "g_VSParam";
const PS_PARAM: &str = "g_PSParam";

/// The buffer a package fills for itself, which the two overlays a zone places read as the only
/// thing the engine tells them beyond the object's own placement. Named the generic way, so nothing
/// but one of their passes fills it this way.
const PARAMETER: &str = "g_Parameter";

/// How far the sheet's texture tiles across the forty thousand units it spans, which puts one period
/// of it every four thousand.
const SHEET_TILING: f32 = 10.0;
const SHEET_SPAN: f32 = 20000.0;
const SHEET_HEIGHT: f32 = 2000.0;
const SHEET_RISE: f32 = 1000.0;

/// The radius the band stands at around the camera.
const BAND_RADIUS: f32 = 2000.0;

/// What the sheet's shadow map is sized to cover: one period of its own texture, so a texel of the
/// map is a fixed share of a cloud however far across the world the sheet is drawn. The near plane
/// is a thousandth of the far one, which is the sheet's own span.
const CLOUD_SHADOW_SPAN: f32 = SHEET_SPAN / SHEET_TILING;
const CLOUD_SHADOW_NEAR: f32 = SHEET_SPAN * 0.001;
const CLOUD_SHADOW_FAR: f32 = SHEET_SPAN;

/// How thin the box may be taken where the light lies along one of the world's own axes.
const CLOUD_SHADOW_FLOOR: f32 = 0.001;

/// How far the view direction is carried toward straight down before a cloud is lit against it, and
/// the alpha the sheet fades toward overhead. One number does both, and it is a single sample: only
/// the sheet's own shaders read it, and only one frame measured holds a sheet.
const CLOUD_FLOOR: f32 = 0.5;
const PROJECTION_INVERSE: &str = "cProjInv";
const VIEW_INVERSE: &str = "cViewInv";
const COMMON_TEX_PARAM: &str = "cCommonTexParam";
pub const POST_INPUT: &str = "sInput";
pub const POST_TABLE: &str = "sLUT";
pub const POST_MEASURE: &str = "sToneMap";
pub const POST_ADAPTED: &str = "sAdaptedLum";
/// What the merge lays over the frame, which is the glare the blur left.
pub const POST_MERGE: &str = "sMerge1";

/// What the grading pass takes of that buffer. Neither lane is stated anywhere: the environment's
/// colour filter set carries a grading beside the tone mapping one, but nothing pairs its fields
/// with these. So the exponent is left where it changes nothing, and the table is left out: three of
/// them ship and nothing states which one binds, so reading one at full strength grades a frame it
/// was never authored over and takes the color out of it. Only that pass reads the buffer this way,
/// which is why it is the only one built with it.
const TONE_MAP: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

/// The buffers the smoothing pass reads. The first is the rectangle of its target the frame was
/// rendered into, which every pass of the chain clamps its reads to; the frame fills the whole of
/// one here, so the corner it names is the far one.
const VIEWPORT_PARAM: &str = "cDynamicViewportResolutionParam";
const FXAA_PARAM: &str = "cFxaaParam";

/// What that pass runs with, read off the game's own upload. The shader takes one less the first
/// for FXAA's subpixel term, and a half is its own complement, so the lane and the term are the
/// same number; the second scales the frame's own contrast into an edge threshold, over a floor the
/// shader carries as `0.0833`.
pub const FXAA_SUBPIX: f32 = 0.5;
pub const FXAA_EDGE: f32 = 0.15;

/// The buffers the occlusion chain reads. The first is what turns a depth buffer reading into the
/// distance in front of the camera it stands for, and the second the rotation that brings a normal
/// out of the world and into the camera's own space. The third is a bare `float4[3]` the reflection
/// gives no member names and no defaults at all.
const VIEW_DEPTH_FACTOR: &str = "cViewDepthFactor";
const VIEW_ROTATION: &str = "cView";
const HDAO_PARAM: &str = "cHDAOParam";

/// What that pass reads past the texel size of what it gathers from, read off the game's own
/// upload since nothing describes the buffer. The accept is a fall measured against the depth it
/// stands at and the four lengths after it are world units in front of the camera, so those carry
/// across whole; the pass reads the reciprocal of all but the reach. The spread is in texels of the
/// gather instead, which transfers only because both sides run that gather at half the frame: the
/// viewer's is [`OCCLUSION_SCALE`], and where the game sizes its own is untraced.
pub const OCCLUSION_SPREAD: f32 = 0.56;
pub const OCCLUSION_ACCEPT: f32 = 0.01;
pub const OCCLUSION_REJECT: f32 = 0.2;
pub const OCCLUSION_NEAR: f32 = 1.0;
pub const OCCLUSION_REACH: f32 = 50.0;
pub const OCCLUSION_BIAS: f32 = 0.1;
pub const OCCLUSION_INTENSITY: f32 = 10.0;
pub const OCCLUSION_POWER: f32 = 0.3;

/// The vertex shader the pass is drawn with. The game pairs these with a `VSSampling`, which reads a
/// quad of positions and coordinates against a scale and a bias no file states; the screen triangle
/// carries its own, and a frame a pass of this graph wrote is already the way round a sampler here
/// reads it.
pub const POST_VERTEX: &str = "\
#version 300 es

layout(location = 0) in vec4 a_position;

out vec2 TEXCOORD;

void main() {
\tTEXCOORD = a_position.xy * 0.5 + 0.5;
\tgl_Position = a_position;
}
";

/// The sky's own, which hands the fragment where it stands in clip space rather than a texture
/// coordinate: the pass unprojects that to find which way the pixel looks. Held at the far plane, so
/// a depth test keeps it behind everything already drawn rather than over it.
pub const SKY_VERTEX: &str = "\
#version 300 es

layout(location = 0) in vec4 a_position;

out vec2 TEXCOORD;

void main() {
\tTEXCOORD = a_position.xy;
\tgl_Position = vec4(a_position.xy, 1.0, 1.0);
}
";

/// The same for the gathering pass, which reads four texels of one square rather than one. The
/// pixel's own texel goes last, since that is the lane the occlusion pass takes its center from, and
/// the other three run round the square so that a lane and the one two along from it stand either
/// side of its middle. That pairing is what the occlusion pass mirrors its taps by.
pub const GATHER_VERTEX: &str = "\
#version 300 es

layout(location = 0) in vec4 a_position;

uniform vec2 u_texel;

out vec4 TEXCOORD;
out vec4 TEXCOORD1;

void main() {
\tvec2 uv = a_position.xy * 0.5 + 0.5;
\tTEXCOORD = vec4(uv + vec2(0.0, u_texel.y), uv + u_texel);
\tTEXCOORD1 = vec4(uv + vec2(u_texel.x, 0.0), uv);
\tgl_Position = a_position;
}
";

/// The same for the passes that halve the frame, which read the four source texels one destination
/// texel covers. The game pairs these with a `VSSampling4`, whose offsets come from a buffer no file
/// states; a halving names its own, since the source texel is half the destination's and the four
/// taps stand at its corners.
pub const SAMPLING_VERTEX: &str = "\
#version 300 es

layout(location = 0) in vec4 a_position;

uniform vec2 u_texel;

out vec4 TEXCOORD;
out vec4 TEXCOORD1;

void main() {
\tvec2 uv = a_position.xy * 0.5 + 0.5;
\tvec2 step = u_texel * 0.25;
\tTEXCOORD = vec4(uv - step, uv + vec2(step.x, -step.y));
\tTEXCOORD1 = vec4(uv + vec2(-step.x, step.y), uv + step);
\tgl_Position = a_position;
}
";

/// `GetDirectionalLight`, and the value that draws a light rather than nothing. The package defaults
/// it to `_Disable`, whose shader writes no light at all.
const GET_DIRECTIONAL_LIGHT: u32 = 0x8115_916d;
const GET_DIRECTIONAL_LIGHT_ENABLE: u32 = 0x51ed_d496;
/// What the shadowed frame will want instead. Left unused until a mask exists to read: drawn with a
/// white stand-in it comes out measurably darker than the enabled variant, and why is not yet known.
const GET_DIRECTIONAL_LIGHT_SHADOW: u32 = 0xd73b_9e89;

/// `ApplyDetailMap`, and the value that lays the tiled detail arrays over a surface. A background
/// package defaults it to `_Disable`, which draws a wall as its own textures and nothing finer.
const APPLY_DETAIL_MAP: u32 = 0x6313_fd87;
const APPLY_DETAIL_MAP_ENABLE: u32 = 0x7a3d_9efd;

/// `SpecularLighting`, and the value that works a specular out rather than moving nought into the
/// target the composite reads it back from. A placed light's package defaults it to `_Disable`.
const SPECULAR_LIGHTING: u32 = 0x0d81_2fa4;
const SPECULAR_LIGHTING_ENABLE: u32 = 0xaba1_f498;

/// Entries the array holds, which is what its own extent divides into twelve registers apiece.
const ENTRIES: usize = 64;

/// Records of `g_ShaderTypeParameter`, which `SV_Target.w` indexes as `(32 + type) / 255`.
const SHADER_TYPES: usize = 256;

/// Dwords of one `g_ShaderTypeParameter` record, and the one the fur pass reads.
const SHADER_TYPE: usize = 32;
const FUR_LENGTH: usize = 12;

/// Where the character family's own profiles start, and what a material with no colour table names
/// its profile with.
const CHARA_TYPES: usize = 32;
const SHADER_ID: u32 = 0x59bd_a0b1;

/// Dwords in a row of the texture a structured buffer is read through, which the backend fixes so
/// that a shader and whatever fills the texture agree without either having to say so.
pub const ROW: usize = hlsl::glsl::ROW as usize;

/// Dwords of one joint's transform: four columns of three floats, densely packed.
const JOINT: usize = 12;

/// The buffer a drawing package reads one record of per object drawn.
const INSTANCING: &str = "g_InstancingData";

/// The buffer a waving shader takes its wind and its sway weight out of.
const WAVING: &str = "g_WavingParam";

/// The buffer holding what the engine decides per object rather than per material.
const INSTANCE: &str = "g_InstanceParameter";

/// What the decal a face paint is drawn through is tinted with. One register named after itself,
/// which a reflection describes as a bare array and hands no fields for a write to land in.
const DECAL: &str = "g_DecalColor";

/// The buffer one placed light is read out of, and the only one that differs from lamp to lamp.
pub const LIGHT: &str = "g_LightParam";

/// Its fields, as every package that reads one by name declares them. `iris.shpk` picks its record
/// out by which eye a vertex belongs to, and a reflection describes a buffer indexed that way as one
/// bare array, so the names have to come from somewhere for a fill to reach it at all.
const INSTANCE_FIELDS: [(&str, u32); 9] = [
    ("m_MulColor", 16),
    ("m_EnvParameter", 16),
    ("m_CameraLight", 32),
    ("m_Wetness", 16),
    ("m_Wind", 16),
    ("m_PrevWind", 16),
    ("m_IrisParam", 32),
    ("m_Param", 16),
    ("m_HeadUpVector", 16),
];

/// The clock every animated package shares, which the engine drives and no file states.
const PBR: &str = "g_PbrParameterCommon";

fn decal_field() -> Vec<hlsl::layout::Member> {
    vec![hlsl::layout::Member {
        name: DECAL.to_owned(),
        offset: 0,
        size: 16,
        kind: "float4".to_owned(),
    }]
}

fn instance_fields() -> Vec<hlsl::layout::Member> {
    INSTANCE_FIELDS
        .iter()
        .scan(0, |offset, (name, size)| {
            let at = *offset;
            *offset += size;
            Some(hlsl::layout::Member {
                name: (*name).to_owned(),
                offset: at,
                size: *size,
                kind: "float4".to_owned(),
            })
        })
        .collect()
}

/// Which pass of the node to take. `Lighting` and `Lamp` take the same one: the sun and a placed
/// light are separate packages reading one buffer differently.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Pass {
    Depth,
    Buffer,
    Blended,
    Lighting,
    Lamp,
    Fur,
    CloudBand,
    CloudSheet,
    /// The sheet again, drawn from the sun's own side into the map the lighting reads a cloud's
    /// shadow out of.
    CloudShadow,
    Composite,
    CompositeBlended,
    /// What a semitransparent surface that lights itself resolves through.
    BlendedLighting,
    /// What water shades itself with, reading the lit frame back rather than filling the G-buffer.
    Water,
    /// A member of the chain that fills what water reads its own reflection through, which reads
    /// the same buffer as the frame-wide one under a layout of its own.
    WaterMirror,
    /// A shaft of light a zone places, added to the frame the lighting left.
    Shaft,
    /// A slab of fog a zone places, blended into that same frame.
    Layer,
    /// The night star dome, standalone shcd like the cloud passes, whose own `cParam` shares a name
    /// with a posteffect pass's.
    Star,
}

impl Pass {
    fn id(self) -> u32 {
        match self {
            Self::Depth => PASS_Z_OPAQUE,
            Self::Buffer => PASS_G_OPAQUE,
            Self::Blended => PASS_G_SEMITRANSPARENCY,
            Self::Lighting | Self::Lamp => PASS_LIGHTING_OPAQUE,
            Self::Fur | Self::CloudBand | Self::CloudSheet | Self::CloudShadow | Self::Star => {
                PASS_7
            }
            Self::Composite => PASS_COMPOSITE_OPAQUE,
            Self::CompositeBlended => PASS_COMPOSITE_SEMITRANSPARENCY,
            Self::BlendedLighting => PASS_LIGHTING_SEMITRANSPARENCY,
            Self::Water | Self::WaterMirror => PASS_WATER,
            Self::Shaft => PASS_SEMITRANSPARENCY,
            Self::Layer => PASS_WATER_Z,
        }
    }
}

/// Where in a vertex an attribute reads from. The mesh supplies every semantic a drawing package
/// asks for, and each shader binds the ones its own signature names.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Field {
    Position,
    Normal,
    Tangent,
    Bitangent,
    Uv,
    Uv1,
    Color,
    Color1,
    Weights,
    Bones,
}

/// What a signature declares an attribute's components as. A draw only validates where the pointer
/// reads with a type of the same class, signedness and all.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Components {
    Float,
    Signed,
    Unsigned,
}

/// One vertex attribute, as the shader's own input signature asks for it.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Attribute {
    pub location: u32,
    pub field: Field,
    pub components: Components,
}

/// What a sampler is declared over. A draw only validates where the texture bound to the unit is of
/// the declaration's own kind, so this is what decides the target it is bound at.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Plane,
    Array,
    Volume,
    Cube,
}

/// A texture the shader samples, named as GLSL has it and identified as the material names it.
pub struct Texture {
    pub name: String,
    /// The package's own resource id, which is the crc a material's samplers use.
    pub id: u32,
    pub kind: Kind,
}

/// A constant buffer the shader binds, and the fields the reflection describes it with.
pub struct Buffer {
    pub name: String,
    members: Vec<hlsl::layout::Member>,
    registers: u32,
    /// What the files decide a buffer holds, worked out once.
    fixed: Option<Vec<u8>>,
}

/// A structured buffer, which GLSL has no such thing as and reads through a texture of dwords.
pub struct Structured {
    pub name: String,
    pub stride: usize,
}

/// One object of a batch, as `g_InstancingData` holds one.
#[derive(Clone, Copy)]
pub struct Instance {
    /// Where the object stands, in world space.
    pub transform: Mat4,
    /// How much sky reaches it, which a zone states per instance in its `.svb`.
    pub sky_visibility: f32,
    /// The colour a scene cycles its material's emissive to and the strength it is taken at, where
    /// one of the scene's animation handlers names the instance.
    pub emissive: Option<Vec4>,
    /// Where a `.ggd` placement starts in the wind cycle, over `0.0..=1.0`. `None` where the
    /// instance carries no such field, which falls back to a position hash.
    pub wind_phase: Option<f32>,
}

impl Default for Instance {
    fn default() -> Self {
        Self {
            transform: Mat4::IDENTITY,
            sky_visibility: 1.0,
            emissive: None,
            wind_phase: None,
        }
    }
}

/// The shape a placed light throws, which is a package of its own reading the one `g_LightParam`
/// differently. A kind whose package a zone has not fetched draws as a point.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum LampKind {
    Point,
    Spot,
    /// A light with length rather than a point, and one with area: the file calls them line and
    /// flat, and each has a package of its own that reads the same buffer differently.
    Line,
    Plane,
}

/// One placed light, as `g_LightParam` reads it. The box is the one a zone's `.lcb` clips the light
/// against: stated in the light's own space, in the same units the placement stands in, so it cuts
/// the volume the light is drawn as without changing how far the light itself carries.
#[derive(Clone, Copy)]
pub struct Lamp {
    /// Takes the light's own space into the world, without scaling it.
    pub placement: Mat4,
    pub min: Vec3,
    pub max: Vec3,
    /// How far the light carries, which is what its falloff is scaled by and where its own pass
    /// drops a pixel.
    pub reach: f32,
    /// Which of its package's falloff variants the light is shaded by, as an index into
    /// `ATTENUATION`. Its record states the power, and the shape it picks is the whole of how far
    /// the light carries into a room.
    pub falloff: usize,
    /// The range its record states, which the falloff is divided by rather than reached to.
    pub range: f32,
    pub color: Vec3,
    pub kind: LampKind,
    /// Which way the light throws, in world space. Its own space points it along positive z: that
    /// is the axis a spot's vertex shader keeps the half of its box on.
    pub direction: Vec3,
    /// The cosine a spot is at full strength within. Nothing but a spot reads it, and a line reads
    /// the same lane as the reciprocal of its own length instead.
    pub inner: f32,
    /// The cosine a spot's cone is cut at, which its own shader compares the direction to a pixel
    /// against. Nothing but a spot reads it.
    pub cone: f32,
}

impl Default for Lamp {
    fn default() -> Self {
        Self {
            placement: Mat4::IDENTITY,
            min: Vec3::splat(-1.0),
            max: Vec3::ONE,
            reach: 1.0,
            falloff: 0,
            range: 1.0,
            color: Vec3::ONE,
            kind: LampKind::Point,
            direction: Vec3::Z,
            inner: 0.0,
            cone: 0.0,
        }
    }
}

/// One volume a zone swaps its own ambient inside: the boxes a roofed part of a town is lit by
/// rather than by the sky over it.
///
/// The shape codes are the file's own - an `EnvShape` is `Ellipsoid = 1`, `Cuboid = 2`,
/// `Cylinder = 3`, and the composite tests for 1 and 3 and takes everything else as the box.
#[derive(Clone, Copy)]
pub struct Volume {
    /// Takes a place in front of the camera into the volume's own space, where it stands as the unit
    /// shape its kind names.
    pub into: Mat4,
    /// How sharply it takes over across each face, in units of its own half extent. The composite
    /// weighs a pixel by this against how far into the volume it stands and drops the volume where
    /// that reaches nought.
    pub fade: Vec3,
    pub shape: f32,
    /// The light inside, which is another place's harmonics rather than the zone's own.
    pub light: [Vec4; 3],
    pub scale: f32,
}

impl Default for Volume {
    fn default() -> Self {
        Self {
            into: Mat4::IDENTITY,
            fade: Vec3::ONE,
            shape: 0.0,
            light: [Vec4::ZERO; 3],
            scale: 1.0,
        }
    }
}

/// The light a place stands in, as `g_AmbientParamArray` holds one entry of it.
///
/// Each set of harmonics is three rows a shader dots against a normal and a one. A zone states its
/// own per time of day in the `.amb` its `EnvLocation` names, the sky's own come out of
/// `skylight.amb`, `scale` is what a zone's `.envb` calls `ambient_light_scale` and the fade's floor
/// what it calls `parameter_1`.
#[derive(Clone)]
pub struct Ambient {
    pub sky: [Vec4; 3],
    /// What the sky's harmonics are taken back up by.
    pub sky_scale: f32,
    pub light: [Vec4; 3],
    pub scale: f32,
    /// How the ambient fades with the depth of the pixel. The composite squares `x * depth + y`
    /// keeping its sign, then clamps that between the floor and one, so a positive ramp leaves the
    /// ambient alone and only the floor bites.
    pub fade: Vec3,
    /// What a sampled reflection is scaled and biased by, and which of the two reflected terms the
    /// frame takes: nought the one weighed by occlusion and `3 * pi / 4`, one the term raw.
    pub reflection: Vec3,
    /// Which cube of the reflection array a place reflects, which the composite reaches as the
    /// slice `0.1 + this`.
    pub capture: f32,
    /// The places inside the zone that light themselves, past the one lighting the whole of it.
    pub volumes: std::sync::Arc<[Volume]>,
}

impl Default for Ambient {
    fn default() -> Self {
        Self {
            // A place with no zone around it still has to state a sky, since that is the only thing
            // a smooth surface has to reflect. Brighter overhead than underfoot, and this viewer's
            // own: nothing on disk states what a model out of any zone stands in.
            sky: [
                Vec4::new(0.0, 0.12, 0.0, 0.26),
                Vec4::new(0.0, 0.13, 0.0, 0.28),
                Vec4::new(0.0, 0.16, 0.0, 0.33),
            ],
            sky_scale: 1.0,
            light: [Vec4::new(0.0, 0.0, 0.0, 0.12); 3],
            scale: 1.0,
            fade: Vec3::new(0.0, 1.0, 0.0),
            reflection: Vec3::X,
            capture: 0.0,
            volumes: std::sync::Arc::from([] as [Volume; 0]),
        }
    }
}

impl Scene {
    /// The planes the projection was built with, which is the only scale the frame states: the
    /// viewer cuts them to the model's own bounding sphere.
    pub fn planes(&self) -> (f32, f32) {
        let (z, w) = (self.projection.z_axis.z, self.projection.w_axis.z);
        (w / z, w / (z + 1.0))
    }
}

impl Ambient {
    /// The harmonics of one channel as the row a normal is dotted against, from the nine
    /// coefficients a file states. The file runs constant, `y`, `z`, `x`, and the row is dotted
    /// against `(normal, 1)`, so the three linear terms are the first three lanes and the constant
    /// the last. What the shader does with the six second-order terms it never reads.
    pub fn row(coefficients: &[f32; 9]) -> Vec4 {
        // Convolved with a cosine lobe rather than handed over raw. Harmonics state the light
        // arriving from each direction; what a surface takes is that gathered over the hemisphere it
        // faces, which weighs the constant term by `pi * Y00` and the three linear ones by
        // `(2pi/3) * Y1m`. Handing them over unscaled leaves the ambient too dark and too flat, by
        // these two factors and by the `2/sqrt(3)` between them.
        const CONSTANT: f32 = 0.886_226_9;
        const LINEAR: f32 = 1.023_326_7;
        Vec4::new(
            coefficients[3] * LINEAR,
            coefficients[1] * LINEAR,
            coefficients[2] * LINEAR,
            coefficients[0] * CONSTANT,
        )
    }
}

/// What the viewer draws with, past what the files decide. Which passes run and how much of a
/// texture is decoded are choices; every constant this once carried now comes out of the game's own
/// upload but the two lanes of the vignette below.
#[derive(Clone, Copy, PartialEq)]
pub struct Look {
    /// The longest edge a model's textures are decoded to, or the file's own where nothing caps it.
    /// Not a shader constant: it decides which mipmap is fetched.
    pub detail: Option<u16>,
    pub antialias: bool,
    pub occlude: bool,
    /// Which of [`OCCLUDERS`] runs, which in the game follows a graphics setting. What this opens
    /// at is a pick.
    pub quality: usize,
    pub bloom: bool,
    pub vignette: bool,
    /// Whether the frame is reflected off itself, which is what a metal surface answers with where
    /// nothing captured an environment for it.
    pub reflect: bool,
    /// Where the corners start darkening, as the squared distance from the middle of the frame with
    /// a corner at one, and how steeply the darkening deepens past that. Both are a guess: no file
    /// states either, and in the game they follow a graphics setting.
    pub onset: f32,
    pub darkening: f32,
}

impl Default for Look {
    fn default() -> Self {
        Self {
            detail: None,
            antialias: true,
            occlude: true,
            quality: 6,
            bloom: true,
            vignette: true,
            reflect: true,
            onset: 0.35,
            darkening: 0.5,
        }
    }
}

impl Look {
    pub fn occluder(&self) -> String {
        let at = self.quality.min(OCCLUDERS.len() - 1) + 1;
        format!("shader/sm5/posteffect/SSAO{at}.shcd")
    }
}

/// What the environment's tone mapping set states at the weather and time the frame stands at, and
/// what the chain reading it answered for the frame before this one. Every field but the last two is
/// the file's own.
#[derive(Clone, Copy)]
pub struct Exposure {
    pub min: f32,
    pub max: f32,
    /// Per second: the buffer holds it scaled by how long a frame took.
    pub rate: f32,
    /// The buffer holds its square.
    pub key: f32,
    /// How much of the curve reaches the frame, and how far the curve bends toward the exposure.
    pub strength: f32,
    pub shoulder: f32,
    pub step: f32,
    /// The exposure `AdaptLum` last answered, which is what the passes here read the frame under.
    pub adapted: f32,
    /// What every pass writing into the lit frame scales its color by and the tone pass divides
    /// back out, which is also where the curve's knee falls. One where nothing divides it back.
    pub encode: f32,
}

/// What the sky pass reads itself against: the hour, which places the sun and picks the slice of the
/// volume the frame stands under, and that volume's own width and height, which fix the coordinates
/// it is read at.
#[derive(Clone, Copy)]
pub struct Sky {
    /// Seconds since midnight.
    pub time: f32,
    /// How far the sun's circle leans, in degrees, which is the zone's own.
    pub tilt: f32,
    pub size: (f32, f32),
    /// Slices the volume holds, which is one an hour however deep it is: the tail of a deeper one
    /// repeats its start so that reading between them wraps past midnight.
    pub depth: f32,
    /// How far the moon's disc reaches up the frame, as a fraction of its height. The one draw that
    /// states it holds 0.00087, which is a disc two and a half pixels across, and that draw covered
    /// no pixels at all, so it is a control rather than a reading.
    pub moon: f32,
    /// What the weather states its moon looks like, and how much of it the hour lets through.
    pub moonlight: Vec4,
    /// How far the disc's own alpha falls off toward its edge, which the weather's starfield set
    /// states beside the moon's color.
    pub moon_fade: f32,
    /// The moon's own day, `1..=32`, which no file states either: a date rather than anything the
    /// hour or the weather derives.
    pub day: f32,
}

/// Where the sun stands, in the coordinates its own pass reads a pixel by. Nothing where it is
/// behind the camera: a direction there projects onto the half of the screen it is not in, and one
/// directly behind lands dead centre, which draws the glow over a sky the sun has left.
pub fn sun_at(scene: &Scene) -> Option<Vec2> {
    let held = scene.sky;
    let at = scene.projection * scene.view * sun(held.time, held.tilt).extend(0.0);
    if at.w <= 0.0 {
        return None;
    }
    let over = at.truncate() / at.w;
    Some(Vec2::new(over.x * 0.5 + 0.5, over.y * 0.5 + 0.5))
}

/// Where the moon's disc stands and how far it reaches, in the coordinates the pass reads a pixel
/// by. Nothing where it is behind the camera, which a projected direction cannot place.
pub fn moon_disc(scene: &Scene) -> Option<Vec4> {
    let held = scene.sky;
    let at = scene.projection * scene.view * moon(held.time, held.tilt).extend(0.0);
    if at.w <= 0.0 {
        return None;
    }
    let over = at.truncate() / at.w;
    let (wide, tall) = scene.size;
    Some(Vec4::new(
        over.x * 0.5 + 0.5,
        over.y * 0.5 + 0.5,
        held.moon * tall / wide,
        held.moon,
    ))
}

/// Which way the moon comes from, which is taken as the far side of the sun's own circle. Measured
/// once, in the only frame that draws one, it stood **5.19 degrees** off that point; one sample
/// cannot say what rule the difference follows, so this is an approximation rather than a reading.
pub fn moon(time: f32, tilt: f32) -> Vec3 {
    -sun(time, tilt)
}

/// Which way the sun comes from at an hour of the day, in world space. It rises due `+x` at six and
/// stands a quarter turn up at noon, and the sky, the clouds and every light that follows it read
/// the same one.
///
/// The circle it runs on leans, and by how much the **zone** states in its own level file. Five
/// captures of four zones, each exact to four decimal places against what the file holds.
pub fn sun(time: f32, tilt: f32) -> Vec3 {
    let turned = turned(time);
    let (flat, up) = tilt.to_radians().sin_cos();
    Vec3::new(turned.cos(), turned.sin() * up, turned.sin() * flat)
}

/// The hour's own turn, before a zone's tilt splits it into up and flat. `moon_roll` reads the
/// same turn straight, with no tilt to split it.
fn turned(time: f32) -> f32 {
    (time / 3600.0 - 6.0) * std::f32::consts::FRAC_PI_2 / 6.0
}

/// What a place with no level file of its own stands under, which is what most zones state.
pub const TILT: f32 = 30.0;

/// How far down the view the sun's own depth maps reach where no level file states it.
pub const SHADOW_REACH: f32 = 400.0;

/// Depth maps the sun draws, into one image as a grid rather than a single column: a column of
/// five tiles at full resolution would ask for a texture taller than WebGL2 guarantees a device
/// can hold. A pixel is read against the nearest whose own box still holds it.
pub const SPLITS: usize = 5;
pub const ATLAS_COLUMNS: usize = 3;
pub const ATLAS_ROWS: usize = SPLITS.div_ceil(ATLAS_COLUMNS);

/// How far down the view the split at `at` reaches, of the whole the sun's maps cover. The game
/// hands each split a share of the whole in the ratio one, two, four and so on doubling, which
/// against the total of all of them is what makes the box after the first always twice the one
/// before.
pub fn shadow_reach(reach: f32, at: usize) -> f32 {
    let whole = (1u32 << SPLITS) - 1;
    let share = (1u32 << (at + 1)) - 1;
    reach * share as f32 / whole as f32
}

/// Where the map's own near clip sits ahead of the first split, and how far back a split's own box
/// reaches into the one before it, so a pixel near the seam is never left with only one to sample.
pub const SHADOW_NEAR: f32 = 0.1;
pub const SHADOW_OVERLAP: f32 = 3.0;

/// Where the split at `at`'s own box starts: the split before its own far bound, or the map's own
/// near clip for the first, pulled back by the overlap.
pub fn shadow_near(reach: f32, at: usize) -> f32 {
    let before = match at {
        0 => SHADOW_NEAR,
        at => shadow_reach(reach, at - 1),
    };
    before - SHADOW_OVERLAP
}

/// How far a face is pushed away from the light before its depth is kept: a slope the map's own
/// step is multiplied by, and a flat push. This is what keeps a surface off its own shadow, since
/// the pass rasterises both of a surface's sides.
pub const SHADOW_SLOPE: f32 = 2.0;
const SHADOW_PUSH: f32 = 3275.0;

/// That flat push for the split at `at`. A step of the map spans more of the world as the split's
/// own box grows, so this is scaled down with it and the push comes to the same distance in the
/// world whichever split it lands in.
pub fn shadow_push(reach: f32, at: usize) -> f32 {
    SHADOW_PUSH / shadow_reach(reach, at)
}

/// Where the sun stands to draw one split of the scene's depth, as a view and an orthographic
/// projection about `focus`. The projection matches the one the frame is drawn with in handing back
/// a nought-to-one depth, which is what the translator's own fixup leaves in the buffer.
pub fn shadow_camera(light: Vec3, view: Mat4, projection: Mat4, reach: f32, at: usize) -> (Mat4, Mat4) {
    // Taken from the frame's own view rather than passed in, so the pass that draws the map and the
    // matrix that reads it cannot be given different boxes.
    let eye = view.inverse().w_axis.truncate();
    let ahead = -view.row(2).truncate().normalize_or(Vec3::Z);
    let near = shadow_near(reach, at);
    let far = shadow_reach(reach, at);
    // The sphere that holds the whole slice of the frame's own frustum this split covers. A box no
    // wider than the split is deep leaves the slice's far corners off the map, and a coordinate off
    // the edge of one band reads the band beside it rather than the split's own.
    let spread = (1.0 / projection.x_axis.x).powi(2) + (1.0 / projection.y_axis.y).powi(2) + 1.0;
    let along = (far + near) * spread * 0.5;
    let reach = (near * near * (spread - 1.0) + (near - along).powi(2)).sqrt();
    let focus = eye + ahead * along;
    let toward = light.normalize_or(Vec3::Y);
    // A light straight overhead leaves the usual up vector parallel to it, and the look-at degenerate.
    let up = match toward.y.abs() > 0.999 {
        true => Vec3::Z,
        false => Vec3::Y,
    };
    let onto = Mat4::orthographic_rh(-reach, reach, -reach, reach, 0.0, reach * 2.0);
    // Snapped to whole texels of the map. The box follows the camera, so without this every step it
    // takes shifts the grid the depth was rasterised on and an edge crawls across its own surface,
    // which reads as a shadow flickering in and out rather than as one standing still.
    let held = Mat4::look_at_rh(focus + toward * reach, focus, up);
    let texel = 2.0 * reach / SHADOW_MAP as f32;
    let seen = held.transform_point3(focus);
    let drift = Vec3::new(
        seen.x - (seen.x / texel).round() * texel,
        seen.y - (seen.y / texel).round() * texel,
        0.0,
    );
    let focus = focus - held.inverse().transform_vector3(drift);
    (Mat4::look_at_rh(focus + toward * reach, focus, up), onto)
}

/// What the environment's light shaft set states at that same weather and time. One set carries both
/// overlays a zone places: a shaft of light takes the first color, and a slab of fog takes the pair
/// as what it looks like at its own surface and far below it, thickening at the stated rate.
///
/// The pairing is the shape of the two shaders rather than anything a file spells out. The set
/// states one more number, `some_param`, which reads one everywhere and which neither shader has a
/// lane for.
#[derive(Clone, Copy, PartialEq, Default)]
pub struct Shaft {
    pub color: Vec3,
    pub radiance: Vec3,
    pub scale: f32,
}

/// What the environment's cloud set states at the weather and time the frame stands at: the two
/// colors a cloud is lit and shaded with, and how far up the band reaches.
#[derive(Clone, Copy, PartialEq)]
pub struct Cloud {
    pub diffuse: Vec3,
    pub ambient: Vec3,
    /// The band's own share of the sky, which the two heights the vertex shader works out sum to.
    pub reach: f32,
}

impl Cloud {
    /// The projection to draw them under: the frame's own, with the far plane pushed past the sheet.
    /// A zone clips at how far it loads, which is a few thousand units, and the sheet is forty
    /// thousand across; moving it costs nothing, since both meshes are held at the far plane
    /// whatever depth they really stand at.
    pub fn frustum(scene: &Scene) -> Mat4 {
        let (near, far) = scene.planes();
        let far = far.max(SHEET_SPAN * 3.0);
        let mut out = scene.projection;
        out.z_axis.z = far / (near - far);
        out.w_axis.z = near * far / (near - far);
        out
    }

    /// Where a cloud mesh stands, which is around the camera rather than in the world: the band is a
    /// cylinder of radius two thousand centred on it, and the sheet a paraboloid forty thousand
    /// across. The sheet is snapped to the grid one period of its texture spans, so that it travels
    /// with the camera without the texture sliding over it.
    pub fn placement(pass: Pass, eye: Vec3) -> Mat4 {
        let column = |x: f32, y: f32, z: f32| glam::Vec4::new(x, y, z, 0.0);
        match pass {
            Pass::CloudBand => Mat4::from_cols(
                column(BAND_RADIUS, 0.0, 0.0),
                column(0.0, BAND_RADIUS + eye.y, 0.0),
                column(0.0, 0.0, BAND_RADIUS),
                glam::Vec4::new(eye.x, -eye.y, eye.z, 1.0),
            ),
            _ => {
                let period = SHEET_SPAN * 2.0 / SHEET_TILING;
                let snap = |held: f32| (held / period).round() * period;
                Mat4::from_cols(
                    column(SHEET_SPAN, 0.0, 0.0),
                    column(0.0, SHEET_HEIGHT, 0.0),
                    column(0.0, 0.0, SHEET_SPAN),
                    glam::Vec4::new(snap(eye.x), SHEET_RISE, snap(eye.z), 1.0),
                )
            }
        }
    }

    /// Where the sun stands to draw the sheet's own shadow, as a view and an orthographic
    /// projection. It looks along the light from where the camera is, turned the shortest way from
    /// the world's own third axis, and its box is what the two horizontal axes throw across the
    /// light: a low sun reaches further along its own heading than a high one. Depth is left to the
    /// sheet's vertex shader, which clamps what would fall outside the planes rather than losing it.
    pub fn shadow_camera(light: Vec3, view: Mat4) -> (Mat4, Mat4) {
        let eye = view.inverse().w_axis.truncate();
        // Taken to the upper half whichever way it was handed in, so the sheet is always overhead.
        let toward = light.normalize_or(Vec3::Y) * light.y.signum();
        let turn = glam::Quat::from_rotation_arc(Vec3::Z, toward);
        let span = |axis: Vec3| {
            CLOUD_SHADOW_SPAN * axis.cross(toward).length().max(CLOUD_SHADOW_FLOOR)
        };
        let (wide, tall) = (span(Vec3::X), span(Vec3::Z));
        (
            Mat4::from_rotation_translation(turn, eye).inverse(),
            Mat4::orthographic_rh(
                -wide,
                wide,
                -tall,
                tall,
                CLOUD_SHADOW_NEAR,
                CLOUD_SHADOW_FAR,
            ),
        )
    }
}

/// White clouds reaching as far as every weather measured has them reach.
impl Default for Cloud {
    fn default() -> Self {
        Self {
            diffuse: Vec3::ONE,
            ambient: Vec3::ONE,
            reach: 0.9,
        }
    }
}

/// The shape every sky the game ships is but one, so a frame whose volume has not arrived still
/// addresses it the way the rest will be addressed.
impl Default for Sky {
    fn default() -> Self {
        Self {
            time: 0.0,
            tilt: TILT,
            size: (8.0, 32.0),
            depth: 24.0,
            moon: 0.050_346,
            moonlight: Vec4::ZERO,
            moon_fade: 0.4,
            day: 17.0,
        }
    }
}

/// What the environment's vertical fog set states at the weather and time the frame stands at. Every
/// field is the file's own, and the two rates are stated per thousand and per seven thousand four
/// hundred units rather than per one.
#[derive(Clone, Copy, PartialEq)]
pub struct Fog {
    /// What a fogged pixel is dragged toward, before the exposure divides it.
    pub color: Vec3,
    /// How opaque it ever gets, which is that color's own alpha.
    pub cap: f32,
    /// How fast the opacity climbs past `start`, and the sky's share past `fade`.
    pub rate: f32,
    pub blend: f32,
    pub start: f32,
    pub fade: f32,

    /// Whether the zone runs the near haze at all, and how far in front of the camera it begins.
    pub haze: f32,
    pub near: f32,
    /// The two layers the haze sums, each thinning away from a height of its own: how fast it
    /// thins, how thick it is at that height, and where that height sits.
    pub layers: [Vec3; 2],
    /// How much of the frame the haze is ever allowed to leave standing.
    pub clear: f32,

    /// What the sun adds to a pixel looking into the haze: its color, how strong, how tightly it
    /// gathers around the sun, and how far out it starts.
    pub glow: Vec3,
    pub glow_strength: f32,
    pub glow_sharpness: f32,
    pub glow_start: f32,
}

/// No fog at all, which is a frame the weather states none for.
impl Default for Fog {
    fn default() -> Self {
        Self {
            color: Vec3::ZERO,
            cap: 0.0,
            rate: 0.0,
            blend: 0.0,
            start: 0.0,
            fade: 0.0,
            haze: 0.0,
            near: 0.0,
            layers: [Vec3::ZERO; 2],
            clear: 0.0,
            glow: Vec3::ZERO,
            glow_strength: 0.0,
            glow_sharpness: 0.0,
            glow_start: 0.0,
        }
    }
}

impl Fog {
    /// Where the table stops changing, which is the later of the two channels' own saturations. One
    /// climbing at nothing never saturates and stands for nothing here.
    pub fn far(&self) -> f32 {
        let held = |from: f32, over: f32, rate: f32| (rate > 0.0).then(|| from + over / rate);
        held(self.start, self.cap, self.rate)
            .into_iter()
            .chain(held(self.fade, 1.0, self.blend))
            .fold(self.start, f32::max)
    }

    /// The table itself, two channels a texel: how opaque the fog is at that distance, and how far
    /// the color it mixes toward has gone from the fog's own to the sky's. The first is linear under
    /// its cap and the second the square of a linear ramp, which is what the game's own tables are.
    pub fn table(&self) -> Vec<f32> {
        let last = FOG_TABLE as f32 - 1.0;
        let span = self.far() - self.start;
        (0..FOG_TABLE)
            .flat_map(|at| {
                let z = self.start + span * at as f32 / last;
                let toward = ((z - self.fade) * self.blend).clamp(0.0, 1.0);
                [
                    ((z - self.start) * self.rate).clamp(0.0, self.cap),
                    toward * toward,
                ]
            })
            .collect()
    }
}

/// What leaves the frame as the composite resolved it: no exposure, and a curve of no strength.
impl Default for Exposure {
    fn default() -> Self {
        Self {
            min: 1.0,
            max: 1.0,
            rate: 0.0,
            key: 1.0,
            strength: 0.0,
            shoulder: 0.0,
            step: 0.0,
            adapted: 1.0,
            encode: 1.0,
        }
    }
}

/// What the frame is scaled by against the root of the exposure the adaptation answered, which is
/// also where the tone curve's knee falls. Every capture reads the same coefficient whatever the
/// zone, the weather or the frame's own luminance.
const ENCODE: f32 = 0.7;

pub fn encode(adapted: f32) -> f32 {
    ENCODE * adapted.max(0.0).sqrt()
}

/// What the engine decides rather than the files. Everything a constant buffer holds that is not the
/// material's own comes from here, so a field that has to be reconstructed is reconstructed once.
#[derive(Clone)]
pub struct Scene {
    pub view: Mat4,
    pub projection: Mat4,
    pub model: Mat4,
    /// The frame in pixels, which is what a screen-wide pass turns a fragment into a texel with.
    pub size: (f32, f32),
    /// Which way the sun comes from, in world space.
    pub light: Vec3,
    /// The light a lamp pass is drawing.
    pub lamp: Lamp,
    /// Which of the sun's depth maps the pass at hand draws or reads, and how far down the view the
    /// whole set of them covers.
    pub split: usize,
    pub reach: f32,
    pub diffuse: Vec3,
    pub specular: Vec3,
    /// What the composite lights a surface with where no light reaches it.
    pub ambient: Ambient,
    /// What the passes past the composite are run with.
    pub look: Look,
    pub exposure: Exposure,
    pub sky: Sky,
    pub fog: Fog,
    pub cloud: Cloud,
    /// What the overlays a zone places carry, where its environment states a set for them.
    pub shaft: Shaft,
    pub star: Star,
    /// The colours the character was made with.
    pub customize: Customize,
    /// How much of the object at hand is drawn, which its dither clip tests each pixel against.
    pub opacity: f32,
    /// Seconds since the viewer opened, which is what every wave and every leaf is a sine of.
    pub clock: f32,
    pub wind: Wind,
    /// How far one tap of a smoothing pass steps, in the coordinate that pass reads.
    pub blur: Blur,
    /// What share of a surface the composite counts as glare.
    pub bloom: Bloom,
    /// What the member of the reflection chain at hand runs at.
    pub reflect: Reflect,
    /// Whether the lighting at hand is reading the buffer a semi-transparent surface filled, which
    /// packs its shader type in a channel of its own.
    pub sheer: bool,
}

/// Which kernel a smoothing pass of the glare chain lays its taps out for. One walks a square of
/// nine around its own texel; the other spreads the halo along a single axis.
#[derive(Clone, Copy)]
pub enum Blur {
    Square(Vec2),
    Along(Vec2),
}

/// What share of a surface's own specular and of what it emits the composite counts as glare, which
/// is what it leaves in the frame's alpha for the bright pass to weigh a pixel by. The weather's
/// own: `g_CommonParameter.m_Misc` takes both out of the wetness set, and three frames measured
/// reproduce from it to six figures.
#[derive(Clone, Copy)]
pub struct Bloom {
    pub specular: f32,
    pub emissive: f32,
}

/// What a model outside a zone is drawn under, since nothing there names an environment. The pair
/// every weather measured holds through the middle of the day.
impl Default for Bloom {
    fn default() -> Self {
        Self {
            specular: 0.04,
            emissive: 0.1,
        }
    }
}

/// Which level of the blurred reflection a pass addresses, and the texels its own taps are stated
/// in, which is the level above the one it writes.
#[derive(Clone, Copy, Default)]
pub struct Reflect {
    pub level: i32,
    pub texel: Vec2,
}

/// One layer of a weather's wind set, as it is stated rather than collapsed into a single heading.
#[derive(Clone, Copy, Default)]
pub struct WindLayer {
    /// Which way this layer leans, in world space.
    pub heading: Vec3,
    /// The reach a texture sample of 1.0 stands for.
    pub max_strength: f32,
    /// The reach a texture sample of 0.0 stands for.
    pub min_strength: f32,
    /// World units the set states for one cycle of this layer's gust, from which `worldScale` is
    /// the plain reading `1.0 / wavelength`. Unconfirmed: the wind texture itself carries several
    /// visible cycles across its own width, so the gust a player actually sees may run coarser
    /// than this by that same factor, and nothing states which the engine intends.
    pub wavelength: f32,
}

/// Radians of phase one sway runs a second. Read off `ffxiv_dx11.exe`: the bg renderer accumulates
/// `frame time * rate` into the phase every frame and wraps it at `2pi`, and `rate` is a field of the
/// environment manager holding a flat `1.0`. A scene can state its own, and which slot of the level
/// header carries it is not placed here, so this is the engine's own default.
pub const WAVING_RATE: f32 = 1.0;

/// Seconds the engine advects a gust texture one whole cycle over, at unit strength. Read off
/// `ffxiv_dx11.exe`: the wind update scales the frame time by `1.0 / 30.0` and steps each layer's
/// `uvOffset` by `heading * max_strength * worldScale` times it, wrapping the pair into `0..1`.
const WIND_SCROLL_INTERVAL: f32 = 30.0;

/// What a layer's stated strength reaches `grass.shpk` at. Read off `ffxiv_dx11.exe`: the renderer
/// keeps it in the slot straight after the wind block and nothing writes it past the constructor.
/// The advection is not taken down by it, only the two the shader leans a blade between.
const WIND_POWER_SCALE: f32 = 0.15;

/// World units a leaf leans by at the far end of one sway. Measured off `m_WindVector` in real
/// frames rather than derived: the reach a wind set sums to is several times this and is not what
/// the engine hands over. Every frame whose `g_WindInfo` holds the calm pair holds this same length
/// and a storm's holds a longer one, so the set does decide it, but the engine gets there by
/// sampling the wind texture itself and leaning each layer by what it reads, and this viewer takes
/// no such read.
pub const WIND_REACH: f32 = 1.467_972;

/// What a character's own wind is capped and scaled by before a strand is swayed along it. Both off
/// `ffxiv_dx11.exe`: the vector is normalised, then scaled by `min(speed, 30) * 0.0005`.
const WIND_SPEED_CAP: f32 = 30.0;
const WIND_SCALE: f32 = 0.0005;

/// Ticks a second the shared animation clock counts, and the mask its accumulator is held to.
const LOOP_TICKS: u16 = 1024;
const LOOP_WRAP: u64 = 0x1f_ffff;

/// What a leaf is swayed by. `bg.shpk`'s `g_WavingParam` is three registers, so `heading` and `reach`
/// hold both wind layers already summed; `grass.shpk`'s `g_WindInfo` keeps a texture-sampled strength
/// per layer instead, which `layers` carries apart for it. A mesh weights the reach by its own stream,
/// which reaches a tenth at most, so the stated strength is already in world units.
#[derive(Clone, Copy)]
pub struct Wind {
    /// Which way a leaf leans, in world space.
    pub heading: Vec3,
    /// What the set's two layers sum to, which the panel shows. A sway itself runs at
    /// [`WIND_REACH`], not at this.
    pub reach: f32,
    /// The two layers `heading` and `reach` are summed from, apart.
    pub layers: [WindLayer; 2],
}

/// What a lone model is shown under, since nothing outside a zone names an environment to take a
/// wind out of. The panel spells all three out.
impl Default for Wind {
    fn default() -> Self {
        let heading = Vec3::new(0.92, 0.0, 0.38);
        Self {
            heading,
            reach: 4.0,
            layers: [WindLayer { heading, max_strength: 2.0, min_strength: 0.0, wavelength: 512.0 }; 2],
        }
    }
}

/// What character creation decides, which no file a model names holds: each is what an albedo is
/// multiplied by or mixed toward. White leaves a texture's own colour where it is, which is what a
/// model outside the character tab is drawn with.
#[derive(Clone, Copy)]
pub struct Customize {
    pub skin: [f32; 4],
    /// A lip tint, whose alpha is the weight it is mixed at rather than an opacity.
    pub lip: [f32; 4],
    pub hair: [f32; 4],
    /// A hair highlight, which a strand is mixed toward by its own mask.
    pub highlight: [f32; 4],
    pub left_eye: [f32; 4],
    pub right_eye: [f32; 4],
    /// What a race feature is tinted with: a limbal ring, an ear tuft, the tattoo the creator names
    /// it after. Not the face paint, which the engine hands its own buffer.
    pub option: [f32; 3],
    /// What the face paint decal is tinted with, and the weight it is laid on at.
    pub decal: [f32; 4],
    /// The face paint itself, which names the texture the engine binds for it.
    pub paint: Option<u16>,
}

impl Default for Customize {
    fn default() -> Self {
        Self {
            skin: [1.0; 4],
            lip: [1.0, 1.0, 1.0, 0.0],
            hair: [1.0; 4],
            highlight: [1.0, 1.0, 1.0, 0.0],
            left_eye: [1.0; 4],
            right_eye: [1.0; 4],
            option: [1.0; 3],
            decal: [1.0, 1.0, 1.0, 0.0],
            paint: None,
        }
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self {
            view: Mat4::IDENTITY,
            projection: Mat4::IDENTITY,
            model: Mat4::IDENTITY,
            size: (1.0, 1.0),
            light: Vec3::Y,
            lamp: Lamp::default(),
            split: 0,
            reach: SHADOW_REACH,
            diffuse: Vec3::ONE,
            specular: Vec3::ONE,
            ambient: Ambient::default(),
            look: Look::default(),
            exposure: Exposure::default(),
            sky: Sky::default(),
            fog: Fog::default(),
            cloud: Cloud::default(),
            shaft: Shaft::default(),
            star: Star::default(),
            customize: Customize::default(),
            opacity: 1.0,
            clock: 0.0,
            wind: Wind::default(),
            blur: Blur::Along(Vec2::ZERO),
            bloom: Bloom::default(),
            reflect: Reflect::default(),
            sheer: false,
        }
    }
}

/// Everything one draw of one material needs, worked out off the files rather than held on the card.
pub struct Program {
    pub vertex: String,
    pub fragment: String,
    pub attributes: Vec<Attribute>,
    pub textures: Vec<Texture>,
    pub buffers: Vec<Buffer>,
    pub structured: Vec<Structured>,
    /// Every target the shader declares, in register order.
    pub outputs: Vec<u32>,
    /// The targets this reading writes, in attachment order: one page of `outputs`.
    pub targets: Vec<u32>,
    /// What each of `outputs` is called.
    pub names: Vec<String>,
    /// Which pass this is, since two packages read the same buffer differently: a sun's attenuation
    /// fades with depth and a lamp's with the square of the distance.
    pub pass: Pass,
}

/// The positional polynomial a package identifies a node by, applied over each group of keys and
/// then over the four results.
fn selector(keys: &[u32]) -> u32 {
    let (mut out, mut mul) = (0u32, 1u32);
    for key in keys {
        out = out.wrapping_add(key.wrapping_mul(mul));
        mul = mul.wrapping_mul(31);
    }
    out
}

/// What a group of keys resolves to: the draw's own value where it sets the category, the material's
/// where it names it, and the package's default otherwise.
fn values(keys: &[shpk::Key], material: &[mtrl::ShaderKey], set: &[(u32, u32)]) -> Vec<u32> {
    keys.iter()
        .map(|key| {
            set.iter()
                .find(|(id, _)| *id == key.id())
                .map(|(_, value)| *value)
                .or_else(|| {
                    material
                        .iter()
                        .find(|held| held.category() == key.id())
                        .map(mtrl::ShaderKey::value)
                })
                .unwrap_or_else(|| key.default_value())
        })
        .collect()
}

/// The shaders this material would draw the pass with, as indices into the package's own list.
fn pair(
    package: &ShaderPackage,
    material: &[mtrl::ShaderKey],
    set: &[(u32, u32)],
    pass: u32,
    technique: u32,
    subview: u32,
) -> Option<(u32, u32)> {
    let mut parts: Vec<u32> = [
        package.system_keys(),
        package.scene_keys(),
        package.material_keys(),
    ]
    .iter()
    .map(|keys| selector(&values(keys, material, set)))
    .collect();
    parts.push(selector(&[technique, subview]));
    let id = selector(&parts);

    // Lookup is by node id, falling back to the alias table: skin and hair only resolve through it.
    let node = package
        .nodes()
        .iter()
        .find(|node| node.id() == id)
        .or_else(|| {
            let alias = package
                .aliases()
                .iter()
                .find(|alias| alias.selector() == id)?;
            package.nodes().get(alias.node() as usize)
        })?;
    let held = node.passes().iter().find(|held| held.id() == pass)?;
    if held.vertex() == shpk::NONE || held.pixel() == shpk::NONE {
        return None;
    }
    // A pass names a shader by its index within its own stage, not within the whole list.
    let base = |want: Stage| {
        package
            .shaders()
            .iter()
            .take_while(|shader| shader.stage() != want)
            .count() as u32
    };
    Some((
        base(Stage::Vertex) + held.vertex(),
        base(Stage::Pixel) + held.pixel(),
    ))
}

/// The two shaders a draw of this material selects, as indices into the package's own list. A
/// package read without its bytecode still carries the tables this reads, so what a draw will
/// translate is known before the blobs it needs are in hand.
pub fn picks(
    package: &ShaderPackage,
    material: &Material,
    set: &[(u32, u32)],
    pass: Pass,
    subview: u32,
) -> Option<(u32, u32)> {
    let held = material.held();
    let technique = package.technique_subview()[0];
    let node = |view| pair(package, held.shader_keys(), set, pass.id(), technique, view);
    // A package of the older generation keys its nodes on `MAIN` rather than on the main subview,
    // and only that one falls back: a shadow request answered by the main node would draw the
    // wrong pass rather than nothing.
    node(subview).or_else(|| (subview == SUB_VIEW_MAIN).then(|| node(MAIN)).flatten())
}

/// One shader's blob, and the program the disassembler read out of it.
fn program<'a>(
    package: &ShaderPackage,
    bytes: &'a [u8],
    index: u32,
) -> Option<(dxbc::shex::Program, &'a [u8])> {
    let shader = package.shaders().get(index as usize)?;
    let start = package.blobs_offset() + usize::try_from(shader.blob_offset()).ok()?;
    let end = start.checked_add(usize::try_from(shader.blob_size()).ok()?)?;
    let blob = bytes.get(start..end)?;
    Some((shex(blob)?, blob))
}

/// The program the disassembler reads out of a blob.
fn shex(blob: &[u8]) -> Option<dxbc::shex::Program> {
    dxbc::scan_dxbc(blob)
        .iter()
        .flat_map(|container| &container.chunks)
        .find_map(|chunk| match chunk.parse() {
            dxbc::chunks::ChunkData::Shader(program) => Some(program),
            _ => None,
        })
}

/// What a blob's own signature chunks declare its inputs and outputs as. A translation without
/// these emits every one of them as a bare register nothing declared.
fn signatures(blob: &[u8], into: &mut hlsl::Names) {
    use dxbc::chunks::ChunkData;

    for chunk in dxbc::scan_dxbc(blob)
        .iter()
        .flat_map(|container| &container.chunks)
    {
        let (held, signature) = match chunk.parse() {
            ChunkData::InputSignature(signature) => (&mut into.inputs, signature),
            ChunkData::OutputSignature(signature) => (&mut into.outputs, signature),
            _ => continue,
        };
        for element in &signature.elements {
            held.entry(element.register).or_insert_with(|| {
                hlsl::Semantic::new(
                    &element.semantic_name,
                    element.semantic_index,
                    element.component_type,
                    element.mask,
                )
            });
        }
    }
}

/// What this shader's registers are called, and what its signatures declare.
fn names(package: &ShaderPackage, index: u32, blob: &[u8]) -> hlsl::Names {
    let mut names = hlsl::Names::default();
    let Some(shader) = package.shaders().get(index as usize) else {
        return names;
    };
    let named = |resource: &shpk::Resource| {
        package
            .name(resource)
            .map(str::to_owned)
            .or_else(|| shaders::names::resolve(resource.id()).map(str::to_owned))
    };
    for resource in shader.textures() {
        if let Some(name) = named(resource) {
            names.textures.insert(resource.slot(), name);
        }
    }
    for resource in shader.samplers() {
        if let Some(name) = named(resource) {
            names.samplers.insert(resource.slot(), name);
        }
    }
    for resource in shader.constants() {
        if let Some(name) = named(resource) {
            names
                .constants
                .insert(resource.slot(), hlsl::Buffer::new(name, Vec::new()));
        }
    }
    signatures(blob, &mut names);
    names
}

/// The buffer layouts a blob's own reflection describes, by name.
fn layouts(blob: &[u8], into: &mut HashMap<String, Vec<hlsl::layout::Member>>) {
    for chunk in dxbc::scan_dxbc(blob)
        .iter()
        .flat_map(|container| &container.chunks)
    {
        if let dxbc::chunks::ChunkData::Rdef(rdef) = chunk.parse() {
            for buffer in &rdef.constant_buffers {
                into.entry(buffer.name.to_string())
                    .or_insert_with(|| hlsl::layout::members(buffer));
            }
        }
    }
}

/// Which field of a vertex a semantic reads from. One the mesh has nothing for is left to the
/// generic attribute, which the draw sets to something the shader can work with.
fn field(semantic: &str) -> Option<Field> {
    Some(match semantic.to_ascii_uppercase().as_str() {
        "POSITION" => Field::Position,
        "NORMAL" => Field::Normal,
        "BINORMAL" => Field::Tangent,
        "TANGENT" => Field::Bitangent,
        "TEXCOORD" => Field::Uv,
        "TEXCOORD1" => Field::Uv1,
        "COLOR" => Field::Color,
        "COLOR1" => Field::Color1,
        "BLENDWEIGHT" => Field::Weights,
        "BLENDINDICES" => Field::Bones,
        _ => return None,
    })
}

/// What target a sampler the translation declared has to be bound at.
fn kind(declaration: &str) -> Kind {
    match declaration {
        "sampler2DArray" | "sampler2DArrayShadow" => Kind::Array,
        "sampler3D" => Kind::Volume,
        "samplerCube" | "samplerCubeShadow" => Kind::Cube,
        _ => Kind::Plane,
    }
}

/// The parameter buffer as this draw sees it: the package's own defaults, with the material's
/// constants written over the spans the package says they occupy.
fn parameters(package: &ShaderPackage, material: &mtrl::Material) -> Vec<u8> {
    let mut out = vec![0u8; package.param_buffer_size() as usize];
    let mut put = |at: usize, values: &[f32]| {
        for (lane, value) in values.iter().enumerate() {
            let offset = at + lane * 4;
            if offset + 4 <= out.len() {
                out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            }
        }
    };
    put(0, package.param_defaults());
    for param in package.material_params() {
        let Some(values) = material
            .constants()
            .iter()
            .find(|held| held.id() == param.id())
            .and_then(|constant| material.constant_values(constant))
        else {
            continue;
        };
        let lanes = param.byte_size() as usize / 4;
        put(
            param.byte_offset() as usize,
            &values[..lanes.min(values.len())],
        );
    }
    // WebGL2 dropped the sampler-object LOD bias `hair.shpk`'s slot-0 sampler otherwise states, so
    // it is added into this term instead: the alpha-discard already reads
    // `g_TextureMipBias + m_MipBias`, and every character-family package declares the field at the
    // same offset. Bias is read off the one normal-map sampler and applied as a material-wide
    // uniform; anisotropy, by contrast, is looked up per texture in `bind()` since raw GL carries it
    // as sampler state rather than a shader parameter.
    let bias = material
        .samplers()
        .iter()
        .find(|sampler| super::material::NORMAL_SAMPLER.contains(&sampler.id()))
        .map(mtrl::Sampler::lod_bias)
        .unwrap_or(0.0);
    if bias != 0.0
        && let Some(param) = package
            .material_params()
            .iter()
            .find(|param| param.id() == shaders::names::hash(b"g_TextureMipBias"))
        && let Some(slot) = out.get_mut(param.byte_offset() as usize..param.byte_offset() as usize + 4)
    {
        let current = f32::from_le_bytes(slot.try_into().expect("four bytes"));
        slot.copy_from_slice(&(current + bias).to_le_bytes());
    }
    out
}

impl Program {
    /// Translates the pair this material would draw with. `target` names a G-buffer channel; the
    /// page holding it is what the fragment shader is emitted with, so a context with four draw
    /// buffers reaches the fifth target through a reading of its own.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        package: &ShaderPackage,
        bytes: &[u8],
        material: &Material,
        set: &[(u32, u32)],
        pass: Pass,
        subview: u32,
        target: usize,
        attachments: usize,
    ) -> Result<Self, String> {
        let pair = picks(package, material, set, pass, subview)
            .ok_or("this material's keys reach no such pass")?;
        Self::assemble(
            package,
            bytes,
            pair,
            Some(material.held()),
            pass,
            target,
            attachments,
        )
    }

    /// Translates a pass of a package no material names: the ones that light and resolve what the
    /// G-buffer holds, which the engine runs over the whole frame rather than over one draw.
    pub fn screen(
        bytes: &[u8],
        pass: Pass,
        attachments: usize,
        keys: &[(u32, u32)],
    ) -> Result<Self, String> {
        let package = ShaderPackage::parse(bytes).map_err(|why| why.to_string())?;
        // A key a package does not declare is never looked up, so one set serves every package here.
        // `keys` is what one package alone asks for, since a key two of them declare would move both.
        let mut set = vec![
            // The shadowed variant, which reads the mask the resolve leaves. Only
            // `directionallighting` declares this key at all, so no other package here moves with
            // it, and the blend it does is `min(mask, fade^2 * (w - mask) + mask)`: at the `w` of
            // one that buffer states, the second term never falls below the mask, so what lands is
            // the mask whatever the cloud term holds.
            (GET_DIRECTIONAL_LIGHT, GET_DIRECTIONAL_LIGHT_SHADOW),
            (SPECULAR_LIGHTING, SPECULAR_LIGHTING_ENABLE),
        ];
        set.extend_from_slice(keys);
        let technique = package.technique_subview()[0];
        let (vs, ps) = pair(&package, &[], &set, pass.id(), technique, SUB_VIEW_MAIN)
            .ok_or("this package reaches no such pass")?;
        Self::assemble(&package, bytes, (vs, ps), None, pass, 0, attachments)
    }

    /// Translates one variant of a package the engine draws with geometry it builds itself, where
    /// the technique picks the variant rather than the package's own default: the cloud package
    /// draws its band and its sheet from two of them, over two different meshes.
    pub fn cloud(bytes: &[u8], pass: Pass, attachments: usize) -> Result<Self, String> {
        let package = ShaderPackage::parse(bytes).map_err(|why| why.to_string())?;
        let technique = match pass {
            Pass::CloudBand => CLOUD_BAND,
            _ => CLOUD_SHEET,
        };
        let subview = match pass {
            Pass::CloudShadow => CLOUD_SHADOW_VIEW,
            _ => package.technique_subview()[1],
        };
        let (vs, ps) = pair(&package, &[], &[], pass.id(), technique, subview)
            .ok_or("the cloud package holds no such technique")?;
        Self::assemble(&package, bytes, (vs, ps), None, pass, 0, attachments)
    }

    /// Translates the star field's tier 0: two standalone `.shcd`, one apiece, over the dome the
    /// engine has nowhere named in a file. Not `assemble`, which wants a package's own node table to
    /// pick a pair out of, and not `sampling`, which stands the fragment over a full-screen triangle
    /// rather than a real mesh: this reads both blobs' own reflection directly and keeps the
    /// attribute list `sampling` has no reason to build.
    pub fn stars(vertex: &[u8], fragment: &[u8]) -> Result<Self, String> {
        let stage = |bytes: &[u8]| -> Result<(dxbc::shex::Program, hlsl::Names, Vec<u8>), String> {
            let code = shcd::ShaderCode::parse(bytes).map_err(|why| why.to_string())?;
            let blob = bytes
                .get(code.blob_offset()..code.blob_offset() + code.blob_size())
                .ok_or("the shader's bytecode runs past the file")?;
            let program = shex(blob).ok_or("no shader in the blob")?;
            let mut names = hlsl::Names::default();
            for (resources, into) in [
                (code.textures(), &mut names.textures),
                (code.samplers(), &mut names.samplers),
            ] {
                for resource in resources {
                    if let Some(name) = code.name(resource) {
                        into.insert(resource.slot(), name.to_owned());
                    }
                }
            }
            for resource in code.constants() {
                if let Some(name) = code.name(resource) {
                    names
                        .constants
                        .insert(resource.slot(), hlsl::Buffer::new(name.to_owned(), Vec::new()));
                }
            }
            signatures(blob, &mut names);
            Ok((program, names, blob.to_vec()))
        };
        let (vertex, vs_names, vs_blob) = stage(vertex)?;
        let (fragment, ps_names, ps_blob) = stage(fragment)?;

        let mut described = HashMap::new();
        layouts(&vs_blob, &mut described);
        layouts(&ps_blob, &mut described);

        let mut extents = hlsl::glsl::extents(&vertex, &vs_names);
        for (name, registers) in hlsl::glsl::extents(&fragment, &ps_names) {
            let held = extents.entry(name).or_insert(0);
            *held = (*held).max(registers);
        }
        let outputs: Vec<u32> = ps_names
            .outputs
            .keys()
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();

        let mut attributes: Vec<Attribute> = vs_names
            .inputs
            .iter()
            .filter_map(|(register, entry)| {
                Some(Attribute {
                    location: *register,
                    field: field(&entry.name)?,
                    components: match entry.kind.as_str() {
                        held if held.starts_with("uint") => Components::Unsigned,
                        held if held.starts_with("int") => Components::Signed,
                        _ => Components::Float,
                    },
                })
            })
            .collect();
        attributes.sort_by_key(|held| held.location);

        let mut textures: Vec<Texture> = Vec::new();
        for (program, names) in [(&vertex, &vs_names), (&fragment, &ps_names)] {
            let declared = hlsl::glsl::declarations(program);
            for (slot, _, name) in hlsl::glsl::textures(program, names) {
                if textures.iter().all(|held| held.name != name) {
                    textures.push(Texture {
                        id: u32::from(slot),
                        name,
                        kind: kind(declared.get(&slot).copied().unwrap_or_default()),
                    });
                }
            }
        }

        let mut structured: Vec<Structured> = Vec::new();
        let mut buffers: Vec<Buffer> = Vec::new();
        for (program, names) in [(&vertex, &vs_names), (&fragment, &ps_names)] {
            for (name, stride) in hlsl::glsl::buffers(program, names) {
                if structured.iter().all(|held| held.name != name) {
                    structured.push(Structured { name, stride: stride as usize });
                }
            }
            for (name, registers) in hlsl::glsl::extents(program, names) {
                // Filled to whichever stage declares the most of it, which is the extent both are
                // spelled at. Taking the first stage's leaves the other reading nought past it.
                if let Some(held) = buffers.iter_mut().find(|held| held.name == name) {
                    held.registers = held.registers.max(registers);
                    continue;
                }
                buffers.push(Buffer {
                    members: described.get(&name).cloned().unwrap_or_default(),
                    name,
                    registers,
                    fixed: None,
                });
            }
        }

        let vs_options = hlsl::glsl::Options { targets: Vec::new(), extents: extents.clone() };
        let ps_options = hlsl::glsl::Options { targets: outputs.clone(), extents };

        Ok(Self {
            vertex: hlsl::glsl(&vertex, &vs_names, hlsl::Reading::Plain, &vs_options)
                .lines
                .join("\n"),
            fragment: hlsl::glsl(&fragment, &ps_names, hlsl::Reading::Plain, &ps_options)
                .lines
                .join("\n"),
            attributes,
            textures,
            buffers,
            structured,
            names: outputs.iter().map(|at| format!("SV_Target{at}")).collect(),
            targets: outputs.clone(),
            outputs,
            pass: Pass::Star,
        })
    }

    /// Translates one of the two readings the zone's grass is drawn with. The default node writes
    /// the albedo off the color map and leaves the normal at nought, so the second stands over the
    /// pixels it kept and fills the rest of the channels.
    pub fn grass(
        bytes: &[u8],
        normal: bool,
        target: usize,
        attachments: usize,
    ) -> Result<Self, String> {
        let package = ShaderPackage::parse(bytes).map_err(|why| why.to_string())?;
        let [technique, subview] = package.technique_subview();
        // The default node stands every blade still; only the AutoPlacement variant reads a wind.
        let set = [(APPLY_WAVING_ANIMATION, APPLY_WAVING_ANIMATION_AUTO_PLACEMENT)];
        let held = |technique| pair(&package, &[], &set, Pass::Buffer.id(), technique, subview);
        let (vs, ps) = held(technique).ok_or("the grass package holds no default node")?;
        let ps = match normal {
            true => held(GRASS_NORMAL).ok_or("the grass package holds no such technique")?.1,
            false => ps,
        };
        Self::assemble(
            &package,
            bytes,
            (vs, ps),
            None,
            Pass::Buffer,
            target,
            attachments,
        )
    }

    /// Translates one member of the game's post chain. A `.shcd` holds one shader and no node table,
    /// so the file is the variant and there is nothing to select; what it wants is a screen-wide
    /// draw of the vertex shader given, and a frame in the range a screen holds, since the pass that
    /// grades one saturates what it reads before it reads its table. The path is taken because two
    /// members read the same buffer as different things.
    pub fn posteffect(path: &str, bytes: &[u8], vertex: &str) -> Result<Self, String> {
        Self::effect(path, bytes, vertex, &HashMap::new())
    }

    /// The same, where the stage it is drawn with declares blocks of its own: GLSL links a block by
    /// name and rejects a pair whose two spellings of one differ, so each stage is written at the
    /// extent both of them reach.
    fn effect(
        path: &str,
        bytes: &[u8],
        vertex: &str,
        shared: &HashMap<String, u32>,
    ) -> Result<Self, String> {
        let code = shcd::ShaderCode::parse(bytes).map_err(|why| why.to_string())?;
        let blob = bytes
            .get(code.blob_offset()..code.blob_offset() + code.blob_size())
            .ok_or("the shader's bytecode runs past the file")?;
        let fragment = shex(blob).ok_or("no shader in the blob")?;

        let mut names = hlsl::Names::default();
        for (resources, into) in [
            (code.textures(), &mut names.textures),
            (code.samplers(), &mut names.samplers),
        ] {
            for resource in resources {
                if let Some(name) = code.name(resource) {
                    into.insert(resource.slot(), name.to_owned());
                }
            }
        }
        for resource in code.constants() {
            if let Some(name) = code.name(resource) {
                names.constants.insert(
                    resource.slot(),
                    hlsl::Buffer::new(name.to_owned(), Vec::new()),
                );
            }
        }
        signatures(blob, &mut names);

        let outputs: Vec<u32> = names
            .outputs
            .keys()
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let mut extents = hlsl::glsl::extents(&fragment, &names);
        for (name, registers) in shared {
            let held = extents.entry(name.clone()).or_default();
            *held = (*held).max(*registers);
        }
        let options = hlsl::glsl::Options {
            targets: outputs.clone(),
            extents: extents.clone(),
        };

        let declared = hlsl::glsl::declarations(&fragment);
        let textures = hlsl::glsl::textures(&fragment, &names)
            .into_iter()
            .filter_map(|(slot, _, name)| {
                let resource = code.textures().iter().find(|held| held.slot() == slot)?;
                Some(Texture {
                    name,
                    id: resource.id(),
                    kind: kind(declared.get(&slot).copied().unwrap_or_default()),
                })
            })
            .collect();
        let buffers = extents
            .into_iter()
            .map(|(name, registers)| Buffer {
                fixed: (name == TONE_MAP_PARAM && path == TONE_ADJUST).then(|| {
                    TONE_MAP
                        .iter()
                        .flat_map(|held| held.to_le_bytes())
                        .collect()
                }),
                name,
                members: Vec::new(),
                registers,
            })
            .collect();

        Ok(Self {
            vertex: vertex.to_owned(),
            fragment: hlsl::glsl(&fragment, &names, hlsl::Reading::Plain, &options)
                .lines
                .join("\n"),
            attributes: Vec::new(),
            textures,
            buffers,
            structured: hlsl::glsl::buffers(&fragment, &names)
                .into_iter()
                .map(|(name, stride)| Structured {
                    name,
                    stride: stride as usize,
                })
                .collect(),
            names: outputs.iter().map(|at| format!("SV_Target{at}")).collect(),
            targets: outputs.clone(),
            outputs,
            pass: Pass::Composite,
        })
    }

    /// The same, drawn with the vertex shader the game pairs it with rather than one the viewer
    /// wrote. A pass reading more of its source than the pixel under it declares its coordinates as
    /// varyings, and the two stages have to spell those identically for the program to link at all,
    /// which is what taking the file rather than writing one gets.
    pub fn sampling(path: &str, bytes: &[u8], vertex: &[u8]) -> Result<Self, String> {
        let code = shcd::ShaderCode::parse(vertex).map_err(|why| why.to_string())?;
        let blob = vertex
            .get(code.blob_offset()..code.blob_offset() + code.blob_size())
            .ok_or("the vertex shader's bytecode runs past the file")?;
        let program = shex(blob).ok_or("no shader in the blob")?;
        let mut names = hlsl::Names::default();
        for resource in code.constants() {
            if let Some(name) = code.name(resource) {
                names.constants.insert(
                    resource.slot(),
                    hlsl::Buffer::new(name.to_owned(), Vec::new()),
                );
            }
        }
        signatures(blob, &mut names);
        let extents = hlsl::glsl::extents(&program, &names);
        let mut held = Self::effect(path, bytes, "", &extents)?;
        // What a member drawn over geometry rather than over a quad reads a vertex through. A quad
        // pass carries a layout of its own and never looks at these.
        held.attributes = names
            .inputs
            .iter()
            .filter_map(|(register, entry)| {
                Some(Attribute {
                    location: *register,
                    field: field(&entry.name)?,
                    components: match entry.kind.as_str() {
                        held if held.starts_with("uint") => Components::Unsigned,
                        held if held.starts_with("int") => Components::Signed,
                        _ => Components::Float,
                    },
                })
            })
            .collect();
        held.attributes.sort_by_key(|held| held.location);
        for (name, registers) in extents {
            match held.buffers.iter_mut().find(|buffer| buffer.name == name) {
                Some(buffer) => buffer.registers = buffer.registers.max(registers),
                None => held.buffers.push(Buffer {
                    name,
                    members: Vec::new(),
                    registers,
                    fixed: None,
                }),
            }
        }
        // Written again now the fragment's own blocks are known: the two stages have to spell a
        // block at the same extent for the program to link.
        held.vertex = hlsl::glsl(
            &program,
            &names,
            hlsl::Reading::Plain,
            &hlsl::glsl::Options {
                targets: Vec::new(),
                extents: held
                    .buffers
                    .iter()
                    .map(|buffer| (buffer.name.clone(), buffer.registers))
                    .collect(),
            },
        )
        .lines
        .join("\n");
        Ok(held)
    }

    fn assemble(
        package: &ShaderPackage,
        bytes: &[u8],
        (vs, ps): (u32, u32),
        material: Option<&mtrl::Material>,
        pass: Pass,
        target: usize,
        attachments: usize,
    ) -> Result<Self, String> {
        let (vertex, vs_blob) =
            program(package, bytes, vs).ok_or("no vertex shader in the blob")?;
        let (fragment, ps_blob) =
            program(package, bytes, ps).ok_or("no pixel shader in the blob")?;
        let vs_names = names(package, vs, vs_blob);
        let ps_names = names(package, ps, ps_blob);
        let mut described = HashMap::new();
        layouts(vs_blob, &mut described);
        layouts(ps_blob, &mut described);

        // A uniform block has to be spelled identically in both stages or the program will not link,
        // and the two disagree on the extent of a shared buffer more often than not.
        let mut extents = hlsl::glsl::extents(&vertex, &vs_names);
        for (name, registers) in hlsl::glsl::extents(&fragment, &ps_names) {
            let held = extents.entry(name).or_insert(0);
            *held = (*held).max(registers);
        }

        let outputs: Vec<u32> = ps_names
            .outputs
            .keys()
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let attachments = attachments.max(1);
        let page = target / attachments;
        let held: Vec<u32> = outputs
            .chunks(attachments)
            .nth(page)
            .unwrap_or_default()
            .to_vec();
        // A reading that skips a channel still has to name it. A draw buffer list can only point the
        // nth output at the nth attachment, so where the outputs do not run straight along the page
        // they are declared at the channels they fill and the rest are left out of the list.
        let targets: Vec<u32> = match held
            .iter()
            .zip(page * attachments..)
            .all(|(target, at)| *target as usize == at)
        {
            true => held,
            false => (page * attachments..(page + 1) * attachments)
                .map(|at| at as u32)
                .collect(),
        };

        let vs_options = hlsl::glsl::Options {
            targets: Vec::new(),
            extents: extents.clone(),
        };
        let ps_options = hlsl::glsl::Options {
            targets: targets.clone(),
            extents,
        };
        let read = |program, names, options| {
            hlsl::glsl(program, names, hlsl::Reading::Plain, options)
                .lines
                .join("\n")
        };

        let mut attributes: Vec<Attribute> = vs_names
            .inputs
            .iter()
            .filter_map(|(register, entry)| {
                Some(Attribute {
                    location: *register,
                    field: field(&entry.name)?,
                    components: match entry.kind.as_str() {
                        held if held.starts_with("uint") => Components::Unsigned,
                        held if held.starts_with("int") => Components::Signed,
                        _ => Components::Float,
                    },
                })
            })
            .collect();
        attributes.sort_by_key(|held| held.location);

        // Bound by the package's own resource id rather than by slot: the same name sits at
        // different slots across variants of one package.
        let mut textures: Vec<Texture> = Vec::new();
        for (shader, program, names) in [(vs, &vertex, &vs_names), (ps, &fragment, &ps_names)] {
            let resources = package
                .shaders()
                .get(shader as usize)
                .map(shpk::Shader::textures)
                .unwrap_or_default();
            let declared = hlsl::glsl::declarations(program);
            for (slot, _, name) in hlsl::glsl::textures(program, names) {
                let Some(resource) = resources.iter().find(|held| held.slot() == slot) else {
                    continue;
                };
                if textures.iter().all(|held| held.name != name) {
                    textures.push(Texture {
                        name,
                        id: resource.id(),
                        kind: kind(declared.get(&slot).copied().unwrap_or_default()),
                    });
                }
            }
        }

        let parameters = material.map(|held| parameters(package, held));
        let mut structured: Vec<Structured> = Vec::new();
        let mut buffers: Vec<Buffer> = Vec::new();
        for (program, names) in [(&vertex, &vs_names), (&fragment, &ps_names)] {
            for (name, stride) in hlsl::glsl::buffers(program, names) {
                if structured.iter().all(|held| held.name != name) {
                    structured.push(Structured {
                        name,
                        stride: stride as usize,
                    });
                }
            }
            for (name, registers) in hlsl::glsl::extents(program, names) {
                // Filled to whichever stage declares the most of it, which is the extent both are
                // spelled at. Taking the first stage's leaves the other reading nought past it.
                if let Some(held) = buffers.iter_mut().find(|held| held.name == name) {
                    held.registers = held.registers.max(registers);
                    continue;
                }
                let fixed = (name == "g_MaterialParameter")
                    .then(|| parameters.clone())
                    .flatten();
                // A buffer holding one bare array named after itself is described with no fields
                // at all, and a fill by field name against that lands nowhere and says nothing.
                let members = match described.get(&name) {
                    Some(held) if !held.is_empty() => held.clone(),
                    _ if name == INSTANCE => instance_fields(),
                    _ if name == DECAL => decal_field(),
                    _ => Vec::new(),
                };
                buffers.push(Buffer {
                    members,
                    name,
                    registers,
                    fixed,
                });
            }
        }

        let names = outputs
            .iter()
            .map(|register| {
                ps_names
                    .outputs
                    .get(register)
                    .map_or_else(|| format!("SV_Target{register}"), |held| held.name.clone())
            })
            .collect();

        Ok(Self {
            vertex: read(&vertex, &vs_names, &vs_options),
            fragment: read(&fragment, &ps_names, &ps_options),
            attributes,
            textures,
            buffers,
            structured,
            outputs,
            targets,
            names,
            pass,
        })
    }

    /// Where in this reading's attachments the wanted target landed.
    pub fn attachment(&self, target: usize) -> Option<usize> {
        let register = self.outputs.get(target)?;
        self.targets.iter().position(|held| held == register)
    }

    /// The buffer this pass reads one record of per object drawn, and how many records it holds.
    pub fn instancing(&self) -> Option<(&Buffer, usize)> {
        self.buffers
            .iter()
            .map(|buffer| (buffer, buffer.instances()))
            .find(|(_, count)| *count > 1)
    }

    /// How many objects one draw of this pass covers. A package with no instancing buffer draws one.
    pub fn batch(&self) -> usize {
        self.instancing().map_or(1, |(_, count)| count)
    }
}

impl Buffer {
    /// Bytes of one record, where the reflection describes one and the buffer holds many.
    fn stride(&self) -> u32 {
        self.members
            .iter()
            .map(|member| member.offset + member.size)
            .max()
            .unwrap_or(0)
            .max(16)
            .div_ceil(16)
            * 16
    }

    /// How many objects one draw covers, which is the instancing buffer's own extent over the
    /// record the reflection describes.
    pub fn instances(&self) -> usize {
        match self.name == INSTANCING {
            true => (self.registers * 16 / self.stride()).max(1) as usize,
            false => 1,
        }
    }

    /// The bytes this buffer holds, filled by field name off the reflection. What the files decide
    /// is worked out once; everything else is the camera, the objects being drawn and the light this
    /// pass carries, and whatever nothing names stays zero.
    pub fn fill(&self, scene: &Scene, pass: Pass, instances: &[Instance]) -> Vec<u8> {
        let Scene {
            view,
            projection,
            model,
            size,
            clock,
            ..
        } = *scene;
        let span = self
            .members
            .iter()
            .map(|member| member.offset + member.size)
            .max()
            .unwrap_or(0)
            .max(self.registers * 16)
            .max(16);
        let mut out = vec![0u8; span.div_ceil(16) as usize * 16];
        if let Some(fixed) = &self.fixed {
            let end = fixed.len().min(out.len());
            out[..end].copy_from_slice(&fixed[..end]);
            return out;
        }

        // A matrix reads as its rows, since a register of the buffer is a row and the machine takes
        // a dot product against one.
        let rows = |matrix: Mat4, count: usize| -> Vec<f32> {
            matrix.transpose().to_cols_array()[..count * 4].to_vec()
        };
        // Aimed at one buffer, since the same name means different things in two of them: a light's
        // diffuse color is also what skin under a stocking is multiplied by.
        let mut put = |buffer: &str, name: &str, values: Vec<f32>| {
            if self.name != buffer {
                return;
            }
            let Some(member) = self.members.iter().find(|held| held.name == name) else {
                return;
            };
            // A field the reflection calls a dword is read back through the bit pattern, so a whole
            // number goes in as one rather than as the float that reads the same.
            let whole = member.kind == "dword" || member.kind.starts_with("uint");
            // The same name is declared at different extents across packages, so a write is cut to
            // the one this buffer states: anything past it is the next field along.
            let end = out.len().min((member.offset + member.size) as usize);
            for (at, value) in values.iter().enumerate() {
                let offset = member.offset as usize + at * 4;
                let bits = match whole {
                    true => (*value as u32).to_le_bytes(),
                    false => value.to_le_bytes(),
                };
                if offset + 4 <= end {
                    out[offset..offset + 4].copy_from_slice(&bits);
                }
            }
        };
        if self.name == INSTANCING {
            self.instancing(scene, instances, &mut out);
            return out;
        }
        // A slab of fog is stood by a translation and nothing else: its own vertices arrive turned
        // and scaled the way the zone placed them, and the buffer it reads holds one register.
        if self.name == INSTANCE && pass == Pass::Layer {
            let held = model.w_axis;
            write(&mut out, 0, &[held.x, held.y, held.z, 0.0]);
            return out;
        }
        if self.name == PARAMETER && matches!(pass, Pass::Shaft | Pass::Layer) {
            let held = scene.shaft;
            match pass {
                // Which way the light the shaft carries travels, which is the way back from where
                // the sun stands, and the color it carries where it points along it.
                Pass::Shaft => {
                    let aim = -scene.light.normalize_or_zero();
                    write(&mut out, 0, &[aim.x, aim.y, aim.z, clock]);
                    write(&mut out, 1, &[held.radiance.x, held.radiance.y, held.radiance.z, 0.0]);
                }
                // Where the eye stands, which the slab measures its own texture off, and the pair
                // of colors it is read between: the first at its own surface, the second far under
                // it. The set's own scale is what the depth below the surface is thickened at.
                _ => {
                    let eye = view.inverse().w_axis;
                    write(&mut out, 0, &[eye.x, eye.y, eye.z, clock]);
                    write(&mut out, 1, &[held.color.x, held.color.y, held.color.z, held.scale]);
                    write(&mut out, 2, &[held.radiance.x, held.radiance.y, held.radiance.z, 0.0]);
                }
            }
            return out;
        }
        if self.name == "g_AmbientParamArray" {
            ambient(&scene.ambient, glam::Mat3::from_mat4(view), view, &mut out);
            return out;
        }
        if self.name == "g_AmbientParam" {
            entry(&scene.ambient, glam::Mat3::from_mat4(view), &mut out, 0);
            return out;
        }
        if self.name == "g_BGAmbientParameter" {
            write(&mut out, 0, &BG_AMBIENT);
            return out;
        }
        if self.name == VIEWPORT_PARAM {
            write(&mut out, 0, &[1.0; 4]);
            return out;
        }
        // Where the grid a blade belongs to stands, which its own placements are measured from. The
        // last lane reaches nothing the readings the scene runs go on to look at.
        if self.name == "g_GrassGridParam" {
            let held = model.w_axis;
            write(&mut out, 0, &[held.x, held.y, held.z, 0.0]);
            return out;
        }
        // The phase grass.shpk adds to a blade's own `COLOR1.y` before the sine, in the same radians
        // as everything else the wind turns. No previous-frame history is tracked, so the buffer
        // meant for one reads the same value; the shader only reads it for a motion vector this
        // viewer does not draw.
        if self.name == "g_WindParam" || self.name == "g_PreviousWindParam" {
            write(&mut out, 0, &[0.0, 0.0, 0.0, scene.clock * WAVING_RATE]);
            return out;
        }
        // Two layers, each a heading, and a strength between `windPowerMin` and `windPowerMin +
        // windPower * sample^2`, where `sample` is a texel of `wind_0{1,2}.tex` at `worldPos.xz *
        // worldScale - uvOffset`. `worldScale` is the plain reading of the file's own stated cycle
        // length. The pair the blade leans between is the stated range rather than the stated
        // strength, both taken down by [`WIND_POWER_SCALE`]. The engine advects `uvOffset` along the
        // layer's heading by its own strength, so a stronger wind runs the field past faster; it
        // wraps the pair every cycle, which keeps the coordinate exact however long the clock has
        // run. `windViewDir` is read by no vertex shader this viewer runs, so it is left at nought.
        if self.name == "g_WindInfo" {
            for (at, layer) in scene.wind.layers.iter().enumerate() {
                let world_scale = match layer.wavelength > 0.0 {
                    true => 1.0 / layer.wavelength,
                    false => 0.0,
                };
                let carried =
                    layer.max_strength * world_scale * scene.clock / WIND_SCROLL_INTERVAL;
                let offset = (Vec2::new(layer.heading.x, layer.heading.z) * carried)
                    .to_array()
                    .map(|held| held.rem_euclid(1.0));
                let offset = Vec2::from_array(offset);
                for base in [at * 3, at * 3 + 6] {
                    write(&mut out, base, &[
                        layer.heading.x,
                        layer.heading.y,
                        layer.heading.z,
                        (layer.max_strength - layer.min_strength) * WIND_POWER_SCALE,
                    ]);
                    write(&mut out, base + 1, &[
                        offset.x,
                        offset.y,
                        world_scale,
                        layer.min_strength * WIND_POWER_SCALE,
                    ]);
                }
            }
            return out;
        }
        // A whole matrix, one register per row, for a buffer the reflection gives no members.
        let write_rows = |out: &mut Vec<u8>, held: &[f32]| {
            for (at, row) in held.chunks(4).enumerate() {
                write(out, at, row);
            }
        };
        let exposure = scene.exposure;
        // What the frame is written at, which every pass reading it back divides through again.
        let encode = exposure.encode.max(f32::EPSILON);
        if self.name == ADAPT_LUM_PARAM {
            // The rate is stated per second and the buffer wants what one frame moves by. The
            // measure reads the frame divided by what wrote it, so what it reports does not follow
            // the exposure and a whole step would reach the answer in one frame.
            let step = (exposure.rate * exposure.step).min(SETTLE);
            write(
                &mut out,
                0,
                &[exposure.min, exposure.max, step, exposure.key * exposure.key],
            );
            return out;
        }
        // The three registers of the camera buffer the reflection chain reads, for a pass the
        // reflection gives no members. Its own projection rather than the viewer's: the game stands
        // the near plane at one and has no far plane at all, and the march's hit test is written
        // against that ordering. The second row is negated with it, since these shaders turn a
        // texture coordinate into a clip one the way D3D counts rows and the buffers they read count
        // them the other way.
        if self.name == CAMERA && self.members.is_empty() {
            let (near, _) = scene.planes();
            let reversed = Mat4::from_cols(
                projection.row(0),
                -projection.row(1),
                Vec4::new(0.0, 0.0, 0.0, near),
                Vec4::new(0.0, 0.0, -1.0, 0.0),
            )
            .transpose();
            for (base, matrix) in [
                (0, rows(view, 3)),
                (14, rows(reversed.inverse(), 4)),
                (18, rows(reversed, 4)),
            ] {
                for (at, row) in matrix.chunks(4).enumerate() {
                    write(&mut out, base + at, row);
                }
            }
            return out;
        }
        // The one register run the water chain's own vertex shader takes an object through, for a
        // pass whose reflection gives no members either. It takes a vertex straight into view space
        // and the projection above carries it from there.
        if self.name == INSTANCE && self.members.is_empty() {
            for (at, row) in rows(view * model, 3).chunks(4).enumerate() {
                write(&mut out, at, row);
            }
            return out;
        }
        // The march steps its ray one cell of this at a time, and the cells it counts are of the
        // buffer it is drawing into rather than of the frame. What the game's own upload holds here
        // could not be recovered: the window it stands in had been written over by the time the
        // capture was taken.
        if matches!(pass, Pass::WaterMirror) && self.name == SCREEN_PARAM {
            let texel = scene.reflect.texel;
            write(&mut out, 0, &[1.0 / texel.x, 1.0 / texel.y, texel.x, texel.y]);
            return out;
        }
        if self.name == SCREEN_PARAM {
            let (width, height) = (size.0.max(1.0), size.1.max(1.0));
            for at in 0..2 {
                write(&mut out, at, &[width, height, 1.0 / width, 1.0 / height]);
            }
            write(&mut out, 2, &[1.0, 1.0, 1.0, 1.0]);
            return out;
        }
        // The same buffer as below under the layout water's own chain names its fields by: a texel
        // and a half-texel of each buffer it addresses, then what the reflection fades toward, the
        // blur weights and how much of the frame a dynamic resolution is standing at. What the
        // chain draws into and the pyramid it walks are both half the frame, which is why the
        // first and third of these are the same register twice, as the game's own upload has them.
        // The mip the march starts at is the last register and stays at nought, which is the top
        // of the pyramid.
        if matches!(pass, Pass::WaterMirror) && self.name == REFLECTION_PARAM {
            let texel = scene.reflect.texel;
            let (width, height) = (size.0.max(1.0), size.1.max(1.0));
            let weights = blur_weights();
            let step = [texel.x, texel.y, texel.x * 0.5, texel.y * 0.5];
            write(&mut out, 0, &step);
            write(&mut out, 1, &[
                1.0 / width,
                1.0 / height,
                0.5 / width,
                0.5 / height,
            ]);
            write(&mut out, 2, &step);
            // What the reflection is taken over by with distance is the zone's own vertical fog,
            // read out of the same four fields the fog pass takes: the color it fades toward, the
            // most of it that ever arrives, then where the fade starts and how much a unit past
            // that adds.
            let fog = scene.fog;
            write(&mut out, 3, &[fog.color.x, fog.color.y, fog.color.z, fog.cap]);
            write(&mut out, 4, &[fog.rate, fog.start, WATER_MIRROR_UNREAD, 1.0]);
            write(&mut out, 5, &weights[..4]);
            write(&mut out, 6, &weights[4..]);
            write(&mut out, 7, &[1.0, 1.0, 0.0, 0.0]);
            return out;
        }
        if self.name == REFLECTION_PARAM {
            let held = scene.reflect;
            // The step a blur takes between its taps, and under it the offset the chain's own vertex
            // shader adds to every coordinate its passes sample with. Both are stated in the texel
            // of the level being read rather than in the frame's.
            let texel = held.texel;
            write(&mut out, 0, &[texel.x * 2.0, texel.y * 2.0, texel.x, texel.y]);
            write(&mut out, 1, &REFLECTION_FADE);
            write(&mut out, 2, &[0.0, REFLECTION_POWER, 0.0, REFLECTION_ROUGHNESS]);
            // A whole number rather than the float that reads the same: the shaders take this lane
            // through its bit pattern.
            out[40..44].copy_from_slice(&held.level.to_le_bytes());
            write(&mut out, 3, &[1.0, 1.0, 0.0, 0.0]);
            return out;
        }
        if self.name == PROJECTION_INVERSE {
            write_rows(&mut out, &rows(projection.inverse(), 4));
            return out;
        }
        if self.name == VIEW_INVERSE {
            write_rows(&mut out, &rows(view.inverse(), 4));
            return out;
        }
        if self.name == SKY_PARAM {
            let held = scene.sky;
            let hours = held.time / 3600.0;
            let held_sun = sun(held.time, held.tilt);
            let eye = view.inverse().w_axis;
            let (wide, tall) = held.size;
            write(&mut out, 0, &[eye.x, eye.y, eye.z, 0.0]);
            write(&mut out, 1, &[held_sun.x, held_sun.y, held_sun.z, 0.0]);
            // Cut to the volume's own texel centers, which is what the frame's numbers work out to
            // for the eight by thirty-two every sky but one is.
            write(&mut out, 2, &[0.5, 1.0 - 0.5 / tall, 0.0, 0.0]);
            write(&mut out, 3, &[(1.0 - 1.0 / wide) * 0.5, -(1.0 - 1.0 / tall), 0.0, 0.0]);
            // The hour's own slice, and the scale the sky is written at with everything else.
            write(&mut out, 4, &[(hours + 0.5) / held.depth, 0.0, encode, 0.0]);
            // The color the sky is mixed toward, at the weight a real frame carried: nought, so it
            // never reaches the frame. Nothing found states either.
            write(&mut out, 5, &[0.0; 4]);
            return out;
        }
        if self.name == SUN_PARAM {
            let (wide, tall) = scene.size;
            // Where the sun stands as this pass reads a pixel's own coordinate, which is a texture
            // one rather than the clip xy the sky takes. The game's runs down the frame and this
            // one up it, so the vertical is the one place the two conventions part.
            let over = sun_at(scene).unwrap_or_default();
            write(&mut out, 0, &[wide / tall, 1.0, over.x, over.y]);
            write(&mut out, 1, &SUN_RAYS);
            write(&mut out, 2, &SUN_FALLOFF);
            write(&mut out, 3, &SUN_CORE);
            write(&mut out, 4, &SUN_HALO);
            return out;
        }
        if self.name == MOON_PARAM {
            let held = scene.sky;
            let Some(disc) = moon_disc(scene) else {
                return out;
            };
            let color = held.moonlight.truncate() * MOON_TINT;
            let roll = moon_roll(held.time);
            let phase = moon_phase(held.day);
            let axis = moon_terminator(phase);
            let (slope, offset) = moon_softness(phase);
            write(&mut out, 0, &[roll.x, roll.y, axis.x, axis.y]);
            write(&mut out, 1, &[slope, offset, 1.0, phase]);
            write(&mut out, 2, &[color.x, color.y, color.z, held.moon_fade]);
            write(&mut out, 3, &[0.0, 0.0, 0.0, held.moonlight.w * MOON_WEIGHT]);
            // What the frame behind the disc is read at, which is where each of its own fragments
            // falls on the screen: the same rectangle the quad was stood over.
            write(&mut out, 4, &[disc.z, -disc.w, disc.x, disc.y]);
            write(&mut out, 5, &[0.0; 4]);
            return out;
        }
        if self.name == "cWorldMatrix" {
            write_rows(&mut out, &rows(model, 3));
            return out;
        }
        if self.name == "cWorldViewProjMatrix" {
            write_rows(&mut out, &rows(projection * view * model, 4));
            return out;
        }
        // A posteffect pass reflects its own quad-sampling buffer under the same bare name, so the
        // pass tells the two apart rather than the members being empty for both.
        if matches!(pass, Pass::Star) && self.name == "cParam" {
            let held = scene.star;
            let twinkle = (clock * STAR_TWINKLE_RATE).fract();
            write(&mut out, 0, &[STAR_PARAM_0[0], STAR_PARAM_0[1], STAR_PARAM_0[2], twinkle]);
            write(&mut out, 1, &[held.horizon, held.point, held.band, held.alpha]);
            write(&mut out, 2, &[STAR_HORIZON, 0.0, 0.0, 0.0]);
            write(&mut out, 3, &STAR_PARAM_3);
            write(&mut out, 4, &[STAR_SKY_SCALE[0], STAR_SKY_SCALE[1], 0.0, 0.0]);
            return out;
        }
        // The star shaders declare this as one bare 64-byte member rather than the four fields
        // every other package's own reflection names, so a fill by name reaches nothing here: the
        // same four fields, written positionally instead.
        if self.name == "g_CommonParameter" && self.members.is_empty() {
            let (width, height) = (size.0.max(1.0), size.1.max(1.0));
            write(&mut out, 0, &[1.0 / width, 1.0 / height, 0.0, 0.0]);
            write(&mut out, 1, &[2.0 / width, 2.0 / height, -1.0, -1.0]);
            let bloom = scene.bloom;
            let [gain, floor] = REFLECTION_WEIGHT;
            write(&mut out, 2, &[bloom.specular, bloom.emissive, gain, floor]);
            write(&mut out, 3, &[encode, 0.0, 0.0, 0.0]);
            return out;
        }
        if matches!(pass, Pass::CloudBand | Pass::CloudSheet | Pass::CloudShadow)
            && let Some(register) = [VS_PARAM, PS_PARAM].iter().position(|held| self.name == *held)
        {
            let held = scene.cloud;
            let sun = sun(scene.sky.time, scene.sky.tilt);
            let eye = view.inverse().w_axis;
            if register == 1 {
                // The two colors go in squared, and come back out under a root: what the shader
                // works out is a light, and it is gathered in the square of the color rather than
                // in the color. The sky ramp and the shadow's own two numbers are the same in
                // every frame measured.
                let squared = |held: Vec3| held * held;
                let (diffuse, ambient) = (squared(held.diffuse), squared(held.ambient));
                write(&mut out, 0, &[sun.x, sun.y, sun.z, CLOUD_FLOOR]);
                write(&mut out, 1, &[diffuse.x, diffuse.y, diffuse.z, 1.0]);
                write(&mut out, 2, &[ambient.x, ambient.y, ambient.z, 0.0]);
                write(&mut out, 3, &[2.0, 0.0, 10.0, -5.0]);
                write(&mut out, 4, &[0.125, 50.0, 1.0, 0.0]);
                return out;
            }
            let up = sun.y.abs();
            match pass {
                // A cylinder the vertex shader flares into a cone: the first pair leans it toward
                // the sun, and the two heights it splits the reach into are how far up the band
                // stands on the near side and on the far one.
                Pass::CloudBand => {
                    let lean = glam::Vec2::new(sun.x, sun.z.abs()).normalize_or_zero() * 0.5;
                    write(&mut out, 0, &[1.0, 1.0, 0.0, 0.0]);
                    write(
                        &mut out,
                        1,
                        &[
                            lean.x,
                            lean.y,
                            held.reach * (0.25 + 0.75 * up),
                            held.reach * 0.75 * (1.0 - up),
                        ],
                    );
                    write(&mut out, 2, &[0.0; 4]);
                    write(&mut out, 3, &[sun.x, sun.y, sun.z, CLOUD_FLOOR]);
                    write(&mut out, 4, &[eye.x, eye.y, eye.z, eye.y / 1000.0]);
                }
                // The sheet tiles its texture ten times across the forty thousand units it spans,
                // and takes the whole of its first layer: the second is what the crossfading
                // variant reads, and every frame measured leaves it no weight at all.
                _ => {
                    write(&mut out, 0, &[SHEET_TILING, SHEET_TILING, 0.0, 0.0]);
                    write(&mut out, 1, &[1.0; 4]);
                    write(&mut out, 2, &[0.0; 4]);
                    write(&mut out, 3, &[sun.x, sun.y, sun.z, 0.0]);
                    write(&mut out, 4, &[eye.x, eye.y, eye.z, 0.0]);
                }
            }
            return out;
        }
        if self.name == FOG_PARAM {
            let held = scene.fog;
            // Scaled the way the sky it fades toward already is: the two are mixed together and
            // have to stand in one space.
            let color = held.color * encode;
            let glow = held.glow * encode;
            let eye = view.inverse().w_axis;
            // What the depth buffer holds and the distance in front of the camera it stands for are
            // one over the other about the planes the projection states. The table is addressed
            // across what the fog spans, on its own texel centers: the first holds where the fog
            // starts and the last where it stops changing.
            let (z, w) = (projection.z_axis.z, projection.w_axis.z);
            let texel = 1.0 / FOG_TABLE as f32;
            let scale = (1.0 - texel) / (held.far() - held.start).max(f32::EPSILON);
            write(&mut out, 0, &[color.x, color.y, color.z, held.cap]);
            // The color carries its own weight here and the set's again in the height buffer. The
            // two only ever multiply, so the file's is folded into the color and this stays one.
            write(&mut out, 1, &[glow.x, glow.y, glow.z, 1.0]);
            // A one in the last lane is what keeps the pass off the froxel volume it would
            // otherwise march, and nothing here builds one.
            let sun = sun(scene.sky.time, scene.sky.tilt);
            write(&mut out, 2, &[sun.x, sun.y, sun.z, 1.0]);
            write(&mut out, 3, &[0.0, 0.0, 0.0, texel * 0.5 - scale * held.start]);
            write(&mut out, 4, &[eye.x, eye.y, eye.z, scale]);
            write(&mut out, 5, &[z / w, 1.0 / w, 0.0, 0.0]);
            return out;
        }
        // What takes a pixel back out to where it stands. The pass hands it the depth as sampled,
        // which the translator's own fixup leaves in the clip space the game's shaders were built
        // for, so the projection goes in as it is.
        if self.name == CLIP_TO_WORLD {
            write_rows(&mut out, &rows((projection * view).inverse(), 4));
            return out;
        }
        // Where a pixel stands in the sun's own map. The pass hands it a view-space position, and
        // rows nought and one answer the coordinate while row two answers the depth to compare, so
        // only those two take the half that turns a clip coordinate into a texture one.
        if self.name == DIRECTIONAL_SHADOW_PARAM {
            let (sun, onto) = shadow_camera(scene.light, view, projection, scene.reach, scene.split);
            // The splits sit in a grid of one image, so the half that turns a clip coordinate into
            // a texture one also takes both lanes into this split's own cell of it.
            let (columns, grid_rows) = (ATLAS_COLUMNS as f32, ATLAS_ROWS as f32);
            let (column, row) = (
                (scene.split % ATLAS_COLUMNS) as f32,
                (scene.split / ATLAS_COLUMNS) as f32,
            );
            let half = Mat4::from_cols(
                Vec4::new(0.5 / columns, 0.0, 0.0, 0.0),
                Vec4::new(0.0, 0.5 / grid_rows, 0.0, 0.0),
                Vec4::new(0.0, 0.0, 1.0, 0.0),
                Vec4::new((column + 0.5) / columns, (row + 0.5) / grid_rows, 0.0, 1.0),
            );
            let map = half * onto * sun * view.inverse();
            put(
                DIRECTIONAL_SHADOW_PARAM,
                "m_ShadowProjectionMatrix",
                rows(map, 4),
            );
            // Where this split stops, as the depth buffer holds it: the resolve draws a quad there
            // and keeps what stands nearer, which is how a pixel reaches the nearest split that
            // still covers it.
            let (z, w) = (projection.z_axis.z, projection.w_axis.z);
            let depth = |at: f32| (w / at.max(f32::EPSILON) - z).clamp(0.0, 1.0);
            put(DIRECTIONAL_SHADOW_PARAM, "m_ShadowDistance", vec![
                depth(shadow_near(scene.reach, scene.split)),
                depth(shadow_reach(scene.reach, scene.split)),
                0.0,
                0.0,
            ]);
            // Read by the lighting rather than by the resolve, and a nought here is what makes the
            // shadowed variant write black whatever the mask holds. A texel of the whole atlas, not
            // of one split's own cell: the game's own measured value is `1/2048, 1/10240` against a
            // 2048x10240 atlas, the reciprocal of the full width and height.
            put(DIRECTIONAL_SHADOW_PARAM, "m_ShadowMapParameter", vec![
                1.0 / (SHADOW_MAP * ATLAS_COLUMNS as i32) as f32,
                1.0 / (SHADOW_MAP * ATLAS_ROWS as i32) as f32,
                0.0,
                1.0,
            ]);
            // What the softening sizes a penumbra with. The second lane turns a depth of the map
            // back into a distance along the light, which is the whole span the box covers since
            // its own near plane sits at nought; the first turns such a distance into a radius of
            // the disc the taps stand on, which the shader states in the shorter side of the whole
            // image. A negative first lane is also what tells it the map is orthographic.
            let cell = 1.0 / (ATLAS_ROWS as f32 * map.row(1).truncate().length());
            let short = (SHADOW_MAP * ATLAS_COLUMNS.min(ATLAS_ROWS) as i32) as f32;
            put(DIRECTIONAL_SHADOW_PARAM, "m_NearFarParam", vec![
                -SUN_SOFTNESS * SHADOW_MAP as f32 / (cell * short),
                -1.0 / map.row(2).truncate().length(),
                0.0,
                1.0,
            ]);
            return out;
        }
        if self.name == SHADOW_BIAS_PARAM {
            // Measured off a frame the game drew: no constant offset, a twentieth of a unit along
            // the normal, and the whole of the slope term.
            write(&mut out, 0, &[0.0, 0.05, 1.0, 0.0]);
            return out;
        }
        if self.name == HEIGHT_FOG_PARAM {
            let held = scene.fog;
            let [near, far] = held.layers;
            write(&mut out, 0, &[held.near, near.x, near.y, near.z]);
            write(&mut out, 1, &[held.clear, far.x, far.y, far.z]);
            write(
                &mut out,
                2,
                &[
                    held.haze,
                    held.glow_strength,
                    held.glow_sharpness,
                    held.glow_start,
                ],
            );
            return out;
        }
        if self.name == COMMON_TEX_PARAM {
            write(&mut out, 0, &[encode, 1.0 / encode, 0.0, 0.0]);
            return out;
        }
        if self.name == TONE_MAP_PARAM {
            // Half a texel of the curve, which is both where its first texel's center falls and the
            // share of the exposure one texel of it spans.
            let half = 0.5 / CURVE as f32;
            write(
                &mut out,
                0,
                &[exposure.strength, exposure.shoulder, encode, 0.0],
            );
            write(&mut out, 1, &[half, 1.0 - half, 1.0 / encode, half / encode]);
            return out;
        }
        if self.name == BRIGHT_PASS_PARAM {
            write(&mut out, 0, &[GLARE_THRESHOLD, 0.0, 0.0, 0.0]);
            return out;
        }
        if self.name == MERGE_WEIGHT {
            // The merge reads the halo alone: the second lane weighs a level of the pyramid the
            // engine keeps beside it, and every frame measured leaves that at nothing.
            write(&mut out, 0, &[1.0, 0.0, GLARE_VEIL, 0.0]);
            return out;
        }
        if self.name == SOFT_FOCUS_PARAM {
            // The last lane is the alpha the pass writes, and the blend reads it back: nought is
            // what makes the halo add to the frame rather than cover it.
            write(&mut out, 0, &[
                1.0 - 0.5 / size.0,
                1.0 - 0.5 / size.1,
                0.0,
                0.0,
            ]);
            return out;
        }
        if self.name == CLOUD_SHADOW_PARAM {
            write(&mut out, 0, &CLOUD_SHADOW_WEIGHTS);
            return out;
        }
        if self.name == SAMPLING_PARAM {
            // The quad this is drawn over already carries clip space and the coordinate to read at,
            // so the position is only turned the way round the pass expects and the coordinate is
            // taken as it stands.
            write(&mut out, 0, &[0.5, -0.5, 0.5, 0.5]);
            write(&mut out, 1, &[1.0, 1.0, 0.0, 0.0]);
            return out;
        }
        if self.name == SAMPLING_OFFSET {
            // The cloud shadow's own four, which stand on a cross rather than on a square: each is a
            // whole texel of the map one way and half a texel the other.
            if matches!(pass, Pass::CloudShadow) {
                let texel = 1.0 / CLOUD_SHADOW_MAP as f32;
                let half = texel * 0.5;
                write(&mut out, 0, &[-texel, -half, texel, half]);
                write(&mut out, 1, &[half, -texel, -half, texel]);
                return out;
            }
            // Which tap lands in which lane is settled by the weight the pass pairs with it, and the
            // middle of the kernel is the first lane here rather than the lone coordinate.
            match scene.blur {
                Blur::Square(texel) => {
                    let held = texel * GAUSS_TAP;
                    write(&mut out, 0, &[0.0, 0.0, -held.x, 0.0]);
                    write(&mut out, 1, &[held.x, 0.0, 0.0, -held.y]);
                    write(&mut out, 2, &[0.0, held.y, -held.x, -held.y]);
                    write(&mut out, 3, &[-held.x, held.y, held.x, -held.y]);
                    write(&mut out, 4, &[held.x, held.y, 0.0, 0.0]);
                }
                Blur::Along(step) => {
                    let [near, mid, far] = TAPS.map(|held| step * held);
                    write(&mut out, 0, &[0.0, 0.0, near.x, near.y]);
                    write(&mut out, 1, &[mid.x, mid.y, far.x, far.y]);
                    write(&mut out, 2, &[-near.x, -near.y, -mid.x, -mid.y]);
                    write(&mut out, 3, &[-far.x, -far.y, 0.0, 0.0]);
                }
            }
            return out;
        }
        if self.name == VIGNETTING_PARAM {
            // The ellipse the darkening spreads over sits halfway between a circle and the frame's
            // own shape, and its axes are taken to unit length so a corner falls at one whatever
            // the frame's shape.
            let shape = (1.0 + size.0 / size.1) * 0.5;
            let span = shape.hypot(1.0);
            let look = scene.look;
            write(
                &mut out,
                0,
                &[shape / span, 1.0 / span, look.onset, look.darkening],
            );
            // The second register is the color a corner is taken toward, which every frame that
            // states it leaves at black.
            return out;
        }
        if self.name == FXAA_PARAM {
            write(
                &mut out,
                0,
                &[1.0 / size.0, 1.0 / size.1, FXAA_SUBPIX, FXAA_EDGE],
            );
            return out;
        }
        if self.name == VIEW_DEPTH_FACTOR {
            // The reading and the distance are one over the other about the plane the projection
            // states, so the pass is given that relation rather than the planes. The normal is left
            // at the scale it arrived with: the occlusion pass has a bias of its own.
            let (z, w) = (projection.z_axis.z, projection.w_axis.z);
            write(&mut out, 0, &[z / w, 1.0 / w, 1.0, 0.0]);
            return out;
        }
        if self.name == VIEW_ROTATION {
            write(&mut out, 0, &rows(view, 3));
            return out;
        }
        if self.name == HDAO_PARAM {
            let texel = OCCLUSION_SCALE as f32;
            write(
                &mut out,
                0,
                &[
                    texel / size.0,
                    texel / size.1,
                    OCCLUSION_SPREAD,
                    OCCLUSION_SPREAD,
                ],
            );
            // The reach goes in the last lane here as well, which no quality the game reads.
            write(
                &mut out,
                1,
                &[
                    1.0 / OCCLUSION_ACCEPT,
                    1.0 / OCCLUSION_REJECT,
                    1.0 / OCCLUSION_NEAR,
                    OCCLUSION_REACH,
                ],
            );
            write(
                &mut out,
                2,
                &[
                    OCCLUSION_REACH,
                    OCCLUSION_BIAS,
                    OCCLUSION_INTENSITY,
                    OCCLUSION_POWER,
                ],
            );
            return out;
        }
        let world_view = view * model;
        let view_projection = projection * view;
        // Nothing here moves between frames, so every previous-frame matrix is the current one and
        // the motion vectors come out as nought.
        let camera = CAMERA;
        for name in ["m_ViewMatrix", "m_ViewMatrixPrev"] {
            put(camera, name, rows(view, 3));
        }
        for name in [
            "m_InverseViewMatrix",
            "m_InverseViewMatrixPrev",
            "m_MainViewToWorldMatrix",
        ] {
            put(camera, name, rows(view.inverse(), 3));
        }
        for name in ["m_ViewProjectionMatrix", "m_ViewProjectionMatrixPrev"] {
            put(camera, name, rows(view_projection, 4));
        }
        for name in [
            "m_InverseViewProjectionMatrix",
            "m_InverseViewProjectionMatrixPrev",
        ] {
            put(camera, name, rows(view_projection.inverse(), 4));
        }
        for name in [
            "m_ProjectionMatrix",
            "m_ProjectionMatrixPrev",
            "m_MainViewToProjectionMatrix",
        ] {
            put(camera, name, rows(projection, 4));
        }
        for name in ["m_InverseProjectionMatrix", "m_InverseProjectionMatrixPrev"] {
            put(camera, name, rows(projection.inverse(), 4));
        }
        put(camera, "m_ProjToProjPrevMatrix", rows(Mat4::IDENTITY, 4));
        put(camera, "m_ViewToViewPrevMatrix", rows(Mat4::IDENTITY, 3));
        // The transform a vertex shader multiplies by before the projection alone, with nothing
        // between the two: it takes an object into view space rather than into the world. The buffer
        // holds this frame's and the last one's.
        put("g_WorldViewMatrix", "g_WorldViewMatrix", {
            let mut held = rows(world_view, 3);
            held.extend(rows(world_view, 3));
            held
        });
        // What takes an object into the world alone, which the engine's own draws carry instead of
        // the object transform a model's instancing buffer holds.
        put("g_WorldMatrix", "g_WorldMatrix", rows(model, 3));
        // The engine drives these and no file states one. A blade reads the last lane of the second
        // as how dry it is, and the two masks as how much of the occlusion buffer reaches it.
        let grass = "g_GrassCommonParam";
        put(grass, "m_GrassNormal", GRASS_UP.to_vec());
        put(grass, "m_Param", vec![0.0, 0.0, 0.0, 1.0]);
        put(grass, "m_SSAOMaskMin", vec![1.0]);
        put(grass, "m_SSAOMaskMax", vec![1.0]);
        // Both scale the wind displacement before anything else does; left at one rather than the
        // nought the buffer would otherwise sit at, which silently stands every blade still.
        put(grass, "m_GrassWindSpeedScale", vec![1.0]);
        put(grass, "m_BushWindSpeedScale", vec![1.0]);
        put(INSTANCE, "m_MulColor", vec![1.0, 1.0, 1.0, scene.opacity]);
        // What a hair strand flutters along. The engine hands over a unit heading scaled by the
        // wind's own speed, capped at thirty and taken down by a factor of two thousand, and leaves
        // the last lane at nought. Heading and speed are this viewer's own, read off the zone's
        // `.envb`; the engine samples an ambient field at the character's position instead, and
        // whether the two are the same quantity is the one thing here nothing states.
        let gust = scene.wind.heading * scene.wind.reach.min(WIND_SPEED_CAP) * WIND_SCALE;
        put(INSTANCE, "m_Wind", vec![gust.x, gust.y, gust.z, 0.0]);
        // Declared by all five character packages; only `character`'s own G pass reads the first
        // lane, scaling the fur march by it. A capture confirms the game writes the same identity
        // here.
        put(INSTANCE, "m_Param", vec![1.0]);
        // One record an eye, picked by the vertex color. The first two lanes scale the coordinate an
        // eye's textures are read at and the third warps it toward the pupil, so ones leave that
        // coordinate where the mesh's own uv put it; nought collapses the eye onto a single texel.
        put(INSTANCE, "m_IrisParam", vec![1.0; 8]);
        // The engine drives this per draw and no material states one, so identity is what leaves a
        // table's own emissive column as it was written.
        put(
            "g_MaterialParameterDynamic",
            "m_EmissiveColor",
            vec![1.0; 3],
        );
        // Where a lit pixel stands in the map the sheet's own shadow was drawn into. The lighting
        // hands this a view-space position and turns the clip coordinate into a texture one itself,
        // so unlike the sun's own map this takes no half of its rows. A weather that draws no sheet
        // leaves the map opaque white, and the term is one wherever this lands.
        let (cloud, onto) = Cloud::shadow_camera(scene.light, view);
        put(
            CLOUD_SHADOW_MATRIX,
            CLOUD_SHADOW_MATRIX,
            rows(onto * cloud * view.inverse(), 4),
        );
        // The weight the character resolve carries a material's own emissive into the frame's alpha
        // at, which the glare pass reads that frame back through. It reaches no color of its own,
        // and the engine drives the weight over a span narrow enough to stand on its low end.
        put(INSTANCE, "m_EnvParameter", vec![0.0, 0.0, 0.0, 0.17]);
        // A fill light standing where the camera does, added over whatever the frame's own lighting
        // left. The first register weighs it: a diffuse and a specular for the character path, then
        // the pair skin takes instead. The second is the rim it draws around a silhouette, how far
        // that rim is leant into the view, and the weight an eye reads its own reflection at.
        put(INSTANCE, "m_CameraLight", vec![
            0.15, 0.15, 0.15, 0.17, 0.01584, 0.9, 0.01584, 0.8,
        ]);
        put("g_ModelParameter", "m_Params", vec![1.0; 4]);
        // The clock every animated package shares. A tick is a thousand-and-twenty-fourth of a
        // second and the engine's accumulator is masked to twenty-one bits, so this runs to exactly
        // 2048 and back, which is what makes the periods the hair shader snaps to divide it evenly.
        put(PBR, "m_LoopTime", vec![
            ((clock * f32::from(LOOP_TICKS)) as u64 & LOOP_WRAP) as f32 / f32::from(LOOP_TICKS),
        ]);
        // What skin showing through a stocking is multiplied by, which is not the light's own color
        // of the same name.
        put("g_SkinMaterialParameter", "m_DiffuseColor", vec![1.0; 3]);

        // The colors a character was made with. The last lane of each hair color is not a hair's
        // own alpha: the pair places the decal a face paint is read through across the face, and
        // every package reading either reads it for that and nothing else.
        let held = scene.customize;
        let customize = "g_CustomizeParameter";
        put(customize, "m_SkinColor", held.skin.to_vec());
        put(customize, "m_LipColor", held.lip.to_vec());
        put(customize, "m_MainColor", vec![
            held.hair[0],
            held.hair[1],
            held.hair[2],
            1.0,
        ]);
        put(customize, "m_MeshColor", vec![
            held.highlight[0],
            held.highlight[1],
            held.highlight[2],
            0.0,
        ]);
        // A face with no paint picked leaves the weight at nought, which is what keeps the flat
        // stand-in an unbound decal sampler answers with off the whole face. What that weight is
        // when one is picked is the swatch's own last lane, which is what the game writes for a lip
        // and what nothing else in the file could be.
        put(DECAL, DECAL, held.decal.to_vec());
        // The last lane of an eye color is not the swatch's own alpha either: it is how strongly
        // `iris` rings the iris edge with the race feature color.
        put(customize, "m_LeftColor", vec![
            held.left_eye[0],
            held.left_eye[1],
            held.left_eye[2],
            0.0,
        ]);
        put(customize, "m_RightColor", vec![
            held.right_eye[0],
            held.right_eye[1],
            held.right_eye[2],
            0.0,
        ]);
        put(customize, "m_OptionColor0", held.option.to_vec());

        // A pixel's own place, which a screen-wide pass has nothing else to work from.
        let (width, height) = (size.0.max(1.0), size.1.max(1.0));
        let common = "g_CommonParameter";
        put(common, "m_RenderTarget", vec![
            1.0 / width,
            1.0 / height,
            0.0,
            0.0,
        ]);
        put(common, "m_Viewport", vec![
            2.0 / width,
            2.0 / height,
            -1.0,
            -1.0,
        ]);
        // The two lanes the composite weighs its glare by before it divides that through by the
        // colour and leaves the share in the frame's alpha, which the weather states, and then the
        // pair a surface takes the brightness it stands in through to weigh what it reflects.
        let held = scene.bloom;
        let [gain, floor] = REFLECTION_WEIGHT;
        put(common, "m_Misc", vec![held.specular, held.emissive, gain, floor]);
        put(common, "m_Misc2", vec![scene.exposure.encode, 0.0, 0.0, 0.0]);
        // Which of the two G-buffers the lighting is reading. The opaque one carries the shader
        // type in the first target's alpha and the transparent one in the second's first channel,
        // and every lighting pass picks between them with this.
        put(
            "g_LightDrawParam",
            "SemiTransparency",
            vec![f32::from(u8::from(scene.sheer))],
        );
        let screen = "g_ScreenParameter";
        put(screen, "m_BackBufferSize", vec![width, height]);
        put(screen, "m_ViewportSize", vec![width, height]);
        for name in ["m_InverseBackBufferSize", "m_InverseViewportSize"] {
            put(screen, name, vec![1.0 / width, 1.0 / height]);
        }
        // Nothing here renders at a resolution other than the one it presents at, and a pass that
        // reads the frame back scales its coordinate by this before sampling.
        for name in ["m_DynamicResolutionScale", "m_DynamicResolutionChangeScale"] {
            put(screen, name, vec![1.0; 2]);
        }

        // The transform water is drawn through: it takes one from here rather than from the buffer
        // every other package names, and at nought every vertex lands on the same point.
        put(INSTANCE, "m_WorldViewMatrix", rows(world_view, 3));
        // The fade a dither clip tests against, and the weight a mesh's own position carries into
        // the wave it is lifted by. One leaves both as the file wrote them.
        put(INSTANCE, "m_Misc", vec![0.0, 0.0, 1.0, 1.0]);

        // A shaft of light reads the same buffer as the object's own placement: where it stands,
        // how it turns, and what it is scaled by. The material's `g_NearClip` is read against the
        // last lane as `saturate(clip * it - 1)`, which fades a shaft out as the eye comes up to
        // it, so what the lane holds is how far away it stands. The colors go in beside it: the
        // second row weighs what the surface itself carries.
        let (scale, turn, _) = model.to_scale_rotation_translation();
        put(INSTANCE, "transform", rows(model, 3));
        put(INSTANCE, "rotate", rows(Mat4::from_quat(turn), 3));
        let held = scene.shaft.color;
        let eye = view.inverse().w_axis.truncate();
        put(INSTANCE, "misc", vec![
            scale.x,
            scale.y,
            scale.z,
            eye.distance(model.w_axis.truncate()),
            held.x,
            held.y,
            held.z,
            0.0,
        ]);

        // Water is a sum of Gerstner waves, and each is a sine of a frequency times this plus a
        // wavenumber times where the vertex stands; the wave maps, the noise and the caustics all
        // scroll along it as well, at rates the material states.
        let water = "g_WaterParameter";
        put(water, "m_WavingParam", vec![clock; 4]);
        for name in ["m_GBufferSize", "m_RenderTargetSize"] {
            put(water, name, vec![width, height, 1.0 / width, 1.0 / height]);
        }
        // `.zw` of each is the half-resolution reciprocal, measured off the game's own frame.
        for name in [
            "m_GBufferPixelSize",
            "m_RenderTargetPixelSize",
            "m_HalfViewPositionPixelSize",
        ] {
            put(water, name, vec![1.0 / width, 1.0 / height, 0.5 / width, 0.5 / height]);
        }
        // How far into the frame a surface may reach for what stands behind it. Sampling past this
        // is folded back in, so the whole frame is what leaves the reading where it was aimed.
        put(water, "m_DynamicViewportResolution", vec![1.0; 4]);
        // The engine's own local-reflection toggle, measured on in a real frame. Neither river
        // shader this viewer reaches happens to read it, but a variant that does should see the
        // game's own value rather than the package's dead default.
        put(water, "m_RLRParam", vec![2.0, 0.0, 0.0, 0.0]);
        // The light the caustics volume adds under the surface, at the strength the material states
        // and nothing else. The lane past it wobbles where that slice is read, and no file or frame
        // states how far, so the slice is taken where the surface puts it.
        put(water, "m_UnderCausticsParam", vec![0.0, 0.0, 1.0, 0.0]);
        // Both lerp toward one against a weight the mesh carries, so a one is the reading the
        // engine's own number would only move away from.
        put(water, "m_Roughness", vec![1.0; 4]);
        put(water, "m_Misc", vec![1.0; 4]);
        // Where the whitecaps wander: a world position goes through the reciprocal here, so a one
        // tiles the noise once per world unit rather than once across it.
        put(water, "m_NoiseSize", vec![
            NOISE_TEXELS,
            NOISE_TEXELS,
            1.0 / NOISE_TEXELS,
            1.0 / NOISE_TEXELS,
        ]);

        // The pair below is read by every one of the twenty-eight shaders holding the buffer, and the
        // two past it by none of them.
        //
        // Both are added to a position the instancing record has already brought into view space, so
        // they are handed over in that space too. The sun draws the same sway under its own view,
        // and a leaf whose shadow leans one way while it leans another is what leaving them in the
        // world looks like.
        let wind = view.transform_vector3(scene.wind.heading * WIND_REACH);
        put(WAVING, "m_WindVector", wind.to_array().to_vec());
        put(WAVING, "m_UpVector", view.transform_vector3(Vec3::Y).to_array().to_vec());
        // All four as the engine writes them once and never again, though no waving shader reads
        // past .xy.
        put(WAVING, "m_WavingParam", vec![1.0, 1.0, 0.2, 1.0]);

        // A light is read in view space: the shader dots its direction against a normal it has just
        // brought out of the G-buffer and through the view matrix.
        let axes = glam::Mat3::from_mat4(view);
        let light = LIGHT;
        let lamp = scene.lamp;
        put(
            light,
            "m_Direction",
            (axes
                * match pass {
                    Pass::Lamp => lamp.direction,
                    _ => scene.light,
                })
            .normalize_or_zero()
            .to_array()
            .to_vec(),
        );
        // Both colors go in squared, as the clouds' own pair does: what the shader gathers is a
        // light, and a frame the game drew states the square of every color a file holds, the sun's
        // and each lamp's alike, to five digits.
        let squared = |held: Vec3| (held * held).to_array().to_vec();
        put(
            light,
            "m_DiffuseColor",
            squared(match pass {
                Pass::Lamp => lamp.color,
                _ => scene.diffuse,
            }),
        );
        put(
            light,
            "m_SpecularColor",
            squared(match pass {
                Pass::Lamp => lamp.color,
                _ => scene.specular,
            }),
        );
        // The two lighting packages read this buffer differently. A sun fades with the depth of the
        // pixel, and the fade is off: the scale is cubed and clamped, so a constant one leaves it
        // alone, and a frame the game drew states the same `(0, 0, 1, 0.05)` - the floor never bites
        // against a ramp already at one. A lamp reads `z` as what its squared distance is taken into
        // the ramp by, and `w` as what that ramp is scaled by against the distance itself: the
        // pixel shader works out `saturate(ramp(d^2 * z) * w / d)`. A spot's own buffer in a frame
        // the game drew reads `(cos(inner), cos(outer), 0.000196, 1.0)`, so `w` is **one** and the
        // falloff is the ramp over the distance with nothing scaling it up. `y` is the cosine that
        // package discards a spot against outright.
        let reach = lamp.reach.max(0.001);
        let (inner, cone) = match pass {
            Pass::Lamp => (lamp.inner, lamp.cone),
            _ => (0.0, 0.0),
        };
        put(
            light,
            "m_Attenuation",
            match pass {
                Pass::Composite | Pass::CompositeBlended => vec![0.0, 0.0, 1.0, 0.05],
                _ => vec![inner, cone, 1.0 / (reach * reach), 1.0],
            },
        );
        put(light, "m_LightFadeValueStatic", vec![1.0]);
        put(light, "m_LightFadeValueDynamic", vec![1.0]);

        // A lamp is drawn as the volume it reaches: its own vertex shader clamps a unit box to the
        // extents the zone clips it against and then projects, so the transform carries the light's
        // whole reach and the extents cut the box back out of it. A spot scales the box by the
        // fourth extent before clamping and keeps only the half in front of itself, so one leaves it
        // where the clamp alone would have put it.
        let volume = lamp.placement * Mat4::from_scale(Vec3::splat(reach));
        let (min, max) = (lamp.min / reach, lamp.max / reach);
        put(
            light,
            "m_Position",
            (view * lamp.placement * Vec3::ZERO.extend(1.0))
                .to_array()
                .to_vec(),
        );
        // A plane throws along its own positive z and lights only what stands in front of it, so the
        // ray back to it runs the other way; its `z` is what the pass divides the depth by, and
        // nought there is a NaN on every pixel the light covers. The mask carries the span the
        // light's own clip box states, which is what puts its edge where the zone put it. The fade
        // is the one number no file states: two is where the ramp reaches full strength at the
        // middle, and anything larger only sharpens the edge.
        let span = lamp.min.abs().max(lamp.max.abs()).max(Vec3::splat(0.001));
        put(light, "m_PlaneRayDirection", vec![0.0, 0.0, -1.0, 0.0]);
        put(light, "m_ShadowTexMask", vec![0.5 / span.x, 0.5 / span.y, 0.0, 0.0]);
        put(light, "m_PlaneFadeScale", vec![2.0, 2.0]);
        put(
            light,
            "m_PlaneInversMatrix",
            rows((view * lamp.placement).inverse(), 3),
        );
        put(light, "m_ClipMin", min.extend(1.0).to_array().to_vec());
        put(light, "m_ClipMax", max.to_array().to_vec());
        put(
            light,
            "m_WorldViewProjectionMatrix",
            rows(view_projection * volume, 4),
        );
        put(
            light,
            "m_WorldViewInversMatrix",
            rows((view * volume).inverse(), 3),
        );
        out
    }

    /// One record per object drawn, at the stride the reflection's own record takes. The transform
    /// takes an object into view space rather than into the world: what a shader multiplies by after
    /// it is the projection alone.
    fn instancing(&self, scene: &Scene, instances: &[Instance], out: &mut [u8]) {
        let stride = self.stride() as usize;
        let mut put = |at: usize, name: &str, values: &[f32]| {
            let Some(member) = self.members.iter().find(|held| held.name == name) else {
                return;
            };
            for (lane, value) in values.iter().enumerate() {
                let offset = at * stride + member.offset as usize + lane * 4;
                if offset + 4 <= out.len() {
                    out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
                }
            }
        };
        let held = [Instance {
            transform: scene.model,
            ..Instance::default()
        }];
        let instances = match instances.is_empty() {
            true => &held[..],
            false => instances,
        };
        for (at, instance) in instances.iter().enumerate().take(self.instances()) {
            let world_view = scene.view * instance.transform;
            put(
                at,
                "m_TransformMatrix",
                &world_view.transpose().to_cols_array()[..12],
            );
            put(at, "m_SkyVisibility", &[instance.sky_visibility]);
            put(at, "m_DitherAlpha", &[1.0]);
            // The phase one object sways at. A `.ggd` placement states where in the cycle it starts;
            // nothing else does, so a layer group placement falls back to a guess off its own place,
            // which at least keeps a stand of the same plant from leaning as one. The noise is the
            // same offset again, which is all the vertical bob reads it for.
            let offset = std::f32::consts::TAU
                * instance.wind_phase.unwrap_or_else(|| {
                    let (x, z) = (instance.transform.w_axis.x, instance.transform.w_axis.z);
                    (x * 0.37 + z * 0.61).rem_euclid(std::f32::consts::TAU) / std::f32::consts::TAU
                });
            put(at, "m_WavingAnimTime", &[scene.clock * WAVING_RATE + offset]);
            put(at, "m_WavingAnimNoize", &[(offset / std::f32::consts::TAU).fract()]);
            // The blend is what carries the colour: the shading lerps from the material's own
            // emissive toward this one by it, so a colour written with the blend at nought never
            // reaches the frame. Left at nought the strength would take the non-emissive branch
            // instead, and every glowing thing a zone places would come out dark.
            let (color, power, blend) = match instance.emissive {
                Some(held) => (held.truncate(), held.w, 1.0),
                None => (Vec3::ZERO, 1.0, 0.0),
            };
            put(at, "m_EmissivePower", &[power]);
            put(at, "m_EmissiveColor", &color.to_array());
            put(at, "m_EmissiveBlend", &[blend]);
        }
    }
}

/// One register of a buffer written as floats.
fn write(out: &mut [u8], register: usize, values: &[f32]) {
    for (at, value) in values.iter().enumerate() {
        let offset = register * 16 + at * 4;
        if offset + 4 <= out.len() {
            out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
    }
}

/// The scale and bias a surface takes the brightness it stands in through, saturated, to weigh
/// everything it reflects: at nought the whole environment term goes and only what it refracts is
/// left. Byte-identical across two captured frames, so the engine holds these rather than any file.
const REFLECTION_WEIGHT: [f32; 2] = [4.0, 0.2];

/// Texels across `common/graphics/texture/-noise.tex`. Read off `ffxiv_dx11.exe`: the water
/// renderer writes that texture's own width, height and their reciprocals into `m_NoiseSize` every
/// frame, and a shader scales a world position by the reciprocal, so one tile of the noise spans
/// this many world units.
const NOISE_TEXELS: f32 = 128.0;

/// The floor a background surface's ambient never falls below, and the gain its occlusion is read
/// at. Byte-identical across five captures of four zones, so the engine holds these rather than any
/// file.
const BG_AMBIENT: [f32; 4] = [0.005, 0.005, 0.005, 10.0];

/// The ambient array, whose header the reflection describes and whose entries it does not: past the
/// spherical harmonics the buffer is an array of a struct named but not laid out, so it goes in by
/// register.
///
/// One entry is filled and the count says one, which is what keeps the composite from walking the
/// array at all: it takes entry `count - 1` at full weight and never enters the loop. The composite
/// reads entry `n` at registers `12 * n + 4` through `12 * n + 15`, so the one entry starts at four.
///
/// The harmonics go in turned by `axes`. The composite dots the light rows against a normal it has
/// just taken through the view matrix and the sky rows against a reflection of that same normal, so
/// both are read in view space; the reflection only goes back to the world to sample the cube.
fn ambient(held: &Ambient, axes: glam::Mat3, view: Mat4, out: &mut [u8]) {
    if out.len() < 8 {
        return;
    }
    let turned = |row: &Vec4| (axes * row.truncate()).extend(row.w);
    // The count reads as a whole number rather than as the float that would print the same.
    let volumes = held.volumes.len().min(ENTRIES - 1);
    out[..4].copy_from_slice(&(1 + volumes as u32).to_le_bytes());
    out[4..8].copy_from_slice(&held.sky_scale.to_le_bytes());
    for (at, row) in held.sky.iter().enumerate() {
        write(out, 1 + at, &turned(row).to_array());
    }
    // Last, since that is the one the composite falls back on: it seeds itself with entry
    // `count - 1` at full weight and walks only the entries before it.
    let global = volumes * 12 + 4;
    entry(held, axes, out, global);
    // No bounding shape, so the entry covers the frame rather than a room.
    write(out, global + 10, &[0.0, 0.0, 0.0, 0.0]);
    write(out, global + 11, &[0.0, 1.0, 0.0, 0.0]);

    // The places that light themselves, each tested against the pixel before the global one is
    // fallen back on. The composite reads entry `n` from `12n + 4`.
    for (index, volume) in held.volumes.iter().take(volumes).enumerate() {
        let at = index * 12 + 4;
        for (row, held) in volume.light.iter().enumerate() {
            write(out, at + row, &turned(held).to_array());
        }
        write(out, at + 3, &[0.0, 0.0, 0.0, volume.scale]);
        // A bounded entry states no attenuation and takes its reflected term raw, which is the pair
        // the composite tests to decide the place lights itself.
        write(out, at + 4, &[held.fade.x, held.fade.y, held.fade.z, 0.0]);
        write(out, at + 5, &[1.0, 0.0, 1.0, held.capture]);
        // Into the volume's own space, where it stands as the unit shape. The buffer holds the
        // three rows a `float3x4` takes, and the pixel arrives in front of the camera, so the view
        // is folded in here rather than at the source.
        // A pixel arrives in front of the camera, so the view is undone before the volume's own
        // transform takes it in.
        let into = volume.into * view.inverse();
        let rows = into.transpose().to_cols_array();
        for row in 0..3 {
            write(out, at + 6 + row, &rows[row * 4..row * 4 + 4]);
        }
        // How sharply it takes over across each face. The composite reads the near widths outright
        // and works the far ones back out by subtracting them, so both sides carry the same number.
        write(
            out,
            at + 9,
            &[volume.fade.x, volume.fade.y, volume.fade.z, volume.fade.x],
        );
        write(
            out,
            at + 10,
            &[volume.fade.y, volume.fade.z, 0.0, volume.shape],
        );
        write(out, at + 11, &[0.0, 1.0, 0.0, 0.0]);
    }
}

/// The ten registers a composite reads one entry of the ambient from. A drawing package that
/// composites itself binds exactly these as `g_AmbientParam`, which is what says where each field
/// sits. An array entry shares the first six and then diverges: what the sky rows sit at here is a
/// bounding volume there, tested against the pixel's position rather than dotted against a
/// direction, and left unread while the shape register stays nought.
fn entry(held: &Ambient, axes: glam::Mat3, out: &mut [u8], at: usize) {
    let turned = |row: &Vec4| (axes * row.truncate()).extend(row.w);
    for (row, held) in held.light.iter().enumerate() {
        write(out, at + row, &turned(held).to_array());
    }
    write(out, at + 3, &[0.0, 0.0, 0.0, held.scale]);
    write(out, at + 4, &[held.fade.x, held.fade.y, held.fade.z, 1.0]);
    write(
        out,
        at + 5,
        &[
            held.reflection.x,
            held.reflection.y,
            held.reflection.z,
            held.capture,
        ],
    );
    for (row, held) in held.sky.iter().enumerate() {
        write(out, at + 6 + row, &turned(held).to_array());
    }
    // The trailing lane scales the ambient skin, eyes and stockings take, as
    // `0.65 + 0.35 * it`, and nothing else reads it. Nought would leave those three at 65% of what
    // the gear beside them gets.
    write(out, at + 9, &[held.sky_scale, 0.0, 0.0, 1.0]);
}

/// The joint transforms a skinned shader reads, as the dwords of the texture standing in for a
/// structured buffer. Each is four columns of three floats.
///
/// A joint's transform is the object's own, composed with what the pose moved that bone by, so a
/// palette of identities stands the model in the pose it is stored in.
pub fn joints(palette: &[Mat4], object: Mat4) -> Vec<u32> {
    let rows = (palette.len().max(1) * JOINT).div_ceil(ROW);
    let mut out = vec![0u32; rows * ROW];
    for (at, joint) in palette.iter().enumerate() {
        let columns = (object * *joint).to_cols_array();
        for column in 0..4 {
            for lane in 0..3 {
                out[at * JOINT + column * 3 + lane] = columns[column * 4 + lane].to_bits();
            }
        }
    }
    out
}

/// The parameter files the table is filled from, and the record each one's first profile lands at: a
/// G pass adds its own family's base to the type its material names.
pub const PARAMETERS: [(usize, &str); 2] = [
    (CHARA_TYPES, "common/graphics/chara_shader_param.spm"),
    (128, "common/graphics/bg_shader_param.spm"),
];

/// The table `SV_Target.w` indexes, as the dwords of the texture standing in for a structured
/// buffer, filled from the parameter files whose profiles it holds.
///
/// Every index a G pass can write has a record, since the one it writes is `(32 + type) / 255` and
/// what a material makes of that is the material's own business. A file the caller has yet to
/// receive leaves its own records at nought, which is the branch a plain surface takes.
pub fn shader_types(files: &[(usize, &spm::ShaderParameters)]) -> Vec<u32> {
    let rows = (SHADER_TYPES * SHADER_TYPE).div_ceil(ROW);
    let mut out = vec![0u32; rows * ROW];
    for (base, file) in files {
        for profile in 0..file.rows().len() {
            let at = (base + profile) * SHADER_TYPE;
            let Some(record) = out.get_mut(at..at + SHADER_TYPE) else {
                continue;
            };
            for (column, held) in file.columns().iter().enumerate() {
                let Some(name) = spm::name(held.id()) else {
                    continue;
                };
                let Some(value) = file.value(profile, column) else {
                    continue;
                };
                if let Some((slot, stated)) = parameter(name, value) {
                    record[slot] = stated;
                }
            }
        }
    }
    out
}

/// Whether any record this material can reach states a fur length. The fur pass discards every pixel
/// whose own record leaves it at nought, so a model reaching none of them has nothing for it to do.
///
/// A material's records are the ones its colour table names a row at a time, plus the one a material
/// carrying no table states outright; both are offsets into the character family's own profiles.
/// Only that family: a background material states its profile through the same constant, and the
/// alpha the pass would march along is that family's emissive flag instead.
pub fn furred(material: &Material, types: &[u32]) -> bool {
    if material.family() == Family::Background {
        return false;
    }
    let held = material.held();
    let table = held.color_table().into_iter().flat_map(|table| {
        (0..table.rows()).filter_map(|row| Some(table.row_values(row)?.shader_index as usize))
    });
    let stated = held
        .constants()
        .iter()
        .find(|constant| constant.id() == SHADER_ID)
        .and_then(|constant| held.constant_values(constant))
        .and_then(|values| values.first().copied())
        .map(|value| value as usize);
    table.chain(stated).any(|profile| {
        types
            .get((CHARA_TYPES + profile) * SHADER_TYPE + FUR_LENGTH)
            .is_some_and(|held| f32::from_bits(*held) > 0.0)
    })
}

/// Where one of the parameters a file names goes in a record, and the dword it goes there as. A file
/// orders its own columns and carries whichever subset of the parameters its family reads; the
/// record's layout is the shaders' own, and every file writes into the same one.
fn parameter(name: &str, value: spm::Value) -> Option<(usize, u32)> {
    let slot = match name {
        "LightingType" => 0,
        "SubSurfaceProfileID" => 1,
        "SubSurfaceWidth" => 2,
        "BackScatterPower" => 3,
        "SheenRate" => 4,
        "SheenTintRate" => 5,
        "SheenAperture" => 6,
        "UseSubSurfaceRate" => 7,
        "HairScatterColorShift" => 8,
        "HairSpecularPrimaryShift" => 9,
        "HairSpecularBackScatterShift" => 10,
        "HairSpecularSecondaryShift" => 11,
        "FurLength" => 12,
        "HairBackScatterRoughnessOffsetRate" => 13,
        "HairSecondaryRoughnessOffsetRate" => 14,
        "SubSurfacePower" => 15,
        _ => return None,
    };
    let held = match value {
        // A specular shift is a lobe center against the sine of an angle, and the files state it in
        // degrees.
        spm::Value::Float(held) if name.starts_with("HairSpecular") => held.to_radians().to_bits(),
        spm::Value::Float(held) => held.to_bits(),
        spm::Value::Unsigned(held) => held,
        spm::Value::Name(held) => lighting(held),
    };
    Some((slot, held))
}

/// The lighting model a record names, as the integer the shaders compare against. Anything else is
/// the default, which is the model a surface with nothing said about it takes.
fn lighting(id: u32) -> u32 {
    match spm::name(id) {
        Some("HAIR") => 1,
        Some("LEGACY") => 2,
        Some("HALF") => 3,
        _ => 0,
    }
}

/// Halfs in a row of the layout every shader addresses, and the rows it addresses. A row address is
/// scaled by a hardcoded `1/32` everywhere, and columns nought through seven are divided by the
/// width the shader queries, so nothing else is readable.
const EXTENDED_ROW: usize = 32;
const EXTENDED_ROWS: usize = 32;

/// Where a legacy row's halfs sit in an extended one. Diffuse, specular and emissive land where they
/// were; the two scalars beside them swap; and the tile index and transform move to the end.
const LEGACY_TO_EXTENDED: [(usize, usize); 16] = [
    (0, 0),
    (1, 1),
    (2, 2),
    (7, 3),
    (4, 4),
    (5, 5),
    (6, 6),
    (3, 7),
    (8, 8),
    (9, 9),
    (10, 10),
    (11, 25),
    (12, 28),
    (13, 29),
    (14, 30),
    (15, 31),
];

/// The color table in the layout the game's own shaders read: eight texels a row, thirty-two rows.
/// Answers the halfs, the texels a row takes, and the rows.
///
/// An extended table is already that. A legacy one states sixteen rows of four texels, so it is
/// widened and each row becomes the pair the shaders address it as, which leaves the row blend a
/// no-op: legacy tables have no second row to blend toward.
pub fn table(held: &mtrl::ColorTable) -> Option<(Vec<u16>, usize, usize)> {
    let rows = held.rows();
    let raw = held.raw();
    if rows == 0 || !raw.len().is_multiple_of(rows * 4) {
        return None;
    }
    if held.kind() != mtrl::ColorTableKind::Legacy {
        return Some((raw.to_vec(), raw.len() / rows / 4, rows));
    }
    let mut values = vec![0u16; EXTENDED_ROWS * EXTENDED_ROW];
    for pair in 0..rows.min(EXTENDED_ROWS / 2) {
        let Some(row) = held.row(pair) else { continue };
        for (from, to) in LEGACY_TO_EXTENDED {
            let Some(half) = row.get(from) else { continue };
            values[pair * 2 * EXTENDED_ROW + to] = *half;
            values[(pair * 2 + 1) * EXTENDED_ROW + to] = *half;
        }
    }
    Some((values, EXTENDED_ROW / 4, EXTENDED_ROWS))
}

#[cfg(test)]
mod test {
    use std::io::Cursor;

    use glam::{Mat3, Mat4, Vec2, Vec3, Vec4};
    use ironworks::file::{File, spm::ShaderParameters};

    use super::{
        ADAPT_LUM_PARAM, ATLAS_COLUMNS, ATLAS_ROWS, Ambient, Buffer, CLOUD_SHADOW_MATRIX, Customize,
        DECAL, DIRECTIONAL_SHADOW_PARAM, Exposure, FOG_PARAM, FXAA_PARAM, Fog, HDAO_PARAM, INSTANCE,
        INSTANCING, JOINT, REFLECTION_PARAM, ROW, SETTLE, SHADER_TYPE, SHADOW_MAP, SPLITS,
        SUN_PARAM, WAVING, WIND_POWER_SCALE, Pass, Reflect, Scene, Sky, Volume, Wind, WindLayer,
        ambient, decal_field, encode, instance_fields, joints, moon_phase, moon_roll, moon_softness,
        moon_terminator, selector, shader_types, sun,
    };

    /// The two buffers of the post chain no reflection describes, against what the game's own
    /// upload holds in them. Nothing else stands between a lane of either and the wrong number.
    #[test]
    fn the_smoothing_and_occlusion_buffers_come_out_as_the_game_holds_them() {
        let scene = Scene {
            size: (1920.0, 1080.0),
            ..Default::default()
        };
        let filled = |name: &str, registers| {
            let held = Buffer {
                name: name.to_owned(),
                members: Vec::new(),
                registers,
                fixed: None,
            };
            held.fill(&scene, Pass::Composite, &[])
                .chunks_exact(4)
                .map(|held| f32::from_le_bytes(held.try_into().unwrap()))
                .collect::<Vec<f32>>()
        };
        assert_eq!(filled(FXAA_PARAM, 1), [
            1.0 / 1920.0,
            1.0 / 1080.0,
            0.5,
            0.15
        ]);
        // The taps run over a gather half the frame across, so the texel the first pair names is
        // that gather's rather than the frame's.
        assert_eq!(filled(HDAO_PARAM, 3), [
            2.0 / 1920.0,
            2.0 / 1080.0,
            0.56,
            0.56,
            100.0,
            5.0,
            1.0,
            50.0,
            50.0,
            0.1,
            10.0,
            0.3
        ]);
    }

    /// Where a screen-wide pass reads the frame it is drawing into. `SV_Position` reaches the body
    /// as `gl_FragCoord`, which counts rows from the corner a texture coordinate counts them from,
    /// so this pair scales a fragment's own place into the frame and never turns it over. Both the
    /// buffer named by its fields and the one the star shaders leave bare hold the same four lanes.
    #[test]
    fn a_pixel_reads_the_frame_at_its_own_row() {
        let scene = Scene {
            size: (1920.0, 1080.0),
            ..Default::default()
        };
        let filled = |members: Vec<(&str, u32, u32)>| {
            let held = Buffer {
                name: "g_CommonParameter".to_owned(),
                members: members
                    .into_iter()
                    .map(|(name, offset, size)| hlsl::layout::Member {
                        name: name.to_owned(),
                        offset,
                        size,
                        kind: "float4".to_owned(),
                    })
                    .collect(),
                registers: 4,
                fixed: None,
            };
            held.fill(&scene, Pass::Composite, &[])
                .chunks_exact(4)
                .map(|held| f32::from_le_bytes(held.try_into().unwrap()))
                .collect::<Vec<f32>>()
        };
        for held in [
            filled(vec![("m_RenderTarget", 0, 16), ("m_Viewport", 16, 16)]),
            filled(Vec::new()),
        ] {
            let corner = |row: usize, at: Vec2| {
                Vec2::new(
                    at.x * held[row * 4] + held[row * 4 + 2],
                    at.y * held[row * 4 + 1] + held[row * 4 + 3],
                )
            };
            let (low, high) = (Vec2::new(0.5, 0.5), Vec2::new(1919.5, 1079.5));
            assert!(
                corner(0, low).abs_diff_eq(Vec2::ZERO, 1e-3),
                "the near corner of the frame reads the near corner of it: {held:?}"
            );
            assert!(
                corner(0, high).abs_diff_eq(Vec2::ONE, 1e-3),
                "and the far one the far: {held:?}"
            );
            assert!(
                corner(1, low).abs_diff_eq(Vec2::NEG_ONE, 1e-3),
                "clip space runs the same way: {held:?}"
            );
            assert!(
                corner(1, high).abs_diff_eq(Vec2::ONE, 1e-3),
                "at both ends: {held:?}"
            );
        }
    }

    /// What the shadow softening sizes a penumbra with, read back through the arithmetic its own
    /// shader does: a unit of distance between a blocker and what it falls on has to come out as
    /// `SUN_SOFTNESS` world units of penumbra, on every split.
    #[test]
    fn the_shadow_penumbra_is_as_wide_as_the_sun_stands() {
        for split in 0..SPLITS {
            let scene = Scene {
                view: Mat4::look_at_rh(Vec3::new(0.0, 3.0, 12.0), Vec3::ZERO, Vec3::Y),
                projection: Mat4::perspective_rh(0.96, 1.6, 0.1, 1000.0),
                light: Vec3::new(0.3, 0.8, 0.5).normalize(),
                reach: 300.0,
                split,
                ..Default::default()
            };
            let held = Buffer {
                name: DIRECTIONAL_SHADOW_PARAM.to_owned(),
                members: [("m_ShadowProjectionMatrix", 0, 64), ("m_NearFarParam", 112, 16)]
                    .into_iter()
                    .map(|(name, offset, size)| hlsl::layout::Member {
                        name: name.to_owned(),
                        offset,
                        size,
                        kind: "float4".to_owned(),
                    })
                    .collect(),
                registers: 8,
                fixed: None,
            };
            let filled: Vec<f32> = held
                .fill(&scene, Pass::Lighting, &[])
                .chunks_exact(4)
                .map(|held| f32::from_le_bytes(held.try_into().unwrap()))
                .collect();
            let row = |at: usize| Vec3::new(filled[at * 4], filled[at * 4 + 1], filled[at * 4 + 2]);
            let [x, y, z, w] = [filled[28], filled[29], filled[30], filled[31]];
            // A world unit along the light moves the map's depth by the length of the row that
            // answers it, and this is what the shader turns that depth back into a distance with.
            let along = |depth: f32| -(depth * y + z) / w;
            let step = row(2).length();
            assert!((along(step) - along(0.0) - 1.0).abs() < 1e-4, "split {split}");
            // Its disc is stated in the shorter side of the whole image, and that many texels of
            // one split's own cell is what the penumbra measures in the world. Against the width
            // the frame was captured at rather than against the constant, which would move with it.
            let short = (SHADOW_MAP * ATLAS_COLUMNS.min(ATLAS_ROWS) as i32) as f32;
            let across = -x * short / (SHADOW_MAP as f32 * ATLAS_ROWS as f32 * row(1).length());
            assert!((across - 0.007).abs() < 1e-6, "split {split}: {across}");
        }
    }

    /// Where a lit pixel is read in the map a cloud's shadow was drawn into, against the matrix a
    /// captured frame of Ishgard bound at the same camera and hour. The frame's own view goes in and
    /// the whole of the game's own matrix has to come out: the box, its planes, and the turn that
    /// stands the map's camera under the light are each only right if every lane of this lands.
    #[test]
    fn the_cloud_shadow_matrix_comes_out_as_the_game_held_it() {
        // Rows of the frame's own world-to-view matrix, as its camera buffer holds them.
        let held = [
            [0.675_344_3, 9.536_745e-7, 0.737_502_75, 47.094_776],
            [-0.298_507_6, 0.914_425_9, 0.273_347_5, -128.917_86],
            [-0.674_391_3, -0.404_753_77, 0.617_552_34, -269.328_43],
        ];
        let column = |at: usize| Vec4::new(held[0][at], held[1][at], held[2][at], 0.0);
        let scene = Scene {
            view: Mat4::from_cols(column(0), column(1), column(2), column(3) + Vec4::W),
            light: Vec3::new(0.806_444_64, 0.512_089_13, 0.295_654_77),
            ..Default::default()
        };
        let buffer = Buffer {
            name: CLOUD_SHADOW_MATRIX.to_owned(),
            members: vec![hlsl::layout::Member {
                name: CLOUD_SHADOW_MATRIX.to_owned(),
                offset: 0,
                size: 64,
                kind: "float4".to_owned(),
            }],
            registers: 4,
            fixed: None,
        };
        let filled: Vec<f32> = buffer
            .fill(&scene, Pass::Lighting, &[])
            .chunks_exact(4)
            .map(|held| f32::from_le_bytes(held.try_into().unwrap()))
            .collect();
        let held = [
            -2.184_979_4e-4,
            -5.585_666e-4,
            -5.960_442_4e-4,
            0.0,
            -3.103_349_6e-4,
            3.582_748_8e-4,
            -2.219_851_2e-4,
            0.0,
            -3.817_188_5e-5,
            -1.543_313_7e-5,
            2.845_579_4e-5,
            -1.001_001e-3,
            0.0,
            0.0,
            0.0,
            1.0,
        ];
        // The lanes that carry the box are exact to a part in ten thousand of their own last digit;
        // the ones that carry nothing land on the same 1e-8 of float noise the game's own frames do.
        for (at, (filled, held)) in filled.iter().zip(held).enumerate() {
            assert!(
                (filled - held).abs() < 2e-8,
                "lane {at}: {filled} against {held}"
            );
        }
    }

    /// The three buffers the exposure chain reads, against the bytes a capture of the running game
    /// held in them. What the environment stated at that time and weather goes in; what the frame
    /// was measured and read under has to come out.
    #[test]
    fn the_exposure_buffers_come_out_as_the_game_held_them() {
        let scene = Scene {
            exposure: Exposure {
                min: 1.0,
                max: 3.525834,
                rate: 2.0,
                key: 0.347417,
                strength: 0.5,
                shoulder: 0.95,
                step: 0.027_392_5,
                adapted: 1.0,
                encode: 0.698_801_4,
            },
            ..Default::default()
        };
        let filled = |name: &str, registers| {
            let held = Buffer {
                name: name.to_owned(),
                members: Vec::new(),
                registers,
                fixed: None,
            };
            held.fill(&scene, Pass::Composite, &[])
                .chunks_exact(4)
                .map(|held| f32::from_le_bytes(held.try_into().unwrap()))
                .collect::<Vec<f32>>()
        };
        let close = |held: &[f32], want: &[f32]| {
            held.iter()
                .zip(want)
                .all(|(held, want)| (held - want).abs() <= want.abs() * 1e-4 + 1e-6)
        };

        // The key goes in squared and the rate scaled by the frame, which is the whole of what the
        // capture proved and neither of which is guessable off the field names.
        assert!(close(
            &filled("cAdaptLumParam", 1),
            &[1.0, 3.525834, 0.054785, 0.120698]
        ));
        assert!(close(&filled("cCommonTexParam", 1), &[0.698801, 1.431022, 0.0, 0.0]));
        // The curve's bounds are half a texel in from either end of a strip 1024 wide, and its last
        // lane is the exposure over twice that. The game held `z` a frame older than the rest, so
        // 1.432421 there rather than the 1.431022 this fills.
        assert!(close(
            &filled("cToneMapParam", 2),
            &[
                0.5,
                0.95,
                0.698801,
                0.0,
                0.00048828125,
                0.99951171875,
                1.431022,
                0.000698741,
            ]
        ));
        // The one number here no file states, against the frame that held the largest adaptation
        // measured. The tolerance is the drift a single frame of the game's own shows.
        assert!((encode(3.0) - 1.21097).abs() < 1.21097 * 4e-3);
    }

    /// A frame long enough that the stated rate would carry the whole of its own measurement. The
    /// measure squares the exposure, so a weight of one leaves the loop swinging between its clamps
    /// rather than settling.
    #[test]
    fn a_slow_frame_carries_no_more_than_the_loop_settles_under() {
        let scene = Scene {
            exposure: Exposure {
                rate: 2.0,
                step: 19.5,
                ..Default::default()
            },
            ..Default::default()
        };
        let held = Buffer {
            name: ADAPT_LUM_PARAM.to_owned(),
            members: Vec::new(),
            registers: 1,
            fixed: None,
        };
        let filled: Vec<f32> = held
            .fill(&scene, Pass::Composite, &[])
            .chunks_exact(4)
            .map(|held| f32::from_le_bytes(held.try_into().unwrap()))
            .collect();
        assert_eq!(filled[2], SETTLE);
    }

    /// Water's own reflection chain reads one buffer for everything it is told, against the bytes a
    /// capture of the running game held in it: the frame there is 2560 by 1440 and the chain runs at
    /// half of it, which is why the game's own upload writes the same register twice. The fog is the
    /// one that zone's own environment states at the hour the capture was taken.
    #[test]
    fn the_water_reflection_buffer_comes_out_as_the_game_held_it() {
        let scene = Scene {
            size: (2560.0, 1440.0),
            reflect: Reflect {
                level: 0,
                texel: Vec2::new(1.0 / 1280.0, 1.0 / 720.0),
            },
            fog: Fog {
                color: Vec3::new(0.734657, 0.543774, 0.758775),
                cap: 0.860343,
                rate: 0.0001,
                start: 200.0,
                ..Fog::default()
            },
            ..Default::default()
        };
        let held = Buffer {
            name: REFLECTION_PARAM.to_owned(),
            members: Vec::new(),
            registers: 9,
            fixed: None,
        };
        let filled: Vec<f32> = held
            .fill(&scene, Pass::WaterMirror, &[])
            .chunks_exact(4)
            .map(|held| f32::from_le_bytes(held.try_into().unwrap()))
            .collect();
        let step = [1.0 / 1280.0, 1.0 / 720.0, 1.0 / 2560.0, 1.0 / 1440.0];
        assert_eq!(filled[0..4], step);
        assert_eq!(filled[4..8], [
            1.0 / 2560.0,
            1.0 / 1440.0,
            1.0 / 5120.0,
            1.0 / 2880.0
        ]);
        assert_eq!(filled[8..12], step);
        assert_eq!(filled[12..16], [0.734657, 0.543774, 0.758775, 0.860343]);
        assert_eq!(filled[16..20], [0.0001, 200.0, 3500.0, 1.0]);
        for (at, want) in [
            0.0891034, 0.0856096, 0.0790276, 0.0700912, 0.0597278, 0.048901, 0.0384669, 0.0290726,
        ]
        .into_iter()
        .enumerate()
        {
            assert!(
                (filled[20 + at] - want).abs() < 1e-6,
                "tap {at} came out {} rather than {want}",
                filled[20 + at]
            );
        }
        assert_eq!(filled[28..32], [1.0, 1.0, 0.0, 0.0]);
        assert_eq!(filled[32], 0.0);
    }

    /// The same two registers under a second zone, against a capture taken there: neither lane the
    /// march reads is the first zone's.
    #[test]
    fn the_water_reflection_fog_is_the_zone_standing_under_it() {
        let scene = Scene {
            fog: Fog {
                color: Vec3::new(0.478431, 0.672222, 0.783333),
                cap: 0.995752,
                rate: 0.00026,
                start: 300.0,
                ..Fog::default()
            },
            ..Default::default()
        };
        let held = Buffer {
            name: REFLECTION_PARAM.to_owned(),
            members: Vec::new(),
            registers: 9,
            fixed: None,
        };
        let filled: Vec<f32> = held
            .fill(&scene, Pass::WaterMirror, &[])
            .chunks_exact(4)
            .map(|held| f32::from_le_bytes(held.try_into().unwrap()))
            .collect();
        assert_eq!(filled[12..16], [0.478431, 0.672222, 0.783333, 0.995752]);
        assert_eq!(filled[16..20], [0.00026, 300.0, 3500.0, 1.0]);
    }

    /// The two members of that chain drawn over the water itself take a vertex into view space
    /// through one register run, and the reflection their file carries names no field for a write to
    /// land in.
    #[test]
    fn water_reflection_takes_a_vertex_into_view_space() {
        let scene = Scene {
            view: Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0)),
            model: Mat4::from_scale(Vec3::new(2.0, 2.0, 2.0)),
            ..Default::default()
        };
        let held = Buffer {
            name: INSTANCE.to_owned(),
            members: Vec::new(),
            registers: 4,
            fixed: None,
        };
        let filled: Vec<f32> = held
            .fill(&scene, Pass::WaterMirror, &[])
            .chunks_exact(4)
            .map(|held| f32::from_le_bytes(held.try_into().unwrap()))
            .collect();
        // One row per register, the way the shader dots them: the scale down the diagonal and the
        // translation in the last lane.
        assert_eq!(filled[0..4], [2.0, 0.0, 0.0, 1.0]);
        assert_eq!(filled[4..8], [0.0, 2.0, 0.0, 2.0]);
        assert_eq!(filled[8..12], [0.0, 0.0, 2.0, 3.0]);
    }

    /// The instance record a character is drawn with, against the bytes a capture of the running
    /// game held in it. Two registers of the eleven are filled here and neither reads as one field:
    /// the camera light spans a pair, and a write cut to four floats would leave the rim at nought.
    #[test]
    fn the_camera_light_comes_out_as_the_game_held_it() {
        let held = Buffer {
            name: INSTANCE.to_owned(),
            members: instance_fields(),
            registers: 11,
            fixed: None,
        };
        let filled: Vec<f32> = held
            .fill(&Scene::default(), Pass::Composite, &[])
            .chunks_exact(4)
            .map(|held| f32::from_le_bytes(held.try_into().unwrap()))
            .collect();
        assert_eq!(filled[4..8], [0.0, 0.0, 0.0, 0.17]);
        assert_eq!(filled[8..12], [0.15, 0.15, 0.15, 0.17]);
        assert_eq!(filled[12..16], [0.01584, 0.9, 0.01584, 0.8]);
    }

    /// The two buffers a face paint reaches, filled the way a character package declares them. The
    /// decal's own buffer holds one bare array named after itself, which a reflection describes with
    /// no fields at all: a write by name against that lands nowhere and reports nothing.
    #[test]
    fn a_face_paint_reaches_both_buffers_it_is_read_through() {
        let member = |name: &str, offset, size| hlsl::layout::Member {
            name: name.to_owned(),
            offset,
            size,
            kind: "float4".to_owned(),
        };
        let scene = Scene {
            customize: Customize {
                hair: [0.1, 0.2, 0.3, 0.4],
                highlight: [0.5, 0.6, 0.7, 0.8],
                decal: [0.9, 0.8, 0.7, 0.6],
                ..Default::default()
            },
            ..Default::default()
        };
        let filled = |name: &str, members: Vec<hlsl::layout::Member>, registers| {
            let held = Buffer {
                name: name.to_owned(),
                members,
                registers,
                fixed: None,
            };
            held.fill(&scene, Pass::Buffer, &[])
                .chunks_exact(4)
                .map(|held| f32::from_le_bytes(held.try_into().unwrap()))
                .collect::<Vec<f32>>()
        };
        assert_eq!(filled(DECAL, decal_field(), 1), [0.9, 0.8, 0.7, 0.6]);
        // The last lane of each hair colour places the decal across the face rather than weighing
        // the hair, so the palette's own alpha has no business in either.
        let held = filled(
            "g_CustomizeParameter",
            vec![member("m_MainColor", 32, 16), member("m_MeshColor", 48, 16)],
            4,
        );
        assert_eq!(held[8..12], [0.1, 0.2, 0.3, 1.0]);
        assert_eq!(held[12..16], [0.5, 0.6, 0.7, 0.0]);
    }

    /// The fog reads a distance out of the depth buffer as `1 / (y * d + x)`, and everything it then
    /// does with that distance rests on those two lanes. Rather than argue the convention, this
    /// pushes a distance through the projection the zone is drawn with and asks for it back.
    #[test]
    fn a_depth_reading_comes_back_the_distance_it_stood_for() {
        let projection = Mat4::perspective_rh(1.0, 1.6, 0.1, 8000.0);
        let scene = Scene {
            projection,
            fog: Fog {
                cap: 0.9,
                rate: 0.0005,
                start: 100.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let held = Buffer {
            name: FOG_PARAM.to_owned(),
            members: Vec::new(),
            registers: 6,
            fixed: None,
        };
        let filled: Vec<f32> = held
            .fill(&scene, Pass::Composite, &[])
            .chunks_exact(4)
            .map(|held| f32::from_le_bytes(held.try_into().unwrap()))
            .collect();
        for want in [1.0f32, 100.0, 500.0, 2000.0] {
            // Where the projection leaves a point that far in front of the camera, and what the
            // shaders' own fixup makes of it: they hand the card a clip depth over the whole of
            // `[-w, w]`, and the buffer holds the half of that between nought and one.
            let clip = projection * glam::Vec4::new(0.0, 0.0, -want, 1.0);
            let depth = (clip.z / clip.w * 2.0 - 1.0) * 0.5 + 0.5;
            let read = 1.0 / (filled[21] * depth + filled[20]);
            assert!((read - want).abs() < want * 1e-3, "{want} came back {read}");
        }
        // Texel nought stands where the fog starts, and the last where its opacity reaches the cap.
        let coordinate = |z: f32| filled[19] * z + filled[15];
        assert!((coordinate(100.0) - 0.5 / 256.0).abs() < 1e-6);
        assert!((coordinate(1900.0) - 255.5 / 256.0).abs() < 1e-6);
    }

    /// A camera turned to face the sun should find it dead centre. The pass measures every pixel
    /// against the place this states, so an error here moves the whole glow rather than distorting
    /// it, which on screen reads as a sun in the wrong part of the sky.
    #[test]
    fn the_sun_lands_where_the_camera_looks() {
        let time = 51_000.0;
        let tilt = 5.0;
        let toward = sun(time, tilt);
        let eye = Vec3::new(-6.535, 18.583, 36.727);
        let projection = Mat4::perspective_rh(55.0f32.to_radians(), 1251.0 / 913.0, 0.1, 8000.0);
        let scene = Scene {
            view: Mat4::look_at_rh(eye, eye + toward, Vec3::Y),
            projection,
            size: (1251.0, 913.0),
            sky: Sky {
                time,
                tilt,
                ..Default::default()
            },
            ..Default::default()
        };
        let held = Buffer {
            name: SUN_PARAM.to_owned(),
            members: Vec::new(),
            registers: 5,
            fixed: None,
        };
        let filled: Vec<f32> = held
            .fill(&scene, Pass::Composite, &[])
            .chunks_exact(4)
            .map(|held| f32::from_le_bytes(held.try_into().unwrap()))
            .collect();
        assert!(
            (filled[2] - 0.5).abs() < 1e-4 && (filled[3] - 0.5).abs() < 1e-4,
            "the sun stands at {}, {} rather than the middle",
            filled[2],
            filled[3]
        );
    }

    /// Against the two days a capture of the running game held `cMoonParam` for: a waning crescent
    /// and a full moon, both at the same hour.
    #[test]
    fn a_days_phase_matches_what_the_game_held_for_it() {
        let crescent = moon_phase(30.0);
        assert!((crescent - 0.1875).abs() < 1e-6);
        let axis = moon_terminator(crescent);
        assert!((axis.x - 0.555_570).abs() < 1e-5);
        assert!((axis.y - -0.831_470).abs() < 1e-5);

        let full = moon_phase(17.0);
        assert!((full - 1.0).abs() < 1e-6);
        let axis = moon_terminator(full);
        assert!(axis.x.abs() < 1e-6);
        assert!((axis.y - 1.0).abs() < 1e-6);
    }

    /// `moon_softness` is a closed form, not a fit: it agrees with what the game held for the
    /// crescent to five significant figures and misses in the sixth, which this tolerance records
    /// rather than hides.
    #[test]
    fn a_days_softness_is_close_to_what_the_game_held_for_it() {
        let (slope, offset) = moon_softness(moon_phase(30.0));
        assert!((slope - 1.461_532).abs() < 1e-5);
        assert!((offset - 0.315_787).abs() < 1e-5);

        let (slope, offset) = moon_softness(moon_phase(17.0));
        assert_eq!(slope, 999_999.0);
        assert!((offset - 0.999_999).abs() < 1e-6);
    }

    /// Two frames a day apart at the same hour hold `cMoonParam[0].xy` identically; this is the
    /// turn that explains why.
    #[test]
    fn moon_roll_matches_what_the_game_held_at_the_hour() {
        let roll = moon_roll(3.0 * 3600.0);
        assert!((roll.x - 0.707_107).abs() < 1e-5);
        assert!((roll.y - -0.707_107).abs() < 1e-5);
    }

    /// The composite reads entry `n` at registers `12 * n + 4` through `12 * n + 15`, and its header
    /// at nought through three. Nothing in the reflection lays the entry out, so this is the whole
    /// statement of where each field goes.
    #[test]
    fn the_ambient_entry_starts_at_the_fourth_register() {
        let held = Ambient {
            sky: [Vec4::splat(1.0), Vec4::splat(2.0), Vec4::splat(3.0)],
            sky_scale: 4.0,
            light: [Vec4::splat(5.0), Vec4::splat(6.0), Vec4::splat(7.0)],
            scale: 8.0,
            fade: Vec3::new(9.0, 10.0, 11.0),
            reflection: Vec3::new(12.0, 13.0, 14.0),
            capture: 15.0,
            volumes: std::sync::Arc::from([] as [Volume; 0]),
        };
        let mut out = vec![0u8; 16 * 16];
        ambient(&held, Mat3::IDENTITY, Mat4::IDENTITY, &mut out);
        let lane = |register: usize, at: usize| {
            let start = register * 16 + at * 4;
            f32::from_le_bytes(out[start..start + 4].try_into().unwrap())
        };
        assert_eq!(u32::from_le_bytes(out[..4].try_into().unwrap()), 1);
        assert_eq!(lane(0, 1), 4.0);
        assert_eq!([lane(1, 0), lane(2, 0), lane(3, 0)], [1.0, 2.0, 3.0]);
        assert_eq!([lane(4, 0), lane(5, 0), lane(6, 0)], [5.0, 6.0, 7.0]);
        assert_eq!(lane(7, 3), 8.0);
        assert_eq!([lane(8, 0), lane(8, 1), lane(8, 2)], [9.0, 10.0, 11.0]);
        assert_eq!(
            [lane(9, 0), lane(9, 1), lane(9, 2), lane(9, 3)],
            [12.0, 13.0, 14.0, 15.0]
        );
        // Past the entry's own twelve registers nothing is written: the next one along is entry one.
        assert!(out[16 * 16..].is_empty());
    }

    /// The row is dotted against a normal and a one, and the file runs constant, `y`, `z`, `x`. Each
    /// term carries the weight a cosine lobe gathers it at, which a real frame's own buffer matches
    /// to seven figures; the ratio between the two weights is `2/sqrt(3)`.
    #[test]
    fn a_harmonic_row_is_convolved_and_puts_the_constant_last() {
        let row = Ambient::row(&[1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let close = |held: f32, want: f32| (held - want).abs() < 1e-6;
        assert!(close(row.w, 0.886_226_9));
        assert!(close(row.x, 4.0 * 1.023_326_7));
        assert!(close(row.y, 2.0 * 1.023_326_7));
        assert!(close(row.z, 3.0 * 1.023_326_7));
        assert!(close(row.x / row.w / 4.0, 2.0 / 3.0f32.sqrt()));
        // The `y` lane is what a normal pointing up reads, beside the constant every normal reads.
        assert!(close(
            row.dot(Vec4::new(0.0, 1.0, 0.0, 1.0)),
            2.0 * 1.023_326_7 + 0.886_226_9
        ));
    }

    #[test]
    fn the_selector_is_a_polynomial_in_thirty_one() {
        assert_eq!(selector(&[]), 0);
        assert_eq!(selector(&[7]), 7);
        assert_eq!(selector(&[1, 1]), 32);
        assert_eq!(selector(&[0, 0, 1]), 961);
        assert_eq!(selector(&[u32::MAX, 2]), u32::MAX.wrapping_add(62));
    }

    /// Four columns of three floats each, which is how the shader rebuilds a row: it takes the first
    /// component of each of its four reads.
    #[test]
    fn a_joint_is_four_columns_of_three() {
        let held = joints(
            &[Mat4::IDENTITY; 2],
            Mat4::from_translation(Vec3::new(4.0, 5.0, 6.0)),
        );
        assert_eq!(held.len(), ROW);
        let value = |lane: usize| f32::from_bits(held[lane]);
        assert_eq!(
            (0..JOINT).map(value).collect::<Vec<_>>(),
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 4.0, 5.0, 6.0]
        );
        assert_eq!(&held[JOINT..JOINT * 2], &held[..JOINT]);
    }

    /// A parameter file of one profile, written the way the shipping ones are.
    fn parameters(columns: &[(u32, u32)], values: &[u32]) -> Vec<u8> {
        // Offsets are counted in words, and the header is three of them.
        let columns_at = 3u16;
        let rows_at = columns_at + 2 * columns.len() as u16;
        let values_at = rows_at + 2;

        let mut bytes = 0x0100_0000u32.to_le_bytes().to_vec();
        bytes.push(columns.len() as u8);
        bytes.push(1);
        bytes.extend(columns_at.to_le_bytes());
        bytes.extend(rows_at.to_le_bytes());
        bytes.extend(values_at.to_le_bytes());
        for (id, kind) in columns {
            bytes.extend(id.to_le_bytes());
            bytes.extend(kind.to_le_bytes());
        }
        bytes.extend(0xB9FD_FB6Cu32.to_le_bytes());
        bytes.extend(0u32.to_le_bytes());
        for value in values {
            bytes.extend(value.to_le_bytes());
        }
        bytes
    }

    /// Nothing in a file says where its parameters go: the record is laid out by what the shaders
    /// read, and a file states whichever of them its own family uses, in whichever order it likes.
    #[test]
    fn a_profile_fills_the_record_the_shaders_read() {
        let file = ShaderParameters::read(Cursor::new(parameters(
            &[
                (0xF33F_F064, 0),
                (0x8FB5_3404, 1),
                (0xE800_1A59, 2),
                (0x4133_8E94, 0),
            ],
            &[13.0f32.to_bits(), 5, 0x56F1_6FCB, 1.0f32.to_bits()],
        )))
        .unwrap();

        let held = shader_types(&[(32, &file)]);
        let record = &held[32 * SHADER_TYPE..33 * SHADER_TYPE];
        assert_eq!(record[0], 2);
        assert_eq!(record[1], 5);
        assert_eq!(f32::from_bits(record[3]), 1.0);
        // The lobe this centers is a Gaussian over a sine, and the file states it in degrees.
        assert_eq!(f32::from_bits(record[9]), 13.0f32.to_radians());
        assert!(held[..32 * SHADER_TYPE].iter().all(|held| *held == 0));
    }

    /// The two numbers the engine's own bg renderer decides for a sway rather than reading out of a
    /// file: the weight it writes once at startup and never again, and the rate its phase runs at.
    #[test]
    fn a_sway_carries_the_weight_and_the_rate_the_engine_states() {
        let member = |name: &str, offset| hlsl::layout::Member {
            name: name.to_owned(),
            offset,
            size: 16,
            kind: "float4".to_owned(),
        };
        let floats = |held: Vec<u8>| -> Vec<f32> {
            held.chunks_exact(4)
                .map(|held| f32::from_le_bytes(held.try_into().unwrap()))
                .collect()
        };
        let waving = Buffer {
            name: WAVING.to_owned(),
            members: vec![
                member("m_WindVector", 0),
                member("m_UpVector", 16),
                member("m_WavingParam", 32),
            ],
            registers: 3,
            fixed: None,
        };
        let scene = Scene {
            wind: Wind {
                heading: Vec3::Z,
                ..Default::default()
            },
            ..Default::default()
        };
        let filled = floats(waving.fill(&scene, Pass::Buffer, &[]));
        // The wind the engine hands over, not the reach the set sums to.
        assert_eq!(filled[..3], [0.0, 0.0, 1.467_972]);
        assert_eq!(filled[8..12], [1.0, 1.0, 0.2, 1.0]);

        let phase = |clock| {
            let held = Buffer {
                name: INSTANCING.to_owned(),
                members: vec![member("m_WavingAnimTime", 0)],
                registers: 1,
                fixed: None,
            };
            floats(held.fill(&Scene { clock, ..Default::default() }, Pass::Buffer, &[]))[0]
        };
        // Two seconds of it, at the one radian a second the engine states.
        assert!((phase(3.0) - phase(1.0) - 2.0).abs() < 1e-5);
    }

    /// What a strand of hair is swayed along and the clock it flutters on, neither of which any file
    /// states and both of which the engine drives.
    #[test]
    fn a_strand_takes_a_capped_wind_and_a_wrapping_clock() {
        let floats = |held: Vec<u8>| -> Vec<f32> {
            held.chunks_exact(4)
                .map(|held| f32::from_le_bytes(held.try_into().unwrap()))
                .collect()
        };
        let instance = Buffer {
            name: INSTANCE.to_owned(),
            members: instance_fields(),
            registers: 11,
            fixed: None,
        };
        let gust = |reach| {
            let scene = Scene {
                wind: Wind {
                    heading: Vec3::Z,
                    reach,
                    ..Default::default()
                },
                ..Default::default()
            };
            floats(instance.fill(&scene, Pass::Buffer, &[]))[20..24].to_vec()
        };
        // A unit heading taken down by two thousand, and a last lane the engine zeroes outright.
        assert!((gust(4.0)[2] - 0.002).abs() < 1e-9);
        assert_eq!(gust(4.0)[0], 0.0);
        assert_eq!(gust(4.0)[3], 0.0);
        // Past thirty the speed stops counting.
        assert!((gust(200.0)[2] - 0.015).abs() < 1e-9);
        assert_eq!(gust(200.0), gust(30.0));

        let loop_time = |clock| {
            let held = Buffer {
                name: "g_PbrParameterCommon".to_owned(),
                members: vec![hlsl::layout::Member {
                    name: "m_LoopTime".to_owned(),
                    offset: 0,
                    size: 4,
                    kind: "float".to_owned(),
                }],
                registers: 1,
                fixed: None,
            };
            floats(held.fill(&Scene { clock, ..Default::default() }, Pass::Buffer, &[]))[0]
        };
        // Held to a tick, and back to nought where the accumulator wraps.
        assert_eq!(loop_time(1.0 + 0.5 / 1024.0), 1.0);
        assert_eq!(loop_time(2048.0), 0.0);
        assert_eq!(loop_time(2049.0), 1.0);
    }

    /// The register water wanders its whitecaps by, which is the noise texture's own size: the
    /// shader multiplies a world position by `.zw`, so those have to be the reciprocal of a real
    /// texture rather than a one.
    #[test]
    fn the_whitecap_noise_tiles_across_the_texture_the_engine_loads() {
        let floats = |held: Vec<u8>| -> Vec<f32> {
            held.chunks_exact(4)
                .map(|held| f32::from_le_bytes(held.try_into().unwrap()))
                .collect()
        };
        let water = Buffer {
            name: "g_WaterParameter".to_owned(),
            members: vec![hlsl::layout::Member {
                name: "m_NoiseSize".to_owned(),
                offset: 0,
                size: 16,
                kind: "float4".to_owned(),
            }],
            registers: 1,
            fixed: None,
        };
        let filled = floats(water.fill(&Scene::default(), Pass::Water, &[]));
        assert_eq!(filled[..2], [128.0, 128.0]);
        assert_eq!(filled[2..4], [1.0 / 128.0, 1.0 / 128.0]);
    }

    /// The buffer the game itself uploads at the Tuliyollal preset, where the zone's own `.envb`
    /// states two layers of strength 8 and 1 at azimuth 90 and 120, both reaching nought at their
    /// weakest. Read out of `~/rdcaps/tuli.zip`.
    #[test]
    fn a_blade_leans_between_the_pair_the_game_hands_it() {
        let floats = |held: Vec<u8>| -> Vec<f32> {
            held.chunks_exact(4)
                .map(|held| f32::from_le_bytes(held.try_into().unwrap()))
                .collect()
        };
        let layer = |azimuth: f32, max_strength, wavelength| {
            let held = f32::to_radians(azimuth);
            WindLayer {
                heading: Vec3::new(-held.sin(), 0.0, held.cos()),
                max_strength,
                min_strength: 0.0,
                wavelength,
            }
        };
        let info = Buffer {
            name: "g_WindInfo".to_owned(),
            members: Vec::new(),
            registers: 12,
            fixed: None,
        };
        let scene = Scene {
            clock: 0.0,
            wind: Wind {
                layers: [layer(90.0, 8.0, 512.0), layer(120.0, 1.0, 128.0)],
                ..Default::default()
            },
            ..Default::default()
        };
        let filled = floats(info.fill(&scene, Pass::Buffer, &[]));
        let close = |held: f32, want: f32| assert!((held - want).abs() < 1e-5, "{held} not {want}");
        close(filled[0], -1.0);
        close(filled[1], 0.0);
        close(filled[2], 0.0);
        close(filled[3], 1.2);
        close(filled[6], 1.0 / 512.0);
        close(filled[7], 0.0);
        close(filled[12], -0.866_025);
        close(filled[13], 0.0);
        close(filled[14], -0.5);
        close(filled[15], 0.15);
        close(filled[18], 1.0 / 128.0);
        close(filled[19], 0.0);
    }

    /// The gust the engine advects: a cycle over `wavelength` world units, carried along the
    /// layer's heading by its own strength, and wrapped every cycle.
    #[test]
    fn a_gust_scrolls_by_the_strength_the_layer_states() {
        let floats = |held: Vec<u8>| -> Vec<f32> {
            held.chunks_exact(4)
                .map(|held| f32::from_le_bytes(held.try_into().unwrap()))
                .collect()
        };
        let layer = WindLayer {
            heading: Vec3::Z,
            max_strength: 8.0,
            min_strength: 2.0,
            wavelength: 512.0,
        };
        let held = |clock| {
            let info = Buffer {
                name: "g_WindInfo".to_owned(),
                members: Vec::new(),
                registers: 12,
                fixed: None,
            };
            let scene = Scene {
                clock,
                wind: Wind { layers: [layer; 2], ..Default::default() },
                ..Default::default()
            };
            floats(info.fill(&scene, Pass::Buffer, &[]))
        };

        let filled = held(30.0);
        // The world-to-uv scale is the stated cycle length and nothing else.
        assert_eq!(filled[6], 1.0 / 512.0);
        assert_eq!(filled[3], (8.0 - 2.0) * WIND_POWER_SCALE);
        assert_eq!(filled[7], 2.0 * WIND_POWER_SCALE);
        // Thirty seconds at strength eight carries the field eight cycles' worth of texels.
        assert!((filled[5] - (8.0 / 512.0)).abs() < 1e-6, "{}", filled[5]);
        assert_eq!(filled[4], 0.0);
        // A clock long enough to have wrapped stays inside the texture rather than drifting off it.
        let far = held(30.0 * 512.0 / 8.0 + 30.0);
        assert!((far[5] - (8.0 / 512.0)).abs() < 1e-3, "{}", far[5]);
    }
}
