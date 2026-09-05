//! The frame the game's own shaders draw into, and the passes that resolve it.
//!
//! A G-buffer written a page of targets at a time, the buffers the lighting passes fill from it, and
//! the composite that turns those into a frame. One model and a whole zone want the same thing here,
//! so this is what they share; what differs is only the draw list that fills the G-buffer.

use std::collections::{BTreeMap, HashMap};

use glow::HasContext;

use super::program;

/// The frame's own buffers, as the lighting and composite passes name them.
const GBUFFER: [u32; 5] = [
    0xebbb_29bd,
    0xe4e5_7422,
    0x7dec_2598,
    0x0aeb_150e,
    0x948f_80ad,
];
const DEPTH: u32 = 0x2c8f_f4b0;
const VIEW_POSITION: u32 = 0xbc61_5663;
/// The same buffer under the name water and river know it by.
const WATER_VIEW_POSITION: u32 = 0x34a0_4363;
/// What water reads for whatever stands behind it, which is the frame as the lighting left it.
const REFRACTION: u32 = 0xa38e_45e1;
/// What water reads its own local reflection through, and all of it: `river.shpk` reaches no cube.
/// [`WaterMirror`] fills it, which is not the frame-wide chain [`Reflection`] runs; where that chain
/// has not drawn, it falls to the neutral answer below.
const REFLECTION_MAP: u32 = 0xc705_a5b6;
const LIGHT_DIFFUSE: u32 = 0x23d0_f850;
const LIGHT_SPECULAR: u32 = 0x6c19_aca4;
const OCCLUSION: u32 = 0x3266_7bd7;
/// What the lighting reads a shadow through. Not the same as the occlusion above: shadows and
/// ambient occlusion are separate consumers and the lighting package declares both.
const SHADOW_MASK: u32 = 0x8187_d13f;
/// The sun's own depth map, which the resolve compares a pixel against.
const SHADOW_DEPTH: u32 = 0x58ad_2b38;
/// What the lighting reads a cloud's shadow through, which the sheet fills from the sun's own side.
/// Where no sheet is drawn it falls to the opaque white below and the term comes to one.
const CLOUD_SHADOW: u32 = 0xb821_f0d3;
/// What the softening gathers that same map's own depths through rather than comparing against
/// them, spelled the way `directionalshadow` spells it.
const GATHER_SAMPLER: &str = "g_SamplerGahter";
/// The table the subsurface blur reads its taps out of. The engine builds it per frame and no file
/// states one, so what ships here is the table read whole off a frame the game drew: 32 taps by 64
/// rows, `.xyz` a weight per channel and `.w` the offset.
const SUBSURFACE_KERNEL: u32 = 0x3b44_510e;
const SUBSURFACE: &[u8] = include_bytes!("subsurface.bin");
const SUBSURFACE_TAPS: i32 = 32;
const SUBSURFACE_ROWS: i32 = 64;
const ATTENUATION: u32 = 0x008c_d1ca;

/// The frame as the composite left it, which is what a semitransparent pass blends over.
const FINAL_COLOR: u32 = 0x8ea9_df48;

/// What every member of the post chain calls the frame it reads.
const INPUT: u32 = 0x527d_95a1;

/// The second buffer the bright pass reads, which nothing here writes. It decodes a strength out of
/// two of the channels and squares the alpha away from one, so what it wants is the stand-in for a
/// weight nothing filled rather than the flat grey every other unfilled sampler answers with.
const GLARE_GEOMETRY: u32 = 0xc028_4c84;

/// What the occlusion chain calls what it reads: the depth buffer and the G-buffer channel holding
/// the normal, then what each of its passes leaves for the next.
const DEPTH_PLANE: u32 = 0x70f6_bb1f;
const NORMAL_PLANE: u32 = 0x4a6b_06c7;
const DEPTH_NORMAL_Z: u32 = 0xe1fb_187c;
const GATHER_DEPTH: u32 = 0x5ee7_1209;
const GATHER_NORMAL_Z: u32 = 0x3d66_0475;

/// The G-buffer channel holding the surface normal. In the world rather than in front of the
/// camera: every pass that lights the frame brings it through the view matrix itself, and so does
/// the one that scales it down here.
const NORMAL_CHANNEL: usize = 0;

/// The ramp a placed light's falloff is read off, indexed by the square of how far the pixel stands
/// from it. The engine binds this rather than any material, and the flat stand-in leaves every light
/// at full strength out to the edge of its own volume, which is a hard circle.
pub const RAMP: (u32, &str, u32) = (
    ATTENUATION,
    "common/graphics/texture/-attenuation.tex",
    glow::LINEAR,
);

/// The textures the game's own shaders read that no material names, and how each is read between its
/// texels: the tiles a character's surface detail is taken from, the spheres its resolve pass shades
/// against, that ramp, the profiles the subsurface term is read off, and the ramps a cel-shaded
/// surface takes its whole light response off, a profile to a row and the cosine along the row.
///
/// The kernel is addressed at whole texels, a profile to a row and a Gaussian to a column, so
/// filtering it would answer with the mean of two profiles and of two Gaussians alike.
pub const ENGINE: [(u32, &str, u32); 17] = [
    // The two tiled arrays a background surface lays over its own textures up close, which its
    // material picks a layer of by `g_DetailID`. Without them a stone wall is its albedo and nothing
    // finer, however near the camera stands.
    (
        0xc8b8_827e,
        "bgcommon/nature/detail/texture/detail_n_array.tex",
        glow::LINEAR,
    ),
    (
        0x9f68_44e2,
        "bgcommon/nature/detail/texture/detail_d_array.tex",
        glow::LINEAR,
    ),
    (
        0x92f0_3e53,
        "chara/common/texture/tile_norm_array.tex",
        glow::LINEAR,
    ),
    (
        0x800b_e99b,
        "chara/common/texture/tile_orb_array.tex",
        glow::LINEAR,
    ),
    (
        0x3334_d3ca,
        "chara/common/texture/sphere_d_array.tex",
        glow::LINEAR,
    ),
    RAMP,
    (
        0x3b44_510e,
        "common/graphics/texture/-sss_kernel_ssst.tex",
        glow::NEAREST,
    ),
    (0x8b73_3c20, "chara/common/texture/-toon.tex", glow::LINEAR),
    (
        0xdce8_add5,
        "bgcommon/nature/moon/texture/moon.tex",
        glow::LINEAR,
    ),
    // What water reads beside the frame behind it: the ramp its reflectance is read off, the volume
    // its caustics are a slice of, and the two that wobble and step where that slice is taken.
    (
        0xba8d_7950,
        "common/graphics/texture/-fresnel.tex",
        glow::LINEAR,
    ),
    (
        0x0efb_24f7,
        "common/graphics/texture/-caustics.tex",
        glow::LINEAR,
    ),
    (
        0xd703_3544,
        "common/graphics/texture/-distortion.tex",
        glow::LINEAR,
    ),
    (
        0x2b85_7ef0,
        "common/graphics/texture/-noise.tex",
        glow::LINEAR,
    ),
    // `grass.shpk`'s own wind field: two 512x512 textures the AutoPlacement vertex shader samples
    // at a world-scaled UV, so unlike the rest of this table they read past a single tile and want
    // to repeat rather than clamp, handled in `Buffers::layered`.
    (WIND_SAMPLE_0, "bgcommon/nature/wind/texture/wind_001.tex", glow::LINEAR),
    (WIND_SAMPLE_1, "bgcommon/nature/wind/texture/wind_002.tex", glow::LINEAR),
    // The angle the shadow softening turns each pixel's disc of taps by, read at whole texels. The
    // game holds sixteen of these and steps to the next one every frame, which only pays off where
    // the frames are accumulated; one alone leaves the dither still rather than crawling.
    (
        0x94a1_94c2,
        "common/graphics/texture/-bn_64_00.tex",
        glow::NEAREST,
    ),
    // The threshold a dither clip tests an object's own opacity against, four by four and read at
    // a quarter of the pixel's own coordinate, so one texel covers one pixel.
    (DITHER, "common/graphics/texture/-dither.tex", glow::NEAREST),
];

const WIND_SAMPLE_0: u32 = 0x78d3_e3b7;
const WIND_SAMPLE_1: u32 = 0x0fd4_d321;
const DITHER: u32 = 0x9f46_7267;

/// The noise field a hair strand takes its flutter out of. Kept out of [`ENGINE`] because the
/// engine builds it rather than reading it: [`super::noise::perlin`] draws the same field, and
/// nothing fetches a path for it.
pub const PERLIN_2D: u32 = 0xc06f_eb5b;

/// The decal a face paint is drawn through, and where the set the creator picks is filed. Kept out
/// of [`ENGINE`] because which one binds is what a character was made with rather than a fixed path.
pub const FACE_PAINT: u32 = 0x0237_cb94;
pub const PAINTS: &str = "chara/common/texture/decal_face/_decal_";

/// The table the frame is graded through, which the grading pass addresses by the color it found.
/// Three of these ship and nothing states which one binds; this is the one that answers a grey with
/// the grey it was given and departs from the identity least of the three.
pub const GRADING: (u32, &str, u32) = (
    0xabc0_472a,
    "common/graphics/texture/-output_lut_d.tex",
    glow::LINEAR,
);

/// Where the passes past the lighting are linked. The lighting and the composite take nought to
/// five, the smoothing and the occlusion six to ten, and each of these is one program of its own:
/// two sharing a slot would relink both every frame.
const POST: usize = 11;
const SKY: usize = 12;
const EXPOSURE: usize = 13;
/// How wide the sun's own depth maps are drawn. One square a split, stacked into one image.
const SHADOW: i32 = program::SHADOW_MAP;

const FOG: usize = 19;
const CLOUD: usize = 20;
const SHADE: usize = 21;
const SCATTER: usize = 22;
const SUN: usize = 23;
const MOON: usize = 24;
/// The glare chain: the bright pass, a smoothing at each of its two levels, the blur, the merge and
/// the pass that adds the halo back. Both halves of the blur share a slot, since they are one
/// program run twice, and so do the two smoothings.
const GLARE: usize = 25;
const VIGNETTE: usize = 30;
/// The reflection chain: the pyramid it walks, the normal and reflectance it reads a pixel off, the
/// march, the two halves of the blur, the resolve and the copy back.
const REFLECT: usize = 31;

/// The night star field's tier 0. Held well clear of the slots above rather than threaded into that
/// range, since several of their own upper bounds are sized off the frame or the lamp count rather
/// than fixed.
const STAR: usize = 70;

/// The sheet drawn from the sun's own side, and the blur that map is left in.
const CLOUD_SHADOW_DRAW: usize = 72;
const CLOUD_SHADOW_BLUR: usize = 73;

/// The four members of water's own reflection chain drawn over a quad: the blur pair, the wide
/// blur's first half and the merge. The two drawn over the water itself are linked by the caller
/// that holds the geometry.
const WATER_REFLECT: usize = 80;

/// One slot per kind of lamp and falloff power, so a frame holding both a linear and a cubic light
/// of the same kind does not relink one of them on every lamp it draws.
const LAMP: usize = 40;

/// The same again for the run that lights a character's clipped fringe: it draws the same lamp
/// programs over its own buffers, and sharing the opaque run's slots would relink one of them at
/// every switch between the two.
const SHEER_LAMP: usize = LAMP + 4 * program::ATTENUATION.len();

/// One level of the pyramid the march walks, reduced off the level above it. A square of four texels
/// becomes one, the nearest of them in the first channel and the furthest in the second, which is
/// what the game's own compute shader leaves and the only reading the chain reads back.
const HIERARCHY: &str = "\
#version 300 es
precision highp float;
precision highp sampler2D;

in vec2 TEXCOORD;

uniform sampler2D u_source;
uniform vec2 u_texel;
uniform float u_level;
/// What the frame's own depth is brought into the game's ordering by, where the first level reads
/// that rather than a level of this.
uniform float u_range;
uniform bool u_first;

out vec2 fragColor;

void main() {
\tvec2 held = vec2(0.0, 1.0);
\tfor (int at = 0; at < 4; ++at) {
\t\tvec2 corner = vec2(float(at & 1), float(at >> 1)) - 0.5;
\t\tvec2 read = u_first
\t\t\t? vec2(1.0 - texture(u_source, TEXCOORD + corner * u_texel).x * u_range)
\t\t\t: textureLod(u_source, TEXCOORD + corner * u_texel, u_level).xy;
\t\theld = vec2(max(held.x, read.x), min(held.y, read.y));
\t}
\tfragColor = held;
}
";

/// How far down the sky is drawn on its own plane, which the fog reads what a distant pixel fades
/// toward out of. Nothing there is finer than the sky itself, and the game takes it down by the same
/// factor.
const OVERHEAD_SCALE: i32 = 4;

/// Channels of the G-buffer, which is what its pages add up to however many a context can write at
/// once, and the channel past the last of them: the frame the composite resolved.
pub const TARGETS: usize = 5;
pub const LIT: usize = TARGETS;

/// The channel past that: what the reflection chain resolved, before the copy laid it back over the
/// frame. Read as data rather than looked at, the way a raw channel is.
pub const REFLECTED: usize = LIT + 1;

/// The channel the fur pass softens. It squares what it reads and takes the root of what it writes,
/// and the surface color is the one channel a drawing package gamma-encodes; the package asks for it
/// by a name every package that shades a surface gives the channel after it.
const FUR_CHANNEL: usize = 2;

/// How many targets a semi-transparent surface writes. Its pass declares the same five as the
/// opaque one but the packing takes four: the tangent frame moves down into the fourth, the motion
/// the opaque third carries has nowhere to go, and the fifth is dropped on the floor.
pub const SHEER: usize = 4;

/// What that buffer starts at: the tangent frame's own nought, and a coverage of one step. A strand
/// end blends a fraction of itself over this, and over black it would decode a direction pointing
/// nowhere the light can reach.
const SHEER_CLEAR: [f32; 4] = [127.0 / 255.0, 127.0 / 255.0, 127.0 / 255.0, 1.0 / 255.0];

/// What stands where the packing has no target, which is the same white the game hands every slot
/// it leaves empty rather than the grey a material's own missing texture answers with.
const SHEER_ABSENT: [u8; 4] = [255, 255, 255, 255];

/// The one structured buffer that is not a joint palette.
pub const TYPES: &str = "g_ShaderTypeParameter";

/// What a texture the material binds nothing to answers with.
const STAND_IN: [u8; 4] = [128, 128, 128, 255];

/// The tables the engine works out a frame at a time and no file holds, each at the value that
/// leaves the term it drives where it was. `g_SamplerToneMapLut` divides the resolved color, so the
/// flat stand-in would double every pixel; a fog weight of nought keeps the color it mixes toward
/// out of the frame entirely.
///
/// The middle two are what the shadowed lighting reads beside its mask, and they are neutral at
/// opposite ends: a cloud shadow multiplies, so nothing overhead is white, while caustics are light
/// added by water, so none of it is black. The flat grey every unnamed sampler otherwise answers
/// with is half a cloud shadow over the whole zone and half a pool's worth of light on top of it.
/// The last is water's own local reflection: its alpha is what the shader lerps by, so zero leaves
/// the term it feeds exactly as it stood without one.
const NEUTRAL: [(u32, [u8; 4]); 5] = [
    (0x342f_2734, [255, 255, 255, 255]),
    (0x6e23_1669, [0, 0, 0, 0]),
    (CLOUD_SHADOW, [255, 255, 255, 255]),
    (0x0efb_24f7, [0, 0, 0, 0]),
    (REFLECTION_MAP, [0, 0, 0, 0]),
];

/// What a buffer nothing here fills answers with where a lighting pass wants a weight: nothing
/// shadowed in the red the lighting reads, nothing faded in the alpha the composite reads.
const UNOCCLUDED: [u8; 4] = [255, 255, 255, 0];

/// What the composite takes a reflection against before a place has said what its sky looks like.
/// Black, not grey: the composite squares this texel and divides by the alpha it was gathered at,
/// so any nonzero colour here survives an alpha of nought.
const UNREFLECTED: [u8; 4] = [0, 0, 0, 0];

/// Texels a face of the reflection cube takes. The sky it is built from is three harmonic rows,
/// which carry nothing finer than this.
const SKY_FACE: i32 = 16;

/// Which way a texel of a cube face looks, in the space the composite samples the cube in.
fn facing(face: usize, u: f32, v: f32) -> glam::Vec3 {
    match face {
        0 => glam::Vec3::new(1.0, -v, -u),
        1 => glam::Vec3::new(-1.0, -v, u),
        2 => glam::Vec3::new(u, 1.0, v),
        3 => glam::Vec3::new(u, -1.0, -v),
        4 => glam::Vec3::new(u, -v, 1.0),
        _ => glam::Vec3::new(-u, -v, -1.0),
    }
    .normalize()
}

/// One triangle covering clip space, which is the geometry a screen-wide pass draws: their vertex
/// shaders pass the position straight through, and one of them reads it back as the place on screen
/// the pixel came from.
const SCREEN: [f32; 12] = [
    -1.0, -1.0, 0.0, 1.0, //
    3.0, -1.0, 0.0, 1.0, //
    -1.0, 3.0, 0.0, 1.0,
];

/// The same triangle for the passes drawn with the game's own sampling vertex shader, which reads
/// where a vertex stands and what it samples out of one attribute rather than working the second out
/// of the first. Kept apart from the triangle above because half the shaders drawn over that one
/// pass the whole attribute through as a position.
const SAMPLED: [f32; 12] = [
    -1.0, -1.0, 0.0, 0.0, //
    3.0, -1.0, 2.0, 0.0, //
    -1.0, 3.0, 0.0, 2.0,
];

/// The corners and triangles of the volume a light covers, which its own vertex shader clamps to the
/// extents the zone clips it against and then projects. Not a screen-wide pass: a light only reaches
/// the pixels its volume covers.
const VOLUME: [f32; 32] = [
    -1.0, -1.0, -1.0, 1.0, //
    1.0, -1.0, -1.0, 1.0, //
    1.0, 1.0, -1.0, 1.0, //
    -1.0, 1.0, -1.0, 1.0, //
    -1.0, -1.0, 1.0, 1.0, //
    1.0, -1.0, 1.0, 1.0, //
    1.0, 1.0, 1.0, 1.0, //
    -1.0, 1.0, 1.0, 1.0,
];
/// The two meshes the engine builds for its clouds, which no file holds: a band of columns around a
/// unit cylinder, and a sheet over a square. Both are drawn as one serpentine strip, and both take a
/// position and a coordinate apiece.
const BAND: (usize, usize) = (25, 5);
const SHEET: usize = 17;

/// A vertex of either: a place and where to read the cloud out of.
const CLOUD_STRIDE: i32 = 24;

/// The band as its columns and rows state it: a unit circle a column at a time, hanging from nought
/// down to minus one, with the texture wrapped three times around it.
fn band() -> Vec<f32> {
    let (columns, rows) = BAND;
    let mut out = Vec::with_capacity(columns * rows * 6);
    for row in 0..rows {
        for column in 0..columns {
            let turn = std::f32::consts::TAU * column as f32 / (columns - 1) as f32;
            let height = -(row as f32) / (rows - 1) as f32;
            out.extend([turn.sin(), height, -turn.cos(), 1.0]);
            out.extend([
                3.0 * column as f32 / (columns - 1) as f32,
                row as f32 / (rows - 1) as f32,
            ]);
        }
    }
    out
}

/// The sheet the same way: a square whose points crowd toward the middle, bent down at the edges by
/// the square of how far out they stand, which is what carries it below the horizon at range.
fn sheet() -> Vec<f32> {
    let mut out = Vec::with_capacity(SHEET * SHEET * 6);
    let at = |step: usize| {
        let held = (step as f32 - (SHEET / 2) as f32) / (SHEET / 2) as f32;
        held.signum() * held * held
    };
    for row in 0..SHEET {
        for column in 0..SHEET {
            let (x, z) = (at(SHEET - 1 - column), at(row));
            out.extend([x, -(x * x + z * z), z, 1.0]);
            out.extend([-x, z]);
        }
    }
    out
}

/// One strip over a grid that many columns wide, turning at the ends rather than restarting: a band
/// runs one way, the next runs back, and the column they share stands on the axis so the triangle
/// that turns between them has no area.
fn strip(columns: usize, rows: usize) -> Vec<u16> {
    let mut out = Vec::new();
    let at = |row: usize, column: usize| (row * columns + column) as u16;
    for row in 0..rows - 1 {
        let forward = row % 2 == 0;
        for step in 0..columns {
            let column = match forward {
                true => step,
                false => columns - 1 - step,
            };
            // Every band but the first starts where the one before it stopped, and needs only the
            // one point that carries the strip down a row.
            if row > 0 && step == 0 {
                out.push(at(row + 1, column));
                continue;
            }
            out.extend([at(row, column), at(row + 1, column)]);
        }
    }
    out
}

/// The night star field's own dome: 6 face patches of a 17x17 grid, 1734 vertices over 9216
/// indices, byte-identical in count to the game's own draw. Position and the twinkle mask's baked
/// randomness (`TEXCOORD1.zw`) are read out of `STAR_MESH` below rather than generated: every
/// formula this project tried for the position (gnomonic, and bilinear slerp between each patch's
/// four measured corners) came within a few percent of the captured vertices and no closer, and a
/// mesh that merely looks spherical is not what was asked for. The grid's own texture coordinates
/// (`TEXCOORD0` and `TEXCOORD1.xy`) are a plain, verified function of a vertex's place in its patch
/// and are computed here instead of shipped.
const STAR_FACES: usize = 6;
const STAR_GRID: usize = 17;
/// Per vertex: `POSITION.xyz`, then `TEXCOORD1.zw`, each an `f16` the capture's own buffer held.
const STAR_MESH: &[u8] = include_bytes!("stars.bin");
/// Position (3) + `TEXCOORD0` (4) + `TEXCOORD1` (4).
const STAR_STRIDE: i32 = 44;

fn dome() -> (Vec<f32>, Vec<u16>) {
    let half = |at: usize| -> f32 {
        let bits = u16::from_le_bytes([STAR_MESH[at * 2], STAR_MESH[at * 2 + 1]]);
        f32::from(half::f16::from_bits(bits))
    };
    let mut vertices = Vec::with_capacity(STAR_FACES * STAR_GRID * STAR_GRID * 11);
    let mut indices = Vec::with_capacity(STAR_FACES * (STAR_GRID - 1) * (STAR_GRID - 1) * 6);
    for panel in 0..STAR_FACES {
        for slow in 0..STAR_GRID {
            for fast in 0..STAR_GRID {
                let vertex = panel * STAR_GRID * STAR_GRID + slow * STAR_GRID + fast;
                let base = vertex * 5;
                let (x, y, z) = (half(base), half(base + 1), half(base + 2));
                let (noise_z, noise_w) = (half(base + 3), half(base + 4));
                let u = slow as f32 * 0.5;
                let v = (STAR_GRID - 1 - fast) as f32 * 0.5;
                vertices.extend([
                    x,
                    y,
                    z,
                    u,
                    v,
                    u * 4.0,
                    v * 4.0,
                    u / 4.0,
                    v / 2.0 + 0.5,
                    noise_z,
                    noise_w,
                ]);
            }
        }
        let start = (panel * STAR_GRID * STAR_GRID) as u16;
        let at = |slow: usize, fast: usize| start + (slow * STAR_GRID + fast) as u16;
        for slow in 0..STAR_GRID - 1 {
            for fast in 0..STAR_GRID - 1 {
                let (a, b) = (at(slow, fast), at(slow, fast + 1));
                let (c, d) = (at(slow + 1, fast), at(slow + 1, fast + 1));
                indices.extend([a, b, c, b, d, c]);
            }
        }
    }
    (vertices, indices)
}

const VOLUME_FACES: [u16; 36] = [
    0, 2, 1, 0, 3, 2, // back
    4, 5, 6, 4, 6, 7, // front
    0, 1, 5, 0, 5, 4, // bottom
    3, 7, 6, 3, 6, 2, // top
    0, 4, 7, 0, 7, 3, // left
    1, 2, 6, 1, 6, 5, // right
];

/// What a pass of the graph covers: the whole frame, what one light reaches, or the whole frame
/// again where the fur pass reads the channel it answers into as the G-buffer left it.
#[derive(Clone, Copy)]
enum Over {
    Screen,
    Volume,
    Softening(glow::Texture),
    /// A member of the post chain reading what the one before it wrote rather than the frame the
    /// composite resolved.
    Reading(glow::Texture),
    /// A pass of the occlusion chain, which is drawn over a fraction of the frame rather than the
    /// whole of it.
    Fraction,
    /// A pass of the exposure chain, which halves what it reads until one texel holds the frame, so
    /// each stands over a size of its own and reads what the one before it left.
    Exposing((i32, i32), Reads),
    /// A pass drawn over the whole of a target smaller than the frame.
    Sized((i32, i32)),
    /// The fog, which reads the frame's own depth, the sky on its plane, and the table it takes the
    /// curve out of.
    Fogging(Fogged),
    /// One of the cloud meshes, over the strip it is drawn as and against the sheet it reads.
    Clouding(Clouded),
    /// The moon, over the disc it stands on and against the sky it blends into.
    Mooning(Mooned),
    /// The star dome, over its own mesh and against the sky it blends into and is occluded by.
    Starring(Starred),
    /// The skin blur, which walks the diffuse light around a pixel and writes the same buffer, so
    /// what it reads is a copy taken before it ran.
    Scattering(glow::Texture),
    /// A lighting pass reading what a semi-transparent surface filled rather than the opaque
    /// G-buffer, which is a packing and a depth of its own.
    Sheering(Sheered),
    /// A pass of the glare chain, over a target of its own and against what the pass before it left.
    Glaring((i32, i32), Glared),
    /// The blur among them and the cloud shadow's own, drawn over the triangle that carries a
    /// coordinate of its own: the vertex shader the game pairs each with builds its taps off that
    /// lane rather than off the position.
    Blurring((i32, i32), glow::Texture),
    /// A member of the reflection chain, over a target of its own and against what the members
    /// before it left. Which member is drawing is carried with them: two of the names its files use
    /// mean different things to different members.
    Reflecting((i32, i32), Member, Reflecting),
}

/// Which member of the reflection chain is drawing.
#[derive(Clone, Copy, PartialEq)]
enum Member {
    Normal,
    Mask,
    March,
    Blur,
    Distort,
    Copy,
}

/// What the reflection chain reads: the frame's depth as a pyramid, the frame itself, the normal and
/// reflectance the members before the march left, and the level of the blurred reflection at hand.
#[derive(Clone, Copy)]
struct Reflecting {
    depth: glow::Texture,
    frame: glow::Texture,
    normal: glow::Texture,
    mask: glow::Texture,
    blurred: glow::Texture,
}

/// What a pass of the glare chain reads, by the names its files give them: the frame or whatever the
/// pass before it left, and the spread glare the merge lays back over the frame.
#[derive(Clone, Copy)]
struct Glared {
    input: glow::Texture,
    merge: Option<glow::Texture>,
}

/// One cloud mesh as the card holds it, what it is drawn with, and how wide the target it is drawn
/// into stands: the sheet is drawn once over the frame and again into a map of its own.
#[derive(Clone, Copy)]
struct Clouded {
    layout: glow::VertexArray,
    indices: i32,
    texture: glow::Texture,
    size: (i32, i32),
}

/// Which of the passes a zone runs over and above the lighting actually drew this frame. A picture
/// alone cannot say whether a weather that states no clouds drew none or the draw quietly failed, so
/// the graph says which of them it ran.
#[derive(Clone, Copy, Default)]
pub struct Drawn {
    pub sky: bool,
    /// Whether the frame was reflected off itself this frame.
    pub reflection: bool,
    /// Whether water's own chain filled the map it reflects itself through.
    pub water: bool,
    pub sun: bool,
    pub moon: bool,
    pub stars: bool,
    pub fog: bool,
    pub vignette: bool,
    /// Whether the sun's own depth was drawn and resolved into a mask this frame.
    pub shadow: bool,
    /// Whether the chain that weights a pixel by how much sky reaches it ran this frame.
    pub occlusion: bool,
    /// The horizon band and the overhead sheet, in that order.
    pub clouds: [bool; 2],
    /// Whether the sheet was drawn again into the map the sun reads a cloud's shadow out of.
    pub cloud_shadow: bool,
}

/// What the fog pass reads, by the names its file gives them.
#[derive(Clone, Copy)]
struct Fogged {
    depth: glow::Texture,
    sky: glow::Texture,
    table: glow::Texture,
}

/// What the moon's disc is drawn over and against: the rectangle it stands on, in clip space, and
/// the sky it blends itself into.
#[derive(Clone, Copy)]
struct Mooned {
    disc: glam::Vec4,
    sky: glow::Texture,
}

/// The star dome's own mesh, and the four textures it reads: per-point colour, the Milky Way band,
/// a scrolling twinkle mask, and the sky it blends into and is occluded against.
#[derive(Clone, Copy)]
struct Starred {
    layout: glow::VertexArray,
    indices: i32,
    color: glow::Texture,
    band: glow::Texture,
    twinkle: glow::Texture,
    sky: glow::Texture,
}

/// What a pass of the exposure chain reads, by the names its file gives them: the frame, the measure
/// the halvings leave, and the exposure the adaptation carries. Each pass takes one or two of the
/// three, and one nothing fills is a pass reading a buffer nothing wrote.
#[derive(Clone, Copy, Default)]
struct Reads {
    input: Option<glow::Texture>,
    measure: Option<glow::Texture>,
    adapted: Option<glow::Texture>,
}

const PRESENT_VERTEX: &str = include_str!("present.vert");
const PRESENT_FRAGMENT: &str = include_str!("present.frag");

/// GL objects with nothing left to draw them, waiting for a context to delete them under. A viewer
/// is dropped between frames, where there is no context, so its objects outlive it by one callback.
static GRAVEYARD: std::sync::OnceLock<std::sync::Mutex<Vec<Dead>>> = std::sync::OnceLock::new();

/// The buffers a semi-transparent surface is drawn and lit through, kept apart from the opaque
/// G-buffer: the passes past the composite still read that one, and this carries a packing of its
/// own besides.
struct Sheer {
    frame: glow::Framebuffer,
    color: Vec<glow::Texture>,
    /// The frame's own depth copied in, so what stands in front of the surface hides it. It is
    /// never laid back over the frame's: a pixel a strand only half reached is the widget's still,
    /// and one counted as covered would be shown at that half's brightness rather than blended.
    depth: glow::Texture,
    position: (glow::Framebuffer, glow::Texture),
}

/// What one lighting pass reads out of those.
#[derive(Clone, Copy)]
struct Sheered {
    color: [glow::Texture; SHEER],
    depth: glow::Texture,
    position: glow::Texture,
}

pub enum Dead {
    Layout(glow::VertexArray),
    Buffer(glow::Buffer),
    Texture(glow::Texture),
    Program(glow::Program),
    Renderbuffer(glow::Renderbuffer),
    Frame(glow::Framebuffer),
    Sampler(glow::Sampler),
}

pub fn graveyard() -> &'static std::sync::Mutex<Vec<Dead>> {
    GRAVEYARD.get_or_init(Default::default)
}

/// What a linked program has already answered about its own names. Each of these is a question put
/// to the driver, and a zone puts the same ones to the same programs thousands of times a frame.
#[derive(Default)]
struct Learned {
    uniforms: HashMap<String, Option<glow::UniformLocation>>,
    samplers: HashMap<String, Sampled>,
    blocks: HashMap<String, Option<(u32, usize)>>,
    bindings: HashMap<u32, u32>,
}

/// Where a program takes the unit a sampler reads from, where it takes how many levels the texture
/// on that unit has, and which unit it was last told to read.
struct Sampled {
    unit: Option<glow::UniformLocation>,
    levels: Option<glow::UniformLocation>,
    at: Option<u32>,
}

type Lessons = std::sync::Mutex<HashMap<glow::Program, Learned>>;
static LEARNED: std::sync::OnceLock<Lessons> = std::sync::OnceLock::new();

fn learned() -> &'static Lessons {
    LEARNED.get_or_init(Default::default)
}

/// Where a program holds the uniform of this name.
pub fn uniform(
    gl: &glow::Context,
    program: glow::Program,
    name: &str,
) -> Option<glow::UniformLocation> {
    let mut learned = learned().lock().unwrap();
    let held = learned.entry(program).or_default();
    if let Some(found) = held.uniforms.get(name) {
        return found.as_ref().cloned();
    }
    let found = unsafe { gl.get_uniform_location(program, name) };
    held.uniforms.insert(name.to_owned(), found);
    held.uniforms[name].as_ref().cloned()
}

/// Points a program's sampler at a unit, where it is not pointed there already. The value is the
/// program's own state, so it stands until something points it elsewhere.
fn point_sampler(gl: &glow::Context, program: glow::Program, name: &str, unit: u32) {
    let mut learned = learned().lock().unwrap();
    let held = learned
        .entry(program)
        .or_default()
        .samplers
        .entry(name.to_owned())
        .or_insert_with(|| unsafe {
            Sampled {
                unit: gl.get_uniform_location(program, name),
                levels: gl.get_uniform_location(program, &format!("{name}_levels")),
                at: None,
            }
        });
    if held.at == Some(unit) {
        return;
    }
    held.at = Some(unit);
    unsafe {
        if let Some(location) = &held.unit {
            gl.uniform_1_i32(Some(location), unit as i32);
        }
        // Nothing in GLSL says how many levels a texture has, so a shader that asks is told.
        if let Some(location) = &held.levels {
            gl.uniform_1_i32(Some(location), 1);
        }
    }
}

/// Whether a program's block already reads the binding point it is about to be pointed at.
pub fn pointed(program: glow::Program, block: u32, at: u32) -> bool {
    learned()
        .lock()
        .unwrap()
        .entry(program)
        .or_default()
        .bindings
        .insert(block, at)
        == Some(at)
}

/// Which block a program holds for the buffer of this name, and how large it declares it. The
/// translator names a block after the buffer it carries, with a suffix that keeps the two apart.
pub fn block(gl: &glow::Context, program: glow::Program, name: &str) -> Option<(u32, usize)> {
    let mut learned = learned().lock().unwrap();
    let held = learned.entry(program).or_default();
    if let Some(found) = held.blocks.get(name) {
        return *found;
    }
    let found = unsafe {
        gl.get_uniform_block_index(program, &format!("{name}_b")).map(|index| {
            let size = gl.get_active_uniform_block_parameter_i32(
                program,
                index,
                glow::UNIFORM_BLOCK_DATA_SIZE,
            );
            (index, size as usize)
        })
    };
    held.blocks.insert(name.to_owned(), found);
    found
}

/// Deletes what an earlier viewer left behind. Called at the top of a draw, because that is the
/// only moment a context exists.
pub fn bury(gl: &glow::Context) {
    let dead: Vec<Dead> = graveyard().lock().unwrap().drain(..).collect();
    if !dead.is_empty() {
        // A handle is the driver's to hand out again, so what was learned about one being deleted
        // cannot be left standing for whatever takes its number next.
        let mut learned = learned().lock().unwrap();
        for dead in &dead {
            if let Dead::Program(program) = dead {
                learned.remove(program);
            }
        }
    }
    for dead in dead {
        unsafe {
            match dead {
                Dead::Layout(layout) => gl.delete_vertex_array(layout),
                Dead::Buffer(buffer) => gl.delete_buffer(buffer),
                Dead::Texture(texture) => gl.delete_texture(texture),
                Dead::Program(program) => gl.delete_program(program),
                Dead::Renderbuffer(held) => gl.delete_renderbuffer(held),
                Dead::Frame(held) => gl.delete_framebuffer(held),
                Dead::Sampler(held) => gl.delete_sampler(held),
            }
        }
    }
}

/// One lighting package translated at each falloff power it declares.
pub type Falloffs = [std::sync::Arc<program::Program>; 3];

/// The passes that light what the G-buffer holds and resolve it into a frame, translated out of the
/// packages that hold them rather than out of the one a material names.
///
/// Each reads what the one before it wrote: the view position comes off the depth buffer, the light
/// off the view position and the G-buffer, and the frame off both.
#[derive(Clone)]
pub struct Lighting {
    pub position: std::sync::Arc<program::Program>,
    pub directional: std::sync::Arc<program::Program>,
    /// One program per falloff power the package declares, in the order a light's own record names
    /// them.
    pub point: Falloffs,
    /// Absent until a zone's own spot package has arrived, and always where nothing places a spot.
    pub spot: Option<Falloffs>,
    /// The same for the two kinds a zone places besides those, each drawn over its own volume.
    pub line: Option<Falloffs>,
    pub plane: Option<Falloffs>,
    /// The same for fur, which only a surface whose own record states a length has any of.
    pub fur: Option<std::sync::Arc<program::Program>>,
    /// What softens the light inside skin, which only a surface the type table marks has any of.
    pub subsurface: Option<std::sync::Arc<program::Program>>,
    /// What turns the sun's own depth into how much of it reaches each pixel. Absent until the
    /// package arrives, and the frame lights unshadowed until it does.
    pub shadow: Option<std::sync::Arc<program::Program>>,
    pub composite: std::sync::Arc<program::Program>,
}

/// The pair that smooths the frame's edges, in the order they run.
pub struct Smoothing {
    pub luma: std::sync::Arc<program::Program>,
    pub fxaa: std::sync::Arc<program::Program>,
}

/// The chain that works out how bright the frame turned out and reads it back through a curve, in
/// the order it runs.
pub struct Exposure {
    pub initial: std::sync::Arc<program::Program>,
    pub iterative: std::sync::Arc<program::Program>,
    pub last: std::sync::Arc<program::Program>,
    pub adapt: std::sync::Arc<program::Program>,
    pub curve: std::sync::Arc<program::Program>,
    pub tone: std::sync::Arc<program::Program>,
}

/// The chain that spreads the bright end of the frame into a halo, in the order it runs.
pub struct Glare {
    pub bright: std::sync::Arc<program::Program>,
    pub gauss: std::sync::Arc<program::Program>,
    pub blur: std::sync::Arc<program::Program>,
    pub merge: std::sync::Arc<program::Program>,
    pub composite: std::sync::Arc<program::Program>,
}

/// The chain that reflects the frame off itself, in the order it runs. Four of its files carry names
/// the path list knows and five ship under none.
pub struct Reflection {
    pub normal: std::sync::Arc<program::Program>,
    pub mask: std::sync::Arc<program::Program>,
    pub march: std::sync::Arc<program::Program>,
    /// The two halves of the blur, across and then down.
    pub blur: [std::sync::Arc<program::Program>; 2],
    pub distort: std::sync::Arc<program::Program>,
    pub copy: std::sync::Arc<program::Program>,
}

/// The chain that fills what water reflects itself through, in the order it runs. Not the one above:
/// `river.shpk` and `water.shpk` reach no cube and read a screen-wide map this leaves, and the two
/// chains share only the depth pyramid and the frame.
pub struct WaterMirror {
    pub mask: std::sync::Arc<program::Program>,
    pub march: std::sync::Arc<program::Program>,
    /// The two halves of the first blur, across and then down.
    pub blur: [std::sync::Arc<program::Program>; 2],
    /// The first half of the second, which is what the merge is handed.
    pub wide: std::sync::Arc<program::Program>,
    pub merge: std::sync::Arc<program::Program>,
}

/// The chain that works out how much of the sky reaches each pixel, in the order it runs.
pub struct Occlusion {
    pub scale: std::sync::Arc<program::Program>,
    pub gather: std::sync::Arc<program::Program>,
    pub occlude: std::sync::Arc<program::Program>,
}

/// A layered texture as the card takes one: its slices one after the next, in RGBA bytes.
#[derive(Clone)]
pub struct Layered {
    pub size: (i32, i32),
    pub layers: i32,
    pub pixels: Vec<u8>,
    pub filter: u32,
    /// What a sampler has to be declared as to read it, which the file states for itself. A draw
    /// only validates where the texture bound to a unit is of the declaration's own kind.
    pub kind: program::Kind,
}

/// What the reflection chain draws into: the frame's depth as a pyramid of maxima, the normal and
/// reflectance a pixel's surface is read off, and the pair the blur carries the marched reflection
/// down. Two of that pair rather than one because each level of the blur reads a level of the other,
/// and a texture cannot be read and drawn into at once.
struct Mirrored {
    depth: (glow::Texture, Vec<glow::Framebuffer>),
    normal: (glow::Framebuffer, glow::Texture),
    mask: (glow::Framebuffer, glow::Texture),
    chain: [(glow::Texture, Vec<glow::Framebuffer>); 2],
    size: (i32, i32),
}

/// What water's own reflection chain draws into: the marched reflection, the buffer each blur half
/// hands the next, and what the merge leaves for water to read. One stencil across all three, since
/// every member past the mask is cut to the pixels the mask stamped.
struct Watered {
    sharp: (glow::Framebuffer, glow::Texture),
    scratch: (glow::Framebuffer, glow::Texture),
    merged: (glow::Framebuffer, glow::Texture),
    stencil: glow::Renderbuffer,
    size: (i32, i32),
}

/// What the two members of that chain drawn over the water itself need: where they write, the
/// pyramid the march walks and the frame it reads, and how much of the screen they cover.
#[derive(Clone, Copy)]
pub struct Watering {
    pub into: glow::Framebuffer,
    pub depth: glow::Texture,
    pub frame: glow::Texture,
    pub size: (i32, i32),
}

/// A linked pair of the game's own shaders, and the source it was built from so a change rebuilds
/// it rather than a stale program drawing on.
pub struct Linked {
    source: std::sync::Arc<program::Program>,
    pub program: glow::Program,
}

/// Everything the graph draws into, and the passes past the G-buffer.
#[derive(Default)]
pub struct Buffers {
    /// One framebuffer per page of the G-buffer's targets, all sharing the depth texture.
    frames: Vec<glow::Framebuffer>,
    color: Vec<glow::Texture>,
    /// A texture rather than a renderbuffer: the lighting passes read the depth back.
    depth: Option<glow::Texture>,
    /// The view position, the light this frame accumulates, and the frame the composite resolves.
    position: Option<(glow::Framebuffer, glow::Texture)>,
    light: Option<(glow::Framebuffer, [glow::Texture; 2])>,
    lit: Option<(glow::Framebuffer, glow::Texture)>,
    /// The same color, over a copy of the depth rather than the depth itself: a pass that reads the
    /// depth back as a texture cannot write through a framebuffer that same texture is attached to.
    bare: Option<glow::Framebuffer>,
    cutoff: Option<glow::Texture>,
    /// The depth the scene leaves as the sun sees it, which a shadow is tested against. One cascade:
    /// the resolve shader takes a single matrix and has no cascade index at all, so the count is a
    /// property of how many times the engine draws, not of what the shader can read.
    shadow: Option<(glow::Framebuffer, glow::Texture)>,
    /// What the resolve leaves for the lighting: one channel, one where the sun reaches.
    mask: Option<(glow::Framebuffer, glow::Texture)>,
    shadowing: bool,
    /// The diffuse light as it stood before the skin blur read it, since a pass cannot read the
    /// channel it writes, and the table of taps that blur walks.
    scattered: Option<glow::Texture>,
    kernel: Option<glow::Texture>,
    /// What the composite left, kept apart from the frame a semitransparent pass writes: that pass
    /// reads the one and writes the other, and a texture cannot be both at once.
    resolved: Option<glow::Texture>,
    /// What a semi-transparent surface fills, made the first time one is drawn: a frame with none in
    /// it carries neither this nor the second lighting run that reads it.
    sheer: Option<Sheer>,
    /// A framebuffer over the channel the fur pass softens, and that channel as the G-buffer left
    /// it: the pass walks a strand across its neighbours to answer for one pixel.
    fur: Option<(glow::Framebuffer, glow::Texture)>,
    /// The frame with each pixel's brightness in its alpha, which is what the smoothing pass reads
    /// its edges off. Filtered rather than point sampled, since that pass reads between texels along
    /// the edge it found.
    smoothed: Option<(glow::Framebuffer, glow::Texture)>,
    /// The occlusion chain's own buffers, at a fraction of the frame: the depth and normal it works
    /// from, the square of four of each its taps address, and how much sky reached the pixel. That
    /// last is filtered, since the lighting reads it back over the whole frame.
    scaled: Option<(glow::Framebuffer, glow::Texture)>,
    gathered: Option<(glow::Framebuffer, [glow::Texture; 2])>,
    occluded: Option<(glow::Framebuffer, glow::Texture)>,
    /// The glare chain's own levels, a quarter of the frame and an eighth. A pair at each because
    /// every pass of the chain reads what the one before it wrote, and filtered because the taps
    /// address between texels and the last of them is read back over the whole frame.
    glared: Option<[(glow::Framebuffer, glow::Texture); 2]>,
    reached: Option<[(glow::Framebuffer, glow::Texture); 2]>,
    /// What that chain spreads: the frame as the sky left it, at half its size. The game keeps its
    /// own copy there rather than at the end, and the passes between write an alpha of their own
    /// over the share of a pixel the composite marked as glare.
    sourced: Option<(glow::Framebuffer, glow::Texture)>,
    /// The reflection chain's own buffers, at a fraction of the frame.
    mirrored: Option<Mirrored>,
    /// Water's own chain's, at the same fraction.
    watered: Option<Watered>,
    /// The program that builds the pyramid the march walks. The game builds it in a compute shader,
    /// which a context this draws through has none of.
    hierarchy: Option<glow::Program>,
    /// Whether that chain ran this frame. Every pass reads the flat stand-in until it has, and again
    /// from the frame the viewer stops asking for it.
    occluding: bool,
    /// The exposure chain's buffers: the frame halved until one texel holds it, the pair the
    /// adaptation carries the answer between, and the curve the tone pass reads the frame through.
    /// The pair is two rather than one because the adaptation reads the frame before it and cannot
    /// sample the texture it is writing.
    luminance: Vec<((i32, i32), glow::Framebuffer, glow::Texture)>,
    adapted: Option<[(glow::Framebuffer, glow::Texture); 2]>,
    /// Which of the pair the last frame wrote, which is the one this frame reads.
    adaptation: usize,
    /// What that one holds, read back because two passes take the exposure as a constant rather than
    /// as the texture the rest of the chain carries it in. Read a frame after it was written, which
    /// is the lag the game runs with anyway.
    exposed: f32,
    measured: f32,
    curve: Option<(glow::Framebuffer, glow::Texture)>,
    /// The sky on a plane of its own, drawn over the whole of it rather than only where the frame
    /// left a hole: the fog fades a distant pixel toward the sky in that direction, which is not the
    /// direction anything happens to have left uncovered.
    overhead: Option<(glow::Framebuffer, glow::Texture)>,
    /// The table the fog reads its curve out of, and the numbers it was built from, so a weather or
    /// an hour that has not moved keeps it.
    haze: Option<(glow::Texture, program::Fog)>,
    size: (i32, i32),
    /// Whether `attach()` finished the graph it started: only true once every fallible allocation
    /// in it has succeeded, so a failure partway cannot make its reuse guard think a half-built
    /// graph is done.
    built: bool,
    /// What the context allows, which is what decides how much of the G-buffer one pass can write.
    attachments: usize,
    types: Option<glow::Texture>,
    /// One texel of the same value per target a sampler can be declared over. A draw is rejected
    /// outright where what is bound to a unit is not of the declaration's own kind, so a plane
    /// cannot stand in for the rest.
    blanks: BTreeMap<u32, glow::Texture>,
    /// The textures the shaders read off the game's own files, by resource id, each with the target
    /// its file was taken onto the card at.
    arrays: BTreeMap<u32, (u32, glow::Texture)>,
    /// The same, for the ones a material names rather than the engine, under the path it named.
    stacked: BTreeMap<String, (u32, glow::Texture)>,
    neutrals: BTreeMap<u32, glow::Texture>,
    /// A sampler that compares nothing, built once for the unit that gathers the sun's map raw.
    gather: Option<glow::Sampler>,
    unoccluded: Option<glow::Texture>,
    reflection: Option<glow::Texture>,
    /// The sky the reflection cube was built from, so a frame asking for the same one keeps it.
    sky: Option<([glam::Vec4; 3], f32)>,
    screen: Option<(glow::VertexArray, glow::Buffer)>,
    sampled: Option<(glow::VertexArray, glow::Buffer)>,
    volume: Option<(glow::VertexArray, glow::Buffer, glow::Buffer)>,
    /// The two meshes the clouds are drawn over, in the order the passes take them: the band first,
    /// then the sheet. Built once, since neither depends on the frame.
    strips: [Option<(glow::VertexArray, glow::Buffer, glow::Buffer, i32)>; 2],
    /// The texture each draws, under the file it came from so a weather that has not moved keeps it.
    sheets: [Option<(String, glow::Texture)>; 2],
    /// The map the sheet's own shadow is drawn into, and the one the blur leaves it in. Fixed at the
    /// size the game draws it, so neither follows the frame.
    clouded: Option<[(glow::Framebuffer, glow::Texture); 2]>,
    /// Whether that map holds this frame's clouds. The lighting reads the opaque white stand-in
    /// until it does, and again from the frame a weather stops stating a sheet.
    clouding: bool,
    /// The star field's own dome, built once since it depends on nothing a frame carries.
    dome: Option<(glow::VertexArray, glow::Buffer, glow::Buffer, i32)>,
    /// Its three textures: colour, band, twinkle. Fixed paths rather than weather-named, so unlike
    /// the cloud sheets there is nothing to check against a stale one.
    star_textures: [Option<glow::Texture>; 3],
    resolvers: BTreeMap<usize, Linked>,
    present: Option<glow::Program>,
    /// One uniform buffer per binding slot, and how many bytes the last fill of it came to.
    blocks: Vec<(glow::Buffer, Vec<u8>)>,
    /// Whether the graph has already brought the frame into the range a screen holds, which is what
    /// keeps the pass that puts it up from bending it a second time.
    toned: bool,
    /// Which of the optional passes ran over the frame, cleared as the lighting starts again.
    drawn: Drawn,
    /// Whether a sky stands behind the frame. The pass that puts one up drops the pixels nothing
    /// drew at, since those belong to the widget rather than to the frame; once a sky has filled
    /// them they are the frame's, and dropping them throws the sky away.
    covered: bool,
}

impl Buffers {
    pub fn size(&self) -> (i32, i32) {
        self.size
    }

    /// The exposure the chain settled on, which is a reading of the frame rather than of any file.
    /// Which of the optional passes ran over the last frame.
    pub fn drawn(&self) -> Drawn {
        self.drawn
    }

    pub fn measured(&self) -> f32 {
        self.measured
    }

    /// One until a chain has run and answered, which is the exposure a frame is drawn under when
    /// nothing is adapting it, and what every pass that divides by this has to see.
    pub fn exposed(&self) -> f32 {
        match self.exposed {
            held if held > 0.0 => held,
            _ => 1.0,
        }
    }

    /// How much of the G-buffer one pass can write. Four until a frame has asked the context, since
    /// that is what a context is promised.
    pub fn attachments(&self) -> usize {
        match self.attachments {
            0 => 4,
            held => held,
        }
    }

    /// What `limit()` answered, once it has: `None` until a frame has actually asked the context.
    pub fn attachments_learned(&self) -> Option<usize> {
        (self.attachments != 0).then_some(self.attachments)
    }

    /// Carries an answer `limit()` already got on another `Buffers` over to this one, so a model
    /// rebuilt fresh (an equipment change, a detail level) does not spend its first frame assuming
    /// four and its second correcting to what the context actually promises: the count is a
    /// property of the context, not of the model, and does not change between them.
    pub fn seed_attachments(&mut self, learned: usize) {
        if self.attachments == 0 {
            self.attachments = learned.clamp(1, TARGETS);
        }
    }

    pub fn pages(&self) -> usize {
        self.frames.len()
    }

    /// What a viewer puts on screen: one channel of the G-buffer, or the frame the composite
    /// resolved past the last of them.
    fn channel(&self, at: usize) -> Option<glow::Texture> {
        match at {
            REFLECTED => self.mirrored.as_ref().map(|held| held.chain[1].0),
            LIT => self.lit.map(|(_, texture)| texture),
            _ => self.color.get(at).copied(),
        }
    }

    /// Draws one of those over the widget and leaves egui's own framebuffer bound behind it. A raw
    /// channel is read as data rather than looked at, so only the frame the composite resolved is
    /// bent toward what a screen holds.
    ///
    /// A pass rather than a blit: the framebuffer a browser hands a callback is multisampled, and
    /// blitting into one of those is an error rather than a resolve. The depth buffer goes with it,
    /// since that is what says which pixels the frame covered and which are egui's to keep.
    pub fn show(
        &mut self,
        gl: &glow::Context,
        at: usize,
        into: Option<glow::Framebuffer>,
        viewport: (i32, i32, i32, i32),
    ) -> Result<(), String> {
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, into);
            // The default framebuffer draws to the back buffer and one of its own draws to its
            // first attachment; naming the wrong one is an error rather than a no-op.
            match into {
                Some(_) => gl.draw_buffers(&[glow::COLOR_ATTACHMENT0]),
                None => gl.draw_buffers(&[glow::BACK]),
            }
            gl.viewport(viewport.0, viewport.1, viewport.2, viewport.3);
            // Back on for the widget, which is drawn in the window's own coordinates, where that
            // clip rect means what it says.
            gl.enable(glow::SCISSOR_TEST);
            gl.color_mask(true, true, true, true);
            gl.disable(glow::CULL_FACE);
            // The frame answers with what it covers in its alpha and a color already multiplied by
            // it, so a pixel the frame owns replaces the widget and one a halo alone reached is
            // added to it.
            gl.enable(glow::BLEND);
            gl.blend_func(glow::ONE, glow::ONE_MINUS_SRC_ALPHA);
            // The frame carries its own depth up with it, so whatever is drawn over the widget
            // afterwards can test against what it covered. The pixels this pass drops would keep
            // what an earlier frame left there, hence the clear, and a fragment writes no depth at
            // all with the test off, hence the test.
            gl.enable(glow::DEPTH_TEST);
            gl.depth_func(glow::ALWAYS);
            gl.depth_mask(true);
            gl.clear_depth_f32(1.0);
            gl.clear(glow::DEPTH_BUFFER_BIT);
        }
        let texture = self
            .channel(at)
            .ok_or_else(|| format!("the frame has no buffer {at}"))?;
        let depth = self.depth.ok_or("no depth buffer")?;
        let program = self.presenter(gl)?;
        let layout = self.screen(gl)?;
        unsafe {
            gl.use_program(Some(program));
            sampler(gl, program, "u_frame", 0, texture);
            sampler(gl, program, "u_depth", 1, depth);
            if let Some(location) = gl.get_uniform_location(program, "u_tone") {
                gl.uniform_1_i32(Some(&location), i32::from(at == LIT && !self.toned));
            }
            if let Some(location) = gl.get_uniform_location(program, "u_cover") {
                gl.uniform_1_i32(Some(&location), i32::from(at == LIT && self.covered));
            }
            gl.bind_vertex_array(Some(layout));
            gl.draw_arrays(glow::TRIANGLES, 0, 3);
            gl.bind_vertex_array(None);
            gl.depth_mask(false);
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::BLEND);
        }
        Ok(())
    }

    /// One channel of the frame read off the card, each component between nought and one.
    ///
    /// The channel's own texture is attached to a framebuffer of this read's own, rather than the
    /// page it was written on being bound and the channel named as an attachment of that: how many
    /// channels a page holds is whatever the context turned out to allow rather than the four one
    /// is promised, so a channel named by its own number can fall past the last attachment its page
    /// has an image at. A read of an attachment with no image leaves the pixels it was given
    /// untouched and raises an error, which is a black frame to anyone who did not ask for one.
    pub fn read(&self, gl: &glow::Context, at: usize) -> Result<Vec<f32>, String> {
        let texture = self
            .channel(at)
            .ok_or_else(|| format!("the frame has no buffer {at}"))?;
        let count = (self.size.0 * self.size.1 * 4) as usize;
        unsafe {
            let held = gl.create_framebuffer()?;
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(held));
            gl.framebuffer_texture_2d(
                glow::READ_FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(texture),
                0,
            );
            gl.read_buffer(glow::COLOR_ATTACHMENT0);
            let status = gl.check_framebuffer_status(glow::READ_FRAMEBUFFER);
            let incomplete = (status != glow::FRAMEBUFFER_COMPLETE).then_some(status);
            while gl.get_error() != glow::NO_ERROR {}
            // The G-buffer holds bytes and the frame the composite resolved does not, and a read is
            // only asked for in a pair the format it is reading answers in.
            let values = match at >= TARGETS {
                true => {
                    let mut values = vec![0f32; count];
                    gl.read_pixels(
                        0,
                        0,
                        self.size.0,
                        self.size.1,
                        glow::RGBA,
                        glow::FLOAT,
                        glow::PixelPackData::Slice(Some(bytemuck::cast_slice_mut(&mut values))),
                    );
                    values
                }
                false => {
                    let mut values = vec![0u8; count];
                    gl.read_pixels(
                        0,
                        0,
                        self.size.0,
                        self.size.1,
                        glow::RGBA,
                        glow::UNSIGNED_BYTE,
                        glow::PixelPackData::Slice(Some(&mut values)),
                    );
                    values
                        .into_iter()
                        .map(|value| f32::from(value) / 255.0)
                        .collect()
                }
            };
            let why = gl.get_error();
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
            gl.delete_framebuffer(held);
            match incomplete.or((why != glow::NO_ERROR).then_some(why)) {
                Some(why) => Err(format!("buffer {at} would not read back: {why:#x}")),
                None => Ok(values),
            }
        }
    }

    /// Asks the context how much of the G-buffer one pass may write, which is only answerable where
    /// there is a context to ask.
    ///
    /// Asked by every frame rather than by the first one to draw the G-buffer: a material's passes
    /// are translated into pages of this size before the callback that draws them, so a frame that
    /// first asked as it drew would have translated its own passes against the four a context is
    /// promised and written that many targets into a buffer of however many it turned out to allow.
    pub fn limit(&mut self, gl: &glow::Context) {
        if self.attachments == 0 {
            let limit = unsafe { gl.get_parameter_i32(glow::MAX_DRAW_BUFFERS) };
            self.attachments = (limit.max(1) as usize).min(TARGETS);
        }
    }

    /// Every buffer of the graph, sized to what is being drawn into.
    ///
    /// The G-buffer is one framebuffer per page of its targets: a context is promised four draw
    /// buffers and a framebuffer no more color attachments than that, so five targets cannot all
    /// hang off one, and what a page cannot hold is written by a reading of its own.
    pub fn attach(&mut self, gl: &glow::Context, size: (i32, i32)) -> Result<(), String> {
        self.limit(gl);
        if self.built && self.size == size {
            return Ok(());
        }
        // Cleared up front, not just on failure: a `?` anywhere below must leave this false, since
        // a half-built graph is what the guard above exists to keep out.
        self.built = false;
        let mut dead = graveyard().lock().unwrap();
        dead.extend(self.color.drain(..).map(Dead::Texture));
        dead.extend(self.depth.take().map(Dead::Texture));
        dead.extend(self.resolved.take().map(Dead::Texture));
        dead.extend(self.frames.drain(..).map(Dead::Frame));
        dead.extend(self.bare.take().map(Dead::Frame));
        dead.extend(self.cutoff.take().map(Dead::Texture));
        if let Some((frame, held)) = self.shadow.take() {
            dead.push(Dead::Frame(frame));
            dead.push(Dead::Texture(held));
        }
        if let Some((frame, held)) = self.mask.take() {
            dead.push(Dead::Frame(frame));
            dead.push(Dead::Texture(held));
        }
        dead.extend(self.scattered.take().map(Dead::Texture));
        if let Some(held) = self.sheer.take() {
            dead.push(Dead::Frame(held.frame));
            dead.extend(held.color.into_iter().map(Dead::Texture));
            dead.push(Dead::Texture(held.depth));
            dead.push(Dead::Frame(held.position.0));
            dead.push(Dead::Texture(held.position.1));
        }
        for (frame, textures) in [
            self.position
                .take()
                .map(|(frame, held)| (frame, vec![held])),
            self.light
                .take()
                .map(|(frame, held)| (frame, held.to_vec())),
            self.lit.take().map(|(frame, held)| (frame, vec![held])),
            self.fur.take().map(|(frame, held)| (frame, vec![held])),
            self.smoothed
                .take()
                .map(|(frame, held)| (frame, vec![held])),
            self.scaled.take().map(|(frame, held)| (frame, vec![held])),
            self.gathered
                .take()
                .map(|(frame, held)| (frame, held.to_vec())),
            self.occluded
                .take()
                .map(|(frame, held)| (frame, vec![held])),
            self.curve.take().map(|(frame, held)| (frame, vec![held])),
            self.overhead
                .take()
                .map(|(frame, held)| (frame, vec![held])),
        ]
        .into_iter()
        .flatten()
        .chain(
            self.luminance
                .drain(..)
                .map(|(_, frame, held)| (frame, held))
                .chain(self.adapted.take().into_iter().flatten())
                .chain(self.glared.take().into_iter().flatten())
                .chain(self.reached.take().into_iter().flatten())
                .chain(self.sourced.take())
                .map(|(frame, held)| (frame, vec![held])),
        )
        {
            dead.push(Dead::Frame(frame));
            dead.extend(textures.into_iter().map(Dead::Texture));
        }
        if let Some(held) = self.watered.take() {
            dead.push(Dead::Renderbuffer(held.stencil));
            for (frame, texture) in [held.sharp, held.scratch, held.merged] {
                dead.push(Dead::Frame(frame));
                dead.push(Dead::Texture(texture));
            }
        }
        if let Some(held) = self.mirrored.take() {
            let (texture, frames) = held.depth;
            dead.push(Dead::Texture(texture));
            dead.extend(frames.into_iter().map(Dead::Frame));
            for (frame, texture) in [held.normal, held.mask] {
                dead.push(Dead::Frame(frame));
                dead.push(Dead::Texture(texture));
            }
            for (texture, frames) in held.chain {
                dead.push(Dead::Texture(texture));
                dead.extend(frames.into_iter().map(Dead::Frame));
            }
        }
        drop(dead);
        self.size = size;

        unsafe {
            let depth = gl.create_texture()?;
            gl.bind_texture(glow::TEXTURE_2D, Some(depth));
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::DEPTH_COMPONENT24 as i32,
                size.0,
                size.1,
                0,
                glow::DEPTH_COMPONENT,
                glow::UNSIGNED_INT,
                glow::PixelUnpackData::Slice(None),
            );
            point(gl);
            self.depth = Some(depth);

            for page in 0..TARGETS.div_ceil(self.attachments) {
                let held = gl.create_framebuffer()?;
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(held));
                let width = (TARGETS - page * self.attachments).min(self.attachments);
                for at in 0..width {
                    // The channels the game stores as bytes, which is what makes the shader type in
                    // the first target's alpha come back the whole number it went in as.
                    let texture = plane(gl, size, glow::RGBA8, glow::RGBA, glow::UNSIGNED_BYTE)?;
                    gl.framebuffer_texture_2d(
                        glow::FRAMEBUFFER,
                        glow::COLOR_ATTACHMENT0 + at as u32,
                        glow::TEXTURE_2D,
                        Some(texture),
                        0,
                    );
                    self.color.push(texture);
                }
                gl.framebuffer_texture_2d(
                    glow::FRAMEBUFFER,
                    glow::DEPTH_ATTACHMENT,
                    glow::TEXTURE_2D,
                    Some(depth),
                    0,
                );
                let status = gl.check_framebuffer_status(glow::FRAMEBUFFER);
                if status != glow::FRAMEBUFFER_COMPLETE {
                    return Err(format!("the G-buffer would not complete: {status:#x}"));
                }
                self.frames.push(held);
            }

            // The view position and the light the frame gathers hold values a byte cannot: one is a
            // place in front of the camera, the other a sum of every light that reached the pixel.
            let position = plane(gl, size, glow::RGBA16F, glow::RGBA, glow::FLOAT)?;
            self.position = Some((frame_of(gl, &[position], None)?, position));
            let light = [
                plane(gl, size, glow::RGBA16F, glow::RGBA, glow::FLOAT)?,
                plane(gl, size, glow::RGBA16F, glow::RGBA, glow::FLOAT)?,
            ];
            self.light = Some((frame_of(gl, &light, None)?, light));
            // The composite encodes what it writes but does not bring it under one, so this holds
            // what a byte cannot. It carries the G-buffer's own depth, since a material resolves
            // itself into it as geometry, and a framebuffer with no depth buffer passes every depth
            // test put to it.
            let lit = plane(gl, size, glow::RGBA16F, glow::RGBA, glow::FLOAT)?;
            self.lit = Some((frame_of(gl, &[lit], Some(depth))?, lit));
            // The same frame again, over a copy of that depth rather than the depth itself: the fog
            // samples the one the G-buffer left, and a texture cannot be read and drawn into at
            // once. The copy is what the pass tests against, so it is blitted in before the draw.
            let cutoff = gl.create_texture()?;
            gl.bind_texture(glow::TEXTURE_2D, Some(cutoff));
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::DEPTH_COMPONENT24 as i32,
                size.0,
                size.1,
                0,
                glow::DEPTH_COMPONENT,
                glow::UNSIGNED_INT,
                glow::PixelUnpackData::Slice(None),
            );
            point(gl);
            self.cutoff = Some(cutoff);
            self.bare = Some(frame_of(gl, &[lit], Some(cutoff))?);
            let shadow = gl.create_texture()?;
            gl.bind_texture(glow::TEXTURE_2D, Some(shadow));
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::DEPTH_COMPONENT16 as i32,
                SHADOW * program::ATLAS_COLUMNS as i32,
                SHADOW * program::ATLAS_ROWS as i32,
                0,
                glow::DEPTH_COMPONENT,
                glow::UNSIGNED_SHORT,
                glow::PixelUnpackData::Slice(None),
            );
            // Compared rather than sampled, and off the edge of the map a pixel is lit: the game's
            // own sampler is LESS_OR_EQUAL, LINEAR and clamped.
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_COMPARE_MODE,
                glow::COMPARE_REF_TO_TEXTURE as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_COMPARE_FUNC,
                glow::LEQUAL as i32,
            );
            self.shadow = Some((frame_of(gl, &[], Some(shadow))?, shadow));
            let mask = plane(gl, size, glow::R8, glow::RED, glow::UNSIGNED_BYTE)?;
            // Over the copy of the frame's depth rather than the depth itself, which the resolve
            // samples: each split is a quad the depth test clips to what stands nearer than it.
            self.mask = Some((frame_of(gl, &[mask], Some(cutoff))?, mask));
            self.resolved = Some(plane(gl, size, glow::RGBA16F, glow::RGBA, glow::FLOAT)?);
            // The channel the fur pass answers into is one of the G-buffer's own, reached through a
            // framebuffer of its own so a pass writing one channel does not have to name a page.
            let softened = plane(gl, size, glow::RGBA8, glow::RGBA, glow::UNSIGNED_BYTE)?;
            let held = frame_of(gl, &self.color[FUR_CHANNEL..=FUR_CHANNEL], None)?;
            self.fur = Some((held, softened));

            let smoothed = plane(gl, size, glow::RGBA16F, glow::RGBA, glow::FLOAT)?;
            smooth(gl, smoothed);
            self.smoothed = Some((frame_of(gl, &[smoothed], None)?, smoothed));

            // Whole floats: the pass that fills the first of these answers with the distance in
            // front of the camera, which it caps at a hundred thousand.
            let held = self.fraction();
            let scaled = plane(gl, held, glow::RG32F, glow::RG, glow::FLOAT)?;
            self.scaled = Some((frame_of(gl, &[scaled], None)?, scaled));
            let gathered = [
                plane(gl, held, glow::RGBA32F, glow::RGBA, glow::FLOAT)?,
                plane(gl, held, glow::RGBA32F, glow::RGBA, glow::FLOAT)?,
            ];
            self.gathered = Some((frame_of(gl, &gathered, None)?, gathered));
            let occluded = plane(gl, held, glow::RGBA8, glow::RGBA, glow::UNSIGNED_BYTE)?;
            smooth(gl, occluded);
            self.occluded = Some((frame_of(gl, &[occluded], None)?, occluded));

            let level = |size| {
                let pair = [
                    plane(gl, size, glow::RGBA16F, glow::RGBA, glow::FLOAT)?,
                    plane(gl, size, glow::RGBA16F, glow::RGBA, glow::FLOAT)?,
                ];
                for held in pair {
                    smooth(gl, held);
                }
                Ok::<_, String>([
                    (frame_of(gl, &pair[..1], None)?, pair[0]),
                    (frame_of(gl, &pair[1..], None)?, pair[1]),
                ])
            };
            self.glared = Some(level(self.spread())?);
            self.reached = Some(level(self.reach())?);
            let held = self.halved();
            let sourced = plane(gl, held, glow::RGBA16F, glow::RGBA, glow::FLOAT)?;
            smooth(gl, sourced);
            self.sourced = Some((frame_of(gl, &[sourced], None)?, sourced));

            // One channel at whole-float width: what the halving accumulates is the reciprocal of a
            // luminance, which runs far past what a byte or a half holds once a pixel is dark.
            let mut level = ((size.0 / 2).max(1), (size.1 / 2).max(1));
            loop {
                let held = plane(gl, level, glow::R32F, glow::RED, glow::FLOAT)?;
                self.luminance
                    .push((level, frame_of(gl, &[held], None)?, held));
                if level == (1, 1) {
                    break;
                }
                level = ((level.0 / 2).max(1), (level.1 / 2).max(1));
            }
            // Four channels rather than the one the pass writes: the exposure is read back off this
            // to fill the two buffers that hold it, and reading a plane back a channel at a time is
            // not something a context has to accept.
            let (first, second) = (
                plane(gl, (1, 1), glow::RGBA32F, glow::RGBA, glow::FLOAT)?,
                plane(gl, (1, 1), glow::RGBA32F, glow::RGBA, glow::FLOAT)?,
            );
            let pair = [
                (frame_of(gl, &[first], None)?, first),
                (frame_of(gl, &[second], None)?, second),
            ];
            // At an exposure of one rather than at nothing. The pass carries the frame before this
            // one toward what it measures, and nought reads as a scene too dark to have been lit,
            // which it would then climb out of over as many frames as the rate takes.
            for (frame, _) in pair {
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(frame));
                gl.draw_buffers(&[glow::COLOR_ATTACHMENT0]);
                gl.clear_color(1.0, 1.0, 1.0, 1.0);
                gl.clear(glow::COLOR_BUFFER_BIT);
            }
            self.adapted = Some(pair);
            self.exposed = 1.0;
            // Addressed by a coordinate the tone pass works out rather than by texel, so it reads
            // between them.
            let curve = plane(gl, (program::CURVE, 1), glow::RGBA16F, glow::RGBA, glow::FLOAT)?;
            smooth(gl, curve);
            self.curve = Some((frame_of(gl, &[curve], None)?, curve));

            // Read back over the whole frame from a plane a fraction of its size, so between texels
            // rather than at their centers.
            let held = self.overhead();
            let overhead = plane(gl, held, glow::RGBA16F, glow::RGBA, glow::FLOAT)?;
            smooth(gl, overhead);
            self.overhead = Some((frame_of(gl, &[overhead], None)?, overhead));
        }
        self.built = true;
        Ok(())
    }

    /// What the occlusion chain draws into, which is the frame taken down by the factor the pass
    /// that fills it is named for.
    fn fraction(&self) -> (i32, i32) {
        let held = program::OCCLUSION_SCALE;
        ((self.size.0 / held).max(1), (self.size.1 / held).max(1))
    }

    /// What the glare chain reads, and the two levels it draws into: where the frame's bright end is
    /// kept and smoothed, and where the blur that spreads it runs.
    fn halved(&self) -> (i32, i32) {
        let held = program::GLARE_SOURCE;
        ((self.size.0 / held).max(1), (self.size.1 / held).max(1))
    }

    fn spread(&self) -> (i32, i32) {
        let held = program::GLARE_SCALE;
        ((self.size.0 / held).max(1), (self.size.1 / held).max(1))
    }

    fn reach(&self) -> (i32, i32) {
        let held = program::GLARE_SPREAD;
        ((self.size.0 / held).max(1), (self.size.1 / held).max(1))
    }

    /// What the sky is drawn onto for the fog to read it back.
    fn overhead(&self) -> (i32, i32) {
        let held = OVERHEAD_SCALE;
        ((self.size.0 / held).max(1), (self.size.1 / held).max(1))
    }

    /// Takes a copy of the resolved frame, which is what the passes drawn over it read. Bound as
    /// the read framebuffer rather than blitted: the draw that follows writes the frame this was
    /// copied from, and a texture being written cannot also be sampled.
    pub fn keep(&self, gl: &glow::Context) -> Result<(), String> {
        let (frame, _) = self.lit.ok_or("no lit frame")?;
        let held = self.resolved.ok_or("no resolved frame")?;
        unsafe {
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(frame));
            gl.read_buffer(glow::COLOR_ATTACHMENT0);
            // Onto the first unit rather than whichever happens to be active, which would be a
            // sampler the pass before this one had just been given.
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(held));
            gl.copy_tex_sub_image_2d(glow::TEXTURE_2D, 0, 0, 0, 0, 0, self.size.0, self.size.1);
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
        }
        Ok(())
    }

    /// Keeps what the glare chain spreads, which the game does the moment the sky has drawn and
    /// nothing else has. Halved on the way, the way the game's own copy is.
    pub fn source(&self, gl: &glow::Context) -> Result<(), String> {
        let (frame, _) = self.lit.ok_or("no lit frame")?;
        let (into, _) = self.sourced.ok_or("no glare source")?;
        blit(gl, frame, into, self.size, self.halved());
        Ok(())
    }

    /// The resolved frame put through the game's own grading pass.
    ///
    /// That pass saturates what it reads before it reads its table, so the shoulder runs first: it
    /// stands where the game's exposure and tone curve would, and running it here rather than at the
    /// present is what keeps it from being applied twice. Each step reads the copy the one before it
    /// left, since a texture being written cannot also be sampled.
    pub fn post(
        &mut self,
        gl: &glow::Context,
        held: &std::sync::Arc<program::Program>,
        scene: &program::Scene,
    ) -> Result<(), String> {
        let (lit, _) = self.lit.ok_or("no lit frame")?;
        let table = held
            .textures
            .iter()
            .find(|texture| texture.name == program::POST_TABLE)
            .and_then(|texture| self.supplied(texture.kind, GRADING.0))
            .ok_or("the grading table has not arrived")?;
        let source = self.resolved.ok_or("no resolved frame")?;
        let layout = self.screen(gl)?;
        unsafe {
            gl.disable(glow::SCISSOR_TEST);
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::CULL_FACE);
            gl.disable(glow::BLEND);
            gl.depth_mask(false);
            gl.color_mask(true, true, true, true);
            gl.viewport(0, 0, self.size.0, self.size.1);
        }

        // Only where the game's own chain has not already done it. That chain is what this stands
        // in for, and a frame put through both is bent twice.
        if !self.toned {
            // The pass that puts a frame up drops the pixels the depth buffer says nothing drew at,
            // which belong to egui. Here every pixel is the frame's, and the depth buffer is
            // attached to what this draws into: sampling it would be reading a buffer it is writing.
            let covered = self.stand_in(gl)?;
            let shoulder = self.presenter(gl)?;
            self.keep(gl)?;
            unsafe {
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(lit));
                gl.draw_buffers(&[glow::COLOR_ATTACHMENT0]);
                gl.use_program(Some(shoulder));
                sampler(gl, shoulder, "u_frame", 0, source);
                sampler(gl, shoulder, "u_depth", 1, covered);
                if let Some(location) = gl.get_uniform_location(shoulder, "u_tone") {
                    gl.uniform_1_i32(Some(&location), 1);
                }
                gl.bind_vertex_array(Some(layout));
                gl.draw_arrays(glow::TRIANGLES, 0, 3);
            }
        }

        self.keep(gl)?;
        let program = link(gl, &mut self.resolvers, POST, held)?;
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(lit));
            gl.draw_buffers(&[glow::COLOR_ATTACHMENT0]);
            gl.use_program(Some(program));
        }
        self.bind(gl, program, held, scene, &[])?;
        // Named rather than looked up by resource id: what a name here does not reach would be
        // bound the flat stand-in, and a table of one grey renders something plausible and wrong.
        for (unit, texture) in held.textures.iter().enumerate() {
            let bound = match texture.name.as_str() {
                program::POST_INPUT => source,
                program::POST_TABLE => table,
                name => {
                    return Err(format!(
                        "the grading pass reads {name}, which nothing fills"
                    ));
                }
            };
            bind(
                gl,
                program,
                &texture.name,
                unit as u32,
                bound,
                target(texture.kind),
                0.0,
            );
        }
        unsafe {
            gl.bind_vertex_array(Some(layout));
            gl.draw_arrays(glow::TRIANGLES, 0, 3);
            gl.bind_vertex_array(None);
        }
        self.toned = true;
        Ok(())
    }

    /// How much of the sky reaches each pixel, worked out off the G-buffer before anything lights
    /// it. Three passes at a fraction of the frame: the depth linearized and the normal brought in
    /// front of the camera, a square of four of those packed into the channels of one texel, and the
    /// taps that read them. What it leaves is what every lighting pass and the composite read back.
    pub fn occlude(
        &mut self,
        gl: &glow::Context,
        held: &Occlusion,
        scene: &program::Scene,
    ) -> Result<(), String> {
        let (scaled, _) = self.scaled.ok_or("no scaled depth")?;
        let (gathered, _) = self.gathered.ok_or("no gathered depth")?;
        let (occluded, _) = self.occluded.ok_or("no occlusion")?;
        let size = self.fraction();
        unsafe {
            gl.disable(glow::SCISSOR_TEST);
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::CULL_FACE);
            gl.disable(glow::BLEND);
            gl.depth_mask(false);
            gl.viewport(0, 0, size.0, size.1);
            // The last two passes discard where the pixel stands past the distance the settings
            // state, and what a discard leaves behind is whatever the buffer last held. A gathered
            // depth of nought stands further from its neighbour than any setting accepts, so a tap
            // that reaches one of these is thrown out rather than read as a valley.
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(gathered));
            gl.draw_buffers(&[glow::COLOR_ATTACHMENT0, glow::COLOR_ATTACHMENT1]);
            gl.clear_color(0.0, 0.0, 0.0, 0.0);
            gl.clear(glow::COLOR_BUFFER_BIT);
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(occluded));
            gl.draw_buffers(&[glow::COLOR_ATTACHMENT0]);
            gl.clear_color(1.0, 1.0, 1.0, 0.0);
            gl.clear(glow::COLOR_BUFFER_BIT);
        }
        self.pass(gl, 8, &held.scale, scaled, scene, Over::Fraction)?;
        self.pass(gl, 9, &held.gather, gathered, scene, Over::Fraction)?;
        self.pass(gl, 10, &held.occlude, occluded, scene, Over::Fraction)?;
        self.occluding = true;
        Ok(())
    }

    /// Leaves every pass reading the flat stand-in again, which is what a frame nobody asked for
    /// occlusion on is lit against.
    pub fn unocclude(&mut self) {
        self.occluding = false;
    }

    /// How much of the sun reaches each pixel, worked out from the depth it left of its own view.
    /// One channel, one where nothing stands between the pixel and the light, which is what a
    /// lighting pass multiplies its own term by.
    pub fn shade(
        &mut self,
        gl: &glow::Context,
        held: &std::sync::Arc<program::Program>,
        scene: &program::Scene,
    ) -> Result<(), String> {
        let (into, _) = self.mask.ok_or("no shadow mask")?;
        let (from, _) = self.lit.ok_or("no lit frame")?;
        unsafe {
            // Ahead of the copy: a blit is one of the few things the scissor still reaches.
            gl.disable(glow::SCISSOR_TEST);
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(from));
            gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(into));
            let (width, height) = self.size;
            gl.blit_framebuffer(
                0,
                0,
                width,
                height,
                0,
                0,
                width,
                height,
                glow::DEPTH_BUFFER_BIT,
                glow::NEAREST,
            );
            gl.disable(glow::CULL_FACE);
            gl.disable(glow::BLEND);
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(into));
            gl.draw_buffers(&[glow::COLOR_ATTACHMENT0]);
            // Lit, so a pixel the pass leaves alone is one the sun reaches rather than one in the
            // dark: an unwritten mask has to read as no shadow at all.
            gl.clear_color(1.0, 1.0, 1.0, 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT);
            // Each split stands its quad where it stops and keeps what is nearer, so the nearest
            // one covering a pixel is the one drawn over it last.
            gl.enable(glow::DEPTH_TEST);
            gl.depth_func(glow::GREATER);
            gl.depth_mask(false);
        }
        let mut drawn = Ok(());
        for split in (0..program::SPLITS).rev() {
            let scene = program::Scene {
                split,
                ..scene.clone()
            };
            drawn = self.pass(gl, SHADE, held, into, &scene, Over::Screen);
            if drawn.is_err() {
                break;
            }
        }
        unsafe {
            gl.disable(glow::DEPTH_TEST);
            gl.depth_func(glow::LEQUAL);
            gl.depth_mask(true);
        }
        self.shadowing = drawn.is_ok();
        self.drawn.shadow = self.shadowing;
        drawn
    }

    /// Leaves the lighting reading a lit mask again.
    pub fn unshade(&mut self) {
        self.shadowing = false;
    }

    /// The sky, drawn over whatever the frame did not cover, and again onto a plane of its own.
    ///
    /// Into the frame it writes no depth and tests against what the geometry left, so it lands only
    /// where nothing drew. Run before the exposure rather than after: the measure reads the whole
    /// frame, and a frame with a black hole where the sky belongs reads as far darker than it is.
    ///
    /// The plane takes the whole sky rather than the holes in the frame, because the fog fades a
    /// distant pixel toward the sky standing behind *it*, which is a direction something already
    /// drew over.
    pub fn sky(
        &mut self,
        gl: &glow::Context,
        held: &std::sync::Arc<program::Program>,
        scene: &program::Scene,
    ) -> Result<(), String> {
        let (lit, _) = self.lit.ok_or("no lit frame")?;
        unsafe {
            gl.disable(glow::SCISSOR_TEST);
            gl.disable(glow::CULL_FACE);
            gl.disable(glow::BLEND);
            gl.disable(glow::DEPTH_TEST);
            gl.depth_mask(false);
        }
        if let Some((frame, _)) = self.overhead {
            let size = self.overhead();
            self.pass(gl, SKY, held, frame, scene, Over::Sized(size))?;
        }
        unsafe {
            // The quad sits at the far plane and the buffer was cleared to it, so this passes where
            // nothing drew and nowhere else. It writes no depth, the way the game's own does.
            gl.enable(glow::DEPTH_TEST);
            gl.depth_func(glow::LEQUAL);
        }
        let held = self.pass(gl, SKY, held, lit, scene, Over::Screen);
        unsafe {
            gl.disable(glow::DEPTH_TEST);
        }
        self.covered = true;
        self.drawn.sky = held.is_ok();
        held
    }

    /// The sun's glow, over the sky and nowhere the frame already covered. The game keeps its own
    /// occlusion in a buffer this graph does not build, so the depth test stands in for it: what the
    /// pass answers is added where nothing drew, and hidden behind anything that did.
    pub fn sun(
        &mut self,
        gl: &glow::Context,
        held: &std::sync::Arc<program::Program>,
        scene: &program::Scene,
    ) -> Result<(), String> {
        let (lit, _) = self.lit.ok_or("no lit frame")?;
        if program::sun_at(scene).is_none() {
            return Ok(());
        }
        unsafe {
            gl.disable(glow::SCISSOR_TEST);
            gl.disable(glow::CULL_FACE);
            gl.enable(glow::BLEND);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE);
            gl.enable(glow::DEPTH_TEST);
            gl.depth_func(glow::LEQUAL);
            gl.depth_mask(false);
        }
        let held = self.pass(gl, SUN, held, lit, scene, Over::Screen);
        unsafe {
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::BLEND);
        }
        self.drawn.sun = held.is_ok();
        held
    }

    /// The moon's own disc, over the sky it blends itself into. Nothing where the weather lets none
    /// of it through, or where it stands behind the camera.
    pub fn moon(
        &mut self,
        gl: &glow::Context,
        held: &std::sync::Arc<program::Program>,
        scene: &program::Scene,
    ) -> Result<(), String> {
        let (lit, _) = self.lit.ok_or("no lit frame")?;
        let Some(disc) = program::moon_disc(scene).filter(|_| scene.sky.moonlight.w > 0.0) else {
            return Ok(());
        };
        let Some((_, sky)) = self.overhead else {
            return Ok(());
        };
        unsafe {
            gl.disable(glow::SCISSOR_TEST);
            gl.disable(glow::CULL_FACE);
            gl.disable(glow::BLEND);
            gl.enable(glow::DEPTH_TEST);
            gl.depth_func(glow::LEQUAL);
            gl.depth_mask(false);
        }
        // The quad the vertex shader stands over, as clip space reads it.
        let disc = glam::Vec4::new(
            disc.x * 2.0 - 1.0,
            disc.y * 2.0 - 1.0,
            disc.z * 2.0,
            disc.w * 2.0,
        );
        let held = self.pass(gl, MOON, held, lit, scene, Over::Mooning(Mooned { disc, sky }));
        unsafe {
            gl.disable(glow::DEPTH_TEST);
        }
        self.drawn.moon = held.is_ok();
        held
    }

    /// One of the two cloud meshes, drawn over the sky and behind everything the frame covered.
    ///
    /// Premultiplied: the shader answers with its color already taken up by how much of the pixel it
    /// covers, so what lands is added rather than mixed.
    ///
    /// Held at the far plane rather than at the distance the mesh really stands: the game draws its
    /// clouds into a buffer of their own and composites that where nothing drew, and a cloud is a
    /// backdrop rather than something at four thousand units that a mountain at six could cover.
    /// Squashing the range says the same thing in one line.
    pub fn cloud(
        &mut self,
        gl: &glow::Context,
        at: usize,
        held: &std::sync::Arc<program::Program>,
        scene: &program::Scene,
    ) -> Result<(), String> {
        let (lit, _) = self.lit.ok_or("no lit frame")?;
        let Some(texture) = self
            .sheets
            .get(at)
            .and_then(Option::as_ref)
            .map(|(_, held)| *held)
        else {
            return Ok(());
        };
        let (layout, _, _, indices) = self.strip(gl, at)?;
        unsafe {
            gl.disable(glow::SCISSOR_TEST);
            gl.disable(glow::CULL_FACE);
            gl.enable(glow::DEPTH_TEST);
            gl.depth_func(glow::LEQUAL);
            gl.depth_range_f32(1.0, 1.0);
            gl.depth_mask(false);
            gl.enable(glow::BLEND);
            gl.blend_equation(glow::FUNC_ADD);
            gl.blend_func(glow::ONE, glow::ONE_MINUS_SRC_ALPHA);
        }
        let drawn = self.pass(
            gl,
            CLOUD + at,
            held,
            lit,
            scene,
            Over::Clouding(Clouded {
                layout,
                indices,
                texture,
                size: self.size,
            }),
        );
        unsafe {
            gl.depth_range_f32(0.0, 1.0);
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::BLEND);
        }
        if let Some(held) = self.drawn.clouds.get_mut(at) {
            *held = drawn.is_ok();
        }
        drawn
    }

    /// The texture one of the cloud draws reads, taken up under the file it came from. Wrapped
    /// rather than clamped: the band's coordinate runs three times round its own circle, and the
    /// sheet's ten times across itself.
    pub fn overcast(
        &mut self,
        gl: &glow::Context,
        at: usize,
        path: &str,
        held: &Layered,
    ) -> Result<(), String> {
        if self.sheets[at].as_ref().is_some_and(|(from, _)| from == path) {
            return Ok(());
        }
        unsafe {
            let texture = gl.create_texture()?;
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA8 as i32,
                held.size.0,
                held.size.1,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(&held.pixels)),
            );
            for (name, value) in [
                (glow::TEXTURE_MIN_FILTER, glow::LINEAR),
                (glow::TEXTURE_MAG_FILTER, glow::LINEAR),
                (glow::TEXTURE_WRAP_S, glow::REPEAT),
                (glow::TEXTURE_WRAP_T, glow::REPEAT),
            ] {
                gl.tex_parameter_i32(glow::TEXTURE_2D, name, value as i32);
            }
            if let Some((_, stale)) = self.sheets[at].replace((path.to_owned(), texture)) {
                graveyard().lock().unwrap().push(Dead::Texture(stale));
            }
        }
        Ok(())
    }

    /// The sheet again, drawn from where the sun stands into a map the lighting reads a cloud's
    /// shadow out of, and blurred there. Before the lighting, which takes it as a weight on the sun.
    ///
    /// The map is cleared to nought in the two channels the blur reads, so wherever the sheet does
    /// not cover it the blur leaves opaque light. A weather that draws no sheet leaves the lighting
    /// on the white stand-in instead, which comes to the same thing without the two passes.
    pub fn cloud_shadow(
        &mut self,
        gl: &glow::Context,
        held: &std::sync::Arc<program::Program>,
        blur: &std::sync::Arc<program::Program>,
        scene: &program::Scene,
    ) -> Result<(), String> {
        let Some(texture) = self.sheets[1].as_ref().map(|(_, held)| *held) else {
            self.clouding = false;
            return Ok(());
        };
        let [(source, drawn), (into, _)] = self.clouded(gl)?;
        let (layout, _, _, indices) = self.strip(gl, 1)?;
        let size = (program::CLOUD_SHADOW_MAP, program::CLOUD_SHADOW_MAP);
        unsafe {
            gl.disable(glow::SCISSOR_TEST);
            gl.disable(glow::CULL_FACE);
            gl.disable(glow::BLEND);
            gl.disable(glow::DEPTH_TEST);
            gl.depth_mask(false);
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(source));
            gl.viewport(0, 0, size.0, size.1);
            gl.clear_color(1.0, 1.0, 0.0, 0.0);
            gl.clear(glow::COLOR_BUFFER_BIT);
        }
        self.pass(
            gl,
            CLOUD_SHADOW_DRAW,
            held,
            source,
            scene,
            Over::Clouding(Clouded {
                layout,
                indices,
                texture,
                size,
            }),
        )?;
        self.pass(
            gl,
            CLOUD_SHADOW_BLUR,
            blur,
            into,
            scene,
            Over::Blurring(size, drawn),
        )?;
        self.clouding = true;
        self.drawn.cloud_shadow = true;
        Ok(())
    }

    /// Leaves the lighting reading opaque light again.
    pub fn unclouded(&mut self) {
        self.clouding = false;
    }

    /// The pair that map is drawn and blurred into, built the first time it is. Wrapped rather than
    /// clamped: the lighting reads it at a coordinate that runs past its own edge.
    fn clouded(
        &mut self,
        gl: &glow::Context,
    ) -> Result<[(glow::Framebuffer, glow::Texture); 2], String> {
        if let Some(held) = self.clouded {
            return Ok(held);
        }
        let size = (program::CLOUD_SHADOW_MAP, program::CLOUD_SHADOW_MAP);
        let one = || -> Result<(glow::Framebuffer, glow::Texture), String> {
            let map = plane(gl, size, glow::RGBA8, glow::RGBA, glow::UNSIGNED_BYTE)?;
            unsafe {
                gl.bind_texture(glow::TEXTURE_2D, Some(map));
                for (name, value) in [
                    (glow::TEXTURE_MIN_FILTER, glow::LINEAR),
                    (glow::TEXTURE_MAG_FILTER, glow::LINEAR),
                    (glow::TEXTURE_WRAP_S, glow::REPEAT),
                    (glow::TEXTURE_WRAP_T, glow::REPEAT),
                ] {
                    gl.tex_parameter_i32(glow::TEXTURE_2D, name, value as i32);
                }
            }
            Ok((frame_of(gl, &[map], None)?, map))
        };
        let held = [one()?, one()?];
        self.clouded = Some(held);
        Ok(held)
    }

    /// One of the cloud meshes on the card, built the first time it is drawn.
    fn strip(
        &mut self,
        gl: &glow::Context,
        at: usize,
    ) -> Result<(glow::VertexArray, glow::Buffer, glow::Buffer, i32), String> {
        if let Some(held) = self.strips[at] {
            return Ok(held);
        }
        let (vertices, indices) = match at {
            0 => (band(), strip(BAND.0, BAND.1)),
            _ => (sheet(), strip(SHEET, SHEET)),
        };
        let held = upload_strip(gl, &vertices, &indices)?;
        self.strips[at] = Some(held);
        Ok(held)
    }

    /// The night star field's tier 0, over the sky and against it: the shader discards any pixel it
    /// would not carry a full stop brighter than the sky already reads there, so this needs the sky
    /// plane standing before it draws. Nothing where a texture has not arrived, or the weather's own
    /// starfield set states no alpha at all - the gate `moon` also opens on.
    ///
    /// No blend: the captured pipeline draws this with `blendEnable` off, so a surviving fragment
    /// replaces the sky pixel outright rather than mixing over it, and the discard is what keeps
    /// that from carving holes in it.
    pub fn stars(
        &mut self,
        gl: &glow::Context,
        held: &std::sync::Arc<program::Program>,
        scene: &program::Scene,
    ) -> Result<(), String> {
        let (lit, _) = self.lit.ok_or("no lit frame")?;
        let Some((_, sky)) = self.overhead else {
            return Ok(());
        };
        let [color, band, twinkle] = self.star_textures;
        let (Some(color), Some(band), Some(twinkle)) = (color, band, twinkle) else {
            return Ok(());
        };
        if scene.star.alpha <= 0.0 {
            return Ok(());
        }
        let (layout, _, _, indices) = self.star_mesh(gl)?;
        unsafe {
            gl.disable(glow::SCISSOR_TEST);
            gl.disable(glow::CULL_FACE);
            gl.disable(glow::BLEND);
            gl.enable(glow::DEPTH_TEST);
            gl.depth_func(glow::LEQUAL);
            gl.depth_range_f32(1.0, 1.0);
            gl.depth_mask(false);
        }
        let drawn = self.pass(
            gl,
            STAR,
            held,
            lit,
            scene,
            Over::Starring(Starred { layout, indices, color, band, twinkle, sky }),
        );
        unsafe {
            gl.depth_range_f32(0.0, 1.0);
            gl.disable(glow::DEPTH_TEST);
        }
        self.drawn.stars = drawn.is_ok();
        drawn
    }

    /// One of the star field's three textures, uploaded the first time it arrives. Wrapped rather
    /// than clamped: every one of them is sampled well past a single tile.
    pub fn starlit(&mut self, gl: &glow::Context, at: usize, held: &Layered) -> Result<(), String> {
        if self.star_textures[at].is_some() {
            return Ok(());
        }
        unsafe {
            let texture = gl.create_texture()?;
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA8 as i32,
                held.size.0,
                held.size.1,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(&held.pixels)),
            );
            for (name, value) in [
                (glow::TEXTURE_MIN_FILTER, glow::LINEAR),
                (glow::TEXTURE_MAG_FILTER, glow::LINEAR),
                (glow::TEXTURE_WRAP_S, glow::REPEAT),
                (glow::TEXTURE_WRAP_T, glow::REPEAT),
            ] {
                gl.tex_parameter_i32(glow::TEXTURE_2D, name, value as i32);
            }
            self.star_textures[at] = Some(texture);
        }
        Ok(())
    }

    /// The star dome on the card, built the first time it is drawn.
    fn star_mesh(
        &mut self,
        gl: &glow::Context,
    ) -> Result<(glow::VertexArray, glow::Buffer, glow::Buffer, i32), String> {
        if let Some(held) = self.dome {
            return Ok(held);
        }
        let (vertices, indices) = dome();
        let held = upload_dome(gl, &vertices, &indices)?;
        self.dome = Some(held);
        Ok(held)
    }

    /// The frame with the weather's own fog over it: a pixel's distance addresses a table, whose two
    /// channels say how opaque the fog is there and how far the color it mixes toward has gone from
    /// the fog's own to the sky's. Near geometry is left alone, the middle distance is dragged toward
    /// the fog color, and the far distance toward the sky itself.
    ///
    /// Before the exposure, like the sky: this is one of the things the frame's brightness is
    /// measured over rather than something done to a frame already read back.
    pub fn fog(
        &mut self,
        gl: &glow::Context,
        held: &std::sync::Arc<program::Program>,
        scene: &program::Scene,
    ) -> Result<(), String> {
        let into = self.bare.ok_or("no lit frame")?;
        let (from, _) = self.lit.ok_or("no lit frame")?;
        let depth = self.depth.ok_or("no depth")?;
        let (_, sky) = self.overhead.ok_or("no sky plane")?;
        let table = self.table(gl, scene.fog)?;
        unsafe {
            // Ahead of the copy: a blit is one of the few things the scissor still reaches.
            gl.disable(glow::SCISSOR_TEST);
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(from));
            gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(into));
            let (width, height) = self.size;
            gl.blit_framebuffer(
                0,
                0,
                width,
                height,
                0,
                0,
                width,
                height,
                glow::DEPTH_BUFFER_BIT,
                glow::NEAREST,
            );
            gl.disable(glow::CULL_FACE);
            gl.depth_mask(false);
            // The shader was built against a reversed depth and drops the pixels its own far plane
            // holds, which here are the ones nothing was drawn into: the sky fogs itself. Keeping
            // the quad at the far plane and letting only nearer pixels through says the same thing
            // in state, and leaves the shader as the file wrote it.
            gl.enable(glow::DEPTH_TEST);
            gl.depth_func(glow::GREATER);
            gl.depth_range_f32(1.0, 1.0);
            // The pass answers with what the pixel fades toward and how far it has gone, which is a
            // mix rather than something added to the frame.
            gl.enable(glow::BLEND);
            gl.blend_equation(glow::FUNC_ADD);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);
        }
        let drawn = self.pass(
            gl,
            FOG,
            held,
            into,
            scene,
            Over::Fogging(Fogged { depth, sky, table }),
        );
        unsafe {
            gl.disable(glow::BLEND);
            gl.disable(glow::DEPTH_TEST);
            gl.depth_func(glow::LESS);
            gl.depth_range_f32(0.0, 1.0);
        }
        self.drawn.fog = drawn.is_ok();
        drawn
    }

    /// The taps the skin blur walks, uploaded once. Point sampled for the reason [`ENGINE`] gives:
    /// a column and a row are addressed as `n / size`, which is the seam between two texels.
    fn subsurface(&mut self, gl: &glow::Context) -> Result<glow::Texture, String> {
        if let Some(held) = self.kernel {
            return Ok(held);
        }
        let held = unsafe {
            let held = gl.create_texture()?;
            gl.bind_texture(glow::TEXTURE_2D, Some(held));
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA16F as i32,
                SUBSURFACE_TAPS,
                SUBSURFACE_ROWS,
                0,
                glow::RGBA,
                glow::HALF_FLOAT,
                glow::PixelUnpackData::Slice(Some(SUBSURFACE)),
            );
            point(gl);
            held
        };
        self.kernel = Some(held);
        Ok(held)
    }

    /// The sun's own depth of the scene, and the frame it is drawn into.
    pub fn shadow(&self) -> Option<(glow::Framebuffer, glow::Texture)> {
        self.shadow
    }

    /// How wide that map is, which the resolve needs to address a texel of it.
    pub fn shadow_size(&self) -> i32 {
        SHADOW
    }

    /// The table the fog reads its curve out of, built again where the weather or the hour has moved
    /// it. Filtered: the pass addresses it by a distance rather than by texel.
    fn table(
        &mut self,
        gl: &glow::Context,
        held: program::Fog,
    ) -> Result<glow::Texture, String> {
        if let Some((texture, built)) = self.haze
            && built == held
        {
            return Ok(texture);
        }
        let texture = match self.haze {
            Some((texture, _)) => texture,
            None => unsafe { gl.create_texture()? },
        };
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RG16F as i32,
                program::FOG_TABLE,
                1,
                0,
                glow::RG,
                glow::FLOAT,
                glow::PixelUnpackData::Slice(Some(bytemuck::cast_slice(&held.table()))),
            );
            point(gl);
        }
        smooth(gl, texture);
        self.haze = Some((texture, held));
        Ok(texture)
    }

    /// The exposure the chain settled on last frame, taken off the plane it was left in. Two of the
    /// passes read it as a constant rather than as a texture, so it has to come back across; reading
    /// it a frame late is what keeps that from waiting on the card, and is the lag the game runs
    /// with regardless.
    fn readback(&mut self, gl: &glow::Context) {
        let Some(pair) = self.adapted else {
            return;
        };
        let mut held = [0f32; 4];
        unsafe {
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(pair[self.adaptation].0));
            gl.read_buffer(glow::COLOR_ATTACHMENT0);
            gl.read_pixels(
                0,
                0,
                1,
                1,
                glow::RGBA,
                glow::FLOAT,
                glow::PixelPackData::Slice(Some(bytemuck::cast_slice_mut(&mut held))),
            );
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
        }
        // A context that would not answer leaves the last exposure standing rather than taking the
        // frame to nothing, and the pass clamps its own answer into the range the file states.
        if held[0].is_finite() && held[0] > 0.0 {
            self.exposed = held[0];
        }
        // What the frame actually measured, which the exposure alone cannot show once it sits on
        // either end of the range the file states.
        if let Some((_, from, _)) = self.luminance.last() {
            let mut lit = [0f32; 4];
            unsafe {
                gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(*from));
                gl.read_buffer(glow::COLOR_ATTACHMENT0);
                gl.read_pixels(
                    0,
                    0,
                    1,
                    1,
                    glow::RGBA,
                    glow::FLOAT,
                    glow::PixelPackData::Slice(Some(bytemuck::cast_slice_mut(&mut lit))),
                );
                gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
            }
            if lit[0].is_finite() {
                self.measured = lit[0];
            }
        }
    }

    /// How bright the frame turned out, and the frame read back through what that makes of it.
    ///
    /// Six passes: the frame halved until one texel holds the harmonic mean of its luminance, that
    /// carried toward the exposure the last frame settled on, a curve built across the range the
    /// result spans, and the frame read through it. What lands is in the range a screen holds, which
    /// is what the passes after this one expect.
    pub fn expose(
        &mut self,
        gl: &glow::Context,
        held: &Exposure,
        scene: &program::Scene,
    ) -> Result<(), String> {
        let (lit, frame) = self.lit.ok_or("no lit frame")?;
        let source = self.resolved.ok_or("no resolved frame")?;
        let pair = self.adapted.ok_or("no adaptation")?;
        let (into, curve) = self.curve.ok_or("no tone curve")?;
        let levels = self.luminance.clone();
        let last = levels.len().checked_sub(1).ok_or("no measure")?;
        self.readback(gl);
        // The exposure the passes are run under is the chain's own rather than the caller's: it is
        // what the card answered, and only this side of the graph knows it.
        let scene = &program::Scene {
            exposure: program::Exposure {
                adapted: self.exposed(),
                ..scene.exposure
            },
            ..scene.clone()
        };
        unsafe {
            gl.disable(glow::SCISSOR_TEST);
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::CULL_FACE);
            gl.disable(glow::BLEND);
            gl.depth_mask(false);
        }

        let mut read = frame;
        for (at, (size, into, plane)) in levels.into_iter().enumerate() {
            // The first reads the frame, the last turns the sum back into a luminance, and every one
            // between them is the same pass over a smaller square. Those share a slot, since they
            // share a source and a linked program is kept by what it was built from.
            let (program, slot) = match at {
                0 => (&held.initial, 0),
                at if at == last => (&held.last, 2),
                _ => (&held.iterative, 1),
            };
            let reads = match at {
                0 => Reads {
                    input: Some(read),
                    ..Default::default()
                },
                _ => Reads {
                    measure: Some(read),
                    ..Default::default()
                },
            };
            self.pass(gl, EXPOSURE + slot, program, into, scene, Over::Exposing(size, reads))?;
            read = plane;
        }

        // Into the plane the last frame did not write, since the pass reads the one it is carrying
        // from and a texture being written cannot also be sampled.
        let next = 1 - self.adaptation;
        self.pass(
            gl,
            EXPOSURE + 3,
            &held.adapt,
            pair[next].0,
            scene,
            Over::Exposing(
                (1, 1),
                Reads {
                    measure: Some(read),
                    adapted: Some(pair[self.adaptation].1),
                    ..Default::default()
                },
            ),
        )?;
        self.adaptation = next;

        self.pass(
            gl,
            EXPOSURE + 4,
            &held.curve,
            into,
            scene,
            Over::Exposing(
                (program::CURVE, 1),
                Reads {
                    adapted: Some(pair[next].1),
                    ..Default::default()
                },
            ),
        )?;

        // The last pass writes the frame it reads, so it reads the copy instead.
        self.keep(gl)?;
        self.pass(
            gl,
            EXPOSURE + 5,
            &held.tone,
            lit,
            scene,
            Over::Exposing(
                self.size,
                Reads {
                    input: Some(source),
                    measure: Some(curve),
                    ..Default::default()
                },
            ),
        )?;
        self.toned = true;
        Ok(())
    }

    /// The bright end of the frame spread into a halo and added back to it.
    ///
    /// Seven passes: the share of each pixel the composite marked as glare, kept where it is bright
    /// enough to count and taken down to a quarter of the frame; that smoothed; halved again; that
    /// smoothed; spread along each axis in turn; shaped; and added to the frame. How far a halo
    /// reaches follows the level the spreading runs at, since its taps are six texels of what it
    /// reads.
    ///
    /// Ahead of the exposure and the grading, since a halo belongs to the frame the lighting left
    /// rather than to what a curve made of it.
    pub fn glare(
        &mut self,
        gl: &glow::Context,
        held: &Glare,
        scene: &program::Scene,
    ) -> Result<(), String> {
        let (lit, _) = self.lit.ok_or("no lit frame")?;
        let (_, source) = self.sourced.ok_or("no glare source")?;
        let [(bright, kept), (merged, halo)] = self.glared.ok_or("no glare buffers")?;
        let [(first, swept), (second, spread)] = self.reached.ok_or("no glare buffers")?;
        unsafe {
            gl.disable(glow::SCISSOR_TEST);
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::CULL_FACE);
            gl.disable(glow::BLEND);
            gl.depth_mask(false);
        }
        let near = self.spread();
        let far = self.reach();
        let texel = |(wide, tall): (i32, i32)| glam::Vec2::new(1.0 / wide as f32, 1.0 / tall as f32);
        self.pass(
            gl,
            GLARE,
            &held.bright,
            bright,
            scene,
            Over::Glaring(
                near,
                Glared {
                    input: source,
                    merge: None,
                },
            ),
        )?;
        let smoothing = program::Scene {
            blur: program::Blur::Square(texel(near)),
            ..scene.clone()
        };
        self.pass(
            gl,
            GLARE + 1,
            &held.gauss,
            merged,
            &smoothing,
            Over::Blurring(near, kept),
        )?;
        // The level below, which the game copies down with a pass of its own.
        blit(gl, merged, second, near, far);
        let smoothing = program::Scene {
            blur: program::Blur::Square(texel(far)),
            ..scene.clone()
        };
        self.pass(
            gl,
            GLARE + 1,
            &held.gauss,
            first,
            &smoothing,
            Over::Blurring(far, spread),
        )?;
        // A tap of the blur is stated in texels of what it reads, and each half reads the target the
        // other wrote, so the step is the same for both and only its direction differs.
        let step = texel(far);
        for (frame, read, along) in [
            (second, swept, glam::Vec2::new(step.x, 0.0)),
            (first, spread, glam::Vec2::new(0.0, step.y)),
        ] {
            let scene = program::Scene {
                blur: program::Blur::Along(along),
                ..scene.clone()
            };
            self.pass(gl, GLARE + 2, &held.blur, frame, &scene, Over::Blurring(far, read))?;
        }
        self.pass(
            gl,
            GLARE + 3,
            &held.merge,
            merged,
            scene,
            Over::Glaring(
                near,
                Glared {
                    input: swept,
                    merge: Some(kept),
                },
            ),
        )?;
        // The halo's own alpha is nought, which is what makes this add rather than cover.
        unsafe {
            gl.enable(glow::BLEND);
            gl.blend_func_separate(
                glow::ONE,
                glow::ONE_MINUS_SRC_ALPHA,
                glow::ZERO,
                glow::ZERO,
            );
        }
        let drawn = self.pass(gl, GLARE + 4, &held.composite, lit, scene, Over::Reading(halo));
        unsafe { gl.disable(glow::BLEND) };
        drawn
    }

    /// What water's own reflection chain draws into, built where the frame has moved under it. The
    /// same fraction of the frame as the chain above, which is what makes the pyramid's texel and
    /// this chain's the one register the game's own upload writes twice.
    ///
    /// The marched reflection carries more than a byte a channel: what it holds is the frame as the
    /// lighting left it, before anything took that into range.
    fn waters(&mut self, gl: &glow::Context) -> Result<&Watered, String> {
        let size = (
            (self.size.0 / program::REFLECTION_SCALE).max(1),
            (self.size.1 / program::REFLECTION_SCALE).max(1),
        );
        if self.watered.as_ref().is_some_and(|held| held.size == size) {
            return Ok(self.watered.as_ref().expect("just measured"));
        }
        let sharp = plane(gl, size, glow::RGBA16F, glow::RGBA, glow::FLOAT)?;
        smooth(gl, sharp);
        let scratch = plane(gl, size, glow::RGBA8, glow::RGBA, glow::UNSIGNED_BYTE)?;
        smooth(gl, scratch);
        let merged = plane(gl, size, glow::RGBA8, glow::RGBA, glow::UNSIGNED_BYTE)?;
        smooth(gl, merged);
        let stencil = unsafe {
            let held = gl.create_renderbuffer()?;
            gl.bind_renderbuffer(glow::RENDERBUFFER, Some(held));
            gl.renderbuffer_storage(glow::RENDERBUFFER, glow::DEPTH24_STENCIL8, size.0, size.1);
            held
        };
        let mut frames = Vec::new();
        for texture in [sharp, scratch, merged] {
            let held = frame_of(gl, &[texture], None)?;
            unsafe {
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(held));
                gl.framebuffer_renderbuffer(
                    glow::FRAMEBUFFER,
                    glow::DEPTH_STENCIL_ATTACHMENT,
                    glow::RENDERBUFFER,
                    Some(stencil),
                );
                let status = gl.check_framebuffer_status(glow::FRAMEBUFFER);
                if status != glow::FRAMEBUFFER_COMPLETE {
                    return Err(format!("water's reflection would not complete: {status:#x}"));
                }
            }
            frames.push(held);
        }
        self.watered = Some(Watered {
            sharp: (frames[0], sharp),
            scratch: (frames[1], scratch),
            merged: (frames[2], merged),
            stencil,
            size,
        });
        Ok(self.watered.as_ref().expect("just built"))
    }

    /// The buffers that chain draws into, with the pyramid built, everything cleared and the state
    /// the two members drawn over the water itself want already standing.
    ///
    /// The frame the march reads is the one the composite resolved, which is what stands behind the
    /// water at the point this runs.
    pub fn watering(
        &mut self,
        gl: &glow::Context,
        scene: &program::Scene,
    ) -> Result<Watering, String> {
        self.hierarchy(gl, scene)?;
        let frame = self.resolved.ok_or("no resolved frame")?;
        let depth = self.mirrors(gl)?.depth.0;
        let held = self.waters(gl)?;
        let (into, size) = (held.sharp.0, held.size);
        let frames = [held.sharp.0, held.scratch.0, held.merged.0];
        unsafe {
            gl.disable(glow::SCISSOR_TEST);
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::BLEND);
            gl.depth_mask(false);
            gl.color_mask(true, true, true, true);
            gl.clear_color(0.0, 0.0, 0.0, 0.0);
            gl.clear_stencil(0);
            gl.stencil_mask(0xff);
            for held in frames {
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(held));
                gl.draw_buffers(&[glow::COLOR_ATTACHMENT0]);
                gl.clear(glow::COLOR_BUFFER_BIT | glow::STENCIL_BUFFER_BIT);
            }
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(into));
            gl.viewport(0, 0, size.0, size.1);
            gl.enable(glow::STENCIL_TEST);
        }
        Ok(Watering {
            into,
            depth,
            frame,
            size,
        })
    }

    /// The four members of that chain drawn over a quad: the blur across and down, the wide blur's
    /// first half, and the merge that picks between the sharp answer and the spread one per pixel.
    ///
    /// The game runs a second wide half after this one, into the same buffer the merge writes and
    /// with nothing between them, so its answer never reaches water; what the merge is handed is the
    /// first half, which is what the game's own bindings hand it.
    pub fn water_mirror(
        &mut self,
        gl: &glow::Context,
        held: &WaterMirror,
        scene: &program::Scene,
    ) -> Result<(), String> {
        let buffers = self.waters(gl)?;
        let (sharp, scratch, merged, size) =
            (buffers.sharp, buffers.scratch, buffers.merged, buffers.size);
        let reads = Reflecting {
            depth: sharp.1,
            frame: sharp.1,
            normal: sharp.1,
            mask: sharp.1,
            blurred: sharp.1,
        };
        let scene = &program::Scene {
            reflect: program::Reflect {
                level: 0,
                texel: glam::Vec2::new(1.0 / size.0 as f32, 1.0 / size.1 as f32),
            },
            ..scene.clone()
        };
        unsafe {
            gl.stencil_func(glow::NOTEQUAL, 0, 0xff);
            gl.stencil_op(glow::KEEP, glow::KEEP, glow::KEEP);
            gl.stencil_mask(0);
        }
        for (at, program, into, source) in [
            (0, &held.blur[0], scratch.0, sharp.1),
            (1, &held.blur[1], sharp.0, scratch.1),
            (2, &held.wide, scratch.0, sharp.1),
        ] {
            self.pass(
                gl,
                WATER_REFLECT + at,
                program,
                into,
                scene,
                Over::Reflecting(size, Member::Blur, Reflecting {
                    blurred: source,
                    ..reads
                }),
            )?;
        }
        let drawn = self.pass(
            gl,
            WATER_REFLECT + 3,
            &held.merge,
            merged.0,
            scene,
            Over::Reflecting(size, Member::Blur, Reflecting {
                frame: sharp.1,
                blurred: scratch.1,
                ..reads
            }),
        );
        unsafe {
            gl.disable(glow::STENCIL_TEST);
            gl.stencil_mask(0xff);
        }
        self.drawn.water = drawn.is_ok();
        drawn
    }

    /// What the reflection chain draws into, built where the frame has moved under it.
    fn mirrors(&mut self, gl: &glow::Context) -> Result<&Mirrored, String> {
        let size = (
            (self.size.0 / program::REFLECTION_SCALE).max(1),
            (self.size.1 / program::REFLECTION_SCALE).max(1),
        );
        if self.mirrored.as_ref().is_some_and(|held| held.size == size) {
            return Ok(self.mirrored.as_ref().expect("just measured"));
        }
        // One level fewer than asked for wherever the frame is too small to halve that many times,
        // since a chain deeper than its own largest edge is not a texture the card will allocate.
        let deepest = 32 - (size.0.max(size.1) as u32).leading_zeros() as i32;
        let normal = plane(gl, size, glow::RGBA8, glow::RGBA, glow::UNSIGNED_BYTE)?;
        smooth(gl, normal);
        let mask = plane(gl, size, glow::RGBA16F, glow::RGBA, glow::FLOAT)?;
        smooth(gl, mask);
        let held = Mirrored {
            depth: pyramid(
                gl,
                size,
                program::REFLECTION_DEPTHS.min(deepest),
                glow::RG16F,
                glow::NEAREST,
            )?,
            normal: (frame_of(gl, &[normal], None)?, normal),
            mask: (frame_of(gl, &[mask], None)?, mask),
            chain: [
                pyramid(
                    gl,
                    size,
                    (program::REFLECTION_LEVELS + 1).min(deepest),
                    glow::RGBA16F,
                    glow::LINEAR,
                )?,
                pyramid(
                    gl,
                    size,
                    (program::REFLECTION_LEVELS + 1).min(deepest),
                    glow::RGBA16F,
                    glow::LINEAR,
                )?,
            ],
            size,
        };
        self.mirrored = Some(held);
        Ok(self.mirrored.as_ref().expect("just built"))
    }

    /// The frame's own depth as the pyramid of maxima the march walks, one level a draw.
    ///
    /// The game builds all four levels in one compute shader, which a context this draws through has
    /// none of, so this is the same reduction written as a draw. The reading it leaves is the game's
    /// rather than the viewer's: the game stands its near plane at one and its far plane at nought,
    /// which is the ordering the march's hit test is written against.
    fn hierarchy(&mut self, gl: &glow::Context, scene: &program::Scene) -> Result<(), String> {
        let program = match self.hierarchy {
            Some(held) => held,
            None => {
                let held = build_pair(gl, program::POST_VERTEX, HIERARCHY)?;
                self.hierarchy = Some(held);
                held
            }
        };
        let depth = self.depth.ok_or("no depth buffer")?;
        let layout = self.screen(gl)?;
        let (near, far) = scene.planes();
        let frame = self.size;
        let held = self.mirrors(gl)?;
        let (texture, frames) = held.depth.clone();
        let size = held.size;
        unsafe {
            gl.use_program(Some(program));
            gl.bind_vertex_array(Some(layout));
            gl.active_texture(glow::TEXTURE0);
        }
        let uniform = |name: &str| unsafe { gl.get_uniform_location(program, name) };
        for (level, into) in frames.iter().enumerate() {
            let source = match level {
                0 => frame,
                _ => (
                    (size.0 >> (level - 1)).max(1),
                    (size.1 >> (level - 1)).max(1),
                ),
            };
            unsafe {
                // The level read is cut down to before the level written is attached: a texture
                // whose sampled levels include the one under the pen is a draw the card rejects.
                gl.bind_framebuffer(glow::FRAMEBUFFER, None);
                gl.bind_texture(glow::TEXTURE_2D, Some(texture));
                for name in [glow::TEXTURE_BASE_LEVEL, glow::TEXTURE_MAX_LEVEL] {
                    gl.tex_parameter_i32(glow::TEXTURE_2D, name, level.saturating_sub(1) as i32);
                }
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(*into));
                gl.draw_buffers(&[glow::COLOR_ATTACHMENT0]);
                gl.viewport(0, 0, (size.0 >> level).max(1), (size.1 >> level).max(1));
                if let Some(at) = uniform("u_texel") {
                    gl.uniform_2_f32(Some(&at), 1.0 / source.0 as f32, 1.0 / source.1 as f32);
                }
                if let Some(at) = uniform("u_level") {
                    gl.uniform_1_f32(Some(&at), level.saturating_sub(1) as f32);
                }
                if let Some(at) = uniform("u_range") {
                    gl.uniform_1_f32(Some(&at), (far - near) / far.max(f32::EPSILON));
                }
                if let Some(at) = uniform("u_first") {
                    gl.uniform_1_i32(Some(&at), i32::from(level == 0));
                }
                sampler(
                    gl,
                    program,
                    "u_source",
                    0,
                    match level {
                        0 => depth,
                        _ => texture,
                    },
                );
                gl.draw_arrays(glow::TRIANGLES, 0, 3);
            }
        }
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.bind_vertex_array(None);
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_BASE_LEVEL, 0);
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAX_LEVEL,
                frames.len() as i32 - 1,
            );
        }
        Ok(())
    }

    /// The frame reflected off itself, which is what a metal surface answers with where nothing
    /// captured an environment for it.
    ///
    /// The march reads the frame while the copy at the end writes it, so the copy taken once here is
    /// what it reads. Every member runs at half the frame, which is what the game runs them at and
    /// what the buffers are sized to.
    pub fn mirror(
        &mut self,
        gl: &glow::Context,
        held: &Reflection,
        scene: &program::Scene,
    ) -> Result<(), String> {
        let (lit, _) = self.lit.ok_or("no lit frame")?;
        self.keep(gl)?;
        unsafe {
            gl.disable(glow::SCISSOR_TEST);
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::CULL_FACE);
            gl.disable(glow::BLEND);
            gl.depth_mask(false);
        }
        self.hierarchy(gl, scene)?;
        let frame = self.resolved.ok_or("no resolved frame")?;
        let mirrors = self.mirrors(gl)?;
        let size = mirrors.size;
        let levels = mirrors.chain[0].1.len();
        let reads = Reflecting {
            depth: mirrors.depth.0,
            frame,
            normal: mirrors.normal.1,
            mask: mirrors.mask.1,
            blurred: mirrors.chain[0].0,
        };
        let (into, masked) = (mirrors.normal.0, mirrors.mask.0);
        let marched = mirrors.chain[0].1[0];
        let chain: [(glow::Texture, Vec<glow::Framebuffer>); 2] = mirrors.chain.clone();
        let scene = &program::Scene {
            reflect: program::Reflect {
                level: 0,
                texel: glam::Vec2::new(1.0 / size.0 as f32, 1.0 / size.1 as f32),
            },
            ..scene.clone()
        };
        self.pass(
            gl,
            REFLECT,
            &held.normal,
            into,
            scene,
            Over::Reflecting(size, Member::Normal, reads),
        )?;
        // The mask drops the pixels no reflection reaches rather than writing a nought to them, so
        // what it leaves behind on those is whatever it wrote last frame unless this clears it.
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(masked));
            gl.clear_color(0.0, 0.0, 0.0, 0.0);
            gl.clear(glow::COLOR_BUFFER_BIT);
        }
        self.pass(
            gl,
            REFLECT + 1,
            &held.mask,
            masked,
            scene,
            Over::Reflecting(size, Member::Mask, reads),
        )?;
        self.pass(
            gl,
            REFLECT + 2,
            &held.march,
            marched,
            scene,
            Over::Reflecting(size, Member::March, reads),
        )?;
        // Each level reads the level above it out of the other half of the pair, so the step a tap
        // is stated in is the texel of that level rather than of the one being written.
        for level in 1..levels {
            let source = (
                (size.0 >> (level - 1)).max(1) as f32,
                (size.1 >> (level - 1)).max(1) as f32,
            );
            let scene = program::Scene {
                reflect: program::Reflect {
                    level: level as i32,
                    texel: glam::Vec2::new(1.0 / source.0, 1.0 / source.1),
                },
                ..scene.clone()
            };
            let over = ((size.0 >> level).max(1), (size.1 >> level).max(1));
            for (at, from) in [(1, 0), (0, 1)] {
                self.pass(
                    gl,
                    REFLECT + 3 + from,
                    &held.blur[from],
                    chain[at].1[level],
                    &scene,
                    Over::Reflecting(
                        over,
                        Member::Blur,
                        Reflecting {
                            blurred: chain[from].0,
                            ..reads
                        },
                    ),
                )?;
            }
        }
        // One level past the last the blur wrote, which is what caps the level the resolve picks
        // per pixel: at nought it would read the marched level and none of the blurred ones.
        let resolving = program::Scene {
            reflect: program::Reflect {
                level: levels as i32,
                ..scene.reflect
            },
            ..scene.clone()
        };
        self.pass(
            gl,
            REFLECT + 5,
            &held.distort,
            chain[1].1[0],
            &resolving,
            Over::Reflecting(size, Member::Distort, reads),
        )?;
        // The reflection is added to the frame by the square of itself, which is the blend the game
        // lays it back with.
        unsafe {
            gl.enable(glow::BLEND);
            gl.blend_equation(glow::FUNC_ADD);
            gl.blend_func_separate(glow::SRC_COLOR, glow::ONE, glow::ZERO, glow::ONE);
        }
        let drawn = self.pass(
            gl,
            REFLECT + 6,
            &held.copy,
            lit,
            scene,
            Over::Reflecting(
                self.size,
                Member::Copy,
                Reflecting {
                    blurred: chain[1].0,
                    ..reads
                },
            ),
        );
        unsafe { gl.disable(glow::BLEND) };
        self.drawn.reflection = drawn.is_ok();
        drawn
    }

    /// The frame with its edges smoothed, which is the last thing the graph does to it.
    ///
    /// Two passes rather than one: the game works each pixel's brightness out in a pass of its own
    /// and leaves it in the alpha the pass after it reads its edges off. Each reads the copy the one
    /// before it left, since a texture being written cannot also be sampled.
    pub fn antialias(
        &mut self,
        gl: &glow::Context,
        held: &Smoothing,
        scene: &program::Scene,
    ) -> Result<(), String> {
        let (lit, _) = self.lit.ok_or("no lit frame")?;
        let (into, smoothed) = self.smoothed.ok_or("no smoothed frame")?;
        unsafe {
            gl.disable(glow::SCISSOR_TEST);
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::CULL_FACE);
            gl.disable(glow::BLEND);
            gl.depth_mask(false);
        }
        self.keep(gl)?;
        self.pass(gl, 6, &held.luma, into, scene, Over::Screen)?;
        self.pass(gl, 7, &held.fxaa, lit, scene, Over::Reading(smoothed))
    }

    /// The frame's four corners taken down toward black, which is the last thing the game does to
    /// one and the last thing done here.
    ///
    /// The pass answers with that color and the share of it a pixel takes, so it is blended over the
    /// frame rather than written; it reads no texture at all, only where the pixel stands. The
    /// game draws it into a quarter of the frame and lays that back over one with an alpha test,
    /// which saves it three quarters of a ramp it can work out per pixel for nothing.
    pub fn vignette(
        &mut self,
        gl: &glow::Context,
        held: &std::sync::Arc<program::Program>,
        scene: &program::Scene,
    ) -> Result<(), String> {
        let (lit, _) = self.lit.ok_or("no lit frame")?;
        unsafe {
            gl.disable(glow::SCISSOR_TEST);
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::CULL_FACE);
            gl.depth_mask(false);
            gl.enable(glow::BLEND);
            gl.blend_equation(glow::FUNC_ADD);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);
        }
        let drawn = self.pass(gl, VIGNETTE, held, lit, scene, Over::Screen);
        unsafe { gl.disable(glow::BLEND) };
        self.drawn.vignette = drawn.is_ok();
        drawn
    }

    /// The framebuffer the composite resolved into, standing on a copy of the depth rather than
    /// the depth itself: what a pass drawn over the frame writes, when that pass also samples the
    /// scene depth, since a texture cannot be attached to the framebuffer it is bound to sample
    /// from.
    pub fn bare(&self) -> Option<glow::Framebuffer> {
        self.bare
    }

    /// The copy of the depth that framebuffer stands on, for a pass that samples it back rather
    /// than only testing against it.
    pub fn cutoff(&self) -> Option<glow::Texture> {
        self.cutoff
    }

    /// The depth the frame's own geometry left, for a pass that samples it back while drawing
    /// through the framebuffer standing on the copy.
    pub fn depth(&self) -> Option<glow::Texture> {
        self.depth
    }

    /// The live frame the composite resolved into, standing on the depth itself: sun, moon and an
    /// effect's own glow all draw here, since none of them sample it back.
    pub fn lit(&self) -> Option<glow::Framebuffer> {
        self.lit.map(|(frame, _)| frame)
    }

    /// Refreshes that copy from what the frame currently holds.
    pub fn cut(&self, gl: &glow::Context) -> Result<(), String> {
        let (frame, _) = self.lit.ok_or("no lit frame")?;
        let into = self.bare.ok_or("no lit frame")?;
        unsafe {
            gl.disable(glow::SCISSOR_TEST);
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(frame));
            gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(into));
            let (width, height) = self.size;
            gl.blit_framebuffer(
                0,
                0,
                width,
                height,
                0,
                0,
                width,
                height,
                glow::DEPTH_BUFFER_BIT,
                glow::NEAREST,
            );
        }
        Ok(())
    }

    /// Clears one page of the G-buffer and points the draw buffers at every attachment it has.
    /// Binds a page and stands its state up, leaving whatever it already holds: a model standing in
    /// someone else's frame fills a buffer that frame has already written into.
    pub fn reopen(&self, gl: &glow::Context, page: usize) {
        let Some(held) = self.frames.get(page) else {
            return;
        };
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(*held));
            // A clear only reaches the attachments the draw buffers name, and a framebuffer starts
            // out naming its first alone: without this the rest keep whatever the texture happened
            // to be created holding.
            let width = (TARGETS - page * self.attachments).min(self.attachments);
            let attachments: Vec<u32> = (0..width)
                .map(|at| glow::COLOR_ATTACHMENT0 + at as u32)
                .collect();
            gl.draw_buffers(&attachments);
            gl.viewport(0, 0, self.size.0, self.size.1);
            // egui leaves the scissor set to the widget's rect in the window's own coordinates, and
            // the frame is a buffer of its own that starts at nought: leaving it on clips the clear
            // and every draw after it to whatever part of the frame the rect happens to overlap.
            gl.disable(glow::SCISSOR_TEST);
            gl.disable(glow::BLEND);
            gl.enable(glow::DEPTH_TEST);
            gl.depth_func(glow::LEQUAL);
            gl.depth_mask(true);
            gl.color_mask(true, true, true, true);
        }
    }

    /// The same, emptied first.
    pub fn open(&self, gl: &glow::Context, page: usize) {
        self.reopen(gl, page);
        if self.frames.get(page).is_none() {
            return;
        }
        unsafe {
            gl.clear_color(0.0, 0.0, 0.0, 0.0);
            // Said rather than assumed: the pass that puts the frame on screen reads this value
            // back as the sign that nothing drew.
            gl.clear_depth_f32(1.0);
            // Every page draws the same geometry against one depth buffer, so only the first of
            // them clears it: what a later page does not draw still covered the pixel.
            gl.clear(match page {
                0 => glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT,
                _ => glow::COLOR_BUFFER_BIT,
            });
        }
    }

    /// The buffers a semi-transparent surface is drawn and lit through, made the first time one is.
    fn sheered(&mut self, gl: &glow::Context) -> Result<(), String> {
        if self.sheer.is_some() {
            return Ok(());
        }
        let size = self.size;
        let depth = unsafe {
            let held = gl.create_texture()?;
            gl.bind_texture(glow::TEXTURE_2D, Some(held));
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::DEPTH_COMPONENT24 as i32,
                size.0,
                size.1,
                0,
                glow::DEPTH_COMPONENT,
                glow::UNSIGNED_INT,
                glow::PixelUnpackData::Slice(None),
            );
            point(gl);
            held
        };
        let color: Vec<glow::Texture> = (0..SHEER)
            .map(|_| plane(gl, size, glow::RGBA8, glow::RGBA, glow::UNSIGNED_BYTE))
            .collect::<Result<_, _>>()?;
        let position = plane(gl, size, glow::RGBA16F, glow::RGBA, glow::FLOAT)?;
        self.sheer = Some(Sheer {
            frame: frame_of(gl, &color, Some(depth))?,
            color,
            depth,
            position: (frame_of(gl, &[position], None)?, position),
        });
        Ok(())
    }

    /// What one of its lighting passes reads.
    fn sheering(&self) -> Result<Sheered, String> {
        let held = self.sheer.as_ref().ok_or("no transparency buffer")?;
        Ok(Sheered {
            color: held
                .color
                .as_slice()
                .try_into()
                .map_err(|_| "the transparency buffer is short a channel")?,
            depth: held.depth,
            position: held.position.1,
        })
    }

    /// Clears that buffer and stands the frame's own depth in it, so a surface behind what the
    /// opaque passes drew is hidden by them and one in front of it settles a depth of its own.
    pub fn sheer(&mut self, gl: &glow::Context) -> Result<(), String> {
        self.sheered(gl)?;
        let from = *self.frames.first().ok_or("no G-buffer")?;
        let (width, height) = self.size;
        let into = self.sheer.as_ref().ok_or("no transparency buffer")?.frame;
        let attachments: Vec<u32> = (0..SHEER)
            .map(|at| glow::COLOR_ATTACHMENT0 + at as u32)
            .collect();
        unsafe {
            gl.disable(glow::SCISSOR_TEST);
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(into));
            gl.draw_buffers(&attachments);
            gl.viewport(0, 0, width, height);
            gl.clear_color(SHEER_CLEAR[0], SHEER_CLEAR[1], SHEER_CLEAR[2], SHEER_CLEAR[3]);
            gl.disable(glow::BLEND);
            gl.color_mask(true, true, true, true);
            gl.clear(glow::COLOR_BUFFER_BIT);
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(from));
            gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(into));
            gl.blit_framebuffer(
                0,
                0,
                width,
                height,
                0,
                0,
                width,
                height,
                glow::DEPTH_BUFFER_BIT,
                glow::NEAREST,
            );
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(into));
            gl.enable(glow::DEPTH_TEST);
            // Strictly nearer: a fragment standing where the opaque passes already drew is one this
            // buffer has nothing to say about, and its own resolve drops it anyway.
            gl.depth_func(glow::LESS);
            gl.depth_mask(true);
        }
        Ok(())
    }

    /// The light a semi-transparent surface gathers, worked out again over the buffer it filled.
    ///
    /// After the composite rather than beside it: both halves of the frame gather their light into
    /// the same two buffers, and the opaque composite has to have read its own before this clears
    /// them. The type index moves channel between the two packings, which is what the pass picks
    /// between by `g_LightDrawParam`.
    pub fn relight(
        &mut self,
        gl: &glow::Context,
        lighting: &Lighting,
        scene: &program::Scene,
        lamps: &[program::Lamp],
    ) -> Result<(), String> {
        let reads = self.sheering()?;
        let position = self
            .sheer
            .as_ref()
            .ok_or("no transparency buffer")?
            .position
            .0;
        let (light, _) = self.light.ok_or("no light buffer")?;
        let mut held = program::Scene {
            sheer: true,
            ..scene.clone()
        };
        unsafe {
            gl.disable(glow::DEPTH_TEST);
            gl.depth_mask(false);
            gl.disable(glow::CULL_FACE);
            gl.disable(glow::BLEND);
        }
        self.pass(
            gl,
            0,
            &lighting.position,
            position,
            &held,
            Over::Sheering(reads),
        )?;
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(light));
            gl.draw_buffers(&[glow::COLOR_ATTACHMENT0, glow::COLOR_ATTACHMENT1]);
            gl.clear_color(0.0, 0.0, 0.0, 0.0);
            gl.clear(glow::COLOR_BUFFER_BIT);
            gl.enable(glow::BLEND);
            gl.blend_func(glow::ONE, glow::ONE);
        }
        self.pass(
            gl,
            1,
            &lighting.directional,
            light,
            &held,
            Over::Sheering(reads),
        )?;
        unsafe {
            gl.enable(glow::CULL_FACE);
            gl.cull_face(glow::FRONT);
            gl.front_face(glow::CCW);
        }
        let mut sorted = lamps.to_vec();
        sorted.sort_by_key(|lamp| (lamp.kind as u8, lamp.falloff));
        let mut standing = None;
        for lamp in &sorted {
            held.lamp = *lamp;
            let (kind, programs) = match lamp.kind {
                program::LampKind::Spot => (1, lighting.spot.as_ref()),
                program::LampKind::Line => (2, lighting.line.as_ref()),
                program::LampKind::Plane => (3, lighting.plane.as_ref()),
                program::LampKind::Point => (0, None),
            };
            let (kind, programs) = match programs {
                Some(held) => (kind, held),
                None => (0, &lighting.point),
            };
            let falloff = lamp.falloff.min(programs.len() - 1);
            let slot = SHEER_LAMP + kind * program::ATTENUATION.len() + falloff;
            let program = &programs[falloff];
            match standing == Some(slot) {
                true => self.again(gl, program, &held),
                false => {
                    self.pass(gl, slot, program, light, &held, Over::Sheering(reads))?;
                    standing = Some(slot);
                }
            }
        }
        unsafe {
            gl.disable(glow::CULL_FACE);
            gl.disable(glow::BLEND);
        }
        Ok(())
    }

    /// The pair that bends a frame toward what a screen holds and puts it up, built the first time
    /// something draws it.
    fn presenter(&mut self, gl: &glow::Context) -> Result<glow::Program, String> {
        if let Some(held) = self.present {
            return Ok(held);
        }
        let held = build_pair(gl, PRESENT_VERTEX, PRESENT_FRAGMENT)?;
        self.present = Some(held);
        Ok(held)
    }

    /// The screen-wide triangle's array, uploaded the first time something draws it.
    fn screen(&mut self, gl: &glow::Context) -> Result<glow::VertexArray, String> {
        if let Some((layout, _)) = self.screen {
            return Ok(layout);
        }
        let held = upload_screen(gl, &SCREEN)?;
        self.screen = Some(held);
        Ok(held.0)
    }

    /// The same triangle carrying what a sampling pass reads at, likewise.
    fn sampled(&mut self, gl: &glow::Context) -> Result<glow::VertexArray, String> {
        if let Some((layout, _)) = self.sampled {
            return Ok(layout);
        }
        let held = upload_screen(gl, &SAMPLED)?;
        self.sampled = Some(held);
        Ok(held.0)
    }

    /// The one texture standing in for whatever a material binds nothing to.
    pub fn stand_in(&mut self, gl: &glow::Context) -> Result<glow::Texture, String> {
        self.blank(gl, glow::TEXTURE_2D)
    }

    /// The same, at whichever target a sampler was declared over.
    fn blank(&mut self, gl: &glow::Context, target: u32) -> Result<glow::Texture, String> {
        if let Some(held) = self.blanks.get(&target) {
            return Ok(*held);
        }
        let held = flat(gl, target, &STAND_IN)?;
        self.blanks.insert(target, held);
        Ok(held)
    }

    /// One of those tables, at the value that leaves the term it drives where it was.
    fn neutral(
        &mut self,
        gl: &glow::Context,
        id: u32,
        value: &[u8; 4],
    ) -> Result<glow::Texture, String> {
        if let Some(held) = self.neutrals.get(&id) {
            return Ok(*held);
        }
        let held = flat(gl, glow::TEXTURE_2D, value)?;
        self.neutrals.insert(id, held);
        Ok(held)
    }

    /// What a lighting pass reads where nothing occluded the pixel. Nothing here computes occlusion,
    /// so every pixel answers the same, and it is not the value a color map would stand in with.
    fn unoccluded(&mut self, gl: &glow::Context) -> Result<glow::Texture, String> {
        if let Some(held) = self.unoccluded {
            return Ok(held);
        }
        let held = flat(gl, glow::TEXTURE_2D, &UNOCCLUDED)?;
        self.unoccluded = Some(held);
        Ok(held)
    }

    /// The table `SV_Target.w` indexes, which every pixel shader that shades a surface reads. Empty
    /// until the files it is filled from arrive, which is the branch a plain surface takes.
    pub fn types(&mut self, gl: &glow::Context) -> Result<glow::Texture, String> {
        if let Some(held) = self.types {
            return Ok(held);
        }
        let held = dwords(gl, &program::shader_types(&[]))?;
        self.types = Some(held);
        Ok(held)
    }

    /// The same table, as the parameter files that have arrived state it.
    pub fn fill_types(&mut self, gl: &glow::Context, values: &[u32]) -> Result<(), String> {
        let held = dwords(gl, values)?;
        if let Some(stale) = self.types.replace(held) {
            graveyard().lock().unwrap().push(Dead::Texture(stale));
        }
        Ok(())
    }

    /// The cube the composite takes reflections against, before a place has said what its sky looks
    /// like. The alpha is the weight it is blended in at, so nought leaves the ambient it would
    /// otherwise replace.
    pub fn reflection(&mut self, gl: &glow::Context) -> Result<glow::Texture, String> {
        if let Some(held) = self.reflection {
            return Ok(held);
        }
        let held = flat(gl, glow::TEXTURE_CUBE_MAP, &UNREFLECTED)?;
        self.reflection = Some(held);
        Ok(held)
    }

    /// The same cube off the sky a place states, which is what a smooth surface has to reflect where
    /// nothing captures the frame around it. A harmonic row dotted against a direction is what that
    /// sky looks like that way, and a mirror reflection is that same sky read the mirror way.
    ///
    /// Gamma-encoded, since a composite squares the texel and divides it by the alpha it was
    /// gathered at rather than reading it as the light it stands for.
    ///
    /// Rebuilt only where the sky changed: a zone states one per time of day, and a frame otherwise
    /// asks for the same cube it asked for last.
    pub fn reflect(&mut self, gl: &glow::Context, held: &program::Ambient) -> Result<(), String> {
        let sky = (held.sky, held.sky_scale);
        if self.sky == Some(sky) {
            return Ok(());
        }
        let mut pixels = Vec::with_capacity((6 * SKY_FACE * SKY_FACE * 4) as usize);
        for face in 0..6 {
            for y in 0..SKY_FACE {
                for x in 0..SKY_FACE {
                    let over = |at: i32| 2.0 * (at as f32 + 0.5) / SKY_FACE as f32 - 1.0;
                    let toward = facing(face, over(x), over(y)).extend(1.0);
                    pixels.extend(held.sky.iter().map(|row| {
                        let held = row.dot(toward) * sky.1;
                        (held.clamp(0.0, 1.0).sqrt() * 255.0).round() as u8
                    }));
                    pixels.push(255);
                }
            }
        }
        unsafe {
            let texture = gl.create_texture()?;
            gl.bind_texture(glow::TEXTURE_CUBE_MAP, Some(texture));
            let face = (SKY_FACE * SKY_FACE * 4) as usize;
            for at in 0..6 {
                gl.tex_image_2d(
                    glow::TEXTURE_CUBE_MAP_POSITIVE_X + at as u32,
                    0,
                    glow::RGBA8 as i32,
                    SKY_FACE,
                    SKY_FACE,
                    0,
                    glow::RGBA,
                    glow::UNSIGNED_BYTE,
                    glow::PixelUnpackData::Slice(Some(&pixels[at * face..(at + 1) * face])),
                );
            }
            for (name, value) in [
                (glow::TEXTURE_MIN_FILTER, glow::LINEAR),
                (glow::TEXTURE_MAG_FILTER, glow::LINEAR),
                (glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE),
                (glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE),
            ] {
                gl.tex_parameter_i32(glow::TEXTURE_CUBE_MAP, name, value as i32);
            }
            if let Some(stale) = self.reflection.replace(texture) {
                graveyard().lock().unwrap().push(Dead::Texture(stale));
            }
        }
        self.sky = Some(sky);
        Ok(())
    }

    /// Takes one of the game's own textures onto the card at the target its own file calls for: a
    /// plane, an array, a volume or a cube. A draw only validates where the texture bound to a unit
    /// is of the declaration's own kind.
    ///
    /// An array repeats, since the shaders that read one scale the coordinate up by a tile factor
    /// and expect the tile to come round again. Everything else is clamped: a plane is addressed
    /// over its whole width, and wrapping would blend its last texel against its first. What is read
    /// between the texels is the caller's to say, since nothing about the texture tells whether it
    /// is an image or a table.
    fn upload(gl: &glow::Context, held: &Layered) -> Result<(u32, glow::Texture), String> {
        let target = target(held.kind);
        let wrap = match held.kind {
            program::Kind::Array => glow::REPEAT,
            _ => glow::CLAMP_TO_EDGE,
        };
        unsafe {
            let texture = gl.create_texture()?;
            gl.bind_texture(target, Some(texture));
            match target {
                glow::TEXTURE_2D => gl.tex_image_2d(
                    target,
                    0,
                    glow::RGBA8 as i32,
                    held.size.0,
                    held.size.1,
                    0,
                    glow::RGBA,
                    glow::UNSIGNED_BYTE,
                    glow::PixelUnpackData::Slice(Some(&held.pixels)),
                ),
                glow::TEXTURE_CUBE_MAP => {
                    let face = (held.size.0 * held.size.1 * 4) as usize;
                    for at in 0..6 {
                        gl.tex_image_2d(
                            glow::TEXTURE_CUBE_MAP_POSITIVE_X + at as u32,
                            0,
                            glow::RGBA8 as i32,
                            held.size.0,
                            held.size.1,
                            0,
                            glow::RGBA,
                            glow::UNSIGNED_BYTE,
                            glow::PixelUnpackData::Slice(
                                held.pixels.get(at * face..(at + 1) * face),
                            ),
                        );
                    }
                }
                _ => gl.tex_image_3d(
                    target,
                    0,
                    glow::RGBA8 as i32,
                    held.size.0,
                    held.size.1,
                    held.layers,
                    0,
                    glow::RGBA,
                    glow::UNSIGNED_BYTE,
                    glow::PixelUnpackData::Slice(Some(&held.pixels)),
                ),
            }
            for (name, value) in [
                (glow::TEXTURE_MIN_FILTER, held.filter),
                (glow::TEXTURE_MAG_FILTER, held.filter),
                (glow::TEXTURE_WRAP_S, wrap),
                (glow::TEXTURE_WRAP_T, wrap),
            ] {
                gl.tex_parameter_i32(target, name, value as i32);
            }
            if target == glow::TEXTURE_3D {
                gl.tex_parameter_i32(target, glow::TEXTURE_WRAP_R, wrap as i32);
            }
            Ok((target, texture))
        }
    }

    /// One of the game's own textures under the resource id the shaders know it by, which is how
    /// the engine's own set is reached.
    pub fn layered(&mut self, gl: &glow::Context, id: u32, held: &Layered) -> Result<(), String> {
        let held = Self::upload(gl, held)?;
        // The wind field is read at a world-scaled UV and the dither at a quarter of the pixel's
        // own coordinate, both of which run well past a single tile, so `upload`'s own `Kind`-based
        // guess of clamping a plane is wrong for these two. No file states either sampler's state
        // at all, so REPEAT is read off the shaders' own math rather than off any capture.
        if matches!(id, WIND_SAMPLE_0 | WIND_SAMPLE_1 | DITHER) {
            unsafe {
                gl.bind_texture(held.0, Some(held.1));
                gl.tex_parameter_i32(held.0, glow::TEXTURE_WRAP_S, glow::REPEAT as i32);
                gl.tex_parameter_i32(held.0, glow::TEXTURE_WRAP_T, glow::REPEAT as i32);
            }
        }
        if let Some((_, stale)) = self.arrays.insert(id, held) {
            graveyard().lock().unwrap().push(Dead::Texture(stale));
        }
        Ok(())
    }

    /// One a material names for itself, under the path that named it. Not under its resource id:
    /// two materials name different cubes at the same sampler, and a zone holds both.
    pub fn stack(&mut self, gl: &glow::Context, path: &str, held: &Layered) -> Result<(), String> {
        let held = Self::upload(gl, held)?;
        if let Some((_, stale)) = self.stacked.insert(path.to_owned(), held) {
            graveyard().lock().unwrap().push(Dead::Texture(stale));
        }
        Ok(())
    }

    /// The texture a material named at this path, where it has arrived and reads through a sampler
    /// of this kind.
    pub fn stacked(&self, kind: program::Kind, path: &str) -> Option<glow::Texture> {
        self.stacked
            .get(path)
            .filter(|(at, _)| *at == target(kind))
            .map(|(_, held)| *held)
    }

    /// The sampler the sun's own map is gathered through. The map is set to compare, and a texture
    /// set to compare answers nothing at all where the sampler reading it does not; the softening
    /// reads it both ways in one draw, so the unit doing the gathering takes a sampler of its own,
    /// which is where that state sits once one is bound.
    fn gathering(&mut self, gl: &glow::Context) -> Result<glow::Sampler, String> {
        if let Some(held) = self.gather {
            return Ok(held);
        }
        let held = unsafe {
            let held = gl.create_sampler()?;
            for (name, value) in [
                (glow::TEXTURE_MIN_FILTER, glow::NEAREST),
                (glow::TEXTURE_MAG_FILTER, glow::NEAREST),
                (glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE),
                (glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE),
                (glow::TEXTURE_COMPARE_MODE, glow::NONE),
            ] {
                gl.sampler_parameter_i32(held, name, value as i32);
            }
            held
        };
        self.gather = Some(held);
        Ok(held)
    }

    /// The file a resource id names, where one has arrived and its own target is the one a sampler
    /// of this kind reads through. A file states how many slices it holds and a package states what
    /// it is sampled as, so the two can disagree; a texture bound at the other target is not of the
    /// declaration's kind, and answers a constant rather than being rejected.
    fn supplied(&self, kind: program::Kind, id: u32) -> Option<glow::Texture> {
        self.arrays
            .get(&id)
            .filter(|(at, _)| *at == target(kind))
            .map(|(_, held)| *held)
    }

    /// What stands in for a sampler of this kind that nothing bound a texture to: the file the
    /// resource id names once that has arrived, and the flat texture of its own target until then.
    pub fn absent(
        &mut self,
        gl: &glow::Context,
        kind: program::Kind,
        id: u32,
    ) -> Result<glow::Texture, String> {
        match self.supplied(kind, id) {
            Some(texture) => Ok(texture),
            None => match kind {
                program::Kind::Cube => self.reflection(gl),
                held => self.blank(gl, target(held)),
            },
        }
    }

    /// Every stand-in a draw may reach for, made before any of them is bound.
    ///
    /// Making a texture binds it to whichever unit happens to be active, so one made partway through
    /// a binding loop takes over the unit the sampler before it was given, and that sampler then
    /// reads a texture of the wrong format.
    pub fn stand_ins(&mut self, gl: &glow::Context) -> Result<(), String> {
        self.unoccluded(gl)?;
        self.types(gl)?;
        self.reflection(gl)?;
        for (id, value) in &NEUTRAL {
            self.neutral(gl, *id, value)?;
        }
        for kind in [
            program::Kind::Plane,
            program::Kind::Array,
            program::Kind::Volume,
        ] {
            self.blank(gl, target(kind))?;
        }
        Ok(())
    }

    /// The buffers a screen-wide pass reads, by the name the package knows each by. One nothing here
    /// fills is the flat stand-in, which is what leaves its term out rather than crashing the draw.
    pub fn engine(&mut self, gl: &glow::Context, id: u32) -> Result<glow::Texture, String> {
        if let Some(at) = GBUFFER.iter().position(|held| *held == id) {
            return self
                .color
                .get(at)
                .copied()
                .ok_or_else(|| format!("the G-buffer has no channel {at}"));
        }
        if let Some(held) = self.supplied(program::Kind::Plane, id) {
            return Ok(held);
        }
        // Before the neutral below, which is what water reads where nothing drew its chain.
        if id == REFLECTION_MAP
            && let Some(held) = self.watered.as_ref()
        {
            return Ok(held.merged.1);
        }
        // Before it as well, and the same shape: opaque white until a sheet has been drawn from the
        // sun's own side, and again from the frame a weather stops stating one.
        if id == CLOUD_SHADOW
            && self.clouding
            && let Some([_, (_, map)]) = self.clouded
        {
            return Ok(map);
        }
        if let Some(held) = self.neutrals.get(&id) {
            return Ok(*held);
        }
        Ok(match id {
            DEPTH | DEPTH_PLANE => self.depth.ok_or("no depth buffer")?,
            NORMAL_PLANE => self
                .color
                .get(NORMAL_CHANNEL)
                .copied()
                .ok_or("the G-buffer has no normal channel")?,
            VIEW_POSITION | WATER_VIEW_POSITION => self.position.ok_or("no view position")?.1,
            LIGHT_DIFFUSE => self.light.ok_or("no light buffer")?.1[0],
            LIGHT_SPECULAR => self.light.ok_or("no light buffer")?.1[1],
            FINAL_COLOR | INPUT | REFRACTION => self.resolved.ok_or("no resolved frame")?,
            DEPTH_NORMAL_Z => self.scaled.ok_or("no scaled depth")?.1,
            GATHER_DEPTH => self.gathered.ok_or("no gathered depth")?.1[0],
            GATHER_NORMAL_Z => self.gathered.ok_or("no gathered depth")?.1[1],
            OCCLUSION if self.occluding => self.occluded.ok_or("no occlusion")?.1,
            SHADOW_DEPTH => self.shadow.ok_or("no shadow map")?.1,
            SUBSURFACE_KERNEL => self.subsurface(gl)?,
            SHADOW_MASK if self.shadowing => self.mask.ok_or("no shadow mask")?.1,
            // White rather than the flat grey every other unfilled sampler answers with: grey here
            // is half the frame in shadow, which is a plausible-looking wrong answer. The last of
            // them is squared and taken from one, where grey would throw three quarters of the
            // glare away and white all of it.
            SHADOW_MASK | OCCLUSION | ATTENUATION | GLARE_GEOMETRY => self.unoccluded(gl)?,
            _ => self.stand_in(gl)?,
        })
    }

    /// The uniform blocks one draw reads, filled and bound.
    pub fn bind(
        &mut self,
        gl: &glow::Context,
        program: glow::Program,
        held: &program::Program,
        scene: &program::Scene,
        instances: &[program::Instance],
    ) -> Result<(), String> {
        for (at, buffer) in held.buffers.iter().enumerate() {
            let Some((block, size)) = block(gl, program, &buffer.name) else {
                continue;
            };
            unsafe {
                let mut data = buffer.fill(scene, held.pass, instances);
                data.resize(size.max(16), 0);
                while self.blocks.len() <= at {
                    self.blocks.push((gl.create_buffer()?, Vec::new()));
                }
                let held = self.blocks[at].0;
                gl.bind_buffer(glow::UNIFORM_BUFFER, Some(held));
                if self.blocks[at].1 != data {
                    gl.buffer_data_u8_slice(glow::UNIFORM_BUFFER, &data, glow::DYNAMIC_DRAW);
                    self.blocks[at].1 = data;
                }
                gl.bind_buffer_base(glow::UNIFORM_BUFFER, at as u32, Some(held));
                if !pointed(program, block, at as u32) {
                    gl.uniform_block_binding(program, block, at as u32);
                }
            }
        }
        Ok(())
    }

    /// One pass of the graph drawn over a framebuffer of its own, reading whatever the passes before
    /// it left behind.
    fn pass(
        &mut self,
        gl: &glow::Context,
        at: usize,
        held: &std::sync::Arc<program::Program>,
        into: glow::Framebuffer,
        scene: &program::Scene,
        over: Over,
    ) -> Result<(), String> {
        let size = match over {
            Over::Fraction => self.fraction(),
            Over::Clouding(held) => held.size,
            Over::Exposing(size, _)
            | Over::Sized(size)
            | Over::Glaring(size, _)
            | Over::Blurring(size, _)
            | Over::Reflecting(size, _, _) => size,
            _ => self.size,
        };
        let program = match self.resolvers.get(&at) {
            Some(linked) if std::sync::Arc::ptr_eq(&linked.source, held) => linked.program,
            _ => {
                let built = build_pair(gl, &held.vertex, &held.fragment)?;
                if let Some(stale) = self.resolvers.insert(
                    at,
                    Linked {
                        source: held.clone(),
                        program: built,
                    },
                ) {
                    graveyard()
                        .lock()
                        .unwrap()
                        .push(Dead::Program(stale.program));
                }
                built
            }
        };
        let layout = match over {
            Over::Clouding(held) => held.layout,
            Over::Starring(held) => held.layout,
            Over::Blurring(..) | Over::Reflecting(..) => self.sampled(gl)?,
            Over::Volume => {
                let held = match self.volume {
                    Some(held) => held,
                    None => {
                        let held = upload_volume(gl)?;
                        self.volume = Some(held);
                        held
                    }
                };
                held.0
            }
            _ => self.screen(gl)?,
        };
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(into));
            let written: Vec<u32> = (0..held.targets.len().max(1))
                .map(|slot| glow::COLOR_ATTACHMENT0 + slot as u32)
                .collect();
            gl.draw_buffers(&written);
            gl.viewport(0, 0, size.0, size.1);
            gl.use_program(Some(program));
            gl.color_mask(true, true, true, true);
        }
        self.bind(gl, program, held, scene, &[])?;
        self.stand_ins(gl)?;
        let mut unit = 0;
        let mut gathering = None;
        for texture in &held.textures {
            let bound = match texture.kind {
                program::Kind::Plane => match over {
                    Over::Softening(held) if texture.id == GBUFFER[3] => held,
                    Over::Reading(held) if texture.id == INPUT => held,
                    Over::Exposing(_, reads) => match texture.name.as_str() {
                        program::POST_INPUT => reads.input,
                        program::POST_MEASURE => reads.measure,
                        program::POST_ADAPTED => reads.adapted,
                        _ => None,
                    }
                    .ok_or_else(|| {
                        format!("the exposure chain reads {}, which nothing fills", texture.name)
                    })?,
                    Over::Scattering(held) if texture.id == LIGHT_DIFFUSE => held,
                    // The transparent packing carries no motion, so its fourth target holds what
                    // the opaque one puts in its fifth and a pass reading the one it dropped reads
                    // the stand-in the game binds there.
                    Over::Sheering(reads) => {
                        match GBUFFER.iter().position(|held| *held == texture.id) {
                            Some(3) => self.neutral(gl, GBUFFER[3], &SHEER_ABSENT)?,
                            Some(at) => reads.color[at.min(SHEER - 1)],
                            None => match texture.id {
                                DEPTH | DEPTH_PLANE => reads.depth,
                                NORMAL_PLANE => reads.color[NORMAL_CHANNEL],
                                VIEW_POSITION => reads.position,
                                _ => self.engine(gl, texture.id)?,
                            },
                        }
                    }
                    Over::Blurring(_, held) if texture.id == INPUT => held,
                    Over::Glaring(_, held) => match texture.name.as_str() {
                        program::POST_INPUT => held.input,
                        program::POST_MERGE => held.merge.ok_or("no glare to merge")?,
                        _ => self.engine(gl, texture.id)?,
                    },
                    Over::Mooning(held) if texture.name == program::SKY_SAMPLER => held.sky,
                    Over::Fogging(reads) => match texture.name.as_str() {
                        program::FOG_DEPTH => reads.depth,
                        program::SKY_SAMPLER => reads.sky,
                        program::FOG_LUT => reads.table,
                        _ => self.engine(gl, texture.id)?,
                    },
                    // Two of the names the chain's files use mean different things to different
                    // members: what the mask calls the blurred reflection is the normal the member
                    // before it wrote, and what the march calls the first two channels of the
                    // G-buffer are that normal and the mask itself.
                    Over::Reflecting(_, member, reads) => match (member, texture.name.as_str()) {
                        (_, program::REFLECTION_DEPTH) => reads.depth,
                        (_, program::REFLECTION_FRAME) => reads.frame,
                        (Member::Mask, program::REFLECTION_BLURRED) => reads.normal,
                        (_, program::REFLECTION_BLURRED) => reads.blurred,
                        (Member::March | Member::Distort, program::REFLECTION_PLANE) => {
                            reads.normal
                        }
                        (Member::March, program::REFLECTION_MASKED) => reads.mask,
                        (_, program::REFLECTION_PLANE) => self.engine(gl, GBUFFER[0])?,
                        _ => self.engine(gl, texture.id)?,
                    },
                    // Both meshes read one sampler under the same name, so which sheet is bound is
                    // the draw's to say rather than the resource id's.
                    Over::Clouding(held) => held.texture,
                    Over::Starring(held) => match texture.name.as_str() {
                        "sSampler0" => held.color,
                        "sSampler1" => held.band,
                        "sSampler2" => held.twinkle,
                        program::SKY_SAMPLER => held.sky,
                        _ => self.engine(gl, texture.id)?,
                    },
                    _ => self.engine(gl, texture.id)?,
                },
                kind => self.absent(gl, kind, texture.id)?,
            };
            bind(
                gl,
                program,
                &texture.name,
                unit,
                bound,
                target(texture.kind),
                0.0,
            );
            if texture.name.ends_with(GATHER_SAMPLER) {
                gathering = Some(unit);
            }
            unit += 1;
        }
        for structured in &held.structured {
            let bound = match structured.name.as_str() {
                TYPES => self.types(gl)?,
                _ => self.stand_in(gl)?,
            };
            sampler(gl, program, &structured.name, unit, bound);
            unit += 1;
        }
        let gathering = match gathering {
            Some(unit) => Some((unit, self.gathering(gl)?)),
            None => None,
        };
        unsafe {
            if let Some((unit, held)) = gathering {
                gl.bind_sampler(unit, Some(held));
            }
            // What a pass reading a square of its own source steps between the corners of it.
            if let Some(location) = gl.get_uniform_location(program, "u_texel") {
                gl.uniform_2_f32(Some(&location), 1.0 / size.0 as f32, 1.0 / size.1 as f32);
            }
            if let Over::Mooning(held) = over
                && let Some(location) = gl.get_uniform_location(program, "u_disc")
            {
                let [x, y, wide, tall] = held.disc.to_array();
                gl.uniform_4_f32(Some(&location), x, y, wide, tall);
            }
            gl.bind_vertex_array(Some(layout));
            match over {
                Over::Volume => gl.draw_elements(
                    glow::TRIANGLES,
                    VOLUME_FACES.len() as i32,
                    glow::UNSIGNED_SHORT,
                    0,
                ),
                Over::Clouding(held) => {
                    gl.draw_elements(glow::TRIANGLE_STRIP, held.indices, glow::UNSIGNED_SHORT, 0)
                }
                Over::Starring(held) => {
                    gl.draw_elements(glow::TRIANGLES, held.indices, glow::UNSIGNED_SHORT, 0)
                }
                _ => gl.draw_arrays(glow::TRIANGLES, 0, 3),
            }
            gl.bind_vertex_array(None);
            // A sampler stands on its unit until something takes it off, so a later pass reading a
            // texture there would read it through this one.
            if let Some((unit, _)) = gathering {
                gl.bind_sampler(unit, None);
            }
        }
        Ok(())
    }

    /// One more draw of the pass just run, with only the light it carries written again. Every lamp
    /// of a kind reads the same program over the same volume, so what that pass bound - its
    /// framebuffer, its textures, the rest of its buffers - stands as it left it.
    fn again(&mut self, gl: &glow::Context, held: &program::Program, scene: &program::Scene) {
        let Some((layout, _, _)) = self.volume else {
            return;
        };
        for (at, buffer) in held.buffers.iter().enumerate() {
            if buffer.name != program::LIGHT {
                continue;
            }
            let Some((block, standing)) = self.blocks.get_mut(at).filter(|held| !held.1.is_empty())
            else {
                continue;
            };
            let mut data = buffer.fill(scene, held.pass, &[]);
            data.resize(standing.len(), 0);
            if *standing == data {
                continue;
            }
            unsafe {
                gl.bind_buffer(glow::UNIFORM_BUFFER, Some(*block));
                gl.buffer_sub_data_u8_slice(glow::UNIFORM_BUFFER, 0, &data);
            }
            *standing = data;
        }
        unsafe {
            gl.bind_vertex_array(Some(layout));
            gl.draw_elements(
                glow::TRIANGLES,
                VOLUME_FACES.len() as i32,
                glow::UNSIGNED_SHORT,
                0,
            );
            gl.bind_vertex_array(None);
        }
    }

    /// The graph past the G-buffer: the view position off the depth, the light off both, and the
    /// frame off the light. Every lamp adds to the same two buffers, so they start at nothing and
    /// each pass adds to what the one before it left.
    pub fn resolve(
        &mut self,
        gl: &glow::Context,
        lighting: &Lighting,
        scene: &program::Scene,
        lamps: &[program::Lamp],
    ) -> Result<(), String> {
        // The frame starts again here, so what ran over the last one is forgotten - except the
        // sun's own pass and the occlusion, which run ahead of this one and would be forgotten
        // before they were read.
        self.drawn = Drawn {
            shadow: self.shadowing,
            occlusion: self.occluding,
            cloud_shadow: self.clouding,
            ..Drawn::default()
        };
        let (position, _) = self.position.ok_or("no view position")?;
        let (light, _) = self.light.ok_or("no light buffer")?;
        let (lit, _) = self.lit.ok_or("no lit frame")?;
        self.toned = false;
        self.covered = false;
        self.reflect(gl, &scene.ambient)?;
        unsafe {
            // A screen-wide pass covers every pixel and reads the depth rather than testing against
            // it, so nothing here is depth tested and nothing writes depth.
            gl.disable(glow::DEPTH_TEST);
            gl.depth_mask(false);
            gl.disable(glow::CULL_FACE);
            gl.disable(glow::BLEND);
        }
        self.pass(gl, 0, &lighting.position, position, scene, Over::Screen)?;
        // A strand is softened where it stands rather than where it is lit, so this runs before any
        // light reads the channel. It walks its neighbours to answer for one pixel, which is why it
        // reads a copy of the channel and writes the channel itself, and it discards where the
        // surface states no fur, leaving those pixels as the G-buffer left them.
        if let Some(fur) = &lighting.fur
            && let Some((into, softened)) = self.fur
        {
            unsafe {
                gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(into));
                gl.read_buffer(glow::COLOR_ATTACHMENT0);
                gl.active_texture(glow::TEXTURE0);
                gl.bind_texture(glow::TEXTURE_2D, Some(softened));
                gl.copy_tex_sub_image_2d(glow::TEXTURE_2D, 0, 0, 0, 0, 0, self.size.0, self.size.1);
                gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
            }
            let held = fur.clone();
            self.pass(gl, 5, &held, into, scene, Over::Softening(softened))?;
        }

        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(light));
            gl.draw_buffers(&[glow::COLOR_ATTACHMENT0, glow::COLOR_ATTACHMENT1]);
            gl.clear_color(0.0, 0.0, 0.0, 0.0);
            gl.clear(glow::COLOR_BUFFER_BIT);
            gl.enable(glow::BLEND);
            gl.blend_func(glow::ONE, glow::ONE);
        }
        self.pass(gl, 1, &lighting.directional, light, scene, Over::Screen)?;
        // One face of the volume, not both. The pass adds what it computes to the buffer, so a box
        // shaded front and back would light every pixel it covers twice over. The far face is the
        // one kept, since it still covers the frame when the camera stands inside the light.
        unsafe {
            gl.enable(glow::CULL_FACE);
            gl.cull_face(glow::FRONT);
            gl.front_face(glow::CCW);
        }
        // Each kind is one pass over a volume of its own, linked once at a slot of its own. A kind
        // whose package has not arrived draws through the point one, which reads neither its length
        // nor its area: a lit box rather than nothing. Gathered by kind so a run of lamps goes
        // through one program: the first sets the pass up whole and the rest write the one buffer
        // that differs and draw. Every lamp adds to the same buffer, so the order costs nothing.
        let mut sorted = lamps.to_vec();
        sorted.sort_by_key(|lamp| (lamp.kind as u8, lamp.falloff));
        let mut held = scene.clone();
        let mut standing = None;
        for lamp in &sorted {
            held.lamp = *lamp;
            let (kind, programs) = match lamp.kind {
                program::LampKind::Spot => (1, lighting.spot.as_ref()),
                program::LampKind::Line => (2, lighting.line.as_ref()),
                program::LampKind::Plane => (3, lighting.plane.as_ref()),
                program::LampKind::Point => (0, None),
            };
            let (kind, programs) = match programs {
                Some(held) => (kind, held),
                None => (0, &lighting.point),
            };
            let falloff = lamp.falloff.min(programs.len() - 1);
            let slot = LAMP + kind * program::ATTENUATION.len() + falloff;
            let program = &programs[falloff];
            match standing == Some(slot) {
                true => self.again(gl, program, &held),
                false => {
                    self.pass(gl, slot, program, light, &held, Over::Volume)?;
                    standing = Some(slot);
                }
            }
        }
        unsafe {
            gl.disable(glow::CULL_FACE);
            gl.disable(glow::BLEND);
        };
        // Skin scatters the light that fell on it before the composite reads it. The pass walks the
        // diffuse channel around a pixel and writes that same channel, so what it reads is a copy
        // taken here; where the type table states no width it hands the copy straight back, leaving
        // the pixel as the lamps left it.
        // The copy is made the first time something scatters rather than with the rest of the
        // frame: a zone has no skin in it and would carry a full-size float buffer for nothing.
        if lighting.subsurface.is_some() && self.scattered.is_none() {
            self.scattered = Some(plane(
                gl,
                self.size,
                glow::RGBA16F,
                glow::RGBA,
                glow::FLOAT,
            )?);
        }
        if let Some(held) = &lighting.subsurface
            && let Some(scattered) = self.scattered
        {
            unsafe {
                gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(light));
                gl.read_buffer(glow::COLOR_ATTACHMENT0);
                gl.active_texture(glow::TEXTURE0);
                gl.bind_texture(glow::TEXTURE_2D, Some(scattered));
                gl.copy_tex_sub_image_2d(glow::TEXTURE_2D, 0, 0, 0, 0, 0, self.size.0, self.size.1);
                gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(light));
                gl.draw_buffers(&[glow::COLOR_ATTACHMENT0]);
            }
            let held = held.clone();
            self.pass(gl, SCATTER, &held, light, scene, Over::Scattering(scattered))?;
        }
        self.pass(gl, 4, &lighting.composite, lit, scene, Over::Screen)
    }
}

impl Drop for Buffers {
    fn drop(&mut self) {
        let mut dead = graveyard().lock().unwrap();
        dead.extend(self.color.drain(..).map(Dead::Texture));
        dead.extend(
            [
                self.types.take(),
                self.unoccluded.take(),
                self.reflection.take(),
                self.resolved.take(),
                self.depth.take(),
                self.haze.take().map(|(held, _)| held),
            ]
            .into_iter()
            .flatten()
            .chain(std::mem::take(&mut self.blanks).into_values())
            .chain(std::mem::take(&mut self.neutrals).into_values())
            .chain(
                std::mem::take(&mut self.arrays)
                    .into_values()
                    .chain(std::mem::take(&mut self.stacked).into_values())
                    .map(|(_, held)| held),
            )
            .map(Dead::Texture),
        );
        dead.extend(self.frames.drain(..).map(Dead::Frame));
        dead.extend(self.bare.take().map(Dead::Frame));
        dead.extend(self.cutoff.take().map(Dead::Texture));
        if let Some((frame, held)) = self.shadow.take() {
            dead.push(Dead::Frame(frame));
            dead.push(Dead::Texture(held));
        }
        if let Some((frame, held)) = self.mask.take() {
            dead.push(Dead::Frame(frame));
            dead.push(Dead::Texture(held));
        }
        dead.extend(self.scattered.take().map(Dead::Texture));
        dead.extend(self.gather.take().map(Dead::Sampler));
        if let Some(held) = self.sheer.take() {
            dead.push(Dead::Frame(held.frame));
            dead.extend(held.color.into_iter().map(Dead::Texture));
            dead.push(Dead::Texture(held.depth));
            dead.push(Dead::Frame(held.position.0));
            dead.push(Dead::Texture(held.position.1));
        }
        for (frame, textures) in [
            self.position
                .take()
                .map(|(frame, held)| (frame, vec![held])),
            self.light
                .take()
                .map(|(frame, held)| (frame, held.to_vec())),
            self.lit.take().map(|(frame, held)| (frame, vec![held])),
            self.fur.take().map(|(frame, held)| (frame, vec![held])),
            self.smoothed
                .take()
                .map(|(frame, held)| (frame, vec![held])),
            self.scaled.take().map(|(frame, held)| (frame, vec![held])),
            self.gathered
                .take()
                .map(|(frame, held)| (frame, held.to_vec())),
            self.occluded
                .take()
                .map(|(frame, held)| (frame, vec![held])),
            self.overhead
                .take()
                .map(|(frame, held)| (frame, vec![held])),
        ]
        .into_iter()
        .flatten()
        .chain(
            self.glared
                .take()
                .into_iter()
                .flatten()
                .chain(self.reached.take().into_iter().flatten())
                .chain(self.sourced.take())
                .map(|(frame, held)| (frame, vec![held])),
        )
        {
            dead.push(Dead::Frame(frame));
            dead.extend(textures.into_iter().map(Dead::Texture));
        }
        for (layout, held) in [self.screen.take(), self.sampled.take()].into_iter().flatten() {
            dead.push(Dead::Layout(layout));
            dead.push(Dead::Buffer(held));
        }
        if let Some((layout, held, faces)) = self.volume.take() {
            dead.push(Dead::Layout(layout));
            dead.push(Dead::Buffer(held));
            dead.push(Dead::Buffer(faces));
        }
        for (layout, held, faces, _) in self.strips.iter_mut().filter_map(Option::take) {
            dead.push(Dead::Layout(layout));
            dead.push(Dead::Buffer(held));
            dead.push(Dead::Buffer(faces));
        }
        dead.extend(
            self.sheets
                .iter_mut()
                .filter_map(Option::take)
                .map(|(_, held)| Dead::Texture(held)),
        );
        dead.extend(self.blocks.drain(..).map(|(held, _)| Dead::Buffer(held)));
        dead.extend(self.present.take().map(Dead::Program));
        dead.extend(
            std::mem::take(&mut self.resolvers)
                .into_values()
                .map(|held| Dead::Program(held.program)),
        );
    }
}

/// One linked pair, kept against the reading it was built from so a change rebuilds it rather than a
/// stale program drawing on. Held by identity rather than by source: a zone links thousands of times
/// a frame, and the two shaders behind one of them run to tens of kilobytes.
pub fn link<K: Ord>(
    gl: &glow::Context,
    into: &mut BTreeMap<K, Linked>,
    key: K,
    held: &std::sync::Arc<program::Program>,
) -> Result<glow::Program, String> {
    if let Some(linked) = into.get(&key)
        && std::sync::Arc::ptr_eq(&linked.source, held)
    {
        return Ok(linked.program);
    }
    let built = build_pair(gl, &held.vertex, &held.fragment)?;
    if let Some(stale) = into.insert(
        key,
        Linked {
            source: held.clone(),
            program: built,
        },
    ) {
        graveyard()
            .lock()
            .unwrap()
            .push(Dead::Program(stale.program));
    }
    Ok(built)
}

/// The draw buffers one reading writes, which is where each of its outputs lands. A list can only
/// point the nth output at the nth attachment, so a channel the reading declares and skips is named
/// as none rather than left holding whatever the draw would have put there.
pub fn written(held: &program::Program) -> Vec<u32> {
    match held.targets.is_empty() {
        true => vec![glow::COLOR_ATTACHMENT0],
        false => held
            .targets
            .iter()
            .enumerate()
            .map(|(at, target)| match held.outputs.contains(target) {
                true => glow::COLOR_ATTACHMENT0 + at as u32,
                false => glow::NONE,
            })
            .collect(),
    }
}

/// The target a texture a sampler of this kind reads has to be bound at.
pub fn target(kind: program::Kind) -> u32 {
    match kind {
        program::Kind::Plane => glow::TEXTURE_2D,
        program::Kind::Array => glow::TEXTURE_2D_ARRAY,
        program::Kind::Volume => glow::TEXTURE_3D,
        program::Kind::Cube => glow::TEXTURE_CUBE_MAP,
    }
}

/// Point sampling and no wrap, which is what every buffer of the graph is read with: a shader
/// addresses texel centers and works out for itself what lies between them.
pub fn point(gl: &glow::Context) {
    for (name, value) in [
        (glow::TEXTURE_MIN_FILTER, glow::NEAREST),
        (glow::TEXTURE_MAG_FILTER, glow::NEAREST),
        (glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE),
        (glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE),
    ] {
        unsafe { gl.tex_parameter_i32(glow::TEXTURE_2D, name, value as i32) };
    }
}

/// For the buffers a pass reads between texels rather than at their centers. Leaves the wrap modes
/// alone, so a texture takes this after `point`.
fn smooth(gl: &glow::Context, texture: glow::Texture) {
    unsafe {
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        for name in [glow::TEXTURE_MIN_FILTER, glow::TEXTURE_MAG_FILTER] {
            gl.tex_parameter_i32(glow::TEXTURE_2D, name, glow::LINEAR as i32);
        }
    }
}

/// The whole of one buffer onto the whole of another, filtered where the two differ in size.
fn blit(
    gl: &glow::Context,
    from: glow::Framebuffer,
    into: glow::Framebuffer,
    (wide, tall): (i32, i32),
    (across, down): (i32, i32),
) {
    unsafe {
        gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(from));
        gl.read_buffer(glow::COLOR_ATTACHMENT0);
        gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(into));
        gl.draw_buffers(&[glow::COLOR_ATTACHMENT0]);
        gl.blit_framebuffer(
            0,
            0,
            wide,
            tall,
            0,
            0,
            across,
            down,
            glow::COLOR_BUFFER_BIT,
            glow::LINEAR,
        );
        gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
    }
}

/// One buffer of the graph, the size of what is being drawn into.
fn plane(
    gl: &glow::Context,
    size: (i32, i32),
    internal: u32,
    format: u32,
    kind: u32,
) -> Result<glow::Texture, String> {
    unsafe {
        let texture = gl.create_texture()?;
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            internal as i32,
            size.0,
            size.1,
            0,
            format,
            kind,
            glow::PixelUnpackData::Slice(None),
        );
        point(gl);
        Ok(texture)
    }
}

/// A buffer of the graph with a chain of levels under it and a framebuffer over each, since a pass
/// of the reflection chain draws into one level while it reads another.
fn pyramid(
    gl: &glow::Context,
    size: (i32, i32),
    levels: i32,
    internal: u32,
    filter: u32,
) -> Result<(glow::Texture, Vec<glow::Framebuffer>), String> {
    let levels = levels.max(1);
    unsafe {
        let texture = gl.create_texture()?;
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        gl.tex_storage_2d(glow::TEXTURE_2D, levels, internal, size.0, size.1);
        for (name, value) in [
            (glow::TEXTURE_MAG_FILTER, filter),
            (
                glow::TEXTURE_MIN_FILTER,
                match filter {
                    glow::LINEAR => glow::LINEAR_MIPMAP_LINEAR,
                    _ => glow::NEAREST_MIPMAP_NEAREST,
                },
            ),
            (glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE),
            (glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE),
        ] {
            gl.tex_parameter_i32(glow::TEXTURE_2D, name, value as i32);
        }
        let mut frames = Vec::new();
        for level in 0..levels {
            let held = gl.create_framebuffer()?;
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(held));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(texture),
                level,
            );
            let status = gl.check_framebuffer_status(glow::FRAMEBUFFER);
            if status != glow::FRAMEBUFFER_COMPLETE {
                return Err(format!("a level of the graph would not complete: {status:#x}"));
            }
            frames.push(held);
        }
        Ok((texture, frames))
    }
}

/// A framebuffer over the textures given, in the order a pass writes them.
fn frame_of(
    gl: &glow::Context,
    color: &[glow::Texture],
    depth: Option<glow::Texture>,
) -> Result<glow::Framebuffer, String> {
    unsafe {
        let held = gl.create_framebuffer()?;
        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(held));
        for (at, texture) in color.iter().enumerate() {
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0 + at as u32,
                glow::TEXTURE_2D,
                Some(*texture),
                0,
            );
        }
        gl.framebuffer_texture_2d(
            glow::FRAMEBUFFER,
            glow::DEPTH_ATTACHMENT,
            glow::TEXTURE_2D,
            depth,
            0,
        );
        let status = gl.check_framebuffer_status(glow::FRAMEBUFFER);
        match status == glow::FRAMEBUFFER_COMPLETE {
            true => Ok(held),
            false => Err(format!(
                "a buffer of the graph would not complete: {status:#x}"
            )),
        }
    }
}

/// A buffer of dwords as the texture a shader reads it through, one dword to a texel.
pub fn dwords(gl: &glow::Context, values: &[u32]) -> Result<glow::Texture, String> {
    unsafe {
        let texture = gl.create_texture()?;
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            glow::R32UI as i32,
            program::ROW as i32,
            (values.len() / program::ROW) as i32,
            0,
            glow::RED_INTEGER,
            glow::UNSIGNED_INT,
            glow::PixelUnpackData::Slice(Some(bytemuck::cast_slice(values))),
        );
        point(gl);
        Ok(texture)
    }
}

/// Binds a texture to a unit and points the shader's own sampler at it.
pub fn sampler(
    gl: &glow::Context,
    program: glow::Program,
    name: &str,
    unit: u32,
    texture: glow::Texture,
) {
    bind(gl, program, name, unit, texture, glow::TEXTURE_2D, 0.0);
}

/// The driver's own anisotropy ceiling, asked for once. `EXT_texture_filter_anisotropic` is an
/// extension on every backend this runs on, WebGL2 included, and the enum is only meaningful once
/// the driver has answered whether it exists.
static MAX_ANISOTROPY: std::sync::OnceLock<f32> = std::sync::OnceLock::new();

pub(super) fn max_anisotropy(gl: &glow::Context) -> f32 {
    *MAX_ANISOTROPY.get_or_init(|| {
        match gl.supported_extensions().contains("EXT_texture_filter_anisotropic") {
            true => unsafe { gl.get_parameter_f32(glow::MAX_TEXTURE_MAX_ANISOTROPY_EXT) },
            false => 0.0,
        }
    })
}

/// The same over whichever target the shader declared the sampler against. A draw only validates if
/// the texture bound to a unit is of the sampler's own type, so the target follows the declaration.
///
/// `aniso` is the anisotropy a material's own sampler asks for; it degrades to whatever the driver
/// actually offers, and to nothing at all where the extension is absent.
pub fn bind(
    gl: &glow::Context,
    program: glow::Program,
    name: &str,
    unit: u32,
    texture: glow::Texture,
    target: u32,
    aniso: f32,
) {
    unsafe {
        gl.active_texture(glow::TEXTURE0 + unit);
        gl.bind_texture(target, Some(texture));
        // Screen-space and G-buffer planes bind through here too, every draw; skip the call
        // entirely where no material sampler asked for anisotropy, rather than setting 1.0 (a no-op
        // value) on textures that never wanted the state touched.
        let ceiling = max_anisotropy(gl);
        if aniso > 0.0 && ceiling > 0.0 {
            gl.tex_parameter_f32(
                target,
                glow::TEXTURE_MAX_ANISOTROPY_EXT,
                aniso.clamp(1.0, ceiling),
            );
        }
    }
    point_sampler(gl, program, name, unit);
}

/// The geometry the screen-wide passes and a light's volume are drawn from, in one array each so a
/// pass can draw either without rebinding anything else.
fn upload_screen(
    gl: &glow::Context,
    vertices: &[f32; 12],
) -> Result<(glow::VertexArray, glow::Buffer), String> {
    unsafe {
        let layout = gl.create_vertex_array()?;
        gl.bind_vertex_array(Some(layout));
        let held = gl.create_buffer()?;
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(held));
        gl.buffer_data_u8_slice(
            glow::ARRAY_BUFFER,
            bytemuck::cast_slice(vertices),
            glow::STATIC_DRAW,
        );
        gl.enable_vertex_attrib_array(0);
        gl.vertex_attrib_pointer_f32(0, 4, glow::FLOAT, false, 16, 0);
        gl.bind_vertex_array(None);
        gl.bind_buffer(glow::ARRAY_BUFFER, None);
        Ok((layout, held))
    }
}

/// One cloud mesh on the card, with its own vertex array: a place and a coordinate a vertex, which
/// is what the cloud package's own signature asks for and nothing else.
fn upload_strip(
    gl: &glow::Context,
    vertices: &[f32],
    indices: &[u16],
) -> Result<(glow::VertexArray, glow::Buffer, glow::Buffer, i32), String> {
    unsafe {
        let layout = gl.create_vertex_array()?;
        gl.bind_vertex_array(Some(layout));
        let held = gl.create_buffer()?;
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(held));
        gl.buffer_data_u8_slice(
            glow::ARRAY_BUFFER,
            bytemuck::cast_slice(vertices),
            glow::STATIC_DRAW,
        );
        for (location, lanes, offset) in [(0, 4, 0), (1, 2, 16)] {
            gl.enable_vertex_attrib_array(location);
            gl.vertex_attrib_pointer_f32(location, lanes, glow::FLOAT, false, CLOUD_STRIDE, offset);
        }
        let faces = gl.create_buffer()?;
        gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(faces));
        gl.buffer_data_u8_slice(
            glow::ELEMENT_ARRAY_BUFFER,
            bytemuck::cast_slice(indices),
            glow::STATIC_DRAW,
        );
        gl.bind_vertex_array(None);
        gl.bind_buffer(glow::ARRAY_BUFFER, None);
        Ok((layout, held, faces, indices.len() as i32))
    }
}

fn upload_dome(
    gl: &glow::Context,
    vertices: &[f32],
    indices: &[u16],
) -> Result<(glow::VertexArray, glow::Buffer, glow::Buffer, i32), String> {
    unsafe {
        let layout = gl.create_vertex_array()?;
        gl.bind_vertex_array(Some(layout));
        let held = gl.create_buffer()?;
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(held));
        gl.buffer_data_u8_slice(
            glow::ARRAY_BUFFER,
            bytemuck::cast_slice(vertices),
            glow::STATIC_DRAW,
        );
        for (location, lanes, offset) in [(0, 3, 0), (1, 4, 12), (2, 4, 28)] {
            gl.enable_vertex_attrib_array(location);
            gl.vertex_attrib_pointer_f32(location, lanes, glow::FLOAT, false, STAR_STRIDE, offset);
        }
        let faces = gl.create_buffer()?;
        gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(faces));
        gl.buffer_data_u8_slice(
            glow::ELEMENT_ARRAY_BUFFER,
            bytemuck::cast_slice(indices),
            glow::STATIC_DRAW,
        );
        gl.bind_vertex_array(None);
        gl.bind_buffer(glow::ARRAY_BUFFER, None);
        Ok((layout, held, faces, indices.len() as i32))
    }
}

fn upload_volume(
    gl: &glow::Context,
) -> Result<(glow::VertexArray, glow::Buffer, glow::Buffer), String> {
    unsafe {
        let layout = gl.create_vertex_array()?;
        gl.bind_vertex_array(Some(layout));
        let held = gl.create_buffer()?;
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(held));
        gl.buffer_data_u8_slice(
            glow::ARRAY_BUFFER,
            bytemuck::cast_slice(&VOLUME),
            glow::STATIC_DRAW,
        );
        gl.enable_vertex_attrib_array(0);
        gl.vertex_attrib_pointer_f32(0, 4, glow::FLOAT, false, 16, 0);
        let faces = gl.create_buffer()?;
        gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(faces));
        gl.buffer_data_u8_slice(
            glow::ELEMENT_ARRAY_BUFFER,
            bytemuck::cast_slice(&VOLUME_FACES),
            glow::STATIC_DRAW,
        );
        gl.bind_vertex_array(None);
        gl.bind_buffer(glow::ARRAY_BUFFER, None);
        Ok((layout, held, faces))
    }
}

/// A one-texel texture of the given target, answering with the same value everywhere. A cube states
/// each of its faces and the targets that take a depth state one slice.
fn flat(gl: &glow::Context, target: u32, value: &[u8; 4]) -> Result<glow::Texture, String> {
    unsafe {
        let texture = gl.create_texture()?;
        gl.bind_texture(target, Some(texture));
        match target {
            glow::TEXTURE_CUBE_MAP => {
                for face in 0..6 {
                    gl.tex_image_2d(
                        glow::TEXTURE_CUBE_MAP_POSITIVE_X + face,
                        0,
                        glow::RGBA as i32,
                        1,
                        1,
                        0,
                        glow::RGBA,
                        glow::UNSIGNED_BYTE,
                        glow::PixelUnpackData::Slice(Some(value)),
                    );
                }
            }
            glow::TEXTURE_2D_ARRAY | glow::TEXTURE_3D => gl.tex_image_3d(
                target,
                0,
                glow::RGBA as i32,
                1,
                1,
                1,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(value)),
            ),
            _ => gl.tex_image_2d(
                target,
                0,
                glow::RGBA as i32,
                1,
                1,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(value)),
            ),
        }
        for (name, held) in [
            (glow::TEXTURE_MIN_FILTER, glow::LINEAR),
            (glow::TEXTURE_MAG_FILTER, glow::LINEAR),
            (glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE),
            (glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE),
        ] {
            gl.tex_parameter_i32(target, name, held as i32);
        }
        Ok(texture)
    }
}

/// How many texture units the fragment stage may read, asked for once. WebGL2 guarantees only 16
/// where desktop GL usually offers 32, and the game's own character shaders read past that.
static MAX_TEXTURE_UNITS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

fn max_texture_units(gl: &glow::Context) -> usize {
    *MAX_TEXTURE_UNITS.get_or_init(|| unsafe {
        gl.get_parameter_i32(glow::MAX_TEXTURE_IMAGE_UNITS).max(0) as usize
    })
}

/// The samplers a fragment source declares, which is what the driver counts at link time. Counted
/// off the source rather than the pair's resource list, since only one of the two stages is being
/// weighed and a texture read through more than one sampler is declared once per sampler.
fn fragment_samplers(source: &str) -> usize {
    source
        .lines()
        .filter(|line| {
            let line = line.trim_start();
            line.starts_with("uniform ") && line.contains("sampler")
        })
        .count()
}

pub fn build_pair(
    gl: &glow::Context,
    vertex: &str,
    fragment: &str,
) -> Result<glow::Program, String> {
    // Answered before compiling rather than after linking: the driver's own message names the limit
    // but not the shader, and a doomed pair costs a compile of both stages to find out.
    let wanted = fragment_samplers(fragment);
    let ceiling = max_texture_units(gl);
    if ceiling > 0 && wanted > ceiling {
        return Err(format!(
            "the fragment shader reads {wanted} textures and this device offers {ceiling}"
        ));
    }
    unsafe {
        let program = gl.create_program()?;
        let mut built = Vec::new();
        for (stage, source) in [
            (glow::VERTEX_SHADER, vertex),
            (glow::FRAGMENT_SHADER, fragment),
        ] {
            let shader = gl.create_shader(stage)?;
            gl.shader_source(shader, source);
            gl.compile_shader(shader);
            if !gl.get_shader_compile_status(shader) {
                let why = gl.get_shader_info_log(shader);
                gl.delete_shader(shader);
                for shader in built {
                    gl.delete_shader(shader);
                }
                gl.delete_program(program);
                let name = match stage {
                    glow::VERTEX_SHADER => "vertex",
                    _ => "fragment",
                };
                // Some implementations reject a shader with no diagnostic at all.
                return Err(match why.trim().is_empty() {
                    true => format!("{name} shader would not compile"),
                    false => format!("{name}: {why}"),
                });
            }
            gl.attach_shader(program, shader);
            built.push(shader);
        }
        gl.link_program(program);
        for shader in built {
            gl.detach_shader(program, shader);
            gl.delete_shader(shader);
        }
        if !gl.get_program_link_status(program) {
            let why = gl.get_program_info_log(program);
            gl.delete_program(program);
            return Err(match why.trim().is_empty() {
                true => "program would not link".to_owned(),
                false => why,
            });
        }
        Ok(program)
    }
}

#[cfg(test)]
mod test {
    /// Only the fragment stage is weighed, and a texture read through two samplers is declared
    /// twice, so the count comes off the source rather than off the pair's resource list.
    #[test]
    fn a_fragment_source_states_how_many_textures_it_reads() {
        let source = "#version 300 es\n             precision highp float;\n             uniform sampler2D g_SamplerNormal;\n             uniform highp sampler2D g_SamplerIndex;\n             uniform usampler2D g_SamplerTable;\n             uniform samplerCube g_SamplerEnv;\n             uniform vec4 g_Params[4];\n             // uniform sampler2D commented_out;\n             void main() {}\n";
        assert_eq!(super::fragment_samplers(source), 4);
        assert_eq!(super::fragment_samplers("void main() {}"), 0);
    }

    use super::{BAND, ENGINE, SHEET, STAR_FACES, STAR_GRID, band, dome, sheet, strip};

    /// The id the file the shadow softening dithers by is bound under. Nothing in the table says
    /// which sampler an entry answers except this number, and the packages derive it from the name.
    #[test]
    fn the_shadow_dither_is_bound_under_the_name_its_shader_reads() {
        let (id, _, _) = ENGINE
            .into_iter()
            .find(|(_, path, _)| path.contains("-bn_64_"))
            .expect("the engine table names a blue noise file");
        assert_eq!(id, shaders::names::hash(b"tBlueNoise"));
    }

    /// Both meshes and both strips, against what the buffers a capture of the running game bound
    /// hold. The engine builds these rather than reading them out of a file, so the counts and the
    /// order are the whole statement of what they are.
    #[test]
    fn the_cloud_meshes_come_out_as_the_game_built_them() {
        let held = band();
        assert_eq!(held.len(), BAND.0 * BAND.1 * 6);
        // A unit circle a column at a time, with the last column back on the first.
        let point = |at: usize| &held[at * 6..at * 6 + 6];
        assert_eq!(point(0), [0.0, 0.0, -1.0, 1.0, 0.0, 0.0]);
        let seam = point(BAND.0 - 1);
        assert!(seam[0].abs() < 1e-6 && (seam[2] + 1.0).abs() < 1e-6);
        // The texture wraps three times around it and once from top to bottom.
        assert_eq!(seam[4], 3.0);
        assert_eq!(point(BAND.0 * (BAND.1 - 1))[1], -1.0);

        let held = sheet();
        assert_eq!(held.len(), SHEET * SHEET * 6);
        let point = |at: usize| &held[at * 6..at * 6 + 6];
        // A square whose points crowd toward the middle, bent down by the square of how far out
        // they stand, and read at a coordinate that runs the other way across.
        assert_eq!(point(0)[0], 1.0);
        assert_eq!(point(SHEET - 1)[0], -1.0);
        assert_eq!(point(SHEET / 2)[0], 0.0);
        assert_eq!(point(1)[0], 0.765_625);
        for at in [0, SHEET / 2, SHEET * SHEET - 1] {
            let held = point(at);
            assert!((held[1] + held[0] * held[0] + held[2] * held[2]).abs() < 1e-6);
            assert_eq!([held[4], held[5]], [-held[0], held[2]]);
        }

        // One strip apiece, turning at the ends rather than restarting.
        assert_eq!(strip(BAND.0, BAND.1).len(), 197);
        assert_eq!(strip(SHEET, SHEET).len(), 529);
        assert_eq!(strip(BAND.0, BAND.1)[..4], [0, 25, 1, 26]);
        assert_eq!(strip(SHEET, SHEET)[..4], [0, 17, 1, 18]);
        // The turn goes down a row at the column the band stopped on, which stands on the axis, so
        // the triangle that turns between them has no area.
        assert_eq!(strip(BAND.0, BAND.1)[48..52], [24, 49, 74, 48]);
    }

    /// The star dome against the capture's own vertex and index buffers: 1734 vertices over 9216
    /// indices, six self-contained panels, and the first vertex and a panel's own centre exactly as
    /// the capture held them.
    #[test]
    fn the_star_dome_comes_out_as_the_capture_held_it() {
        let (vertices, indices) = dome();
        assert_eq!(vertices.len(), STAR_FACES * STAR_GRID * STAR_GRID * 11);
        assert_eq!(indices.len(), STAR_FACES * (STAR_GRID - 1) * (STAR_GRID - 1) * 6);
        assert_eq!(vertices.len(), 1734 * 11);
        assert_eq!(indices.len(), 9216);

        let vertex = |at: usize| &vertices[at * 11..at * 11 + 11];
        let v0 = vertex(0);
        assert_eq!(v0[0..3], [0.577_148_44, 0.0, 0.816_406_25]);
        assert_eq!(v0[3..7], [0.0, 8.0, 0.0, 32.0]);
        assert_eq!(v0[7..9], [0.0, 4.5]);

        // A face's own centre, an axis-aligned unit vector.
        let mid = vertex(8 * STAR_GRID + 8);
        assert_eq!(mid[0..3], [1.0, 0.0, 0.0]);
        assert_eq!(mid[3..7], [4.0, 4.0, 16.0, 16.0]);

        // Each panel's own 289 vertices and 1536 indices, self-contained the way the capture held
        // them: no index of one panel ever names a vertex of another.
        for panel in 0..STAR_FACES {
            let held = &indices[panel * 1536..(panel + 1) * 1536];
            let lo = *held.iter().min().unwrap() as usize;
            let hi = *held.iter().max().unwrap() as usize;
            assert_eq!((lo, hi), (panel * 289, panel * 289 + 288));
        }
    }
}
