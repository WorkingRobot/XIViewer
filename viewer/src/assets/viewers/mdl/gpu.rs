//! The GL side of the model viewer.
//!
//! Everything here runs inside an [`egui_glow`] paint callback, which is the only place a
//! `glow::Context` is reachable: the context is neither `Send` nor `Sync` on wasm, so it cannot be
//! captured, and eframe's copy of it is not threaded down to a viewer. Uploads therefore happen on
//! the first frame that draws rather than when the file is decoded, and freeing happens in a
//! graveyard the next callback drains.

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::{Arc, Mutex};

use egui::TextureId;
use glow::HasContext;

use super::deferred::{self, Layered, Linked, TARGETS, TYPES, build_pair, dwords, sampler};
use super::grid::{Grid, Ground};
use super::material::{Family, Glass};
use super::super::avfx;
use super::{Table, Vertex, program};

pub use super::deferred::{
    Dead, Exposure, Glare, LIT, Lighting, Occlusion, Reflection, Smoothing, bury, graveyard,
};

/// Where the two passes a semi-transparent surface draws through are linked, past every channel a
/// draw of the opaque half is keyed by.
const SHEER_BUFFER: usize = deferred::REFLECTED + 1;
const SHEER_RESOLVE: usize = SHEER_BUFFER + 1;

/// Attribute locations, in the order [`Vertex`] stores them.
const ATTRIBUTES: [(u32, i32, i32); 4] = [(0, 3, 0), (1, 3, 12), (2, 4, 24), (3, 2, 56)];
const COLOR: u32 = 4;
const COLOR_OFFSET: i32 = 88;

/// The influences a skinned vertex carries, which the shader reads as integers.
const INFLUENCES: [(u32, i32); 2] = [(5, 96), (6, 104)];

/// Where each semantic a drawing package asks for sits in a [`Vertex`], and how wide it is. The
/// bytes are read as integers where the shader's own signature declares them so.
const FIELDS: [(program::Field, i32, i32, u32); 10] = [
    (program::Field::Position, 3, 0, glow::FLOAT),
    (program::Field::Normal, 3, 12, glow::FLOAT),
    (program::Field::Tangent, 4, 24, glow::FLOAT),
    (program::Field::Bitangent, 4, 40, glow::FLOAT),
    (program::Field::Uv, 4, 56, glow::FLOAT),
    (program::Field::Uv1, 4, 72, glow::FLOAT),
    (program::Field::Color, 4, 88, glow::UNSIGNED_BYTE),
    (program::Field::Color1, 4, 92, glow::UNSIGNED_BYTE),
    (program::Field::Weights, 4, 96, glow::UNSIGNED_SHORT),
    (program::Field::Bones, 4, 104, glow::UNSIGNED_SHORT),
];

/// The color table, which the game's own shaders address as a texture of their own.
const TABLE: u32 = 0x2005_679f;

/// Texture units, in the order the shader's samplers declare them.
const NORMAL_UNIT: u32 = 0;
const INDEX_UNIT: u32 = 1;
const MASK_UNIT: u32 = 2;
const DIFFUSE_UNIT: u32 = 3;
const TABLE_UNIT: u32 = 4;
const JOINTS_UNIT: u32 = 5;

/// Texels per color-table row. This viewer's own packing, not the game's.
pub const TABLE_COLUMNS: i32 = 4;

const VERTEX_SOURCE: &str = include_str!("model.vert");
const FRAGMENT_SOURCE: &str = include_str!("model.frag");

/// A mesh's geometry, once it is on the card.
struct Buffers {
    layout: glow::VertexArray,
    vertices: glow::Buffer,
    indices: glow::Buffer,
    /// The middle of the mesh's own extent, which is what orders the passes drawn over the frame.
    center: glam::Vec3,
}

/// One material drawn with the shaders the game would draw it with.
pub struct Shaded {
    /// The buffer pass, one reading per page of its targets: a context promised four draw buffers
    /// fills a five-target G-buffer by running the pass more than once.
    pub buffer: Vec<Arc<program::Program>>,
    /// The depth pass, which runs first so the buffer pass shades nothing it covers.
    pub depth: Option<Arc<program::Program>>,
    /// The same geometry as the light sees it, which is the depth a shadow is tested against. The
    /// package answers this under a subview of its own rather than as a pass of the main one.
    pub shadow: Option<Arc<program::Program>>,
    /// What the material resolves itself into the frame with, drawn as its own geometry over what
    /// the lighting left. A semitransparent package has only this: it writes no G-buffer at all,
    /// and what it blends over is the frame the composite already resolved.
    pub resolve: Option<Arc<program::Program>>,
    /// The pair that draws what the buffer pass clipped away: the surface into a buffer of its own,
    /// and what blends that over the frame the opaque half left.
    pub sheer: Option<(Arc<program::Program>, Arc<program::Program>)>,
    /// The color table in the game's own layout: its halfs, the texels a row takes, and the rows.
    pub table: Option<Arc<(Vec<u16>, usize, usize)>>,
    /// The textures the material binds, by the resource id the package knows each by.
    pub textures: Vec<(u32, Option<Bound>)>,
}

impl Shaded {
    /// What the material bound at one of its package's samplers.
    pub fn bound(&self, id: u32) -> Option<&Bound> {
        self.textures
            .iter()
            .find(|(held, _)| *held == id)
            .and_then(|(_, held)| held.as_ref())
    }
}

/// A texture a material named, where it has arrived. A plane is on the card as an egui texture;
/// anything with slices is in the graph's own store, since egui holds nothing but planes.
#[derive(Clone)]
pub enum Bound {
    /// The anisotropy the material's own sampler asks for, since egui's texture manager owns the
    /// object and has no field for it; `0.0` asks for none.
    Plane(TextureId, f32),
    Stacked(Arc<str>),
}

impl Bound {
    pub fn plane(&self) -> Option<(TextureId, f32)> {
        match self {
            Self::Plane(held, aniso) => Some((*held, *aniso)),
            Self::Stacked(_) => None,
        }
    }

    pub fn stacked(&self) -> Option<&str> {
        match self {
            Self::Stacked(held) => Some(held),
            Self::Plane(..) => None,
        }
    }
}

/// What one draw call needs beyond its geometry: the material it uses, and the egui textures that
/// material resolved to.
pub struct Surface {
    pub material: usize,
    pub shaded: Option<Shaded>,
    /// Which of the mesh's indices to draw, so a hidden part costs no triangles.
    pub runs: Vec<Range<i32>>,
    pub family: Family,
    pub normal: Option<TextureId>,
    pub index: Option<TextureId>,
    pub mask: Option<TextureId>,
    pub diffuse: Option<TextureId>,
    pub alpha_threshold: f32,
    pub diffuse_color: [f32; 3],
    pub emissive_color: [f32; 3],
    pub normal_scale: f32,
    pub cull: bool,
    /// Set where the material's own package states how its glass reaches the frame, which is the
    /// one package that does.
    pub glass: Option<Glass>,
}

/// What a mesh draws as while its material is still being fetched: bare geometry, nothing tinted
/// away and nothing clipped.
impl Default for Surface {
    fn default() -> Self {
        Self {
            material: 0,
            shaded: None,
            runs: Vec::new(),
            family: Family::Background,
            normal: None,
            index: None,
            mask: None,
            diffuse: None,
            alpha_threshold: 0.0,
            diffuse_color: [1.0; 3],
            emissive_color: [0.0; 3],
            normal_scale: 1.0,
            glass: None,
            cull: false,
        }
    }
}

/// What the shader draws instead of a shaded surface. Discriminants are the values `model.frag`
/// compares `u_debug` against.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Debug {
    None = 0,
    Normals = 1,
    Uv = 2,
    Geometry = 3,
    Tangents = 4,
    Bitangents = 5,
    Handedness = 6,
    Color = 7,
    Alpha = 8,
    Meshes = 9,
}

/// One frame of camera and material bindings, rebuilt every time the widget draws.
pub struct Frame {
    pub view: [f32; 16],
    pub projection: [f32; 16],
    /// Which G-buffer channel the game-shader path puts on screen, or the lit frame past the last
    /// of them.
    pub target: usize,
    /// What the game's own shaders are given that no file says: the camera in the convention they
    /// were compiled for, the frame's size, and the one light this viewer lights with.
    pub scene: program::Scene,
    /// The passes that light the G-buffer, once their packages have arrived.
    pub lighting: Option<Arc<Lighting>>,
    /// The pass that grades the frame they resolve, once its shader and its table have arrived.
    pub post: Option<Arc<program::Program>>,
    /// The chain that spreads the bright end of it into a halo, likewise.
    pub glare: Option<Arc<Glare>>,
    /// The pair that smooths its edges, once both their shaders have arrived and the viewer asks
    /// for them.
    pub smoothing: Option<Arc<Smoothing>>,
    /// The chain that works out how much sky reaches each pixel, on the same terms.
    pub occlusion: Option<Arc<Occlusion>>,
    /// The chain that reflects the frame off itself, where the viewer is drawing with it.
    pub reflection: Option<Arc<Reflection>>,
    /// The one that darkens its corners, which runs after all of them.
    pub vignette: Option<Arc<program::Program>>,
    pub eye: [f32; 3],
    /// Key, fill and rim directions, in world space. Built once a frame from the camera, so a
    /// surface is lit by one set of lights rather than by a set of its own.
    pub lights: [f32; 9],
    pub surfaces: Vec<Surface>,
    /// What each mesh's blend indices name, in the model's own space: one palette per mesh, since a
    /// mesh's indices run over its own bone table.
    pub joints: Vec<Vec<glam::Mat4>>,
    pub debug: Debug,
    /// The floor to rule under the model, where the viewer asks for one.
    pub grid: Option<Ground>,
    /// The emote's own particles, drawn into the frame the composite resolved so the character
    /// occludes them and the chain past it spreads their glow. Taken by the pass that draws them,
    /// which is what fills in the depth and the size only the buffers know.
    pub effects: Mutex<Vec<(Arc<Mutex<avfx::gpu::Particles>>, avfx::gpu::Frame)>>,
}

/// Geometry waiting for a context to upload it under.
#[derive(Default)]
pub struct Pending {
    pub meshes: Vec<(Vec<Vertex>, Vec<u16>)>,
}

/// The card's side of drawing one model with the game's own shaders: the frame of the graph, and one
/// linked program per material.
#[derive(Default)]
struct Game {
    /// One palette per mesh, since a mesh's blend indices name its own bone table.
    joints: Vec<glow::Texture>,
    programs: BTreeMap<(usize, bool, usize), Linked>,
    tables: BTreeMap<usize, (glow::Texture, Table)>,
    /// The array these shaders bind their attributes into. An array holds the enable flags and the
    /// pointers, and a mesh's own array holds the layout the preview path was uploaded with, so
    /// laying a shader's own semantics over it would leave the preview reading the wrong fields.
    layout: Option<glow::VertexArray>,
    failure: Option<String>,
}

/// Everything the callback owns, shared with the viewer that built it.
pub struct Model {
    pending: Option<Pending>,
    program: Option<glow::Program>,
    game: Game,
    /// The frame this model's own viewport draws into. A model standing in someone else's frame
    /// fills theirs instead and leaves this one unattached.
    buffers: deferred::Buffers,
    meshes: Vec<Buffers>,
    /// Color tables arrive with their materials, which is long after the geometry, so they queue
    /// rather than travelling with it.
    queued: Vec<(usize, Vec<f32>)>,
    /// Meshes whose indices a shape key rewrote, waiting for a context to upload them under.
    rewritten: Vec<(usize, Vec<u16>)>,
    /// The game's own layered textures, waiting for the same.
    arrays: Vec<(u32, Layered)>,
    /// The same, for the ones the material names rather than the engine, under the naming path.
    stacks: Vec<(Arc<str>, Layered)>,
    /// The table the shading passes index, waiting for the same.
    types: Option<Vec<u32>>,
    grid: Grid,
    tables: BTreeMap<usize, (glow::Texture, f32)>,
    /// Why the shader would not build, kept so the viewer can say so rather than draw nothing.
    failure: Option<String>,
}

impl Model {
    pub fn new(pending: Pending) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            pending: Some(pending),
            program: None,
            game: Game::default(),
            buffers: deferred::Buffers::default(),
            meshes: Vec::new(),
            queued: Vec::new(),
            rewritten: Vec::new(),
            arrays: Vec::new(),
            stacks: Vec::new(),
            types: None,
            grid: Grid::default(),
            tables: BTreeMap::new(),
            failure: None,
        }))
    }

    pub fn failure(&self) -> Option<&str> {
        self.failure.as_deref()
    }

    /// The game shader pipeline's own failure, taken rather than borrowed: leaving it would relink
    /// and refail the same program every frame once the caller stops routing surfaces its way.
    pub fn take_shader_failure(&mut self) -> Option<String> {
        self.game.failure.take()
    }

    /// How much of the G-buffer one pass can write. Four until a frame has asked the context, since
    /// that is what a context is promised.
    pub fn attachments(&self) -> usize {
        self.buffers.attachments()
    }

    /// What the context answered, once it has.
    pub fn attachments_learned(&self) -> Option<usize> {
        self.buffers.attachments_learned()
    }

    /// Carries that answer over to a `Model` built fresh, so it draws its first frame at the real
    /// count rather than the four a context is merely promised.
    pub fn seed_attachments(&mut self, learned: usize) {
        self.buffers.seed_attachments(learned);
    }

    /// Hands a material's color table over for the next draw to upload.
    pub fn queue_table(&mut self, material: usize, values: Vec<f32>) {
        self.queued.push((material, values));
    }

    /// Hands a mesh's indices over for the next draw to upload, replacing the ones it holds.
    pub fn queue_indices(&mut self, mesh: usize, indices: Vec<u16>) {
        self.rewritten.push((mesh, indices));
    }

    /// Hands one of the game's own layered textures over, under the resource id its shaders name it
    /// by.
    pub fn queue_array(&mut self, id: u32, held: Layered) {
        self.arrays.push((id, held));
    }

    /// The same, for a texture the material names, under the path that named it.
    pub fn queue_stack(&mut self, path: Arc<str>, held: Layered) {
        self.stacks.push((path, held));
    }

    /// Hands the table the shading passes index over, replacing the one the frame stood in with.
    pub fn queue_types(&mut self, values: Vec<u32>) {
        self.types = Some(values);
    }

    /// Everything a draw needs standing that belongs to the model rather than to the frame it is
    /// drawn into. Answers whether it is worth drawing at all.
    fn stage(&mut self, gl: &glow::Context, surfaces: usize) -> bool {
        if self.failure.is_some() {
            return false;
        }
        if let Some(pending) = self.pending.take()
            && let Err(why) = self.upload(gl, pending)
        {
            self.failure = Some(why);
            return false;
        }
        for (mesh, indices) in std::mem::take(&mut self.rewritten) {
            let Some(buffers) = self.meshes.get(mesh) else {
                continue;
            };
            // Through the mesh's own vertex array, since binding an element buffer rewrites
            // whichever array is current, and egui leaves its own bound around a callback.
            unsafe {
                gl.bind_vertex_array(Some(buffers.layout));
                gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(buffers.indices));
                gl.buffer_data_u8_slice(
                    glow::ELEMENT_ARRAY_BUFFER,
                    bytemuck::cast_slice(&indices),
                    glow::STATIC_DRAW,
                );
                gl.bind_vertex_array(None);
            }
        }
        for (material, values) in std::mem::take(&mut self.queued) {
            let rows = values.len() as i32 / (TABLE_COLUMNS * 4);
            match upload_table(gl, &values, rows) {
                Ok(texture) => {
                    self.tables.insert(material, (texture, rows as f32));
                }
                Err(why) => log::error!("assets/mdl: color table: {why}"),
            }
        }
        // A zip would truncate instead, and a mesh drawn under another mesh's material shows as a
        // texturing bug rather than as the bookkeeping error it is.
        if self.meshes.len() != surfaces {
            self.failure = Some(format!(
                "{} meshes against {surfaces} surfaces",
                self.meshes.len(),
            ));
            return false;
        }
        true
    }

    /// Fills a G-buffer someone else owns with this model's own surfaces, and nothing else: the
    /// lighting, the composite and every pass past it belong to the frame it is standing in.
    pub fn fill(
        &mut self,
        gl: &glow::Context,
        painter: &egui_glow::Painter,
        held: (&[Surface], &[Vec<glam::Mat4>]),
        buffers: &mut deferred::Buffers,
        scene: &program::Scene,
    ) -> Result<(), String> {
        if !self.stage(gl, held.0.len()) {
            return Ok(());
        }
        let supplied = (
            std::mem::take(&mut self.arrays),
            std::mem::take(&mut self.stacks),
            self.types.take(),
        );
        supply(gl, buffers, supplied);
        self.game
            .fill(gl, painter, held, &self.meshes, buffers, scene)
    }

    /// The surfaces of this model that answer into the frame rather than into the buffer: drawn
    /// once the host has resolved its own lighting over what [`Self::fill`] left.
    #[allow(clippy::too_many_arguments)]
    pub fn over(
        &mut self,
        gl: &glow::Context,
        painter: &egui_glow::Painter,
        surfaces: &[Surface],
        buffers: &mut deferred::Buffers,
        lighting: &deferred::Lighting,
        lamps: &[program::Lamp],
        scene: &program::Scene,
    ) -> Result<(), String> {
        if self.failure.is_some() || self.meshes.len() != surfaces.len() {
            return Ok(());
        }
        self.game
            .resolve(gl, painter, surfaces, &self.meshes, buffers, scene)?;
        self.game.sheer(
            gl, painter, surfaces, &self.meshes, buffers, lighting, lamps, scene,
        )
    }

    pub fn draw(
        &mut self,
        gl: &glow::Context,
        painter: &egui_glow::Painter,
        frame: &Frame,
        info: &egui::PaintCallbackInfo,
    ) {
        bury(gl);
        // Before anything is drawn, shaded or not: the G-buffer is only attached where a frame
        // draws into it, and by then this frame's passes have already been translated.
        self.buffers.limit(gl);
        if !self.stage(gl, frame.surfaces.len()) {
            return;
        }
        let held = (
            std::mem::take(&mut self.arrays),
            std::mem::take(&mut self.stacks),
            self.types.take(),
        );
        supply(gl, &mut self.buffers, held);
        let Some(program) = self.program else {
            return;
        };

        if frame.surfaces.iter().any(|held| held.shaded.is_some()) {
            self.game
                .draw(gl, painter, frame, &self.meshes, &mut self.buffers, info);
            // Only over the frame the composite resolved: a raw channel is data, and a grid ruled
            // across it would be read as part of it.
            if frame.target >= LIT {
                self.ground(gl, painter, frame, info);
            }
            return;
        }

        // In the model's own space, since this path lights a vertex where the file put it and only
        // then takes it through the camera. A model with nothing to skin leaves the palettes alone
        // and reads none of them.
        if frame.joints.iter().any(|held| !held.is_empty())
            && let Err(why) = self.game.palettes(gl, &frame.joints, glam::Mat4::IDENTITY)
        {
            self.failure = Some(why);
            return;
        }

        unsafe {
            gl.enable(glow::DEPTH_TEST);
            gl.depth_func(glow::LESS);
            gl.depth_mask(true);
            gl.clear(glow::DEPTH_BUFFER_BIT);
            gl.disable(glow::BLEND);
            gl.use_program(Some(program));

            let view = gl.get_uniform_location(program, "u_view");
            gl.uniform_matrix_4_f32_slice(view.as_ref(), false, &frame.view);
            let projection = gl.get_uniform_location(program, "u_projection");
            gl.uniform_matrix_4_f32_slice(projection.as_ref(), false, &frame.projection);
            let eye = gl.get_uniform_location(program, "u_eye");
            gl.uniform_3_f32_slice(eye.as_ref(), &frame.eye);
            let lights = gl.get_uniform_location(program, "u_lights[0]");
            gl.uniform_3_f32_slice(lights.as_ref(), &frame.lights);
            for (name, unit) in [
                ("u_normal_map", NORMAL_UNIT),
                ("u_index_map", INDEX_UNIT),
                ("u_mask_map", MASK_UNIT),
                ("u_diffuse_map", DIFFUSE_UNIT),
                ("u_table", TABLE_UNIT),
                ("u_joints", JOINTS_UNIT),
            ] {
                let slot = gl.get_uniform_location(program, name);
                gl.uniform_1_i32(slot.as_ref(), unit as i32);
            }
            let debug = gl.get_uniform_location(program, "u_debug");
            gl.uniform_1_i32(debug.as_ref(), frame.debug as i32);
            let have = gl.get_uniform_location(program, "u_have");
            let family = gl.get_uniform_location(program, "u_family");
            let mesh = gl.get_uniform_location(program, "u_mesh");
            let threshold = gl.get_uniform_location(program, "u_alpha_threshold");
            let rows = gl.get_uniform_location(program, "u_table_rows");
            let diffuse = gl.get_uniform_location(program, "u_diffuse_color");
            let emissive = gl.get_uniform_location(program, "u_emissive_color");
            let scale = gl.get_uniform_location(program, "u_normal_scale");
            let skinned = gl.get_uniform_location(program, "u_skinned");

            for (at, (buffers, surface)) in self.meshes.iter().zip(&frame.surfaces).enumerate() {
                if surface.runs.is_empty() {
                    continue;
                }
                match surface.cull {
                    true => {
                        gl.enable(glow::CULL_FACE);
                        gl.cull_face(glow::BACK);
                        gl.front_face(glow::CCW);
                    }
                    false => gl.disable(glow::CULL_FACE),
                }

                let table = self.tables.get(&surface.material).copied();
                let mut bound = 0;
                for (unit, id) in [
                    (NORMAL_UNIT, surface.normal),
                    (INDEX_UNIT, surface.index),
                    (MASK_UNIT, surface.mask),
                    (DIFFUSE_UNIT, surface.diffuse),
                ] {
                    let texture = id.and_then(|id| painter.texture(id));
                    gl.active_texture(glow::TEXTURE0 + unit);
                    gl.bind_texture(glow::TEXTURE_2D, texture);
                    // The game-shader pass may have left this same egui-owned texture object at a
                    // material's own anisotropy: it is texture-object state, not per-draw state,
                    // so it survives toggling "Game shaders" off and this plain pass would
                    // otherwise inherit it silently.
                    if texture.is_some() && deferred::max_anisotropy(gl) > 0.0 {
                        gl.tex_parameter_f32(glow::TEXTURE_2D, glow::TEXTURE_MAX_ANISOTROPY_EXT, 1.0);
                    }
                    bound |= i32::from(texture.is_some()) << unit;
                }
                gl.active_texture(glow::TEXTURE0 + JOINTS_UNIT);
                gl.bind_texture(glow::TEXTURE_2D, self.game.joints.get(at).copied());
                gl.uniform_1_i32(
                    skinned.as_ref(),
                    i32::from(frame.joints.get(at).is_some_and(|held| !held.is_empty())),
                );
                gl.active_texture(glow::TEXTURE0 + TABLE_UNIT);
                gl.bind_texture(glow::TEXTURE_2D, table.map(|(texture, _)| texture));
                bound |= i32::from(table.is_some()) << TABLE_UNIT;

                gl.uniform_1_i32(have.as_ref(), bound);
                gl.uniform_1_i32(family.as_ref(), surface.family as i32);
                gl.uniform_1_i32(mesh.as_ref(), at as i32);
                gl.uniform_1_f32(threshold.as_ref(), surface.alpha_threshold);
                gl.uniform_1_f32(rows.as_ref(), table.map_or(0.0, |(_, rows)| rows));
                gl.uniform_3_f32_slice(diffuse.as_ref(), &surface.diffuse_color);
                gl.uniform_3_f32_slice(emissive.as_ref(), &surface.emissive_color);
                gl.uniform_1_f32(scale.as_ref(), surface.normal_scale);

                gl.bind_vertex_array(Some(buffers.layout));
                for run in &surface.runs {
                    let offset = run.start * size_of::<u16>() as i32;
                    gl.draw_elements(
                        glow::TRIANGLES,
                        run.end - run.start,
                        glow::UNSIGNED_SHORT,
                        offset,
                    );
                }
            }

            gl.bind_vertex_array(None);
            gl.depth_mask(false);
        }

        self.ground(gl, painter, frame, info);
        // This path draws straight into the buffer egui bound, so the depth the model just left is
        // what a glow is tested against. Nothing there can be sampled back, which leaves the
        // soft-particle variant its unbound sampler.
        let held = info.viewport_in_pixels();
        let size = (held.width_px.max(1), held.height_px.max(1));
        for (particles, effect) in frame.effects.lock().unwrap().iter_mut() {
            effect.tested = true;
            effect.scene.size = (size.0 as f32, size.1 as f32);
            particles.lock().unwrap().draw(gl, painter, effect);
        }
    }

    /// The floor, over the model and against the depth whichever path drew it left behind. Both
    /// leave that depth in the buffer egui bound before the callback: the preview path draws
    /// straight into it, and the pass that puts the game path's frame up carries the G-buffer's own
    /// depth over with it.
    fn ground(
        &mut self,
        gl: &glow::Context,
        painter: &egui_glow::Painter,
        frame: &Frame,
        info: &egui::PaintCallbackInfo,
    ) {
        let Some(ground) = &frame.grid else {
            return;
        };
        let held = info.viewport_in_pixels();
        let viewport = (
            held.left_px,
            held.from_bottom_px,
            held.width_px.max(1),
            held.height_px.max(1),
        );
        if let Err(why) = self
            .grid
            .draw(gl, ground, painter.intermediate_fbo(), viewport)
        {
            self.failure = Some(why);
        }
    }

    fn upload(&mut self, gl: &glow::Context, pending: Pending) -> Result<(), String> {
        // `antialias` on the canvas is a hint the implementation may ignore, and nothing short of a
        // live context says whether it did. DEPTH_BITS is not asked alongside it: a core profile
        // dropped that query, and asking raises an error the frame is then blamed for.
        let samples = unsafe { gl.get_parameter_i32(glow::SAMPLES) };
        log::info!(
            "assets/mdl: {} meshes on {:?}, {samples} samples",
            pending.meshes.len(),
            gl.version()
        );
        self.program = Some(build(gl)?);
        for (vertices, indices) in &pending.meshes {
            self.meshes.push(upload_mesh(gl, vertices, indices)?);
        }
        Ok(())
    }
}

impl Game {
    /// The joint palettes, which every skinned shader reads through a texture of dwords. Rewritten
    /// each frame, since a joint carries the camera as well as the pose.
    fn palettes(
        &mut self,
        gl: &glow::Context,
        joints: &[Vec<glam::Mat4>],
        object: glam::Mat4,
    ) -> Result<(), String> {
        let held = joints
            .iter()
            .map(|palette| dwords(gl, &program::joints(palette, object)))
            .collect::<Result<Vec<_>, _>>()?;
        let stale = std::mem::replace(&mut self.joints, held);
        graveyard()
            .lock()
            .unwrap()
            .extend(stale.into_iter().map(Dead::Texture));
        Ok(())
    }

    fn layout(&mut self, gl: &glow::Context) -> Result<glow::VertexArray, String> {
        if let Some(held) = self.layout {
            return Ok(held);
        }
        let held = unsafe { gl.create_vertex_array()? };
        self.layout = Some(held);
        Ok(held)
    }

    /// The material's color table, in the layout its own shaders address it in. Re-uploaded
    /// whenever `held` names a different table than the one already resident, which is what makes a
    /// changed dye visible: the table itself is otherwise cached forever, by material index alone.
    fn table(
        &mut self,
        gl: &glow::Context,
        material: usize,
        held: &Table,
    ) -> Result<glow::Texture, String> {
        if let Some((texture, resident)) = self.tables.get(&material)
            && Arc::ptr_eq(resident, held)
        {
            return Ok(*texture);
        }
        let (values, columns, rows) = &**held;
        unsafe {
            let texture = match self.tables.remove(&material) {
                Some((texture, _)) => texture,
                None => gl.create_texture()?,
            };
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA16F as i32,
                *columns as i32,
                *rows as i32,
                0,
                glow::RGBA,
                glow::HALF_FLOAT,
                glow::PixelUnpackData::Slice(Some(bytemuck::cast_slice(values))),
            );
            // Filtered, because the shader addresses a row pair by landing between the two of them
            // and leaves the mix to the sampler. Every other read it makes is of a texel center,
            // which filtering answers exactly.
            for (name, value) in [
                (glow::TEXTURE_MIN_FILTER, glow::LINEAR),
                (glow::TEXTURE_MAG_FILTER, glow::LINEAR),
                (glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE),
                (glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE),
            ] {
                gl.tex_parameter_i32(glow::TEXTURE_2D, name, value as i32);
            }
            self.tables.insert(material, (texture, held.clone()));
            Ok(texture)
        }
    }

    fn draw(
        &mut self,
        gl: &glow::Context,
        painter: &egui_glow::Painter,
        frame: &Frame,
        meshes: &[Buffers],
        buffers: &mut deferred::Buffers,
        info: &egui::PaintCallbackInfo,
    ) {
        let held = info.viewport_in_pixels();
        let size = (held.width_px.max(1), held.height_px.max(1));
        // egui draws into whatever it bound before the callback, and that has to be bound again
        // whether or not the frame drew. Asking the painter rather than the context is what makes
        // this work on the web: glow keeps its own map of the resources it created, and a
        // framebuffer read back out of WebGL is a JS object it cannot find in there.
        let bound = painter.intermediate_fbo();
        let drawn = self.render(gl, painter, frame, meshes, buffers, size);
        let shown = buffers.show(
            gl,
            frame.target,
            bound,
            (held.left_px, held.from_bottom_px, size.0, size.1),
        );
        self.failure = drawn.and(shown).err();
    }

    /// The G-buffer a page at a time, and each page's depth pass before its buffer pass: the game
    /// runs those as two passes over the whole draw rather than as two draws of one surface.
    ///
    /// A mesh whose program will not link or bind is skipped rather than aborting the pass: one
    /// material a real driver rejects (and a software one does not) would otherwise blank every mesh
    /// behind it in the loop, and everything the frame does with the buffer along with it, every
    /// frame from then on. The first such failure is returned once the loop has had its turn, which
    /// is what still reaches the "game shaders would not build" banner.
    fn fill(
        &mut self,
        gl: &glow::Context,
        painter: &egui_glow::Painter,
        held: (&[Surface], &[Vec<glam::Mat4>]),
        meshes: &[Buffers],
        buffers: &mut deferred::Buffers,
        scene: &program::Scene,
    ) -> Result<(), String> {
        let (surfaces, joints) = held;
        self.palettes(gl, joints, scene.view * scene.model)?;
        let mut failed: Option<String> = None;
        for page in 0..buffers.pages() {
            buffers.reopen(gl, page);
            for depth in [true, false] {
                for (at, (mesh, surface)) in meshes.iter().zip(surfaces).enumerate() {
                    let Some(shaded) = &surface.shaded else {
                        continue;
                    };
                    if surface.runs.is_empty() {
                        continue;
                    }
                    let held = match depth {
                        true => shaded.depth.as_ref(),
                        false => shaded.buffer.get(page),
                    };
                    let Some(held) = held.filter(|held| depth || !held.targets.is_empty()) else {
                        continue;
                    };
                    let program = match deferred::link(
                        gl,
                        &mut self.programs,
                        (surface.material, depth, page),
                        held,
                    ) {
                        Ok(program) => program,
                        Err(why) => {
                            let why = format!(
                                "material {} depth={depth} page={page} attachments={}: {why}",
                                surface.material,
                                buffers.attachments()
                            );
                            log::error!("assets/mdl: {why}");
                            failed.get_or_insert(why);
                            continue;
                        }
                    };
                    unsafe {
                        gl.use_program(Some(program));
                        // Nearer-or-equal, every pass settling its own depth rather than taking
                        // the depth pass's: the two do not clip the same fragments, so a buffer
                        // pass gated on the depth the other settled loses every fragment only its
                        // own test keeps.
                        gl.depth_mask(true);
                        gl.depth_func(glow::LEQUAL);
                        gl.color_mask(!depth, !depth, !depth, !depth);
                        let written: Vec<u32> = (0..held.targets.len().max(1))
                            .map(|at| glow::COLOR_ATTACHMENT0 + at as u32)
                            .collect();
                        gl.draw_buffers(&written);
                        match surface.cull {
                            true => {
                                gl.enable(glow::CULL_FACE);
                                gl.cull_face(glow::BACK);
                                gl.front_face(glow::CCW);
                            }
                            false => gl.disable(glow::CULL_FACE),
                        }
                    }
                    if let Err(why) =
                        self.bind(gl, painter, program, held, surface, at, mesh, buffers, scene)
                    {
                        let why = format!(
                            "material {} depth={depth} page={page} attachments={}: {why}",
                            surface.material,
                            buffers.attachments()
                        );
                        log::error!("assets/mdl: {why}");
                        failed.get_or_insert(why);
                    }
                }
            }
        }
        match failed {
            Some(why) => Err(why),
            None => Ok(()),
        }
    }

    /// The emote's own particles, over the frame the composite resolved.
    ///
    /// Drawn through the framebuffer standing on the copy of the depth, which leaves the live one
    /// free to be sampled: a glow is tested against the depth the character settled, and the
    /// soft-particle variant reads that same depth back.
    fn particles(
        &self,
        gl: &glow::Context,
        painter: &egui_glow::Painter,
        frame: &Frame,
        buffers: &deferred::Buffers,
        size: (i32, i32),
    ) -> Result<(), String> {
        let mut held = frame.effects.lock().unwrap();
        let (Some(into), Some(depth)) = (buffers.bare(), buffers.depth()) else {
            return Ok(());
        };
        if held.is_empty() {
            return Ok(());
        }
        buffers.cut(gl)?;
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(into));
            gl.viewport(0, 0, size.0, size.1);
        }
        for (particles, effect) in held.iter_mut() {
            effect.tested = true;
            effect.depth = Some(depth);
            effect.scene.size = (size.0 as f32, size.1 as f32);
            particles.lock().unwrap().draw(gl, painter, effect);
        }
        unsafe {
            gl.depth_func(glow::LESS);
            gl.depth_mask(true);
            gl.disable(glow::BLEND);
            gl.disable(glow::DEPTH_TEST);
        }
        Ok(())
    }

    fn render(
        &mut self,
        gl: &glow::Context,
        painter: &egui_glow::Painter,
        frame: &Frame,
        meshes: &[Buffers],
        buffers: &mut deferred::Buffers,
        size: (i32, i32),
    ) -> Result<(), String> {
        buffers.attach(gl, size)?;
        buffers.stand_ins(gl)?;
        // Only the callback knows how many pixels the widget really covers, and a screen-wide pass
        // has nothing else to turn a fragment into a texel with.
        let scene = program::Scene {
            size: (size.0 as f32, size.1 as f32),
            ..frame.scene.clone()
        };
        // Emptied here rather than a page at a time inside the fill, which is what lets a model
        // stand in a frame someone else has already written.
        for page in 0..buffers.pages() {
            buffers.open(gl, page);
        }
        let failed = self
            .fill(
                gl,
                painter,
                (&frame.surfaces, &frame.joints),
                meshes,
                buffers,
                &scene,
            )
            .err();
        // Only where the lit frame is what is being shown: a raw channel is a page of the G-buffer
        // and owes nothing to the passes past it.
        if let Some(lighting) = frame.lighting.as_ref().filter(|_| frame.target >= TARGETS) {
            // Before anything reads it: every lighting pass and the composite take the occlusion as
            // a weight on what they work out.
            match frame.occlusion.as_ref() {
                Some(held) => buffers.occlude(gl, held, &scene)?,
                None => buffers.unocclude(),
            }
            buffers
                .resolve(gl, lighting, &scene, &[frame.scene.lamp])?;
            self.resolve(gl, painter, &frame.surfaces, meshes, buffers, &scene)?;
            self.sheer(
                gl,
                painter,
                &frame.surfaces,
                meshes,
                buffers,
                lighting,
                &[frame.scene.lamp],
                &scene,
            )?;
            // Over the frame the composite left and before anything spreads or grades it, which is
            // where the game runs it.
            if let Some(reflection) = frame.reflection.as_ref() {
                buffers.mirror(gl, reflection, &scene)?;
            }
            // Over the composite and before the chain that spreads the bright end of it, which is
            // where the game draws them: a glow behind the character is hidden by it, and one in
            // front blooms with everything else.
            self.particles(gl, painter, frame, buffers, size)?;
            if let Some(glare) = frame.glare.as_ref() {
                buffers.source(gl)?;
                buffers.glare(gl, glare, &scene)?;
            }
            if let Some(post) = frame.post.as_ref() {
                buffers.post(gl, post, &scene)?;
            }
            if let Some(smoothing) = frame.smoothing.as_ref() {
                buffers.antialias(gl, smoothing, &scene)?;
            }
            // Last, over the graded frame, which is where the game draws it.
            if let Some(vignette) = frame.vignette.as_ref() {
                buffers.vignette(gl, vignette, &scene)?;
            }
        }
        match failed {
            Some(why) => Err(why),
            None => Ok(()),
        }
    }

    /// Every material resolved into the frame as its own geometry, after the lighting.
    ///
    /// A material that settled its own depth goes first and resolves against it; one that did not
    /// reads the frame instead, so the copy it reads is taken once the rest have drawn. Depth
    /// tested against what the G-buffer covered and writing none of its own, so the surfaces in
    /// front of a piece of glass hide it and the pieces behind it do not.
    ///
    /// A package with no opaque pass fills the buffer through a semitransparent one and settles no
    /// depth of its own, so it belongs in the second leg however much of the buffer it wrote: the
    /// alpha its composite states is what makes it sheer, and only that leg applies it.
    ///
    /// The ones reading the frame are drawn back to front and each takes its own copy: a pass of
    /// theirs writes the whole composited color rather than blending, so one drawn over another
    /// reading the same copy would erase it.
    fn resolve(
        &mut self,
        gl: &glow::Context,
        painter: &egui_glow::Painter,
        surfaces: &[Surface],
        meshes: &[Buffers],
        buffers: &mut deferred::Buffers,
        scene: &program::Scene,
    ) -> Result<(), String> {
        let (opaque, mut blended): (Vec<usize>, Vec<usize>) = surfaces
            .iter()
            .enumerate()
            .filter(|(_, surface)| {
                surface
                    .shaded
                    .as_ref()
                    .is_some_and(|shaded| shaded.resolve.is_some())
                    && !surface.runs.is_empty()
            })
            .map(|(at, _)| at)
            .partition(|at| {
                surfaces[*at]
                    .shaded
                    .as_ref()
                    .is_some_and(|shaded| shaded.depth.is_some())
            });
        // The camera looks down negative z, so the farthest is the least.
        let away = |at: &usize| match meshes.get(*at) {
            Some(mesh) => (scene.view * scene.model).transform_point3(mesh.center).z,
            None => 0.0,
        };
        blended.sort_by(|left, right| away(left).total_cmp(&away(right)));

        for (behind, held) in [(false, &opaque), (true, &blended)] {
            if held.is_empty() {
                continue;
            }
            // Tested against a copy of the depth rather than the depth itself: water reads the
            // depth back, and the live one is the framebuffer's own attachment.
            buffers.cut(gl)?;
            let into = buffers.bare().ok_or("no lit frame")?;
            let size = buffers.size();
            unsafe {
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(into));
                gl.draw_buffers(&[glow::COLOR_ATTACHMENT0]);
                gl.viewport(0, 0, size.0, size.1);
                gl.color_mask(true, true, true, true);
                gl.enable(glow::DEPTH_TEST);
                // A surface that filled the G-buffer resolves against the depth its own pre-pass
                // settled, so a mesh layered over itself keeps the fragment the buffer pass kept.
                // One that wrote no depth of its own has nothing to match and only wants to be
                // hidden by what stands in front of it.
                gl.depth_func(match behind {
                    true => glow::LEQUAL,
                    false => glow::EQUAL,
                });
                gl.depth_mask(false);
                // A pass drawn over the frame states in its alpha how much of itself reaches it,
                // and nothing downstream would ever apply that. One that reads the frame back
                // instead resolves what it found and states one, which blends to the same pixel.
                match behind {
                    true => {
                        gl.enable(glow::BLEND);
                        gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);
                    }
                    false => gl.disable(glow::BLEND),
                }
            }
            // Water reads whatever stands behind it out of the frame it is about to write, so the
            // copy is taken once before the leg rather than between its draws.
            if !behind {
                buffers.keep(gl)?;
            }
            for at in held {
                if behind {
                    buffers.keep(gl)?;
                }
                let surface = &surfaces[*at];
                let Some(mesh) = meshes.get(*at) else {
                    continue;
                };
                // Glass states its own blend. Its pass hands over what the frame behind is to be
                // scaled by rather than what to mix into it, so blending that on coverage lays a lit
                // card over the frame and the halo chain then spreads it.
                if behind {
                    unsafe {
                        match surface.glass {
                            Some(Glass::Mul) => gl.blend_func(glow::DST_COLOR, glow::ZERO),
                            Some(Glass::Add) => gl.blend_func(glow::SRC_ALPHA, glow::ONE),
                            None => gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA),
                        }
                    }
                }
                let held = surface
                    .shaded
                    .as_ref()
                    .and_then(|shaded| shaded.resolve.as_ref())
                    .ok_or("no pass to resolve with")?;
                let program =
                    deferred::link(gl, &mut self.programs, (surface.material, false, LIT), held)?;
                unsafe {
                    gl.use_program(Some(program));
                    match surface.cull {
                        true => {
                            gl.enable(glow::CULL_FACE);
                            gl.cull_face(glow::BACK);
                            gl.front_face(glow::CCW);
                        }
                        false => gl.disable(glow::CULL_FACE),
                    }
                }
                self.bind(gl, painter, program, held, surface, *at, mesh, buffers, scene)?;
            }
        }
        unsafe { gl.disable(glow::BLEND) };
        Ok(())
    }

    /// The half of a surface the buffer pass clipped away, drawn through a buffer of its own.
    ///
    /// A material states one alpha to be drawn as opaque past, and the pass that keeps the rest
    /// states another far under it; between the two stands the fringe of a strand of hair, which
    /// the opaque half never draws. Minification averages a normal map's alpha toward its mean and
    /// so widens that band, which is what thins a head of hair as the camera stands back from it.
    ///
    /// The surface fills a G-buffer of its own, the light is gathered again over that, and each
    /// material blends itself into the frame against the depth the opaque half settled.
    #[allow(clippy::too_many_arguments)]
    fn sheer(
        &mut self,
        gl: &glow::Context,
        painter: &egui_glow::Painter,
        surfaces: &[Surface],
        meshes: &[Buffers],
        buffers: &mut deferred::Buffers,
        lighting: &deferred::Lighting,
        lamps: &[program::Lamp],
        scene: &program::Scene,
    ) -> Result<(), String> {
        let held: Vec<(usize, Arc<program::Program>, Arc<program::Program>)> = surfaces
            .iter()
            .enumerate()
            .filter(|(_, surface)| !surface.runs.is_empty())
            .filter_map(|(at, surface)| {
                let (buffer, resolve) = surface.shaded.as_ref()?.sheer.as_ref()?;
                Some((at, buffer.clone(), resolve.clone()))
            })
            .collect();
        if held.is_empty() {
            return Ok(());
        }
        buffers.sheer(gl)?;
        for (at, buffer, _) in &held {
            let surface = &surfaces[*at];
            let Some(mesh) = meshes.get(*at) else {
                continue;
            };
            let program = deferred::link(
                gl,
                &mut self.programs,
                (surface.material, false, SHEER_BUFFER),
                buffer,
            )?;
            unsafe {
                gl.use_program(Some(program));
                // Cut to what this buffer holds: the pass declares the fifth target the opaque one
                // writes and leaves it at nought, and there is no channel here for it to land in.
                let written: Vec<u32> = (0..buffer.targets.len().clamp(1, deferred::SHEER))
                    .map(|slot| glow::COLOR_ATTACHMENT0 + slot as u32)
                    .collect();
                gl.draw_buffers(&written);
                match surface.cull {
                    true => {
                        gl.enable(glow::CULL_FACE);
                        gl.cull_face(glow::BACK);
                        gl.front_face(glow::CCW);
                    }
                    false => gl.disable(glow::CULL_FACE),
                }
            }
            self.bind(gl, painter, program, buffer, surface, *at, mesh, buffers, scene)?;
        }

        buffers.relight(gl, lighting, scene, lamps)?;

        // Tested against a copy of the depth rather than the depth itself: a surface here also
        // samples it, and the live one is the framebuffer's own attachment.
        buffers.cut(gl)?;
        let into = buffers.bare().ok_or("no lit frame")?;
        let size = buffers.size();
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(into));
            gl.draw_buffers(&[glow::COLOR_ATTACHMENT0]);
            gl.viewport(0, 0, size.0, size.1);
            gl.color_mask(true, true, true, true);
            gl.enable(glow::DEPTH_TEST);
            // Against the depth the opaque half settled rather than this pass's own, and strictly
            // nearer: the resolve reads no coverage back out of the buffer, so every strand over a
            // pixel has to blend its own for the layers to add up.
            gl.depth_func(glow::LESS);
            gl.depth_mask(false);
            gl.enable(glow::BLEND);
            // The color by what the surface covers, and the frame's alpha left as it stands: that
            // channel holds the share of a pixel the composite counted as glare, not an opacity,
            // and a coverage written there blooms every strand the opaque half clipped away.
            gl.blend_func_separate(
                glow::SRC_ALPHA,
                glow::ONE_MINUS_SRC_ALPHA,
                glow::ZERO,
                glow::ONE,
            );
        }
        for (at, _, resolve) in &held {
            let surface = &surfaces[*at];
            let Some(mesh) = meshes.get(*at) else {
                continue;
            };
            let program = deferred::link(
                gl,
                &mut self.programs,
                (surface.material, false, SHEER_RESOLVE),
                resolve,
            )?;
            unsafe {
                gl.use_program(Some(program));
                match surface.cull {
                    true => {
                        gl.enable(glow::CULL_FACE);
                        gl.cull_face(glow::BACK);
                        gl.front_face(glow::CCW);
                    }
                    false => gl.disable(glow::CULL_FACE),
                }
            }
            self.bind(gl, painter, program, resolve, surface, *at, mesh, buffers, scene)?;
        }
        unsafe {
            gl.disable(glow::BLEND);
            gl.disable(glow::CULL_FACE);
        }
        Ok(())
    }

    /// What one draw of one material binds, and the geometry it covers. A texture the material has
    /// nothing for is the frame's own where the graph holds one under that name, and the flat
    /// stand-in otherwise.
    #[allow(clippy::too_many_arguments)]
    fn bind(
        &mut self,
        gl: &glow::Context,
        painter: &egui_glow::Painter,
        program: glow::Program,
        held: &program::Program,
        surface: &Surface,
        at: usize,
        mesh: &Buffers,
        buffers: &mut deferred::Buffers,
        scene: &program::Scene,
    ) -> Result<(), String> {
        let shaded = surface.shaded.as_ref().ok_or("nothing to draw with")?;
        let palette = *self.joints.get(at).ok_or("no joint palette")?;
        let layout = self.layout(gl)?;
        buffers.bind(gl, program, held, scene, &[])?;
        // Before anything is bound: making a texture binds it to whichever unit happens to be
        // active, so one made partway through the loop below takes over the unit the sampler
        // before it was just given.
        let table = match &shaded.table {
            Some(table) => Some(self.table(gl, surface.material, table)?),
            None => None,
        };
        let mut unit = 0;
        for texture in &held.textures {
            let mut aniso = 0.0;
            let bound = match texture.kind {
                program::Kind::Plane => {
                    let held = match texture.id {
                        TABLE => table,
                        id => {
                            let plane = shaded.bound(id).and_then(Bound::plane);
                            aniso = plane.map_or(0.0, |(_, aniso)| aniso);
                            plane.and_then(|(held, _)| painter.texture(held))
                        }
                    };
                    match held {
                        Some(held) => held,
                        None => buffers.engine(gl, texture.id)?,
                    }
                }
                kind => match shaded
                    .bound(texture.id)
                    .and_then(Bound::stacked)
                    .and_then(|path| buffers.stacked(kind, path))
                {
                    Some(held) => held,
                    None => buffers.absent(gl, kind, texture.id)?,
                },
            };
            deferred::bind(
                gl,
                program,
                &texture.name,
                unit,
                bound,
                deferred::target(texture.kind),
                aniso,
            );
            unit += 1;
        }
        // By name, not by position: a character's buffer pass reads the joint palette and the
        // shader-type table both, and they hold different things.
        for structured in &held.structured {
            let bound = match structured.name.as_str() {
                TYPES => buffers.types(gl)?,
                _ => palette,
            };
            sampler(gl, program, &structured.name, unit, bound);
            unit += 1;
        }
        unsafe {
            gl.bind_vertex_array(Some(layout));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(mesh.vertices));
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(mesh.indices));
            for location in 0..16 {
                gl.disable_vertex_attrib_array(location);
            }
            for held in &held.attributes {
                attribute(gl, held);
            }
            for run in &surface.runs {
                gl.draw_elements(
                    glow::TRIANGLES,
                    run.end - run.start,
                    glow::UNSIGNED_SHORT,
                    run.start * size_of::<u16>() as i32,
                );
            }
            gl.bind_vertex_array(None);
        }
        Ok(())
    }
}

impl Drop for Game {
    fn drop(&mut self) {
        let mut dead = graveyard().lock().unwrap();
        dead.extend(self.tables.values().map(|(texture, _)| Dead::Texture(*texture)));
        dead.extend(
            std::mem::take(&mut self.joints)
                .into_iter()
                .map(Dead::Texture),
        );
        dead.extend(self.layout.take().map(Dead::Layout));
        dead.extend(
            std::mem::take(&mut self.programs)
                .into_values()
                .map(|held| Dead::Program(held.program)),
        );
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        graveyard().lock().unwrap().extend(
            self.meshes
                .drain(..)
                .flat_map(|held| {
                    [
                        Dead::Layout(held.layout),
                        Dead::Buffer(held.vertices),
                        Dead::Buffer(held.indices),
                    ]
                })
                .chain(
                    std::mem::take(&mut self.tables)
                        .into_values()
                        .map(|(texture, _)| Dead::Texture(texture)),
                )
                .chain(self.program.take().map(Dead::Program)),
        );
    }
}

/// Points one attribute at the field of a [`Vertex`] its semantic names. The mesh keeps its
/// influences unsigned and a shader declares them either way, so the pointer's own type follows the
/// signature: a draw is rejected outright where the two differ in class or in sign.
/// The game's own textures a model has been handed, the ones its materials name that egui cannot
/// hold, and the table the shading passes index.
type Supplied = (Vec<(u32, Layered)>, Vec<(Arc<str>, Layered)>, Option<Vec<u32>>);

/// Those, uploaded into whichever frame the model is being drawn into.
fn supply(gl: &glow::Context, buffers: &mut deferred::Buffers, (arrays, stacks, types): Supplied) {
    for (id, held) in arrays {
        if let Err(why) = buffers.layered(gl, id, &held) {
            log::error!("assets/mdl: texture array {id:#010x}: {why}");
        }
    }
    for (path, held) in stacks {
        if let Err(why) = buffers.stack(gl, &path, &held) {
            log::error!("assets/mdl: {path}: {why}");
        }
    }
    if let Some(values) = types
        && let Err(why) = buffers.fill_types(gl, &values)
    {
        log::error!("assets/mdl: shader types: {why}");
    }
}

pub fn attribute(gl: &glow::Context, held: &program::Attribute) {
    let Some((_, lanes, offset, kind)) = FIELDS.iter().find(|(field, ..)| *field == held.field)
    else {
        return;
    };
    let stride = size_of::<Vertex>() as i32;
    unsafe {
        gl.enable_vertex_attrib_array(held.location);
        match held.components {
            program::Components::Float => gl.vertex_attrib_pointer_f32(
                held.location,
                *lanes,
                *kind,
                *kind == glow::UNSIGNED_BYTE,
                stride,
                *offset,
            ),
            program::Components::Unsigned => {
                gl.vertex_attrib_pointer_i32(held.location, *lanes, *kind, stride, *offset)
            }
            program::Components::Signed => gl.vertex_attrib_pointer_i32(
                held.location,
                *lanes,
                match *kind {
                    glow::UNSIGNED_BYTE => glow::BYTE,
                    _ => glow::SHORT,
                },
                stride,
                *offset,
            ),
        }
    }
}

/// One mesh's buffers, with the attribute layout captured in a vertex array of its own.
///
/// The array is not an optimisation. egui leaves its own vertex array bound while a callback runs,
/// so setting attribute pointers without one would rewrite egui's layout to point at model
/// geometry, and every widget drawn afterwards would read vertices out of this mesh.
fn upload_mesh(
    gl: &glow::Context,
    vertices: &[Vertex],
    indices: &[u16],
) -> Result<Buffers, String> {
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

        let stride = size_of::<Vertex>() as i32;
        for (location, size, offset) in ATTRIBUTES {
            gl.enable_vertex_attrib_array(location);
            gl.vertex_attrib_pointer_f32(location, size, glow::FLOAT, false, stride, offset);
        }
        gl.enable_vertex_attrib_array(COLOR);
        gl.vertex_attrib_pointer_f32(COLOR, 4, glow::UNSIGNED_BYTE, true, stride, COLOR_OFFSET);
        for (location, offset) in INFLUENCES {
            gl.enable_vertex_attrib_array(location);
            gl.vertex_attrib_pointer_i32(location, 4, glow::UNSIGNED_SHORT, stride, offset);
        }

        let drawn = gl.create_buffer()?;
        gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(drawn));
        gl.buffer_data_u8_slice(
            glow::ELEMENT_ARRAY_BUFFER,
            bytemuck::cast_slice(indices),
            glow::STATIC_DRAW,
        );

        gl.bind_vertex_array(None);
        gl.bind_buffer(glow::ARRAY_BUFFER, None);
        let (low, high) = vertices.iter().fold(
            (glam::Vec3::INFINITY, glam::Vec3::NEG_INFINITY),
            |(low, high), vertex| {
                let held = glam::Vec3::from_array(vertex.position);
                (low.min(held), high.max(held))
            },
        );
        Ok(Buffers {
            layout,
            vertices: held,
            indices: drawn,
            center: (low + high) * 0.5,
        })
    }
}

/// The color table, one RGBA texel per field group. Point sampled: the row pair is mixed in the
/// shader rather than by the sampler, so a row's own values stay exact.
fn upload_table(gl: &glow::Context, values: &[f32], rows: i32) -> Result<glow::Texture, String> {
    unsafe {
        let texture = gl.create_texture()?;
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            glow::RGBA16F as i32,
            TABLE_COLUMNS,
            rows,
            0,
            glow::RGBA,
            glow::FLOAT,
            glow::PixelUnpackData::Slice(Some(bytemuck::cast_slice(values))),
        );
        for (name, value) in [
            (glow::TEXTURE_MIN_FILTER, glow::NEAREST),
            (glow::TEXTURE_MAG_FILTER, glow::NEAREST),
            (glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE),
            (glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE),
        ] {
            gl.tex_parameter_i32(glow::TEXTURE_2D, name, value as i32);
        }
        Ok(texture)
    }
}

fn build(gl: &glow::Context) -> Result<glow::Program, String> {
    build_pair(gl, VERTEX_SOURCE, FRAGMENT_SOURCE)
}
