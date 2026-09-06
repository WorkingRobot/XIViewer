//! The GL side of the effect viewer.
//!
//! Everything runs inside an [`egui_glow`] paint callback, uploads happen on the first frame that
//! draws them, and dead objects go to the same graveyard the model viewer uses. What each particle is
//! shaded by is the game's own pair, translated on demand: a sprite goes through `apricot_shape`,
//! which reads a stream the viewer has already placed in the world, and a model through
//! `apricot_model`, which reads the effect's own mesh with a transform buffer beside it.
//!
//! Both write two targets. The first is the color the frame is blended with; the second is a depth
//! of field coefficient and a weight, which nothing downstream of a viewer reads, so it is written
//! into an attachment of its own and dropped.

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::sync::{Arc, Mutex};

use egui::TextureId;
use glow::HasContext;

use super::super::mdl::gpu::{Dead, bury, graveyard};
use super::program::{self, Field, Instance, Program, Scene};
use super::sim::{Blend, Mesh, Shading, Shape, Vertex};

/// One vertex of the stream the sprite packages read. The color and the uv sets are integers the
/// shader scales by a thousandth, which is what its precision key means.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Sprite {
    /// Where the corner stands in the world, and how far towards the camera its depth is pulled.
    pub position: [f32; 4],
    pub color: [i16; 4],
    pub uv01: [i16; 4],
    pub uv23: [i16; 4],
    pub extra: [f32; 4],
}

/// Everything drawn under one particle definition and one blend at once.
#[derive(Clone)]
pub struct Batch {
    pub shape: Shape,
    /// The effect's textures, in the order it lists them.
    pub textures: Vec<Option<TextureId>>,
    pub blend: Blend,
    pub def: usize,
    pub shading: Arc<Shading>,
    /// The corners the sprite packages draw, already in the world.
    pub vertices: Vec<Sprite>,
    /// One record per model drawn, which the model packages read one of per draw.
    pub instances: Vec<Instance>,
}

/// One frame of camera and batches, rebuilt every time the widget draws.
pub struct Frame {
    pub scene: Scene,
    pub batches: Vec<Batch>,
    /// The packages the batches resolve against, once they have been fetched.
    pub packages: Arc<Packages>,
    /// Tested against whatever depth the caller already bound, rather than disabled outright: the
    /// standalone preview draws into a frame with none, and a zone draws into the one its own
    /// geometry left, so a glow standing behind a wall does not show through it.
    pub tested: bool,
    /// A copy of the opaque depth, for the soft-particle apricot_model variant that samples it.
    /// Never the live attachment `tested` compares against: reading and writing the same depth in
    /// one draw is the feedback loop `blended()` once fell into. `None` where the caller has no
    /// scene depth to copy, which leaves the sampler unbound.
    pub depth: Option<glow::Texture>,
}

/// The two apricot packages an effect is drawn with.
#[derive(Default)]
pub struct Packages {
    pub shape: Option<Vec<u8>>,
    pub model: Option<Vec<u8>>,
}

struct Buffers {
    layout: glow::VertexArray,
    vertices: glow::Buffer,
    indices: glow::Buffer,
    count: i32,
}

/// A linked pair, kept against the source it was built from so a change rebuilds it rather than a
/// stale program drawing on.
struct Linked {
    program: glow::Program,
    held: Program,
}

pub struct Particles {
    pending: Option<Vec<Mesh>>,
    /// One entry per model the effect carries, uploaded once.
    models: Vec<Buffers>,
    /// The stream the sprite packages draw, rewritten every frame.
    stream: Option<(glow::VertexArray, glow::Buffer)>,
    capacity: usize,
    /// One linked program per particle definition, since the keys are the particle's own.
    programs: BTreeMap<usize, Linked>,
    /// The uniform blocks a draw binds, one per buffer the shader names.
    blocks: Vec<glow::Buffer>,
    /// Why a shader would not build, kept so the viewer can say so rather than draw nothing.
    failure: Option<String>,
}

impl Particles {
    pub fn new(models: Vec<Mesh>) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self::new_plain(models)))
    }

    /// Unwrapped, for a caller that already runs behind its own lock: a zone's renderer holds one
    /// per placed file directly rather than through a second one of its own.
    pub(crate) fn new_plain(models: Vec<Mesh>) -> Self {
        Self {
            pending: Some(models),
            models: Vec::new(),
            stream: None,
            capacity: 0,
            programs: BTreeMap::new(),
            blocks: Vec::new(),
            failure: None,
        }
    }

    pub fn failure(&self) -> Option<&str> {
        self.failure.as_deref()
    }

    pub fn draw(&mut self, gl: &glow::Context, painter: &egui_glow::Painter, frame: &Frame) {
        bury(gl);
        if self.failure.is_some() {
            return;
        }
        if let Some(pending) = self.pending.take()
            && let Err(why) = self.upload(gl, pending)
        {
            self.failure = Some(why);
            return;
        }
        if let Err(why) = self.render(gl, painter, frame) {
            log::error!("assets/avfx: {why}");
            self.failure = Some(why);
        }
    }

    fn render(
        &mut self,
        gl: &glow::Context,
        painter: &egui_glow::Painter,
        frame: &Frame,
    ) -> Result<(), String> {
        // Both packages write a second target: a depth of field coefficient and a weight, neither of
        // which anything downstream of a viewer reads. What egui hands the callback may be
        // multisampled, so nothing is attached to it and the write goes nowhere.
        unsafe {
            match frame.tested {
                true => {
                    gl.enable(glow::DEPTH_TEST);
                    gl.depth_func(glow::LEQUAL);
                }
                false => gl.disable(glow::DEPTH_TEST),
            }
            gl.depth_mask(false);
            gl.disable(glow::CULL_FACE);
        }

        let mut stream: Vec<Sprite> = Vec::new();
        for batch in &frame.batches {
            stream.extend_from_slice(&batch.vertices);
        }
        if !stream.is_empty() {
            self.upload_stream(gl, &stream)?;
        }

        let mut at = 0;
        for batch in &frame.batches {
            let vertices = batch.vertices.len();
            if let Err(why) = self.batch(gl, painter, frame, batch, at) {
                log::warn!("assets/avfx: particle {}: {why}", batch.def);
            }
            at += vertices;
        }

        unsafe {
            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            gl.blend_equation(glow::FUNC_ADD);
            gl.disable(glow::BLEND);
            gl.depth_mask(false);
        }
        Ok(())
    }

    /// One particle definition drawn: its own pair, linked once and kept, with the buffers it names
    /// filled and the effect's textures bound to the samplers it asks for by name.
    fn batch(
        &mut self,
        gl: &glow::Context,
        painter: &egui_glow::Painter,
        frame: &Frame,
        batch: &Batch,
        at: usize,
    ) -> Result<(), String> {
        let sprite = batch.shape == Shape::Sprite;
        let shading = &batch.shading;
        let bytes = match sprite {
            true => frame.packages.shape.as_ref(),
            false => frame.packages.model.as_ref(),
        }
        .ok_or("the package has not arrived")?;

        if let Entry::Vacant(slot) = self.programs.entry(batch.def) {
            let keys = [shading.keys.clone(), shading.lights.clone()].concat();
            let held = Program::build(bytes, &keys, shading.sprite)?;
            let program = build_pair(gl, &held.vertex, &held.fragment)?;
            slot.insert(Linked { program, held });
        }
        let linked = &self.programs[&batch.def];
        let program = linked.program;
        unsafe { gl.use_program(Some(program)) };

        for (unit, texture) in linked.held.textures.iter().enumerate() {
            // The engine's own depth, not a particle's: nothing in the effect names it, so it is
            // matched by id and bound from the caller rather than looked up in `shading.textures`.
            if texture.id == program::id("g_SamplerDepth") {
                unsafe {
                    gl.active_texture(glow::TEXTURE0 + unit as u32);
                    gl.bind_texture(glow::TEXTURE_2D, frame.depth);
                    if let Some(location) = gl.get_uniform_location(program, &texture.name) {
                        gl.uniform_1_i32(Some(&location), unit as i32);
                    }
                }
                continue;
            }
            let held = shading
                .textures
                .iter()
                .find(|(id, ..)| *id == texture.id)
                .and_then(|(_, index, filter, wrap)| {
                    let held = batch.textures.get(*index).copied().flatten()?;
                    Some((painter.texture(held)?, *filter, *wrap))
                });
            unsafe {
                gl.active_texture(glow::TEXTURE0 + unit as u32);
                gl.bind_texture(glow::TEXTURE_2D, held.map(|(texture, ..)| texture));
                if let Some((_, filter, wrap)) = held {
                    for (name, value) in [
                        (glow::TEXTURE_MIN_FILTER, filter),
                        (glow::TEXTURE_MAG_FILTER, filter),
                        (glow::TEXTURE_WRAP_S, wrap[0]),
                        (glow::TEXTURE_WRAP_T, wrap[1]),
                    ] {
                        gl.tex_parameter_i32(glow::TEXTURE_2D, name, value as i32);
                    }
                }
                if let Some(location) = gl.get_uniform_location(program, &texture.name) {
                    gl.uniform_1_i32(Some(&location), unit as i32);
                }
                if let Some(location) =
                    gl.get_uniform_location(program, &format!("{}_levels", texture.name))
                {
                    gl.uniform_1_i32(Some(&location), 1);
                }
            }
        }

        blend(gl, batch.blend);
        match sprite {
            true => self.draw_stream(gl, batch, at, frame),
            false => self.draw_models(gl, batch, frame),
        }
    }

    /// Every corner of every sprite under one definition, drawn as one call out of the stream the
    /// whole frame was written into.
    fn draw_stream(
        &mut self,
        gl: &glow::Context,
        batch: &Batch,
        at: usize,
        frame: &Frame,
    ) -> Result<(), String> {
        if batch.vertices.is_empty() {
            return Ok(());
        }
        let Some((layout, buffer)) = self.stream else {
            return Ok(());
        };
        let linked = &self.programs[&batch.def];
        unsafe {
            gl.bind_vertex_array(Some(layout));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(buffer));
        }
        let stride = size_of::<Sprite>() as i32;
        let base = at as i32 * stride;
        for held in &linked.held.attributes {
            let (size, kind, offset, integer) = match held.field {
                Field::Position => (4, glow::FLOAT, 0, false),
                Field::Color => (4, glow::SHORT, 16, held.integer),
                Field::Uv01 => (4, glow::SHORT, 24, held.integer),
                Field::Uv23 => (4, glow::SHORT, 32, held.integer),
                _ => (4, glow::FLOAT, 40, false),
            };
            unsafe {
                gl.enable_vertex_attrib_array(held.location);
                match integer {
                    true => gl.vertex_attrib_pointer_i32(
                        held.location,
                        size,
                        kind,
                        stride,
                        base + offset,
                    ),
                    false => gl.vertex_attrib_pointer_f32(
                        held.location,
                        size,
                        kind,
                        false,
                        stride,
                        base + offset,
                    ),
                }
            }
        }
        let instance = Instance {
            calculate: batch.shading.calculate,
            depth_offset: batch.shading.depth_offset,
            ..Instance::default()
        };
        self.bind(gl, batch.def, &frame.scene, &instance)?;
        unsafe { gl.draw_arrays(glow::TRIANGLES, 0, batch.vertices.len() as i32) };
        Ok(())
    }

    /// One draw per model drawn, because the package reads one transform at a time: its instance
    /// buffer holds one record and its vertex shader has no instance index to pick another with.
    fn draw_models(
        &mut self,
        gl: &glow::Context,
        batch: &Batch,
        frame: &Frame,
    ) -> Result<(), String> {
        let Shape::Model(model) = batch.shape else {
            return Ok(());
        };
        let Some(buffers) = self.models.get(model) else {
            return Ok(());
        };
        let (layout, vertices, count) = (buffers.layout, buffers.vertices, buffers.count);
        let linked = &self.programs[&batch.def];
        unsafe {
            gl.bind_vertex_array(Some(layout));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vertices));
        }
        let stride = size_of::<Vertex>() as i32;
        for held in &linked.held.attributes {
            let (size, kind, normalize, offset) = match held.field {
                Field::Position => (4, glow::FLOAT, false, 0),
                Field::Normal => (4, glow::UNSIGNED_BYTE, true, 16),
                Field::Tangent => (4, glow::UNSIGNED_BYTE, true, 20),
                Field::Color => (4, glow::UNSIGNED_BYTE, true, 24),
                Field::Uv01 => (4, glow::FLOAT, false, 28),
                _ => (4, glow::FLOAT, false, 44),
            };
            unsafe {
                gl.enable_vertex_attrib_array(held.location);
                gl.vertex_attrib_pointer_f32(held.location, size, kind, normalize, stride, offset);
            }
        }
        for instance in &batch.instances {
            let instance = Instance {
                calculate: batch.shading.calculate,
                ..*instance
            };
            self.bind(gl, batch.def, &frame.scene, &instance)?;
            unsafe {
                gl.draw_elements(glow::TRIANGLES, count, glow::UNSIGNED_SHORT, 0);
            }
        }
        Ok(())
    }

    /// The uniform blocks one draw reads, filled and bound.
    fn bind(
        &mut self,
        gl: &glow::Context,
        def: usize,
        scene: &Scene,
        instance: &Instance,
    ) -> Result<(), String> {
        let linked = &self.programs[&def];
        let program = linked.program;
        for (at, buffer) in linked.held.buffers.iter().enumerate() {
            let Some(block) =
                (unsafe { gl.get_uniform_block_index(program, &format!("{}_b", buffer.name)) })
            else {
                continue;
            };
            let mut data = buffer.fill(scene, instance);
            unsafe {
                let size = gl.get_active_uniform_block_parameter_i32(
                    program,
                    block,
                    glow::UNIFORM_BLOCK_DATA_SIZE,
                ) as usize;
                data.resize(size.max(16), 0);
                while self.blocks.len() <= at {
                    self.blocks.push(gl.create_buffer()?);
                }
                let held = self.blocks[at];
                gl.bind_buffer(glow::UNIFORM_BUFFER, Some(held));
                gl.buffer_data_u8_slice(glow::UNIFORM_BUFFER, &data, glow::DYNAMIC_DRAW);
                gl.bind_buffer_base(glow::UNIFORM_BUFFER, at as u32, Some(held));
                gl.uniform_block_binding(program, block, at as u32);
            }
        }
        Ok(())
    }

    fn upload_stream(&mut self, gl: &glow::Context, stream: &[Sprite]) -> Result<(), String> {
        unsafe {
            let (layout, buffer) = match self.stream {
                Some(held) => held,
                None => {
                    let layout = gl.create_vertex_array()?;
                    let buffer = gl.create_buffer()?;
                    *self.stream.insert((layout, buffer))
                }
            };
            gl.bind_vertex_array(Some(layout));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(buffer));
            let bytes: &[u8] = bytemuck::cast_slice(stream);
            match stream.len() <= self.capacity {
                true => gl.buffer_sub_data_u8_slice(glow::ARRAY_BUFFER, 0, bytes),
                false => {
                    gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytes, glow::STREAM_DRAW);
                    self.capacity = stream.len();
                }
            }
        }
        Ok(())
    }

    fn upload(&mut self, gl: &glow::Context, meshes: Vec<Mesh>) -> Result<(), String> {
        log::info!("assets/avfx: {} models on {:?}", meshes.len(), gl.version());
        for mesh in &meshes {
            self.models.push(upload(gl, mesh)?);
        }
        Ok(())
    }
}

impl Drop for Particles {
    fn drop(&mut self) {
        graveyard().lock().unwrap().extend(
            self.models
                .drain(..)
                .flat_map(|held| {
                    [
                        Dead::Layout(held.layout),
                        Dead::Buffer(held.vertices),
                        Dead::Buffer(held.indices),
                    ]
                })
                .chain(
                    self.stream
                        .take()
                        .into_iter()
                        .flat_map(|(layout, buffer)| [Dead::Layout(layout), Dead::Buffer(buffer)]),
                )
                .chain(self.blocks.drain(..).map(Dead::Buffer))
                .chain(
                    std::mem::take(&mut self.programs)
                        .into_values()
                        .map(|held| Dead::Program(held.program)),
                ),
        );
    }
}

/// How a blend's source and destination are weighted. Both packages hand over a color the shader has
/// not scaled by its own opacity, so the source factor carries it. The destination alpha is left
/// alone rather than blended into: in a zone that channel is the share of a pixel the composite
/// counted as glare, and a particle drawn over it must not rewrite that mask.
fn blend(gl: &glow::Context, blend: Blend) {
    unsafe {
        gl.blend_equation(match blend {
            Blend::Subtract => glow::FUNC_REVERSE_SUBTRACT,
            _ => glow::FUNC_ADD,
        });
        match blend {
            Blend::Opaque => {
                gl.disable(glow::BLEND);
                return;
            }
            Blend::Alpha => gl.blend_func_separate(
                glow::SRC_ALPHA,
                glow::ONE_MINUS_SRC_ALPHA,
                glow::ZERO,
                glow::ONE,
            ),
            Blend::Multiply => {
                gl.blend_func_separate(glow::ZERO, glow::SRC_COLOR, glow::ZERO, glow::ONE)
            }
            Blend::Screen => {
                gl.blend_func_separate(glow::ONE, glow::ONE_MINUS_SRC_COLOR, glow::ZERO, glow::ONE)
            }
            Blend::Subtract | Blend::Add => {
                gl.blend_func_separate(glow::SRC_ALPHA, glow::ONE, glow::ZERO, glow::ONE)
            }
        }
        gl.enable(glow::BLEND);
    }
}

/// One model's buffers, with its own vertex array: egui leaves its own bound while a callback runs,
/// so setting attribute pointers without one would rewrite egui's layout.
fn upload(gl: &glow::Context, mesh: &Mesh) -> Result<Buffers, String> {
    unsafe {
        let layout = gl.create_vertex_array()?;
        gl.bind_vertex_array(Some(layout));

        let vertices = gl.create_buffer()?;
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(vertices));
        gl.buffer_data_u8_slice(
            glow::ARRAY_BUFFER,
            bytemuck::cast_slice(&mesh.vertices),
            glow::STATIC_DRAW,
        );

        let indices = gl.create_buffer()?;
        gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(indices));
        gl.buffer_data_u8_slice(
            glow::ELEMENT_ARRAY_BUFFER,
            bytemuck::cast_slice(&mesh.indices),
            glow::STATIC_DRAW,
        );

        gl.bind_vertex_array(None);
        gl.bind_buffer(glow::ARRAY_BUFFER, None);
        Ok(Buffers {
            layout,
            vertices,
            indices,
            count: mesh.indices.len() as i32,
        })
    }
}

fn build_pair(gl: &glow::Context, vertex: &str, fragment: &str) -> Result<glow::Program, String> {
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
                return Err(why);
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
            return Err(why);
        }
        Ok(program)
    }
}

/// The corners of one sprite, in the world: the quad the viewer bills against the camera, at the
/// particle's own place and turn, with its color and uv sets written as the fixed point the shader
/// reads them back through.
///
/// The sprite packages put no transform on a texture coordinate, so each set arrives transformed,
/// over a corner centered the way an effect's own models write theirs.
pub fn quad(
    center: glam::Vec3,
    right: glam::Vec3,
    up: glam::Vec3,
    color: [f32; 4],
    uv: &[[f32; 4]; program::UV_SETS * program::UV_REGISTERS],
    into: &mut Vec<Sprite>,
) {
    let fixed = |value: f32| (value * program::FIXED).clamp(-32767.0, 32767.0) as i16;
    let tint = [
        fixed(color[0]),
        fixed(color[1]),
        fixed(color[2]),
        fixed(color[3]),
    ];
    let corner = |x: f32, y: f32| {
        let (u, v) = (x, -y);
        let set = |at: usize| {
            let rows = &uv[at * program::UV_REGISTERS..];
            [
                fixed(rows[0][0] * u + rows[0][1] * v + rows[0][3]),
                fixed(rows[1][0] * u + rows[1][1] * v + rows[1][3]),
            ]
        };
        let (first, second, third, fourth) = (set(0), set(1), set(2), set(3));
        Sprite {
            position: [
                center.x + right.x * x + up.x * y,
                center.y + right.y * x + up.y * y,
                center.z + right.z * x + up.z * y,
                0.0,
            ],
            color: tint,
            uv01: [first[0], first[1], second[0], second[1]],
            uv23: [third[0], third[1], fourth[0], fourth[1]],
            extra: [0.0; 4],
        }
    };
    let corners = [
        corner(-0.5, -0.5),
        corner(0.5, -0.5),
        corner(0.5, 0.5),
        corner(-0.5, 0.5),
    ];
    into.extend([
        corners[0], corners[1], corners[2], corners[0], corners[2], corners[3],
    ]);
}

#[cfg(test)]
mod test {
    use super::{program, quad};

    #[test]
    fn a_uv_set_reaches_the_corners_it_is_written_into() {
        let mut uv = program::UV_IDENTITY;
        uv[0] = [2.0, 0.0, 0.0, 0.5];
        uv[1] = [0.0, 2.0, 0.0, 0.5];
        let mut into = Vec::new();
        quad(
            glam::Vec3::ZERO,
            glam::Vec3::X,
            glam::Vec3::Y,
            [1.0; 4],
            &uv,
            &mut into,
        );
        // The first corner takes the bottom left, which the second set leaves where it was.
        assert_eq!(into[0].uv01, [-500, 1500, 0, 1000]);
    }

    /// A half-scale set covers the middle half of its texture, which is what the game bakes into a
    /// sprite's own corners: elpfall draw 61053 reads 0.25 and 0.75 down every quad it draws.
    #[test]
    fn a_half_scale_set_bakes_the_middle_of_the_texture() {
        let mut uv = program::UV_IDENTITY;
        uv[0] = [0.5, 0.0, 0.0, 0.5];
        uv[1] = [0.0, 0.5, 0.0, 0.5];
        let mut into = Vec::new();
        quad(
            glam::Vec3::ZERO,
            glam::Vec3::X,
            glam::Vec3::Y,
            [1.0; 4],
            &uv,
            &mut into,
        );
        assert_eq!([into[0].uv01[0], into[0].uv01[1]], [250, 750]);
        assert_eq!([into[2].uv01[0], into[2].uv01[1]], [750, 250]);
    }
}
