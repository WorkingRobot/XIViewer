//! Baseline glTF export: the geometry, materials and skeleton of a model exactly as it currently
//! draws, posed to whatever the animation tab has it standing in.
//!
//! Split into a sync core (`gather`, `bake`, `assemble`) and a thin async wrapper (`finish`) so the
//! geometry and GLB-writing logic can be exercised without a `Backend`.

use std::collections::{BTreeMap, HashMap};

use anyhow::{Context, Result, bail};
use glam::{Mat4, Vec3};
use image::imageops::{self, FilterType};
use image::{DynamicImage, GenericImageView, ImageFormat, Rgba, RgbaImage};
use ironworks::file::mdl::VertexAttributeKind;
use serde_json::{Value, json};

use super::material::{Family, Material, Role};
use super::program;
use super::{Rendered, Slot, Vertex, build, detail, draws};
use crate::data::FileProvider;

/// The long edge a baked material texture is capped to. A dressed character bakes up to four of
/// these per material, and wasm's 32-bit address space is the tighter of the two limits.
const MAX_TEXTURE_DIM: u32 = 1024;

pub(super) struct Scene {
    pieces: Vec<PieceMesh>,
    materials: Vec<MaterialInfo>,
    skeleton: Option<Skeleton>,
    stature: f32,
    /// The wearer's own colours, which most character packages tint their albedo with and no
    /// texture holds.
    customize: program::Customize,
}

struct PieceMesh {
    name: String,
    primitives: Vec<Primitive>,
}

struct Primitive {
    material: usize,
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    /// Absent where the source mesh declares no binormal at all, since deriving one would be
    /// invented rather than read.
    tangents: Option<Vec<[f32; 4]>>,
    uv0: Vec<[f32; 2]>,
    colors: Vec<[u8; 4]>,
    joints0: Option<Vec<[u32; 4]>>,
    weights0: Option<Vec<[f32; 4]>>,
    joints1: Option<Vec<[u32; 4]>>,
    weights1: Option<Vec<[f32; 4]>>,
    indices: Vec<u32>,
}

struct MaterialInfo {
    name: String,
    /// Whether the package tints what stands behind rather than covering it, which is a surface
    /// that has to blend however opaque its own alpha reads.
    glass: bool,
    /// The package as the file names it, which per-package shading keys on: the channel a mask
    /// means is stated by the package, not by the family.
    package: String,
    family: Family,
    alpha_threshold: f32,
    cull: bool,
    textures: [Option<String>; 4],
    table: Option<Vec<f32>>,
    diffuse_color: [f32; 3],
    emissive_color: [f32; 3],
    normal_scale: f32,
}

struct Skeleton {
    names: Vec<String>,
    parents: Vec<Option<usize>>,
    /// Every bone's placement this frame, in the model's own space.
    world: Vec<Mat4>,
    /// The matrix that carries a bind-pose vertex into each bone's own rest frame.
    rest_inverse: Vec<Mat4>,
}

/// Gathers geometry, materials and the current pose off a model exactly as its own side panel has
/// it: the detail level on screen, the parts the user has toggled, whatever shape keys are active.
/// Touches no network; a material this reads has already had to finish loading to be drawn at all.
pub(super) fn gather(rendered: &Rendered) -> Result<Scene> {
    if rendered.animation.rides().is_some() {
        bail!("exporting a mounted character is not supported");
    }

    let customize = rendered.customize.get();
    let level = rendered.level.borrow();
    let enabled_shapes = rendered.shapes.borrow();
    let slots = rendered.slots.borrow();

    let mut materials = Vec::with_capacity(level.materials.len());
    for (index, path) in level.materials.iter().enumerate() {
        match slots.get(index) {
            Some(Some(Slot::Ready(material))) => materials.push(material_info(path, material)),
            // A material can legitimately be absent from the install (a stated variant with no file
            // behind it), which is not reason enough to fail an export of everything else that did
            // load: draw that one piece untextured instead.
            Some(Some(Slot::Failed(why))) => {
                log::warn!(
                    "assets/mdl: export: material {path} failed to load: {why}, exporting untextured"
                );
                materials.push(placeholder_material(path));
            }
            _ => bail!("material {path} has not finished loading yet"),
        }
    }

    let skeleton = match level.skinned {
        true => {
            let (names, parents, rest_inverse) = rendered
                .animation
                .rig()
                .context("the skeleton has not finished loading yet")?;
            let pose = rendered.animation.pose(&[], &[], false);
            if pose.world.len() != names.len() {
                bail!("the skeleton has not finished loading yet");
            }
            Some(Skeleton {
                names,
                parents,
                world: pose.world,
                rest_inverse,
            })
        }
        false => None,
    };
    let joints = skeleton.as_ref().map(|skeleton| {
        skeleton
            .names
            .iter()
            .enumerate()
            .map(|(bone, name)| (name.as_str(), bone as u32))
            .collect::<HashMap<_, _>>()
    });
    let fallback_joint = skeleton.as_ref().map_or(0, |skeleton| skeleton.names.len() as u32);

    let mut pieces: Vec<PieceMesh> = rendered
        .pieces
        .iter()
        .map(|piece| PieceMesh {
            name: piece_name(&piece.path),
            primitives: Vec::new(),
        })
        .collect();

    let mut missing = 0usize;
    let mut wanted = 0usize;
    let mut level_index = 0usize;

    for (piece_index, piece) in rendered.pieces.iter().enumerate() {
        let model = piece.container.model(detail(rendered.lod.get()));
        let bone_names = model.bone_names().unwrap_or_default();
        for mesh in model.meshes() {
            if !draws(&mesh) {
                continue;
            }
            let Ok(attributes) = mesh.attributes() else {
                continue;
            };
            let Ok(mesh_indices) = mesh.indices() else {
                continue;
            };
            let Ok((vertices, mut indices)) = build(&attributes, mesh_indices) else {
                continue;
            };
            let mesh_skinned = attributes
                .iter()
                .any(|attribute| attribute.kind as u8 == VertexAttributeKind::BlendIndices as u8);
            let tangent_present = attributes
                .iter()
                .any(|attribute| attribute.kind as u8 == VertexAttributeKind::Tangent1 as u8);

            let table: Vec<String> = mesh
                .bone_table()
                .iter()
                .map(|bone| bone_names.get(usize::from(*bone)).cloned().unwrap_or_default())
                .collect();
            let mut vertices = vertices;
            if let Some(deform) = piece.deform.as_deref() {
                deform.apply(&mut vertices, &table);
            }

            let Some(level_mesh) = level.meshes.get(level_index) else {
                bail!("mesh accounting drifted out of sync with the model's own level");
            };
            if !level_mesh.base.is_empty() {
                let mut patched = level_mesh.base.clone();
                for shape in level
                    .shapes
                    .iter()
                    .filter(|shape| enabled_shapes.contains(&shape.name))
                {
                    for (mesh_at, values) in &shape.rewrites {
                        if *mesh_at != level_index {
                            continue;
                        }
                        for (offset, vertex) in values {
                            if let Some(slot) = patched.get_mut(usize::from(*offset)) {
                                *slot = *vertex;
                            }
                        }
                    }
                }
                indices = patched;
            }

            let mut kept: Vec<u16> = Vec::new();
            for part in &level_mesh.parts {
                if part.shown.get() {
                    let Some(run) = indices.get(part.range.clone()) else {
                        bail!("a submesh names indices past the end of its own mesh");
                    };
                    kept.extend_from_slice(run);
                }
            }
            let material = level_mesh.material;
            level_index += 1;
            if kept.is_empty() {
                continue;
            }

            let mut remap: HashMap<u16, u32> = HashMap::new();
            let mut compact: Vec<Vertex> = Vec::new();
            let mut compact_indices: Vec<u32> = Vec::with_capacity(kept.len());
            for old in kept {
                let at = match remap.get(&old) {
                    Some(at) => *at,
                    None => {
                        let Some(vertex) = vertices.get(usize::from(old)) else {
                            bail!("an index names none of its mesh's own vertices");
                        };
                        compact.push(*vertex);
                        let at = (compact.len() - 1) as u32;
                        remap.insert(old, at);
                        at
                    }
                };
                compact_indices.push(at);
            }

            let primitive = primitive_from(
                &compact,
                compact_indices,
                material,
                tangent_present,
                (materials
                    .get(material)
                    .is_some_and(|held| held.package == "iris.shpk"))
                .then_some(&customize),
                mesh_skinned.then_some(&table),
                joints.as_ref(),
                fallback_joint,
                &mut missing,
                &mut wanted,
            );
            pieces[piece_index].primitives.push(primitive);
        }
    }
    if level_index != level.meshes.len() {
        bail!("walked {level_index} meshes but the model's own level built {}", level.meshes.len());
    }
    if wanted > 0 {
        log::info!("assets/mdl: export: {missing} of {wanted} bone influences are named by no skeleton");
    }

    Ok(Scene {
        pieces,
        materials,
        skeleton,
        stature: rendered.stature.get(),
        customize,
    })
}

fn piece_name(path: &str) -> String {
    let stem = crate::utils::file_name(path);
    stem.strip_suffix(".mdl")
        .or_else(|| stem.strip_suffix(".mtrl"))
        .unwrap_or(stem)
        .to_owned()
}

fn material_info(path: &str, material: &Material) -> MaterialInfo {
    MaterialInfo {
        name: piece_name(path),
        glass: material.glass().is_some(),
        package: material.shader().to_owned(),
        family: material.family(),
        alpha_threshold: material.alpha_threshold(),
        cull: material.cull(),
        textures: [Role::Normal, Role::Index, Role::Mask, Role::Diffuse]
            .map(|role| material.texture(role).cloned()),
        table: material.table().map(<[f32]>::to_vec),
        diffuse_color: material.diffuse(),
        emissive_color: material.emissive(),
        normal_scale: material.normal_scale(),
    }
}

/// What a material that failed to load exports as: flat grey, untextured, cut at nothing.
fn placeholder_material(path: &str) -> MaterialInfo {
    MaterialInfo {
        name: piece_name(path),
        glass: false,
        package: String::new(),
        family: Family::Character,
        alpha_threshold: 0.0,
        cull: true,
        textures: [None, None, None, None],
        table: None,
        diffuse_color: [0.5, 0.5, 0.5],
        emissive_color: [0.0, 0.0, 0.0],
        normal_scale: 1.0,
    }
}

/// A vertex's true tangent, reconstructed from the binormal the file actually declares (see
/// `mdl-tangent-frame`): `T = normalize(B - N(N.B))`, `cross(B', N) * sign(w)`, with the same sign
/// carried into the fourth channel so a standard glTF consumer's own `cross(N, T) * T.w`
/// reconstructs the same bitangent. `None` where the frame is degenerate and inventing one would
/// be worse than leaving it out.
fn tangent_frame(vertex: &Vertex) -> Option<[f32; 4]> {
    let normal = Vec3::from_array(vertex.normal);
    if normal.length_squared() < 1e-12 {
        return None;
    }
    let normal = normal.normalize();
    let raw = Vec3::new(vertex.tangent[0], vertex.tangent[1], vertex.tangent[2]) * 2.0 - Vec3::ONE;
    let across = raw - normal * normal.dot(raw);
    if across.length_squared() < 1e-6 {
        return None;
    }
    let bitangent = across.normalize();
    let sign = if vertex.tangent[3] * 2.0 - 1.0 >= 0.0 { 1.0 } else { -1.0 };
    let tangent = bitangent.cross(normal) * sign;
    Some([tangent.x, tangent.y, tangent.z, sign])
}

/// One vertex's eight packed influences, unpacked, dropped where the skeleton names no such bone,
/// renormalized over what is left, and split across two glTF weight sets where more than four
/// survive. A vertex that loses every influence stays at bind, matching `Skin::palette`.
fn skin_vertex(
    vertex: &Vertex,
    local_bones: &[String],
    joints: &HashMap<&str, u32>,
    fallback: u32,
    missing: &mut usize,
    wanted: &mut usize,
) -> ([u32; 4], [f32; 4], [u32; 4], [f32; 4], bool) {
    let mut influences: Vec<(u32, f32)> = Vec::with_capacity(8);
    for at in 0..8usize {
        let lane = at & 3;
        let shift = if at < 4 { 0 } else { 8 };
        let weight = f32::from((vertex.weights[lane] >> shift) & 0xFF) / 255.0;
        if weight <= 0.0 {
            continue;
        }
        *wanted += 1;
        let local = usize::from((vertex.bones[lane] >> shift) & 0xFF);
        let named = local_bones.get(local).filter(|name| !name.is_empty());
        match named.and_then(|name| joints.get(name.as_str())) {
            Some(joint) => influences.push((*joint, weight)),
            None => *missing += 1,
        }
    }
    let sum: f32 = influences.iter().map(|(_, weight)| weight).sum();
    if sum <= 0.0 {
        influences.clear();
        influences.push((fallback, 1.0));
    } else {
        for (_, weight) in &mut influences {
            *weight /= sum;
        }
    }
    let second = influences.len() > 4;
    let mut joints0 = [fallback; 4];
    let mut weights0 = [0.0; 4];
    let mut joints1 = [fallback; 4];
    let mut weights1 = [0.0; 4];
    for (at, (joint, weight)) in influences.into_iter().enumerate().take(8) {
        match at < 4 {
            true => {
                joints0[at] = joint;
                weights0[at] = weight;
            }
            false => {
                joints1[at - 4] = joint;
                weights1[at - 4] = weight;
            }
        }
    }
    (joints0, weights0, joints1, weights1, second)
}

#[allow(clippy::too_many_arguments)]
fn primitive_from(
    vertices: &[Vertex],
    indices: Vec<u32>,
    material: usize,
    tangent_present: bool,
    // The wearer's colours, where this primitive's own package picks between two of them per
    // vertex rather than tinting every texel the same.
    eyes: Option<&program::Customize>,
    skinning: Option<&Vec<String>>,
    joints: Option<&HashMap<&str, u32>>,
    fallback: u32,
    missing: &mut usize,
    wanted: &mut usize,
) -> Primitive {
    let mut positions = Vec::with_capacity(vertices.len());
    let mut normals = Vec::with_capacity(vertices.len());
    let mut tangents = tangent_present.then(Vec::new);
    let mut uv0 = Vec::with_capacity(vertices.len());
    let mut colors = Vec::with_capacity(vertices.len());
    let skinning = skinning.zip(joints);
    let mut joints0 = skinning.is_some().then(Vec::new);
    let mut weights0 = skinning.is_some().then(Vec::new);
    let mut joints1 = Vec::new();
    let mut weights1 = Vec::new();
    let mut needs_second = false;

    for vertex in vertices {
        positions.push(vertex.position);
        normals.push(vertex.normal);
        if let Some(tangents) = &mut tangents {
            tangents.push(tangent_frame(vertex).unwrap_or_else(|| safe_tangent(vertex.normal)));
        }
        uv0.push([vertex.uv[0], vertex.uv[1]]);
        colors.push(match eyes {
            Some(customize) => eye_color(vertex.color, customize),
            None => vertex_color(vertex.color),
        });
        if let Some((local_bones, joints)) = skinning {
            let (j0, w0, j1, w1, second) =
                skin_vertex(vertex, local_bones, joints, fallback, missing, wanted);
            joints0.as_mut().unwrap().push(j0);
            weights0.as_mut().unwrap().push(w0);
            joints1.push(j1);
            weights1.push(w1);
            needs_second |= second;
        }
    }

    Primitive {
        material,
        positions,
        normals,
        tangents,
        uv0,
        colors,
        joints0,
        weights0,
        joints1: needs_second.then_some(joints1),
        weights1: needs_second.then_some(weights1),
        indices,
    }
}

/// What a vertex's own colour contributes to a glTF one. Only the alpha is an opacity: `model.frag`
/// reads `v_color.a` and nothing else, so the rgb carries per-family shader inputs rather than a
/// tint, and a consumer multiplying `COLOR_0` into the base colour would paint the surface with
/// them. An iris is the one case that wants an actual colour there; [`eye_color`] supplies it.
fn vertex_color(raw: [u8; 4]) -> [u8; 4] {
    [255, 255, 255, raw[3]]
}

/// Any unit vector orthogonal to `normal`, for the rare vertex whose declared frame is degenerate.
fn safe_tangent(normal: [f32; 3]) -> [f32; 4] {
    let normal = Vec3::from_array(normal).normalize_or(Vec3::Y);
    let guess = if normal.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let tangent = guess.cross(normal).normalize_or(Vec3::X);
    [tangent.x, tangent.y, tangent.z, 1.0]
}

enum BakedMaterial {
    Flat {
        base_color: [f32; 4],
        roughness: f32,
        metalness: f32,
    },
    Baked {
        base_color: RgbaImage,
        metallic_roughness: RgbaImage,
        emissive: Option<RgbaImage>,
        normal: Option<RgbaImage>,
        /// Whether any texel came out short of opaque, which is what tells a surface that fades
        /// from one the alpha channel only ever cuts a hole in.
        sheer: bool,
    },
}

/// Bakes a material's shading into standard glTF PBR maps. Most character materials carry no
/// diffuse texture at all: their color comes from the id map picking a row of the color table, so
/// that lookup has to run once per output texel or the baseline export is grey. Lighting-only terms
/// (sheen, the rim light, a tinted specular) have no channel in glTF's metallic-roughness model and
/// are dropped rather than approximated.
fn bake(
    material: &MaterialInfo,
    images: &BTreeMap<String, DynamicImage>,
    customize: &program::Customize,
) -> BakedMaterial {
    let get = |role: Role| material.textures[role as usize].as_ref().and_then(|path| images.get(path));
    let normal = get(Role::Normal);
    let index = get(Role::Index);
    let mask = get(Role::Mask);
    let diffuse = get(Role::Diffuse);

    if normal.is_none() && index.is_none() && mask.is_none() && diffuse.is_none() && material.table.is_none() {
        return BakedMaterial::Flat {
            base_color: [0.72, 0.72, 0.72, 1.0],
            roughness: 0.5,
            metalness: 0.0,
        };
    }

    let (long_w, long_h) = [normal, index, mask, diffuse]
        .into_iter()
        .flatten()
        .map(GenericImageView::dimensions)
        .max_by_key(|(w, h)| u64::from(*w) * u64::from(*h))
        .unwrap_or((256, 256));
    let scale = (MAX_TEXTURE_DIM as f32 / long_w.max(long_h) as f32).min(1.0);
    let width = ((long_w as f32 * scale).round() as u32).max(1);
    let height = ((long_h as f32 * scale).round() as u32).max(1);

    // Nearest, not the resize this app otherwise reaches for: interpolating the index map would mix
    // row *indices*, landing on rows the color table never declared.
    let resize = |image: Option<&DynamicImage>| {
        image.map(|image| imageops::resize(&image.to_rgba8(), width, height, FilterType::Nearest))
    };
    let normal_img = resize(normal);
    let index_img = resize(index);
    let mask_img = resize(mask);
    let diffuse_img = resize(diffuse);

    let mut base_color = RgbaImage::new(width, height);
    let mut metallic_roughness = RgbaImage::new(width, height);
    let mut emissive_img = RgbaImage::new(width, height);
    let mut has_emissive = false;
    for y in 0..height {
        for x in 0..width {
            let sample = |image: &Option<RgbaImage>| image.as_ref().map(|image| image.get_pixel(x, y).0);
            let shaded = shade_texel(
                material,
                sample(&normal_img),
                sample(&index_img),
                sample(&mask_img),
                sample(&diffuse_img),
                customize,
            );
            base_color.put_pixel(
                x,
                y,
                Rgba([
                    to_srgb(shaded.albedo[0]),
                    to_srgb(shaded.albedo[1]),
                    to_srgb(shaded.albedo[2]),
                    (shaded.opacity.clamp(0.0, 1.0) * 255.0).round() as u8,
                ]),
            );
            metallic_roughness.put_pixel(
                x,
                y,
                Rgba([255, (shaded.roughness * 255.0).round() as u8, (shaded.metalness * 255.0).round() as u8, 255]),
            );
            has_emissive |= shaded.emissive != [0.0; 3];
            emissive_img.put_pixel(
                x,
                y,
                Rgba([to_srgb(shaded.emissive[0]), to_srgb(shaded.emissive[1]), to_srgb(shaded.emissive[2]), 255]),
            );
        }
    }

    let normal_out = normal_img.map(|image| {
        let mut out = RgbaImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                out.put_pixel(x, y, Rgba(repack_normal(image.get_pixel(x, y).0)));
            }
        }
        out
    });

    let sheer = base_color.pixels().any(|texel| texel.0[3] < 255);
    BakedMaterial::Baked {
        base_color,
        metallic_roughness,
        emissive: has_emissive.then_some(emissive_img),
        normal: normal_out,
        sheer,
    }
}

struct Shaded {
    albedo: [f32; 3],
    opacity: f32,
    roughness: f32,
    metalness: f32,
    emissive: [f32; 3],
}

/// The occlusion a package folds into its own albedo, out of the mask map. Measured per package,
/// not per family: `iris` and `skin` fold in none at all, and a blanket rule multiplied the eyes to
/// black by a channel that is nought across every texel of their mask.
fn occlusion(package: &str, mask: Option<[f32; 4]>) -> f32 {
    let Some(mask) = mask else {
        return 1.0;
    };
    let squared = |channel: f32| channel * channel;
    match package {
        "hair.shpk" => squared(mask[3]),
        "character.shpk" => squared(mask[2]),
        _ => 1.0,
    }
}

/// What the wearer's own colours tint a package's albedo with. Hair is mixed between the two hair
/// colours by the highlight blend its mask holds; skin is tinted outright. An iris is neither: its
/// two colours are picked between per vertex, so [`eye_color`] carries them instead.
fn tint(package: &str, mask: Option<[f32; 4]>, customize: &program::Customize) -> [f32; 3] {
    match package {
        "hair.shpk" => {
            let blend = mask.map_or(0.0, |mask| mask[2]);
            std::array::from_fn(|i| {
                customize.hair[i] + (customize.highlight[i] - customize.hair[i]) * blend
            })
        }
        "skin.shpk" => [customize.skin[0], customize.skin[1], customize.skin[2]],
        _ => [1.0; 3],
    }
}

/// The colour an iris vertex carries, which is the side its own colour selects rather than a tint:
/// `iris.shpk` computes `m_LeftColor * COLOR.x + m_RightColor * COLOR.y`, so the same sum travels
/// as `COLOR_0` and a glTF consumer's own multiply lands the two eyes apart.
fn eye_color(raw: [u8; 4], customize: &program::Customize) -> [u8; 4] {
    let (left, right) = (f32::from(raw[0]) / 255.0, f32::from(raw[1]) / 255.0);
    let mixed: [f32; 3] =
        std::array::from_fn(|i| customize.left_eye[i] * left + customize.right_eye[i] * right);
    [
        to_srgb(mixed[0]),
        to_srgb(mixed[1]),
        to_srgb(mixed[2]),
        raw[3],
    ]
}

/// The static half of `model.frag`'s shading: everything but the three lights and the tone curve,
/// which glTF's own renderer supplies. Vertex-color opacity (dye, wind) is not a texel and travels
/// as `COLOR_0` instead, multiplying this texture's alpha the same way `v_color.a` does on screen.
fn shade_texel(
    material: &MaterialInfo,
    normal: Option<[u8; 4]>,
    index: Option<[u8; 4]>,
    mask: Option<[u8; 4]>,
    diffuse: Option<[u8; 4]>,
    customize: &program::Customize,
) -> Shaded {
    let mut albedo = [0.72f32; 3];
    let mut roughness = 0.5f32;
    let mut metalness = 0.0f32;
    let mut emissive = [0.0f32; 3];

    let sampled = normal.map_or([0.5, 0.5, 1.0, 0.0], |raw| raw.map(|channel| f32::from(channel) / 255.0));

    if let Some(table) = &material.table {
        let rows = table.len() / 16;
        if rows >= 1 {
            let alpha = sampled[3];
            let index_rg = index.map(|raw| (f32::from(raw[0]) / 255.0, f32::from(raw[1]) / 255.0));
            let (lower, upper, blend) = pick_rows(material.family, rows, alpha, index_rg);
            let mix = |channel: usize| {
                let a = table_texel(table, lower, channel);
                let b = table_texel(table, upper, channel);
                std::array::from_fn::<f32, 4, _>(|component| a[component] + (b[component] - a[component]) * blend)
            };
            let first = mix(0);
            let second = mix(1);
            let third = mix(2);
            albedo = [first[0], first[1], first[2]];
            roughness = first[3];
            metalness = second[3];
            emissive = [third[0], third[1], third[2]];
        }
    }

    if let Some(diffuse) = diffuse {
        let linear = [
            (f32::from(diffuse[0]) / 255.0).powi(2),
            (f32::from(diffuse[1]) / 255.0).powi(2),
            (f32::from(diffuse[2]) / 255.0).powi(2),
        ];
        albedo = match material.family == Family::Legacy {
            true => linear,
            false => [albedo[0] * linear[0], albedo[1] * linear[1], albedo[2] * linear[2]],
        };
    }

    let unit_mask = mask.map(|mask| mask.map(|channel| f32::from(channel) / 255.0));
    if let Some(mask) = unit_mask
        && material.family != Family::Background
    {
        let shaded = occlusion(&material.package, Some(mask));
        albedo = [albedo[0] * shaded, albedo[1] * shaded, albedo[2] * shaded];
        if material.family != Family::Legacy {
            let bias = roughness * 2.0 - 1.0;
            roughness = mask[1] + bias * if bias < 0.0 { mask[1] } else { 1.0 - mask[1] };
        }
    }

    // The wearer's own colours, which no texture holds.
    let tinted = tint(&material.package, unit_mask, customize);
    albedo = [albedo[0] * tinted[0], albedo[1] * tinted[1], albedo[2] * tinted[2]];
    // A lip is a tint laid over the skin by the weight the face's own normal map states, not a
    // colour of its own.
    if material.package == "skin.shpk" {
        let weight = sampled[3] * customize.lip[3];
        albedo = std::array::from_fn(|i| albedo[i] + (customize.lip[i] - albedo[i]) * weight);
    }

    let mut opacity = 1.0f32;
    if normal.is_some() && material.family != Family::Background {
        opacity *= if material.family == Family::Hair { sampled[3] } else { sampled[2] };
    }
    if material.family == Family::Background
        && let Some(diffuse) = diffuse
    {
        opacity *= f32::from(diffuse[3]) / 255.0;
    }

    let albedo = std::array::from_fn(|i| albedo[i] * material.diffuse_color[i]);
    let emissive = std::array::from_fn(|i| emissive[i] + material.emissive_color[i]);

    Shaded {
        albedo,
        opacity,
        roughness: roughness.clamp(0.0, 1.0),
        metalness: metalness.clamp(0.0, 1.0),
        emissive,
    }
}

/// The row pair a texel blends between, and how far across it sits: the index map in seventeenths
/// for an extended table, the compatibility path's own alpha for a legacy one.
fn pick_rows(family: Family, rows: usize, alpha: f32, index: Option<(f32, f32)>) -> (usize, usize, f32) {
    if rows < 2 {
        return (0, 0, 0.0);
    }
    let (lower, blend) = if family == Family::Legacy {
        let position = alpha.clamp(0.0, 1.0) * (rows.min(16) as f32 - 1.0);
        (position as usize, position.fract())
    } else if let Some((r, g)) = index {
        (((255.0 * r + 8.0) / 17.0) as usize * 2, (1.0 - g).clamp(0.0, 1.0))
    } else {
        (0, 0.0)
    };
    let lower = lower.min(rows - 1);
    (lower, (lower + 1).min(rows - 1), blend)
}

fn table_texel(table: &[f32], row: usize, column: usize) -> [f32; 4] {
    let at = row * 16 + column * 4;
    [table[at], table[at + 1], table[at + 2], table[at + 3]]
}

/// glTF declares `baseColorTexture` and `emissiveTexture` sRGB-encoded, so a compliant renderer
/// decodes with the real sRGB EOTF, not the viewer's own `sqrt` approximation of one.
fn to_srgb(linear: f32) -> u8 {
    let linear = linear.clamp(0.0, 1.0);
    let encoded = if linear <= 0.003_130_8 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    };
    (encoded * 255.0).round() as u8
}

fn repack_normal(raw: [u8; 4]) -> [u8; 4] {
    let x = f32::from(raw[0]) / 255.0 * 2.0 - 1.0;
    let y = f32::from(raw[1]) / 255.0 * 2.0 - 1.0;
    let z = (1.0 - (x * x + y * y)).max(1e-4).sqrt();
    [
        ((x * 0.5 + 0.5) * 255.0).round() as u8,
        ((y * 0.5 + 0.5) * 255.0).round() as u8,
        ((z * 0.5 + 0.5) * 255.0).round() as u8,
        255,
    ]
}

/// Every texture path a scene's materials name, for the caller to fetch before baking.
fn texture_paths(scene: &Scene) -> Vec<String> {
    let mut paths: Vec<String> = scene
        .materials
        .iter()
        .flat_map(|material| material.textures.iter().flatten().cloned())
        .collect();
    paths.sort();
    paths.dedup();
    paths
}

struct Writer {
    bin: Vec<u8>,
    buffer_views: Vec<Value>,
    accessors: Vec<Value>,
    images: Vec<Value>,
    textures: Vec<Value>,
    samplers: Vec<Value>,
    sampler: Option<u32>,
}

impl Writer {
    fn new() -> Self {
        Self {
            bin: Vec::new(),
            buffer_views: Vec::new(),
            accessors: Vec::new(),
            images: Vec::new(),
            textures: Vec::new(),
            samplers: Vec::new(),
            sampler: None,
        }
    }

    fn pad(&mut self) {
        while !self.bin.len().is_multiple_of(4) {
            self.bin.push(0);
        }
    }

    fn view(&mut self, bytes: &[u8], stride: Option<u32>, target: Option<u32>) -> u32 {
        self.pad();
        let offset = self.bin.len() as u32;
        self.bin.extend_from_slice(bytes);
        let index = self.buffer_views.len() as u32;
        let mut view = json!({
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": bytes.len() as u32,
        });
        if let Some(stride) = stride {
            view["byteStride"] = json!(stride);
        }
        if let Some(target) = target {
            view["target"] = json!(target);
        }
        self.buffer_views.push(view);
        index
    }

    #[allow(clippy::too_many_arguments)]
    fn accessor(&mut self, view: u32, component_type: u32, count: u32, kind: &str, normalized: bool, min: Option<Value>, max: Option<Value>) -> u32 {
        let index = self.accessors.len() as u32;
        let mut accessor = json!({
            "bufferView": view,
            "componentType": component_type,
            "count": count,
            "type": kind,
        });
        if normalized {
            accessor["normalized"] = json!(true);
        }
        if let Some(min) = min {
            accessor["min"] = min;
        }
        if let Some(max) = max {
            accessor["max"] = max;
        }
        self.accessors.push(accessor);
        index
    }

    /// Registers a baked PNG as a texture, reusing the one repeat-wrapped sampler every baked
    /// material's maps share.
    fn texture(&mut self, png: &[u8]) -> u32 {
        let view = self.view(png, None, None);
        let image_index = self.images.len() as u32;
        self.images.push(json!({ "mimeType": "image/png", "bufferView": view }));
        if self.sampler.is_none() {
            let index = self.samplers.len() as u32;
            self.samplers.push(json!({
                "magFilter": 9729,
                "minFilter": 9987,
                "wrapS": 10497,
                "wrapT": 10497,
            }));
            self.sampler = Some(index);
        }
        let texture_index = self.textures.len() as u32;
        self.textures
            .push(json!({ "sampler": self.sampler.unwrap(), "source": image_index }));
        texture_index
    }
}

const FLOAT: u32 = 5126;
const UNSIGNED_SHORT: u32 = 5123;
const UNSIGNED_BYTE: u32 = 5121;
const ARRAY_BUFFER: u32 = 34962;
const ELEMENT_ARRAY_BUFFER: u32 = 34963;

/// Writes a scene and its baked materials to a self-contained `.glb`. `baked` is one entry per
/// `Scene`'s own material list, in order.
fn assemble(scene: &Scene, baked: &[BakedMaterial]) -> Result<Vec<u8>> {
    let mut writer = Writer::new();

    let materials: Vec<Value> = scene
        .materials
        .iter()
        .zip(baked)
        .map(|(info, baked)| material_json(info, baked, &mut writer))
        .collect();

    let mut nodes = Vec::new();
    let mut skins = Vec::new();
    let mut root_children = Vec::new();

    let has_skin = match &scene.skeleton {
        Some(skeleton) => {
            let (mut joints, roots) = skeleton_nodes(skeleton, &mut nodes);
            root_children.extend(roots);
            let fallback_node = nodes.len() as u32;
            nodes.push(json!({ "name": "export_identity" }));
            root_children.push(fallback_node);
            joints.push(fallback_node);

            let mut ibm_bytes = Vec::with_capacity((skeleton.names.len() + 1) * 64);
            for matrix in &skeleton.rest_inverse {
                ibm_bytes.extend_from_slice(bytemuck::bytes_of(&matrix.to_cols_array()));
            }
            ibm_bytes.extend_from_slice(bytemuck::bytes_of(&Mat4::IDENTITY.to_cols_array()));
            let view = writer.view(&ibm_bytes, None, None);
            let ibm = writer.accessor(view, FLOAT, (skeleton.names.len() + 1) as u32, "MAT4", false, None, None);

            skins.push(json!({ "joints": joints, "inverseBindMatrices": ibm }));
            true
        }
        None => false,
    };

    let mut meshes = Vec::new();
    for piece in &scene.pieces {
        let (skinned, unskinned): (Vec<&Primitive>, Vec<&Primitive>) =
            piece.primitives.iter().partition(|primitive| primitive.joints0.is_some());
        for (group, tag) in [(skinned, ""), (unskinned, " (static)")] {
            if group.is_empty() {
                continue;
            }
            let primitives: Vec<Value> = group
                .into_iter()
                .map(|primitive| primitive_json(primitive, &mut writer))
                .collect();
            let mesh_index = meshes.len() as u32;
            meshes.push(json!({ "primitives": primitives, "name": format!("{}{tag}", piece.name) }));
            let node_index = nodes.len() as u32;
            let mut node = json!({ "name": format!("{}{tag}", piece.name), "mesh": mesh_index });
            // Only the skinned group carries joints, so only it names the skin; a piece with no
            // bone data at all draws exactly where the file put it.
            if tag.is_empty() && has_skin {
                node["skin"] = json!(0);
            }
            nodes.push(node);
            root_children.push(node_index);
        }
    }

    let root_index = nodes.len() as u32;
    let mut root = json!({ "name": "Model", "children": root_children });
    if scene.stature != 1.0 {
        root["matrix"] = json!(Mat4::from_scale(Vec3::splat(scene.stature)).to_cols_array());
    }
    nodes.push(root);

    let buffer_length = writer.bin.len() as u32;
    let document = json!({
        "asset": { "version": "2.0", "generator": "EXDViewer" },
        "scene": 0,
        "scenes": [{ "nodes": [root_index] }],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "textures": writer.textures,
        "images": writer.images,
        "samplers": writer.samplers,
        "skins": skins,
        "accessors": writer.accessors,
        "bufferViews": writer.buffer_views,
        "buffers": [{ "byteLength": buffer_length }],
    });

    write_glb(&document, &writer.bin)
}

/// Writes one node per bone and returns every one of their indices, in the model's own bone
/// order: `JOINTS_0`/`JOINTS_1` reference a skin's `joints` array positionally, and that array is
/// built straight from this order, so a caller must not reorder or filter it down to roots alone.
/// The second return is the roots alone, for hanging the rig off the scene's own root node.
fn skeleton_nodes(skeleton: &Skeleton, nodes: &mut Vec<Value>) -> (Vec<u32>, Vec<u32>) {
    let base = nodes.len() as u32;
    let indices: Vec<u32> = (0..skeleton.names.len() as u32).map(|at| base + at).collect();
    let mut children: Vec<Vec<u32>> = vec![Vec::new(); skeleton.names.len()];
    for (bone, parent) in skeleton.parents.iter().enumerate() {
        if let Some(parent) = parent {
            children[*parent].push(indices[bone]);
        }
    }
    let mut roots = Vec::new();
    for (bone, name) in skeleton.names.iter().enumerate() {
        let local = match skeleton.parents[bone] {
            Some(parent) => skeleton.world[parent].inverse() * skeleton.world[bone],
            None => {
                roots.push(indices[bone]);
                skeleton.world[bone]
            }
        };
        nodes.push(json!({
            "name": name,
            "children": children[bone],
            "matrix": local.to_cols_array(),
        }));
    }
    (indices, roots)
}

fn primitive_json(primitive: &Primitive, writer: &mut Writer) -> Value {
    let mut attributes = serde_json::Map::new();

    let position_view = writer.view(bytemuck::cast_slice(&primitive.positions), Some(12), Some(ARRAY_BUFFER));
    let (min, max) = bounds(&primitive.positions);
    attributes.insert(
        "POSITION".into(),
        json!(writer.accessor(position_view, FLOAT, primitive.positions.len() as u32, "VEC3", false, Some(min), Some(max))),
    );

    let normal_view = writer.view(bytemuck::cast_slice(&primitive.normals), Some(12), Some(ARRAY_BUFFER));
    attributes.insert(
        "NORMAL".into(),
        json!(writer.accessor(normal_view, FLOAT, primitive.normals.len() as u32, "VEC3", false, None, None)),
    );

    if let Some(tangents) = &primitive.tangents {
        let view = writer.view(bytemuck::cast_slice(tangents), Some(16), Some(ARRAY_BUFFER));
        attributes.insert(
            "TANGENT".into(),
            json!(writer.accessor(view, FLOAT, tangents.len() as u32, "VEC4", false, None, None)),
        );
    }

    let uv_view = writer.view(bytemuck::cast_slice(&primitive.uv0), Some(8), Some(ARRAY_BUFFER));
    attributes.insert(
        "TEXCOORD_0".into(),
        json!(writer.accessor(uv_view, FLOAT, primitive.uv0.len() as u32, "VEC2", false, None, None)),
    );

    let color_view = writer.view(bytemuck::cast_slice(&primitive.colors), Some(4), Some(ARRAY_BUFFER));
    attributes.insert(
        "COLOR_0".into(),
        json!(writer.accessor(color_view, UNSIGNED_BYTE, primitive.colors.len() as u32, "VEC4", true, None, None)),
    );

    if let (Some(joints0), Some(weights0)) = (&primitive.joints0, &primitive.weights0) {
        let joints_narrow: Vec<[u16; 4]> = joints0.iter().map(|j| j.map(|v| v as u16)).collect();
        let view = writer.view(bytemuck::cast_slice(&joints_narrow), Some(8), Some(ARRAY_BUFFER));
        attributes.insert(
            "JOINTS_0".into(),
            json!(writer.accessor(view, UNSIGNED_SHORT, joints_narrow.len() as u32, "VEC4", false, None, None)),
        );
        let view = writer.view(bytemuck::cast_slice(weights0), Some(16), Some(ARRAY_BUFFER));
        attributes.insert(
            "WEIGHTS_0".into(),
            json!(writer.accessor(view, FLOAT, weights0.len() as u32, "VEC4", false, None, None)),
        );
    }
    if let (Some(joints1), Some(weights1)) = (&primitive.joints1, &primitive.weights1) {
        let joints_narrow: Vec<[u16; 4]> = joints1.iter().map(|j| j.map(|v| v as u16)).collect();
        let view = writer.view(bytemuck::cast_slice(&joints_narrow), Some(8), Some(ARRAY_BUFFER));
        attributes.insert(
            "JOINTS_1".into(),
            json!(writer.accessor(view, UNSIGNED_SHORT, joints_narrow.len() as u32, "VEC4", false, None, None)),
        );
        let view = writer.view(bytemuck::cast_slice(weights1), Some(16), Some(ARRAY_BUFFER));
        attributes.insert(
            "WEIGHTS_1".into(),
            json!(writer.accessor(view, FLOAT, weights1.len() as u32, "VEC4", false, None, None)),
        );
    }

    // The source format's own indices are 16-bit, and compacting only ever drops vertices, so a
    // primitive never needs wider than that.
    let narrow_indices: Vec<u16> = primitive.indices.iter().map(|&index| index as u16).collect();
    let index_view = writer.view(bytemuck::cast_slice(&narrow_indices), None, Some(ELEMENT_ARRAY_BUFFER));
    let indices_accessor =
        writer.accessor(index_view, UNSIGNED_SHORT, narrow_indices.len() as u32, "SCALAR", false, None, None);

    json!({
        "attributes": attributes,
        "indices": indices_accessor,
        "material": primitive.material,
    })
}

fn bounds(positions: &[[f32; 3]]) -> (Value, Value) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for position in positions {
        for axis in 0..3 {
            min[axis] = min[axis].min(position[axis]);
            max[axis] = max[axis].max(position[axis]);
        }
    }
    (json!(min), json!(max))
}

fn material_json(info: &MaterialInfo, baked: &BakedMaterial, writer: &mut Writer) -> Value {
    let mut pbr = serde_json::Map::new();
    let mut material = json!({ "name": info.name, "pbrMetallicRoughness": {} });

    match baked {
        BakedMaterial::Flat { base_color, roughness, metalness } => {
            pbr.insert("baseColorFactor".into(), json!(base_color));
            pbr.insert("metallicFactor".into(), json!(metalness));
            pbr.insert("roughnessFactor".into(), json!(roughness));
        }
        BakedMaterial::Baked { base_color, metallic_roughness, emissive, normal, .. } => {
            let index = writer.texture(&tex_png(base_color));
            pbr.insert("baseColorTexture".into(), json!({ "index": index }));
            pbr.insert("metallicFactor".into(), json!(1.0));
            pbr.insert("roughnessFactor".into(), json!(1.0));

            let index = writer.texture(&tex_png(metallic_roughness));
            pbr.insert("metallicRoughnessTexture".into(), json!({ "index": index }));

            if let Some(emissive) = emissive {
                let index = writer.texture(&tex_png(emissive));
                material["emissiveTexture"] = json!({ "index": index });
                material["emissiveFactor"] = json!([1.0, 1.0, 1.0]);
            }
            if let Some(normal) = normal {
                let index = writer.texture(&tex_png(normal));
                material["normalTexture"] = json!({ "index": index });
                if info.normal_scale != 1.0 {
                    material["normalTexture"]["scale"] = json!(info.normal_scale);
                }
            }
        }
    }
    material["pbrMetallicRoughness"] = Value::Object(pbr);
    let sheer = match baked {
        BakedMaterial::Baked { sheer, .. } => *sheer,
        BakedMaterial::Flat { base_color, .. } => base_color[3] < 1.0,
    };
    if info.alpha_threshold > 0.0 {
        material["alphaMode"] = json!("MASK");
        material["alphaCutoff"] = json!(info.alpha_threshold);
    } else if sheer || info.glass {
        material["alphaMode"] = json!("BLEND");
    }
    material["doubleSided"] = json!(!info.cull);
    material
}

fn tex_png(image: &RgbaImage) -> Vec<u8> {
    crate::utils::tex_loader::write(DynamicImage::ImageRgba8(image.clone()), ImageFormat::Png)
        .unwrap_or_default()
}

fn write_glb(document: &Value, bin: &[u8]) -> Result<Vec<u8>> {
    let mut json_chunk = serde_json::to_vec(document).context("failed to serialize the glTF document")?;
    while !json_chunk.len().is_multiple_of(4) {
        json_chunk.push(b' ');
    }
    let mut bin_chunk = bin.to_vec();
    while !bin_chunk.len().is_multiple_of(4) {
        bin_chunk.push(0);
    }

    let total = 12 + 8 + json_chunk.len() + 8 + bin_chunk.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());

    out.extend_from_slice(&(json_chunk.len() as u32).to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(&json_chunk);

    out.extend_from_slice(&(bin_chunk.len() as u32).to_le_bytes());
    out.extend_from_slice(b"BIN\0");
    out.extend_from_slice(&bin_chunk);

    Ok(out)
}

/// Fetches every texture a scene's materials name, no larger than the bake ever keeps, and writes
/// the result. The only part of the export that touches the network, so it is what runs after
/// `gather` has already taken everything else it needs off the model, letting the caller hold no
/// reference to it across the wait.
pub(super) async fn finish(scene: Scene, files: &dyn FileProvider) -> Result<Vec<u8>> {
    let mut images = BTreeMap::new();
    for path in texture_paths(&scene) {
        match files.read_texture(&path, Some(MAX_TEXTURE_DIM as u16)).await {
            Ok(decoded) => {
                images.insert(path, DynamicImage::ImageRgba8(decoded.image));
            }
            Err(why) => log::warn!("assets/mdl: export: {path}: {why}"),
        }
    }
    let baked: Vec<BakedMaterial> = scene
        .materials
        .iter()
        .map(|material| bake(material, &images, &scene.customize))
        .collect();
    assemble(&scene, &baked)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coloured() -> program::Customize {
        program::Customize {
            skin: [0.8, 0.6, 0.5, 1.0],
            lip: [1.0, 0.0, 0.0, 0.5],
            hair: [0.2, 0.1, 0.0, 1.0],
            highlight: [1.0, 1.0, 0.0, 1.0],
            left_eye: [1.0, 0.0, 0.0, 1.0],
            right_eye: [0.0, 1.0, 0.0, 1.0],
            ..program::Customize::default()
        }
    }

    /// The measured rule is stated per package: a family-wide one multiplied the eyes by a channel
    /// that is nought across every texel of their mask, which drew them black.
    #[test]
    fn only_the_packages_that_state_an_occlusion_fold_one_in() {
        let mask = Some([1.0, 1.0, 0.5, 0.25]);
        assert_eq!(occlusion("hair.shpk", mask), 0.0625);
        assert_eq!(occlusion("character.shpk", mask), 0.25);
        assert_eq!(occlusion("iris.shpk", mask), 1.0);
        assert_eq!(occlusion("skin.shpk", mask), 1.0);
        assert_eq!(occlusion("character.shpk", None), 1.0);
    }

    #[test]
    fn hair_is_mixed_between_its_two_colours_and_skin_is_tinted_outright() {
        let held = coloured();
        // The blend is the mask's blue: nought is all hair colour, one is all highlight.
        assert_eq!(tint("hair.shpk", Some([0.0, 0.0, 0.0, 0.0]), &held), [0.2, 0.1, 0.0]);
        assert_eq!(tint("hair.shpk", Some([0.0, 0.0, 1.0, 0.0]), &held), [1.0, 1.0, 0.0]);
        assert_eq!(tint("skin.shpk", None, &held), [0.8, 0.6, 0.5]);
        assert_eq!(tint("character.shpk", None, &held), [1.0; 3]);
    }

    #[test]
    fn an_iris_vertex_carries_the_eye_its_own_colour_selects() {
        let held = coloured();
        // COLOR.x picks the left eye and COLOR.y the right, which is how one face draws two.
        assert_eq!(eye_color([255, 0, 0, 255], &held), [255, 0, 0, 255]);
        assert_eq!(eye_color([0, 255, 0, 128], &held), [0, 255, 0, 128]);
    }

    #[test]
    fn a_vertex_colour_carries_its_opacity_and_tints_nothing() {
        // An iris mesh's own colour is the side select, not a colour: exporting it as one paints
        // one eye red and the other green.
        assert_eq!(vertex_color([255, 0, 0, 255]), [255, 255, 255, 255]);
        assert_eq!(vertex_color([0, 255, 0, 128]), [255, 255, 255, 128]);
    }

    fn vertex(position: [f32; 3], normal: [f32; 3]) -> Vertex {
        // `Vertex`'s fields are private to `mdl` but visible to this descendant module.
        Vertex {
            position,
            normal,
            tangent: [0.5, 0.5, 1.0, 1.0],
            bitangent: [0.5, 0.5, 1.0, 1.0],
            uv: [0.0, 0.0, 0.0, 0.0],
            uv1: [0.0; 4],
            color: [255; 4],
            color1: [0; 4],
            weights: [255, 0, 0, 0],
            bones: [0, 0, 0, 0],
        }
    }

    #[test]
    fn a_degenerate_binormal_omits_a_tangent_rather_than_nan() {
        assert!(tangent_frame(&vertex([0.0; 3], [0.0, 0.0, 1.0])).is_none());
    }

    #[test]
    fn the_binormal_reconstructs_the_games_own_tangent() {
        // kind-6 raw = (1,0,0) biased -> unbias to +X, handedness +1: matches `mdl-tangent-frame`'s
        // measured `T = w * cross(B, N)` with B=+X, N=+Z, giving T=cross(X,Z)=-Y.
        let mut v = vertex([0.0; 3], [0.0, 0.0, 1.0]);
        v.tangent = [1.0, 0.5, 0.5, 1.0];
        let frame = tangent_frame(&v).unwrap();
        assert!((frame[0] - 0.0).abs() < 1e-5);
        assert!((frame[1] - -1.0).abs() < 1e-5);
        assert!((frame[2] - 0.0).abs() < 1e-5);
        assert!((frame[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn eight_influences_split_and_renormalize_across_two_weight_sets() {
        let mut v = vertex([0.0; 3], [0.0, 1.0, 0.0]);
        // Every one of the eight lanes carries influence `at` at equal weight, so after
        // renormalizing each of the eight should read 0.125.
        v.weights = [0x_FF_FF, 0x_FF_FF, 0x_FF_FF, 0x_FF_FF];
        v.bones = [0x_0100, 0x_0302, 0x_0504, 0x_0706];
        let local: Vec<String> = (0..8).map(|n| format!("b{n}")).collect();
        let joints: HashMap<&str, u32> = local.iter().map(|name| (name.as_str(), name[1..].parse().unwrap())).collect();
        let mut missing = 0;
        let mut wanted = 0;
        let (j0, w0, j1, w1, second) = skin_vertex(&v, &local, &joints, 99, &mut missing, &mut wanted);
        assert!(second);
        assert_eq!(missing, 0);
        assert_eq!(wanted, 8);
        for weight in w0.iter().chain(w1.iter()) {
            assert!((weight - 0.125).abs() < 1e-5);
        }
        let mut seen: Vec<u32> = j0.iter().chain(j1.iter()).copied().collect();
        seen.sort_unstable();
        assert_eq!(seen, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn a_bone_named_by_no_skeleton_falls_back_to_bind() {
        let v = vertex([0.0; 3], [0.0, 1.0, 0.0]);
        let local = vec!["ghost".to_owned()];
        let joints: HashMap<&str, u32> = HashMap::new();
        let mut missing = 0;
        let mut wanted = 0;
        let (j0, w0, _, _, _) = skin_vertex(&v, &local, &joints, 42, &mut missing, &mut wanted);
        assert_eq!(missing, 1);
        assert_eq!(j0[0], 42);
        assert!((w0[0] - 1.0).abs() < 1e-6);
        assert_eq!(w0[1..], [0.0; 3]);
    }

    #[test]
    fn joint_hierarchy_composes_to_the_posed_world() {
        // Two bones, the second scaled non-uniformly under the first: a TRS decomposition of this
        // would not recompose correctly, which is why joint nodes carry `matrix` instead.
        let world = vec![
            Mat4::from_scale(glam::Vec3::new(1.0, 2.0, 1.0)),
            Mat4::from_scale(glam::Vec3::new(1.0, 2.0, 1.0))
                * Mat4::from_translation(glam::Vec3::new(0.0, 1.0, 0.0))
                * Mat4::from_scale(glam::Vec3::new(3.0, 1.0, 1.0)),
        ];
        let skeleton = Skeleton {
            names: vec!["root".into(), "child".into()],
            parents: vec![None, Some(0)],
            world: world.clone(),
            rest_inverse: vec![Mat4::IDENTITY, Mat4::IDENTITY],
        };
        let mut nodes = Vec::new();
        skeleton_nodes(&skeleton, &mut nodes);
        let local: [f32; 16] = nodes[1]["matrix"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let recomposed = world[0] * Mat4::from_cols_array(&local);
        for (a, b) in recomposed.to_cols_array().iter().zip(world[1].to_cols_array()) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    /// Polls a future to completion on the current thread with no real waker, which is enough for
    /// the local install's own I/O: nothing here needs to run concurrently with anything else.
    fn block_on<F: std::future::Future>(future: F) -> F::Output {
        use std::task::Wake;
        struct NoopWaker;
        impl Wake for NoopWaker {
            fn wake(self: std::sync::Arc<Self>) {}
        }
        let waker = std::task::Waker::from(std::sync::Arc::new(NoopWaker));
        let mut cx = std::task::Context::from_waker(&waker);
        let mut future = std::pin::pin!(future);
        loop {
            match future.as_mut().poll(&mut cx) {
                std::task::Poll::Ready(value) => return value,
                std::task::Poll::Pending => std::thread::sleep(std::time::Duration::from_millis(2)),
            }
        }
    }

    const SQPACK: &str = "/home/asriel/.xlcore/ffxiv/game/sqpack";

    fn read_local(path: &str) -> Vec<u8> {
        use ironworks::sqpack::{Install, SqPack};
        use std::io::Read;
        let pack = SqPack::new(Install::at_sqpack(SQPACK));
        let mut stream = pack.file(path).unwrap();
        let mut bytes = Vec::new();
        stream.read_to_end(&mut bytes).unwrap();
        bytes
    }

    /// Real data end to end, run manually (`cargo test -p viewer --lib -- --ignored
    /// export::tests::a_real --nocapture`): the body model, its skeleton off the local install,
    /// posed by the bust slider so `world` differs from bind by a real, non-uniform-scale margin
    /// (the case a TRS-decomposed joint node would get wrong), baked with real textures and
    /// written as a `.glb` to the scratchpad for the standalone `gltf`-crate check.
    #[test]
    #[ignore = "reads the real local FFXIV install"]
    fn a_real_body_exports_posed_and_skinned() {
        let path = "chara/human/c0101/obj/body/b0001/model/c0101b0001_top.mdl";
        let bytes = read_local(path);
        let rendered = super::super::compose(&[super::super::Source {
            path: path.to_owned(),
            bytes,
            variant: 0,
            material: None,
            deform: None,
            skin: None,
            rigid: false,
        }])
        .unwrap();

        let backend = block_on(crate::backend::Backend::new(crate::settings::BackendConfig {
            api_url: "https://exd.camora.dev".to_owned(),
            location: crate::settings::InstallLocation::Sqpack(SQPACK.to_owned()),
            schema: crate::settings::SchemaLocation::Local("/home/asriel/Code/EXDSchema".to_owned()),
        }))
        .unwrap();

        let ctx = egui::Context::default();
        for _ in 0..500 {
            // `spawn_local`'s futures only run when something ticks `poll_promise`'s own local
            // executor, which the real app does once a frame from `tick_promises`; materials and
            // the skeleton both land off `Rendered::poll`, which wants a real `Ui` to call it with.
            crate::utils::tick_promises(&ctx);
            let _ = ctx.run_ui(egui::RawInput::default(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| rendered.poll(ui, &backend));
            });
            let ready = rendered
                .slots
                .borrow()
                .iter()
                .all(|slot| matches!(slot, Some(super::Slot::Ready(_))));
            if ready && rendered.animation.rig().is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        rendered.animation.rig().expect("the skeleton never landed");
        assert!(
            rendered.slots.borrow().iter().all(|slot| matches!(slot, Some(super::Slot::Ready(_)))),
            "not every material finished loading"
        );

        rendered.animation.shaped(glam::Vec3::new(1.0, 1.6, 1.0));
        let scene = gather(&rendered).expect("gather");

        let bones = scene.skeleton.as_ref().expect("body carries a skeleton");
        let bust = bones
            .names
            .iter()
            .position(|name| name == "j_mune_l")
            .expect("body names j_mune_l");
        let root = bones.names.iter().position(|name| name == "n_root").expect("body names n_root");
        let bust_delta = (bones.world[bust] * bones.rest_inverse[bust] - Mat4::IDENTITY)
            .to_cols_array()
            .iter()
            .fold(0.0f32, |acc, value| acc.max(value.abs()));
        let root_delta = (bones.world[root] * bones.rest_inverse[root] - Mat4::IDENTITY)
            .to_cols_array()
            .iter()
            .fold(0.0f32, |acc, value| acc.max(value.abs()));
        println!("bust joint delta from bind: {bust_delta}, root joint delta from bind: {root_delta}");
        assert!(bust_delta > 0.05, "the bust slider should move its own joint off bind");
        assert!(root_delta < 1e-4, "a bone the slider does not touch should stay at bind");

        let vertices: usize = scene.pieces.iter().flat_map(|piece| &piece.primitives).map(|p| p.positions.len()).sum();
        let triangles: usize = scene.pieces.iter().flat_map(|piece| &piece.primitives).map(|p| p.indices.len() / 3).sum();
        println!(
            "pieces: {}, primitives: {}, vertices (post-compaction, visible parts only): {vertices}, triangles: {triangles}, materials: {}, joints: {}",
            scene.pieces.len(),
            scene.pieces.iter().map(|piece| piece.primitives.len()).sum::<usize>(),
            scene.materials.len(),
            bones.names.len(),
        );
        assert!(vertices > 0 && triangles > 0);

        let files = crate::data::sqpack::SqpackFileProvider::new(SQPACK);
        let bytes = block_on(finish(scene, &files)).expect("finish");
        println!("glb size: {} bytes", bytes.len());
        assert_eq!(&bytes[0..4], b"glTF");

        let dir = std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| ".".to_owned());
        let out = std::path::Path::new(&dir).join("c0101b0001.glb");
        std::fs::write(&out, &bytes).unwrap();
        println!("wrote {}", out.display());
    }

    /// A body worn with a hairstyle: two pieces, two skeletons merged into one rig (the hair's own
    /// bones hang off the body's, per `est-extra-skeletons`), and a hair material whose family
    /// takes a different bake path than the body's. Exercises `Scene::pieces` actually holding more
    /// than one piece, which the single-piece test above cannot.
    #[test]
    #[ignore = "reads the real local FFXIV install"]
    fn a_body_and_hair_export_as_two_pieces_on_one_rig() {
        let body_path = "chara/human/c0101/obj/body/b0001/model/c0101b0001_top.mdl";
        let hair_path = "chara/human/c0101/obj/hair/h0001/model/c0101h0001_hir.mdl";
        let rendered = super::super::compose(&[
            super::super::Source {
                path: body_path.to_owned(),
                bytes: read_local(body_path),
                variant: 0,
                material: None,
                deform: None,
                skin: None,
                rigid: false,
            },
            super::super::Source {
                path: hair_path.to_owned(),
                bytes: read_local(hair_path),
                variant: 0,
                material: None,
                deform: None,
                skin: None,
                rigid: false,
            },
        ])
        .unwrap();

        let backend = block_on(crate::backend::Backend::new(crate::settings::BackendConfig {
            api_url: "https://exd.camora.dev".to_owned(),
            location: crate::settings::InstallLocation::Sqpack(SQPACK.to_owned()),
            schema: crate::settings::SchemaLocation::Local("/home/asriel/Code/EXDSchema".to_owned()),
        }))
        .unwrap();

        let ctx = egui::Context::default();
        for _ in 0..800 {
            crate::utils::tick_promises(&ctx);
            let _ = ctx.run_ui(egui::RawInput::default(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| rendered.poll(ui, &backend));
            });
            let ready = rendered
                .slots
                .borrow()
                .iter()
                .all(|slot| matches!(slot, Some(super::Slot::Ready(_))));
            if ready && rendered.animation.rig().is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        rendered.animation.rig().expect("the skeleton never landed");
        assert!(
            rendered.slots.borrow().iter().all(|slot| matches!(slot, Some(super::Slot::Ready(_)))),
            "not every material finished loading"
        );

        let scene = gather(&rendered).expect("gather");
        assert_eq!(scene.pieces.len(), 2, "one node per piece");
        for piece in &scene.pieces {
            assert!(!piece.primitives.is_empty(), "every piece should draw something");
            for primitive in &piece.primitives {
                assert!(primitive.joints0.is_some(), "both the body and this hairstyle carry bone data");
            }
        }
        let bones = scene.skeleton.as_ref().unwrap();
        println!(
            "pieces: {:?}, joints (body + hair's own merged in): {}, materials: {}",
            scene.pieces.iter().map(|p| (p.name.clone(), p.primitives.len())).collect::<Vec<_>>(),
            bones.names.len(),
            scene.materials.len(),
        );
        // The est merge appends the hair's own bones after the body's 106, so a hairstyle with
        // extra bones of its own should grow the rig past the body-only run's count.
        assert!(bones.names.len() >= 106);

        let files = crate::data::sqpack::SqpackFileProvider::new(SQPACK);
        let bytes = block_on(finish(scene, &files)).expect("finish");
        println!("glb size: {} bytes", bytes.len());
        let dir = std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| ".".to_owned());
        let out = std::path::Path::new(&dir).join("c0101_body_hair.glb");
        std::fs::write(&out, &bytes).unwrap();
        println!("wrote {}", out.display());
    }
}
