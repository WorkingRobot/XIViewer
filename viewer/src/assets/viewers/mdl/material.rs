//! Getting from the name a mesh gives its material to something the shader can bind.

use std::io::Cursor;

use anyhow::Result;
use half::f16;
use ironworks::file::{File, imc, mtrl};

use super::gpu::TABLE_COLUMNS;

/// What the shader does with a texture. A material names its samplers by hash, and the two shader
/// families the browser meets most name the same three roles differently.
#[derive(Clone, Copy)]
pub enum Role {
    Normal,
    Index,
    Mask,
    Diffuse,
}

/// `GlassBlendMode`, and the value that adds rather than multiplies.
const GLASS_BLEND_MODE: u32 = 0x9f2a_6183;
const GLASS_BLEND_ADD: u32 = 0x105a_09de;

/// How a glass pass reaches the frame behind it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Glass {
    Mul,
    Add,
}

/// Which set of meanings a material's textures carry. Every family binds the same four sampler
/// slots, so the slot a texture arrives in does not say what its channels are.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Family {
    /// A mask map: red scales the specular color, green states the roughness, blue an occlusion.
    /// The color table is indexed by the index map.
    Character,
    /// The three-texture set the game keeps a compatibility path for. The mask slot holds a
    /// specular map, whose blue scales the specular strength, and the color table is indexed by the
    /// normal map's alpha.
    Legacy,
    /// A specular map, of which only the red channel means what a mask's does.
    Background,
    /// The normal map carries its cutout in the alpha channel, where every other character family
    /// keeps it in the blue, and the mask's alpha is what shades a strand.
    Hair,
}

/// `g_SamplerNormal`'s two id variants. The only sampler this viewer takes an address mode, LOD
/// bias or anisotropy hint from: every family's normal map states the same values as its other
/// samplers, so one read stands for the whole material.
pub(super) const NORMAL_SAMPLER: [u32; 2] = [0x0C5E_C1F1, 0xAAB4_D9E9];

const ROLES: [(u32, Role); 7] = [
    (NORMAL_SAMPLER[0], Role::Normal),
    (NORMAL_SAMPLER[1], Role::Normal),
    (0x565F_8FD8, Role::Index),
    (0x8A4E_82B6, Role::Mask),
    (0x1BBC_2F12, Role::Mask),
    (0x1153_06BE, Role::Diffuse),
    (0x1E6F_EF9C, Role::Diffuse),
];

/// Material constants, by the crc32 of their name, with what a package that declares one leaves it
/// at. A `.shpk` carries the defaults but not the names, so the names come from Meddle's table.
const ALPHA_THRESHOLD: u32 = 0x29AC_0223;
const DIFFUSE_COLOR: u32 = 0x2C2A_34DD;
const EMISSIVE_COLOR: u32 = 0x38A6_4362;
const NORMAL_SCALE: u32 = 0xB554_5FBB;

/// What to clip at when a character material leaves its own threshold at zero. Hair and eyelashes
/// are authored as opaque quads with the cutout in the normal map's blue channel, so without a
/// floor they draw as rectangles.
const CUTOUT: f32 = 0.5;

/// Bit 0 of a material's shader flags.
const HIDE_BACKFACES: u32 = 1;

/// Packages this viewer has nothing to draw with. One is occlusion geometry the game never shows as
/// a surface; the other takes its color from the wearer's customization, which no file the browser
/// can reach from a model carries. Shading either of them lights bare geometry into a white shell
/// over the face it belongs to.
const UNDRAWN: [&str; 2] = ["characterocclusion.shpk", "charactertattoo.shpk"];

pub struct Material {
    held: mtrl::Material,
    shader: String,
    family: Family,
    textures: [Option<String>; 4],
    alpha_threshold: f32,
    clip: f32,
    diffuse: [f32; 3],
    emissive: [f32; 3],
    normal_scale: f32,
    cull: bool,
    /// Taken once, when the color table is handed to the context.
    table: Option<Vec<f32>>,
    rows: usize,
}

impl Material {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let material = mtrl::Material::read(Cursor::new(bytes.to_vec()))?;
        let shader = material.shader().to_owned();

        let mut textures: [Option<String>; 4] = Default::default();
        for sampler in material.samplers() {
            let Some(role) = ROLES
                .iter()
                .find(|(id, _)| *id == sampler.id())
                .map(|(_, role)| *role)
            else {
                continue;
            };
            let Some(texture) = sampler
                .texture_index()
                .and_then(|index| material.textures().get(usize::from(index)))
            else {
                continue;
            };
            textures[role as usize] = Some(texture.path().to_owned());
        }

        let constant = |id: u32| {
            material
                .constants()
                .iter()
                .find(|constant| constant.id() == id)
                .and_then(|constant| material.constant_values(constant))
        };
        let declared = constant(ALPHA_THRESHOLD)
            .and_then(|values| values.first().copied())
            .unwrap_or(0.0);
        let triple = |id, fallback: [f32; 3]| {
            constant(id)
                .and_then(|values| values.first_chunk::<3>().copied())
                .unwrap_or(fallback)
        };
        let diffuse = triple(DIFFUSE_COLOR, [1.0; 3]);
        let emissive = triple(EMISSIVE_COLOR, [0.0; 3]);
        let normal_scale = constant(NORMAL_SCALE)
            .and_then(|values| values.first().copied())
            .unwrap_or(1.0);
        // The compatibility path is the one that still binds a diffuse map, having no color table
        // row to take a diffuse color from.
        let family =
            if shader == "characterlegacy.shpk" && textures[Role::Diffuse as usize].is_some() {
                Family::Legacy
            } else if shader == "hair.shpk" {
                Family::Hair
            } else if shader.starts_with("character")
                || matches!(shader.as_str(), "skin.shpk" | "iris.shpk")
            {
                Family::Character
            } else {
                Family::Background
            };
        // Only the character families hide a cutout in the normal map; a bg normal map's third
        // channel is something else, and clipping on it would erase the surface.
        let cutout = family != Family::Background && textures[Role::Normal as usize].is_some();
        let alpha_threshold = match cutout {
            true => declared.max(CUTOUT),
            false => declared,
        };

        let cull = material.shader_flags() & HIDE_BACKFACES != 0;
        let (table, rows) = match material.color_table() {
            Some(table) => (pack(table), table.rows()),
            None => (None, 0),
        };

        Ok(Self {
            held: material,
            clip: declared,
            shader,
            family,
            textures,
            alpha_threshold,
            diffuse,
            emissive,
            normal_scale,
            cull,
            table,
            rows,
        })
    }

    pub fn texture(&self, role: Role) -> Option<&String> {
        self.textures[role as usize].as_ref()
    }

    /// The material as the file states it, for the path that runs the game's own shaders and so
    /// needs every sampler and constant rather than the four roles this viewer's own shading knows.
    pub fn held(&self) -> &mtrl::Material {
        &self.held
    }

    /// The package this material names, as a path under the shader tree.
    pub fn package(&self) -> String {
        format!("shader/sm5/shpk/{}", self.shader)
    }

    /// Every texture the material binds, by the sampler id the package knows it as.
    pub fn bound(&self) -> impl Iterator<Item = (u32, &str)> {
        self.held.samplers().iter().filter_map(|sampler| {
            let texture = sampler
                .texture_index()
                .and_then(|index| self.held.textures().get(usize::from(index)))?;
            Some((sampler.id(), texture.path()))
        })
    }

    /// The anisotropy the material's own sampler of this id asks for, `0.0` where it does not. The
    /// corpus only ever states 16x where this is set.
    pub fn anisotropic(&self, id: u32) -> f32 {
        match self.held.samplers().iter().find(|sampler| sampler.id() == id) {
            Some(sampler) if sampler.anisotropic() => 16.0,
            _ => 0.0,
        }
    }

    /// The address mode the material's own sampler of this id asks for, `Repeat` where none states
    /// otherwise. All three axes agree in the live corpus, so U stands for the sampler's own mode.
    pub fn wrap(&self, id: u32) -> mtrl::AddressMode {
        self.held
            .samplers()
            .iter()
            .find(|sampler| sampler.id() == id)
            .map_or(mtrl::AddressMode::Repeat, mtrl::Sampler::address_u)
    }

    pub fn family(&self) -> Family {
        self.family
    }

    /// How a glass surface reaches what is already drawn. `characterglass.shpk` is the one package
    /// that states a blend of its own, and it defaults to a multiply: its pass hands over the colour
    /// the frame behind is to be scaled by rather than one to be mixed into it, so blending it on
    /// coverage lays a lit card where the game tints what stands behind, and the halo chain then
    /// spreads that card.
    pub fn glass(&self) -> Option<Glass> {
        if !self.shader.ends_with("/characterglass.shpk") {
            return None;
        }
        let stated = self
            .held()
            .shader_keys()
            .iter()
            .find(|key| key.category() == GLASS_BLEND_MODE)
            .map(|key| key.value());
        Some(match stated == Some(GLASS_BLEND_ADD) {
            true => Glass::Add,
            false => Glass::Mul,
        })
    }

    pub fn drawn(&self) -> bool {
        !UNDRAWN.contains(&self.shader.as_str())
    }

    pub fn textures(&self) -> impl Iterator<Item = &String> {
        self.textures.iter().flatten()
    }

    pub fn alpha_threshold(&self) -> f32 {
        self.alpha_threshold
    }

    /// What the material itself states to clip at, which is what the game's own passes read, before
    /// the floor a draw without them puts under it.
    pub fn clip(&self) -> f32 {
        self.clip
    }

    pub fn diffuse(&self) -> [f32; 3] {
        self.diffuse
    }

    pub fn emissive(&self) -> [f32; 3] {
        self.emissive
    }

    pub fn normal_scale(&self) -> f32 {
        self.normal_scale
    }

    pub fn cull(&self) -> bool {
        self.cull
    }

    /// Kept rather than handed over, so a detail level built after this material arrived can be
    /// given it too.
    pub fn table(&self) -> Option<&[f32]> {
        self.table.as_deref()
    }

    pub fn summary(&self) -> String {
        if !self.drawn() {
            return format!("{}, not drawn", self.shader);
        }
        let named = self.textures.iter().flatten().count();
        match self.rows {
            0 => format!("{}, {named} textures", self.shader),
            rows => format!("{}, {named} textures, {rows} color rows", self.shader),
        }
    }
}

/// The color table as the fragment shader reads it: four RGBA texels a row, grouping the fields
/// that are used together. The game's own eight-texel layout carries several more, none of which
/// this shading model has anything to do with.
fn pack(table: &mtrl::ColorTable) -> Option<Vec<f32>> {
    let rows = table.rows();
    if rows == 0 {
        return None;
    }
    let extended = table.kind() == mtrl::ColorTableKind::Extended;
    let mut values = Vec::with_capacity(rows * TABLE_COLUMNS as usize * 4);
    for index in 0..rows {
        let row = table.row_values(index)?;
        // A compatibility row has no roughness field and states a specular exponent in its place;
        // the conversion is the one the game's own compatibility pass makes.
        let roughness = match extended {
            true => row.roughness,
            false => (-f32::from(f16::from_bits(*table.row(index)?.get(7)?)) / 15.0).exp2(),
        };
        values.extend(row.diffuse);
        values.push(roughness.clamp(0.0, 1.0));
        values.extend(row.specular);
        values.push(row.metalness);
        values.extend(row.emissive);
        values.push(row.sheen_rate);
        values.extend([row.sheen_tint, row.sheen_aperture, 0.0, 0.0]);
    }
    Some(values)
}

/// The file a material name points at. Character models name theirs by filename alone, against a
/// directory the name itself spells out; everything else states a whole path.
///
/// A worn piece states which of a set's colourways it is worn in, and that is the directory its own
/// materials sit in. The ones it borrows from the body it is worn over are not its to vary, and a
/// piece being inspected rather than worn states nothing, so both take the base.
///
/// A body drawn from another body's model reads its own skin all the same, which is `skin`: the one
/// mesh the game holds is named for the body it was modelled on and every body wearing it names a
/// material of its own.
pub fn path(model: &str, name: &str, variant: u16, skin: Option<u16>) -> Option<String> {
    let name = name.trim_start_matches('/');
    if name.contains('/') {
        return Some(name.to_owned());
    }
    if let Some(held) = weapon(model, name, variant) {
        return Some(held);
    }
    let stem = name.strip_prefix("mt_")?;
    let kind = stem.as_bytes().first().copied()? as char;
    let set = stem.as_bytes().get(5).copied()? as char;
    let mut body: u32 = stem.get(1..5)?.parse().ok()?;
    let part: u32 = stem.get(6..10)?.parse().ok()?;
    let worn = variant.max(1);
    let mut name = name.to_owned();
    if let ('c', 'b', Some(skin)) = (kind, set, skin) {
        body = u32::from(skin);
        name = format!("mt_c{body:04}b{part:04}{}", stem.get(10..)?);
    }
    let directory = match (kind, set) {
        ('c', 'e') => format!("chara/equipment/e{part:04}/material/v{worn:04}"),
        ('c', 'a') => format!("chara/accessory/a{part:04}/material/v{worn:04}"),
        ('c', 'b') => format!("chara/human/c{body:04}/obj/body/b{part:04}/material/v0001"),
        ('c', 'h') => format!("chara/human/c{body:04}/obj/hair/h{part:04}/material/v0001"),
        ('c', 't') => format!("chara/human/c{body:04}/obj/tail/t{part:04}/material/v0001"),
        ('c', 'f') => format!("chara/human/c{body:04}/obj/face/f{part:04}/material"),
        ('c', 'z') => format!("chara/human/c{body:04}/obj/zear/z{part:04}/material"),
        ('d', 'e') => {
            format!("chara/demihuman/d{body:04}/obj/equipment/e{part:04}/material/v{worn:04}")
        }
        ('m', 'b') => format!("chara/monster/m{body:04}/obj/body/b{part:04}/material/v{worn:04}"),
        _ => return None,
    };
    Some(format!("{directory}/{name}"))
}

/// The file a weapon's material sits in, which `Weapon::ResolveMtrlPath` builds out of the weapon
/// being drawn rather than out of the material's own name: a model is free to name a set that is
/// not the one it is filed under, and the name's own digits are rewritten to match the directory.
fn weapon(model: &str, name: &str, variant: u16) -> Option<String> {
    let (set, rest) = model.strip_prefix("chara/weapon/w")?.split_once("/obj/body/b")?;
    let set: u32 = set.parse().ok()?;
    let base: u32 = rest.get(..4)?.parse().ok()?;
    // Every machinist gun hangs the same aetherotransformer off its `_c` material, and the game
    // answers that one with the base set's rather than the gun's own.
    if set / 100 == 20 && name.as_bytes().get(14) == Some(&b'c') {
        return Some("chara/weapon/w2001/obj/body/b0001/material/v0001/mt_w2001b0001_c.mtrl".into());
    }
    let shared = shared_set(set);
    let name = match shared == set {
        true => name.to_owned(),
        false => format!("mt_w{shared:04}{}", name.strip_prefix("mt_")?.get(5..)?),
    };
    let worn = variant.max(1);
    Some(format!(
        "chara/weapon/w{shared:04}/obj/body/b{base:04}/material/v{worn:04}/{name}"
    ))
}

/// The set a weapon's materials and its `.imc` are filed under, which is not its own for the
/// off-hand half of a paired weapon: `Weapon::ResolveMtrlPath` and `ResolveImcPath` both map a set
/// whose last two digits pass fifty back by fifty, in the six families that pair across two hands.
pub(crate) fn shared_set(set: u32) -> u32 {
    const PAIRED: [u32; 6] = [3, 16, 18, 26, 30, 31];
    match PAIRED.contains(&(set / 100)) && set % 100 > 50 {
        true => set - 50,
        false => set,
    }
}

/// Whether the imc's own colourway is what files this material, which is what makes an entry
/// naming material nought leave it with no file to read at all. Everything a piece states as its
/// own is filed there; the skin it borrows from the body it is worn over is not, and that is the
/// one name `Human::ResolveMtrlPath` answers without asking the imc first.
pub fn colourwayed(model: &str, name: &str) -> bool {
    match model.starts_with("chara/equipment/") || model.starts_with("chara/accessory/") {
        true => name.trim_start_matches('/').as_bytes().get(8) != Some(&b'b'),
        false => true,
    }
}

/// The material variant a worn piece's `.imc` says `variant` actually draws with. Several variants
/// commonly share one material to avoid duplicate files, so the folder a piece's material sits in
/// is not always its own variant number. Nought is the entry stating that the slot draws no
/// material at all.
///
/// `None` wherever nothing states one: a piece looked at rather than worn, no imc to ask, one that
/// will not read, or one silent about this variant. That leaves the folder `variant` alone named.
pub fn resolve_variant(path: &str, variant: u16, imc_bytes: Option<&[u8]>) -> Option<u16> {
    if variant == 0 {
        return None;
    }
    let image_change = imc::ImageChange::read(Cursor::new(imc_bytes?.to_vec())).ok()?;
    let entry = image_change.entry(super::imc_part(path), variant)?;
    Some(u16::from(entry.material_id()))
}

#[cfg(test)]
mod tests {
    use ironworks::Ironworks;
    use ironworks::sqpack::{Install, SqPack};

    use super::{colourwayed, resolve_variant};

    const SQPACK: &str = "/home/asriel/.xlcore/ffxiv/game/sqpack";

    /// The one material a slot naming no material still draws: a piece borrows the skin showing
    /// through it from the body it is worn over, which is filed under that body rather than under
    /// the piece's own colourway. A weapon's own material spells a `b` in the same place and is
    /// not one of those.
    #[test]
    fn a_borrowed_skin_is_not_filed_under_the_wearers_colourway() {
        let worn = "chara/equipment/e0028/model/c0101e0028_top.mdl";
        assert!(colourwayed(worn, "mt_c0101e0028_top_a.mtrl"));
        assert!(!colourwayed(worn, "mt_c0101b0001_a.mtrl"));
        assert!(colourwayed(
            "chara/weapon/w5341/obj/body/b0001/model/w5341b0001.mdl",
            "mt_w5341b0001_a.mtrl"
        ));
    }

    /// Tataru's `ModelHead` names `e0005` variant 224, which has no `v0224` material on disk. Its
    /// own `e0005.imc` states variant 224's material_id as 26, and `v0026` does exist: the failing
    /// glTF export was asking `chara/equipment/e0005/material/v0224/...` for a folder the imc never
    /// pointed there.
    #[test]
    #[ignore = "reads the real local FFXIV install"]
    fn a_shared_variant_resolves_to_its_own_material() {
        let ironworks = Ironworks::new().with_resource(SqPack::new(Install::at_sqpack(SQPACK)));
        let imc = ironworks
            .file::<Vec<u8>>("chara/equipment/e0005/e0005.imc")
            .unwrap();
        let path = "chara/equipment/e0005/model/c0101e0005_met.mdl";
        assert_eq!(resolve_variant(path, 224, Some(&imc)), Some(26));
        assert_eq!(resolve_variant(path, 1, Some(&imc)), Some(1));
        assert_eq!(resolve_variant(path, 224, None), None);
    }
}
