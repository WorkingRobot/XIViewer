//! Replacing a color table's own values with a stain's, the way a worn piece's dye table names.
//!
//! `chara/base_material/stainingtemplate.stm` holds the legacy, two-scalar templates a legacy color
//! table's dye rows point at; `stainingtemplate_gud.stm` holds the modern, nine-scalar ones an
//! extended table's dye rows point at. Both are read once and kept together, since a dye row's
//! template id alone does not say which file it lives in.

use std::io::Cursor;
use std::sync::Arc;

use anyhow::Result;
use half::f16;
use ironworks::file::{File, mtrl, stm};

use super::Table;
use crate::backend::Backend;

const REGULAR: &str = "chara/base_material/stainingtemplate.stm";
const GOOD: &str = "chara/base_material/stainingtemplate_gud.stm";

/// Both staining template files the game ships.
pub struct Templates {
    regular: stm::StainingTemplates,
    good: stm::StainingTemplates,
}

impl Templates {
    pub async fn read(backend: &Backend) -> Result<Self> {
        let files = backend.files();
        let regular = files.read(REGULAR).await?;
        let good = files.read(GOOD).await?;
        Ok(Self {
            regular: stm::StainingTemplates::read(Cursor::new(regular))?,
            good: stm::StainingTemplates::read(Cursor::new(good))?,
        })
    }

    /// A stain's values for the template a dye row names, trying both files since a row's id alone
    /// does not say which it came from.
    pub(super) fn pack(&self, template: u16, stain: u8) -> Option<stm::DyePack> {
        let template = u32::from(template);
        self.regular
            .template(template)
            .or_else(|| self.good.template(template))?
            .dye(stain)
    }
}

/// A dyeable field, how to read a stain's value for it, and where that lands in the extended row
/// layout every shader addresses.
type ColorField = (mtrl::DyeField, fn(&stm::DyePack) -> [f32; 3], [usize; 3]);
type ScalarField = (mtrl::DyeField, fn(&stm::DyePack) -> f32, usize);

/// Where diffuse, specular and emissive sit. Unchanged by the legacy crossover: only halves 3 and 7
/// swap meaning between the layouts.
const COLORS: [ColorField; 3] = [
    (mtrl::DyeField::Diffuse, |pack| pack.diffuse, [0, 1, 2]),
    (mtrl::DyeField::Specular, |pack| pack.specular, [4, 5, 6]),
    (mtrl::DyeField::Emissive, |pack| pack.emissive, [8, 9, 10]),
];

/// The scalars a modern, nine-scalar template carries, which only an extended table's dye row can
/// name: a legacy row's five bits reach no further than [`mtrl::DyeField::Metalness`].
const SCALARS: [ScalarField; 7] = [
    (mtrl::DyeField::Roughness, |pack| pack.roughness, 16),
    (mtrl::DyeField::SheenRate, |pack| pack.sheen_rate, 12),
    (mtrl::DyeField::SheenTint, |pack| pack.sheen_tint, 13),
    (mtrl::DyeField::SheenAperture, |pack| pack.sheen_aperture, 14),
    (mtrl::DyeField::Anisotropy, |pack| pack.anisotropy, 19),
    (
        mtrl::DyeField::SphereIndex,
        |pack| f32::from(pack.sphere_index),
        27,
    ),
    (mtrl::DyeField::SphereMask, |pack| pack.sphere_mask, 21),
];

/// Where [`mtrl::DyeField::Scalar3`] and [`mtrl::DyeField::Metalness`] land, which is the one place
/// legacy and extended disagree: `program::table` widens a legacy row's shininess (half 7) onto
/// extended half 3, and its specular mask (half 3) onto extended half 7, so a legacy dye row's
/// second scalar has to follow the specular mask there rather than to the modern metalness slot.
fn scalar3_and_metalness(kind: mtrl::ColorTableKind) -> [(mtrl::DyeField, usize); 2] {
    match kind {
        mtrl::ColorTableKind::Legacy => {
            [(mtrl::DyeField::Scalar3, 3), (mtrl::DyeField::Metalness, 7)]
        }
        _ => [
            (mtrl::DyeField::Scalar3, 3),
            (mtrl::DyeField::Metalness, 18),
        ],
    }
}

/// Replaces the fields a dye row names in one row of a color table already widened to the extended
/// layout, which is a no-op for a field the row does not dye.
fn apply(row: &mut [u16], kind: mtrl::ColorTableKind, dye: mtrl::DyeRow, pack: &stm::DyePack) {
    let half = |v: f32| f16::from_f32(v).to_bits();
    for (field, read, offsets) in COLORS {
        if !dye.dyes(field) {
            continue;
        }
        for (offset, v) in offsets.into_iter().zip(read(pack)) {
            if let Some(slot) = row.get_mut(offset) {
                *slot = half(v);
            }
        }
    }
    for (field, offset) in scalar3_and_metalness(kind) {
        if dye.dyes(field)
            && let Some(slot) = row.get_mut(offset)
        {
            *slot = half(match field {
                mtrl::DyeField::Scalar3 => pack.scalar3,
                _ => pack.metalness,
            });
        }
    }
    for (field, read, offset) in SCALARS {
        if dye.dyes(field)
            && let Some(slot) = row.get_mut(offset)
        {
            *slot = half(read(pack));
        }
    }
}

/// Where a shader addresses every row of the layout [`super::program::table`] widens a legacy table
/// into, halves per row.
const ROW: usize = 32;

/// The color table the game's own shaders read, with the wearer's stains replacing whatever fields
/// the material's own dye table names. `None` where nothing was picked or nothing in the table is
/// dyed, so the caller keeps drawing the table it already has.
pub fn table(
    base: &Table,
    colors: &mtrl::ColorTable,
    templates: &Templates,
    stains: [Option<u8>; 2],
) -> Option<Table> {
    if stains == [None, None] {
        return None;
    }
    let (raw, columns, rows) = &**base;
    let mut values = raw.clone();
    let mut dyed = false;
    for index in 0..colors.rows() {
        let Some(row) = colors.dye_row(index) else {
            continue;
        };
        if row.template() == 0 {
            continue;
        }
        let Some(Some(stain)) = stains.get(usize::from(row.channel())) else {
            continue;
        };
        let Some(pack) = templates.pack(row.template(), *stain) else {
            continue;
        };
        let targets: &[usize] = match colors.kind() {
            mtrl::ColorTableKind::Legacy => &[index * 2, index * 2 + 1],
            _ => &[index],
        };
        for &target in targets {
            let Some(slice) = values.get_mut(target * ROW..(target + 1) * ROW) else {
                continue;
            };
            apply(slice, colors.kind(), row, &pack);
        }
        dyed = true;
    }
    dyed.then(|| Arc::new((values, *columns, *rows)))
}

#[cfg(test)]
pub(super) mod test {
    use std::io::Cursor;

    use half::f16;
    use ironworks::file::{mtrl, stm, File};

    use super::{table, Templates};

    fn scalar(value: f32) -> Vec<u8> {
        f16::from_f32(value).to_le_bytes().into()
    }

    fn color(value: f32) -> Vec<u8> {
        [value, value + 1.0, value + 2.0]
            .into_iter()
            .flat_map(scalar)
            .collect()
    }

    /// A staining template file with one modern, nine-scalar template.
    fn stm_bytes(key: u32, columns: &[Vec<u8>]) -> Vec<u8> {
        let mut body = Vec::new();
        let mut end = 0;
        for column in columns {
            end += column.len();
            body.extend(u16::try_from(end / 2).unwrap().to_le_bytes());
        }
        body.extend(columns.concat());

        let mut bytes = Vec::new();
        bytes.extend(0x534Du16.to_le_bytes());
        bytes.extend(0x0201u16.to_le_bytes());
        bytes.extend(1u16.to_le_bytes());
        bytes.extend([3, 9]);
        bytes.extend(key.to_le_bytes());
        bytes.extend(0u32.to_le_bytes());
        bytes.extend(body);
        bytes
    }

    pub(in super::super) fn templates(key: u32) -> Templates {
        // diffuse, specular, emissive, then the nine scalars in template order.
        let columns = [
            color(0.4),
            color(0.04),
            Vec::new(),
            scalar(0.3),  // scalar3
            scalar(0.8),  // metalness
            scalar(0.5),  // roughness
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ];
        let good = stm::StainingTemplates::read(Cursor::new(stm_bytes(key, &columns))).unwrap();
        let empty = stm::StainingTemplates::read(Cursor::new(stm_bytes(9999, &vec![Vec::new(); 12])))
            .unwrap();
        Templates {
            regular: empty,
            good,
        }
    }

    /// A material carrying nothing but a full-size extended color table, every row zero, and a
    /// dye table naming the rows in `dye` by index.
    pub(in super::super) fn extended_material(dye: &[(u16, u8, u16)]) -> Vec<u8> {
        let mut table = vec![0u16; 32 * 32];
        for &(template, channel, fields) in dye {
            let bits = u32::from(fields) | (u32::from(template) << 16) | (u32::from(channel) << 27);
            table.extend([bits as u16, (bits >> 16) as u16]);
        }
        material(0x53, &table)
    }

    /// The same, at the legacy table's own size, with one dyed row.
    fn legacy_material(dye: u16) -> Vec<u8> {
        let mut table = vec![0u16; 16 * 16];
        table.push(dye);
        material(0x00, &table)
    }

    fn material(logs: u32, table: &[u16]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend(0x0103_0000u32.to_le_bytes());
        bytes.extend(0u16.to_le_bytes());
        bytes.extend(u16::try_from(table.len() * 2).unwrap().to_le_bytes());
        bytes.extend(1u16.to_le_bytes());
        bytes.extend(0u16.to_le_bytes());
        bytes.extend([0, 0, 0, 4]);
        bytes.push(0);
        bytes.extend((0xC | (logs << 4)).to_le_bytes());
        bytes.extend(table.iter().flat_map(|half| half.to_le_bytes()));
        bytes.extend([0; 12]);
        bytes
    }

    fn colors(bytes: Vec<u8>) -> mtrl::Material {
        mtrl::Material::read(Cursor::new(bytes)).unwrap()
    }

    fn half_at(values: &[u16], at: usize) -> f32 {
        f16::from_bits(values[at]).to_f32()
    }

    /// A literal, rounded the way `apply` rounds it, so an assertion compares what a half can
    /// state rather than the f32 the test wrote down.
    fn rounded(value: f32) -> f32 {
        f16::from_f32(value).to_f32()
    }

    #[test]
    fn dyes_the_fields_an_extended_row_names() {
        let templates = templates(1100);
        let held = colors(extended_material(&[(1100, 0, 0x19)]));
        let colors = held.color_table().unwrap();
        let base = std::sync::Arc::new((vec![0u16; 32 * 32], 8, 32));

        let dyed = table(&base, colors, &templates, [Some(3), None]).unwrap();
        assert_eq!(half_at(&dyed.0, 0), rounded(0.4));
        assert_eq!(half_at(&dyed.0, 1), rounded(1.4));
        assert_eq!(half_at(&dyed.0, 2), rounded(2.4));
        assert_eq!(half_at(&dyed.0, 3), rounded(0.3));
        assert_eq!(half_at(&dyed.0, 18), rounded(0.8));
        // The row past the dyed one, and every field the row's own bits leave alone, are untouched.
        assert_eq!(half_at(&dyed.0, 32), 0.0);
        assert_eq!(half_at(&dyed.0, 4), 0.0);
    }

    #[test]
    fn a_channel_the_wearer_left_undyed_keeps_the_base_row() {
        let templates = templates(1100);
        let held = colors(extended_material(&[(1100, 1, 0x1)]));
        let colors = held.color_table().unwrap();
        let base = std::sync::Arc::new((vec![0u16; 32 * 32], 8, 32));

        assert!(table(&base, colors, &templates, [Some(3), None]).is_none());
        let dyed = table(&base, colors, &templates, [None, Some(3)]).unwrap();
        assert_eq!(half_at(&dyed.0, 0), rounded(0.4));
    }

    /// A legacy row's second scalar is the specular mask, which sits at half 3 before the row is
    /// widened and lands on extended half 7, not the modern metalness slot.
    #[test]
    fn a_legacy_row_dyes_the_widened_specular_mask_not_metalness() {
        let templates = templates(100);
        let held = colors(legacy_material((100 << 5) | 0x19));
        let colors = held.color_table().unwrap();
        let base = std::sync::Arc::new((vec![0u16; 32 * 32], 8, 32));

        let dyed = table(&base, colors, &templates, [Some(3), None]).unwrap();
        // The doubled pair of widened rows both take the dye.
        for row in [0, 1] {
            assert_eq!(half_at(&dyed.0, row * 32 + 3), rounded(0.3), "row {row} scalar3");
            assert_eq!(half_at(&dyed.0, row * 32 + 7), rounded(0.8), "row {row} metalness->specular mask");
            assert_eq!(half_at(&dyed.0, row * 32 + 18), 0.0, "row {row} modern metalness untouched");
        }
    }

    #[test]
    fn nothing_picked_leaves_the_table_alone() {
        let templates = templates(1100);
        let held = colors(extended_material(&[(1100, 0, 0x1)]));
        let colors = held.color_table().unwrap();
        let base = std::sync::Arc::new((vec![0u16; 32], 8, 1));
        assert!(table(&base, colors, &templates, [None, None]).is_none());
    }
}
