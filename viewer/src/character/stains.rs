//! The dyes the game names, out of `Stain`.
//!
//! A row's own id is the stain a `.stm` template reads a color by, so the sheet exists only to
//! name and swatch what a template already carries a value for.

use anyhow::Result;
use egui::{Color32, Mesh, Painter, Rect, Shape};
use ironworks::excel::Language;

use crate::backend::Backend;
use crate::excel::provider::{ExcelProvider, ExcelSheet};

/// `Stain`'s columns, as byte offsets. The color's leading byte is unused; the other three are
/// red, green and blue in file order, and the sheet states no alpha. `IsMetallic` packs into bit
/// 0 of the byte at 22; `IsHousingApplicable` (unused here) is bit 1 of the same byte. `Shade` and
/// `SubOrder` are the shelf a dye sits on in the game's own dye picker and its place on that
/// shelf: the sheet's own row order is not either, since a colour added in a later patch is
/// appended past whatever shelf it actually belongs on.
const NAME: u32 = 4;
const COLOR: u32 = 8;
const SHADE: u32 = 20;
const SUB_ORDER: u32 = 21;
const IS_METALLIC: u32 = 22;

/// One dye the picker offers.
pub struct Stain {
    pub id: u8,
    pub name: String,
    pub color: Color32,
    /// `Stain.IsMetallic`.
    pub metallic: bool,
    /// `Stain.Shade`: which shelf of the game's own dye picker this dye sits on.
    pub shade: u8,
}

const CORNER: f32 = 2.0;

/// The flat fill every swatch gets, plus a diagonal sheen for a metallic dye. The gradient is
/// inset by the fill's own corner radius so its straight edges never poke past the rounded fill
/// into the surrounding theme.
pub fn paint(painter: &Painter, rect: Rect, color: Color32, metallic: bool) {
    painter.rect_filled(rect, CORNER, color);
    if !metallic {
        return;
    }
    let inset = rect.shrink(CORNER);
    let mut mesh = Mesh::default();
    mesh.colored_vertex(inset.left_top(), Color32::from_white_alpha(70));
    mesh.colored_vertex(inset.right_top(), Color32::TRANSPARENT);
    mesh.colored_vertex(inset.right_bottom(), Color32::from_black_alpha(60));
    mesh.colored_vertex(inset.left_bottom(), Color32::TRANSPARENT);
    mesh.add_triangle(0, 1, 2);
    mesh.add_triangle(0, 2, 3);
    painter.add(Shape::mesh(mesh));
}

/// Every stain the game names, ordered the way its own dye picker shelves them: by `Shade`, then
/// by `SubOrder` within it. Row 0 is the unstained slot and is never returned.
pub async fn read(backend: &Backend, language: Language) -> Result<Vec<Stain>> {
    let excel = backend.excel();
    let sheet = excel.get_sheet("Stain", language).await?;
    let mut found = Vec::new();
    for id in sheet.get_row_ids() {
        let Ok(id) = u8::try_from(id) else { continue };
        if id == 0 {
            continue;
        }
        let Ok(row) = sheet.get_row(u32::from(id)) else {
            continue;
        };
        let Ok(name) = row.read_string(NAME) else {
            continue;
        };
        let name = name.to_string();
        if name.is_empty() {
            continue;
        }
        let color: u32 = row.read(COLOR).unwrap_or(0);
        let [_, r, g, b] = color.to_be_bytes();
        let metallic = row.read_packed_bool(IS_METALLIC, 0).unwrap_or(false);
        let shade = row.read::<u8>(SHADE).unwrap_or(0);
        let order: u8 = row.read(SUB_ORDER).unwrap_or(0);
        found.push((
            shade,
            order,
            Stain {
                id,
                name,
                color: Color32::from_rgb(r, g, b),
                metallic,
                shade,
            },
        ));
    }
    found.sort_by_key(|(shade, order, _)| (*shade, *order));
    let found: Vec<Stain> = found.into_iter().map(|(.., stain)| stain).collect();
    log::info!("character: {} dyes to pick from", found.len());
    Ok(found)
}
