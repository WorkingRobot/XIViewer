//! `.fpeb` facial parameter edits: the eye size a face wears.
//!
//! Nothing animates `j_f_noanim_eyesize_l` or `_r`: the name says so, and this table is the only
//! thing that states them. The file is a FlatBuffers buffer, which is what the client's own loader
//! verifies it as, and it nests model code, then face, then one entry per eye shape. Every
//! parameter names its bone by an FNV-1a hash of the bone's own name, and across all 2,248 of them
//! only those two bones are ever named.

/// The bones the table scales, left then right.
pub const BONES: [&str; 2] = ["j_f_noanim_eyesize_l", "j_f_noanim_eyesize_r"];

pub const PATH: &str = "chara/xls/facial_param_edit/facial_param_edit.fpeb";

/// Which customisation picks the entry, as `Customize` numbers them.
pub const EYE_SHAPE: u32 = 16;

const BASIS: u32 = 0x811C_9DC5;
const PRIME: u32 = 0x0100_0193;

fn hashed(name: &str) -> u32 {
    name.bytes()
        .fold(BASIS, |held, byte| (held ^ u32::from(byte)).wrapping_mul(PRIME))
}

/// A FlatBuffers buffer, read far enough to walk this one table. Every read is bounds-checked, so a
/// truncated or foreign file answers nothing rather than panicking.
struct Buf<'a>(&'a [u8]);

impl Buf<'_> {
    fn u16(&self, at: usize) -> Option<u16> {
        let held = self.0.get(at..at + 2)?;
        Some(u16::from_le_bytes([held[0], held[1]]))
    }

    fn u32(&self, at: usize) -> Option<u32> {
        let held = self.0.get(at..at + 4)?;
        Some(u32::from_le_bytes([held[0], held[1], held[2], held[3]]))
    }

    fn i32(&self, at: usize) -> Option<i32> {
        self.u32(at).map(|held| held as i32)
    }

    fn f32(&self, at: usize) -> Option<f32> {
        self.u32(at).map(f32::from_bits)
    }

    /// Where a table stands, out of the offset at `at`.
    fn table(&self, at: usize) -> Option<usize> {
        at.checked_add(self.u32(at)? as usize)
    }

    /// Where a table's `index`th field sits. Nothing where its vtable is shorter than that or
    /// leaves the slot at zero, which is FlatBuffers stating the field is at its default.
    fn field(&self, table: usize, index: usize) -> Option<usize> {
        // The offset back to the vtable is signed, and this file states plenty of them negative;
        // subtracting it as an unsigned would wrap and quietly answer the default for every field.
        let vtable = usize::try_from(table as i64 - i64::from(self.i32(table)?)).ok()?;
        let slot = vtable.checked_add(4 + 2 * index)?;
        (slot + 2 <= vtable + self.u16(vtable)? as usize).then_some(())?;
        match self.u16(slot)? {
            0 => None,
            held => table.checked_add(held as usize),
        }
    }

    /// The tables the `index`th field's own vector holds.
    fn vector(&self, table: usize, index: usize) -> Vec<usize> {
        let held = || -> Option<Vec<usize>> {
            let at = self.table(self.field(table, index)?)?;
            let count = self.u32(at)? as usize;
            (0..count)
                .map(|item| self.table(at + 4 + 4 * item))
                .collect()
        };
        held().unwrap_or_default()
    }

    /// A table's own `index`th field as a number, which is nought where the vtable leaves it out.
    fn number(&self, table: usize, index: usize) -> u32 {
        self.field(table, index)
            .and_then(|at| self.u32(at))
            .unwrap_or_default()
    }
}

/// What to scale each of [`BONES`] by for a body, its face and the eye shape it was made with.
/// `None` where the table names none of the three, which leaves the bones at the size the skeleton
/// rests them in.
pub fn scales(bytes: &[u8], code: u16, face: u16, shape: u16) -> Option<[f32; 2]> {
    let buf = Buf(bytes);
    (bytes.get(4..8)? == b"FPEB").then_some(())?;
    let root = buf.table(0)?;
    let race = buf
        .vector(root, 2)
        .into_iter()
        .find(|held| buf.number(*held, 0) == u32::from(code))?;
    let held = buf
        .vector(race, 2)
        .into_iter()
        .find(|held| buf.number(*held, 0) == u32::from(face))?;
    let group = buf.vector(held, 2).into_iter().next()?;
    let entry = buf
        .vector(group, 1)
        .into_iter()
        .find(|held| buf.number(*held, 0) == u32::from(shape))?;

    let mut found = [1.0; 2];
    for param in buf.vector(entry, 1) {
        let named = buf.number(param, 0);
        let Some(at) = BONES.iter().position(|bone| hashed(bone) == named) else {
            continue;
        };
        if let Some(scale) = buf.field(param, 3).and_then(|at| buf.f32(at)) {
            found[at] = scale;
        }
    }
    Some(found)
}


use anyhow::{Context, Result};
use egui::RichText;

use super::{Preview, facts, line, section, table};

const COLUMNS: [(&str, usize); 5] = [
    ("Body", 7),
    ("Face", 6),
    ("Eye shape", 11),
    ("Bone", 22),
    ("Scale", 0),
];

pub struct Rendered {
    identity: Vec<(&'static str, String)>,
    rows: Vec<(u32, u32, u32, &'static str, f32)>,
}

pub fn decode(path: &str, bytes: &[u8]) -> Result<Preview> {
    let buf = Buf(bytes);
    (bytes.get(4..8) == Some(b"FPEB")).then_some(()).context("a facial parameter edit")?;
    let root = buf.table(0).context("a root table")?;
    let mut rows = Vec::new();
    for race in buf.vector(root, 2) {
        let code = buf.number(race, 0);
        for face in buf.vector(race, 2) {
            let id = buf.number(face, 0);
            for group in buf.vector(face, 2) {
                for entry in buf.vector(group, 1) {
                    let shape = buf.number(entry, 0);
                    for param in buf.vector(entry, 1) {
                        let named = buf.number(param, 0);
                        let Some(at) = BONES.iter().position(|bone| hashed(bone) == named) else {
                            continue;
                        };
                        let scale = buf
                            .field(param, 3)
                            .and_then(|at| buf.f32(at))
                            .unwrap_or(1.0);
                        rows.push((code, id, shape, BONES[at], scale));
                    }
                }
            }
        }
    }
    let bodies = {
        let mut held: Vec<u32> = rows.iter().map(|(code, ..)| *code).collect();
        held.sort_unstable();
        held.dedup();
        held.len()
    };
    let identity = vec![
        ("Bodies", bodies.to_string()),
        ("Parameters", rows.len().to_string()),
        (
            "Left alone",
            rows.iter().filter(|(.., scale)| *scale == 1.0).count().to_string(),
        ),
    ];
    log::info!("assets/fpeb: {path} {bodies} bodies, {} parameters", rows.len());
    Ok(Preview::Fpeb(Box::new(Rendered { identity, rows })))
}

pub fn ui(ui: &mut egui::Ui, file: &Rendered) {
    section(ui, "Eye size");
    ui.label(
        RichText::new(
            "Nothing animates either bone, so this table is the whole of what sizes them. A body, \
             face or eye shape it states nothing for leaves them at rest.",
        )
        .weak(),
    );
    ui.add_space(4.0);
    table(ui, &COLUMNS, file.rows.len(), |ui, index| {
        let (code, face, shape, bone, scale) = &file.rows[index];
        ui.label(
            RichText::new(line(
                &COLUMNS,
                [
                    &format!("c{code:04}"),
                    face.to_string().as_str(),
                    shape.to_string().as_str(),
                    bone,
                    &format!("{scale:.3}"),
                ],
            ))
            .monospace(),
        );
    });
}

impl Rendered {
    pub fn details_ui(&self, ui: &mut egui::Ui) {
        facts(ui, "fpeb_identity", &self.identity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two bones the file names, by the hashes it names them with. Measured off the file: no
    /// other hash appears in any of its 2,248 parameters.
    #[test]
    fn the_table_names_its_bones_by_hash() {
        assert_eq!(hashed(BONES[0]), 0x3C65_F7AB);
        assert_eq!(hashed(BONES[1]), 0x4A66_0DB5);
    }

    /// The real table, off the install: `c0101` face 1 states one scale per eye shape, and the
    /// first shape leaves the eyes alone where the second takes them to 0.95.
    #[test]
    #[ignore = "reads the real local FFXIV install"]
    fn the_real_table_states_a_scale_per_eye_shape() {
        use ironworks::Ironworks;
        use ironworks::sqpack::{Install, SqPack};

        let install = Ironworks::new().with_resource(Box::new(SqPack::new(Install::at_sqpack(
            "/home/asriel/.xlcore/ffxiv/game/sqpack",
        ))));
        let bytes: Vec<u8> = install.file(PATH).expect("the table");
        assert_eq!(scales(&bytes, 101, 1, 0), Some([1.0, 1.0]));
        assert_eq!(scales(&bytes, 101, 1, 1), Some([0.95, 0.95]));
        assert_eq!(scales(&bytes, 101, 1, 4), Some([0.9, 0.9]));
        // A body, face or shape the table says nothing about leaves the eyes where they rest.
        assert_eq!(scales(&bytes, 9999, 1, 0), None);
        assert_eq!(scales(&bytes, 101, 99, 0), None);
    }

    /// Nothing but a real buffer answers, and nothing panics on one that is not.
    #[test]
    fn a_buffer_that_is_not_one_answers_nothing() {
        assert_eq!(scales(b"", 101, 1, 0), None);
        assert_eq!(scales(b"not a buffer at all", 101, 1, 0), None);
        assert_eq!(scales(&[0u8; 64], 101, 1, 0), None);
    }
}
