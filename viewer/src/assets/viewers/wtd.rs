//! `.wtd` weapon type tables: which motion class, or which attachment, each weapon model set is in.
//!
//! A four-byte header whose second half counts the entries, then one four-byte model set and one
//! three-letter code packed into four bytes for each. The table states one set per *run* of them,
//! and a lookup between two entries clamps into the lower, which is what [`code`] answers.

use anyhow::{Context, Result};
use egui::RichText;

use super::{Preview, facts, line, section, table};

const COLUMNS: [(&str, usize); 3] = [("From set", 9), ("Code", 6), ("Directory", 0)];

pub struct Rendered {
    identity: Vec<(&'static str, String)>,
    rows: Vec<(u32, String)>,
}

/// The sets and codes a table states, in the order it states them.
pub fn types(bytes: &[u8]) -> Option<Vec<(u32, String)>> {
    let count = usize::from(u16::from_le_bytes(bytes.get(2..4)?.try_into().ok()?));
    (0..count)
        .map(|at| {
            let entry = bytes.get(4 + at * 8..12 + at * 8)?;
            let set = u32::from_le_bytes(entry[..4].try_into().ok()?);
            let code = String::from_utf8(entry[4..7].iter().rev().copied().collect()).ok()?;
            Some((set, code))
        })
        .collect()
}

/// The code a weapon model set reads out of one: the last entry at or below it.
pub fn code(types: &[(u32, String)], set: u16) -> Option<&str> {
    let at = types.partition_point(|(held, _)| *held <= u32::from(set));
    types
        .get(at.saturating_sub(1))
        .map(|(_, code)| code.as_str())
}

pub fn decode(path: &str, bytes: &[u8]) -> Result<Preview> {
    let rows = types(bytes).context("a weapon type table")?;
    let distinct = {
        let mut held: Vec<&str> = rows.iter().map(|(_, code)| code.as_str()).collect();
        held.sort_unstable();
        held.dedup();
        held.len()
    };
    let identity = vec![
        ("Entries", rows.len().to_string()),
        ("Codes", distinct.to_string()),
        (
            "First set",
            rows.first().map_or_else(String::new, |(set, _)| set.to_string()),
        ),
        (
            "Last set",
            rows.last().map_or_else(String::new, |(set, _)| set.to_string()),
        ),
    ];
    log::info!("assets/wtd: {path} {} entries, {distinct} codes", rows.len());
    Ok(Preview::Wtd(Box::new(Rendered { identity, rows })))
}

pub fn ui(ui: &mut egui::Ui, file: &Rendered) {
    section(ui, "Weapon model sets");
    ui.label(
        RichText::new(
            "Each row states the code every set from it up to the next row reads, so a set between \
             two rows takes the lower one's.",
        )
        .weak(),
    );
    ui.add_space(4.0);
    table(ui, &COLUMNS, file.rows.len(), |ui, index| {
        let (set, code) = &file.rows[index];
        ui.label(
            RichText::new(line(
                &COLUMNS,
                [
                    set.to_string().as_str(),
                    code.as_str(),
                    &format!("bt_{code}_*"),
                ],
            ))
            .monospace(),
        );
    });
}

impl Rendered {
    pub fn details_ui(&self, ui: &mut egui::Ui) {
        facts(ui, "wtd_identity", &self.identity);
    }
}
