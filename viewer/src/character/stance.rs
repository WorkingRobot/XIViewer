//! What a body stands in: the pose the weapon it holds puts it in, the motions it draws and
//! sheathes with, and how long the game takes to blend one motion into another.
//!
//! A weapon's motion class is its model set's own entry in `chara/xls/weapontype/motion.wtd`, a
//! sorted table of set to three-letter code the client binary-searches and clamps to the last
//! entry at or below the set it asks for. Main hand and off hand each read one, and the pair names
//! the directory the drawn packs are filed in: `bt_swd_sld`, `bt_2ax_emp`, and so on, the same
//! spelling `bt_%c%c%c_%c%c%c` builds. Empty hands read `emp`, and `bt_emp_emp`'s own idle pack
//! holds no animation at all, which is the game stating that bare hands have no drawn pose.
//!
//! Blend lengths are the game's own: `MotionTimeline` gives each motion name a blend group and
//! `MotionTimelineBlendTable` gives every ordered pair of groups a frame count, which the client
//! reads at a floor of one frame and 30 frames a second. A motion the sheet does not name is in
//! group 0, and a pair the table does not state falls back to the one out of group 0, then to 0
//! to 0: the client fills its matrix in place from the top, so every unstated pair in row 0 is
//! already the 0 to 0 one by the time any other row reads it.
//!
//! Which body a pack is actually read out of is `chara/xls/animation/papLoadTable.plt`'s answer.
//! Most bodies file no animation of their own and most classes share another's, so the client asks
//! the table for a body, animation set and class directory and reads the same pack name out of
//! what it gets back.

use std::collections::HashMap;

use anyhow::{Context, Result};
use ironworks::excel::Language;

use crate::assets::viewers::wtd;
use crate::backend::Backend;
use crate::excel::provider::{ExcelProvider, ExcelSheet as _};

/// The animation set a player character's own packs are filed under.
const SET: u16 = 1;

/// The table naming which motion class each weapon model set is in.
const MOTION_TYPES: &str = "chara/xls/weapontype/motion.wtd";

/// The table naming where each body's packs are really filed.
const PACK_TABLE: &str = "chara/xls/animation/papLoadTable.plt";

/// What both hands read as with nothing in them.
const EMPTY: &str = "emp";

/// The directory every body's own unarmed packs are filed in, whatever it holds.
pub const COMMON: &str = "bt_common";

/// The motion a body stands in sheathed and drawn, and the partial motions it draws and sheathes
/// with, which are the same names in every class's own packs.
pub const SHEATHED: &str = "cbnm_id0";
pub const DRAWN: &str = "cbbm_id0";
pub const DRAW: &str = "cbbp_a_activ";
pub const SHEATHE: &str = "cbbp_a_deact";

/// Frames a second the blend table counts in.
const FPS: f32 = 30.0;

/// `MotionTimeline`'s filename and blend group, and `MotionTimelineBlendTable`'s destination
/// group, source group and player-character frame count, as byte offsets.
const FILENAME: u32 = 0;
const GROUP: u32 = 4;
const DEST: u32 = 0;
const SOURCE: u32 = 1;
const FRAMES: u32 = 2;
/// The frame counts the other kinds of body read, which the table is padded past its last real
/// row with rows of nothing but zeroes; the client tells those apart by every field being nought
/// and files no blend for them, so the pair they would spell keeps whatever a real row said.
const OTHERS: [u32; 3] = [3, 4, 5];

/// The path a pack sits at: the body its model id spells, the animation set, the class directory
/// and the name inside it, which the client formats as `%s/animation/a%04d/%s/%s.pap`. A model id
/// carries the kind it is in its high half, which the client switches monsters, demihumans and
/// weapons on; asked about a body, the table only ever answers with one, so this spells that.
fn path(model: u32, set: u16, held: &str, file: &str) -> String {
    let code = model & 0xffff;
    format!("chara/human/c{code:04}/animation/a{set:04}/{held}/{file}.pap")
}

/// Everything a change of stance is resolved out of: which class a weapon puts the body in, where
/// its packs are filed, and how long the game blends one motion into another.
pub struct Stance {
    /// Weapon model set to motion class, in the table's own ascending order.
    classes: Vec<(u32, String)>,
    /// Where each body reads a pack out of.
    packs: Packs,
    /// Which blend group each motion's own name is in.
    groups: HashMap<String, u8>,
    /// Frames the blend from one group into another runs for.
    blends: HashMap<(u8, u8), u8>,
}

impl Stance {
    pub async fn read(backend: &Backend, language: Language) -> Result<Self> {
        let classes = wtd::types(&backend.files().read(MOTION_TYPES).await?)
            .context("weapon motion types")?;
        let packs =
            Packs::read(&backend.files().read(PACK_TABLE).await?).context("pap load table")?;

        let excel = backend.excel();
        let motions = excel.get_sheet("MotionTimeline", language).await?;
        let mut groups = HashMap::new();
        for id in motions.get_row_ids() {
            let Ok(row) = motions.get_row(id) else {
                continue;
            };
            if let (Ok(name), Ok(group)) = (row.read_string(FILENAME), row.read::<u8>(GROUP))
                && !name.to_string().is_empty()
            {
                groups.insert(name.to_string(), group);
            }
        }

        let table = excel.get_sheet("MotionTimelineBlendTable", language).await?;
        let mut blends = HashMap::new();
        for id in table.get_row_ids() {
            let Ok(row) = table.get_row(id) else {
                continue;
            };
            let read = |at| row.read::<u8>(at).unwrap_or_default();
            let (dest, source, frames) = (read(DEST), read(SOURCE), read(FRAMES));
            let stated = OTHERS.iter().fold(dest | source | frames, |held, at| held | read(*at));
            if stated != 0 {
                blends.insert((source, dest), frames);
            }
        }
        log::info!(
            "character: {} weapon classes, {} blend groups, {} blends, {} packs filed",
            classes.len(),
            groups.len(),
            blends.len(),
            packs.packs.len()
        );
        Ok(Self {
            classes,
            packs,
            groups,
            blends,
        })
    }

    /// The motion class a weapon model set is in. Nothing held is an empty hand.
    pub fn class(&self, set: Option<u16>) -> &str {
        set.and_then(|set| wtd::code(&self.classes, set)).unwrap_or(EMPTY)
    }

    /// The directory a pair of weapons files its drawn packs under.
    pub fn directory(&self, main: Option<u16>, off: Option<u16>) -> String {
        format!("bt_{}_{}", self.class(main), self.class(off))
    }

    /// Where the body `code` names really reads `file` out of, under the class directory `held`.
    pub fn pack(&self, code: u16, held: &str, file: &str) -> String {
        let (model, set, held) = self.packs.filed(u32::from(code), SET, held, file);
        path(model, set, held, file)
    }

    /// The idle pack every body plays with nothing drawn.
    pub fn sheathed_pack(&self, code: u16) -> String {
        self.pack(code, COMMON, "resident/idle")
    }

    /// How long the game blends `from` into `to`, in seconds. A pair the table says nothing about
    /// takes the fallbacks the client's own lookup fills the matrix in with, and a stated zero
    /// still runs for a frame.
    pub fn fade(&self, from: &str, to: &str) -> f32 {
        let (source, dest) = (self.group(from), self.group(to));
        let frames = [(source, dest), (0, dest), (0, 0)]
            .iter()
            .find_map(|pair| self.blends.get(pair))
            .copied()
            .unwrap_or_default();
        f32::from(frames.max(1)) / FPS
    }

    /// The blend group a motion's own name is in. A name the sheet does not carry reads as group
    /// 0, which is what the client's own lookup answers for one it cannot find a row for.
    fn group(&self, motion: &str) -> u8 {
        self.groups.get(motion).copied().unwrap_or_default()
    }
}

/// The flag on a pack record that makes its name stand for every one it prefixes, up to its `*`.
const WILDCARD: u32 = 0x400;

/// One pack the table knows, as the class directory it is filed under and the name inside it.
struct Pack {
    dir: usize,
    name: String,
    wildcard: bool,
}

impl Pack {
    fn holds(&self, name: &str) -> bool {
        match self.wildcard {
            true => match self.name.split_once('*') {
                Some((prefix, _)) => name.starts_with(prefix),
                None => false,
            },
            false => self.name == name,
        }
    }
}

/// A body and animation set, and which packs it reads out of somewhere else: one bit per four
/// packs, and the redirections of every body sharing a run of bits sit one after another.
struct Body {
    model: u32,
    set: u16,
    base: u16,
    redirected: [u64; 20],
}

impl Body {
    /// Where this body's redirection for the pack at `at` sits, if it has one. The bit's own
    /// four redirections follow every earlier set bit's four, from the body's own base.
    fn redirect(&self, at: usize) -> Option<usize> {
        let (group, word) = (at >> 2, at >> 8);
        let bit = group & 63;
        let held = *self.redirected.get(word)?;
        if held >> bit & 1 == 0 {
            return None;
        }
        let rank = self.redirected[..word]
            .iter()
            .map(|word| word.count_ones())
            .sum::<u32>()
            + (held & ((1 << bit) - 1)).count_ones();
        Some(4 * (usize::from(self.base) + rank as usize) + (at & 3))
    }
}

/// Where a redirected pack is read from instead.
struct Redirect {
    model: u32,
    set: u16,
    dir: usize,
}

/// `papLoadTable.plt`: two counts and the bounds of its two string blocks, then a twelve-byte
/// record for every pack it knows, a hundred and sixty-eight for every body, and eight for every
/// redirection. Directories and pack names are both offsets into the string blocks.
pub struct Packs {
    /// Class directory names against the offsets the records spell them as, in the block's order.
    dirs: Vec<(usize, String)>,
    packs: Vec<Pack>,
    bodies: Vec<Body>,
    redirects: Vec<Redirect>,
}

impl Packs {
    fn read(bytes: &[u8]) -> Option<Self> {
        let half = |at: usize| Some(u16::from_le_bytes(bytes.get(at..at + 2)?.try_into().ok()?));
        let word = |at: usize| Some(u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?));
        let long = |at: usize| Some(u64::from_le_bytes(bytes.get(at..at + 8)?.try_into().ok()?));

        let spelled = word(8)? as usize;
        let named = word(12)? as usize;
        let dirs = names(bytes.get(spelled..named)?);
        let tails = bytes.get(named..)?;

        let packs = (0..usize::from(half(0)?))
            .map(|at| {
                let at = 16 + at * 12;
                Some(Pack {
                    dir: word(at + 4)? as usize,
                    name: name(tails, word(at + 8)? as usize)?,
                    wildcard: word(at)? & WILDCARD != 0,
                })
            })
            .collect::<Option<Vec<_>>>()?;

        let after = 16 + packs.len() * 12;
        let bodies = (0..usize::from(half(2)?))
            .map(|at| {
                let at = after + at * 168;
                let mut redirected = [0; 20];
                for (index, word) in redirected.iter_mut().enumerate() {
                    *word = long(at + 8 + index * 8)?;
                }
                Some(Body {
                    model: word(at)?,
                    set: half(at + 4)?,
                    base: half(at + 6)?,
                    redirected,
                })
            })
            .collect::<Option<Vec<_>>>()?;

        let after = after + bodies.len() * 168;
        let redirects = (0..spelled.checked_sub(after)? / 8)
            .map(|at| {
                let at = after + at * 8;
                Some(Redirect {
                    model: word(at)?,
                    set: half(at + 4)?,
                    dir: usize::from(half(at + 6)?),
                })
            })
            .collect::<Option<Vec<_>>>()?;

        Some(Self {
            dirs,
            packs,
            bodies,
            redirects,
        })
    }

    /// Which body, animation set and class directory the game reads `file` out of, asked for by a
    /// body under `held`. Anything the table says nothing about is read where it was asked for.
    fn filed<'a>(&'a self, model: u32, set: u16, held: &'a str, file: &str) -> (u32, u16, &'a str) {
        let asked = (model, set, held);
        let (Some(spelled), Some(body)) = (
            self.spelled(held),
            self.bodies
                .iter()
                .find(|body| body.model == model && body.set == set),
        ) else {
            return asked;
        };
        let Some(at) = self
            .packs
            .iter()
            .position(|pack| pack.dir == spelled && pack.holds(file))
        else {
            return asked;
        };
        match body.redirect(at).and_then(|at| self.redirects.get(at)) {
            Some(read) => (read.model, read.set, self.spells(read.dir).unwrap_or(held)),
            None => asked,
        }
    }

    fn spelled(&self, name: &str) -> Option<usize> {
        self.dirs
            .iter()
            .find_map(|(at, held)| (held == name).then_some(*at))
    }

    fn spells(&self, at: usize) -> Option<&str> {
        self.dirs
            .binary_search_by_key(&at, |(at, _)| *at)
            .ok()
            .map(|found| self.dirs[found].1.as_str())
    }
}

/// Every name of a block of NUL-terminated ones, against the offset it starts at.
fn names(block: &[u8]) -> Vec<(usize, String)> {
    let mut found = Vec::new();
    let mut at = 0;
    while let Some(held) = name(block, at) {
        let next = at + held.len() + 1;
        found.push((at, held));
        at = next;
    }
    found
}

/// The one name a block of NUL-terminated ones holds at `at`.
fn name(block: &[u8], at: usize) -> Option<String> {
    let rest = block.get(at..)?;
    let end = rest.iter().position(|byte| *byte == 0)?;
    Some(std::str::from_utf8(&rest[..end]).ok()?.to_owned())
}

#[cfg(test)]
mod tests {
    use ironworks::Ironworks;
    use ironworks::file::File as _;
    use ironworks::file::pap::AnimationPack;
    use ironworks::sqpack::{Install, SqPack};

    use super::*;

    fn table(entries: &[(u32, &str)]) -> Vec<u8> {
        let mut bytes = vec![1, 0];
        bytes.extend((entries.len() as u16).to_le_bytes());
        for (set, code) in entries {
            bytes.extend(set.to_le_bytes());
            bytes.extend([code.as_bytes()[2], code.as_bytes()[1], code.as_bytes()[0], 0]);
        }
        bytes
    }

    fn stance(entries: &[(u32, &str)]) -> Stance {
        Stance {
            classes: wtd::types(&table(entries)).expect("the table reads"),
            packs: Packs::read(&plt()).expect("the pack table reads"),
            groups: HashMap::new(),
            blends: HashMap::new(),
        }
    }

    const DIRS: [&str; 2] = ["bt_a", "bt_b"];
    const NAMES: [&str; 5] = [
        "resident/idle",
        "battle/*",
        "resident/sub",
        "resident/move_a",
        "resident/move_b",
    ];

    /// A table of `papLoadTable.plt`'s own shape: eight packs over two class directories, so two
    /// groups of four, and two bodies redirecting them from different bases.
    fn plt() -> Vec<u8> {
        let packs: [(usize, usize, u32); 8] = [
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, WILDCARD),
            (0, 2, 0),
            (1, 2, 0),
            (0, 3, 0),
            (1, 3, 0),
            (0, 4, 0),
        ];
        let bodies: [(u32, u16, u16, u64); 2] = [(101, 1, 0, 0b11), (201, 1, 2, 0b10)];
        let filed: [(u32, u16, usize); 12] = [
            (101, 1, 0),
            (201, 1, 0),
            (101, 8, 1),
            (0, 0, 0),
            (301, 1, 1),
            (0, 0, 0),
            (401, 1, 0),
            (0, 0, 0),
            (501, 1, 0),
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0),
        ];
        let offset = |list: &[&str], want: usize| -> u32 {
            list[..want].iter().map(|held| held.len() as u32 + 1).sum()
        };
        let block = |list: &[&str]| -> Vec<u8> {
            list.iter()
                .flat_map(|held| held.bytes().chain([0]))
                .collect()
        };

        let spelled = 16 + packs.len() * 12 + bodies.len() * 168 + filed.len() * 8;
        let mut bytes = Vec::new();
        bytes.extend((packs.len() as u16).to_le_bytes());
        bytes.extend((bodies.len() as u16).to_le_bytes());
        bytes.extend(0u32.to_le_bytes());
        bytes.extend((spelled as u32).to_le_bytes());
        bytes.extend(((spelled + block(&DIRS).len()) as u32).to_le_bytes());
        for (dir, name, flags) in packs {
            bytes.extend(flags.to_le_bytes());
            bytes.extend(offset(&DIRS, dir).to_le_bytes());
            bytes.extend(offset(&NAMES, name).to_le_bytes());
        }
        for (model, set, base, redirected) in bodies {
            bytes.extend(model.to_le_bytes());
            bytes.extend(set.to_le_bytes());
            bytes.extend(base.to_le_bytes());
            bytes.extend(redirected.to_le_bytes());
            bytes.extend([0; 152]);
        }
        for (model, set, dir) in filed {
            bytes.extend(model.to_le_bytes());
            bytes.extend(set.to_le_bytes());
            bytes.extend((offset(&DIRS, dir) as u16).to_le_bytes());
        }
        bytes.extend(block(&DIRS));
        bytes.extend(block(&NAMES));
        bytes
    }

    /// The redirection the table states for a pack, and the three ways of asking it something it
    /// does not state, which all read the pack where it was asked for.
    #[test]
    fn a_body_reads_a_pack_out_of_the_body_the_table_sends_it_to() {
        let held = Packs::read(&plt()).expect("the table reads");
        assert_eq!(
            held.filed(101, 1, "bt_a", "resident/idle"),
            (101, 1, "bt_a")
        );
        assert_eq!(
            held.filed(101, 1, "bt_b", "resident/idle"),
            (201, 1, "bt_a")
        );
        assert_eq!(
            held.filed(999, 1, "bt_a", "resident/idle"),
            (999, 1, "bt_a")
        );
        assert_eq!(
            held.filed(101, 1, "bt_c", "resident/idle"),
            (101, 1, "bt_c")
        );
        assert_eq!(held.filed(101, 1, "bt_a", "elsewhere"), (101, 1, "bt_a"));
    }

    /// A pack in the second group of four, which is only reached by counting the bits set before
    /// it, and read for two bodies whose redirections start at different bases.
    #[test]
    fn a_redirection_is_counted_from_the_bits_set_before_it() {
        let held = Packs::read(&plt()).expect("the table reads");
        assert_eq!(held.filed(101, 1, "bt_b", "resident/sub"), (301, 1, "bt_b"));
        assert_eq!(
            held.filed(101, 1, "bt_b", "resident/move_a"),
            (401, 1, "bt_a")
        );
        assert_eq!(held.filed(201, 1, "bt_b", "resident/sub"), (501, 1, "bt_a"));
        assert_eq!(
            held.filed(201, 1, "bt_a", "resident/idle"),
            (201, 1, "bt_a"),
            "a group the body leaves unset stays where it was asked for"
        );
    }

    /// A name ending in `*` stands for every one it prefixes, and the redirection can move the
    /// animation set as well as the body.
    #[test]
    fn a_starred_name_stands_for_everything_it_prefixes() {
        let held = Packs::read(&plt()).expect("the table reads");
        assert_eq!(
            held.filed(101, 1, "bt_b", "battle/battle_start"),
            (101, 8, "bt_b")
        );
        assert_eq!(held.filed(101, 1, "bt_b", "battl"), (101, 1, "bt_b"));
    }

    #[test]
    fn a_stance_names_the_pack_the_table_files_it_under() {
        let held = stance(&[(101, "sld"), (201, "swd")]);
        assert_eq!(
            held.pack(101, "bt_b", "resident/idle"),
            "chara/human/c0201/animation/a0001/bt_a/resident/idle.pap"
        );
    }

    /// The three-letter codes `motion.wtd` itself carries for the first weapon sets it names.
    #[test]
    fn a_weapon_set_reads_the_class_the_table_files_it_under() {
        let held = stance(&[(101, "sld"), (201, "swd"), (301, "clw"), (401, "2ax")]);
        assert_eq!(held.class(Some(201)), "swd");
        assert_eq!(held.class(Some(250)), "swd", "between two entries reads the lower");
        assert_eq!(held.class(Some(401)), "2ax");
        assert_eq!(held.class(Some(50)), "sld", "below the first still reads one");
        assert_eq!(held.class(None), "emp");
    }

    #[test]
    fn a_pair_of_weapons_names_the_directory_their_packs_are_filed_in() {
        let held = stance(&[(101, "sld"), (201, "swd"), (401, "2ax")]);
        assert_eq!(held.directory(Some(201), Some(101)), "bt_swd_sld");
        assert_eq!(held.directory(Some(401), None), "bt_2ax_emp");
        assert_eq!(held.directory(None, None), "bt_emp_emp");
    }

    /// The blend table's own fallback order, which fills every unstated pair from the entry into
    /// group 0 before the one that is both, and never from the one out of it: the client fills its
    /// matrix in place, so row 0 is already filled by the time any other row falls back through it.
    #[test]
    fn an_unstated_blend_falls_back_the_way_the_client_fills_its_matrix() {
        let mut held = stance(&[]);
        held.groups.insert("from".to_owned(), 7);
        held.groups.insert("to".to_owned(), 6);
        held.blends.insert((0, 0), 3);
        assert_eq!(held.fade("from", "to"), 3.0 / FPS);
        held.blends.insert((7, 0), 9);
        assert_eq!(held.fade("from", "to"), 3.0 / FPS, "the one out of group 0 is never reached");
        held.blends.insert((0, 6), 4);
        assert_eq!(held.fade("from", "to"), 4.0 / FPS);
        held.blends.insert((7, 6), 12);
        assert_eq!(held.fade("from", "to"), 12.0 / FPS);
    }

    /// A motion neither name is a `MotionTimeline` row for still blends: both read as group 0, and
    /// the pair that is both is what the whole matrix falls back through.
    #[test]
    fn a_motion_the_sheet_never_names_blends_out_of_group_nought() {
        let mut held = stance(&[]);
        held.blends.insert((0, 0), 10);
        assert_eq!(held.fade("cbem_sp63", "cbem_dance16_2lp"), 10.0 / FPS);
    }

    /// Polls a future to completion on the current thread with no real waker, which is enough for
    /// the local install's own I/O.
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

    /// The install's own tables, read end to end: the classes a gladius and a shield are in, and
    /// the twelve frames the blend table gives a change of standing pose.
    #[test]
    #[ignore = "reads the real local FFXIV install"]
    fn the_installs_own_tables_name_a_gladius_and_shield_stance() {
        let backend = block_on(crate::backend::Backend::new(crate::settings::BackendConfig {
            api_url: "https://exd.camora.dev".to_owned(),
            location: crate::settings::InstallLocation::Sqpack(
                "/home/asriel/.xlcore/ffxiv/game/sqpack".to_owned(),
            ),
            schema: crate::settings::SchemaLocation::Local("/home/asriel/Code/EXDSchema".to_owned()),
        }))
        .expect("the local install");
        let held = block_on(Stance::read(&backend, Language::English)).expect("the tables");
        assert_eq!(held.class(Some(201)), "swd", "a gladius");
        assert_eq!(held.class(Some(101)), "sld", "a shield");
        assert_eq!(held.directory(Some(201), Some(101)), "bt_swd_sld");
        assert_eq!(held.directory(Some(401), None), "bt_2ax_emp", "a two-hander");
        assert_eq!(held.fade(SHEATHED, DRAWN), 12.0 / FPS);
        assert_eq!(held.fade(DRAWN, DRAW), 4.0 / FPS);
    }

    /// What the install's own tables price the two changes the creator makes at. An emote motion
    /// no `MotionTimeline` row names still blends, out of the pair every unstated one falls back
    /// through; a change of facial pose takes the one into the group every `cfxf_` name is in,
    /// whether or not anything was on the face before it.
    #[test]
    #[ignore = "reads the real local FFXIV install"]
    fn the_installs_own_tables_price_an_emote_and_a_change_of_face() {
        let backend = block_on(crate::backend::Backend::new(crate::settings::BackendConfig {
            api_url: "https://exd.camora.dev".to_owned(),
            location: crate::settings::InstallLocation::Sqpack(
                "/home/asriel/.xlcore/ffxiv/game/sqpack".to_owned(),
            ),
            schema: crate::settings::SchemaLocation::Local("/home/asriel/Code/EXDSchema".to_owned()),
        }))
        .expect("the local install");
        let held = block_on(Stance::read(&backend, Language::English)).expect("the tables");
        assert_eq!(held.fade(SHEATHED, "cbem_dance16_2lp"), 10.0 / FPS);
        assert_eq!(held.fade("cfxf_smile", "cfxf_grin"), 5.0 / FPS);
        assert_eq!(held.fade("", "cfxf_grin"), 5.0 / FPS, "the face was at rest");
        assert_eq!(held.fade("cfxf_grin", ""), 10.0 / FPS, "and back to rest again");
    }

    const SQPACK: &str = "/home/asriel/.xlcore/ffxiv/game/sqpack";

    fn install() -> Ironworks {
        Ironworks::new().with_resource(Box::new(SqPack::new(Install::at_sqpack(SQPACK))))
    }

    fn installed_packs(install: &Ironworks) -> Packs {
        let bytes: Vec<u8> = install.file(PACK_TABLE).expect("the pack table");
        Packs::read(&bytes).expect("a readable pack table")
    }

    /// The four motion names this module stands on, read out of the packs it names them in. Which
    /// of `sub.pap`'s partial motions the game plays on a draw is inferred from their names, the
    /// half-second they run and the upper-body slot `ActionTimeline` rows 1 and 2 state; that
    /// they are the only two in the pack, in every class, is what this holds. `papLoadTable.plt`
    /// settles which body files a pack and nothing about what is inside one, so it cannot settle
    /// this.
    #[test]
    #[ignore = "reads the real local FFXIV install"]
    fn the_packs_a_stance_names_hold_the_motions_it_asks_them_for() {
        let install = install();
        let packs = installed_packs(&install);
        let filed = |held: &str, file: &str| {
            let (model, set, held) = packs.filed(101, SET, held, file);
            path(model, set, held, file)
        };
        let names = |path: String| -> Vec<String> {
            let bytes: Vec<u8> = install.file(&path).expect("the pack");
            AnimationPack::read(std::io::Cursor::new(bytes))
                .expect("a readable pack")
                .animations()
                .iter()
                .map(|animation| animation.name().to_owned())
                .collect()
        };
        assert!(names(filed(COMMON, "resident/idle")).contains(&SHEATHED.to_owned()));
        for class in ["bt_swd_sld", "bt_2ax_emp", "bt_clw_clw", "bt_2gn_emp"] {
            let idle = names(filed(class, "resident/idle"));
            assert!(idle.contains(&DRAWN.to_owned()), "{class}: {idle:?}");
            let sub = names(filed(class, "resident/sub"));
            assert!(sub.contains(&DRAW.to_owned()), "{class}: {sub:?}");
            assert!(sub.contains(&SHEATHE.to_owned()), "{class}: {sub:?}");
        }
    }

    /// A class the game files no packs of its own for reads another's, so a battle emote asked
    /// for under the class a lone sword puts a body in is really read out of the sword and shield
    /// one, which is the only place it is filed at all.
    #[test]
    #[ignore = "reads the real local FFXIV install"]
    fn a_class_with_no_packs_of_its_own_reads_a_battle_emote_out_of_the_one_it_shares() {
        let packs = installed_packs(&install());
        let sqpack = SqPack::new(Install::at_sqpack(SQPACK));
        let filed = |held: &str, file: &str| {
            let (model, set, held) = packs.filed(101, SET, held, file);
            path(model, set, held, file)
        };
        let bare = filed("bt_emp_emp", "emote/battle02");
        assert_eq!(
            bare,
            "chara/human/c0101/animation/a0001/bt_emp_emp/emote/battle02.pap"
        );
        assert!(
            !sqpack.exists(&bare).unwrap_or(false),
            "bare hands are moved nowhere and file no battle emote, so there is none to play"
        );
        let held = filed("bt_swd_emp", "emote/battle02");
        assert_eq!(
            held,
            "chara/human/c0101/animation/a0001/bt_swd_sld/emote/battle02.pap"
        );
        assert!(sqpack.exists(&held).unwrap_or(false));
        assert!(
            !sqpack
                .exists(&path(101, SET, "bt_swd_emp", "emote/battle02"))
                .unwrap_or(false),
            "the class it was asked for files nothing of its own"
        );
        // The table moves every emote a lone sword asks for, whether or not that class files one:
        // the ones every body shares are not there either way, and fall back to the shared
        // directory on not being found.
        let shared = filed("bt_swd_emp", "emote/goodbye_st");
        assert_eq!(
            shared,
            "chara/human/c0101/animation/a0001/bt_swd_sld/emote/goodbye_st.pap"
        );
        assert!(!sqpack.exists(&shared).unwrap_or(false));
        assert!(
            sqpack
                .exists(&filed(COMMON, "emote/goodbye_st"))
                .unwrap_or(false)
        );
    }

    /// What the table moves a request onto, against the install. A redirection is only reached by
    /// counting the bits set before it, so a rank or base read off by one lands on another body's
    /// answer, and this is what says it landed on the right one. Plenty of what it moves is
    /// missing at both ends: the pack list is one table for every body and keeps names for classes
    /// the game no longer ships, `bt_axe_sld` among them. Exactly one move goes the wrong way,
    /// onto a pack only the body that asked for it holds, and the table itself is what says so.
    #[test]
    #[ignore = "reads the real local FFXIV install"]
    fn the_table_moves_a_pack_onto_one_the_install_holds() {
        let sqpack = SqPack::new(Install::at_sqpack(SQPACK));
        let packs = installed_packs(&install());
        let (mut moved, mut gained, mut lost) = (0, 0, Vec::new());
        for body in packs.bodies.iter().filter(|body| body.model >> 16 == 0) {
            for pack in packs.packs.iter().filter(|pack| !pack.wildcard) {
                let Some(held) = packs.spells(pack.dir) else {
                    continue;
                };
                let filed = packs.filed(body.model, body.set, held, &pack.name);
                if filed == (body.model, body.set, held) {
                    continue;
                }
                moved += 1;
                let asked = path(body.model, body.set, held, &pack.name);
                let onto = path(filed.0, filed.1, filed.2, &pack.name);
                let (before, after) = (
                    sqpack.exists(&asked).unwrap_or(false),
                    sqpack.exists(&onto).unwrap_or(false),
                );
                match (before, after) {
                    (false, true) => gained += 1,
                    (true, false) => lost.push(format!("{asked} -> {onto}")),
                    _ => (),
                }
            }
        }
        println!(
            "{moved} moved, {gained} onto a pack that ships, {} lost",
            lost.len()
        );
        assert!(gained > moved / 2);
        assert_eq!(
            lost,
            [concat!(
                "chara/human/c1801/animation/a0001/bt_common/resident/mount.pap -> ",
                "chara/human/c0801/animation/a0001/bt_common/resident/mount.pap"
            )]
        );
    }

    /// The bases tile the redirections: each body's own start is the number of bits set by every
    /// body before it, and four redirections per bit fills the block the header bounds exactly.
    #[test]
    #[ignore = "reads the real local FFXIV install"]
    fn the_redirections_tile_the_block_the_header_bounds() {
        let packs = installed_packs(&install());
        let mut run = 0;
        for body in &packs.bodies {
            assert_eq!(usize::from(body.base), run, "c{:04}", body.model);
            run += body
                .redirected
                .iter()
                .map(|word| word.count_ones() as usize)
                .sum::<usize>();
        }
        assert_eq!(4 * run, packs.redirects.len());
    }

    /// Every race reaches a drawn pose without walking a lineage: the table itself names the body
    /// each one's sword and shield packs are filed under.
    #[test]
    #[ignore = "reads the real local FFXIV install"]
    fn every_race_reads_the_class_it_stands_in_off_the_table() {
        let install = install();
        let packs = installed_packs(&install);
        for code in [101, 201, 501, 1201, 1501, 1801] {
            let (model, set, held) = packs.filed(code, SET, "bt_swd_sld", "resident/idle");
            let path = path(model, set, held, "resident/idle");
            assert!(
                install.file::<Vec<u8>>(&path).is_ok(),
                "c{code:04} reads no drawn sword and shield pose from {path}"
            );
        }
    }

    #[test]
    fn a_blend_of_no_frames_still_runs_for_one() {
        let mut held = stance(&[]);
        held.groups.insert("from".to_owned(), 9);
        held.groups.insert("to".to_owned(), 8);
        held.blends.insert((9, 8), 0);
        assert_eq!(held.fade("from", "to"), 1.0 / FPS);
        assert_eq!(
            held.fade("from", "elsewhere"),
            1.0 / FPS,
            "a name with no row of its own still blends, out of group 0"
        );
    }
}
