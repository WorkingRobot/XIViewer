//! The emotes the game names, out of `Emote` and `ActionTimeline`.
//!
//! An emote names seven timelines by slot: the pose it stands in, the motion that plays it in, and
//! then the same emote sat on the ground, sat on a chair, mounted and asleep. A timeline's key is
//! the tail of a pack path under the body's own animation directory, which is what turns thousands
//! of numbered files into a named, iconned list.
//!
//! The game does not mask an emote down to the upper body: it plays a different motion. The three
//! seated slots name a `u_` pack of their own holding one `cbep_u_` motion, which is a partial
//! naming only the bones it moves, so whatever the body is held in shows through the rest. 232 of
//! the 300 emotes the sheet names state one for at least one of those slots, and every one of them
//! is filed under `Stance` 1 where the standing motion is filed under 0.

use std::collections::HashSet;

use anyhow::Result;
use ironworks::excel::Language;

use crate::backend::Backend;
use crate::excel::provider::{ExcelProvider, ExcelRow, ExcelSheet};

/// `Emote`'s name, icon and the two timelines a standing character plays, and `ActionTimeline`'s
/// key, as byte offsets.
const NAME: u32 = 0;
const ICON: u32 = 4;
const STANDING: u32 = 16;
const START: u32 = 18;
/// The slots a body whose lower half is already committed reads: sat on the ground, sat on a
/// chair, and the partial a rider plays.
///
/// `Priority` is what says which is which, not the order. `emote/jmn` sits a body on the ground at
/// priority 8 and every `j_` variant is filed at 8 beside it; `emote/sit` sits it in a chair at 9
/// and every `s_` variant is filed at 9. So `j_` is the ground - `jimen` - and `s_` is the chair.
const GROUND: u32 = 20;
const CHAIR: u32 = 22;
const MOUNTED: u32 = 24;
const KEY: u32 = 0;

/// The motion the emote that sits a body down starts with. Matching the key rather than the
/// emote's own name keeps this off whichever language the sheet was read in.
const CHAIR_START: &str = "_chair_start";
const GROUND_START: &str = "_ground_start";

/// How an alternate names the motion that plays it in and the one it then holds.
const LEADS_IN: &str = "_start";
const HOLDS: &str = "_loop";

/// What a body's lower half is committed to, which decides both the pose it holds and which of an
/// emote's own slots it plays. A rider is not one of these: a mount states its own seat.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub enum Posture {
    #[default]
    Standing,
    Chair,
    Ground,
}

impl Posture {
    pub const ALL: [Self; 3] = [Self::Standing, Self::Chair, Self::Ground];

    pub fn name(self) -> &'static str {
        match self {
            Self::Standing => "Standing",
            Self::Chair => "Sitting",
            Self::Ground => "On the ground",
        }
    }
}

/// One emote the creator's own list would show.
pub struct Emote {
    pub name: String,
    pub icon: u32,
    /// The pose it holds and the motion that plays it in, as the keys they are filed under. An
    /// emote that only makes a face states one and no other.
    standing: Option<String>,
    start: Option<String>,
    /// The partial the same emote is played as while mounted, where it names one.
    mounted: Option<String>,
    /// The whole-body variants for a seat, which replace the standing motion rather than laying
    /// over it: the body is already sat down and the emote states how it looks that way.
    chair: Option<String>,
    ground: Option<String>,
}

impl Emote {
    /// The keys the packs a body plays this from are filed under: the motion it starts with, and
    /// the pose it settles into once that has played through. An emote that holds a pose forever
    /// states the motion that plays it in apart from the pose itself; one that only moves states
    /// the motion alone.
    pub fn keys(&self) -> (Option<&str>, Option<&str>) {
        match (&self.start, &self.standing) {
            (Some(start), standing) => (Some(start), standing.as_deref()),
            (None, standing) => (standing.as_deref(), None),
        }
    }

    /// The key of the partial this emote is played as while mounted, which is laid over the pose
    /// the mount holds the rider in rather than replacing it.
    pub fn mounted(&self) -> Option<&str> {
        self.mounted.as_deref()
    }

    /// The key this emote plays from in a seat, where it names one. Nothing is laid over here: a
    /// seated variant is a whole motion of a body that is already sat down.
    pub fn seated(&self, at: Posture) -> Option<&str> {
        match at {
            Posture::Standing => None,
            Posture::Chair => self.chair.as_deref(),
            Posture::Ground => self.ground.as_deref(),
        }
    }

    /// The expression this emote is, for the ones that only make a face. Those are filed under the
    /// face skeleton a character wears rather than under its body, and the last segment of the key
    /// is what names one there.
    pub fn expression(&self) -> Option<&str> {
        let key = self.standing.as_deref()?.strip_prefix("facial/")?;
        key.rsplit('/').next()
    }
}

/// One pose a posture rests in: the motion that leads into it, where the sheet names one, and the
/// pose it settles into. A body on its feet settles into the idle its own weapons state names,
/// which is no pack of its own, so both are None for that one.
#[derive(Clone, Default, PartialEq, Eq, Debug)]
pub struct Pose {
    pub start: Option<String>,
    pub settle: Option<String>,
}

/// The poses `/cpose` steps a posture through: the one it rests in, then the alternates the sheet
/// numbers.
#[derive(Default, Clone)]
pub struct Poses {
    standing: Vec<Pose>,
    chair: Vec<Pose>,
    ground: Vec<Pose>,
}

impl Poses {
    pub fn of(&self, at: Posture) -> &[Pose] {
        match at {
            Posture::Standing => &self.standing,
            Posture::Chair => &self.chair,
            Posture::Ground => &self.ground,
        }
    }
}

/// The alternates one posture cycles, in the order they number themselves. An alternate numbers
/// itself and stops there: `pose01_center_loop` carries more than a number and is something else.
fn cycled(keys: &HashSet<String>, prefix: &str) -> Vec<Pose> {
    let numbered = |key: &String, tail: &str| {
        key.strip_prefix(prefix)
            .and_then(|rest| rest.strip_suffix(tail))
            .is_some_and(|held| held.len() == 2 && held.bytes().all(|byte| byte.is_ascii_digit()))
    };
    let mut found: Vec<String> = keys
        .iter()
        .filter(|key| numbered(key, HOLDS))
        .cloned()
        .collect();
    found.sort();
    found
        .into_iter()
        .map(|settle| {
            let start = format!("{}{LEADS_IN}", settle.strip_suffix(HOLDS).unwrap_or(&settle));
            Pose {
                start: keys.contains(&start).then_some(start),
                settle: Some(settle),
            }
        })
        .collect()
}

/// Every emote the game both names and animates, in name order, and the poses each seat cycles.
pub async fn read(backend: &Backend, language: Language) -> Result<(Vec<Emote>, Poses)> {
    let excel = backend.excel();
    let emotes = excel.get_sheet("Emote", language).await?;
    let timelines = excel.get_sheet("ActionTimeline", language).await?;

    // An emote that can be aimed at somebody ships twice, as two rows of each sheet, and the game
    // picks between them on whether a target exists. Nothing here ever has one, so the pair
    // collapses to the untargeted half.
    let mut keys = HashSet::new();
    for id in timelines.get_row_ids() {
        if let Some(key) = key(&timelines, id) {
            keys.insert(key);
        }
    }

    let mut found = Vec::new();
    for id in emotes.get_row_ids() {
        let Ok(row) = emotes.get_row(id) else {
            continue;
        };
        let (Ok(name), Ok(icon)) = (row.read_string(NAME), row.read::<u32>(ICON)) else {
            continue;
        };
        let name = name.to_string();
        if name.is_empty() || icon == 0 {
            continue;
        }
        let slot = |at| {
            let timeline = row.read::<u16>(at).ok().filter(|timeline| *timeline > 0)?;
            Some(untargeted(&keys, key(&timelines, u32::from(timeline))?))
        };
        let (standing, start) = (slot(STANDING), slot(START));
        if standing.is_none() && start.is_none() {
            continue;
        }
        found.push(Emote {
            name,
            icon,
            standing,
            start,
            mounted: slot(MOUNTED),
            chair: slot(CHAIR),
            ground: slot(GROUND),
        });
    }
    // The pose a seat settles into is the one its own sitting emote holds; the motion that emote
    // starts with is what says which seat it is, in whatever language the sheet was read in.
    let mut poses = Poses::default();
    // A body on its feet rests in whatever its weapons state names, which no pack here holds.
    poses.standing.push(Pose::default());
    for held in &found {
        let (Some(start), Some(settle)) = (held.start.clone(), held.standing.clone()) else {
            continue;
        };
        let seat = match (start.ends_with(CHAIR_START), start.ends_with(GROUND_START)) {
            (true, _) => &mut poses.chair,
            (_, true) => &mut poses.ground,
            _ => continue,
        };
        if seat.is_empty() {
            seat.push(Pose {
                start: Some(start),
                settle: Some(settle),
            });
        }
    }
    poses.standing.extend(cycled(&keys, "emote/pose"));
    poses.chair.extend(cycled(&keys, "emote/s_pose"));
    poses.ground.extend(cycled(&keys, "emote/j_pose"));
    log::info!(
        "character: {} poses standing, {} sitting, {} on the ground",
        poses.standing.len(),
        poses.chair.len(),
        poses.ground.len()
    );

    found.sort_by(|left, right| left.name.cmp(&right.name));
    found.dedup_by(|left, right| {
        left.name == right.name
            && left.standing == right.standing
            && left.start == right.start
            && left.mounted == right.mounted
            && left.chair == right.chair
            && left.ground == right.ground
    });
    log::info!("character: {} emotes to play", found.len());
    Ok((found, poses))
}

/// The half of a targeted emote the game plays with nothing aimed at: it names its own motions, so
/// the heart a `Dote` throws stays unthrown rather than flying at a target that is not there.
fn untargeted(keys: &HashSet<String>, key: String) -> String {
    let alone = format!("{key}_no_target");
    match keys.contains(&alone) {
        true => alone,
        false => key,
    }
}

fn key(timelines: &impl ExcelSheet, id: u32) -> Option<String> {
    let row: ExcelRow<'_> = timelines.get_row(id).ok()?;
    let key = row.read_string(KEY).ok()?.to_string();
    (!key.is_empty()).then_some(key)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{Emote, Posture, cycled, untargeted};

    /// `Dote` and `All Saints' Charm` each ship a second row naming a motion of their own, and that
    /// is the one to play; `Chuckle` ships no such thing and keeps the key it states.
    #[test]
    fn a_targeted_emote_falls_to_the_half_that_needs_no_target() {
        let keys: HashSet<String> = ["emote_sp/sp04", "emote_sp/sp04_no_target", "emote/laugh"]
            .into_iter()
            .map(ToOwned::to_owned)
            .collect();
        assert_eq!(
            untargeted(&keys, "emote_sp/sp04".to_owned()),
            "emote_sp/sp04_no_target"
        );
        assert_eq!(untargeted(&keys, "emote/laugh".to_owned()), "emote/laugh");
        // The untargeted half names no half of its own, so it is a fixed point.
        assert_eq!(
            untargeted(&keys, "emote_sp/sp04_no_target".to_owned()),
            "emote_sp/sp04_no_target"
        );
    }

    fn emote(standing: Option<&str>, start: Option<&str>) -> Emote {
        Emote {
            name: String::new(),
            icon: 0,
            standing: standing.map(ToOwned::to_owned),
            start: start.map(ToOwned::to_owned),
            mounted: None,
            chair: None,
            ground: None,
        }
    }

    /// A seat reads the slot it is filed under and nothing else: standing reads none of them, so a
    /// body on its feet plays the whole-body motion rather than a variant.
    #[test]
    fn a_seat_reads_only_its_own_slot() {
        let mut clap = emote(Some("emote/clap"), None);
        clap.chair = Some("emote/s_clap".to_owned());
        clap.ground = Some("emote/j_clap".to_owned());
        assert_eq!(clap.seated(Posture::Standing), None);
        assert_eq!(clap.seated(Posture::Chair), Some("emote/s_clap"));
        assert_eq!(clap.seated(Posture::Ground), Some("emote/j_clap"));

        // 120-odd emotes name one seat and not the other, and an unnamed one is nothing to play.
        let sit_ups = emote(Some("emote/loop_emot09_loop"), None);
        assert_eq!(sit_ups.seated(Posture::Chair), None);
    }

    /// The alternates a seat cycles come out of the sheet's own keys, in the order they number
    /// themselves, and nothing else in the sheet is one of them.
    #[test]
    fn a_seat_cycles_the_keys_that_number_themselves() {
        let keys: HashSet<String> = [
            "emote/j_pose03_loop",
            "emote/j_pose01_loop",
            "emote/j_pose02_loop",
            "emote/j_pose02_start",
            "emote/s_pose01_loop",
            "emote/clap",
        ]
        .into_iter()
        .map(ToOwned::to_owned)
        .collect();
        let held = cycled(&keys, "emote/j_pose");
        let settles: Vec<&str> = held.iter().filter_map(|p| p.settle.as_deref()).collect();
        assert_eq!(
            settles,
            [
                "emote/j_pose01_loop",
                "emote/j_pose02_loop",
                "emote/j_pose03_loop"
            ]
        );
        // Only the second states a motion to lead in with, so the others go straight to the pose.
        let starts: Vec<Option<&str>> = held.iter().map(|p| p.start.as_deref()).collect();
        assert_eq!(starts, [None, Some("emote/j_pose02_start"), None]);
        assert_eq!(cycled(&keys, "emote/s_pose").len(), 1);
    }

    /// `pose01_center_loop` numbers itself and then carries more, so it is not one of the poses
    /// `/cpose` steps through.
    #[test]
    fn an_alternate_numbers_itself_and_stops() {
        let keys: HashSet<String> = [
            "emote/pose01_loop",
            "emote/pose01_center_loop",
            "emote/pose02_loop",
        ]
        .into_iter()
        .map(ToOwned::to_owned)
        .collect();
        let held = cycled(&keys, "emote/pose");
        let settles: Vec<&str> = held.iter().filter_map(|p| p.settle.as_deref()).collect();
        assert_eq!(settles, ["emote/pose01_loop", "emote/pose02_loop"]);
    }

    /// A pose held forever states the motion that plays it in apart from the pose itself, and the
    /// motion comes first.
    #[test]
    fn an_emote_starts_before_it_settles() {
        let sit = emote(Some("emote/sit"), Some("event_base/event_base_chair_start"));
        assert_eq!(
            sit.keys(),
            (Some("event_base/event_base_chair_start"), Some("emote/sit"))
        );
        assert_eq!(sit.expression(), None);

        let wave = emote(Some("emote/goodbye_st"), None);
        assert_eq!(wave.keys(), (Some("emote/goodbye_st"), None));
    }

    /// An emote that only makes a face is filed under the face a character wears, not under its
    /// body, and the last segment of the key is what names it there.
    #[test]
    fn an_emote_that_only_makes_a_face_names_an_expression() {
        assert_eq!(emote(Some("facial/pose/smile"), None).expression(), Some("smile"));
        assert_eq!(emote(Some("facial/pose/base"), None).expression(), Some("base"));
        assert_eq!(emote(Some("emote/bow"), None).expression(), None);
    }

    /// What the mounted slot really names, off the real install: Blow Bubbles states `u_sp63` for
    /// every seated slot, and that pack holds one `cbep_u_sp63` moving a fraction of the bones the
    /// whole-body `cbem_sp63` does. That is the whole of how the game restricts an emote to the
    /// upper half - a motion of its own, not a mask over the standing one.
    #[test]
    #[ignore = "reads the real local FFXIV install"]
    fn the_mounted_slot_names_a_partial_that_moves_fewer_bones() {
        use ironworks::file::File as _;
        use ironworks::file::pap::AnimationPack;
        use ironworks::sqpack::{Install, SqPack};

        let install = ironworks::Ironworks::new()
            .with_resource(SqPack::new(Install::at_sqpack(
                "/home/asriel/.xlcore/ffxiv/game/sqpack",
            )));
        let moved = |key: &str| -> (String, usize) {
            let path =
                format!("chara/human/c0101/animation/a0001/bt_common/{key}.pap");
            let bytes: Vec<u8> = install.file(&path).expect(&path);
            let pack = AnimationPack::read(std::io::Cursor::new(bytes)).expect("a readable pack");
            let bindings = pack.parse_animations().expect("its motions");
            (
                pack.animations()[0].name().to_owned(),
                bindings[0].bones().len(),
            )
        };

        let (whole, all) = moved("emote_sp/sp63");
        let (partial, some) = moved("emote_sp/u_sp63");
        assert_eq!(whole, "cbem_sp63");
        assert_eq!(partial, "cbep_u_sp63");
        assert!(some < all, "{partial} moves {some} bones, {whole} moves {all}");
    }
}
