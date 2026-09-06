//! The characters the game itself stands in the world, out of `ENpcBase` and `ENpcResident`.
//!
//! A base row carries the whole of what the creator would have picked, in the creator's own
//! numbering, plus a model quad per slot. The resident row of the same id is what it is called.
//!
//! A row that names no race is not a human at all: its `ModelChara` states a body of its own, the
//! same way a mount's does. See [`Stands`].

use std::collections::{BTreeMap, BTreeSet};

use anyhow::Result;
use ironworks::excel::Language;

use super::{Gear, HIGHLIGHT_COLOR, HIGHLIGHTS, LEFT_EYE_COLOR, LIPSTICK, ODD_EYES, Outfit};
use crate::backend::Backend;
use crate::excel::provider::{ExcelProvider, ExcelRow, ExcelSheet};

/// What `ENpcResident` calls a base row.
const SINGULAR: u32 = 0;

/// Where `ENpcBase` writes out the whole of what the creator would have picked. It is the game's
/// own customisation array laid down byte for byte, so a byte's place in it is also the menu the
/// creator drives it from, and everything below is that place rather than an offset into the row.
const CUSTOMIZE: u32 = 202;
const RACE: u32 = 0;
const GENDER: u32 = 1;
const BODY: u32 = 2;
const TRIBE: u32 = 4;

/// What `BODY` reads for a child. The elderly are stated apart from both and are an adult body
/// under an old face, so nothing here tells them from anyone else.
const CHILD: u32 = 4;

/// The customisations stated as the menu's own position, which counts from one where a menu counts
/// from nought, each with the mask picking it out of a byte two menus share.
const LISTED: [(u32, u8); 6] = [
    (14, 0xFF), // Eyebrows
    (16, 0x7F), // Eye shape
    (17, 0xFF), // Nose
    (18, 0xFF), // Jaw
    (19, 0x7F), // Mouth
    (22, 0xFF), // Tail or ear shape
];
/// The ones stated outright: a palette index, a slider's own place, a mask of features, or the
/// number the file tree files a set under. A face and a hairstyle are the last of those and not a
/// position: the artillerist's face is 216, which is `f0216` on disk and a face no menu offers.
const STATED: [(u32, u8); 13] = [
    (3, 0xFF),  // Height
    (5, 0xFF),  // Face
    (6, 0xFF),  // Hairstyle
    (8, 0xFF),  // Skin colour
    (9, 0xFF),  // Eye colour
    (10, 0xFF), // Hair colour
    (12, 0xFF), // Facial features
    (13, 0xFF), // Tattoo colour
    (20, 0xFF), // Lip colour
    (21, 0xFF), // Muscle tone
    (23, 0xFF), // Bust size
    (24, 0x7F), // Face paint
    (25, 0xFF), // Face paint colour
];
/// The bytes the creator ticks a box for rather than offering a menu of its own, and the one it
/// shares with the eye colour: an eye is odd exactly where the two are not the same colour.
const HIGHLIGHTS_AT: u32 = 7;
const HIGHLIGHT_COLOR_AT: u32 = 11;
const LEFT_EYE_AT: u32 = 15;
const EYE_AT: u32 = 9;
/// Iris size, which is the top bit of the byte the eye shape menu holds the rest of. The creator
/// numbers its menu fifteen even though the byte of that number is the left eye's colour.
const IRIS_AT: u32 = 16;
const IRIS: u32 = 15;
/// Lipstick, which is the top bit of the byte the mouth menu holds the rest of.
const LIPSTICK_AT: u32 = 19;

/// Where a row states what it wears: ten model quads in `Slot::ALL` order, then a `Stain` row id
/// per slot for each of the two channels a modern item can carry. The second channel trails all ten
/// of the first rather than sitting beside it.
struct Wearing {
    models: u32,
    dyes: u32,
    dyes2: u32,
}

/// `ENpcBase`'s own, which sit either side of the customise array.
const OWN: Wearing = Wearing {
    models: 148,
    dyes: CUSTOMIZE + 31,
    dyes2: CUSTOMIZE + 41,
};
/// `NpcEquip`'s, which states those ten slots and little else.
const EQUIP: Wearing = Wearing {
    models: 16,
    dyes: 64,
    dyes2: 74,
};

/// The body a row states apart from the creator's numbering, and the row it is dressed out of.
const MODEL_CHARA: u32 = 190;
const NPC_EQUIP: u32 = 192;

/// `ModelChara`'s numbered body, the kind of body it is, the set under it, and the variant.
const MODEL: u32 = 12;
const KIND: u32 = 16;
const BASE: u32 = 17;
const VARIANT: u32 = 18;

/// The kinds of body `ModelChara` states that are drawn from a directory of their own. Kind one is
/// a human, which the row that named it already says everything about; kind four is filed
/// elsewhere again and nothing here resolves it.
const DEMIHUMAN: u8 = 2;
const MONSTER: u8 = 3;

/// `BNpcBase`'s own links: the body, the customise array kept apart from it, and what it wears.
const BNPC_MODEL_CHARA: u32 = 14;
const BNPC_CUSTOMIZE: u32 = 16;
const BNPC_EQUIP: u32 = 18;

/// The `ENpcBase` a cutscene stands in for a party member it has no live one of. A viewer never has
/// one, so this is what every `PartyMember`, `PartyMemberAlt` and `Unknown82` participant draws:
/// `sub_141B26310` reaches for it wherever the roster it indexes answers nothing and the
/// participant does not force an id of its own.
pub const PARTY_STAND_IN: u32 = 1_034_882;
/// The `ENpcBase` a `StableChocobo` participant draws, which its own setup writes over whatever the
/// participant names rather than falling back to.
pub const STABLED_CHOCOBO: u32 = 1_006_001;

/// One of the game's own characters, as far as building it goes.
#[derive(Clone)]
pub struct Npc {
    pub name: String,
    pub race: u32,
    pub tribe: u32,
    pub female: bool,
    pub child: bool,
    /// What each of the creator's menus was left at, by the `Customize` it drives.
    pub choices: Vec<(u32, u32)>,
    pub outfit: Outfit,
    /// What each of `Slot::ALL` is dyed, one id per channel a modern item can carry. `ENpcBase` and
    /// `NpcEquip` state ten quads, not eleven: an NPC never wears facewear, so that slot is always
    /// `None` here rather than read from anything.
    pub stains: [[Option<u8>; 2]; 11],
}

/// What a character id builds.
#[derive(Clone)]
pub enum Stands {
    /// A body out of the creator's own numbering, dressed in what the row states.
    Human(Box<Npc>),
    /// A body of its own, drawn from every model under one directory at one variant: what
    /// `ModelChara` names for a monster or a demihuman.
    Beast { under: String, variant: u16 },
}

/// Which sheet an id is a row of. Only a `BattleNpc` participant reads the second.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Roll {
    Event,
    Battle,
}

/// What each of the creator's menus was left at, off a customise array laid down byte for byte
/// at `at`.
fn choices(row: &ExcelRow<'_>, at: u32) -> Vec<(u32, u32)> {
    let byte = |slot: u32| u32::from(row.read::<u8>(at + slot).unwrap_or(0));
    let (left, right) = (byte(LEFT_EYE_AT), byte(EYE_AT));
    LISTED
        .into_iter()
        .map(|(slot, mask)| (slot, (byte(slot) & u32::from(mask)).saturating_sub(1)))
        .chain(
            STATED
                .into_iter()
                .map(|(slot, mask)| (slot, byte(slot) & u32::from(mask))),
        )
        .chain([
            (IRIS, byte(IRIS_AT) >> 7),
            (LIPSTICK, byte(LIPSTICK_AT) >> 7),
            (HIGHLIGHTS, byte(HIGHLIGHTS_AT) >> 7),
            (HIGHLIGHT_COLOR, byte(HIGHLIGHT_COLOR_AT)),
            (ODD_EYES, u32::from(left != right)),
            (LEFT_EYE_COLOR, left),
        ])
        .collect()
}

/// What a row dresses each slot in, and what each is dyed.
fn worn(row: &ExcelRow<'_>, held: &Wearing) -> (Outfit, [[Option<u8>; 2]; 11]) {
    let mut outfit = [None; 11];
    let mut stains = [[None; 2]; 11];
    for slot in 0..10u32 {
        outfit[slot as usize] = row
            .read::<u32>(held.models + slot * 4)
            .ok()
            .filter(|quad| *quad != u32::MAX)
            .and_then(|quad| Gear::read(u64::from(quad)));
        let stain = |at: u32| row.read::<u8>(at + slot).ok().filter(|id| *id != 0);
        stains[slot as usize] = [stain(held.dyes), stain(held.dyes2)];
    }
    (outfit, stains)
}

/// The human a customise array at `at` states, where it states one at all.
fn human(row: &ExcelRow<'_>, at: u32, name: String) -> Option<Npc> {
    let byte = |slot: u32| u32::from(row.read::<u8>(at + slot).unwrap_or(0));
    let (race, tribe, gender) = (byte(RACE), byte(TRIBE), byte(GENDER));
    if race == 0 || tribe == 0 {
        return None;
    }
    Some(Npc {
        name,
        race,
        tribe,
        female: gender != 0,
        child: byte(BODY) == CHILD,
        choices: choices(row, at),
        outfit: [None; 11],
        stains: [[None; 2]; 11],
    })
}

/// Where the models a `ModelChara` row names sit, and the variant they are worn at.
fn beast(row: &ExcelRow<'_>) -> Option<(String, u16)> {
    let (model, kind, base, variant) = (
        row.read::<u16>(MODEL).ok()?,
        row.read::<u8>(KIND).ok()?,
        row.read::<u8>(BASE).ok()?,
        row.read::<u8>(VARIANT).ok()?,
    );
    let under = match kind {
        MONSTER => format!("chara/monster/m{model:04}/obj/body/b{base:04}/model/"),
        DEMIHUMAN => format!("chara/demihuman/d{model:04}/obj/equipment/e{base:04}/model/"),
        _ => return None,
    };
    Some((under, u16::from(variant)))
}

/// What each of the wanted rows builds. An id no sheet holds, or that states neither a race nor a
/// body this resolves, is left out.
///
/// A row's own model quads are what it wears; the `NpcEquip` it names dresses it only where it
/// states none of its own. That row is otherwise a scripted change of clothes rather than a
/// default: `NpcEquip` is a Lua method on an actor, bound at `sub_140C93460`.
pub async fn stand_in(
    backend: &Backend,
    language: Language,
    wanted: &BTreeSet<(Roll, u32)>,
) -> Result<BTreeMap<(Roll, u32), Stands>> {
    let excel = backend.excel();
    let bases = excel.get_sheet("ENpcBase", language).await?;
    let residents = excel.get_sheet("ENpcResident", language).await?;
    let battle = excel.get_sheet("BNpcBase", language).await?;
    let customizes = excel.get_sheet("BNpcCustomize", language).await?;
    let models = excel.get_sheet("ModelChara", language).await?;
    let equips = excel.get_sheet("NpcEquip", language).await?;

    let mut found = BTreeMap::new();
    for (roll, id) in wanted {
        // A battle character keeps its customise array and its equipment in rows of their own; an
        // event one writes both out in place.
        let (customize, dressed, chara) = match roll {
            Roll::Event => {
                let Ok(row) = bases.get_row(*id) else { continue };
                let name = residents
                    .get_row(*id)
                    .ok()
                    .and_then(|held| held.read_string(SINGULAR).ok().map(|name| name.to_string()))
                    .unwrap_or_default();
                let equip = row.read::<u16>(NPC_EQUIP).unwrap_or(0);
                let bare = (0..10).all(|slot| row.read::<u32>(OWN.models + slot * 4).is_ok_and(|quad| quad == 0));
                let dressed = match bare && equip != 0 {
                    true => equips
                        .get_row(u32::from(equip))
                        .ok()
                        .map(|held| worn(&held, &EQUIP)),
                    false => Some(worn(&row, &OWN)),
                };
                (
                    human(&row, CUSTOMIZE, name),
                    dressed,
                    row.read::<u16>(MODEL_CHARA).unwrap_or(0),
                )
            }
            Roll::Battle => {
                let Ok(row) = battle.get_row(*id) else { continue };
                let held = row.read::<u16>(BNPC_CUSTOMIZE).unwrap_or(0);
                let equip = row.read::<u16>(BNPC_EQUIP).unwrap_or(0);
                (
                    customizes
                        .get_row(u32::from(held))
                        .ok()
                        .and_then(|held| human(&held, 0, String::new())),
                    equips
                        .get_row(u32::from(equip))
                        .ok()
                        .map(|held| worn(&held, &EQUIP)),
                    row.read::<u16>(BNPC_MODEL_CHARA).unwrap_or(0),
                )
            }
        };
        let stands = match customize {
            Some(mut npc) => {
                if let Some((outfit, stains)) = dressed {
                    npc.outfit = outfit;
                    npc.stains = stains;
                }
                Stands::Human(Box::new(npc))
            }
            None => {
                let Some((under, variant)) = models
                    .get_row(u32::from(chara))
                    .ok()
                    .filter(|_| chara != 0)
                    .and_then(|held| beast(&held))
                else {
                    continue;
                };
                Stands::Beast { under, variant }
            }
        };
        found.insert((*roll, *id), stands);
    }
    Ok(found)
}

/// Every named character the game builds out of a human body. The unnamed ones are left out: a
/// list of sixty thousand rows is not something to search by name.
pub async fn read(backend: &Backend, language: Language) -> Result<Vec<Npc>> {
    let excel = backend.excel();
    let bases = excel.get_sheet("ENpcBase", language).await?;
    let residents = excel.get_sheet("ENpcResident", language).await?;

    let mut found = Vec::new();
    for id in bases.get_row_ids() {
        let Ok(row) = bases.get_row(id) else {
            continue;
        };
        let name = residents
            .get_row(id)
            .ok()
            .and_then(|held| held.read_string(SINGULAR).ok().map(|name| name.to_string()))
            .unwrap_or_default();
        if name.is_empty() {
            continue;
        }
        let Some(mut npc) = human(&row, CUSTOMIZE, name) else {
            continue;
        };
        (npc.outfit, npc.stains) = worn(&row, &OWN);
        found.push(npc);
    }
    found.sort_by(|left, right| left.name.cmp(&right.name));
    log::info!("character: {} named characters to stand in", found.len());
    Ok(found)
}
