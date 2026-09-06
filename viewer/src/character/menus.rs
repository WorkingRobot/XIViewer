//! What the character creator itself offers, out of `CharaMakeType`, `CharaMakeCustomize` and the
//! `Race` and `Tribe` sheets.
//!
//! A `CharaMakeType` row is one race, clan and gender, holding a menu per customisation that names
//! which it drives in `Customize`. The two menus read here state their choices differently: the face
//! menu holds icon ids outright, and the hair menu holds `CharaMakeCustomize` rows, which carry the
//! set number the file tree uses alongside the icon.
//!
//! Every offset is measured from the file rather than taken from EXDSchema, which names the fields
//! but not where they sit: a menu is 452 bytes, of which `SubMenuParam` is 12 to 436.

use std::collections::BTreeMap;
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};

use anyhow::Result;
use ironworks::excel::Language;
#[cfg(target_arch = "wasm32")]
use web_time::{Duration, Instant};

use super::{Gear, Outfit, Slot};
use crate::backend::Backend;
use crate::excel::provider::{ExcelProvider, ExcelRow, ExcelSheet};
use crate::utils::yield_to_ui;

/// How long the walk over every item holds the thread before letting the interface draw.
const MAX_FRAME_TIME: Duration = Duration::from_millis(250);

/// Menus a row holds, and the bytes one menu is.
const MENUS: u32 = 28;
const STRIDE: u32 = 452;
/// A menu's name, what it drives, where it opens, how it is picked from and how many it offers, as
/// byte offsets into one menu, then the first of its choices.
const LOBBY: u32 = 0;
const CUSTOMIZE: u32 = 8;
const INIT: u32 = 436;
const KIND: u32 = 437;
const COUNT: u32 = 438;
const PARAMS: u32 = 12;
const PARAM_COUNT: u32 = 106;
/// What `Lobby` calls a menu.
const LOBBY_TEXT: u32 = 0;
/// Where a row names the icon each facial feature is offered under, seven to a face, in the order
/// the face menu offers the faces.
const FEATURES: u32 = 12668;
const PER_FACE: u32 = 7;
/// Where a row names the race, clan and gender it is for.
const RACE: u32 = 13064;
const TRIBE: u32 = 13068;
const GENDER: u32 = 13072;

/// Which customisation a menu drives, as the creator's own numbering has it.
const FACE: i32 = 5;
const HAIR: i32 = 6;

/// `Masculine` and `Feminine`, which is how both sheets name a race and a clan.
const MASCULINE: u32 = 0;
const FEMININE: u32 = 4;

/// Where `Race` names the item worn in each of [`Slot::RACIAL`], masculine then feminine.
const RSE: u32 = 8;
/// `Item`'s model quad, name, icon, equip slot category and race restriction.
const MODEL: u32 = 24;
const ITEM_NAME: u32 = 12;
const ITEM_ICON: u32 = 136;
const SLOTS: u32 = 154;
const RESTRICTION: u32 = 80;
/// Where `EquipSlotCategory` states each of [`Slot::ALL`] but facewear, which the sheet has no
/// column for at all. It runs over every slot the game dresses an item in, of which the five a
/// body is dressed in are not adjacent, and it names the left ring ahead of the right one where
/// the models are named the other way round.
const FILLS: [u32; 10] = [2, 3, 4, 6, 7, 8, 9, 10, 11, 12];
/// `Glasses`' name, model quad and icon. The sheet states no race or gender restriction, so
/// everything it offers suits every one the game will let wear it at all.
const GLASSES_NAME: u32 = 12;
const GLASSES_MODEL: u32 = 24;
const GLASSES_ICON: u32 = 28;
/// `EquipRaceCategory`'s eight races, then the two genders packed into one byte after them.
const WORN_BY: u32 = 0;
const GENDERS: u32 = 8;
/// `CharaMakeClassEquip`'s class, after the seven quads it dresses that class in.
const CLASS_JOB: u32 = 56;
/// `ClassJob`'s name as the creator writes it, rather than the lowercase one it is filed under.
const JOB_NAME: u32 = 16;

/// How a menu is picked from, as the row states it beside the count.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Choices the creator names rather than draws: a jaw or a nose is offered as `Type 1` and an
    /// iris as `Large` or `Small`, and each names the `Lobby` row it is called by.
    List,
    /// Choices the creator draws as icons, either named outright or held by a `CharaMakeCustomize`
    /// row alongside the number the file tree uses.
    Icons,
    /// A palette out of `human.cmp`.
    Color,
    /// A palette that colours two things at once, which is the eyes and nothing else.
    DoubleColor,
    /// Several choices at once, each a facial feature the face model draws as its own part.
    Checks,
    /// A run the creator draws as a bar.
    Slider,
}

impl Kind {
    fn read(kind: u8) -> Option<Self> {
        match kind {
            0 => Some(Self::List),
            1 => Some(Self::Icons),
            2 => Some(Self::Color),
            3 => Some(Self::DoubleColor),
            4 => Some(Self::Checks),
            5 => Some(Self::Slider),
            _ => None,
        }
    }
}

/// One of the customisations the creator offers, as the row lays it out.
#[derive(Clone)]
pub struct Menu {
    pub name: String,
    /// Which of `Customize` it drives, which is the only stable name a menu has.
    pub customize: u32,
    pub kind: Kind,
    pub count: u32,
    /// What a bar runs between, which the row states beside the two labels bounding it and which
    /// is not the count: every slider counts a hundred and runs nought to a hundred.
    pub range: [u32; 2],
    /// Where the creator opens the menu, which is not its first choice: a Midlander man starts at
    /// the middle of the height bar and at the fifty-fifth hair colour, which is a brown.
    pub init: u32,
    /// What each choice is, where the menu draws them: a `CharaMakeCustomize` row, or an icon
    /// outright where the number is too large to be one.
    pub params: Vec<i32>,
    /// What each choice is called, where the menu names them instead. The same numbers would read
    /// as `CharaMakeCustomize` rows, and four hairstyles are what a jaw menu draws that way.
    pub labels: Vec<String>,
}

/// One race, clan and gender the creator offers, and what it offers for them.
#[derive(Clone)]
pub struct Body {
    pub race: u32,
    pub tribe: u32,
    pub female: bool,
    /// The icon each face is offered under, by face number.
    pub faces: BTreeMap<u16, u32>,
    /// The icon each hair set is offered under, by the set number the file tree uses.
    pub hairs: BTreeMap<u16, u32>,
    /// The icon each of the seven facial features is offered under, by the place the face it
    /// belongs to sits in the face menu. Hrothgar faces number 5 to 8, so the place is not the id.
    pub features: Vec<[u32; 7]>,
    /// Everything the creator offers this body, in the order it offers it.
    pub menus: Vec<Menu>,
}

/// One of the classes the creator starts a character in, and what it dresses them in.
pub struct Job {
    pub name: String,
    pub outfit: Outfit,
}

/// One piece of equipment the game names, and what wearing it does to a character's slots.
#[derive(Clone)]
pub struct Piece {
    pub name: String,
    pub gear: Gear,
    pub icon: u32,
    /// The slots it covers itself, so nothing else is drawn in them.
    pub hides: [bool; 10],
    /// The races and genders it is made for, one bit each. A character outside them still has a
    /// model to wear: the game bars the pairing, the files do not.
    races: u8,
    genders: u8,
}

impl Piece {
    /// Whether the game would let this race and gender wear it.
    pub fn suits(&self, race: u32, female: bool) -> bool {
        self.races & 1 << (race.clamp(1, 8) - 1) != 0 && self.genders & 1 << u8::from(female) != 0
    }
}

#[derive(Default)]
pub struct Creator {
    pub bodies: Vec<Body>,
    /// What each race and clan is called, masculine then feminine.
    pub races: BTreeMap<u32, (String, String)>,
    pub tribes: BTreeMap<u32, (String, String)>,
    /// What each choice a menu names is: the number the file tree uses for it, and the icon the
    /// creator offers it under, by `CharaMakeCustomize` row.
    pub offered: BTreeMap<u32, (u16, u32)>,
    /// The clothing a race wears when it wears nothing else, by race and gender.
    pub attire: BTreeMap<(u32, bool), Outfit>,
    pub jobs: Vec<Job>,
    /// Everything there is to wear, by the slot it is worn in.
    pub pieces: [Vec<Piece>; 11],
}

impl Creator {
    /// What to call a race or clan, in the gender that is being built.
    pub fn named(named: &BTreeMap<u32, (String, String)>, id: u32, female: bool) -> String {
        match named.get(&id) {
            Some((male, girl)) => match female {
                true => girl.clone(),
                false => male.clone(),
            },
            None => id.to_string(),
        }
    }

    pub fn body(&self, tribe: u32, female: bool) -> Option<&Body> {
        self.bodies
            .iter()
            .find(|body| body.tribe == tribe && body.female == female)
    }
}

pub async fn read(backend: &Backend, language: Language) -> Result<Creator> {
    let excel = backend.excel();
    let types = excel.get_sheet("CharaMakeType", language).await?;
    let customize = excel.get_sheet("CharaMakeCustomize", language).await?;
    let lobby = excel.get_sheet("Lobby", language).await?;

    // Set number and icon of every choice the creator offers, whatever menu holds it.
    let mut offered = BTreeMap::new();
    for id in customize.get_row_ids() {
        let Ok(row) = customize.get_row(id) else {
            continue;
        };
        if let (Ok(icon), Ok(feature)) = (row.read::<u32>(0), row.read::<u8>(14)) {
            offered.insert(id, (u16::from(feature), icon));
        }
    }

    let mut bodies = Vec::new();
    for id in types.get_row_ids() {
        let Ok(row) = types.get_row(id) else {
            continue;
        };
        let (Ok(race), Ok(tribe), Ok(gender)) = (
            row.read::<i32>(RACE),
            row.read::<i32>(TRIBE),
            row.read::<i8>(GENDER),
        ) else {
            continue;
        };
        if race <= 0 || tribe <= 0 {
            continue;
        }
        let menus = menus(&row, &lobby);
        let faces = menus
            .iter()
            .find(|menu| menu.customize == FACE as u32)
            .map_or(0, |menu| menu.count);
        bodies.push(Body {
            race: race as u32,
            tribe: tribe as u32,
            female: gender != 0,
            faces: params(&row, FACE)
                .iter()
                .filter_map(|icon| Some((face(*icon)?, *icon as u32)))
                .collect(),
            hairs: params(&row, HAIR)
                .iter()
                .filter_map(|param| offered.get(&(*param as u32)).copied())
                .collect(),
            features: (0..faces)
                .map(|face| {
                    std::array::from_fn(|feature| {
                        let at = FEATURES + (face * PER_FACE + feature as u32) * 4;
                        row.read::<i32>(at).unwrap_or(0).max(0) as u32
                    })
                })
                .collect(),
            menus,
        });
    }

    Ok(Creator {
        bodies,
        offered,
        races: names(backend, "Race", language).await?,
        tribes: names(backend, "Tribe", language).await?,
        attire: attire(backend, language).await?,
        jobs: jobs(backend, language).await?,
        pieces: Default::default(),
    })
}

/// Everything the game names that a body can be dressed in, by the slot it goes in. A ring is the
/// one thing that goes in either of two, so a category fills a list of slots rather than one.
///
/// Read on its own, since walking every item there is takes long enough that waiting for it would
/// hold up the character it is going to dress.
pub async fn pieces(backend: &Backend, language: Language) -> Result<[Vec<Piece>; 11]> {
    let excel = backend.excel();
    let items = excel.get_sheet("Item", language).await?;
    let categories = excel.get_sheet("EquipSlotCategory", language).await?;
    let restrictions = excel.get_sheet("EquipRaceCategory", language).await?;

    let mut worn = BTreeMap::new();
    for id in categories.get_row_ids() {
        let Ok(row) = categories.get_row(id) else {
            continue;
        };
        let mut fills = Vec::new();
        let mut hides = [false; 10];
        for (slot, at) in FILLS.into_iter().enumerate() {
            match row.read::<i8>(at) {
                Ok(1) => fills.push(slot),
                Ok(-1) => hides[slot] = true,
                _ => {}
            }
        }
        if !fills.is_empty() {
            worn.insert(id, (fills, hides));
        }
    }

    let mut allowed = BTreeMap::new();
    for id in restrictions.get_row_ids() {
        let Ok(row) = restrictions.get_row(id) else {
            continue;
        };
        let races = (0..8).fold(0u8, |races, race| {
            let worn = row.read_bool(WORN_BY + race).unwrap_or(false);
            races | u8::from(worn) << race
        });
        let genders = (0..2).fold(0u8, |genders, gender| {
            let worn = row.read_packed_bool(GENDERS, gender).unwrap_or(false);
            genders | u8::from(worn) << gender
        });
        allowed.insert(id, (races, genders));
    }

    let mut found: [Vec<Piece>; 11] = Default::default();
    let mut drawn = Instant::now();
    for id in items.get_row_ids() {
        if drawn.elapsed() >= MAX_FRAME_TIME {
            yield_to_ui().await;
            drawn = Instant::now();
        }
        let Ok(row) = items.get_row(id) else {
            continue;
        };
        let Some((fills, hides)) = row
            .read::<u8>(SLOTS)
            .ok()
            .and_then(|category| worn.get(&u32::from(category)))
        else {
            continue;
        };
        let Some(gear) = row.read::<u64>(MODEL).ok().and_then(Gear::read) else {
            continue;
        };
        let Ok(name) = row.read_string(ITEM_NAME) else {
            continue;
        };
        let (races, genders) = row
            .read::<u8>(RESTRICTION)
            .ok()
            .and_then(|worn| allowed.get(&u32::from(worn)))
            .copied()
            .unwrap_or_default();
        let piece = Piece {
            name: name.to_string(),
            gear,
            icon: row.read::<u16>(ITEM_ICON).unwrap_or(0).into(),
            hides: *hides,
            races,
            genders,
        };
        for slot in fills {
            found[*slot].push(piece.clone());
        }
    }

    // A slot of its own: `EquipSlotCategory` has no column for facewear at all, and the game
    // resolves its model through the plain equipment convention rather than `Item`'s.
    let glasses = excel.get_sheet("Glasses", language).await?;
    for id in glasses.get_row_ids() {
        let Ok(row) = glasses.get_row(id) else {
            continue;
        };
        let Some(gear) = row.read::<u32>(GLASSES_MODEL).ok().map(u64::from).and_then(Gear::read)
        else {
            continue;
        };
        let Ok(name) = row.read_string(GLASSES_NAME) else {
            continue;
        };
        let name = name.to_string();
        if name.is_empty() {
            continue;
        }
        found[Slot::Facewear as usize].push(Piece {
            name,
            gear,
            icon: row.read::<u32>(GLASSES_ICON).unwrap_or(0),
            hides: [false; 10],
            races: u8::MAX,
            genders: u8::MAX,
        });
    }

    for pieces in &mut found {
        pieces.sort_by(|left, right| left.name.cmp(&right.name));
    }
    log::info!(
        "character: {} pieces to wear",
        found.iter().map(Vec::len).sum::<usize>()
    );
    Ok(found)
}

/// What each race stands in when it is wearing nothing else. `Race` names an item per slot and
/// gender, and the item's model quad is the set and the variant it is worn at.
async fn attire(backend: &Backend, language: Language) -> Result<BTreeMap<(u32, bool), Outfit>> {
    let excel = backend.excel();
    let races = excel.get_sheet("Race", language).await?;
    let items = excel.get_sheet("Item", language).await?;
    let mut dressed = BTreeMap::new();
    for id in races.get_row_ids() {
        let Ok(race) = races.get_row(id) else {
            continue;
        };
        for female in [false, true] {
            let mut outfit = Outfit::default();
            for (at, slot) in Slot::RACIAL.into_iter().enumerate() {
                let at = RSE + at as u32 * 8 + u32::from(female) * 4;
                outfit[slot as usize] = race
                    .read::<i32>(at)
                    .ok()
                    .filter(|item| *item > 0)
                    .and_then(|item| items.get_row(item as u32).ok())
                    .and_then(|item| item.read::<u64>(MODEL).ok())
                    .and_then(Gear::read);
            }
            dressed.insert((id, female), outfit);
        }
    }
    Ok(dressed)
}

/// The classes a character can be started as, and the gear each of them is started in. The sheet
/// states its five armour quads in [`Slot::ALL`]'s own order, ahead of the two it holds.
async fn jobs(backend: &Backend, language: Language) -> Result<Vec<Job>> {
    let excel = backend.excel();
    let equipped = excel.get_sheet("CharaMakeClassEquip", language).await?;
    let classes = excel.get_sheet("ClassJob", language).await?;
    let mut found = Vec::new();
    for id in equipped.get_row_ids() {
        let Ok(row) = equipped.get_row(id) else {
            continue;
        };
        let Ok(class) = row.read::<i32>(CLASS_JOB) else {
            continue;
        };
        let mut outfit = Outfit::default();
        for slot in Slot::GEAR {
            outfit[slot as usize] = row.read::<u64>(slot as u32 * 8).ok().and_then(Gear::read);
        }
        found.push(Job {
            name: classes
                .get_row(class as u32)
                .ok()
                .and_then(|row| row.read_string(JOB_NAME).ok().map(|name| name.to_string()))
                .unwrap_or_else(|| class.to_string()),
            outfit,
        });
    }
    found.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(found)
}

/// A sheet's masculine and feminine names, by row.
async fn names(
    backend: &Backend,
    sheet: &str,
    language: Language,
) -> Result<BTreeMap<u32, (String, String)>> {
    let sheet = backend.excel().get_sheet(sheet, language).await?;
    let mut named = BTreeMap::new();
    for id in sheet.get_row_ids() {
        let Ok(row) = sheet.get_row(id) else {
            continue;
        };
        if let (Ok(male), Ok(female)) = (row.read_string(MASCULINE), row.read_string(FEMININE)) {
            let (male, female) = (male.to_string(), female.to_string());
            if !male.is_empty() {
                named.insert(id, (male, female));
            }
        }
    }
    Ok(named)
}

/// Which face an icon is offered for, which is the last two digits of the icon's own number rather
/// than where it sits in the menu. Hrothgar are what tells the two apart: both of theirs offer four
/// faces numbered 5 to 8, so reading them off their positions would draw four other faces entirely,
/// and one of the two codes ships no lower face at all.
pub fn face(icon: i32) -> Option<u16> {
    match icon % 100 {
        0 => None,
        id => Some(id as u16),
    }
}

/// Everything one body's row offers, in the order the creator offers it. A menu with nothing to
/// choose from is one the row leaves empty rather than one it names.
fn menus(row: &ExcelRow<'_>, lobby: &impl ExcelSheet) -> Vec<Menu> {
    let mut found = Vec::new();
    for menu in 0..MENUS {
        let at = menu * STRIDE;
        let (Ok(count), Ok(kind), Ok(init), Ok(named), Ok(customize)) = (
            row.read::<u8>(at + COUNT),
            row.read::<u8>(at + KIND),
            row.read::<u8>(at + INIT),
            row.read::<u32>(at + LOBBY),
            row.read::<i32>(at + CUSTOMIZE),
        ) else {
            continue;
        };
        let (Some(kind), true) = (Kind::read(kind), count > 0) else {
            continue;
        };
        let params: Vec<i32> = (0..PARAM_COUNT.min(u32::from(count)))
            .filter_map(|param| row.read::<i32>(at + PARAMS + param * 4).ok())
            .collect();
        found.push(Menu {
            name: text(lobby, named).unwrap_or_else(|| format!("Customize {customize}")),
            customize: customize.max(0) as u32,
            kind,
            count: u32::from(count),
            range: match kind {
                Kind::Slider => [
                    params.get(2).copied().unwrap_or(0).max(0) as u32,
                    params.get(3).copied().unwrap_or(0).max(0) as u32,
                ],
                _ => [0, u32::from(count).saturating_sub(1)],
            },
            init: u32::from(init),
            labels: match kind {
                Kind::List => params
                    .iter()
                    .enumerate()
                    .map(|(at, param)| {
                        text(lobby, (*param).max(0) as u32).unwrap_or_else(|| (at + 1).to_string())
                    })
                    .collect(),
                _ => Vec::new(),
            },
            params,
        });
    }
    found
}

/// What `Lobby` calls a row, where it calls it anything.
fn text(lobby: &impl ExcelSheet, row: u32) -> Option<String> {
    lobby
        .get_row(row)
        .ok()?
        .read_string(LOBBY_TEXT)
        .ok()
        .map(|text| text.to_string())
        .filter(|text| !text.is_empty())
}

/// The choices the menu driving one customisation offers, as the row states them.
fn params(row: &ExcelRow<'_>, customize: i32) -> Vec<i32> {
    for menu in 0..MENUS {
        let at = menu * STRIDE;
        if row.read::<i32>(at + CUSTOMIZE).ok() != Some(customize) {
            continue;
        }
        return (0..PARAM_COUNT)
            .filter_map(|param| row.read::<i32>(at + PARAMS + param * 4).ok())
            .take_while(|param| *param != 0)
            .collect();
    }
    Vec::new()
}
