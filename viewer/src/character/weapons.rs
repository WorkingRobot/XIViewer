//! Weapons: which model an item is, where it attaches, and which stance it plays.
//!
//! `Item.ModelMain`/`ModelSub` pack a weapon's set, base and variant the same way `Gear` packs
//! equipment, but sixteen bits a field rather than eight: measured against real weapons (id 100
//! "Dated Bronze Gladius" reads `201/1/4`, id 124 "Dated Bone Hora" reads `301/9/2`), the top
//! sixteen bits are unused in every weapon Item sampled. `ModelSub` is the other hand's model for
//! most of them, but a fist weapon's is a hands equipment id: see [`FISTS`]. Where a weapon
//! attaches to the skeleton comes from the character's own `.atch` file, keyed by a three-letter
//! tag `chara/xls/weapontype/attach.wtd` gives every weapon model set.

use anyhow::{Context, Result};
use glam::{EulerRot, Mat4, Quat, Vec3};
use ironworks::excel::Language;
use ironworks::file::File;
use ironworks::file::atch::AttachPoints;
use ironworks::file::imc::ImageChange;
use std::io::Cursor;

use crate::backend::Backend;
use crate::character::Gear;
use crate::excel::provider::{ExcelProvider, ExcelSheet as _};

/// `Item`'s model quads, name, icon and `EquipSlotCategory`, as byte offsets.
const MODEL_MAIN: u32 = 24;
const MODEL_SUB: u32 = 32;
const NAME: u32 = 12;
const ICON: u32 = 136;
const SLOT_CATEGORY: u32 = 154;
/// Where `EquipSlotCategory` states whether a row fills the main hand or the off hand.
const MAIN_HAND: u32 = 0;
const OFF_HAND: u32 = 1;

/// The sets a fist weapon is filed under. `DrawDataContainer::LoadWeapon` reads the off-hand model
/// of a main hand in this range as the main's own plus fifty rather than off the item, and
/// `LoadEquipment` then draws the hands from what the item's `ModelSub` names.
const FISTS: std::ops::RangeInclusive<u16> = 1601..=1650;

/// The table naming which `.atch` point each weapon model set hangs from.
const ATTACH_TYPES: &str = "chara/xls/weapontype/attach.wtd";

/// A weapon model: the set its directory is filed under, the body within it, and the material
/// colourway. Packed the same shape as [`super::Gear`] but sixteen bits a field rather than eight.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Weapon {
    pub set: u16,
    pub base: u16,
    pub variant: u16,
}

impl Weapon {
    pub fn read(quad: u64) -> Option<Self> {
        (quad != 0).then_some(Self {
            set: quad as u16,
            base: (quad >> 16) as u16,
            variant: (quad >> 32) as u16,
        })
    }

    pub fn model(&self) -> String {
        format!(
            "chara/weapon/w{0:04}/obj/body/b{1:04}/model/w{0:04}b{1:04}.mdl",
            self.set, self.base
        )
    }
}

/// One weapon the game names, and what wearing it takes.
#[derive(Clone)]
pub struct Piece {
    pub name: String,
    pub icon: u32,
    pub weapon: Weapon,
    /// The other hand's own model, where this item carries one of its own: a fist weapon or a
    /// twinblade is one item with two, rather than a second item worn in the other slot.
    pub off_hand: Option<Weapon>,
    /// The gauntlets a fist weapon is worn with, which the game draws in the hands slot over
    /// whatever is worn there.
    pub gauntlets: Option<Gear>,
    /// Whether this item's own `EquipSlotCategory` covers the off hand, leaving nothing there to
    /// pick by hand.
    pub covers_off_hand: bool,
}

/// Weapon model set to `.atch` point, in the table's own ascending order.
pub type Tags = Vec<(u32, String)>;

/// The point a weapon model set hangs from, which the table states one of per run of sets.
pub(super) use crate::assets::viewers::wtd::code as tag;

/// Every weapon and shield the game names, split by which hand it is picked for, in one list per
/// hand, and the table saying where each model set hangs.
pub type Pieces = (Vec<Piece>, Vec<Piece>, Tags);

/// Every weapon and shield the game names, split by which hand it is picked for. A category that
/// covers the off hand rather than filling it (a fist weapon's second knuckle) never lists
/// anything for that hand: the item's own `off_hand` supplies it instead.
pub async fn read(backend: &Backend, language: Language) -> Result<Pieces> {
    let tags = crate::assets::viewers::wtd::types(&backend.files().read(ATTACH_TYPES).await?)
        .context("weapon attach types")?;
    let excel = backend.excel();
    let items = excel.get_sheet("Item", language).await?;
    let categories = excel.get_sheet("EquipSlotCategory", language).await?;

    let mut hands = std::collections::BTreeMap::new();
    for id in categories.get_row_ids() {
        let Ok(row) = categories.get_row(id) else {
            continue;
        };
        let main = row.read::<i8>(MAIN_HAND).ok() == Some(1);
        let off = row.read::<i8>(OFF_HAND).ok();
        if main || off == Some(1) || off == Some(-1) {
            hands.insert(id, (main, off == Some(1), off == Some(-1)));
        }
    }

    let (mut main_hand, mut off_hand) = (Vec::new(), Vec::new());
    for id in items.get_row_ids() {
        let Ok(row) = items.get_row(id) else {
            continue;
        };
        let Some(&(fills_main, fills_off, covers_off)) = row
            .read::<u8>(SLOT_CATEGORY)
            .ok()
            .and_then(|category| hands.get(&u32::from(category)))
        else {
            continue;
        };
        let Some(weapon) = row.read::<u64>(MODEL_MAIN).ok().and_then(Weapon::read) else {
            continue;
        };
        let Ok(name) = row.read_string(NAME) else {
            continue;
        };
        let name = name.to_string();
        if name.is_empty() {
            continue;
        }
        // A fist weapon's `ModelSub` is a hands equipment id rather than a weapon, and the knuckle
        // in the other hand is the main's own set plus fifty.
        let sub = row.read::<u64>(MODEL_SUB).ok().unwrap_or(0);
        let fists = FISTS.contains(&weapon.set);
        let piece = Piece {
            name,
            icon: row.read::<u16>(ICON).unwrap_or(0).into(),
            weapon,
            off_hand: match fists {
                true => Some(Weapon { set: weapon.set + 50, ..weapon }),
                false => Weapon::read(sub),
            },
            gauntlets: fists.then(|| Gear::read(sub)).flatten(),
            covers_off_hand: covers_off,
        };
        if fills_main {
            main_hand.push(piece.clone());
        }
        if fills_off {
            off_hand.push(piece);
        }
    }
    main_hand.sort_by(|left, right| left.name.cmp(&right.name));
    off_hand.sort_by(|left, right| left.name.cmp(&right.name));
    log::info!(
        "character: {} main hand, {} off hand weapons, {} attach points",
        main_hand.len(),
        off_hand.len(),
        tags.len()
    );
    Ok((main_hand, off_hand, tags))
}

/// Where a race's `.atch` file is filed, which names its weapon and tool attach points.
pub fn atch_path(code: u16) -> String {
    format!("chara/xls/attachoffset/c{code:04}.atch")
}

/// One placement a weapon's attach point takes.
pub struct Attach {
    pub bone: String,
    pub scale: f32,
    pub offset: [f32; 3],
    pub rotation: [f32; 3],
}

impl Attach {
    /// The same placement on the other side of the body: the bone's own `_l` or `_r` flipped, and
    /// the placement reflected across the plane between them.
    fn mirrored(self) -> Self {
        Self {
            bone: across(&self.bone).unwrap_or(self.bone),
            offset: [self.offset[0], self.offset[1], -self.offset[2]],
            rotation: [-self.rotation[0], -self.rotation[1], self.rotation[2]],
            ..self
        }
    }

    /// Where this puts a weapon relative to the bone it hangs from. The three angles compose as
    /// `Rz * Ry * Rx`, which is how the client builds the point's own quaternion; the offset is
    /// the bone's own and is added after the turn.
    pub fn placement(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(
            Vec3::splat(self.scale),
            Quat::from_euler(
                EulerRot::ZYX,
                self.rotation[2],
                self.rotation[1],
                self.rotation[0],
            ),
            Vec3::from_array(self.offset),
        )
    }
}

/// The placement `tag` takes in the drawn or the sheathed state, out of a race's own `.atch` file,
/// for the slot it is worn in. State 0 is drawn and state 1 is sheathed: measured over every point
/// c0101.atch carries, state 0 is the bare, unoffset bone in 106 of 143 and state 1 is a placement
/// of its own in 96 of 143, and no point ever states only the first.
///
/// Nothing states a left-hand placement of its own. The client mirrors one instead, whenever the
/// slot being worn is the second and the point is no accessory, or the point is an accessory and
/// the slot is not the second: the bone's own trailing `_l` or `_r` is flipped, and the placement
/// is reflected with it.
pub fn attach(bytes: &[u8], tag: &str, drawn: bool, off_hand: bool) -> Option<Attach> {
    let file = AttachPoints::read(Cursor::new(bytes.to_vec())).ok()?;
    let point = file.point(tag)?;
    let state = file.states(point)?.get(usize::from(!drawn))?;
    let placement = Attach {
        bone: state.bone().to_owned(),
        scale: state.scale(),
        offset: state.offset(),
        rotation: state.rotation(),
    };
    Some(match off_hand != file.accessory(point) {
        true => placement.mirrored(),
        false => placement,
    })
}

/// The lowest effect id filed apart from any one weapon, out of `Weapon::ResolveVfxPath`: below
/// this the effect is the weapon's own, at or above it the shared one every weapon reads from.
const SHARED_VFX: u8 = 100;

/// Where the effect a weapon's own `.imc` names is filed, for the variant it is worn at. The game
/// only plays this while the weapon is drawn, which is what makes a relic glow in a battle stance
/// and not out of one.
pub fn vfx_path(weapon: &Weapon, bytes: &[u8]) -> Option<String> {
    let file = ImageChange::read(Cursor::new(bytes.to_vec())).ok()?;
    let vfx = file.entry(0, weapon.variant)?.vfx_id();
    match vfx {
        0 => None,
        SHARED_VFX.. => Some(format!("vfx/weapon/eff/vw{vfx:04}.avfx")),
        _ => Some(format!(
            "chara/weapon/w{:04}/obj/body/b{:04}/vfx/eff/vw{vfx:04}.avfx",
            weapon.set, weapon.base
        )),
    }
}

/// The same bone on the other side of the body, for a name that carries a side at all.
fn across(bone: &str) -> Option<String> {
    let side = match bone.as_bytes() {
        [.., b'_', b'l'] => 'r',
        [.., b'_', b'r'] => 'l',
        _ => return None,
    };
    let mut across = bone.to_owned();
    across.pop();
    across.push(side);
    Some(across)
}

/// The bone on the other side of the body from `bone`, for a point that names it in one of its own
/// states. A summon states no hand of its own and hangs from the point its model set names, so two
/// of one set would stack in the same hand; this is what the file itself offers as the other one.
/// Nothing where the name carries no side, or where the point never names the mirror.
pub fn other_hand(bytes: &[u8], tag: &str, bone: &str) -> Option<String> {
    let other = across(bone)?;
    let file = AttachPoints::read(Cursor::new(bytes.to_vec())).ok()?;
    let point = file.point(tag)?;
    file.states(point)?
        .iter()
        .any(|state| state.bone() == other)
        .then_some(other)
}

/// The bone a weapon hangs from when nothing names an attach point for it: the plain right or left
/// hand null bone, whichever `main` says.
pub fn fallback_bone(main: bool) -> &'static str {
    match main {
        true => "n_buki_r",
        false => "n_buki_l",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The order a point's own three angles compose in, which the client builds by hand out of
    /// their halved sines and cosines: a sheathed sword states `[pi, 0, pi/2]`, and only this
    /// order turns the model's own length onto the hip the way round the game hangs it.
    /// The off hand is worn at the same point read the other way round: the bone's own side
    /// swapped, and the placement reflected with it.
    #[test]
    fn the_off_hand_wears_the_main_hands_point_mirrored() {
        let point = Attach {
            bone: "n_buki_r".to_owned(),
            scale: 0.9,
            offset: [0.1, 0.2, 0.3],
            rotation: [0.4, 0.5, 0.6],
        }
        .mirrored();
        assert_eq!(point.bone, "n_buki_l");
        assert_eq!(point.offset, [0.1, 0.2, -0.3]);
        assert_eq!(point.rotation, [-0.4, -0.5, 0.6]);
        assert_eq!(point.scale, 0.9);

        let bare = Attach {
            bone: "n_throw".to_owned(),
            scale: 1.0,
            offset: [0.0; 3],
            rotation: [0.0; 3],
        }
        .mirrored();
        assert_eq!(bare.bone, "n_throw", "a bone that names no side keeps it");
    }

    #[test]
    fn an_attach_point_turns_a_weapon_the_way_the_client_composes_its_angles() {
        use std::f32::consts::{FRAC_PI_2, PI};
        let sheathed = Attach {
            bone: String::new(),
            scale: 1.0,
            offset: [0.0; 3],
            rotation: [PI, 0.0, FRAC_PI_2],
        };
        let along = sheathed.placement().transform_vector3(Vec3::Y);
        assert!((along - Vec3::X).length() < 1e-5, "{along:?}");
    }

    #[test]
    fn a_weapon_quad_packs_set_base_and_variant_in_sixteen_bits_each() {
        assert!(Weapon::read(0).is_none());
        let gladius = Weapon::read(0x0000_0004_0001_00c9).unwrap();
        assert_eq!((gladius.set, gladius.base, gladius.variant), (201, 1, 4));
        assert_eq!(
            gladius.model(),
            "chara/weapon/w0201/obj/body/b0001/model/w0201b0001.mdl"
        );

        let hora = Weapon::read(0x0000_0002_0009_012d).unwrap();
        assert_eq!((hora.set, hora.base, hora.variant), (301, 9, 2));
    }

    /// The split `Weapon::ResolveVfxPath` makes: below a hundred the effect is filed under the
    /// weapon's own directory, at or above it under the one every weapon shares.
    #[test]
    fn an_effect_is_the_weapons_own_until_it_is_shared() {
        let weapon = Weapon {
            set: 201,
            base: 1,
            variant: 0,
        };
        let entry = |vfx: u8| {
            let mut bytes = vec![0, 0, 1, 0];
            bytes.extend([0, 0, 0, 0, vfx, 0]);
            bytes
        };
        assert_eq!(vfx_path(&weapon, &entry(0)), None);
        assert_eq!(
            vfx_path(&weapon, &entry(2)).as_deref(),
            Some("chara/weapon/w0201/obj/body/b0001/vfx/eff/vw0002.avfx")
        );
        assert_eq!(
            vfx_path(&weapon, &entry(150)).as_deref(),
            Some("vfx/weapon/eff/vw0150.avfx")
        );
    }

    /// One weapon the install itself gives an effect: `w0401b0080`'s third variant names vfx 2,
    /// and the file that resolves to is one the install ships.
    #[test]
    #[ignore = "reads the real local FFXIV install"]
    fn a_weapon_variant_that_names_an_effect_resolves_a_file_the_install_holds() {
        let install = ironworks::Ironworks::new().with_resource(ironworks::sqpack::SqPack::new(
            ironworks::sqpack::Install::at_sqpack("/home/asriel/.xlcore/ffxiv/game/sqpack"),
        ));
        let imc: Vec<u8> = install
            .file("chara/weapon/w0401/obj/body/b0080/b0080.imc")
            .expect("the imc");
        let weapon = Weapon {
            set: 401,
            base: 80,
            variant: 2,
        };
        let path = vfx_path(&weapon, &imc).expect("this variant names an effect");
        assert_eq!(path, "chara/weapon/w0401/obj/body/b0080/vfx/eff/vw0002.avfx");
        assert!(install.file::<Vec<u8>>(&path).is_ok(), "{path}");

        let plain = Weapon { variant: 0, ..weapon };
        assert_eq!(vfx_path(&plain, &imc), None, "the default variant has none");
    }

    #[test]
    fn a_fist_weapon_wears_the_set_past_its_own_and_gauntlets() {
        // "Ultimate Omega Knuckles": set 1601, and a `ModelSub` naming equipment set 8808.
        let knuckles = Weapon::read(0x0000_0002_0002_0641).unwrap();
        assert!(FISTS.contains(&knuckles.set));
        assert_eq!(
            Weapon { set: knuckles.set + 50, ..knuckles }.model(),
            "chara/weapon/w1651/obj/body/b0002/model/w1651b0002.mdl"
        );
        let gauntlets = Gear::read(0x0000_0000_0002_2268).unwrap();
        assert_eq!((gauntlets.set, gauntlets.variant), (8808, 2));

        // "Dated Bone Hora", whose own `ModelSub` is the second knuckle rather than gauntlets.
        assert!(!FISTS.contains(&301));
    }

    /// The points `attach.wtd` itself names for the first weapon sets it carries, and the clamp a
    /// set it states nothing of reads under.
    #[test]
    #[ignore = "reads the real local FFXIV install"]
    fn a_weapon_set_reads_the_point_the_table_hangs_it_from() {
        let install = ironworks::Ironworks::new().with_resource(ironworks::sqpack::SqPack::new(
            ironworks::sqpack::Install::at_sqpack("/home/asriel/.xlcore/ffxiv/game/sqpack"),
        ));
        let bytes: Vec<u8> = install.file(ATTACH_TYPES).expect("the table");
        let tags = crate::assets::viewers::wtd::types(&bytes).expect("a readable table");
        assert_eq!(tag(&tags, 101), Some("sld"), "a shield");
        assert_eq!(tag(&tags, 201), Some("swd"), "a gladius");
        assert_eq!(tag(&tags, 1601), Some("clg"), "a fist weapon");
        assert_eq!(
            tag(&tags, 1651),
            Some("clg"),
            "the second knuckle clamps into the first"
        );

        // Every point a combat weapon set is sent to is one the body's own `.atch` carries. A few
        // of the tool points are in no player race's file, and those hang off the bare hand bone.
        let bytes: Vec<u8> = install.file(&atch_path(101)).expect("the attach points");
        let points = AttachPoints::read(Cursor::new(bytes)).expect("a readable file");
        for set in (101..=2801).step_by(100) {
            let held = tag(&tags, set).expect("every set reads a point");
            assert!(points.point(held).is_some(), "c0101.atch has no {held}");
        }
    }

    const SQPACK: &str = "/home/asriel/.xlcore/ffxiv/game/sqpack";

    /// Where a second summon of one model set goes, off the real install. Cheer On: Orange summons
    /// `w1980` twice, whose point `nmf` hangs it from `n_buki_r`; the point's own states name
    /// `n_buki_l` too, which is the hand the second one is moved to. A point that never names the
    /// mirror offers none, and neither does a bone whose name carries no side.
    #[test]
    #[ignore = "reads the real local FFXIV install"]
    fn a_second_summon_of_one_set_reads_the_other_hand_off_the_point() {
        use ironworks::Ironworks;
        use ironworks::sqpack::{Install, SqPack};

        let install =
            Ironworks::new().with_resource(Box::new(SqPack::new(Install::at_sqpack(SQPACK))));
        let bytes: Vec<u8> = install
            .file("chara/xls/attachoffset/c0101.atch")
            .expect("the attach points");
        assert_eq!(
            other_hand(&bytes, "nmf", "n_buki_r").as_deref(),
            Some("n_buki_l")
        );
        assert_eq!(
            other_hand(&bytes, "sld", "n_buki_tate_l").as_deref(),
            Some("n_buki_tate_r"),
            "a shield is worn left and its point names the right"
        );
        assert_eq!(
            other_hand(&bytes, "nmf", "n_throw"),
            None,
            "a bone that names no side has no other one"
        );
        assert_eq!(
            other_hand(&bytes, "avt", "n_buki_r"),
            None,
            "a point that only ever throws names neither hand"
        );
    }
}
