//! Dressing a character out of the files its body, face and hair are filed under.
//!
//! A code names a body, and everything that body can wear sits under it: `obj/body`, `obj/face` and
//! `obj/hair` each hold a numbered set, and a set's `model` directory holds every piece of it. What
//! a set is made of is read from that directory rather than from a list of suffixes, since a face is
//! several models and which ones it carries is the file tree's to say.
//!
//! What each set is offered under comes from the creator's own menus, in [`menus`].

mod emotes;
mod gating;
mod menus;
mod mounts;
mod npcs;
mod palette;
mod stains;
pub mod stand;
mod stance;
mod weapons;

use std::cell::{Cell, Ref, RefCell};
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;
use std::sync::Arc;

use anyhow::Result;
use egui::{
    Align, CentralPanel, CollapsingHeader, Color32, Layout, Popup, PopupCloseBehavior, RectAlign,
    RichText, ScrollArea, TextEdit, containers::panel::Panel,
};
use glam::{Mat4, Vec3};
use ironworks::excel::Language;

use crate::assets::viewers::{fpeb, mdl};
use crate::backend::Backend;
use crate::data::get_icon_path;
use crate::data::listing::{Listed, Listing};
use crate::excel::provider::ExcelProvider;
use crate::settings::{LANGUAGE, api_base};
use crate::utils::{
    CollapsibleSidePanel, FuzzyMatcher, IconManager, ManagedIcon, Side, TrackedPromise,
    icon_context_menu,
};

/// The set every character is built from. The game holds one body mesh, `c0101b0001`, and stands
/// every other one on it: no other code ships a model of the set at all, while twelve of them ship
/// its material, which is the skin their own body is drawn with.
const BODY_SET: u16 = 1;
/// What each body differs from the one it is built on by.
pub(super) const DEFORMERS: &str = "chara/xls/boneDeformer/human.pbd";

/// How big a set's icon is drawn, and how far apart the grid sets them.
const ICON: f32 = 40.0;
const GAP: f32 = 4.0;
/// How many rows an icon menu shows before it scrolls, so a long one (a hairstyle) does not push
/// everything under it down the panel.
const ICON_ROWS: usize = 3;

/// How big a piece of equipment's icon is drawn beside its name, and how many of them a slot's
/// picker shows at once.
const PIECE: f32 = 24.0;
const SHOWN: usize = 10;

/// How wide the picker panel may grow. A panel takes the width its widest row asks for and keeps
/// it, so a piece name long enough to run on would otherwise take the view beside it for good.
const PANEL_WIDTH: f32 = 380.0;
const PANEL_MIN_WIDTH: f32 = 180.0;

/// Which customisation each of the creator's menus drives, as `Customize` numbers them. Every one
/// of these is measured from `CharaMakeType` rather than named by any file.
pub(super) const FACE: u32 = 5;
pub(super) const HAIRSTYLE: u32 = 6;
const SKIN_COLOR: u32 = 8;
const EYE_COLOR: u32 = 9;
const HAIR_COLOR: u32 = 10;
const FEATURES: u32 = 12;
/// The top bit of the facial-features byte, which is on wherever a race's tattoo is: the game packs
/// it beside the seven feature checks rather than offering a menu of its own.
const LEGACY_TATTOO: u32 = 0x80;
const TATTOO_COLOR: u32 = 13;
const LIP_COLOR: u32 = 20;
pub(super) const FACE_PAINT: u32 = 24;
const FACE_PAINT_COLOR: u32 = 25;
pub(super) const HEIGHT: u32 = 3;
/// Muscle tone, on a body the creator offers no tail or ears; every other race spends the same
/// customisation on the length of whatever its [`TAIL`] menu shapes.
const MUSCLE_TONE: u32 = 21;
const BUST: u32 = 23;
/// A tail, or a Viera's ears: the game files both under the one customisation, and under one
/// numbered set beneath the body that grows it.
pub(super) const TAIL: u32 = 22;

/// What the creator ticks beside a menu rather than offering as a menu of its own, keyed past every
/// menu number so the two can never collide. The game holds each of them all the same: a highlight
/// is the top bit of one byte and a colour of its own in another, and an eye is odd when the two
/// eyes are not left the one colour.
pub const HIGHLIGHTS: u32 = 100;
pub const HIGHLIGHT_COLOR: u32 = 101;
pub const ODD_EYES: u32 = 102;
pub const LEFT_EYE_COLOR: u32 = 103;
pub const LIPSTICK: u32 = 104;

/// Where the light half of a split palette begins. A lip and a face paint are offered twice over,
/// the same colours at two weights, and light is the half worn the more lightly of the two.
const HALF: u32 = 128;

/// The parts of a face the creator deforms, and the shape keys each is named with. A choice picks
/// the nth shape the model declares for that part, counting the first choice as the face's own.
const SHAPED: [(u32, &str); 6] = [
    (14, "shp_brw"),
    (16, "shp_eye"),
    (15, "shp_irs"),
    (19, "shp_mth"),
    (17, "shp_nse"),
    (18, "shp_chk"),
];

/// What a face calls the parts a facial feature draws as, one letter each. The creator splits them
/// across two menus and the model declares them as one run.
const FEATURE: &str = "atr_fv_";
const FEATURE_LETTERS: [char; 7] = ['a', 'b', 'c', 'd', 'e', 'f', 'g'];

/// How big a colour swatch is drawn.
const SWATCH: f32 = 18.0;

/// Smallclothes, which is what everything else is worn over.
const SMALLCLOTHES: Gear = Gear { set: 0, variant: 1 };

/// The directory and letter a slot's models are filed under, as [`Slot::filed`] answers it.
type Filed = (&'static str, char);

/// A slot a character wears something in, as the file names abbreviate it. The five it is dressed
/// in come first, in the order every sheet states them, and what it is adorned with after.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Slot {
    Head,
    Body,
    Hands,
    Legs,
    Feet,
    Ears,
    Neck,
    Wrists,
    RingLeft,
    RingRight,
    Facewear,
}

impl Slot {
    /// The five a body is dressed in.
    pub const GEAR: [Slot; 5] = [Self::Head, Self::Body, Self::Hands, Self::Legs, Self::Feet];
    /// What it is adorned with beyond its gear, which hides nothing of it. Every one but facewear
    /// is filed apart from the gear as well; facewear is not, sharing the head's own listing.
    pub const ADORNMENT: [Slot; 6] = [
        Self::Ears,
        Self::Neck,
        Self::Wrists,
        Self::RingLeft,
        Self::RingRight,
        Self::Facewear,
    ];
    pub const ALL: [Slot; 11] = [
        Self::Head,
        Self::Body,
        Self::Hands,
        Self::Legs,
        Self::Feet,
        Self::Ears,
        Self::Neck,
        Self::Wrists,
        Self::RingLeft,
        Self::RingRight,
        Self::Facewear,
    ];
    /// The slots a race has clothing of its own for, in the order `Race` states them.
    pub const RACIAL: [Slot; 4] = [Self::Body, Self::Hands, Self::Legs, Self::Feet];

    fn name(self) -> &'static str {
        match self {
            Self::Head => "Head",
            Self::Body => "Body",
            Self::Hands => "Hands",
            Self::Legs => "Legs",
            Self::Feet => "Feet",
            Self::Ears => "Ears",
            Self::Neck => "Neck",
            Self::Wrists => "Wrists",
            Self::RingLeft => "Left ring",
            Self::RingRight => "Right ring",
            Self::Facewear => "Facewear",
        }
    }

    fn suffix(self) -> &'static str {
        match self {
            // Facewear shares the head's own suffix: the game resolves it through the same fixed
            // entry of the suffix table `ResolveMdlPath` loads for a head set.
            Self::Head | Self::Facewear => "met",
            Self::Body => "top",
            Self::Hands => "glv",
            Self::Legs => "dwn",
            Self::Feet => "sho",
            Self::Ears => "ear",
            Self::Neck => "nek",
            Self::Wrists => "wrs",
            Self::RingLeft => "ril",
            Self::RingRight => "rir",
        }
    }

    /// Whether this is one of the pieces worn over the gear rather than the gear itself: it hides
    /// nothing under it, has no attire default, and shows "None" rather than "Bare" where its
    /// picker is empty.
    pub(super) fn adornment(self) -> bool {
        Self::ADORNMENT.contains(&self)
    }

    /// The directory and letter a worn set here is filed under. Facewear is the one adornment that
    /// files as plain equipment rather than as an accessory.
    pub(super) fn filed(self) -> Filed {
        match self.adornment() && self != Self::Facewear {
            true => ("accessory", 'a'),
            false => ("equipment", 'e'),
        }
    }
}

/// A set and the variant it is worn at, which is how a model quad states a piece of equipment.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Gear {
    pub set: u16,
    pub variant: u16,
}

impl Gear {
    pub fn read(quad: u64) -> Option<Self> {
        (quad != 0).then_some(Self {
            set: quad as u16,
            variant: (quad >> 16) as u16,
        })
    }
}

/// What a character wears, by slot.
pub type Outfit = [Option<Gear>; 11];

/// The model each slot of one set is worn as, where the code carries one at all.
type Models = [Option<String>; 11];

/// What the creator dresses a character in before anything is picked for them.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum Attire {
    #[default]
    Race,
    Job,
    Smallclothes,
    /// What the character being stood in for wears, which is not one of the creator's own.
    Npc,
}

/// The bytes of every file read so far, by path, and one batch of them as they land.
type Files = BTreeMap<String, Vec<u8>>;
type Read = Vec<(String, Vec<u8>)>;
/// The race a `.atch` fetch was asked for, alongside the bytes once it lands.
type AtchRead = (u16, Vec<u8>);

/// The stance the body was last put in: the class directory its weapons name, whether they were
/// drawn, and whether the pose it settled on has been named yet.
struct Stood {
    held: String,
    drawn: bool,
    told: Cell<bool>,
}

/// The emotes the game names and the poses each seat cycles, which are read together.
type Played = (Vec<emotes::Emote>, emotes::Poses);

/// What a picked set is made of, and what to call it.
pub(super) struct Set {
    id: u16,
    parts: Vec<String>,
}

pub struct CharacterBuilder {
    listing: Option<Rc<Listing>>,
    /// What the creator offers, and which of its races, clans and genders is being built.
    creator: menus::Creator,
    reading: Option<TrackedPromise<Result<menus::Creator>>>,
    reading_pieces: Option<TrackedPromise<Result<[Vec<menus::Piece>; 11]>>>,
    race: u32,
    tribe: u32,
    female: bool,
    /// Whether the body is the child one the game grows a race's own children on, which only some
    /// races are built.
    child: bool,
    /// Whether the sets on hand are the picked clan's. A pick drops it rather than the sets
    /// themselves, so a clan this version ships nothing for is read once and not on every frame.
    stood: bool,
    /// The code the picked clan and gender resolve to, the body whose skin it is drawn with, and
    /// the sets it carries.
    code: u16,
    skin: Option<u16>,
    body: Vec<String>,
    faces: Vec<Set>,
    hairs: Vec<Set>,
    tails: Vec<Set>,
    face: u16,
    hair: u16,
    attire: Attire,
    job: usize,
    /// What has been picked by hand for a slot, over whatever the attire puts there, and which
    /// slot's picker is open. Both index [`menus::Creator::pieces`].
    chosen: [Option<usize>; 11],
    picking: Option<Slot>,
    search: [String; 11],
    matched: RefCell<[(Option<String>, Vec<usize>); 11]>,
    matcher: FuzzyMatcher,
    /// What has been picked for each of the creator's menus, by the `Customize` it drives. A menu
    /// the map says nothing about is at its first choice, which is what the row's own defaults are.
    choices: BTreeMap<u32, u32>,
    /// The colours the creator offers, read once.
    made: Option<palette::Made>,
    reading_made: Option<TrackedPromise<Result<palette::Made>>>,
    /// The dyes the game names, read once.
    dyes: Vec<stains::Stain>,
    reading_dyes: Option<TrackedPromise<Result<Vec<stains::Stain>>>>,
    /// The staining templates a dye's values come from, read once.
    dye_templates: Option<Rc<mdl::DyeTemplates>>,
    reading_dye_templates: Option<TrackedPromise<Result<mdl::DyeTemplates>>>,
    /// What has been picked to stain each slot with, one id per channel a modern item can carry.
    /// Zero is the unstained slot, matching a `.stm` template's own numbering.
    stains: [[Option<u8>; 2]; 11],
    /// What each body differs from the one it is built on by, which is both what says where a
    /// borrowed model comes from and what shapes it onto the body wearing it.
    deformers: Option<Rc<mdl::Deformers>>,
    reading_deformers: Option<TrackedPromise<Result<mdl::Deformers>>>,
    /// What a worn piece leaves showing of the body under it.
    worn_over: Option<gating::Worn>,
    /// Whether the visor of the hat being worn is raised.
    visor: bool,
    reading_worn: Option<TrackedPromise<Result<gating::Worn>>>,
    /// The game's own characters, and which of them is being stood in.
    npcs: Vec<npcs::Npc>,
    reading_npcs: Option<TrackedPromise<Result<Vec<npcs::Npc>>>>,
    npc: Option<usize>,
    npc_search: String,
    npcs_matched: RefCell<(Option<String>, Vec<usize>)>,
    /// The emotes the game names, and which of them is being played.
    emotes: Vec<emotes::Emote>,
    reading_emotes: Option<TrackedPromise<Result<Played>>>,
    /// The poses each seat cycles through, and where in one the body currently stands.
    poses: emotes::Poses,
    posture: emotes::Posture,
    pose: usize,
    emote: Option<usize>,
    emote_search: String,
    emotes_matched: RefCell<(Option<String>, Vec<usize>)>,
    /// The mounts the game names, and which of them the character is seated on.
    mounts: Vec<mounts::Mount>,
    reading_mounts: Option<TrackedPromise<Result<Vec<mounts::Mount>>>>,
    mount: Option<usize>,
    /// Which of the mount's own seats the character rides in, for one that seats more than one.
    mount_seat: usize,
    mount_search: String,
    mounts_matched: RefCell<(Option<String>, Vec<usize>)>,
    /// Every weapon the game names, split by which hand it can be picked for, and which of each
    /// list is worn.
    weapons_main: Vec<weapons::Piece>,
    weapons_off: Vec<weapons::Piece>,
    /// Which `.atch` point each weapon model set hangs from.
    weapon_tags: weapons::Tags,
    reading_weapons: Option<TrackedPromise<Result<weapons::Pieces>>>,
    main_hand: Option<usize>,
    off_hand: Option<usize>,
    main_search: String,
    off_search: String,
    main_matched: RefCell<(Option<String>, Vec<usize>)>,
    off_matched: RefCell<(Option<String>, Vec<usize>)>,
    /// Whether the weapon on screen is drawn, which is a whole stance rather than a placement.
    drawn: bool,
    /// Which motion class each weapon puts the body in, and how long the game blends one motion
    /// into another.
    stance: Option<Rc<stance::Stance>>,
    reading_stance: Option<TrackedPromise<Result<stance::Stance>>>,
    /// The stance the body was last put in, so one it is already standing in is not started over
    /// on every frame.
    stood_in: RefCell<Option<Stood>>,
    /// The effects the drawn weapons were last carrying, so they are named once rather than on
    /// every frame they play.
    glowed: RefCell<Vec<String>>,
    /// What was last logged a weapon's placement, so a stance held for a thousand frames names its
    /// bone and offset once rather than on every one of them.
    logged: Cell<(bool, Option<usize>, Option<usize>, bool)>,
    /// The race's own `.atch` file, which says where a weapon it names a tag for hangs, kept
    /// against the code it was fetched for so a change of race asks again.
    atch: Option<(u16, Rc<Vec<u8>>)>,
    reading_atch: Option<TrackedPromise<Result<AtchRead>>>,
    /// The eye-size table, which is one file for every body the game ships.
    facial: Option<Rc<Vec<u8>>>,
    reading_facial: Option<TrackedPromise<Result<Vec<u8>>>>,
    /// The models each set is worn as under the current code, by slot. A set number means one
    /// thing filed as equipment and another filed as an accessory, so the two are kept apart. The
    /// picker asks about every set it lists, and a directory listing is too dear to pay for one on
    /// every frame.
    sets: RefCell<BTreeMap<(Filed, u16), Models>>,
    /// The files the model on screen was built from, so a pick that changes nothing costs nothing.
    worn: Vec<(String, u16)>,
    /// The stains each entry of `worn` is dressed in, in the same order: `[None; 2]` for the face,
    /// hair, tail and mount entries `wearing` adds itself, and the picked slot's own for the rest.
    worn_stains: Vec<[Option<u8>; 2]>,
    /// Whether the last fetch for `worn` failed, so the same equipment is asked for again rather
    /// than left stuck: `worn` alone can't tell a landed dress from a failed one.
    worn_failed: bool,
    /// What each borrowed body is shaped onto this one by, kept rather than rebuilt. The model
    /// keeps a piece across a change of clothes by the deform it was built with, and one built
    /// afresh is a different one however equal it is.
    shaped: RefCell<BTreeMap<u16, Option<Arc<mdl::Deform>>>>,
    /// Every file read so far, so a change of clothes only asks for what it newly needs.
    held: Files,
    /// Batches of files still on their way. A batch is never abandoned: dropping one cancels the
    /// request under it, and what it was fetching is worth keeping whether or not the character has
    /// changed clothes since.
    fetching: Vec<TrackedPromise<Result<Read>>>,
    model: Option<Result<Box<mdl::Rendered>, String>>,
    /// The props the playing emote's own timeline wants held, each by path, material variant and
    /// the weapon set it hangs from, read off the model itself once a frame so `worn` picks them up
    /// the same way it does a weapon. A motion that puts one thing in each hand summons twice.
    props: Vec<(String, u16, u16)>,
}

impl Default for CharacterBuilder {
    fn default() -> Self {
        Self {
            listing: None,
            creator: menus::Creator::default(),
            reading: None,
            reading_pieces: None,
            race: 1,
            tribe: 1,
            female: false,
            child: false,
            stood: false,
            code: 101,
            skin: None,
            body: Vec::new(),
            faces: Vec::new(),
            hairs: Vec::new(),
            tails: Vec::new(),
            face: 1,
            hair: 1,
            attire: Attire::default(),
            job: 0,
            chosen: [None; 11],
            picking: None,
            search: Default::default(),
            matched: Default::default(),
            matcher: FuzzyMatcher::new(),
            choices: BTreeMap::new(),
            made: None,
            reading_made: None,
            dyes: Vec::new(),
            reading_dyes: None,
            dye_templates: None,
            reading_dye_templates: None,
            stains: [[None; 2]; 11],
            deformers: None,
            reading_deformers: None,
            worn_over: None,
            visor: false,
            reading_worn: None,
            npcs: Vec::new(),
            reading_npcs: None,
            npc: None,
            npc_search: String::new(),
            npcs_matched: Default::default(),
            emotes: Vec::new(),
            reading_emotes: None,
            poses: emotes::Poses::default(),
            posture: emotes::Posture::default(),
            pose: 0,
            emote: None,
            emote_search: String::new(),
            emotes_matched: Default::default(),
            mounts: Vec::new(),
            reading_mounts: None,
            mount: None,
            mount_seat: 0,
            mount_search: String::new(),
            mounts_matched: Default::default(),
            weapons_main: Vec::new(),
            weapons_off: Vec::new(),
            weapon_tags: Vec::new(),
            reading_weapons: None,
            main_hand: None,
            off_hand: None,
            main_search: String::new(),
            off_search: String::new(),
            main_matched: Default::default(),
            off_matched: Default::default(),
            drawn: false,
            stance: None,
            reading_stance: None,
            stood_in: RefCell::new(None),
            glowed: RefCell::new(Vec::new()),
            logged: Cell::new((false, None, None, false)),
            atch: None,
            reading_atch: None,
            facial: None,
            reading_facial: None,
            sets: RefCell::new(BTreeMap::new()),
            worn: Vec::new(),
            worn_stains: Vec::new(),
            worn_failed: false,
            shaped: RefCell::new(BTreeMap::new()),
            held: Files::new(),
            fetching: Vec::new(),
            model: None,
            props: Vec::new(),
        }
    }
}

impl CharacterBuilder {
    /// Drop everything that came from the install, so a reconnect reads it all again.
    pub fn reset(&mut self) {
        self.listing = None;
        self.creator = menus::Creator::default();
        self.reading = None;
        self.reading_pieces = None;
        self.made = None;
        self.reading_made = None;
        self.dyes.clear();
        self.reading_dyes = None;
        self.dye_templates = None;
        self.reading_dye_templates = None;
        self.deformers = None;
        self.reading_deformers = None;
        self.worn_over = None;
        self.reading_worn = None;
        self.npcs.clear();
        self.reading_npcs = None;
        self.npc = None;
        self.npcs_matched.take();
        self.emotes.clear();
        self.poses = emotes::Poses::default();
        self.posture = emotes::Posture::default();
        self.pose = 0;
        self.reading_emotes = None;
        self.emote = None;
        self.emotes_matched.take();
        self.mounts.clear();
        self.reading_mounts = None;
        self.mount = None;
        self.mount_seat = 0;
        self.mounts_matched.take();
        self.weapons_main.clear();
        self.weapons_off.clear();
        self.weapon_tags.clear();
        self.reading_weapons = None;
        self.main_hand = None;
        self.off_hand = None;
        self.main_matched.take();
        self.off_matched.take();
        self.logged.set((false, None, None, false));
        self.stood_in.take();
        self.glowed.take();
        self.atch = None;
        self.reading_atch = None;
        self.facial = None;
        self.reading_facial = None;
        self.stood = false;
        self.body.clear();
        self.faces.clear();
        self.hairs.clear();
        // What was picked by hand is where a piece sat in a list that is about to be read again.
        self.chosen = [None; 11];
        self.matched.take();
        self.sets.borrow_mut().clear();
        self.shaped.borrow_mut().clear();
        self.worn.clear();
        self.worn_stains.clear();
        self.worn_failed = false;
        self.held.clear();
        self.fetching.clear();
        self.model = None;
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, backend: &Backend, icons: &IconManager) {
        self.poll(ui.ctx(), backend);
        self.side_panel(ui, backend, icons);
        CentralPanel::default().show(ui, |ui| {
            if CollapsibleSidePanel::is_collapsed(ui.ctx(), "character_pick") {
                Panel::top("character_reexpand").show(ui, |ui| {
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        CollapsibleSidePanel::draw_arrow(ui, "character_pick", Side::Left);
                    });
                    ui.add_space(4.0);
                });
            }
            match &self.model {
                Some(Ok(model)) => mdl::ui(ui, model, backend),
                Some(Err(why)) => {
                    ui.centered_and_justified(|ui| {
                        ui.colored_label(Color32::LIGHT_RED, why);
                    });
                }
                None => {
                    ui.centered_and_justified(|ui| {
                        ui.spinner();
                    });
                }
            }
        });
    }

    /// Asks the install for everything the tab is built out of: what the creator offers, the
    /// colours it offers them in, the deformers, the emotes and mounts, what a worn piece covers,
    /// and the characters the game stands itself.
    fn read(&mut self, ctx: &egui::Context, backend: &Backend) {
        let language = LANGUAGE.get(ctx);
        let creator = backend.clone();
        self.reading = Some(TrackedPromise::spawn_local(async move {
            menus::read(&creator, language).await
        }));
        let colors = backend.clone();
        self.reading_made = Some(TrackedPromise::spawn_local(async move {
            palette::Made::read(&colors).await
        }));
        let dyed = backend.clone();
        self.reading_dyes = Some(TrackedPromise::spawn_local(async move {
            stains::read(&dyed, language).await
        }));
        let templates = backend.clone();
        self.reading_dye_templates = Some(TrackedPromise::spawn_local(async move {
            mdl::DyeTemplates::read(&templates).await
        }));
        let shaped = backend.files().clone();
        self.reading_deformers = Some(TrackedPromise::spawn_local(async move {
            mdl::Deformers::read(&shaped.read(DEFORMERS).await?)
        }));
        let played = backend.clone();
        self.reading_emotes = Some(TrackedPromise::spawn_local(async move {
            emotes::read(&played, language).await
        }));
        let stood = backend.clone();
        self.reading_mounts = Some(TrackedPromise::spawn_local(async move {
            mounts::read(&stood, language).await
        }));
        let gated = backend.clone();
        self.reading_worn = Some(TrackedPromise::spawn_local(async move {
            gating::Worn::read(&gated).await
        }));
        let standing = backend.clone();
        self.reading_npcs = Some(TrackedPromise::spawn_local(async move {
            npcs::read(&standing, language).await
        }));
        let stanced = backend.clone();
        self.reading_stance = Some(TrackedPromise::spawn_local(async move {
            stance::Stance::read(&stanced, language).await
        }));
    }

    fn poll(&mut self, ctx: &egui::Context, backend: &Backend) {
        if self.listing.is_none() {
            match backend.listing(&api_base(ctx)) {
                Listed::Loading => return,
                // Everything else the install answers is asked for the moment the listing lands,
                // which happens once: a reconnect drops it and asks again.
                Listed::Ready(listing) => {
                    self.listing = Some(listing);
                    self.read(ctx, backend);
                }
                Listed::Failed(why) => {
                    self.model = Some(Err(why.to_string()));
                    return;
                }
            }
        }
        let Some(listing) = self.listing.clone() else {
            return;
        };

        if let Some(promise) = self.reading_npcs.take() {
            match promise.try_take() {
                Ok(Ok(read)) => {
                    self.npcs = read;
                    self.npcs_matched.take();
                }
                Ok(Err(why)) => log::warn!("character: no characters to stand in: {why}"),
                Err(promise) => self.reading_npcs = Some(promise),
            }
        }
        if let Some(promise) = self.reading_stance.take() {
            match promise.try_take() {
                Ok(Ok(read)) => self.stance = Some(Rc::new(read)),
                Ok(Err(why)) => log::warn!("character: nothing states a weapon's stance: {why}"),
                Err(promise) => self.reading_stance = Some(promise),
            }
        }
        if let Some(promise) = self.reading_worn.take() {
            match promise.try_take() {
                Ok(Ok(read)) => self.worn_over = Some(read),
                Ok(Err(why)) => log::warn!("character: nothing gates what is worn: {why}"),
                Err(promise) => self.reading_worn = Some(promise),
            }
        }
        if let Some(promise) = self.reading_emotes.take() {
            match promise.try_take() {
                Ok(Ok(read)) => {
                    let (read, poses) = read;
                    self.emotes = read;
                    self.poses = poses;
                    self.emotes_matched.take();
                }
                Ok(Err(why)) => log::warn!("character: no emotes to play: {why}"),
                Err(promise) => self.reading_emotes = Some(promise),
            }
        }
        if let Some(promise) = self.reading_mounts.take() {
            match promise.try_take() {
                Ok(Ok(read)) => {
                    self.mounts = read;
                    self.mounts_matched.take();
                }
                Ok(Err(why)) => log::warn!("character: no mounts to stand on: {why}"),
                Err(promise) => self.reading_mounts = Some(promise),
            }
        }
        if let Some(promise) = self.reading_made.take() {
            match promise.try_take() {
                Ok(Ok(read)) => self.made = Some(read),
                Ok(Err(why)) => log::warn!("character: no colours to pick from: {why}"),
                Err(promise) => self.reading_made = Some(promise),
            }
        }
        if let Some(promise) = self.reading_dyes.take() {
            match promise.try_take() {
                Ok(Ok(read)) => self.dyes = read,
                Ok(Err(why)) => log::warn!("character: no dyes to pick from: {why}"),
                Err(promise) => self.reading_dyes = Some(promise),
            }
        }
        if let Some(promise) = self.reading_dye_templates.take() {
            match promise.try_take() {
                Ok(Ok(read)) => self.dye_templates = Some(Rc::new(read)),
                Ok(Err(why)) => log::warn!("character: no staining templates to dye with: {why}"),
                Err(promise) => self.reading_dye_templates = Some(promise),
            }
        }
        if let Some(promise) = self.reading_deformers.take() {
            match promise.try_take() {
                Ok(Ok(read)) => self.deformers = Some(Rc::new(read)),
                Ok(Err(why)) => self.model = Some(Err(why.to_string())),
                Err(promise) => self.reading_deformers = Some(promise),
            }
        }
        if let Some(promise) = self.reading.take() {
            match promise.try_take() {
                Ok(Ok(read)) => {
                    self.creator = read;
                    self.stood = false;
                    // Only once the character is dressed: both walk the same sheet, and the one
                    // that gets there first is the one that finishes first.
                    let backend = backend.clone();
                    let language = LANGUAGE.get(ctx);
                    self.reading_pieces = Some(TrackedPromise::spawn_local(async move {
                        menus::pieces(&backend, language).await
                    }));
                }
                Ok(Err(why)) => self.model = Some(Err(why.to_string())),
                Err(promise) => self.reading = Some(promise),
            }
        }
        if let Some(promise) = self.reading_pieces.take() {
            match promise.try_take() {
                Ok(Ok(read)) => {
                    self.creator.pieces = read;
                    // A picker open while they were still arriving matched against nothing.
                    self.matched.take();
                    // Sequenced behind the equipment walk rather than alongside it, for the same
                    // reason `reading_pieces` waits on `reading`: three promises racing the same
                    // sheet would starve whichever lands last.
                    let backend = backend.clone();
                    let language = LANGUAGE.get(ctx);
                    self.reading_weapons = Some(TrackedPromise::spawn_local(async move {
                        weapons::read(&backend, language).await
                    }));
                }
                Ok(Err(why)) => log::warn!("character: nothing to pick equipment from: {why}"),
                Err(promise) => self.reading_pieces = Some(promise),
            }
        }
        if let Some(promise) = self.reading_weapons.take() {
            match promise.try_take() {
                Ok(Ok((main_hand, off_hand, tags))) => {
                    self.weapons_main = main_hand;
                    self.weapons_off = off_hand;
                    self.weapon_tags = tags;
                    self.main_matched.take();
                    self.off_matched.take();
                }
                Ok(Err(why)) => log::warn!("character: nothing to pick a weapon from: {why}"),
                Err(promise) => self.reading_weapons = Some(promise),
            }
        }
        if let Some(promise) = self.reading_atch.take() {
            match promise.try_take() {
                Ok(Ok((code, bytes))) => {
                    log::info!(
                        "character: attach points landed for c{code:04}, {} bytes",
                        bytes.len()
                    );
                    self.atch = Some((code, Rc::new(bytes)));
                }
                Ok(Err(why)) => {
                    log::warn!("character: nothing says where a weapon attaches: {why}");
                }
                Err(promise) => self.reading_atch = Some(promise),
            }
        }

        // Nothing is built until the deformers have landed: what a body wears a piece as, and its
        // own model, are both the tree's to answer.
        let Some(deformers) = self.deformers.clone() else {
            return;
        };

        // Checked every frame rather than folded into `!self.stood`: the code it settles on can
        // change more than once before that flag next goes false, and a fetch spawned for a code
        // already left behind must not be mistaken for the one now on screen.
        if let Some(promise) = self.reading_facial.take() {
            match promise.try_take() {
                Ok(Ok(bytes)) => {
                    log::info!("character: the eye-size table landed, {} bytes", bytes.len());
                    self.facial = Some(Rc::new(bytes));
                }
                Ok(Err(why)) => log::warn!("character: no eye sizes to read: {why}"),
                Err(promise) => self.reading_facial = Some(promise),
            }
        }
        if self.facial.is_none() && self.reading_facial.is_none() {
            let files = backend.files().clone();
            let fetch = async move { anyhow::Ok(files.read(fpeb::PATH).await?) };
            self.reading_facial = Some(TrackedPromise::spawn_local(fetch));
        }

        let stale = self.atch.as_ref().is_none_or(|(held, _)| *held != self.code);
        if self.reading_atch.is_none() && stale {
            let files = backend.files().clone();
            let code = self.code;
            let path = weapons::atch_path(code);
            let fetch = async move { anyhow::Ok((code, files.read(&path).await?)) };
            self.reading_atch = Some(TrackedPromise::spawn_local(fetch));
        }

        if !self.stood {
            self.stood = true;
            // Only some races are built a child, and a code the deformers do not carry names
            // nothing on disk at all.
            let wanted = resolve(self.tribe, self.female, self.child);
            let code = match deformers.knows(wanted) {
                true => wanted,
                false => resolve(self.tribe, self.female, false),
            };
            // Which model a set is worn as is the code's to say, so the answers held for the last
            // one say nothing about this one.
            if code != self.code {
                self.sets.borrow_mut().clear();
                self.shaped.borrow_mut().clear();
            }
            self.code = code;
            self.skin = skin(&listing, &deformers, self.code);
            self.body = body(&listing, &deformers, self.code);
            self.faces = sets(&listing, &self.code, "face");
            self.hairs = sets(&listing, &self.code, "hair");
            self.tails = grown(&listing, self.code);
            // Both name the files the character is built from, so both are read out of what the
            // menus have been left at rather than kept alongside it. Where nothing has been picked
            // that is the creator's own opening choice, which is not the first of either: a
            // Midlander man does not start bald and white-haired.
            for customize in [FACE, HAIRSTYLE] {
                let Some(menu) = self
                    .creator
                    .body(self.tribe, self.female)
                    .and_then(|body| {
                        body.menus.iter().find(|menu| menu.customize == customize)
                    })
                    .cloned()
                else {
                    continue;
                };
                let held = match self.choices.get(&customize) {
                    Some(id) => *id as u16,
                    None => {
                        self.choice_of(&menu, menu.init.min(menu.count.saturating_sub(1)))
                            .id
                    }
                };
                match customize {
                    FACE => self.face = held,
                    _ => self.hair = held,
                }
            }
            self.face = pick(&self.faces, self.face);
            self.hair = pick(&self.hairs, self.hair);
            // The two feature menus are halves of the one byte the game holds them in, so where
            // each opens has to be laid into its own half of it before either menu reads it. Left
            // until the creator has landed: a body it cannot answer for yet would seed nothing,
            // and nothing is what the menus would then be held at.
            let features = self.creator.body(self.tribe, self.female).map(|body| {
                body.menus
                    .iter()
                    .filter(|menu| menu.customize == FEATURES)
                    .fold((0, 0), |(mask, first), menu| {
                        (mask | menu.init << first, first + menu.count)
                    })
                    .0
            });
            if let Some(features) = features {
                self.choices.entry(FEATURES).or_insert(features);
            }
        }

        // Read off the model rather than driven from here: an emote's own timeline is what says
        // whether it wants a prop held, and when.
        self.props = match self.model.as_ref() {
            Some(Ok(model)) => model.wanted_props(),
            _ => Vec::new(),
        };
        let full = self.wearing(&listing, &deformers);
        let wanted: Vec<(String, u16)> = full
            .iter()
            .map(|(path, variant, _)| (path.clone(), *variant))
            .collect();
        // Kept apart from `wanted` and updated every frame the dressed-comparison passes over: a
        // dye pick does not touch the model's own files, and gating it the same way would ask the
        // whole level to rebuild for a color that only the resolve pass reads.
        if !full.is_empty() {
            self.worn_stains = full.iter().map(|(_, _, stains)| *stains).collect();
        }
        if (wanted != self.worn || self.worn_failed) && !wanted.is_empty() {
            self.worn = wanted;
            self.worn_failed = false;
            let missing: Vec<String> = self
                .worn
                .iter()
                .map(|(path, _)| path)
                .filter(|path| !self.held.contains_key(*path))
                .cloned()
                .collect();
            // A worn piece's own imc says which material_id its variant actually draws with, which
            // can differ from the variant itself. Fetched alongside so it is already in hand the
            // first time this piece is dressed, and tolerantly: a missing or unreadable imc just
            // leaves the variant number to stand for its own material_id, same as it did before.
            let missing_imc: Vec<String> = self
                .worn
                .iter()
                .filter(|(_, variant)| *variant != 0)
                .filter_map(|(path, _)| mdl::imc_path(path))
                .filter(|path| !self.held.contains_key(path))
                .collect();
            match missing.is_empty() && missing_imc.is_empty() {
                true => self.dress(),
                false => {
                    let files = backend.files().clone();
                    self.fetching.push(TrackedPromise::spawn_local(async move {
                        let mut read = Vec::with_capacity(missing.len());
                        for path in missing {
                            let bytes = files.read(&path).await?;
                            read.push((path, bytes));
                        }
                        for path in missing_imc {
                            if let Ok(bytes) = files.read(&path).await {
                                read.push((path, bytes));
                            }
                        }
                        Ok(read)
                    }));
                }
            }
        }
        let mut landed = false;
        let mut waiting = Vec::new();
        for promise in std::mem::take(&mut self.fetching) {
            match promise.try_take() {
                Ok(Ok(read)) => {
                    self.held.extend(read);
                    landed = true;
                }
                Ok(Err(why)) => {
                    self.model = Some(Err(why.to_string()));
                    self.worn_failed = true;
                }
                Err(promise) => waiting.push(promise),
            }
        }
        self.fetching = waiting;
        if landed {
            self.dress();
        }

        // Cheap enough to hand over on every frame: it walks the parts of one character and the
        // model keeps what it was already at, so nothing is rebuilt where nothing was picked.
        if let Some(Ok(model)) = &self.model {
            let (customize, hidden, shapes, stature, bust) = self.made();
            model.made(customize, hidden, shapes, stature, bust);
            model.hinged(self.raised());
            // Neither eye bone is animated by anything, so the table is the whole of what sizes
            // them; a body, face or eye shape it says nothing about leaves them at rest.
            let shape = self.choices.get(&fpeb::EYE_SHAPE).copied().unwrap_or_default();
            model.eyed(
                self.facial
                    .as_ref()
                    .and_then(|held| fpeb::scales(held, self.code, self.face, shape as u16))
                    .unwrap_or([1.0; 2]),
            );
            model.seated(self.mount_seat);
            model.dye(self.dye_templates.clone(), self.worn_stains.clone());
            // A weapon does not reach its back until the motion putting it there has run: the
            // pack states no command to move it, so what keeps it in hand is that motion still
            // playing. Drawing is the other way round, and takes it in hand at once.
            let sheathing = model.acting().is_some_and(|name| name == stance::SHEATHE);
            let carried = self.attachments(self.drawn || sheathing);
            model.glowing(self.effects(&carried));
            model.carried(carried, self.drawn);
            if let Some(stance) = self.stance.clone() {
                model.blending(move |from, to| stance.fade(from, to));
            }
            self.stand(model);
        }
    }

    /// Puts the body in the pose its weapons and the stance toggle state, playing the transition
    /// between the two over it. A class the install files no drawn idle for keeps the sheathed
    /// one: `bt_swd_emp` ships the draw and sheathe motions and no idle to hold after them, and
    /// `bt_emp_emp`'s idle pack holds no animation at all, which is bare hands having no drawn
    /// pose to take.
    fn stand(&self, model: &mdl::Rendered) {
        let Some(stance) = &self.stance else {
            return;
        };
        // A body holding a pose of its own keeps it until it is put back on its feet, which is
        // what forgets this and asks for the weapons' own idle again.
        if self.resting().is_some_and(|pose| pose.settle.is_some()) {
            return;
        }
        let held = self.directory();
        let mut stood = self.stood_in.borrow_mut();
        if let Some(stood) = stood
            .as_ref()
            .filter(|stood| stood.held == held && stood.drawn == self.drawn)
        {
            // Named once the pack has landed rather than when it was asked for: a class with no
            // drawn pose of its own settles into the sheathed one, and this is what says so.
            if let Some(name) = model.standing().filter(|_| !stood.told.replace(true)) {
                log::info!("character: {} settled into {name}", stood.held);
            }
            return;
        }

        let mut poses = Vec::new();
        if self.drawn {
            poses.push((stance.pack(self.code, &held, "resident/idle"), stance::DRAWN));
        }
        poses.push((stance.sheathed_pack(self.code), stance::SHEATHED));

        let wanted = poses[0].1;
        let fade = model.standing().map_or(0.0, |from| stance.fade(&from, wanted));
        log::info!("character: {held} asks for {wanted}, blending over {fade:.3}s");
        model.stand(&poses, fade);

        if stood.as_ref().is_some_and(|stood| stood.drawn != self.drawn) {
            let over = match self.drawn {
                true => stance::DRAW,
                false => stance::SHEATHE,
            };
            let packs = vec![stance.pack(self.code, &held, "resident/sub")];
            let fade = stance.fade(stance::DRAWN, over);
            log::info!("character: {over} over it, blending over {fade:.3}s");
            model.act(&packs, over, fade);
        }
        *stood = Some(Stood {
            held,
            drawn: self.drawn,
            told: Cell::new(false),
        });
    }

    /// The class directory the weapons in hand file this body's packs under.
    fn directory(&self) -> String {
        let wielded = self.wielded();
        let set = |hand: usize| wielded.get(hand).map(|weapon| weapon.set);
        match &self.stance {
            Some(stance) => stance.directory(set(0), set(1)),
            None => stance::COMMON.to_owned(),
        }
    }

    /// Where an emote's own key is filed for this body, nearest first: under the class directory
    /// the weapons in hand put it in, which is the only place a battle emote is filed at all, and
    /// then under the one every body shares. Both go through the table saying which body really
    /// holds a pack, since a class that ships none of its own reads another's.
    /// The pose the body rests in: which posture it is in, and where in that posture's own cycle
    /// it stands.
    fn resting(&self) -> Option<&emotes::Pose> {
        self.poses.of(self.posture).get(self.pose)
    }

    /// The pack the body settles into, which is what an emote played out of that pose returns to.
    fn pose_pack(&self) -> Option<String> {
        let key = self.resting()?.settle.as_deref()?;
        Some(self.stance.as_ref()?.pack(self.code, stance::COMMON, key))
    }

    /// Takes the body into the pose its posture and place in that posture's own cycle name, by way
    /// of the motion that leads into it rather than snapping to the pose itself.
    fn sit(&self, model: &mdl::Rendered) {
        let Some(pose) = self.resting() else {
            return;
        };
        let Some(settle) = pose.settle.as_deref() else {
            // The idle a body on its feet rests in is its weapons' to name, so forgetting what it
            // was standing in is what asks for that again.
            self.stood_in.borrow_mut().take();
            return;
        };
        let held = self.pose_pack();
        match pose.start.as_deref() {
            Some(start) => {
                log::info!("character: {start} into {settle}");
                model.play(&self.emote_packs(start), held.as_deref());
            }
            None => {
                log::info!("character: straight into {settle}");
                model.play(&self.emote_packs(settle), None);
            }
        }
    }

    fn emote_packs(&self, key: &str) -> Vec<String> {
        let Some(stance) = &self.stance else {
            return Vec::new();
        };
        let held = self.directory();
        let shared = stance.pack(self.code, stance::COMMON, key);
        let mut found = vec![stance.pack(self.code, &held, key)];
        if found[0] != shared {
            found.push(shared);
        }
        found
    }

    /// Where each wielded weapon hangs this frame: the model it is worn as, the bone it hangs from
    /// and its own placement relative to that bone. Falls back to the plain hand null bone at no
    /// offset where the race's `.atch` file has not landed yet or names this weapon's job nothing.
    fn attachments(&self, drawn: bool) -> Vec<(String, String, Mat4)> {
        let mut found = Vec::new();
        if let Some(main) = self.main_hand.and_then(|at| self.weapons_main.get(at)) {
            let atch = self
                .atch
                .as_ref()
                .filter(|(code, _)| *code == self.code)
                .map(|(_, bytes)| bytes);
            // Logged once a stance, a wielded weapon or whether the atch file has landed actually
            // changes, rather than every frame the pose is recomputed: this is the bone and offset
            // a stance change moves a weapon to.
            let key = (drawn, self.main_hand, self.off_hand, atch.is_some());
            let log = self.logged.get() != key;
            self.logged.set(key);
            let tag = |weapon: &weapons::Weapon| weapons::tag(&self.weapon_tags, weapon.set);
            found.push(self.attach(main.weapon.model(), tag(&main.weapon), true, drawn, atch, log));
            let off = match main.covers_off_hand {
                true => main.off_hand,
                false => self
                    .off_hand
                    .and_then(|at| self.weapons_off.get(at))
                    .map(|piece| piece.weapon),
            };
            if let Some(weapon) = off {
                found.push(self.attach(weapon.model(), tag(&weapon), false, drawn, atch, log));
            }
        }
        // An emote's own prop hangs off the point its model set names, the same table a weapon
        // reads, and always at the drawn placement: it is summoned into a hand rather than worn,
        // so the stance toggle is nothing to it. One that holds a thing in each hand is moved
        // into them by a pack of its own rather than by where it hangs.
        let atch = self
            .atch
            .as_ref()
            .filter(|(code, _)| *code == self.code)
            .map(|(_, bytes)| bytes);
        for (path, _, set) in &self.props {
            let tag = weapons::tag(&self.weapon_tags, *set);
            let mut placed = self.attach(path.clone(), tag, true, true, atch, false);
            // A motion summoning two of one model names the same point for both, and nothing in
            // the command says which hand either goes to, so the second would stack on the first.
            // The point's own states name the bone on the other side, and state no offset at any
            // of them, so moving the second there is the whole of it.
            if found.iter().any(|(_, bone, _)| *bone == placed.1)
                && let Some(other) = tag
                    .zip(atch)
                    .and_then(|(tag, bytes)| weapons::other_hand(bytes, tag, &placed.1))
            {
                log::info!("character: a second {path} hangs from {other} instead");
                placed.1 = other;
            }
            found.push(placed);
        }
        found
    }

    /// The bone each drawn weapon's own effect plays from, for the ones whose `.imc` names one.
    /// The game only plays a weapon's effect in a battle stance, so nothing sheathed carries one.
    fn effects(&self, carried: &[(String, String, Mat4)]) -> Vec<(String, String)> {
        if !self.drawn {
            self.glowed.take();
            return Vec::new();
        }
        let found: Vec<(String, String)> = self
            .wielded()
            .into_iter()
            .filter_map(|weapon| {
                let model = weapon.model();
                let imc = mdl::imc_path(&model).and_then(|path| self.held.get(&path))?;
                let path = weapons::vfx_path(&weapon, imc)?;
                let (_, bone, _) = carried.iter().find(|(held, ..)| *held == model)?;
                Some((path, bone.clone()))
            })
            .collect();
        let named: Vec<String> = found.iter().map(|(path, _)| path.clone()).collect();
        if *self.glowed.borrow() != named {
            for path in &named {
                log::info!("character: a drawn weapon plays {path}");
            }
            *self.glowed.borrow_mut() = named;
        }
        found
    }

    /// One weapon's own placement: the bone its attach point names and the offset, rotation and
    /// scale it takes there, out of the race's `.atch` file at the current stance.
    fn attach(
        &self,
        path: String,
        tag: Option<&str>,
        main: bool,
        drawn: bool,
        atch: Option<&Rc<Vec<u8>>>,
        log: bool,
    ) -> (String, String, Mat4) {
        let stance = if drawn { "drawn" } else { "sheathed" };
        let placed = tag
            .zip(atch)
            .and_then(|(tag, bytes)| weapons::attach(bytes, tag, drawn, !main));
        let Some(placement) = placed else {
            let bone = weapons::fallback_bone(main);
            if log {
                log::info!("character: {path} hangs from {bone} at no named offset, {stance}");
            }
            return (path, bone.to_owned(), Mat4::IDENTITY);
        };
        let local = placement.placement();
        if log {
            log::info!(
                "character: {path} hangs from {} at offset {:?} scale {}, {stance}",
                placement.bone,
                placement.offset,
                placement.scale
            );
        }
        (path, placement.bone, local)
    }

    /// What the creator's menus have been left at, and what the shaders and the model make of it:
    /// the colours to tint with, the parts to leave undrawn and the shape keys to deform by.
    fn made(&self) -> (mdl::Customize, BTreeSet<String>, BTreeSet<String>, f32, Vec3) {
        appearance(
            &self.creator,
            self.made.as_ref(),
            self.tribe,
            self.female,
            &self.choices,
            self.covered(),
            self.paint(),
        )
    }

    /// How far the visor of the hat being worn has been raised, which is nothing at all where the
    /// set states no gimmick or the box is unticked.
    fn raised(&self) -> [f32; 3] {
        let (outfit, _) = self.dressed();
        match (self.visor, outfit[Slot::Head as usize], &self.worn_over) {
            (true, Some(hat), Some(worn)) => worn.visor(hat.set),
            _ => [0.0; 3],
        }
    }

    /// The seams the outfit covers, which draw nothing rather than through what is over them.
    /// Only a piece whose own model is on hand covers anything: where a slot falls back to the
    /// body what would be hidden is the very skin standing in for the piece, and a seam hidden
    /// for a piece still on its way is a hole in the character already on screen.
    fn covered(&self) -> BTreeSet<String> {
        let Some(worn) = &self.worn_over else {
            return BTreeSet::new();
        };
        let (outfit, _) = self.dressed();
        let sets = self.sets.borrow();
        let mut arrived = Outfit::default();
        for slot in Slot::ALL {
            let Some(gear) = outfit[slot as usize] else {
                continue;
            };
            let held = sets
                .get(&(slot.filed(), gear.set))
                .and_then(|found| found[slot as usize].as_ref())
                .is_some_and(|path| self.held.contains_key(path));
            if held {
                arrived[slot as usize] = Some(gear);
            }
        }
        worn.covers(&arrived, self.race)
            .into_iter()
            .map(str::to_owned)
            .collect()
    }

    /// Where a menu has been left, which is where the creator opens it until it is picked from.
    /// Not bounded by the count: a lip colour past the dark half is where the light one starts.
    fn choice(&self, menu: &menus::Menu) -> u32 {
        self.choices
            .get(&menu.customize)
            .copied()
            .unwrap_or(menu.init)
    }

    /// The face paint the creator has been left at, as the number the file tree files the set
    /// under. Nought is the empty box a face wearing none is offered under.
    fn paint(&self) -> Option<u16> {
        let menu = self
            .creator
            .body(self.tribe, self.female)
            .and_then(|body| body.menus.iter().find(|menu| menu.customize == FACE_PAINT))?;
        let held = match self.choices.get(&FACE_PAINT) {
            Some(id) => *id as u16,
            None => {
                self.choice_of(menu, menu.init.min(menu.count.saturating_sub(1)))
                    .id
            }
        };
        (held > 0).then_some(held)
    }

    /// The tail or ear set the creator has been left at, which is one past where the choice sits.
    fn tail(&self) -> u16 {
        let at = self
            .creator
            .body(self.tribe, self.female)
            .and_then(|body| body.menus.iter().find(|menu| menu.customize == TAIL))
            .map_or(0, |menu| self.choice(menu) as u16 + 1);
        pick(&self.tails, at)
    }

    /// What one of the creator's own boxes has been left at, and whether it is ticked.
    fn held(&self, key: u32) -> u32 {
        self.choices.get(&key).copied().unwrap_or(0)
    }

    fn ticked(&self, key: u32) -> bool {
        self.held(key) != 0
    }

    /// The piece picked by hand for a slot, if the list it was picked from is still the one held.
    fn picked(&self, slot: Slot) -> Option<&menus::Piece> {
        self.creator.pieces[slot as usize].get(self.chosen[slot as usize]?)
    }

    /// What the character is dressed in: the attire, then anything picked by hand over it, then
    /// the slots those pieces cover themselves, which draw nothing at all rather than falling back
    /// to the body's own model. A slot picked for is never covered, since a pick is an instruction.
    fn dressed(&self) -> (Outfit, [bool; 11]) {
        let mut outfit = self.outfit();
        let mut hidden = [false; 11];
        for slot in Slot::ALL {
            if let Some(piece) = self.picked(slot) {
                outfit[slot as usize] = Some(piece.gear);
            }
        }
        for slot in Slot::ALL {
            let Some(piece) = self.picked(slot) else {
                continue;
            };
            for (at, covered) in piece.hides.iter().enumerate() {
                if *covered && self.chosen[at].is_none() {
                    outfit[at] = None;
                    hidden[at] = true;
                }
            }
        }
        // A fist weapon's gauntlets are the visible half of it: its own knuckle models are
        // three-index stubs, and the game draws the hands from the item's `ModelSub` over whatever
        // is worn there.
        if let Some(gauntlets) = self.gauntlets() {
            outfit[Slot::Hands as usize] = Some(gauntlets);
        }
        (outfit, hidden)
    }

    /// The gauntlets the wielded fist weapon puts on the hands, where one is wielded at all.
    fn gauntlets(&self) -> Option<Gear> {
        self.main_hand
            .and_then(|at| self.weapons_main.get(at))?
            .gauntlets
    }

    /// The outfit the picked attire dresses the character in.
    fn outfit(&self) -> Outfit {
        match self.attire {
            Attire::Race => self
                .creator
                .attire
                .get(&(self.race, self.female))
                .copied()
                .unwrap_or_default(),
            Attire::Job => self
                .creator
                .jobs
                .get(self.job)
                .map(|job| job.outfit)
                .unwrap_or_default(),
            Attire::Smallclothes => {
                let mut outfit = Outfit::default();
                for slot in Slot::RACIAL {
                    outfit[slot as usize] = Some(SMALLCLOTHES);
                }
                outfit
            }
            Attire::Npc => self
                .npc
                .and_then(|npc| self.npcs.get(npc))
                .map(|npc| npc.outfit)
                .unwrap_or_default(),
        }
    }

    /// Every model the character is drawn from, each with the variant it is worn at. A slot draws
    /// exactly one of them: the equipment worn in it where there is any, and the body's own model
    /// for that slot otherwise. Those two are the very same mesh wherever a race's smallclothes are
    /// its bare skin, which is what drawing both of them showed as z-fighting.
    ///
    /// The face leads, since the first file is what names the skeleton the rest are posed on and a
    /// piece of equipment worn by a race that has no model of its own is filed under another's code.
    fn wearing(
        &self,
        listing: &Listing,
        deformers: &mdl::Deformers,
    ) -> Vec<(String, u16, [Option<u8>; 2])> {
        if self.body.is_empty() {
            return Vec::new();
        }
        let (outfit, hidden) = self.dressed();
        let hair = match (outfit[Slot::Head as usize], &self.worn_over) {
            (Some(hat), Some(worn)) => worn.keeps_hair(hat.set, self.race),
            _ => true,
        };
        let mut found: Vec<_> = held(&self.faces, self.face)
            .into_iter()
            .chain(match hair {
                true => held(&self.hairs, self.hair),
                false => Vec::new(),
            })
            .chain(held(&self.tails, self.tail()))
            .map(|path| (path, 0, [None, None]))
            .collect();
        for slot in Slot::ALL {
            let worn = outfit[slot as usize].and_then(|gear| {
                self.worn_as(listing, deformers, slot.filed(), gear.set)[slot as usize]
                    .clone()
                    .map(|path| (path, gear.variant, self.stains[slot as usize]))
            });
            match worn {
                // An adornment is the only thing its slot ever draws, so a piece worn over it
                // states whether it is there at all rather than what stands in for it.
                Some(_) if slot.adornment() && !self.bared(&outfit, slot) => {}
                Some(part) => found.push(part),
                // Nothing stands in for a bare head: the body ships no model for it, and the face
                // and the hair are what draw one.
                None if hidden[slot as usize] => {}
                None if !self.bared(&outfit, slot) => {}
                None => found.extend(part(&self.body, slot).map(|path| (path, 0, [None, None]))),
            }
        }
        found.extend(
            self.ridden(listing)
                .into_iter()
                .map(|(path, variant)| (path, variant, [None, None])),
        );
        found.extend(
            self.wielded()
                .into_iter()
                .map(|weapon| (weapon.model(), weapon.variant, [None, None])),
        );
        found.extend(
            self.props
                .iter()
                .map(|(path, variant, _)| (path.clone(), *variant, [None, None])),
        );
        found
    }

    /// The weapon in each hand: the picked main hand item, its own off hand where wielding it
    /// leaves nothing to pick there, and otherwise the separately picked off hand item.
    fn wielded(&self) -> Vec<weapons::Weapon> {
        let Some(main) = self.main_hand.and_then(|at| self.weapons_main.get(at)) else {
            return Vec::new();
        };
        let mut found = vec![main.weapon];
        found.extend(match main.covers_off_hand {
            true => main.off_hand,
            false => self
                .off_hand
                .and_then(|at| self.weapons_off.get(at))
                .map(|piece| piece.weapon),
        });
        found
    }

    /// The mount the character is seated on. A mount is a whole body rather than anything worn, so
    /// every model under it is drawn, and it comes after the character: the first file is what
    /// names the skeleton the rest are posed on.
    fn ridden(&self, listing: &Listing) -> Vec<(String, u16)> {
        let Some(mount) = self.mount.and_then(|at| self.mounts.get(at)) else {
            return Vec::new();
        };
        let mut found = listing.under(&mount.under);
        found.retain(|path| path.ends_with(".mdl"));
        found.sort();
        found
            .into_iter()
            .map(|path| (path, mount.variant))
            .collect()
    }

    /// Whether the body's own model for a slot still draws, which is what a piece worn over it
    /// states rather than anything the two meshes could be told apart by: where a race's
    /// smallclothes are its bare skin the two are the very same geometry.
    fn bared(&self, outfit: &Outfit, slot: Slot) -> bool {
        let Some(worn) = &self.worn_over else {
            return true;
        };
        Slot::GEAR.into_iter().all(|over| {
            outfit[over as usize].is_none_or(|gear| worn.shows(over, gear.set, slot, self.race))
        })
    }

    /// The model each slot of a set is worn as under the current code, answered out of the memo
    /// and read off the listing the first time a set is asked about.
    fn worn_as(
        &self,
        listing: &Listing,
        deformers: &mdl::Deformers,
        filed: Filed,
        set: u16,
    ) -> Ref<'_, Models> {
        if !self.sets.borrow().contains_key(&(filed, set)) {
            let found = equipment(listing, deformers, self.code, filed, set);
            self.sets.borrow_mut().insert((filed, set), found);
        }
        Ref::map(self.sets.borrow(), |sets| &sets[&(filed, set)])
    }

    /// Puts what has arrived on screen, keeping the character that is already there where there is
    /// one so a change of clothes neither moves the view nor asks for anything twice.
    fn dress(&mut self) {
        let wielded: Vec<String> = self.wielded().iter().map(weapons::Weapon::model).collect();
        let parts: Vec<_> = self
            .worn
            .iter()
            .filter_map(|(path, variant)| {
                let deform = made_for(path).and_then(|made_for| {
                    self.shaped
                        .borrow_mut()
                        .entry(made_for)
                        .or_insert_with(|| {
                            let deformers = self.deformers.as_ref()?;
                            deformers.between(made_for, self.code).map(Arc::new)
                        })
                        .clone()
                });
                let material = mdl::material::resolve_variant(
                    path,
                    *variant,
                    mdl::imc_path(path)
                        .and_then(|imc| self.held.get(&imc))
                        .map(Vec::as_slice),
                );
                Some(mdl::Source {
                    path: path.clone(),
                    bytes: self.held.get(path)?.clone(),
                    variant: *variant,
                    material,
                    deform,
                    skin: self.skin,
                    rigid: wielded.contains(path)
                        || self.props.iter().any(|(held, ..)| held == path),
                })
            })
            .collect();
        if parts.len() != self.worn.len() {
            return;
        }
        match &mut self.model {
            Some(Ok(model)) => {
                if let Err(why) = model.redress(&parts) {
                    self.model = Some(Err(why.to_string()));
                }
            }
            _ => {
                self.model = Some(
                    mdl::compose(&parts)
                        .map(Box::new)
                        .map_err(|why| why.to_string()),
                )
            }
        }
        if let Some(Ok(model)) = &self.model {
            model.built_on(self.lineage());
        }
    }

    /// Whether the game builds a child of the picked clan at all.
    fn builds_a_child(&self) -> bool {
        self.deformers
            .as_ref()
            .is_some_and(|deformers| deformers.knows(resolve(self.tribe, self.female, true)))
    }

    /// This body and every one it is built on, as the animation directories name them. Few bodies
    /// carry animation of their own: a child's is the one child body's, and a Highlander man's is
    /// the Midlander's.
    fn lineage(&self) -> Vec<String> {
        let Some(deformers) = &self.deformers else {
            return Vec::new();
        };
        deformers
            .lineage(self.code)
            .map(|code| format!("c{code:04}"))
            .collect()
    }

    /// One slot to dress: what is in it, and, while its picker is open, everything the game names
    /// that could be. A piece the code has no model of would draw the body's own part instead of
    /// what was asked for, so it is offered but not pickable; one the game bars this race or
    /// gender from is picked all the same, since only the game bars it and the files do not.
    fn slot_ui(
        &mut self,
        ui: &mut egui::Ui,
        backend: &Backend,
        icons: &IconManager,
        listing: &Listing,
        slot: Slot,
    ) {
        let at = slot as usize;
        let (outfit, hidden) = self.dressed();
        let worn = match (
            outfit[at]
                .and_then(|gear| self.creator.pieces[at].iter().find(|piece| piece.gear == gear)),
            outfit[at],
        ) {
            (Some(piece), _) => piece.name.clone(),
            (None, Some(gear)) => format!("Set {}", gear.set),
            (None, None) => match (hidden[at], slot.adornment()) {
                (true, _) => "Covered".to_owned(),
                (_, true) => "None".to_owned(),
                _ => "Bare".to_owned(),
            },
        };
        let open = self.picking == Some(slot);
        let visored = slot == Slot::Head
            && outfit[at].is_some_and(|gear| {
                self.worn_over
                    .as_ref()
                    .is_some_and(|worn| worn.visored(gear.set))
            });
        // Offered on every slot rather than only where a worn piece's material states a dye row:
        // that is not known until the material has been fetched, and swatches that appear once it
        // has would move everything under it. Facewear carries no dye row at all, on any material
        // any facewear item ships, so it gets no swatches to begin with.
        let dyeable = slot != Slot::Facewear;
        let reserve = match dyeable {
            true => 2.0 * SWATCH + 2.0 * ui.spacing().item_spacing.x,
            false => 0.0,
        };
        let mut clicked = false;
        ui.horizontal(|ui| {
            let button = egui::Button::selectable(open, format!("{}: {worn}", slot.name()))
                .truncate()
                .min_size(egui::vec2((ui.available_width() - reserve).max(0.0), 0.0));
            clicked = ui.add(button).clicked();
            if dyeable {
                for channel in 0..2u8 {
                    self.dye_swatch(ui, slot, channel);
                }
            }
        });
        if clicked {
            self.picking = (!open).then_some(slot);
            if let Some(slot) = self.picking {
                log::info!("character: picking {}", slot.name());
            }
        }
        if visored {
            ui.checkbox(&mut self.visor, "Visor");
        }
        if !open {
            return;
        }
        ui.horizontal(|ui| {
            ui.add(
                TextEdit::singleline(&mut self.search[at])
                    .hint_text("Search")
                    .desired_width(ui.available_width() - 60.0),
            );
            if ui
                .add_enabled(self.chosen[at].is_some(), egui::Button::new("Attire"))
                .on_hover_text("Wear what the attire puts here")
                .clicked()
            {
                self.chosen[at] = None;
            }
        });

        let mut picked = None;
        {
            let deformers = self.deformers.clone();
            let query = self.search[at].clone();
            let matched = self.matches(slot, &query);
            let step = PIECE + 2.0 * ui.spacing().button_padding.y + ui.spacing().item_spacing.y;
            ScrollArea::vertical()
                .id_salt(("character_pieces", at))
                .max_height(step * SHOWN as f32)
                .show_rows(ui, step, matched.len(), |ui, rows| {
                    for row in rows {
                        let index = matched[row];
                        let piece = &self.creator.pieces[at][index];
                        let held = deformers.as_ref().is_none_or(|deformers| {
                            self.worn_as(listing, deformers, slot.filed(), piece.gear.set)[at]
                                .is_some()
                        });
                        let suits = piece.suits(self.race, self.female);
                        let name = match suits {
                            true => RichText::new(&piece.name),
                            false => RichText::new(&piece.name).color(Color32::KHAKI),
                        };
                        let icon = get_icon_path(backend.icons(), piece.icon, false, Language::None);
                        let excel = backend.excel().clone();
                        let source = icons.get_or_insert_icon(&icon, ui.ctx(), || {
                            let excel = excel.clone();
                            let icon = icon.clone();
                            TrackedPromise::spawn_local(async move { excel.get_icon(&icon).await })
                        });
                        let loaded = match &source {
                            ManagedIcon::Loaded(source) => Some(source.clone()),
                            _ => None,
                        };
                        let button = match source {
                            ManagedIcon::Loaded(source) => egui::Button::image_and_text(
                                egui::Image::new(source)
                                    .maintain_aspect_ratio(true)
                                    .fit_to_exact_size(egui::Vec2::splat(PIECE)),
                                name,
                            ),
                            _ => egui::Button::new(name),
                        };
                        // One line to a row, since the rows are scrolled by a fixed step and a name
                        // long enough to wrap would walk the list out from under it.
                        let response = ui.add_enabled(
                            held,
                            button
                                .truncate()
                                .selected(self.chosen[at] == Some(index))
                                .min_size(egui::vec2(ui.available_width(), PIECE)),
                        );
                        let response = match (held, suits) {
                            (false, _) => {
                                response.on_disabled_hover_text("This body has no model of it")
                            }
                            (_, false) => response
                                .on_hover_text("The game does not offer this to this race and gender"),
                            _ => response,
                        };
                        icon_context_menu(
                            &response,
                            icons,
                            excel,
                            backend.files().clone(),
                            piece.icon,
                            &icon,
                            loaded,
                        );
                        if response.clicked() {
                            picked = Some(index);
                        }
                    }
                });
        }
        if let Some(index) = picked {
            log::info!(
                "character: chose {} for {}",
                self.creator.pieces[at][index].name,
                slot.name()
            );
            self.chosen[at] = Some(index);
        }
    }

    /// One slot's swatch for one of the two channels a modern item can carry, and the popup it
    /// opens onto every dye the game names. Picking one dyes only the fields a worn piece's own
    /// material states, so a stain with no dyeable row to land in changes nothing.
    fn dye_swatch(&mut self, ui: &mut egui::Ui, slot: Slot, channel: u8) {
        let at = slot as usize;
        let current = self.stains[at][usize::from(channel)];
        let dye = current.and_then(|id| self.dyes.iter().find(|dye| dye.id == id));
        let (rect, response) =
            ui.allocate_exact_size(egui::Vec2::splat(SWATCH), egui::Sense::click());
        stains::paint(
            ui.painter(),
            rect,
            dye.map_or(Color32::TRANSPARENT, |dye| dye.color),
            dye.is_some_and(|dye| dye.metallic),
        );
        ui.painter().rect_stroke(
            rect,
            2.0,
            ui.visuals().widgets.inactive.fg_stroke,
            egui::StrokeKind::Inside,
        );
        let response = response.on_hover_text(dye.map_or("No dye", |dye| dye.name.as_str()));
        let mut picked = None;
        const DYE_GAP: f32 = 1.0;
        let popup_id = Popup::default_response_id(&response);
        Popup::from_toggle_button_response(&response)
            .close_behavior(PopupCloseBehavior::CloseOnClickOutside)
            .align(RectAlign::BOTTOM_START)
            .show(|ui| {
                ui.spacing_mut().item_spacing = egui::Vec2::splat(DYE_GAP);
                ScrollArea::vertical().max_height(10.0 * (SWATCH + DYE_GAP)).show(ui, |ui| {
                    let mut cell = |ui: &mut egui::Ui, color, metallic, name: &str, hit: Option<u8>| {
                        let (rect, response) = ui.allocate_exact_size(
                            egui::Vec2::splat(SWATCH),
                            egui::Sense::click(),
                        );
                        stains::paint(ui.painter(), rect, color, metallic);
                        if current == hit {
                            ui.painter().rect_stroke(
                                rect,
                                2.0,
                                ui.visuals().selection.stroke,
                                egui::StrokeKind::Inside,
                            );
                        }
                        if response.on_hover_text(name).clicked() {
                            picked = Some(hit);
                        }
                    };
                    // Each shelf its own row, exactly as wide as its own swatches: a `Grid` would
                    // pad every row to the widest shelf's column count instead.
                    ui.vertical(|ui| {
                        ui.horizontal(|ui| cell(ui, Color32::TRANSPARENT, false, "No dye", None));
                        for shelf in self.dyes.chunk_by(|left, right| left.shade == right.shade) {
                            ui.horizontal(|ui| {
                                for dye in shelf {
                                    cell(ui, dye.color, dye.metallic, &dye.name, Some(dye.id));
                                }
                            });
                        }
                    });
                });
            });
        if let Some(hit) = picked {
            self.stains[at][usize::from(channel)] = hit;
            Popup::close_id(ui.ctx(), popup_id);
        }
    }

    /// Everything the creator offers this body, in its own order and under its own names. A face
    /// and a hairstyle name the files the character is built from, so those are kept where the
    /// rest of the choices are not.
    fn appearance(
        &mut self,
        ui: &mut egui::Ui,
        backend: &Backend,
        icons: &IconManager,
    ) -> Option<Pick> {
        let body = self.creator.body(self.tribe, self.female).cloned()?;
        let palettes = self
            .made
            .as_ref()
            .map(|made| made.palettes(self.tribe, self.female));
        // Which face is being worn, since the icons a facial feature is offered under are the
        // face's own and are held by where it sits in the menu rather than by its number.
        let face = body
            .menus
            .iter()
            .find(|menu| menu.customize == FACE)
            .and_then(|menu| {
                (0..menu.count).position(|index| self.choice_of(menu, index).id == self.face)
            })
            .unwrap_or(0);
        let mut picked = None;
        for (at, menu) in body.menus.iter().enumerate() {
            ui.add_space(8.0);
            let name = RichText::new(&menu.name).strong();
            // A colour the character is not wearing has nothing to pick: what puts it on is the
            // box beside it, or the paint the colour is for.
            let worn = match menu.customize {
                LIP_COLOR => self.ticked(LIPSTICK),
                FACE_PAINT_COLOR => self.paint().is_some(),
                TATTOO_COLOR => self.held(FEATURES) & LEGACY_TATTOO != 0,
                _ => true,
            };
            // Lip colour keeps its heading unworn: the "Lipstick" checkbox under it is its own
            // affordance to turn it on. Face paint and tattoo colour have no such box, so their
            // heading goes with the grid rather than standing bare over nothing.
            let headed = worn || menu.customize == LIP_COLOR;
            if menu.kind != menus::Kind::Slider && headed {
                ui.label(name.clone());
            }
            let current = self.choice(menu);
            match menu.kind {
                menus::Kind::Slider => {
                    let [low, high] = menu.range;
                    let mut held = current.clamp(low, high);
                    ui.horizontal(|ui| {
                        if ui.add(egui::Slider::new(&mut held, low..=high)).changed() {
                            picked = Some(Pick::Made(menu.customize, held));
                        }
                        ui.label(name);
                    });
                }
                menus::Kind::List => {
                    ui.horizontal_wrapped(|ui| {
                        for (index, label) in menu.labels.iter().enumerate() {
                            if ui.selectable_label(current as usize == index, label).clicked() {
                                picked = Some(Pick::Made(menu.customize, index as u32));
                            }
                        }
                    });
                }
                menus::Kind::Checks => {
                    // Both menus that drive the features are halves of one run of parts, and the
                    // icons are that whole run: this one's start in it is what precedes it.
                    let first = body
                        .menus
                        .iter()
                        .take_while(|held| !std::ptr::eq(*held, menu))
                        .filter(|held| held.customize == FEATURES)
                        .map(|held| held.count as usize)
                        .sum::<usize>();
                    let shown = body.features.get(face);
                    let bits: Vec<Choice> = (0..menu.count)
                        .map(|bit| Choice {
                            at: first as u32 + bit,
                            id: bit as u16 + 1,
                            icon: shown
                                .and_then(|icons| icons.get(first + bit as usize))
                                .copied()
                                .filter(|icon| *icon > 0),
                        })
                        .collect();
                    grid(ui, &format!("character_checks_{at}"), &bits, |ui, held| {
                        let on = current & 1 << held.at != 0;
                        chip(ui, backend, icons, held, on)
                            .then_some(Pick::Made(menu.customize, current ^ 1 << held.at))
                    })
                    .inspect(|choice| picked = Some(*choice));
                }
                menus::Kind::Color | menus::Kind::DoubleColor => {
                    let Some(palettes) = &palettes else {
                        ui.spinner();
                        continue;
                    };
                    let swatches = match menu.customize {
                        SKIN_COLOR => &palettes.skin,
                        HAIR_COLOR => &palettes.hair,
                        EYE_COLOR => &palettes.eyes,
                        LIP_COLOR => &palettes.lips,
                        FACE_PAINT_COLOR => &palettes.face_paint,
                        _ => &palettes.features,
                    };
                    if menu.customize == LIP_COLOR {
                        let mut on = self.ticked(LIPSTICK);
                        if ui.checkbox(&mut on, "Lipstick").changed() {
                            picked = Some(Pick::Made(LIPSTICK, u32::from(on)));
                        }
                    }
                    // The half a colour belongs to is the top bit of its own index, so switching
                    // halves is that bit and nothing else.
                    let mut half = 0;
                    if worn && matches!(menu.customize, LIP_COLOR | FACE_PAINT_COLOR) {
                        half = current / HALF;
                        ui.horizontal(|ui| {
                            for (at, name) in [(0, "Dark"), (1, "Light")] {
                                if ui.selectable_label(half == at, name).clicked() {
                                    picked = Some(Pick::Made(
                                        menu.customize,
                                        current % HALF + at * HALF,
                                    ));
                                }
                            }
                        });
                    }
                    // A second colour the creator only offers once its own box is ticked: a strand
                    // is mixed between two hair colours, and an eye takes one each. Shown beside
                    // the first where there is room for both rather than under it.
                    let paired = match menu.customize {
                        HAIR_COLOR => Some((HIGHLIGHTS, HIGHLIGHT_COLOR, "Highlights", &palettes.highlights)),
                        EYE_COLOR => Some((ODD_EYES, LEFT_EYE_COLOR, "Odd eyes", &palettes.eyes)),
                        _ => None,
                    };
                    let grid_width = palette::COLUMNS as f32 * (SWATCH + 2.0);
                    let side_by_side = worn
                        && paired.is_some()
                        && ui.available_width() >= 2.0 * grid_width + ui.spacing().item_spacing.x;
                    if side_by_side {
                        let (box_of, color, name, second) = paired.unwrap();
                        let mut on = self.ticked(box_of);
                        let offered = half * HALF..half * HALF + menu.count;
                        // A fixed height for both headers, so the left grid (no header of its own,
                        // the menu's name already stands above both columns) starts on the same
                        // line as the right one, whose header is this checkbox.
                        let header = ui.spacing().interact_size.y;
                        ui.columns(2, |columns| {
                            // Nothing to draw here, only a blank row the same height as the
                            // checkbox beside it.
                            columns[0].horizontal(|ui| ui.set_min_height(header));
                            if let Some(index) = colors(
                                &mut columns[0],
                                ("character_colors", at),
                                swatches,
                                offered,
                                current,
                            ) {
                                picked = Some(Pick::Made(menu.customize, index));
                            }
                            columns[1].horizontal(|ui| {
                                ui.set_min_height(header);
                                if ui.checkbox(&mut on, name).changed() {
                                    picked = Some(Pick::Made(box_of, u32::from(on)));
                                }
                            });
                            if on && let Some(index) = colors(
                                &mut columns[1],
                                ("character_second", at),
                                second,
                                0..menu.count,
                                self.held(color),
                            ) {
                                picked = Some(Pick::Made(color, index));
                            }
                        });
                    } else {
                        if worn {
                            let offered = half * HALF..half * HALF + menu.count;
                            if let Some(index) =
                                colors(ui, ("character_colors", at), swatches, offered, current)
                            {
                                picked = Some(Pick::Made(menu.customize, index));
                            }
                        }
                        if let Some((box_of, color, name, second)) = paired {
                            let mut on = self.ticked(box_of);
                            if ui.checkbox(&mut on, name).changed() {
                                picked = Some(Pick::Made(box_of, u32::from(on)));
                            }
                            if on {
                                let held = self.held(color);
                                let offered = 0..menu.count;
                                if let Some(index) =
                                    colors(ui, ("character_second", at), second, offered, held)
                                {
                                    picked = Some(Pick::Made(color, index));
                                }
                            }
                        }
                    }
                }
                // What a choice is worth is the number the file tree files it under and not where
                // it sits: an NPC states a face of 216, which is a face on disk that no menu lists.
                menus::Kind::Icons => {
                    let choices: Vec<Choice> = (0..menu.count)
                        .map(|index| self.choice_of(menu, index))
                        .collect();
                    ScrollArea::vertical()
                        .id_salt(("character_menu_scroll", menu.customize))
                        .max_height((ICON + GAP) * ICON_ROWS as f32)
                        .show(ui, |ui| {
                            grid(ui, &format!("character_menu_{}", menu.customize), &choices, |ui, held| {
                                chip(ui, backend, icons, held, u32::from(held.id) == current)
                                    .then_some(Pick::Choice(menu.customize, held.id))
                            })
                        })
                        .inner
                        .inspect(|choice| picked = Some(*choice));
                }
            }
        }
        picked
    }

    /// The game's own characters, searched by name. Picking one is picking everything at once: it
    /// carries the whole of what the creator would have been left at, plus what it is wearing.
    fn npcs_ui(&mut self, ui: &mut egui::Ui) -> Option<Pick> {
        if self.reading_npcs.is_some() {
            ui.spinner();
        }
        ui.add(
            TextEdit::singleline(&mut self.npc_search)
                .hint_text("Search")
                .desired_width(f32::INFINITY),
        );
        let mut picked = None;
        let query = self.npc_search.clone();
        let matched = self.npcs_matching(&query);
        let row = ui.text_style_height(&egui::TextStyle::Body) + ui.spacing().button_padding.y * 2.0;
        let step = row + ui.spacing().item_spacing.y;
        ScrollArea::vertical()
            .id_salt("character_npcs")
            .max_height(step * SHOWN as f32)
            .show_rows(ui, step, matched.len(), |ui, rows| {
                for at in rows {
                    let index = matched[at];
                    let name = &self.npcs[index].name;
                    let button = egui::Button::selectable(self.npc == Some(index), name.as_str())
                        .truncate()
                        .min_size(egui::vec2(ui.available_width(), row));
                    if ui.add(button).on_hover_text(name).clicked() {
                        picked = Some(Pick::Npc(index));
                    }
                }
            });
        picked
    }

    /// Which characters a search names, kept the way a slot's own list is.
    fn npcs_matching(&self, query: &str) -> Ref<'_, Vec<usize>> {
        if self.npcs_matched.borrow().0.as_deref() != Some(query) {
            let found = self.matcher.match_list_indirect(
                (!query.is_empty()).then_some(query),
                self.npcs
                    .iter()
                    .enumerate()
                    .map(|(index, npc)| (index, npc.name.as_str())),
                |npc| npc.1,
            );
            *self.npcs_matched.borrow_mut() = (
                Some(query.to_owned()),
                found.into_iter().map(|(index, _)| index).collect(),
            );
        }
        Ref::map(self.npcs_matched.borrow(), |(_, rows)| rows)
    }

    /// The emotes the game names, searched by name and drawn under their own icons. Standing and
    /// its unique variant are here rather than in a control of their own: both are idles, so both
    /// are looked up exactly as an emote is.
    fn emotes_ui(
        &mut self,
        ui: &mut egui::Ui,
        backend: &Backend,
        icons: &IconManager,
    ) -> Option<Pick> {
        if self.reading_emotes.is_some() {
            ui.spinner();
        }
        ui.add(
            TextEdit::singleline(&mut self.emote_search)
                .hint_text("Search")
                .desired_width(f32::INFINITY),
        );
        let query = self.emote_search.clone();
        let matched = self.emotes_matching(&query);
        listed(
            ui,
            backend,
            icons,
            "character_emotes",
            &matched,
            self.emote,
            // A rider can only play what the sheet names a partial for, since the mount holds its
            // lower half; an emote that is nothing but a face is always its own to make.
            |index| {
                let emote = &self.emotes[index];
                let playable = self.mount.is_none()
                    || emote.mounted().is_some()
                    || emote.expression().is_some();
                (emote.name.as_str(), emote.icon, playable)
            },
        )
        .map(Pick::Emote)
    }

    /// The mounts the game names, one of which the character rides. A mount is a body of its own
    /// and names the same bones a rider does, so the two are posed apart and the rider is carried
    /// to whichever seat its own skeleton names is picked, for the ones that seat more than one.
    fn mounts_ui(
        &mut self,
        ui: &mut egui::Ui,
        backend: &Backend,
        icons: &IconManager,
    ) -> Option<Pick> {
        if self.reading_mounts.is_some() {
            ui.spinner();
        }
        ui.add(
            TextEdit::singleline(&mut self.mount_search)
                .hint_text("Search")
                .desired_width(f32::INFINITY),
        );
        let query = self.mount_search.clone();
        let matched = self.mounts_matching(&query);
        let picked = listed(
            ui,
            backend,
            icons,
            "character_mounts",
            &matched,
            self.mount,
            |index| {
                let mount = &self.mounts[index];
                (mount.name.as_str(), mount.icon, true)
            },
        )
        .map(|index| Pick::Mount((self.mount != Some(index)).then_some(index)));
        if picked.is_none()
            && let Some(mount) = self.mount.and_then(|at| self.mounts.get(at))
            && mount.extra_seats > 0
        {
            return self.seat_ui(ui, mount.extra_seats);
        }
        picked
    }

    /// Which of a mount's own seats the character rides in, for one seating more than one. Seat
    /// zero is the one the mount's skeleton names first, whatever the game calls it in its own UI.
    /// Where the body sits and which of that seat's own poses it holds, which is what `/cpose`
    /// steps through. A rider reads none of this: a mount states the seat it holds one in.
    fn posture_ui(&self, ui: &mut egui::Ui) -> Option<Pick> {
        if self.mount.is_some() {
            return None;
        }
        let mut picked = None;
        ui.add_space(8.0);
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new("Posture").strong());
            for held in emotes::Posture::ALL {
                if held != emotes::Posture::Standing && self.poses.of(held).is_empty() {
                    continue;
                }
                if ui
                    .selectable_label(self.posture == held, held.name())
                    .clicked()
                {
                    picked = Some(Pick::Posture(held));
                }
            }
        });
        let poses = self.poses.of(self.posture);
        if poses.len() > 1 {
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new("Pose").strong());
                for at in 0..poses.len() {
                    if ui
                        .selectable_label(self.pose == at, (at + 1).to_string())
                        .clicked()
                    {
                        picked = Some(Pick::Pose(at));
                    }
                }
            });
        }
        picked
    }

    /// Takes the body into or out of a seat. Sitting plays the pose it settles into; standing back
    /// up forgets what it was standing in, so the idle its weapons state names is asked for again.
    fn seated_changed(&self) {
        if let Some(Ok(model)) = &self.model {
            self.sit(model);
        }
    }

    fn seat_ui(&self, ui: &mut egui::Ui, extra_seats: u8) -> Option<Pick> {
        let mut picked = None;
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new("Seat").strong());
            for seat in 0..=usize::from(extra_seats) {
                if ui
                    .selectable_label(self.mount_seat == seat, (seat + 1).to_string())
                    .clicked()
                {
                    picked = Some(Pick::Seat(seat));
                }
            }
        });
        picked
    }

    /// Which mounts a search names, kept the way a slot's own list is.
    fn mounts_matching(&self, query: &str) -> Ref<'_, Vec<usize>> {
        if self.mounts_matched.borrow().0.as_deref() != Some(query) {
            let found = self.matcher.match_list_indirect(
                (!query.is_empty()).then_some(query),
                self.mounts
                    .iter()
                    .enumerate()
                    .map(|(index, mount)| (index, mount.name.as_str())),
                |mount| mount.1,
            );
            *self.mounts_matched.borrow_mut() = (
                Some(query.to_owned()),
                found.into_iter().map(|(index, _)| index).collect(),
            );
        }
        Ref::map(self.mounts_matched.borrow(), |(_, rows)| rows)
    }

    /// Which emotes a search names, kept the way a slot's own list is.
    fn emotes_matching(&self, query: &str) -> Ref<'_, Vec<usize>> {
        if self.emotes_matched.borrow().0.as_deref() != Some(query) {
            let found = self.matcher.match_list_indirect(
                (!query.is_empty()).then_some(query),
                self.emotes
                    .iter()
                    .enumerate()
                    .map(|(index, emote)| (index, emote.name.as_str())),
                |emote| emote.1,
            );
            *self.emotes_matched.borrow_mut() = (
                Some(query.to_owned()),
                found.into_iter().map(|(index, _)| index).collect(),
            );
        }
        Ref::map(self.emotes_matched.borrow(), |(_, rows)| rows)
    }

    /// The weapon in each hand, searched by name, and the stance it is held in. An off hand a
    /// wielded weapon covers itself is left unoffered: there is nothing to pick there.
    fn weapons_ui(
        &mut self,
        ui: &mut egui::Ui,
        backend: &Backend,
        icons: &IconManager,
    ) -> Option<Pick> {
        let mut picked = None;
        if self.reading_weapons.is_some() {
            ui.spinner();
        }
        ui.horizontal_wrapped(|ui| {
            for (drawn, name) in [(false, "Sheathed"), (true, "Drawn")] {
                if ui.selectable_label(self.drawn == drawn, name).clicked() {
                    picked = Some(Pick::Stance(drawn));
                }
            }
        });
        ui.add(
            TextEdit::singleline(&mut self.main_search)
                .hint_text("Search")
                .desired_width(f32::INFINITY),
        );
        {
            let query = self.main_search.clone();
            let matched = self.main_hand_matching(&query);
            if let Some(index) = listed(
                ui,
                backend,
                icons,
                "character_weapons_main",
                &matched,
                self.main_hand,
                |index| {
                    let piece = &self.weapons_main[index];
                    (piece.name.as_str(), piece.icon, true)
                },
            ) {
                picked = Some(Pick::Weapon((self.main_hand != Some(index)).then_some(index)));
            }
        }

        let covers_off_hand = self
            .main_hand
            .and_then(|at| self.weapons_main.get(at))
            .is_some_and(|piece| piece.covers_off_hand);
        if !covers_off_hand {
            ui.add_space(4.0);
            ui.label("Off hand");
            ui.add(
                TextEdit::singleline(&mut self.off_search)
                    .hint_text("Search")
                    .desired_width(f32::INFINITY),
            );
            let query = self.off_search.clone();
            let matched = self.off_hand_matching(&query);
            if let Some(index) = listed(
                ui,
                backend,
                icons,
                "character_weapons_off",
                &matched,
                self.off_hand,
                |index| {
                    let piece = &self.weapons_off[index];
                    (piece.name.as_str(), piece.icon, true)
                },
            ) {
                picked = Some(Pick::OffHand((self.off_hand != Some(index)).then_some(index)));
            }
        }
        picked
    }

    /// Which main hand weapons a search names, kept the way a slot's own list is.
    fn main_hand_matching(&self, query: &str) -> Ref<'_, Vec<usize>> {
        if self.main_matched.borrow().0.as_deref() != Some(query) {
            let found = self.matcher.match_list_indirect(
                (!query.is_empty()).then_some(query),
                self.weapons_main
                    .iter()
                    .enumerate()
                    .map(|(index, piece)| (index, piece.name.as_str())),
                |piece| piece.1,
            );
            *self.main_matched.borrow_mut() = (
                Some(query.to_owned()),
                found.into_iter().map(|(index, _)| index).collect(),
            );
        }
        Ref::map(self.main_matched.borrow(), |(_, rows)| rows)
    }

    /// Which off hand weapons a search names, kept the way a slot's own list is.
    fn off_hand_matching(&self, query: &str) -> Ref<'_, Vec<usize>> {
        if self.off_matched.borrow().0.as_deref() != Some(query) {
            let found = self.matcher.match_list_indirect(
                (!query.is_empty()).then_some(query),
                self.weapons_off
                    .iter()
                    .enumerate()
                    .map(|(index, piece)| (index, piece.name.as_str())),
                |piece| piece.1,
            );
            *self.off_matched.borrow_mut() = (
                Some(query.to_owned()),
                found.into_iter().map(|(index, _)| index).collect(),
            );
        }
        Ref::map(self.off_matched.borrow(), |(_, rows)| rows)
    }

    /// What one choice of a menu is: the number the file tree uses for it, and the icon it is
    /// offered under. A menu either names icons outright or names rows that carry one.
    fn choice_of(&self, menu: &menus::Menu, index: u32) -> Choice {
        let param = menu.params.get(index as usize).copied().unwrap_or(0);
        // The icons a tail or a pair of ears is offered under end at 91 to 94 whatever the body,
        // so the choice is where it sits and the set it names is one past that.
        if menu.customize == TAIL {
            return Choice {
                at: index,
                id: index as u16,
                icon: (param > 0).then_some(param as u32),
            };
        }
        // A row's icon stands whatever it is: nought is the empty box wearing no face paint is
        // offered under, not a choice the creator left undrawn. One no row holds falls back to
        // naming the icon outright, and to its own number where it names nothing.
        match self.creator.offered.get(&(param.max(0) as u32)) {
            Some((id, icon)) => Choice {
                at: index,
                id: *id,
                icon: Some(*icon),
            },
            None => Choice {
                at: index,
                id: menus::face(param).unwrap_or(index as u16 + 1),
                icon: (param > 0).then_some(param as u32),
            },
        }
    }

    /// Which of a slot's pieces its search names, kept since matching every name again on every
    /// frame costs more than reading them all did.
    fn matches(&self, slot: Slot, query: &str) -> Ref<'_, Vec<usize>> {
        let at = slot as usize;
        if self.matched.borrow()[at].0.as_deref() != Some(query) {
            let found = self.matcher.match_list_indirect(
                (!query.is_empty()).then_some(query),
                self.creator.pieces[at]
                    .iter()
                    .enumerate()
                    .map(|(index, piece)| (index, piece.name.as_str())),
                |piece| piece.1,
            );
            self.matched.borrow_mut()[at] = (
                Some(query.to_owned()),
                found.into_iter().map(|(index, _)| index).collect(),
            );
        }
        Ref::map(self.matched.borrow(), |matched| &matched[at].1)
    }

    fn side_panel(&mut self, ui: &mut egui::Ui, backend: &Backend, icons: &IconManager) {
        let listing = self.listing.clone();
        let picked = CollapsibleSidePanel::new("character_pick", Side::Left)
            .min_width(PANEL_MIN_WIDTH)
            .max_width(PANEL_WIDTH)
            .show(ui, |ui, is_open| {
                let mut picked = None;
                if !is_open {
                    return picked;
                }
                Panel::top("character_header").show(ui, |ui| {
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.with_layout(Layout::right_to_left(Align::Min), |ui| {
                            CollapsibleSidePanel::draw_arrow(ui, "character_pick", Side::Left);
                            ui.vertical_centered_justified(|ui| ui.heading("Character"));
                        });
                    });
                    ui.add_space(4.0);
                });
                ScrollArea::vertical().show(ui, |ui| {
                    ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Truncate);
                    ui.label(RichText::new("Race").strong());
                    for race in self.creator.races.keys() {
                        if !self.creator.bodies.iter().any(|body| body.race == *race) {
                            continue;
                        }
                        let name = menus::Creator::named(&self.creator.races, *race, self.female);
                        if ui.selectable_label(self.race == *race, name).clicked() {
                            picked = Some(Pick::Race(*race));
                        }
                    }
                    ui.add_space(8.0);
                    ui.label(RichText::new("Clan").strong());
                    for body in &self.creator.bodies {
                        if body.race != self.race || body.female != self.female {
                            continue;
                        }
                        let name =
                            menus::Creator::named(&self.creator.tribes, body.tribe, self.female);
                        if ui
                            .selectable_label(self.tribe == body.tribe, name)
                            .clicked()
                        {
                            picked = Some(Pick::Tribe(body.tribe));
                        }
                    }
                    ui.add_space(8.0);
                    ui.horizontal_wrapped(|ui| {
                        for (female, name) in [(false, "Male"), (true, "Female")] {
                            if ui.selectable_label(self.female == female, name).clicked() {
                                picked = Some(Pick::Gender(female));
                            }
                        }
                        // Only some races are built a child, and the rest would draw the adult
                        // under a lit button.
                        if self.builds_a_child()
                            && ui.selectable_label(self.child, "Child").clicked()
                        {
                            picked = Some(Pick::Child(!self.child));
                        }
                    });
                    ui.add_space(8.0);
                    ui.label(RichText::new("Attire").strong());
                    ui.horizontal_wrapped(|ui| {
                        for (attire, name) in [
                            (Attire::Race, "Race"),
                            (Attire::Job, "Job"),
                            (Attire::Smallclothes, "Smallclothes"),
                            (Attire::Npc, "Theirs"),
                        ] {
                            if ui.selectable_label(self.attire == attire, name).clicked() {
                                picked = Some(Pick::Attire(attire));
                            }
                        }
                    });
                    if self.attire == Attire::Job {
                        for (at, job) in self.creator.jobs.iter().enumerate() {
                            if ui.selectable_label(self.job == at, &job.name).clicked() {
                                picked = Some(Pick::Job(at));
                            }
                        }
                    }
                    if let Some(listing) = &listing {
                        ui.add_space(8.0);
                        section(ui, "Equipment", |ui| {
                            if self.reading_pieces.is_some() {
                                ui.spinner();
                            }
                            for slot in Slot::GEAR {
                                self.slot_ui(ui, backend, icons, listing, slot);
                            }
                            None
                        });
                        ui.add_space(8.0);
                        section(ui, "Accessories", |ui| {
                            for slot in Slot::ADORNMENT {
                                self.slot_ui(ui, backend, icons, listing, slot);
                            }
                            None
                        });
                    }
                    // Beside the weapon's own sheathed and drawn, since both say what the body is
                    // doing rather than what it is wearing, and neither wants to sit under a list.
                    self.posture_ui(ui)
                        .inspect(|posture| picked = Some(*posture));
                    ui.add_space(8.0);
                    section(ui, "Weapon", |ui| self.weapons_ui(ui, backend, icons))
                        .inspect(|pick| picked = Some(*pick));
                    ui.add_space(8.0);
                    section(ui, "Customization", |ui| self.appearance(ui, backend, icons))
                        .inspect(|made| picked = Some(*made));
                    ui.add_space(8.0);
                    section(ui, "Emote", |ui| self.emotes_ui(ui, backend, icons))
                        .inspect(|emote| picked = Some(*emote));
                    ui.add_space(8.0);
                    section(ui, "Mount", |ui| self.mounts_ui(ui, backend, icons))
                        .inspect(|mount| picked = Some(*mount));
                    ui.add_space(8.0);
                    section(ui, "Stand in for", |ui| self.npcs_ui(ui))
                        .inspect(|npc| picked = Some(*npc));
                });
                picked
            })
            .and_then(|panel| panel.inner);
        // A body's faces and hair are its own, so clearing them is what reads them again.
        match picked {
            Some(Pick::Race(race)) => {
                self.race = race;
                self.tribe = self
                    .creator
                    .bodies
                    .iter()
                    .find(|body| body.race == race)
                    .map_or(self.tribe, |body| body.tribe);
                self.stood = false;
            }
            Some(Pick::Tribe(tribe)) => {
                self.tribe = tribe;
                self.stood = false;
            }
            Some(Pick::Gender(female)) => {
                self.female = female;
                self.stood = false;
            }
            Some(Pick::Child(child)) => {
                self.child = child;
                self.stood = false;
            }
            Some(Pick::Attire(attire)) => self.attire = attire,
            Some(Pick::Job(job)) => self.job = job,
            Some(Pick::Npc(npc)) => {
                self.npc = Some(npc);
                if let Some(held) = self.npcs.get(npc) {
                    self.race = held.race;
                    self.tribe = held.tribe;
                    self.female = held.female;
                    self.child = held.child;
                    self.choices = held.choices.iter().copied().collect();
                    self.attire = Attire::Npc;
                    self.chosen = [None; 11];
                    self.stains = held.stains;
                    self.stood = false;
                }
            }
            Some(Pick::Emote(emote)) => {
                self.emote = Some(emote);
                if let (Some(Ok(model)), Some(emote)) = (&self.model, self.emotes.get(emote)) {
                    let mounted = self.mount.is_some().then(|| emote.mounted()).flatten();
                    let seated = emote.seated(self.posture);
                    match (emote.expression(), mounted, seated) {
                        (Some(name), _, _) => model.express(name),
                        // A rider keeps the pose the mount holds it in, so the emote it plays is
                        // the partial the sheet names for that rather than the whole-body one.
                        (None, Some(key), _) => model.play_over(&self.emote_packs(key)),
                        // A body already sat down states a whole motion of its own, and settles
                        // back into the pose it was sitting in rather than standing up.
                        (None, None, Some(key)) => {
                            model.play(&self.emote_packs(key), self.pose_pack().as_deref())
                        }
                        (None, None, None) => {
                            let (start, settles) = emote.keys();
                            let packs = start.map(|key| self.emote_packs(key)).unwrap_or_default();
                            let settles = settles.and_then(|key| {
                                Some(self.stance.as_ref()?.pack(self.code, stance::COMMON, key))
                            });
                            model.play(&packs, settles.as_deref());
                        }
                    }
                }
            }
            Some(Pick::Posture(posture)) => {
                self.posture = posture;
                self.pose = 0;
                self.seated_changed();
            }
            Some(Pick::Pose(pose)) => {
                self.pose = pose;
                self.seated_changed();
            }
            Some(Pick::Mount(mount)) => {
                self.mount = mount;
                self.mount_seat = 0;
                // Nothing sits down on a mount: it states the seat it holds a rider in.
                self.posture = emotes::Posture::Standing;
                self.pose = 0;
            }
            Some(Pick::Seat(seat)) => self.mount_seat = seat,
            Some(Pick::Weapon(weapon)) => self.main_hand = weapon,
            Some(Pick::OffHand(weapon)) => self.off_hand = weapon,
            Some(Pick::Stance(drawn)) => self.drawn = drawn,
            Some(Pick::Made(customize, choice)) => {
                self.choices.insert(customize, choice);
            }
            Some(Pick::Choice(customize, id)) => {
                self.choices.insert(customize, u32::from(id));
                match customize {
                    FACE => self.face = id,
                    HAIRSTYLE => self.hair = id,
                    _ => {}
                }
            }
            None => {}
        }
    }
}

#[derive(Clone, Copy)]
enum Pick {
    Race(u32),
    Tribe(u32),
    Gender(bool),
    Child(bool),
    Attire(Attire),
    Job(usize),
    /// A menu left at a choice, by the customisation it drives.
    Made(u32, u32),
    /// The same, where what a menu holds is the number the file tree files the choice under.
    Choice(u32, u16),
    Emote(usize),
    /// Where the body sits, and which of that seat's own poses it holds.
    Posture(emotes::Posture),
    Pose(usize),
    /// A mount to seat the character on, or none to stand it on the ground again.
    Mount(Option<usize>),
    /// Which of the mount's own seats to ride in.
    Seat(usize),
    Npc(usize),
    /// A weapon to wield in the main hand, or none to go unarmed.
    Weapon(Option<usize>),
    /// A weapon to wield in the off hand, or none to leave it empty.
    OffHand(Option<usize>),
    /// Whether the weapon is drawn, which is a whole stance rather than a placement.
    Stance(bool),
}

/// How far along a bar a menu has been left, over the range the row states for it rather than over
/// its count: every slider counts a hundred and runs nought to a hundred, so its middle is exactly
/// the middle.
fn slid(menu: &menus::Menu, at: u32) -> f32 {
    let [low, high] = menu.range;
    match high > low {
        true => (at.clamp(low, high) - low) as f32 / (high - low) as f32,
        false => 0.0,
    }
}

/// One choice a menu offers: where it sits in the menu, the number the file tree files it under,
/// and the icon the creator draws it as.
struct Choice {
    at: u32,
    id: u16,
    icon: Option<u32>,
}

/// The model code a clan and gender are built on. Nothing in the files pairs the two: the bodies
/// are numbered in the order the game's races were first built, which is not the order `Tribe`
/// states them in, and neither sheet names a code. Counting which code ships the hair a clan is
/// offered does not tell them apart either: a man's hair is filed under every man's body, so five
/// codes score the whole of Wildwood's forty-eight and the last of them wins.
///
/// A woman is built on the body after her man's, and both Hyur clans have one of their own where
/// every other race shares.
const BUILT_ON: [u16; 16] = [1, 3, 5, 5, 11, 11, 7, 7, 9, 9, 13, 13, 15, 15, 17, 17];

/// The variant a code ends in: the body the race is grown to. A child is one shape shared by every
/// race that has one, which is why the deformers build every `04` on `c0104` rather than on the
/// adult of its own race.
const ADULT: u16 = 1;
const CHILD: u16 = 4;


/// Where a menu has been left, which is where the creator opens it until it is picked from. Not
/// bounded by the count: a lip colour past the dark half is where the light one starts.
fn choice(choices: &BTreeMap<u32, u32>, menu: &menus::Menu) -> u32 {
    choices.get(&menu.customize).copied().unwrap_or(menu.init)
}

/// What one of the creator's own boxes has been left at, and whether it is ticked.
fn left_at(choices: &BTreeMap<u32, u32>, key: u32) -> u32 {
    choices.get(&key).copied().unwrap_or(0)
}

fn ticked(choices: &BTreeMap<u32, u32>, key: u32) -> bool {
    left_at(choices, key) != 0
}

/// The colours a body was made with, the seams it hides, the shapes it wears, how tall it stands
/// and how full its chest is: everything the creator's own menus decide, off where each has been
/// left. `covered` is what the outfit hides on top of that, and `paint` the face paint it wears.
#[allow(clippy::too_many_arguments)]
pub fn appearance(
creator: &menus::Creator,
made: Option<&palette::Made>,
tribe: u32,
female: bool,
choices: &BTreeMap<u32, u32>,
covered: BTreeSet<String>,
paint: Option<u16>,
) -> (mdl::Customize, BTreeSet<String>, BTreeSet<String>, f32, Vec3) {
    let mut customize = mdl::Customize::default();
    // Every feature the face declares, less the ones the creator has been left on.
    let mut hidden: BTreeSet<String> = FEATURE_LETTERS
        .iter()
        .map(|letter| format!("{FEATURE}{letter}"))
        .collect();
    hidden.extend(covered);
    let mut shapes = BTreeSet::new();
    let mut stature = 1.0;
    let mut bust = Vec3::ONE;
    let mut tone = 1.0;
    let Some(body) = creator.body(tribe, female) else {
        return (customize, hidden, shapes, stature, bust);
    };
    // A body that is offered a tail or a pair of ears lengthens those with the customisation
    // the rest spend on muscle, and is left at the tone a race with none to set is given.
    let muscled = !body.menus.iter().any(|menu| menu.customize == TAIL);
    let palettes = made.map(|made| made.palettes(tribe, female));
    for menu in &body.menus {
        let at = choice(choices, menu) as usize;
        if let Some(palettes) = &palettes {
            let color = match menu.customize {
                SKIN_COLOR => Some((&palettes.skin, &mut customize.skin)),
                HAIR_COLOR => Some((&palettes.hair, &mut customize.hair)),
                LIP_COLOR => Some((&palettes.lips, &mut customize.lip)),
                EYE_COLOR => Some((&palettes.eyes, &mut customize.right_eye)),
                _ => None,
            };
            if let Some((swatches, held)) = color {
                *held = swatches.shaded(at);
            }
            // A strand is mixed between the two hair colours by its mask, so with the
            // highlight left off both are the one colour: leaving it white is what drew brown
            // hair silver. Eyes go the same way, one colour unless they are made odd.
            if menu.customize == HAIR_COLOR {
                customize.highlight = match ticked(choices, HIGHLIGHTS) {
                    true => palettes.highlights.shaded(left_at(choices, HIGHLIGHT_COLOR) as usize),
                    false => customize.hair,
                };
            }
            if menu.customize == EYE_COLOR {
                customize.left_eye = match ticked(choices, ODD_EYES) {
                    true => palettes.eyes.shaded(left_at(choices, LEFT_EYE_COLOR) as usize),
                    false => customize.right_eye,
                };
            }
            // What the shaders call the option colour is the race feature: a limbal ring, a
            // Miqo'te's ear tuft, the tattoo the creator names it after. Face paint has a
            // colour of its own, which the game hands to the decal rather than to this.
            if menu.customize == TATTOO_COLOR {
                let [red, green, blue, _] = palettes.features.shaded(at);
                customize.option = [red, green, blue];
            }
            if menu.customize == FACE_PAINT_COLOR {
                customize.decal = palettes.face_paint.shaded(at);
            }
        }
        // Only the lane, since the muscle tone menu comes before the skin colour that would
        // otherwise write over what it left.
        if menu.customize == MUSCLE_TONE && muscled {
            tone = slid(menu, at as u32);
        }
        if menu.customize == HEIGHT
            && let Some(palettes) = &palettes
        {
            let [short, tall] = palettes.height;
            stature = short + (tall - short) * slid(menu, at as u32);
        }
        if menu.customize == BUST
            && let Some(palettes) = &palettes
        {
            let [small, full] = palettes.bust;
            bust = Vec3::from(small).lerp(Vec3::from(full), slid(menu, at as u32));
        }
        if menu.customize == FEATURES {
            // The two menus that share this one number are halves of the same run of parts,
            // and each states where in it its own toggles start.
            let first = body
                .menus
                .iter()
                .take_while(|held| !std::ptr::eq(*held, menu))
                .filter(|held| held.customize == FEATURES)
                .map(|held| held.count as usize)
                .sum::<usize>();
            for bit in first..first + menu.count as usize {
                if at & 1 << bit != 0 && let Some(letter) = FEATURE_LETTERS.get(bit) {
                    hidden.remove(&format!("{FEATURE}{letter}"));
                }
            }
        }
        if let Some((_, prefix)) = SHAPED.iter().find(|(held, _)| *held == menu.customize)
            && at > 0
            && let Some(letter) = FEATURE_LETTERS.get(at - 1)
        {
            shapes.insert(format!("{prefix}_{letter}"));
        }
    }
    customize.skin[3] = tone;
    customize.paint = paint;
    if customize.paint.is_none() {
        customize.decal[3] = 0.0;
    }
    // Lip colour carries its own opacity, and the creator's own box is what it is worn at all.
    if !ticked(choices, LIPSTICK) {
        customize.lip[3] = 0.0;
    }
    (customize, hidden, shapes, stature, bust)
}


pub(super) fn resolve(tribe: u32, female: bool, child: bool) -> u16 {
    let body = BUILT_ON
        .get(tribe.max(1) as usize - 1)
        .copied()
        .unwrap_or(1);
    (body + u16::from(female)) * 100 + if child { CHILD } else { ADULT }
}

/// The body a model was made for, which its own file name states.
pub(super) fn made_for(model: &str) -> Option<u16> {
    let name = model.rsplit('/').next()?;
    name.strip_prefix('c')?.get(..4)?.parse().ok()
}

fn root(code: u16) -> String {
    format!("chara/human/c{code:04}")
}

/// Every model one numbered set of a kind holds. A face is several files and a body one, and which
/// is which is the directory's to say rather than a list of suffixes here.
pub(super) fn parts(listing: &Listing, under: &str, id: u16) -> Vec<String> {
    let letter = under.rsplit('/').next().unwrap_or_default().as_bytes()[0] as char;
    let mut found = listing.under(&format!("{under}/{letter}{id:04}/model/"));
    found.retain(|path| path.ends_with(".mdl"));
    found.sort();
    found
}

/// The numbered sets of a kind the code carries, each with the models it holds.
pub(super) fn sets(listing: &Listing, code: &u16, kind: &str) -> Vec<Set> {
    let under = format!("{}/obj/{kind}", root(*code));
    let letter = kind.as_bytes()[0] as char;
    listing
        .under(&format!("{under}/"))
        .iter()
        .filter(|path| path.ends_with(".mdl"))
        .filter_map(|path| {
            let rest = path.strip_prefix(&format!("{under}/{letter}"))?;
            rest.get(..4)?.parse::<u16>().ok()
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|id| Set {
            id,
            parts: parts(listing, &under, id),
        })
        .collect()
}

/// The sets of whichever part the body grows: a tail where it has one, a Viera's ears where it does
/// not. Both are numbered the same way and neither body ships the other.
pub(super) fn grown(listing: &Listing, code: u16) -> Vec<Set> {
    ["tail", "zear"]
        .into_iter()
        .map(|kind| sets(listing, &code, kind))
        .find(|found| !found.is_empty())
        .unwrap_or_default()
}

/// The model a code wears a set as, by slot. A set is one directory, so the whole of it is listed
/// once and answered for every slot at once.
///
/// Few bodies have a model of their own for a given slot: the rest wear the nearest one they are
/// built on, which is what the deformers between the two then shape onto them.
pub(super) fn equipment(
    listing: &Listing,
    deformers: &mdl::Deformers,
    code: u16,
    filed: Filed,
    set: u16,
) -> Models {
    let (kind, letter) = filed;
    let under = format!("chara/{kind}/{letter}{set:04}/model");
    let held = listing.under(&under);
    Slot::ALL.map(|slot| {
        if slot.filed() != filed {
            return None;
        }
        deformers.lineage(code).find_map(|code| {
            let path = format!("{under}/c{code:04}{letter}{set:04}_{}.mdl", slot.suffix());
            held.contains(&path).then_some(path)
        })
    })
}

/// The body whose skin another one's model is drawn with: its own where it ships a material for
/// the set, and the nearest body it is built on where it does not. Elezen, Miqo'te, Roegadyn women
/// and Lalafell women ship none, and each of those is a body whose skin is its parent's.
pub(super) fn skin(listing: &Listing, deformers: &mdl::Deformers, code: u16) -> Option<u16> {
    deformers.lineage(code).find(|code| {
        let under = format!("{}/obj/body/b{BODY_SET:04}/material/", root(*code));
        !listing.under(&under).is_empty()
    })
}

/// The models a body is built out of. The game holds one of them, `c0101b0001`, and stands every
/// other body on it deformed, which is why no other code ships a model of the set while twelve of
/// them ship the skin it is drawn with.
pub(super) fn body(listing: &Listing, deformers: &mdl::Deformers, code: u16) -> Vec<String> {
    deformers
        .lineage(code)
        .map(|code| parts(listing, &format!("{}/obj/body", root(code)), BODY_SET))
        .find(|parts| !parts.is_empty())
        .unwrap_or_default()
}

/// Which of a set's models covers one slot.
pub(super) fn part(parts: &[String], slot: Slot) -> Option<String> {
    let tail = format!("_{}.mdl", slot.suffix());
    parts.iter().find(|path| path.ends_with(&tail)).cloned()
}

/// The picked set if the code still carries it, and its lowest otherwise.
pub(super) fn pick(sets: &[Set], wanted: u16) -> u16 {
    match sets.iter().any(|set| set.id == wanted) {
        true => wanted,
        false => sets.first().map_or(wanted, |set| set.id),
    }
}

pub(super) fn held(sets: &[Set], wanted: u16) -> Vec<String> {
    sets.iter()
        .find(|set| set.id == wanted)
        .map(|set| set.parts.clone())
        .unwrap_or_default()
}

/// A searched list to pick one row of, each drawn with the icon the game offers it under. The rows
/// are the indices a search left, and what comes back is the one clicked. A row states its own
/// name, icon and whether it can be taken at all, which is what grays out the rest.
fn listed<'a>(
    ui: &mut egui::Ui,
    backend: &Backend,
    icons: &IconManager,
    id: &str,
    rows: &[usize],
    chosen: Option<usize>,
    held: impl Fn(usize) -> (&'a str, u32, bool),
) -> Option<usize> {
    let mut picked = None;
    let step = PIECE + 2.0 * ui.spacing().button_padding.y + ui.spacing().item_spacing.y;
    ScrollArea::vertical()
        .id_salt(id)
        .max_height(step * SHOWN as f32)
        .show_rows(ui, step, rows.len(), |ui, drawn| {
            for row in drawn {
                let index = rows[row];
                let (name, icon, playable) = held(index);
                let path = get_icon_path(backend.icons(), icon, false, Language::None);
                let excel = backend.excel().clone();
                let source = icons.get_or_insert_icon(&path, ui.ctx(), || {
                    let excel = excel.clone();
                    let path = path.clone();
                    TrackedPromise::spawn_local(async move { excel.get_icon(&path).await })
                });
                let loaded = match &source {
                    ManagedIcon::Loaded(source) => Some(source.clone()),
                    _ => None,
                };
                let button = match source {
                    ManagedIcon::Loaded(source) => egui::Button::image_and_text(
                        egui::Image::new(source)
                            .maintain_aspect_ratio(true)
                            .fit_to_exact_size(egui::Vec2::splat(PIECE)),
                        name,
                    ),
                    _ => egui::Button::new(name),
                };
                let response = ui.add_enabled(
                    playable,
                    button
                        .truncate()
                        .selected(chosen == Some(index))
                        .min_size(egui::vec2(ui.available_width(), PIECE)),
                );
                icon_context_menu(
                    &response,
                    icons,
                    excel,
                    backend.files().clone(),
                    icon,
                    &path,
                    loaded,
                );
                if response.clicked() {
                    picked = Some(index);
                }
            }
        });
    picked
}

/// A collapsing section that starts open, so nothing looks different until it is folded.
fn section(
    ui: &mut egui::Ui,
    title: &str,
    body: impl FnOnce(&mut egui::Ui) -> Option<Pick>,
) -> Option<Pick> {
    CollapsingHeader::new(title)
        .default_open(true)
        .show(ui, body)
        .body_returned
        .flatten()
}

/// Sets to pick from, laid out as many to a row as the panel is wide enough for. Every cell is the
/// same size whether or not the creator offers an icon for what is in it, so one it does not name
/// leaves a gap in the numbering rather than a break in the grid.
fn grid<T, R>(
    ui: &mut egui::Ui,
    id: &str,
    sets: &[T],
    mut cell: impl FnMut(&mut egui::Ui, &T) -> Option<R>,
) -> Option<R> {
    let step = ICON + GAP + ui.spacing().button_padding.x * 2.0;
    let columns = ((ui.available_width() / step) as usize).max(1);
    egui::Grid::new(id)
        .spacing(egui::Vec2::splat(GAP))
        .show(ui, |ui| {
            let mut picked = None;
            for (at, set) in sets.iter().enumerate() {
                if at > 0 && at % columns == 0 {
                    ui.end_row();
                }
                picked = cell(ui, set).or(picked);
            }
            picked
        })
        .inner
}

/// A palette to pick a colour out of, laid out the way the creator lays one out, answering the
/// index picked. What is offered is a window on the file's own run rather than the whole of it: a
/// lip colour is one of ninety-six, in whichever half of the palette the dark and light box picked.
fn colors(
    ui: &mut egui::Ui,
    id: (&str, usize),
    swatches: &palette::Swatches,
    offered: std::ops::Range<u32>,
    current: u32,
) -> Option<u32> {
    let mut picked = None;
    egui::Grid::new(id)
        .spacing(egui::Vec2::splat(2.0))
        // Without this, an empty column falls back to the UI's default interact size (~40px)
        // rather than a swatch's own 18px, and 8 of those locks the panel wide for good.
        .min_col_width(SWATCH)
        .min_row_height(SWATCH)
        .show(ui, |ui| {
            for (place, index) in offered.enumerate() {
                if place > 0 && place % palette::COLUMNS == 0 {
                    ui.end_row();
                }
                let Some(color) = swatches.shown(index as usize) else {
                    continue;
                };
                let (rect, response) =
                    ui.allocate_exact_size(egui::Vec2::splat(SWATCH), egui::Sense::click());
                ui.painter().rect_filled(rect, 2.0, color);
                if current == index {
                    ui.painter().rect_stroke(
                        rect,
                        2.0,
                        ui.visuals().selection.stroke,
                        egui::StrokeKind::Inside,
                    );
                }
                if response.clicked() {
                    picked = Some(index);
                }
            }
        });
    picked
}

/// One set to pick from: the icon the creator offers it under where there is one, and its number
/// where there is not, since a set the menus do not list still has a model on disk.
fn chip(
    ui: &mut egui::Ui,
    backend: &Backend,
    icons: &IconManager,
    choice: &Choice,
    selected: bool,
) -> bool {
    let Some(icon) = choice.icon else {
        return numbered(ui, choice, selected, "No icon");
    };
    let path = get_icon_path(backend.icons(), icon, false, Language::None);
    let excel = backend.excel().clone();
    let held = icons.get_or_insert_icon(&path, ui.ctx(), || {
        let excel = excel.clone();
        let path = path.clone();
        TrackedPromise::spawn_local(async move { excel.get_icon(&path).await })
    });
    match held {
        // Sized rather than fitted, so a cell is the same size whichever way it is drawn: a grid
        // that grows as its icons land walks every control under it down the panel.
        ManagedIcon::Loaded(source) => {
            let response = ui
                .add_sized(
                    egui::Vec2::splat(ICON),
                    egui::Button::image(
                        egui::Image::new(source.clone())
                            .maintain_aspect_ratio(true)
                            .shrink_to_fit(),
                    )
                    .selected(selected),
                )
                .on_hover_text(choice.id.to_string());
            icon_context_menu(
                &response,
                icons,
                excel,
                backend.files().clone(),
                icon,
                &path,
                Some(source),
            );
            response.clicked()
        }
        // An icon that has not landed yet is not one the creator never named, and saying so would
        // have every chip claim it has no icon for as long as the icons take to arrive.
        ManagedIcon::Failed(_) => numbered(ui, choice, selected, "No icon"),
        _ => numbered(ui, choice, selected, "Loading"),
    }
}

fn numbered(ui: &mut egui::Ui, choice: &Choice, selected: bool, why: &str) -> bool {
    ui.add_sized(
        egui::Vec2::splat(ICON),
        egui::Button::new((choice.at + 1).to_string()).selected(selected),
    )
    .on_hover_text(why)
    .clicked()
}

#[cfg(test)]
mod tests {
    use super::{Gear, resolve};

    /// Tataru's own `ModelHead` quad: set 5 (`e0005`), variant 224. The dye that goes with it is a
    /// separate `ENpcBase` column, not packed into this word.
    #[test]
    fn gear_reads_set_and_variant() {
        let gear = Gear::read(0x00E0_0005).unwrap();
        assert_eq!(gear.set, 5);
        assert_eq!(gear.variant, 224);
        assert!(Gear::read(0).is_none());
    }

    /// A clan and gender name the body they are built on, and the variant says how grown it is.
    #[test]
    fn a_clan_names_its_body() {
        assert_eq!(resolve(1, false, false), 101);
        assert_eq!(resolve(1, true, false), 201);
        assert_eq!(resolve(3, false, false), 501);
        assert_eq!(resolve(3, true, false), 601);
        assert_eq!(resolve(16, true, false), 1801);
    }

    /// The child body is the same clan's, one variant along, which is the only thing that tells
    /// Alphinaud from an adult Wildwood Elezen.
    #[test]
    fn a_child_is_the_same_body_grown_less() {
        assert_eq!(resolve(3, false, true), 504);
        assert_eq!(resolve(3, true, true), 604);
        assert_eq!(resolve(1, false, true), 104);
        assert_eq!(resolve(8, true, true), 804);
    }
}

