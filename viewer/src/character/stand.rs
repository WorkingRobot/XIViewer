//! The characters a scene stands, built out of the game's own rows rather than out of the creator's
//! pickers.
//!
//! Two participants naming the same row at the same height are one build however many places they
//! stand in: what differs between them is the transform, and that belongs to the scene rather than
//! to the model.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;
use std::sync::Arc;

use anyhow::Result;
use ironworks::excel::Language;
use ironworks::file::layer::Transform;

pub use super::npcs::{PARTY_STAND_IN, Roll, STABLED_CHOCOBO};
use super::npcs::Stands;
use super::{
    DEFORMERS, FACE, HAIRSTYLE, HEIGHT, Slot, TAIL, appearance, body, equipment, gating, grown,
    made_for, menus, npcs, palette, part, pick, resolve, sets,
};
use crate::assets::viewers::layer::scene;
use crate::assets::viewers::mdl;
use crate::backend::Backend;
use crate::data::listing::{Listed, Listing};
use crate::settings::{LANGUAGE, api_base};
use crate::utils::TrackedPromise;

/// The heights the game's own slider stands a character at, which is what a participant's `height`
/// of one through five names. Nought leaves the row's own byte alone.
const HEIGHTS: [u32; 5] = [0, 25, 50, 75, 100];

/// One character a scene wants standing: which row it draws, how tall, and where.
pub struct Wanted {
    pub roll: Roll,
    pub id: u32,
    /// Which height to stand it at over whatever the row states, as the game's own slider numbers
    /// one. Nought leaves the row be.
    pub height: u8,
    pub at: Transform,
    /// The `CTAL` participant standing it, which is what a cutscene's own timeline addresses.
    pub participant: u32,
}

/// A build: the row and the height, since those are the whole of what makes two characters differ.
type Key = (Roll, u32, u8);

/// One batch of files a character is still waiting on, and the rows every character was read from.
type Fetching = TrackedPromise<Result<Vec<(String, Vec<u8>)>>>;
type Rows = TrackedPromise<Result<BTreeMap<(Roll, u32), Stands>>>;

/// The motions each pack a scene loads holds, by the path it was read from, each with whether it
/// lays over the pose the body is in rather than replacing it.
type Naming = TrackedPromise<Result<Vec<(String, Vec<(String, bool)>)>>>;

/// Everything every character in a scene reads, which is read once for the whole cast.
struct Reference {
    creator: menus::Creator,
    made: palette::Made,
    deformers: mdl::Deformers,
    dye_templates: Rc<mdl::DyeTemplates>,
    worn_over: gating::Worn,
}

/// The same, before the shared handle a model is dyed through is put on it: a promise's answer has
/// to be able to cross threads, and an `Rc` cannot.
type Reading = (
    menus::Creator,
    palette::Made,
    mdl::Deformers,
    mdl::DyeTemplates,
    gating::Worn,
);

async fn reference(backend: &Backend, language: Language) -> Result<Reading> {
    // One after another rather than at once: several of these walk the same sheet, and the one
    // that reaches it first is the one that finishes first.
    Ok((
        menus::read(backend, language).await?,
        palette::Made::read(backend).await?,
        mdl::Deformers::read(&backend.files().read(DEFORMERS).await?)?,
        mdl::DyeTemplates::read(backend).await?,
        gating::Worn::read(backend).await?,
    ))
}

/// One character on its way to the scene.
struct Build {
    stands: Stands,
    /// The body it is grown on, and the body whose skin it is drawn with.
    code: u16,
    skin: Option<u16>,
    /// The models it is drawn from, each with the variant and the dyes it is worn at.
    worn: Vec<(String, u16, [Option<u8>; 2])>,
    /// The seams the outfit covers, which draw nothing rather than through what is over them.
    covered: BTreeSet<String>,
    held: BTreeMap<String, Vec<u8>>,
    fetching: Option<Fetching>,
    /// What each borrowed body is shaped onto this one by, kept rather than rebuilt.
    shaped: RefCell<BTreeMap<u16, Option<Arc<mdl::Deform>>>>,
    /// One model per participant standing this row: a cutscene drives each participant's own
    /// animation, and that state lives in the model, so two of them cannot share one.
    models: BTreeMap<u32, Rc<mdl::Rendered>>,
    /// The bodies its motions are read from, nearest first, kept for picking the pack a named
    /// motion comes out of.
    lineage: Vec<String>,
    failure: Option<String>,
}

/// The characters a scene stands, from the ids their participants name through to the models the
/// scene draws.
#[derive(Default)]
pub struct Cast {
    wanted: Vec<Wanted>,
    listing: Option<Rc<Listing>>,
    reference: Option<Rc<Reference>>,
    reading_reference: Option<TrackedPromise<Result<Reading>>>,
    reading_rows: Option<Rows>,
    asked: bool,
    builds: BTreeMap<Key, Build>,
    /// The participants a timeline has taken out of the frame, which a cutscene does to the double
    /// standing in for someone the moment the other takes over.
    hidden: BTreeSet<u32>,
    /// How much of each participant a fade leaves drawn. Absent is whole.
    faded: BTreeMap<u32, f32>,
    packs: Packs,
    failure: Option<String>,
}

/// Every motion the packs a scene loads hold, by name, so a timeline naming one plays out of the
/// pack that has it rather than by opening every candidate in turn.
#[derive(Default)]
struct Packs {
    queue: Vec<String>,
    reading: Option<Naming>,
    named: BTreeMap<String, Vec<String>>,
    /// The motions that lay over the pose the body is in rather than replacing it.
    over: BTreeSet<String>,
    /// What has been queued, so a pack two characters share is read once.
    asked: BTreeSet<String>,
    /// Which bodies have had their resident set queued, so the listing is walked once a body
    /// rather than once a frame.
    bodies: BTreeSet<String>,
}

impl Packs {
    fn ask(&mut self, paths: impl IntoIterator<Item = String>) {
        for path in paths {
            if self.asked.insert(path.clone()) {
                self.queue.push(path);
            }
        }
    }
}

/// The packs a body has resident whatever it is doing, which is where the motions a cutscene names
/// but its own `CTRL` does not list are filed: the idle it stands in, the additive gestures a
/// conversation lays over it, and the rest of the unarmed set.
fn resident(listing: &Listing, code: &str) -> Vec<String> {
    let kind = match code.as_bytes().first() {
        Some(b'c') => "human",
        Some(b'm') => "monster",
        Some(b'd') => "demihuman",
        _ => return Vec::new(),
    };
    let mut found = listing.under(&format!(
        "chara/{kind}/{code}/animation/a0001/bt_common/resident/"
    ));
    found.retain(|path| path.ends_with(".pap"));
    found
}

impl Cast {
    pub fn new(wanted: Vec<Wanted>) -> Self {
        Self {
            wanted,
            ..Self::default()
        }
    }

    /// The packs a scene loads its motions out of, which a cutscene states for itself.
    pub fn loads(&mut self, paths: Vec<String>) {
        self.packs.ask(paths);
    }

    /// The models built so far, against how many participants the cast stands.
    pub fn built(&self) -> (usize, usize) {
        let built = self.builds.values().map(|held| held.models.len()).sum();
        (built, self.wanted.len())
    }

    /// The model standing for a participant, once it has been built.
    pub fn model(&self, participant: u32) -> Option<&Rc<mdl::Rendered>> {
        self.wanted
            .iter()
            .find(|wanted| wanted.participant == participant)
            .and_then(|wanted| self.builds.get(&(wanted.roll, wanted.id, wanted.height)))
            .and_then(|held| held.models.get(&participant))
    }

    /// Which of the packs the scene loads a participant plays `motion` out of: the one filed
    /// under the nearest body it is built on, since a race that authors none of its own is posed
    /// from the one it borrows its clothes from.
    pub fn holding(&self, participant: u32, motion: &str) -> Option<String> {
        let held = self.packs.named.get(motion)?;
        let lineage = self
            .wanted
            .iter()
            .find(|wanted| wanted.participant == participant)
            .and_then(|wanted| self.builds.get(&(wanted.roll, wanted.id, wanted.height)))
            .map(|build| build.lineage.as_slice())
            .unwrap_or_default();
        lineage
            .iter()
            .find_map(|code| held.iter().find(|path| path.contains(code.as_str())))
            .or_else(|| held.first())
            .cloned()
    }

    /// Whether a motion lays over the pose the body is in rather than replacing it, which is what
    /// its own clip states.
    pub fn lays_over(&self, motion: &str) -> bool {
        self.packs.over.contains(motion)
    }

    /// Whether every pack asked for has been read, past which a motion the index does not name is
    /// filed nowhere the scene loads rather than not read yet.
    pub fn loaded(&self) -> bool {
        self.packs.queue.is_empty() && self.packs.reading.is_none()
    }

    /// Whether a participant is drawn. Everyone is until a timeline says otherwise, which is the
    /// state the game's own clip reverts to.
    pub fn show(&mut self, participant: u32, shown: bool) {
        match shown {
            true => self.hidden.remove(&participant),
            false => self.hidden.insert(participant),
        };
    }

    /// How much of a participant is drawn, which a fade takes down over its own length.
    pub fn fade(&mut self, participant: u32, opacity: f32) {
        self.faded.insert(participant, opacity);
    }

    /// Moves a participant to where its own timeline puts it now.
    pub fn place(&mut self, participant: u32, at: Transform) {
        for wanted in &mut self.wanted {
            if wanted.participant == participant {
                wanted.at = at;
            }
        }
    }

    /// Why the cast could not be read, where it could not be.
    pub fn failure(&self) -> Option<&str> {
        self.failure.as_deref()
    }

    /// Where each character stands, for the scene to draw. One build stands in as many places as
    /// participants named it.
    pub fn standing(&self) -> Vec<scene::Standing> {
        self.wanted
            .iter()
            .filter(|wanted| !self.hidden.contains(&wanted.participant))
            .filter_map(|wanted| {
                let held = self.builds.get(&(wanted.roll, wanted.id, wanted.height))?;
                Some(scene::Standing {
                    model: held.models.get(&wanted.participant)?.clone(),
                    at: wanted.at,
                    opacity: self
                        .faded
                        .get(&wanted.participant)
                        .copied()
                        .unwrap_or(1.0),
                })
            })
            .collect()
    }

    /// Asks for whatever the cast still needs, and stands up whatever has landed.
    pub fn poll(&mut self, ctx: &egui::Context, backend: &Backend) {
        if self.wanted.is_empty() || self.failure.is_some() {
            return;
        }
        if self.listing.is_none() {
            match backend.listing(&api_base(ctx)) {
                Listed::Loading => return,
                Listed::Ready(listing) => self.listing = Some(listing),
                Listed::Failed(why) => {
                    self.failure = Some(why.to_string());
                    return;
                }
            }
        }
        let language = LANGUAGE.get(ctx);
        if !self.asked {
            self.asked = true;
            let held = backend.clone();
            self.reading_reference =
                Some(TrackedPromise::spawn_local(
                    async move { reference(&held, language).await },
                ));
            let held = backend.clone();
            let ids: BTreeSet<(Roll, u32)> = self
                .wanted
                .iter()
                .map(|wanted| (wanted.roll, wanted.id))
                .collect();
            self.reading_rows = Some(TrackedPromise::spawn_local(async move {
                npcs::stand_in(&held, language, &ids).await
            }));
        }
        if let Some(promise) = self.packs.reading.take() {
            match promise.try_take() {
                Ok(read) => {
                    for (path, names) in read.unwrap_or_default() {
                        for (name, over) in names {
                            if over {
                                self.packs.over.insert(name.clone());
                            }
                            self.packs.named.entry(name).or_default().push(path.clone());
                        }
                    }
                }
                Err(promise) => self.packs.reading = Some(promise),
            }
        }
        if self.packs.reading.is_none() && !self.packs.queue.is_empty() {
            let files = backend.files().clone();
            let paths = std::mem::take(&mut self.packs.queue);
            self.packs.reading = Some(TrackedPromise::spawn_local(async move {
                let mut read = Vec::with_capacity(paths.len());
                for path in paths {
                    // A pack the scene lists but the install does not ship is skipped: what the
                    // rest hold is still worth having.
                    if let Ok(bytes) = files.read(&path).await
                        && let Ok(names) = mdl::motion_names(&bytes)
                    {
                        read.push((path, names));
                    }
                }
                Ok(read)
            }));
        }
        if let Some(promise) = self.reading_reference.take() {
            match promise.try_take() {
                Ok(Ok((creator, made, deformers, dye_templates, worn_over))) => {
                    self.reference = Some(Rc::new(Reference {
                        creator,
                        made,
                        deformers,
                        dye_templates: Rc::new(dye_templates),
                        worn_over,
                    }));
                }
                Ok(Err(why)) => self.failure = Some(why.to_string()),
                Err(promise) => self.reading_reference = Some(promise),
            }
        }
        if let Some(promise) = self.reading_rows.take() {
            match promise.try_take() {
                Ok(Ok(read)) => self.open(read),
                Ok(Err(why)) => self.failure = Some(why.to_string()),
                Err(promise) => self.reading_rows = Some(promise),
            }
        }
        let (Some(listing), Some(reference)) = (self.listing.clone(), self.reference.clone()) else {
            return;
        };
        for (key, build) in &mut self.builds {
            let standing: Vec<u32> = self
                .wanted
                .iter()
                .filter(|wanted| (wanted.roll, wanted.id, wanted.height) == *key)
                .map(|wanted| wanted.participant)
                .collect();
            build.poll(ctx, &listing, &reference, backend, &standing);
        }
        let bodies: Vec<String> = self
            .builds
            .values()
            .flat_map(|build| build.lineage.iter())
            .filter(|code| !self.packs.bodies.contains(*code))
            .cloned()
            .collect();
        for code in bodies {
            let held = resident(&listing, &code);
            self.packs.bodies.insert(code);
            self.packs.ask(held);
        }
    }

    /// Opens a build per row and height a participant asked for.
    fn open(&mut self, rows: BTreeMap<(Roll, u32), Stands>) {
        let mut wanted: BTreeSet<Key> = self
            .wanted
            .iter()
            .map(|wanted| (wanted.roll, wanted.id, wanted.height))
            .collect();
        wanted.retain(|(roll, id, _)| rows.contains_key(&(*roll, *id)));
        for key in wanted {
            // The same row stood at two heights is two builds: the height lands in the customise
            // array, so the two are different characters however much else they share.
            let Some(stands) = rows.get(&(key.0, key.1)) else {
                continue;
            };
            self.builds.insert(key, Build::new(stands.clone(), key.2));
        }
    }
}

impl Build {
    fn new(stands: Stands, height: u8) -> Self {
        let mut stands = stands;
        // The height a participant states stands in for whatever the row says, at the place in the
        // customise array the creator's own slider drives.
        let stood = usize::from(height)
            .checked_sub(1)
            .and_then(|at| HEIGHTS.get(at));
        if let (Stands::Human(npc), Some(held)) = (&mut stands, stood) {
            for (menu, at) in &mut npc.choices {
                if *menu == HEIGHT {
                    *at = *held;
                }
            }
        }
        Self {
            stands,
            code: 0,
            skin: None,
            worn: Vec::new(),
            covered: BTreeSet::new(),
            held: BTreeMap::new(),
            fetching: None,
            shaped: RefCell::new(BTreeMap::new()),
            models: BTreeMap::new(),
            lineage: Vec::new(),
            failure: None,
        }
    }

    fn poll(
        &mut self,
        ctx: &egui::Context,
        listing: &Listing,
        reference: &Reference,
        backend: &Backend,
        standing: &[u32],
    ) {
        if self.failure.is_some() {
            return;
        }
        if self.worn.is_empty() {
            self.dress(listing, &reference.deformers, &reference.worn_over);
            if self.worn.is_empty() {
                self.failure = Some("nothing on disk under what it names".to_owned());
                return;
            }
        }
        if let Some(promise) = self.fetching.take() {
            match promise.try_take() {
                Ok(Ok(read)) => self.held.extend(read),
                Ok(Err(why)) => self.failure = Some(why.to_string()),
                Err(promise) => {
                    self.fetching = Some(promise);
                    return;
                }
            }
        }
        // A worn piece's own imc says which material_id its variant actually draws with, which can
        // differ from the variant itself. Fetched alongside, and tolerantly: a missing one leaves
        // the variant number to stand for its own material_id.
        let missing: Vec<String> = self
            .worn
            .iter()
            .map(|(path, ..)| path.clone())
            .filter(|path| !self.held.contains_key(path))
            .collect();
        let optional: Vec<String> = self
            .worn
            .iter()
            .filter(|(_, variant, _)| *variant != 0)
            .filter_map(|(path, ..)| mdl::imc_path(path))
            .filter(|path| !self.held.contains_key(path))
            .collect();
        if !missing.is_empty() || !optional.is_empty() {
            if self.models.is_empty() && self.fetching.is_none() {
                let files = backend.files().clone();
                self.fetching = Some(TrackedPromise::spawn_local(async move {
                    let mut read = Vec::with_capacity(missing.len());
                    for path in missing {
                        read.push((path.clone(), files.read(&path).await?));
                    }
                    for path in optional {
                        if let Ok(bytes) = files.read(&path).await {
                            read.push((path, bytes));
                        }
                    }
                    Ok(read)
                }));
            }
            return;
        }
        // One a frame: composing several at once is a whole model's upload apiece, and a cutscene
        // stands as many of a row as its own participants name.
        if let Some(participant) = standing
            .iter()
            .find(|participant| !self.models.contains_key(participant))
        {
            self.compose(reference, *participant);
        }
        for model in self.models.values() {
            model.poll(ctx, backend);
        }
    }

    /// Every model this character is drawn from, once the listing and the deformers have answered.
    fn dress(&mut self, listing: &Listing, deformers: &mdl::Deformers, gating: &gating::Worn) {
        let npc = match &self.stands {
            Stands::Beast { under, variant } => {
                self.lineage = monster_code(under).into_iter().collect();
                let mut found = listing.under(under);
                found.retain(|path| path.ends_with(".mdl"));
                found.sort();
                self.worn = found
                    .into_iter()
                    .map(|path| (path, *variant, [None; 2]))
                    .collect();
                return;
            }
            Stands::Human(npc) => npc,
        };
        let wanted = resolve(npc.tribe, npc.female, npc.child);
        self.code = match deformers.knows(wanted) {
            true => wanted,
            false => resolve(npc.tribe, npc.female, false),
        };
        self.skin = super::skin(listing, deformers, self.code);
        self.lineage = deformers
            .lineage(self.code)
            .map(|code| format!("c{code:04}"))
            .collect();
        let body = body(listing, deformers, self.code);
        if body.is_empty() {
            return;
        }
        let choices: BTreeMap<u32, u32> = npc.choices.iter().copied().collect();
        let at = |menu: u32| choices.get(&menu).copied().unwrap_or(0) as u16;
        // A tail or a pair of ears is numbered one past where its own menu sits.
        let grown = grown(listing, self.code);
        let held = [
            (sets(listing, &self.code, "face"), at(FACE)),
            (sets(listing, &self.code, "hair"), at(HAIRSTYLE)),
            (grown, at(TAIL) + 1),
        ];
        // The face leads, since the first file is what names the skeleton the rest are posed on.
        let mut worn: Vec<(String, u16, [Option<u8>; 2])> = Vec::new();
        let hat = npc.outfit[Slot::Head as usize];
        for (at, (sets, wanted)) in held.into_iter().enumerate() {
            // A hat states for itself whether the hair under it still draws.
            if at == 1 && hat.is_some_and(|hat| !gating.keeps_hair(hat.set, npc.race)) {
                continue;
            }
            worn.extend(
                super::held(&sets, pick(&sets, wanted))
                    .into_iter()
                    .map(|path| (path, 0, [None; 2])),
            );
        }
        // Whether the body's own model for a slot still draws is what a piece worn over it states,
        // rather than anything the two meshes could be told apart by: where a race's smallclothes
        // are its bare skin the two are the very same geometry.
        let bared = |slot: Slot| {
            Slot::GEAR.into_iter().all(|over| {
                npc.outfit[over as usize]
                    .is_none_or(|gear| gating.shows(over, gear.set, slot, npc.race))
            })
        };
        let mut arrived = super::Outfit::default();
        for slot in Slot::ALL {
            let held = npc.outfit[slot as usize].and_then(|gear| {
                let path = equipment(listing, deformers, self.code, slot.filed(), gear.set)
                    [slot as usize]
                    .clone()?;
                Some((path, gear))
            });
            match held {
                // An adornment is the only thing its slot ever draws, so a piece worn over it
                // states whether it is there at all rather than what stands in for it.
                Some(_) if slot.adornment() && !bared(slot) => {}
                Some((path, gear)) => {
                    arrived[slot as usize] = Some(gear);
                    worn.push((path, gear.variant, npc.stains[slot as usize]));
                }
                // Nothing stands in for a bare head: the body ships no model for it, and the face
                // and the hair are what draw one.
                None if !bared(slot) => {}
                None => worn.extend(part(&body, slot).map(|path| (path, 0, [None; 2]))),
            }
        }
        self.covered = gating
            .covers(&arrived, npc.race)
            .into_iter()
            .map(str::to_owned)
            .collect();
        self.worn = worn;
    }

    /// A model for one participant, once every file it is drawn from has landed.
    fn compose(&mut self, reference: &Reference, participant: u32) {
        let parts: Vec<mdl::Source> = self
            .worn
            .iter()
            .filter_map(|(path, variant, _)| {
                let deform = made_for(path).and_then(|made_for| {
                    self.shaped
                        .borrow_mut()
                        .entry(made_for)
                        .or_insert_with(|| {
                            reference
                                .deformers
                                .between(made_for, self.code)
                                .map(Arc::new)
                        })
                        .clone()
                });
                Some(mdl::Source {
                    path: path.clone(),
                    bytes: self.held.get(path)?.clone(),
                    variant: *variant,
                    material: mdl::material::resolve_variant(
                        path,
                        *variant,
                        mdl::imc_path(path)
                            .and_then(|imc| self.held.get(&imc))
                            .map(Vec::as_slice),
                    ),
                    deform,
                    skin: self.skin,
                    rigid: false,
                })
            })
            .collect();
        if parts.len() != self.worn.len() {
            return;
        }
        let model = match mdl::compose(&parts) {
            Ok(model) => model,
            Err(why) => {
                self.failure = Some(why.to_string());
                return;
            }
        };
        model.placed();
        model.built_on(self.lineage.clone());
        if let Stands::Human(npc) = &self.stands {
            let choices: BTreeMap<u32, u32> = npc.choices.iter().copied().collect();
            let paint = choices.get(&super::FACE_PAINT).copied().unwrap_or(0) as u16;
            let (customize, hidden, shapes, stature, bust) = appearance(
                &reference.creator,
                Some(&reference.made),
                npc.tribe,
                npc.female,
                &choices,
                self.covered.clone(),
                (paint > 0).then_some(paint),
            );
            model.made(customize, hidden, shapes, stature, bust);
            model.dye(
                Some(reference.dye_templates.clone()),
                self.worn.iter().map(|(_, _, stains)| *stains).collect(),
            );
        }
        self.models.insert(participant, Rc::new(model));
    }
}

/// The body a beast's own files sit under, which is what names the packs it is posed from.
fn monster_code(under: &str) -> Option<String> {
    under.split('/').nth(2).filter(|held| !held.is_empty()).map(str::to_owned)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn a_body_is_named_by_the_directory_its_own_files_sit_under() {
        assert_eq!(
            monster_code("chara/monster/m0886/obj/body/b0001/model/"),
            Some("m0886".to_owned())
        );
        // The set a demihuman wears is a directory of its own, and is not the body.
        assert_eq!(
            monster_code("chara/demihuman/d1003/obj/equipment/e0001/model/"),
            Some("d1003".to_owned())
        );
    }

}
