//! Posing a model on the skeleton it is skinned to.
//!
//! A mesh's blend indices name slots of its own bone table, and that table names bones the way a
//! skeleton does, so the palette a skinned shader reads is matched up by name rather than by
//! position. Each joint carries the pose a motion puts its bone in against the pose the model is
//! stored in, which leaves a bone the skeleton does not name standing where the file put it.
//!
//! The skeleton is guessed from the model's own path and fetched on the first frame that draws a
//! skinned mesh, the way the model's `.imc` is. The packs are read off the install's own listing,
//! since nothing in the model, the skeleton or the sheets names the ones a model can play.
//!
//! A body's own skeleton names none of the bones a face, a hairstyle or a piece of headgear moves
//! on: those hang off skeletons of their own, and `.est` is what says which. They are merged into
//! the body's rather than posed apart, since each is stated as bones hanging off one the body
//! already names.

use crate::assets::viewers::skeleton::Laid;
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Cursor;
use std::rc::Rc;

use anyhow::Result;
use egui::{Color32, RichText};
use glam::{Mat4, Quat, Vec3};
use ironworks::file::File;
use ironworks::file::est::ExtraSkeletonTemplate;
use ironworks::file::pap::{AnimationPack, Binding};
use ironworks::file::sklb::{SkeletonBinary, Transform};
use ironworks::file::tmb::{CommandKind, Item, Timeline};

use super::super::skeleton::{Placement, Rig, middle};
use super::super::{link, placed, section};
use crate::backend::Backend;
use crate::data::listing::{Listed, Listing};
use crate::settings::api_base;
use crate::utils::{TrackedPromise, file_name};

/// What the picker calls standing the model where its own file put it.
const REST: &str = "Reference pose";
/// How tall the pack list is allowed to get. A human carries thousands of them.
const PACK_LIST_HEIGHT: f32 = 240.0;

/// The bone a body hangs off, which is what a pose is centred on. A tail carries many bones a long
/// way out and swings them, and averaging every bone instead walks the frame around with it.
const ANCHOR: &str = "n_hara";

/// The pair of bones the creator's bust slider scales, which are leaves of the body's own skeleton.
const BUST: [&str; 2] = ["j_mune_l", "j_mune_r"];

/// The bones a visor hinges on, each turned about its own Z by one of the three angles the
/// gimmick states for the set. A head that names none of them raises nothing.
const VISOR: [&str; 3] = ["j_ex_met_va", "j_ex_met_vb", "j_ex_met_vc"];

/// The bone a mount seats its rider on, and the ones an extra rider is seated on beyond it. A
/// mount names them `n_mount`, then `n_mount_second` for a second seat or `n_mount_a`,
/// `n_mount_b`, ... for a third and beyond; both spellings are real, so a seat is anything
/// starting with this rather than one fixed suffix.
const SEAT: &str = "n_mount";

/// The rig a model is skinned to, ready to answer a mesh's bone table with a palette.
pub struct Skin {
    rig: Rig,
    /// Where each bone rests, inverted: what takes a vertex out of the pose the model is stored in.
    rest: Vec<Mat4>,
    /// Which bone the skeleton calls each name.
    named: HashMap<String, usize>,
    /// The bone a pose is centred on, where it rests, and how far the furthest bone stands from it,
    /// which a pose's own are read against.
    anchor: Option<usize>,
    home: Vec3,
    spread: f32,
    /// Where a mount seats each of its riders, nearest first, in the order its own skeleton
    /// names them.
    seats: Vec<usize>,
}

impl Skin {
    fn new(rig: Rig) -> Self {
        let world = rig.world(rig.reference());
        let rest = world
            .iter()
            .map(|placement| placement.matrix().inverse())
            .collect();
        // A name that collided on merge names more than one bone here; the first is the one the
        // rig itself resolves a bare lookup to, so a mesh's own table has to agree with it.
        let mut named: HashMap<String, usize> = HashMap::new();
        for (bone, name) in rig.names().iter().enumerate() {
            named.entry(name.clone()).or_insert(bone);
        }
        let anchor = named.get(ANCHOR).copied();
        let (home, spread) = middle(&world, anchor);
        let seats = rig
            .names()
            .iter()
            .enumerate()
            .filter(|(_, name)| {
                name.strip_prefix(SEAT)
                    .is_some_and(|rest| rest.is_empty() || rest.starts_with('_'))
            })
            .map(|(bone, _)| bone)
            .collect();
        Self {
            rig,
            rest,
            named,
            anchor,
            home,
            spread,
            seats,
        }
    }

    /// What each slot of one mesh's bone table moves a vertex by, in the model's own space.
    fn palette(&self, table: &[String], posed: &[Placement]) -> Vec<Mat4> {
        table
            .iter()
            .map(|name| match self.named.get(name) {
                Some(bone) => posed[*bone].matrix() * self.rest[*bone],
                None => Mat4::IDENTITY,
            })
            .collect()
    }
}

/// A skeleton as its own file states it, held unbuilt so several can be merged into one rig.
struct Skeleton {
    names: Vec<String>,
    parents: Vec<i16>,
    reference: Vec<Transform>,
}

impl Skeleton {
    fn read(bytes: &[u8]) -> Result<Self> {
        let file = SkeletonBinary::read(Cursor::new(bytes.to_vec()))?;
        let skeleton = file.parse_skeleton()?;
        Ok(Self {
            names: skeleton.bones().to_vec(),
            parents: skeleton.parent_indices().to_vec(),
            reference: skeleton.reference_pose().to_vec(),
        })
    }
}

/// A skeleton a part is posed on beyond the body's own, which is one table each.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Extra {
    Face,
    Hair,
    Head,
    Body,
}

impl Extra {
    const ALL: [Self; 4] = [Self::Face, Self::Hair, Self::Head, Self::Body];

    fn table(self) -> &'static str {
        match self {
            Self::Face => "chara/xls/charadb/faceSkeletonTemplate.est",
            Self::Hair => "chara/xls/charadb/hairSkeletonTemplate.est",
            Self::Head => "chara/xls/charadb/extra_met.est",
            Self::Body => "chara/xls/charadb/extra_top.est",
        }
    }

    /// The directory its skeletons are filed under, and the letter their files carry.
    fn filed(self) -> (&'static str, char) {
        match self {
            Self::Face => ("face", 'f'),
            Self::Hair => ("hair", 'h'),
            Self::Head => ("met", 'm'),
            Self::Body => ("top", 't'),
        }
    }
}

/// One frame of a model's pose, worked out once for everything that reads it.
#[derive(Default)]
pub struct Pose {
    /// The palette each mesh's blend indices read, in the model's own space.
    pub joints: Vec<Vec<Mat4>>,
    /// The rig itself, drawn where it was asked for.
    pub skeleton: Vec<placed::Batch>,
    /// How far the pose has carried the bones from where the model rests.
    pub drift: Vec3,
    /// How much further from the middle of them the pose flings the bones than the rest pose does,
    /// which the geometry hung on them reaches by too.
    pub stretch: f32,
    /// Every bone's own placement this frame, in the model's own space, for whatever wants a joint
    /// on its own rather than through a mesh's palette. Empty until the rig has landed.
    pub world: Vec<Mat4>,
    /// Where this rig seats a rider, for the one that is a mount.
    seat: Option<Placement>,
}

/// What a pack names the motion a model stands in, whatever rig it is built on.
const IDLE: &str = "_id0";

/// The seated idle a mount's own per-seat pack names. Picked by exact name rather than by
/// [`Motions::standing`]'s suffix guess: the pack also carries an additive breathing layer and,
/// for the seat a driver takes, a mount-up transition, and nothing about their own names rules
/// them out as reliably as this one being named for what it is.
const RIDE_IDLE: &str = "cbnm_mt_id0";

/// One `cfxf_` clip a motion's own timeline plays over it, and where in the motion's own clock
/// (in seconds) it is held across: a `C010` states this as a frame count and a start/end fraction
/// of the clip itself, which only means anything once scaled by the timeline it is read from.
#[derive(Clone)]
struct Expression {
    name: String,
    /// Seconds into the motion's own clock the hold starts and ends.
    window: (f32, f32),
    /// The clip's own position, normalized start to end, that the window plays across.
    span: (f32, f32),
}

/// What a motion's own timeline lays over it on the face: the poses it runs through, and the
/// library they come out of. A longer emote states several poses in turn, so the face is a
/// schedule against the body's own clock rather than one clip held for the whole of it.
#[derive(Clone, Default)]
struct Companion {
    /// The `TMPP` the timeline names, which is the facial pack past `nonresident/`. Absent leaves
    /// the pose to be found by its own name, or in the library every face keeps resident.
    library: Option<String>,
    expressions: Vec<Expression>,
}

impl Companion {
    /// Which pose is in force `at` seconds into the motion: the last one the clock has reached,
    /// or, before the first, the one a previous turn round the loop left the face holding. Read
    /// without assuming the timeline states its commands in time order, which it does not.
    fn at(&self, at: f32) -> Option<&Expression> {
        let latest = |left: &&Expression, right: &&Expression| {
            left.window.0.total_cmp(&right.window.0)
        };
        self.expressions
            .iter()
            .filter(|held| held.window.0 <= at)
            .max_by(latest)
            .or_else(|| self.expressions.iter().max_by(latest))
    }
}

/// The motions a pack holds, and the name each of its animations gives one.
struct Motions {
    /// Animation names, each with the motion it plays.
    named: Vec<(String, usize)>,
    /// The companion each of `named`'s own timeline plays over it, parallel to it. An emote often
    /// states its facial expression this way rather than by a name the creator picks.
    companions: Vec<Companion>,
    bindings: Vec<Binding>,
}

impl Motions {
    fn read(bytes: &[u8]) -> Result<Self> {
        let file = AnimationPack::read(Cursor::new(bytes.to_vec()))?;
        let bindings = file.parse_animations()?;
        let (mut named, mut companions) = (Vec::new(), Vec::new());
        for (animation, timeline) in file.animations().iter().zip(file.timelines()) {
            let Some(motion) = usize::try_from(animation.havok_index())
                .ok()
                .filter(|motion| bindings.get(*motion).is_some())
            else {
                continue;
            };
            named.push((animation.name().to_owned(), motion));
            let duration = bindings[motion].motion().duration();
            companions.push(companion(timeline, duration));
        }
        Ok(Self {
            named,
            companions,
            bindings,
        })
    }

    /// The motion the picker is on.
    fn binding(&self, motion: usize) -> Option<&Binding> {
        let (_, at) = self.named.get(motion)?;
        self.bindings.get(*at)
    }

    /// The companion the motion at `motion` names, if its own timeline lays any pose over it.
    fn companion(&self, motion: usize) -> Option<&Companion> {
        self.companions
            .get(motion)
            .filter(|held| !held.expressions.is_empty())
    }

    /// Which motion the pack opens on: the idle where it names one, since a monster's pack leads
    /// with a special rather than with the motion it stands in. Otherwise the first that stands on
    /// its own, since the first of a human's idle pack is a delta over whatever else is playing and
    /// a model posed on one alone scatters. A pack of nothing but deltas, which every facial one
    /// is, opens on its first.
    fn standing(&self) -> Option<usize> {
        let alone = |at: &usize| self.binding(*at).is_some_and(|held| held.blend_hint() == 0);
        (0..self.named.len())
            .find(|at| alone(at) && self.named[*at].0.ends_with(IDLE))
            .or_else(|| (0..self.named.len()).find(alone))
            .or((!self.named.is_empty()).then_some(0))
    }
}

/// Every `cfxf_` pose a motion's own timeline plays over it, and the library they come out of:
/// `duration` is the motion's own, in seconds, which is what the timeline's frame units are scaled
/// against. An emote runs through several poses in turn, each one taking the face from the time
/// its own command states.
fn companion(timeline: &[u8], duration: f32) -> Companion {
    let Ok(timeline) = Timeline::read(Cursor::new(timeline.to_vec())) else {
        return Companion::default();
    };
    let frames = timeline.items().iter().find_map(|item| match item {
        Item::Header(header) => Some(header.duration()),
        _ => None,
    });
    let Some(frames) = frames.filter(|frames| *frames > 0) else {
        return Companion::default();
    };
    let scale = duration / f32::from(frames);
    let mut companion = Companion::default();
    for item in timeline.items() {
        let command = match item {
            Item::FaceLibrary(library) => {
                companion.library = library.path().map(ToOwned::to_owned);
                continue;
            }
            Item::Command(command) => command,
            _ => continue,
        };
        let (path, hold, span) = match command.kind() {
            CommandKind::C009(animation) => (animation.motion(), animation.duration(), (0.0, 1.0)),
            CommandKind::C010(animation) => (
                animation.motion(),
                animation.duration(),
                // `0x01` enables the start and end frames; without it the whole clip plays.
                match animation.flags() & 0x01 != 0 {
                    true => (animation.animation_start(), animation.animation_end()),
                    false => (0.0, 1.0),
                },
            ),
            _ => continue,
        };
        let Some(name) = path.and_then(|path| path.strip_prefix("cfxf_")) else {
            continue;
        };
        let start = f32::from(command.time()) * scale;
        companion.expressions.push(Expression {
            name: name.to_owned(),
            window: (start, start + hold as f32 * scale),
            span,
        });
    }
    companion
}

/// How long a change from one motion to another blends over, in seconds, by the names their own
/// packs give them.
type Blend = dyn Fn(&str, &str) -> f32;

/// A clip on its way out from under whatever replaced it, kept whole so it can go on being
/// sampled across the fade.
struct Leaving {
    path: String,
    pack: Rc<Motions>,
    motion: usize,
    time: f32,
}

/// One motion playing on the rig: the pack it comes from, which of that pack's motions, and how
/// far into it.
#[derive(Default)]
struct Layer {
    /// The pack to play, as the user or an emote has it.
    wanted: RefCell<String>,
    pack: RefCell<Option<Fetch<Rc<Motions>>>>,
    /// What to hold once this pack has played through, which is how an emote states the pose it
    /// settles into apart from the motion that gets it there.
    then: RefCell<Option<String>>,
    /// Which of the pack's motions to open on, by name. A face keeps the ones it uses often in a
    /// pack together, so an expression is not always a file of its own.
    opening: RefCell<Option<String>>,
    /// Which motion is playing, indexing [`Motions::named`]. None leaves the bones where the
    /// skeleton rests, which is what a file being inspected shows.
    motion: Cell<Option<usize>>,
    time: Cell<f32>,
    /// Packs still to try, in order, each with the motion it is wanted for, if the one loading now
    /// lands without its own. A name and its file disagree often enough that a guess has to be
    /// verified rather than trusted on sight, and a pack the install lists can still ship empty.
    /// A candidate wanted for no motion in particular is only given way to by one that 404s.
    retry: RefCell<Vec<(String, Option<String>)>>,
    /// The clip the last change is fading out of, sampled under the incoming one until the fade
    /// closes. A layer with nothing wanted fades this one out to nothing instead, which is what
    /// lets the layers under it back through.
    leaving: RefCell<Option<Leaving>>,
    /// How far into the fade, in seconds, and how long it runs. No length is a hard cut.
    fade: Cell<f32>,
    over: Cell<f32>,
    /// Whether the length is still owed: the blend table is keyed by the two motions' own names,
    /// and the incoming one is not named until its pack lands, so the fade is held shut until then.
    pricing: Cell<bool>,
    /// How long to fade this layer out over once the clip has played through with nothing queued
    /// behind it. No length leaves it looping, which is what a pose held forever wants.
    settle: Cell<f32>,
}

impl Layer {
    /// `fade` is how long to cross-fade over where the caller already knows the length, a cut at
    /// nought; `None` leaves it to be priced once the incoming pack lands and names the motion it
    /// opens on, which is the only thing the blend table answers to.
    fn load(&self, path: &str, motion: Option<&str>, then: Option<&str>, fade: Option<f32>) {
        // Nothing to play is never left to the blend table: there is no incoming motion for it to
        // be keyed by, so a layer let go of takes the length it was handed or none at all.
        let fade = match path.is_empty() {
            true => Some(fade.unwrap_or_default()),
            false => fade,
        };
        // A change asked for while the last one is still being fetched has no clip of its own to
        // hand over yet, and letting that clear what is already on its way out is what snapped the
        // body back to its reference pose: keep the outgoing clip until something replaces it.
        match (fade, self.leaving_clip()) {
            (Some(0.0), _) => *self.leaving.borrow_mut() = None,
            (_, Some(clip)) => *self.leaving.borrow_mut() = Some(clip),
            (_, None) => {}
        }
        self.fade.set(0.0);
        self.over.set(fade.unwrap_or_default());
        self.pricing.set(fade.is_none());
        self.settle.set(0.0);
        path.clone_into(&mut self.wanted.borrow_mut());
        *self.pack.borrow_mut() = None;
        *self.then.borrow_mut() = then.map(ToOwned::to_owned);
        *self.opening.borrow_mut() = motion.map(ToOwned::to_owned);
        *self.retry.borrow_mut() = Vec::new();
        self.motion.set(None);
        self.time.set(0.0);
    }

    /// Plays the first of `candidates` that holds the motion it names through once, then fades the
    /// layer back out over `fade`, which is what an action laid over a base pose does rather than
    /// loop on top of it forever.
    fn once(&self, candidates: Vec<(String, String)>, fade: f32) {
        self.seek(candidates, Some(fade));
        self.settle.set(fade);
    }

    /// Plays the first of `candidates` the install actually holds, on whatever motion its pack
    /// opens on, settling into `then` once that has played through. A candidate that is not there
    /// gives way to the next, which is how an emote filed under the class directory a body's
    /// weapons put it in falls back to the one every body shares.
    fn plays(&self, candidates: &[String], then: Option<&str>, fade: Option<f32>) {
        let mut candidates = candidates.iter();
        let first = candidates.next().map(String::as_str).unwrap_or_default();
        self.load(first, None, then, fade);
        *self.retry.borrow_mut() = candidates.map(|path| (path.clone(), None)).collect();
    }

    /// The two motions a change of clip is between, once the incoming one has landed and named
    /// itself: what the layer is leaving and what it took up, for the caller that can price the
    /// blend. Answered once per change.
    fn changed(&self) -> Option<(String, String)> {
        if !self.pricing.get() {
            return None;
        }
        let pack = self.pack.borrow();
        let (to, _) = pack
            .as_ref()
            .and_then(Fetch::ready)?
            .named
            .get(self.motion.get()?)?;
        let from = self.leaving.borrow().as_ref().and_then(|held| {
            Some(held.pack.named.get(held.motion)?.0.clone())
        });
        self.pricing.set(false);
        Some((from.unwrap_or_default(), to.clone()))
    }

    /// Takes the length the blend table priced the change at.
    fn priced(&self, over: f32) {
        self.over.set(over);
    }

    /// What is playing now, ready to go on being sampled after the layer has moved off it.
    fn leaving_clip(&self) -> Option<Leaving> {
        let pack = self.pack.borrow();
        Some(Leaving {
            path: self.wanted.borrow().clone(),
            pack: Rc::clone(pack.as_ref().and_then(Fetch::ready)?),
            motion: self.motion.get()?,
            time: self.time.get(),
        })
    }

    /// How much of the incoming clip shows: none until the fade opens, all of it once it has
    /// closed. A layer with nothing wanted reads this as how far its outgoing clip has faded out,
    /// and one whose change has yet to be priced shows none of it, since the fade has not opened.
    fn share(&self) -> f32 {
        match (self.pricing.get(), self.over.get() > 0.0) {
            (true, _) => 0.0,
            (false, true) => (self.fade.get() / self.over.get()).clamp(0.0, 1.0),
            (false, false) => 1.0,
        }
    }

    /// Loads the first of `candidates` that opens on the motion it names, keeping the rest to try
    /// in turn if it lands without one. An empty list leaves the layer at rest.
    fn seek(&self, mut candidates: Vec<(String, String)>, fade: Option<f32>) {
        match candidates.is_empty() {
            true => self.load("", None, None, fade),
            false => {
                let (path, motion) = candidates.remove(0);
                self.load(&path, Some(&motion), None, fade);
                *self.retry.borrow_mut() =
                    candidates.into_iter().map(|(path, motion)| (path, Some(motion))).collect();
            }
        }
    }

    /// Whether the layer has run out of candidates without landing on `opening`'s motion,
    /// however it got there: nothing was ever wanted, or every candidate `seek` was given has
    /// landed and none of them named it.
    fn spent(&self) -> bool {
        if self.wanted.borrow().is_empty() {
            return true;
        }
        let landed = matches!(
            self.pack.borrow().as_ref(),
            Some(Fetch::Ready(_)) | Some(Fetch::Failed(_))
        );
        landed && self.retry.borrow().is_empty() && self.motion.get().is_none()
    }

    /// Takes up the pack once it lands, opening on the motion asked for. A pack that lands
    /// without it, or never arrives at all, gives way to the next candidate `seek` queued, or
    /// what was queued behind it once that runs out too: not every race ships the motion an
    /// emote starts with, and a file's name is not always its content.
    fn poll(&self, backend: &Backend) {
        let wanted = self.wanted.borrow().clone();
        let mut held = self.pack.borrow_mut();
        if wanted.is_empty() || !matches!(held.as_ref(), None | Some(Fetch::Fetching(_))) {
            return;
        }
        Fetch::poll(&mut held, backend, &wanted, |bytes| {
            Motions::read(bytes).map(Rc::new)
        });
        let ready = held.as_ref().and_then(Fetch::ready);
        let opening = self.opening.borrow().clone();
        let motion = ready.and_then(|motions| match opening.as_deref() {
            Some(name) => motions.named.iter().position(|(held, _)| held == name),
            None => motions.standing(),
        });
        let failed = matches!(held.as_ref(), Some(Fetch::Failed(_)));
        let missed = opening.is_some() && ready.is_some() && motion.is_none();
        drop(held);
        if failed || missed {
            let next = {
                let mut retry = self.retry.borrow_mut();
                (!retry.is_empty()).then(|| retry.remove(0))
            };
            if let Some((next, motion)) = next {
                next.clone_into(&mut self.wanted.borrow_mut());
                *self.opening.borrow_mut() = motion;
                *self.pack.borrow_mut() = None;
                self.motion.set(None);
                self.time.set(0.0);
                return;
            }
        }
        self.motion.set(motion);
        if failed {
            let then = self.then.borrow_mut().take();
            if let Some(then) = then {
                self.load(&then, None, None, Some(self.over.get()));
            }
        }
    }

    /// Which of the pack's motions is playing, with the rest pose as the way out of playing any.
    fn motion_ui(&self, ui: &mut egui::Ui, id: &str) {
        let pack = self.pack.borrow();
        let Some(motions) = pack.as_ref().and_then(Fetch::ready) else {
            return;
        };
        let motion = self.motion.get();
        egui::ComboBox::from_id_salt(id)
            .selected_text(match motion.and_then(|at| motions.named.get(at)) {
                Some((name, _)) => name.as_str(),
                None => REST,
            })
            .show_ui(ui, |ui| {
                if ui.selectable_label(motion.is_none(), REST).clicked() {
                    self.motion.set(None);
                    self.time.set(0.0);
                }
                for (at, (name, _)) in motions.named.iter().enumerate() {
                    if ui.selectable_label(motion == Some(at), name).clicked() {
                        self.motion.set(Some(at));
                        self.time.set(0.0);
                    }
                }
            });
    }

    /// How far the motion on screen runs, or nothing if none is playing.
    fn duration(&self) -> Option<f32> {
        let pack = self.pack.borrow();
        let motions = pack.as_ref().and_then(Fetch::ready)?;
        let binding = self.motion.get().and_then(|at| motions.binding(at))?;
        Some(binding.motion().duration().max(f32::EPSILON))
    }

    /// The expression the motion now playing lays over the face `at` seconds in, if its own
    /// timeline lays any.
    fn expression(&self, at: f32) -> Option<Expression> {
        let pack = self.pack.borrow();
        let motions = pack.as_ref().and_then(Fetch::ready)?;
        motions.companion(self.motion.get()?)?.at(at).cloned()
    }

    /// The facial library the motion now playing names, for the packs that name their own rather
    /// than leave it to the timeline they are filed under.
    fn library(&self) -> Option<String> {
        let pack = self.pack.borrow();
        let motions = pack.as_ref().and_then(Fetch::ready)?;
        motions.companion(self.motion.get()?)?.library.clone()
    }

    /// The pack playing, the name its own file gives the motion, and how far into it, in
    /// seconds: for whatever reads a motion's timeline directly rather than through the pose it
    /// drives.
    fn playing(&self) -> Option<(String, String, f32)> {
        let pack = self.pack.borrow();
        let motions = pack.as_ref().and_then(Fetch::ready)?;
        let (name, _) = motions.named.get(self.motion.get()?)?;
        Some((self.wanted.borrow().clone(), name.clone(), self.time.get()))
    }

    /// Runs the clock on by `step`, taking up whatever was queued behind the motion once it has
    /// played through. Nothing queued means it loops, unless the clip was played once, in which
    /// case the layer fades back out from under itself.
    fn advance(&self, step: f32) {
        self.fading(step);
        let Some(duration) = self.duration() else {
            return;
        };
        let time = self.time.get() + step.min(duration);
        if time <= duration {
            self.time.set(time);
            return;
        }
        let then = self.then.borrow_mut().take();
        let settle = self.settle.get();
        match (then, settle > 0.0) {
            (Some(then), _) => self.load(&then, None, None, Some(self.over.get())),
            (None, true) => self.load("", None, None, Some(settle)),
            (None, false) => self.time.set(time - duration),
        }
    }

    /// Runs the fade on by `step`, and the outgoing clip's own clock with it. The fade only opens
    /// once the incoming pack has landed, since a clip that is still being fetched has nothing to
    /// fade towards; a layer with nothing wanted is fading out to whatever is under it and opens
    /// straight away.
    fn fading(&self, step: f32) {
        // A change still waiting on its pack, or on the length the blend table prices it at, has
        // nothing to fade towards yet, so what it left goes on showing whole until it does.
        if self.pricing.get() || (self.motion.get().is_none() && !self.wanted.borrow().is_empty()) {
            return;
        }
        if self.fade.get() >= self.over.get() {
            *self.leaving.borrow_mut() = None;
            return;
        }
        self.fade.set(self.fade.get() + step);
        let mut leaving = self.leaving.borrow_mut();
        if let Some(held) = leaving.as_mut() {
            let duration = held
                .pack
                .binding(held.motion)
                .map_or(f32::EPSILON, |binding| {
                    binding.motion().duration().max(f32::EPSILON)
                });
            // A clip being cross-faded out of is still running, so it wraps; one the layer is
            // leaving with nothing behind it holds its last frame rather than starting over
            // under the fade.
            held.time = match self.wanted.borrow().is_empty() {
                true => (held.time + step).min(duration),
                false => (held.time + step.min(duration)) % duration,
            };
        }
        if self.share() >= 1.0 {
            *leaving = None;
        }
    }

    /// Puts this layer's clock `at` seconds into its clip, wrapped into the clip's own length the
    /// way [`Self::advance`] wraps one that plays through. A layer whose pack has not landed keeps
    /// the clock it had, since nothing yet says how long the clip runs.
    fn run_to(&self, at: f32) {
        let Some(duration) = self.duration() else {
            return;
        };
        self.time.set(at.rem_euclid(duration));
    }

    /// Sets this layer's clock from `at`, the other layer's own, against the window `expression`
    /// states, rather than running one of its own: a facial clip a fraction of a second long
    /// otherwise loops many times over while the body it belongs to plays once.
    fn hold(&self, expression: &Expression, at: f32) {
        let Some(duration) = self.duration() else {
            return;
        };
        self.time.set(held(expression, at, duration));
    }
}

/// Every candidate paired with the `cfxf_` motion an expression is looked for by, which is the
/// same one in whichever of them holds it.
fn opening(candidates: Vec<String>, name: &str) -> Vec<(String, String)> {
    let motion = format!("cfxf_{name}");
    candidates
        .into_iter()
        .map(|path| (path, motion.clone()))
        .collect()
}

/// Where a `duration`-second clip should sit to hold `expression` against `at`, the other clip's
/// own time in seconds: clamped, so it settles at the window's own edge rather than snap back to
/// it before the window opens or past it once it has closed.
fn held(expression: &Expression, at: f32, duration: f32) -> f32 {
    let (start, end) = expression.window;
    let fraction = ((at - start) / (end - start).max(f32::EPSILON)).clamp(0.0, 1.0);
    let (from, to) = expression.span;
    (from + (to - from) * fraction) * duration
}

/// One file on its way in, and what it decoded to.
enum Fetch<T> {
    Fetching(TrackedPromise<Result<Vec<u8>>>),
    Ready(T),
    Failed(String),
}

impl<T> Fetch<T> {
    /// Asks for `path` if nothing has, and reads it once it lands.
    fn poll(
        held: &mut Option<Self>,
        backend: &Backend,
        path: &str,
        read: impl FnOnce(&[u8]) -> Result<T>,
    ) {
        match held {
            None => {
                let files = backend.files().clone();
                let wanted = path.to_owned();
                *held = Some(Self::Fetching(TrackedPromise::spawn_local(async move {
                    files.read(&wanted).await
                })));
            }
            Some(Self::Fetching(promise)) => {
                let Some(result) = promise.try_get() else {
                    return;
                };
                let landed = result
                    .as_ref()
                    .map_err(ToString::to_string)
                    .and_then(|bytes| read(bytes).map_err(|why| why.to_string()));
                *held = Some(match landed {
                    Ok(value) => Self::Ready(value),
                    Err(why) => Self::Failed(why),
                });
            }
            Some(_) => {}
        }
    }

    fn ready(&self) -> Option<&T> {
        match self {
            Self::Ready(value) => Some(value),
            _ => None,
        }
    }
}

/// Every name a pack's own animation table states, which is how a timeline naming a motion says
/// which pack to play it out of, and whether each lays over the pose the body is already in rather
/// than replacing it, which is what the clip's own blend hint states.
pub fn motion_names(bytes: &[u8]) -> Result<Vec<(String, bool)>> {
    let file = AnimationPack::read(Cursor::new(bytes.to_vec()))?;
    let bindings = file.parse_animations().unwrap_or_default();
    Ok(file
        .animations()
        .iter()
        .map(|animation| {
            let at = usize::try_from(animation.havok_index()).unwrap_or(usize::MAX);
            let over = bindings
                .get(at)
                .is_some_and(|binding| binding.blend_hint() == 1);
            (animation.name().to_owned(), over)
        })
        .collect())
}

/// The `cfxf_` names a pap's own animation table states, regardless of whether its own timeline
/// also names a companion.
fn pose_names(bytes: &[u8]) -> Result<Vec<String>> {
    let file = AnimationPack::read(Cursor::new(bytes.to_vec()))?;
    Ok(file
        .animations()
        .iter()
        .filter_map(|animation| animation.name().strip_prefix("cfxf_").map(ToOwned::to_owned))
        .collect())
}

/// What a lookup against [`Poses`] found.
enum PoseLookup {
    /// The index is still being built; ask again once more of it has landed.
    Pending,
    Found(String),
    /// The index finished without ever seeing the name.
    Miss,
}

/// Every `.pap` under a face's own tree, read one at a time and kept for the rest of the
/// session. Built only once a pose neither the filename guess nor the shared library could
/// confirm asks for it, since walking the whole tree costs hundreds of fetches a face rarely
/// needs.
#[derive(Default)]
enum Poses {
    #[default]
    Unbuilt,
    Building {
        queue: Vec<String>,
        current: Option<String>,
        fetch: Option<Fetch<Vec<String>>>,
        found: HashMap<String, String>,
    },
    Ready(HashMap<String, String>),
}

impl Poses {
    /// Looks `name` up, advancing the walk by one fetch if it has not been seen yet. `paths` is
    /// only read on the first call, to seed the walk from the listing already fetched.
    fn advance(&mut self, backend: &Backend, paths: Vec<String>, name: &str) -> PoseLookup {
        if matches!(self, Self::Unbuilt) {
            let mut queue = paths;
            queue.reverse();
            *self = Self::Building {
                queue,
                current: None,
                fetch: None,
                found: HashMap::new(),
            };
        }
        loop {
            match self {
                Self::Ready(found) => {
                    return match found.get(name) {
                        Some(path) => PoseLookup::Found(path.clone()),
                        None => PoseLookup::Miss,
                    };
                }
                Self::Building {
                    queue,
                    current,
                    fetch,
                    found,
                } => {
                    if let Some(path) = found.get(name) {
                        return PoseLookup::Found(path.clone());
                    }
                    let path = match current.clone() {
                        Some(path) => path,
                        None => match queue.pop() {
                            Some(next) => {
                                *current = Some(next.clone());
                                next
                            }
                            None => {
                                let found = std::mem::take(found);
                                *self = Self::Ready(found);
                                continue;
                            }
                        },
                    };
                    Fetch::poll(fetch, backend, &path, pose_names);
                    match fetch.take() {
                        Some(Fetch::Ready(names)) => {
                            for pose in names {
                                found.entry(pose).or_insert_with(|| path.clone());
                            }
                            *current = None;
                        }
                        Some(Fetch::Failed(_)) => *current = None,
                        other => {
                            *fetch = other;
                            return PoseLookup::Pending;
                        }
                    }
                }
                Self::Unbuilt => unreachable!(),
            }
        }
    }
}

/// One pack a model can be played from, as the listing names it.
struct Pack {
    path: String,
    /// What the picker calls it: the path with everything every pack shares cut off the front.
    label: String,
}

/// A rig's own bones, each one's parent, and the matrix that carries a bind-pose vertex into that
/// bone's own rest frame.
pub type RigInfo = (Vec<String>, Vec<Option<usize>>, Vec<Mat4>);

/// What plays a model: the skeleton it is skinned to, the motions laid over it, and the clock.
pub struct Animation {
    /// The `c0101` of the model's own path, which everything it plays is filed under.
    code: Option<String>,
    /// Where the model's own path says its skeleton is, and the file that came of it.
    skeleton: Option<String>,
    base: RefCell<Option<Fetch<Skeleton>>>,
    /// Which extra skeleton each part on screen asks for, out of the parts' own file names.
    needs: RefCell<Vec<(Extra, u16)>>,
    /// The tables naming which skeleton a set is posed on, fetched only where a part needs one.
    tables: RefCell<[Option<Fetch<ExtraSkeletonTemplate>>; 4]>,
    /// Every extra skeleton asked for so far, by path, kept across a change of clothes.
    extras: RefCell<BTreeMap<String, Option<Fetch<Skeleton>>>>,
    /// The rig everything is posed on, and which extras it was built from, so it is built again as
    /// more of them land.
    skin: RefCell<Option<Skin>>,
    built: RefCell<Vec<String>>,
    /// Whether the bones this rig cannot name have been counted since it was last built.
    counted: Cell<bool>,
    /// The bodies to read packs from, nearest first, which is the model's own until it is told
    /// what it is built on.
    built_on: RefCell<Vec<String>>,
    packs: RefCell<Option<Result<Vec<Pack>, Rc<str>>>>,
    /// Cuts the pack list down while the picker is open.
    filter: RefCell<String>,
    /// What the body does, the one-shot laid over it, and the expression over that. A facial
    /// motion states a delta on bones the body's own motions never touch, so the two play at once
    /// rather than in turn; an action is a partial motion that owns the bones it names for as long
    /// as it runs and gives them back to the base once it has.
    body: Layer,
    action: Layer,
    face: Layer,
    /// The `cfxf_` companion last used to drive `face` on the body's own say-so, so a change of it
    /// is what asks for another rather than every frame re-loading the same pack.
    linked: RefCell<Option<String>>,
    /// How long the game blends one motion into another, by the names their own packs give them,
    /// for the caller that has read the tables. Nothing on hand leaves every change a hard cut.
    blend: RefCell<Option<Box<Blend>>>,
    /// The timeline the body's own pack is filed under, and the facial library its `TMPP` names.
    /// A pack that states one of its own needs none of this; the rest leave it to the timeline of
    /// the same key under `chara/action/`.
    keyed: RefCell<String>,
    library: RefCell<Option<Fetch<Option<String>>>>,
    /// Whether a caller has said what the body stands in, so the pack list stops picking for it.
    stood: Cell<bool>,
    /// Whether the face holds a pose the creator asked for by name. The game plays such a pose
    /// from an `ActionTimeline` row of its own, where the face a stance lays over its idle is a
    /// bare command inside the pack with no row at all, so the one does not displace the other.
    picked: Cell<bool>,
    /// Whether `face` is still the pack `linked` put it on, so its clock tracks the body's own
    /// rather than running free. `express` and a manual face pick from the picker both drop this,
    /// since a pose the creator asked for by name is not the body's to hold or let go of.
    synced: Cell<bool>,
    /// A pose `express` or the body's own companion asked for that neither the filename guess
    /// nor the shared library could confirm, waiting on `poses` to be built far enough to answer.
    pending: RefCell<Option<String>>,
    /// Every `.pap` under the face's own tree, read one at a time and kept for the session: the
    /// last resort once a pose's name and its likely files disagree.
    poses: RefCell<Poses>,
    /// What the bust bones are scaled by, three axes in their own frame.
    bust: Cell<Vec3>,
    /// How far a raised visor has turned, one angle per bone it hinges on.
    visor: Cell<[f32; 3]>,
    running: Cell<bool>,
    /// The mount the body is seated on, posed on a rig of its own. A mount names the same bones a
    /// body does, so the two cannot be merged the way an extra skeleton is.
    mounted: Option<Box<Animation>>,
    /// Which seat this rig is in: on a mount, the bone a rider is carried to; on a rider, which
    /// of the mount's own per-seat packs it plays. A mount that seats more than one names a pose
    /// of its own for each.
    seat: Cell<usize>,
}

impl Animation {
    pub fn new<'a>(models: impl IntoIterator<Item = &'a str>) -> Self {
        let models: Vec<&str> = models.into_iter().collect();
        let code = models.iter().find_map(|model| code(model));
        let mount = ridden(code.as_deref(), &models);
        let worn = worn_by(mount.as_deref(), &models);
        Self {
            skeleton: code.as_deref().and_then(skeleton_path),
            base: RefCell::new(None),
            needs: RefCell::new(needed(&worn)),
            tables: Default::default(),
            extras: Default::default(),
            skin: RefCell::new(None),
            built: Default::default(),
            counted: Cell::new(false),
            built_on: RefCell::new(code.iter().cloned().collect()),
            packs: RefCell::new(None),
            filter: RefCell::new(String::new()),
            body: Layer {
                // A guess to stand in until the listing lands and `listed` picks properly: the
                // mount's own seat 0, or the plain idle where there is no mount to guess a code
                // for. Never the whistle a mount is called with; that is not a pose to hold.
                wanted: RefCell::new(
                    code.as_deref()
                        .and_then(|code| match &mount {
                            Some(mount) => {
                                seat_paths(code, mount, 0).pop().or_else(|| pack_path(code))
                            }
                            None => pack_path(code),
                        })
                        .unwrap_or_default(),
                ),
                ..Default::default()
            },
            action: Default::default(),
            face: Default::default(),
            linked: RefCell::new(None),
            blend: RefCell::new(None),
            keyed: RefCell::new(String::new()),
            library: RefCell::new(None),
            stood: Cell::new(false),
            picked: Cell::new(false),
            synced: Cell::new(false),
            pending: RefCell::new(None),
            poses: Default::default(),
            bust: Cell::new(Vec3::ONE),
            visor: Cell::new([0.0; 3]),
            running: Cell::new(true),
            mounted: mount.map(|mount| Box::new(Animation::new(filed_under(&mount, &models)))),
            seat: Cell::new(0),
            code,
        }
    }

    /// The `101` of `c0101`, which is what the extra skeleton tables key their answers on.
    fn body_code(&self) -> Option<u16> {
        self.code.as_deref()?.get(1..)?.parse().ok()
    }

    /// Whether a model is one this body is drawn from, which its file name states. Asked of a
    /// mount, which is the one body that never borrows a model from another.
    fn owns(&self, model: &str) -> bool {
        code(model) == self.code
    }

    /// The mount the body is seated on, where it is on one.
    pub fn rides(&self) -> Option<&str> {
        self.mounted.as_ref()?.code.as_deref()
    }

    /// The rig everything is posed on, once it has landed: its bones, each one's parent, and the
    /// matrix that carries a bind-pose vertex into that bone's own rest frame.
    pub fn rig(&self) -> Option<RigInfo> {
        let skin = self.skin.borrow();
        let skin = skin.as_ref()?;
        let parents = (0..skin.rig.bones())
            .map(|bone| skin.rig.parent(bone))
            .collect();
        Some((skin.rig.names().to_vec(), parents, skin.rest.clone()))
    }

    /// Whether the rigs on hand are the ones a set of models is posed on: the body the first of
    /// them names, and the mount it is ridden on. Neither can be pointed elsewhere once built, so
    /// a change to either is what asks for a rig of its own.
    pub fn poses<'a>(&self, models: impl IntoIterator<Item = &'a str>) -> bool {
        let models: Vec<&str> = models.into_iter().collect();
        let code = models.iter().find_map(|model| code(model));
        let mount = ridden(code.as_deref(), &models);
        code == self.code && mount == self.mounted.as_ref().and_then(|held| held.code.clone())
    }

    /// Points the extra skeletons at what is being worn now, keeping everything already fetched:
    /// a hat that comes back off a picker is not worth asking for twice.
    pub fn rewear<'a>(&self, models: impl IntoIterator<Item = &'a str>) {
        let models: Vec<&str> = models.into_iter().collect();
        let mount = self.mounted.as_ref().and_then(|held| held.code.as_deref());
        if let (Some(mounted), Some(mount)) = (&self.mounted, mount) {
            mounted.rewear(filed_under(mount, &models));
        }
        *self.needs.borrow_mut() = needed(&worn_by(mount, &models));
    }

    /// Asks for the skeleton, the listing and the pack, and takes up whichever has landed. Only
    /// called for a model that carries bone indices, so nothing is fetched for one that could not
    /// be posed.
    pub fn poll(&self, ctx: &egui::Context, backend: &Backend) {
        if let Some(mounted) = &self.mounted {
            mounted.poll(ctx, backend);
        }
        if let Some(path) = &self.skeleton {
            Fetch::poll(&mut self.base.borrow_mut(), backend, path, Skeleton::read);
        }
        self.poll_extras(backend);
        let mut held = self.packs.borrow_mut();
        if held.is_none() {
            *held = match backend.listing(&api_base(ctx)) {
                Listed::Loading => None,
                Listed::Ready(listing) => Some(Ok(self.listed(&listing))),
                Listed::Failed(why) => Some(Err(why)),
            };
        }
        drop(held);
        for layer in self.layers() {
            layer.poll(backend);
        }
        self.poll_fades();
        self.poll_ordering(backend);
        self.poll_library(backend);
        self.poll_companion();
        self.poll_pose(backend);
        if self.running.get() {
            let step = ctx.input(|input| input.stable_dt);
            self.body.advance(step);
            self.action.advance(step);
            // A body command names a window of its own clock to hold the face against rather than
            // let it loop on one of its own; nothing named, or a face the creator has since picked
            // by hand, leaves it free to run on its own clock instead.
            match self.body.expression(self.body.time.get()) {
                Some(expression) if self.synced.get() => {
                    // The clock is the body's, but the fade between one pose and the next is this
                    // layer's own and still has to run.
                    self.face.fading(step);
                    self.face.hold(&expression, self.body.time.get());
                }
                _ => self.face.advance(step),
            }
            // Nothing else asks for a frame while the pointer is still, so playback has to.
            ctx.request_repaint();
        }
    }

    /// The layers in the order they are laid: the base first, then whatever owns the bones it
    /// names over it, then the face over that.
    fn layers(&self) -> [&Layer; 3] {
        [&self.body, &self.action, &self.face]
    }

    /// What to price a change of clip against, which the caller reading the game's own tables
    /// hands over as soon as it has them. Taken once: the tables do not change under a body.
    pub fn blending(&self, blend: impl Fn(&str, &str) -> f32 + 'static) {
        let mut held = self.blend.borrow_mut();
        if held.is_none() {
            *held = Some(Box::new(blend));
        }
    }

    /// Prices whatever change of clip has just landed. The blend table answers to the two motions'
    /// own names, so a layer holds what it was playing until the incoming pack names its own.
    fn poll_fades(&self) {
        let blend = self.blend.borrow();
        let Some(blend) = blend.as_ref() else {
            return;
        };
        for layer in self.layers() {
            if let Some((from, to)) = layer.changed() {
                let over = blend(&from, &to);
                let from = match from.is_empty() {
                    true => "nothing".to_owned(),
                    false => from,
                };
                log::info!("mdl: {from} into {to} blends over {over:.3}s");
                layer.priced(over);
            }
        }
    }

    /// No length of its own where the blend table is on hand to price the change once it lands,
    /// and a hard cut where it is not.
    fn priced(&self) -> Option<f32> {
        self.blend.borrow().is_none().then_some(0.0)
    }

    /// How long a layer fades back out over once nothing is wanted of it: the blend out of the
    /// motion it is playing into none, which unlike a change of clip has both ends named already.
    fn released(&self, layer: &Layer) -> f32 {
        let blend = self.blend.borrow();
        match (blend.as_ref(), layer.playing()) {
            (Some(blend), Some((_, from, _))) => blend(&from, ""),
            _ => 0.0,
        }
    }

    /// Asks for the skeleton each playing motion's tracks are ordered by. A facial motion names a
    /// face skeleton of its own, whose bones the body's skeleton does not carry.
    fn poll_ordering(&self, backend: &Backend) {
        let Some(code) = self.code.as_deref() else {
            return;
        };
        let wanted: Vec<String> = self
            .layers()
            .iter()
            .filter_map(|layer| ordering(code, &layer.wanted.borrow()))
            .collect();
        for path in wanted {
            let mut extras = self.extras.borrow_mut();
            let held = extras.entry(path.clone()).or_default();
            Fetch::poll(held, backend, &path, Skeleton::read);
        }
    }

    /// The bodies to read packs from, nearest first. A body the game files no animation under is
    /// played from the one it is built on, which is the same tree that says where it borrows its
    /// clothes from.
    pub fn built_on(&self, lineage: Vec<String>) {
        if !lineage.is_empty() {
            *self.built_on.borrow_mut() = lineage;
        }
    }

    /// Every pack the lineage this body is built on files, nearest first, opened on the nearest
    /// one's own idle (or its ride pack, mounted). A race rarely authors every motion its own
    /// body plays: `battle_dead_1` ships only under `c0101`, so a Lalafell's own directory alone
    /// would never offer it, and every other body's list is unioned in rather than replaced by
    /// the first non-empty one. Where two bodies both ship a pack of the same name, the nearer's
    /// is kept.
    fn listed(&self, listing: &Listing) -> Vec<Pack> {
        let mut listed: Vec<Pack> = Vec::new();
        let mut named: HashSet<String> = HashSet::new();
        for code in self.built_on.borrow().iter() {
            let Some(root) = pack_root(code) else {
                continue;
            };
            for pack in found(&root, listing.under(&root)) {
                if named.insert(pack.label.clone()) {
                    listed.push(pack);
                }
            }
        }
        listed.sort_by(|left, right| left.label.cmp(&right.label));
        let idle = match self
            .mounted
            .as_ref()
            .and_then(|mounted| mounted.code.as_deref())
        {
            Some(mount) => self.ride_pack(mount, self.seat.get(), &listed),
            None => {
                let exists = |path: Option<String>| {
                    path.filter(|path| listed.iter().any(|pack| pack.path == *path))
                };
                self.built_on
                    .borrow()
                    .iter()
                    .find_map(|code| exists(pack_path(code)))
                    .map(|path| (path, None))
            }
        };
        // The placeholder set at construction is only ever a guess, so the conventional pack
        // always overrides it once the listing is in; a weapon is named none at all, and only
        // then does the listing's own first pack stand in. A caller that has already said what
        // the body stands in keeps it: the conventional idle is the guess, not the answer.
        if let Some((path, motion)) = idle
            .or_else(|| listed.first().map(|pack| (pack.path.clone(), None)))
            .filter(|_| !self.stood.get())
        {
            self.body.load(&path, motion, None, Some(0.0));
        }
        listed
    }

    /// Asks for the tables the parts on screen need, for the skeletons those tables name, and
    /// builds the rig again whenever another of them lands.
    fn poll_extras(&self, backend: &Backend) {
        let Some(body) = self.body_code() else {
            return;
        };
        for kind in Extra::ALL {
            if self.needs.borrow().iter().any(|(held, _)| *held == kind) {
                let mut tables = self.tables.borrow_mut();
                Fetch::poll(&mut tables[kind as usize], backend, kind.table(), |bytes| {
                    Ok(ExtraSkeletonTemplate::read(Cursor::new(bytes.to_vec()))?)
                });
            }
        }
        for path in self.named(body) {
            let mut extras = self.extras.borrow_mut();
            let held = extras.entry(path.clone()).or_default();
            Fetch::poll(held, backend, &path, Skeleton::read);
        }

        let base = self.base.borrow();
        let Some(base) = base.as_ref().and_then(Fetch::ready) else {
            return;
        };
        let extras = self.extras.borrow();
        let landed: Vec<String> = self
            .named(body)
            .into_iter()
            .filter(|path| extras[path].as_ref().and_then(Fetch::ready).is_some())
            .collect();
        if landed == *self.built.borrow() && self.skin.borrow().is_some() {
            return;
        }
        let mut rig = Rig::new(&base.names, &base.parents, &base.reference);
        for path in &landed {
            let Some(held) = extras[path].as_ref().and_then(Fetch::ready) else {
                continue;
            };
            rig = rig.merged(path, &held.names, &held.parents, &held.reference);
        }
        *self.skin.borrow_mut() = Some(Skin::new(rig));
        *self.built.borrow_mut() = landed;
        self.counted.set(false);
    }

    /// Where every extra skeleton the parts need is filed, for the ones whose table has landed and
    /// names one. A set the table says nothing about is worn on the body's own bones.
    fn named(&self, body: u16) -> Vec<String> {
        let tables = self.tables.borrow();
        let mut found: Vec<String> = self
            .needs
            .borrow()
            .iter()
            .filter_map(|(kind, set)| {
                let id = tables[*kind as usize]
                    .as_ref()
                    .and_then(Fetch::ready)?
                    .skeleton(body, *set)
                    .filter(|id| *id > 0)?;
                let (under, letter) = kind.filed();
                Some(format!(
                    "chara/human/c{body:04}/skeleton/{under}/{letter}{id:04}/skl_c{body:04}{letter}{id:04}.sklb"
                ))
            })
            .collect();
        found.sort();
        found.dedup();
        found
    }

    /// What the bust bones are scaled by, which `human.cmp` states as a pair of bounds a slider
    /// runs between.
    pub fn shaped(&self, bust: Vec3) {
        self.bust.set(bust);
    }

    /// How far a raised visor has turned, in radians, one angle per bone it hinges on.
    pub fn hinged(&self, visor: [f32; 3]) {
        self.visor.set(visor);
    }

    /// Which of the mount's own seats the rider takes, for the one that is a mount seating more
    /// than one. A body that is not riding has nowhere to put this. A change of seat asks for the
    /// pose that seat plays rather than waiting for the pack list to notice on its own.
    pub fn seated(&self, seat: usize) {
        if let Some(mounted) = &self.mounted {
            mounted.seat.set(seat);
        }
        let Some(mount) = self
            .mounted
            .as_ref()
            .and_then(|mounted| mounted.code.as_deref())
        else {
            return;
        };
        if self.seat.replace(seat) == seat {
            return;
        }
        let packs = self.packs.borrow();
        if let Some(packs) = packs.as_ref().and_then(|packs| packs.as_ref().ok())
            && let Some((path, motion)) = self.ride_pack(mount, seat, packs)
        {
            self.body.load(&path, motion, None, self.priced());
        }
    }

    /// The pose a mount's own seat plays, out of the packs given: its own, by exact name, where
    /// the mount ships one, else the plain standing idle every body has. Neither is the whistle a
    /// mount is called with, which holds no seated pose at all.
    fn ride_pack(
        &self,
        mount: &str,
        seat: usize,
        packs: &[Pack],
    ) -> Option<(String, Option<&'static str>)> {
        let exists =
            |path: Option<String>| path.filter(|path| packs.iter().any(|pack| pack.path == *path));
        self.built_on
            .borrow()
            .iter()
            .find_map(|code| {
                seat_paths(code, mount, seat)
                    .into_iter()
                    .find_map(|path| exists(Some(path)))
            })
            .map(|path| (path, Some(RIDE_IDLE)))
            .or_else(|| {
                self.built_on
                    .borrow()
                    .iter()
                    .find_map(|code| exists(pack_path(code)))
                    .map(|path| (path, None))
            })
    }

    /// The pack, motion name and time the body is playing, for an emote's own timeline commands
    /// (props, sound, vfx) rather than the face's: those are read against whatever the body is
    /// doing, not the expression laid over it.
    pub fn body_playing(&self) -> Option<(String, String, f32)> {
        self.body.playing()
    }

    /// Plays the first of `packs` the install holds, settling into `then` once it has played
    /// through and cross-fading out of whatever was playing over the length the blend table prices
    /// the change at.
    ///
    /// A pack of facial motions plays over whatever the body is doing rather than in place of it,
    /// so which of the two it lands on is the pack's to say.
    pub fn play(&self, packs: &[String], then: Option<&str>) {
        // Nothing named is nothing to play: what is on screen keeps playing rather than the model
        // dropping to the pose its own file was stored in.
        let Some(first) = packs.first() else {
            return;
        };
        let face = facial(first);
        let fade = self.priced();
        if face {
            self.synced.set(false);
            self.face.plays(packs, then, fade);
        } else {
            self.body.plays(packs, then, fade);
        }
        // Forces the next poll to re-read the companion rather than see the same name it had
        // last time and assume nothing changed, which is what left a re-picked emote's face
        // stuck on whatever frame it was already at.
        *self.linked.borrow_mut() = None;
        // A body motion the creator asked for carries its own face, so it takes the one an
        // expression was holding rather than being refused by it; a facial pack picked by hand
        // is itself the expression.
        self.picked.set(face);
        self.running.set(true);
    }

    /// Stands the body in the first of `poses` whose own pack holds the motion it names,
    /// cross-fading out of whatever it was standing in over `fade` seconds. A base pose is picked
    /// by what the character is doing rather than by what a pack opens on, which is why each names
    /// its motion; a pack that is missing or that ships empty gives way to the next, which is how
    /// a stance with no pose of its own falls back to one that has.
    pub fn stand(&self, poses: &[(String, &str)], fade: f32) {
        let candidates = poses
            .iter()
            .map(|(path, motion)| (path.clone(), (*motion).to_owned()))
            .collect();
        self.stood.set(true);
        self.body.seek(candidates, Some(fade));
        self.running.set(true);
    }

    /// Puts the body's clip `seconds` in rather than wherever wall time has run it to, which is
    /// what a transport seeking by a cutscene's own frame numbering asks for. Read after whatever
    /// asked for the clip, since taking one up is what puts its clock back to nought.
    pub fn plays_at(&self, seconds: f32) {
        self.body.run_to(seconds);
    }

    /// What the body is standing in, by the name its own pack gives the motion.
    pub fn standing(&self) -> Option<String> {
        self.body.playing().map(|(_, name, _)| name)
    }

    /// The motion laid over the body right now, by the name its own pack gives it. Drawing and
    /// sheathing a weapon are what put one there.
    pub fn acting(&self) -> Option<String> {
        self.action.playing().map(|(_, name, _)| name)
    }

    /// Lays `motion` over whatever the body is doing for as long as it runs, out of the first of
    /// `packs` that holds it, fading in and back out over `fade` seconds. A partial motion names
    /// only the bones it moves, so the base keeps every other one for the whole of it.
    pub fn act(&self, packs: &[String], motion: &str, fade: f32) {
        let candidates = packs
            .iter()
            .map(|path| (path.clone(), motion.to_owned()))
            .collect();
        self.action.once(candidates, fade);
        self.running.set(true);
    }

    /// Plays the first of `packs` the install holds over whatever the body is doing rather than in
    /// place of it. An emote played seated or mounted is a partial naming only the bones above the
    /// waist, so the pose the mount holds the rider in shows through everything it leaves alone.
    pub fn play_over(&self, packs: &[String]) {
        if packs.is_empty() {
            return;
        }
        self.action.plays(packs, None, self.priced());
        self.running.set(true);
        *self.linked.borrow_mut() = None;
    }

    /// Puts an expression on the face the character wears. A file's own name is only a guess at
    /// what it holds, so every candidate is opened on the `cfxf_` name itself and skipped if it
    /// does not carry it: the path the game loads a pose on demand from first, then anything else
    /// of that name the listing knows, then the library a face keeps resident, then the rest of
    /// the face's own tree if none of them knew it. A name filed nowhere leaves the face as it
    /// rests, which is the game's own neutral pose.
    pub fn express(&self, name: &str) {
        let Some(root) = self.face_root() else {
            return;
        };
        let file = format!("{name}.pap");
        let asked = format!("{root}nonresident/{file}");
        let listed: Vec<String> = match self.packs.borrow().as_ref() {
            Some(Ok(packs)) => packs
                .iter()
                .filter(|pack| pack.path.starts_with(&root) && file_name(&pack.path) == file)
                .map(|pack| pack.path.clone())
                .filter(|path| *path != asked)
                .collect(),
            _ => Vec::new(),
        };
        let mut candidates = vec![asked];
        candidates.extend(listed);
        candidates.push(format!("{root}resident/face.pap"));
        self.face.seek(opening(candidates, name), self.priced());
        *self.pending.borrow_mut() = Some(name.to_owned());
        self.picked.set(true);
        self.synced.set(false);
        self.running.set(true);
    }

    /// Asks for the timeline the body's own pack is filed under, which is what names the facial
    /// library its poses come out of. A pack that names one of its own, or one whose path holds no
    /// key to look up, needs none of this.
    fn poll_library(&self, backend: &Backend) {
        let wanted = action_key(&self.body.wanted.borrow())
            .map(|key| format!("chara/action/{key}.tmb"))
            .unwrap_or_default();
        let mut keyed = self.keyed.borrow_mut();
        if *keyed != wanted {
            wanted.clone_into(&mut keyed);
            *self.library.borrow_mut() = None;
        }
        drop(keyed);
        if wanted.is_empty() {
            return;
        }
        Fetch::poll(&mut self.library.borrow_mut(), backend, &wanted, |bytes| {
            let timeline = Timeline::read(Cursor::new(bytes.to_vec()))?;
            Ok(timeline.items().iter().find_map(|item| match item {
                Item::FaceLibrary(library) => library.path().map(ToOwned::to_owned),
                _ => None,
            }))
        });
    }

    /// The facial library the body's own motion plays out of, or nothing where it names none.
    /// `None` while the timeline that would name it is still being read, so the face waits for the
    /// answer rather than settling for a guess it would never take back.
    fn face_library(&self) -> Option<Option<String>> {
        if let Some(library) = self.body.library() {
            return Some(Some(library));
        }
        if self.keyed.borrow().is_empty() {
            return Some(None);
        }
        match self.library.borrow().as_ref() {
            Some(Fetch::Ready(library)) => Some(library.clone()),
            Some(Fetch::Failed(_)) => Some(None),
            _ => None,
        }
    }

    /// Drives the face from the `cfxf_` poses the body's own motion lays over it, the way an emote
    /// like Joy carries its own expression rather than leaving the creator to pick one. An emote
    /// runs through several in turn, so this is read against the body's own clock and a change of
    /// pose is what asks for another; a body motion that lays none resets the face to rest rather
    /// than leaving it holding whatever it played last. A pose the creator asked for by name holds
    /// through all of it, since what a stance lays over its own idle is no pose anyone picked.
    fn poll_companion(&self) {
        if self.picked.get() {
            return;
        }
        let wanted = self.body.expression(self.body.time.get());
        let name = wanted.as_ref().map(|held| held.name.clone());
        if name == *self.linked.borrow() {
            return;
        }
        let Some(name) = name else {
            *self.linked.borrow_mut() = None;
            self.synced.set(false);
            self.face.load("", None, None, Some(self.released(&self.face)));
            return;
        };
        let Some(root) = self.face_root() else {
            return;
        };
        let Some(library) = self.face_library() else {
            return;
        };
        let held = self.packs.borrow();
        let Some(Ok(packs)) = held.as_ref() else {
            return;
        };
        let candidates: Vec<String> = library
            .iter()
            .map(|library| format!("{root}nonresident/{library}.pap"))
            .chain([
                format!("{root}nonresident/{name}.pap"),
                format!("{root}resident/face.pap"),
            ])
            .filter(|candidate| packs.iter().any(|pack| pack.path == *candidate))
            .collect();
        drop(held);
        log::info!("mdl: the body lays cfxf_{name} over the face");
        self.face.seek(opening(candidates, &name), self.priced());
        *self.pending.borrow_mut() = Some(name.clone());
        *self.linked.borrow_mut() = Some(name);
        self.synced.set(true);
    }

    /// Falls back to the lazily-built name index for a pose `express` or `poll_companion` could
    /// not confirm any faster way, once the face layer has run out of candidates to try on its
    /// own.
    fn poll_pose(&self, backend: &Backend) {
        let Some(name) = self.pending.borrow().clone() else {
            return;
        };
        if self.face.motion.get().is_some() {
            *self.pending.borrow_mut() = None;
            return;
        }
        if !self.face.spent() {
            return;
        }
        let Some(root) = self.face_root() else {
            *self.pending.borrow_mut() = None;
            return;
        };
        let held = self.packs.borrow();
        let Some(Ok(packs)) = held.as_ref() else {
            return;
        };
        let paths: Vec<String> = packs
            .iter()
            .filter(|pack| pack.path.starts_with(&root) && pack.path.ends_with(".pap"))
            .map(|pack| pack.path.clone())
            .collect();
        drop(held);
        match self.poses.borrow_mut().advance(backend, paths, &name) {
            PoseLookup::Pending => {}
            PoseLookup::Found(path) => {
                self.face.load(&path, Some(&format!("cfxf_{name}")), None, self.priced());
                *self.pending.borrow_mut() = None;
            }
            PoseLookup::Miss => *self.pending.borrow_mut() = None,
        }
    }

    /// Where the packs of the face the character wears are filed.
    fn face_root(&self) -> Option<String> {
        let code = self.code.as_deref()?;
        let body = self.body_code()?;
        let (_, set) = *self
            .needs
            .borrow()
            .iter()
            .find(|(kind, _)| *kind == Extra::Face)?;
        let id = self.tables.borrow()[Extra::Face as usize]
            .as_ref()
            .and_then(Fetch::ready)?
            .skeleton(body, set)
            .filter(|id| *id > 0)?;
        Some(format!("chara/human/{code}/animation/f{id:04}/"))
    }

    /// Where the model stands this frame: a walk of each rig it is drawn on. A mesh is posed by
    /// the rig of the body whose file it came from, and a rider's is then carried to the seat its
    /// mount names.
    pub fn pose(&self, tables: &[Vec<String>], worn: &[&str], skeleton: bool) -> Pose {
        let Some(mounted) = &self.mounted else {
            return self.walked(tables, &[], None, skeleton);
        };
        let ridden: Vec<bool> = worn.iter().map(|path| mounted.owns(path)).collect();
        let rider: Vec<bool> = ridden.iter().map(|held| !held).collect();
        let mount = mounted.walked(tables, &ridden, None, skeleton);
        let mut pose = self.walked(tables, &rider, mount.seat.as_ref(), skeleton);
        for (mesh, joints) in pose.joints.iter_mut().enumerate() {
            if ridden[mesh] {
                joints.clone_from(&mount.joints[mesh]);
            }
        }
        pose.skeleton.extend(mount.skeleton);
        // Both bodies were measured standing at the origin, so the frame is moved half the lift the
        // seat carries the rider by and widened by the other half.
        let lift = mount.seat.map_or(Vec3::ZERO, |seat| seat.translation());
        pose.drift += lift * 0.5;
        pose.stretch += lift.length() * 0.5;
        pose
    }

    /// Where one rig stands this frame, and everything read off it. `poses` says which meshes this
    /// rig answers for, the rest being another's to pose; an empty one is every mesh. `at` carries
    /// the whole rig somewhere, which is where a mount seats its rider.
    fn walked(
        &self,
        tables: &[Vec<String>],
        poses: &[bool],
        at: Option<&Placement>,
        skeleton: bool,
    ) -> Pose {
        let mine = |mesh: usize| poses.get(mesh).copied().unwrap_or(true);
        let skin = self.skin.borrow();
        let Some(skin) = skin.as_ref() else {
            return Pose {
                joints: (0..tables.len())
                    .map(|mesh| match mine(mesh) {
                        true => vec![Mat4::IDENTITY; tables[mesh].len()],
                        false => Vec::new(),
                    })
                    .collect(),
                ..Default::default()
            };
        };
        if !self.counted.replace(true) {
            // A bone the rig cannot name poses nothing and leaves its vertices where the file put
            // them, which is a face standing still while the head it hangs on turns.
            let named: Vec<&Vec<String>> = tables
                .iter()
                .enumerate()
                .filter(|(mesh, _)| mine(*mesh))
                .map(|(_, table)| table)
                .collect();
            let wanted: usize = named.iter().map(|table| table.len()).sum();
            let missing = named
                .iter()
                .flat_map(|table| table.iter())
                .filter(|name| !skin.named.contains_key(*name))
                .count();
            log::info!("mdl: {missing} of {wanted} bones are named by no skeleton");
        }
        let base = self.base.borrow();
        let extras = self.extras.borrow();
        let mut locals = skin.rig.reference().to_vec();
        let mut lay = |path: &str, binding: &Binding, time: f32, weight: f32| {
            // Lalafell file no drawn idle and no draw motion of their own, so both are read out of
            // a body twice their height; taking that body's bone offsets with them is what tore
            // the rig apart.
            let foreign = filed_body(path).is_some_and(|body| Some(body) != self.code.as_deref());
            let ordered = self.code.as_deref().and_then(|code| ordering(code, path));
            let held = match &ordered {
                Some(path) => extras.get(path).and_then(Option::as_ref),
                None => base.as_ref(),
            };
            let Some(names) = held.and_then(Fetch::ready).map(|held| &held.names) else {
                return;
            };
            skin.rig.lay(
                &mut locals,
                binding,
                names,
                Laid {
                    origin: ordered.as_deref(),
                    time,
                    weight,
                    retarget: foreign,
                },
            );
        };
        for layer in self.layers() {
            // The incoming clip is laid over whatever is already there at the share the fade has
            // opened to, so a layer replacing a clip cross-fades and one that had nothing playing
            // fades up out of the layers under it alike.
            let share = layer.share();
            if let Some(leaving) = layer.leaving.borrow().as_ref()
                && let Some(binding) = leaving.pack.binding(leaving.motion)
            {
                // Nothing wanted means the layer is on its way out from over the ones under it, so
                // what is left of the clip it was playing is all there is to lay. A clip that is a
                // delta on what is under it gives up its share as the incoming one takes it, since
                // the two are added rather than one laid over the other.
                let weight = match layer.wanted.borrow().is_empty() || binding.blend_hint() != 0 {
                    true => 1.0 - share,
                    false => 1.0,
                };
                lay(&leaving.path, binding, leaving.time, weight);
            }
            let pack = layer.pack.borrow();
            let Some(binding) = layer
                .motion
                .get()
                .and_then(|motion| pack.as_ref().and_then(Fetch::ready)?.binding(motion))
            else {
                continue;
            };
            lay(&layer.wanted.borrow(), binding, layer.time.get(), share);
        }
        for (name, angle) in VISOR.iter().zip(self.visor.get()) {
            if angle != 0.0
                && let Some(bone) = skin.named.get(*name)
                && let Some(local) = locals.get_mut(*bone)
            {
                let turned = Quat::from_array(local.rotation) * Quat::from_rotation_z(angle);
                local.rotation = turned.to_array();
            }
        }
        let mut posed = skin.rig.world(&locals);
        let bust = self.bust.get();
        if bust != Vec3::ONE {
            for bone in BUST.iter().filter_map(|name| skin.named.get(*name)) {
                posed[*bone] = posed[*bone].scaled(bust);
            }
        }
        let (center, spread) = middle(&posed, skin.anchor);
        // A seat past what this rig's own skeleton names is a vehicle-class mount whose extra
        // riders have no bone of their own; falling back to the first keeps them on the mount at
        // all rather than carrying nothing.
        let seat = skin
            .seats
            .get(self.seat.get())
            .or_else(|| skin.seats.first())
            .map(|bone| posed[*bone]);
        if let Some(at) = at {
            for placement in &mut posed {
                *placement = placement.carried(at);
            }
        }
        Pose {
            joints: (0..tables.len())
                .map(|mesh| match mine(mesh) {
                    true => skin.palette(&tables[mesh], &posed),
                    false => Vec::new(),
                })
                .collect(),
            skeleton: match skeleton {
                true => skin.rig.batches(&posed, None),
                false => Vec::new(),
            },
            drift: center - skin.home,
            stretch: (spread - skin.spread).max(0.0),
            world: posed.iter().map(Placement::matrix).collect(),
            seat,
        }
    }

    /// Which packs are loaded, which motion each of them plays, play and pause, and the scrubber.
    /// Only the pickers are offered until a motion is picked: with none the model stands where its
    /// own file put it, and there is nothing to play.
    pub fn ui(&self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| self.row(ui));
        if let Some(mounted) = &self.mounted {
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new("Mount").strong());
                mounted.row(ui);
            });
        }
    }

    /// One rig's pickers and clock. Everything in here is named after the body it plays, since a
    /// mounted character draws two of these rows.
    fn row(&self, ui: &mut egui::Ui) {
        ui.push_id(self.code.as_deref().unwrap_or_default(), |ui| self.picked(ui));
    }

    fn picked(&self, ui: &mut egui::Ui) {
        self.packs_ui(ui);
        self.body.motion_ui(ui, "mdl_motion");
        self.face.motion_ui(ui, "mdl_face_motion");
        let scrubbed = match self.body.duration() {
            Some(duration) => Some((&self.body, duration)),
            None => self.face.duration().map(|duration| (&self.face, duration)),
        };
        let Some((layer, duration)) = scrubbed else {
            return;
        };
        let running = self.running.get();
        if ui.button(if running { "Pause" } else { "Play" }).clicked() {
            self.running.set(!running);
        }
        let mut time = layer.time.get().clamp(0.0, duration);
        if ui
            .add(
                egui::Slider::new(&mut time, 0.0..=duration)
                    .fixed_decimals(2)
                    .suffix(" s"),
            )
            .changed()
        {
            layer.time.set(time);
        }
    }

    /// Every pack filed under the model's own animation directory. A human carries thousands, so
    /// the list is filtered rather than scrolled.
    fn packs_ui(&self, ui: &mut egui::Ui) {
        let packs = self.packs.borrow();
        let Some(Ok(packs)) = packs.as_ref() else {
            return;
        };
        let held = [
            self.body.wanted.borrow().clone(),
            self.face.wanted.borrow().clone(),
        ];
        let mut picked = None;
        egui::ComboBox::from_id_salt("mdl_pack")
            .selected_text(match packs.iter().find(|pack| pack.path == held[0]) {
                Some(pack) => pack.label.as_str(),
                None => file_name(&held[0]),
            })
            .show_ui(ui, |ui| {
                let mut filter = self.filter.borrow_mut();
                ui.add(
                    egui::TextEdit::singleline(&mut *filter)
                        .desired_width(f32::INFINITY)
                        .hint_text("filter"),
                );
                let matching: Vec<&Pack> = packs
                    .iter()
                    .filter(|pack| pack.label.contains(&*filter))
                    .collect();
                let row = ui.text_style_height(&egui::TextStyle::Body)
                    + ui.spacing().button_padding.y * 2.0;
                egui::ScrollArea::vertical()
                    .max_height(PACK_LIST_HEIGHT)
                    .show_rows(ui, row, matching.len(), |ui, rows| {
                        for pack in &matching[rows] {
                            if ui
                                .selectable_label(held.contains(&pack.path), &pack.label)
                                .clicked()
                            {
                                picked = Some(pack.path.clone());
                            }
                        }
                    });
            });
        if let Some(path) = picked {
            self.play(std::slice::from_ref(&path), None);
        }
    }

    /// The files it is posed from: the skeleton it found, and the pack to take motions out of.
    pub fn details_ui(&self, ui: &mut egui::Ui, follow: &mut Option<String>) {
        section(ui, "Animation");
        match &self.skeleton {
            Some(path) => {
                if link(ui, file_name(path), path) {
                    *follow = Some(path.clone());
                }
                if let Some(Fetch::Failed(why)) = self.base.borrow().as_ref() {
                    ui.label(RichText::new(why).color(Color32::LIGHT_RED));
                }
            }
            None => {
                ui.label(RichText::new("this model's path names no skeleton").weak());
            }
        }
        let mut wanted = self.body.wanted.borrow().clone();
        if ui
            .add(egui::TextEdit::singleline(&mut wanted).hint_text("animation pack"))
            .changed()
        {
            self.body.load(&wanted, None, None, self.priced());
        }
        for layer in self.layers() {
            if let Some(Fetch::Failed(why)) = layer.pack.borrow().as_ref() {
                ui.label(RichText::new(why).color(Color32::LIGHT_RED));
            }
        }
        match self.packs.borrow().as_ref() {
            Some(Ok(packs)) => {
                ui.label(RichText::new(format!("{} packs listed", packs.len())).weak());
            }
            Some(Err(why)) => {
                ui.label(RichText::new(why.as_ref()).color(Color32::LIGHT_RED));
            }
            None => {}
        }
    }
}

/// The models a mount is drawn from, which are the ones filed under its own code.
fn filed_under<'a>(mount: &str, models: &[&'a str]) -> Vec<&'a str> {
    models
        .iter()
        .copied()
        .filter(|model| code(model).as_deref() == Some(mount))
        .collect()
}

/// The models the rider is drawn from, which is everything the mount is not. A body wears models
/// filed under other bodies' codes wherever it ships none of its own, so what a rider is drawn from
/// cannot be read off the codes its files carry.
fn worn_by<'a>(mount: Option<&str>, models: &[&'a str]) -> Vec<&'a str> {
    let Some(mount) = mount else {
        return models.to_vec();
    };
    models
        .iter()
        .copied()
        .filter(|model| code(model).as_deref() != Some(mount))
        .collect()
}

/// The mount a body is being drawn seated on. Only a human rides one, and only one of them is
/// ridden at a time.
fn ridden(rig: Option<&str>, models: &[&str]) -> Option<String> {
    if !rig.is_some_and(|code| code.starts_with('c')) {
        return None;
    }
    models.iter().find_map(|model| {
        code(model).filter(|held| matches!(held.as_bytes().first(), Some(b'm' | b'd')))
    })
}

/// The `m0911` of a model's path, which is what its skeleton and its animations are filed under.
/// The body a pack is filed under, out of its own path, which `code` cannot answer for: an
/// animation names its body in a directory rather than in its file name.
fn filed_body(path: &str) -> Option<&str> {
    let rest = path.strip_prefix("chara/human/")?;
    let held = rest.get(..5)?;
    (held.starts_with('c') && held[1..].bytes().all(|byte| byte.is_ascii_digit())).then_some(held)
}

pub fn code(model: &str) -> Option<String> {
    let name = file_name(model);
    let code = name.get(..5)?;
    let (letter, digits) = code.split_at(1);
    let known = matches!(letter, "c" | "m" | "d" | "w");
    (known && digits.bytes().all(|byte| byte.is_ascii_digit())).then(|| code.to_owned())
}

/// Which extra skeleton each part asks for, out of the parts' own file names. Every model of one
/// face is posed on the same one, so the answers are worth deduplicating before they are looked up.
fn needed(models: &[&str]) -> Vec<(Extra, u16)> {
    let mut found: Vec<_> = models.iter().filter_map(|model| extra(model)).collect();
    found.dedup();
    found
}

/// What one part asks for: `c0101f0002_fac` names the face set it draws, and a piece of equipment
/// names the set it belongs to and, in its suffix, which of the two tables covers that slot.
fn extra(model: &str) -> Option<(Extra, u16)> {
    let name = file_name(model).strip_suffix(".mdl")?;
    let rest = name.get(5..)?;
    let set = rest.get(1..5)?.parse().ok()?;
    let kind = match (rest.as_bytes().first()?, rest.get(5..)?) {
        (b'f', _) => Extra::Face,
        (b'h', _) => Extra::Hair,
        (b'e', "_met") => Extra::Head,
        (b'e', "_top") => Extra::Body,
        _ => return None,
    };
    Some((kind, set))
}

/// The `f0003` of a pack filed under a face skeleton's own directory. Those hold the motions that
/// move a face, and their tracks are ordered by that skeleton's bones rather than the body's.
fn face_set(pack: &str) -> Option<&str> {
    let set = pack.split_once("/animation/")?.1.split('/').next()?;
    let named = set.len() == 5
        && set.starts_with('f')
        && set[1..].bytes().all(|byte| byte.is_ascii_digit());
    named.then_some(set)
}

/// Whether a pack moves a face rather than a body.
fn facial(pack: &str) -> bool {
    face_set(pack).is_some()
}

/// The key a body's pack is filed under, which is what the timeline naming its facial library is
/// filed under too: the path past the animation set and the weapon class, both of which say which
/// body plays the motion rather than which motion it is.
fn action_key(pack: &str) -> Option<&str> {
    let (set, rest) = pack.split_once("/animation/")?.1.split_once('/')?;
    let named =
        set.len() == 5 && set.starts_with('a') && set[1..].bytes().all(|byte| byte.is_ascii_digit());
    named.then_some(())?;
    rest.split_once('/')?.1.strip_suffix(".pap")
}

/// Where the skeleton a pack's tracks are ordered by is filed, or nothing where that is the
/// model's own base skeleton.
fn ordering(code: &str, pack: &str) -> Option<String> {
    let set = face_set(pack)?;
    Some(format!(
        "chara/human/{code}/skeleton/face/{set}/skl_{code}{set}.sklb"
    ))
}

/// Where the model class a code names files its skeletons and animations.
fn tree(code: &str) -> Option<&'static str> {
    match code.as_bytes().first()? {
        b'c' => Some("human"),
        b'm' => Some("monster"),
        b'd' => Some("demihuman"),
        b'w' => Some("weapon"),
        _ => None,
    }
}

fn skeleton_path(code: &str) -> Option<String> {
    let tree = tree(code)?;
    Some(format!(
        "chara/{tree}/{code}/skeleton/base/b0001/skl_{code}b0001.sklb"
    ))
}

/// Where every pack a model class can play is filed, whatever animation set names it.
fn pack_root(code: &str) -> Option<String> {
    Some(format!("chara/{}/{code}/animation/", tree(code)?))
}

/// The pack a model class idles from, which is what stands in until the listing lands. A weapon has
/// none of its own: it is moved by whoever holds it.
fn pack_path(code: &str) -> Option<String> {
    let tree = tree(code)?;
    let resident = match tree {
        "monster" => "monster",
        "weapon" => return None,
        _ => "idle",
    };
    Some(format!(
        "chara/{tree}/{code}/animation/a0001/bt_common/resident/{resident}.pap"
    ))
}

/// The packs a mount names for one of its own seats, in the order to try them. A two-seater's
/// driver leans and sits differently from its passenger, and a bench seating several turns some of
/// them toward the one driving rather than facing forward, so a mount that seats more than one
/// files a numbered pack per seat, 1-based; every other mount files its rider's pose under one
/// unnumbered pack, which a numbered seat also falls back to. `bt_common/mount/mount_start.pap` is
/// not a pose to fall back to at all, whatever seat is asked for, since its one motion is the
/// whistle a mount is called with.
fn seat_paths(code: &str, mount: &str, seat: usize) -> Vec<String> {
    let Some(tree) = tree(code) else {
        return Vec::new();
    };
    let root = format!("chara/{tree}/{code}/animation/a0001/mt_{mount}/resident/mount");
    vec![format!("{root}{:02}.pap", seat + 1), format!("{root}.pap")]
}

/// The packs under a model's animation directory, named by what tells them apart. Every pack of a
/// model sits under the same animation set and the same weapon class, and a segment they all share
/// says nothing; one that would leave a bare file name has gone too far.
fn found(root: &str, paths: Vec<String>) -> Vec<Pack> {
    let mut packs: Vec<Pack> = paths
        .into_iter()
        .filter_map(|path| {
            let label = path.strip_prefix(root)?.strip_suffix(".pap")?.to_owned();
            Some(Pack { path, label })
        })
        .collect();
    packs.sort_by(|left, right| left.label.cmp(&right.label));
    while let Some((head, _)) = packs.first().and_then(|pack| pack.label.split_once('/')) {
        let head = format!("{head}/");
        if !packs.iter().all(|pack| {
            pack.label
                .strip_prefix(&head)
                .is_some_and(|rest| rest.contains('/'))
        }) {
            break;
        }
        for pack in &mut packs {
            pack.label.drain(..head.len());
        }
    }
    packs
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::rc::Rc;

    use glam::{Mat4, Vec3};
    use ironworks::file::File as _;
    use ironworks::file::sklb::Transform;
    use ironworks::file::tmb::{Item, Timeline};

    use super::super::super::skeleton::{Rig, middle};
    use super::{
        Animation, Companion, Expression, Extra, Fetch, Layer, Leaving, Motions, PoseLookup,
        Poses, Skeleton, Skin, action_key, code, extra, facial, found, held, opening, ordering,
        pack_path, pack_root, seat_paths, skeleton_path,
    };

    fn transform(translation: [f32; 3]) -> Transform {
        Transform {
            translation: [translation[0], translation[1], translation[2], 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0, 0.0],
        }
    }

    fn rig() -> Rig {
        Rig::new(
            &["n_root".to_owned(), "j_kubi".to_owned()],
            &[-1, 0],
            &[transform([0.0, 1.0, 0.0]), transform([0.0, 2.0, 0.0])],
        )
    }

    /// A pack under a face skeleton's own directory moves a face, and its tracks are ordered by
    /// that skeleton rather than by the body's.
    #[test]
    fn a_pack_filed_under_a_face_is_ordered_by_it() {
        let face = "chara/human/c0101/animation/f0003/nonresident/smile.pap";
        assert!(facial(face));
        assert_eq!(
            ordering("c0101", face).as_deref(),
            Some("chara/human/c0101/skeleton/face/f0003/skl_c0101f0003.sklb")
        );

        for body in [
            "chara/human/c0101/animation/a0001/bt_common/resident/idle.pap",
            "chara/monster/m0911/animation/a0001/bt_common/resident/monster.pap",
            "chara/weapon/w2616/animation/a0001/wp_common/resident/weapon.pap",
        ] {
            assert!(!facial(body), "{body}");
            assert_eq!(ordering("c0101", body), None, "{body}");
        }
    }

    /// A tail swinging is not the body moving.
    #[test]
    fn a_pose_stands_where_its_anchor_does_however_far_a_tail_swings() {
        let skin = Skin::new(Rig::new(
            &[
                "n_root".to_owned(),
                "n_hara".to_owned(),
                "j_sits".to_owned(),
            ],
            &[-1, 0, 1],
            &[
                transform([0.0, 0.0, 0.0]),
                transform([0.0, 1.0, 0.0]),
                transform([0.0, 0.0, -4.0]),
            ],
        ));
        assert_eq!(skin.anchor, Some(1));

        let mut locals = skin.rig.reference().to_vec();
        locals[2] = transform([3.0, 0.0, -4.0]);
        let swung = skin.rig.world(&locals);
        assert_eq!(middle(&swung, skin.anchor).0, skin.home);
        assert_ne!(
            middle(&swung, None).0,
            middle(&skin.rig.world(skin.rig.reference()), None).0
        );
    }

    /// The rest pose against itself is no movement at all, whatever order the table names the
    /// bones in, and a bone the skeleton does not name stands still.
    #[test]
    fn a_model_at_rest_stands_where_the_file_put_it() {
        let rig = rig();
        let skin = Skin::new(rig);
        let table = ["j_kubi".to_owned(), "n_root".to_owned(), "j_ago".to_owned()];
        let posed = skin.rig.world(skin.rig.reference());
        for joint in skin.palette(&table, &posed) {
            assert!(
                joint.abs_diff_eq(Mat4::IDENTITY, 1e-5),
                "a joint at rest moved: {joint}"
            );
        }
    }

    /// A bone the motion moved carries that movement, and only that bone.
    #[test]
    fn a_posed_bone_carries_what_the_pose_moved_it_by() {
        let rig = rig();
        let skin = Skin::new(rig);
        let mut locals = skin.rig.reference().to_vec();
        locals[1] = transform([0.0, 5.0, 0.0]);
        let posed = skin.rig.world(&locals);
        let held = skin.palette(&["n_root".to_owned(), "j_kubi".to_owned()], &posed);
        assert!(held[0].abs_diff_eq(Mat4::IDENTITY, 1e-5));
        assert_eq!(held[1].w_axis.truncate(), Vec3::new(0.0, 3.0, 0.0));
    }

    /// A mount's seats are `n_mount` and whatever else the skeleton names after it, in the order
    /// it lists them; a name that only starts the same, like a decorative `n_mounted_light`, is
    /// not a seat.
    #[test]
    fn a_mount_names_its_seats_in_skeleton_order() {
        let names = [
            "n_root",
            "n_mount",
            "n_mount_a",
            "n_mounted_light",
            "n_mount_b",
        ]
        .map(ToOwned::to_owned);
        let reference: Vec<_> = names.iter().map(|_| transform([0.0, 0.0, 0.0])).collect();
        let rig = Rig::new(&names, &[-1, 0, 0, 0, 0], &reference);
        let skin = Skin::new(rig);
        assert_eq!(skin.seats, [1, 2, 4]);
    }

    /// A face is skinned to bones the body's own skeleton has never heard of, and its own skeleton
    /// hangs them off one the body does name.
    #[test]
    fn a_face_bone_is_posed_once_its_own_skeleton_is_merged_in() {
        let base = rig();
        assert_eq!(base.bones(), 2);
        let merged = base.merged(
            "face",
            &[
                "j_kubi".to_owned(),
                "j_f_ago".to_owned(),
                "j_nowhere".to_owned(),
                "j_f_orphan".to_owned(),
            ],
            &[-1, 0, -1, 2],
            &[
                // The head where the face's own file put it, which is nowhere near where the
                // body's chain carries it: the base's placement has to win.
                transform([0.0, 9.0, 0.0]),
                transform([0.0, 1.0, 0.0]),
                transform([0.0, 1.0, 0.0]),
                transform([0.0, 1.0, 0.0]),
            ],
        );
        // The body's own bones keep their places, since a motion's tracks name them by index, and
        // a bone hanging off nothing the merge could find is left out rather than put at the origin.
        assert_eq!(merged.names(), ["n_root", "j_kubi", "j_f_ago"]);

        let skin = Skin::new(merged);
        let mut locals = skin.rig.reference().to_vec();
        locals[1] = transform([0.0, 5.0, 0.0]);
        let posed = skin.rig.world(&locals);
        let held = skin.palette(&["j_f_ago".to_owned()], &posed);
        assert_eq!(held[0].w_axis.truncate(), Vec3::new(0.0, 3.0, 0.0));
    }

    /// A mesh's own table still means the body's bone by a name a merge had to keep apart from an
    /// extra's: `Skin`'s own lookup has to agree with `Rig::bone`'s, or a mesh skinned to the
    /// base's `j_ago` would draw off the face's instead of the body's the moment one collided.
    #[test]
    fn a_meshs_bare_lookup_still_means_the_bases_own_bone() {
        let base = rig();
        let merged = base.merged(
            "face",
            &["j_kubi".to_owned(), "j_kubi".to_owned()],
            &[-1, 0],
            &[transform([0.0, 9.0, 0.0]), transform([0.0, 0.5, 0.0])],
        );
        assert_eq!(merged.bones(), 3);
        let base_kubi = merged.bone("j_kubi").expect("the body keeps its own");
        let skin = Skin::new(merged);
        assert_eq!(skin.named["j_kubi"], base_kubi);
    }

    #[test]
    fn a_part_names_the_extra_skeleton_it_is_posed_on() {
        let named = |path| extra(path).map(|(kind, set)| (kind as usize, set));
        assert_eq!(
            named("chara/human/c0101/obj/face/f0002/model/c0101f0002_fac.mdl"),
            Some((Extra::Face as usize, 2))
        );
        assert_eq!(
            named("chara/human/c0101/obj/hair/h0115/model/c0101h0115_hir.mdl"),
            Some((Extra::Hair as usize, 115))
        );
        assert_eq!(
            named("chara/equipment/e0279/model/c0101e0279_met.mdl"),
            Some((Extra::Head as usize, 279))
        );
        assert_eq!(
            named("chara/equipment/e0279/model/c0101e0279_top.mdl"),
            Some((Extra::Body as usize, 279))
        );
        // Gloves are worn on the body's own bones, and so is its own smallclothes top.
        assert_eq!(named("chara/equipment/e0279/model/c0101e0279_glv.mdl"), None);
        assert_eq!(
            named("chara/human/c0101/obj/body/b0001/model/c0101b0001_top.mdl"),
            None
        );
    }

    #[test]
    fn a_model_names_the_files_it_is_animated_from() {
        assert_eq!(
            code("chara/monster/m0911/obj/body/b0001/model/m0911b0001.mdl").as_deref(),
            Some("m0911")
        );
        assert_eq!(
            code("chara/equipment/e0971/model/c0201e0971_top.mdl").as_deref(),
            Some("c0201")
        );
        assert_eq!(
            code("bg/ffxiv/wil_w1/twn/w1t2/bgparts/w1t2_a1_bui1.mdl"),
            None
        );
        assert_eq!(
            skeleton_path("m0911").as_deref(),
            Some("chara/monster/m0911/skeleton/base/b0001/skl_m0911b0001.sklb")
        );
        assert_eq!(
            pack_path("c0101").as_deref(),
            Some("chara/human/c0101/animation/a0001/bt_common/resident/idle.pap")
        );
        assert_eq!(pack_path("w2616"), None);
        assert_eq!(
            pack_root("m0430").as_deref(),
            Some("chara/monster/m0430/animation/")
        );
        assert_eq!(
            seat_paths("c0101", "m0547", 3),
            [
                "chara/human/c0101/animation/a0001/mt_m0547/resident/mount04.pap",
                "chara/human/c0101/animation/a0001/mt_m0547/resident/mount.pap"
            ]
        );
    }

    /// m0430's own directory, which is the shape the pickers are named from: the set and the weapon
    /// class go, and the two `mon_sp001` under different directories stay apart.
    #[test]
    fn packs_are_named_by_what_tells_them_apart() {
        let root = "chara/monster/m0430/animation/";
        let paths = [
            "a0001/bt_common/mon_sp/m0430/hide/mon_sp001.pap",
            "a0001/bt_common/mon_sp/m0430/mon_sp001.pap",
            "a0001/bt_common/resident/monster.pap",
            "a0001/bt_common/warp/warp_start.pap",
            "a0001/bt_common/skl_m0430b0001.sklb",
        ]
        .map(|tail| format!("{root}{tail}"));

        let packs = found(root, paths.to_vec());
        assert_eq!(
            packs.iter().map(|pack| &pack.label).collect::<Vec<_>>(),
            [
                "mon_sp/m0430/hide/mon_sp001",
                "mon_sp/m0430/mon_sp001",
                "resident/monster",
                "warp/warp_start",
            ]
        );
        assert_eq!(
            packs[2].path,
            format!("{root}a0001/bt_common/resident/monster.pap")
        );
    }

    /// Trimming the shared front off one pack would leave a bare file name saying nothing.
    #[test]
    fn a_lone_pack_keeps_the_directory_that_names_it() {
        let root = "chara/weapon/w2616/animation/";
        let packs = found(
            root,
            vec![format!("{root}a0001/wp_common/resident/weapon.pap")],
        );
        assert_eq!(packs[0].label, "resident/weapon");
    }

    #[test]
    fn seek_queues_the_rest_as_retries() {
        let layer = Layer::default();
        layer.seek(opening(vec!["a.pap".to_owned(), "b.pap".to_owned()], "salute"), Some(0.0));
        assert_eq!(*layer.wanted.borrow(), "a.pap");
        assert_eq!(
            *layer.retry.borrow(),
            vec![("b.pap".to_owned(), Some("cfxf_salute".to_owned()))]
        );
        assert_eq!(layer.opening.borrow().as_deref(), Some("cfxf_salute"));
    }

    #[test]
    fn seek_with_nothing_to_try_rests() {
        let layer = Layer::default();
        layer.seek(Vec::new(), Some(0.0));
        assert!(layer.wanted.borrow().is_empty());
        assert!(layer.opening.borrow().is_none());
    }

    #[test]
    fn spent_waits_for_a_landing_with_nothing_left_to_try() {
        let layer = Layer::default();
        assert!(layer.spent(), "nothing wanted yet");
        layer.seek(opening(vec!["a.pap".to_owned()], "salute"), Some(0.0));
        assert!(!layer.spent(), "still fetching, no candidates behind it");
        *layer.pack.borrow_mut() = Some(Fetch::Failed("boom".to_owned()));
        assert!(
            layer.spent(),
            "landed with nothing left to try and no motion found"
        );
    }

    #[test]
    fn spent_stays_false_while_a_retry_is_queued() {
        let layer = Layer::default();
        layer.seek(opening(vec!["a.pap".to_owned(), "b.pap".to_owned()], "salute"), Some(0.0));
        *layer.pack.borrow_mut() = Some(Fetch::Failed("boom".to_owned()));
        assert!(!layer.spent(), "b.pap is still queued behind a.pap");
    }

    /// `c1801`'s own Joy emote, measured against the install: a `Header duration: 122` timeline
    /// naming `C010 { duration: 103, animation_start: 0.0, animation_end: 1.0 }` over a body clip
    /// 4.0 seconds long, held against a face clip 2.0 seconds long. Before the window the face
    /// sits at its own first frame, inside it the two clocks track each other, and past it the
    /// face holds its last rather than snapping back to loop on a clock of its own.
    #[test]
    fn held_tracks_the_bodys_clock_across_the_window_and_clamps_past_it() {
        let scale = 4.0 / 122.0;
        let satisfied = expression("satisfied", 0.0, 103.0 * scale);
        assert_eq!(held(&satisfied, -1.0, 2.0), 0.0);
        assert!((held(&satisfied, 103.0 * scale * 0.5, 2.0) - 1.0).abs() < 1e-4);
        assert!((held(&satisfied, 103.0 * scale, 2.0) - 2.0).abs() < 1e-4);
        assert_eq!(held(&satisfied, 4.0, 2.0), 2.0);
    }

    fn expression(name: &str, from: f32, to: f32) -> Expression {
        Expression {
            name: name.to_owned(),
            window: (from, to),
            span: (0.0, 1.0),
        }
    }

    /// Airquotes, as the install states it: four commands laid over one four-second motion, and
    /// the timeline writes them at 10, 40, 30, 20 rather than in the order the clock reaches
    /// them. The face runs through all four, and before the first it holds the one a turn round
    /// the loop left it on.
    #[test]
    fn the_face_runs_through_every_pose_the_body_lays_over_it() {
        let scale = 4.0 / 120.0;
        let at = |frame: f32| frame * scale;
        let companion = Companion {
            library: Some("emot/airquotes".to_owned()),
            expressions: vec![
                expression("laugh", at(10.0), at(32.0)),
                expression("smile", at(40.0), at(110.0)),
                expression("laugh", at(30.0), at(52.0)),
                expression("smile", at(20.0), at(42.0)),
            ],
        };
        let name = |time: f32| companion.at(time).map(|held| held.name.as_str());
        assert_eq!(name(0.0), Some("smile"), "what the last turn left behind");
        assert_eq!(name(at(10.0)), Some("laugh"));
        assert_eq!(name(at(25.0)), Some("smile"));
        assert_eq!(name(at(35.0)), Some("laugh"));
        assert_eq!(name(at(120.0)), Some("smile"));
        assert_eq!(Companion::default().at(0.0).map(|held| held.name.as_str()), None);
    }

    /// The key a timeline is filed under is the pack path past the animation set and the weapon
    /// class, which is what `chara/action/<key>.tmb` names the facial library under.
    #[test]
    fn a_packs_key_is_what_names_its_facial_library() {
        assert_eq!(
            action_key("chara/human/c0101/animation/a0001/bt_common/emote/airquotes.pap"),
            Some("emote/airquotes")
        );
        assert_eq!(
            action_key("chara/human/c0101/animation/a0001/bt_swd_sld/resident/idle.pap"),
            Some("resident/idle")
        );
        assert_eq!(
            action_key("chara/human/c0101/animation/f0002/nonresident/emot/airquotes.pap"),
            None,
            "a facial pack is filed under the face, not under an animation set"
        );
        assert_eq!(action_key("chara/human/c0101/animation/a0001/bt_common.pap"), None);
    }

    /// A pack holding nothing, for the fade arithmetic, which never reads what is playing.
    fn empty_pack() -> Rc<Motions> {
        Rc::new(Motions {
            named: Vec::new(),
            companions: Vec::new(),
            bindings: Vec::new(),
        })
    }

    fn leaving(layer: &Layer) {
        *layer.leaving.borrow_mut() = Some(Leaving {
            path: "a.pap".to_owned(),
            pack: empty_pack(),
            motion: 0,
            time: 0.0,
        });
    }

    #[test]
    fn a_change_with_no_length_cuts_straight_to_the_new_clip() {
        let layer = Layer::default();
        layer.motion.set(Some(0));
        *layer.pack.borrow_mut() = Some(Fetch::Ready(empty_pack()));
        layer.load("b.pap", None, None, Some(0.0));
        assert!(layer.leaving.borrow().is_none());
        assert_eq!(layer.share(), 1.0);
    }

    /// A pack naming one motion, for the change the blend table is asked to price.
    fn named_pack(name: &str) -> Rc<Motions> {
        Rc::new(Motions {
            named: vec![(name.to_owned(), 0)],
            companions: Vec::new(),
            bindings: Vec::new(),
        })
    }

    /// A change the caller left to the blend table holds what it was playing whole until both
    /// motions are named and a length comes back, and only then opens.
    #[test]
    fn a_change_left_to_the_blend_table_waits_for_its_length() {
        let layer = Layer::default();
        layer.motion.set(Some(0));
        *layer.pack.borrow_mut() = Some(Fetch::Ready(named_pack("cbnm_id0")));
        layer.load("b.pap", None, None, None);
        assert_eq!(layer.share(), 0.0, "none of the incoming clip shows yet");
        assert_eq!(layer.changed(), None, "its pack has not landed to be named");

        *layer.pack.borrow_mut() = Some(Fetch::Ready(named_pack("cbem_sp63")));
        layer.motion.set(Some(0));
        assert_eq!(
            layer.changed(),
            Some(("cbnm_id0".to_owned(), "cbem_sp63".to_owned()))
        );
        assert_eq!(layer.changed(), None, "asked once per change");
        layer.priced(0.4);
        layer.advance(0.2);
        assert_eq!(layer.share(), 0.5);
    }

    /// Candidates for one motion the caller cannot name: the first is what plays, and the rest
    /// wait for it to turn out not to be there.
    #[test]
    fn plays_queues_the_rest_behind_whatever_the_first_pack_opens_on() {
        let layer = Layer::default();
        let candidates = ["class.pap".to_owned(), "common.pap".to_owned()];
        layer.plays(&candidates, None, Some(0.0));
        assert_eq!(*layer.wanted.borrow(), "class.pap");
        assert!(layer.opening.borrow().is_none());
        assert_eq!(*layer.retry.borrow(), vec![("common.pap".to_owned(), None)]);
    }

    #[test]
    fn a_fade_holds_shut_until_the_incoming_pack_lands() {
        let layer = Layer::default();
        leaving(&layer);
        layer.over.set(0.4);
        "b.pap".clone_into(&mut layer.wanted.borrow_mut());
        layer.advance(0.2);
        assert_eq!(layer.share(), 0.0, "nothing has landed to fade towards");
        layer.motion.set(Some(0));
        layer.advance(0.2);
        assert_eq!(layer.share(), 0.5);
        layer.advance(0.2);
        assert!(layer.leaving.borrow().is_none(), "the fade closed");
    }

    #[test]
    fn a_layer_with_nothing_playing_fades_its_clip_up_out_of_the_ones_under_it() {
        let layer = Layer::default();
        layer.once(
            vec![("a.pap".to_owned(), "cbbp_a_activ".to_owned())],
            0.4,
        );
        assert!(layer.leaving.borrow().is_none(), "nothing was playing");
        assert_eq!(layer.share(), 0.0);
        layer.motion.set(Some(0));
        layer.advance(0.2);
        assert_eq!(layer.share(), 0.5);
        layer.advance(0.2);
        assert_eq!(layer.share(), 1.0);
    }

    #[test]
    fn a_released_layer_fades_out_from_over_the_ones_under_it() {
        let layer = Layer::default();
        layer.motion.set(Some(0));
        *layer.pack.borrow_mut() = Some(Fetch::Ready(empty_pack()));
        "a.pap".clone_into(&mut layer.wanted.borrow_mut());
        layer.load("", None, None, Some(0.5));
        assert!(layer.wanted.borrow().is_empty());
        layer.advance(0.25);
        assert_eq!(layer.share(), 0.5, "half of the outgoing clip is left");
        layer.advance(0.25);
        assert!(layer.leaving.borrow().is_none());
    }

    /// Polls a future to completion on the current thread with no real waker, which is enough for
    /// the local install's own I/O: nothing here needs to run concurrently with anything else.
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

    const SQPACK: &str = "/home/asriel/.xlcore/ffxiv/game/sqpack";

    fn local_backend() -> crate::backend::Backend {
        block_on(crate::backend::Backend::new(crate::settings::BackendConfig {
            api_url: "https://exd.camora.dev".to_owned(),
            location: crate::settings::InstallLocation::Sqpack(SQPACK.to_owned()),
            schema: crate::settings::SchemaLocation::Local("/home/asriel/Code/EXDSchema".to_owned()),
        }))
        .unwrap()
    }

    /// Drives a layer's own polling loop against a real backend until it lands on a motion, runs
    /// out of candidates, or the budget below runs out.
    fn settle(layer: &Layer, backend: &crate::backend::Backend) {
        let ctx = egui::Context::default();
        for _ in 0..500 {
            crate::utils::tick_promises(&ctx);
            layer.poll(backend);
            if layer.spent() || layer.motion.get().is_some() {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    /// What picking a real emote lands on: Battle Stance is filed under the class directory the
    /// weapons put the body in and under no other, so the shared directory queued behind it is
    /// never reached; Bee's Knees is only under the shared one, so the class directory ahead of it
    /// is passed over. Each names the motion the change is then priced from.
    #[test]
    #[ignore = "reads the real local FFXIV install"]
    fn a_real_emote_is_played_from_the_class_directory_or_the_one_queued_behind_it() {
        let backend = local_backend();
        let filed =
            |dir: &str, key: &str| format!("chara/human/c0101/animation/a0001/{dir}/{key}.pap");
        let played = |key: &str| {
            let layer = Layer::default();
            layer.plays(&[filed("bt_swd_sld", key), filed("bt_common", key)], None, None);
            settle(&layer, &backend);
            (layer.wanted.borrow().clone(), layer.changed())
        };

        let (path, changed) = played("emote/battle02");
        assert_eq!(path, filed("bt_swd_sld", "emote/battle02"));
        assert_eq!(changed, Some((String::new(), "cbbm_emot02".to_owned())));

        let (path, changed) = played("emote/dance16_loop");
        assert_eq!(path, filed("bt_common", "emote/dance16_loop"));
        assert_eq!(changed, Some((String::new(), "cbem_dance16_2lp".to_owned())));
    }

    /// `salute.pap` really carries `cfxf_bow`, per `dump`ing the real file: exactly the case the
    /// The draw motion played through once: at the end the layer fades out from over the base
    /// holding its last frame, rather than starting the clip over underneath the fade.
    #[test]
    #[ignore = "reads the real local FFXIV install"]
    fn a_one_shot_holds_its_last_frame_while_it_fades_out() {
        let backend = local_backend();
        let layer = Layer::default();
        layer.once(
            vec![(
                "chara/human/c0101/animation/a0001/bt_swd_sld/resident/sub.pap".to_owned(),
                "cbbp_a_activ".to_owned(),
            )],
            0.2,
        );
        settle(&layer, &backend);
        let duration = layer.duration().expect("the draw motion");
        for _ in 0..40 {
            layer.advance(duration / 10.0);
            if layer.wanted.borrow().is_empty() {
                break;
            }
        }
        assert!(layer.wanted.borrow().is_empty(), "it played through");
        let at = |layer: &Layer| layer.leaving.borrow().as_ref().map(|held| held.time);
        let held = at(&layer).expect("fading out from over the base");
        layer.advance(duration / 10.0);
        let after = at(&layer).expect("still fading out");
        assert!(after >= held, "the released clip ran backwards: {held} -> {after}");
        assert!(after <= duration, "past its own end: {after} > {duration}");
    }

    /// filename-first bug got wrong. Seeking it first with `resident/face.pap` behind it, which
    /// does carry `cfxf_salute`, should miss the filename guess and land on the fallback instead.
    #[test]
    #[ignore = "reads the real local FFXIV install"]
    fn a_filename_guess_that_lands_on_the_wrong_pose_falls_back() {
        let backend = local_backend();
        let root = "chara/human/c0101/animation/f0206/";
        let layer = Layer::default();
        layer.seek(
            opening(
                vec![
                    format!("{root}nonresident/emot/salute.pap"),
                    format!("{root}resident/face.pap"),
                ],
                "salute",
            ),
            Some(0.0),
        );
        settle(&layer, &backend);
        assert_eq!(
            *layer.wanted.borrow(),
            format!("{root}resident/face.pap"),
            "the wrong-named guess should have been abandoned"
        );
        assert!(
            layer.motion.get().is_some(),
            "the fallback names cfxf_salute and should have landed on it"
        );
    }

    /// `nonresident/comeon.pap` (not the `emot/` one) is self-consistent: its name really matches
    /// its own `cfxf_comeon`. The filename guess should be kept rather than spent on the fallback.
    #[test]
    #[ignore = "reads the real local FFXIV install"]
    fn a_filename_guess_that_matches_is_kept() {
        let backend = local_backend();
        let root = "chara/human/c0101/animation/f0206/";
        let layer = Layer::default();
        layer.seek(
            opening(
                vec![
                    format!("{root}nonresident/comeon.pap"),
                    format!("{root}resident/face.pap"),
                ],
                "comeon",
            ),
            Some(0.0),
        );
        settle(&layer, &backend);
        assert_eq!(
            *layer.wanted.borrow(),
            format!("{root}nonresident/comeon.pap"),
            "the matching guess should never have been abandoned"
        );
        assert!(layer.motion.get().is_some());
        assert_eq!(
            layer.retry.borrow().len(),
            1,
            "resident/face.pap should still be queued, not yet fetched"
        );
    }

    /// The lazy index has to walk past files that do not carry the name it is after before it
    /// reaches the one that does, and has to come back with a clean miss for a name in none of
    /// them: `act_emot27` names `cfxf_emot_eeh`, which only `nonresident/eeh.pap` carries, while
    /// `loop_emot32_loop`'s `cfxf_lookback_l` is nowhere in the tree at all.
    #[test]
    #[ignore = "reads the real local FFXIV install"]
    fn the_pose_index_walks_past_misses_to_a_real_hit_and_reports_a_true_one() {
        let backend = local_backend();
        let root = "chara/human/c0101/animation/f0206/";
        let paths: Vec<String> = [
            "nonresident/angry.pap",
            "nonresident/bow.pap",
            "nonresident/eeh.pap",
            "nonresident/kiss.pap",
        ]
        .into_iter()
        .map(|tail| format!("{root}{tail}"))
        .collect();

        let mut poses = Poses::default();
        let found = loop {
            match poses.advance(&backend, paths.clone(), "emot_eeh") {
                PoseLookup::Found(path) => break path,
                PoseLookup::Miss => panic!("emot_eeh is really in nonresident/eeh.pap"),
                PoseLookup::Pending => std::thread::sleep(std::time::Duration::from_millis(5)),
            }
        };
        assert_eq!(found, format!("{root}nonresident/eeh.pap"));

        let mut poses = Poses::default();
        loop {
            match poses.advance(&backend, paths.clone(), "lookback_l") {
                PoseLookup::Found(path) => panic!("lookback_l should not exist, found in {path}"),
                PoseLookup::Miss => break,
                PoseLookup::Pending => std::thread::sleep(std::time::Duration::from_millis(5)),
            }
        }
    }

    /// Airquotes off the real install: the emote's own timeline names its facial library through
    /// `chara/action/emote/airquotes.tmb`, and the pack that names holds every pose the emote runs
    /// through. The stance idle a drawn weapon puts a body in names no library of its own and no
    /// timeline under `chara/action/` either, so its own pose has to be found by name instead.
    #[test]
    #[ignore = "reads the real local FFXIV install"]
    fn a_real_airquotes_names_the_library_its_poses_come_out_of() {
        let backend = local_backend();
        let read = |path: &str| block_on(backend.files().read(path)).expect(path);

        let key = action_key("chara/human/c0101/animation/a0001/bt_common/emote/airquotes.pap");
        assert_eq!(key, Some("emote/airquotes"));
        let timeline = read("chara/action/emote/airquotes.tmb");
        let library = Timeline::read(Cursor::new(timeline))
            .expect("a real timeline should parse")
            .items()
            .iter()
            .find_map(|item| match item {
                Item::FaceLibrary(library) => library.path().map(ToOwned::to_owned),
                _ => None,
            });
        assert_eq!(library.as_deref(), Some("emot/airquotes"));

        let motions = Motions::read(&read("chara/human/c0101/animation/a0001/bt_common/emote/airquotes.pap"))
            .expect("a real animation pack should parse");
        let at = motions
            .named
            .iter()
            .position(|(name, _)| name == "cbem_airquotes")
            .expect("cbem_airquotes should be named");
        let companion = motions.companion(at).expect("airquotes lays poses on the face");
        let mut wanted: Vec<&str> = companion
            .expressions
            .iter()
            .map(|held| held.name.as_str())
            .collect();
        wanted.sort_unstable();
        wanted.dedup();
        assert_eq!(wanted, ["laugh", "smile"]);

        let face = Motions::read(&read("chara/human/c0101/animation/f0002/nonresident/emot/airquotes.pap"))
            .expect("the library the timeline names should parse");
        for name in wanted {
            assert!(
                face.named.iter().any(|(held, _)| held == &format!("cfxf_{name}")),
                "the library should hold cfxf_{name}"
            );
        }

        let drawn = Motions::read(&read("chara/human/c0101/animation/a0001/bt_swd_sld/resident/idle.pap"))
            .expect("a real animation pack should parse");
        let at = drawn
            .named
            .iter()
            .position(|(name, _)| name == "cbbm_id0")
            .expect("cbbm_id0 should be named");
        let companion = drawn.companion(at).expect("the drawn idle lays a pose on the face");
        assert_eq!(companion.library, None);
        assert!(block_on(backend.files().read("chara/action/resident/idle.tmb")).is_err());
    }

    /// Drives the base skeleton fetch and the extra-skeleton merge until the rig lands.
    fn settle_rig(animation: &Animation, backend: &crate::backend::Backend) {
        let ctx = egui::Context::default();
        let Some(path) = animation.skeleton.clone() else {
            panic!("a human model always names its own base skeleton");
        };
        for _ in 0..2000 {
            crate::utils::tick_promises(&ctx);
            Fetch::poll(
                &mut animation.base.borrow_mut(),
                backend,
                &path,
                Skeleton::read,
            );
            animation.poll_extras(backend);
            if animation.rig().is_some() {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        panic!("the rig never landed");
    }

    /// Viera's own face skeleton (`c1801f0002`) carries a `j_ago` that is not the body's jaw: the
    /// real merge, off the real install, has to keep both rather than let the face's vanish onto
    /// whichever one the body already named.
    #[test]
    #[ignore = "reads the real local FFXIV install"]
    fn a_real_vieras_face_keeps_its_own_jaw_distinct_from_the_bodys() {
        let backend = local_backend();
        let animation = Animation::new([
            "chara/human/c1801/obj/body/b0001/model/c1801b0001_top.mdl",
            "chara/human/c1801/obj/face/f0002/model/c1801f0002_fac.mdl",
        ]);
        settle_rig(&animation, &backend);
        let (names, parents, _) = animation.rig().expect("settled above");
        let agos: Vec<usize> = names
            .iter()
            .enumerate()
            .filter(|(_, name)| *name == "j_ago")
            .map(|(bone, _)| bone)
            .collect();
        assert_eq!(
            agos.len(),
            2,
            "the body's own jaw and the face's own must both survive the merge, found {names:?}"
        );
        let kao = names
            .iter()
            .position(|name| name == "j_kao")
            .expect("j_kao merges in as the face's own root");
        assert!(
            agos.iter().any(|bone| parents[*bone] == Some(kao)),
            "the face's own j_ago hangs off j_kao, same as the real file states"
        );
    }

    /// `cbem_joy`, off the real install: `Motions::read` has to carry `cfxf_satisfied`'s window
    /// through in seconds, scaled off the timeline's own `Header duration: 122` against `duration:
    /// 103`, rather than the bare frame counts the file states.
    #[test]
    #[ignore = "reads the real local FFXIV install"]
    fn a_real_joy_emote_holds_its_face_across_the_bodys_own_window() {
        let backend = local_backend();
        let path = "chara/human/c1801/animation/a0001/bt_common/emote/joy.pap";
        let bytes =
            block_on(backend.files().read(path)).expect("joy.pap should read off the real install");
        let motions = Motions::read(&bytes).expect("a real animation pack should parse");
        let at = motions
            .named
            .iter()
            .position(|(name, _)| name == "cbem_joy")
            .expect("cbem_joy should be named");
        let companion = motions
            .companion(at)
            .expect("cbem_joy names a facial companion");
        let [held] = companion.expressions.as_slice() else {
            panic!("cbem_joy lays one pose over the face, found {:?}", companion.expressions.len());
        };
        assert_eq!(held.name, "satisfied");
        assert_eq!(held.window.0, 0.0);
        assert_eq!(held.span, (0.0, 1.0));
        let body_duration = motions
            .binding(at)
            .expect("cbem_joy has a binding")
            .motion()
            .duration();
        let expected_end = 103.0 / 122.0 * body_duration;
        assert!(
            (held.window.1 - expected_end).abs() < 1e-4,
            "window end {} should scale duration 103 against Header duration 122, expected {expected_end}",
            held.window.1
        );
    }

    /// An animation names the body it is filed under in a directory, which is what says whether a
    /// rig is wearing its own bone offsets or another body's.
    #[test]
    fn a_pack_states_the_body_it_is_filed_under() {
        use super::filed_body;
        assert_eq!(
            filed_body("chara/human/c0701/animation/a0001/bt_swd_sld/resident/idle.pap"),
            Some("c0701")
        );
        assert_eq!(
            filed_body("chara/human/c0101/animation/a0001/bt_common/emote/clap.pap"),
            Some("c0101")
        );
        // A prop or a mount is filed nowhere near a body, and answers for none.
        assert_eq!(filed_body("chara/weapon/w1980/animation/a0001/idle.pap"), None);
        assert_eq!(filed_body("chara/human/cabbage/animation/x.pap"), None);
    }

    /// What one `cfxf_` pose is worth, off the real install, on the rig the viewer actually poses:
    /// the face skeleton merged into the body's. `grin.pap` holds a single frame of deltas, and
    /// composing it once on the merged rig moves no face bone further than it does on the face
    /// skeleton alone, which is the pose the file states and nothing more.
    #[test]
    #[ignore = "reads the real local FFXIV install"]
    fn a_real_facial_pose_composes_to_what_its_own_file_states() {
        use glam::Quat;
        use ironworks::Ironworks;
        use ironworks::file::est::ExtraSkeletonTemplate;
        use ironworks::file::pap::AnimationPack;
        use ironworks::file::sklb::SkeletonBinary;
        use ironworks::sqpack::{Install, SqPack};

        let install =
            Ironworks::new().with_resource(Box::new(SqPack::new(Install::at_sqpack(SQPACK))));
        let read = |path: &str| install.file::<Vec<u8>>(path).expect(path);
        let parsed = |path: &str| {
            SkeletonBinary::read(Cursor::new(read(path)))
                .expect(path)
                .parse_skeleton()
                .expect("a readable tagfile")
        };
        // A face model ships no animation of its own; `.est` is what says which skeleton poses it,
        // and that is where its expressions are filed. The two rows are the creator's own default
        // and a Rava Viera, which resolve to different skeletons.
        let est: Vec<u8> = install
            .file("chara/xls/charadb/faceSkeletonTemplate.est")
            .expect("the face template");
        let template = ExtraSkeletonTemplate::read(Cursor::new(est)).expect("a readable est");

        for (code, chosen, set, widest) in [(101u16, 5u16, 6u16, 12.0f32), (1801, 1, 2, 32.0)] {
            assert_eq!(template.skeleton(code, chosen), Some(set));
            let scope = format!("f{set:04}");

            let body =
                parsed(&format!("chara/human/c{code:04}/skeleton/base/b0001/skl_c{code:04}b0001.sklb"));
            let face = parsed(&format!(
                "chara/human/c{code:04}/skeleton/face/{scope}/skl_c{code:04}{scope}.sklb"
            ));
            let base = Rig::new(body.bones(), body.parent_indices(), body.reference_pose());
            let merged = base.merged(
                &scope,
                face.bones(),
                face.parent_indices(),
                face.reference_pose(),
            );
            let alone = Rig::new(face.bones(), face.parent_indices(), face.reference_pose());

            let pack = AnimationPack::read(Cursor::new(read(&format!(
                "chara/human/c{code:04}/animation/{scope}/nonresident/grin.pap"
            ))))
            .expect("the pack");
            let bindings = pack.parse_animations().expect("its motions");
            let binding = &bindings[0];
            assert_ne!(binding.blend_hint(), 0, "a facial pack is a pack of deltas");

            let moved = |rig: &Rig, origin: Option<&str>| -> Vec<(String, f32)> {
                let mut locals = rig.reference().to_vec();
                rig.lay(&mut locals, binding, face.bones(), Laid { origin, weight: 1.0, ..Laid::default() });
                let rest = rig.world(rig.reference());
                let posed = rig.world(&locals);
                face.bones()
                    .iter()
                    .filter_map(|name| {
                        let bone = rig
                            .bone(&format!("{scope}\u{0}{name}"))
                            .or_else(|| rig.bone(name))?;
                        let from = rest[bone].matrix().to_scale_rotation_translation().2;
                        let to = posed[bone].matrix().to_scale_rotation_translation().2;
                        Some((name.clone(), from.distance(to)))
                    })
                    .collect()
            };

            // A grin widens the lips and leaves the jaw shut, which is what tells it from a laugh.
            // How far the lips go is stated per face; the jaw holds on both.
            let (mut turned, mut jaw) = (0.0f32, 0.0f32);
            {
                let mut locals = alone.reference().to_vec();
                alone.lay(&mut locals, binding, face.bones(), Laid { weight: 1.0, ..Laid::default() });
                for (at, local) in locals.iter().enumerate() {
                    let from = Quat::from_array(alone.reference()[at].rotation);
                    let by = from.angle_between(Quat::from_array(local.rotation));
                    turned = turned.max(by);
                    if face.bones()[at].ends_with("ago") {
                        jaw = jaw.max(by);
                    }
                }
            }
            assert!(
                turned.to_degrees() < widest,
                "c{code:04} turns {} deg, over the {widest} its lips state",
                turned.to_degrees()
            );
            assert!(jaw.to_degrees() < 0.1, "c{code:04} opens its jaw {} deg", jaw.to_degrees());

            let held = moved(&alone, None);
            let over = moved(&merged, Some(&scope));
            assert_eq!(held.len(), face.bones().len());
            for ((name, one), (_, two)) in held.iter().zip(&over) {
                assert!(
                    (one - two).abs() < 1e-4,
                    "{name} moves {one} on its own skeleton and {two} merged"
                );
                assert!(*one < 0.02, "c{code:04} {name} moves {one} m, which is not a face");
            }
        }
    }
}
