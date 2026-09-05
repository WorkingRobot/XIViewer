//! Plays a cutscene's own camera over the level its `CTDS` names: the shots each `CTTL` states,
//! sequenced in the order the file lists them since nothing else states another order.
//!
//! The camera comes from `C004` plus the `TMFC` curve set its `curve_id` names. Its targets carry a
//! role apiece and hang off one another: the eye stands where the last [`EYE`] target does, aimed
//! at the last [`LOOK_AT`] one with the last [`UP`] one over it, and each of those rides whichever
//! `CTAL` participant the shot's own bindings name. See the ironworks `C004` doc for the rest,
//! including the focal length and roll fields on the set's target `0xff`.
//!
//! Each actor is driven off the same timelines: a `TMAC` names the `CTAL` participant its tracks
//! run against, and the commands they reach place it (`C018`), play a motion on its body (`C010`,
//! `C040`) and put an expression on its face (`C090`). A motion is named rather than filed, so
//! which pack holds it is looked up against the `.pap` files the cutscene's own `CTRL` loads.

use std::cell::RefCell;
use std::collections::BTreeMap;

use egui::{
    Align, Button, CentralPanel, Color32, FontId, Layout, Rect, RichText, ScrollArea,
    containers::panel::Panel, pos2, vec2,
};
use glam::{Mat4, Quat, Vec3, Vec4};
use ironworks::excel::Language;
use ironworks::file::cutb::{Cutscene, Node};
use ironworks::file::layer::{HelperKind, HelperObject, Instance, InstanceData, Transform};
use ironworks::file::lvb::LevelFile;
use ironworks::file::tmb::{Channel, CommandKind, Curves, Item, Timeline};

use super::{music, sound};
use crate::assets::viewers::layer;
use crate::assets::viewers::layer::scene;
use crate::backend::Backend;
use crate::character::stand;
use crate::data::FileProviderExt;
use crate::excel::provider::{ExcelProvider, ExcelSheet};
use crate::quests::sestring;
use crate::settings::LANGUAGE;
use crate::sheet::SheetColumnDefinition;
use crate::utils::{PromiseKind, TrackedPromise};

/// The roles a camera's curve set gives its targets. The frame the rest hang off is role 1, which
/// the shot binds but the camera never reads a position off.
const EYE: u8 = 2;
const LOOK_AT: u8 = 3;
const UP: u8 = 4;

/// Which `C004` binding names each role's participant, and where the flag holding role 1 to a
/// participant's position alone sits. Each role spends five fields: two participants with a
/// sub-index apiece, then that flag.
const ROLES: [(u8, usize); 3] = [(1, 0), (EYE, 6), (LOOK_AT, 11)];
const RIG_UPRIGHT: usize = 4;

/// Where the flag holding a shot to the bind it opened with sits, past role 1's own five fields.
/// Nought in nine shots in ten, which is a camera that re-binds every frame.
const HELD_BIND: usize = 5;

/// How far a target's parents are followed, past which a file naming a loop of them stops rather
/// than hangs. Deeper than any set the game ships.
const DEPTH: u8 = 16;

/// Where a set's own fields sit, past its targets' transform channels.
const CAMERA_FIELDS: u8 = 0xFF;
const FOCAL_LENGTH_TAG: u8 = 0x34;
const ROLL_TAG: u8 = 0x35;

/// Half the sensor height the game turns a focal length into a vertical field of view against, at
/// a frame it fixes at sixteen by nine.
const HALF_SENSOR: f32 = 7.001_51;

/// Which channel of a `C049`'s curve set carries each part of the effect's color. The client
/// hands them to the same setter a placed effect's own `Rgba` reaches, in this order.
const RED: u8 = 0x0C;
const GREEN: u8 = 0x0B;
const BLUE: u8 = 0x0A;
const ALPHA: u8 = 0x0D;

/// Frames a second. A cutscene's own numbering runs at this: a `C010` naming `cbfm_arms` ends at
/// frame 155, and that pack's own binding gives the clip 5.1666665 seconds.
pub(super) const FRAMES_A_SECOND: f32 = 30.0;

/// A camera pose, already in world space and degrees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    pub position: Vec3,
    pub forward: Vec3,
    pub up: Vec3,
    pub fov_degrees: f32,
    pub near: f32,
    pub far: f32,
}

impl Pose {
    pub fn drive(self) -> scene::Drive {
        scene::Drive {
            position: self.position,
            forward: self.forward,
            up: self.up,
            fov_degrees: self.fov_degrees,
            near: self.near,
            far: self.far,
        }
    }
}

/// The vertical field of view a focal length turns into, in degrees.
fn field_of_view_degrees(focal_mm: f32) -> f32 {
    (2.0 * (HALF_SENSOR / focal_mm).atan()).to_degrees()
}

/// The up vector a roll leaves, turned about the eye's own forward axis. A positive roll takes it
/// towards the eye's right, which is the other way round from how the file states it.
fn banked(forward: Vec3, up: Vec3, roll_deg: f32) -> Vec3 {
    Quat::from_axis_angle(forward, roll_deg.to_radians()) * up
}

fn curve_value(set: &Curves, target: u8, channel: Channel, time: f32) -> f32 {
    set.channel(target, channel)
        .and_then(|curve| curve.at(time))
        .unwrap_or(0.0)
}

fn camera_field(set: &Curves, tag: u8, time: f32) -> Option<f32> {
    set.curves()
        .iter()
        .find(|curve| curve.target() == CAMERA_FIELDS && curve.tag() & 0x3F == tag)
        .and_then(|curve| curve.at(time))
}

/// The color a `C049`'s curve set states at a time. Its five channels sit on the set's one
/// target; the fifth reaches nothing the client draws with.
fn lit(set: &Curves, time: f32) -> Vec4 {
    let channel = |tag: u8| {
        set.curves()
            .iter()
            .find(|curve| curve.target() == 0 && curve.tag() & 0x3F == tag)
            .and_then(|curve| curve.at(time))
            .unwrap_or(1.0)
    };
    Vec4::new(
        channel(RED),
        channel(GREEN),
        channel(BLUE),
        channel(ALPHA).clamp(0.0, 1.0),
    )
}

/// One target of a camera's curve set: what it stands for, what it hangs off, and where the
/// participant its role binds stands, with whether it turns with that participant as well.
struct Target {
    role: u8,
    parent: Option<u8>,
    bound: Option<(Mat4, bool)>,
}

/// The targets a shot's curve set drives, with each role's binding resolved onto the first target
/// carrying it - the only one the shot binds.
fn rig(
    set: &Curves,
    bindings: &[u32; 17],
    participants: &[Instance],
    placed: &dyn Fn(u32) -> Option<Transform>,
) -> BTreeMap<u8, Target> {
    let mut targets = BTreeMap::new();
    for curve in set.curves().iter().filter(|curve| curve.target() != CAMERA_FIELDS) {
        targets.entry(curve.target()).or_insert(Target {
            role: curve.role(),
            parent: curve.parent(),
            bound: None,
        });
    }
    for (role, slot) in ROLES {
        // The second participant of a pair stands in where the first names nothing: the game skips
        // a role's binding only when neither of the two resolves.
        let Some(participant) = [bindings[slot], bindings[slot + 2]]
            .into_iter()
            .find_map(|id| participants.iter().find(|held| held.id() == id))
        else {
            continue;
        };
        let Some(target) = targets.values_mut().find(|target| target.role == role) else {
            continue;
        };
        let turns = role == ROLES[0].0 && bindings[RIG_UPRIGHT] != 1;
        let at = placed(participant.id()).unwrap_or_else(|| stands_at(participant));
        target.bound = Some((scene::matrix(at), turns));
    }
    targets
}

/// Where a target stands at a time, in the frame its parents and its own binding put it in. A
/// bound target that does not turn keeps its parent's facing and takes only the participant's
/// position.
fn world(
    set: &Curves,
    targets: &BTreeMap<u8, Target>,
    index: u8,
    time: f32,
    depth: u8,
) -> Mat4 {
    let Some(target) = targets.get(&index) else {
        return Mat4::IDENTITY;
    };
    let channels = |channels: [Channel; 3]| {
        Vec3::from_array(channels.map(|channel| curve_value(set, index, channel, time)))
    };
    let local = Mat4::from_rotation_translation(
        Quat::from_mat3(&scene::rotation(
            channels([Channel::RotationX, Channel::RotationY, Channel::RotationZ])
                .to_array()
                .map(f32::to_radians),
        )),
        channels([
            Channel::TranslationX,
            Channel::TranslationY,
            Channel::TranslationZ,
        ]),
    );
    let parent = match target.parent.filter(|_| depth < DEPTH) {
        Some(parent) => world(set, targets, parent, time, depth + 1),
        None => Mat4::IDENTITY,
    };
    frame(parent, target.bound) * local
}

/// The frame a target's own channels sit in: the placement its role binds, whole where the role
/// turns with the participant and as its position over the parent's own facing where it does not.
fn frame(parent: Mat4, bound: Option<(Mat4, bool)>) -> Mat4 {
    match bound {
        Some((placement, true)) => placement,
        Some((placement, false)) => Mat4::from_cols(
            parent.x_axis,
            parent.y_axis,
            parent.z_axis,
            placement.w_axis,
        ),
        None => parent,
    }
}

/// Where the last target of a role stands, which is the one the camera reads: the game walks its
/// targets from the end.
fn stands(set: &Curves, targets: &BTreeMap<u8, Target>, role: u8, time: f32) -> Option<Vec3> {
    let index = *targets
        .iter()
        .rev()
        .find(|(_, target)| target.role == role)?
        .0;
    Some(world(set, targets, index, time, 0).w_axis.truncate())
}

/// The camera's pose at a time within the shot's own span, held past either end the way
/// [`ironworks::file::tmb::Curve::at`] holds a curve.
fn eye_pose(
    set: &Curves,
    targets: &BTreeMap<u8, Target>,
    time: f32,
    near: f32,
    far: f32,
) -> Option<Pose> {
    let position = stands(set, targets, EYE, time)?;
    let forward = (stands(set, targets, LOOK_AT, time)? - position).normalize_or_zero();
    let up = stands(set, targets, UP, time)
        .map(|over| over - position)
        .unwrap_or(Vec3::Y);
    let roll = camera_field(set, ROLL_TAG, time).unwrap_or(0.0);
    let fov_degrees = camera_field(set, FOCAL_LENGTH_TAG, time)
        .filter(|focal| *focal > 0.0)
        .map(field_of_view_degrees)
        .unwrap_or(55.0);
    Some(Pose {
        position,
        forward,
        up: banked(forward, up, roll),
        fov_degrees,
        near,
        far,
    })
}

/// What a cutscene's timelines ask of one participant, in the cutscene's own global frame
/// numbering. Each list is what holds from its own time on, so the one to run at a time is the
/// last to have started.
#[derive(Default)]
pub struct Part {
    /// Where it stands. Empty where nothing places it, which leaves its `CTAL` record's own
    /// transform standing.
    placed: Vec<(f32, Transform)>,
    /// The motion its body plays.
    motions: Vec<(f32, Cue)>,
    /// The `cfxf_` expression its face wears.
    faces: Vec<(f32, String)>,
    /// The effects it fires.
    effects: Vec<(f32, Burst)>,
    /// Whether it is drawn. Empty where nothing states it, which leaves it drawn.
    shown: Vec<(f32, bool)>,
    /// The fades it runs. Empty where nothing states one, which leaves it whole.
    faded: Vec<(f32, Fade)>,
}

/// One `C094`: what the participant fades from and to, and over how many of the cutscene's own
/// frames. The client ramps it linearly and holds either end past it, and the four parts its
/// filter can pick between are the same in every file that names one, so nothing here splits them.
struct Fade {
    from: f32,
    to: f32,
    over: f32,
}

impl Fade {
    /// How much is drawn `along` frames into the fade.
    fn at(&self, along: f32) -> f32 {
        let held = (along / self.over.max(f32::EPSILON)).clamp(0.0, 1.0);
        self.from + (self.to - self.from) * held
    }
}

/// One effect a timeline fires on a participant: which file, the curve set stating its color, and
/// where the timeline holding it ends, past which the effect is done.
struct Burst {
    path: String,
    node: usize,
    curves: i16,
    until: f32,
}

/// One motion a timeline names for a body: which, how far into the clip to open, in seconds, and
/// how long the command runs for, in the cutscene's own frames. Only a motion laid over the pose
/// reads the last: what replaces the pose holds until something else replaces it.
struct Cue {
    motion: String,
    from: f32,
    runs: f32,
}

/// One `C048` subtitle: when it stands, the row it names, and how long it stands for.
pub struct Subtitle {
    /// When it stands, in the cutscene's own global frame numbering.
    pub at: f32,
    /// The key of the row it names, in the sheet the cutscene's `CTIS` node holds.
    pub key: String,
    /// Who says it. The client's own parsers stop at the line number, so this is the viewer's
    /// reading of the rest of the key rather than a field.
    pub speaker: String,
    /// How long the line stands in each language, in milliseconds, nought where that language
    /// states nothing.
    lengths: Vec<i32>,
}

impl Subtitle {
    /// How long the line stands in a language, in seconds.
    pub fn runs(&self, language: Language) -> f32 {
        caption_slot(language)
            .and_then(|slot| self.lengths.get(slot))
            .map(|length| *length as f32 / 1000.0)
            .unwrap_or(0.0)
    }
}

/// Who a key says the line: what it names past the sheet's own id, skipping the line number, which
/// a key writes on either side of the name.
fn speaker_of(key: &str, sheet_upper: &str) -> String {
    let rest = key.strip_prefix("TEXT_").unwrap_or(key);
    let rest = rest
        .strip_prefix(sheet_upper)
        .and_then(|rest| rest.strip_prefix('_'))
        .unwrap_or(rest);
    rest.split('_')
        .find(|part| !part.bytes().all(|byte| byte.is_ascii_digit()))
        .unwrap_or_default()
        .to_owned()
}

/// The id a cutscene's line keys open with, which is the last part of the sheet its `CTIS` names.
fn sheet_id(sheet: Option<&str>) -> String {
    sheet
        .and_then(|name| name.rsplit('/').next())
        .unwrap_or_default()
        .to_uppercase()
}

/// Which of a `C048`'s captions a language reads: the client indexes them by the language itself,
/// in the order `ja`, `en`, `de`, `fr`, `chs`, a slot it rejects, `ko`, `tc`.
fn caption_slot(language: Language) -> Option<usize> {
    Some(match language {
        Language::Japanese => 0,
        Language::English => 1,
        Language::German => 2,
        Language::French => 3,
        Language::ChineseSimplified => 4,
        Language::Korean => 6,
        _ => return None,
    })
}

impl Part {
    /// Puts each list in the order it runs. A timeline lists its commands in neither the order it
    /// plays them nor any order at all, so what holds at a time cannot be read off one as it comes
    /// out of the file.
    /// The fade holding at a time, and how far into it that time is: the last one to have started,
    /// since a fade holds its own end until another replaces it. The client multiplies every fade
    /// running on one object together; this reads the last alone, which is the same answer wherever
    /// two of them do not overlap.
    fn fading(&self, time: f32) -> Option<(usize, &Fade, f32)> {
        let (index, (at, fade)) = self
            .faded
            .iter()
            .enumerate()
            .rev()
            .find(|(_, (at, _))| *at <= time)?;
        Some((index, fade, time - at))
    }

    fn order(&mut self) {
        self.placed.sort_by(|left, right| left.0.total_cmp(&right.0));
        self.motions.sort_by(|left, right| left.0.total_cmp(&right.0));
        self.faces.sort_by(|left, right| left.0.total_cmp(&right.0));
        self.effects.sort_by(|left, right| left.0.total_cmp(&right.0));
        self.shown.sort_by(|left, right| left.0.total_cmp(&right.0));
        self.faded.sort_by(|left, right| left.0.total_cmp(&right.0));
    }
}

/// Which motion holds at a time on one of the two layers a body plays on: the last to have
/// started, of the ones that lay over the pose or of the ones that replace it.
///
/// A motion that replaces the pose holds until another replaces it; one that lays over it runs
/// only for as long as its own command states, past which nothing lays over anything.
fn started<'a>(
    motions: &'a [(f32, Cue)],
    time: f32,
    over: bool,
    lays_over: &dyn Fn(&str) -> bool,
) -> Option<(usize, &'a Cue)> {
    motions
        .iter()
        .enumerate()
        .rev()
        .find(|(_, (at, held))| {
            *at <= time && lays_over(&held.motion) == over && (!over || time < at + held.runs)
        })
        .map(|(index, (_, held))| (index, held))
}

/// Which of a list's entries holds at a time: the last to have started.
fn latest<T>(held: &[(f32, T)], time: f32) -> Option<(usize, &T)> {
    held.iter()
        .enumerate()
        .filter(|(_, (at, _))| *at <= time)
        .last()
        .map(|(index, (_, held))| (index, held))
}

/// The participant each of a timeline's commands runs against, out of the actors it drives.
fn addressed(timeline: &Timeline) -> BTreeMap<i16, u32> {
    let tracks: BTreeMap<i16, &[i16]> = timeline
        .items()
        .iter()
        .filter_map(|item| match item {
            Item::Track(track) => Some((track.id(), track.commands())),
            _ => None,
        })
        .collect();
    let mut held = BTreeMap::new();
    for item in timeline.items() {
        let Item::Actor(actor) = item else { continue };
        for track in actor.tracks() {
            for command in tracks.get(track).into_iter().flat_map(|held| held.iter()) {
                held.insert(*command, actor.participant());
            }
        }
    }
    held
}

/// Files a named motion where it belongs: a face wears the poses its own packs name and a body
/// plays the rest. A `cfx` name that is not a pose is the face's own blink or lip clip, which the
/// body would only corrupt, so nothing plays it.
fn cue(part: &mut Part, at: f32, motion: Option<&str>, from: f32, runs: f32) {
    let Some(motion) = motion.filter(|motion| !motion.is_empty()) else {
        return;
    };
    match motion.strip_prefix("cfxf_") {
        Some(pose) => part.faces.push((at, pose.to_owned())),
        None if motion.starts_with("cfx") => {}
        None => part.motions.push((
            at,
            Cue {
                motion: motion.to_owned(),
                from,
                runs,
            },
        )),
    }
}

/// Reads one timeline's commands into the parts its participants play, offset into the cutscene's
/// own global frame numbering.
fn parts_of(
    timeline: &Timeline,
    node: usize,
    offset: f32,
    span: f32,
    parts: &mut BTreeMap<u32, Part>,
) {
    let addressed = addressed(timeline);
    for item in timeline.items() {
        let Item::Command(command) = item else {
            continue;
        };
        let Some(participant) = addressed.get(&command.id()) else {
            continue;
        };
        let at = offset + f32::from(command.time());
        let part = parts.entry(*participant).or_default();
        match command.kind() {
            CommandKind::C018(placed) => part.placed.push((
                at,
                Transform::new(placed.translation(), placed.rotation(), placed.scale()),
            )),
            // `0x01` enables the start and end frames; without it the clip plays from its own
            // start.
            CommandKind::C010(play) => {
                let from = match play.flags() & 0x01 != 0 {
                    true => play.animation_start() / FRAMES_A_SECOND,
                    false => 0.0,
                };
                cue(part, at, play.motion(), from, play.duration().max(0) as f32);
            }
            // Neither states how long it runs: a `C040` opens with a one where a `C010` opens with
            // its own length, and only a motion laid over the pose has an end of its own.
            CommandKind::C040(play) => cue(part, at, play.motion(), 0.0, f32::INFINITY),
            CommandKind::C090(wear) => cue(part, at, wear.motion(), 0.0, f32::INFINITY),
            CommandKind::C049(effect) => {
                if let Some(path) = effect.path().filter(|path| !path.is_empty()) {
                    part.effects.push((
                        at,
                        Burst {
                            path: path.to_owned(),
                            node,
                            curves: effect.curve_id().try_into().unwrap_or(0),
                            until: offset + span,
                        },
                    ));
                }
            }
            CommandKind::C019(state) => part.shown.push((at, state.visibility() & 0xFF != 0)),
            CommandKind::C094(fade) => part.faded.push((
                at,
                Fade {
                    from: fade.start_visibility(),
                    to: fade.end_visibility(),
                    // The clip takes its length off the low half of the field, as every command
                    // whose first word is a duration does.
                    over: f32::from(fade.fade_time() as u16),
                },
            )),
            _ => {}
        }
    }
}

/// The subtitles one `CTTL` states, offset into the cutscene's own global frame numbering.
fn subtitles_of(timeline: &Timeline, offset: f32, sheet_upper: &str, held: &mut Vec<Subtitle>) {
    for item in timeline.items() {
        let Item::Command(command) = item else {
            continue;
        };
        let CommandKind::C048(subtitle) = command.kind() else {
            continue;
        };
        let Some(key) = subtitle.key().filter(|key| !key.is_empty()) else {
            continue;
        };
        held.push(Subtitle {
            at: offset + f32::from(command.time()),
            speaker: speaker_of(key, sheet_upper),
            key: key.to_owned(),
            lengths: subtitle
                .captions()
                .iter()
                .map(|caption| match caption.enabled() != 0 {
                    true => caption.duration(),
                    false => 0,
                })
                .collect(),
        });
    }
}

/// Every `C063` a timeline plays, as a cue naming the container and the entry inside it. The
/// command files its own path, so nothing has to be derived.
fn sounds_of(timeline: &Timeline, offset: f32, held: &mut Vec<sound::Cue>) {
    for item in timeline.items() {
        let Item::Command(command) = item else {
            continue;
        };
        let CommandKind::C063(played) = command.kind() else {
            continue;
        };
        let Some(path) = played.path().filter(|path| !path.is_empty()) else {
            continue;
        };
        let Ok(entry) = usize::try_from(played.sound_index()) else {
            continue;
        };
        held.push(sound::Cue {
            at: offset + f32::from(command.time()),
            paths: vec![path.to_owned()],
            entry,
            label: "effect".to_owned(),
            holds: None,
        });
    }
}

/// One shot: a `C004` command and the `CTTL` node it came from, in the cutscene's own global
/// frame numbering (its segment's own start added on).
pub struct Shot {
    pub node: usize,
    pub name: Option<String>,
    pub start: f32,
    pub duration: f32,
    curves: i16,
    bindings: [u32; 17],
    near: f32,
    far: f32,
}

impl Shot {
    /// When to read the placement of whatever the shot binds: where the actor stands now, unless
    /// the shot says to hold the bind it opened with. Nine shots in ten re-bind, so a camera
    /// riding someone who walks follows rather than lags a whole shot behind.
    fn bound_at(&self, time: f32) -> f32 {
        match self.bindings[HELD_BIND] == 1 {
            true => self.start,
            false => time,
        }
    }
}

/// The command ids a timeline's own actors and tracks reach, so a shot nothing plays is told apart
/// from one its own structure never offers. Empty where the timeline names no actors at all, which
/// a filter reads as "nothing is excluded" rather than "everything is".
fn reachable_commands(timeline: &Timeline) -> std::collections::BTreeSet<i16> {
    let mut reachable = std::collections::BTreeSet::new();
    for item in timeline.items() {
        let Item::ActorList(list) = item else {
            continue;
        };
        for actor_id in list.actors() {
            let Some(Item::Actor(actor)) = timeline
                .items()
                .iter()
                .find(|item| matches!(item, Item::Actor(a) if a.id() == *actor_id))
            else {
                continue;
            };
            for track_id in actor.tracks() {
                let Some(Item::Track(track)) = timeline
                    .items()
                    .iter()
                    .find(|item| matches!(item, Item::Track(t) if t.id() == *track_id))
                else {
                    continue;
                };
                reachable.extend(track.commands());
            }
        }
    }
    reachable
}

/// How long a `CTTL` plays for: what its own header states, or where nothing does, past the
/// furthest a shot it holds runs.
fn timeline_span(timeline: &Timeline, shots: &[(i16, f32, f32)]) -> f32 {
    let stated = timeline.items().iter().find_map(|item| match item {
        Item::Header(header) => Some(f32::from(header.duration())),
        _ => None,
    });
    stated.unwrap_or_else(|| {
        shots
            .iter()
            .map(|(_, start, duration)| start + duration)
            .fold(0.0, f32::max)
    })
}

/// One `C004` read out of a timeline: its own id (for the reachability filter), when it starts and
/// runs, its name, which `TMFC` drives it, what it binds, and its stated clip planes.
type RawShot = (i16, f32, f32, Option<String>, i16, [u32; 17], f32, f32);

/// The `C004` shots one `CTTL` holds, filtered to the ones its own actor tracks reach where any
/// are, in the order they run.
fn shots_of(timeline: &Timeline) -> Vec<RawShot> {
    let reachable = reachable_commands(timeline);
    let all: Vec<RawShot> = timeline
        .items()
        .iter()
        .filter_map(|item| {
            let Item::Command(command) = item else {
                return None;
            };
            let CommandKind::C004(camera) = command.kind() else {
                return None;
            };
            Some((
                command.id(),
                f32::from(command.time()),
                camera.duration().max(0) as f32,
                camera.name().map(str::to_owned),
                camera.curve_id().try_into().unwrap_or(0),
                *camera.bindings(),
                camera.near_plane(),
                camera.far_plane(),
            ))
        })
        .collect();
    let mut kept: Vec<_> = all
        .iter()
        .filter(|(id, ..)| reachable.is_empty() || reachable.contains(id))
        .cloned()
        .collect();
    if kept.is_empty() {
        kept = all;
    }
    kept.sort_by(|a, b| a.1.total_cmp(&b.1));
    kept
}

/// The shot active at a time, the last one to start at or before it. `None` before the first shot
/// anywhere in the cutscene has started.
fn active_shot(shots: &[Shot], time: f32) -> Option<&Shot> {
    shots
        .iter()
        .filter(|shot| shot.start <= time)
        .max_by(|a, b| a.start.total_cmp(&b.start))
}

/// A cutscene's camera, sequenced. Holds no bytes of its own past the shot list: [`Self::pose_at`]
/// reads the curves back out of the `Cutscene` it was built from.
pub struct Player {
    shots: Vec<Shot>,
    /// What each participant its timelines address does over the whole of it.
    parts: BTreeMap<u32, Part>,
    /// The lines its timelines put on screen, in the order they run.
    subtitles: Vec<Subtitle>,
    /// The sounds its timelines play, in the order they run.
    sounds: Vec<sound::Cue>,
    duration: f32,
}

impl Player {
    pub fn new(cutscene: &Cutscene) -> Self {
        let mut shots = Vec::new();
        let mut parts = BTreeMap::new();
        let mut subtitles = Vec::new();
        let mut sounds = Vec::new();
        let sheet_upper = sheet_id(dialogue_sheet(cutscene));
        let mut offset = 0.0;
        for (node, held) in cutscene.nodes().iter().enumerate() {
            let Node::Timeline(timeline) = held else {
                continue;
            };
            subtitles_of(timeline, offset, &sheet_upper, &mut subtitles);
            sounds_of(timeline, offset, &mut sounds);
            let local = shots_of(timeline);
            let span = timeline_span(
                timeline,
                &local
                    .iter()
                    .map(|(id, start, duration, ..)| (*id, *start, *duration))
                    .collect::<Vec<_>>(),
            );
            parts_of(timeline, node, offset, span.max(1.0), &mut parts);
            for (_, start, duration, name, curves, bindings, near, far) in local {
                shots.push(Shot {
                    node,
                    name,
                    start: offset + start,
                    duration,
                    curves,
                    bindings,
                    near,
                    far,
                });
            }
            offset += span.max(1.0);
        }
        for part in parts.values_mut() {
            part.order();
        }
        subtitles.sort_by(|left, right| left.at.total_cmp(&right.at));
        sounds.sort_by(|left, right| left.at.total_cmp(&right.at));
        Self {
            duration: offset,
            shots,
            parts,
            subtitles,
            sounds,
        }
    }

    /// Every sound the cutscene plays: the effects its `C063` commands file, and the voice line
    /// each subtitle key names in a language, both in the order they run.
    fn cues(&self, slug: &str, language: Language) -> Vec<sound::Cue> {
        let mut cues = self.sounds.clone();
        for subtitle in &self.subtitles {
            let paths = sound::voice_paths(&subtitle.key, slug, language);
            if paths.is_empty() {
                continue;
            }
            cues.push(sound::Cue {
                at: subtitle.at,
                paths,
                entry: 0,
                label: subtitle.speaker.clone(),
                holds: None,
            });
        }
        cues.sort_by(|left, right| left.at.total_cmp(&right.at));
        cues
    }

    /// Every line the cutscene puts on screen, in the order it runs.
    pub fn subtitles(&self) -> &[Subtitle] {
        &self.subtitles
    }

    /// The line standing at a time: the last to have started, while its own length holds. A line
    /// stating no length stands until the next one replaces it, which is what the client's own
    /// countdown leaves a zero doing.
    fn subtitle_at(&self, time: f32, language: Language) -> Option<(usize, &Subtitle)> {
        let (at, held) = self
            .subtitles
            .iter()
            .enumerate()
            .rev()
            .find(|(_, held)| held.at <= time)?;
        let runs = held.runs(language);
        (runs <= 0.0 || time < held.at + runs * FRAMES_A_SECOND).then_some((at, held))
    }

    /// What each participant its timelines address does.
    pub fn parts(&self) -> &BTreeMap<u32, Part> {
        &self.parts
    }

    /// Where a participant stands at a time, where its own timeline places it.
    fn placed(&self, participant: u32, time: f32) -> Option<Transform> {
        latest(&self.parts.get(&participant)?.placed, time).map(|(_, at)| *at)
    }

    /// The participants nothing is drawing at a time, which is what a prop is taken out of the
    /// frame by: the scene places one per participant and a cast is hidden through its own model.
    fn unplaced(&self, time: f32) -> std::collections::BTreeSet<u32> {
        self.parts
            .iter()
            .filter(|(_, part)| latest(&part.shown, time).is_some_and(|(_, on)| !on))
            .map(|(participant, _)| *participant)
            .collect()
    }

    /// Every shot, in the order it plays.
    pub fn shots(&self) -> &[Shot] {
        &self.shots
    }

    /// How long the whole cutscene plays for, in frames.
    pub fn duration(&self) -> f32 {
        self.duration
    }

    /// Every effect running at a time: the ones a participant has fired that its own timeline has
    /// not run past, each standing where that participant now does. A firing keeps its id across
    /// frames, so the particles it has run out are its own.
    pub fn firing(&self, cutscene: &Cutscene, time: f32) -> Vec<scene::Fired> {
        let participants = participants(cutscene);
        let mut fired = Vec::new();
        for (participant, part) in &self.parts {
            let at = self.placed(*participant, time).or_else(|| {
                participants
                    .iter()
                    .find(|held| held.id() == *participant)
                    .map(stands_at)
            });
            let Some(at) = at.map(scene::matrix) else {
                continue;
            };
            for (index, (start, burst)) in part.effects.iter().enumerate() {
                if time < *start || time >= burst.until {
                    continue;
                }
                let along = time - start;
                fired.push(scene::Fired {
                    id: u64::from(*participant) << 32 | index as u64,
                    path: burst.path.clone(),
                    at,
                    frame: along as i32,
                    tint: set_of(cutscene, burst.node, burst.curves)
                        .map(|set| lit(set, along))
                        .unwrap_or(Vec4::ONE),
                });
            }
        }
        fired
    }

    /// The camera at a time, or `None` before any shot has started.
    pub fn pose_at(&self, cutscene: &Cutscene, time: f32) -> Option<Pose> {
        let shot = active_shot(&self.shots, time)?;
        let set = set_of(cutscene, shot.node, shot.curves)?;
        let bound = shot.bound_at(time);
        let targets = rig(set, &shot.bindings, participants(cutscene), &|participant| {
            self.placed(participant, bound)
        });
        eye_pose(set, &targets, time - shot.start, shot.near, shot.far)
    }
}

/// The curve set of one id, out of the timeline that holds it.
fn set_of(cutscene: &Cutscene, node: usize, id: i16) -> Option<&Curves> {
    let Some(Node::Timeline(timeline)) = cutscene.nodes().get(node) else {
        return None;
    };
    timeline.items().iter().find_map(|item| match item {
        Item::Curves(held) if held.id() == id => Some(held),
        _ => None,
    })
}

/// The helper a participant is written as, where it is one.
fn helper(participant: &Instance) -> Option<&HelperObject> {
    match participant.data() {
        InstanceData::HelperObject(helper) => Some(helper),
        _ => None,
    }
}

/// What a participant stands for, in as few words as its record states.
pub fn stands_for(participant: &Instance) -> String {
    let Some(helper) = helper(participant) else {
        return format!("{:?}", participant.kind());
    };
    match helper.kind() {
        HelperKind::EventNpc | HelperKind::BattleNpc => {
            format!("{:?} {}", helper.kind(), helper.base_id())
        }
        HelperKind::Weapon => format!("Weapon {}", helper.weapon().pattern_id()),
        kind => helper
            .nested()
            .and_then(|nested| layer::asset(nested.data()))
            .and_then(|asset| asset.rsplit('/').next())
            .map(str::to_owned)
            .unwrap_or_else(|| format!("{kind:?}")),
    }
}

/// Whether a kind's own setup reads the placement stated beside it. The rest are built by copying
/// the participant record's header wholesale, transform and all, so the placement never reaches
/// them: `sub_141B26310` calls the placement-aware `sub_141B282F0` for every other kind, and for
/// these a plain copy of the record's first 0x30 bytes.
fn takes_placement(kind: HelperKind) -> bool {
    !matches!(
        kind,
        HelperKind::BgPart | HelperKind::SharedGroup | HelperKind::Weapon | HelperKind::Unknown85
    )
}

/// Where a participant stands: the transform its record states apart from the instance's own wins
/// where the flag says so and the kind reads it, the way the game's own setup takes it.
fn stands_at(participant: &Instance) -> Transform {
    helper(participant)
        .filter(|helper| takes_placement(helper.kind()))
        .and_then(HelperObject::placement)
        .filter(|placement| placement.flags() & 1 != 0)
        .map(|placement| placement.transform())
        .unwrap_or_else(|| participant.transform())
}

/// What a prop participant draws itself from, where its nested instance names one. Only the two
/// kinds that build background out of it: `Unknown85` names a nested shared group as well, and its
/// own setup takes the path alone, as a kind of instance this view has no notion of.
fn drawn_from(participant: &Instance) -> Option<scene::Asset> {
    let helper = helper(participant)
        .filter(|helper| matches!(helper.kind(), HelperKind::BgPart | HelperKind::SharedGroup))?;
    let asset = match helper.nested()?.data() {
        InstanceData::BgPart(part) => scene::Asset::Model(part.asset_path().clone()),
        InstanceData::SharedGroup(group) => scene::Asset::Group(group.asset_path().clone()),
        _ => return None,
    };
    let (scene::Asset::Model(path) | scene::Asset::Group(path)) = &asset;
    (!path.is_empty()).then_some(asset)
}

/// The scenery a cutscene brings with it: the participants naming a model or a shared group, at the
/// transforms their own records state. The nested instance carries the asset and nothing else - its
/// own transform is all zeroes in every shipping file.
fn props(cutscene: &Cutscene) -> Vec<scene::Prop> {
    participants(cutscene)
        .iter()
        .filter_map(|participant| {
            Some(scene::Prop {
                asset: drawn_from(participant)?,
                transform: stands_at(participant),
                id: participant.id(),
            })
        })
        .collect()
}

/// The character each participant stands for, with no live one on hand to copy. `sub_141B26310`
/// takes the live character its kind names and falls back to a row: a party member to one fixed
/// stand-in unless the record forces an id of its own, a stabled chocobo to one whichever id it
/// names, and the player to the record's own - which every shipping file leaves at a row stating
/// no race, no equipment and no `ModelChara`, so nothing is drawn for one.
fn stands_as(participant: &Instance) -> Option<stand::Wanted> {
    let helper = helper(participant)?;
    let (roll, id) = match helper.kind() {
        HelperKind::EventNpc | HelperKind::Player => (stand::Roll::Event, helper.base_id()),
        HelperKind::BattleNpc => (stand::Roll::Battle, helper.base_id()),
        HelperKind::PartyMember | HelperKind::PartyMemberAlt | HelperKind::Unknown82 => (
            stand::Roll::Event,
            match helper.forces_base_id() {
                true => helper.base_id(),
                false => stand::PARTY_STAND_IN,
            },
        ),
        HelperKind::StableChocobo => (stand::Roll::Event, stand::STABLED_CHOCOBO),
        _ => return None,
    };
    (id != 0).then(|| stand::Wanted {
        roll,
        id,
        height: helper.height(),
        at: stands_at(participant),
        participant: participant.id(),
    })
}

/// Everyone a cutscene stands, at the transforms their own records state.
fn cast(cutscene: &Cutscene) -> Vec<stand::Wanted> {
    participants(cutscene).iter().filter_map(stands_as).collect()
}

/// What a `CTAL` holds, as a count of each kind its participants stand for.
pub fn roll_call(participants: &[Instance]) -> String {
    let mut held: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for participant in participants {
        let named = match helper(participant) {
            Some(helper) => format!("{:?}", helper.kind()),
            None => format!("{:?}", participant.kind()),
        };
        *held.entry(named).or_default() += 1;
    }
    let mut lines: Vec<(usize, String)> = held
        .into_iter()
        .map(|(named, count)| (count, named))
        .collect();
    lines.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)));
    lines
        .iter()
        .take(4)
        .map(|(count, named)| format!("{count} {named}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The `.pap` packs a cutscene loads, which is where the motions its timelines name live.
fn packs(cutscene: &Cutscene) -> Vec<String> {
    cutscene
        .nodes()
        .iter()
        .filter_map(|node| match node {
            Node::Resources(list) => Some(list),
            _ => None,
        })
        .flatten()
        .map(|resource| resource.path().to_owned())
        .filter(|path| path.ends_with(".pap"))
        .collect()
}

/// The sheet a cutscene's `CTIS` reads its lines out of.
fn dialogue_sheet(cutscene: &Cutscene) -> Option<&str> {
    cutscene.nodes().iter().find_map(|node| match node {
        Node::Sheet(name) if !name.is_empty() => Some(name.as_str()),
        _ => None,
    })
}

/// The `CTAL` a cutscene holds, empty where it names none.
fn participants(cutscene: &Cutscene) -> &[Instance] {
    cutscene
        .nodes()
        .iter()
        .find_map(|node| match node {
            Node::Participants(participants) => Some(participants.as_slice()),
            _ => None,
        })
        .unwrap_or_default()
}

/// The `CTAL` participants a cutscene names, as points to mark rather than characters to draw.
pub fn markers(cutscene: &Cutscene) -> Vec<(Vec3, String)> {
    participants(cutscene)
        .iter()
        .map(|participant| {
            (
                Vec3::from_array(stands_at(participant).translation()),
                format!("{} · {:#x}", stands_for(participant), participant.id()),
            )
        })
        .collect()
}

/// The `.lvb` a `CTDS` names its level by: the same shape the Assets tab's own Zones tab resolves.
fn level_path(level: &str) -> String {
    format!("bg/{level}.lvb")
}

enum Fetch {
    Idle,
    Loading(Box<TrackedPromise<anyhow::Result<LevelFile>>>),
    Ready(Box<scene::Scene>),
    Failed(String),
}

/// The lines a cutscene's own sheet holds, by the key a `C048` names them with, as the strings the
/// sheet states rather than the text they format to: which payloads a line spells out is a setting.
type Said = BTreeMap<String, Vec<u8>>;

enum Tracks {
    Idle,
    Loading(Box<TrackedPromise<anyhow::Result<music::Music>>>),
    Ready(music::Music),
    Failed(String),
}

enum Lines {
    Idle,
    Loading(Box<TrackedPromise<anyhow::Result<Said>>>),
    Ready(Said),
    Failed(String),
}

/// Reads a cutscene's dialogue sheet: two string columns of key and text, keyed by the first.
async fn read_lines(backend: Backend, sheet: String, language: Language) -> anyhow::Result<Said> {
    let opened = backend.excel().get_sheet(&sheet, language).await?;
    let columns = SheetColumnDefinition::from_sheet(&opened);
    let [key, text, ..] = columns.as_slice() else {
        anyhow::bail!("{sheet} is not a two column text sheet");
    };
    let mut held = BTreeMap::new();
    for row_id in opened.get_row_ids() {
        let Ok(row) = opened.get_row(row_id) else {
            continue;
        };
        let (Ok(key), Ok(text)) = (
            row.read_string(u32::from(key.offset())),
            row.read_string(u32::from(text.offset())),
        ) else {
            continue;
        };
        held.insert(
            key.format().to_string(),
            text.as_bytes().to_vec(),
        );
    }
    Ok(held)
}

struct State {
    fetch: Fetch,
    lines: Lines,
    /// Which language the lines were read in, so a change of it reads them again.
    lines_for: Option<Language>,
    /// Whether to put the cutscene's own lines over the frame, and whether to frame it as the
    /// sixteen by nine the camera's field of view is worked out against.
    subtitles: bool,
    framed: bool,
    /// Everyone standing in the scene, from the rows their participants name through to the models
    /// the scene draws.
    cast: stand::Cast,
    time: f32,
    playing: bool,
    /// Frames a second, which the transport bar can move off the rate the file's own numbering
    /// runs at.
    fps: f32,
    /// Which cue each participant's body and face are holding, so a cue is issued when it changes
    /// rather than every frame: taking a clip up is what puts its own clock back to nought.
    bodies: BTreeMap<u32, usize>,
    overs: BTreeMap<u32, usize>,
    faces: BTreeMap<u32, usize>,
    /// The firings already logged, so one effect is reported when it starts rather than every
    /// frame it runs for, and the same for whether each participant is drawn.
    burst: std::collections::BTreeSet<u64>,
    shown: BTreeMap<u32, bool>,
    /// Which fade each participant is running, so one is reported when it starts rather than every
    /// frame it steps through.
    fades: BTreeMap<u32, Option<usize>>,
    /// What the cutscene sounds, and how far through it the cues have been fired.
    stage: sound::Stage,
    cues: Vec<sound::Cue>,
    cues_for: Option<Language>,
    sounded: f32,
    /// The music the cutscene's own quest states, and which of it to play under the frame.
    tracks: Tracks,
    track: usize,
    /// Whether to fire the cues, and whether to keep a track under them. Either one opens the
    /// mixer; a browser only grants that from inside the click that ticked the box.
    effects: bool,
    music: bool,
}

impl Default for State {
    fn default() -> Self {
        Self {
            fetch: Fetch::Idle,
            lines: Lines::Idle,
            lines_for: None,
            subtitles: true,
            framed: true,
            cast: stand::Cast::default(),
            time: 0.0,
            playing: false,
            fps: FRAMES_A_SECOND,
            bodies: BTreeMap::new(),
            overs: BTreeMap::new(),
            faces: BTreeMap::new(),
            burst: std::collections::BTreeSet::new(),
            shown: BTreeMap::new(),
            fades: BTreeMap::new(),
            stage: sound::Stage::default(),
            cues: Vec::new(),
            cues_for: None,
            sounded: 0.0,
            tracks: Tracks::Idle,
            track: 0,
            effects: false,
            music: false,
        }
    }
}

/// A cutscene's own "Play" tab: the level its `CTDS` names, with the camera driven by its shots
/// instead of the free orbit camera.
pub struct Tab {
    level: String,
    /// The cutscene's own path, and which expansion it sits under: the client spends the same on
    /// a voice path.
    path: String,
    slug: String,
    player: Player,
    state: RefCell<State>,
}

impl Tab {
    pub fn new(level: String, path: &str, cutscene: &Cutscene) -> Self {
        Self {
            level,
            path: path.to_owned(),
            slug: path.split('/').nth(1).unwrap_or("ffxiv").to_owned(),
            player: Player::new(cutscene),
            state: RefCell::new(State::default()),
        }
    }
}

pub fn ui(ui: &mut egui::Ui, tab: &Tab, cutscene: &Cutscene, backend: &Backend) -> Option<String> {
    if tab.level.is_empty() {
        ui.centered_and_justified(|ui| {
            ui.label(RichText::new("This cutscene names no level").weak());
        });
        return None;
    }

    let mut state = tab.state.borrow_mut();
    if matches!(&state.fetch, Fetch::Idle) {
        let files = backend.files().clone();
        let path = level_path(&tab.level);
        state.fetch = Fetch::Loading(Box::new(TrackedPromise::spawn_local(async move {
            files.file::<LevelFile>(&path).await
        })));
    }
    if matches!(&state.fetch, Fetch::Loading(promise) if promise.try_get().is_some()) {
        let Fetch::Loading(promise) = std::mem::replace(&mut state.fetch, Fetch::Idle) else {
            unreachable!()
        };
        state.fetch = match promise.block_and_take() {
            Ok(file) => {
                let mut scene = layer::level_scene(&tab.level, file);
                scene.place("Cutscene", props(cutscene));
                state.cast = stand::Cast::new(cast(cutscene));
                state.cast.loads(packs(cutscene));
                Fetch::Ready(Box::new(scene))
            }
            Err(error) => Fetch::Failed(error.to_string()),
        };
    }

    let language = LANGUAGE.get(ui.ctx());
    if state.lines_for != Some(language) {
        state.lines_for = Some(language);
        state.lines = Lines::Idle;
    }
    if matches!(&state.lines, Lines::Idle)
        && let Some(sheet) = dialogue_sheet(cutscene)
    {
        let backend = backend.clone();
        let sheet = sheet.to_owned();
        state.lines = Lines::Loading(Box::new(TrackedPromise::spawn_local(async move {
            read_lines(backend, sheet, language).await
        })));
    }
    if matches!(&state.lines, Lines::Loading(promise) if promise.try_get().is_some()) {
        let Lines::Loading(promise) = std::mem::replace(&mut state.lines, Lines::Idle) else {
            unreachable!()
        };
        state.lines = match promise.block_and_take() {
            Ok(held) => Lines::Ready(held),
            Err(error) => Lines::Failed(error.to_string()),
        };
    }

    if state.cues_for != Some(language) {
        state.cues_for = Some(language);
        state.cues = tab.player.cues(&tab.slug, language);
        state.stage.silence();
    }
    if state.music && matches!(&state.tracks, Tracks::Idle) {
        let backend = backend.clone();
        let path = tab.path.clone();
        state.tracks = Tracks::Loading(Box::new(TrackedPromise::spawn_local(async move {
            music::resolve(backend, language, path).await
        })));
    }
    if matches!(&state.tracks, Tracks::Loading(promise) if promise.try_get().is_some()) {
        let Tracks::Loading(promise) = std::mem::replace(&mut state.tracks, Tracks::Idle) else {
            unreachable!()
        };
        state.tracks = match promise.block_and_take() {
            Ok(found) => Tracks::Ready(found),
            Err(error) => Tracks::Failed(error.to_string()),
        };
    }

    let held = &mut *state;
    if held.effects {
        held.stage.want(backend, &held.cues);
    }
    if let Some(cue) = under(held, tab.player.duration()) {
        held.stage.want(backend, std::slice::from_ref(&cue));
    }
    held.stage.poll(held.time);

    let pose = tab.player.pose_at(cutscene, state.time);
    state.cast.poll(ui.ctx(), backend);
    perform(tab.player.parts(), &mut state);
    let standing = state.cast.standing();
    let firing = tab.player.firing(cutscene, state.time);
    let unplaced = tab.player.unplaced(state.time);
    state.burst.retain(|id| firing.iter().any(|held| held.id == *id));
    for held in &firing {
        if state.burst.insert(held.id) {
            log::info!(
                "cutb: {:#x} fires {} at frame {:.0}",
                held.id >> 32,
                held.path,
                state.time
            );
        }
    }

    Panel::left("cutb_shots")
        .default_size(200.0)
        .show(ui, |ui| {
            shots_ui(ui, tab, &mut state, language);
        });
    Panel::bottom("cutb_transport").show(ui, |ui| {
        ui.add_space(4.0);
        transport(ui, tab, &mut state, pose.as_ref());
        ui.add_space(4.0);
    });
    CentralPanel::default().show(ui, |ui| match &mut state.fetch {
        Fetch::Idle | Fetch::Loading(_) => {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("Reading the level…");
            });
        }
        Fetch::Failed(error) => {
            ui.colored_label(egui::Color32::RED, error.clone());
        }
        Fetch::Ready(scene) => {
            scene.stand(standing);
            scene.hide(unplaced);
            scene.fire(firing);
            if let Some(pose) = pose {
                scene.drive(pose.drive());
            }
            scene.mark(markers(cutscene));
            let frame = ui.max_rect();
            scene::ui(ui, scene, backend);
            overlay(ui, tab, &state, frame, language);
        }
    });
    None
}

/// The frame a cutscene composes against, sixteen by nine: the aspect the client turns a shot's
/// focal length into a field of view against, and the one it clamps the subtitle addon's own width
/// to (`Client::UI::Agent::AgentTalkSubtitle.OpenSubtitleAddon`). The scene carries that field to
/// whatever aspect it is drawn at by keeping the horizontal one, so anything past the frame's own
/// height is over-draw.
const FRAME_ASPECT: f32 = 16.0 / 9.0;

/// The frame `ui/uld/TalkSubtitle.uld` is laid out in, which the client scales to the screen.
const DESIGN: egui::Vec2 = egui::Vec2::new(1280.0, 720.0);

/// Where the subtitle addon attaches, as a fraction of the frame: centred across it, and 95 parts
/// in a hundred down it.
const SUBTITLE_AT: egui::Vec2 = egui::Vec2::new(0.50, 0.95);

/// What `TalkSubtitle.uld`'s three text nodes state: one run at size 18, with a copy a pixel above
/// and a pixel below it. Their fills are `UIColor` rows 1 and 7, white over black in the theme the
/// game opens in.
const TEXT_SIZE: f32 = 18.0;
const EDGE: f32 = 1.0;
const SAID: Color32 = Color32::WHITE;
const SHADOW: Color32 = Color32::BLACK;

/// Puts the cutscene's own frame and its lines over the scene.
fn overlay(ui: &mut egui::Ui, tab: &Tab, state: &State, frame: Rect, language: Language) {
    let painter = ui.painter_at(frame);
    let inner = match state.framed {
        true => framed(frame),
        false => frame,
    };
    if state.framed {
        let bar = Color32::BLACK;
        painter.rect_filled(
            Rect::from_min_max(frame.min, pos2(frame.max.x, inner.min.y)),
            0.0,
            bar,
        );
        painter.rect_filled(
            Rect::from_min_max(pos2(frame.min.x, inner.max.y), frame.max),
            0.0,
            bar,
        );
    }
    if !state.subtitles {
        return;
    }
    let Some((_, subtitle)) = tab.player.subtitle_at(state.time, language) else {
        return;
    };
    let Lines::Ready(lines) = &state.lines else {
        return;
    };
    let text = lines
        .get(&subtitle.key)
        .map(|text| sestring(ui, text))
        .filter(|text| !text.is_empty());
    let Some(text) = text else {
        return;
    };
    let scale = inner.height() / DESIGN.y;
    let font = FontId::proportional(TEXT_SIZE * scale);
    let laid = ui
        .ctx()
        .fonts_mut(|fonts| fonts.layout(text, font, SAID, DESIGN.x * scale));
    let at = inner.min + inner.size() * SUBTITLE_AT - vec2(laid.size().x * 0.5, laid.size().y * 0.5);
    for (offset, color) in [
        (-EDGE * scale, SHADOW),
        (EDGE * scale, SHADOW),
        (0.0, SAID),
    ] {
        painter.galley(at + vec2(0.0, offset), laid.clone(), color);
    }
}

/// The largest sixteen by nine rect the frame holds, centred in it.
fn framed(frame: Rect) -> Rect {
    let size = match frame.width() / frame.height() < FRAME_ASPECT {
        true => vec2(frame.width(), frame.width() / FRAME_ASPECT),
        false => vec2(frame.height() * FRAME_ASPECT, frame.height()),
    };
    Rect::from_center_size(frame.center(), size)
}

/// Puts every participant where its own timeline has it now, and plays what that timeline names.
///
/// A cue is issued only where it has changed: taking a clip up is what puts its own clock back to
/// nought, so a cue reissued every frame would hold every actor on its first pose.
fn perform(parts: &BTreeMap<u32, Part>, state: &mut State) {
    let time = state.time;
    for (participant, part) in parts {
        if let Some((_, at)) = latest(&part.placed, time) {
            state.cast.place(*participant, *at);
        }
        let shown = latest(&part.shown, time).is_none_or(|(_, on)| *on);
        // Everyone is drawn until a timeline says otherwise, so a participant nothing has stated
        // yet is not a change to report.
        if state.shown.insert(*participant, shown).unwrap_or(true) != shown {
            log::info!(
                "cutb: {participant:#x} is {}",
                match shown {
                    true => "drawn",
                    false => "out of frame",
                }
            );
        }
        state.cast.show(*participant, shown);
        let fading = part.fading(time);
        state.cast.fade(
            *participant,
            fading.map_or(1.0, |(_, fade, along)| fade.at(along)),
        );
        let at = fading.map(|(at, ..)| at);
        if state.fades.insert(*participant, at) != Some(at)
            && let Some((_, fade, _)) = fading
        {
            log::info!(
                "cutb: {participant:#x} fades {:.2} to {:.2} over {:.0}f",
                fade.from,
                fade.to,
                fade.over
            );
        }
        let Some(model) = state.cast.model(*participant).cloned() else {
            continue;
        };
        let lays_over = |motion: &str| state.cast.lays_over(motion);
        if let Some((at, held)) = started(&part.motions, time, false, &lays_over) {
            if state.bodies.get(participant) != Some(&at)
                && let Some(pack) = state.cast.holding(*participant, &held.motion)
            {
                state.bodies.insert(*participant, at);
                log::info!(
                    "cutb: {participant:#x} plays {} from {:.2}s out of {pack}",
                    held.motion,
                    held.from
                );
                model.stand(&[(pack, &held.motion)], 0.0);
            }
            // Every frame rather than only where the cue changes: the clip runs on wall time and
            // the transport on the cutscene's own frames, so a seek would otherwise replay the
            // clip from its start instead of landing inside it.
            model.plays_at(held.from + (time - part.motions[at].0) / FRAMES_A_SECOND);
        }
        match started(&part.motions, time, true, &lays_over) {
            Some((at, held)) => {
                if state.overs.get(participant) != Some(&at)
                    && let Some(pack) = state.cast.holding(*participant, &held.motion)
                {
                    state.overs.insert(*participant, at);
                    log::info!("cutb: {participant:#x} lays {} over it", held.motion);
                    model.act(&[pack], &held.motion, 0.0);
                }
            }
            None => {
                if state.overs.remove(participant).is_some() {
                    model.act(&[], "", 0.0);
                }
            }
        }
        if let Some((at, name)) = latest(&part.faces, time)
            && state.faces.get(participant) != Some(&at)
        {
            state.faces.insert(*participant, at);
            model.express(name);
        }
    }
}

/// The shots a cutscene cuts between and the lines it says, each a list to seek by. They share the
/// panel rather than run on from one another: a cutscene holds scores of each, so a single column
/// would leave the second one off the foot of the panel.
fn shots_ui(ui: &mut egui::Ui, tab: &Tab, state: &mut State, language: Language) {
    let active = active_shot(tab.player.shots(), state.time).map(|shot| shot.start);
    let speaking = tab
        .player
        .subtitle_at(state.time, language)
        .map(|(at, _)| at);
    let lines = match &state.lines {
        Lines::Ready(held) => Some(held),
        _ => None,
    };
    let mut seek = None;
    let split = !tab.player.subtitles().is_empty();
    let height = match split {
        true => ui.available_height() * 0.5,
        false => ui.available_height(),
    };

    ui.label(RichText::new("Shots").strong());
    ScrollArea::vertical()
        .id_salt("cutb_shot_list")
        .max_height(height)
        .auto_shrink(false)
        .show(ui, |ui| {
            ui.with_layout(Layout::top_down_justified(Align::Min), |ui| {
                for shot in tab.player.shots() {
                    let current = active == Some(shot.start);
                    let label = format!(
                        "{} · node {} · {:.0}f",
                        shot.name.as_deref().unwrap_or("-"),
                        shot.node,
                        shot.duration,
                    );
                    if ui.add(Button::selectable(current, label)).clicked() {
                        seek = Some(shot.start);
                    }
                }
                if tab.player.shots().is_empty() {
                    ui.label(RichText::new("This cutscene's timelines hold no camera").weak());
                }
            });
        });
    if split {
        ui.add_space(4.0);
        ui.label(RichText::new("Lines").strong());
        ScrollArea::vertical()
            .id_salt("cutb_line_list")
            .auto_shrink(false)
            .show(ui, |ui| {
                ui.with_layout(Layout::top_down_justified(Align::Min), |ui| {
                    for (at, subtitle) in tab.player.subtitles().iter().enumerate() {
                        let said = lines
                            .and_then(|lines| lines.get(&subtitle.key))
                            .map(|text| sestring(ui, text).replace('\n', " "))
                            .filter(|text| !text.is_empty())
                            .unwrap_or_else(|| subtitle.key.clone());
                        let label =
                            format!("{:.0}f · {} · {said}", subtitle.at, subtitle.speaker);
                        if ui
                            .add(Button::selectable(speaking == Some(at), label))
                            .on_hover_text(&subtitle.key)
                            .clicked()
                        {
                            seek = Some(subtitle.at);
                        }
                    }
                });
            });
    }
    if let Some(at) = seek {
        state.time = at;
        state.playing = false;
    }
}

/// Plays whatever the clock has just run over. A cutscene's own clock scrubs and the mixer's does
/// not, so anything but playing forward stops the lot rather than leaving it ringing.
fn fire(state: &mut State, duration: f32) {
    let time = state.time;
    if !state.playing || time < state.sounded {
        state.stage.silence();
        state.sounded = time;
        return;
    }
    if state.effects {
        for cue in &state.cues {
            if cue.at > state.sounded && cue.at <= time {
                state.stage.play(cue, time);
            }
        }
    }
    if let Some(cue) = under(state, duration) {
        state.stage.under(&cue, 0.0);
    }
    state.sounded = time;
}

/// The track to keep playing under the frame: whichever of the quest's the panel has picked. It is
/// held for the whole cutscene, since nothing states where it stops.
fn under(state: &State, duration: f32) -> Option<sound::Cue> {
    if !state.music {
        return None;
    }
    let Tracks::Ready(found) = &state.tracks else {
        return None;
    };
    let track = found.tracks.get(state.track)?;
    Some(sound::Cue {
        at: 0.0,
        paths: vec![track.path.clone()],
        entry: 0,
        label: "music".to_owned(),
        holds: Some(duration.max(1.0)),
    })
}

/// Opens or closes the mixer to match the two toggles. Creating it and resuming it both have to
/// happen inside the click that asked for sound, which is the only user gesture a browser counts.
fn listen(state: &mut State) {
    match state.effects || state.music {
        true => state.stage.enable(),
        false => state.stage.disable(),
    }
    state.stage.silence();
}

/// The music row: a toggle, whatever the cutscene's own quest names, and which of it to play.
fn music_ui(ui: &mut egui::Ui, state: &mut State) {
    if ui
        .checkbox(&mut state.music, "Music")
        .on_hover_text(
            "Play the music the quest naming this cutscene states. Nothing in the file itself \
             says what a cutscene plays under, so this is quest-wide rather than per-cutscene.",
        )
        .clicked()
    {
        listen(state);
    }
    if !state.music {
        return;
    }
    match &state.tracks {
        Tracks::Idle | Tracks::Loading(_) => {
            ui.label(RichText::new("reading the quest\u{2026}").weak());
        }
        Tracks::Failed(why) => {
            ui.label(RichText::new(format!("music: {why}")).weak());
        }
        Tracks::Ready(found) if found.tracks.is_empty() => {
            ui.label(
                RichText::new(match found.quests {
                    0 => "no quest names this cutscene".to_owned(),
                    quests => format!("{quests} quests name it, none with music"),
                })
                .weak(),
            );
        }
        Tracks::Ready(found) => {
            let mut picked = state.track.min(found.tracks.len() - 1);
            egui::ComboBox::from_id_salt("cutb_music")
                .selected_text(label_of(&found.tracks[picked]))
                .show_ui(ui, |ui| {
                    for (at, track) in found.tracks.iter().enumerate() {
                        ui.selectable_value(&mut picked, at, label_of(track))
                            .on_hover_text(format!("{} \u{b7} quest {}", track.path, track.quest));
                    }
                });
            if picked != state.track {
                state.track = picked;
                state.stage.silence();
            }
            if state.stage.sounding_under() {
                ui.label(RichText::new("under").weak());
            }
        }
    }
}

/// What to call a track: the instruction the quest names it with, marked where the quest's own
/// script plays it in the scene that plays this cutscene.
fn label_of(track: &music::Track) -> String {
    let leaf = track.path.rsplit('/').next().unwrap_or(&track.path);
    match track.scripted {
        true => format!("{} \u{b7} {leaf}", track.instruction),
        false => format!("{} \u{b7} {leaf} (elsewhere in the quest)", track.instruction),
    }
}

fn transport(ui: &mut egui::Ui, tab: &Tab, state: &mut State, pose: Option<&Pose>) {
    let duration = tab.player.duration();
    if state.playing {
        state.time += ui.input(|input| input.stable_dt).min(0.25) * state.fps;
        if state.time >= duration {
            state.time = duration;
            state.playing = false;
        }
        ui.ctx().request_repaint();
    }

    fire(state, duration);

    ui.horizontal_wrapped(|ui| {
        if ui.button("⏮").on_hover_text("Back to the start").clicked() {
            state.time = 0.0;
            state.playing = false;
        }
        if ui
            .add(Button::new(if state.playing { "⏸" } else { "▶" }))
            .clicked()
        {
            state.playing = !state.playing;
        }
        ui.spacing_mut().slider_width = 200.0;
        ui.add(egui::Slider::new(&mut state.time, 0.0..=duration.max(1.0)).text("frame"));
        ui.add(egui::Slider::new(&mut state.fps, 5.0..=60.0).text("fps")).on_hover_text(
            "How fast to play the cutscene's own frames. Thirty is the rate its own numbering \
             runs at.",
        );
    });
    // A second row of its own, so neither the toggles nor the readouts a cutscene grows can push
    // the transport's own buttons off the row they sit on.
    ui.horizontal_wrapped(|ui| {
        ui.checkbox(&mut state.framed, "16:9").on_hover_text(
            "Frame the shot as sixteen by nine, which is the aspect its focal length is turned \
             into a field of view against",
        );
        ui.checkbox(&mut state.subtitles, "Lines").on_hover_text(
            "Put the cutscene's own subtitles over the frame, out of the sheet its CTIS node names",
        );
        if ui
            .checkbox(&mut state.effects, "Sound")
            .on_hover_text(
                "Play the sounds the cutscene's own C063 commands file, and the voice line each \
                 subtitle key names. Cues fire while it plays; a pause or a seek stops them.",
            )
            .clicked()
        {
            listen(state);
        }
        if state.effects {
            let mut volume = state.stage.volume();
            ui.spacing_mut().slider_width = 80.0;
            if ui
                .add(egui::Slider::new(&mut volume, 0.0..=1.0).show_value(false).text("🔊"))
                .changed()
            {
                state.stage.set_volume(volume);
            }
            let (read, wanted) = state.stage.read();
            ui.label(
                RichText::new(format!(
                    "{read}/{wanted} read, {} sounding",
                    state.stage.playing()
                ))
                .weak(),
            )
            .on_hover_text(format!(
                "Sound files read of the ones {} cues ask for",
                state.cues.len()
            ));
            if state.stage.missing() > 0 {
                ui.label(RichText::new(format!("{} missing", state.stage.missing())).weak())
                    .on_hover_text("Cues naming a file the install does not hold");
            }
        }
        if let Some(why) = state.stage.error() {
            ui.colored_label(egui::Color32::LIGHT_RED, format!("sound: {why}"));
        }
        music_ui(ui, state);
        if let Lines::Failed(why) = &state.lines {
            ui.colored_label(egui::Color32::LIGHT_RED, format!("lines: {why}"));
        }
        ui.label(
            RichText::new(match pose {
                Some(pose) => format!(
                    "eye {:.1}, {:.1}, {:.1} · {:.1}\u{b0}",
                    pose.position.x, pose.position.y, pose.position.z, pose.fov_degrees
                ),
                None => "no shot active yet".to_owned(),
            })
            .weak(),
        );
        let (built, wanted) = state.cast.built();
        if wanted > 0 {
            ui.label(RichText::new(format!("{built}/{wanted} standing")).weak())
                .on_hover_text(
                    "Characters built out of the rows their participants name, against how many \
                     rows the cast holds",
                );
        }
        if !state.cast.loaded() {
            ui.label(RichText::new("reading the motion packs\u{2026}").weak())
                .on_hover_text(
                    "A cutscene names the motions it plays rather than filing them, so nothing \
                     acts until the packs it loads have been read",
                );
        }
        if let Some(why) = state.cast.failure() {
            ui.colored_label(egui::Color32::LIGHT_RED, format!("cast: {why}"));
        }
    });
}

#[cfg(test)]
mod test {
    use super::*;

    fn close(a: Vec3, b: Vec3) -> bool {
        (a - b).length() < 1e-4
    }

    #[test]
    fn a_focal_length_of_the_half_sensor_gives_a_right_angle() {
        // atan(1) doubled is a quarter turn: the vertical frame exactly spans the lens.
        let fov = field_of_view_degrees(HALF_SENSOR);
        assert!((fov - 90.0).abs() < 1e-3);
    }

    #[test]
    fn a_longer_focal_length_narrows_the_field_of_view() {
        assert!(field_of_view_degrees(70.0) < field_of_view_degrees(35.0));
    }

    #[test]
    fn roll_turns_up_about_the_eye_s_own_forward() {
        assert!(close(banked(Vec3::NEG_Z, Vec3::Y, 0.0), Vec3::Y));
        // "The other way round": a positive roll field turns up towards +X, not -X.
        assert!(close(banked(Vec3::NEG_Z, Vec3::Y, 90.0), Vec3::X));
    }

    #[test]
    fn a_binding_that_does_not_turn_keeps_the_parent_s_facing() {
        let parent = Mat4::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let placement = Mat4::from_rotation_translation(
            Quat::from_rotation_y(std::f32::consts::PI),
            Vec3::new(3.0, 4.0, 5.0),
        );
        let held = frame(parent, Some((placement, false)));
        assert!(close(held.w_axis.truncate(), Vec3::new(3.0, 4.0, 5.0)));
        assert!(close(
            held.transform_vector3(Vec3::Z),
            parent.transform_vector3(Vec3::Z)
        ));
        let turning = frame(parent, Some((placement, true)));
        assert!(close(turning.transform_vector3(Vec3::Z), Vec3::NEG_Z));
        assert!(close(turning.w_axis.truncate(), Vec3::new(3.0, 4.0, 5.0)));
    }

    fn shot(start: f32, duration: f32) -> Shot {
        Shot {
            node: 0,
            name: None,
            start,
            duration,
            curves: 0,
            bindings: [0xffff_ffff; 17],
            near: 0.1,
            far: 1000.0,
        }
    }

    #[test]
    fn the_active_shot_is_the_last_one_that_has_started() {
        let shots = vec![shot(0.0, 30.0), shot(30.0, 60.0), shot(120.0, 10.0)];
        assert!(active_shot(&shots, -1.0).is_none());
        assert_eq!(active_shot(&shots, 0.0).unwrap().start, 0.0);
        assert_eq!(active_shot(&shots, 45.0).unwrap().start, 30.0);
        // Past every shot's own duration, the last one to start still holds: a cut is what ends a
        // shot, not its own stated length.
        assert_eq!(active_shot(&shots, 200.0).unwrap().start, 120.0);
    }

    #[test]
    fn a_later_shot_starting_before_an_earlier_one_ends_preempts_it() {
        let shots = vec![shot(0.0, 100.0), shot(10.0, 5.0)];
        assert_eq!(active_shot(&shots, 50.0).unwrap().start, 10.0);
    }

    #[test]
    fn a_shot_reads_its_bound_actor_where_it_stands_now_unless_it_holds_the_bind() {
        let mut held = shot(30.0, 60.0);
        held.bindings[HELD_BIND] = 0;
        assert_eq!(held.bound_at(75.0), 75.0);
        held.bindings[HELD_BIND] = 1;
        assert_eq!(held.bound_at(75.0), 30.0);
    }

    fn placed(x: f32) -> Transform {
        Transform::new([x, 0.0, 0.0], [0.0; 3], [1.0; 3])
    }

    #[test]
    fn what_holds_at_a_time_is_the_last_entry_to_have_started() {
        let held = vec![(0.0, placed(1.0)), (100.0, placed(2.0)), (200.0, placed(3.0))];
        assert!(latest(&held, -1.0).is_none());
        assert_eq!(latest(&held, 0.0).unwrap().0, 0);
        assert_eq!(latest(&held, 99.0).unwrap().1.translation()[0], 1.0);
        assert_eq!(latest(&held, 100.0).unwrap().1.translation()[0], 2.0);
        assert_eq!(latest(&held, 5000.0).unwrap().1.translation()[0], 3.0);
    }

    #[test]
    fn a_pose_goes_to_the_face_and_a_motion_to_the_body() {
        let mut part = Part::default();
        cue(&mut part, 10.0, Some("cfxf_salute"), 0.0, 30.0);
        cue(&mut part, 20.0, Some("cbfm_arms"), 3.0, 30.0);
        // The face's own blink and lip clips move bones the body does not carry.
        cue(&mut part, 30.0, Some("cfxb_blink1"), 0.0, 30.0);
        cue(&mut part, 40.0, Some("cfxl_lip_nor1"), 0.0, 30.0);
        cue(&mut part, 50.0, None, 0.0, 30.0);
        assert_eq!(part.faces.len(), 1);
        assert_eq!(part.faces[0], (10.0, "salute".to_owned()));
        assert_eq!(part.motions.len(), 1);
        assert_eq!(part.motions[0].1.motion, "cbfm_arms");
        assert_eq!(part.motions[0].1.from, 3.0);
        assert_eq!(part.motions[0].1.runs, 30.0);
    }

    #[test]
    fn a_part_runs_in_time_order_whatever_order_the_file_lists_it_in() {
        let mut part = Part {
            placed: vec![(100.0, placed(2.0)), (0.0, placed(1.0))],
            ..Part::default()
        };
        cue(&mut part, 50.0, Some("cbfm_arms"), 0.0, 30.0);
        cue(&mut part, 10.0, Some("cbnm_id0"), 0.0, 30.0);
        cue(&mut part, 80.0, Some("cfxf_salute"), 0.0, 30.0);
        cue(&mut part, 20.0, Some("cfxf_angry"), 0.0, 30.0);
        part.order();
        assert_eq!(latest(&part.placed, 50.0).unwrap().1.translation()[0], 1.0);
        assert_eq!(latest(&part.motions, 60.0).unwrap().1.motion, "cbfm_arms");
        assert_eq!(latest(&part.faces, 90.0).unwrap().1, "salute");
        assert_eq!(latest(&part.faces, 50.0).unwrap().1, "angry");
    }

    #[test]
    fn a_motion_laid_over_the_pose_neither_replaces_it_nor_outlives_its_own_command() {
        let mut part = Part::default();
        cue(&mut part, 0.0, Some("cbnm_id0"), 0.0, f32::INFINITY);
        cue(&mut part, 10.0, Some("cbfa_add_yes"), 0.0, 20.0);
        cue(&mut part, 50.0, Some("cbfm_arms"), 0.0, 60.0);
        part.order();
        let over = |motion: &str| motion.starts_with("cbfa_");
        let base = |time| started(&part.motions, time, false, &over).map(|(_, held)| &held.motion);
        let laid = |time| started(&part.motions, time, true, &over).map(|(_, held)| &held.motion);
        // The nod lays over the idle rather than becoming it, and is gone once it has run.
        assert_eq!(base(20.0).unwrap(), "cbnm_id0");
        assert_eq!(laid(20.0).unwrap(), "cbfa_add_yes");
        assert!(laid(30.0).is_none());
        assert_eq!(base(70.0).unwrap(), "cbfm_arms");
        assert!(laid(70.0).is_none());
    }

    fn said(at: f32, key: &str, english_ms: i32) -> Subtitle {
        Subtitle {
            at,
            speaker: speaker_of(key, "A_00000"),
            key: key.to_owned(),
            lengths: vec![0, english_ms, 0, 0, 0, 0, 0, 0],
        }
    }

    #[test]
    fn a_line_stands_for_its_own_language_s_length_and_a_lengthless_one_until_the_next() {
        let player = Player {
            shots: Vec::new(),
            parts: BTreeMap::new(),
            subtitles: vec![
                said(0.0, "TEXT_A_00000_000010_ALPHA", 2000),
                said(120.0, "TEXT_A_00000_000020_BETA", 0),
                said(300.0, "TEXT_A_00000_000030_ALPHA", 1000),
            ],
            sounds: Vec::new(),
            duration: 0.0,
        };
        let standing =
            |time| player.subtitle_at(time, Language::English).map(|(at, _)| at);
        // Two seconds of the cutscene's own frames, then nothing until the next line.
        assert_eq!(standing(30.0), Some(0));
        assert_eq!(standing(59.0), Some(0));
        assert_eq!(standing(61.0), None);
        // A line stating no length holds until something replaces it.
        assert_eq!(standing(299.0), Some(1));
        assert_eq!(standing(301.0), Some(2));
        assert_eq!(standing(400.0), None);
        assert!(standing(-1.0).is_none());
        // A language the file states no length for leaves every line lengthless.
        assert_eq!(
            player
                .subtitle_at(400.0, Language::Japanese)
                .map(|(at, _)| at),
            Some(2)
        );
    }

    #[test]
    fn a_key_names_its_speaker_on_either_side_of_its_line_number() {
        let named = |key| speaker_of(key, &sheet_id(Some("cut_scene/070/VoiceMan_07003")));
        assert_eq!(named("TEXT_VOICEMAN_07003_003100_GALUF"), "GALUF");
        let quest = |key| speaker_of(key, &sheet_id(Some("quest/000/ManFst000_00083")));
        // Half the corpus writes the name ahead of the number rather than after it.
        assert_eq!(quest("TEXT_MANFST000_00083_BREMONDT_000_37"), "BREMONDT");
        assert_eq!(quest("TEXT_MANFST000_00083_000010_MOTHERCRYSTAL"), "MOTHERCRYSTAL");
        // A key naming nothing but its number leaves the speaker unsaid.
        assert_eq!(quest("TEXT_MANFST000_00083_000010"), "");
    }

    #[test]
    fn the_frame_is_the_widest_sixteen_by_nine_the_view_holds() {
        let tall = framed(Rect::from_min_size(egui::Pos2::ZERO, vec2(1600.0, 1200.0)));
        assert!((tall.width() - 1600.0).abs() < 1e-3);
        assert!((tall.height() - 900.0).abs() < 1e-3);
        assert!((tall.center().y - 600.0).abs() < 1e-3);
        let wide = framed(Rect::from_min_size(egui::Pos2::ZERO, vec2(2400.0, 900.0)));
        assert!((wide.width() - 1600.0).abs() < 1e-3);
        assert!((wide.height() - 900.0).abs() < 1e-3);
    }

    fn fade(at: f32, from: f32, to: f32, over: f32) -> (f32, Fade) {
        (at, Fade { from, to, over })
    }

    #[test]
    fn a_fade_ramps_between_its_ends_and_holds_either_side_of_them() {
        let part = Part {
            faded: vec![fade(100.0, 1.0, 0.0, 20.0)],
            ..Part::default()
        };
        let drawn = |time| part.fading(time).map(|(_, fade, along)| fade.at(along));
        // Nothing has faded it yet, so it is whole.
        assert!(drawn(99.0).is_none());
        assert_eq!(drawn(100.0), Some(1.0));
        assert_eq!(drawn(110.0), Some(0.5));
        assert_eq!(drawn(120.0), Some(0.0));
        // Past its own length it holds the end it reached rather than reverting.
        assert_eq!(drawn(400.0), Some(0.0));
    }

    #[test]
    fn the_last_fade_to_have_started_is_the_one_that_holds() {
        let mut part = Part {
            faded: vec![fade(200.0, 0.0, 1.0, 10.0), fade(100.0, 1.0, 0.0, 10.0)],
            ..Part::default()
        };
        part.order();
        let drawn = |time| part.fading(time).map(|(_, fade, along)| fade.at(along));
        // A fade out followed by a fade back in leaves it whole, which a product of the two
        // would not.
        assert_eq!(drawn(150.0), Some(0.0));
        assert_eq!(drawn(210.0), Some(1.0));
    }

    #[test]
    fn a_participant_stands_where_its_own_timeline_last_put_it() {
        let player = Player {
            shots: Vec::new(),
            subtitles: Vec::new(),
            sounds: Vec::new(),
            parts: BTreeMap::from([(
                7,
                Part {
                    placed: vec![(0.0, placed(1.0)), (100.0, placed(2.0))],
                    ..Part::default()
                },
            )]),
            duration: 0.0,
        };
        assert_eq!(player.placed(7, 50.0).unwrap().translation()[0], 1.0);
        assert_eq!(player.placed(7, 150.0).unwrap().translation()[0], 2.0);
        // A participant its timelines never place keeps whatever its own record states.
        assert!(player.placed(8, 50.0).is_none());
    }
}
