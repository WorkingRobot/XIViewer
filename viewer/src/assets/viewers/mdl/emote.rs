//! Props, sound and vfx an emote's own timeline states, read out of the `.pap` its body motion
//! plays rather than out of anything the creator or the sheets name.
//!
//! A body motion is a Havok animation driven by a `.tmb` timeline embedded in the same `.pap`.
//! `C043`/`C198` summon a held prop, built the same way a weapon is
//! (`chara/weapon/w####/obj/body/b####/model/w####b####.mdl`); `C063` plays a sound; `C012`/`C173`
//! play a `.avfx`. None of the three name a bone of their own for a prop: measured against
//! `chara/xls/attachoffset/c0101.atch`, a tool held in the main hand (food, an axe, a hammer)
//! rests at `n_buki_r` with no offset, which is the same fallback a weapon takes with no `.atch`
//! tag resolved for it; a prop meant for the hip, the back or the off hand (a card, a drum, a
//! harp) is not, and nothing in the file says which is which, so this places every prop there.
//!
//! Timeline time is frames at a fixed 30 fps, measured across four packs by comparing `TMDH`'s own
//! duration against the Havok motion's, in seconds, that the same pack plays (330/11, 60/2,
//! 145/4.8333, 690/23 all divide out to exactly 30).

use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;

use glam::{EulerRot, Mat4, Quat, Vec3};
use ironworks::file::File;
use ironworks::file::pap::{AnimationPack, Binding};
use ironworks::file::sklb::SkeletonBinary;
use ironworks::file::tmb::{CommandKind, Item, Timeline};

use super::super::skeleton::Rig;

use crate::audio::{self, Mixer};
use crate::backend::Backend;
use crate::utils::TrackedPromise;

const FPS: f32 = 30.0;

/// The bone a prop, or a vfx bound to the character or the prop it summons, hangs from when
/// nothing names an attach point for it.
const FALLBACK_BONE: &str = "n_buki_r";

/// The bone a vfx bound to the plain character (`BindType::Character`) hangs from with no bone id
/// of its own: the same bone a pose is centred on.
const ROOT_BONE: &str = "n_hara";

fn secs(frames: i32) -> f32 {
    frames as f32 / FPS
}

fn prop(start: f32, held: f32, set: i16, base: i16, variant: i32) -> Prop {
    let (set, base) = (u16::try_from(set).unwrap_or(0), u16::try_from(base).unwrap_or(0));
    Prop {
        start,
        end: start + held,
        set,
        base,
        path: prop_model(set, base),
        variant: u16::try_from(variant).unwrap_or(0),
    }
}

/// `chara/weapon/w####/obj/body/b####/model/w####b####.mdl`, the same path a real weapon is
/// carried as.
fn prop_model(set: u16, base: u16) -> String {
    format!("chara/weapon/w{set:04}/obj/body/b{base:04}/model/w{set:04}b{base:04}.mdl")
}

/// The rig a prop is skinned to, filed beside its model.
fn prop_skeleton(set: u16, base: u16) -> String {
    format!("chara/weapon/w{set:04}/skeleton/base/b{base:04}/skl_w{set:04}b{base:04}.sklb")
}

/// The pack a prop moves out of, which is the one every weapon of that set shares.
fn prop_pack(set: u16) -> String {
    format!("chara/weapon/w{set:04}/animation/a0001/wp_common/resident/weapon.pap")
}

/// The body a pack is filed under, which its own path names.
fn body_code(pack: &str) -> Option<&str> {
    pack.strip_prefix("chara/human/")?.split('/').next()
}

/// What a body motion is named past the four letters saying which kind of motion it is, which is
/// what the prop's own pack names its animation for.
fn motion_key(motion: &str) -> &str {
    match motion.split_once('_') {
        Some((_, key)) => key,
        None => motion,
    }
}

/// `C012`/`C173`'s `BindType`, out of VFXEditor's `C012.cs`: which of the two default (`-1`) bind
/// ids this corpus carries resolves to a bone. A non-default id is left unresolved, since nothing
/// in the file states which bone a numeric id names.
fn bind_bone(bind_type: u8, bind_id: i16) -> Option<&'static str> {
    if bind_id != -1 {
        return None;
    }
    match bind_type {
        0 => Some(ROOT_BONE), // Character
        // Weapon, Offhand, Summon (the prop this timeline itself summons).
        1..=3 => Some(FALLBACK_BONE),
        _ => None,
    }
}

fn vec3(values: &[f32], default: f32) -> Vec3 {
    match values {
        [x, y, z] => Vec3::new(*x, *y, *z),
        _ => Vec3::splat(default),
    }
}

fn vec4(values: &[f32], default: f32) -> [f32; 4] {
    match values {
        [x, y, z, w] => [*x, *y, *z, *w],
        _ => [default; 4],
    }
}

fn local_transform(scale: &[f32], rotation: &[f32], position: &[f32]) -> Mat4 {
    let rotation = vec3(rotation, 0.0);
    Mat4::from_scale_rotation_translation(
        vec3(scale, 1.0),
        Quat::from_euler(EulerRot::ZYX, rotation.z, rotation.y, rotation.x),
        vec3(position, 0.0),
    )
}

/// A held prop's window, the weapon set and body it is filed under, and its material variant.
struct Prop {
    start: f32,
    end: f32,
    set: u16,
    base: u16,
    path: String,
    variant: u16,
}

/// A sound's own start, for firing once as the clock crosses it.
struct Sound {
    id: i16,
    start: f32,
    path: String,
}

/// A vfx the timeline fires: the file, where it is bound, its own tint, and when it starts. A
/// command that states a length of its own is over at the end of it; one that starts a loop and
/// leaves it running states none, and the effect's own length is what ends it instead.
struct Vfx {
    id: i16,
    start: f32,
    end: Option<f32>,
    bone: &'static str,
    path: String,
    local: Mat4,
    tint: [f32; 4],
}

/// One vfx running right now: the bone it hangs from, its placement relative to that bone, and how
/// far into its own run it is. The id is the firing's own, so a player handed the list again every
/// frame keeps each firing's particles.
pub struct Firing<'a> {
    pub id: u64,
    pub bone: &'a str,
    pub path: &'a str,
    pub local: Mat4,
    pub tint: [f32; 4],
    pub since: f32,
}

/// Everything one motion's timeline states, read once and kept until the motion or the pack
/// changes.
#[derive(Default)]
struct Events {
    props: Vec<Prop>,
    sounds: Vec<Sound>,
    vfx: Vec<Vfx>,
}

impl Events {
    fn read(bytes: &[u8], animation_name: &str) -> anyhow::Result<Self> {
        let pack = AnimationPack::read(Cursor::new(bytes.to_vec()))?;
        let index = pack
            .animations()
            .iter()
            .position(|animation| animation.name() == animation_name)
            .ok_or_else(|| anyhow::anyhow!("{animation_name}: not in this pack"))?;
        let timeline = Timeline::read(Cursor::new(pack.timelines()[index].clone()))?;

        let mut events = Self::default();
        for item in timeline.items() {
            let Item::Command(command) = item else {
                continue;
            };
            let start = secs(i32::from(command.time()));
            match command.kind() {
                CommandKind::C043(c) => events.props.push(prop(
                    start,
                    secs(c.duration()),
                    c.weapon_id(),
                    c.body_id(),
                    c.variant_id(),
                )),
                CommandKind::C198(c) => events.props.push(prop(
                    start,
                    secs(c.duration()),
                    c.model_id(),
                    c.body_id(),
                    c.variant(),
                )),
                CommandKind::C063(c) => {
                    if let Some(path) = c.path() {
                        events.sounds.push(Sound {
                            id: command.id(),
                            start,
                            path: path.to_owned(),
                        });
                    }
                }
                CommandKind::C012(c) => {
                    if c.path().is_some()
                        && let Some(bone) = bind_bone(c.bind_type_1(), c.bind_id_1())
                    {
                        events.vfx.push(Vfx {
                            id: command.id(),
                            start,
                            end: Some(start + secs(c.duration())),
                            bone,
                            path: c.path().unwrap_or_default().to_owned(),
                            local: local_transform(c.scale(), c.rotation(), c.position()),
                            tint: vec4(c.rgba(), 1.0),
                        });
                    }
                }
                CommandKind::C173(c) => {
                    if c.path().is_some()
                        && let Some(bone) = bind_bone(c.bind_type_1(), c.bind_id_1())
                    {
                        // No duration of its own: the command starts a loop and leaves it running
                        // rather than waiting on it, so what ends it is the effect's own length.
                        events.vfx.push(Vfx {
                            id: command.id(),
                            start,
                            end: None,
                            bone,
                            path: c.path().unwrap_or_default().to_owned(),
                            local: Mat4::IDENTITY,
                            tint: [1.0; 4],
                        });
                    }
                }
                _ => {}
            }
        }
        Ok(events)
    }

    /// Every prop the clock is inside the window of, in the order the timeline summons them: a
    /// motion that puts one thing in each hand summons twice rather than once.
    fn active_props(&self, time: f32) -> impl Iterator<Item = &Prop> {
        self.props
            .iter()
            .filter(move |prop| time >= prop.start && time < prop.end.max(prop.start + f32::EPSILON))
    }

    fn active_vfx(&self, time: f32) -> impl Iterator<Item = &Vfx> {
        self.vfx
            .iter()
            .filter(move |vfx| time >= vfx.start && vfx.end.is_none_or(|end| time < end))
    }

    /// Sounds whose start crossed between `since` (exclusive) and `time` (inclusive), wrapping
    /// once at the timeline's own duration for a motion that loops.
    fn due_sounds(&self, since: f32, time: f32) -> impl Iterator<Item = &Sound> {
        let wrapped = time < since;
        self.sounds.iter().filter(move |sound| match wrapped {
            false => sound.start > since && sound.start <= time,
            true => sound.start > since || sound.start <= time,
        })
    }
}

/// A prop's own rig: the skeleton it is skinned to, and the pack that walks its bones through the
/// emote. A prop that puts one thing in each hand is one model on this rig rather than two hung
/// apart, so which hand each of its bones ends up in is the pack's to say, not the attach point's.
struct Rigging {
    rig: Rig,
    /// Animation names, each with the motion it plays.
    named: Vec<(String, usize)>,
    bindings: Vec<Binding>,
    /// Where the model rests, inverted: what takes a vertex out of the pose the file stored it in.
    rest: Vec<Mat4>,
}

impl Rigging {
    fn read(skeleton: &[u8], pack: &[u8]) -> anyhow::Result<Self> {
        let held = SkeletonBinary::read(Cursor::new(skeleton.to_vec()))?.parse_skeleton()?;
        let rig = Rig::new(held.bones(), held.parent_indices(), held.reference_pose());
        let file = AnimationPack::read(Cursor::new(pack.to_vec()))?;
        let bindings = file.parse_animations()?;
        let named = file
            .animations()
            .iter()
            .filter_map(|animation| {
                let motion = usize::try_from(animation.havok_index()).ok()?;
                bindings.get(motion)?;
                Some((animation.name().to_owned(), motion))
            })
            .collect();
        let rest = rig
            .world(rig.reference())
            .iter()
            .map(|placement| placement.matrix().inverse())
            .collect();
        Ok(Self {
            rig,
            named,
            bindings,
            rest,
        })
    }

    /// Which of the pack's animations this body plays: the one named for both the motion and the
    /// body's own code, then the only one that names the code at all, since a pack files one
    /// animation per race and a motion's own name is not always spelled the way its prop's is.
    fn motion(&self, key: &str, code: &str) -> Option<usize> {
        let wanted = format!("cbew_{key}_{code}");
        let held = format!("_{code}");
        self.named
            .iter()
            .position(|(name, _)| *name == wanted)
            .or_else(|| {
                let mut named = self
                    .named
                    .iter()
                    .enumerate()
                    .filter(|(_, (name, _))| name.ends_with(&held));
                named.next().filter(|_| named.next().is_none()).map(|(at, _)| at)
            })
    }

    /// What each slot of the model's own bone table moves a vertex by, in the prop's own space.
    fn joints(&self, motion: usize, table: &[String], time: f32) -> Option<Vec<Mat4>> {
        let binding = self.bindings.get(self.named.get(motion)?.1)?;
        let mut locals = self.rig.reference().to_vec();
        self.rig
            .lay(&mut locals, binding, self.rig.names(), None, time, 1.0, false);
        let posed = self.rig.world(&locals);
        Some(
            table
                .iter()
                .map(|name| match self.rig.bone(name) {
                    Some(bone) => posed[bone].matrix() * self.rest[bone],
                    None => Mat4::IDENTITY,
                })
                .collect(),
        )
    }
}

enum Fetch {
    Fetching(TrackedPromise<anyhow::Result<Vec<u8>>>),
    Ready(Events),
    Failed,
}

/// A prop's own rig on its way in, and the motion this body plays out of it.
enum Rigged {
    Fetching(TrackedPromise<anyhow::Result<(Vec<u8>, Vec<u8>)>>),
    Ready(Rigging, usize),
    Failed,
}

enum SoundFetch {
    Fetching(TrackedPromise<anyhow::Result<audio::Decoded>>),
    Ready(Arc<audio::Decoded>),
    Failed,
}

/// One emote's props, sound and vfx, tracked against the body motion currently playing: which
/// pack this was read from, the clock it was last polled at, and the voices its sounds are
/// playing through.
#[derive(Default)]
pub struct Cue {
    key: Option<(String, String)>,
    fetch: Option<Fetch>,
    last_time: f32,
    loop_count: u32,
    decode: HashMap<String, SoundFetch>,
    voices: Option<Mixer<(i16, u32)>>,
    voices_failed: bool,
    /// The rig the prop now held is posed on, by the set and body it is filed under. A prop that
    /// ships none is carried whole at the point it hangs from instead.
    rigged: Option<((u16, u16), Rigged)>,
}

impl Cue {
    /// Takes up whatever the body is playing, fetching and parsing its pack's timeline once, and
    /// fires whichever sounds the clock has crossed since the last poll.
    pub fn poll(&mut self, backend: &Backend, playing: Option<(String, String, f32)>) {
        let Some((pack, name, time)) = playing else {
            return;
        };
        let key = (pack.clone(), name.clone());
        if self.key.as_ref() != Some(&key) {
            self.key = Some(key);
            self.fetch = None;
            // Below any real command time, so a sound at frame 0 still counts as due once the
            // pack finishes fetching rather than needing a loop back around to be crossed.
            self.last_time = -1.0;
        }
        match &mut self.fetch {
            None => {
                let files = backend.files().clone();
                let wanted = pack.clone();
                self.fetch = Some(Fetch::Fetching(TrackedPromise::spawn_local(async move {
                    files.read(&wanted).await
                })));
            }
            Some(Fetch::Fetching(promise)) => {
                if let Some(result) = promise.try_get() {
                    self.fetch = Some(match result.as_ref().map_err(ToString::to_string) {
                        Ok(bytes) => match Events::read(bytes, &name) {
                            Ok(events) => Fetch::Ready(events),
                            Err(_) => Fetch::Failed,
                        },
                        Err(_) => Fetch::Failed,
                    });
                }
            }
            Some(_) => {}
        }

        let held = match &self.fetch {
            Some(Fetch::Ready(events)) => events
                .active_props(time)
                .next()
                .map(|prop| (prop.set, prop.base)),
            _ => None,
        };
        self.poll_rig(backend, held, &name);

        let Some(Fetch::Ready(events)) = &self.fetch else {
            return;
        };
        if time < self.last_time {
            self.loop_count += 1;
        }
        let due: Vec<(i16, String)> = events
            .due_sounds(self.last_time, time)
            .map(|sound| (sound.id, sound.path.clone()))
            .collect();
        self.last_time = time;
        if due.is_empty() {
            return;
        }

        if self.voices.is_none() && !self.voices_failed {
            match Mixer::new() {
                Ok(mixer) => self.voices = Some(mixer),
                Err(why) => {
                    log::warn!("assets/mdl: no emote sound: {why}");
                    self.voices_failed = true;
                }
            }
        }
        let loop_count = self.loop_count;
        let Some(voices) = &mut self.voices else {
            return;
        };
        voices.unlock();
        voices.retain(|(_, held)| *held + 1 >= loop_count);
        for (id, path) in due {
            match self.decode.get(&path) {
                Some(SoundFetch::Ready(decoded)) => {
                    let decoded = decoded.clone();
                    if let Err(why) = voices.play((id, loop_count), decoded, 1.0) {
                        log::warn!("assets/mdl: emote sound play failed: {why}");
                    }
                }
                Some(SoundFetch::Fetching(_) | SoundFetch::Failed) => {}
                None => {
                    let files = backend.files().clone();
                    let wanted = path.clone();
                    self.decode.insert(
                        path,
                        SoundFetch::Fetching(TrackedPromise::spawn_local(async move {
                            let bytes = files.read(&wanted).await?;
                            let container =
                                ironworks::file::scd::SoundContainer::read(Cursor::new(bytes))?;
                            let entry = container
                                .entries()
                                .first()
                                .ok_or_else(|| anyhow::anyhow!("{wanted}: no audio streams"))?;
                            audio::decode_data(entry.format(), entry.data())
                        })),
                    );
                }
            }
        }
        for fetch in self.decode.values_mut() {
            if !matches!(fetch, SoundFetch::Fetching(_)) {
                continue;
            }
            let SoundFetch::Fetching(promise) = std::mem::replace(fetch, SoundFetch::Failed) else {
                unreachable!()
            };
            *fetch = match promise.try_take() {
                Ok(Ok(decoded)) => SoundFetch::Ready(Arc::new(decoded)),
                Ok(Err(why)) => {
                    log::warn!("assets/mdl: emote sound decode failed: {why}");
                    SoundFetch::Failed
                }
                Err(promise) => SoundFetch::Fetching(promise),
            };
        }
    }

    /// Asks for the rig the prop now held is posed on, and takes it up once both its skeleton and
    /// its pack have landed. A prop the game files no pack for is carried whole instead, which is
    /// what a failed fetch leaves it as.
    fn poll_rig(&mut self, backend: &Backend, held: Option<(u16, u16)>, motion: &str) {
        let Some((set, base)) = held else {
            self.rigged = None;
            return;
        };
        if self.rigged.as_ref().is_none_or(|(worn, _)| *worn != (set, base)) {
            let files = backend.files().clone();
            let (skeleton, pack) = (prop_skeleton(set, base), prop_pack(set));
            self.rigged = Some((
                (set, base),
                Rigged::Fetching(TrackedPromise::spawn_local(async move {
                    Ok((files.read(&skeleton).await?, files.read(&pack).await?))
                })),
            ));
        }
        let Some((_, Rigged::Fetching(promise))) = &mut self.rigged else {
            return;
        };
        let Some(landed) = promise.try_get() else {
            return;
        };
        let code = self
            .key
            .as_ref()
            .and_then(|(pack, _)| body_code(pack))
            .unwrap_or_default()
            .to_owned();
        let read = landed
            .as_ref()
            .ok()
            .and_then(|(skeleton, pack)| Rigging::read(skeleton, pack).ok())
            .and_then(|rigging| {
                let at = rigging.motion(motion_key(motion), &code)?;
                log::info!("assets/mdl: the prop plays {}", rigging.named[at].0);
                Some(Rigged::Ready(rigging, at))
            });
        self.rigged = Some(((set, base), read.unwrap_or(Rigged::Failed)));
    }

    /// The models an emote's own timeline wants held right now, each by the path it is worn as, its
    /// material variant and the weapon set it is filed under. Several where the motion summons
    /// several, in the order it names them.
    pub fn active_props(&self, time: f32) -> Vec<(String, u16, u16)> {
        let Some(Fetch::Ready(events)) = &self.fetch else {
            return Vec::new();
        };
        events
            .active_props(time)
            .map(|prop| (prop.path.clone(), prop.variant, prop.set))
            .collect()
    }

    /// Where each slot of the held prop's own bone table stands `time` seconds into the motion, in
    /// the prop's own space, for the prop `path` names. Nothing where it ships no pack of its own to
    /// move it, and nothing for any prop but the one the rig was fetched for.
    pub fn joints(&self, path: &str, table: &[String], time: f32) -> Option<Vec<Mat4>> {
        let ((set, base), Rigged::Ready(rigging, motion)) = self.rigged.as_ref()? else {
            return None;
        };
        (prop_model(*set, *base) == path).then_some(())?;
        rigging.joints(*motion, table, time)
    }

    /// The vfx running right now, each with how far into its own clock it has reached. A firing is
    /// told apart by the command that started it and the turn round the motion it started on, so
    /// one that survives a loop is a new firing rather than the same one carried on.
    pub fn firing(&self, time: f32) -> impl Iterator<Item = Firing<'_>> {
        let ready = match &self.fetch {
            Some(Fetch::Ready(events)) => Some(events),
            _ => None,
        };
        let turn = u64::from(self.loop_count) << 32;
        ready
            .into_iter()
            .flat_map(move |events| events.active_vfx(time))
            .map(move |vfx| Firing {
                id: turn | u64::from(vfx.id as u16),
                bone: vfx.bone,
                path: &vfx.path,
                local: vfx.local,
                tint: vfx.tint,
                since: time - vfx.start,
            })
    }
}

#[cfg(test)]
mod tests {
    use ironworks::Ironworks;
    use ironworks::sqpack::{Install, SqPack};

    use super::*;

    const SQPACK: &str = "/home/asriel/.xlcore/ffxiv/game/sqpack";

    /// Blow Bubbles, off the real install: the emote summons one prop, and the pack that prop
    /// ships walks its two bones into the character's own two hands. Measured as where each of
    /// them lands against the hand bone the body's own motion puts there at the same moment,
    /// carried on the point `attach.wtd` sends the prop's set to.
    #[test]
    #[ignore = "reads the real local FFXIV install"]
    fn a_real_props_own_pack_walks_it_into_the_hands_the_body_holds_it_in() {
        let install =
            Ironworks::new().with_resource(SqPack::new(Install::at_sqpack(SQPACK)));
        let read = |path: &str| install.file::<Vec<u8>>(path).expect(path);

        let body = "chara/human/c0101/animation/a0001/bt_common/emote_sp/sp63.pap";
        let events = Events::read(&read(body), "cbem_sp63").expect("a readable timeline");
        let [prop] = events.props.as_slice() else {
            panic!("sp63 summons one prop, found {}", events.props.len());
        };
        assert_eq!((prop.set, prop.base), (1949, 1));

        let rigging = Rigging::read(
            &read(&prop_skeleton(prop.set, prop.base)),
            &read(&prop_pack(prop.set)),
        )
        .expect("the prop's own rig");
        let motion = rigging
            .motion(motion_key("cbem_sp63"), "c0101")
            .expect("the pack names this body's own animation");
        assert_eq!(rigging.named[motion].0, "cbew_sp63_c0101");
        assert_eq!(
            rigging.motion(motion_key("cbep_u_sp63"), "c0101"),
            Some(motion),
            "the upper-body motion is spelled another way and reads the same animation"
        );

        // The body's own rig at the same moment, which is what the prop is measured against.
        let skeleton = SkeletonBinary::read(Cursor::new(read(
            "chara/human/c0101/skeleton/base/b0001/skl_c0101b0001.sklb",
        )))
        .expect("the body skeleton")
        .parse_skeleton()
        .expect("a readable tagfile");
        let rig = Rig::new(
            skeleton.bones(),
            skeleton.parent_indices(),
            skeleton.reference_pose(),
        );
        let pack = AnimationPack::read(Cursor::new(read(body))).expect("the body pack");
        let bindings = pack.parse_animations().expect("the body motion");

        let table = ["n_body".to_owned(), "n_head".to_owned()];
        let mut apart = [0.0f32; 2];
        let times = [0.0, 2.0, 5.0, 10.0, 16.0];
        for time in times {
            let mut locals = rig.reference().to_vec();
            rig.lay(&mut locals, &bindings[0], rig.names(), None, time, 1.0, false);
            let posed = rig.world(&locals);
            let held = posed[rig.bone("j_sebo_a").expect("the spine")].matrix();
            let joints = rigging.joints(motion, &table, time).expect("the prop's pose");
            for (at, hand) in ["n_buki_l", "n_buki_r"].iter().enumerate() {
                let bone = posed[rig.bone(hand).expect("a hand")].matrix();
                let stands = (held * joints[at]).to_scale_rotation_translation().2;
                apart[at] += stands.distance(bone.to_scale_rotation_translation().2);
            }
        }
        for (at, hand) in ["n_buki_l", "n_buki_r"].iter().enumerate() {
            let mean = apart[at] / times.len() as f32;
            assert!(mean < 0.02, "{} stands {mean} from {hand}", table[at]);
        }
    }

    /// Cheer On: Orange, off the real install: the emote summons the same penlight twice, and the
    /// second names attach state 4, which every point in `c0101.atch` sends to `n_throw`. Measured
    /// as where that bone stands against each hand over the motion.
    #[test]
    #[ignore = "reads the real local FFXIV install"]
    fn where_the_throw_bone_stands_over_cheer_on() {
        let install = Ironworks::new().with_resource(SqPack::new(Install::at_sqpack(SQPACK)));
        let read = |path: &str| install.file::<Vec<u8>>(path).expect(path);

        let skeleton = SkeletonBinary::read(Cursor::new(read(
            "chara/human/c0101/skeleton/base/b0001/skl_c0101b0001.sklb",
        )))
        .expect("the body skeleton")
        .parse_skeleton()
        .expect("a readable tagfile");
        let rig = Rig::new(
            skeleton.bones(),
            skeleton.parent_indices(),
            skeleton.reference_pose(),
        );
        let body = "chara/human/c0101/animation/a0001/bt_common/emote_sp/sp78_loop.pap";
        let pack = AnimationPack::read(Cursor::new(read(body))).expect("the body pack");
        let bindings = pack.parse_animations().expect("the body motion");
        let duration = bindings[0].motion().duration();

        for step in 0..=10 {
            let time = duration * step as f32 / 10.0;
            let mut locals = rig.reference().to_vec();
            rig.lay(&mut locals, &bindings[0], rig.names(), None, time, 1.0, false);
            let posed = rig.world(&locals);
            let at = |name: &str| {
                posed[rig.bone(name).expect(name)]
                    .matrix()
                    .to_scale_rotation_translation()
                    .2
            };
            let throw = at("n_throw");
            println!(
                "t={time:5.2} n_throw {throw:?} l {:.4} r {:.4}",
                throw.distance(at("n_buki_l")),
                throw.distance(at("n_buki_r"))
            );
        }
    }

    /// Cheer On: Orange, off the real install: the loop fires one `.avfx` eight times over its four
    /// seconds, each a firing of its own on its own clock, and every one of them names a file the
    /// install holds. `C173` states no length, so what is checked here is that the window a firing
    /// runs in is the motion's rather than one made up.
    #[test]
    #[ignore = "reads the real local FFXIV install"]
    fn a_real_emote_fires_its_own_effect_file_over_and_over() {
        let install = Ironworks::new().with_resource(SqPack::new(Install::at_sqpack(SQPACK)));
        let read = |path: &str| install.file::<Vec<u8>>(path).expect(path);

        let body = "chara/human/c0101/animation/a0001/bt_common/emote_sp/sp78_loop.pap";
        let events = Events::read(&read(body), "cbem_sp78_2lp").expect("a readable timeline");
        let sparkle = "vfx/emote_sp/emt_sp078/eff/emt_sp078_2lpc0c.avfx";
        assert_eq!(
            events.vfx.iter().filter(|vfx| vfx.path == sparkle).count(),
            8
        );
        for vfx in &events.vfx {
            assert!(vfx.end.is_none(), "{}: C173 states no length", vfx.path);
            assert!(
                install.file::<Vec<u8>>(&vfx.path).is_ok(),
                "{} is not in the install",
                vfx.path
            );
        }
        // Four seconds in, every one of them has started and none has been cut short.
        assert_eq!(events.active_vfx(4.0).count(), events.vfx.len());
        assert_eq!(
            events.active_vfx(0.0).count(),
            1,
            "only the sync marker fires at frame nought"
        );
    }

    /// Cheer On: Orange summons the same penlight twice at once, which is why one prop is not
    /// enough: both windows cover the whole four-second loop, and both name `w1980b0001`.
    #[test]
    #[ignore = "reads the real local FFXIV install"]
    fn a_real_emote_summons_the_same_model_into_both_hands() {
        let install = Ironworks::new().with_resource(SqPack::new(Install::at_sqpack(SQPACK)));
        let read = |path: &str| install.file::<Vec<u8>>(path).expect(path);

        let body = "chara/human/c0101/animation/a0001/bt_common/emote_sp/sp78_loop.pap";
        let events = Events::read(&read(body), "cbem_sp78_2lp").expect("a readable timeline");
        let held: Vec<&Prop> = events.active_props(2.0).collect();
        assert_eq!(held.len(), 2, "one in each hand");
        assert_eq!(held[0].path, held[1].path);
        assert_eq!(held[0].path, prop_model(1980, 1));
        assert_eq!((held[0].variant, held[1].variant), (1, 1));

        // Blow Bubbles summons one, and its own pack is what puts a bone in each hand instead.
        let alone = "chara/human/c0101/animation/a0001/bt_common/emote_sp/sp63.pap";
        let events = Events::read(&read(alone), "cbem_sp63").expect("a readable timeline");
        assert_eq!(events.active_props(2.0).count(), 1);
    }
}
