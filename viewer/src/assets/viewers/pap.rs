//! `.pap` animation packs: the motions one skeleton can play, the timeline driving each, and the
//! motions themselves played back on the skeleton the pack is built for.

use crate::assets::viewers::skeleton::Laid;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::io::Cursor;

use anyhow::Result;
use egui::{Color32, RichText, ScrollArea};
use ironworks::file::pap::{AnimationPack, Binding, ModelType};
use ironworks::file::sklb::SkeletonBinary;
use ironworks::file::{File, tmb};

use super::{
    Preview, chara, facts, heading, link, placed, section, skeleton::Rig, tmb as timeline,
};
use crate::assets::Bytes;
use crate::backend::Backend;
use crate::utils::{TrackedPromise, file_name};

/// Bytes a pack with no motions of its own holds where the tagfile would be.
const NO_HAVOK: usize = 8;

/// One animation of the pack, and the timeline it plays alongside.
struct Animation {
    name: String,
    kind: u16,
    havok_index: i16,
    face: bool,
    timeline: tmb::Timeline,
    items: timeline::Items,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Animations,
    Playback,
}

/// An animation pack, decoded and ready to draw.
pub struct Rendered {
    identity: Vec<(&'static str, String)>,
    animations: Vec<Animation>,
    /// The motions the pack holds, or why the tagfile gave none.
    motions: std::result::Result<Vec<Binding>, String>,
    /// The model the skeleton behind these animations is filed under, as in the `c0101` of a path.
    model: Option<String>,
    play: Play,
    tab: Cell<Tab>,
}

/// What the playback view is showing, and the skeletons it has fetched to show it on.
#[derive(Default)]
struct Play {
    /// Which animation is on screen, indexing [`Rendered::animations`].
    animation: Cell<usize>,
    /// How far into it, in seconds.
    time: Cell<f32>,
    running: Cell<bool>,
    rigs: RefCell<HashMap<String, Loading>>,
}

enum Loading {
    Fetching(TrackedPromise<Result<Vec<u8>>>),
    Ready(Box<Loaded>),
    Failed(String),
}

struct Loaded {
    rig: Rig,
    view: placed::View,
}

/// The model these animations are built for, written the way its own files are named.
fn model(kind: ModelType, id: u16) -> String {
    match kind {
        ModelType::Human => chara::described(id),
        ModelType::Monster => format!("m{id:04}"),
        ModelType::DemiHuman => format!("d{id:04}"),
        ModelType::Weapon => format!("w{id:04}"),
        ModelType::Unknown(_) => id.to_string(),
    }
}

fn model_type(kind: ModelType) -> String {
    match kind {
        ModelType::Unknown(value) => format!("Unknown ({value})"),
        named => format!("{named:?}"),
    }
}

/// The `c0101` a pack's model code spells out, which is the directory its skeletons sit under.
fn model_code(kind: ModelType, id: u16) -> Option<String> {
    let letter = match kind {
        ModelType::Human => 'c',
        ModelType::Monster => 'm',
        ModelType::DemiHuman => 'd',
        ModelType::Weapon => 'w',
        ModelType::Unknown(_) => return None,
    };
    Some(format!("{letter}{id:04}"))
}

/// Where a motion's skeleton is filed, best guess first.
///
/// A motion states the rig it was authored against, which is often another character's: the name is
/// taken only for the sub-skeleton it picks out under the pack's own model, and the pack's base
/// skeleton is what everything else falls back to.
fn skeleton_paths(code: &str, motion: &str) -> Vec<String> {
    let directory = match code.as_bytes().first() {
        Some(b'c') => "human",
        Some(b'm') => "monster",
        Some(b'd') => "demihuman",
        Some(b'w') => "weapon",
        _ => return Vec::new(),
    };
    let base = format!("chara/{directory}/{code}/skeleton/base/b0001/skl_{code}b0001.sklb");
    match face(code, motion) {
        Some(sub) => vec![
            format!("chara/{directory}/{code}/skeleton/face/{sub}/skl_{code}{sub}.sklb"),
            base,
        ],
        None => vec![base],
    }
}

/// The `f0212` of `c0101f0212_0:mdl:j_kao`, for the motions that drive a face rather than a body.
fn face<'a>(code: &str, motion: &'a str) -> Option<&'a str> {
    let rig = motion.split(":mdl:").next()?;
    let rig = rig.rsplit_once('_').map_or(rig, |(head, _)| head);
    let sub = rig.strip_prefix(code)?;
    let named = sub.len() == 5
        && sub.starts_with('f')
        && sub[1..].bytes().all(|byte| byte.is_ascii_digit());
    (named && motion.ends_with(":j_kao")).then_some(sub)
}

pub fn decode(path: &str, bytes: &[u8]) -> Result<Preview> {
    let file = AnimationPack::read(Cursor::new(bytes.to_vec()))?;

    let animations = file
        .animations()
        .iter()
        .zip(file.timelines())
        .map(|(animation, bytes)| {
            let timeline = tmb::Timeline::read(Cursor::new(bytes.clone()))?;
            Ok(Animation {
                name: animation.name().to_owned(),
                kind: animation.animation_type(),
                havok_index: animation.havok_index(),
                face: animation.face(),
                items: timeline::Items::new(&timeline),
                timeline,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let motions = match file.havok().len() > NO_HAVOK {
        true => file.parse_animations().map_err(|e| e.to_string()),
        false => Err("this pack carries no motions of its own".to_owned()),
    };

    let identity = vec![
        ("Version", format!("{:#010x}", file.version())),
        ("Model", model(file.model_type(), file.model_id())),
        ("Model type", model_type(file.model_type())),
        ("Variant", file.variant().to_string()),
        ("Animations", animations.len().to_string()),
        (
            "Motions",
            match &motions {
                Ok(motions) => motions.len().to_string(),
                Err(e) => e.clone(),
            },
        ),
        ("Havok", Bytes(file.havok().len()).to_string()),
    ];

    log::info!(
        "assets/pap: {path} {} animations, {} bytes of havok",
        animations.len(),
        file.havok().len()
    );

    Ok(Preview::Pap(Box::new(Rendered {
        identity,
        animations,
        motions,
        model: model_code(file.model_type(), file.model_id()),
        play: Play::default(),
        tab: Cell::new(Tab::Animations),
    })))
}

pub fn ui(ui: &mut egui::Ui, file: &Rendered, backend: &Backend) -> Option<String> {
    let mut follow = None;
    ui.horizontal(|ui| {
        for (tab, label) in [(Tab::Animations, "Animations"), (Tab::Playback, "Playback")] {
            if ui.selectable_label(file.tab.get() == tab, label).clicked() {
                file.tab.set(tab);
            }
        }
    });
    ui.add_space(4.0);

    if file.tab.get() == Tab::Animations {
        section(ui, "Animations");
        ScrollArea::both().auto_shrink(false).show(ui, |ui| {
            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
            for animation in &file.animations {
                heading(ui, &animation.name);
                ui.label(
                    RichText::new(format!(
                        "type {}, havok motion {}, {}, {} items",
                        animation.kind,
                        animation.havok_index,
                        match animation.face {
                            true => "face",
                            false => "body",
                        },
                        animation.timeline.items().len()
                    ))
                    .monospace()
                    .weak(),
                );
                if let Some(path) = animation.items.ui(ui, &animation.timeline) {
                    follow = Some(path);
                }
            }
        });
        return follow;
    }

    let motions = match &file.motions {
        Ok(motions) => motions,
        Err(e) => {
            ui.centered_and_justified(|ui| {
                ui.colored_label(Color32::RED, format!("No motions to play: {e}"));
            });
            return follow;
        }
    };

    file.picker(ui, motions);
    let index = file.play.animation.get();
    let Some(binding) = file
        .animations
        .get(index)
        .and_then(|animation| usize::try_from(animation.havok_index).ok())
        .and_then(|motion| motions.get(motion))
    else {
        ui.label(RichText::new("This animation names no motion").weak());
        return follow;
    };

    file.transport(ui, binding);
    file.scene(ui, binding, backend, &mut follow);
    follow
}

impl Rendered {
    pub fn details_ui(&self, ui: &mut egui::Ui) {
        ScrollArea::vertical()
            .auto_shrink(false)
            .show(ui, |ui| facts(ui, "pap_identity", &self.identity));
    }

    /// Which animation is playing, as a row of every one the pack holds.
    fn picker(&self, ui: &mut egui::Ui, motions: &[Binding]) {
        ScrollArea::horizontal()
            .id_salt("pap_picker")
            .max_height(ui.text_style_height(&egui::TextStyle::Button) + 8.0)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    for (index, animation) in self.animations.iter().enumerate() {
                        let named = usize::try_from(animation.havok_index)
                            .is_ok_and(|motion| motion < motions.len());
                        let label = ui
                            .add_enabled_ui(named, |ui| {
                                ui.selectable_label(
                                    self.play.animation.get() == index,
                                    &animation.name,
                                )
                            })
                            .inner;
                        if label.clicked() {
                            self.play.animation.set(index);
                            self.play.time.set(0.0);
                        }
                    }
                });
            });
        ui.add_space(4.0);
    }

    /// Play, pause and the scrubber, which is also what advances the clock.
    fn transport(&self, ui: &mut egui::Ui, binding: &Binding) {
        let duration = binding.motion().duration().max(f32::EPSILON);
        let mut time = self.play.time.get().clamp(0.0, duration);

        if self.play.running.get() {
            time += ui.input(|input| input.stable_dt).min(duration);
            if time > duration {
                time -= duration;
            }
            // Nothing else asks for a frame while the pointer is still, so playback has to.
            ui.ctx().request_repaint();
        }

        ui.horizontal(|ui| {
            let running = self.play.running.get();
            if ui.button(if running { "Pause" } else { "Play" }).clicked() {
                self.play.running.set(!running);
            }
            if ui.button("Restart").clicked() {
                time = 0.0;
            }
            ui.add(
                egui::Slider::new(&mut time, 0.0..=duration)
                    .fixed_decimals(3)
                    .suffix(" s"),
            );
            ui.label(
                RichText::new(format!(
                    "{} frames, {} tracks, blend {}",
                    binding.motion().frames(),
                    binding.bones().len(),
                    binding.blend_hint()
                ))
                .weak(),
            );
        });
        self.play.time.set(time);
        ui.add_space(4.0);
    }

    /// The skeleton the motion drives, fetched on first use, with the motion posed onto it.
    fn scene(
        &self,
        ui: &mut egui::Ui,
        binding: &Binding,
        backend: &Backend,
        follow: &mut Option<String>,
    ) {
        let Some(code) = &self.model else {
            ui.label(RichText::new("This pack names no model to find a skeleton by").weak());
            return;
        };
        let candidates = skeleton_paths(code, binding.skeleton());

        let mut rigs = self.play.rigs.borrow_mut();
        for (index, path) in candidates.iter().enumerate() {
            rigs.entry(path.clone()).or_insert_with(|| {
                let files = backend.files().clone();
                let wanted = path.clone();
                Loading::Fetching(TrackedPromise::spawn_local(async move {
                    files.read(&wanted).await
                }))
            });
            if let Some(Loading::Fetching(promise)) = rigs.get(path)
                && let Some(result) = promise.try_get()
            {
                let landed = match result {
                    Ok(bytes) => match load(bytes) {
                        Ok(loaded) => Loading::Ready(Box::new(loaded)),
                        Err(e) => Loading::Failed(e.to_string()),
                    },
                    Err(e) => Loading::Failed(e.to_string()),
                };
                rigs.insert(path.clone(), landed);
            }

            match rigs.get(path) {
                Some(Loading::Ready(loaded)) => {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Skeleton").weak());
                        if link(ui, file_name(path), path) {
                            *follow = Some(path.clone());
                        }
                        // The tracks then index a face rig's bones against a body's, which draws
                        // something plausible and wrong.
                        if binding.skeleton().ends_with("j_kao")
                            && !path.contains("/skeleton/face/")
                        {
                            ui.label(
                                RichText::new("(a face motion, with no face skeleton to draw on)")
                                    .weak(),
                            );
                        }
                    });
                    ui.add_space(4.0);
                    let mut locals = loaded.rig.reference().to_vec();
                    loaded.rig.lay(
                        &mut locals,
                        binding,
                        loaded.rig.names(),
                        Laid {
                            time: self.play.time.get(),
                            weight: 1.0,
                            ..Laid::default()
                        },
                    );
                    let world = loaded.rig.world(&locals);
                    loaded.view.replace(loaded.rig.batches(&world, None));
                    loaded.view.ui(ui);
                    return;
                }
                Some(Loading::Failed(e)) => {
                    if index + 1 == candidates.len() {
                        ui.centered_and_justified(|ui| {
                            ui.colored_label(Color32::RED, format!("No skeleton at {path}: {e}"));
                        });
                        return;
                    }
                }
                _ => {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label(RichText::new(format!("Loading {path}")).weak());
                    });
                    ui.ctx().request_repaint();
                    return;
                }
            }
        }
        ui.label(RichText::new("This motion names no skeleton to draw it on").weak());
    }
}

fn load(bytes: &[u8]) -> Result<Loaded> {
    let file = SkeletonBinary::read(Cursor::new(bytes.to_vec()))?;
    let skeleton = file.parse_skeleton()?;
    let rig = Rig::new(
        skeleton.bones(),
        skeleton.parent_indices(),
        skeleton.reference_pose(),
    );
    let view = rig.view(rig.reference());
    Ok(Loaded { rig, view })
}

#[cfg(test)]
mod tests {
    use super::{face, skeleton_paths};

    /// A motion states the rig it was authored on, which is another character's often enough that
    /// only the part naming a sub-skeleton of this one is worth reading.
    #[test]
    fn reads_a_face_only_out_of_this_models_own_rig() {
        assert_eq!(face("c0101", "c0101f0212_0:mdl:j_kao"), Some("f0212"));
        assert_eq!(face("c0101", "c0201f0212_0:mdl:j_kao"), None);
        assert_eq!(face("c0101", "c0101_0:mdl:n_root"), None);
        assert_eq!(face("d1018", "d1018f0000_0:mdl:n_root"), None);
        assert_eq!(face("c0101", "j_kao"), None);
    }

    #[test]
    fn falls_back_to_the_models_base_skeleton() {
        assert_eq!(
            skeleton_paths("m0089", "n_root"),
            ["chara/monster/m0089/skeleton/base/b0001/skl_m0089b0001.sklb"]
        );
        assert_eq!(
            skeleton_paths("c0101", "c0101f0212_0:mdl:j_kao"),
            [
                "chara/human/c0101/skeleton/face/f0212/skl_c0101f0212.sklb",
                "chara/human/c0101/skeleton/base/b0001/skl_c0101b0001.sklb"
            ]
        );
        assert!(skeleton_paths("x0001", "n_root").is_empty());
    }
}
