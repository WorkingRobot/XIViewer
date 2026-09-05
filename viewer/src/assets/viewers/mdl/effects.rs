//! The `.avfx` an emote's own timeline fires, drawn as the game's own particles.
//!
//! A firing is one command's own start, so a motion firing the same file eight times over its loop
//! runs eight of them at once, each on its own clock. None of the commands that start a loop state
//! a length, so what ends one is the length the file itself states.

use std::collections::HashMap;
use std::io::Cursor;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use glam::{Mat4, Vec3, Vec4};
use ironworks::file::File as _;
use ironworks::file::avfx::{Avfx, Block};

use super::super::avfx::{self, Shaders, Textures, gpu, program, sim};
use crate::backend::Backend;
use crate::settings::AVFX_FRAME_RATE;
use crate::utils::TrackedPromise;

/// How many files to keep parsed and on the card at once. The busiest motion in the game fires
/// seven distinct `.avfx` (`emote_sp/sp46`), and a change of motion has the outgoing one's files in
/// hand alongside the incoming one's, so twice that never evicts a file still firing.
const KEPT: usize = 16;

/// One firing to draw: the file, where it stands in the world, how far into its own run it is, and
/// the tint the command that started it states.
pub struct Fired {
    pub id: u64,
    pub path: String,
    pub at: Mat4,
    pub since: f32,
    pub tint: Vec4,
}

enum File {
    Fetching(TrackedPromise<Result<Vec<u8>>>),
    Ready(Box<Held>),
    Failed,
}

/// A file parsed once, with the card-side geometry its model particles draw.
struct Held {
    effect: sim::Effect,
    particles: Arc<Mutex<gpu::Particles>>,
    /// The bone each of the file's own binders hangs it from, for the binders whose id a capture
    /// has actually pinned. Empty where none of them are known, which is most files.
    bound: Vec<&'static str>,
}

/// Where a binder's own id hangs its effect. A file states a numeric id per binder and the client
/// resolves it against a list of attachments the character carries, keyed by id rather than
/// indexed; nothing here reads that list, so only the ids a game capture has placed are answered.
///
/// 77 and 78 are the two hands, off the Cheer On: Blue capture, which is what puts a light in each.
/// 43 and 44 are the two eyes, off the Frighten capture: that file holds one emitter and two
/// binders, so its two glows are its two bind points and nothing else.
fn bound(id: i32) -> Option<&'static str> {
    match id {
        43 => Some("j_f_eye_l"),
        44 => Some("j_f_eye_r"),
        77 => Some("n_buki_r"),
        78 => Some("n_buki_l"),
        _ => None,
    }
}

/// A binder states its bind point under `BPTP`/`BPID`, nested inside its own property tree.
fn binders(file: &Avfx) -> Vec<&'static str> {
    fn dig(block: &Block, name: &str, into: &mut Vec<i32>) {
        if block.name().as_str() == name
            && let Some(value) = block.i32()
        {
            into.push(value);
        }
        for held in block.blocks() {
            dig(held, name, into);
        }
    }
    file.binders()
        .iter()
        .filter_map(|binder| {
            let (mut kind, mut id) = (Vec::new(), Vec::new());
            dig(binder, "BPTP", &mut kind);
            dig(binder, "BPID", &mut id);
            // Kind 3 is the one that hangs off the character; 0 states no attachment at all.
            (kind.first() == Some(&3)).then_some(())?;
            bound(*id.first()?)
        })
        .collect()
}

/// A file on its way in or in hand, with the last poll it was fired on.
struct Kept {
    file: File,
    fired: u64,
}

/// Every effect an emote is firing, kept across frames: the files parsed once each, the particles
/// each firing has run out, and the textures and packages they are all drawn through.
#[derive(Default)]
pub struct Effects {
    files: HashMap<String, Kept>,
    running: HashMap<u64, sim::State>,
    textures: Textures,
    shaders: Shaders,
    /// Counts polls, so the least recently fired file is the one to give up.
    polls: u64,
}

impl Effects {
    /// The bones the file at `path` hangs itself from, where its own binders name any this knows.
    /// Empty until the file has landed, and for every file whose ids are unpinned.
    pub fn bound(&self, path: &str) -> &[&'static str] {
        match self.files.get(path).map(|kept| &kept.file) {
            Some(File::Ready(held)) => &held.bound,
            _ => &[],
        }
    }

    /// Takes up whatever is firing this frame: asks for any file not in hand, steps each firing to
    /// where its own clock has reached, and forgets the ones no longer named.
    pub fn poll(&mut self, ctx: &egui::Context, backend: &Backend, fired: &[Fired]) {
        if fired.is_empty() && self.files.is_empty() {
            return;
        }
        self.shaders.poll(backend);
        self.polls += 1;
        let polls = self.polls;
        for held in fired {
            self.files
                .entry(held.path.clone())
                .or_insert_with(|| {
                    let files = backend.files().clone();
                    let wanted = held.path.clone();
                    Kept {
                        file: File::Fetching(TrackedPromise::spawn_local(async move {
                            files.read(&wanted).await
                        })),
                        fired: polls,
                    }
                })
                .fired = polls;
        }
        // A file no longer fired is kept against the emote being picked again, but only so many:
        // its particles hold buffers on the card, and the creator's own list is a few hundred
        // emotes long.
        while self.files.len() > KEPT {
            let Some(stalest) = self
                .files
                .iter()
                .filter(|(_, kept)| kept.fired < polls)
                .min_by_key(|(_, kept)| kept.fired)
                .map(|(path, _)| path.clone())
            else {
                break;
            };
            self.files.remove(&stalest);
        }
        for (path, kept) in self.files.iter_mut() {
            let file = &mut kept.file;
            let File::Fetching(promise) = file else {
                continue;
            };
            let Some(landed) = promise.try_get() else {
                continue;
            };
            *file = match landed
                .as_ref()
                .map_err(ToString::to_string)
                .and_then(|bytes| {
                    Avfx::read(Cursor::new(bytes.clone())).map_err(|why| why.to_string())
                }) {
                Ok(read) => {
                    let mut effect = sim::Effect::read(&read);
                    // Nothing reads the models again once they are on the card: a particle already
                    // carries the index it draws.
                    let models = std::mem::take(&mut effect.models);
                    log::info!("assets/mdl: the emote fires {path}, {} frames", effect.length);
                    File::Ready(Box::new(Held {
                        bound: binders(&read),
                        effect,
                        particles: gpu::Particles::new(models),
                    }))
                }
                Err(why) => {
                    log::warn!("assets/mdl: {path}: {why}");
                    File::Failed
                }
            };
        }

        let wanted: Vec<String> = self
            .files
            .values()
            .filter_map(|kept| match &kept.file {
                File::Ready(held) => Some(held.effect.textures.clone()),
                _ => None,
            })
            .flatten()
            .collect();
        self.textures.poll(ctx, backend, &wanted);

        let rate = AVFX_FRAME_RATE.get(ctx);
        let Self { files, running, .. } = self;
        running.retain(|id, _| fired.iter().any(|held| held.id == *id));
        for held in fired {
            let Some(File::Ready(file)) = files.get(&held.path).map(|kept| &kept.file) else {
                continue;
            };
            let end = match file.effect.bounded {
                true => file.effect.length,
                false => sim::LONGEST,
            };
            let frame = (held.since * rate) as i32;
            file.effect
                .seek(running.entry(held.id).or_default(), frame.clamp(0, end));
        }
    }

    /// What to draw this frame, one entry per file however many firings it has: a draw is the
    /// file's own programs and geometry, so every firing of one goes into a single stream.
    pub fn frames(
        &self,
        fired: &[Fired],
        view: Mat4,
        projection: Mat4,
        size: (f32, f32),
        eye: Vec3,
    ) -> Vec<(Arc<Mutex<gpu::Particles>>, gpu::Frame)> {
        // A sprite is set into the screen's plane, which is what the camera's own axes are for.
        let axes = glam::Mat3::from_mat4(view).transpose();
        let (right, up) = (axes.x_axis, axes.y_axis);
        self.files
            .iter()
            .filter_map(|(path, kept)| {
                let File::Ready(file) = &kept.file else {
                    return None;
                };
                let bound = self.textures.bound(&file.effect.textures);
                let drawn: Vec<sim::Drawn> = fired
                    .iter()
                    .filter(|held| held.path == *path)
                    .filter_map(|held| {
                        let state = self.running.get(&held.id)?;
                        let (scale, rotation, translation) = held.at.to_scale_rotation_translation();
                        let scale = scale.abs().max_element().max(0.001);
                        Some(
                            file.effect
                                .drawn(state)
                                .into_iter()
                                .map(move |item| item.placed(rotation, translation, scale, held.tint)),
                        )
                    })
                    .flatten()
                    .collect();
                let batches = avfx::batches(&file.effect, drawn, &bound, view, eye, right, up);
                (!batches.is_empty()).then(|| {
                    (
                        file.particles.clone(),
                        gpu::Frame {
                            scene: program::Scene {
                                view,
                                projection,
                                size,
                                light: (eye - Vec3::ZERO).normalize_or(Vec3::Y),
                                fade_range: file.effect.fade_range,
                                ..program::Scene::default()
                            },
                            batches,
                            packages: self.shaders.resolved(),
                            // Drawn after the character has been composited, which leaves no depth
                            // to test against and nothing to copy for the soft-particle variant.
                            tested: false,
                            depth: None,
                        },
                    )
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    /// Only the ids two captures have actually placed are answered; everything else is left where
    /// the command bound it rather than guessed at.
    #[test]
    fn only_a_pinned_bind_point_names_a_bone() {
        assert_eq!(super::bound(77), Some("n_buki_r"));
        assert_eq!(super::bound(78), Some("n_buki_l"));
        assert_eq!(super::bound(43), Some("j_f_eye_l"));
        assert_eq!(super::bound(44), Some("j_f_eye_r"));
        // The corpus carries 5, 8-11, 16, 25-30, 32, 33, 42, 107 and 108 besides, and no capture
        // places any of them.
        for id in [0, 5, 16, 30, 42, 107, 108] {
            assert_eq!(super::bound(id), None, "{id} is not pinned by anything");
        }
    }
}
