//! A rig: the bones a skeleton names, the tree its parent indices make, and a pose drawn in space.
//!
//! Both `.sklb` and `.pap` end up here, one drawing the reference pose and the other whatever a
//! motion samples to. Bones are drawn as a marker at each joint and a stick back to its parent,
//! which is a rig rather than the character it moves.

use std::cell::Cell;
use std::collections::HashMap;

use egui::{RichText, Sense};
use glam::{Mat4, Quat, Vec3};
use ironworks::file::pap::Binding;
use ironworks::file::sklb::Transform;

use super::{line, placed, table};

/// Space one level of the tree sets its bones in by.
const INDENT: usize = 2;

/// Marker at a joint, and half the width of a bone, both as a fraction of the rig's own extent.
const JOINT: f32 = 0.012;
const BONE: f32 = 0.005;

const BONE_COLOR: [f32; 4] = [0.62, 0.66, 0.72, 1.0];
const JOINT_COLOR: [f32; 4] = [0.90, 0.62, 0.30, 1.0];
const PICKED_COLOR: [f32; 4] = [0.35, 0.85, 0.95, 1.0];

/// Where a bone ended up once every transform above it has been applied.
#[derive(Clone, Copy)]
pub struct Placement {
    translation: Vec3,
    rotation: Quat,
    scale: Vec3,
}

impl Placement {
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }

    /// This placement carried by another, which is how a rider hangs off the seat its mount names.
    pub fn carried(&self, by: &Self) -> Self {
        Self {
            translation: by.translation + by.rotation * (by.scale * self.translation),
            rotation: by.rotation * self.rotation,
            scale: by.scale * self.scale,
        }
    }

    pub fn translation(&self) -> Vec3 {
        self.translation
    }

    /// Scaled about its own origin, along its own axes, which is what a proportion slider does to
    /// the one pair of bones it names.
    pub fn scaled(&self, by: Vec3) -> Self {
        Self {
            scale: self.scale * by,
            ..*self
        }
    }
}

fn axes(values: [f32; 4], count: usize) -> String {
    values[..count]
        .iter()
        .map(|value| format!("{value:.3}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// A skeleton's bones, ready to list and to draw.
pub struct Rig {
    names: Vec<String>,
    /// Which bone the skeleton calls each name, since everything else names one that way.
    at: HashMap<String, usize>,
    parents: Vec<i16>,
    reference: Vec<Transform>,
    /// Each bone and the depth it hangs at, ordered so a bone follows its parent.
    rows: Vec<(usize, usize)>,
    /// Widths the table pads its cells to, the first sized to the deepest name.
    columns: Vec<(&'static str, usize)>,
    /// How far across the rig reaches, which is what the markers are sized against.
    span: f32,
}

impl Rig {
    pub fn new(bones: &[String], parent_indices: &[i16], reference_pose: &[Transform]) -> Self {
        let at = bones
            .iter()
            .enumerate()
            .map(|(bone, name)| (name.clone(), bone))
            .collect();
        Self::built(
            bones.to_vec(),
            parent_indices.to_vec(),
            reference_pose.to_vec(),
            at,
        )
    }

    /// Shared by [`Self::new`] and [`Self::merged`], which differ only in where `at` came from: a
    /// merge keeps a name apart from the base's own rather than handing every lookup of it to
    /// whichever bone claimed it first.
    fn built(
        names: Vec<String>,
        parents: Vec<i16>,
        reference: Vec<Transform>,
        at: HashMap<String, usize>,
    ) -> Self {
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); names.len()];
        let mut roots = Vec::new();
        for bone in 0..names.len() {
            match parent_of(&parents, bone) {
                Some(parent) => children[parent].push(bone),
                None => roots.push(bone),
            }
        }

        // A bone is always written after its parent, so one pass down the list places every one.
        let mut rows = Vec::with_capacity(names.len());
        let mut depths = vec![0usize; names.len()];
        let mut stack = roots;
        stack.reverse();
        while let Some(bone) = stack.pop() {
            let depth = depths[bone];
            rows.push((bone, depth));
            for &child in children[bone].iter().rev() {
                depths[child] = depth + 1;
                stack.push(child);
            }
        }

        let widest = rows
            .iter()
            .map(|(bone, depth)| depth * INDENT + names[*bone].chars().count())
            .max()
            .unwrap_or(0);
        let columns = vec![
            ("Bone", widest + 2),
            ("Index", 6),
            ("Parent", 7),
            ("Translation", 26),
            ("Rotation", 34),
            ("Scale", 26),
        ];

        let mut rig = Self {
            names,
            at,
            parents,
            reference,
            rows,
            columns,
            span: 1.0,
        };
        let span = extent(&rig.world(&rig.reference));
        rig.span = span;
        rig
    }

    /// This rig with another skeleton's bones hung off the ones it already names, scoped to
    /// `origin` so a name it shares with the base or a previous extra for an unrelated bone is
    /// kept apart rather than merged into it. A name both carry stays this one's only where it is
    /// the extra skeleton's own structural root: an extra skeleton states the head where its own
    /// file put it rather than where the body's chain carries it.
    pub fn merged(
        &self,
        origin: &str,
        names: &[String],
        parents: &[i16],
        reference: &[Transform],
    ) -> Self {
        let mut held = self.names.clone();
        let mut hung = self.parents.clone();
        let mut rest = self.reference.clone();
        let mut at = self.at.clone();
        for (bone, name) in names.iter().enumerate() {
            let root = parent_of(parents, bone).is_none();
            if root && at.contains_key(name) {
                continue;
            }
            // A bone whose parent is nowhere to be found would stand at the world origin, which is
            // further from where it belongs than leaving it out entirely.
            let Some(parent) = parent_of(parents, bone).and_then(|parent| {
                let name = &names[parent];
                at.get(&scoped(origin, name))
                    .or_else(|| at.get(name))
                    .copied()
            }) else {
                continue;
            };
            hung.push(parent as i16);
            rest.push(reference[bone]);
            let key = match at.contains_key(name) {
                true => scoped(origin, name),
                false => name.clone(),
            };
            at.insert(key, held.len());
            held.push(name.clone());
        }
        Self::built(held, hung, rest, at)
    }

    pub fn bones(&self) -> usize {
        self.names.len()
    }

    /// What the skeleton calls each of its bones, which is how anything else names one.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Which bone the skeleton calls `name`.
    pub fn bone(&self, name: &str) -> Option<usize> {
        self.at.get(name).copied()
    }

    /// Which bone `name` reaches for a motion authored against `origin`'s own skeleton,
    /// preferring a bone `origin` had to keep apart from the base's over one they both happened
    /// to call the same thing.
    fn resolve(&self, origin: Option<&str>, name: &str) -> Option<usize> {
        origin
            .and_then(|origin| self.at.get(&scoped(origin, name)))
            .or_else(|| self.at.get(name))
            .copied()
    }

    /// Which bone `bone`'s own transform composes onto, if any.
    pub fn parent(&self, bone: usize) -> Option<usize> {
        parent_of(&self.parents, bone)
    }

    pub fn reference(&self) -> &[Transform] {
        &self.reference
    }

    /// Each bone's placement with everything above it applied, `locals` being one transform per
    /// bone in the skeleton's own order.
    pub fn world(&self, locals: &[Transform]) -> Vec<Placement> {
        let mut world: Vec<Placement> = Vec::with_capacity(self.names.len());
        for bone in 0..self.names.len() {
            let local = locals.get(bone).unwrap_or(&IDENTITY);
            let translation = Vec3::from_slice(&local.translation);
            let rotation = Quat::from_array(local.rotation);
            let scale = Vec3::from_slice(&local.scale);
            world.push(
                match parent_of(&self.parents, bone).map(|parent| world[parent]) {
                    Some(parent) => Placement {
                        translation: parent.translation
                            + parent.rotation * (parent.scale * translation),
                        rotation: parent.rotation * rotation,
                        scale: parent.scale * scale,
                    },
                    None => Placement {
                        translation,
                        rotation,
                        scale,
                    },
                },
            );
        }
        world
    }

    /// Lays a motion's tracks over `locals` at `time`, `weight` of the way there.
    ///
    /// A motion's tracks are in the order of the skeleton it was authored against, which need not
    /// be this rig, so `ordering` names that skeleton's bones and the two are matched by name.
    /// `origin` is that skeleton's own path where it is an extra one, so a name it shares with the
    /// base or another extra reaches the bone kept apart for it rather than whichever claimed the
    /// name first. A motion that blends states a delta in each bone's own frame rather than a pose
    /// of its own, so it composes onto whatever is already there.
    pub fn lay(
        &self,
        locals: &mut [Transform],
        binding: &Binding,
        ordering: &[String],
        origin: Option<&str>,
        time: f32,
        weight: f32,
        retarget: bool,
    ) {
        if weight <= 0.0 {
            return;
        }
        let blends = binding.blend_hint() != 0;
        for (track, sampled) in binding.motion().sample(time).into_iter().enumerate() {
            let Some(bone) = binding
                .bones()
                .get(track)
                .and_then(|bone| usize::try_from(*bone).ok())
                .and_then(|bone| ordering.get(bone))
                .and_then(|name| self.resolve(origin, name))
            else {
                continue;
            };
            let mut laid = match blends {
                true => over(&locals[bone], &sampled),
                false => sampled,
            };
            // A clip filed under another body states that body's own bone offsets, and a rig of
            // different proportions wearing them comes apart at every joint. What retargets is the
            // rotation; the lengths between the joints are the rig's own to keep.
            if retarget {
                let rest = &self.reference[bone];
                laid.translation = rest.translation;
                laid.scale = rest.scale;
            }
            locals[bone] = match weight >= 1.0 {
                true => laid,
                false => mix(&locals[bone], &laid, weight),
            };
        }
    }

    /// A marker at every joint and a stick from every bone back to its parent. The counts are fixed
    /// by the rig, so a bone of no length keeps its place as a stick of no size.
    pub fn batches(&self, world: &[Placement], picked: Option<usize>) -> Vec<placed::Batch> {
        let joint = (self.span * JOINT).max(f32::EPSILON);
        let bone = (self.span * BONE).max(f32::EPSILON);

        let mut instances = Vec::with_capacity(world.len() * 2);
        for (index, placement) in world.iter().enumerate() {
            instances.push(placed::Instance {
                center: placement.translation.to_array(),
                scale: [joint; 3],
                turn: placement.rotation.to_array(),
                color: match picked == Some(index) {
                    true => PICKED_COLOR,
                    false => JOINT_COLOR,
                },
            });
        }
        for (index, placement) in world.iter().enumerate() {
            let start = match parent_of(&self.parents, index) {
                Some(parent) => world[parent].translation,
                None => placement.translation,
            };
            let along = placement.translation - start;
            let length = along.length();
            instances.push(placed::Instance {
                center: ((start + placement.translation) * 0.5).to_array(),
                scale: [bone, bone, length * 0.5],
                turn: match length > f32::EPSILON {
                    true => Quat::from_rotation_arc(Vec3::Z, along / length).to_array(),
                    false => Quat::IDENTITY.to_array(),
                },
                color: match picked == Some(index) {
                    true => PICKED_COLOR,
                    false => BONE_COLOR,
                },
            });
        }

        vec![placed::Batch {
            shape: placed::Shape::Box,
            instances,
        }]
    }

    /// A view framed on the pose it is built with.
    pub fn view(&self, locals: &[Transform]) -> placed::View {
        placed::View::new(self.batches(&self.world(locals), None))
    }

    /// The bone tree, with the transform each one rests at. Clicking a row picks it out.
    pub fn tree_ui(&self, ui: &mut egui::Ui, locals: &[Transform], picked: &Cell<Option<usize>>) {
        table(ui, &self.columns, self.rows.len(), |ui, row| {
            let (bone, depth) = self.rows[row];
            let local = locals.get(bone).unwrap_or(&IDENTITY);
            let cells = [
                format!(
                    "{:indent$}{}",
                    "",
                    self.names[bone],
                    indent = depth * INDENT
                ),
                bone.to_string(),
                match parent_of(&self.parents, bone) {
                    Some(parent) => parent.to_string(),
                    None => "-".to_owned(),
                },
                axes(local.translation, 3),
                axes(local.rotation, 4),
                axes(local.scale, 3),
            ];
            let text =
                RichText::new(line(&self.columns, cells.iter().map(String::as_str))).monospace();
            let response = ui.add(
                egui::Label::new(match picked.get() == Some(bone) {
                    true => text.color(ui.visuals().hyperlink_color),
                    false => text,
                })
                .sense(Sense::click()),
            );
            if response.clicked() {
                picked.set((picked.get() != Some(bone)).then_some(bone));
            }
        });
    }
}

/// A delta applied in the frame the transform under it leaves, which is what a blending motion
/// states rather than a pose of its own.
fn over(base: &Transform, delta: &Transform) -> Transform {
    let rotation = Quat::from_array(base.rotation);
    let scale = Vec3::from_slice(&base.scale);
    let translation = Vec3::from_slice(&base.translation)
        + rotation * (scale * Vec3::from_slice(&delta.translation));
    let scale = scale * Vec3::from_slice(&delta.scale);
    Transform {
        translation: [
            translation.x,
            translation.y,
            translation.z,
            base.translation[3],
        ],
        rotation: (rotation * Quat::from_array(delta.rotation)).to_array(),
        scale: [scale.x, scale.y, scale.z, base.scale[3]],
    }
}

/// One transform `weight` of the way to another: the straight line for translation and scale, the
/// shortest arc for rotation.
fn mix(from: &Transform, to: &Transform, weight: f32) -> Transform {
    let translation =
        Vec3::from_slice(&from.translation).lerp(Vec3::from_slice(&to.translation), weight);
    let scale = Vec3::from_slice(&from.scale).lerp(Vec3::from_slice(&to.scale), weight);
    let rotation = Quat::from_array(from.rotation).slerp(Quat::from_array(to.rotation), weight);
    Transform {
        translation: [translation.x, translation.y, translation.z, to.translation[3]],
        rotation: rotation.to_array(),
        scale: [scale.x, scale.y, scale.z, to.scale[3]],
    }
}

/// A name kept apart from whatever the base or another extra already called the same thing,
/// since two skeletons can each have their own unrelated bone of that name.
fn scoped(origin: &str, name: &str) -> String {
    format!("{origin}\u{0}{name}")
}

/// A bone with nothing above it, and the one case a file could get wrong: a parent at or past the
/// bone itself, which the walks here read in order and so could not reach.
fn parent_of(parents: &[i16], bone: usize) -> Option<usize> {
    usize::try_from(*parents.get(bone)?)
        .ok()
        .filter(|parent| *parent < bone)
}

const IDENTITY: Transform = Transform {
    translation: [0.0; 4],
    rotation: [0.0, 0.0, 0.0, 1.0],
    scale: [1.0, 1.0, 1.0, 0.0],
};

/// Where a pose stands and how far the furthest bone reaches from there, which is what a pose is
/// framed and clipped by. `anchor` is the bone the body hangs off; without one this falls back to
/// the middle of every bone, which a long tail drags around each time it swings.
pub fn middle(world: &[Placement], anchor: Option<usize>) -> (Vec3, f32) {
    if world.is_empty() {
        return (Vec3::ZERO, 0.0);
    }
    let center = match anchor.and_then(|bone| world.get(bone)) {
        Some(placement) => placement.translation,
        None => {
            world
                .iter()
                .map(|placement| placement.translation)
                .sum::<Vec3>()
                / world.len() as f32
        }
    };
    let reach = world
        .iter()
        .map(|placement| placement.translation.distance(center))
        .fold(0.0, f32::max);
    (center, reach)
}

/// How far across the pose reaches, which sizes the markers drawn on it.
fn extent(world: &[Placement]) -> f32 {
    let mut low = Vec3::splat(f32::INFINITY);
    let mut high = Vec3::splat(f32::NEG_INFINITY);
    for placement in world {
        low = low.min(placement.translation);
        high = high.max(placement.translation);
    }
    match low.x <= high.x {
        true => (high - low).length().max(1e-3),
        false => 1.0,
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::FRAC_PI_2;

    use glam::{Quat, Vec3};

    use super::{Rig, Transform, mix, over};

    fn transform(translation: [f32; 3]) -> Transform {
        Transform {
            translation: [translation[0], translation[1], translation[2], 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0, 0.0],
        }
    }

    /// A chain of three offset bones, so the world walk has something to accumulate.
    fn rig(parents: Vec<i16>) -> Rig {
        Rig::new(
            &["a".to_owned(), "b".to_owned(), "c".to_owned()],
            &parents,
            &[
                transform([1.0, 0.0, 0.0]),
                transform([0.0, 2.0, 0.0]),
                transform([0.0, 0.0, 3.0]),
            ],
        )
    }

    #[test]
    fn composes_each_bone_onto_its_parent() {
        let rig = rig(vec![-1, 0, 1]);
        let world = rig.world(rig.reference());
        assert_eq!(world[0].translation.to_array(), [1.0, 0.0, 0.0]);
        assert_eq!(world[1].translation.to_array(), [1.0, 2.0, 0.0]);
        assert_eq!(world[2].translation.to_array(), [1.0, 2.0, 3.0]);
    }

    /// A marker per joint and a stick per bone, whether or not the bone has any length.
    #[test]
    fn draws_every_bone_whatever_its_length() {
        let rig = rig(vec![-1, 0, 0]);
        let world = rig.world(rig.reference());
        let batches = rig.batches(&world, None);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].instances.len(), 6);
        assert!(
            batches[0]
                .instances
                .iter()
                .all(|instance| instance.turn.iter().all(|value| value.is_finite())),
            "a bone of no length turned to nowhere"
        );
    }

    /// A blending motion states a delta in the frame the bone it drives already sits in, so its
    /// translation turns with that bone rather than with the parent.
    #[test]
    fn a_delta_is_applied_in_the_frame_below_it() {
        let base = Transform {
            translation: [1.0, 0.0, 0.0, 0.0],
            rotation: Quat::from_rotation_z(FRAC_PI_2).to_array(),
            scale: [2.0, 2.0, 2.0, 0.0],
        };
        let delta = Transform {
            translation: [1.0, 0.0, 0.0, 0.0],
            rotation: Quat::from_rotation_z(FRAC_PI_2).to_array(),
            scale: [0.5, 0.5, 0.5, 0.0],
        };
        let held = over(&base, &delta);
        assert!(Vec3::from_slice(&held.translation).abs_diff_eq(Vec3::new(1.0, 2.0, 0.0), 1e-5));
        assert!(
            Quat::from_array(held.rotation)
                .abs_diff_eq(Quat::from_rotation_z(2.0 * FRAC_PI_2), 1e-5)
        );
        assert_eq!(Vec3::from_slice(&held.scale), Vec3::ONE);
    }

    /// The neutral face states an identity delta, so a blending motion of nothing must leave the
    /// pose under it exactly as it rests.
    #[test]
    fn a_delta_of_nothing_leaves_the_pose_alone() {
        let base = Transform {
            translation: [1.0, -2.0, 3.0, 0.0],
            rotation: Quat::from_rotation_y(0.7).to_array(),
            scale: [1.5, 1.5, 1.5, 0.0],
        };
        let held = over(
            &base,
            &Transform {
                translation: [0.0; 4],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0, 1.0, 1.0, 0.0],
            },
        );
        assert!(Vec3::from_slice(&held.translation)
            .abs_diff_eq(Vec3::from_slice(&base.translation), 1e-6));
        assert!(Quat::from_array(held.rotation).abs_diff_eq(Quat::from_array(base.rotation), 1e-6));
        assert_eq!(held.scale, base.scale);
    }

    /// A face skeleton is merged onto a body's by name, so a motion authored against either
    /// reaches the same bones through the names its own skeleton lists.
    #[test]
    fn a_merged_bone_is_reached_by_the_name_its_own_skeleton_lists() {
        let body = Rig::new(
            &["n_root".to_owned(), "j_kao".to_owned()],
            &[-1, 0],
            &[transform([0.0, 0.0, 0.0]), transform([0.0, 1.0, 0.0])],
        );
        let merged = body.merged(
            "face",
            &["j_kao".to_owned(), "j_f_face".to_owned()],
            &[-1, 0],
            &[transform([0.0, 9.0, 0.0]), transform([0.0, 0.1, 0.0])],
        );
        assert_eq!(merged.bone("j_kao"), Some(1));
        assert_eq!(merged.bone("j_f_face"), Some(2));
        assert_eq!(merged.bone("j_f_ago"), None);
    }

    /// The extra skeleton's own root aliases onto the base's, same name and all, but a bone one
    /// level under it that only happens to share a name with an unrelated base bone must stay its
    /// own: Viera's face skeleton carries a `j_ago` that is not the body's jaw.
    #[test]
    fn a_colliding_non_root_bone_stays_apart_from_the_bases_own() {
        let base = Rig::new(
            &["n_root".to_owned(), "j_kao".to_owned(), "j_ago".to_owned()],
            &[-1, 0, 1],
            &[
                transform([0.0, 0.0, 0.0]),
                transform([0.0, 1.0, 0.0]),
                transform([0.0, 0.1, 0.0]),
            ],
        );
        let merged = base.merged(
            "face",
            &[
                "j_kao".to_owned(),
                "j_ago".to_owned(),
                "j_f_noanim_ago".to_owned(),
            ],
            &[-1, 0, 1],
            &[
                transform([0.0, 9.0, 0.0]),
                transform([0.0, 0.2, 0.0]),
                transform([0.0, 0.05, 0.0]),
            ],
        );
        assert_eq!(merged.bones(), 5);
        let base_ago = merged.bone("j_ago").expect("the body keeps its own jaw");
        assert_eq!(base_ago, 2);
        let face_ago = merged
            .resolve(Some("face"), "j_ago")
            .expect("the face's own j_ago must not be dropped");
        assert_ne!(face_ago, base_ago, "the two j_agos collapsed into one bone");
        assert_eq!(merged.parent(face_ago), merged.bone("j_kao"));
        let noanim = merged
            .resolve(Some("face"), "j_f_noanim_ago")
            .expect("added with nothing to collide with");
        assert_eq!(merged.parent(noanim), Some(face_ago));
    }

    #[test]
    fn a_mix_stands_at_either_end_and_halfway_between() {
        let turned = Transform {
            translation: [4.0, 0.0, 0.0, 0.0],
            rotation: Quat::from_rotation_z(FRAC_PI_2).to_array(),
            scale: [3.0, 3.0, 3.0, 0.0],
        };
        let rest = transform([0.0, 0.0, 0.0]);
        assert_eq!(mix(&rest, &turned, 0.0).translation, rest.translation);
        assert_eq!(mix(&rest, &turned, 1.0).translation, turned.translation);

        let half = mix(&rest, &turned, 0.5);
        assert_eq!(half.translation[0], 2.0);
        assert_eq!(half.scale[0], 2.0);
        let angle = Quat::from_array(half.rotation).to_axis_angle().1;
        assert!((angle - FRAC_PI_2 / 2.0).abs() < 1e-5, "{angle}");
    }

    /// A parent index at or past its own bone would leave the ordered walks below it unreachable,
    /// so it reads as a root instead.
    #[test]
    fn a_parent_that_is_not_above_its_bone_is_a_root() {
        let rig = rig(vec![-1, 2, 1]);
        let world = rig.world(rig.reference());
        assert_eq!(world[1].translation.to_array(), [0.0, 2.0, 0.0]);
        assert_eq!(world[2].translation.to_array(), [0.0, 2.0, 3.0]);
        assert_eq!(rig.rows.len(), 3);
    }
}
