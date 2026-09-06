//! The rig a carried weapon moves on.
//!
//! A weapon is not posed on the character's own rig. It ships a skeleton of its own and one pack
//! every weapon of its set shares, and that pack names the same motions everywhere: measured over a
//! sample spanning the whole id range, all 678 of them file `cbnw_id0` and `cbbw_id0`, and most file
//! `cbbw_activ` and `cbbw_deact` besides. Those four are what close a grimoire, stow a set of
//! nouliths and fold an astrologian's globe.

use std::collections::HashMap;

use glam::Mat4;

use super::emote::{Rigging, weapon_pack, weapon_skeleton};
use crate::backend::Backend;
use crate::utils::TrackedPromise;

/// Where a weapon rests, sheathed then drawn.
const IDLE: [&str; 2] = ["cbnw_id0", "cbbw_id0"];

/// What carries it there, on the same footing. A set is free to file neither, and one that does not
/// simply arrives in the pose.
const ENTER: [&str; 2] = ["cbbw_deact", "cbbw_activ"];

/// The set and body a weapon model path names, which is what its rig is filed under.
pub fn worn(path: &str) -> Option<(u16, u16)> {
    let rest = path.strip_prefix("chara/weapon/w")?;
    let (set, rest) = rest.split_once("/obj/body/b")?;
    let (base, _) = rest.split_once('/')?;
    Some((set.parse().ok()?, base.parse().ok()?))
}

enum Rigged {
    Fetching(TrackedPromise<anyhow::Result<(Vec<u8>, Vec<u8>)>>),
    Ready(Box<Rigging>),
    Failed,
}

/// The rigs of the weapons carried right now, by the set and body each is filed under.
#[derive(Default)]
pub struct Wield {
    held: HashMap<(u16, u16), Rigged>,
    /// The stance last seen and the clock it turned on, so the motion between the two runs once
    /// from its own beginning rather than from wherever the frame clock stands.
    turned: Option<(bool, f64)>,
}

impl Wield {
    /// Asks for the rig of every set carried this frame and drops the ones no longer held, noting
    /// when the stance last changed.
    pub fn poll(&mut self, backend: &Backend, worn: &[(u16, u16)], drawn: bool, now: f64) {
        if self.turned.is_none_or(|(held, _)| held != drawn) {
            self.turned = Some((drawn, now));
        }
        self.held.retain(|held, _| worn.contains(held));
        for &(set, base) in worn {
            self.held.entry((set, base)).or_insert_with(|| {
                let files = backend.files().clone();
                let (skeleton, pack) = (weapon_skeleton(set, base), weapon_pack(set));
                Rigged::Fetching(TrackedPromise::spawn_local(async move {
                    Ok((files.read(&skeleton).await?, files.read(&pack).await?))
                }))
            });
        }
        for (&(set, _), rigged) in &mut self.held {
            let Rigged::Fetching(promise) = rigged else {
                continue;
            };
            let Some(landed) = promise.try_get() else {
                continue;
            };
            *rigged = match landed
                .as_ref()
                .ok()
                .and_then(|(skeleton, pack)| Rigging::read(skeleton, pack).ok())
            {
                Some(rigging) => {
                    log::info!("assets/mdl: w{set:04} moves on a rig of its own");
                    Rigged::Ready(Box::new(rigging))
                }
                None => Rigged::Failed,
            };
        }
    }

    /// What each slot of a carried weapon's own bone table moves a vertex by, in the weapon's own
    /// space: the motion that carries it into the stance while that one still runs, and the idle it
    /// settles in after. Nothing for a set whose rig has not landed, which leaves it carried whole.
    pub fn joints(&self, set: u16, base: u16, table: &[String], now: f64) -> Option<Vec<Mat4>> {
        let Rigged::Ready(rigging) = self.held.get(&(set, base))? else {
            return None;
        };
        let (drawn, turned) = self.turned?;
        let since = (now - turned) as f32;
        let stance = usize::from(drawn);
        let entering = rigging
            .named(ENTER[stance])
            .filter(|motion| rigging.duration(*motion).is_some_and(|held| since < held));
        match entering {
            Some(motion) => rigging.joints(motion, table, since),
            None => {
                let motion = rigging.named(IDLE[stance])?;
                let duration = rigging.duration(motion)?;
                rigging.joints(motion, table, since % duration)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use ironworks::Ironworks;
    use ironworks::file::File;
    use ironworks::file::sklb::SkeletonBinary;
    use ironworks::sqpack::{Install, SqPack};

    const SQPACK: &str = "/home/asriel/.xlcore/ffxiv/game/sqpack";

    #[test]
    fn a_weapon_model_path_names_the_set_and_body_its_rig_is_filed_under() {
        assert_eq!(
            super::worn("chara/weapon/w1710/obj/body/b0001/model/w1710b0001.mdl"),
            Some((1710, 1))
        );
        assert_eq!(super::worn("chara/human/c0101/obj/body/b0001/model/x.mdl"), None);
    }

    /// A weapon that opens, off the real install: its own pack names both idles, and the two put its
    /// bones in different places. That difference is the thing itself - a grimoire shut while it
    /// hangs and open once it is drawn - and the motion that carries it between them is filed too.
    #[test]
    #[ignore = "reads the real local FFXIV install"]
    fn a_weapons_own_pack_holds_it_shut_sheathed_and_open_drawn() {
        let install = Ironworks::new().with_resource(SqPack::new(Install::at_sqpack(SQPACK)));
        let read = |path: &str| install.file::<Vec<u8>>(path).expect(path);
        let (set, base) = (1710, 1);
        let skeleton = read(&super::weapon_skeleton(set, base));
        let rigging = super::Rigging::read(&skeleton, &read(&super::weapon_pack(set)))
            .expect("the weapon's own rig");

        let held = SkeletonBinary::read(Cursor::new(skeleton))
            .expect("the skeleton")
            .parse_skeleton()
            .expect("a readable tagfile");
        let table = held.bones().to_vec();
        assert!(!table.is_empty(), "the weapon carries bones of its own");

        let shut = rigging.named(super::IDLE[0]).expect("a sheathed idle");
        let open = rigging.named(super::IDLE[1]).expect("a drawn idle");
        assert_ne!(
            rigging.joints(shut, &table, 0.0).expect("the shut pose"),
            rigging.joints(open, &table, 0.0).expect("the open pose"),
            "the two idles leave every bone in one place"
        );
        assert!(
            rigging.named(super::ENTER[1]).is_some(),
            "w{set:04} files the motion that opens it"
        );
    }
}
